"""
Assignment Task 3: Jointly Trained Soft Mixture-of-Experts Restoration.

Reuses (not duplicates):
  - practise/data_pipeline.py, practise/utils.py
  - Assignment_Task1/Assignment_Task1.py  -- UniversalAutoencoder (the experts)
  - Assignment_Task2/Assignment_Task2.py  -- CorruptionClassifier (the gate)

HARD DEPENDENCY: this file cannot train for real until Task 2 has produced
  Assignment_Task2/classifier_best_params.json + checkpoints/classifier_final.pt
  Assignment_Task2/specialist_arch_params.json + checkpoints/specialist_*.pt
Until then, `smoketest` mode builds fresh (randomly-initialized) stand-ins
to prove the pipeline is wired correctly -- it does NOT substitute for the
real initialization-from-Task-2 requirement.

w = softmax(G(x_tilde) / tau)                         over [clean, salt, blur, occlusion]
x_hat = w0*x_tilde + w1*A_salt(x_tilde) + w2*A_blur(x_tilde) + w3*A_occlusion(x_tilde)

Training procedure (assignment-mandated, not optional):
  1. Initialize gate <- Task 2 classifier, experts <- Task 2 specialists.
  2. Warm-up: freeze experts, train gate only.
  3. Joint fine-tune: unfreeze everything, smaller LR, combined loss:
       L = lambda1*L1 + lambda_s*(1-SSIM) + lambda_c*CE + lambda_b*L_balance
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "practise"))
sys.path.insert(0, str(PROJECT_ROOT / "Assignment_Task1"))
sys.path.insert(0, str(PROJECT_ROOT / "Assignment_Task2"))

from utils import SSIMLoss, get_device, set_seed  # noqa: E402
from data_pipeline import (  # noqa: E402
    CORRUPTION_TYPES, PetTrainDataset, PetManifestDataset, list_clean_images,
    train_val_split, generate_validation_manifest,
)
from Assignment_Task1 import UniversalAutoencoder, IMAGES_DIR  # noqa: E402
from Assignment_Task2 import CorruptionClassifier  # noqa: E402

HERE = Path(__file__).resolve().parent
CHECKPOINT_DIR = HERE / "checkpoints"
BEST_PARAMS_PATH = HERE / "best_params.json"

TASK2_DIR = PROJECT_ROOT / "Assignment_Task2"
CLASSIFIER_PARAMS_PATH = TASK2_DIR / "classifier_best_params.json"
CLASSIFIER_CKPT_PATH = TASK2_DIR / "checkpoints" / "classifier_final.pt"
SPECIALIST_ARCH_PATH = TASK2_DIR / "specialist_arch_params.json"
SPECIALIST_CKPT_DIR = TASK2_DIR / "checkpoints"

# Warm-up LR is fixed (not an Optuna dimension) -- the assignment only
# requires the JOINT fine-tuning LR to be searched.
WARMUP_LR = 1e-3


# ==========================================================================
# Model
# ==========================================================================

class SoftMoE(nn.Module):
    """gate: CorruptionClassifier (reused as-is -- its 4 logits become routing
    weights via softmax(/tau) instead of argmax). experts: UniversalAutoencoder
    instances, one each for salt/blur/occlusion. Clean is an identity branch,
    not a network -- there's nothing to learn for "leave it alone"."""

    def __init__(self, gate: CorruptionClassifier, expert_salt: UniversalAutoencoder,
                 expert_blur: UniversalAutoencoder, expert_occlusion: UniversalAutoencoder):
        super().__init__()
        self.gate = gate
        self.expert_salt = expert_salt
        self.expert_blur = expert_blur
        self.expert_occlusion = expert_occlusion

    def freeze_experts(self, freeze: bool = True):
        for expert in (self.expert_salt, self.expert_blur, self.expert_occlusion):
            for p in expert.parameters():
                p.requires_grad = not freeze

    def forward(self, x_tilde: torch.Tensor, temperature: float = 1.0):
        logits = self.gate(x_tilde)
        weights = F.softmax(logits / temperature, dim=1)  # (B, 4): [clean, salt, blur, occ]

        w_clean = weights[:, 0].view(-1, 1, 1, 1)
        w_salt = weights[:, 1].view(-1, 1, 1, 1)
        w_blur = weights[:, 2].view(-1, 1, 1, 1)
        w_occ = weights[:, 3].view(-1, 1, 1, 1)

        out_salt = self.expert_salt(x_tilde)
        out_blur = self.expert_blur(x_tilde)
        out_occ = self.expert_occlusion(x_tilde)

        x_hat = w_clean * x_tilde + w_salt * out_salt + w_blur * out_blur + w_occ * out_occ
        return x_hat, weights, logits


def balance_loss(weights: torch.Tensor) -> torch.Tensor:
    """L_balance = sum_k (w_bar_k - 1/4)^2 -- penalizes the gate for sending
    nearly everything to one branch (routing collapse)."""
    w_bar = weights.mean(dim=0)
    target = torch.full_like(w_bar, 1.0 / weights.size(1))
    return ((w_bar - target) ** 2).sum()


class MoELoss(nn.Module):
    def __init__(self, lambda1=0.8, lambda_s=0.2, lambda_c=0.1, lambda_b=0.01):
        super().__init__()
        self.lambda1 = lambda1
        self.lambda_s = lambda_s
        self.lambda_c = lambda_c
        self.lambda_b = lambda_b
        self.l1 = nn.L1Loss()
        self.ssim_loss = SSIMLoss()
        self.ce = nn.CrossEntropyLoss()

    def forward(self, x_hat, x_clean, weights, gate_logits, true_labels):
        l1_term = self.l1(x_hat, x_clean)
        ssim_term = self.ssim_loss(x_hat, x_clean)
        ce_term = self.ce(gate_logits, true_labels)
        bal_term = balance_loss(weights)
        total = (self.lambda1 * l1_term + self.lambda_s * ssim_term +
                 self.lambda_c * ce_term + self.lambda_b * bal_term)
        return total, {"l1": l1_term.item(), "ssim": ssim_term.item(),
                        "ce": ce_term.item(), "balance": bal_term.item()}


# ==========================================================================
# Loading pretrained Task 2 components (the mandatory initialization)
# ==========================================================================

def build_and_load_pretrained_moe(device) -> SoftMoE:
    """Builds the gate + 3 experts using Task 2's ACTUAL architecture configs
    and loads Task 2's trained weights -- this is the only correct way to
    start Task 3, per the assignment."""
    missing = [p for p in [CLASSIFIER_PARAMS_PATH, CLASSIFIER_CKPT_PATH,
                            SPECIALIST_ARCH_PATH] if not p.exists()]
    if missing:
        raise SystemExit(
            "Missing Task 2 outputs, cannot build a real Task 3 model:\n" +
            "\n".join(f"  - {p}" for p in missing) +
            "\nFinish Task 2 (classifier-optuna/train, specialist-optuna/train) first."
        )

    with open(CLASSIFIER_PARAMS_PATH) as f:
        clf_params = json.load(f)
    with open(SPECIALIST_ARCH_PATH) as f:
        spec_params = json.load(f)

    gate = CorruptionClassifier(clf_params["base_channels"], clf_params["dropout"]).to(device)
    gate.load_state_dict(torch.load(CLASSIFIER_CKPT_PATH, map_location=device))

    experts = {}
    for corruption in ["salt_pepper", "blur", "occlusion"]:
        ckpt_path = SPECIALIST_CKPT_DIR / f"specialist_{corruption}.pt"
        if not ckpt_path.exists():
            raise SystemExit(f"Missing specialist checkpoint: {ckpt_path}")
        expert = UniversalAutoencoder(
            base_channels=spec_params["base_channels"],
            bottleneck_spatial=spec_params["bottleneck_spatial"],
            dropout=spec_params["dropout"],
            activation=spec_params["activation"],
        ).to(device)
        expert.load_state_dict(torch.load(ckpt_path, map_location=device))
        experts[corruption] = expert

    return SoftMoE(gate, experts["salt_pepper"], experts["blur"], experts["occlusion"])


# ==========================================================================
# Training
# ==========================================================================

def warmup_train(model: SoftMoE, loader, optimizer, loss_fn, device,
                  temperature: float = 1.0, epochs: int = 3):
    """Experts frozen -- the gate alone learns to route against ALREADY-GOOD
    experts, instead of both learning against each other's noise at once."""
    model.freeze_experts(True)
    model.train()
    for epoch in range(epochs):
        total_loss, n = 0.0, 0
        for batch in loader:
            x = batch["corrupted"].to(device)
            y = batch["clean"].to(device)
            labels = torch.as_tensor(batch["label"]).to(device)

            optimizer.zero_grad()
            x_hat, weights, gate_logits = model(x, temperature)
            loss, _ = loss_fn(x_hat, y, weights, gate_logits, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * x.size(0)
            n += x.size(0)
        print(f"[warmup] epoch {epoch+1}/{epochs}  loss={total_loss/max(n,1):.4f}")


def joint_finetune(model: SoftMoE, loader, optimizer, loss_fn, device,
                    temperature: float = 1.0, epochs: int = 10):
    """Everything unfrozen, smaller LR (set on the optimizer by the caller)."""
    model.freeze_experts(False)
    model.train()
    for epoch in range(epochs):
        total_loss, n = 0.0, 0
        collapse_warned = False
        for batch in loader:
            x = batch["corrupted"].to(device)
            y = batch["clean"].to(device)
            labels = torch.as_tensor(batch["label"]).to(device)

            optimizer.zero_grad()
            x_hat, weights, gate_logits = model(x, temperature)
            loss, _ = loss_fn(x_hat, y, weights, gate_logits, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * x.size(0)
            n += x.size(0)

            if not collapse_warned and weights.mean(dim=0).max().item() > 0.9:
                print(f"  WARNING: possible routing collapse "
                      f"(max mean branch weight={weights.mean(dim=0).max().item():.3f})")
                collapse_warned = True
        print(f"[joint] epoch {epoch+1}/{epochs}  loss={total_loss/max(n,1):.4f}")


@torch.no_grad()
def evaluate_moe(model: SoftMoE, loader, loss_fn, device, temperature: float = 1.0):
    model.eval()
    total_loss, n = 0.0, 0
    for batch in loader:
        x = batch["corrupted"].to(device)
        y = batch["clean"].to(device)
        labels = torch.as_tensor(batch["label"]).to(device)
        x_hat, weights, gate_logits = model(x, temperature)
        loss, _ = loss_fn(x_hat, y, weights, gate_logits, labels)
        total_loss += loss.item() * x.size(0)
        n += x.size(0)
    return total_loss / max(n, 1)


# ==========================================================================
# Routing analysis (required report deliverable)
# ==========================================================================

@torch.no_grad()
def routing_weights_by_corruption(model: SoftMoE, loader, device, temperature: float = 1.0):
    """Returns {(corruption, severity): mean_weight_vector} -- the exact
    table/heatmap shape the report needs."""
    model.eval()
    accum = {}
    for batch in loader:
        x = batch["corrupted"].to(device)
        weights = F.softmax(model.gate(x) / temperature, dim=1).cpu().numpy()
        corruptions = batch["corruption"]
        severities = batch.get("severity", ["none"] * len(corruptions))
        for i, (corr, sev) in enumerate(zip(corruptions, severities)):
            accum.setdefault((corr, sev), []).append(weights[i])
    return {k: np.mean(np.stack(v), axis=0) for k, v in accum.items()}


@torch.no_grad()
def check_expert_health(routing_summary: dict, dead_threshold: float = 0.05,
                         dominant_threshold: float = 0.6):
    """Flags experts that never get used (dead) or that dominate even
    unrelated (non-matching) corruption types (over-dominant)."""
    branch_names = ["clean", "salt_pepper", "blur", "occlusion"]
    overall = np.mean(list(routing_summary.values()), axis=0)
    report = {"overall_mean_weight": dict(zip(branch_names, overall.tolist()))}

    dead = [branch_names[i] for i, w in enumerate(overall) if w < dead_threshold]
    report["dead_experts"] = dead

    dominant_elsewhere = []
    for (corruption, severity), w in routing_summary.items():
        for i, branch in enumerate(branch_names):
            if branch != "clean" and corruption != branch and w[i] > dominant_threshold:
                dominant_elsewhere.append({
                    "branch": branch, "on_corruption": corruption, "severity": severity,
                    "weight": float(w[i]),
                })
    report["unexpected_dominance_cases"] = dominant_elsewhere
    return report


# ==========================================================================
# Optuna: joint fine-tuning hyperparameters
# ==========================================================================

def moe_objective(train_ids, val_manifest, device, max_epochs=3):
    def objective(trial):
        lr = trial.suggest_float("lr", 1e-6, 1e-4, log=True)
        temperature = trial.suggest_float("temperature", 0.3, 3.0)
        lambda_c = trial.suggest_float("lambda_c", 0.01, 0.5, log=True)
        lambda_b = trial.suggest_float("lambda_b", 0.001, 0.1, log=True)
        lambda1 = trial.suggest_float("lambda1", 0.5, 0.9)
        lambda_s = 1.0 - lambda1

        model = build_and_load_pretrained_moe(device)
        loss_fn = MoELoss(lambda1=lambda1, lambda_s=lambda_s, lambda_c=lambda_c, lambda_b=lambda_b)

        train_ds = PetTrainDataset(IMAGES_DIR, train_ids)
        train_loader = DataLoader(train_ds, batch_size=16, shuffle=True, drop_last=True)
        val_ds = PetManifestDataset(IMAGES_DIR, val_manifest)
        val_loader = DataLoader(val_ds, batch_size=16, shuffle=False)

        warmup_opt = torch.optim.Adam(model.gate.parameters(), lr=WARMUP_LR)
        warmup_train(model, train_loader, warmup_opt, loss_fn, device, temperature, epochs=1)

        joint_opt = torch.optim.Adam(model.parameters(), lr=lr)
        best_val = float("inf")
        for epoch in range(max_epochs):
            joint_finetune(model, train_loader, joint_opt, loss_fn, device, temperature, epochs=1)
            val_loss = evaluate_moe(model, val_loader, loss_fn, device, temperature)
            best_val = min(best_val, val_loss)

            # Routing-collapse-aware pruning: a trial whose gate collapsed onto
            # one branch is not a useful trial, prune it regardless of loss.
            routing = routing_weights_by_corruption(model, val_loader, device, temperature)
            health = check_expert_health(routing)
            if len(health["dead_experts"]) >= 2:
                import optuna
                raise optuna.TrialPruned()

            trial.report(val_loss, epoch)
            if trial.should_prune():
                import optuna
                raise optuna.TrialPruned()

        return best_val

    return objective


def run_moe_optuna_study(n_trials: int = 20, max_epochs_per_trial: int = 3):
    import optuna

    set_seed(42)
    device = get_device()
    stems = list_clean_images(IMAGES_DIR)
    train_ids, val_ids = train_val_split(stems, seed=42)
    val_manifest = generate_validation_manifest(val_ids, seed=42)

    study = optuna.create_study(direction="minimize", pruner=optuna.pruners.MedianPruner())
    objective = moe_objective(train_ids, val_manifest, device, max_epochs_per_trial)
    study.optimize(objective, n_trials=n_trials)

    print("\n=== Task 3 Optuna study complete ===")
    print(f"Trials run: {len(study.trials)}")
    print(f"Best trial: #{study.best_trial.number}  best val loss: {study.best_value:.4f}")
    print(f"Best params: {study.best_params}")

    with open(BEST_PARAMS_PATH, "w") as f:
        json.dump(study.best_params, f, indent=2)
    print(f"Saved to {BEST_PARAMS_PATH}")
    return study


def train_final_moe(best_params: dict, warmup_epochs: int = 5, joint_epochs: int = 20):
    set_seed(42)
    device = get_device()
    stems = list_clean_images(IMAGES_DIR)
    train_ids, val_ids = train_val_split(stems, seed=42)
    val_manifest = generate_validation_manifest(val_ids, seed=42)

    model = build_and_load_pretrained_moe(device)
    print(f"SoftMoE total params: {sum(p.numel() for p in model.parameters()):,}")

    lambda1 = best_params["lambda1"]
    loss_fn = MoELoss(lambda1=lambda1, lambda_s=1.0 - lambda1,
                       lambda_c=best_params["lambda_c"], lambda_b=best_params["lambda_b"])
    temperature = best_params["temperature"]

    train_ds = PetTrainDataset(IMAGES_DIR, train_ids)
    train_loader = DataLoader(train_ds, batch_size=16, shuffle=True, drop_last=True)
    val_ds = PetManifestDataset(IMAGES_DIR, val_manifest)
    val_loader = DataLoader(val_ds, batch_size=16, shuffle=False)

    warmup_opt = torch.optim.Adam(model.gate.parameters(), lr=WARMUP_LR)
    warmup_train(model, train_loader, warmup_opt, loss_fn, device, temperature, epochs=warmup_epochs)

    joint_opt = torch.optim.Adam(model.parameters(), lr=best_params["lr"])
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    ckpt_path = CHECKPOINT_DIR / "moe_final.pt"

    best_val = float("inf")
    for epoch in range(joint_epochs):
        joint_finetune(model, train_loader, joint_opt, loss_fn, device, temperature, epochs=1)
        val_loss = evaluate_moe(model, val_loader, loss_fn, device, temperature)
        print(f"epoch {epoch+1}/{joint_epochs}  val_loss={val_loss:.4f}")
        if val_loss < best_val:
            best_val = val_loss
            torch.save(model.state_dict(), ckpt_path)
            print(f"  -> saved new best checkpoint to {ckpt_path}")

    routing = routing_weights_by_corruption(model, val_loader, device, temperature)
    health = check_expert_health(routing)
    print("\nRouting health check:")
    print(json.dumps(health, indent=2))

    return model


# ==========================================================================
# Smoke test -- fresh (randomly-initialized) stand-ins, proves plumbing only
# ==========================================================================

def smoke_test():
    set_seed(42)
    device = get_device()
    stems = list_clean_images(IMAGES_DIR)
    train_ids, val_ids = train_val_split(stems, seed=42)

    tiny_arch = dict(base_channels=16, bottleneck_spatial=8, dropout=0.1, activation="relu")
    gate = CorruptionClassifier(base_channels=16, dropout=0.2).to(device)
    expert_salt = UniversalAutoencoder(**tiny_arch).to(device)
    expert_blur = UniversalAutoencoder(**tiny_arch).to(device)
    expert_occ = UniversalAutoencoder(**tiny_arch).to(device)
    model = SoftMoE(gate, expert_salt, expert_blur, expert_occ)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"SoftMoE total params (tiny stand-in): {n_params:,}")
    print("NOTE: these are randomly-initialized stand-ins, NOT loaded from Task 2 -- "
          "this only proves the training/eval code paths are correct.")

    train_ds = PetTrainDataset(IMAGES_DIR, train_ids[:32])
    loader = DataLoader(train_ds, batch_size=8, shuffle=True, drop_last=True)
    loss_fn = MoELoss()

    warmup_opt = torch.optim.Adam(model.gate.parameters(), lr=WARMUP_LR)
    warmup_train(model, loader, warmup_opt, loss_fn, device, temperature=1.0, epochs=1)

    joint_opt = torch.optim.Adam(model.parameters(), lr=1e-5)
    joint_finetune(model, loader, joint_opt, loss_fn, device, temperature=1.0, epochs=1)

    val_manifest = generate_validation_manifest(val_ids[:16], seed=42)
    val_ds = PetManifestDataset(IMAGES_DIR, val_manifest)
    val_loader = DataLoader(val_ds, batch_size=8, shuffle=False)

    val_loss = evaluate_moe(model, val_loader, loss_fn, device)
    print(f"tiny val loss: {val_loss:.4f}")

    routing = routing_weights_by_corruption(model, val_loader, device)
    print("Routing weights by (corruption, severity):")
    for k, v in routing.items():
        print(f"  {k}: {np.round(v, 3)}")

    health = check_expert_health(routing)
    print("Expert health check:", json.dumps(health, indent=2))

    print("\nSmoke test passed.")


# ==========================================================================
# CLI
# ==========================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Task 3: Soft Mixture-of-Experts Restoration")
    parser.add_argument("mode", choices=["smoketest", "optuna", "train", "evaluate-routing"],
                         nargs="?", default="smoketest")
    parser.add_argument("--n-trials", type=int, default=20)
    parser.add_argument("--trial-epochs", type=int, default=3)
    parser.add_argument("--warmup-epochs", type=int, default=5)
    parser.add_argument("--joint-epochs", type=int, default=20)
    args = parser.parse_args()

    if args.mode == "smoketest":
        smoke_test()

    elif args.mode == "optuna":
        run_moe_optuna_study(n_trials=args.n_trials, max_epochs_per_trial=args.trial_epochs)

    elif args.mode == "train":
        if not BEST_PARAMS_PATH.exists():
            raise SystemExit(f"Run `optuna` first (missing {BEST_PARAMS_PATH}).")
        with open(BEST_PARAMS_PATH) as f:
            best_params = json.load(f)
        train_final_moe(best_params, warmup_epochs=args.warmup_epochs,
                         joint_epochs=args.joint_epochs)

    elif args.mode == "evaluate-routing":
        device = get_device()
        model = build_and_load_pretrained_moe(device)
        ckpt_path = CHECKPOINT_DIR / "moe_final.pt"
        if ckpt_path.exists():
            model.load_state_dict(torch.load(ckpt_path, map_location=device))
        else:
            print(f"WARNING: {ckpt_path} not found, evaluating warm-started model as-is.")

        stems = list_clean_images(IMAGES_DIR)
        _, val_ids = train_val_split(stems, seed=42)
        val_manifest = generate_validation_manifest(val_ids, seed=42)
        val_ds = PetManifestDataset(IMAGES_DIR, val_manifest)
        val_loader = DataLoader(val_ds, batch_size=32, shuffle=False)

        routing = routing_weights_by_corruption(model, val_loader, device)
        health = check_expert_health(routing)
        print(json.dumps(health, indent=2))
