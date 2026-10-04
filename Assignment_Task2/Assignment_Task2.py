"""
Assignment Task 2: Corruption Classification + Hard-Routed Specialist Autoencoders.

Reuses (not duplicates):
  - practise/data_pipeline.py  -- dataset, manifests, BalancedBatchSampler
  - practise/utils.py          -- SSIM/L1 loss, psnr, seeding
  - Assignment_Task1/Assignment_Task1.py -- UniversalAutoencoder + its exact
    conv_down_block building block, so the Task 2 classifier and specialists
    share the SAME architectural "shape" (GroupNorm, k3/s2/p1 strided conv,
    activation choice) as the finalized Task 1 design, not the stale
    BatchNorm/LeakyReLU-fixed version that used to live in practise/.

Three parts, matching the assignment:

  PART A -- CorruptionClassifier: conv stack (Task 1's down-block, reused)
            -> Global Average Pooling -> Linear(4). Balanced-batch trained.
            Optuna tunes: lr, batch_size, base_channels, dropout, weight_decay.
            (Activation is fixed to "relu" here -- the assignment's required
            classifier hyperparameter list doesn't include it, so it's kept
            out of the search to avoid an unrequested extra dimension.)

  PART B -- Specialists: literally UniversalAutoencoder from Task 1, ONE
            shared Optuna search (on pooled corrupted data) to pick a common
            architecture, then trained as 3 INDEPENDENT copies, each only on
            its own corruption type.

  PART C -- HardRoutedSystem: classifier predicts -> routes to matching
            specialist. Clean predictions bypass untouched (identity).
            Evaluated in two modes: oracle (true label routes) and predicted
            (classifier routes) -- the gap between them is the required
            evidence of classifier-error-caused restoration failure.
"""

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "practise"))
sys.path.insert(0, str(PROJECT_ROOT / "Assignment_Task1"))

from utils import ReconstructionLoss, get_device, psnr, set_seed  # noqa: E402
from tracking import Tracker  # noqa: E402
from data_pipeline import (  # noqa: E402
    CORRUPTION_TYPES, IMG_SIZE, PetTrainDataset, PetManifestDataset,
    BalancedBatchSampler, list_clean_images, train_val_split,
    generate_validation_manifest, generate_test_manifest,
    load_image_as_array, sample_corruption_params, apply_corruption, to_chw_tensor,
)
from Assignment_Task1 import (  # noqa: E402
    UniversalAutoencoder, conv_down_block, GROUP_NORM_GROUPS, IMAGES_DIR,
    train_one_epoch as ae_train_one_epoch, evaluate as ae_evaluate,
)

HERE = Path(__file__).resolve().parent
CHECKPOINT_DIR = HERE / "checkpoints"
CLASSIFIER_BEST_PARAMS_PATH = HERE / "classifier_best_params.json"
SPECIALIST_ARCH_PARAMS_PATH = HERE / "specialist_arch_params.json"
LOG_DIR = HERE / "logs"

# The four bottleneck options from the design discussion, and nothing else.
# name -> (bottleneck_spatial, base_channels). Channels at the bottleneck are
# base_channels * 2^(depth-1), so each option lands on exactly the named size:
#   8x8x128: depth 4, base 16 -> 16*8  = 128
#   8x8x256: depth 4, base 32 -> 32*8  = 256
#   4x4x128: depth 5, base 8  ->  8*16 = 128
#   4x4x256: depth 5, base 16 -> 16*16 = 256
BOTTLENECK_CONFIGS = {
    "8x8x128": (8, 16),
    "8x8x256": (8, 32),
    "4x4x128": (4, 8),
    "4x4x256": (4, 16),
}


# ==========================================================================
# PART A: Classifier
# ==========================================================================

class CorruptionClassifier(nn.Module):
    """Reuses Task 1's exact conv_down_block (GroupNorm + activation, k3/s2/p1)
    for consistency, then Global Average Pooling instead of flatten -- a 4-way
    "which corruption is present" decision doesn't need positional detail,
    just "is this pattern present anywhere", which GAP captures with far
    fewer parameters. Fixed depth=4 (128->64->32->16->8); unlike the
    autoencoder, the classifier has no bottleneck-size requirement to search."""

    def __init__(self, base_channels: int = 32, dropout: float = 0.3,
                 num_classes: int = 4, activation: str = "relu"):
        super().__init__()
        channels = [base_channels * (2 ** i) for i in range(4)]
        layers = []
        in_ch = 3
        for out_ch in channels:
            layers.append(conv_down_block(in_ch, out_ch, activation))
            in_ch = out_ch
        self.conv = nn.Sequential(*layers)
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(channels[-1], num_classes)
        self.feature_dim = channels[-1]

    def features(self, x):
        """Exposes pooled pre-logit features -- useful if Task 3's gating
        network wants to reuse this backbone."""
        feat = self.conv(x)
        return self.gap(feat).flatten(1)

    def forward(self, x):
        pooled = self.features(x)
        return self.fc(self.dropout(pooled))


def train_classifier_one_epoch(model, dataset, sampler, device, optimizer, ce_loss):
    model.train()
    loader = DataLoader(dataset, batch_sampler=sampler)
    total_loss, n = 0.0, 0
    for batch in loader:
        x = batch["corrupted"].to(device)
        y = torch.as_tensor(batch["label"]).to(device)
        optimizer.zero_grad()
        logits = model(x)
        loss = ce_loss(logits, y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * x.size(0)
        n += x.size(0)
    return total_loss / max(n, 1)


@torch.no_grad()
def evaluate_classifier(model, loader, device):
    """Accuracy, macro P/R/F1, per-class metrics, normalized confusion
    matrix -- computed manually, no sklearn dependency needed."""
    model.eval()
    num_classes = 4
    confusion = np.zeros((num_classes, num_classes), dtype=np.int64)
    for batch in loader:
        x = batch["corrupted"].to(device)
        y = torch.as_tensor(batch["label"]).numpy()
        preds = model(x).argmax(dim=1).cpu().numpy()
        for t, p in zip(y, preds):
            confusion[t, p] += 1

    accuracy = np.trace(confusion) / max(confusion.sum(), 1)
    precisions, recalls, f1s = [], [], []
    for c in range(num_classes):
        tp = confusion[c, c]
        fp = confusion[:, c].sum() - tp
        fn = confusion[c, :].sum() - tp
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        precisions.append(precision)
        recalls.append(recall)
        f1s.append(f1)

    normalized_confusion = confusion / np.maximum(confusion.sum(axis=1, keepdims=True), 1)

    return {
        "accuracy": float(accuracy),
        "macro_precision": float(np.mean(precisions)),
        "macro_recall": float(np.mean(recalls)),
        "macro_f1": float(np.mean(f1s)),
        "per_class": {CORRUPTION_TYPES[c]: {"precision": precisions[c], "recall": recalls[c],
                                             "f1": f1s[c]} for c in range(num_classes)},
        "confusion_matrix": confusion.tolist(),
        "confusion_matrix_normalized": normalized_confusion.tolist(),
    }


def classifier_objective(images_dir, train_ids, val_manifest, device, max_epochs=3):
    def objective(trial):
        lr = trial.suggest_categorical("lr", [0.01, 0.003, 0.001, 0.0003, 0.0001])
        batch_size = trial.suggest_categorical("batch_size", [8, 16, 32, 64])
        base_channels = trial.suggest_categorical("base_channels", [16, 32, 64])
        dropout = trial.suggest_float("dropout", 0.0, 0.5)
        weight_decay = trial.suggest_float("weight_decay", 1e-6, 1e-3, log=True)

        model = CorruptionClassifier(base_channels, dropout).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
        ce_loss = nn.CrossEntropyLoss()

        train_ds = PetTrainDataset(images_dir, train_ids)
        sampler = BalancedBatchSampler(train_ds, batch_size=batch_size)
        val_ds = PetManifestDataset(images_dir, val_manifest)
        val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

        best_acc = 0.0
        for epoch in range(max_epochs):
            sampler.resample_epoch(epoch)
            train_classifier_one_epoch(model, train_ds, sampler, device, optimizer, ce_loss)
            metrics = evaluate_classifier(model, val_loader, device)
            best_acc = max(best_acc, metrics["accuracy"])

            trial.report(metrics["accuracy"], epoch)
            if trial.should_prune():
                import optuna
                raise optuna.TrialPruned()
        return best_acc

    return objective


def run_classifier_optuna_study(n_trials: int = 30, max_epochs_per_trial: int = 3):
    import optuna

    set_seed(42)
    device = get_device()
    stems = list_clean_images(IMAGES_DIR)
    train_ids, val_ids = train_val_split(stems, seed=42)
    val_manifest = generate_validation_manifest(val_ids, seed=42)

    study = optuna.create_study(
        study_name="task2_classifier", storage=f"sqlite:///{HERE / 'optuna_study.db'}",
        load_if_exists=True, direction="maximize", pruner=optuna.pruners.MedianPruner(),
    )
    objective = classifier_objective(IMAGES_DIR, train_ids, val_manifest, device,
                                      max_epochs_per_trial)
    study.optimize(objective, n_trials=n_trials)

    print("\n=== Classifier Optuna study complete ===")
    print(f"Trials run: {len(study.trials)}")
    print(f"Best trial: #{study.best_trial.number}  best val accuracy: {study.best_value:.4f}")
    print(f"Best params: {study.best_params}")

    with open(CLASSIFIER_BEST_PARAMS_PATH, "w") as f:
        json.dump(study.best_params, f, indent=2)
    print(f"Saved to {CLASSIFIER_BEST_PARAMS_PATH}")
    return study


def train_final_classifier(best_params: dict, epochs: int = 30):
    set_seed(42)
    device = get_device()
    stems = list_clean_images(IMAGES_DIR)
    train_ids, val_ids = train_val_split(stems, seed=42)
    val_manifest = generate_validation_manifest(val_ids, seed=42)

    model = CorruptionClassifier(best_params["base_channels"], best_params["dropout"]).to(device)
    print(f"Classifier params: {sum(p.numel() for p in model.parameters()):,}")
    optimizer = torch.optim.Adam(model.parameters(), lr=best_params["lr"],
                                  weight_decay=best_params["weight_decay"])
    ce_loss = nn.CrossEntropyLoss()

    train_ds = PetTrainDataset(IMAGES_DIR, train_ids)
    sampler = BalancedBatchSampler(train_ds, batch_size=best_params["batch_size"])
    val_ds = PetManifestDataset(IMAGES_DIR, val_manifest)
    val_loader = DataLoader(val_ds, batch_size=best_params["batch_size"], shuffle=False)

    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    ckpt_path = CHECKPOINT_DIR / "classifier_final.pt"

    LOG_DIR.mkdir(parents=True, exist_ok=True)
    tracker = Tracker("task2_classifier", "final_train", {**best_params, "epochs": epochs})
    log_path = LOG_DIR / "classifier_train_log.csv"
    best_acc = 0.0
    with open(log_path, "w", newline="") as log_file:
        writer = csv.writer(log_file)
        writer.writerow(["epoch", "train_loss", "val_accuracy", "val_macro_f1"])
        for epoch in range(epochs):
            sampler.resample_epoch(epoch)
            train_loss = train_classifier_one_epoch(model, train_ds, sampler, device,
                                                      optimizer, ce_loss)
            metrics = evaluate_classifier(model, val_loader, device)
            writer.writerow([epoch + 1, train_loss, metrics["accuracy"], metrics["macro_f1"]])
            log_file.flush()
            tracker.log_metrics({"train_loss": train_loss, "val_accuracy": metrics["accuracy"],
                                 "val_macro_f1": metrics["macro_f1"]}, step=epoch + 1)
            print(f"epoch {epoch+1}/{epochs}  train_loss={train_loss:.4f}  "
                  f"val_acc={metrics['accuracy']:.4f}  val_macro_f1={metrics['macro_f1']:.4f}")
            if metrics["accuracy"] > best_acc:
                best_acc = metrics["accuracy"]
                torch.save(model.state_dict(), ckpt_path)
                print(f"  -> saved new best checkpoint to {ckpt_path}")
    print(f"Training log written to {log_path}")
    tracker.log_artifact(ckpt_path)
    tracker.close()

    print("\nFinal validation metrics (best checkpoint):")
    print(json.dumps(metrics, indent=2))
    return model


# ==========================================================================
# PART B: Specialists (reuse UniversalAutoencoder, filtered per-corruption data)
# ==========================================================================

class SingleCorruptionDataset(Dataset):
    """Always applies ONE fixed corruption type (randomized severity) --
    used to train/search one specialist."""

    def __init__(self, images_dir, stems, corruption: str, img_size=IMG_SIZE):
        self.images_dir = images_dir
        self.stems = stems
        self.corruption = corruption
        self.img_size = img_size

    def __len__(self):
        return len(self.stems)

    def __getitem__(self, idx):
        stem = self.stems[idx]
        clean = load_image_as_array(self.images_dir, stem, self.img_size)
        rng = np.random.default_rng()
        params = sample_corruption_params(self.corruption, rng)
        corrupted = apply_corruption(clean, self.corruption, params, rng)
        return {"corrupted": to_chw_tensor(corrupted), "clean": to_chw_tensor(clean)}


class PooledCorruptionDataset(Dataset):
    """For the SHARED architecture search only: each item is corrupted with
    one of the 3 non-clean types, chosen uniformly at random, so the shared
    Optuna search isn't biased toward any single corruption type."""

    NON_CLEAN = ["salt_pepper", "blur", "occlusion"]

    def __init__(self, images_dir, stems, img_size=IMG_SIZE):
        self.images_dir = images_dir
        self.stems = stems
        self.img_size = img_size

    def __len__(self):
        return len(self.stems)

    def __getitem__(self, idx):
        stem = self.stems[idx]
        clean = load_image_as_array(self.images_dir, stem, self.img_size)
        rng = np.random.default_rng()
        corruption = self.NON_CLEAN[int(rng.integers(0, 3))]
        params = sample_corruption_params(corruption, rng)
        corrupted = apply_corruption(clean, corruption, params, rng)
        return {"corrupted": to_chw_tensor(corrupted), "clean": to_chw_tensor(clean)}


def specialist_shared_objective(images_dir, train_ids, val_stems, device, max_epochs=3):
    """Searches architecture + training hyperparams on POOLED (any-corruption)
    data to find one shared config, per the assignment's suggested shortcut."""

    def objective(trial):
        lr = trial.suggest_categorical("lr", [0.01, 0.003, 0.001, 0.0003, 0.0001])
        batch_size = trial.suggest_categorical("batch_size", [8, 16, 32, 64])
        bottleneck = trial.suggest_categorical("bottleneck", list(BOTTLENECK_CONFIGS))
        bottleneck_spatial, base_channels = BOTTLENECK_CONFIGS[bottleneck]
        dropout = trial.suggest_float("dropout", 0.0, 0.3)
        alpha = trial.suggest_float("alpha", 0.5, 0.95)
        activation = trial.suggest_categorical("activation", ["relu", "leaky_relu"])

        model = UniversalAutoencoder(base_channels, bottleneck_spatial, dropout,
                                      activation).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
        loss_fn = ReconstructionLoss(alpha=alpha)

        train_ds = PooledCorruptionDataset(images_dir, train_ids)
        train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, drop_last=True)
        val_ds = PooledCorruptionDataset(images_dir, val_stems)
        val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

        best_val = float("inf")
        for epoch in range(max_epochs):
            ae_train_one_epoch(model, train_loader, optimizer, loss_fn, device)
            val_loss, _ = ae_evaluate(model, val_loader, loss_fn, device)
            best_val = min(best_val, val_loss)

            trial.report(val_loss, epoch)
            if trial.should_prune():
                import optuna
                raise optuna.TrialPruned()
        return best_val

    return objective


def run_specialist_shared_optuna_study(n_trials: int = 20, max_epochs_per_trial: int = 3):
    import optuna

    set_seed(42)
    device = get_device()
    stems = list_clean_images(IMAGES_DIR)
    train_ids, val_ids = train_val_split(stems, seed=42)

    study = optuna.create_study(
        study_name="task2_specialist_shared", storage=f"sqlite:///{HERE / 'optuna_study.db'}",
        load_if_exists=True, direction="minimize", pruner=optuna.pruners.MedianPruner(),
    )
    objective = specialist_shared_objective(IMAGES_DIR, train_ids, val_ids, device,
                                             max_epochs_per_trial)
    study.optimize(objective, n_trials=n_trials)

    print("\n=== Specialist shared-architecture Optuna study complete ===")
    print(f"Trials run: {len(study.trials)}")
    print(f"Best trial: #{study.best_trial.number}  best val loss: {study.best_value:.4f}")
    print(f"Best params: {study.best_params}")

    # Expand the named bottleneck into the keys the rest of the code reads
    # (train_specialist, evaluate-routing, and Task 3 all use these names).
    best = dict(study.best_params)
    spatial, base = BOTTLENECK_CONFIGS[best.pop("bottleneck")]
    best.update({"bottleneck_spatial": spatial, "base_channels": base})
    with open(SPECIALIST_ARCH_PARAMS_PATH, "w") as f:
        json.dump(best, f, indent=2)
    print(f"Saved to {SPECIALIST_ARCH_PARAMS_PATH}")
    return study


def train_specialist(corruption: str, arch_params: dict, epochs: int = 30):
    """Trains ONE specialist, independently, only on `corruption`-corrupted inputs."""
    assert corruption in ("salt_pepper", "blur", "occlusion")
    set_seed(42)
    device = get_device()
    stems = list_clean_images(IMAGES_DIR)
    train_ids, val_ids = train_val_split(stems, seed=42)

    model = UniversalAutoencoder(
        base_channels=arch_params["base_channels"],
        bottleneck_spatial=arch_params["bottleneck_spatial"],
        dropout=arch_params["dropout"],
        activation=arch_params["activation"],
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=arch_params["lr"], weight_decay=1e-5)
    loss_fn = ReconstructionLoss(alpha=arch_params["alpha"])

    train_ds = SingleCorruptionDataset(IMAGES_DIR, train_ids, corruption)
    train_loader = DataLoader(train_ds, batch_size=arch_params["batch_size"], shuffle=True)
    val_ds = SingleCorruptionDataset(IMAGES_DIR, val_ids, corruption)
    val_loader = DataLoader(val_ds, batch_size=arch_params["batch_size"], shuffle=False)

    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    ckpt_path = CHECKPOINT_DIR / f"specialist_{corruption}.pt"

    LOG_DIR.mkdir(parents=True, exist_ok=True)
    tracker = Tracker("task2_specialists", f"specialist_{corruption}", {**arch_params, "epochs": epochs})
    log_path = LOG_DIR / f"specialist_{corruption}_train_log.csv"
    best_val = float("inf")
    with open(log_path, "w", newline="") as log_file:
        writer = csv.writer(log_file)
        writer.writerow(["epoch", "train_loss", "val_loss", "val_psnr"])
        for epoch in range(epochs):
            train_loss = ae_train_one_epoch(model, train_loader, optimizer, loss_fn, device)
            val_loss, val_psnr = ae_evaluate(model, val_loader, loss_fn, device)
            writer.writerow([epoch + 1, train_loss, val_loss, val_psnr])
            log_file.flush()
            tracker.log_metrics({"train_loss": train_loss, "val_loss": val_loss,
                                 "val_psnr": val_psnr}, step=epoch + 1)
            print(f"[{corruption}] epoch {epoch+1}/{epochs}  train_loss={train_loss:.4f}  "
                  f"val_loss={val_loss:.4f}  val_psnr={val_psnr:.2f}dB")
            if val_loss < best_val:
                best_val = val_loss
                torch.save(model.state_dict(), ckpt_path)
    print(f"Training log written to {log_path}")
    tracker.log_artifact(ckpt_path)
    tracker.close()

    return model


def train_all_specialists(arch_params: dict, epochs: int = 30):
    models = {}
    for corruption in ["salt_pepper", "blur", "occlusion"]:
        print(f"\n=== Training specialist: {corruption} ===")
        models[corruption] = train_specialist(corruption, arch_params, epochs)
    return models


# ==========================================================================
# PART C: Hard routing + oracle vs predicted evaluation
# ==========================================================================

class HardRoutedSystem:
    def __init__(self, classifier: CorruptionClassifier, specialists: dict, device):
        self.classifier = classifier.to(device).eval()
        self.specialists = {k: v.to(device).eval() for k, v in specialists.items()}
        self.device = device

    @torch.no_grad()
    def restore(self, x_tilde: torch.Tensor, true_label: torch.Tensor = None,
                mode: str = "predicted"):
        x_tilde = x_tilde.to(self.device)
        if mode == "oracle":
            assert true_label is not None
            route = true_label.to(self.device)
            probs = None
        else:
            logits = self.classifier(x_tilde)
            probs = torch.softmax(logits, dim=1)
            route = probs.argmax(dim=1)

        out = x_tilde.clone()
        for label_idx, corruption in enumerate(CORRUPTION_TYPES):
            if corruption == "clean":
                continue
            mask = (route == label_idx)
            if mask.any():
                out[mask] = self.specialists[corruption](x_tilde[mask])
        return out, route, probs


@torch.no_grad()
def evaluate_routing(system: HardRoutedSystem, test_manifest, device, mode: str,
                      batch_size: int = 32):
    """Breaks down L1 error and PSNR by (corruption, severity) -- the exact
    table shape the report needs. Also returns classifier accuracy on the
    test set when mode='predicted'."""
    ds = PetManifestDataset(IMAGES_DIR, test_manifest)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False)

    results = {}  # (corruption, severity) -> list of (l1, psnr_val)
    n_correct, n_total = 0, 0

    for batch in loader:
        x = batch["corrupted"].to(device)
        y = batch["clean"].to(device)
        true_label = torch.as_tensor(batch["label"])

        out, route, probs = system.restore(x, true_label=true_label, mode=mode)

        if mode == "predicted":
            n_correct += (route.cpu() == true_label).sum().item()
            n_total += len(true_label)

        l1_per_sample = (out - y).abs().mean(dim=[1, 2, 3]).cpu().numpy()
        for i in range(len(l1_per_sample)):
            key = (batch["corruption"][i], batch["severity"][i])
            sample_psnr = psnr(out[i:i+1], y[i:i+1])
            results.setdefault(key, []).append((float(l1_per_sample[i]), sample_psnr))

    summary = {}
    for key, vals in results.items():
        l1s = [v[0] for v in vals]
        psnrs = [v[1] for v in vals]
        summary[key] = {"mean_l1": float(np.mean(l1s)), "mean_psnr": float(np.mean(psnrs)),
                         "n": len(vals)}

    out = {"per_corruption_severity": summary}
    if mode == "predicted":
        out["classifier_accuracy_on_test"] = n_correct / max(n_total, 1)
    return out


@torch.no_grad()
def find_routing_failure_cases(system: HardRoutedSystem, test_manifest, device, n: int = 4):
    """Finds examples where the classifier misrouted AND that misrouting
    measurably hurt reconstruction (predicted error notably worse than
    oracle error on the SAME image) -- exactly the evidence the report needs."""
    ds = PetManifestDataset(IMAGES_DIR, test_manifest)
    loader = DataLoader(ds, batch_size=16, shuffle=False)

    failures = []
    for batch in loader:
        x = batch["corrupted"].to(device)
        y = batch["clean"].to(device)
        true_label = torch.as_tensor(batch["label"])

        out_oracle, _, _ = system.restore(x, true_label=true_label, mode="oracle")
        out_pred, route_pred, _ = system.restore(x, mode="predicted")

        oracle_err = (out_oracle - y).abs().mean(dim=[1, 2, 3])
        pred_err = (out_pred - y).abs().mean(dim=[1, 2, 3])
        degradation = (pred_err - oracle_err).cpu().numpy()
        misrouted = (route_pred.cpu() != true_label).numpy()

        for i in range(len(misrouted)):
            if misrouted[i] and degradation[i] > 0:
                failures.append({
                    "stem": batch["stem"][i],
                    "true_corruption": batch["corruption"][i],
                    "severity": batch["severity"][i],
                    "predicted_class": CORRUPTION_TYPES[route_pred[i].item()],
                    "oracle_l1": float(oracle_err[i]),
                    "predicted_l1": float(pred_err[i]),
                    "degradation": float(degradation[i]),
                })
        if len(failures) >= n * 3:  # gather a few extra, then pick the worst
            break

    failures.sort(key=lambda f: -f["degradation"])
    return failures[:n]


# ==========================================================================
# Smoke test
# ==========================================================================

def smoke_test():
    set_seed(42)
    device = get_device()
    stems = list_clean_images(IMAGES_DIR)
    train_ids, val_ids = train_val_split(stems, seed=42)

    print("--- Classifier ---")
    clf = CorruptionClassifier(base_channels=16, dropout=0.2).to(device)
    print(f"params={sum(p.numel() for p in clf.parameters()):,}")
    tiny_train = PetTrainDataset(IMAGES_DIR, train_ids[:32])
    sampler = BalancedBatchSampler(tiny_train, batch_size=8)
    optimizer = torch.optim.Adam(clf.parameters(), lr=1e-3)
    ce_loss = nn.CrossEntropyLoss()
    loss = train_classifier_one_epoch(clf, tiny_train, sampler, device, optimizer, ce_loss)
    print(f"training loss after 1 tiny epoch: {loss:.4f}")

    val_manifest = generate_validation_manifest(val_ids[:16], seed=42)
    val_ds = PetManifestDataset(IMAGES_DIR, val_manifest)
    val_loader = DataLoader(val_ds, batch_size=8, shuffle=False)
    metrics = evaluate_classifier(clf, val_loader, device)
    print(f"val accuracy: {metrics['accuracy']:.3f}")
    print(f"confusion matrix: {metrics['confusion_matrix']}")

    print("\n--- Specialists (tiny, 1 epoch each) ---")
    tiny_arch = {"base_channels": 16, "bottleneck_spatial": 8, "dropout": 0.1,
                 "lr": 1e-3, "alpha": 0.8, "batch_size": 8, "activation": "relu"}
    specialists = {}
    for corruption in ["salt_pepper", "blur", "occlusion"]:
        model = UniversalAutoencoder(tiny_arch["base_channels"], tiny_arch["bottleneck_spatial"],
                                      tiny_arch["dropout"], tiny_arch["activation"]).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=tiny_arch["lr"])
        loss_fn = ReconstructionLoss(alpha=tiny_arch["alpha"])
        ds = SingleCorruptionDataset(IMAGES_DIR, train_ids[:16], corruption)
        loader = DataLoader(ds, batch_size=8, shuffle=True)
        loss = ae_train_one_epoch(model, loader, optimizer, loss_fn, device)
        print(f"[{corruption}] loss after 1 tiny epoch: {loss:.4f}")
        specialists[corruption] = model

    print("\n--- Hard routing ---")
    system = HardRoutedSystem(clf, specialists, device)
    test_manifest = generate_test_manifest(val_ids[:2], seed=42)
    oracle_results = evaluate_routing(system, test_manifest, device, mode="oracle", batch_size=8)
    predicted_results = evaluate_routing(system, test_manifest, device, mode="predicted",
                                          batch_size=8)
    print("Oracle routing (sample):", list(oracle_results["per_corruption_severity"].items())[:2])
    print("Predicted routing classifier accuracy on tiny test:",
          predicted_results["classifier_accuracy_on_test"])

    failures = find_routing_failure_cases(system, test_manifest, device, n=2)
    print(f"Found {len(failures)} failure-case candidates (tiny test, may be 0 -- expected,"
          f" classifier is untrained).")

    print("\nSmoke test passed.")


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Task 2: Classifier + Hard-Routed Specialists")
    parser.add_argument("mode", choices=[
        "smoketest", "classifier-optuna", "classifier-train",
        "specialist-optuna", "specialist-train", "evaluate-routing",
    ], nargs="?", default="smoketest")
    parser.add_argument("--n-trials", type=int, default=30)
    parser.add_argument("--trial-epochs", type=int, default=3)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--corruption", type=str, default="all",
                         choices=["all", "salt_pepper", "blur", "occlusion"])
    args = parser.parse_args()

    if args.mode == "smoketest":
        smoke_test()

    elif args.mode == "classifier-optuna":
        run_classifier_optuna_study(n_trials=args.n_trials, max_epochs_per_trial=args.trial_epochs)

    elif args.mode == "classifier-train":
        if not CLASSIFIER_BEST_PARAMS_PATH.exists():
            raise SystemExit(f"Run `classifier-optuna` first (missing {CLASSIFIER_BEST_PARAMS_PATH}).")
        with open(CLASSIFIER_BEST_PARAMS_PATH) as f:
            best_params = json.load(f)
        train_final_classifier(best_params, epochs=args.epochs)

    elif args.mode == "specialist-optuna":
        run_specialist_shared_optuna_study(n_trials=args.n_trials,
                                            max_epochs_per_trial=args.trial_epochs)

    elif args.mode == "specialist-train":
        if not SPECIALIST_ARCH_PARAMS_PATH.exists():
            raise SystemExit(f"Run `specialist-optuna` first (missing {SPECIALIST_ARCH_PARAMS_PATH}).")
        with open(SPECIALIST_ARCH_PARAMS_PATH) as f:
            arch_params = json.load(f)
        if args.corruption == "all":
            train_all_specialists(arch_params, epochs=args.epochs)
        else:
            train_specialist(args.corruption, arch_params, epochs=args.epochs)

    elif args.mode == "evaluate-routing":
        device = get_device()
        clf = CorruptionClassifier().to(device)
        clf.load_state_dict(torch.load(CHECKPOINT_DIR / "classifier_final.pt",
                                        map_location=device))
        with open(SPECIALIST_ARCH_PARAMS_PATH) as f:
            arch_params = json.load(f)
        specialists = {}
        for corruption in ["salt_pepper", "blur", "occlusion"]:
            m = UniversalAutoencoder(arch_params["base_channels"], arch_params["bottleneck_spatial"],
                                      arch_params["dropout"], arch_params["activation"]).to(device)
            m.load_state_dict(torch.load(CHECKPOINT_DIR / f"specialist_{corruption}.pt",
                                          map_location=device))
            specialists[corruption] = m

        system = HardRoutedSystem(clf, specialists, device)
        stems = list_clean_images(IMAGES_DIR)
        _, val_ids = train_val_split(stems, seed=42)
        test_manifest = generate_test_manifest(val_ids, seed=42)  # placeholder: swap for real held-out test set

        oracle = evaluate_routing(system, test_manifest, device, mode="oracle")
        predicted = evaluate_routing(system, test_manifest, device, mode="predicted")
        print("=== Oracle routing ===")
        print(json.dumps(oracle, indent=2))
        print("=== Predicted routing ===")
        print(json.dumps(predicted, indent=2))

        failures = find_routing_failure_cases(system, test_manifest, device, n=4)
        print("=== Top failure cases ===")
        print(json.dumps(failures, indent=2))
