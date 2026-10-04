"""
Assignment Task 4: Style-conditioned face-to-sketch cGAN (FS2K).

Architecture decisions (all documented choices, see notes):
  - Style conditioning A: a learned embedding for the 3 styles, broadcast
    spatially and concatenated to the input of BOTH networks.
  - Generator: U-Net with 5 downsampling stages (128 -> 4), skip connections,
    upsample+conv decoder (no checkerboard artifacts), InstanceNorm, dropout
    in the first three decoder blocks, tanh output on [-1, 1].
  - Discriminator: PatchGAN with two stride-2 layers and two stride-1 layers,
    receptive field about 34 px at 128 px (matches pix2pix's relative patch size).
  - Loss: BCE-with-logits adversarial loss, plus lambda_l1 * L1 for the generator.

Data: fs2k_data.py (official split, 15% stratified validation, paired augmentation).
"""

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader

HERE = Path(__file__).resolve().parent
PROJECT_ROOT = HERE.parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(PROJECT_ROOT / "practise"))

from utils import get_device, set_seed, ssim  # noqa: E402
from tracking import Tracker  # noqa: E402
from fs2k_data import IMG_SIZE, NUM_STYLES, FS2KPairDataset, make_splits  # noqa: E402

CHECKPOINT_DIR = HERE / "checkpoints"
LOG_DIR = HERE / "logs"
SAMPLE_DIR = HERE / "samples"
BEST_PARAMS_PATH = HERE / "best_params.json"
MAX_CHANNELS = 512
N_FIXED_SAMPLES = 4


# ==========================================================================
# Style conditioning (option A)
# ==========================================================================

def broadcast_style(style_emb: nn.Embedding, style: torch.Tensor, h: int, w: int):
    """(B,) style ids -> (B, E, H, W) learned embedding repeated over the image."""
    vec = style_emb(style)                       # (B, E)
    return vec[:, :, None, None].expand(-1, -1, h, w)


# ==========================================================================
# Generator: U-Net
# ==========================================================================

def down_block(in_c: int, out_c: int, norm: bool = True) -> nn.Sequential:
    layers = [nn.Conv2d(in_c, out_c, kernel_size=4, stride=2, padding=1)]
    if norm:
        layers.append(nn.InstanceNorm2d(out_c, affine=True))
    layers.append(nn.LeakyReLU(0.2, inplace=True))
    return nn.Sequential(*layers)


def up_block(in_c: int, out_c: int, dropout: float = 0.0) -> nn.Sequential:
    layers = [
        nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
        nn.Conv2d(in_c, out_c, kernel_size=3, stride=1, padding=1),
        nn.InstanceNorm2d(out_c, affine=True),
        nn.ReLU(inplace=True),
    ]
    if dropout > 0:
        layers.append(nn.Dropout(dropout))
    return nn.Sequential(*layers)


class UNetGenerator(nn.Module):
    """G(photo, style) -> sketch. Channel widths: base, 2*base, 4*base, 8*base, 8*base
    (capped at MAX_CHANNELS). Spatial sizes: 64, 32, 16, 8, 4 for the encoder."""

    def __init__(self, base_channels: int = 64, dropout: float = 0.5, style_dim: int = 16,
                 conditioning: str = "input"):
        """conditioning='input'  : style broadcast and concatenated to the photo (option A)
           conditioning='bottleneck': style broadcast and concatenated at the U-Net
                                       bottleneck only (option B, for comparison)."""
        super().__init__()
        assert conditioning in ("input", "bottleneck"), conditioning
        self.conditioning = conditioning
        self.style_emb = nn.Embedding(NUM_STYLES, style_dim)
        ch = [min(base_channels * (2 ** i), MAX_CHANNELS) for i in range(5)]
        in_style = style_dim if conditioning == "input" else 0
        bottleneck_style = style_dim if conditioning == "bottleneck" else 0

        self.downs = nn.ModuleList([
            down_block(3 + in_style, ch[0], norm=False),   # 128 -> 64
            down_block(ch[0], ch[1]),                       # 64  -> 32
            down_block(ch[1], ch[2]),                       # 32  -> 16
            down_block(ch[2], ch[3]),                       # 16  -> 8
            down_block(ch[3], ch[4]),                       # 8   -> 4 (bottleneck)
        ])

        # Decoder: upsample, conv, then concatenate the encoder skip of the same size.
        self.ups = nn.ModuleList()
        cur = ch[4] + bottleneck_style
        for i in [3, 2, 1, 0]:
            drop = dropout if i >= 1 else 0.0  # dropout in the first three decoder blocks
            self.ups.append(up_block(cur, ch[i], drop))
            cur = ch[i] * 2                    # after concatenating skip d_i
        self.final = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),  # 64 -> 128
            nn.Conv2d(cur, 3, kernel_size=3, stride=1, padding=1),
            nn.Tanh(),
        )

    def forward(self, photo: torch.Tensor, style: torch.Tensor) -> torch.Tensor:
        _, _, h, w = photo.shape
        if self.conditioning == "input":
            s = broadcast_style(self.style_emb, style, h, w)
            x = torch.cat([photo, s], dim=1)
        else:
            x = photo

        skips = []
        for down in self.downs:
            x = down(x)
            skips.append(x)
        x = skips[4]                            # bottleneck, 4x4
        if self.conditioning == "bottleneck":
            s4 = broadcast_style(self.style_emb, style, x.size(2), x.size(3))
            x = torch.cat([x, s4], dim=1)

        for up, i in zip(self.ups, [3, 2, 1, 0]):
            x = up(x)
            x = torch.cat([x, skips[i]], dim=1)  # U-Net skip

        return self.final(x)


# ==========================================================================
# Discriminator: PatchGAN with style conditioning
# ==========================================================================

class PatchDiscriminator(nn.Module):
    """D(photo, sketch, style) -> 30x30 map of real/fake logits at 128 px.
    Each output value sees a 34x34 patch of the input pair."""

    def __init__(self, base_channels: int = 64, style_dim: int = 16):
        super().__init__()
        self.style_emb = nn.Embedding(NUM_STYLES, style_dim)
        c = base_channels
        self.net = nn.Sequential(
            nn.Conv2d(3 + 3 + style_dim, c, 4, 2, 1), nn.LeakyReLU(0.2, inplace=True),        # 128 -> 64
            nn.Conv2d(c, 2 * c, 4, 2, 1), nn.InstanceNorm2d(2 * c, affine=True),
            nn.LeakyReLU(0.2, inplace=True),                                                   # 64 -> 32
            nn.Conv2d(2 * c, 4 * c, 4, 1, 1), nn.InstanceNorm2d(4 * c, affine=True),
            nn.LeakyReLU(0.2, inplace=True),                                                   # 32 -> 31
            nn.Conv2d(4 * c, 1, 4, 1, 1),                                                      # 31 -> 30
        )

    def forward(self, photo, sketch, style):
        _, _, h, w = photo.shape
        s = broadcast_style(self.style_emb, style, h, w)
        return self.net(torch.cat([photo, sketch, s], dim=1))


# ==========================================================================
# Training
# ==========================================================================

bce = nn.BCEWithLogitsLoss()


def train_step(G, D, batch, opt_g, opt_d, lambda_l1: float, device):
    """One adversarial step. Images are scaled to [-1, 1] to match tanh."""
    photo = batch["photo"].to(device) * 2 - 1
    sketch = batch["sketch"].to(device) * 2 - 1
    style = batch["style"].to(device).long()

    # --- Discriminator: real pairs -> 1, generated pairs -> 0
    with torch.no_grad():
        fake = G(photo, style)
    real_logits = D(photo, sketch, style)
    fake_logits = D(photo, fake, style)
    d_real = bce(real_logits, torch.ones_like(real_logits))
    d_fake = bce(fake_logits, torch.zeros_like(fake_logits))
    d_loss = 0.5 * (d_real + d_fake)
    opt_d.zero_grad()
    d_loss.backward()
    opt_d.step()

    # --- Generator: fool D, and stay close to the true sketch
    fake = G(photo, style)
    fake_logits = D(photo, fake, style)
    g_adv = bce(fake_logits, torch.ones_like(fake_logits))
    g_l1 = F.l1_loss(fake, sketch)
    g_loss = g_adv + lambda_l1 * g_l1
    opt_g.zero_grad()
    g_loss.backward()
    opt_g.step()

    return {"d_real": d_real.item(), "d_fake": d_fake.item(),
            "g_adv": g_adv.item(), "g_l1": g_l1.item()}


@torch.no_grad()
def evaluate(G, loader, device):
    """Validation L1 and SSIM against the true sketch, both in [0, 1]."""
    G.eval()
    total_l1, total_ssim, n = 0.0, 0.0, 0
    for batch in loader:
        photo = batch["photo"].to(device) * 2 - 1
        target = batch["sketch"].to(device)
        style = batch["style"].to(device).long()
        out = (G(photo, style) + 1) / 2
        k = photo.size(0)
        total_l1 += F.l1_loss(out, target, reduction="sum").item() / (3 * IMG_SIZE * IMG_SIZE)
        total_ssim += ssim(out, target).item() * k
        n += k
    G.train()
    return total_l1 / max(n, 1), total_ssim / max(n, 1)


def make_loaders(batch_size: int):
    train_pairs, val_pairs, _ = make_splits()
    train_loader = DataLoader(FS2KPairDataset(train_pairs, augment=True),
                              batch_size=batch_size, shuffle=True, drop_last=True)
    val_loader = DataLoader(FS2KPairDataset(val_pairs, augment=False),
                            batch_size=batch_size, shuffle=False)
    return train_loader, val_loader, val_pairs


def variant_tag(params: dict) -> str:
    """File suffix per conditioning option: option A keeps the original names."""
    return "" if params.get("conditioning", "input") == "input" else "_" + params["conditioning"]


def build_models(params: dict, device):
    G = UNetGenerator(params["base_channels"], params["dropout"], params["style_dim"],
                      params.get("conditioning", "input")).to(device)
    D = PatchDiscriminator(params["base_channels"], params["style_dim"]).to(device)
    return G, D


# ==========================================================================
# Optuna
# ==========================================================================

def objective_factory(max_epochs: int):
    def objective(trial):
        import optuna

        params = {
            "lr_g": trial.suggest_float("lr_g", 1e-4, 4e-4, log=True),
            "lr_d": trial.suggest_float("lr_d", 1e-4, 4e-4, log=True),
            "batch_size": trial.suggest_categorical("batch_size", [8, 16, 32]),
            "base_channels": trial.suggest_categorical("base_channels", [32, 64]),
            "dropout": trial.suggest_float("dropout", 0.0, 0.5),
            "style_dim": trial.suggest_categorical("style_dim", [8, 16, 32]),
            "lambda_l1": trial.suggest_float("lambda_l1", 10.0, 200.0, log=True),
        }
        set_seed(42)
        device = get_device()
        train_loader, val_loader, _ = make_loaders(params["batch_size"])
        G, D = build_models(params, device)
        opt_g = torch.optim.Adam(G.parameters(), lr=params["lr_g"], betas=(0.5, 0.999))
        opt_d = torch.optim.Adam(D.parameters(), lr=params["lr_d"], betas=(0.5, 0.999))

        best_val = float("inf")
        for epoch in range(max_epochs):
            for batch in train_loader:
                train_step(G, D, batch, opt_g, opt_d, params["lambda_l1"], device)
            val_l1, _ = evaluate(G, val_loader, device)
            best_val = min(best_val, val_l1)
            trial.report(val_l1, epoch)
            if trial.should_prune():
                raise optuna.TrialPruned()
        return best_val

    return objective


def run_optuna_study(n_trials: int, max_epochs_per_trial: int):
    import optuna

    study = optuna.create_study(
        study_name="task4_cgan", storage=f"sqlite:///{HERE / 'optuna_study.db'}",
        load_if_exists=True, direction="minimize", pruner=optuna.pruners.MedianPruner(),
    )
    study.optimize(objective_factory(max_epochs_per_trial), n_trials=n_trials)

    print("\n=== Task 4 Optuna study complete ===")
    print(f"Trials run: {len(study.trials)}")
    print(f"Best trial: #{study.best_trial.number}  best val L1: {study.best_value:.4f}")
    print(f"Best params: {study.best_params}")
    with open(BEST_PARAMS_PATH, "w") as f:
        json.dump(study.best_params, f, indent=2)
    print(f"Saved to {BEST_PARAMS_PATH}")
    return study


# ==========================================================================
# Final training with logging
# ==========================================================================

def tensor_to_uint8(t: torch.Tensor) -> np.ndarray:
    return (t.clamp(0, 1).permute(1, 2, 0).cpu().numpy() * 255).round().astype(np.uint8)


@torch.no_grad()
def save_samples(G, fixed_batch, epoch: int, device, tag: str = ""):
    """Writes [photo | generated | true sketch] rows for the same fixed validation
    photos at every logging interval, so progress can be compared over time."""
    G.eval()
    photo = fixed_batch["photo"].to(device)
    style = fixed_batch["style"].to(device).long()
    gen = (G(photo * 2 - 1, style) + 1) / 2
    G.train()
    rows = []
    for i in range(photo.size(0)):
        rows.append(np.concatenate([tensor_to_uint8(photo[i].cpu()),
                                     tensor_to_uint8(gen[i].cpu()),
                                     tensor_to_uint8(fixed_batch["sketch"][i])], axis=1))
    SAMPLE_DIR.mkdir(parents=True, exist_ok=True)
    Image.fromarray(np.concatenate(rows, axis=0)).save(SAMPLE_DIR / f"epoch_{epoch:04d}{tag}.png")


def train_final(params: dict, epochs: int, sample_every: int = 10):
    set_seed(42)
    device = get_device()
    train_loader, val_loader, val_pairs = make_loaders(params["batch_size"])
    G, D = build_models(params, device)
    print(f"Generator params: {sum(p.numel() for p in G.parameters()):,}")
    print(f"Discriminator params: {sum(p.numel() for p in D.parameters()):,}")

    opt_g = torch.optim.Adam(G.parameters(), lr=params["lr_g"], betas=(0.5, 0.999))
    opt_d = torch.optim.Adam(D.parameters(), lr=params["lr_d"], betas=(0.5, 0.999))

    fixed_loader = DataLoader(FS2KPairDataset(val_pairs[:N_FIXED_SAMPLES], augment=False),
                              batch_size=N_FIXED_SAMPLES, shuffle=False)
    fixed_batch = next(iter(fixed_loader))

    LOG_DIR.mkdir(parents=True, exist_ok=True)
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    tag = variant_tag(params)
    tracker = Tracker("task4_cgan", f"train{tag or '_input'}", {**params, "epochs": epochs})
    log_path = LOG_DIR / f"train_log{tag}.csv"
    best_val = float("inf")

    with open(log_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["epoch", "d_real", "d_fake", "g_adv", "g_l1", "val_l1", "val_ssim"])
        for epoch in range(epochs):
            sums = {"d_real": 0.0, "d_fake": 0.0, "g_adv": 0.0, "g_l1": 0.0}
            steps = 0
            for batch in train_loader:
                losses = train_step(G, D, batch, opt_g, opt_d, params["lambda_l1"], device)
                for k in sums:
                    sums[k] += losses[k]
                steps += 1
            means = {k: v / max(steps, 1) for k, v in sums.items()}
            val_l1, val_ssim = evaluate(G, val_loader, device)

            writer.writerow([epoch + 1, means["d_real"], means["d_fake"],
                             means["g_adv"], means["g_l1"], val_l1, val_ssim])
            f.flush()
            tracker.log_metrics({"d_real": means["d_real"], "d_fake": means["d_fake"],
                                 "g_adv": means["g_adv"], "g_l1": means["g_l1"],
                                 "val_l1": val_l1, "val_ssim": val_ssim}, step=epoch + 1)
            print(f"epoch {epoch+1}/{epochs}  d_real={means['d_real']:.3f} d_fake={means['d_fake']:.3f} "
                  f"g_adv={means['g_adv']:.3f} g_l1={means['g_l1']:.3f}  "
                  f"val_l1={val_l1:.4f} val_ssim={val_ssim:.4f}")

            if val_l1 < best_val:
                best_val = val_l1
                torch.save(G.state_dict(), CHECKPOINT_DIR / f"generator_best{tag}.pt")
                torch.save(D.state_dict(), CHECKPOINT_DIR / f"discriminator_best{tag}.pt")
            if (epoch + 1) % sample_every == 0 or epoch == 0:
                save_samples(G, fixed_batch, epoch + 1, device, tag)

    torch.save(G.state_dict(), CHECKPOINT_DIR / f"generator_last{tag}.pt")
    tracker.log_artifact(CHECKPOINT_DIR / f"generator_best{tag}.pt")
    tracker.close()
    print(f"Best validation L1: {best_val:.4f}. Log: {log_path}")


# ==========================================================================
# ONNX export (generator only -- the discriminator is training-only)
# ==========================================================================

def export_generator_onnx(params: dict, out_path: Path = CHECKPOINT_DIR / "generator.onnx"):
    device = torch.device("cpu")
    G = UNetGenerator(params["base_channels"], params["dropout"], params["style_dim"],
                      params.get("conditioning", "input")).to(device)
    G.load_state_dict(torch.load(CHECKPOINT_DIR / f"generator_best{variant_tag(params)}.pt",
                                 map_location=device))
    G.eval()
    dummy_photo = torch.randn(1, 3, IMG_SIZE, IMG_SIZE)
    dummy_style = torch.zeros(1, dtype=torch.long)
    torch.onnx.export(G, (dummy_photo, dummy_style), str(out_path),
                      input_names=["photo", "style"], output_names=["sketch"],
                      opset_version=17)
    print(f"Exported generator to {out_path}")


# ==========================================================================
# Smoke test: real batch through every step, shapes checked
# ==========================================================================

def smoke_test():
    set_seed(42)
    device = get_device()
    params = {"base_channels": 16, "dropout": 0.2, "style_dim": 8,
              "lr_g": 2e-4, "lr_d": 2e-4, "lambda_l1": 100.0}
    G, D = build_models(params, device)

    train_loader, val_loader, _ = make_loaders(batch_size=4)
    batch = next(iter(train_loader))
    photo = batch["photo"].to(device) * 2 - 1
    style = batch["style"].to(device).long()

    out = G(photo, style)
    assert out.shape == photo.shape, f"generator shape {tuple(out.shape)}"
    assert out.min() >= -1 and out.max() <= 1, "tanh output out of range"
    logits = D(photo, out.detach(), style)
    assert logits.shape == (photo.size(0), 1, 30, 30), f"discriminator shape {tuple(logits.shape)}"
    print(f"generator output {tuple(out.shape)} range [{out.min():.2f}, {out.max():.2f}]")
    print(f"discriminator patch logits {tuple(logits.shape)}")

    opt_g = torch.optim.Adam(G.parameters(), lr=params["lr_g"], betas=(0.5, 0.999))
    opt_d = torch.optim.Adam(D.parameters(), lr=params["lr_d"], betas=(0.5, 0.999))
    losses = train_step(G, D, batch, opt_g, opt_d, params["lambda_l1"], device)
    print("one training step losses:", {k: round(v, 4) for k, v in losses.items()})

    val_l1, val_ssim = evaluate(G, val_loader, device)
    print(f"tiny validation: L1={val_l1:.4f} SSIM={val_ssim:.4f}")
    print("Smoke test passed.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Task 4: Style-conditioned face-to-sketch cGAN")
    parser.add_argument("mode", choices=["smoketest", "optuna", "train", "export-onnx"],
                        nargs="?", default="smoketest")
    parser.add_argument("--n-trials", type=int, default=20)
    parser.add_argument("--trial-epochs", type=int, default=15)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--sample-every", type=int, default=10)
    parser.add_argument("--conditioning", choices=["input", "bottleneck"], default="input",
                        help="input = option A (default); bottleneck = option B, for comparison")
    args = parser.parse_args()

    if args.mode == "smoketest":
        smoke_test()
    elif args.mode == "optuna":
        run_optuna_study(args.n_trials, args.trial_epochs)
    elif args.mode in ("train", "export-onnx"):
        if not BEST_PARAMS_PATH.exists():
            raise SystemExit(f"Run `optuna` first (missing {BEST_PARAMS_PATH}).")
        with open(BEST_PARAMS_PATH) as f:
            params = json.load(f)
        # The Optuna search tuned option A. Option B reuses those hyperparameters,
        # so the comparison isolates the conditioning location.
        params["conditioning"] = args.conditioning
        if args.mode == "train":
            train_final(params, args.epochs, args.sample_every)
        else:
            export_generator_onnx(params)
