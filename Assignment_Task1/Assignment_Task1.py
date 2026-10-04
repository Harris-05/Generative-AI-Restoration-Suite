
import argparse
import csv
import json
import sys
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

# Reuse the already-tested data pipeline / loss utilities from practise/
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "practise"))
from utils import ReconstructionLoss, get_device, psnr, set_seed  # noqa: E402
from tracking import Tracker  # noqa: E402
from data_pipeline import (  # noqa: E402
    IMG_SIZE, PetTrainDataset, PetManifestDataset, list_clean_images,
    train_val_split, generate_validation_manifest,
)

GROUP_NORM_GROUPS = 8  # divides every channel count we use (16/32/64 x 2^n) evenly
IMAGES_DIR = str(PROJECT_ROOT / "oxford-iiit-pet" / "images")
CHECKPOINT_DIR = Path(__file__).resolve().parent / "checkpoints"
BEST_PARAMS_PATH = Path(__file__).resolve().parent / "best_params.json"
HERE = Path(__file__).resolve().parent
LOG_DIR = HERE / "logs"


# --------------------------------------------------------------------------
# Model
# --------------------------------------------------------------------------

def make_activation(activation: str) -> nn.Module:
    if activation == "relu":
        return nn.ReLU(inplace=True)
    if activation == "leaky_relu":
        return nn.LeakyReLU(0.2, inplace=True)
    raise ValueError(f"Unknown activation: {activation}")


def conv_down_block(in_ch: int, out_ch: int, activation: str = "relu") -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=2, padding=1),
        nn.GroupNorm(GROUP_NORM_GROUPS, out_ch),
        make_activation(activation),
    )


def conv_up_block(in_ch: int, out_ch: int, final: bool = False,
                   activation: str = "relu") -> nn.Sequential:
    layers = [
        nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
        nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=1, padding=1),
    ]
    if final:
        layers.append(nn.Sigmoid())
    else:
        layers += [nn.GroupNorm(GROUP_NORM_GROUPS, out_ch), make_activation(activation)]
    return nn.Sequential(*layers)


class UniversalAutoencoder(nn.Module):
    """x_hat = D(E(x_tilde)).

    depth is derived from bottleneck_spatial (8 -> 4 stages, 4 -> 5 stages),
    since 128 / 2^4 = 8 and 128 / 2^5 = 4.
    """

    def __init__(self, base_channels: int = 32, bottleneck_spatial: int = 8,
                 dropout: float = 0.1, activation: str = "relu", img_size: int = IMG_SIZE):
        super().__init__()
        assert bottleneck_spatial in (4, 8), "bottleneck_spatial must be 4 or 8"
        depth = 4 if bottleneck_spatial == 8 else 5
        assert img_size % (2 ** depth) == 0

        # channel progression: base, 2*base, 4*base, ... one value per stage
        channels = [base_channels * (2 ** i) for i in range(depth)]
        self.bottleneck_channels = channels[-1]
        self.bottleneck_spatial = bottleneck_spatial
        self.depth = depth
        self.activation = activation

        # --- Encoder ---
        encoder_layers = []
        in_ch = 3
        for out_ch in channels:
            encoder_layers.append(conv_down_block(in_ch, out_ch, activation))
            in_ch = out_ch
        self.encoder = nn.Sequential(*encoder_layers)
        self.bottleneck_dropout = nn.Dropout2d(dropout)

        # --- Decoder (mirror of encoder) ---
        decoder_layers = []
        rev_channels = list(reversed(channels))
        for i in range(len(rev_channels) - 1):
            decoder_layers.append(conv_up_block(rev_channels[i], rev_channels[i + 1],
                                                  activation=activation))
        decoder_layers.append(conv_up_block(rev_channels[-1], 3, final=True))
        self.decoder = nn.Sequential(*decoder_layers)

    def forward(self, x):
        latent = self.encoder(x)
        latent = self.bottleneck_dropout(latent)
        return self.decoder(latent)

    def latent_shape_str(self) -> str:
        return f"{self.bottleneck_spatial}x{self.bottleneck_spatial}x{self.bottleneck_channels}"


# --------------------------------------------------------------------------
# Training / evaluation
# --------------------------------------------------------------------------

def train_one_epoch(model, loader, optimizer, loss_fn, device):
    model.train()
    total_loss, n = 0.0, 0
    for batch in loader:
        x = batch["corrupted"].to(device)
        y = batch["clean"].to(device)
        optimizer.zero_grad()
        pred = model(x)
        loss = loss_fn(pred, y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * x.size(0)
        n += x.size(0)
    return total_loss / max(n, 1)


@torch.no_grad()
def evaluate(model, loader, loss_fn, device):
    model.eval()
    total_loss, total_psnr, n = 0.0, 0.0, 0
    for batch in loader:
        x = batch["corrupted"].to(device)
        y = batch["clean"].to(device)
        pred = model(x)
        loss = loss_fn(pred, y)
        total_loss += loss.item() * x.size(0)
        total_psnr += psnr(pred, y) * x.size(0)
        n += x.size(0)
    return total_loss / max(n, 1), total_psnr / max(n, 1)


# --------------------------------------------------------------------------
# Optuna
# --------------------------------------------------------------------------

def build_objective(images_dir, train_ids, val_manifest, device, max_epochs=3):
    def objective(trial):
        lr = trial.suggest_categorical("lr", [0.01, 0.003, 0.001, 0.0003, 0.0001])
        batch_size = trial.suggest_categorical("batch_size", [8, 16, 32, 64])
        bottleneck_spatial = trial.suggest_categorical("bottleneck_spatial", [4, 8])
        base_channels = trial.suggest_categorical("base_channels", [16, 32, 64])
        dropout = trial.suggest_float("dropout", 0.0, 0.3)
        alpha = trial.suggest_float("alpha", 0.5, 0.95)
        activation = trial.suggest_categorical("activation", ["relu", "leaky_relu"])

        model = UniversalAutoencoder(base_channels, bottleneck_spatial, dropout,
                                      activation).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
        loss_fn = ReconstructionLoss(alpha=alpha)

        train_ds = PetTrainDataset(images_dir, train_ids)
        train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, drop_last=True)
        val_ds = PetManifestDataset(images_dir, val_manifest)
        val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

        best_val = float("inf")
        for epoch in range(max_epochs):
            train_one_epoch(model, train_loader, optimizer, loss_fn, device)
            val_loss, _ = evaluate(model, val_loader, loss_fn, device)
            best_val = min(best_val, val_loss)
            print(f"trial {trial.number} epoch {epoch+1}/{max_epochs}  val_loss={val_loss:.4f}")
            trial.report(val_loss, epoch)
            if trial.should_prune():
                import optuna
                raise optuna.TrialPruned()

        return best_val

    return objective


def run_optuna_study(n_trials: int = 30, max_epochs_per_trial: int = 3):
    import optuna

    set_seed(42)
    device = get_device()
    stems = list_clean_images(IMAGES_DIR)
    train_ids, val_ids = train_val_split(stems, seed=42)
    val_manifest = generate_validation_manifest(val_ids, seed=42)

    study = optuna.create_study(
        study_name="task1_universal_ae", storage=f"sqlite:///{HERE / 'optuna_study.db'}",
        load_if_exists=True, direction="minimize", pruner=optuna.pruners.MedianPruner(),
    )
    objective = build_objective(IMAGES_DIR, train_ids, val_manifest, device, max_epochs_per_trial)
    study.optimize(objective, n_trials=n_trials)

    print("\n=== Optuna study complete ===")
    print(f"Trials run: {len(study.trials)}")
    print(f"Best trial: #{study.best_trial.number}")
    print(f"Best val loss: {study.best_value:.4f}")
    print(f"Best params: {study.best_params}")

    BEST_PARAMS_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(BEST_PARAMS_PATH, "w") as f:
        json.dump(study.best_params, f, indent=2)
    print(f"\nSaved best params to {BEST_PARAMS_PATH}")

    return study


# --------------------------------------------------------------------------
# Final full-schedule training with the winning config
# --------------------------------------------------------------------------

def train_final_model(best_params: dict, epochs: int = 50):
    set_seed(42)
    device = get_device()
    stems = list_clean_images(IMAGES_DIR)
    train_ids, val_ids = train_val_split(stems, seed=42)
    val_manifest = generate_validation_manifest(val_ids, seed=42)

    model = UniversalAutoencoder(
        base_channels=best_params["base_channels"],
        bottleneck_spatial=best_params["bottleneck_spatial"],
        dropout=best_params["dropout"],
        activation=best_params["activation"],
    ).to(device)
    print(f"Final model bottleneck: {model.latent_shape_str()}")
    print(f"Total parameters: {sum(p.numel() for p in model.parameters()):,}")

    optimizer = torch.optim.Adam(model.parameters(), lr=best_params["lr"], weight_decay=1e-5)
    loss_fn = ReconstructionLoss(alpha=best_params["alpha"])

    train_ds = PetTrainDataset(IMAGES_DIR, train_ids)
    train_loader = DataLoader(train_ds, batch_size=best_params["batch_size"], shuffle=True)
    val_ds = PetManifestDataset(IMAGES_DIR, val_manifest)
    val_loader = DataLoader(val_ds, batch_size=best_params["batch_size"], shuffle=False)

    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    ckpt_path = CHECKPOINT_DIR / "task1_final.pt"

    tracker = Tracker("task1_universal_ae", "final_train", {**best_params, "epochs": epochs})
    log_path = LOG_DIR / "train_log.csv"
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    best_val = float("inf")
    with open(log_path, "w", newline="") as log_file:
        writer = csv.writer(log_file)
        writer.writerow(["epoch", "train_loss", "val_loss", "val_psnr"])
        for epoch in range(epochs):
            train_loss = train_one_epoch(model, train_loader, optimizer, loss_fn, device)
            val_loss, val_psnr = evaluate(model, val_loader, loss_fn, device)
            writer.writerow([epoch + 1, train_loss, val_loss, val_psnr])
            log_file.flush()
            tracker.log_metrics({"train_loss": train_loss, "val_loss": val_loss,
                                 "val_psnr": val_psnr}, step=epoch + 1)
            print(f"epoch {epoch+1}/{epochs}  train_loss={train_loss:.4f}  "
                  f"val_loss={val_loss:.4f}  val_psnr={val_psnr:.2f}dB")
            if val_loss < best_val:
                best_val = val_loss
                torch.save(model.state_dict(), ckpt_path)
                print(f"  -> saved new best checkpoint to {ckpt_path}")
    print(f"Training log written to {log_path}")
    tracker.log_artifact(ckpt_path)
    tracker.log_artifact(log_path)
    tracker.close()

    return model


# --------------------------------------------------------------------------
# Smoke test -- tiny real run proving the pipeline is wired correctly
# --------------------------------------------------------------------------

def smoke_test():
    set_seed(42)
    device = get_device()
    stems = list_clean_images(IMAGES_DIR)
    train_ids, val_ids = train_val_split(stems, seed=42)

    configs = [
        (16, 8, "relu"),
        (16, 4, "relu"),
        (16, 8, "leaky_relu"),
    ]
    for base_channels, bottleneck_spatial, activation in configs:
        model = UniversalAutoencoder(base_channels, bottleneck_spatial, dropout=0.1,
                                      activation=activation).to(device)
        n_params = sum(p.numel() for p in model.parameters())
        print(f"\n[{model.latent_shape_str()}, {activation}] params={n_params:,}")

        tiny_train = PetTrainDataset(IMAGES_DIR, train_ids[:32])
        loader = DataLoader(tiny_train, batch_size=8, shuffle=True)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        loss_fn = ReconstructionLoss(alpha=0.8)

        loss = train_one_epoch(model, loader, optimizer, loss_fn, device)
        print(f"  training loss after 1 tiny epoch: {loss:.4f}")

        x = torch.rand(2, 3, IMG_SIZE, IMG_SIZE).to(device)
        out = model(x)
        assert out.shape == x.shape, f"shape mismatch: {out.shape} vs {x.shape}"
        assert out.min() >= 0 and out.max() <= 1, "sigmoid output out of [0,1] range"
        print(f"  shape check OK: {tuple(out.shape)}, range [{out.min():.3f}, {out.max():.3f}]")

    print("\nSmoke test passed.")


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Task 1: Universal Denoising Autoencoder")
    parser.add_argument("mode", choices=["smoketest", "optuna", "train"], nargs="?",
                         default="smoketest")
    parser.add_argument("--n-trials", type=int, default=30, help="Optuna trial count")
    parser.add_argument("--trial-epochs", type=int, default=3, help="Epochs per Optuna trial")
    parser.add_argument("--epochs", type=int, default=50, help="Epochs for final training")
    args = parser.parse_args()

    if args.mode == "smoketest":
        smoke_test()
    elif args.mode == "optuna":
        run_optuna_study(n_trials=args.n_trials, max_epochs_per_trial=args.trial_epochs)
    elif args.mode == "train":
        if not BEST_PARAMS_PATH.exists():
            raise SystemExit(
                f"No {BEST_PARAMS_PATH} found. Run `python Assignment_Task1.py optuna` first."
            )
        with open(BEST_PARAMS_PATH) as f:
            best_params = json.load(f)
        train_final_model(best_params, epochs=args.epochs)                  