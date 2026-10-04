"""
Task 3 evaluation on the OFFICIAL Oxford pets test list (annotations/test.txt).

Reports the routing weights per (corruption, severity) and the expert-health
check, using the tuned temperature from best_params.json. Needs:
  - Assignment_Task3/checkpoints/moe_final.pt and best_params.json
  - the Task 2 classifier and three specialist checkpoints it is built from

Outputs go to Assignment_Task3/evaluation/.

Usage:
  python evaluate_task3.py --limit 300
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader

HERE = Path(__file__).resolve().parent
PROJECT_ROOT = HERE.parent
sys.path.insert(0, str(PROJECT_ROOT / "practise"))
sys.path.insert(0, str(PROJECT_ROOT / "Assignment_Task1"))
sys.path.insert(0, str(PROJECT_ROOT / "Assignment_Task2"))
sys.path.insert(0, str(HERE))

from utils import get_device, psnr, ssim  # noqa: E402
from data_pipeline import PetManifestDataset, generate_test_manifest  # noqa: E402
from official_split import official_test_stems  # noqa: E402
from Assignment_Task3 import (  # noqa: E402
    build_and_load_pretrained_moe, routing_weights_by_corruption, check_expert_health,
    BEST_PARAMS_PATH, CHECKPOINT_DIR, IMAGES_DIR, CLASSIFIER_CKPT_PATH,
    SPECIALIST_CKPT_DIR, SPECIALIST_ARCH_PATH,
)

MOE_CKPT = CHECKPOINT_DIR / "moe_final.pt"
OUT_DIR = HERE / "evaluation"


def to_u8(t: torch.Tensor) -> np.ndarray:
    return (t.clamp(0, 1).permute(1, 2, 0).numpy() * 255).round().astype(np.uint8)


def save_example_figure(stored, dominant, spread, out_dir: Path):
    """Rows of clean | corrupted | restored for the dominant and spread examples,
    so the report can show both routing behaviours side by side."""
    out_dir.mkdir(parents=True, exist_ok=True)
    wanted = [("dominant", r) for r in dominant] + [("spread", r) for r in spread]
    rows = []
    for _, target in wanted:
        for record, y, x, x_hat in stored:
            if record is target:
                rows.append(np.concatenate([to_u8(y), to_u8(x), to_u8(x_hat)], axis=1))
                break
    if rows:
        Image.fromarray(np.concatenate(rows, axis=0)).save(out_dir / "routing_examples.png")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=300, help="0 = full official test set")
    args = parser.parse_args()

    required = [MOE_CKPT, BEST_PARAMS_PATH, CLASSIFIER_CKPT_PATH, SPECIALIST_ARCH_PATH] + [
        SPECIALIST_CKPT_DIR / f"specialist_{c}.pt" for c in ["salt_pepper", "blur", "occlusion"]]
    missing = [p for p in required if not p.exists()]
    if missing:
        print("Task 3 evaluation SKIPPED. Missing:")
        for p in missing:
            print(f"  - {p}")
        return

    device = get_device()
    with open(BEST_PARAMS_PATH) as f:
        temperature = json.load(f)["temperature"]
    model = build_and_load_pretrained_moe(device)
    model.load_state_dict(torch.load(MOE_CKPT, map_location=device))
    model.eval()

    stems = official_test_stems()
    if args.limit > 0:
        stems = stems[:args.limit]
    manifest = generate_test_manifest(stems, seed=42)
    loader = DataLoader(PetManifestDataset(IMAGES_DIR, manifest), batch_size=32, shuffle=False)

    # One pass over the test grid: routing weights AND reconstruction metrics.
    groups = {}          # (corruption, severity) -> lists of weights and metrics
    per_image = []       # one record per entry, used to pick examples
    stored = []          # (record, clean, corrupted, restored) for example images
    with torch.no_grad():
        for batch in loader:
            x = batch["corrupted"].to(device)
            y = batch["clean"].to(device)
            x_hat, weights, _ = model(x, temperature)
            weights = weights.cpu().numpy()
            for i in range(x.size(0)):
                key = (batch["corruption"][i], batch["severity"][i])
                g = groups.setdefault(key, {"w": [], "l1": [], "psnr": [], "ssim": []})
                out = x_hat[i:i + 1]
                g["w"].append(weights[i])
                g["l1"].append((out - y[i:i + 1]).abs().mean().item())
                g["psnr"].append(psnr(out, y[i:i + 1]))
                g["ssim"].append(ssim(out, y[i:i + 1]).item())
                record = {"stem": batch["stem"][i], "corruption": key[0], "severity": key[1],
                          "weights": weights[i].tolist()}
                per_image.append(record)
                stored.append((record, y[i].cpu(), x[i].cpu(), x_hat[i].cpu()))

    routing = {k: np.mean(np.stack(v["w"]), axis=0) for k, v in groups.items()}
    health = check_expert_health(routing)

    reconstruction = {}
    for (corruption, severity), g in groups.items():
        finite_psnr = [p for p in g["psnr"] if np.isfinite(p)]
        reconstruction[f"{corruption}|{severity}"] = {
            "n": len(g["l1"]), "l1": float(np.mean(g["l1"])),
            "psnr": float(np.mean(finite_psnr)) if finite_psnr else float("inf"),
            "ssim": float(np.mean(g["ssim"])),
        }

    # Examples: one expert dominates (max weight > 0.8) versus weight spread (max < 0.5).
    dominant = [r for r in per_image if max(r["weights"]) > 0.8][:4]
    spread = [r for r in per_image if max(r["weights"]) < 0.5][:4]
    examples = {"dominant": dominant, "spread": spread}

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    serializable = {
        "temperature": temperature,
        "routing": {f"{c}|{s}": v.tolist() for (c, s), v in routing.items()},
        "reconstruction": reconstruction,
        "health": health,
        "examples": examples,
    }
    with open(OUT_DIR / "task3_routing_test.json", "w") as f:
        json.dump(serializable, f, indent=2)

    save_example_figure(stored, dominant, spread, OUT_DIR / "figures")

    print(f"Routing and reconstruction at temperature={temperature:.3f} "
          f"on {len(stems)} official test images")
    print("branch order: [clean, salt_pepper, blur, occlusion]")
    for key in sorted(routing):
        w = routing[key]
        rec = reconstruction[f"{key[0]}|{key[1]}"]
        print(f"  {key[0]:12s} {key[1]:8s} " + "  ".join(f"{v:.3f}" for v in w)
              + f"   L1={rec['l1']:.4f} SSIM={rec['ssim']:.4f}")
    print("\nExpert health:", json.dumps(health, indent=2))
    print(f"Examples: {len(dominant)} dominant, {len(spread)} spread. Saved to {OUT_DIR}")


if __name__ == "__main__":
    main()
