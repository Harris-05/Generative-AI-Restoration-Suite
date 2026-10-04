"""
Task 4 evaluation on the OFFICIAL FS2K test split (anno_test.json, 1,046 pairs).

Reports L1 and SSIM against the true sketch, per style, and saves sample grids
(photo | generated | true sketch) for the first four test pairs of each style.
Needs Assignment_Task4/checkpoints/generator_best.pt and best_params.json.

Outputs go to Assignment_Task4/evaluation/.

Usage:
  python evaluate_task4.py
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
sys.path.insert(0, str(HERE))

from utils import get_device, ssim  # noqa: E402
from fs2k_data import FS2KPairDataset, make_splits, NUM_STYLES  # noqa: E402
from Assignment_Task4 import UNetGenerator, variant_tag  # noqa: E402

PARAMS_PATH = HERE / "best_params.json"
OUT_DIR = HERE / "evaluation"


def to_u8(t: torch.Tensor) -> np.ndarray:
    return (t.clamp(0, 1).permute(1, 2, 0).cpu().numpy() * 255).round().astype(np.uint8)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--conditioning", choices=["input", "bottleneck"], default="input",
                        help="which trained option to evaluate (A = input, B = bottleneck)")
    args = parser.parse_args()

    if not PARAMS_PATH.exists():
        print(f"Task 4 evaluation SKIPPED. Missing {PARAMS_PATH}")
        return
    with open(PARAMS_PATH) as f:
        params = json.load(f)
    params["conditioning"] = args.conditioning
    tag = variant_tag(params)
    gen_ckpt = HERE / "checkpoints" / f"generator_best{tag}.pt"
    if not gen_ckpt.exists():
        print(f"Task 4 evaluation SKIPPED ({args.conditioning}). Missing {gen_ckpt}")
        return

    device = get_device()
    G = UNetGenerator(params["base_channels"], params["dropout"], params["style_dim"],
                      args.conditioning).to(device)
    G.load_state_dict(torch.load(gen_ckpt, map_location=device))
    G.eval()

    _, _, test_pairs = make_splits()
    loader = DataLoader(FS2KPairDataset(test_pairs, augment=False), batch_size=16, shuffle=False)

    per_style = {s: {"l1": [], "ssim": []} for s in range(NUM_STYLES)}
    samples = {s: [] for s in range(NUM_STYLES)}
    with torch.no_grad():
        for batch in loader:
            photo = batch["photo"].to(device)
            target = batch["sketch"].to(device)
            style = batch["style"].to(device).long()
            out = (G(photo * 2 - 1, style) + 1) / 2
            for i in range(photo.size(0)):
                s = int(batch["style"][i])
                per_style[s]["l1"].append((out[i] - target[i]).abs().mean().item())
                per_style[s]["ssim"].append(ssim(out[i:i + 1], target[i:i + 1]).item())
                if len(samples[s]) < 4:
                    samples[s].append(np.concatenate([to_u8(photo[i].cpu()), to_u8(out[i].cpu()),
                                                       to_u8(target[i].cpu())], axis=1))

    results = {}
    for s in range(NUM_STYLES):
        results[f"style{s + 1}"] = {"n": len(per_style[s]["l1"]),
                                    "l1": float(np.mean(per_style[s]["l1"])),
                                    "ssim": float(np.mean(per_style[s]["ssim"]))}
    all_l1 = [v for s in per_style.values() for v in s["l1"]]
    all_ssim = [v for s in per_style.values() for v in s["ssim"]]
    results["overall"] = {"n": len(all_l1), "l1": float(np.mean(all_l1)), "ssim": float(np.mean(all_ssim))}

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    with open(OUT_DIR / f"task4_results{tag}.json", "w") as f:
        json.dump(results, f, indent=2)
    for s in range(NUM_STYLES):
        if samples[s]:
            Image.fromarray(np.concatenate(samples[s], axis=0)).save(
                OUT_DIR / f"samples_style{s + 1}{tag}.png")

    print("FS2K official test split")
    for name, r in results.items():
        print(f"  {name:8s} n={r['n']:4d}  L1={r['l1']:.4f}  SSIM={r['ssim']:.4f}")
    print(f"Saved results and sample grids to {OUT_DIR}")


if __name__ == "__main__":
    main()
