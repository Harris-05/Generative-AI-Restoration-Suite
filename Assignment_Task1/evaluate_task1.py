"""
Task 1 evaluation on the OFFICIAL Oxford pets test list (annotations/test.txt).

Each test image is evaluated on the fixed test grid: clean + 3 severities for
each of salt-and-pepper, blur, and occlusion (10 entries per image).

Every entry is scored three ways, so the restoration can be judged against
doing nothing:
  model   the autoencoder's restored output
  input   the corrupted image itself (the baseline: no restoration)
  median  a 3x3 median filter on the corrupted image (salt-and-pepper only,
          the standard simple denoiser for this corruption)

Outputs (in Assignment_Task1/evaluation/):
  task1_results.json   per (corruption, severity): model, input and median metrics
  task1_results.md     the comparison table for the report
  examples/            12 examples (clean | corrupted | restored | error map)

Usage:
  python evaluate_task1.py --limit 300     # quick check on the first 300 test images
  python evaluate_task1.py --limit 0       # full official test set (slow on CPU)
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from scipy.ndimage import median_filter
from torch.utils.data import DataLoader

HERE = Path(__file__).resolve().parent
PROJECT_ROOT = HERE.parent
sys.path.insert(0, str(PROJECT_ROOT / "practise"))
sys.path.insert(0, str(HERE))

from utils import get_device, psnr, ssim  # noqa: E402
from data_pipeline import PetManifestDataset, generate_test_manifest  # noqa: E402
from official_split import official_test_stems  # noqa: E402
from Assignment_Task1 import UniversalAutoencoder, IMAGES_DIR  # noqa: E402

CKPT_PATH = HERE / "checkpoints" / "task1_final.pt"
PARAMS_PATH = HERE / "best_params.json"
OUT_DIR = HERE / "evaluation"
CORRUPTION_ORDER = ["clean", "salt_pepper", "blur", "occlusion"]
SEVERITY_ORDER = ["none", "low", "medium", "high"]


def to_u8(t: torch.Tensor) -> np.ndarray:
    return (t.clamp(0, 1).permute(1, 2, 0).cpu().numpy() * 255).round().astype(np.uint8)


def error_map_u8(restored: torch.Tensor, clean: torch.Tensor) -> np.ndarray:
    """Absolute error averaged over channels, scaled to this image's maximum."""
    err = (restored - clean).abs().mean(dim=0)
    err = err / (err.max() + 1e-8)
    gray = (err.cpu().numpy() * 255).round().astype(np.uint8)
    return np.stack([gray] * 3, axis=-1)


def median_baseline(corrupted: torch.Tensor) -> torch.Tensor:
    """3x3 median filter per channel, applied to one CHW tensor."""
    hwc = corrupted.permute(1, 2, 0).cpu().numpy()
    filtered = median_filter(hwc, size=(3, 3, 1))
    return torch.from_numpy(filtered.transpose(2, 0, 1).copy()).float()


def metrics(pred: torch.Tensor, target: torch.Tensor) -> dict:
    """L1, PSNR and SSIM for one image pair (CHW tensors on the same device)."""
    return {
        "l1": (pred - target).abs().mean().item(),
        "psnr": psnr(pred.unsqueeze(0), target.unsqueeze(0)),
        "ssim": ssim(pred.unsqueeze(0), target.unsqueeze(0)).item(),
    }


def save_example_grid(examples: list, out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for idx, ex in enumerate(examples):
        row = np.concatenate([to_u8(ex["clean"]), to_u8(ex["corrupted"]),
                               to_u8(ex["restored"]), error_map_u8(ex["restored"], ex["clean"])],
                              axis=1)
        rows.append(row)
        Image.fromarray(row).save(out_dir / f"example_{idx:02d}_{ex['corruption']}.png")
    Image.fromarray(np.concatenate(rows, axis=0)).save(out_dir / "all_examples.png")


def _mean(values: list) -> float:
    finite = [v for v in values if np.isfinite(v)]
    return float(np.mean(finite)) if finite else float("nan")


def _fmt(value: float, digits: int) -> str:
    return "n/a" if not np.isfinite(value) else f"{value:.{digits}f}"


def _gain(model: float, baseline: float, digits: int) -> str:
    """Model minus baseline. Clean inputs are identical to their target (infinite PSNR),
    so gains are not defined for that row."""
    if not (np.isfinite(model) and np.isfinite(baseline)):
        return "n/a"
    return f"{model - baseline:+.{digits}f}"


def write_markdown(summary: dict, path: Path):
    header = ("| Corruption | Severity | n | PSNR model | PSNR input | gain | "
              "PSNR median | SSIM model | SSIM input | gain | SSIM median | L1 model | L1 input |")
    lines = [header, "|" + "---|" * 13]
    for corruption in CORRUPTION_ORDER:
        for severity in SEVERITY_ORDER:
            key = f"{corruption}|{severity}"
            if key not in summary:
                continue
            s = summary[key]
            m, i = s["model"], s["input"]
            med = s.get("median", {"psnr": float("nan"), "ssim": float("nan")})
            lines.append(
                f"| {corruption} | {severity} | {s['n']} "
                f"| {_fmt(m['psnr'], 2)} | {_fmt(i['psnr'], 2)} | {_gain(m['psnr'], i['psnr'], 2)} "
                f"| {_fmt(med['psnr'], 2)} "
                f"| {_fmt(m['ssim'], 4)} | {_fmt(i['ssim'], 4)} | {_gain(m['ssim'], i['ssim'], 4)} "
                f"| {_fmt(med['ssim'], 4)} "
                f"| {_fmt(m['l1'], 4)} | {_fmt(i['l1'], 4)} |")
    lines.append("")
    lines.append("gain = model minus baseline (positive means the model is better than the baseline). "
                 "Clean rows have no corruption, so their input is identical to the target and gains are n/a.")
    path.write_text("\n".join(lines) + "\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=300, help="0 = full official test set")
    args = parser.parse_args()

    for p in (CKPT_PATH, PARAMS_PATH):
        if not p.exists():
            raise SystemExit(f"Missing {p}. Train Task 1 first.")

    with open(PARAMS_PATH) as f:
        params = json.load(f)
    device = get_device()
    model = UniversalAutoencoder(params["base_channels"], params["bottleneck_spatial"],
                                 params["dropout"], params["activation"]).to(device)
    model.load_state_dict(torch.load(CKPT_PATH, map_location=device))
    model.eval()

    stems = official_test_stems()
    if args.limit > 0:
        stems = stems[:args.limit]
    manifest = generate_test_manifest(stems, seed=42)
    loader = DataLoader(PetManifestDataset(IMAGES_DIR, manifest), batch_size=32, shuffle=False)
    print(f"Evaluating {len(stems)} official test images -> {len(manifest)} test entries")

    per_group = {}
    examples_per_type = {c: [] for c in CORRUPTION_ORDER[1:]}
    with torch.no_grad():
        for batch in loader:
            x = batch["corrupted"].to(device)
            y = batch["clean"].to(device)
            out = model(x)
            for i in range(x.size(0)):
                corruption = batch["corruption"][i]
                severity = batch["severity"][i]
                key = f"{corruption}|{severity}"
                g = per_group.setdefault(key, {"model": [], "input": [], "median": []})

                g["model"].append(metrics(out[i], y[i]))
                g["input"].append(metrics(x[i], y[i]))
                if corruption == "salt_pepper":
                    g["median"].append(metrics(median_baseline(x[i]).to(device), y[i]))

                if corruption != "clean" and severity == "high" and len(examples_per_type[corruption]) < 4:
                    examples_per_type[corruption].append({
                        "clean": y[i].cpu(), "corrupted": x[i].cpu(),
                        "restored": out[i].cpu(), "corruption": corruption,
                    })

    summary = {}
    for key, g in per_group.items():
        entry = {"n": len(g["model"])}
        for name in ("model", "input", "median"):
            if g[name]:
                entry[name] = {
                    "l1": _mean([m["l1"] for m in g[name]]),
                    "psnr": _mean([m["psnr"] for m in g[name]]),
                    "ssim": _mean([m["ssim"] for m in g[name]]),
                }
        summary[key] = entry

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    with open(OUT_DIR / "task1_results.json", "w") as f:
        json.dump(summary, f, indent=2, default=lambda v: None if not np.isfinite(v) else v)
    write_markdown(summary, OUT_DIR / "task1_results.md")

    examples = [ex for c in CORRUPTION_ORDER[1:] for ex in examples_per_type[c]]
    save_example_grid(examples, OUT_DIR / "examples")

    print((OUT_DIR / "task1_results.md").read_text())
    print(f"Saved results and {len(examples)} example rows to {OUT_DIR}")


if __name__ == "__main__":
    main()
