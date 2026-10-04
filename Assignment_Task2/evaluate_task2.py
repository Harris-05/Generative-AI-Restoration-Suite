"""
Task 2 evaluation on the OFFICIAL Oxford pets test list (annotations/test.txt).

Part 1 (always runs if the classifier exists): classifier accuracy, macro
precision/recall/F1, per-class metrics, and normalized confusion matrix.

Part 2 (runs only when the three specialist checkpoints exist): oracle-routing
and predicted-routing restoration error per (corruption, severity), plus the
top classifier-caused failure cases.

Outputs go to Assignment_Task2/evaluation/.

Usage:
  python evaluate_task2.py --limit 300   # first 300 official test images
  python evaluate_task2.py --limit 0     # full official test set
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
sys.path.insert(0, str(HERE))

from utils import get_device  # noqa: E402
from data_pipeline import PetManifestDataset, generate_test_manifest  # noqa: E402
from official_split import official_test_stems  # noqa: E402
from Assignment_Task1 import UniversalAutoencoder  # noqa: E402
from Assignment_Task2 import (  # noqa: E402
    CorruptionClassifier, evaluate_classifier, HardRoutedSystem, evaluate_routing,
    find_routing_failure_cases, IMAGES_DIR, CHECKPOINT_DIR,
    CLASSIFIER_BEST_PARAMS_PATH, SPECIALIST_ARCH_PARAMS_PATH,
)

OUT_DIR = HERE / "evaluation"
SPECIALISTS = ["salt_pepper", "blur", "occlusion"]


def to_u8(t: torch.Tensor) -> np.ndarray:
    return (t.clamp(0, 1).permute(1, 2, 0).cpu().numpy() * 255).round().astype(np.uint8)


@torch.no_grad()
def save_failure_grids(system, manifest, failures, out_dir: Path):
    """One row per failure: clean | corrupted | oracle-routed | predicted-routed.
    Shows directly how a wrong classifier decision changes the restoration."""
    if not failures:
        return
    out_dir.mkdir(parents=True, exist_ok=True)
    ds = PetManifestDataset(IMAGES_DIR, manifest)
    rows = []
    for f in failures:
        idx = next(i for i, e in enumerate(manifest)
                   if e["stem"] == f["stem"] and e["corruption"] == f["true_corruption"]
                   and e.get("severity", "none") == f["severity"])
        item = ds[idx]
        x = item["corrupted"].unsqueeze(0)
        label = torch.tensor([item["label"]])
        oracle, _, _ = system.restore(x, true_label=label, mode="oracle")
        predicted, _, _ = system.restore(x, mode="predicted")
        rows.append(np.concatenate([to_u8(item["clean"]), to_u8(item["corrupted"]),
                                    to_u8(oracle[0].cpu()), to_u8(predicted[0].cpu())], axis=1))
    Image.fromarray(np.concatenate(rows, axis=0)).save(out_dir / "failure_cases.png")


def jsonable(obj):
    """Converts tuple keys (corruption, severity) to strings so JSON can save them."""
    if isinstance(obj, dict):
        return {("|".join(map(str, k)) if isinstance(k, tuple) else str(k)): jsonable(v)
                for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [jsonable(v) for v in obj]
    return obj


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=300, help="0 = full official test set")
    args = parser.parse_args()

    if not (CLASSIFIER_BEST_PARAMS_PATH.exists() and (CHECKPOINT_DIR / "classifier_final.pt").exists()):
        raise SystemExit("Missing classifier checkpoint. Finish Task 2 classifier training first.")

    device = get_device()
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    with open(CLASSIFIER_BEST_PARAMS_PATH) as f:
        clf_params = json.load(f)
    clf = CorruptionClassifier(clf_params["base_channels"], clf_params["dropout"]).to(device)
    clf.load_state_dict(torch.load(CHECKPOINT_DIR / "classifier_final.pt", map_location=device))
    clf.eval()

    stems = official_test_stems()
    if args.limit > 0:
        stems = stems[:args.limit]
    manifest = generate_test_manifest(stems, seed=42)
    loader = DataLoader(PetManifestDataset(IMAGES_DIR, manifest), batch_size=32, shuffle=False)
    print(f"Evaluating on {len(stems)} official test images ({len(manifest)} entries)")

    cls_metrics = evaluate_classifier(clf, loader, device)
    print("\n=== Classifier (test) ===")
    print(f"accuracy={cls_metrics['accuracy']:.4f}  macro_precision={cls_metrics['macro_precision']:.4f}  "
          f"macro_recall={cls_metrics['macro_recall']:.4f}  macro_f1={cls_metrics['macro_f1']:.4f}")
    for name, m in cls_metrics["per_class"].items():
        print(f"  {name:12s} P={m['precision']:.3f} R={m['recall']:.3f} F1={m['f1']:.3f}")
    print("normalized confusion matrix (rows = true class):")
    for row in cls_metrics["confusion_matrix_normalized"]:
        print("  " + "  ".join(f"{v:.3f}" for v in row))
    with open(OUT_DIR / "task2_classifier_test.json", "w") as f:
        json.dump(cls_metrics, f, indent=2)

    specialist_paths = [CHECKPOINT_DIR / f"specialist_{c}.pt" for c in SPECIALISTS]
    if not (SPECIALIST_ARCH_PARAMS_PATH.exists() and all(p.exists() for p in specialist_paths)):
        print("\nRouting evaluation SKIPPED: specialist checkpoints not found yet.")
        print("  Expected: " + ", ".join(str(p.name) for p in specialist_paths))
        return

    with open(SPECIALIST_ARCH_PARAMS_PATH) as f:
        arch = json.load(f)
    specialists = {}
    for corruption, path in zip(SPECIALISTS, specialist_paths):
        m = UniversalAutoencoder(arch["base_channels"], arch["bottleneck_spatial"],
                                 arch["dropout"], arch["activation"]).to(device)
        m.load_state_dict(torch.load(path, map_location=device))
        specialists[corruption] = m
    system = HardRoutedSystem(clf, specialists, device)

    oracle = evaluate_routing(system, manifest, device, mode="oracle")
    predicted = evaluate_routing(system, manifest, device, mode="predicted")
    failures = find_routing_failure_cases(system, manifest, device, n=4)
    save_failure_grids(system, manifest, failures, OUT_DIR / "figures")
    results = {"oracle": oracle, "predicted": predicted, "failure_cases": failures}
    with open(OUT_DIR / "task2_routing_test.json", "w") as f:
        json.dump(jsonable(results), f, indent=2)

    print("\n=== Routing (test) ===")
    print(f"classifier accuracy on routed test set: {predicted['classifier_accuracy_on_test']:.4f}")
    for mode_name, res in [("oracle", oracle), ("predicted", predicted)]:
        print(f"-- {mode_name} --")
        for key, s in res["per_corruption_severity"].items():
            print(f"  {key[0]:12s} {key[1]:8s} L1={s['mean_l1']:.4f} PSNR={s['mean_psnr']:.2f}")
    print(f"\nTop classifier-caused failures: {len(failures)} found, saved to {OUT_DIR}")


if __name__ == "__main__":
    main()
