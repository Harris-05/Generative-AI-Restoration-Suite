"""
Export the Task 2 classifier and its three specialist autoencoders to ONNX,
and verify parity with PyTorch for each.

  python export_onnx.py              # trained checkpoints -> onnx/
  python export_onnx.py --selftest   # random weights, temp folder (checks the pipeline only)
"""

import argparse
import json
import sys
import tempfile
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
PROJECT_ROOT = HERE.parent
sys.path.insert(0, str(PROJECT_ROOT / "practise"))
sys.path.insert(0, str(PROJECT_ROOT / "Assignment_Task1"))
sys.path.insert(0, str(HERE))

from onnx_utils import check_parity, export_onnx  # noqa: E402
from data_pipeline import PetManifestDataset, generate_test_manifest  # noqa: E402
from official_split import official_test_stems  # noqa: E402
from Assignment_Task1 import UniversalAutoencoder, IMAGES_DIR  # noqa: E402
from Assignment_Task2 import CorruptionClassifier, CHECKPOINT_DIR, CLASSIFIER_BEST_PARAMS_PATH, SPECIALIST_ARCH_PARAMS_PATH  # noqa: E402

SPECIALISTS = ["salt_pepper", "blur", "occlusion"]
ONNX_DIR = HERE / "onnx"


def real_inputs():
    manifest = generate_test_manifest(official_test_stems()[:2], seed=42)
    ds = PetManifestDataset(IMAGES_DIR, manifest)
    return (torch.stack([ds[i]["corrupted"] for i in range(2)]),)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--selftest", action="store_true")
    args = parser.parse_args()

    if args.selftest:
        clf = CorruptionClassifier(16, 0.2)
        spec_params = {"base_channels": 16, "bottleneck_spatial": 8, "dropout": 0.1, "activation": "relu"}
        specialists = {c: UniversalAutoencoder(spec_params["base_channels"], spec_params["bottleneck_spatial"],
                                               spec_params["dropout"], spec_params["activation"])
                       for c in SPECIALISTS}
        inputs = (torch.rand(2, 3, 128, 128),)
        out_dir = Path(tempfile.mkdtemp())
    else:
        needed = [CLASSIFIER_BEST_PARAMS_PATH, SPECIALIST_ARCH_PARAMS_PATH,
                  CHECKPOINT_DIR / "classifier_final.pt"] + [CHECKPOINT_DIR / f"specialist_{c}.pt" for c in SPECIALISTS]
        missing = [p for p in needed if not p.exists()]
        if missing:
            raise SystemExit("Missing: " + ", ".join(str(p) for p in missing))
        with open(CLASSIFIER_BEST_PARAMS_PATH) as f:
            cp = json.load(f)
        with open(SPECIALIST_ARCH_PARAMS_PATH) as f:
            sp = json.load(f)
        clf = CorruptionClassifier(cp["base_channels"], cp["dropout"])
        clf.load_state_dict(torch.load(CHECKPOINT_DIR / "classifier_final.pt", map_location="cpu"))
        specialists = {}
        for c in SPECIALISTS:
            m = UniversalAutoencoder(sp["base_channels"], sp["bottleneck_spatial"], sp["dropout"], sp["activation"])
            m.load_state_dict(torch.load(CHECKPOINT_DIR / f"specialist_{c}.pt", map_location="cpu"))
            specialists[c] = m
        inputs = real_inputs()
        out_dir = ONNX_DIR

    print("classifier:")
    clf_path = out_dir / "classifier.onnx"
    export_onnx(clf, inputs, clf_path, ["corrupted"], ["logits"])
    check_parity(clf, inputs, clf_path, ["corrupted"])

    for c in SPECIALISTS:
        print(f"specialist {c}:")
        path = out_dir / f"specialist_{c}.onnx"
        export_onnx(specialists[c], inputs, path, ["corrupted"], ["restored"])
        check_parity(specialists[c], inputs, path, ["corrupted"])

    print(f"exported to {out_dir}")


if __name__ == "__main__":
    main()
