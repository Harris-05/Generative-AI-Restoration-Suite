"""
Export the Task 1 universal autoencoder to ONNX and verify parity with PyTorch.

  python export_onnx.py              # trained checkpoint -> onnx/universal_autoencoder.onnx
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
sys.path.insert(0, str(HERE))

from onnx_utils import check_parity, export_onnx  # noqa: E402
from data_pipeline import PetManifestDataset, generate_test_manifest  # noqa: E402
from official_split import official_test_stems  # noqa: E402
from Assignment_Task1 import UniversalAutoencoder, IMAGES_DIR  # noqa: E402

CKPT_PATH = HERE / "checkpoints" / "task1_final.pt"
PARAMS_PATH = HERE / "best_params.json"
ONNX_DIR = HERE / "onnx"


def build(params: dict) -> UniversalAutoencoder:
    return UniversalAutoencoder(params["base_channels"], params["bottleneck_spatial"],
                                params["dropout"], params["activation"])


def real_inputs():
    """Two real corrupted test images, so parity is checked on realistic data."""
    manifest = generate_test_manifest(official_test_stems()[:2], seed=42)
    ds = PetManifestDataset(IMAGES_DIR, manifest)
    return (torch.stack([ds[i]["corrupted"] for i in range(2)]),)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--selftest", action="store_true")
    args = parser.parse_args()

    if args.selftest:
        params = {"base_channels": 16, "bottleneck_spatial": 8, "dropout": 0.1, "activation": "relu"}
        model = build(params)
        inputs = (torch.rand(2, 3, 128, 128),)
        out_dir = Path(tempfile.mkdtemp())
    else:
        if not (CKPT_PATH.exists() and PARAMS_PATH.exists()):
            raise SystemExit(f"Missing {CKPT_PATH} or {PARAMS_PATH}. Train Task 1 first.")
        with open(PARAMS_PATH) as f:
            params = json.load(f)
        model = build(params)
        model.load_state_dict(torch.load(CKPT_PATH, map_location="cpu"))
        inputs = real_inputs()
        out_dir = ONNX_DIR

    path = out_dir / "universal_autoencoder.onnx"
    export_onnx(model, inputs, path, ["corrupted"], ["restored"])
    print(f"exported {path}")
    check_parity(model, inputs, path, ["corrupted"])


if __name__ == "__main__":
    main()
