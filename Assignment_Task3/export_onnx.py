"""
Export the complete Task 3 soft mixture-of-experts pipeline to ONNX and verify
parity with PyTorch. The graph contains the gate, the three experts, the
identity branch, and the blend. It outputs the restored image and the four
routing weights (the UI shows which experts contributed).

The temperature from best_params.json is baked into the graph as a constant.

  python export_onnx.py              # trained checkpoints -> onnx/moe.onnx
  python export_onnx.py --selftest   # random weights, temp folder (checks the pipeline only)
"""

import argparse
import json
import sys
import tempfile
from pathlib import Path

import torch
import torch.nn as nn

HERE = Path(__file__).resolve().parent
PROJECT_ROOT = HERE.parent
sys.path.insert(0, str(PROJECT_ROOT / "practise"))
sys.path.insert(0, str(PROJECT_ROOT / "Assignment_Task1"))
sys.path.insert(0, str(PROJECT_ROOT / "Assignment_Task2"))
sys.path.insert(0, str(HERE))

from onnx_utils import check_parity, export_onnx  # noqa: E402
from data_pipeline import PetManifestDataset, generate_test_manifest  # noqa: E402
from official_split import official_test_stems  # noqa: E402
from Assignment_Task1 import UniversalAutoencoder, IMAGES_DIR  # noqa: E402
from Assignment_Task2 import CorruptionClassifier  # noqa: E402
from Assignment_Task3 import SoftMoE, build_and_load_pretrained_moe, BEST_PARAMS_PATH, CHECKPOINT_DIR  # noqa: E402

MOE_CKPT = CHECKPOINT_DIR / "moe_final.pt"
ONNX_DIR = HERE / "onnx"


class MoEExport(nn.Module):
    """Wraps SoftMoE so the graph has one image input and two outputs."""

    def __init__(self, moe: SoftMoE, temperature: float):
        super().__init__()
        self.moe = moe
        self.temperature = temperature

    def forward(self, x_tilde):
        x_hat, weights, _ = self.moe(x_tilde, self.temperature)
        return x_hat, weights


def real_inputs():
    manifest = generate_test_manifest(official_test_stems()[:2], seed=42)
    ds = PetManifestDataset(IMAGES_DIR, manifest)
    return (torch.stack([ds[i]["corrupted"] for i in range(2)]),)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--selftest", action="store_true")
    args = parser.parse_args()

    if args.selftest:
        tiny = dict(base_channels=16, bottleneck_spatial=8, dropout=0.1, activation="relu")
        moe = SoftMoE(CorruptionClassifier(16, 0.2),
                      UniversalAutoencoder(**tiny), UniversalAutoencoder(**tiny), UniversalAutoencoder(**tiny))
        temperature = 1.0
        inputs = (torch.rand(2, 3, 128, 128),)
        out_dir = Path(tempfile.mkdtemp())
    else:
        if not (MOE_CKPT.exists() and BEST_PARAMS_PATH.exists()):
            raise SystemExit(f"Missing {MOE_CKPT} or {BEST_PARAMS_PATH}. Train Task 3 first.")
        with open(BEST_PARAMS_PATH) as f:
            temperature = json.load(f)["temperature"]
        moe = build_and_load_pretrained_moe("cpu")
        moe.load_state_dict(torch.load(MOE_CKPT, map_location="cpu"))
        inputs = real_inputs()
        out_dir = ONNX_DIR

    model = MoEExport(moe, temperature)
    path = out_dir / "moe.onnx"
    export_onnx(model, inputs, path, ["corrupted"], ["restored", "routing_weights"])
    print(f"exported {path}")
    check_parity(model, inputs, path, ["corrupted"])


if __name__ == "__main__":
    main()
