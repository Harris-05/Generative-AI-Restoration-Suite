"""
Export the Task 4 generator to ONNX and verify parity with PyTorch.
Only the generator is exported; the discriminator is a training component.

  python export_onnx.py                          # option A checkpoint -> onnx/generator.onnx
  python export_onnx.py --conditioning bottleneck  # option B checkpoint -> onnx/generator_bottleneck.onnx
  python export_onnx.py --selftest               # random weights, temp folder (checks the pipeline only)
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
from fs2k_data import FS2KPairDataset, make_splits, IMG_SIZE, NUM_STYLES  # noqa: E402
from Assignment_Task4 import UNetGenerator, variant_tag  # noqa: E402

PARAMS_PATH = HERE / "best_params.json"
CHECKPOINT_DIR = HERE / "checkpoints"
ONNX_DIR = HERE / "onnx"


def real_inputs():
    """Two real test photos (already in the [-1, 1] range the generator expects)
    and their style labels."""
    _, _, test_pairs = make_splits()
    ds = FS2KPairDataset(test_pairs[:2], augment=False)
    photo = torch.stack([ds[i]["photo"] for i in range(2)]) * 2 - 1
    style = torch.tensor([ds[i]["style"] for i in range(2)], dtype=torch.long)
    return photo, style


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--conditioning", choices=["input", "bottleneck"], default="input")
    parser.add_argument("--selftest", action="store_true")
    args = parser.parse_args()

    if args.selftest:
        params = {"base_channels": 16, "dropout": 0.1, "style_dim": 8, "conditioning": args.conditioning}
        G = UNetGenerator(16, 0.1, 8, args.conditioning)
        inputs = (torch.rand(2, 3, IMG_SIZE, IMG_SIZE) * 2 - 1,
                  torch.randint(0, NUM_STYLES, (2,), dtype=torch.long))
        out_dir = Path(tempfile.mkdtemp())
    else:
        if not PARAMS_PATH.exists():
            raise SystemExit(f"Missing {PARAMS_PATH}. Run the Task 4 Optuna search first.")
        with open(PARAMS_PATH) as f:
            params = json.load(f)
        params["conditioning"] = args.conditioning
        ckpt = CHECKPOINT_DIR / f"generator_best{variant_tag(params)}.pt"
        if not ckpt.exists():
            raise SystemExit(f"Missing {ckpt}. Train this option first.")
        G = UNetGenerator(params["base_channels"], params["dropout"], params["style_dim"], args.conditioning)
        G.load_state_dict(torch.load(ckpt, map_location="cpu"))
        inputs = real_inputs()
        out_dir = ONNX_DIR

    path = out_dir / f"generator{variant_tag(params)}.onnx"
    export_onnx(G, inputs, path, ["photo", "style"], ["sketch"])
    print(f"exported {path}")
    check_parity(G, inputs, path, ["photo", "style"])


if __name__ == "__main__":
    main()
