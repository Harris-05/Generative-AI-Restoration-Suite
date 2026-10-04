"""
Shared ONNX export and parity check used by every task's export_onnx.py.

Parity means: run the same inputs through the PyTorch model and the exported
ONNX graph (ONNX Runtime), then compare the outputs. The assignment requires
them to match; we accept a maximum absolute difference of 1e-4 for float32.
"""

from pathlib import Path

import numpy as np
import onnxruntime as ort
import torch

PARITY_TOLERANCE = 1e-4


def export_onnx(model: torch.nn.Module, example_inputs: tuple, path: Path,
                input_names: list, output_names: list):
    """Exports with the legacy TorchScript-based exporter, which handles these
    architectures reliably, and allows a dynamic batch dimension."""
    model.eval()
    path.parent.mkdir(parents=True, exist_ok=True)
    dynamic_axes = {name: {0: "batch"} for name in input_names + output_names}
    torch.onnx.export(
        model, example_inputs, str(path),
        input_names=input_names, output_names=output_names,
        opset_version=17, dynamic_axes=dynamic_axes, dynamo=False,
    )


def check_parity(model: torch.nn.Module, inputs: tuple, path: Path,
                 input_names: list) -> float:
    """Returns the largest absolute difference between PyTorch and ONNX Runtime
    outputs. Raises AssertionError if it exceeds PARITY_TOLERANCE."""
    model.eval()
    with torch.no_grad():
        ref = model(*inputs)
    ref = ref if isinstance(ref, tuple) else (ref,)

    session = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    feed = {name: tensor.cpu().numpy() for name, tensor in zip(input_names, inputs)}
    outputs = session.run(None, feed)

    worst = 0.0
    for got, expected in zip(outputs, ref):
        diff = float(np.max(np.abs(got - expected.cpu().numpy())))
        worst = max(worst, diff)
    status = "PASS" if worst <= PARITY_TOLERANCE else "FAIL"
    print(f"  parity {status}: max |PyTorch - ONNX| = {worst:.2e} (tolerance {PARITY_TOLERANCE:.0e})")
    assert worst <= PARITY_TOLERANCE, f"ONNX output differs from PyTorch by {worst:.2e}"
    return worst
