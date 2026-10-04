"""
Model loading and image conversion for the backend.

Models are ONNX files in Backend/models/, exported by each task's export_onnx.py.
Any model whose file is missing is reported as not loaded, and its endpoint returns
503 instead of crashing. Images use the same preprocessing as training: RGB, bilinear
resize to 128x128, values in [0, 1].
"""

import base64
import io
import json
from pathlib import Path

import numpy as np
import onnxruntime as ort
from PIL import Image

MODELS_DIR = Path(__file__).resolve().parent / "models"
IMG_SIZE = 128
CLASS_KEYS = ["clean", "salt_pepper", "blur", "occlusion"]
MODEL_FILES = {
    "universal_autoencoder": "universal_autoencoder.onnx",
    "classifier": "classifier.onnx",
    "specialist_salt_pepper": "specialist_salt_pepper.onnx",
    "specialist_blur": "specialist_blur.onnx",
    "specialist_occlusion": "specialist_occlusion.onnx",
    "moe": "moe.onnx",
    "generator": "generator.onnx",
}


class ModelStore:
    def __init__(self):
        available = ort.get_available_providers()
        self.providers = [p for p in ["CUDAExecutionProvider", "CPUExecutionProvider"] if p in available]
        self.device = "cuda" if "CUDAExecutionProvider" in self.providers else "cpu"
        self.sessions = {}
        for name, filename in MODEL_FILES.items():
            path = MODELS_DIR / filename
            if path.exists():
                self.sessions[name] = ort.InferenceSession(str(path), providers=self.providers)
        config_path = MODELS_DIR / "config.json"
        self.config = json.loads(config_path.read_text()) if config_path.exists() else {}

    def loaded(self, name: str) -> bool:
        return name in self.sessions

    def status(self) -> dict:
        return {name: name in self.sessions for name in MODEL_FILES}

    def run(self, name: str, feeds: dict) -> list:
        return self.sessions[name].run(None, feeds)


def decode_image(data: bytes) -> np.ndarray:
    """Bytes -> (3, 128, 128) float32 in [0, 1]. Raises ValueError if not an image."""
    try:
        with Image.open(io.BytesIO(data)) as im:
            im = im.convert("RGB").resize((IMG_SIZE, IMG_SIZE), Image.BILINEAR)
            arr = np.asarray(im, dtype=np.float32) / 255.0
    except Exception as exc:
        raise ValueError("The file could not be read as an image.") from exc
    return arr.transpose(2, 0, 1).copy()


def encode_png(chw: np.ndarray, from_range: str = "unit") -> str:
    """(3, H, W) array -> base64 PNG. from_range='signed' means values in [-1, 1]."""
    arr = np.asarray(chw, dtype=np.float32)
    if from_range == "signed":
        arr = (arr + 1.0) / 2.0
    arr = (np.clip(arr, 0.0, 1.0).transpose(1, 2, 0) * 255).round().astype(np.uint8)
    buffer = io.BytesIO()
    Image.fromarray(arr).save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode("ascii")


def softmax(logits: np.ndarray) -> np.ndarray:
    shifted = logits - logits.max()
    exp = np.exp(shifted)
    return exp / exp.sum()
