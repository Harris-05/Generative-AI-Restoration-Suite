"""
FastAPI backend for the Restoration Studio frontend.

Endpoints (the contract in Frontend/app/src/lib/api.js):
  GET  /health                 device, loaded models, softmax temperature
  GET  /samples                clean reference images for the Universal workspace
  POST /universal-restoration  upload a corrupted image, or sample_id + corruption + severity
  POST /hard-routing           classifier routes the image to one specialist
  POST /soft-mixture           gating network blends all branches (needs moe.onnx)
  POST /face-to-sketch         photo + style -> sketch (needs generator.onnx)

Run locally:  uvicorn main:app --reload --port 8000
"""

import json
import os
import time
import zlib
from pathlib import Path

import numpy as np
from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from corruption import apply_corruption, psnr, ssim
from inference import CLASS_KEYS, ModelStore, decode_image, encode_png, softmax

HERE = Path(__file__).resolve().parent
SAMPLES_DIR = HERE / "samples"
MAX_UPLOAD_BYTES = 10 * 1024 * 1024
ALLOWED_TYPES = {"image/jpeg", "image/png"}
CORRUPTIONS = {"salt_pepper", "blur", "occlusion"}
SEVERITIES = {"low", "medium", "high"}

app = FastAPI(title="Restoration Studio API", version="0.1.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=os.environ.get("ALLOWED_ORIGINS", "http://localhost:5173").split(","),
    allow_methods=["*"],
    allow_headers=["*"],
)
if SAMPLES_DIR.exists():
    app.mount("/static/samples", StaticFiles(directory=SAMPLES_DIR), name="samples")

models = ModelStore()


def read_upload(upload: UploadFile) -> bytes:
    if upload.content_type not in ALLOWED_TYPES:
        raise HTTPException(415, "Unsupported file type. Use JPG or PNG.")
    data = upload.file.read(MAX_UPLOAD_BYTES + 1)
    if len(data) > MAX_UPLOAD_BYTES:
        raise HTTPException(413, "File too large (max 10 MB).")
    return data


def image_tensor(upload: UploadFile) -> np.ndarray:
    try:
        return decode_image(read_upload(upload))
    except ValueError as exc:
        raise HTTPException(400, str(exc))


def require(name: str, label: str):
    if not models.loaded(name):
        raise HTTPException(503, f"The {label} model is not loaded. Export it into Backend/models/.")


def load_sample(sample_id: str) -> np.ndarray:
    index = json.loads((SAMPLES_DIR / "index.json").read_text())
    entry = next((s for s in index if s["id"] == sample_id), None)
    if entry is None:
        raise HTTPException(404, f"Unknown sample '{sample_id}'.")
    return decode_image((SAMPLES_DIR / entry["file"]).read_bytes())


@app.get("/health")
def health():
    return {
        "status": "ok",
        "device": models.device,
        "softmax_temperature": models.config.get("softmax_temperature"),
        "models": models.status(),
    }


@app.get("/samples")
def samples(request: Request):
    index_path = SAMPLES_DIR / "index.json"
    if not index_path.exists():
        return []
    base = str(request.base_url).rstrip("/")
    return [
        {"id": s["id"], "label": s["label"], "url": f"{base}/static/samples/{s['file']}"}
        for s in json.loads(index_path.read_text())
    ]


@app.post("/universal-restoration")
def universal_restoration(
    image: UploadFile | None = File(None),
    sample_id: str | None = Form(None),
    corruption: str | None = Form(None),
    severity: str | None = Form(None),
):
    require("universal_autoencoder", "universal restoration")
    clean = None
    if sample_id:
        if corruption not in CORRUPTIONS or severity not in SEVERITIES:
            raise HTTPException(400, "Choose a corruption type and a severity level.")
        clean = load_sample(sample_id)
        seed = zlib.crc32(f"{sample_id}|{corruption}|{severity}".encode())
        corrupted = apply_corruption(clean, corruption, severity, seed)
    elif image is not None:
        corrupted = image_tensor(image)
    else:
        raise HTTPException(400, "Send an image, or a sample_id with a corruption and severity.")

    start = time.perf_counter()
    (restored,) = models.run("universal_autoencoder", {"corrupted": corrupted[None]})
    elapsed_ms = (time.perf_counter() - start) * 1000
    restored = restored[0]

    response = {
        "input_png": encode_png(corrupted),
        "restored_png": encode_png(restored),
        "inference_ms": elapsed_ms,
    }
    if clean is not None:
        response["original_png"] = encode_png(clean)
        restored_clipped = np.clip(restored, 0, 1)
        score_psnr = psnr(restored_clipped, clean)
        if np.isfinite(score_psnr):  # JSON cannot carry infinity (a perfect match)
            response["psnr"] = score_psnr
        response["ssim"] = ssim(restored_clipped, clean)
    return response


@app.post("/hard-routing")
def hard_routing(image: UploadFile = File(...)):
    require("classifier", "classifier")
    x = image_tensor(image)

    start = time.perf_counter()
    (logits,) = models.run("classifier", {"corrupted": x[None]})
    probs = softmax(logits[0])
    predicted = CLASS_KEYS[int(np.argmax(probs))]

    if predicted == "clean":
        restored = x
    else:
        specialist = f"specialist_{predicted}"
        require(specialist, f"{predicted} specialist")
        (out,) = models.run(specialist, {"corrupted": x[None]})
        restored = out[0]
    elapsed_ms = (time.perf_counter() - start) * 1000

    return {
        "probabilities": {k: float(p) for k, p in zip(CLASS_KEYS, probs)},
        "predicted": predicted,
        "expert": predicted,
        "input_png": encode_png(x),
        "restored_png": encode_png(restored),
        "inference_ms": elapsed_ms,
    }


@app.post("/soft-mixture")
def soft_mixture(image: UploadFile = File(...)):
    require("moe", "soft mixture-of-experts")
    x = image_tensor(image)

    start = time.perf_counter()
    restored, weights = models.run("moe", {"corrupted": x[None]})
    elapsed_ms = (time.perf_counter() - start) * 1000
    weights = weights[0]

    return {
        "weights": {k: float(w) for k, w in zip(CLASS_KEYS, weights)},
        "dominant": CLASS_KEYS[int(np.argmax(weights))],
        "input_png": encode_png(x),
        "restored_png": encode_png(restored[0]),
        "inference_ms": elapsed_ms,
    }


@app.post("/face-to-sketch")
def face_to_sketch(image: UploadFile = File(...), style: int = Form(...)):
    require("generator", "face-to-sketch generator")
    if style not in (0, 1, 2):
        raise HTTPException(400, "Style must be 0, 1 or 2 (Style 1, 2 or 3).")
    x = image_tensor(image)

    start = time.perf_counter()
    (sketch,) = models.run(
        "generator",
        {"photo": (x * 2.0 - 1.0)[None].astype(np.float32), "style": np.array([style], dtype=np.int64)},
    )
    elapsed_ms = (time.perf_counter() - start) * 1000

    return {"sketch_png": encode_png(sketch[0], from_range="signed"), "inference_ms": elapsed_ms}
