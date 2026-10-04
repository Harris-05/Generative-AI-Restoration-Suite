# Restoration Studio backend

FastAPI service that runs the four models through ONNX Runtime. The frontend in
`Frontend/app` calls it at `http://localhost:8000`.

## Setup

```bash
cd Backend
pip install -r requirements.txt
python make_samples.py            # creates the clean reference images (run once)
```

## Models

Copy each exported model into `models/` (the export scripts write them to each task's `onnx/` folder):

| File | From | Used by |
|---|---|---|
| `universal_autoencoder.onnx` | Assignment_Task1 | `/universal-restoration` |
| `classifier.onnx` | Assignment_Task2 | `/hard-routing` |
| `specialist_salt_pepper.onnx`, `specialist_blur.onnx`, `specialist_occlusion.onnx` | Assignment_Task2 | `/hard-routing` |
| `moe.onnx` | Assignment_Task3 | `/soft-mixture` |
| `generator.onnx` | Assignment_Task4 | `/face-to-sketch` |

A missing model does not stop the server. Its endpoint returns 503 with the file it needs.

Set `softmax_temperature` in `models/config.json` to the tuned value from
`Assignment_Task3/best_params.json`.

## Run

```bash
uvicorn main:app --reload --port 8000
```

Open `http://localhost:8000/docs` for the interactive API page.

## Endpoints

| Method | Path | Purpose |
|---|---|---|
| GET | `/health` | device, loaded models, softmax temperature |
| GET | `/samples` | clean reference images |
| POST | `/universal-restoration` | upload an image, or `sample_id` + `corruption` + `severity` |
| POST | `/hard-routing` | upload an image |
| POST | `/soft-mixture` | upload an image |
| POST | `/face-to-sketch` | upload an image + `style` (0, 1 or 2) |

Uploads must be JPG or PNG and at most 10 MB. Images are resized to 128×128, as in training.

## Notes

- Corruptions for samples use fixed seeds, so the same sample and severity always give the same
  corrupted image.
- PSNR and SSIM are returned only for samples, which have a clean reference. The SSIM here uses a
  Gaussian window and may differ slightly from the evaluation scripts' value.
