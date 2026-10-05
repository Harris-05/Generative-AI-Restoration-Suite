# Generative AI Restoration Suite

Four generative-AI systems, trained, evaluated, exported to ONNX, and served through one browser
application with a FastAPI backend and a React frontend. Everything runs in Docker containers.

- **Task 1 — Universal denoising autoencoder.** One convolutional autoencoder restores images damaged by
  salt-and-pepper noise, Gaussian blur, or rectangular occlusion, without being told which corruption is present.
- **Task 2 — Corruption classifier with hard-routed specialists.** A classifier names the corruption, and
  one of three specialist autoencoders restores the image. Clean images bypass the specialists.
- **Task 3 — Soft mixture-of-experts.** A gating network blends the identity branch and the three
  specialists with continuous, differentiable weights, initialised from Task 2 and trained jointly.
- **Task 4 — Style-conditioned face-to-sketch cGAN.** A U-Net generator and a PatchGAN discriminator turn a
  face photograph into a sketch in one of three styles.

The full technical report is in `Report/` (LaTeX, IEEE format). Hyperparameters were tuned with
Optuna, and training runs are tracked with MLflow.

---

## Repository layout

```
.
├── practise/                 shared data pipeline, losses, MLflow tracker, ONNX helpers
├── Assignment_Task1/         Task 1: autoencoder, Optuna search, training, evaluation, ONNX export
├── Assignment_Task2/         Task 2: classifier, specialists, hard routing, evaluation, ONNX export
├── Assignment_Task3/         Task 3: soft mixture-of-experts, routing analysis, ONNX export
├── Assignment_Task4/         Task 4: cGAN, FS2K pipeline, Optuna search, training, evaluation, ONNX export
├── Backend/                  FastAPI service that runs the ONNX models
├── Frontend/app/             React + Tailwind interface (designed in Google Stitch)
├── Report/                   IEEE LaTeX report and figure generation
├── docker-compose.yml        runs the frontend and backend together
├── oxford-iiit-pet/          dataset for Tasks 1–3 (not in the repository, see below)
└── FS2K/                     dataset for Task 4 (not in the repository, see below)
```

---

## Datasets

Neither dataset is stored in this repository. Download them and place them as shown.

**Oxford-IIIT Pet (Tasks 1–3)**
- Download the images and annotations from https://www.robots.ox.ac.uk/~vgg/data/pets/.
- Place them so that the project contains:
  - `oxford-iiit-pet/images/` with the `.jpg` images
  - `oxford-iiit-pet/annotations/trainval.txt` and `oxford-iiit-pet/annotations/test.txt`
- The development split comes from `trainval.txt`. The official `test.txt` is used only for evaluation.

**FS2K (Task 4)**
- Download the dataset from the FS2K authors' release (the link is in the dataset's README).
- Extract it so that the project contains `FS2K/FS2K/` with `photo/`, `sketch/`, `anno_train.json`,
  and `anno_test.json`. The nested `FS2K/FS2K` folder is intentional.

---

## Trained models

The trained weights are not stored in Git. The ONNX models needed to run the application are the
files below. Put them in `Backend/models/`.

| File | Used by |
|---|---|
| `universal_autoencoder.onnx` | Universal Restoration |
| `classifier.onnx`, `specialist_salt_pepper.onnx`, `specialist_blur.onnx`, `specialist_occlusion.onnx` | Hard-Routed Restoration |
| `moe.onnx` | Soft Mixture-of-Experts |
| `generator.onnx` | Face-to-Sketch |

**Download:** the link to the model files is added here before submission.

To produce them yourself, follow the training and export steps below.

---

## Requirements

- Python 3.12 or newer (3.14 was used for development)
- Node.js 20 or newer (for the frontend without Docker)
- Docker Desktop (for the one-command deployment)
- A GPU is optional. Training was run on Google Colab (T4 GPU). CPU training is possible but slow.

Install the Python packages once:

```bash
pip install torch numpy pillow scipy optuna onnx onnxruntime mlflow
```

---

## Run the application with Docker (one command)

This is the required deployment. It starts the backend and the frontend in containers.

1. Clone the repository and place the four ONNX models in `Backend/models/` (see above).
2. From the repository root, run:
   ```bash
   docker compose up --build
   ```
3. Open **http://localhost:8080** in a browser. The backend API is at http://localhost:8000, with
   interactive documentation at http://localhost:8000/docs.

The header shows "Backend: connected" when the models are loaded. Any model that is missing makes only its
own workspace return an error, and the message names the file it needs.

To stop the application, press `Ctrl+C`, then run `docker compose down`.

---

## Run the application without Docker

**Backend:**
```bash
cd Backend
pip install -r requirements.txt
python make_samples.py            # creates the clean sample images (run once)
python -m uvicorn main:app --reload --port 8000
```

**Frontend** (in a second terminal):
```bash
cd Frontend/app
npm install
npm run dev                       # http://localhost:5173
```

---

## Reproduce the results

Run the steps for each task in order. Training is long, so the GPU runs were done on Colab.
Each command writes its outputs inside the task's folder.

### Task 1 — Universal autoencoder
```bash
cd Assignment_Task1
python Assignment_Task1.py optuna --n-trials 30 --trial-epochs 3   # hyperparameter search -> best_params.json
python Assignment_Task1.py train --epochs 50                       # final training -> checkpoints/task1_final.pt
python evaluate_task1.py --limit 0                                 # official test set -> evaluation/
python export_onnx.py                                              # -> onnx/universal_autoencoder.onnx
```
`--limit 0` evaluates the full official test set. Use a smaller number for a quick check.

### Task 2 — Classifier and specialists
```bash
cd Assignment_Task2
python Assignment_Task2.py classifier-optuna --n-trials 30 --trial-epochs 3
python Assignment_Task2.py classifier-train --epochs 30            # -> checkpoints/classifier_final.pt
python Assignment_Task2.py specialist-optuna --n-trials 20 --trial-epochs 3
python Assignment_Task2.py specialist-train --epochs 30            # -> checkpoints/specialist_*.pt
python evaluate_task2.py --limit 0                                 # classifier and routing on the test set
python export_onnx.py                                              # -> onnx/classifier.onnx and specialists
```

### Task 3 — Soft mixture-of-experts
```bash
cd Assignment_Task3
python Assignment_Task3.py optuna --n-trials 20 --trial-epochs 3   # needs Task 2 checkpoints
python Assignment_Task3.py train --warmup-epochs 5 --joint-epochs 20
python evaluate_task3.py --limit 0                                 # routing weights and reconstruction
python export_onnx.py                                              # -> onnx/moe.onnx (whole pipeline)
```

### Task 4 — Face-to-sketch cGAN
```bash
cd Assignment_Task4
python Assignment_Task4.py optuna --n-trials 20 --trial-epochs 15  # 15-epoch trials on the FS2K training split
python Assignment_Task4.py train --epochs 200                      # option A (input conditioning)
python Assignment_Task4.py train --epochs 200 --conditioning bottleneck   # option B, for comparison
python evaluate_task4.py                                           # official FS2K test split
python evaluate_task4.py --conditioning bottleneck
python export_onnx.py                                              # -> onnx/generator.onnx
python export_onnx.py --conditioning bottleneck
```

### Figures and experiment tracking
```bash
python Report/make_figures.py                 # builds the report figures from the logs and evaluations
mlflow ui --backend-store-uri sqlite:///mlflow/mlflow.db
```
Open http://localhost:5000 to browse the MLflow runs (hyperparameters, per-epoch metrics, checkpoints).

### Verifying the ONNX exports
Every `export_onnx.py` runs a parity check that compares PyTorch and ONNX Runtime outputs on real test
images. The check passes when the largest difference is below 1e-4. Run the self-test without trained
weights using `python export_onnx.py --selftest`.

---

## Building the report

The report is written in LaTeX (IEEE conference format). Compile it from inside `Report/`:
```bash
cd Report
pdflatex report.tex
pdflatex report.tex
```
Run `pdflatex` twice so the references resolve. The figures are read from `Report/figs/`.

---

## Reproducibility notes

- Random seeds: 42 for every data split and training run.
- Development data is the official `trainval` list (80/20 split). The official `test` list is used only for
  the final evaluation.
- Corruptions for the test set use the fixed severities from the assignment (`practise/data_pipeline.py`).
- Optuna studies are stored as SQLite files (`optuna_study.db`) in each task folder.

---

## Demonstration video

Demonstration video (unlisted): https://youtu.be/6Q2rip2AkxE

---

## Author

Muhammad Harris (i232535) — i232535@isb.nu.edu.pk
