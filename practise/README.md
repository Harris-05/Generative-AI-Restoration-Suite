# `practise/` — reference & practice code, NOT the submission

This folder is scaffolding to learn from and build on — every file has been
smoke-tested (tiny, 1-epoch, real-data runs) to prove the shapes/losses/
optimizers are wired correctly, but **none of it has been actually trained**.
Per the assignment's own AI-use policy, you're required to read, verify, test,
and understand every part of this before it counts as your work — treat this
as a first draft to critique and rebuild, not a final answer.

## Files

| File | What it is | Depends on |
|---|---|---|
| `utils.py` | seeding, pure-PyTorch SSIM, `ReconstructionLoss` (L1+SSIM) | — |
| `data_pipeline.py` | corruption functions, `PetTrainDataset` (dynamic), manifests + `PetManifestDataset` (deterministic val/test), `BalancedBatchSampler` | `utils` (indirectly) |
| `task1_autoencoder.py` | `UniversalAutoencoder`, training loop, Optuna objective | `utils`, `data_pipeline` |
| `task2_classifier_specialists.py` | `CorruptionClassifier` (GAP head), specialist training (reuses `UniversalAutoencoder`), `HardRoutedSystem` (oracle/predicted modes) | `task1_autoencoder`, `data_pipeline` |
| `task3_moe.py` | `GatingNetwork` (reuses classifier backbone), `SoftMoE`, warm-up + joint fine-tune, balance loss, routing analysis | `task1_autoencoder`, `task2_classifier_specialists` |
| `task4_gan.py` | `UNetGenerator` + `PatchDiscriminator`, both style-conditioned, GAN training step | `utils` only (independent of 1–3) |

Run any file directly (`python practise/task1_autoencoder.py`) to execute its
`__main__` smoke test.

## Installs still needed

```
pip install optuna torch  # torch already installed; optuna is not yet
```
Deliberately **not** using `torchvision` (not installed, and PIL+numpy+scipy
cover everything needed) or `pytorch-msssim` (SSIM is hand-implemented in
`utils.py`) or `opencv`/`cv2` (not installed) — one less dependency to manage
in your Docker image later.

You'll also need, once you actually run real training:
- **MLflow or Weights & Biases** — not wired in yet; add logging calls inside
  the training loops (`train_one_epoch`, `warmup_train`, `joint_finetune`,
  `train_gan`) once you pick one.
- **onnx / onnxruntime** — for the export + parity-check step.
- **FS2K dataset** — not present in this project folder yet (only
  `oxford-iiit-pet` is). `task4_gan.py`'s `PairedSketchDataset` is written
  against an expected `photo_dir` / `sketch_dir` layout with matching
  filename stems; you'll need to download FS2K and either match that layout
  or adjust the class to its actual folder/metadata structure.

## What's deliberately NOT done yet (design decisions you still own)

- No architecture has been validated against real trained results — the
  channel counts, bottleneck sizes, etc. are starting points from our
  discussion, not proven-good numbers. That's what Optuna is for.
- No skip-connection experiment for Task 1 (currently: none, per the safer
  default we discussed) — if you want to test a justified limited skip, that
  comparison isn't written.
- Task 2's specialist training doesn't yet do the "shared Optuna search to
  find one architecture, then train 3 independent copies" orchestration —
  `train_specialist()` takes `arch_params` as an argument; you still need to
  write the loop that runs it 3 times with different `train_ids` filters and
  saves 3 separate checkpoints.
- Task 3's `__main__` block uses freshly-initialized (random) gate/experts
  for the smoke test — real use requires loading Task 2's actual trained
  checkpoints first (the `load_from_classifier` / `load_state_dict` calls are
  stubbed with comments showing where they go).
- No ONNX export code anywhere yet.
- No app/backend/frontend/Docker code — none of that was in scope for this pass.

---

# What's left overall (the honest full list)

Everything below this line is still ahead of you. Roughly in order:

## 1. Finish the research trail (ongoing, not a one-time step)
Keep the notes file we talked about — as you actually run training and see
real results, write down what worked, what didn't, and why. This is the raw
material for your report; it doesn't get written after the fact from memory.

## 2. Actually run training + Optuna, for real, on your hardware
- Task 1: run `run_optuna_study()`, then `train_final_model()` with the winner.
- Task 2: run the classifier Optuna search, train it fully; run one shared
  Optuna search for the specialist architecture, then train all 3 specialists
  independently to convergence; evaluate oracle vs. predicted routing.
- Task 3: load Task 2's checkpoints into the gate/experts, warm-up, joint
  fine-tune, run Optuna over the joint-training hyperparameters, generate the
  routing-weight tables/heatmaps.
- Task 4: get FS2K downloaded and organized, build the stratified 15% val
  split (seed 42), run the (shorter-epoch) Optuna search, retrain the winner
  for the full schedule.

You have no GPU detected in this environment (`torch.cuda.is_available()` is
`False`) — full training runs here will be slow. Worth checking whether you
have access to a GPU machine (lab, cloud credits, Colab) before committing to
full-scale Optuna studies and full training schedules on CPU.

## 3. Experiment tracking
Wire up MLflow or W&B around every training loop — hyperparameters, losses,
checkpoints, and sample visual outputs need to be logged, not just printed.

## 4. Evaluation deliverables
- Task 1: per-corruption × per-severity result tables, ≥12 example grids
  (clean/corrupted/reconstructed/error-map), 4 discussed failure cases.
- Task 2: accuracy/macro-P/R/F1/per-class/confusion matrix for the classifier;
  oracle vs. predicted routing comparison; classifier-error-caused-failure
  discussion.
- Task 3: routing weight tables by corruption × severity, dominant-vs-spread
  examples, dead/dominant-expert check, routing heatmap.
- Task 4: D-real/D-fake/G-adv/G-L1 curves, fixed-validation-photo sample
  progression over training, generated-sketch grids per style.

## 5. ONNX export + parity verification, all 4 tasks
Export: Task 1 model, Task 2 classifier + 3 specialists, Task 3's full
pipeline, Task 4 generator only. For each, run the same input through both
the PyTorch model and the ONNX model and confirm the outputs match closely.

## 6. Application design + build
- Design the UI in Google Stitch first (screenshot/export it for the report).
- React + Tailwind frontend with 4 workspaces (Universal Restoration,
  Hard-Routed Restoration, Soft Mixture-of-Experts Restoration,
  Face-to-Sketch Generator).
- FastAPI backend: health-check, universal-restoration, hard-routing,
  soft-mixture, face-to-sketch endpoints — validate uploads, preprocess,
  load ONNX models, run inference, return result + routing/timing info.

## 7. Dockerize
Dockerfiles for frontend + backend, plus a `docker-compose.yml` that starts
everything with one command. Test it actually works from a clean clone.

## 8. GitHub repo
Push code, configs, dependency file, all scripts, Optuna studies, ONNX
export code, app code, Dockerfiles/compose, and a real README with execution
instructions. Use Git LFS or a documented download link for model weights —
don't commit the dataset or large checkpoints directly.

## 9. Demo video (5–7 min, YouTube, unlisted/public link only)
Show: startup, upload, runtime corruption, universal restoration, hard
routing, soft weights, face-to-sketch, downloading a result, and the
experiment-tracking dashboard.

## 10. The IEEE-format LaTeX report
Written from your actual results and notes (step 1), not written first —
intro, related work, dataset prep, architecture/loss/training per task,
Optuna results, experimental results, application architecture, limitations,
conclusion, AI-use appendix, GitHub link, YouTube link.

## 11. Confirm the real deadline
The PDF says "March 16, 2024," which is almost certainly stale — check
Google Classroom for the actual date before you plan backward from it.
