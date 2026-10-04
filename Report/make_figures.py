"""
Builds every report figure from the logs and evaluation outputs of Tasks 1-4.

Run from anywhere:  python Report/make_figures.py

Figures are written to Report/figs/ under the names report.tex expects. Any
figure whose input does not exist yet is skipped with a message, so you can run
this after each stage of training and see what is still missing.
"""

import csv
import json
import shutil
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from PIL import Image  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
FIG_DIR = Path(__file__).resolve().parent / "figs"
BRANCHES = ["clean", "salt-pepper", "blur", "occlusion"]


def read_csv(path: Path):
    with open(path) as f:
        rows = list(csv.DictReader(f))
    return {k: [float(r[k]) for r in rows] for k in rows[0]} if rows else None


def skip(name: str, reason: str):
    print(f"  skipped {name}: {reason}")


def save_fig(fig, name: str):
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(FIG_DIR / name, dpi=200)
    plt.close(fig)
    print(f"  wrote {name}")


def copy_image(src: Path, name: str):
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    shutil.copy(src, FIG_DIR / name)
    print(f"  wrote {name}")


# --------------------------------------------------------------------------
# Training curves
# --------------------------------------------------------------------------

def task1_training_curve():
    path = ROOT / "Assignment_Task1" / "logs" / "train_log.csv"
    data = read_csv(path) if path.exists() else None
    if not data:
        return skip("t1_training_curve.png", f"no log at {path}")
    fig, ax = plt.subplots(figsize=(6, 3.5))
    ax.plot(data["epoch"], data["train_loss"], label="train loss")
    ax.plot(data["epoch"], data["val_loss"], label="validation loss")
    ax.set_xlabel("epoch")
    ax.set_ylabel("L1 + SSIM loss")
    ax.set_title("Task 1: universal denoising autoencoder")
    ax.legend()
    save_fig(fig, "t1_training_curve.png")


def task2_classifier_curve():
    path = ROOT / "Assignment_Task2" / "logs" / "classifier_train_log.csv"
    data = read_csv(path) if path.exists() else None
    if not data:
        return skip("t2_classifier_curve.png", f"no log at {path}")
    fig, ax = plt.subplots(figsize=(6, 3.5))
    ax.plot(data["epoch"], data["train_loss"], label="train cross-entropy")
    ax2 = ax.twinx()
    ax2.plot(data["epoch"], data["val_accuracy"], color="tab:green", label="validation accuracy")
    ax.set_xlabel("epoch")
    ax.set_ylabel("cross-entropy")
    ax2.set_ylabel("accuracy")
    ax.set_title("Task 2: corruption classifier")
    save_fig(fig, "t2_classifier_curve.png")


def task2_specialist_curves():
    fig, ax = plt.subplots(figsize=(6, 3.5))
    found = False
    for corruption in ["salt_pepper", "blur", "occlusion"]:
        path = ROOT / "Assignment_Task2" / "logs" / f"specialist_{corruption}_train_log.csv"
        data = read_csv(path) if path.exists() else None
        if data:
            found = True
            ax.plot(data["epoch"], data["val_loss"], label=f"{corruption} (validation)")
    if not found:
        plt.close(fig)
        return skip("t2_specialist_curves.png", "no specialist logs yet")
    ax.set_xlabel("epoch")
    ax.set_ylabel("validation loss")
    ax.set_title("Task 2: specialist autoencoders")
    ax.legend()
    save_fig(fig, "t2_specialist_curves.png")


def task3_training_curve():
    path = ROOT / "Assignment_Task3" / "logs" / "moe_train_log.csv"
    data = read_csv(path) if path.exists() else None
    if not data:
        return skip("t3_training_curve.png", f"no log at {path}")
    fig, ax = plt.subplots(figsize=(6, 3.5))
    ax.plot(data["epoch"], data["val_loss"], label="validation loss")
    ax.set_xlabel("joint fine-tuning epoch")
    ax.set_ylabel("MoE loss")
    ax.set_title("Task 3: soft mixture-of-experts")
    ax.legend()
    save_fig(fig, "t3_training_curve.png")


def task4_loss_curves():
    path = ROOT / "Assignment_Task4" / "logs" / "train_log.csv"
    data = read_csv(path) if path.exists() else None
    if not data:
        return skip("t4_loss_curves.png", f"no log at {path}")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9, 3.5))
    ax1.plot(data["epoch"], data["d_real"], label="D real")
    ax1.plot(data["epoch"], data["d_fake"], label="D fake")
    ax1.plot(data["epoch"], data["g_adv"], label="G adversarial")
    ax1.set_xlabel("epoch")
    ax1.set_ylabel("BCE loss")
    ax1.legend()
    ax2.plot(data["epoch"], data["g_l1"], label="G L1 (training)")
    ax2.plot(data["epoch"], data["val_l1"], label="validation L1")
    ax2.set_xlabel("epoch")
    ax2.set_ylabel("L1")
    ax2.legend()
    fig.suptitle("Task 4: cGAN training")
    save_fig(fig, "t4_loss_curves.png")


def task4_conditioning_comparison():
    logs = {"A: input concatenation": ROOT / "Assignment_Task4" / "logs" / "train_log.csv",
            "B: bottleneck injection": ROOT / "Assignment_Task4" / "logs" / "train_log_bottleneck.csv"}
    fig, ax = plt.subplots(figsize=(6, 3.5))
    found = 0
    for label, path in logs.items():
        data = read_csv(path) if path.exists() else None
        if data:
            found += 1
            ax.plot(data["epoch"], data["val_l1"], label=label)
    if found < 2:
        plt.close(fig)
        return skip("t4_conditioning_comparison.png", "need both option A and option B logs")
    ax.set_xlabel("epoch")
    ax.set_ylabel("validation L1")
    ax.set_title("Task 4: style conditioning comparison")
    ax.legend()
    save_fig(fig, "t4_conditioning_comparison.png")


# --------------------------------------------------------------------------
# Results figures
# --------------------------------------------------------------------------

def task1_examples():
    src = ROOT / "Assignment_Task1" / "evaluation" / "examples" / "all_examples.png"
    if not src.exists():
        return skip("t1_examples.png", "run evaluate_task1.py first")
    copy_image(src, "t1_examples.png")


def task2_confusion():
    path = ROOT / "Assignment_Task2" / "evaluation" / "task2_classifier_test.json"
    if not path.exists():
        return skip("t2_confusion.png", "run evaluate_task2.py first")
    with open(path) as f:
        matrix = np.array(json.load(f)["confusion_matrix_normalized"])
    fig, ax = plt.subplots(figsize=(4.5, 4))
    im = ax.imshow(matrix, vmin=0, vmax=1, cmap="Blues")
    labels = ["clean", "salt", "blur", "occlusion"]
    ax.set_xticks(range(4), labels, rotation=30)
    ax.set_yticks(range(4), labels)
    ax.set_xlabel("predicted class")
    ax.set_ylabel("true class")
    for i in range(4):
        for j in range(4):
            ax.text(j, i, f"{matrix[i, j]:.2f}", ha="center", va="center",
                    color="white" if matrix[i, j] > 0.5 else "black")
    fig.colorbar(im, ax=ax, fraction=0.046)
    ax.set_title("Task 2: normalized confusion matrix")
    save_fig(fig, "t2_confusion.png")


def task2_failures():
    src = ROOT / "Assignment_Task2" / "evaluation" / "figures" / "failure_cases.png"
    if not src.exists():
        return skip("t2_failures.png", "run evaluate_task2.py with trained specialists first")
    copy_image(src, "t2_failures.png")


def task3_routing_heatmap():
    path = ROOT / "Assignment_Task3" / "evaluation" / "task3_routing_test.json"
    if not path.exists():
        return skip("t3_routing_heatmap.png", "run evaluate_task3.py first")
    with open(path) as f:
        routing = json.load(f)["routing"]
    rows = sorted(routing)
    matrix = np.array([routing[r] for r in rows])
    fig, ax = plt.subplots(figsize=(6, 0.4 * len(rows) + 1.5))
    im = ax.imshow(matrix, vmin=0, vmax=1, cmap="magma", aspect="auto")
    ax.set_xticks(range(4), BRANCHES)
    ax.set_yticks(range(len(rows)), [r.replace("|", " / ") for r in rows])
    for i in range(len(rows)):
        for j in range(4):
            ax.text(j, i, f"{matrix[i, j]:.2f}", ha="center", va="center",
                    color="white" if matrix[i, j] < 0.5 else "black")
    fig.colorbar(im, ax=ax, fraction=0.046, label="mean routing weight")
    ax.set_title("Task 3: routing weights by corruption and severity")
    save_fig(fig, "t3_routing_heatmap.png")


def task3_routing_examples():
    src = ROOT / "Assignment_Task3" / "evaluation" / "figures" / "routing_examples.png"
    if not src.exists():
        return skip("t3_routing_examples.png", "run evaluate_task3.py first")
    copy_image(src, "t3_routing_examples.png")


def task4_samples():
    """Three epochs of the fixed validation samples, side by side (first, middle, last)."""
    sample_dir = ROOT / "Assignment_Task4" / "samples"
    files = sorted(p for p in sample_dir.glob("epoch_*.png") if "_bottleneck" not in p.name) \
        if sample_dir.exists() else []
    if not files:
        return skip("t4_samples.png", "no sample images yet (training writes them)")
    picks = [files[0], files[len(files) // 2], files[-1]]
    images = [np.array(Image.open(p).convert("RGB")) for p in picks]
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    Image.fromarray(np.concatenate(images, axis=1)).save(FIG_DIR / "t4_samples.png")
    print(f"  wrote t4_samples.png (epochs {[p.stem for p in picks]}; "
          "columns per epoch: photo | generated | true sketch)")


def main():
    print("Training curves:")
    task1_training_curve()
    task2_classifier_curve()
    task2_specialist_curves()
    task3_training_curve()
    task4_loss_curves()
    task4_conditioning_comparison()
    print("Results figures:")
    task1_examples()
    task2_confusion()
    task2_failures()
    task3_routing_heatmap()
    task3_routing_examples()
    task4_samples()


if __name__ == "__main__":
    main()
