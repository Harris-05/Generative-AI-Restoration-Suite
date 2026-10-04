"""
Shared data pipeline for Tasks 1-3 (Oxford-IIIT Pet based restoration).

Covers:
  - listing clean images, 80/20 train/val split (seed=42)
  - the 4 runtime corruption definitions (clean / salt_pepper / blur / occlusion)
  - PetTrainDataset: DYNAMIC corruption, freshly sampled every __getitem__ call
    (never writes corrupted copies to disk -- matches the assignment's requirement)
  - deterministic validation + test corruption manifests, generated once and
    reused every time (so results are reproducible / comparable across models)
  - PetManifestDataset: replays a stored manifest exactly
  - BalancedBatchSampler: forces equal per-class counts within every batch,
    for training the Task 2 corruption classifier

Only depends on PIL + numpy + scipy + torch (no torchvision needed).
"""

import json
import os
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

import numpy as np
import torch
from PIL import Image
from scipy.ndimage import gaussian_filter
from torch.utils.data import Dataset, Sampler

IMG_SIZE = 128
CORRUPTION_TYPES = ["clean", "salt_pepper", "blur", "occlusion"]
CORRUPTION_TO_IDX = {c: i for i, c in enumerate(CORRUPTION_TYPES)}

# Fixed test severities, exactly as specified in the assignment.
TEST_SALT_PROBS = [0.03, 0.08, 0.15]
TEST_BLUR_CONFIGS = [(3, 0.7), (5, 1.5), (7, 2.5)]  # (kernel_size, sigma)
TEST_OCCLUSION_CONFIGS = [
    (1, 0.10),  # 1 rectangle,  ~10% area
    (2, 0.20),  # 2 rectangles, ~20% area
    (3, 0.35),  # 3 rectangles, ~35% area
]


# --------------------------------------------------------------------------
# Dataset listing / splitting
# --------------------------------------------------------------------------

def list_clean_images(images_dir: str) -> List[str]:
    """Returns the DEVELOPMENT image stems: the official trainval list only.

    The official test list (annotations/test.txt) is excluded on purpose, so
    no test image is ever used for training or validation. The 41 images that
    appear in neither official list are excluded too."""
    from official_split import official_trainval_stems, official_test_stems

    available = {f.stem for f in Path(images_dir).glob("*.jpg")}
    dev = sorted(s for s in official_trainval_stems() if s in available)
    assert not set(dev) & set(official_test_stems()), "test images leaked into development set"
    return dev


def train_val_split(stems: List[str], val_fraction: float = 0.2, seed: int = 42):
    rng = random.Random(seed)
    shuffled = stems[:]
    rng.shuffle(shuffled)
    n_val = int(len(shuffled) * val_fraction)
    val_ids = shuffled[:n_val]
    train_ids = shuffled[n_val:]
    return train_ids, val_ids


def load_image_as_array(images_dir: str, stem: str, size: int = IMG_SIZE) -> np.ndarray:
    """Loads, converts to RGB, resizes to (size, size), returns float32 array
    in [0, 1] with shape (H, W, 3)."""
    path = Path(images_dir) / f"{stem}.jpg"
    with Image.open(path) as im:
        im = im.convert("RGB").resize((size, size), Image.BILINEAR)
        arr = np.asarray(im, dtype=np.float32) / 255.0
    return arr


# --------------------------------------------------------------------------
# Corruption functions -- operate on HWC float32 arrays in [0, 1]
# --------------------------------------------------------------------------

def apply_salt_pepper(img: np.ndarray, prob: float, rng: np.random.Generator) -> np.ndarray:
    """Replace `prob` fraction of pixels with black or white, 50/50."""
    out = img.copy()
    h, w, _ = out.shape
    mask = rng.random((h, w)) < prob
    salt = rng.random((h, w)) < 0.5  # which of the selected pixels go white vs black
    out[mask & salt] = 1.0
    out[mask & ~salt] = 0.0
    return out


def apply_gaussian_blur(img: np.ndarray, kernel_size: int, sigma: float) -> np.ndarray:
    """Gaussian blur with an *exact* requested kernel size and sigma.
    scipy's gaussian_filter takes `truncate` (kernel radius = truncate * sigma),
    so we back-solve truncate to hit the requested kernel_size."""
    radius = kernel_size // 2
    truncate = max(radius / sigma, 1e-3)
    out = np.empty_like(img)
    for c in range(img.shape[2]):
        out[..., c] = gaussian_filter(img[..., c], sigma=sigma, truncate=truncate)
    return np.clip(out, 0.0, 1.0)


def apply_occlusion(img: np.ndarray, num_rects: int, target_area_frac: float,
                     rng: np.random.Generator) -> tuple[np.ndarray, list]:
    """Insert `num_rects` black rectangles whose combined area is ~target_area_frac
    of the image. Returns (corrupted_img, rect_coords) so coords can be logged
    in manifests for reproducibility."""
    out = img.copy()
    h, w, _ = out.shape
    total_pixels = h * w
    target_pixels = target_area_frac * total_pixels
    # split target area across rectangles with some randomness, then clip
    if num_rects == 1:
        fracs = [1.0]
    else:
        splits = np.sort(rng.random(num_rects - 1))
        fracs = np.diff([0.0, *splits, 1.0])

    coords = []
    for frac in fracs:
        rect_pixels = max(int(target_pixels * frac), 16)  # avoid zero-size rects
        # pick an aspect ratio close to square-ish but randomized
        aspect = rng.uniform(0.5, 2.0)
        rect_h = int(np.sqrt(rect_pixels * aspect))
        rect_w = int(rect_pixels / max(rect_h, 1))
        rect_h = min(max(rect_h, 4), h)
        rect_w = min(max(rect_w, 4), w)

        y0 = int(rng.integers(0, h - rect_h + 1))
        x0 = int(rng.integers(0, w - rect_w + 1))
        out[y0:y0 + rect_h, x0:x0 + rect_w, :] = 0.0
        coords.append({"y0": y0, "x0": x0, "h": rect_h, "w": rect_w})

    return out, coords


def sample_corruption_params(corruption: str, rng: np.random.Generator) -> dict:
    """Sample the *training-time* random severity ranges given in the assignment."""
    if corruption == "clean":
        return {}
    if corruption == "salt_pepper":
        return {"prob": float(rng.uniform(0.02, 0.15))}
    if corruption == "blur":
        kernel_size = int(rng.choice([3, 5, 7]))
        sigma = float(rng.uniform(0.5, 2.5))
        return {"kernel_size": kernel_size, "sigma": sigma}
    if corruption == "occlusion":
        num_rects = int(rng.integers(1, 4))  # 1..3 inclusive
        area_frac = float(rng.uniform(0.10, 0.35))
        return {"num_rects": num_rects, "area_frac": area_frac}
    raise ValueError(corruption)


def apply_corruption(img: np.ndarray, corruption: str, params: dict,
                      rng: np.random.Generator) -> np.ndarray:
    if corruption == "clean":
        return img.copy()
    if corruption == "salt_pepper":
        return apply_salt_pepper(img, params["prob"], rng)
    if corruption == "blur":
        return apply_gaussian_blur(img, params["kernel_size"], params["sigma"])
    if corruption == "occlusion":
        out, _coords = apply_occlusion(img, params["num_rects"], params["area_frac"], rng)
        return out
    raise ValueError(corruption)


def to_chw_tensor(img: np.ndarray) -> torch.Tensor:
    return torch.from_numpy(img.transpose(2, 0, 1).copy()).float()


# --------------------------------------------------------------------------
# Training dataset -- DYNAMIC corruption, resampled every call
# --------------------------------------------------------------------------

class PetTrainDataset(Dataset):
    """Every __getitem__ call samples a fresh corruption type (uniform over
    the 4 classes) and fresh severity, per the assignment. Corrupted images
    are never written to disk."""

    def __init__(self, images_dir: str, stems: List[str], img_size: int = IMG_SIZE,
                 forced_labels: Optional[List[int]] = None):
        self.images_dir = images_dir
        self.stems = stems
        self.img_size = img_size
        # forced_labels: used by BalancedBatchSampler to override the random
        # 25/25/25/25 choice with an externally-assigned, batch-balanced label,
        # while severity within that label is still randomly sampled.
        self.forced_labels = forced_labels

    def __len__(self):
        return len(self.stems)

    def set_forced_labels(self, labels: List[int]):
        assert len(labels) == len(self.stems)
        self.forced_labels = labels

    def __getitem__(self, idx: int):
        stem = self.stems[idx]
        clean = load_image_as_array(self.images_dir, stem, self.img_size)

        rng = np.random.default_rng()  # non-deterministic on purpose (training)
        if self.forced_labels is not None:
            label_idx = self.forced_labels[idx]
        else:
            label_idx = int(rng.integers(0, 4))  # equal probability over 4 classes
        corruption = CORRUPTION_TYPES[label_idx]
        params = sample_corruption_params(corruption, rng)
        corrupted = apply_corruption(clean, corruption, params, rng)

        return {
            "corrupted": to_chw_tensor(corrupted),
            "clean": to_chw_tensor(clean),
            "label": label_idx,
            "corruption": corruption,
        }


# --------------------------------------------------------------------------
# Deterministic manifests for validation / test
# --------------------------------------------------------------------------

def generate_validation_manifest(stems: List[str], seed: int = 42) -> list:
    """One fixed corruption assignment per validation image, generated once."""
    rng = np.random.default_rng(seed)
    manifest = []
    for stem in stems:
        label_idx = int(rng.integers(0, 4))
        corruption = CORRUPTION_TYPES[label_idx]
        params = sample_corruption_params(corruption, rng)
        manifest.append({
            "stem": stem,
            "corruption": corruption,
            "label": label_idx,
            "params": params,
            "seed": seed,
        })
    return manifest


def generate_test_manifest(stems: List[str], seed: int = 42) -> list:
    """Full grid per test image: clean + 3 salt severities + 3 blur configs +
    3 occlusion configs = 10 entries per image, using the FIXED severities
    specified in the assignment (not random)."""
    rng = np.random.default_rng(seed)  # only used for occlusion rect placement
    manifest = []
    for stem in stems:
        manifest.append({"stem": stem, "corruption": "clean", "label": 0,
                          "params": {}, "severity": "none", "seed": seed})

        for i, prob in enumerate(TEST_SALT_PROBS):
            severity = ["low", "medium", "high"][i]
            manifest.append({"stem": stem, "corruption": "salt_pepper", "label": 1,
                              "params": {"prob": prob}, "severity": severity, "seed": seed})

        for i, (k, sigma) in enumerate(TEST_BLUR_CONFIGS):
            severity = ["low", "medium", "high"][i]
            manifest.append({"stem": stem, "corruption": "blur", "label": 2,
                              "params": {"kernel_size": k, "sigma": sigma},
                              "severity": severity, "seed": seed})

        for i, (num_rects, area_frac) in enumerate(TEST_OCCLUSION_CONFIGS):
            severity = ["low", "medium", "high"][i]
            manifest.append({"stem": stem, "corruption": "occlusion", "label": 3,
                              "params": {"num_rects": num_rects, "area_frac": area_frac},
                              "severity": severity, "seed": seed})
    return manifest


def save_manifest(manifest: list, path: str):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(manifest, f, indent=2)


def load_manifest(path: str) -> list:
    with open(path) as f:
        return json.load(f)


class PetManifestDataset(Dataset):
    """Replays a stored manifest exactly -- same corrupted image every time
    this is used, which is what makes val/test scores comparable across runs
    and across models."""

    def __init__(self, images_dir: str, manifest: list, img_size: int = IMG_SIZE):
        self.images_dir = images_dir
        self.manifest = manifest
        self.img_size = img_size

    def __len__(self):
        return len(self.manifest)

    def __getitem__(self, idx: int):
        entry = self.manifest[idx]
        clean = load_image_as_array(self.images_dir, entry["stem"], self.img_size)
        # each manifest entry carries its own seed so a rectangle's position
        # (for occlusion) is reproducible across repeated runs/evaluations
        rng = np.random.default_rng(entry["seed"] + idx)
        corrupted = apply_corruption(clean, entry["corruption"], entry["params"], rng)
        return {
            "corrupted": to_chw_tensor(corrupted),
            "clean": to_chw_tensor(clean),
            "label": entry["label"],
            "corruption": entry["corruption"],
            "severity": entry.get("severity", "none"),
            "stem": entry["stem"],
        }


# --------------------------------------------------------------------------
# Balanced batch sampler -- for Task 2 classifier training
# --------------------------------------------------------------------------

class BalancedBatchSampler(Sampler):
    """Forces every batch to contain (as close to) equal counts of each of the
    4 corruption classes. Works by pre-assigning a balanced label per dataset
    index for the epoch (call `resample_epoch` each epoch), then grouping
    indices by class and interleaving them into batches.

    Usage:
        sampler = BalancedBatchSampler(dataset, batch_size=32)
        sampler.resample_epoch(epoch)              # call once per epoch
        loader = DataLoader(dataset, batch_sampler=sampler)
    """

    def __init__(self, dataset: PetTrainDataset, batch_size: int, seed: int = 42):
        assert batch_size % 4 == 0, "batch_size should be divisible by 4 for clean balancing"
        self.dataset = dataset
        self.batch_size = batch_size
        self.per_class = batch_size // 4
        self.seed = seed
        self.epoch = 0
        self._labels = None
        self.resample_epoch(0)

    def resample_epoch(self, epoch: int):
        self.epoch = epoch
        rng = random.Random(self.seed + epoch)
        n = len(self.dataset)
        # assign as balanced a set of labels as possible across the whole dataset
        base = n // 4
        labels = ([0] * base + [1] * base + [2] * base + [3] * base)
        labels += rng.choices(range(4), k=n - len(labels))  # top up remainder
        rng.shuffle(labels)
        self._labels = labels
        self.dataset.set_forced_labels(labels)

    def __iter__(self):
        rng = random.Random(self.seed + self.epoch + 1)
        buckets = {c: [] for c in range(4)}
        for idx, label in enumerate(self._labels):
            buckets[label].append(idx)
        for c in buckets:
            rng.shuffle(buckets[c])

        n_batches = min(len(buckets[c]) for c in buckets) // self.per_class
        for b in range(n_batches):
            batch = []
            for c in range(4):
                start = b * self.per_class
                batch.extend(buckets[c][start:start + self.per_class])
            rng.shuffle(batch)
            yield batch

    def __len__(self):
        n_per_class = min(sum(1 for l in self._labels if l == c) for c in range(4))
        return n_per_class // self.per_class


if __name__ == "__main__":
    # Smoke test -- verifies the pipeline runs against the real dataset on disk.
    images_dir = str(Path(__file__).resolve().parents[1] / "oxford-iiit-pet" / "images")
    stems = list_clean_images(images_dir)
    print(f"Found {len(stems)} clean images.")
    train_ids, val_ids = train_val_split(stems, seed=42)
    print(f"train={len(train_ids)}  val={len(val_ids)}")

    ds = PetTrainDataset(images_dir, train_ids[:8])
    sample = ds[0]
    print("train sample:", sample["corrupted"].shape, sample["corruption"])

    val_manifest = generate_validation_manifest(val_ids[:8], seed=42)
    val_ds = PetManifestDataset(images_dir, val_manifest)
    vsample = val_ds[0]
    print("val sample:", vsample["corrupted"].shape, vsample["corruption"], vsample["severity"])

    test_manifest = generate_test_manifest(val_ids[:2], seed=42)
    print(f"test manifest entries for 2 images: {len(test_manifest)} (expect 20)")
