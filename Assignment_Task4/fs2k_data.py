"""
Task 4 data pipeline for FS2K (face photo -> sketch).

Dataset layout (after download, nested as FS2K/FS2K/):
  photo/photo{1,2,3}/image####.jpg
  sketch/sketch{1,2,3}/sketch####.jpg   (sketch2 uses .png)
  anno_train.json   1,058 entries  (official training split)
  anno_test.json    1,046 entries  (official test split, untouched until final eval)

Each annotation has image_name "photoN/imageNNNN" and style in {0, 1, 2}
(README: Style 1 / 2 / 3). The style is taken from the annotation, NOT from
the folder, because folders do not map one-to-one to styles.

Provides:
  - build_pairs(): photo path, sketch path, style for every annotation entry
  - make_splits(): official train split -> 85% train / 15% validation,
    stratified by style, seed 42. Official test split returned unchanged.
  - FS2KPairDataset: loads photo and sketch at 128x128 in [0, 1], with
    PAIRED augmentation (same flip/crop applied to both images).
"""

import json
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import Dataset

PROJECT_ROOT = Path(__file__).resolve().parents[1]
FS2K_ROOT = PROJECT_ROOT / "FS2K" / "FS2K"
SPLIT_DIR = Path(__file__).resolve().parent / "splits"

IMG_SIZE = 128
SPLIT_SEED = 42
VAL_FRACTION = 0.15
NUM_STYLES = 3
IMAGE_EXTS = (".jpg", ".png", ".jpeg")


_folder_index = {}


def _find_file(folder: Path, stem: str) -> Path:
    """Finds stem.<ext> in folder, ignoring case in the extension (.jpg, .JPG, ...).
    Linux file systems such as Colab's are case-sensitive, Windows is not."""
    key = str(folder)
    if key not in _folder_index:
        _folder_index[key] = {p.stem.lower(): p for p in folder.iterdir()
                              if p.suffix.lower() in IMAGE_EXTS}
    try:
        return _folder_index[key][stem.lower()]
    except KeyError:
        raise FileNotFoundError(f"No image named {stem} (jpg/png/jpeg, any case) in {folder}")


def load_annotations(split: str) -> list:
    """split: 'train' or 'test'."""
    with open(FS2K_ROOT / f"anno_{split}.json") as f:
        return json.load(f)


def build_pairs(annotations: list) -> list:
    """Maps each annotation to its photo file, sketch file, and style.
    Raises FileNotFoundError if any pair is missing, so bad data fails loudly."""
    pairs = []
    for a in annotations:
        folder, stem = a["image_name"].split("/")       # "photo1", "image0110"
        number = stem.replace("image", "")               # "0110"
        sketch_folder = folder.replace("photo", "sketch")  # "sketch1"
        pairs.append({
            "image_name": a["image_name"],
            "photo": _find_file(FS2K_ROOT / "photo" / folder, stem),
            "sketch": _find_file(FS2K_ROOT / "sketch" / sketch_folder, "sketch" + number),
            "style": int(a["style"]),
        })
    return pairs


def make_splits(seed: int = SPLIT_SEED, val_fraction: float = VAL_FRACTION):
    """Returns (train_pairs, val_pairs, test_pairs).
    Validation is taken from the official training split, stratified by style:
    each style contributes round(val_fraction * its_count) images."""
    train_pairs = build_pairs(load_annotations("train"))
    test_pairs = build_pairs(load_annotations("test"))

    rng = random.Random(seed)
    by_style = {s: [] for s in range(NUM_STYLES)}
    for p in train_pairs:
        by_style[p["style"]].append(p)

    final_train, val = [], []
    for style in range(NUM_STYLES):
        items = sorted(by_style[style], key=lambda p: p["image_name"])
        rng.shuffle(items)
        n_val = round(len(items) * val_fraction)
        val.extend(items[:n_val])
        final_train.extend(items[n_val:])

    return final_train, val, test_pairs


def save_split_manifest(train_pairs, val_pairs, test_pairs, path: Path = None):
    """Writes the exact image names in each split, so the split can be
    verified later and reported in the paper."""
    path = path or SPLIT_DIR / f"fs2k_split_seed{SPLIT_SEED}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    manifest = {}
    for name, pairs in [("train", train_pairs), ("val", val_pairs), ("test", test_pairs)]:
        manifest[name] = {
            "count": len(pairs),
            "style_counts": {str(s): sum(1 for p in pairs if p["style"] == s)
                             for s in range(NUM_STYLES)},
            "image_names": [p["image_name"] for p in pairs],
        }
    with open(path, "w") as f:
        json.dump(manifest, f, indent=2)
    return path


def load_rgb_tensor(path: Path, size: int = IMG_SIZE) -> torch.Tensor:
    """(3, size, size) float tensor in [0, 1]. Converts everything to RGB
    so grayscale or RGBA files don't break the channel count."""
    with Image.open(path) as im:
        im = im.convert("RGB").resize((size, size), Image.BILINEAR)
        arr = np.asarray(im, dtype=np.float32) / 255.0
    return torch.from_numpy(arr.transpose(2, 0, 1).copy())


def paired_augment(photo: torch.Tensor, sketch: torch.Tensor):
    """Applies the SAME random flip and crop to the photo and its sketch.
    Transforming them independently would break the pixel-level pairing."""
    if random.random() < 0.5:
        photo = torch.flip(photo, dims=[2])
        sketch = torch.flip(sketch, dims=[2])

    if random.random() < 0.5:
        scale = random.uniform(0.85, 1.0)
        crop = int(IMG_SIZE * scale)
        y0 = random.randint(0, IMG_SIZE - crop)
        x0 = random.randint(0, IMG_SIZE - crop)
        photo = F.interpolate(photo[:, y0:y0 + crop, x0:x0 + crop].unsqueeze(0),
                              size=(IMG_SIZE, IMG_SIZE), mode="bilinear",
                              align_corners=False).squeeze(0)
        sketch = F.interpolate(sketch[:, y0:y0 + crop, x0:x0 + crop].unsqueeze(0),
                               size=(IMG_SIZE, IMG_SIZE), mode="bilinear",
                               align_corners=False).squeeze(0)

    return photo, sketch


class FS2KPairDataset(Dataset):
    def __init__(self, pairs: list, augment: bool = False):
        self.pairs = pairs
        self.augment = augment

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        p = self.pairs[idx]
        photo = load_rgb_tensor(p["photo"])
        sketch = load_rgb_tensor(p["sketch"])
        if self.augment:
            photo, sketch = paired_augment(photo, sketch)
        return {
            "photo": photo,
            "sketch": sketch,
            "style": p["style"],
            "image_name": p["image_name"],
        }


if __name__ == "__main__":
    train_pairs, val_pairs, test_pairs = make_splits()
    print(f"train={len(train_pairs)}  val={len(val_pairs)}  test={len(test_pairs)}")
    for name, pairs in [("train", train_pairs), ("val", val_pairs), ("test", test_pairs)]:
        counts = {s: sum(1 for p in pairs if p["style"] == s) for s in range(NUM_STYLES)}
        print(f"  {name} style counts: {counts}")

    manifest_path = save_split_manifest(train_pairs, val_pairs, test_pairs)
    print(f"split manifest saved to {manifest_path}")

    ds = FS2KPairDataset(train_pairs[:4], augment=False)
    sample = ds[0]
    print(f"sample: photo {tuple(sample['photo'].shape)}, sketch {tuple(sample['sketch'].shape)}, "
          f"style {sample['style']}, name {sample['image_name']}")

    # Pairing check: feed the SAME image as photo and sketch. Paired augmentation
    # must keep them identical; if they ever differ, the transforms were not shared.
    random.seed(0)
    x = load_rgb_tensor(train_pairs[0]["photo"])
    for _ in range(20):
        a, b = paired_augment(x.clone(), x.clone())
        assert torch.allclose(a, b), "paired augmentation desynchronized photo and sketch"
    print("paired augmentation check: OK (20 random draws kept photo and sketch identical)")
