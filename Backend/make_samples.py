"""
Creates the clean reference images for the Universal Restoration workspace.

Takes evenly spaced images from the official Oxford pets test list, resizes them to
128x128 the same way training did, and writes them to Backend/samples/ with an index.

  python make_samples.py            # 8 samples
  python make_samples.py --count 12
"""

import argparse
import json
from pathlib import Path

from PIL import Image

HERE = Path(__file__).resolve().parent
PETS = HERE.parent / "oxford-iiit-pet"
SAMPLES_DIR = HERE / "samples"


def label_for(stem: str) -> str:
    """'Egyptian_Mau_123' -> 'Egyptian Mau'."""
    parts = stem.split("_")[:-1]
    return " ".join(parts) if parts else stem


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--count", type=int, default=8)
    args = parser.parse_args()

    with open(PETS / "annotations" / "test.txt") as f:
        stems = [line.split()[0] for line in f if line.strip()]
    step = max(len(stems) // args.count, 1)
    chosen = stems[::step][: args.count]

    SAMPLES_DIR.mkdir(parents=True, exist_ok=True)
    index = []
    for stem in chosen:
        with Image.open(PETS / "images" / f"{stem}.jpg") as im:
            im = im.convert("RGB").resize((128, 128), Image.BILINEAR)
            filename = f"{stem}.png"
            im.save(SAMPLES_DIR / filename)
        index.append({"id": stem, "label": label_for(stem), "file": filename})

    (SAMPLES_DIR / "index.json").write_text(json.dumps(index, indent=2))
    print(f"wrote {len(index)} samples to {SAMPLES_DIR}")


if __name__ == "__main__":
    main()
