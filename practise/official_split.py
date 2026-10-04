"""
Official Oxford-IIIT Pet split lists, read from annotations/{trainval,test}.txt.

Use these for evaluation. The training pipeline (data_pipeline.list_clean_images)
currently ignores them and uses every image, which leaks the official test set
into training -- see the audit notes.
"""

from pathlib import Path

PETS_ANNOTATIONS = Path(__file__).resolve().parents[1] / "oxford-iiit-pet" / "annotations"


def _read_stems(filename: str) -> list:
    with open(PETS_ANNOTATIONS / filename) as f:
        return [line.split()[0] for line in f if line.strip() and not line.startswith("#")]


def official_test_stems() -> list:
    return _read_stems("test.txt")


def official_trainval_stems() -> list:
    return _read_stems("trainval.txt")
