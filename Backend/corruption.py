"""
Corruptions for the Universal Restoration workspace, using the assignment's test
severities. Each sample and severity gives the same corrupted image every time
(the seed is fixed), so results can be reproduced.

  salt_pepper: pixel probability 3% / 8% / 15%, half black and half white
  blur:        Gaussian kernel 3/5/7 with sigma 0.7/1.5/2.5
  occlusion:   1/2/3 black rectangles covering about 10/20/35% of the image
"""

import numpy as np
from scipy.ndimage import gaussian_filter

SALT = {"low": 0.03, "medium": 0.08, "high": 0.15}
BLUR = {"low": (3, 0.7), "medium": (5, 1.5), "high": (7, 2.5)}
OCCLUSION = {"low": (1, 0.10), "medium": (2, 0.20), "high": (3, 0.35)}


def apply_corruption(chw: np.ndarray, corruption: str, severity: str, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    img = chw.transpose(1, 2, 0).copy()
    h, w, _ = img.shape

    if corruption == "salt_pepper":
        mask = rng.random((h, w)) < SALT[severity]
        white = rng.random((h, w)) < 0.5
        img[mask & white] = 1.0
        img[mask & ~white] = 0.0

    elif corruption == "blur":
        kernel, sigma = BLUR[severity]
        truncate = (kernel // 2) / sigma
        img = np.stack([gaussian_filter(img[..., c], sigma=sigma, truncate=truncate) for c in range(3)], axis=-1)
        img = np.clip(img, 0.0, 1.0)

    elif corruption == "occlusion":
        count, area_fraction = OCCLUSION[severity]
        target = area_fraction * h * w
        if count == 1:
            shares = [1.0]
        else:
            cuts = np.sort(rng.random(count - 1))
            shares = np.diff(np.concatenate([[0.0], cuts, [1.0]]))
        for share in shares:
            rect_pixels = max(int(target * share), 16)
            aspect = rng.uniform(0.5, 2.0)
            rect_h = min(max(int(np.sqrt(rect_pixels * aspect)), 4), h)
            rect_w = min(max(int(rect_pixels / max(rect_h, 1)), 4), w)
            y0 = int(rng.integers(0, h - rect_h + 1))
            x0 = int(rng.integers(0, w - rect_w + 1))
            img[y0:y0 + rect_h, x0:x0 + rect_w, :] = 0.0

    else:
        raise ValueError(f"Unknown corruption: {corruption}")

    return img.transpose(2, 0, 1).astype(np.float32).copy()


def psnr(a: np.ndarray, b: np.ndarray) -> float:
    mse = float(np.mean((a - b) ** 2))
    return float("inf") if mse == 0 else 20 * np.log10(1.0) - 10 * np.log10(mse)


def ssim(a: np.ndarray, b: np.ndarray) -> float:
    """SSIM with a Gaussian window (sigma 1.5) and the standard constants. This is an
    approximation of the evaluation's 11x11 window, so values can differ slightly."""
    c1, c2 = (0.01) ** 2, (0.03) ** 2
    scores = []
    for c in range(a.shape[0]):
        x, y = a[c].astype(np.float64), b[c].astype(np.float64)
        mu_x, mu_y = gaussian_filter(x, 1.5), gaussian_filter(y, 1.5)
        sxx = gaussian_filter(x * x, 1.5) - mu_x**2
        syy = gaussian_filter(y * y, 1.5) - mu_y**2
        sxy = gaussian_filter(x * y, 1.5) - mu_x * mu_y
        num = (2 * mu_x * mu_y + c1) * (2 * sxy + c2)
        den = (mu_x**2 + mu_y**2 + c1) * (sxx + syy + c2)
        scores.append(float(np.mean(num / den)))
    return float(np.mean(scores))
