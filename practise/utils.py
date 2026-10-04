"""
Shared building blocks used across Task 1-3 (and reused conceptually in Task 4):
  - reproducible seeding
  - a pure-PyTorch differentiable SSIM (no extra dependency like pytorch-msssim needed)
  - a combined L1 + SSIM reconstruction loss, matching:
        L = alpha * L1(x, x_hat) + (1 - alpha) * (1 - SSIM(x, x_hat))

This file has no dependency on torchvision on purpose -- torchvision isn't
installed in this environment, and we don't need it (PIL + numpy is enough
for loading/resizing images).
"""

import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def set_seed(seed: int = 42):
    """Make runs reproducible. Assignment mandates seed=42 for the data splits,
    so we reuse the same seed constant everywhere for consistency unless a
    step explicitly calls for a different one (e.g. per-image manifest seeds)."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


# --------------------------------------------------------------------------
# SSIM (Structural Similarity Index) -- pure PyTorch, differentiable.
# Standard windowed-Gaussian implementation (Wang et al., 2004).
# --------------------------------------------------------------------------

def _gaussian_window(window_size: int, sigma: float) -> torch.Tensor:
    coords = torch.arange(window_size, dtype=torch.float32) - window_size // 2
    g = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
    g = g / g.sum()
    return g


def _create_ssim_window(window_size: int, channels: int) -> torch.Tensor:
    g_1d = _gaussian_window(window_size, sigma=1.5).unsqueeze(1)
    g_2d = g_1d @ g_1d.t()  # outer product -> 2D gaussian kernel
    window = g_2d.expand(channels, 1, window_size, window_size).contiguous()
    return window


def ssim(img1: torch.Tensor, img2: torch.Tensor, window_size: int = 11,
         val_range: float = 1.0) -> torch.Tensor:
    """
    Computes mean SSIM between two batches of images.
    img1, img2: (B, C, H, W), values expected in [0, val_range]
    Returns a scalar tensor (mean SSIM over the batch), in [-1, 1] (usually close to [0, 1]).
    """
    channels = img1.size(1)
    window = _create_ssim_window(window_size, channels).to(img1.device).to(img1.dtype)

    pad = window_size // 2
    mu1 = F.conv2d(img1, window, padding=pad, groups=channels)
    mu2 = F.conv2d(img2, window, padding=pad, groups=channels)

    mu1_sq = mu1 ** 2
    mu2_sq = mu2 ** 2
    mu1_mu2 = mu1 * mu2

    sigma1_sq = F.conv2d(img1 * img1, window, padding=pad, groups=channels) - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2, window, padding=pad, groups=channels) - mu2_sq
    sigma12 = F.conv2d(img1 * img2, window, padding=pad, groups=channels) - mu1_mu2

    c1 = (0.01 * val_range) ** 2
    c2 = (0.03 * val_range) ** 2

    ssim_map = ((2 * mu1_mu2 + c1) * (2 * sigma12 + c2)) / (
        (mu1_sq + mu2_sq + c1) * (sigma1_sq + sigma2_sq + c2)
    )
    return ssim_map.mean()


class SSIMLoss(nn.Module):
    """Wraps ssim() as a loss-style module returning (1 - SSIM)."""

    def __init__(self, window_size: int = 11):
        super().__init__()
        self.window_size = window_size

    def forward(self, img1: torch.Tensor, img2: torch.Tensor) -> torch.Tensor:
        return 1.0 - ssim(img1, img2, window_size=self.window_size)


class ReconstructionLoss(nn.Module):
    """
    L = alpha * L1(x, x_hat) + (1 - alpha) * (1 - SSIM(x, x_hat))

    alpha is a *tensor-free* float here for simplicity; Optuna will search over it,
    and we just re-instantiate this per trial with the chosen alpha.
    """

    def __init__(self, alpha: float = 0.8):
        super().__init__()
        self.alpha = alpha
        self.l1 = nn.L1Loss()
        self.ssim_loss = SSIMLoss()

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        l1_term = self.l1(pred, target)
        ssim_term = self.ssim_loss(pred, target)
        return self.alpha * l1_term + (1 - self.alpha) * ssim_term


def psnr(pred: torch.Tensor, target: torch.Tensor, max_val: float = 1.0) -> float:
    """Peak Signal-to-Noise Ratio -- a standard extra metric worth reporting
    alongside SSIM/L1 in the results tables (higher is better)."""
    mse = F.mse_loss(pred, target).item()
    if mse == 0:
        return float("inf")
    return 20 * np.log10(max_val) - 10 * np.log10(mse)
