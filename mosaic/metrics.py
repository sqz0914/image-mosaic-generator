"""Similarity metrics between the original image and its mosaic reconstruction."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from skimage.metrics import structural_similarity


@dataclass
class Scores:
    mse: float
    ssim: float


def mse(a: np.ndarray, b: np.ndarray) -> float:
    """Mean squared error over all pixels and channels."""
    diff = a.astype(np.float64) - b.astype(np.float64)
    return float(np.mean(diff**2))


def ssim(a: np.ndarray, b: np.ndarray) -> float:
    """Structural similarity over the three color channels jointly."""
    # win_size must be odd and no larger than the smallest spatial dimension.
    smallest = min(a.shape[0], a.shape[1])
    win = min(7, smallest if smallest % 2 == 1 else smallest - 1)
    if win < 3:
        return float("nan")

    return float(
        structural_similarity(a, b, channel_axis=2, data_range=255, win_size=win)
    )


def score(reference: np.ndarray, candidate: np.ndarray) -> Scores:
    """Both metrics for one candidate image against the reference."""
    if reference.shape != candidate.shape:
        raise ValueError(
            f"shape mismatch: reference {reference.shape} vs candidate {candidate.shape}"
        )
    return Scores(mse=mse(reference, candidate), ssim=ssim(reference, candidate))
