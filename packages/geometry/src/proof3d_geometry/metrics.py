"""Shape metrics computed on masks. Appearance metrics (LPIPS, DINOv2, FLIP) live on the GPU
worker; they only run where silhouettes agree, using ``agreement_mask`` from here (PRD 5.4)."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from scipy import ndimage

BoolArray = NDArray[np.bool_]
FloatArray = NDArray[np.float64]


def silhouette_iou(a: BoolArray, b: BoolArray) -> float:
    """Intersection over union of two masks. Two empty masks agree perfectly."""
    a, b = np.asarray(a, dtype=bool), np.asarray(b, dtype=bool)
    if a.shape != b.shape:
        raise ValueError(f"mask shapes differ: {a.shape} vs {b.shape}")
    union = np.count_nonzero(a | b)
    if union == 0:
        return 1.0
    return np.count_nonzero(a & b) / union


def soft_iou(a: FloatArray, b: FloatArray) -> float:
    """Fuzzy IoU on [0, 1] maps; smooth enough for derivative-free optimisation."""
    union = float(np.maximum(a, b).sum())
    if union == 0.0:
        return 1.0
    return float(np.minimum(a, b).sum()) / union


def blur_mask(mask: BoolArray | FloatArray, sigma_px: float) -> FloatArray:
    m = np.asarray(mask, dtype=np.float64)
    return m if sigma_px <= 0 else np.asarray(ndimage.gaussian_filter(m, sigma_px))


def agreement_mask(render: BoolArray, photo: BoolArray, erode_px: int = 2) -> BoolArray:
    """Pixels where both silhouettes agree, eroded so boundary misalignment cannot leak into
    pixel metrics."""
    both: BoolArray = np.logical_and(np.asarray(render, dtype=bool), np.asarray(photo, dtype=bool))
    if erode_px <= 0:
        return both
    return np.asarray(ndimage.binary_erosion(both, iterations=erode_px), dtype=bool)


def mask_centroid_and_area(mask: BoolArray) -> tuple[FloatArray, int]:
    ys, xs = np.nonzero(mask)
    if xs.size == 0:
        return np.array([np.nan, np.nan]), 0
    return np.array([xs.mean() + 0.5, ys.mean() + 0.5]), int(xs.size)
