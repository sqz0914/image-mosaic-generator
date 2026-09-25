"""Image preprocessing: fixed working resolution, grid-aligned cropping, quantization."""

from __future__ import annotations

import cv2
import numpy as np

DEFAULT_LONG_EDGE = 768


def as_rgb_uint8(img: np.ndarray) -> np.ndarray:
    """Coerce any incoming array to a contiguous HxWx3 uint8 RGB image."""
    arr = np.asarray(img)

    if arr.dtype != np.uint8:
        # Floats coming from Gradio / skimage are usually in [0, 1].
        if np.issubdtype(arr.dtype, np.floating) and arr.max() <= 1.0:
            arr = arr * 255.0
        arr = np.clip(arr, 0, 255).astype(np.uint8)

    if arr.ndim == 2:
        arr = cv2.cvtColor(arr, cv2.COLOR_GRAY2RGB)
    elif arr.shape[2] == 4:
        arr = cv2.cvtColor(arr, cv2.COLOR_RGBA2RGB)
    elif arr.shape[2] == 1:
        arr = cv2.cvtColor(arr[:, :, 0], cv2.COLOR_GRAY2RGB)

    return np.ascontiguousarray(arr)


def to_working_resolution(img: np.ndarray, long_edge: int = DEFAULT_LONG_EDGE) -> np.ndarray:
    """Resize so the longest side equals `long_edge`, preserving aspect ratio.

    Fixing the working resolution keeps timings comparable across input images and
    keeps the cost of the pipeline independent of how large a photo the user uploads.
    """
    img = as_rgb_uint8(img)
    h, w = img.shape[:2]
    scale = long_edge / max(h, w)

    if np.isclose(scale, 1.0):
        return img

    new_w, new_h = max(1, round(w * scale)), max(1, round(h * scale))
    # INTER_AREA is the correct choice for downscaling; it averages over the source
    # footprint instead of point-sampling, which matters for the per-cell means later.
    interp = cv2.INTER_AREA if scale < 1 else cv2.INTER_CUBIC
    return cv2.resize(img, (new_w, new_h), interpolation=interp)


def grid_shape(img: np.ndarray, n_cols: int) -> tuple[int, int, int]:
    """Return (n_rows, n_cols, cell_px) for square cells across roughly `n_cols` columns.

    The requested count sets the *cell size*; the actual grid then takes every whole cell
    that fits on each axis. Capping the column count at the request instead would leave a
    partial column's worth of pixels unused -- asking for 37 columns of a 768px image
    gives 20px cells, of which 38 fit, so honoring 37 exactly would discard 28 pixels
    rather than 8. Taking all of them keeps the crop below one cell on every axis.
    """
    h, w = img.shape[:2]
    cell = max(1, w // max(1, n_cols))
    return max(1, h // cell), max(1, w // cell), cell


def crop_to_grid(img: np.ndarray, n_cols: int) -> tuple[np.ndarray, int, int, int]:
    """Center-crop `img` so it divides exactly into square cells.

    Returns (cropped, n_rows, n_cols, cell_px). At most `cell_px - 1` pixels are
    removed per axis, which is the "slight crop so a grid can be applied" the
    assignment allows, and it lets the grid statistics use a pure reshape with no
    padding or ragged edge cells.
    """
    img = as_rgb_uint8(img)
    n_rows, n_cols, cell = grid_shape(img, n_cols)

    target_h, target_w = n_rows * cell, n_cols * cell
    h, w = img.shape[:2]
    top, left = (h - target_h) // 2, (w - target_w) // 2

    cropped = img[top : top + target_h, left : left + target_w]
    return np.ascontiguousarray(cropped), n_rows, n_cols, cell


def quantize_colors(arr: np.ndarray, k: int = 12) -> np.ndarray:
    """Reduce `arr` to at most `k` distinct colors by k-means clustering in RGB space.

    Works on any (..., 3) uint8 array, so it can be applied either to a full image or,
    much more cheaply, to the per-cell mean colors. Simplifying the palette before tile
    matching produces flatter, more poster-like mosaics.
    """
    arr = np.asarray(arr)
    if k <= 1:
        return arr.astype(np.uint8)

    original_shape = arr.shape
    samples = arr.reshape(-1, 3).astype(np.float32)
    k = min(k, len(np.unique(samples, axis=0)))
    if k <= 1:
        return arr.astype(np.uint8)

    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 20, 1.0)
    _, labels, centers = cv2.kmeans(
        samples, k, None, criteria, attempts=3, flags=cv2.KMEANS_PP_CENTERS
    )
    quantized = centers[labels.ravel()]
    return quantized.reshape(original_shape).clip(0, 255).astype(np.uint8)
