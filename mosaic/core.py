"""Grid division, cell classification and mosaic reconstruction.

Every grid operation here is a whole-array NumPy expression -- there is not a single
Python loop over cells on the fast path. `build_mosaic_loop` is the deliberately naive
counterpart kept for the performance comparison in `benchmark.py`.
"""

from __future__ import annotations

import time
from dataclasses import dataclass

import cv2
import numpy as np

from .preprocess import crop_to_grid, quantize_colors, to_working_resolution
from .tiles import LUMA, TileSet, get_tile_set

# Guard rail: grid_size * tile_px can explode (a 128 grid of 32px tiles is 4096x4096).
# Beyond this the browser, not the algorithm, becomes the bottleneck.
MAX_OUTPUT_PX = 2048

COLOR_MODES = ("Tile only", "Blend", "Modulate")
MATCH_SPACES = ("RGB", "CIE-Lab")


@dataclass
class MosaicResult:
    """Everything the UI and the report need from one run of the pipeline."""

    original: np.ndarray    # preprocessed + grid-cropped input (the comparison reference)
    segmented: np.ndarray   # per-cell mean colors, upsampled -- the "grid" visualization
    mosaic: np.ndarray      # tile reconstruction re-rendered at `original`'s resolution
    mosaic_full: np.ndarray # tile reconstruction at the requested tile resolution
    indices: np.ndarray     # (n_rows, n_cols) chosen tile index per cell
    n_rows: int
    n_cols: int
    cell_px: int
    tile_px: int
    elapsed: float          # seconds, pipeline only (excludes tile-set generation)


# --------------------------------------------------------------------------------------
# Step 2: grid statistics -- vectorized
# --------------------------------------------------------------------------------------


def cell_statistics(
    img: np.ndarray, n_rows: int, n_cols: int, cell: int, with_std: bool = True
) -> tuple[np.ndarray, np.ndarray | None]:
    """Mean color and internal contrast of every grid cell, with no loops.

    The image has already been cropped to exactly (n_rows*cell, n_cols*cell), so the grid
    is a pure reshape: splitting each axis into (n_blocks, block_len) and then reducing
    over the two intra-block axes gives every cell's statistics in one pass. Note that
    the reshape is a view, not a copy -- no pixel data moves, only the strides change.

    The standard deviation is the expensive half: it needs per-pixel luminance, so unlike
    the mean it cannot avoid materializing the image as floats. It is only consulted when
    texture-aware matching is switched on, hence the two paths below.

    Returns (mean (n_rows, n_cols, 3) float32, std (n_rows, n_cols) float32 or None).
    """
    blocks = img.reshape(n_rows, cell, n_cols, cell, 3)

    if not with_std:
        # Reduce straight off the uint8 view with a float32 accumulator: casting the
        # image first would copy every pixel for nothing. The accumulation is exact --
        # a cell holds at most 96*96 pixels, so the largest channel sum is 255*9216 =
        # 2.35M, well inside the 2**24 integers float32 represents exactly -- so this
        # returns the correctly rounded float32 mean.
        return blocks.mean(axis=(1, 3), dtype=np.float32), None

    # Contrast within a cell, measured on luminance: a proxy for "how much detail did
    # this cell lose", used for texture-aware tile selection below. The float copy this
    # needs is shared with the mean, so asking for both costs one cast, not two.
    fblocks = blocks.astype(np.float32)
    mean = fblocks.mean(axis=(1, 3))
    std = (fblocks @ LUMA).std(axis=(1, 3))

    return mean, std.astype(np.float32)


def _to_match_space(rgb: np.ndarray, space: str) -> np.ndarray:
    """Convert an (..., 3) RGB array in [0,255] into the chosen matching space."""
    if space != "CIE-Lab":
        return rgb.astype(np.float32)

    flat = np.ascontiguousarray(rgb.reshape(1, -1, 3).astype(np.float32) / 255.0)
    lab = cv2.cvtColor(flat, cv2.COLOR_RGB2LAB)
    return lab.reshape(rgb.shape)


# --------------------------------------------------------------------------------------
# Step 3: classification -- vectorized nearest-tile search
# --------------------------------------------------------------------------------------


def match_tiles(
    cell_mean: np.ndarray,
    cell_std: np.ndarray | None,
    tile_set: TileSet,
    space: str = "RGB",
    detail_weight: float = 0.0,
) -> np.ndarray:
    """Assign a tile index to every cell by nearest-neighbor search.

    Distances are computed for all cells against all tiles at once by broadcasting to
    (n_rows, n_cols, n_tiles) -- for a 128x128 grid against 16 tiles that is a single
    262k-element reduction, which is why grid size barely moves the runtime.

    An intensity-mode tile set is matched on brightness only, so monochrome tile art is
    driven purely by tone. A color-mode set is matched on full color distance.

    `detail_weight` mixes in a texture term: cells with high internal contrast are pulled
    toward tiles that are themselves busy (edges, wedges), and flat cells toward flat
    tiles. At 0.0 the match is purely photometric.
    """
    if tile_set.mode == "intensity":
        if space == "CIE-Lab":
            # L* is perceptual lightness on [0,100]; rescale so the weighting below
            # stays comparable with the RGB branch.
            cell_feat = _to_match_space(cell_mean, space)[..., :1] * 2.55
            key_feat = _to_match_space(tile_set.keys, space)[..., :1] * 2.55
        else:
            cell_feat = (cell_mean @ LUMA)[..., None]
            key_feat = (tile_set.keys @ LUMA)[..., None]
    else:
        cell_feat = _to_match_space(cell_mean, space)
        key_feat = _to_match_space(tile_set.keys, space)

    # (n_rows, n_cols, 1, F) - (1, 1, N, F) -> (n_rows, n_cols, N)
    diff = cell_feat[:, :, None, :] - key_feat[None, None, :, :]
    dist = np.einsum("rcnf,rcnf->rcn", diff, diff)
    # Normalize to roughly [0,1] so `detail_weight` behaves like a proper mixing factor.
    dist /= 255.0**2 * cell_feat.shape[-1]

    if detail_weight > 0 and cell_std is not None:
        tile_std = _tile_detail(tile_set)
        cell_detail = np.clip(cell_std / 96.0, 0, 1)
        detail_diff = cell_detail[:, :, None] - tile_std[None, None, :]
        dist = (1.0 - detail_weight) * dist + detail_weight * detail_diff**2

    return dist.argmin(axis=-1).astype(np.int32)


def _tile_detail(tile_set: TileSet) -> np.ndarray:
    """(N,) normalized internal contrast of each tile.

    Recomputed per call rather than cached: it is a reduction over N small tiles, so it
    is negligible beside the per-cell work it feeds, and it only runs at all when
    texture-aware matching is switched on.
    """
    luma = tile_set.tiles.astype(np.float32) @ LUMA
    return np.clip(luma.reshape(len(luma), -1).std(axis=1) / 96.0, 0, 1)


# --------------------------------------------------------------------------------------
# Step 3b: reconstruction -- one gather, one transpose, one reshape
# --------------------------------------------------------------------------------------


def reconstruct(
    indices: np.ndarray,
    tile_set: TileSet,
    cell_mean: np.ndarray,
    color_mode: str = "Modulate",
    strength: float = 0.85,
) -> np.ndarray:
    """Paste the chosen tile into every cell and stitch the grid back together.

    `tiles[indices]` gathers an entire (n_rows, n_cols, T, T, 3) array of tile pixels in
    one indexing operation. Interleaving the cell axes with the within-tile axes then
    collapses to the output image -- the inverse of the reshape used to build the grid.

    Color modes (all applied before the reshape, broadcasting the cell color over the
    tile's pixels):
      "Tile only" -- the raw tile, so intensity sets render as monochrome tile art.
      "Blend"     -- tile * (1-s) + cell_color * s. At s=1 this reproduces the segmented
                     image exactly, which makes it the upper bound on similarity score.
      "Modulate"  -- keeps the tile's luminance and takes the cell's chrominance:
                     tile_luma + (cell - luma(cell)). This is a YCbCr recombination.
                     Because LUMA sums to 1, the chroma offset contributes zero net
                     luminance by construction, so the output carries exactly the
                     *tile's* brightness -- tinting never darkens or lightens it. The
                     cell's own brightness is then preserved only to the extent that the
                     classifier picked a tile matching it, which is what tile keys being
                     measured means (see tiles._finalize). The multiplicative form,
                     tile * cell / luma(cell), is not equivalent: it clips on saturated
                     colors and darkens the mosaic.
    """
    tiles = tile_set.tiles.astype(np.float32)
    gathered = tiles[indices]                       # (n_rows, n_cols, T, T, 3)

    if color_mode != "Tile only" and strength > 0:
        cell = cell_mean[:, :, None, None, :]       # (n_rows, n_cols, 1, 1, 3)
        if color_mode == "Blend":
            target = cell
        else:
            tile_luma = (gathered @ LUMA)[..., None]
            cell_luma = (cell_mean @ LUMA)[:, :, None, None, None]
            target = tile_luma + (cell - cell_luma)
        gathered = gathered * (1.0 - strength) + target * strength

    n_rows, n_cols, t = gathered.shape[0], gathered.shape[1], gathered.shape[2]
    out = gathered.transpose(0, 2, 1, 3, 4).reshape(n_rows * t, n_cols * t, 3)
    return np.clip(out, 0, 255).astype(np.uint8)


def _fit_tile_px(tile_px: int, n_rows: int, n_cols: int) -> int:
    """Shrink the tile resolution if the reconstruction would be absurdly large."""
    longest = max(n_rows, n_cols)
    if longest * tile_px <= MAX_OUTPUT_PX:
        return tile_px
    return max(4, MAX_OUTPUT_PX // longest)


def segmented_view(cell_mean: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    """The classified grid itself: cell means blown back up to the original size."""
    cells = np.clip(cell_mean, 0, 255).astype(np.uint8)
    h, w = shape
    return cv2.resize(cells, (w, h), interpolation=cv2.INTER_NEAREST)


# --------------------------------------------------------------------------------------
# Grid operations -- the part vectorization actually changes
# --------------------------------------------------------------------------------------
#
# These two functions are the subject of the performance comparison. They take an image
# already cropped to the grid and do the three grid operations: measure every cell,
# classify it, and paste in its tile. Everything around them (decoding, resizing,
# cropping, scoring) is shared by both implementations, so timing the whole pipeline
# would dilute the difference with work that vectorization does not touch.


@dataclass
class GridOutput:
    cell_mean: np.ndarray
    indices: np.ndarray
    mosaic_full: np.ndarray
    tile_px: int


def grid_ops(
    cropped: np.ndarray,
    n_rows: int,
    n_cols: int,
    cell: int,
    tile_set_name: str,
    tile_px: int,
    color_mode: str = "Modulate",
    strength: float = 0.85,
    match_space: str = "RGB",
    detail_weight: float = 0.0,
    quantize_k: int = 0,
) -> GridOutput:
    """Vectorized grid operations: statistics, classification, reconstruction."""
    tile_set = get_tile_set(tile_set_name, int(tile_px))

    cell_mean, cell_std = cell_statistics(
        cropped, n_rows, n_cols, cell, with_std=detail_weight > 0
    )

    if quantize_k and quantize_k > 1:
        # Quantizing the cell means rather than the full image gives the same flattened
        # palette for a fraction of the k-means cost.
        cell_mean = quantize_colors(cell_mean.astype(np.uint8), int(quantize_k)).astype(np.float32)

    indices = match_tiles(cell_mean, cell_std, tile_set, match_space, detail_weight)

    effective_px = _fit_tile_px(int(tile_px), n_rows, n_cols)
    if effective_px != tile_set.tile_size:
        tile_set = get_tile_set(tile_set_name, effective_px)

    mosaic_full = reconstruct(indices, tile_set, cell_mean, color_mode, strength)
    return GridOutput(cell_mean, indices, mosaic_full, effective_px)


def grid_ops_loop(
    cropped: np.ndarray,
    n_rows: int,
    n_cols: int,
    cell: int,
    tile_set_name: str,
    tile_px: int,
    color_mode: str = "Modulate",
    strength: float = 0.85,
    match_space: str = "RGB",
    detail_weight: float = 0.0,
    quantize_k: int = 0,
) -> GridOutput:
    """Loop-based equivalent of `grid_ops`, kept for the performance comparison.

    Same algorithm and same output, but the grid is walked cell by cell in Python and
    the nearest tile is found with an inner loop over the tile set: the straightforward
    way to express the pipeline, and the baseline the vectorized path is measured
    against. Only the RGB / no-detail path is implemented, since that is what the
    benchmark exercises.
    """
    tile_set = get_tile_set(tile_set_name, int(tile_px))

    src = cropped.astype(np.float32)
    keys = tile_set.keys
    intensity = tile_set.mode == "intensity"
    key_feat = keys @ LUMA if intensity else keys

    cell_mean = np.zeros((n_rows, n_cols, 3), dtype=np.float32)
    indices = np.zeros((n_rows, n_cols), dtype=np.int32)

    for r in range(n_rows):
        for c in range(n_cols):
            block = src[r * cell : (r + 1) * cell, c * cell : (c + 1) * cell]
            mean = block.reshape(-1, 3).mean(axis=0)
            cell_mean[r, c] = mean

            feat = float(mean @ LUMA) if intensity else mean
            best_idx, best_dist = 0, float("inf")
            for t in range(len(keys)):
                delta = feat - key_feat[t]
                dist = float(delta * delta) if intensity else float((delta * delta).sum())
                if dist < best_dist:
                    best_idx, best_dist = t, dist
            indices[r, c] = best_idx

    if quantize_k and quantize_k > 1:
        cell_mean = quantize_colors(cell_mean.astype(np.uint8), int(quantize_k)).astype(np.float32)

    effective_px = _fit_tile_px(int(tile_px), n_rows, n_cols)
    if effective_px != tile_set.tile_size:
        tile_set = get_tile_set(tile_set_name, effective_px)

    t_px = tile_set.tile_size
    mosaic_full = np.zeros((n_rows * t_px, n_cols * t_px, 3), dtype=np.uint8)

    for r in range(n_rows):
        for c in range(n_cols):
            tile = tile_set.tiles[indices[r, c]].astype(np.float32)
            if color_mode != "Tile only" and strength > 0:
                col = cell_mean[r, c]
                if color_mode == "Blend":
                    target = col
                else:
                    target = (tile @ LUMA)[..., None] + (col - float(col @ LUMA))
                tile = tile * (1.0 - strength) + target * strength
            mosaic_full[r * t_px : (r + 1) * t_px, c * t_px : (c + 1) * t_px] = np.clip(
                tile, 0, 255
            ).astype(np.uint8)

    return GridOutput(cell_mean, indices, mosaic_full, effective_px)


def scoring_render(
    out: GridOutput,
    cell: int,
    tile_set_name: str,
    color_mode: str,
    strength: float,
) -> np.ndarray:
    """Re-render the mosaic with one tile pixel per source pixel, for the metrics.

    The displayed mosaic uses whatever tile resolution the user picked, so its size
    rarely matches the source. Scoring it would then mean resampling one of the two --
    and that quietly corrupts the comparison, because the *kind* of resampling changes
    with the grid: coarse grids need the mosaic enlarged (which blurs it, flattering the
    score) while fine grids need it shrunk (which does not). Scores stop being comparable
    across grid sizes, and the trend the metric exists to show can invert.

    Rendering the same tile choices at `cell` pixels per tile sidesteps this entirely:
    the result is exactly the source's shape, so every grid size is scored at the same
    fixed resolution with no resampling anywhere. It costs one extra reconstruction,
    skipped when the displayed mosaic already happens to be the right size.
    """
    if out.tile_px == cell:
        return out.mosaic_full

    tile_set = get_tile_set(tile_set_name, cell)
    return reconstruct(out.indices, tile_set, out.cell_mean, color_mode, strength)


# --------------------------------------------------------------------------------------
# Full pipeline
# --------------------------------------------------------------------------------------


def _build(grid_fn, image, grid_size, tile_set_name, tile_px, color_mode, strength,
           match_space, detail_weight, quantize_k, long_edge) -> MosaicResult:
    """Shared pipeline body; `grid_fn` selects the vectorized or loop-based grid ops."""
    start = time.perf_counter()

    work = to_working_resolution(image, long_edge)
    cropped, n_rows, n_cols, cell = crop_to_grid(work, int(grid_size))

    out = grid_fn(cropped, n_rows, n_cols, cell, tile_set_name, tile_px,
                  color_mode, strength, match_space, detail_weight, quantize_k)

    elapsed = time.perf_counter() - start

    h, w = cropped.shape[:2]
    return MosaicResult(
        original=cropped,
        segmented=segmented_view(out.cell_mean, (h, w)),
        mosaic=scoring_render(out, cell, tile_set_name, color_mode, strength),
        mosaic_full=out.mosaic_full,
        indices=out.indices,
        n_rows=n_rows,
        n_cols=n_cols,
        cell_px=cell,
        tile_px=out.tile_px,
        elapsed=elapsed,
    )


def build_mosaic(
    image: np.ndarray,
    grid_size: int = 48,
    tile_set_name: str = "Halftone Dots",
    tile_px: int = 16,
    color_mode: str = "Modulate",
    strength: float = 0.85,
    match_space: str = "RGB",
    detail_weight: float = 0.0,
    quantize_k: int = 0,
    long_edge: int = 768,
) -> MosaicResult:
    """Run the whole vectorized pipeline and return every intermediate the UI shows."""
    return _build(grid_ops, image, grid_size, tile_set_name, tile_px, color_mode,
                  strength, match_space, detail_weight, quantize_k, long_edge)


def build_mosaic_loop(
    image: np.ndarray,
    grid_size: int = 48,
    tile_set_name: str = "Halftone Dots",
    tile_px: int = 16,
    color_mode: str = "Modulate",
    strength: float = 0.85,
    match_space: str = "RGB",
    detail_weight: float = 0.0,
    quantize_k: int = 0,
    long_edge: int = 768,
) -> MosaicResult:
    """Loop-based counterpart of `build_mosaic`, used by the performance comparison."""
    return _build(grid_ops_loop, image, grid_size, tile_set_name, tile_px, color_mode,
                  strength, match_space, detail_weight, quantize_k, long_edge)
