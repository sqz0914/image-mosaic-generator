"""Correctness tests for the mosaic pipeline."""

from __future__ import annotations

import cv2
import numpy as np
import pytest

from mosaic import TILE_SETS, build_mosaic, build_mosaic_loop, get_tile_set
from mosaic.core import cell_statistics, reconstruct, match_tiles
from mosaic.metrics import score
from mosaic.preprocess import crop_to_grid, quantize_colors, to_working_resolution


@pytest.fixture(scope="module")
def sample() -> np.ndarray:
    """A deterministic image with flat regions, gradients and hard edges."""
    h, w = 480, 640
    yy, xx = np.mgrid[0:h, 0:w]
    img = np.stack([xx / w * 255, yy / h * 255, (xx + yy) / (w + h) * 255], -1).astype(np.uint8)
    cv2.circle(img, (200, 200), 120, (250, 40, 40), -1)
    cv2.rectangle(img, (400, 250), (600, 430), (20, 200, 90), -1)
    return img


# ---------------------------------------------------------------- preprocessing


@pytest.mark.parametrize("n_cols", [7, 16, 31, 32, 64, 97])
def test_crop_to_grid_divides_exactly(sample, n_cols):
    """The crop must leave an image that splits into whole square cells."""
    cropped, n_rows, cols, cell = crop_to_grid(sample, n_cols)
    h, w = cropped.shape[:2]

    assert h == n_rows * cell
    assert w == cols * cell
    assert h % cell == 0 and w % cell == 0


def test_crop_discards_less_than_one_cell(sample):
    """Cropping is 'slight': never more than one cell's worth per axis."""
    work = to_working_resolution(sample, 768)
    cropped, _, _, cell = crop_to_grid(work, 37)

    assert work.shape[0] - cropped.shape[0] < cell
    assert work.shape[1] - cropped.shape[1] < cell


def test_working_resolution_preserves_aspect(sample):
    out = to_working_resolution(sample, 512)
    assert max(out.shape[:2]) == 512
    assert out.shape[1] / out.shape[0] == pytest.approx(sample.shape[1] / sample.shape[0], rel=0.01)


@pytest.mark.parametrize(
    "img",
    [
        np.full((100, 100), 128, np.uint8),              # grayscale
        np.full((100, 100, 4), 200, np.uint8),           # RGBA
        np.zeros((640, 120, 3), np.uint8),               # extreme portrait
        np.zeros((120, 640, 3), np.uint8),               # extreme landscape
    ],
)
def test_awkward_inputs_survive(img):
    result = build_mosaic(img, grid_size=24, tile_px=8)
    assert result.mosaic.shape == result.original.shape
    assert result.mosaic.dtype == np.uint8


def test_quantize_reduces_palette(sample):
    out = quantize_colors(sample, k=8)
    assert len(np.unique(out.reshape(-1, 3), axis=0)) <= 8


# ---------------------------------------------------------------- grid statistics


def test_cell_statistics_match_manual_means():
    """The reshape trick must agree with an explicit per-cell average."""
    rng = np.random.default_rng(3)
    img = rng.integers(0, 256, (48, 64, 3), dtype=np.uint8)
    n_rows, n_cols, cell = 6, 8, 8

    mean, std = cell_statistics(img, n_rows, n_cols, cell)

    for r in range(n_rows):
        for c in range(n_cols):
            block = img[r * cell : (r + 1) * cell, c * cell : (c + 1) * cell]
            assert mean[r, c] == pytest.approx(block.reshape(-1, 3).mean(0), abs=1e-3)

    assert mean.shape == (n_rows, n_cols, 3)
    assert std.shape == (n_rows, n_cols)


def test_flat_cells_have_zero_variance():
    img = np.full((32, 32, 3), 77, np.uint8)
    mean, std = cell_statistics(img, 4, 4, 8)

    assert np.allclose(mean, 77)
    assert np.allclose(std, 0)


# ---------------------------------------------------------------- tile sets


@pytest.mark.parametrize("name", TILE_SETS)
@pytest.mark.parametrize("tile_px", [8, 16])
def test_tile_sets_well_formed(name, tile_px):
    ts = get_tile_set(name, tile_px)

    assert ts.tiles.shape == (ts.n_tiles, tile_px, tile_px, 3)
    assert ts.tiles.dtype == np.uint8
    assert ts.keys.shape == (ts.n_tiles, 3)
    assert ts.mode in ("intensity", "color")
    # Keys are sorted by brightness, which the intensity matcher relies on.
    assert np.all(np.diff(ts.luma_keys) >= 0)


@pytest.mark.parametrize("name", TILE_SETS)
def test_tile_sets_span_the_tonal_range(name):
    """A set that cannot reach dark or bright cells silently caps mosaic quality."""
    luma = get_tile_set(name, 16).luma_keys

    assert luma.min() < 40, "no tile dark enough for shadow cells"
    assert luma.max() > 215, "no tile bright enough for highlight cells"
    assert np.diff(luma).max() < 55, "gap in the ramp leaves a tone unrepresentable"


# ---------------------------------------------------------------- reconstruction


def test_reconstruct_shape_and_placement():
    """Each cell must receive its own tile at the right offset."""
    ts = get_tile_set("Halftone Dots", 8)
    indices = np.array([[0, 3], [5, 1]], dtype=np.int32)
    cell_mean = np.zeros((2, 2, 3), np.float32)

    out = reconstruct(indices, ts, cell_mean, color_mode="Tile only")

    assert out.shape == (2 * 8, 2 * 8, 3)
    for r in range(2):
        for c in range(2):
            block = out[r * 8 : (r + 1) * 8, c * 8 : (c + 1) * 8]
            assert np.array_equal(block, ts.tiles[indices[r, c]])


def test_blend_at_full_strength_reproduces_cell_colors():
    """Blend at full strength paints flat cell colors, reproducing the segmented view."""
    ts = get_tile_set("Geometric Blocks", 8)
    cell_mean = np.array([[[200.0, 100.0, 50.0]]], np.float32)
    indices = np.zeros((1, 1), np.int32)

    out = reconstruct(indices, ts, cell_mean, color_mode="Blend", strength=1.0)

    assert np.allclose(out, np.array([200, 100, 50]), atol=1)


def test_modulate_preserves_cell_brightness():
    """Tinting must not shift overall brightness, only hue."""
    img = np.full((64, 64, 3), 0, np.uint8)
    img[:, :32] = (200, 60, 60)
    img[:, 32:] = (60, 60, 200)

    plain = build_mosaic(img, grid_size=8, tile_set_name="Halftone Dots",
                         tile_px=16, color_mode="Tile only")
    tinted = build_mosaic(img, grid_size=8, tile_set_name="Halftone Dots",
                          tile_px=16, color_mode="Modulate", strength=1.0)

    luma = np.array([0.299, 0.587, 0.114], np.float32)
    a, _ = cell_statistics(plain.mosaic_full, plain.n_rows, plain.n_cols, plain.tile_px)
    b, _ = cell_statistics(tinted.mosaic_full, tinted.n_rows, tinted.n_cols, tinted.tile_px)

    # Tinting changes hue, so compare brightness only.
    assert np.abs(a @ luma - b @ luma).mean() < 6


# ---------------------------------------------------------------- vectorized == loop


@pytest.mark.parametrize("grid_size", [16, 32])
@pytest.mark.parametrize("tile_set_name", ["Halftone Dots", "LEGO Studs"])
@pytest.mark.parametrize("color_mode", ["Tile only", "Modulate"])
def test_loop_and_vectorized_agree(sample, grid_size, tile_set_name, color_mode):
    """The fast path must be a pure optimization, not a different algorithm."""
    fast = build_mosaic(sample, grid_size=grid_size, tile_set_name=tile_set_name,
                        tile_px=8, color_mode=color_mode)
    slow = build_mosaic_loop(sample, grid_size=grid_size, tile_set_name=tile_set_name,
                             tile_px=8, color_mode=color_mode)

    assert (fast.n_rows, fast.n_cols, fast.cell_px) == (slow.n_rows, slow.n_cols, slow.cell_px)
    assert np.array_equal(fast.indices, slow.indices)
    # float32 reductions associate differently between the two, so allow one level.
    assert np.abs(fast.mosaic_full.astype(int) - slow.mosaic_full.astype(int)).max() <= 1


# ---------------------------------------------------------------- metrics


def test_identical_images_score_perfectly(sample):
    s = score(sample, sample)
    assert s.mse == 0
    assert s.ssim == pytest.approx(1.0)


@pytest.mark.parametrize("color_mode", ["Tile only", "Modulate", "Blend"])
def test_finer_grids_lower_the_error(sample, color_mode):
    """The core sanity check: more cells must mean a closer reconstruction.

    MSE is asserted rather than SSIM. For a textured tile set SSIM *falls* as the grid
    gets finer, because smaller cells pack more tile detail and more cell edges into
    SSIM's 7x7 window -- a property of the metric, not a defect in the mosaic.
    """
    errors = [
        score(r.original, r.mosaic).mse
        for r in (
            build_mosaic(sample, grid_size=g, tile_set_name="Halftone Dots",
                         tile_px=16, color_mode=color_mode)
            for g in (16, 32, 64, 128)
        )
    ]
    assert errors == sorted(errors, reverse=True), f"MSE should fall with grid size: {errors}"


def test_segmented_view_tracks_the_original(sample):
    """The segmented view is flat cell colors, so it must beat any tile substitution."""
    r = build_mosaic(sample, grid_size=48, tile_set_name="LEGO Studs", tile_px=8)

    assert score(r.original, r.segmented).mse <= score(r.original, r.mosaic).mse


def test_score_rejects_mismatched_shapes(sample):
    with pytest.raises(ValueError):
        score(sample, sample[:100])


# ---------------------------------------------------------------- guard rails


def test_output_size_is_capped():
    """A huge grid x tile combination must not produce a gigapixel image."""
    img = np.zeros((768, 768, 3), np.uint8)
    r = build_mosaic(img, grid_size=128, tile_px=32)

    assert max(r.mosaic_full.shape[:2]) <= 2048


def test_match_tiles_returns_valid_indices(sample):
    ts = get_tile_set("ASCII Art", 8)
    cropped, n_rows, n_cols, cell = crop_to_grid(to_working_resolution(sample), 32)
    mean, std = cell_statistics(cropped, n_rows, n_cols, cell)

    for space in ("RGB", "CIE-Lab"):
        idx = match_tiles(mean, std, ts, space=space, detail_weight=0.3)
        assert idx.shape == (n_rows, n_cols)
        assert idx.min() >= 0 and idx.max() < ts.n_tiles
