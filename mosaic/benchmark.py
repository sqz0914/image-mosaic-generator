"""Timing harness: how the pipeline scales with grid size, vectorized vs loop-based.

The Performance tab in `app.py` is the only consumer; it calls `run` and
`stage_breakdown` for the tables and `plot` for the chart.
"""

from __future__ import annotations

import time
from dataclasses import dataclass

import cv2
import numpy as np

from .core import build_mosaic, grid_ops, grid_ops_loop
from .metrics import score
from .preprocess import crop_to_grid, to_working_resolution

DEFAULT_GRID_SIZES = (16, 32, 64, 128)
DEFAULT_TILE_SET = "Halftone Dots"
DEFAULT_TILE_PX = 16

# Categorical slots 1 and 2 of the reference palette, validated for this pair
# (CVD dE 24.7, normal-vision dE 33.6, both >= 3:1 on the light surface).
C_VECTORIZED = "#2a78d6"
C_LOOP = "#eb6834"
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_MUTED = "#52514e"
GRID_INK = "#dcdbd6"


@dataclass
class Row:
    grid_size: int
    n_cells: int
    vectorized_ms: float     # grid operations only
    vectorized_sd: float
    loop_ms: float | None    # grid operations only
    loop_sd: float | None
    pipeline_ms: float       # end-to-end, including preprocessing and resizing

    @property
    def speedup(self) -> float | None:
        if not self.loop_ms or not self.vectorized_ms:
            return None
        return self.loop_ms / self.vectorized_ms


def _time(fn, repeats: int, *args, **kwargs) -> tuple[float, float]:
    """Return (mean_ms, stdev_ms) over `repeats` runs, after one warmup run.

    The warmup matters: it pays for tile-set generation (which is cached) and for
    NumPy's first-touch allocation, neither of which belongs in the per-call cost.
    """
    fn(*args, **kwargs)

    samples = []
    for _ in range(repeats):
        start = time.perf_counter()
        fn(*args, **kwargs)
        samples.append((time.perf_counter() - start) * 1000.0)

    return float(np.mean(samples)), float(np.std(samples))


def run(
    image: np.ndarray,
    grid_sizes: tuple[int, ...] = DEFAULT_GRID_SIZES,
    repeats: int = 5,
    loop_repeats: int = 2,
    include_loop: bool = True,
    tile_set_name: str = DEFAULT_TILE_SET,
    tile_px: int = DEFAULT_TILE_PX,
    long_edge: int = 768,
    **kwargs,
) -> list[Row]:
    """Benchmark the grid operations across `grid_sizes`, vectorized vs loop-based.

    Cropping is done once per grid size and excluded from the timed region: decoding and
    resizing are identical for both implementations, and at coarse grids they cost more
    than the grid work itself, so including them would understate the difference between
    the two by mixing in work neither one can avoid. The end-to-end pipeline time is
    reported separately, since that is what a user actually waits for.
    """
    work = to_working_resolution(image, long_edge)

    rows = []
    for g in grid_sizes:
        cropped, n_rows, n_cols, cell = crop_to_grid(work, g)
        args = (cropped, n_rows, n_cols, cell, tile_set_name, tile_px)

        vec_mean, vec_sd = _time(grid_ops, repeats, *args, **kwargs)

        loop_mean = loop_sd = None
        if include_loop:
            loop_mean, loop_sd = _time(grid_ops_loop, loop_repeats, *args, **kwargs)

        pipeline_ms, _ = _time(
            build_mosaic, repeats, image, grid_size=g, tile_set_name=tile_set_name,
            tile_px=tile_px, long_edge=long_edge, **kwargs
        )

        rows.append(Row(g, n_rows * n_cols, vec_mean, vec_sd, loop_mean, loop_sd, pipeline_ms))

    return rows


def as_table(rows: list[Row]) -> list[list[str]]:
    """Render results as rows for a Gradio dataframe."""
    out = []
    for r in rows:
        out.append([
            f"{r.grid_size}",
            f"{r.n_cells:,}",
            f"{r.vectorized_ms:.2f}",
            f"{r.loop_ms:.1f}" if r.loop_ms else "—",
            f"{r.speedup:.2f}×" if r.speedup else "—",
            f"{r.pipeline_ms:.1f}",
        ])
    return out


TABLE_HEADERS = [
    "Grid", "Squares", "Vectorized (ms)", "Loop (ms)", "Speedup", "Whole pipeline (ms)"
]
QUALITY_HEADERS = ["Grid", "Squares", "MSE", "SSIM"]


def quality(
    image: np.ndarray,
    grid_sizes: tuple[int, ...] = DEFAULT_GRID_SIZES,
    **mosaic_kwargs,
) -> list[list[str]]:
    """How closely the mosaic matches the original, across the same grid sizes.

    Takes the Mosaic tab's own settings so the sweep describes the style on screen, and
    reports both metrics because they disagree: MSE falls as the grid gets finer, while
    SSIM can move the other way for a high-contrast tile set, whose texture it reads as
    structural mismatch.
    """
    rows = []
    for g in grid_sizes:
        r = build_mosaic(image, grid_size=g, **mosaic_kwargs)
        s = score(r.original, r.mosaic)
        rows.append([
            f"{g}",
            f"{r.n_rows * r.n_cols:,}",
            f"{s.mse:,.0f}",
            f"{s.ssim:.4f}",
        ])
    return rows


def stage_breakdown(
    image: np.ndarray,
    grid_sizes: tuple[int, ...] = DEFAULT_GRID_SIZES,
    repeats: int = 5,
    tile_set_name: str = DEFAULT_TILE_SET,
    tile_px: int = DEFAULT_TILE_PX,
    long_edge: int = 768,
) -> str:
    """Per-stage timings for the vectorized path, as a Markdown table.

    This is what explains the scaling. The three stages answer to different things:
    statistics sweeps every *pixel* once regardless of how the grid is cut, classification
    is proportional to *cells* but so cheap it never matters, and reconstruction is
    proportional to *output pixels* and dominates everything else.
    """
    from .core import cell_statistics, match_tiles, reconstruct
    from .tiles import get_tile_set

    work = to_working_resolution(image, long_edge)
    tile_set = get_tile_set(tile_set_name, tile_px)

    head = (
        "| Grid | Squares | Measuring squares (ms) | Picking tiles (ms) | Rebuilding image (ms) |\n"
        "|-----:|--------:|-----------------------:|-------------------:|----------------------:|\n"
    )
    body = ""
    for g in grid_sizes:
        cropped, n_rows, n_cols, cell = crop_to_grid(work, g)
        mean, _ = cell_statistics(cropped, n_rows, n_cols, cell, with_std=False)
        idx = match_tiles(mean, None, tile_set)

        stats_ms, _ = _time(cell_statistics, repeats, cropped, n_rows, n_cols, cell, False)
        match_ms, _ = _time(match_tiles, repeats, mean, None, tile_set)
        recon_ms, _ = _time(reconstruct, repeats, idx, tile_set, mean)

        body += (
            f"| {g} | {n_rows * n_cols:,} | {stats_ms:.2f} | {match_ms:.3f} | {recon_ms:.2f} |\n"
        )

    return head + body


def plot(rows: list[Row]) -> np.ndarray:
    """Two panels: absolute cost per grid size, and the speedup that buys.

    Time is on a log scale because the two implementations differ by two orders of
    magnitude -- on a linear axis the vectorized series would be flat against zero and
    its own scaling would be invisible.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    grids = [r.grid_size for r in rows]
    vec = [r.vectorized_ms for r in rows]
    has_loop = all(r.loop_ms for r in rows)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.2), dpi=140)
    fig.patch.set_facecolor(SURFACE)

    for ax in (ax1, ax2):
        ax.set_facecolor(SURFACE)
        ax.grid(True, color=GRID_INK, linewidth=0.8, alpha=0.9)
        ax.set_axisbelow(True)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        for side in ("left", "bottom"):
            ax.spines[side].set_color(GRID_INK)
        ax.tick_params(colors=INK_MUTED, labelsize=9)
        ax.set_xscale("log", base=2)
        ax.set_xticks(grids)
        ax.set_xticklabels([str(g) for g in grids])
        ax.set_xlabel("Grid size (cells across)", color=INK_MUTED, fontsize=9)

    # --- Panel 1: wall time -------------------------------------------------------
    ax1.set_yscale("log")
    ax1.plot(grids, vec, color=C_VECTORIZED, lw=2, marker="o", ms=7,
             label="Vectorized", zorder=3)
    if has_loop:
        loop = [r.loop_ms for r in rows]
        ax1.plot(grids, loop, color=C_LOOP, lw=2, marker="o", ms=7,
                 label="Loop-based", zorder=3)
        # Direct labels at the right-hand end: identity without a round trip to a legend.
        ax1.annotate("Loop-based", (grids[-1], loop[-1]), textcoords="offset points",
                     xytext=(-6, 10), ha="right", color=C_LOOP, fontsize=9, fontweight="bold")
    ax1.annotate("Vectorized", (grids[-1], vec[-1]), textcoords="offset points",
                 xytext=(-6, -16), ha="right", color=C_VECTORIZED, fontsize=9,
                 fontweight="bold")

    ax1.set_ylabel("Grid-operation time (ms, log scale)", color=INK_MUTED, fontsize=9)
    ax1.set_title("Grid-operation time by grid size", color=INK, fontsize=11,
                  fontweight="bold", loc="left", pad=12)
    if has_loop:
        ax1.legend(frameon=False, fontsize=9, labelcolor=INK_MUTED, loc="center left")

    # --- Panel 2: speedup ---------------------------------------------------------
    if has_loop:
        speed = [r.speedup for r in rows]
        ax2.plot(grids, speed, color=C_VECTORIZED, lw=2, marker="o", ms=7, zorder=3)
        best = int(np.argmax(speed))
        # Label only the peak, not every point. Right-align when the peak is the last
        # point, so the text grows inward instead of off the edge of the axes.
        at_end = best == len(grids) - 1
        ax2.annotate(f"{speed[best]:.1f}× faster", (grids[best], speed[best]),
                     textcoords="offset points", xytext=(-4 if at_end else 0, 12),
                     ha="right" if at_end else "center",
                     color=C_VECTORIZED, fontsize=10, fontweight="bold")
        ax2.set_ylim(0, max(speed) * 1.25)
        ax2.set_ylabel("Loop time / vectorized time", color=INK_MUTED, fontsize=9)
        ax2.set_title("Speedup from vectorization", color=INK, fontsize=11,
                      fontweight="bold", loc="left", pad=12)
    else:
        ax2.axis("off")

    fig.tight_layout()
    fig.canvas.draw()

    buf = np.asarray(fig.canvas.buffer_rgba())[:, :, :3].copy()
    plt.close(fig)

    return buf
