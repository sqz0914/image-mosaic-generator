"""Gradio interface for the interactive image mosaic generator.

Run locally with hot reload:   gradio app.py
Or as a plain script:          python app.py
"""

from __future__ import annotations

import gradio as gr
import numpy as np

from mosaic import TILE_SETS, build_mosaic, contact_sheet, get_tile_set
from mosaic import benchmark as bench
from mosaic.core import COLOR_MODES, MATCH_SPACES
from mosaic.metrics import score

IDLE_MESSAGE = "_Upload an image, then press **Generate mosaic**._"
BENCH_IDLE_MESSAGE = "_Press **Generate mosaic** on the Mosaic tab to benchmark it._"


# --------------------------------------------------------------------------------------
# Callbacks
# --------------------------------------------------------------------------------------


def generate(
    image: np.ndarray | None,
    grid_size: int,
    tile_set_name: str,
    tile_px: int,
    color_mode: str,
    strength: float,
    match_space: str,
    detail_weight: float,
    use_quantization: bool,
    quantize_k: int,
):
    """Build the mosaic and everything shown beside it.

    The trailing return value is the visibility of the results panel: it stays hidden
    until a run actually produces something to show.
    """
    # Pressing Generate with an empty input is a prompt, not an error popup.
    if image is None:
        return None, None, None, [], IDLE_MESSAGE, gr.update(visible=False)

    result = build_mosaic(
        image,
        grid_size=int(grid_size),
        tile_set_name=tile_set_name,
        tile_px=int(tile_px),
        color_mode=color_mode,
        strength=float(strength),
        match_space=match_space,
        detail_weight=float(detail_weight),
        quantize_k=int(quantize_k) if use_quantization else 0,
    )

    s = score(result.original, result.mosaic)

    summary = (
        f"**{result.n_cols} × {result.n_rows} grid** "
        f"({result.n_rows * result.n_cols:,} squares) &nbsp;·&nbsp; "
        f"{result.cell_px}px squares → {result.tile_px}px tiles &nbsp;·&nbsp; "
        f"output {result.mosaic_full.shape[1]} × {result.mosaic_full.shape[0]} px"
        f" &nbsp;·&nbsp; **{result.elapsed * 1000:.0f} ms**\n\n"
        f"Similarity to the original: **MSE {s.mse:,.0f}** (lower is better) "
        f"&nbsp;·&nbsp; **SSIM {s.ssim:.4f}** (higher is better; 1.0 is identical)"
    )

    return (
        result.original, result.segmented, result.mosaic_full,
        summary, gr.update(visible=True),
    )


def preview_tiles(tile_set_name: str, tile_px: int):
    ts = get_tile_set(tile_set_name, int(tile_px))
    caption = f"**{ts.name}** — {ts.n_tiles} tiles at {ts.tile_size}px. {ts.description}"
    return contact_sheet(ts), caption


def run_benchmark(
    image: np.ndarray | None,
    tile_set_name: str,
    tile_px: int,
    color_mode: str,
    strength: float,
    match_space: str,
    detail_weight: float,
    use_quantization: bool,
    quantize_k: int,
):
    """Sweep grid sizes for both quality and speed, using the Mosaic tab's settings.

    Runs off the same click as `generate` and takes the same settings, so both tables
    describe the mosaic on screen. Grid size is the swept variable and so comes from the
    benchmark's own range rather than the Mosaic slider.
    """
    if image is None:
        return [], [], None, "", BENCH_IDLE_MESSAGE, gr.update(visible=False)

    tile_px = int(tile_px)
    settings = dict(
        tile_set_name=tile_set_name,
        tile_px=tile_px,
        color_mode=color_mode,
        strength=float(strength),
        match_space=match_space,
        detail_weight=float(detail_weight),
        quantize_k=int(quantize_k) if use_quantization else 0,
    )

    quality_rows = bench.quality(image, **settings)
    rows = bench.run(image, tile_set_name=tile_set_name, tile_px=tile_px)

    analysis = (
        "Timings cover the **grid operations only** (measuring squares → picking tiles → "
        "rebuilding the image). Decoding, resizing and cropping are identical for both "
        "implementations, so they are measured separately in the last column.\n\n"
        + bench.stage_breakdown(image, tile_set_name=tile_set_name, tile_px=tile_px)
    )
    status = (
        f"Measured on a {image.shape[1]} × {image.shape[0]} image at a 768px working "
        f"resolution, {tile_set_name} tiles at {tile_px}px."
    )

    return (
        quality_rows, bench.as_table(rows), bench.plot(rows), analysis, status,
        gr.update(visible=True),
    )


# --------------------------------------------------------------------------------------
# Interface
# --------------------------------------------------------------------------------------

with gr.Blocks(title="Image Mosaic Generator") as demo:
    gr.Markdown(
        "# Interactive Image Mosaic Generator\n"
        "Divide an image into a grid, classify every cell, and rebuild it out of tiles."
    )

    with gr.Tab("Mosaic"):
        with gr.Row():
            with gr.Column(scale=1):
                image_in = gr.Image(
                    label="Input image", type="numpy", height=260, sources=["upload"]
                )

                grid_size = gr.Slider(8, 128, value=48, step=4, label="Grid size (cells across)")
                tile_set_name = gr.Dropdown(
                    TILE_SETS, value="Halftone Dots", label="Tile set"
                )
                tile_px = gr.Slider(8, 32, value=16, step=4, label="Tile resolution (px)")

                with gr.Accordion("Color", open=True):
                    color_mode = gr.Radio(
                        list(COLOR_MODES), value="Modulate", label="Color mode"
                    )
                    strength = gr.Slider(
                        0.0, 1.0, value=0.85, step=0.05, label="Color strength"
                    )

                with gr.Accordion("Advanced", open=False):
                    match_space = gr.Radio(
                        list(MATCH_SPACES), value="RGB", label="Matching color space"
                    )
                    detail_weight = gr.Slider(
                        0.0, 1.0, value=0.0, step=0.05,
                        label="Texture matching (0 = tone only)",
                    )
                    use_quantization = gr.Checkbox(False, label="Quantize colors first")
                    quantize_k = gr.Slider(2, 32, value=12, step=1, label="Palette size (k)")

                run_btn = gr.Button("Generate mosaic", variant="primary")

            with gr.Column(scale=2):
                summary = gr.Markdown(IDLE_MESSAGE)
                with gr.Column(visible=False) as results:
                    with gr.Row():
                        original_out = gr.Image(label="1 · Original (preprocessed)", height=240)
                        segmented_out = gr.Image(label="2 · Segmented grid", height=240)
                    mosaic_out = gr.Image(label="3 · Mosaic", height=460)

    with gr.Tab("Tile sets"):
        gr.Markdown(
            "Every tile is generated in code, so any tile set can be rendered at any "
            "resolution. Tiles are ordered by brightness — that ramp is what the "
            "classifier searches."
        )
        with gr.Row():
            preview_name = gr.Dropdown(TILE_SETS, value="ASCII Art", label="Tile set")
            preview_px = gr.Slider(8, 32, value=16, step=4, label="Tile resolution (px)")
        preview_caption = gr.Markdown()
        preview_out = gr.Image(label="Tiles, ordered by brightness", height=380)

    with gr.Tab("Performance"):
        gr.Markdown(
            "Sweeps the grid from coarse to fine, measuring how closely the mosaic matches "
            "the original and how long it takes to build — the vectorized implementation "
            "against a loop-based one. Runs whenever you generate a mosaic, using that "
            "tab's settings."
        )
        bench_status = gr.Markdown(BENCH_IDLE_MESSAGE)
        with gr.Column(visible=False) as bench_results:
            quality_table = gr.Dataframe(
                headers=bench.QUALITY_HEADERS, interactive=False,
                label="Quality — MSE lower is better, SSIM higher is better",
            )
            bench_table = gr.Dataframe(
                headers=bench.TABLE_HEADERS, label="Timings", interactive=False
            )
            bench_plot = gr.Image(label="Scaling and speedup", height=380)
            bench_notes = gr.Markdown()

    # ---------------------------------------------------------------- wiring

    controls = [
        image_in, grid_size, tile_set_name, tile_px, color_mode, strength,
        match_space, detail_weight, use_quantization, quantize_k,
    ]
    outputs = [original_out, segmented_out, mosaic_out, summary, results]

    bench_inputs = [
        image_in, tile_set_name, tile_px, color_mode, strength,
        match_space, detail_weight, use_quantization, quantize_k,
    ]
    bench_outputs = [
        quality_table, bench_table, bench_plot, bench_notes, bench_status, bench_results,
    ]

    # Chained rather than a second handler on the same click: the mosaic is ~0.1s and the
    # sweep a few seconds, so `then` lets the mosaic paint first instead of making the
    # user wait out the measurements to see it.
    run_btn.click(generate, controls, outputs).then(
        run_benchmark, bench_inputs, bench_outputs
    )

    for control in (preview_name, preview_px):
        event = control.release if isinstance(control, gr.Slider) else control.change
        event(preview_tiles, [preview_name, preview_px], [preview_out, preview_caption])
    demo.load(preview_tiles, [preview_name, preview_px], [preview_out, preview_caption])


if __name__ == "__main__":
    demo.launch(theme=gr.themes.Soft())
