---
title: Interactive Image Mosaic Generator
colorFrom: blue
colorTo: green
sdk: gradio
sdk_version: 6.28.0
python_version: "3.12"
app_file: app.py
pinned: false
short_description: Rebuild any image out of procedurally generated tiles.
---

# Interactive Image Mosaic Generator

Divides an image into a grid, classifies every cell by color or brightness, and rebuilds
the image out of tiles drawn from a procedurally generated tile set. Every grid operation
is a vectorized NumPy array expression — no Python loop touches a cell on the fast path.

**▶ Live demo:** <https://huggingface.co/spaces/sqz0914/image-mosaic-generator>

The Space sleeps when idle, so the first load after a quiet spell takes roughly half a
minute to wake.

## Quick start

```bash
# The virtual environment is Python 3.14; see "Environment" below if you need to rebuild it.
./.venv/bin/pip install -r requirements.txt

./.venv/bin/python -m gradio app.py     # hot-reloading dev server on :7860
./.venv/bin/python -m pytest tests/ -q  # 46 tests
```

Timings live in the app's **Performance** tab — run it there on whichever image you have
loaded rather than from a script.

Keep your own test images in `assets/examples/` — it is git-ignored, since Hugging Face
requires binary files to go through Xet/LFS and the app reads nothing from that folder.
Upload an image through the input box on the Mosaic tab, set the controls, and press
**Generate mosaic**.

## What it does

| Stage | Implementation |
|---|---|
| **Preprocess** | Resize to a 768px long edge, then center-crop to a whole number of square cells (less than one cell discarded per axis). |
| **Grid statistics** | `img.reshape(rows, cell, cols, cell, 3).mean(axis=(1, 3))` — one reshape, one reduction, every cell at once. |
| **Classification** | Broadcast cells against tile keys to a `(rows, cols, n_tiles)` distance array; `argmin` picks the tile. RGB or CIE-Lab. |
| **Reconstruction** | `tiles[indices]` gathers every tile in one indexing operation; a transpose and reshape stitch the grid back together. |
| **Scoring** | MSE and SSIM against the source, swept across grid sizes on the Performance tab. |

## Tile sets

All four are generated in code, so any set renders at any tile resolution and the
repository carries no binary tile assets. The **Tile sets** tab previews any of them at
any resolution.

- **ASCII Art** — a character density ramp over four ink/paper combinations, so the set
  spans the full tonal range instead of only the dark quarter a stroke font can ink.
- **Halftone Dots** — a newspaper screen; dot *area* grows linearly with brightness.
- **Geometric Blocks** — bars, wedges and diamonds, including inverted marks to cover the
  bright end of the coverage ramp.
- **LEGO Studs** — studded plates in a fixed brick palette; the one set matched on full
  color rather than brightness.

## Interface

Three tabs: **Mosaic** (upload an image, set the controls, press Generate; shows the
original, the segmented grid, the mosaic and its MSE/SSIM), **Tile sets**
(contact-sheet previews at any resolution), and **Performance** (sweeps grid sizes for
quality and for vectorized-vs-loop timings, using the Mosaic tab's settings).

## Layout

```
app.py              Gradio interface
mosaic/
  preprocess.py     resizing, grid-aligned cropping, color quantization
  core.py           grid statistics, classification, reconstruction + loop reference
  tiles.py          procedural tile-set generation and previews
  metrics.py        MSE / SSIM
  benchmark.py      timing harness and chart, feeding the Performance tab
tests/              46 tests, including vectorized == loop equivalence
```

## Environment

Developed on Python 3.14.7 (macOS, arm64). To rebuild the virtual environment:

```bash
rm -rf .venv && python3.14 -m venv .venv
./.venv/bin/pip install -r requirements.txt
```

`requirements.txt` pins exact versions. Those pins need **Python 3.12 or newer** — numpy
2.5.3 declares `requires_python >= 3.12` — which is why the Space is set to 3.12. On an
older interpreter, relax the numpy and scikit-image pins.

## Deploying to Hugging Face Spaces

As of 2026, creating a Gradio Space on a personal account requires a PRO plan. (A free
account in good standing may instead host up to 2 Gradio Spaces on ZeroGPU hardware.)
This Space runs on **CPU basic**, which has no hourly cost — the app is pure NumPy and
OpenCV and never needs a GPU.

The YAML header at the top of this file is the Space configuration, so the repository can
be pushed as-is:

```bash
git init && git add . && git commit -m "Interactive image mosaic generator"

# Create a Gradio Space at https://huggingface.co/new-space, then:
git remote add space https://huggingface.co/spaces/<your-username>/<space-name>
git push space main
```

The Space installs `requirements.txt`, which includes pytest.
