"""Procedurally generated tile sets.

Every tile set is built from code rather than shipped as image assets, so the tile
resolution is a free parameter and the repository stays free of binaries. A tile set is
just a stack of square RGB tiles plus one "key" color per tile; matching a grid cell to a
tile is then a nearest-neighbor search against those keys.

Sets are cached per (name, tile_size) because regenerating them on every UI interaction
would dominate the runtime of small grids.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache

import cv2
import numpy as np

# Perceptual weights used whenever a color has to collapse to a single brightness.
LUMA = np.array([0.299, 0.587, 0.114], dtype=np.float32)


@dataclass(frozen=True)
class TileSet:
    """A stack of square tiles and the color each one stands for.

    tiles : (N, T, T, 3) uint8 -- the tile images themselves.
    keys  : (N, 3) float32     -- mean RGB of each tile, what cells are matched against.
    mode  : "intensity" cells are matched on brightness alone (monochrome tile art);
            "color"     cells are matched on full RGB distance (colored tile art).
    """

    name: str
    tiles: np.ndarray
    keys: np.ndarray
    mode: str
    description: str = ""

    @property
    def tile_size(self) -> int:
        return int(self.tiles.shape[1])

    @property
    def n_tiles(self) -> int:
        return int(self.tiles.shape[0])

    @property
    def luma_keys(self) -> np.ndarray:
        """(N,) brightness of each tile key."""
        return self.keys @ LUMA


def _finalize(name: str, tiles: np.ndarray, mode: str, description: str) -> TileSet:
    """Attach keys to a freshly generated tile stack.

    The key is the tile's own mean color, so a cell replaced by its best-matching tile
    preserves that cell's average brightness/color as closely as the set allows. That is
    exactly the quantity MSE and SSIM reward, which is why keys are measured rather than
    assigned by hand.
    """
    tiles = np.ascontiguousarray(tiles.astype(np.uint8))
    keys = tiles.reshape(len(tiles), -1, 3).mean(axis=1).astype(np.float32)

    # Order by brightness so the set reads as a ramp in previews and so intensity
    # matching can rely on a monotonic key array.
    order = np.argsort(keys @ LUMA)
    return TileSet(name, tiles[order], keys[order], mode, description)


def _blank(n: int, size: int) -> np.ndarray:
    return np.zeros((n, size, size, 3), dtype=np.uint8)


# --------------------------------------------------------------------------------------
# Tile set generators
# --------------------------------------------------------------------------------------

ASCII_RAMP = " .:-=+ic*#%@"


def _render_glyph(ch: str, hi: int, ink: int, paper: int) -> np.ndarray:
    """One character drawn in `ink` on a `paper` background, at working resolution."""
    tile = np.full((hi, hi, 3), paper, dtype=np.uint8)
    if ch == " ":
        return tile

    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = hi / 30.0
    thickness = max(1, int(hi / 24))

    (tw, th), _ = cv2.getTextSize(ch, font, scale, thickness)
    org = ((hi - tw) // 2, (hi + th) // 2)
    cv2.putText(tile, ch, org, font, scale, (ink,) * 3, thickness, cv2.LINE_AA)
    return tile


# (ink, paper) pairs. Hershey fonts are stroke-based, so even '@' inks only about a
# quarter of its cell: one ink/paper pair spans barely 65 luma levels. Stacking four
# pairs -- bright ink on dark paper, then dark ink on bright paper -- tiles the full
# 0-255 range with no gap wider than ~25 levels. Without this the set simply cannot
# represent a midtone, and every midtone cell is forced to a wrong brightness.
ASCII_PHASES = ((255, 0), (255, 88), (0, 168), (0, 255))


def _ascii_tiles(size: int) -> np.ndarray:
    """A printed-page ramp: characters over four ink/paper combinations."""
    # Render large and downsample: drawing at 4x and area-averaging gives far smoother
    # glyphs than drawing directly at tile resolution.
    hi = max(64, size * 4)

    tiles = [
        _render_glyph(ch, hi, ink=ink, paper=paper)
        for ink, paper in ASCII_PHASES
        for ch in ASCII_RAMP
    ]

    return np.stack([cv2.resize(t, (size, size), interpolation=cv2.INTER_AREA) for t in tiles])


def _halftone_tiles(size: int, n: int = 14) -> np.ndarray:
    """Newspaper-style screen: a centered dot whose radius encodes brightness."""
    hi = max(64, size * 4)
    tiles = _blank(n, hi)

    # Radius grows with sqrt(level) so that dot *area* -- and therefore mean brightness --
    # increases linearly across the set, giving an even intensity ramp.
    max_r = hi * 0.72  # large enough for the final dot to cover the whole tile
    for i in range(n):
        r = int(round(max_r * np.sqrt(i / (n - 1))))
        if r > 0:
            cv2.circle(tiles[i], (hi // 2, hi // 2), r, (255, 255, 255), -1, cv2.LINE_AA)

    return np.stack([cv2.resize(t, (size, size), interpolation=cv2.INTER_AREA) for t in tiles])


def _geometric_tiles(size: int) -> np.ndarray:
    """Mixed primitives -- bars, wedges, diamonds -- covering a range of ink coverage."""
    hi = max(64, size * 4)
    white = (255, 255, 255)
    shapes: list[np.ndarray] = []

    def canvas() -> np.ndarray:
        return np.zeros((hi, hi, 3), dtype=np.uint8)

    shapes.append(canvas())  # empty

    # Concentric square outlines of increasing weight.
    for frac in (0.25, 0.45):
        t = canvas()
        m = int(hi * (0.5 - frac / 2))
        cv2.rectangle(t, (m, m), (hi - m, hi - m), white, max(1, hi // 12))
        shapes.append(t)

    # Diagonal bars in both directions -- these carry edge orientation, which is what
    # high-variance cells get steered toward.
    for pts in ((0, 0, hi, hi), (hi, 0, 0, hi)):
        t = canvas()
        cv2.line(t, (pts[0], pts[1]), (pts[2], pts[3]), white, max(2, hi // 6), cv2.LINE_AA)
        shapes.append(t)

    # Half-plane wedges.
    for tri in (
        [(0, 0), (hi, 0), (0, hi)],
        [(hi, 0), (hi, hi), (0, hi)],
    ):
        t = canvas()
        cv2.fillPoly(t, [np.array(tri, dtype=np.int32)], white)
        shapes.append(t)

    # Diamond, then filled squares at growing scale.
    t = canvas()
    c = hi // 2
    cv2.fillPoly(t, [np.array([(c, 0), (hi, c), (c, hi), (0, c)], dtype=np.int32)], white)
    shapes.append(t)

    for frac in (0.4, 0.55, 0.7, 0.85, 1.0):
        t = canvas()
        m = int(hi * (1 - frac) / 2)
        cv2.rectangle(t, (m, m), (hi - m - 1, hi - m - 1), white, -1)
        shapes.append(t)

    # Inverted shapes -- dark marks on a light ground -- fill the bright end of the
    # coverage ramp, which the additive shapes above leave sparse.
    for frac in (0.55, 0.35):
        t = np.full((hi, hi, 3), 255, dtype=np.uint8)
        m = int(hi * (1 - frac) / 2)
        cv2.rectangle(t, (m, m), (hi - m - 1, hi - m - 1), (0, 0, 0), -1)
        shapes.append(t)

    return np.stack(
        [cv2.resize(s, (size, size), interpolation=cv2.INTER_AREA) for s in shapes]
    )


LEGO_PALETTE = [
    (13, 13, 13), (35, 42, 110), (0, 85, 191), (108, 192, 227),
    (0, 133, 43), (150, 191, 77), (255, 205, 3), (245, 125, 32),
    (180, 0, 0), (255, 255, 255), (161, 165, 162), (99, 95, 98),
    (149, 185, 11), (172, 120, 186), (231, 172, 200), (88, 57, 39),
]


def _lego_tiles(size: int) -> np.ndarray:
    """A studded plate per palette color -- the one genuinely color-driven set."""
    hi = max(64, size * 4)
    tiles = []

    for base in LEGO_PALETTE:
        t = np.full((hi, hi, 3), base, dtype=np.uint8)
        c, r = hi // 2, int(hi * 0.30)

        shade = np.array(base, dtype=np.float32)
        # Stud face slightly lighter than the plate, with a darker rim underneath it,
        # so the tile still reads as 3D once shrunk to a handful of pixels.
        cv2.circle(t, (c, c), r, tuple(np.clip(shade * 0.75, 0, 255).astype(int).tolist()), -1, cv2.LINE_AA)
        cv2.circle(t, (c, c - hi // 40), r - max(1, hi // 32),
                   tuple(np.clip(shade * 1.25 + 12, 0, 255).astype(int).tolist()), -1, cv2.LINE_AA)
        tiles.append(t)

    return np.stack([cv2.resize(t, (size, size), interpolation=cv2.INTER_AREA) for t in tiles])


# --------------------------------------------------------------------------------------
# Registry
# --------------------------------------------------------------------------------------

_GENERATORS = {
    "ASCII Art": (_ascii_tiles, "intensity", "Character density ramp, light on dark."),
    "Halftone Dots": (_halftone_tiles, "intensity", "Newspaper screen; dot area tracks brightness."),
    "Geometric Blocks": (_geometric_tiles, "intensity", "Bars, wedges and diamonds across a coverage ramp."),
    "LEGO Studs": (_lego_tiles, "color", "Studded plates in a fixed brick palette; matched on color."),
}

TILE_SETS = list(_GENERATORS)


@lru_cache(maxsize=64)
def get_tile_set(name: str, tile_size: int = 16) -> TileSet:
    """Build (and cache) the named tile set at the requested tile resolution."""
    if name not in _GENERATORS:
        raise ValueError(f"unknown tile set {name!r}; expected one of {TILE_SETS}")

    generator, mode, description = _GENERATORS[name]
    return _finalize(name, generator(int(tile_size)), mode, description)


def contact_sheet(tile_set: TileSet, cols: int = 8, cell_px: int = 48) -> np.ndarray:
    """Render a tile set as a labelled preview grid for the UI."""
    n = tile_set.n_tiles
    rows = int(np.ceil(n / cols))
    pad = 6

    sheet = np.full(
        (rows * (cell_px + pad) + pad, cols * (cell_px + pad) + pad, 3), 24, dtype=np.uint8
    )

    for i in range(n):
        r, c = divmod(i, cols)
        y = pad + r * (cell_px + pad)
        x = pad + c * (cell_px + pad)
        # Nearest-neighbor keeps the tile's real pixel structure visible when enlarged.
        tile = cv2.resize(tile_set.tiles[i], (cell_px, cell_px), interpolation=cv2.INTER_NEAREST)
        sheet[y : y + cell_px, x : x + cell_px] = tile

    return sheet
