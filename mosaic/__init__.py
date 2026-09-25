"""Interactive image mosaic generator.

The pipeline is: preprocess -> grid statistics -> tile classification -> reconstruction,
with every grid operation expressed as a vectorized NumPy array operation.
"""

from .preprocess import crop_to_grid, quantize_colors, to_working_resolution
from .core import MosaicResult, build_mosaic, build_mosaic_loop
from .tiles import TILE_SETS, TileSet, contact_sheet, get_tile_set
from .metrics import score

__all__ = [
    "crop_to_grid",
    "quantize_colors",
    "to_working_resolution",
    "MosaicResult",
    "build_mosaic",
    "build_mosaic_loop",
    "TILE_SETS",
    "TileSet",
    "contact_sheet",
    "get_tile_set",
    "score",
]
