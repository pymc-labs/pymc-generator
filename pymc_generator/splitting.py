"""Deterministic cell-grouped assignment, separate from scientific generation."""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping


def assign_cell_split(
    world_cells: Mapping[str, tuple[str, int]], *, validation_fraction: float = 0.2, seed: int = 0
) -> dict[str, str]:
    """Assign whole cells on a fixed snapshot, independent of discovery order.

    Each cell key is (stable generation namespace, local cell ID). The caller
    must use the same namespace for all outputs from one generation cell grid,
    even when that grid is packaged across tasks/files. A namespace must not be
    reused for a different grid. Shared local integers alone are not identities.
    At least two distinct cells are needed for a nonempty train/validation split.
    This returns membership only; no raw data or temporal masks are modified.
    """
    if isinstance(validation_fraction, bool) or not math.isfinite(validation_fraction):
        raise ValueError("validation_fraction must be finite and strictly between 0 and 1")
    if not 0 < validation_fraction < 1:
        raise ValueError("validation_fraction must be strictly between 0 and 1")
    if type(seed) is not int or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    for world, cell in world_cells.items():
        if not isinstance(world, str) or not world:
            raise ValueError("world identity must be a nonempty string")
        if (
            not isinstance(cell, tuple)
            or len(cell) != 2
            or not isinstance(cell[0], str)
            or not cell[0]
            or type(cell[1]) is not int
            or cell[1] < 0
        ):
            raise ValueError("cell identity must be (nonempty namespace, nonnegative integer)")
    cells = set(world_cells.values())
    if len(cells) < 2:
        raise ValueError("At least two distinct cells are required to assign a split")

    def rank(cell):
        encoded = json.dumps(["cell-split-v1", seed, *cell], separators=(",", ":")).encode()
        return hashlib.sha256(encoded).digest(), cell

    ordered = sorted(cells, key=rank)
    count = max(1, min(len(cells) - 1, round(len(cells) * validation_fraction)))
    validation = set(ordered[:count])
    return {
        world: "validation" if world_cells[world] in validation else "train"
        for world in sorted(world_cells)
    }
