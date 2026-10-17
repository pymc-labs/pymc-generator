"""Structural-split runs: tasks spread evenly over saturation x adstock splits.

A design lists its splits (one saturation family and one adstock family each) and the
texture regimes its tasks take turns with inside every split. Task ``i`` of a run with
``S`` splits and ``R`` regimes belongs to split ``i % S``, is that split's task
``j = i // S`` and uses regime ``j % R``: the splits fill evenly while the run
progresses, and every split covers every regime. The task's generator configuration is
the member preset ``<saturation>-<adstock>-<regime>`` (``_`` written as ``-``).

Seeds: every split has its own generative seed, derived from the run's base seed (by
default the launch time as ``YYYYMMDDHHMMSS`` in UTC) and the split's two families: the
first 31 bits of ``sha256("<base seed>:<saturation>:<adstock>")``. Task ``j`` of a split
generates with ``split seed + j``. Controls of a design must never lower the outcome, so
the worker discards a task's worlds when a control goes below zero and draws the task
again: its ``k``-th redraw uses ``task seed + k * 2**32``, a seed no first draw of the
run can use (first draws stay below ``2**31 + tasks per split``).

The task-to-split mapping and the seeds are part of the run contract: the reader
re-derives every task's seed from its run's request, so a design must never change once
runs of it exist (add a new design name instead).

Stdlib only: the run contract (``live_dashboard.publication``) and the launcher import it.
"""

from __future__ import annotations

import hashlib
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import UTC, datetime
from types import MappingProxyType
from typing import Any

SPLIT_SEED_BITS = 31
REDRAW_SEED_STRIDE = 2**32
# Redraws of one task after its first draw, before its task is published with no worlds.
MAX_REDRAWS = 5
SEED_RULE = (
    "split seed + the task's index within its split, split seed = first 31 bits of "
    "sha256('<base seed>:<saturation>:<adstock>'); the k-th redraw adds k * 2**32"
)


@dataclass(frozen=True)
class StructuralSplit:
    """One saturation family and one adstock (carryover) family, by library name."""

    saturation: str
    adstock: str

    @property
    def name(self) -> str:
        return f"{self.saturation}-{self.adstock}".replace("_", "-")

    def seed(self, base_seed: int) -> int:
        """This split's generative seed for a run with ``base_seed``."""
        text = f"{base_seed}:{self.saturation}:{self.adstock}"
        digest = hashlib.sha256(text.encode()).digest()
        return int.from_bytes(digest[:8], "big") >> (64 - SPLIT_SEED_BITS)

    def to_dict(self, base_seed: int) -> dict[str, Any]:
        return {
            "saturation": self.saturation,
            "adstock": self.adstock,
            "seed": self.seed(base_seed),
        }


def member_name(split: StructuralSplit, regime: str) -> str:
    """The preset that generates the tasks of ``split`` in ``regime``."""
    return f"{split.name}-{regime}"


@dataclass(frozen=True)
class SplitDesign:
    """Splits and regimes of a structural-split run (see the module docstring)."""

    splits: tuple[StructuralSplit, ...]
    regimes: tuple[str, ...]

    def __post_init__(self) -> None:
        # A run records a design as a mix of its members, which needs two or more.
        if not self.splits or not self.regimes or len(self.splits) * len(self.regimes) < 2:
            raise ValueError("a design needs splits and regimes making two or more members")
        if len(set(self.splits)) != len(self.splits):
            raise ValueError("a design's splits must be distinct")
        if len(set(self.regimes)) != len(self.regimes):
            raise ValueError("a design's regimes must be distinct")

    def split_of(self, task_index: int) -> StructuralSplit:
        return self.splits[task_index % len(self.splits)]

    def index_in_split(self, task_index: int) -> int:
        return task_index // len(self.splits)

    def regime_of(self, task_index: int) -> str:
        return self.regimes[self.index_in_split(task_index) % len(self.regimes)]

    def member_of(self, task_index: int) -> str:
        return member_name(self.split_of(task_index), self.regime_of(task_index))

    @property
    def cycle(self) -> tuple[str, ...]:
        """Members of tasks ``0 .. S*R - 1``: task ``i`` uses ``cycle[i % len(cycle)]``.

        Both the split (``i % S``) and the regime (``(i // S) % R``) of a task depend only
        on ``i % (S * R)``, so the run is the preset mix of this cycle.
        """
        return tuple(self.member_of(i) for i in range(len(self.splits) * len(self.regimes)))

    @property
    def members(self) -> tuple[str, ...]:
        """Every member preset of the design, split-major."""
        return tuple(member_name(split, regime) for split in self.splits for regime in self.regimes)

    def task_seed(self, base_seed: int, task_index: int) -> int:
        return self.split_of(task_index).seed(base_seed) + self.index_in_split(task_index)

    def check(self, base_seed: int, n_tasks: int) -> None:
        """``ValueError`` unless the splits get equal task counts and no seed repeats.

        A split's tasks use the consecutive seeds ``[split seed, split seed + n_tasks / S)``;
        two splits whose ranges overlap would generate identical batches.
        """
        n_splits = len(self.splits)
        if n_tasks % n_splits:
            raise ValueError(
                f"{n_tasks} tasks do not divide evenly over {n_splits} structural splits"
            )
        per_split = n_tasks // n_splits
        starts = sorted(split.seed(base_seed) for split in self.splits)
        if any(later - earlier < per_split for earlier, later in zip(starts, starts[1:])):
            raise ValueError(
                f"two structural splits' seeds overlap for base seed {base_seed}; "
                "choose another seed"
            )

    def seed_context(self, base_seed: int, task_index: int) -> dict[str, Any]:
        """The split, its seed and the regime of task ``task_index``."""
        split = self.split_of(task_index)
        return {
            "split": {
                **split.to_dict(base_seed),
                "task_in_split": self.index_in_split(task_index),
            },
            "regime": self.regime_of(task_index),
        }


def redraw_seed(task_seed: int, redraw: int) -> int:
    """The seed of a task's ``redraw``-th redraw (0 is the first draw)."""
    return task_seed + redraw * REDRAW_SEED_STRIDE


def launch_seed(now: datetime) -> int:
    """A base seed from the launch time: ``YYYYMMDDHHMMSS`` in UTC, as an integer."""
    return int(now.astimezone(UTC).strftime("%Y%m%d%H%M%S"))


# Library family names (pymc_generator.sampler.SATURATION_FAMILY_KEYS and
# CARRYOVER_FAMILY_KEYS); this module cannot import the library.
SATURATIONS = ("linear", "hill", "logistic", "michaelis_menten", "tanh")
ADSTOCKS = ("none", "geometric")
# Texture regimes (live_dashboard.generation._REGIMES): everything smooth, everything
# hectic, each input on its own, spiky channels and controls over a smooth baseline, and a
# jagged baseline under smooth channels and controls.
REGIMES = ("calm", "hectic", "mixed", "spiky-inputs", "hectic-baseline")

# Design name (a run preset) -> design. Read-only, and never edited for an existing name.
DESIGNS: Mapping[str, SplitDesign] = MappingProxyType(
    {
        "structural-splits": SplitDesign(
            splits=tuple(
                StructuralSplit(saturation, adstock)
                for saturation in SATURATIONS
                for adstock in ADSTOCKS
            ),
            regimes=REGIMES,
        )
    }
)
