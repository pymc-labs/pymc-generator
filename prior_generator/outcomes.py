"""Outcome-space distributions: every world pooled along the QUANTITY axis.

The existing diagnostics describe a corpus in *parameter* space (edge
marginals, drawn coefficients) or in *signal* space (per-channel CV, Spearman,
warmup ratio). Neither answers the magnitude question you ask of a prior
before trusting it: **how large are the outcomes, and how large are the pieces
that add up to them?**

This module answers exactly that. For each named quantity it pools all worlds
and returns

* ``series`` — every drawn value as ``(n_units, n_time_steps)``, zero padding
  removed, so ``.values`` is the flat distribution of "all values for Y";
* per-unit stats over time — ``unit_mean``, ``unit_std``, ``unit_min``,
  ``unit_max``, ``unit_total`` — one row per unit, i.e. the distribution
  *across worlds* rather than across time;
* ``unit_share`` — ``Σ_t value / Σ_t sales``, the fraction of total sales the
  unit accounts for, for every quantity that lives on the Y scale.

A **unit** is one world for a scalar quantity (``sales``, ``baseline``, ...)
and one ``(world, column)`` pair for a column quantity (per channel, per
control, per latent demand, per indirect-effect source). Inactive padded
columns are dropped; a structurally-null channel stays in as an exact zero and
is counted by :attr:`QuantityDistribution.zero_unit_fraction`.

The quantities in :data:`ADDITIVE_QUANTITIES` are the exact per-node
decomposition of sales, so their ``unit_share`` values sum to 1.0 per world
(:meth:`OutcomeDistributions.additive_share_total`) — the share table is a
true budget, not a set of loosely related ratios.

Usage::

    from prior_generator import outcome_distributions, sample_prior_predictive

    corpus = sample_prior_predictive(cfg)
    dist = outcome_distributions(corpus)

    print(dist.table())                       # pooled value quantiles
    print(dist.table(of="share"))             # per-unit share-of-sales budget
    y = dist["sales"].values                  # every Y value, all worlds
    media = dist["channel_contribution"]      # per (world, channel)
    media.quantiles(of="share")               # how big media effects get

Scale note: worlds have arbitrary sales levels, so pooled RAW values mix
scales. ``normalize="sales_scale"`` (or ``"sales_mean"``) divides every
Y-scale quantity by the world's own scale, making the pooled distribution
comparable across worlds. Shares are ratios and are unaffected.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from typing import TYPE_CHECKING, Any, Literal

import numpy as np

if TYPE_CHECKING:
    import pandas as pd

    from .worlds import SCM

__all__ = [
    "ADDITIVE_QUANTITIES",
    "DEFAULT_QUANTILES",
    "OUTCOME_QUANTITIES",
    "OutcomeDistributions",
    "QuantityDistribution",
    "outcome_distributions",
]

#: Quantile levels reported by default (matches the report style used by
#: ``signal_diagnostics``, widened to the 5/95 tails).
DEFAULT_QUANTILES: tuple[float, ...] = (0.05, 0.25, 0.5, 0.75, 0.95)

Normalize = Literal["none", "sales_scale", "sales_mean"]
StatName = Literal["value", "mean", "std", "min", "max", "total", "share"]


# ---------------------------------------------------------------------------
# Quantity table
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class _Spec:
    """One reported quantity and where it comes from.

    ``columns`` names the trailing axis kind ("" for a scalar-per-world
    quantity); ``on_y_scale`` marks quantities measured in sales units (only
    those get a share); ``additive`` marks membership of the exact per-node
    decomposition of sales.
    """

    name: str
    corpus_key: str  # "" => derived, see _quantity_values
    world_key: str  # "" => derived
    columns: str  # "" | "channel" | "control" | "latent" | "source"
    on_y_scale: bool
    additive: bool
    doc: str


_SPECS: tuple[_Spec, ...] = (
    _Spec("sales", "sales_raw", "sales", "", True, False, "observed Y"),
    _Spec("baseline", "baseline_raw", "baseline", "", True, False, "all non-media effects on Y"),
    _Spec(
        "baseline_intrinsic",
        "baseline_intrinsic",
        "baseline_intrinsic",
        "",
        True,
        True,
        "intercept random walk",
    ),
    _Spec("sales_noise", "sales_noise", "sales_noise", "", True, True, "observation noise"),
    _Spec(
        "control_contribution",
        "control_contribution",
        "control_contribution",
        "control",
        True,
        True,
        "per-control Z->B effect",
    ),
    _Spec(
        "confounder_contribution",
        "confounder_contribution",
        "confounder_contribution",
        "latent",
        True,
        True,
        "per-latent D->B effect",
    ),
    _Spec(
        "channel_contribution",
        "contributions_raw",
        "contributions",
        "channel",
        True,
        True,
        "per-channel direct C->Y contribution",
    ),
    _Spec(
        "indirect_by_source",
        "indirect_effects_by_source",
        "indirect_effects_by_source",
        "source",
        True,
        True,
        "indirect media effect split by input edge type",
    ),
    _Spec(
        "media_contribution",
        "",
        "",
        "",
        True,
        False,
        "total media effect: Σ_k direct + indirect",
    ),
    _Spec(
        "indirect_effects",
        "indirect_effects",
        "indirect_effects",
        "",
        True,
        False,
        "total indirect media effect",
    ),
    _Spec("spend", "spend_raw", "channels", "channel", False, False, "observed channel spend"),
    _Spec("controls", "controls", "controls", "control", False, False, "observed control series"),
    _Spec("demand", "demand", "demand", "latent", False, False, "latent demand series"),
)

_BY_NAME: dict[str, _Spec] = {spec.name: spec for spec in _SPECS}

#: Reported quantity names, in report order.
OUTCOME_QUANTITIES: tuple[str, ...] = tuple(spec.name for spec in _SPECS)

#: The quantities whose per-world shares form an exact budget summing to 1.
ADDITIVE_QUANTITIES: frozenset[str] = frozenset(spec.name for spec in _SPECS if spec.additive)

#: Corpus mask key per column kind. "source" is never padded (always 3).
_MASK_KEYS: dict[str, str] = {
    "channel": "treatment_active_mask",
    "control": "covariate_active_mask",
    "latent": "latent_active_mask",
}

_COLUMN_PREFIX: dict[str, str] = {"channel": "C", "control": "Z", "latent": "D"}
_SOURCE_LABELS: tuple[str, ...] = ("cc", "zc", "dc")


def _label(kind: str, index: int) -> str:
    """Human label for column ``index`` of a quantity of kind ``kind``."""
    if kind == "":
        return ""
    if kind == "source":
        return _SOURCE_LABELS[index]
    return f"{_COLUMN_PREFIX[kind]}{index + 1}"


def _q_key(level: float) -> str:
    return f"q{level * 100:g}"


def _stat_dict(values: np.ndarray, levels: tuple[float, ...]) -> dict[str, float]:
    """mean/std/min/max plus the requested quantiles, over finite entries."""
    flat = np.asarray(values, dtype=np.float64).ravel()
    finite = flat[np.isfinite(flat)]
    if finite.size == 0:
        nan = float("nan")
        out = {"n": 0.0, "mean": nan, "std": nan, "min": nan, "max": nan}
        out.update({_q_key(level): nan for level in levels})
        return out
    out = {
        "n": float(finite.size),
        "mean": float(finite.mean()),
        "std": float(finite.std()),
        "min": float(finite.min()),
        "max": float(finite.max()),
    }
    out.update({_q_key(level): float(q) for level, q in zip(levels, np.quantile(finite, levels))})
    return out


def _resolve_levels(levels: Sequence[float] | None, stored: tuple[float, ...]) -> tuple[float, ...]:
    """The requested quantile levels as floats; ``stored`` when unset.

    Every report compares the resolved levels against the levels the
    distribution was built with to decide whether the cached pooled statistics
    still answer the question, so the normalization to ``float`` is load
    bearing: ``[0.05, 0.25]`` and ``(0.05, 0.25)`` are the same request and
    must not look like two different ones.
    """
    return tuple(float(level) for level in levels) if levels else stored


def _reject_ambiguous_selector(selector: np.ndarray, n_available: int, kind: str) -> None:
    """Refuse a 0/1 integer selector: mask or positions cannot be inferred.

    NumPy tells a row mask from a list of positions by dtype alone, and this
    corpus stores its flags as ``uint8`` (``is_val``, the active masks), so
    ``worlds=corpus["is_val"]`` reads as "world 1, world 1, world 0, ..."
    rather than "the validation worlds". Both readings produce a plausible,
    correctly-shaped selection of real rows, so nothing downstream can catch
    the mistake — the numbers are simply wrong. Whenever both readings are
    possible (integer dtype, one value per world/unit, every value in
    ``{0, 1}``) the caller has to spell out which one it means.
    """
    if selector.dtype.kind not in "iu" or selector.ndim != 1:
        return
    if selector.size != n_available or selector.size == 0:
        return
    if not bool(np.all((selector == 0) | (selector == 1))):
        return
    raise ValueError(
        f"ambiguous {kind} selector: a 1-D integer array of length {n_available} "
        "whose values are all 0 or 1 is either a row mask or a list of positions, "
        "and only the dtype tells them apart. Say which: arr.astype(bool) (or "
        f"arr == 1) masks the {kind}s where arr is 1; np.flatnonzero(arr) passes "
        "those same rows as positions."
    )


# ---------------------------------------------------------------------------
# Results
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class QuantityDistribution:
    """The distribution of one quantity over every world in the analysed set.

    Attributes
    ----------
    name, columns, on_y_scale, additive
        Copied from the quantity table; ``columns`` is "" for a
        one-unit-per-world quantity, else the column kind
        ("channel" / "control" / "latent" / "source").
    world_index, column_index
        ``(n_units,)`` int arrays locating each unit. ``world_index`` indexes
        the analysed world set (see :attr:`OutcomeDistributions.world_ids` to
        map back to corpus rows); ``column_index`` is -1 for scalar
        quantities.
    unit_mean, unit_std, unit_min, unit_max, unit_total
        ``(n_units,)`` float64 stats over the time axis — the spread of these
        IS the across-world spread of the quantity.
    unit_share
        ``(n_units,)`` ``unit_total / Σ_t sales`` of the unit's world; all-NaN
        when the quantity is not on the Y scale, and NaN for a world whose
        sales sum to zero — that share is undefined, not zero.
    pooled
        mean/std/min/max/quantiles over every individual value of the units
        held here; :meth:`select` recomputes it for the retained subset.
    series
        ``(n_units, n_time_steps)`` float32 of all retained values, or None
        when built with ``keep_series=False``.
    """

    name: str
    columns: str
    on_y_scale: bool
    additive: bool
    n_worlds: int
    quantile_levels: tuple[float, ...]
    world_index: np.ndarray
    column_index: np.ndarray
    unit_mean: np.ndarray
    unit_std: np.ndarray
    unit_min: np.ndarray
    unit_max: np.ndarray
    unit_total: np.ndarray
    unit_share: np.ndarray
    pooled: dict[str, float]
    series: np.ndarray | None

    # -- basics -------------------------------------------------------------
    @property
    def n_units(self) -> int:
        return int(self.world_index.size)

    @property
    def values(self) -> np.ndarray:
        """Flat distribution of every individual value (padding excluded)."""
        if self.series is None:
            raise ValueError(
                f"{self.name}: raw values were dropped (keep_series=False); "
                "per-unit stats and the pooled report at the levels it was built "
                f"with ({', '.join(_q_key(level) for level in self.quantile_levels)}) "
                "are still available"
            )
        return self.series.reshape(-1)

    @property
    def zero_unit_fraction(self) -> float:
        """Fraction of units that are identically zero over the whole window.

        For ``channel_contribution`` this is the share of active channels with
        no ``C->Y`` edge (structural nulls) — they legitimately contribute
        nothing, and they drag every other statistic toward zero, so they are
        reported rather than silently filtered.
        """
        if self.n_units == 0:
            return float("nan")
        zero = (self.unit_min == 0.0) & (self.unit_max == 0.0)
        return float(zero.mean())

    def labels(self) -> tuple[str, ...]:
        """Per-unit column label (``()``-free; "" for scalar quantities)."""
        return tuple(_label(self.columns, int(i)) for i in self.column_index)

    # -- reductions ---------------------------------------------------------
    def stat(self, of: StatName) -> np.ndarray:
        """The per-unit array named by ``of`` (``"value"`` → flat values)."""
        if of == "value":
            return self.values
        try:
            return {
                "mean": self.unit_mean,
                "std": self.unit_std,
                "min": self.unit_min,
                "max": self.unit_max,
                "total": self.unit_total,
                "share": self.unit_share,
            }[of]
        except KeyError:
            raise ValueError(
                f"unknown stat {of!r}; expected one of value, mean, std, min, max, total, share"
            ) from None

    def quantiles(
        self, *, of: StatName = "value", levels: tuple[float, ...] | None = None
    ) -> dict[str, float]:
        """mean/std/min/max/quantiles of the chosen statistic.

        The pooled value report is cached at construction, so asking for the
        levels this distribution was built with is answered from the cache —
        that is the only report available under ``keep_series=False``, where
        the raw values needed to re-quantile them are gone.
        """
        levels = _resolve_levels(levels, self.quantile_levels)
        if of == "value" and levels == self.quantile_levels:
            return dict(self.pooled)
        return _stat_dict(self.stat(of), levels)

    def summary(self, levels: tuple[float, ...] | None = None) -> dict[str, Any]:
        """JSON-serializable summary of this quantity.

        ``levels`` other than the ones this distribution was built with force
        the pooled value report to be recomputed from the raw values, which
        requires ``keep_series=True``.
        """
        levels = _resolve_levels(levels, self.quantile_levels)
        cached = levels == self.quantile_levels
        out: dict[str, Any] = {
            "n_units": self.n_units,
            "n_worlds": self.n_worlds,
            "columns": self.columns or None,
            "additive": self.additive,
            "on_y_scale": self.on_y_scale,
            "zero_unit_fraction": self.zero_unit_fraction,
            "value": dict(self.pooled) if cached else _stat_dict(self.values, levels),
            "unit_mean": _stat_dict(self.unit_mean, levels),
            "unit_std": _stat_dict(self.unit_std, levels),
        }
        out["share"] = _stat_dict(self.unit_share, levels) if self.on_y_scale else None
        return out

    def select(self, mask: np.ndarray) -> QuantityDistribution:
        """A new distribution keeping only the units where ``mask`` is true.

        Boolean ``(n_units,)`` or an integer index array. Use it to answer
        conditional magnitude questions, e.g. direct channels only::

            media = dist["channel_contribution"]
            direct = media.select(media.unit_max > 0)

        A 0/1 integer array as long as ``n_units`` is rejected: it is a mask
        written as integers as often as it is a list of positions, and the two
        readings select different units. Requires the raw values
        (``keep_series=True``) — the pooled report has to be recomputed for
        the subset, and reporting the full population's one instead would be
        silently wrong.
        """
        if self.series is None:
            raise ValueError(
                f"{self.name}: cannot select a subset without the raw values "
                "(keep_series=False); the pooled report would still describe every "
                "unit, not the selected ones. Rebuild with keep_series=True"
            )
        idx = np.asarray(mask)
        if idx.dtype == bool:
            if idx.shape != (self.n_units,):
                raise ValueError(f"boolean mask must have shape ({self.n_units},), got {idx.shape}")
        else:
            _reject_ambiguous_selector(idx, self.n_units, "unit")
            idx = idx.astype(np.int64, copy=False)
            if idx.size and (idx.min() < 0 or idx.max() >= self.n_units):
                raise IndexError(f"unit index out of range for n_units={self.n_units}")
        series = self.series[idx]
        pooled = _stat_dict(series.reshape(-1), self.quantile_levels)
        return replace(
            self,
            world_index=self.world_index[idx],
            column_index=self.column_index[idx],
            unit_mean=self.unit_mean[idx],
            unit_std=self.unit_std[idx],
            unit_min=self.unit_min[idx],
            unit_max=self.unit_max[idx],
            unit_total=self.unit_total[idx],
            unit_share=self.unit_share[idx],
            pooled=pooled,
            series=series,
        )


@dataclass(frozen=True)
class OutcomeDistributions:
    """Every reported quantity's distribution over one set of worlds."""

    quantities: dict[str, QuantityDistribution]
    n_worlds: int
    n_time_steps: int
    normalize: str
    quantile_levels: tuple[float, ...]
    world_ids: np.ndarray

    def __getitem__(self, name: str) -> QuantityDistribution:
        try:
            return self.quantities[name]
        except KeyError:
            raise KeyError(
                f"quantity {name!r} not computed; available: {sorted(self.quantities)}"
            ) from None

    def __iter__(self):
        return iter(self.quantities.values())

    def __len__(self) -> int:
        return len(self.quantities)

    @property
    def names(self) -> tuple[str, ...]:
        return tuple(self.quantities)

    def summary(self, levels: tuple[float, ...] | None = None) -> dict[str, Any]:
        """JSON-serializable report over every quantity."""
        levels = _resolve_levels(levels, self.quantile_levels)
        return {
            "n_worlds": self.n_worlds,
            "n_time_steps": self.n_time_steps,
            "normalize": self.normalize,
            "quantiles": list(levels),
            "quantities": {name: dist.summary(levels) for name, dist in self.quantities.items()},
        }

    def additive_share_total(self) -> np.ndarray:
        """Per-world sum of every additive quantity's share — exactly 1.0.

        The generator's decomposition is exact, so this is a float-error check
        on the share budget as much as a diagnostic. Requires the full
        additive set (do not restrict ``quantities`` if you need it).

        A world whose sales sum to zero has no budget to split: every share of
        it is undefined, and the total comes back as NaN rather than as a 0.0
        that would read like a catastrophically broken decomposition.
        """
        missing = sorted(ADDITIVE_QUANTITIES - set(self.quantities))
        if missing:
            raise ValueError(f"additive share budget is incomplete; missing {missing}")
        total = np.zeros(self.n_worlds, dtype=np.float64)
        for dist in self.quantities.values():
            if dist.additive:
                total += np.bincount(
                    dist.world_index,
                    weights=dist.unit_share,
                    minlength=self.n_worlds,
                )
        return total

    def table(self, *, of: StatName = "value", levels: tuple[float, ...] | None = None) -> str:
        """Fixed-width text table of one statistic per quantity.

        ``of="value"`` shows the pooled magnitude of every drawn value;
        ``of="share"`` shows the share-of-sales budget; ``of="mean"`` shows
        the across-world spread of per-world levels.
        """
        levels = _resolve_levels(levels, self.quantile_levels)
        cols = ["mean", *(_q_key(level) for level in levels)]
        rows: list[tuple[str, str, str, list[str]]] = []
        for name, dist in self.quantities.items():
            if of == "share" and not dist.on_y_scale:
                continue
            stats = dist.quantiles(of=of, levels=levels)
            rows.append(
                (
                    name,
                    str(dist.n_units),
                    f"{dist.zero_unit_fraction:.2f}",
                    [_fmt(stats[c]) for c in cols],
                )
            )
        name_w = max(len("quantity"), *(len(r[0]) for r in rows)) if rows else len("quantity")
        unit_w = max(len("units"), *(len(r[1]) for r in rows)) if rows else len("units")
        val_w = max(
            8,
            *(len(v) for r in rows for v in r[3]),
            *(len(c) for c in cols),
        )
        header = f"{'quantity':<{name_w}}  {'units':>{unit_w}}  {'zero':>4}  " + "  ".join(
            f"{c:>{val_w}}" for c in cols
        )
        scale = "" if self.normalize == "none" else f", normalize={self.normalize}"
        lines = [
            f"outcome distributions — {of} — {self.n_worlds} worlds × "
            f"{self.n_time_steps} steps{scale}",
            header,
            "-" * len(header),
        ]
        for name, units, zero, vals in rows:
            lines.append(
                f"{name:<{name_w}}  {units:>{unit_w}}  {zero:>4}  "
                + "  ".join(f"{v:>{val_w}}" for v in vals)
            )
        return "\n".join(lines)

    def to_frame(self) -> pd.DataFrame:
        """Long-form per-unit table: one row per (quantity, world, column)."""
        import pandas as pd

        frames = []
        for name, dist in self.quantities.items():
            frames.append(
                pd.DataFrame(
                    {
                        "quantity": name,
                        "world": self.world_ids[dist.world_index],
                        "column": dist.column_index,
                        "label": list(dist.labels()),
                        "mean": dist.unit_mean,
                        "std": dist.unit_std,
                        "min": dist.unit_min,
                        "max": dist.unit_max,
                        "total": dist.unit_total,
                        "share": dist.unit_share,
                    }
                )
            )
        return pd.concat(frames, ignore_index=True)


def _fmt(value: float) -> str:
    """Compact fixed-width number for the text table."""
    if not np.isfinite(value):
        return "nan"
    magnitude = abs(value)
    if magnitude != 0.0 and (magnitude < 1e-3 or magnitude >= 1e6):
        return f"{value:.2e}"
    return f"{value:,.3f}" if magnitude < 1e3 else f"{value:,.1f}"


# ---------------------------------------------------------------------------
# Input adapters
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class _Dense:
    """Padded dense arrays + active-column masks for the analysed worlds."""

    arrays: dict[str, np.ndarray]
    masks: dict[str, np.ndarray]
    n_worlds: int
    n_time_steps: int
    sales_scale: np.ndarray
    world_ids: np.ndarray


def _world_index(n_available: int, worlds: Any) -> np.ndarray:
    """Normalize a world selector into an int64 index array.

    A boolean array is a row mask, anything else integral is a list of
    positions — except when both readings fit, which
    :func:`_reject_ambiguous_selector` refuses rather than guesses.
    """
    if worlds is None:
        return np.arange(n_available, dtype=np.int64)
    if isinstance(worlds, slice):
        return np.arange(n_available, dtype=np.int64)[worlds]
    idx = np.asarray(worlds)
    if idx.dtype == bool:
        if idx.shape != (n_available,):
            raise ValueError(
                f"boolean world mask must have shape ({n_available},), got {idx.shape}"
            )
        return np.flatnonzero(idx).astype(np.int64)
    _reject_ambiguous_selector(idx, n_available, "world")
    idx = idx.astype(np.int64, copy=False).reshape(-1)
    if idx.size and (idx.min() < 0 or idx.max() >= n_available):
        raise IndexError(f"world index out of range for {n_available} worlds")
    return idx


_CORPUS_KEYS: dict[str, str] = {spec.name: spec.corpus_key for spec in _SPECS if spec.corpus_key}


def _dense_from_corpus(corpus: Mapping[str, Any], worlds: Any) -> _Dense:
    required = [*_CORPUS_KEYS.values(), "sales_scale", *_MASK_KEYS.values()]
    missing = sorted({key for key in required if key not in corpus})
    if missing:
        raise KeyError(f"corpus is missing keys required for outcome distributions: {missing}")
    sales = np.asarray(corpus["sales_raw"])
    idx = _world_index(int(sales.shape[0]), worlds)
    if idx.size == 0:
        raise ValueError("world selection is empty")
    arrays = {name: np.asarray(corpus[key])[idx] for name, key in _CORPUS_KEYS.items()}
    masks = {kind: np.asarray(corpus[key])[idx].astype(bool) for kind, key in _MASK_KEYS.items()}
    masks["source"] = np.ones((idx.size, 3), dtype=bool)
    return _Dense(
        arrays=arrays,
        masks=masks,
        n_worlds=int(idx.size),
        n_time_steps=int(sales.shape[1]),
        sales_scale=np.asarray(corpus["sales_scale"], dtype=np.float64)[idx],
        world_ids=idx,
    )


_WORLD_KEYS: dict[str, str] = {spec.name: spec.world_key for spec in _SPECS if spec.world_key}


def _dense_from_worlds(scms: Sequence[SCM], worlds: Any) -> _Dense:
    idx = _world_index(len(scms), worlds)
    if idx.size == 0:
        raise ValueError("world selection is empty")
    chosen = [scms[int(i)] for i in idx]
    horizons = {int(w.data["sales"].shape[0]) for w in chosen}
    if len(horizons) > 1:
        raise ValueError(f"worlds must share n_time_steps to be pooled; got {sorted(horizons)}")
    n_time_steps = horizons.pop()
    widths = {
        "channel": max(w.n_treatments for w in chosen),
        "control": max(w.n_covariates for w in chosen),
        "latent": max(w.n_latent for w in chosen),
    }
    per_world = {
        "channel": [w.n_treatments for w in chosen],
        "control": [w.n_covariates for w in chosen],
        "latent": [w.n_latent for w in chosen],
    }
    n_worlds = int(idx.size)
    arrays: dict[str, np.ndarray] = {}
    for name, key in _WORLD_KEYS.items():
        kind = _BY_NAME[name].columns
        if kind == "":
            arrays[name] = np.stack([np.asarray(w.data[key], dtype=np.float64) for w in chosen])
            continue
        width = 3 if kind == "source" else widths[kind]
        dense = np.zeros((n_worlds, n_time_steps, width), dtype=np.float64)
        for row, world in enumerate(chosen):
            block = np.asarray(world.data[key], dtype=np.float64)
            dense[row, :, : block.shape[1]] = block
        arrays[name] = dense
    masks = {
        kind: (np.arange(widths[kind])[None, :] < np.asarray(per_world[kind])[:, None])
        for kind in widths
    }
    masks["source"] = np.ones((n_worlds, 3), dtype=bool)
    # The corpus defines sales_scale over supported observations only; a bare
    # SCM has no train/query split, so the full-window std is the analogue.
    sales_scale = np.asarray([np.asarray(w.data["sales"]).std() for w in chosen], dtype=np.float64)
    return _Dense(
        arrays=arrays,
        masks=masks,
        n_worlds=n_worlds,
        n_time_steps=n_time_steps,
        sales_scale=sales_scale,
        world_ids=idx,
    )


def _quantity_values(dense: _Dense, spec: _Spec) -> np.ndarray:
    """Float64 array for one quantity: ``(n_worlds, T[, K])``."""
    if spec.name == "media_contribution":
        channel = dense.arrays["channel_contribution"].astype(np.float64, copy=False)
        indirect = dense.arrays["indirect_effects"].astype(np.float64, copy=False)
        return np.asarray(channel.sum(axis=2) + indirect, dtype=np.float64)
    return dense.arrays[spec.name].astype(np.float64, copy=False)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def outcome_distributions(
    source: Mapping[str, Any] | Sequence[SCM],
    *,
    quantities: Sequence[str] | None = None,
    worlds: Any = None,
    normalize: Normalize = "none",
    quantiles: Sequence[float] = DEFAULT_QUANTILES,
    keep_series: bool = True,
) -> OutcomeDistributions:
    """Pool worlds along the quantity axis instead of the parameter axis.

    Parameters
    ----------
    source
        A corpus mapping (``sample_prior_predictive`` /
        ``DataGenerator.generate`` / ``load_corpus``) or a sequence of
        :class:`~prior_generator.worlds.SCM` worlds sharing one horizon.
    quantities
        Subset of :data:`OUTCOME_QUANTITIES` to compute; default all, and an
        empty subset is rejected — there is nothing to report. Note
        :meth:`OutcomeDistributions.additive_share_total` needs the full
        additive set.
    worlds
        World selector — boolean mask over corpus rows, integer index array,
        or slice. Use it to condition on anything you can express as a row
        mask (``corpus["cell_id"] == 3``, ``corpus["is_val"] == 1``, an
        active-channel count). Corpus flags are stored as ``uint8``, so
        compare or cast them (``== 1`` / ``.astype(bool)``) instead of passing
        them raw: a 0/1 integer array as long as the corpus reads equally well
        as a list of positions and is rejected as ambiguous.
    normalize
        ``"none"`` keeps raw units. ``"sales_scale"`` divides every Y-scale
        quantity by the world's ``sales_scale`` (std of supported sales);
        ``"sales_mean"`` divides by mean sales. Exogenous inputs (``spend``,
        ``controls``, ``demand``) are never rescaled — they do not live in
        sales units. Shares are ratios and are unaffected.
    quantiles
        Quantile levels for the pooled/marginal reports.
    keep_series
        Retain all raw values (float32, roughly the size of the corpus
        arrays). Set False for very large corpora: per-unit stats and the
        pooled quantiles at ``quantiles`` are still computed, while
        ``.values``, :meth:`QuantityDistribution.select` and reports at other
        quantile levels then raise.

    Returns
    -------
    OutcomeDistributions
    """
    levels = tuple(float(q) for q in quantiles)
    if not levels or min(levels) < 0.0 or max(levels) > 1.0:
        raise ValueError(f"quantile levels must be a non-empty subset of [0, 1], got {levels}")
    if normalize not in ("none", "sales_scale", "sales_mean"):
        raise ValueError(
            f"normalize must be 'none', 'sales_scale' or 'sales_mean', got {normalize!r}"
        )
    names = OUTCOME_QUANTITIES if quantities is None else tuple(quantities)
    unknown = sorted(set(names) - set(OUTCOME_QUANTITIES))
    if unknown:
        raise ValueError(f"unknown quantities {unknown}; expected {list(OUTCOME_QUANTITIES)}")
    if not names:
        raise ValueError(
            f"quantities is empty; name at least one of {list(OUTCOME_QUANTITIES)} "
            "or pass None for all of them"
        )

    if isinstance(source, Mapping):
        dense = _dense_from_corpus(source, worlds)
    elif isinstance(source, Sequence):
        if not source:
            raise ValueError("no worlds to summarize")
        if not hasattr(source[0], "data"):
            raise TypeError("sequence source must contain SCM worlds")
        dense = _dense_from_worlds(source, worlds)
    else:
        raise TypeError(f"source must be a corpus mapping or a sequence of SCM, got {type(source)}")

    raw_sales = dense.arrays["sales"].astype(np.float64, copy=False)
    if normalize == "sales_scale":
        scale = dense.sales_scale.copy()
    elif normalize == "sales_mean":
        scale = np.abs(raw_sales.mean(axis=1))
    else:
        scale = np.ones(dense.n_worlds, dtype=np.float64)
    # A degenerate scale would turn a finite world into inf/nan; leave it at 1.
    scale[~(scale > 0.0)] = 1.0
    sales_total = raw_sales.sum(axis=1) / scale

    out: dict[str, QuantityDistribution] = {}
    for name in OUTCOME_QUANTITIES:
        if name not in names:
            continue
        spec = _BY_NAME[name]
        arr = _quantity_values(dense, spec)
        if spec.on_y_scale and normalize != "none":
            arr = arr / (scale[:, None, None] if arr.ndim == 3 else scale[:, None])
        if spec.columns:
            mask = dense.masks[spec.columns]
            # (worlds, T, K) -> (worlds, K, T) so masking picks whole series.
            series = np.ascontiguousarray(arr.transpose(0, 2, 1)[mask])
            world_index, column_index = (a.astype(np.int64) for a in np.nonzero(mask))
        else:
            series = arr
            world_index = np.arange(dense.n_worlds, dtype=np.int64)
            column_index = np.full(dense.n_worlds, -1, dtype=np.int64)
        unit_total = series.sum(axis=1)
        if spec.on_y_scale:
            denominator = sales_total[world_index]
            unit_share = np.divide(
                unit_total,
                denominator,
                out=np.full(unit_total.shape, np.nan),
                where=denominator != 0.0,
            )
        else:
            unit_share = np.full(unit_total.shape, np.nan)
        out[name] = QuantityDistribution(
            name=name,
            columns=spec.columns,
            on_y_scale=spec.on_y_scale,
            additive=spec.additive,
            n_worlds=dense.n_worlds,
            quantile_levels=levels,
            world_index=world_index,
            column_index=column_index,
            unit_mean=series.mean(axis=1),
            unit_std=series.std(axis=1),
            unit_min=series.min(axis=1),
            unit_max=series.max(axis=1),
            unit_total=unit_total,
            unit_share=unit_share,
            pooled=_stat_dict(series, levels),
            series=series.astype(np.float32) if keep_series else None,
        )

    return OutcomeDistributions(
        quantities=out,
        n_worlds=dense.n_worlds,
        n_time_steps=dense.n_time_steps,
        normalize=normalize,
        quantile_levels=levels,
        world_ids=dense.world_ids,
    )
