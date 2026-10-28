"""Per-world descriptors: one row of interpretable statistics per world.

``data_diagnostics`` explains what happens *inside* a set of worlds. This
module turns every world into one fixed-width row of named statistics so that
worlds can be compared *with each other*: generated corpora, lists of
:class:`~pymc_generator.worlds.SCM` worlds and observed datasets
(:class:`ObservedWorlds`) all produce the same row definitions.

Every statistic is computed inside one world over its active series, then
reduced across the active nodes of one role (treatments, covariates) with a
named reducer. Nothing is pooled across worlds. Three states are kept apart:

* ``valid`` — a value exists. Positive or negative infinity is a valid, ordered
  value (a nonzero constant series has an infinite level-to-variation ratio).
* ``ineligible`` — the question does not apply to this world (no active
  covariates, fewer than two active treatments for a pair statistic, no
  observed outcome).
* ``undefined`` — the question applies but has no answer (``0 / 0``, a lag
  longer than the usable horizon, every contributing series constant).

Descriptors are observable by default: they need only treatments, covariates
and outcome. ``include_truth=True`` adds generating-process quantities
(contribution shares) that only generated worlds carry; they are marked
``observable=False`` and are never fabricated for observed data.

This module is purely post-hoc. It draws no random numbers, never mutates its
source and writes nothing.
"""

from __future__ import annotations

import re
import unicodedata
from collections.abc import Hashable, Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import IO, TYPE_CHECKING, Any

import numpy as np

from .diagnostics import (
    _MIN_PAIRS,
    _WORLD_COUNTERFACTUAL,
    ACTIVITY_STATISTICS,
    _acf_world,
    _cell,
    _check_zero_padding,
    _compute_slots,
    _contribution_budget,
    _corpus_source,
    _numeric_float64,
    _render_table,
    _require_finite,
    _series_lags,
    _validate_lags,
    _validate_world_selector,
    _worlds_source,
    activity_runs,
    lagged_dependence,
    temporal_statistics,
)
from .outcomes import DEFAULT_QUANTILES

if TYPE_CHECKING:  # pragma: no cover - typing only
    import os

    import pandas as pd

    from .worlds import SCM

__all__ = [
    "DEFAULT_STATISTICS",
    "DESCRIPTOR_STATUSES",
    "DESCRIPTOR_VERSION",
    "METADATA_COLUMNS",
    "PAIR_ROLES",
    "REDUCERS",
    "STATISTICS",
    "TRUTH_FEATURES",
    "FeatureDefinition",
    "ObservedWorlds",
    "WorldDescriptors",
    "world_descriptors",
]

#: Bumped whenever the definition of any statistic changes. Descriptor rows
#: from different versions are never compared or concatenated.
DESCRIPTOR_VERSION = 1

#: Status codes stored in :attr:`WorldDescriptors.status`, by position.
DESCRIPTOR_STATUSES: tuple[str, ...] = ("valid", "ineligible", "undefined")
VALID, INELIGIBLE, UNDEFINED = 0, 1, 2

#: Integer per-world metadata usable as strata; never a distance feature.
METADATA_COLUMNS: tuple[str, ...] = ("n_treatments_active", "n_covariates_active", "n_time_steps")

#: Legacy per-series statistics, unchanged by opt-in diagnostic additions.
_LEGACY_NODE_STATISTICS: tuple[str, ...] = (
    "level_to_variation",
    "cv",
    "zero_fraction",
    "roughness",
    "spike",
    "acf",
    "diff_acf",
)
#: Opt-in temporal statistics on original-unit levels and first differences.
_TEMPORAL_STATISTICS: tuple[str, ...] = (
    "level_std",
    "difference_std",
    "time_correlation",
    "overlap_pearson",
    "diff_overlap_pearson",
)
#: Per-series statistics, reduced over the active nodes of a role.
NODE_STATISTICS: tuple[str, ...] = (
    _LEGACY_NODE_STATISTICS + _TEMPORAL_STATISTICS + ACTIVITY_STATISTICS
)
#: Per-role statistics (one value per role, no reducer).
ROLE_STATISTICS: tuple[str, ...] = ("constant_fraction",)
#: Legacy pair statistics retain their unordered within-role pair universe.
_LEGACY_PAIR_STATISTICS: tuple[str, ...] = ("abs_pearson", "diff_abs_pearson")
#: Signed lagged pair statistics, selected with explicit ``dependence_lags``.
_LAGGED_PAIR_STATISTICS: tuple[str, ...] = ("lagged_pearson", "diff_lagged_pearson")
#: Pairwise statistics, reduced over the eligible pairs of a pair role.
PAIR_STATISTICS: tuple[str, ...] = _LEGACY_PAIR_STATISTICS + _LAGGED_PAIR_STATISTICS
#: Original default feature selection; new diagnostics are strictly opt-in.
DEFAULT_STATISTICS: tuple[str, ...] = (
    _LEGACY_NODE_STATISTICS + ROLE_STATISTICS + _LEGACY_PAIR_STATISTICS
)
#: Every selectable statistic family, in feature order.
STATISTICS: tuple[str, ...] = NODE_STATISTICS + ROLE_STATISTICS + PAIR_STATISTICS

#: Reducers across the active nodes (or pairs) of a role.
REDUCERS: tuple[str, ...] = ("median", "min", "max")

#: Roles reduced across active nodes; the outcome is a single series.
_REDUCED_ROLES: tuple[str, ...] = ("treatment", "covariate")

#: Pair roles: which two node sets a pairwise statistic runs over.
PAIR_ROLES: tuple[str, ...] = (
    "treatment_pair",
    "covariate_pair",
    "treatment_covariate",
    "treatment_outcome",
)

#: Generating-process features (``include_truth=True``) -> top-cut row of the
#: ``base_direct_plus_indirect`` contribution budget. Net shares of outcome.
TRUTH_FEATURES: dict[str, str] = {
    "truth_treatment_share": "treatment_total",
    "truth_covariate_share": "covariates_baseline_alloc_total",
    "truth_latent_share": "latent_unobserved_baseline_alloc_total",
    "truth_baseline_share": "B_intrinsic",
    "truth_noise_share": "Y_noise",
}

_DEFAULT_CHUNK = 2048

_STAT_TEXT: dict[str, str] = {
    "level_to_variation": "|mean| / std of the level series (inf for a nonzero constant)",
    "cv": "std / |mean| of the level series (0 for an all-zero series)",
    "zero_fraction": "fraction of time steps exactly equal to zero",
    "roughness": "std of first differences / (sqrt(2) * std); 1 white noise, -> 0 smooth",
    "spike": "max |step - median step| / IQR(step); robust single-jump ratio",
    "acf": "sample autocorrelation of the level series",
    "diff_acf": "sample autocorrelation of the first-differenced series",
    "constant_fraction": "fraction of active series that are constant (exact range floor)",
    "abs_pearson": "|Pearson correlation| of the level series",
    "diff_abs_pearson": "|Pearson correlation| of the first-differenced series",
    "level_std": "population standard deviation of levels in original units",
    "difference_std": "population standard deviation of first differences in original units",
    "time_correlation": "signed Pearson correlation of levels with the time index",
    "overlap_pearson": "signed Pearson correlation of separately centered overlapping levels",
    "diff_overlap_pearson": (
        "signed Pearson correlation of separately centered overlapping first differences"
    ),
    "leading_inactive_length": "length of the leading exactly-zero spell, or zero",
    "trailing_inactive_length": "length of the trailing exactly-zero spell, or zero",
    "longest_active_length": "length of the longest exactly-nonzero spell, or zero",
    "longest_inactive_length": "length of the longest exactly-zero spell, or zero",
    "active_spell_count": "number of exactly-nonzero spells, including censored spells",
    "inactive_spell_count": "number of exactly-zero spells, including censored spells",
    "complete_active_spell_count": "number of complete interior exactly-nonzero spells",
    "complete_inactive_spell_count": "number of complete interior exactly-zero spells",
    "censored_active_spell_count": "number of boundary-censored exactly-nonzero spells",
    "censored_inactive_spell_count": "number of boundary-censored exactly-zero spells",
    "complete_active_length_median": "median length of complete interior exactly-nonzero spells",
    "complete_inactive_length_median": "median length of complete interior exactly-zero spells",
    "switching_rate": "fraction of adjacent time steps switching exact-zero activity state",
    "interior_inactive_fraction": (
        "total length of complete interior exactly-zero spells divided by the full window length"
    ),
    "lagged_pearson": "signed lagged Pearson correlation of levels",
    "diff_lagged_pearson": "signed lagged Pearson correlation of first differences",
}

_PAIR_TEXT: dict[str, str] = {
    "treatment_pair": "over pairs of active treatments",
    "covariate_pair": "over pairs of active covariates",
    "treatment_covariate": "over (active treatment, active covariate) pairs",
    "treatment_outcome": "over (active treatment, outcome) pairs",
}


# ---------------------------------------------------------------------------
# Definitions
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class FeatureDefinition:
    """What one descriptor column means, exactly.

    Two collections are comparable on a feature only when its definitions are
    equal: same name, statistic, role, view, lag, reducer, observability and
    descriptor version. ``description`` is wording only and never compared.

    Attributes
    ----------
    name
        Column name, e.g. ``covariate_level_to_variation_median``.
    role
        ``treatment``, ``covariate``, ``outcome``, a pair role from
        :data:`PAIR_ROLES`, or ``truth``.
    statistic
        One of :data:`STATISTICS`, or ``share`` for truth features.
    view
        ``levels`` or ``differences``.
    lag
        Lag for autocorrelation, overlap correlation or lagged pair statistics,
        otherwise ``None``. Signed pair lags use left[t] versus right[t + lag].
    reducer
        Reducer across nodes/pairs from :data:`REDUCERS`; ``None`` for a
        single-series or per-role statistic.
    observable
        True when the value needs only treatments, covariates and outcome.
    description
        Human wording.
    version
        :data:`DESCRIPTOR_VERSION` the definition was computed under.
    """

    name: str
    role: str
    statistic: str
    view: str
    lag: int | None
    reducer: str | None
    observable: bool
    description: str = field(compare=False)
    version: int = DESCRIPTOR_VERSION


def _node_token(statistic: str, lag: int | None) -> str:
    return f"{statistic}_lag_{lag}" if lag is not None else statistic


def _definitions(
    statistics: tuple[str, ...],
    lags: tuple[int, ...],
    dependence_lags: tuple[int, ...],
    reducers: tuple[str, ...],
    include_truth: bool,
) -> tuple[FeatureDefinition, ...]:
    out: list[FeatureDefinition] = []

    def node_items() -> list[tuple[str, int | None]]:
        items: list[tuple[str, int | None]] = []
        for stat in NODE_STATISTICS:
            if stat not in statistics:
                continue
            if stat in ("acf", "diff_acf", "overlap_pearson", "diff_overlap_pearson"):
                items.extend((stat, lag) for lag in lags)
            else:
                items.append((stat, None))
        return items

    nodes = node_items()
    for role in _REDUCED_ROLES:
        for stat, lag in nodes:
            for reducer in reducers:
                token = _node_token(stat, lag)
                lag_text = f" at lag {lag}" if lag is not None else ""
                out.append(
                    FeatureDefinition(
                        name=f"{role}_{token}_{reducer}",
                        role=role,
                        statistic=stat,
                        view=(
                            "differences"
                            if stat in ("diff_acf", "difference_std", "diff_overlap_pearson")
                            else "levels"
                        ),
                        lag=lag,
                        reducer=reducer,
                        observable=True,
                        description=f"{reducer} over active {role}s of {_STAT_TEXT[stat]}{lag_text}",
                    )
                )
    for stat, lag in nodes:
        token = _node_token(stat, lag)
        lag_text = f" at lag {lag}" if lag is not None else ""
        out.append(
            FeatureDefinition(
                name=f"outcome_{token}",
                role="outcome",
                statistic=stat,
                view=(
                    "differences"
                    if stat in ("diff_acf", "difference_std", "diff_overlap_pearson")
                    else "levels"
                ),
                lag=lag,
                reducer=None,
                observable=True,
                description=f"outcome {_STAT_TEXT[stat]}{lag_text}",
            )
        )
    if "constant_fraction" in statistics:
        for role in _REDUCED_ROLES:
            out.append(
                FeatureDefinition(
                    name=f"{role}_constant_fraction",
                    role=role,
                    statistic="constant_fraction",
                    view="levels",
                    lag=None,
                    reducer=None,
                    observable=True,
                    description=f"{_STAT_TEXT['constant_fraction']} among active {role}s",
                )
            )
    for stat in PAIR_STATISTICS:
        if stat not in statistics:
            continue
        pair_lags: tuple[int | None, ...] = (
            dependence_lags if stat in _LAGGED_PAIR_STATISTICS else (None,)
        )
        for pair_role in PAIR_ROLES:
            for lag in pair_lags:
                token = _node_token(stat, lag)
                pair_text = _PAIR_TEXT[pair_role]
                lag_text = ""
                if lag is not None:
                    if pair_role in ("treatment_pair", "covariate_pair") and lag != 0:
                        role = "treatments" if pair_role == "treatment_pair" else "covariates"
                        pair_text = f"over both ordered distinct directions of active {role}"
                    lag_text = f" at lag {lag} (left[t] versus right[t + lag])"
                for reducer in reducers:
                    out.append(
                        FeatureDefinition(
                            name=f"{pair_role}_{token}_{reducer}",
                            role=pair_role,
                            statistic=stat,
                            view=(
                                "differences"
                                if stat in ("diff_abs_pearson", "diff_lagged_pearson")
                                else "levels"
                            ),
                            lag=lag,
                            reducer=reducer,
                            observable=True,
                            description=f"{reducer} {pair_text} of {_STAT_TEXT[stat]}{lag_text}",
                        )
                    )
    if include_truth:
        for name, key in TRUTH_FEATURES.items():
            out.append(
                FeatureDefinition(
                    name=name,
                    role="truth",
                    statistic="share",
                    view="levels",
                    lag=None,
                    reducer=None,
                    observable=False,
                    description=f"net share of outcome from contribution row {key!r}",
                )
            )
    return tuple(out)


# ---------------------------------------------------------------------------
# The descriptor table
# ---------------------------------------------------------------------------


def _identity_keys(source_ids: np.ndarray, world_ids: np.ndarray) -> list[tuple[str, int]]:
    return [(str(s), int(w)) for s, w in zip(source_ids.tolist(), world_ids.tolist())]


@dataclass(frozen=True)
class WorldDescriptors:
    """One row per world, one column per :class:`FeatureDefinition`.

    Rows are identified by ``(source_id, world_id)``, which must be unique.
    ``group_id`` is the generation cell a world was drawn from (``-1`` when
    unknown); two rows share a group only when both ``source_id`` and a known
    ``group_id`` are equal. Values are float64 with NaN exactly where the
    status is not ``valid``; valid values may be ``+inf``/``-inf``.

    Build one with :func:`world_descriptors`; the constructor is public so an
    external table can be wrapped, and validates every invariant above.
    """

    source_ids: np.ndarray
    world_ids: np.ndarray
    group_ids: np.ndarray
    metadata: dict[str, np.ndarray]
    definitions: tuple[FeatureDefinition, ...]
    values: np.ndarray
    status: np.ndarray

    def __post_init__(self) -> None:
        source_ids = np.asarray(self.source_ids, dtype=object)
        world_ids = np.asarray(self.world_ids)
        group_ids = np.asarray(self.group_ids)
        values = np.asarray(self.values, dtype=np.float64)
        status = np.asarray(self.status)
        if source_ids.ndim != 1:
            raise ValueError("source_ids must be one-dimensional")
        n = int(source_ids.size)
        if not all(isinstance(s, str) and s for s in source_ids.tolist()):
            raise TypeError("source_ids must be non-empty strings")
        int64_max = np.iinfo(np.int64).max
        for name, arr in (("world_ids", world_ids), ("group_ids", group_ids)):
            if arr.shape != (n,) or arr.dtype.kind not in "iu":
                raise ValueError(f"{name} must be an integer array of shape ({n},)")
            if arr.dtype.kind == "u" and arr.size and int(arr.max()) > int64_max:
                raise ValueError(f"{name} must fit in int64")
        if (world_ids < 0).any():
            raise ValueError("world_ids must be >= 0")
        if (group_ids < -1).any():
            raise ValueError("group_ids must be >= 0, or -1 for an unknown group")
        definitions = tuple(self.definitions)
        if not all(isinstance(d, FeatureDefinition) for d in definitions):
            raise TypeError("definitions must be FeatureDefinition instances")
        names = [d.name for d in definitions]
        if len(set(names)) != len(names):
            raise ValueError("feature names must be unique")
        n_features = len(definitions)
        if values.shape != (n, n_features) or status.shape != (n, n_features):
            raise ValueError(f"values and status must have shape ({n}, {n_features})")
        if (
            status.dtype.kind not in "iu"
            or not np.isin(status, (VALID, INELIGIBLE, UNDEFINED)).all()
        ):
            raise ValueError(f"status codes must index {DESCRIPTOR_STATUSES}")
        if (np.isnan(values) != (status != VALID)).any():
            raise ValueError("values must be NaN exactly where status is not 'valid'")
        metadata = dict(self.metadata)
        if set(metadata) != set(METADATA_COLUMNS):
            raise ValueError(f"metadata must hold exactly {list(METADATA_COLUMNS)}")
        clean_meta: dict[str, np.ndarray] = {}
        for column in METADATA_COLUMNS:
            arr = np.asarray(metadata[column])
            if arr.shape != (n,) or arr.dtype.kind not in "iu":
                raise ValueError(f"metadata {column!r} must be an integer array of shape ({n},)")
            clean_meta[column] = arr.astype(np.int64)
        keys = _identity_keys(source_ids, world_ids)
        if len(set(keys)) != len(keys):
            raise ValueError(
                "(source_id, world_id) must be unique; give each collection its own source_id"
            )
        object.__setattr__(self, "source_ids", source_ids)
        object.__setattr__(self, "world_ids", world_ids.astype(np.int64))
        object.__setattr__(self, "group_ids", group_ids.astype(np.int64))
        object.__setattr__(self, "metadata", clean_meta)
        object.__setattr__(self, "definitions", definitions)
        object.__setattr__(self, "values", values)
        object.__setattr__(self, "status", status.astype(np.uint8))

    # -- basics -------------------------------------------------------------
    @property
    def features(self) -> tuple[str, ...]:
        return tuple(d.name for d in self.definitions)

    @property
    def n_worlds(self) -> int:
        return int(self.world_ids.size)

    def __len__(self) -> int:
        return self.n_worlds

    @property
    def sources(self) -> tuple[str, ...]:
        """Distinct source ids in first-appearance order."""
        return tuple(dict.fromkeys(str(s) for s in self.source_ids.tolist()))

    def feature_index(self, name: str) -> int:
        for index, definition in enumerate(self.definitions):
            if definition.name == name:
                return index
        lookalike = [f for f in self.features if name.lower() in f.lower()][:5]
        hint = f"; similar: {lookalike}" if lookalike else ""
        raise KeyError(f"unknown descriptor feature {name!r}{hint}")

    def definition(self, name: str) -> FeatureDefinition:
        return self.definitions[self.feature_index(name)]

    def column(self, name: str) -> np.ndarray:
        """``(n_worlds,)`` values of one feature (NaN where not valid)."""
        return self.values[:, self.feature_index(name)]

    def column_status(self, name: str) -> np.ndarray:
        """``(n_worlds,)`` status labels of one feature."""
        codes = self.status[:, self.feature_index(name)]
        labels: np.ndarray = np.asarray(DESCRIPTOR_STATUSES, dtype=object)[codes]
        return labels

    def grouping(self, column: str) -> np.ndarray:
        """Per-row value of a grouping column: ``source_id`` or a metadata column."""
        if column == "source_id":
            return self.source_ids
        if column in self.metadata:
            return self.metadata[column]
        raise KeyError(
            f"unknown grouping column {column!r}; expected 'source_id' or one of "
            f"{list(METADATA_COLUMNS)}"
        )

    # -- reshaping ----------------------------------------------------------
    def take(self, rows: Any) -> WorldDescriptors:
        """Rows by NumPy semantics, identities kept.

        A boolean array is a row mask of exactly ``n_worlds`` entries; integers
        (a scalar or a 1-D array, negatives counting from the end) are positions
        without repeats; a slice is a slice. An empty selection gives an empty
        table rather than an error, so filtering a thin region is safe.
        """
        n = self.n_worlds
        if isinstance(rows, slice):
            idx = np.arange(n, dtype=np.int64)[rows]
        else:
            arr = np.asarray(rows)
            if arr.dtype == bool:
                if arr.shape != (n,):
                    raise ValueError(f"a boolean row mask needs shape ({n},), got {arr.shape}")
                idx = np.flatnonzero(arr)
            elif arr.size == 0:
                idx = np.zeros(0, dtype=np.int64)
            elif arr.dtype.kind in "iu" and arr.ndim <= 1:
                if arr.dtype.kind == "u" and int(arr.max()) >= n:
                    raise IndexError(f"row positions must lie in [-{n}, {n})")
                idx = arr.reshape(-1).astype(np.int64)
                if ((idx < -n) | (idx >= n)).any():
                    raise IndexError(f"row positions must lie in [-{n}, {n})")
                idx = np.where(idx < 0, idx + n, idx)
                if np.unique(idx).size != idx.size:
                    raise ValueError("row positions repeat; a world cannot be taken twice")
            else:
                raise TypeError(
                    "rows must be a slice, a boolean mask or integer positions, "
                    f"got dtype {arr.dtype} with shape {arr.shape}"
                )
        return WorldDescriptors(
            source_ids=self.source_ids[idx],
            world_ids=self.world_ids[idx],
            group_ids=self.group_ids[idx],
            metadata={k: v[idx] for k, v in self.metadata.items()},
            definitions=self.definitions,
            values=self.values[idx],
            status=self.status[idx],
        )

    def select(self, features: Sequence[str]) -> WorldDescriptors:
        """The same rows restricted to ``features``, in the given order."""
        if isinstance(features, str):
            raise TypeError("features must be a sequence of names, not a bare string")
        chosen = [self.feature_index(name) for name in features]
        if not chosen:
            raise ValueError("features is empty")
        if len(set(chosen)) != len(chosen):
            raise ValueError("features repeats an entry")
        return WorldDescriptors(
            source_ids=self.source_ids,
            world_ids=self.world_ids,
            group_ids=self.group_ids,
            metadata=self.metadata,
            definitions=tuple(self.definitions[i] for i in chosen),
            values=self.values[:, chosen],
            status=self.status[:, chosen],
        )

    @classmethod
    def concat(cls, parts: Sequence[WorldDescriptors]) -> WorldDescriptors:
        """Stack collections with equal definitions and disjoint identities.

        Parts whose definitions are equal but listed in a different order are
        aligned to the first part's column order.
        """
        parts = list(parts)
        if not parts:
            raise ValueError("concat needs at least one WorldDescriptors")
        first = parts[0].definitions
        names = tuple(d.name for d in first)
        mine = {d.name: d for d in first}
        aligned = [parts[0]]
        for part in parts[1:]:
            theirs = {d.name: d for d in part.definitions}
            if theirs != mine:
                differing = sorted(
                    n for n in set(mine) | set(theirs) if mine.get(n) != theirs.get(n)
                )
                raise ValueError(
                    "descriptor definitions differ between parts "
                    f"(first differing features: {differing[:5]}); compute every part "
                    "with the same statistics, lags, reducers and include_truth"
                )
            aligned.append(part if part.definitions == first else part.select(names))
        return cls(
            source_ids=np.concatenate([p.source_ids for p in aligned]),
            world_ids=np.concatenate([p.world_ids for p in aligned]),
            group_ids=np.concatenate([p.group_ids for p in aligned]),
            metadata={
                k: np.concatenate([p.metadata[k] for p in aligned]) for k in METADATA_COLUMNS
            },
            definitions=first,
            values=np.concatenate([p.values for p in aligned]),
            status=np.concatenate([p.status for p in aligned]),
        )

    # -- reporting ----------------------------------------------------------
    def to_frame(self, *, with_status: bool = False) -> pd.DataFrame:
        """One row per world: identity, metadata, then every feature.

        ``with_status=True`` adds a ``status:<feature>`` label column after
        each feature, so ``ineligible`` and ``undefined`` stay distinguishable
        after the NaN they both read as.
        """
        import pandas as pd

        data: dict[str, Any] = {
            "source_id": self.source_ids,
            "world_id": self.world_ids,
            "group_id": self.group_ids,
        }
        data.update(self.metadata)
        labels = np.asarray(DESCRIPTOR_STATUSES, dtype=object)
        for index, name in enumerate(self.features):
            data[name] = self.values[:, index]
            if with_status:
                data[f"status:{name}"] = labels[self.status[:, index]]
        return pd.DataFrame(data)

    def table(self, features: Sequence[str] | None = None) -> str:
        """Per-feature availability: valid / finite / infinite / undefined / ineligible."""
        if isinstance(features, str):
            raise TypeError("features must be a sequence of names, not a bare string")
        names = self.features if features is None else tuple(features)
        rows = []
        for name in names:
            index = self.feature_index(name)
            codes = self.status[:, index]
            col = self.values[:, index]
            valid = codes == VALID
            finite = valid & np.isfinite(col)
            median = float(np.median(col[finite])) if finite.any() else None
            rows.append(
                (
                    name,
                    str(int(valid.sum())),
                    [
                        str(int(finite.sum())),
                        str(int((valid & ~np.isfinite(col)).sum())),
                        str(int((codes == UNDEFINED).sum())),
                        str(int((codes == INELIGIBLE).sum())),
                        _cell(median),
                    ],
                )
            )
        title = (
            f"world descriptors — {self.n_worlds} worlds from {list(self.sources)} — "
            f"{len(self.definitions)} features (v{DESCRIPTOR_VERSION})"
        )
        return _render_table(
            title,
            ["finite", "infinite", "undefined", "ineligible", "median"],
            rows,
            count_header="valid",
        )


# ---------------------------------------------------------------------------
# Observed datasets
# ---------------------------------------------------------------------------


#: ``dataset.csv`` layout of :func:`~pymc_generator.bundles.write_scm_bundle`:
#: role -> exact label pattern, read when no roles are given. The bundle's
#: ``week`` column is never read as time.
_BUNDLE_LAYOUT: dict[str, str] = {
    "treatments": r"treatment_C\d+",
    "covariates": r"covariate_Z\d+",
    "outcome": r"outcome_Y",
}

#: One series of a role, in messages.
_ROLE_NOUN: dict[str, str] = {
    "treatments": "treatment",
    "covariates": "covariate",
    "outcome": "outcome",
    "time": "time",
}


def _column_positions(labels: Iterable[Hashable]) -> dict[Hashable, list[int]]:
    """Positions of each column label, labels in table order.

    Labels match exactly (hash and equality), so a MultiIndex column is named
    by its full tuple and never by a partial key.
    """
    positions: dict[Hashable, list[int]] = {}
    for position, label in enumerate(labels):
        positions.setdefault(label, []).append(position)
    return positions


def _bundle_roles(labels: Iterable[Hashable]) -> dict[str, list[Hashable]]:
    """Bundle-layout labels per role, each in table order (``labels`` are unique).

    Only ``str`` labels are read, matched exactly. A label that matches only
    once Unicode format characters (category ``Cf``: zero-width spaces,
    byte-order marks) are removed and ``strip()`` and ``casefold()`` applied is
    a near miss: it raises, since skipping it would silently drop a series.
    """
    found: dict[str, list[Hashable]] = {role: [] for role in _BUNDLE_LAYOUT}
    for label in labels:
        if not isinstance(label, str):
            continue
        role = next((r for r, p in _BUNDLE_LAYOUT.items() if re.fullmatch(p, label)), None)
        if role is not None:
            found[role].append(label)
            continue
        folded = "".join(c for c in label if unicodedata.category(c) != "Cf").strip().casefold()
        # The patterns hold only ASCII letters, '_' and '\d': casefolding them is exact.
        near = next(
            (r for r, p in _BUNDLE_LAYOUT.items() if re.fullmatch(p.casefold(), folded)), None
        )
        if near is not None:
            raise ValueError(
                f"column {label!r} reads as a bundle {_ROLE_NOUN[near]} label only once "
                "whitespace, case and invisible characters are ignored; rename it to the exact "
                "bundle label (treatment_C1, covariate_Z1, outcome_Y, ...) or pass treatments=, "
                "covariates= and outcome= explicitly"
            )
    return found


def _label_list(value: Any, role: str) -> list[Hashable]:
    """Ordered column labels (a list, tuple, pandas Index, 1-D array, ...) as a list."""
    if isinstance(value, (str, bytes)):
        raise TypeError(
            f"{role} must be a sequence of column labels, not a bare string; write "
            f"{role}=[{value!r}]"
        )
    if (
        isinstance(value, (Mapping, set, frozenset))
        or not isinstance(value, Iterable)
        or (isinstance(value, np.ndarray) and value.ndim != 1)
    ):
        raise TypeError(
            f"{role} must be an ordered collection of column labels (a list, tuple, pandas "
            f"Index or 1-D array), got {type(value).__name__}"
        )
    # Labels of a numpy array are numpy scalars; messages show them as Python values.
    labels = [
        label.item() if isinstance(label, np.generic) and label.dtype.kind in "biufcSU" else label
        for label in value
    ]
    for label in labels:
        if not isinstance(label, Hashable):
            raise TypeError(f"{role} holds {label!r}, which is not a column label")
    return labels


def _explicit_roles(
    treatments: Any,
    covariates: Any,
    outcome: Any,
    positions: Mapping[Hashable, list[int]],
    multiindex: bool,
) -> dict[str, list[Hashable]]:
    given = {"treatments": treatments, "covariates": covariates, "outcome": outcome}
    missing = [role for role, value in given.items() if value is None]
    if missing:
        raise ValueError(
            f"{' and '.join(missing)} {'is' if len(missing) == 1 else 'are'} missing: give "
            "treatments, covariates and outcome together, or none of them to read the bundle "
            "layout (treatment_C1.., covariate_Z1.., outcome_Y)"
        )
    roles = {role: _label_list(given[role], role) for role in ("treatments", "covariates")}
    _check_one_label(outcome, "outcome", positions, multiindex)
    roles["outcome"] = [outcome]
    return roles


def _check_one_label(
    value: Any, role: str, positions: Mapping[Hashable, list[int]], multiindex: bool
) -> None:
    """``value`` is one label: never a list, a tuple only when it is a column label."""
    if isinstance(value, tuple) and value in positions:
        return
    if isinstance(value, (list, tuple)) or not isinstance(value, Hashable):
        flatten = (
            "; name a MultiIndex column by its full tuple, or flatten the columns first"
            if multiindex
            else ""
        )
        raise TypeError(
            f"{role} must be one column label, got {type(value).__name__} {value!r}; a dataset "
            f"has one {_ROLE_NOUN[role]} column{flatten}"
        )


def _check_labels(
    positions: Mapping[Hashable, list[int]],
    used: Sequence[tuple[str, Hashable]],
    multiindex: bool,
) -> None:
    """Every used label names exactly one column, and plays one role."""
    for role, label in used:
        noun = _ROLE_NOUN[role]
        count = len(positions.get(label, ()))
        if count == 0:
            shown = list(positions)
            hint = "; MultiIndex columns are named by their full tuple" if multiindex else ""
            raise KeyError(
                f"{noun} column {label!r} not found; the columns are "
                f"{shown[:12]}{' ...' if len(shown) > 12 else ''}{hint}"
            )
        if count > 1:
            raise ValueError(
                f"{noun} column {label!r} appears {count} times; rename or drop the "
                "duplicates so that each label names one series"
            )
    seen: dict[Hashable, str] = {}
    for role, label in used:
        if label in seen:
            where = f"twice in {role}" if seen[label] == role else f"as {seen[label]} and {role}"
            raise ValueError(f"column {label!r} is used {where}; each column plays one role")
        seen[label] = role


def _dataset_series(
    frame: Any, labels: Sequence[Hashable], positions: Mapping[Hashable, list[int]]
) -> tuple[dict[Hashable, Any], Any]:
    """One pandas Series per used label, and the row labels errors name.

    Mapping values are taken by position, so their row labels are positions.
    """
    import pandas as pd

    if isinstance(frame, pd.DataFrame):
        return {label: frame.iloc[:, positions[label][0]] for label in labels}, frame.index
    lengths: dict[Hashable, int] = {}
    for label in labels:
        try:
            ndim = np.ndim(frame[label])
        except ValueError:  # a ragged nested sequence
            ndim = -1
        if ndim != 1:
            got = "a scalar" if ndim == 0 else "a ragged sequence" if ndim < 0 else f"{ndim}-D"
            raise ValueError(
                f"mapping value {label!r} must be one-dimensional, one entry per time step; "
                f"got {got}"
            )
        lengths[label] = len(frame[label])
    if len(set(lengths.values())) > 1:
        raise ValueError(
            f"mapping values differ in length {lengths}; every series needs one entry per time step"
        )
    indexed = [label for label in labels if isinstance(frame[label], pd.Series)]
    for label in indexed[1:]:
        if not frame[label].index.equals(frame[indexed[0]].index):
            raise ValueError(
                f"Series {label!r} and {indexed[0]!r} have different indexes; mapping values "
                "are taken by position, so align them first (reindex every Series to one index)"
            )
    series = {
        label: frame[label].reset_index(drop=True)
        if isinstance(frame[label], pd.Series)
        else pd.Series(frame[label])
        for label in labels
    }
    return series, pd.RangeIndex(next(iter(lengths.values())))


def _row_label(rows: Any, position: int) -> str:
    label = rows[position]
    return repr(label.item() if isinstance(label, np.generic) else label)


def _refuse_rows(bad: np.ndarray, where: str, what: str, rows: Any) -> None:
    if bad.any():
        count = int(np.count_nonzero(bad))
        raise ValueError(
            f"{where} has {count} {what} value{'s' if count > 1 else ''}; the first is at row "
            f"{_row_label(rows, int(np.flatnonzero(bad)[0]))}. Impute or trim them first"
        )


def _series_values(series: Any, label: Hashable, role: str, rows: Any) -> np.ndarray:
    """Validated values of one column, in the numpy dtype pandas converts it to."""
    from pandas.api.types import is_bool_dtype, is_numeric_dtype

    where = f"column {label!r} ({role})"
    if is_bool_dtype(series.dtype):
        raise TypeError(
            f"{where} is boolean (dtype {series.dtype}); series must be numeric levels, so "
            "cast flags first (for example .astype(float))"
        )
    if not is_numeric_dtype(series.dtype):
        raise TypeError(
            f"{where} is not numeric (dtype {series.dtype}); convert it first. If it was read "
            "from CSV, check sep=, decimal= and thousands= (for example decimal=',' for values "
            "written as 1,5)"
        )
    # Before conversion: a nullable column with missing values converts to
    # objects or NaN depending on the pandas version.
    _refuse_rows(series.isna().to_numpy(dtype=bool), where, "missing", rows)
    values: np.ndarray = series.to_numpy()
    if values.dtype == object:  # e.g. Arrow decimals
        values = series.to_numpy(dtype=np.float64)
    if values.dtype.kind not in "fiu":
        raise TypeError(f"{where} must hold real numbers, got dtype {values.dtype}")
    _refuse_rows(~np.isfinite(values), where, "non-finite", rows)
    if not values.any():
        drop = "" if role == "outcome" else f"drop the inactive {_ROLE_NOUN[role]}: "
        raise ValueError(
            f"{where} is zero at every time step; {drop}an all-zero series has no level or "
            "variation to describe"
        )
    return values


def _check_time(series: Any, label: Hashable, rows: Any) -> None:
    """Time stamps must parse, be finite and strictly increase down the rows."""
    import pandas as pd
    from pandas.api.types import is_bool_dtype, is_datetime64_any_dtype, is_numeric_dtype

    where = f"time column {label!r}"
    dtype = series.dtype
    numeric = is_numeric_dtype(dtype) and not is_bool_dtype(dtype)
    if isinstance(dtype, pd.PeriodDtype):
        stamps = series.dt.to_timestamp()
    elif numeric or is_datetime64_any_dtype(dtype) or dtype.kind == "m":  # "m": timedeltas
        stamps = series
    else:
        try:
            stamps = pd.to_datetime(series, format="ISO8601", utc=True)
        except (ValueError, TypeError) as exc:
            raise ValueError(
                f"{where} is not ISO 8601 date text{_unparsed_row(series, rows)}; read it with "
                "parse_dates= and date_format= (from_csv), or convert it to datetimes or "
                "numbers first"
            ) from exc
    if numeric:  # missing values read as NaN
        bad = ~np.isfinite(stamps.to_numpy(dtype=np.float64, na_value=np.nan))
    else:
        bad = stamps.isna().to_numpy(dtype=bool)
    if bad.any():
        raise ValueError(
            f"{where} is missing or non-finite at row "
            f"{_row_label(rows, int(np.flatnonzero(bad)[0]))}; every row needs a finite time"
        )
    later = (
        stamps.iloc[1:].reset_index(drop=True) > stamps.iloc[:-1].reset_index(drop=True)
    ).to_numpy(dtype=bool)
    if not later.all():
        first = int(np.flatnonzero(~later)[0]) + 1
        raise ValueError(
            f"{where} does not strictly increase: row {_row_label(rows, first)} is not later "
            "than the row before it. Rows must be in time order, one per time step: sort the "
            "rows by time (and merge repeated time steps) first"
        )


def _unparsed_row(series: Any, rows: Any) -> str:
    """``": row R holds V"`` for the first value ISO 8601 parsing refuses, else ``""``."""
    import pandas as pd

    try:
        parsed = pd.to_datetime(series, format="ISO8601", utc=True, errors="coerce")
    except (ValueError, TypeError):
        return ""
    bad = parsed.isna().to_numpy(dtype=bool) & series.notna().to_numpy(dtype=bool)
    if not bad.any():
        return ""
    first = int(np.flatnonzero(bad)[0])
    return f": row {_row_label(rows, first)} holds {series.iloc[first]!r}"


@dataclass(frozen=True)
class ObservedWorlds:
    """Observed datasets as worlds: series only, no generating process.

    Arrays use the corpus names and axes. One dataset is one world, so a
    single ``(n_time_steps, n_treatments)`` table is passed as
    ``treatment_raw[None]``. All worlds in one instance share a horizon;
    describe other horizons separately and :meth:`WorldDescriptors.concat`.

    A dataset table is read with :meth:`from_frame` (a DataFrame or a mapping
    of columns) or :meth:`from_csv`, and one world is written as a table with
    :meth:`to_frame`. The table format:

    * **Layout.** One table is one dataset, so one world: one row per time
      step, one column per series. Rows are assumed to be in time order and
      equally spaced; only an optional ``time`` column is checked (strictly
      increasing).
    * **Roles.** ``treatments`` and ``covariates`` list one or more column
      labels each and ``outcome`` names one; all three roles are required.
      Give all three, or none to read the bundle layout of
      :func:`~pymc_generator.bundles.write_scm_bundle`: ``str`` labels that
      are exactly ``treatment_C<k>``, ``covariate_Z<k>`` and ``outcome_Y``,
      each role in table order. A label that matches only once whitespace,
      case or invisible characters are ignored is refused, never skipped. No
      column is read as time unless named with ``time=``, not even ``week``.
    * **Values.** Real-valued (integer or float) columns of any numpy, pandas
      nullable or Arrow dtype, with no missing or non-finite value and not
      zero at every step (an all-zero series is an inactive one: drop it).
      Booleans, text and complex numbers are refused. Raw levels are described
      as given: nothing is scaled, differenced or imputed.
    * **Resolution.** Each role keeps the resolution of its columns' common
      dtype (``np.result_type``): float32 columns stay float32-resolution,
      float32 mixed with float64 is float64-resolution.

    Parameters
    ----------
    treatment_raw
        ``(n_worlds, n_time_steps, n_treatments)`` treatment levels.
    covariates
        Optional ``(n_worlds, n_time_steps, n_covariates)`` covariate levels.
        ``None`` means no covariates (covariate features are ineligible).
    outcome_raw
        Optional ``(n_worlds, n_time_steps)`` outcome. ``None`` makes every
        outcome feature ineligible rather than inventing a series.
    treatment_active_mask, covariate_active_mask
        Optional 0/1 ``(n_worlds, width)`` masks for padded columns; default
        all active. Inactive columns must be exactly zero. Values must be
        finite: impute or trim missing observations before describing them.

    Notes
    -----
    Arrays are stored as float64. ``resolution`` keeps the machine epsilon of
    each role's input dtype as ``(treatment, covariates, outcome)`` — float32
    data stay float32-resolution — which sets the rounding floor used to
    recognize constant series.

    :meth:`to_frame` writes each role in the float dtype of its resolution,
    so ``from_frame(o.to_frame(i))`` reproduces world ``i``'s active columns
    and resolution exactly. Through a CSV file, :meth:`from_csv` reads
    float64 values written by pandas back bit for bit, but a float32 world
    comes back as the float64 values of its printed decimals (float64
    resolution).
    """

    treatment_raw: Any
    covariates: Any = None
    outcome_raw: Any = None
    treatment_active_mask: Any = None
    covariate_active_mask: Any = None
    resolution: tuple[float, float, float] = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        roles = (self.treatment_raw, self.covariates, self.outcome_raw)
        eps = tuple(
            _resolution(np.asarray(a).dtype) if a is not None else _resolution(np.dtype(np.float64))
            for a in roles
        )
        treatment = _numeric_float64(self.treatment_raw, "treatment_raw")
        if treatment.ndim != 3:
            raise ValueError(
                "treatment_raw must be (n_worlds, n_time_steps, n_treatments); "
                "wrap one dataset as treatment_raw[None]"
            )
        n_worlds, n_time_steps, n_treatments = treatment.shape
        if n_worlds == 0 or n_time_steps == 0:
            raise ValueError("ObservedWorlds needs at least one world and one time step")
        if self.covariates is None:
            covariates = np.zeros((n_worlds, n_time_steps, 0), dtype=np.float64)
        else:
            covariates = _numeric_float64(self.covariates, "covariates")
            if covariates.ndim != 3 or covariates.shape[:2] != (n_worlds, n_time_steps):
                raise ValueError(
                    f"covariates must be ({n_worlds}, {n_time_steps}, n_covariates), "
                    f"got {covariates.shape}"
                )
        outcome = None
        if self.outcome_raw is not None:
            outcome = _numeric_float64(self.outcome_raw, "outcome_raw")
            if outcome.shape != (n_worlds, n_time_steps):
                raise ValueError(
                    f"outcome_raw must be ({n_worlds}, {n_time_steps}), got {outcome.shape}"
                )
        masks = {}
        for name, arr, width in (
            ("treatment_active_mask", self.treatment_active_mask, n_treatments),
            ("covariate_active_mask", self.covariate_active_mask, covariates.shape[2]),
        ):
            if arr is None:
                masks[name] = np.ones((n_worlds, width), dtype=bool)
                continue
            raw = np.asarray(arr)
            if raw.shape != (n_worlds, width):
                raise ValueError(f"{name} must be ({n_worlds}, {width}), got {raw.shape}")
            if raw.dtype.kind not in "buif" or not np.all((raw == 0) | (raw == 1)):
                raise ValueError(f"{name} must be a binary 0/1 mask")
            masks[name] = raw.astype(bool)
        _check_zero_padding(treatment, masks["treatment_active_mask"], "treatment_raw")
        _check_zero_padding(covariates, masks["covariate_active_mask"], "covariates")
        _require_finite(treatment, "treatment_raw")
        _require_finite(covariates, "covariates")
        if outcome is not None:
            _require_finite(outcome, "outcome_raw")
        object.__setattr__(self, "treatment_raw", treatment)
        object.__setattr__(self, "covariates", covariates)
        object.__setattr__(self, "outcome_raw", outcome)
        object.__setattr__(self, "treatment_active_mask", masks["treatment_active_mask"])
        object.__setattr__(self, "covariate_active_mask", masks["covariate_active_mask"])
        object.__setattr__(self, "resolution", eps)

    @property
    def n_worlds(self) -> int:
        return int(self.treatment_raw.shape[0])

    @property
    def n_time_steps(self) -> int:
        return int(self.treatment_raw.shape[1])

    # -- dataset tables -----------------------------------------------------
    @classmethod
    def from_frame(
        cls,
        frame: pd.DataFrame | Mapping[Hashable, Any],
        *,
        treatments: Iterable[Hashable] | None = None,
        covariates: Iterable[Hashable] | None = None,
        outcome: Hashable | None = None,
        time: Hashable | None = None,
    ) -> ObservedWorlds:
        """Read one dataset, as one world, from a table with one row per time step.

        Parameters
        ----------
        frame
            A pandas DataFrame, or a mapping ``label -> 1-D array-like``.
            Mapping values are taken by position: they must have equal
            lengths and, when several are Series, identical indexes; errors
            then name rows by position.
        treatments, covariates
            Column labels, in the order of the result's columns: a list,
            tuple, pandas Index or 1-D array; never a bare string, a set or a
            mapping.
        outcome
            One column label. A tuple is one label only when the table has
            it as a column (MultiIndex columns are named by their full tuple).
        time
            Optional column label of the time stamps, checked and not
            stored: datetimes (time-zone aware or not), timedeltas and
            numbers are used as they are, periods as their start times, and
            text is parsed as ISO 8601 (offsets converted to UTC). Every row
            needs a finite stamp, strictly later than the row before.

        Returns
        -------
        ObservedWorlds
            One world, every column active: ``treatment_raw`` of shape
            ``(1, T, len(treatments))``, ``covariates`` of shape
            ``(1, T, len(covariates))`` and ``outcome_raw`` of shape ``(1, T)``.

        Notes
        -----
        Checks run in this order and their errors name the column (and, for
        values, the count and the first row label): every used label is one
        column of the table (labels match exactly, so a MultiIndex column is
        named by its full tuple); no label plays two roles, ``time``
        included; at least one treatment and one covariate; then per column,
        booleans (numpy, pandas ``boolean``, Arrow) and other non-numeric
        dtypes are refused, missing values are refused before conversion,
        values convert with ``Series.to_numpy()`` (as float64 when that gives
        objects), and non-finite values and all-zero columns are refused;
        finally ``time``.
        """
        import pandas as pd

        if not isinstance(frame, (pd.DataFrame, Mapping)):
            raise TypeError(
                "frame must be a pandas DataFrame or a mapping of column label -> 1-D values, "
                f"got {type(frame).__name__}"
            )
        if isinstance(frame, pd.DataFrame):
            positions = _column_positions(frame.columns)
            multiindex = isinstance(frame.columns, pd.MultiIndex)
        else:
            positions = _column_positions(frame)
            multiindex = False
        defaults = treatments is None and covariates is None and outcome is None
        if defaults:
            roles = _bundle_roles(positions)
        else:
            roles = _explicit_roles(treatments, covariates, outcome, positions, multiindex)
        used = [(role, label) for role, labels in roles.items() for label in labels]
        if time is not None:
            _check_one_label(time, "time", positions, multiindex)
            used.append(("time", time))
        _check_labels(positions, used, multiindex)
        for role, labels in roles.items():
            if labels:
                continue
            noun = _ROLE_NOUN[role]
            if defaults:
                raise ValueError(
                    f"no {noun} column: with no roles given the bundle layout is read "
                    "(treatment_C1.., covariate_Z1.., outcome_Y) and no label is a bundle "
                    f"{noun} label; name your columns with treatments=[...], covariates=[...] "
                    "and outcome=..."
                )
            raise ValueError(f"{role} is empty; a dataset needs at least one {noun} column")
        series, rows = _dataset_series(frame, [label for _, label in used], positions)
        if len(rows) == 0:
            raise ValueError("the dataset has no rows; it needs one row per time step")
        values = {
            role: [_series_values(series[label], label, role, rows) for label in labels]
            for role, labels in roles.items()
        }
        if time is not None:
            _check_time(series[time], time, rows)
        treatment, covariate = (
            np.stack(values[role], axis=1, dtype=np.result_type(*values[role]))
            for role in ("treatments", "covariates")
        )
        return cls(
            treatment_raw=treatment[None],
            covariates=covariate[None],
            outcome_raw=np.array(values["outcome"][0])[None],
        )

    @classmethod
    def from_csv(
        cls,
        path: str | os.PathLike[str] | IO[Any],
        *,
        treatments: Iterable[Hashable] | None = None,
        covariates: Iterable[Hashable] | None = None,
        outcome: Hashable | None = None,
        time: Hashable | None = None,
        **read_csv_kwargs: Any,
    ) -> ObservedWorlds:
        """Read one dataset from a CSV file: :func:`pandas.read_csv`, then :meth:`from_frame`.

        Parameters
        ----------
        path
            A path, URL or buffer, as :func:`pandas.read_csv` takes it.
        treatments, covariates, outcome, time
            Column roles, as in :meth:`from_frame`: all three of
            ``treatments``, ``covariates`` and ``outcome``, or none of them to
            read the bundle layout.
        **read_csv_kwargs
            Passed to :func:`pandas.read_csv`: ``sep=``, ``decimal=`` and
            ``thousands=`` for the number format, ``parse_dates=`` and
            ``date_format=`` for a time column, ``index_col=`` to name rows in
            errors, ``dtype_backend=``, ...

        Returns
        -------
        ObservedWorlds

        Notes
        -----
        Unless ``float_precision`` is given, floats are parsed with
        ``float_precision="round_trip"``, so float64 values written by pandas
        read back bit for bit; the default parser misses the last bit of many
        of them. Engines and options that cannot take it (``engine="python"``
        or ``"pyarrow"``, regex or multi-character separators, ``skipfooter``,
        ...) read without it. ``sep=None`` sniffs the separator, which only the
        python engine does: it reads with ``engine="python"`` unless an engine
        is given, and decodes with ``encoding="utf-8-sig"`` unless an encoding
        is given or the source is an already decoded text stream, so that a
        byte-order mark does not stay in the first label. The python engine is
        not exact for float64.
        """
        import pandas as pd

        kwargs = dict(read_csv_kwargs)
        sniff = "sep" in kwargs and kwargs["sep"] is None
        if sniff:
            if kwargs.get("engine") is None:
                kwargs["engine"] = "python"
            # The sniffing reader keeps a byte-order mark in the first label;
            # utf-8-sig drops it and reads plain UTF-8 identically. A text
            # stream that is already decoded keeps its own encoding.
            if "encoding" not in kwargs and getattr(path, "encoding", None) is None:
                kwargs["encoding"] = "utf-8-sig"
        if sniff or "float_precision" in kwargs:
            frame = pd.read_csv(path, **kwargs)
        else:
            try:
                frame = pd.read_csv(path, float_precision="round_trip", **kwargs)
            except ValueError as exc:
                # pandas refuses the option while validating options, before
                # reading: a buffer is still at its start for the second read.
                if "float_precision" not in str(exc):
                    raise
                frame = pd.read_csv(path, **kwargs)
        return cls.from_frame(
            frame, treatments=treatments, covariates=covariates, outcome=outcome, time=time
        )

    def to_frame(self, world: int = 0) -> pd.DataFrame:
        """One world as a dataset table in the bundle layout.

        Columns: ``week`` (``0 .. T-1``, int64), ``treatment_C1``, ... (the
        active treatments in slot order), ``covariate_Z1``, ... (the active
        covariates) and ``outcome_Y``. Each role is written in the float dtype
        whose machine epsilon is its stored resolution (float16, float32 or
        float64), so ``ObservedWorlds.from_frame(o.to_frame(i))`` reproduces
        world ``i``'s active columns and resolution exactly. Written to CSV
        and read back, a float32 world becomes the float64 values of its
        printed decimals.

        Parameters
        ----------
        world
            Position of the world, ``0 <= world < n_worlds``.

        Returns
        -------
        pandas.DataFrame

        Raises
        ------
        ValueError
            When ``world`` is not an integer position, or the world is not a
            dataset :meth:`from_frame` reads back: no active treatment, no
            active covariate, no outcome, or an active column or the outcome
            zero at every time step.
        """
        import pandas as pd

        if (
            isinstance(world, (bool, np.bool_))
            or not isinstance(world, (int, np.integer))
            or not 0 <= world < self.n_worlds
        ):
            raise ValueError(f"world must be an integer in [0, {self.n_worlds}), got {world!r}")
        position = int(world)
        active = {
            "treatment": np.flatnonzero(self.treatment_active_mask[position]),
            "covariate": np.flatnonzero(self.covariate_active_mask[position]),
        }
        for noun, slots in active.items():
            if slots.size == 0:
                raise ValueError(
                    f"world {position} has no active {noun}; a dataset needs at least one"
                )
        if self.outcome_raw is None:
            raise ValueError("these worlds have no outcome; a dataset needs one")
        treatment_dtype, covariate_dtype, outcome_dtype = (
            _resolution_dtype(eps) for eps in self.resolution
        )
        data: dict[str, np.ndarray] = {"week": np.arange(self.n_time_steps, dtype=np.int64)}
        for noun, prefix, block, dtype in (
            ("treatment", "treatment_C", self.treatment_raw[position], treatment_dtype),
            ("covariate", "covariate_Z", self.covariates[position], covariate_dtype),
        ):
            for k, slot in enumerate(active[noun].tolist(), start=1):
                if not block[:, slot].any():
                    raise ValueError(
                        f"world {position}: active {noun} column {slot} is zero at every time "
                        f"step; mark it inactive in {noun}_active_mask (from_frame refuses "
                        "all-zero columns)"
                    )
                data[f"{prefix}{k}"] = block[:, slot].astype(dtype)
        outcome = self.outcome_raw[position]
        if not outcome.any():
            raise ValueError(
                f"world {position}: the outcome is zero at every time step (from_frame refuses "
                "all-zero columns)"
            )
        data["outcome_Y"] = outcome.astype(outcome_dtype)
        return pd.DataFrame(data)


# ---------------------------------------------------------------------------
# Chunked extraction
# ---------------------------------------------------------------------------


def _resolution(dtype: np.dtype) -> float:
    """Machine epsilon of a dtype's values once converted to float64.

    Statistics run in float64, so a finer dtype (x86 longdouble) cannot set a
    floor below float64 rounding; exact integer data get float64's.
    """
    eps64 = float(np.finfo(np.float64).eps)
    if np.issubdtype(dtype, np.floating):
        return max(float(np.finfo(dtype).eps), eps64)
    return eps64


def _resolution_dtype(eps: float) -> np.dtype:
    """The float dtype (float16, float32 or float64) whose machine epsilon is ``eps``."""
    for dtype in (np.float16, np.float32, np.float64):
        if float(np.finfo(dtype).eps) == eps:
            return np.dtype(dtype)
    raise ValueError(f"no float16, float32 or float64 dtype has machine epsilon {eps}")


@dataclass(frozen=True)
class _Chunk:
    """Observable float64 arrays for one chunk of worlds."""

    treatment: np.ndarray  # (W, T, K)
    covariates: np.ndarray  # (W, T, M)
    outcome: np.ndarray | None  # (W, T)
    treatment_mask: np.ndarray  # (W, K) bool
    covariate_mask: np.ndarray  # (W, M) bool
    truth: dict[str, np.ndarray] | None  # truth feature -> (W,) net share (NaN undefined)
    # Machine epsilon of the source data per role (treatment, covariates,
    # outcome): a scalar, or (W,) when worlds of one chunk differ in dtype.
    resolution: tuple[Any, Any, Any]


_CORPUS_OBSERVABLE_KEYS = (
    "treatment_raw",
    "covariates",
    "outcome_raw",
    "treatment_active_mask",
    "covariate_active_mask",
)


def _truth_shares(source: Any) -> dict[str, np.ndarray]:
    budget = _contribution_budget(source, "base_direct_plus_indirect", DEFAULT_QUANTILES)
    shares = budget.measures["net_share"]
    return {name: shares[:, budget._index(key)] for name, key in TRUTH_FEATURES.items()}


def _check_padding(arr: np.ndarray, mask: np.ndarray, name: str, positions: np.ndarray) -> None:
    """Inactive columns must be exactly zero; errors name the SOURCE world position."""
    inactive = ~mask
    if not inactive.any():
        return
    bad = ~np.all(arr.transpose(0, 2, 1)[inactive] == 0.0, axis=1)
    if bad.any():
        worlds, columns = np.nonzero(inactive)
        first = int(np.flatnonzero(bad)[0])
        raise ValueError(
            f"{name} has non-zero inactive padding at world {int(positions[worlds[first]])}, "
            f"column {int(columns[first])}: inactive columns must be exactly zero"
        )


def _corpus_layout(corpus: Mapping[str, Any]) -> tuple[int, int]:
    missing = [k for k in _CORPUS_OBSERVABLE_KEYS if k not in corpus]
    if missing:
        raise KeyError(f"corpus is missing keys required for world descriptors: {missing}")
    outcome = np.asarray(corpus["outcome_raw"])
    if outcome.ndim != 2:
        raise ValueError(f"outcome_raw must be (n_tasks, n_time_steps), got shape {outcome.shape}")
    n_total, n_time_steps = int(outcome.shape[0]), int(outcome.shape[1])
    treatment = np.asarray(corpus["treatment_raw"])
    covariates = np.asarray(corpus["covariates"])
    for key, arr in (("treatment_raw", treatment), ("covariates", covariates)):
        if arr.ndim != 3 or arr.shape[:2] != (n_total, n_time_steps):
            raise ValueError(f"{key} must be ({n_total}, {n_time_steps}, width), got {arr.shape}")
    for key, width in (
        ("treatment_active_mask", treatment.shape[2]),
        ("covariate_active_mask", covariates.shape[2]),
    ):
        mask = np.asarray(corpus[key])
        if mask.shape != (n_total, width):
            raise ValueError(f"{key} must be ({n_total}, {width}), got {mask.shape}")
    return n_total, n_time_steps


def _mask_chunk(raw: Any, key: str) -> np.ndarray:
    arr = np.asarray(raw)
    if arr.dtype.kind not in "buif" or not np.all((arr == 0) | (arr == 1)):
        raise ValueError(f"{key} must be a binary 0/1 mask")
    return arr.astype(bool)


def _corpus_chunk(
    corpus: Mapping[str, Any],
    idx: np.ndarray,
    include_truth: bool,
    resolution: tuple[float, float, float],
) -> _Chunk:
    treatment = _numeric_float64(np.asarray(corpus["treatment_raw"])[idx], "treatment_raw")
    covariates = _numeric_float64(np.asarray(corpus["covariates"])[idx], "covariates")
    outcome = _numeric_float64(np.asarray(corpus["outcome_raw"])[idx], "outcome_raw")
    tmask = _mask_chunk(np.asarray(corpus["treatment_active_mask"])[idx], "treatment_active_mask")
    cmask = _mask_chunk(np.asarray(corpus["covariate_active_mask"])[idx], "covariate_active_mask")
    _check_padding(treatment, tmask, "treatment_raw", idx)
    _check_padding(covariates, cmask, "covariates", idx)
    for name, arr in (("treatment_raw", treatment), ("covariates", covariates)):
        _require_finite(arr, name)
    _require_finite(outcome, "outcome_raw")
    truth = None
    if include_truth:
        # Slice first: the diagnostic extractor converts whole arrays to float64.
        sub = {
            key: np.asarray(value)[idx]
            for key, value in corpus.items()
            if isinstance(value, np.ndarray)
            and value.ndim >= 1
            and value.shape[0] == np.asarray(corpus["outcome_raw"]).shape[0]
        }
        try:
            truth = _truth_shares(_corpus_source(sub, None, need_counterfactual=False))
        except ValueError as exc:
            shown = ", ".join(str(int(i)) for i in idx[:8]) + (", ..." if idx.size > 8 else "")
            raise ValueError(
                f"truth extraction failed for the chunk of source worlds [{shown}]; "
                f"'world k' below is the k-th world of that list: {exc}"
            ) from exc
    return _Chunk(treatment, covariates, outcome, tmask, cmask, truth, resolution)


def _worlds_chunk(
    scms: Sequence[SCM], idx: np.ndarray, include_truth: bool, widths: tuple[int, int]
) -> _Chunk:
    """Observable arrays straight from ``SCM.data``, padded to the sequence widths.

    Only the observables are read unless truth is requested, so optional audit
    keys never make extraction depend on which worlds share a chunk.
    """
    chosen = [scms[int(i)] for i in idx]
    n_time = int(np.asarray(chosen[0].data["outcome"]).shape[0])
    blocks = {}
    for key, width in (("treatments", widths[0]), ("covariates", widths[1])):
        dense = np.zeros((len(chosen), n_time, width), dtype=np.float64)
        for row, (world, position) in enumerate(zip(chosen, idx)):
            arr = _require_finite(
                _numeric_float64(world.data[key], f"{key} (world {int(position)})"),
                f"{key} (world {int(position)})",
            )
            if arr.ndim != 2 or arr.shape[0] != n_time:
                raise ValueError(
                    f"{key} must be (n_time_steps, width) in world {int(position)}, got {arr.shape}"
                )
            dense[row, :, : arr.shape[1]] = arr
        blocks[key] = dense
    outcome = np.stack(
        [
            _require_finite(
                _numeric_float64(w.data["outcome"], f"outcome (world {int(i)})"),
                f"outcome (world {int(i)})",
            )
            for w, i in zip(chosen, idx)
        ]
    )
    counts = {
        "treatments": np.array([int(w.n_treatments) for w in chosen]),
        "covariates": np.array([int(w.n_covariates) for w in chosen]),
    }
    resolution = tuple(
        np.array([_resolution(np.asarray(w.data[key]).dtype) for w in chosen])
        for key in ("treatments", "covariates", "outcome")
    )
    truth = None
    if include_truth:
        try:
            truth = _truth_shares(_worlds_source(chosen, None, need_counterfactual=False))
        except ValueError as exc:
            shown = ", ".join(str(int(i)) for i in idx[:8]) + (", ..." if idx.size > 8 else "")
            raise ValueError(
                f"truth extraction failed for the chunk of source worlds [{shown}]; "
                f"'world k' below is the k-th world of that list: {exc}"
            ) from exc
    return _Chunk(
        treatment=blocks["treatments"],
        covariates=blocks["covariates"],
        outcome=outcome,
        treatment_mask=np.arange(widths[0])[None, :] < counts["treatments"][:, None],
        covariate_mask=np.arange(widths[1])[None, :] < counts["covariates"][:, None],
        truth=truth,
        resolution=(resolution[0], resolution[1], resolution[2]),
    )


def _observed_chunk(observed: ObservedWorlds, idx: np.ndarray) -> _Chunk:
    return _Chunk(
        treatment=observed.treatment_raw[idx],
        covariates=observed.covariates[idx],
        outcome=None if observed.outcome_raw is None else observed.outcome_raw[idx],
        treatment_mask=observed.treatment_active_mask[idx],
        covariate_mask=observed.covariate_active_mask[idx],
        truth=None,
        resolution=observed.resolution,
    )


# ---------------------------------------------------------------------------
# Statistics for one chunk
# ---------------------------------------------------------------------------


def _reduce(
    values: np.ndarray,
    valid: np.ndarray,
    eligible: np.ndarray,
    reducer: str,
) -> tuple[np.ndarray, np.ndarray]:
    """Reduce ``(W, n)`` per-node values over valid entries, extended reals.

    Status is ``ineligible`` with no eligible entry, ``undefined`` with no
    valid entry (or when the extended-real reduction itself is undefined,
    e.g. the median of ``-inf`` and ``+inf``).
    """
    n_worlds = values.shape[0]
    out = np.full(n_worlds, np.nan)
    status = np.full(n_worlds, UNDEFINED, dtype=np.uint8)
    if values.shape[1] == 0:
        status[:] = INELIGIBLE
        return out, status
    status[~eligible.any(axis=1)] = INELIGIBLE
    rows = valid.any(axis=1)
    if rows.any():
        masked = np.where(valid[rows], values[rows], np.nan)
        with np.errstate(invalid="ignore", over="ignore"):
            if reducer == "median":
                reduced = np.nanmedian(masked, axis=1)
                # NumPy averages the two middle values by summing first.
                # Repair only overflowed finite midpoints; real infinities
                # and all ordinary legacy reductions keep their semantics.
                for row in np.flatnonzero(np.isinf(reduced)):
                    ordered = np.sort(masked[row, ~np.isnan(masked[row])])
                    low, high = ordered[(ordered.size - 1) // 2], ordered[ordered.size // 2]
                    if np.isfinite(low) and np.isfinite(high):
                        reduced[row] = low / 2.0 + high / 2.0
            elif reducer == "min":
                reduced = np.nanmin(masked, axis=1)
            else:
                reduced = np.nanmax(masked, axis=1)
        defined = ~np.isnan(reduced)
        target = np.flatnonzero(rows)
        out[target[defined]] = reduced[defined]
        status[target[defined]] = VALID
    return out, status


@dataclass(frozen=True)
class _Flatness:
    """Rounding-safe degeneracy flags per ``(world, series)``.

    ``np.std`` of an exactly constant series is the rounding residue of its
    mean, which reaches a few ``eps * |c|`` — so any std-based floor misses a
    share of true constants. The exact range does not: ``max - min`` is exactly
    0 for a constant and at most one ULP for one-ULP jitter. First differences
    inherit up to one ULP of the LEVEL magnitude each, so their floor is scaled
    by the level magnitude, not by the (tiny) differences. A mean whose size is
    within its own summation error counts as zero.
    """

    level: np.ndarray  # the level series is constant
    steps: np.ndarray  # the first differences are constant (a linear ramp, or constant)
    zero_mean: np.ndarray  # |mean| is within summation error of zero


#: Rounding floor in units of the source resolution (machine epsilon): a few
#: ULPs, so one- or two-ULP storage jitter still reads as constant.
_FLAT_ULPS = 4.0


def _flatness(block: np.ndarray, slots: np.ndarray, resolution: np.ndarray) -> _Flatness:
    """``resolution`` is the ``(W, P)`` machine epsilon of each series' source dtype."""
    tol = _FLAT_ULPS * resolution
    n_time = block.shape[2]
    scale = np.maximum(np.abs(slots[..., 4]), np.abs(slots[..., 5]))
    level = (slots[..., 5] - slots[..., 4]) <= tol * scale
    if n_time >= 2:
        steps_raw = np.diff(block, axis=2)
        spread = steps_raw.max(axis=2) - steps_raw.min(axis=2)
        # Each step inherits the rounding of two level values.
        steps = level | (spread <= 2.0 * tol * scale)
    else:
        steps = level.copy()
    # The mean is taken in float64 over exactly converted and rescaled values: its
    # summation error grows as T * eps64, while input-storage rounding (e.g. data
    # demeaned in float32) stays at a few eps_in * max|x| whatever T is.
    eps64 = float(np.finfo(np.float64).eps)
    zero_floor = min(max(n_time, 1), 32) * resolution + max(n_time, 1) * eps64
    # A series of one sign (not all zero) has a mean of that sign: never zero.
    lo, hi = slots[..., 4], slots[..., 5]
    one_signed = ((lo >= 0.0) & (hi > 0.0)) | ((hi <= 0.0) & (lo < 0.0))
    zero_mean = (np.abs(slots[..., 0]) <= zero_floor * scale) & ~one_signed
    return _Flatness(level=level, steps=steps, zero_mean=zero_mean)


def _node_statistics(
    block: np.ndarray,
    eligible: np.ndarray,
    lags: tuple[int, ...],
    slots: np.ndarray,
    slot_valid: np.ndarray,
    flat: _Flatness,
) -> dict[tuple[str, int | None], tuple[np.ndarray, np.ndarray]]:
    """``(statistic, lag) -> ((W, P) values, (W, P) valid)`` for every node."""
    n_worlds, n_series, n_time = block.shape
    base = slot_valid[..., 0]
    std = np.where(flat.level, 0.0, slots[..., 1])
    abs_mean = np.where(flat.zero_mean, 0.0, np.abs(slots[..., 0]))
    out: dict[tuple[str, int | None], tuple[np.ndarray, np.ndarray]] = {}

    with np.errstate(divide="ignore", invalid="ignore"):
        ltv = np.where(std > 0.0, abs_mean / np.where(std > 0.0, std, 1.0), np.inf)
        cv = np.where(abs_mean > 0.0, std / np.where(abs_mean > 0.0, abs_mean, 1.0), 0.0)
    # 0/0: an all-zero series has no level-to-variation ratio at all.
    ltv_valid = base & ~((std == 0.0) & (abs_mean == 0.0))
    # Signal-gate convention (signal_diagnostics._coefficient_of_variation):
    # all-zero series have CV 0; a varying series with zero mean has none.
    cv_valid = base & ~((abs_mean == 0.0) & (std > 0.0))
    out[("level_to_variation", None)] = (np.where(ltv_valid, ltv, np.nan), ltv_valid)
    out[("cv", None)] = (np.where(cv_valid, cv, np.nan), cv_valid)
    zero_frac = (block == 0.0).mean(axis=2) if n_time else np.full(std.shape, np.nan)
    out[("zero_fraction", None)] = (np.where(base, zero_frac, np.nan), base)
    # A constant has no texture (the _hf_ratio sd == 0 convention) and a ramp is
    # perfectly smooth; neither has an outlying step. Rounding residue otherwise
    # shows up as roughness ~1 and a spike of a few units.
    # Too few steps and roughness/spike are fixed by arithmetic, not by the data
    # (roughness is 0 at T=2, spike 1 at T=3): the data_diagnostics pair floor.
    enough_steps = n_time - 1 >= _MIN_PAIRS
    rough_valid = slot_valid[..., 6] & enough_steps
    spike_valid = slot_valid[..., 7] & enough_steps
    roughness = np.where(flat.steps, 0.0, slots[..., 6])
    spike = np.where(flat.steps, 0.0, slots[..., 7])
    out[("roughness", None)] = (np.where(rough_valid, roughness, np.nan), rough_valid)
    out[("spike", None)] = (np.where(spike_valid, spike, np.nan), spike_valid)

    series = block.reshape(n_worlds * n_series, n_time)
    acf, acf_valid = _acf_world(series, lags)
    dacf, dacf_valid = _acf_world(np.diff(series, axis=1), lags)
    for column, lag in enumerate(lags):
        for stat, values, valid, degenerate in (
            ("acf", acf, acf_valid, flat.level),
            ("diff_acf", dacf, dacf_valid, flat.steps),
        ):
            v = values[:, column].reshape(n_worlds, n_series)
            ok = valid[:, column].reshape(n_worlds, n_series) & eligible & ~degenerate
            out[(stat, lag)] = (np.where(ok, v, np.nan), ok)
    return out


def _abs_pearson(block: np.ndarray, nonconstant: np.ndarray) -> np.ndarray:
    """``(W, P, P)`` |Pearson| per world over the given non-constant series."""
    centred = block - block.mean(axis=2, keepdims=True)
    norms = np.sqrt((centred**2).sum(axis=2))
    scaled = centred / np.where(nonconstant & (norms > 0.0), norms, 1.0)[..., None]
    # Unit vectors: a projection outside [-1, 1] is rounding, so clipping is repair.
    corr: np.ndarray = np.abs(np.clip(np.einsum("wpt,wqt->wpq", scaled, scaled), -1.0, 1.0))
    return corr


def _pair_index(role: str, n_treat: int, n_cov: int) -> tuple[np.ndarray, np.ndarray]:
    t = np.arange(n_treat)
    c = n_treat + np.arange(n_cov)
    y = n_treat + n_cov
    if role == "treatment_pair":
        i, j = np.triu_indices(n_treat, k=1)
        return t[i], t[j]
    if role == "covariate_pair":
        i, j = np.triu_indices(n_cov, k=1)
        return c[i], c[j]
    if role == "treatment_covariate":
        ti, ci = np.meshgrid(t, c, indexing="ij")
        return ti.reshape(-1), ci.reshape(-1)
    return t, np.full(n_treat, y)


def _chunk_features(
    chunk: _Chunk,
    definitions: tuple[FeatureDefinition, ...],
    statistics: tuple[str, ...],
    lags: tuple[int, ...],
    dependence_lags: tuple[int, ...],
) -> tuple[np.ndarray, np.ndarray]:
    n_worlds, n_time, n_treat = chunk.treatment.shape
    n_cov = chunk.covariates.shape[2]
    outcome = chunk.outcome if chunk.outcome is not None else np.zeros((n_worlds, n_time))
    raw = np.concatenate(
        [
            chunk.treatment.transpose(0, 2, 1),
            chunk.covariates.transpose(0, 2, 1),
            outcome[:, None, :],
        ],
        axis=1,
    )
    eligible = np.concatenate(
        [
            chunk.treatment_mask,
            chunk.covariate_mask,
            np.full((n_worlds, 1), chunk.outcome is not None),
        ],
        axis=1,
    )
    columns = {
        "treatment": slice(0, n_treat),
        "covariate": slice(n_treat, n_treat + n_cov),
        "outcome": slice(n_treat + n_cov, n_treat + n_cov + 1),
    }

    values = np.full((n_worlds, len(definitions)), np.nan)
    status = np.full((n_worlds, len(definitions)), UNDEFINED, dtype=np.uint8)
    # Each series is judged at its own source resolution: a float32 covariate
    # never loosens the floor for a float64 treatment in the same world.
    resolution = np.empty(eligible.shape, dtype=np.float64)
    for role, eps in zip(("treatment", "covariate", "outcome"), chunk.resolution):
        resolution[:, columns[role]] = np.reshape(np.asarray(eps, dtype=np.float64), (-1, 1))
    nodes: dict[tuple[str, int | None], tuple[np.ndarray, np.ndarray]] = {}
    pearson: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    if any(s in statistics for s in DEFAULT_STATISTICS):
        # Preserve legacy scale-invariant calculations exactly. New amplitude
        # and activity diagnostics below must instead receive original levels.
        _, exponent = np.frexp(np.abs(raw).max(axis=2))
        block = np.ascontiguousarray(np.ldexp(raw, -exponent[..., None]))
        slots, slot_valid = _compute_slots(block, eligible)
        flat = _flatness(block, slots, resolution)
        if any(s in statistics for s in _LEGACY_NODE_STATISTICS):
            nodes = _node_statistics(block, eligible, lags, slots, slot_valid, flat)
        constant = flat.level & slot_valid[..., 1]
        for stat, offset, degenerate in (
            ("abs_pearson", 0, flat.level),
            ("diff_abs_pearson", 1, flat.steps),
        ):
            # Too-short views are undefined below; skip the empty-slice arithmetic.
            if stat in statistics and n_time - offset >= _MIN_PAIRS:
                view = np.diff(block, axis=2) if offset else block
                pearson[stat] = (_abs_pearson(view, ~degenerate), ~degenerate)

    source_values = raw.transpose(0, 2, 1)
    if any(s in statistics for s in _TEMPORAL_STATISTICS):
        temporal = temporal_statistics(
            source_values, lags=lags, eligible=eligible, resolution=resolution
        )
        for stat in _TEMPORAL_STATISTICS:
            if stat not in statistics:
                continue
            if stat in ("overlap_pearson", "diff_overlap_pearson"):
                for position, lag in enumerate(temporal.lags):
                    nodes[(stat, lag)] = (
                        temporal.values[stat][..., position],
                        temporal.valid[stat][..., position],
                    )
            else:
                nodes[(stat, None)] = (temporal.values[stat], temporal.valid[stat])
    if any(s in statistics for s in ACTIVITY_STATISTICS):
        # Exact nonzero activity is a property of raw data. Rescaling can
        # underflow tiny nonzero values within a wide-range series into zero.
        activity = activity_runs(source_values, eligible=eligible)
        for stat in ACTIVITY_STATISTICS:
            if stat in statistics:
                nodes[(stat, None)] = (activity.values[stat], activity.valid[stat])

    lagged: dict[tuple[str, str, int | None], tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
    lagged_statistics = tuple(s for s in _LAGGED_PAIR_STATISTICS if s in statistics)
    if lagged_statistics:
        views = tuple(
            "differences" if s == "diff_lagged_pearson" else "levels" for s in lagged_statistics
        )
        lag_groups = (
            tuple(lag for lag in dependence_lags if lag == 0),
            tuple(lag for lag in dependence_lags if lag != 0),
        )
        for pair_role in PAIR_ROLES:
            for pair_lags in lag_groups:
                if not pair_lags:
                    continue
                i, j = _pair_index(pair_role, n_treat, n_cov)
                if pair_lags[0] != 0 and pair_role in ("treatment_pair", "covariate_pair"):
                    i, j = np.concatenate((i, j)), np.concatenate((j, i))
                dependence = lagged_dependence(
                    source_values,
                    pairs=np.column_stack((i, j)),
                    lags=pair_lags,
                    views=views,
                    eligible=eligible,
                    resolution=resolution,
                )
                for view_position, stat in enumerate(lagged_statistics):
                    for lag_position, lag in enumerate(dependence.lags):
                        lagged[(stat, pair_role, lag)] = (
                            dependence.correlation[:, view_position, :, lag_position],
                            dependence.valid[:, view_position, :, lag_position],
                            dependence.eligible,
                        )

    for f, d in enumerate(definitions):
        if d.role == "truth":
            if chunk.truth is None:  # pragma: no cover - guarded by the caller
                raise AssertionError("truth features requested without truth arrays")
            share = chunk.truth[d.name]
            ok = ~np.isnan(share)
            values[ok, f] = share[ok]
            status[ok, f] = VALID
            continue
        if d.statistic in NODE_STATISTICS:
            node_values, node_valid = nodes[(d.statistic, d.lag)]
            cols = columns[d.role]
            if d.role == "outcome":
                if chunk.outcome is None:
                    status[:, f] = INELIGIBLE
                    continue
                ok = node_valid[:, cols][:, 0]
                values[ok, f] = node_values[:, cols][ok, 0]
                status[ok, f] = VALID
                continue
            col, code = _reduce(
                node_values[:, cols], node_valid[:, cols], eligible[:, cols], str(d.reducer)
            )
            values[:, f], status[:, f] = col, code
            continue
        if d.statistic == "constant_fraction":
            cols = columns[d.role]
            active = eligible[:, cols]
            n_active = active.sum(axis=1)
            has = n_active > 0
            values[has, f] = (constant[:, cols] & active)[has].sum(axis=1) / n_active[has]
            status[has, f] = VALID
            status[~has, f] = INELIGIBLE
            continue
        if d.statistic in _LAGGED_PAIR_STATISTICS:
            pair_values, pair_valid, pair_eligible = lagged[(d.statistic, d.role, d.lag)]
            values[:, f], status[:, f] = _reduce(
                pair_values, pair_valid, pair_eligible, str(d.reducer)
            )
            continue
        # pair statistics
        n_view = n_time - (1 if d.statistic == "diff_abs_pearson" else 0)
        i, j = _pair_index(d.role, n_treat, n_cov)
        if i.size == 0:
            status[:, f] = INELIGIBLE
            continue
        pair_eligible = eligible[:, i] & eligible[:, j]
        if n_view < _MIN_PAIRS:
            status[:, f] = np.where(pair_eligible.any(axis=1), UNDEFINED, INELIGIBLE)
            continue
        corr, nonconstant = pearson[d.statistic]
        pair_valid = pair_eligible & nonconstant[:, i] & nonconstant[:, j]
        pair_values = corr[:, i, j]
        col, code = _reduce(pair_values, pair_valid, pair_eligible, str(d.reducer))
        values[:, f], status[:, f] = col, code
    values[status != VALID] = np.nan
    return values, status


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def _validate_choice(values: Any, allowed: Sequence[str], what: str) -> tuple[str, ...]:
    if isinstance(values, str) or not isinstance(values, Sequence):
        raise TypeError(f"{what} must be a non-string sequence, got {type(values).__name__}")
    names = tuple(str(v) for v in values)
    if not names:
        raise ValueError(f"{what} is empty; choose from {list(allowed)}")
    unknown = [n for n in names if n not in allowed]
    if unknown:
        raise ValueError(f"unknown {what} {unknown}; expected {list(allowed)}")
    if len(set(names)) != len(names):
        raise ValueError(f"{what} repeats an entry: {list(names)}")
    # Canonical order, so equal requests produce equal definitions.
    return tuple(a for a in allowed if a in names)


def _id_override(values: Any, n: int, what: str, *, minimum: int) -> np.ndarray:
    arr = np.asarray(values)
    if arr.shape != (n,) or arr.dtype.kind not in "iu":
        raise ValueError(f"{what} must be an integer array with one entry per selected world ({n})")
    if arr.dtype.kind == "u" and arr.size and int(arr.max()) > np.iinfo(np.int64).max:
        raise ValueError(f"{what} must fit in int64")
    if (arr < minimum).any():
        raise ValueError(f"{what} must be >= {minimum}")
    return arr.astype(np.int64)


def world_descriptors(
    source: Mapping[str, Any] | Sequence[SCM] | ObservedWorlds,
    *,
    statistics: Sequence[str] = DEFAULT_STATISTICS,
    lags: Sequence[int] = (1,),
    dependence_lags: Sequence[int] | None = None,
    reducers: Sequence[str] = REDUCERS,
    include_truth: bool = False,
    worlds: Any = None,
    source_id: str,
    world_ids: Any = None,
    group_ids: Any = None,
    chunk_size: int = _DEFAULT_CHUNK,
) -> WorldDescriptors:
    """Describe every world with one row of named, scale-aware statistics.

    Parameters
    ----------
    source
        A corpus mapping (``sample_prior_predictive`` /
        ``DataGenerator.generate`` / ``load_corpus``), a sequence of
        :class:`~pymc_generator.worlds.SCM` worlds sharing one horizon, or
        :class:`ObservedWorlds`.
    statistics
        Statistic families from :data:`STATISTICS`. Defaults to the unchanged
        legacy families in :data:`DEFAULT_STATISTICS`; temporal, activity and
        signed lagged pair features are opt-in. Per-series statistics are
        computed for every active treatment, covariate and the outcome;
        treatment/covariate values are then reduced with every reducer.
    lags
        Positive lags for legacy ``acf`` / ``diff_acf`` and opt-in
        ``overlap_pearson`` / ``diff_overlap_pearson``. Legacy ACF keeps its
        global-mean, full-denominator definition; overlap correlations center
        each overlapping segment separately.
    dependence_lags
        Explicit signed integer lags for ``lagged_pearson`` and
        ``diff_lagged_pearson``; required and nonempty when either is selected.
        At lag h, correlate left[t] with right[t + h]. Treatment-covariate and
        treatment-outcome pairs always put treatment on the left. Within-role
        nonzero lags reduce both ordered distinct directions; lag zero uses
        unordered pairs. No causal or carryover interpretation is implied.
    reducers
        Reducers across active nodes and pairs, from :data:`REDUCERS`.
    include_truth
        Add :data:`TRUTH_FEATURES`, the net contribution shares of outcome.
        Generated sources only; they are not observable.
    worlds
        World selector (None, slice, position(s) or boolean mask; no
        repeats). Selected positions become the default ``world_ids``.
    source_id
        Required name of this collection. Any non-empty string; it only
        identifies rows. Identity is ``(source_id, world_id)`` and groups are
        ``(source_id, group_id)``, so collections described separately must get
        different names — there is no default, because a shared default would
        make unrelated worlds look identical to the between-world analyses.
    world_ids
        Optional per-selected-world integer ids overriding source positions
        (e.g. original ids of an already-sliced shard).
    group_ids
        Optional per-selected-world generation-group ids (``-1`` unknown).
        Defaults to a corpus's ``cell_id`` and to unknown otherwise.
    chunk_size
        Worlds processed at once; bounds float64 working memory.

    Returns
    -------
    WorldDescriptors

    Notes
    -----
    Moments use population ``std`` (``ddof=0``), exactly as ``data_diagnostics``;
    roughness, spike and ACF reuse its definitions. Level-to-variation and CV are
    invariant to multiplying a series by a positive constant but not to
    shifting it.
    Opt-in ``level_std`` and ``difference_std`` retain original units and scale
    with the absolute input multiplier. ``time_correlation`` is signed Pearson
    correlation with ``0, ..., T-1``. Activity summaries use exact raw nonzero
    values, include boundary-censored spells in total counts and longest
    lengths, and exclude them from complete-spell medians. No complete spell
    makes its median undefined, not zero. ``interior_inactive_fraction`` divides
    complete interior inactive length by the full window length.

    Degenerate series are detected from exact ranges, never from a computed
    ``std``, whose rounding residue misses true constants. With ``eps`` the
    machine epsilon of the input dtype, a series is constant when
    ``max - min <= 4 * eps * max|x|``, its steps are constant (a linear ramp)
    when their range is at most ``8 * eps * max|x|``, and its mean is zero when
    ``|mean| <= (min(T, 32) * eps + T * eps64) * max|x|`` (input rounding plus
    float64 summation error) and it takes both signs. Nonzero constants give
    ``level_to_variation = inf`` (an all-zero series has none: it is undefined),
    ``cv = 0``, roughness and spike 0, and undefined ACF and correlations; ramps
    give roughness and spike 0 and undefined differenced correlations. No absolute
    epsilon is applied. Legacy scale-invariant statistics first rescale each
    series exactly by a power of two, so magnitudes from ``1e-300`` to ``1e300``
    describe identically. New diagnostics handle their own safe scaling without
    changing amplitude units or exact activity states. Statuses are exactly
    independent of ``chunk_size``; values agree to floating-point rounding.
    """
    stats = _validate_choice(statistics, STATISTICS, "statistics")
    reds = _validate_choice(reducers, REDUCERS, "reducers")
    if not isinstance(source_id, str) or not source_id:
        raise ValueError("source_id must be a non-empty string")
    if isinstance(chunk_size, bool) or not isinstance(chunk_size, (int, np.integer)):
        raise TypeError("chunk_size must be a positive integer")
    if chunk_size < 1:
        raise ValueError("chunk_size must be a positive integer")

    extract: Any
    groups_default: np.ndarray | None = None
    if isinstance(source, ObservedWorlds):
        if include_truth:
            raise ValueError(
                "include_truth needs generated worlds; observed data carry no contribution truth"
            )
        observed = source
        n_total, n_time_steps = observed.n_worlds, observed.n_time_steps

        def extract(chunk_idx: np.ndarray) -> _Chunk:
            return _observed_chunk(observed, chunk_idx)

    elif isinstance(source, Mapping):
        corpus = source
        n_total, n_time_steps = _corpus_layout(corpus)
        if "cell_id" in corpus:
            cells = np.asarray(corpus["cell_id"])
            if cells.shape != (n_total,) or cells.dtype.kind not in "iu" or (cells < 0).any():
                raise ValueError("corpus cell_id must be non-negative integers, one per world")
            groups_default = cells.astype(np.int64)
        corpus_resolution = (
            _resolution(np.asarray(corpus["treatment_raw"]).dtype),
            _resolution(np.asarray(corpus["covariates"]).dtype),
            _resolution(np.asarray(corpus["outcome_raw"]).dtype),
        )

        def extract(chunk_idx: np.ndarray) -> _Chunk:
            return _corpus_chunk(corpus, chunk_idx, include_truth, corpus_resolution)

    elif isinstance(source, Sequence) and not isinstance(source, str):
        scms = source
        if not scms:
            raise ValueError("no worlds to describe")
        if not all(hasattr(w, "data") for w in scms):
            raise TypeError("sequence source must contain SCM worlds")
        horizons = {int(np.asarray(w.data["outcome"]).shape[0]) for w in scms}
        if len(horizons) > 1:
            raise ValueError(
                f"worlds must share n_time_steps; got {sorted(horizons)}. Describe each "
                "horizon separately and combine with WorldDescriptors.concat"
            )
        n_total, n_time_steps = len(scms), horizons.pop()
        if include_truth:
            # Truth reads every audit array; a key present in only some worlds would
            # otherwise fail or pass depending on which worlds share a chunk.
            for key in _WORLD_COUNTERFACTUAL.values():
                present = [key in w.data for w in scms]
                if any(present) and not all(present):
                    raise ValueError(
                        f"SCM key {key!r} is present in some worlds and missing in others "
                        f"(first missing: world {present.index(False)}); describe a "
                        "homogeneous sequence"
                    )
        scm_widths = (
            max(int(w.n_treatments) for w in scms),
            max(int(w.n_covariates) for w in scms),
        )

        def extract(chunk_idx: np.ndarray) -> _Chunk:
            return _worlds_chunk(scms, chunk_idx, include_truth, scm_widths)

    else:
        raise TypeError(
            "source must be a corpus mapping, a sequence of SCM or ObservedWorlds, "
            f"got {type(source).__name__}"
        )
    if n_time_steps == 0:
        raise ValueError("source has no time steps to describe")
    # Canonical lag order, so equal requests produce equal definitions.
    lag_axis = tuple(sorted(_validate_lags(lags, n_time_steps)))
    dependence_axis = (
        () if dependence_lags is None else tuple(sorted(_series_lags(dependence_lags, signed=True)))
    )
    if any(s in stats for s in _LAGGED_PAIR_STATISTICS) and not dependence_axis:
        raise ValueError(
            "dependence_lags must explicitly select signed integer lags for lagged pair statistics"
        )
    idx = _validate_world_selector(worlds, n_total)
    n = int(idx.size)

    if world_ids is None:
        ids = idx.astype(np.int64)
    else:
        ids = _id_override(world_ids, n, "world_ids", minimum=0)
    if group_ids is not None:
        groups = _id_override(group_ids, n, "group_ids", minimum=-1)
    elif groups_default is not None:
        groups = groups_default[idx]
    else:
        groups = np.full(n, -1, dtype=np.int64)

    definitions = _definitions(stats, lag_axis, dependence_axis, reds, include_truth)
    if not definitions:
        raise ValueError(
            f"no features to compute: statistics {list(stats)} with lags {list(lag_axis)} "
            f"define no column (a {n_time_steps}-step horizon has no default lags)"
        )
    values = np.full((n, len(definitions)), np.nan)
    status = np.zeros((n, len(definitions)), dtype=np.uint8)
    n_treat_active = np.zeros(n, dtype=np.int64)
    n_cov_active = np.zeros(n, dtype=np.int64)
    for start in range(0, n, int(chunk_size)):
        rows = slice(start, min(start + int(chunk_size), n))
        chunk_idx = idx[rows]
        chunk = extract(chunk_idx)
        values[rows], status[rows] = _chunk_features(
            chunk, definitions, stats, lag_axis, dependence_axis
        )
        n_treat_active[rows] = chunk.treatment_mask.sum(axis=1)
        n_cov_active[rows] = chunk.covariate_mask.sum(axis=1)

    return WorldDescriptors(
        source_ids=np.full(n, source_id, dtype=object),
        world_ids=ids,
        group_ids=groups,
        metadata={
            "n_treatments_active": n_treat_active,
            "n_covariates_active": n_cov_active,
            "n_time_steps": np.full(n, n_time_steps, dtype=np.int64),
        },
        definitions=definitions,
        values=values,
        status=status,
    )
