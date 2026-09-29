"""Between-world analysis of descriptor tables.

Every function here reads :class:`~pymc_generator.descriptors.WorldDescriptors`
tables and nothing else. A table is any collection of worlds under arbitrary
``source_id`` names; nothing here knows where the rows came from. The results
describe: nothing here generates data, alters a prior or picks a best
setting, and no p-value or significance is reported.

* :func:`summarize_descriptors` — availability counts and finite-value
  statistics per stratum and feature.
* :func:`bin_counts` — worlds and distinct generation groups per cell of a
  fixed one- or two-feature grid.
* :class:`DescriptorScale` — a per-feature center and scale fitted on one
  table and applied unchanged to any other.
* :func:`nearest_worlds` — for each query row, the closest reference rows in
  standardized units, under explicit exclusions.
* :func:`compare_descriptors` — where query rows fall in a reference
  distribution, per stratum.

``reference`` and ``query`` are directional roles, not kinds of data: the
reference defines ranks, ranges and scales, and nothing is ever fitted on the
query. A feature is compared only when its
:class:`~pymc_generator.descriptors.FeatureDefinition` is equal in both tables.
The three descriptor states stay apart throughout: ``+inf``/``-inf`` are valid,
ordered values, while ``undefined`` and ``ineligible`` rows are counted
separately and never enter a statistic. A missing generation group
(``group_id == -1``) is reported as unknown, never treated as a group of one.

The only random numbers drawn are the optional, seeded subsamples of
:func:`nearest_worlds`.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, fields
from typing import TYPE_CHECKING, Any

import numpy as np

from .descriptors import (
    INELIGIBLE,
    METADATA_COLUMNS,
    UNDEFINED,
    VALID,
    FeatureDefinition,
    WorldDescriptors,
)
from .diagnostics import _cell, _finite_stats, _render_table, _strict_json_value
from .outcomes import DEFAULT_QUANTILES, _q_key, _validate_quantiles

if TYPE_CHECKING:  # pragma: no cover - typing only
    import pandas as pd

__all__ = [
    "NEIGHBOR_STATUSES",
    "BinCounts",
    "DescriptorComparison",
    "DescriptorScale",
    "DescriptorSummary",
    "NearestWorlds",
    "bin_counts",
    "compare_descriptors",
    "nearest_worlds",
    "summarize_descriptors",
]

#: Per-query-row outcome of :func:`nearest_worlds`, by status code.
NEIGHBOR_STATUSES: tuple[str, ...] = (
    "ok",
    "missing_feature",
    "not_sampled",
    "no_candidates",
    "group_unknown",
)
_OK, _MISSING_FEATURE, _NOT_SAMPLED, _NO_CANDIDATES, _GROUP_UNKNOWN = 0, 1, 2, 3, 4

#: Columns a table can be stratified or matched on.
_STRATUM_COLUMNS: tuple[str, ...] = ("source_id", *METADATA_COLUMNS)

#: Why a row is outside a :func:`bin_counts` grid, per binned feature.
_OUTSIDE_REASONS: tuple[str, ...] = ("below", "above", "undefined", "ineligible")

#: Availability counts of one feature over a set of rows, in report order.
_COUNT_KEYS: tuple[str, ...] = (
    "n_valid",
    "n_finite",
    "n_posinf",
    "n_neginf",
    "n_undefined",
    "n_ineligible",
)

#: Per stratum x feature contrasts of :func:`compare_descriptors`.
_CONTRAST_KEYS: tuple[str, ...] = (
    "median_shift",
    "ks_distance",
    "fraction_below_range",
    "fraction_above_range",
)

#: Upper bound on the eight-byte elements one block of :func:`nearest_worlds`
#: works on: the query-minus-reference differences plus the per-pair arrays.
_BLOCK_ELEMENTS = 1 << 24
#: Per-pair eight-byte working arrays beside the differences: distance, merged
#: distance and position, and the selection indices.
_PAIR_OVERHEAD = 5
#: Below this plain RMS, squares may have underflowed (|d| < ~1e-154): such pairs
#: are recomputed at an exact power-of-two scale.
_RMS_UNDERFLOW = 1e-140


# ---------------------------------------------------------------------------
# Shared validation and grouping
# ---------------------------------------------------------------------------


def _require_table(value: Any, what: str) -> WorldDescriptors:
    if not isinstance(value, WorldDescriptors):
        raise TypeError(f"{what} must be WorldDescriptors, got {type(value).__name__}")
    return value


def _features(descriptors: WorldDescriptors, features: Sequence[str] | None) -> tuple[str, ...]:
    """Selected feature names: None for every feature, else known unique names in order."""
    if features is None:
        names = descriptors.features
    else:
        if isinstance(features, str) or not isinstance(features, Sequence):
            raise TypeError(f"features must be a sequence of names, not {type(features).__name__}")
        names = tuple(features)
        for name in names:
            if not isinstance(name, str):
                raise TypeError(f"feature names must be strings, got {type(name).__name__}")
            descriptors.feature_index(name)
        if len(set(names)) != len(names):
            raise ValueError(f"features repeats an entry: {list(names)}")
    if not names:
        raise ValueError("features is empty; name at least one descriptor column")
    return names


def _columns(value: Any, allowed: tuple[str, ...], what: str) -> tuple[str, ...]:
    """A duplicate-free subset of ``allowed``, in the caller's order."""
    if isinstance(value, str) or not isinstance(value, Sequence):
        raise TypeError(f"{what} must be a sequence of column names, not {type(value).__name__}")
    names = tuple(value)
    unknown = [c for c in names if c not in allowed]
    if unknown:
        raise ValueError(f"unknown {what} columns {unknown}; choose from {list(allowed)}")
    if len(set(names)) != len(names):
        raise ValueError(f"{what} repeats a column: {list(names)}")
    return names


def _positive_int(value: Any, what: str) -> int:
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
        raise TypeError(f"{what} must be a positive integer, got {type(value).__name__}")
    if value < 1:
        raise ValueError(f"{what} must be a positive integer, got {value}")
    return int(value)


def _check_compatible(
    left: Sequence[FeatureDefinition],
    right: Sequence[FeatureDefinition],
    features: Sequence[str],
    labels: tuple[str, str] = ("reference", "query"),
    hint: str = "pass features= naming columns both tables share",
) -> None:
    """Every feature must exist on both sides with an equal definition."""
    sides = ({d.name: d for d in left}, {d.name: d for d in right})
    for label, table in zip(labels, sides):
        absent = [name for name in features if name not in table]
        if absent:
            raise KeyError(f"features {absent[:5]} are missing from the {label}; {hint}")
    differing = [name for name in features if sides[0][name] != sides[1][name]]
    if differing:
        first_left, first_right = sides[0][differing[0]], sides[1][differing[0]]
        fields_differing = [
            f.name
            for f in fields(FeatureDefinition)
            if getattr(first_left, f.name) != getattr(first_right, f.name)
        ]
        raise ValueError(
            f"descriptor definitions differ between the {labels[0]} and the {labels[1]} for "
            f"{differing[:5]} ({differing[0]!r} differs in {fields_differing}); a feature is "
            "compared only when its definitions are equal"
        )


def _key_codes(columns: Sequence[np.ndarray], n_rows: int) -> tuple[np.ndarray, np.ndarray]:
    """Dense lexicographic rank of every row's key over ``columns``, and each key's first row."""
    first = np.zeros(0, dtype=np.int64)
    index = np.zeros(n_rows, dtype=np.int64)
    for column in columns:
        distinct, codes = np.unique(column, return_inverse=True)
        # Re-ranking after every column keeps the codes dense (below n_rows), so the
        # combined code stays below n_rows * n_distinct and keeps the key order.
        _, first, index = np.unique(
            index * distinct.size + codes.reshape(-1), return_index=True, return_inverse=True
        )
    return first, index.reshape(-1).astype(np.int64)


def _stratify(
    columns: Sequence[np.ndarray], n_rows: int
) -> tuple[list[tuple[Any, ...]], np.ndarray]:
    """Sorted distinct row keys over ``columns`` and each row's key position."""
    if not columns:
        return [()], np.zeros(n_rows, dtype=np.int64)
    first, index = _key_codes(columns, n_rows)
    return list(zip(*(np.asarray(column)[first].tolist() for column in columns))), index


def _strata(
    descriptors: WorldDescriptors, by: tuple[str, ...]
) -> tuple[list[tuple[Any, ...]], np.ndarray]:
    """Sorted strata of the (validated) ``by`` columns and each row's stratum."""
    return _stratify([descriptors.grouping(column) for column in by], descriptors.n_worlds)


def _stratum_rows(index: np.ndarray, n_strata: int) -> list[np.ndarray]:
    """Row positions of every stratum, ascending within each."""
    order = np.argsort(index, kind="stable")
    bounds = np.searchsorted(index[order], np.arange(n_strata + 1))
    return [order[bounds[s] : bounds[s + 1]] for s in range(n_strata)]


def _stratum_label(by: tuple[str, ...], key: tuple[Any, ...]) -> str:
    if not by:
        return "all worlds"
    return ", ".join(f"{column}={value}" for column, value in zip(by, key))


def _stratum_position(strata: tuple[tuple[Any, ...], ...], stratum_index: Any) -> int:
    if isinstance(stratum_index, bool) or not isinstance(stratum_index, (int, np.integer)):
        raise TypeError(f"stratum_index must be an integer, got {type(stratum_index).__name__}")
    if not 0 <= stratum_index < len(strata):
        raise IndexError(f"stratum_index {stratum_index} is out of range for {len(strata)} strata")
    return int(stratum_index)


def _feature_position(features: tuple[str, ...], feature: str) -> int:
    try:
        return features.index(feature)
    except ValueError:
        raise KeyError(
            f"feature {feature!r} is not in this report; available: {list(features)}"
        ) from None


def _table_layout(
    by: tuple[str, ...],
    strata: tuple[tuple[Any, ...], ...],
    features: tuple[str, ...],
    feature: str | None,
) -> tuple[str, list[tuple[str, int, str]]]:
    """Subtitle and ``(row label, stratum, feature)`` rows of a per-stratum report table."""
    if feature is None:
        if len(strata) > 1:
            raise ValueError(
                f"{len(strata)} strata by {list(by)}; pass feature= to tabulate one feature "
                "across them"
            )
        if not strata:
            return "no worlds", []
        return _stratum_label(by, strata[0]), [(name, 0, name) for name in features]
    _feature_position(features, feature)
    rows = [(_stratum_label(by, key), s, feature) for s, key in enumerate(strata)]
    return f"{feature} — {len(strata)} strata by {list(by)}", rows


def _group_codes(descriptors: WorldDescriptors) -> np.ndarray:
    """Per-row code of the known generation group ``(source_id, group_id)``; -1 unknown."""
    codes = np.full(descriptors.n_worlds, -1, dtype=np.int64)
    known = descriptors.group_ids >= 0
    if known.any():
        _, codes[known] = _key_codes(
            [descriptors.source_ids[known], descriptors.group_ids[known]], int(known.sum())
        )
    return codes


def _column_stats(
    values: np.ndarray, codes: np.ndarray, levels: tuple[float, ...]
) -> dict[str, Any]:
    """Availability counts and finite-value statistics of one feature over some rows."""
    valid = codes == VALID
    finite = _finite_stats(values, valid, levels)
    n_finite = finite.pop("n")
    return {
        "n_valid": int(valid.sum()),
        "n_finite": n_finite,
        # Values are NaN wherever the status is not valid, so every infinity is valid.
        "n_posinf": int(np.isposinf(values).sum()),
        "n_neginf": int(np.isneginf(values).sum()),
        "n_undefined": int((codes == UNDEFINED).sum()),
        "n_ineligible": int((codes == INELIGIBLE).sum()),
        **finite,
    }


def _stat_keys(levels: tuple[float, ...]) -> tuple[str, ...]:
    return (*_COUNT_KEYS, "mean", "std", "min", "max", *(_q_key(level) for level in levels))


def _frame_value(value: Any) -> Any:
    return np.nan if value is None else value


# ---------------------------------------------------------------------------
# Summaries
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class DescriptorSummary:
    """Availability and finite-value statistics per stratum and feature.

    ``n_groups`` counts the distinct known ``(source_id, group_id)`` pairs of a
    stratum and ``n_unknown_group`` its rows without a known group; the two are
    never merged. Moments and quantiles use the finite valid values only, and
    ``n_posinf``/``n_neginf`` count the valid infinities they leave out.
    """

    by: tuple[str, ...]
    strata: tuple[tuple[Any, ...], ...]
    features: tuple[str, ...]
    quantile_levels: tuple[float, ...]
    n_worlds: np.ndarray
    n_groups: np.ndarray
    n_unknown_group: np.ndarray
    records: tuple[tuple[dict[str, Any], ...], ...]

    def stats(self, stratum_index: int, feature: str) -> dict[str, Any]:
        """Counts and finite-value statistics of ``feature`` in one stratum."""
        s = _stratum_position(self.strata, stratum_index)
        return {
            "n_worlds": int(self.n_worlds[s]),
            "n_groups": int(self.n_groups[s]),
            "n_unknown_group": int(self.n_unknown_group[s]),
            **self.records[s][_feature_position(self.features, feature)],
        }

    def to_frame(self) -> pd.DataFrame:
        """One row per stratum x feature; the stratum columns are named by ``by``."""
        import pandas as pd

        keys = ("n_worlds", "n_groups", "n_unknown_group", *_stat_keys(self.quantile_levels))
        rows = []
        for s, key in enumerate(self.strata):
            for name in self.features:
                stats = self.stats(s, name)
                rows.append([*key, name, *(_frame_value(stats[k]) for k in keys)])
        return pd.DataFrame(rows, columns=[*self.by, "feature", *keys])

    def table(self, feature: str | None = None) -> str:
        """Every feature of the single stratum, or one ``feature`` across strata."""
        levels = [_q_key(level) for level in self.quantile_levels]
        columns = ["valid", "+inf", "-inf", "undefined", "ineligible", "groups", "mean", *levels]
        counts = ("n_valid", "n_posinf", "n_neginf", "n_undefined", "n_ineligible", "n_groups")
        subtitle, layout = _table_layout(self.by, self.strata, self.features, feature)
        rows = []
        for label, s, name in layout:
            stats = self.stats(s, name)
            cells = [str(stats[c]) for c in counts]
            cells += [_cell(stats[c]) for c in ("mean", *levels)]
            rows.append((label, str(stats["n_worlds"]), cells))
        return _render_table(
            f"descriptor summary — {subtitle}", columns, rows, count_header="worlds"
        )

    def summary(self) -> dict[str, Any]:
        """Strictly JSON-safe report (``json.dumps(..., allow_nan=False)``)."""
        payload: dict[str, Any] = {
            "by": list(self.by),
            "features": list(self.features),
            "quantiles": list(self.quantile_levels),
            "strata": [
                {
                    "key": dict(zip(self.by, key)),
                    "n_worlds": self.n_worlds[s],
                    "n_groups": self.n_groups[s],
                    "n_unknown_group": self.n_unknown_group[s],
                    "features": dict(zip(self.features, self.records[s])),
                }
                for s, key in enumerate(self.strata)
            ],
        }
        sanitized: dict[str, Any] = _strict_json_value(payload)
        return sanitized


def summarize_descriptors(
    descriptors: WorldDescriptors,
    *,
    features: Sequence[str] | None = None,
    by: Sequence[str] = (),
    quantiles: Sequence[float] = DEFAULT_QUANTILES,
) -> DescriptorSummary:
    """Availability counts and finite-value statistics per stratum and feature.

    Parameters
    ----------
    descriptors
        Any descriptor table.
    features
        Feature names to summarize; None for every feature.
    by
        Stratum columns, a subset of ``source_id`` and
        :data:`~pymc_generator.descriptors.METADATA_COLUMNS`. Empty for one
        stratum holding every row. Strata are sorted by key.
    quantiles
        Quantile levels of the finite valid values.

    Returns
    -------
    DescriptorSummary
    """
    _require_table(descriptors, "descriptors")
    names = _features(descriptors, features)
    columns = _columns(by, _STRATUM_COLUMNS, "by")
    levels = _validate_quantiles(quantiles)
    strata, index = _strata(descriptors, columns)
    positions = np.asarray([descriptors.feature_index(name) for name in names], dtype=np.int64)
    groups = _group_codes(descriptors)
    n_worlds, n_groups, n_unknown, records = [], [], [], []
    for rows in _stratum_rows(index, len(strata)):
        values = descriptors.values[np.ix_(rows, positions)]
        codes = descriptors.status[np.ix_(rows, positions)]
        stratum_groups = groups[rows]
        n_worlds.append(rows.size)
        n_groups.append(np.unique(stratum_groups[stratum_groups >= 0]).size)
        n_unknown.append(int((stratum_groups < 0).sum()))
        records.append(
            tuple(_column_stats(values[:, j], codes[:, j], levels) for j in range(len(names)))
        )
    return DescriptorSummary(
        by=columns,
        strata=tuple(strata),
        features=names,
        quantile_levels=levels,
        n_worlds=np.asarray(n_worlds, dtype=np.int64),
        n_groups=np.asarray(n_groups, dtype=np.int64),
        n_unknown_group=np.asarray(n_unknown, dtype=np.int64),
        records=tuple(records),
    )


# ---------------------------------------------------------------------------
# Fixed-edge bins
# ---------------------------------------------------------------------------


def _edges(name: str, value: Any) -> np.ndarray:
    edges = np.asarray(value)
    if edges.dtype.kind not in "fiu":
        raise TypeError(f"bins[{name!r}] must be real-valued edges")
    if edges.ndim != 1 or edges.size < 2:
        raise ValueError(f"bins[{name!r}] must be a one-dimensional sequence of at least two edges")
    edges = edges.astype(np.float64)
    if not np.isfinite(edges).all():
        raise ValueError(
            f"bins[{name!r}] edges must be finite; values beyond them are counted as "
            "'below'/'above'"
        )
    if not (np.diff(edges) > 0).all():
        raise ValueError(f"bins[{name!r}] edges must be strictly increasing")
    return edges


def _bin_rows(
    descriptors: WorldDescriptors, features: tuple[str, ...], edges: tuple[np.ndarray, ...]
) -> tuple[np.ndarray, np.ndarray]:
    """Per row and feature: the bin (-1 outside) and the ``_OUTSIDE_REASONS`` code (-1 inside)."""
    shape = (descriptors.n_worlds, len(features))
    bins = np.full(shape, -1, dtype=np.int64)
    reasons = np.full(shape, -1, dtype=np.int8)
    for j, (name, e) in enumerate(zip(features, edges)):
        position = descriptors.feature_index(name)
        values, codes = descriptors.values[:, position], descriptors.status[:, position]
        valid = codes == VALID
        below = valid & (values < e[0])
        above = valid & (values > e[-1])
        inside = valid & ~below & ~above
        # [e_i, e_{i+1}): an interior edge opens the bin to its right; the last
        # edge closes the final bin.
        bins[inside, j] = np.minimum(
            np.searchsorted(e, values[inside], side="right") - 1, e.size - 2
        )
        reasons[below, j] = 0
        reasons[above, j] = 1
        reasons[codes == UNDEFINED, j] = 2
        reasons[codes == INELIGIBLE, j] = 3
    return bins, reasons


def _interval_labels(edges: np.ndarray) -> list[str]:
    last = edges.size - 2
    return [
        f"[{lo:g}, {hi:g}{']' if i == last else ')'}"
        for i, (lo, hi) in enumerate(zip(edges[:-1].tolist(), edges[1:].tolist()))
    ]


@dataclass(frozen=True)
class BinCounts:
    """Worlds and distinct generation groups per cell of a fixed grid, per stratum.

    Cells are ``[e_i, e_{i+1})`` except the last of each feature, which is
    closed ``[e_{n-1}, e_n]``. A row is in the grid only when every binned
    feature is valid and within its edges. ``outside[feature][reason]`` counts,
    per stratum, the rows that feature keeps out (``below``, ``above``,
    ``undefined``, ``ineligible``); ``n_outside`` counts each such row once.
    ``group_counts`` counts distinct known ``(source_id, group_id)`` pairs.
    """

    features: tuple[str, ...]
    definitions: tuple[FeatureDefinition, ...]
    edges: tuple[np.ndarray, ...]
    by: tuple[str, ...]
    strata: tuple[tuple[Any, ...], ...]
    counts: np.ndarray
    group_counts: np.ndarray
    n_stratum_worlds: np.ndarray
    n_outside: np.ndarray
    outside: dict[str, dict[str, np.ndarray]]

    def locate(self, descriptors: WorldDescriptors) -> np.ndarray:
        """``(n_worlds, n_features)`` bin of every row under the same edges; -1 outside."""
        _require_table(descriptors, "descriptors")
        _check_compatible(
            self.definitions,
            descriptors.definitions,
            self.features,
            labels=("binned table", "located table"),
            hint="locate needs every binned feature",
        )
        bins, _ = _bin_rows(descriptors, self.features, self.edges)
        return bins

    def to_frame(self) -> pd.DataFrame:
        """One row per stratum x cell: bounds, ``n_worlds``, ``n_groups``, ``fraction``."""
        import pandas as pd

        grid = self.counts.shape[1:]
        n_strata, n_cells = len(self.strata), int(np.prod(grid))
        data: dict[str, Any] = {}
        for j, column in enumerate(self.by):
            data[column] = np.repeat(np.asarray([key[j] for key in self.strata]), n_cells)
        for name, edges, cell_bins in zip(
            self.features, self.edges, np.unravel_index(np.arange(n_cells), grid)
        ):
            data[f"{name}_lower"] = np.tile(edges[:-1][cell_bins], n_strata)
            data[f"{name}_upper"] = np.tile(edges[1:][cell_bins], n_strata)
        n_worlds = self.counts.reshape(-1)
        totals = np.repeat(self.n_stratum_worlds, n_cells)
        data["n_worlds"] = n_worlds
        data["n_groups"] = self.group_counts.reshape(-1)
        data["fraction"] = np.divide(
            n_worlds, totals, out=np.full(totals.shape, np.nan), where=totals > 0
        )
        return pd.DataFrame(data)

    def table(self) -> str:
        """Cell counts of every stratum, then why its other rows are outside the grid."""
        labels = [_interval_labels(edges) for edges in self.edges]
        grid_name = " x ".join(self.features)
        if not self.strata:
            return _render_table(f"bin counts — {grid_name}", ["groups"], [])
        lines: list[str] = []
        for s, key in enumerate(self.strata):
            total = int(self.n_stratum_worlds[s])
            title = (
                f"bin counts — {grid_name} — {_stratum_label(self.by, key)} — {total} worlds, "
                f"{int(self.n_outside[s])} outside the grid"
            )
            if len(self.features) == 1:
                rows = [
                    (
                        label,
                        str(int(count)),
                        [str(int(groups)), _cell(count / total if total else None)],
                    )
                    for label, count, groups in zip(labels[0], self.counts[s], self.group_counts[s])
                ]
                body = _render_table(title, ["groups", "fraction"], rows, count_header="worlds")
            else:
                rows = [
                    (label, str(int(row.sum())), [str(int(count)) for count in row])
                    for label, row in zip(labels[0], self.counts[s])
                ]
                body = _render_table(
                    f"{title}; rows {self.features[0]}, columns {self.features[1]}",
                    labels[1],
                    rows,
                    count_header="worlds",
                )
            outside = "; ".join(
                f"{name}: "
                + ", ".join(
                    f"{reason} {int(self.outside[name][reason][s])}" for reason in _OUTSIDE_REASONS
                )
                for name in self.features
            )
            if lines:
                lines.append("")
            lines += [body, f"outside — {outside}"]
        return "\n".join(lines)

    def summary(self) -> dict[str, Any]:
        """Strictly JSON-safe report (``json.dumps(..., allow_nan=False)``)."""
        payload: dict[str, Any] = {
            "features": list(self.features),
            "edges": dict(zip(self.features, self.edges)),
            "by": list(self.by),
            "strata": [
                {
                    "key": dict(zip(self.by, key)),
                    "n_worlds": self.n_stratum_worlds[s],
                    "n_outside": self.n_outside[s],
                    "counts": self.counts[s],
                    "group_counts": self.group_counts[s],
                    "outside": {
                        name: {reason: counts[s] for reason, counts in reasons.items()}
                        for name, reasons in self.outside.items()
                    },
                }
                for s, key in enumerate(self.strata)
            ],
        }
        sanitized: dict[str, Any] = _strict_json_value(payload)
        return sanitized


def bin_counts(
    descriptors: WorldDescriptors,
    bins: Mapping[str, Sequence[float]],
    *,
    by: Sequence[str] = (),
) -> BinCounts:
    """Count worlds and distinct generation groups per cell of a fixed grid.

    Parameters
    ----------
    descriptors
        Any descriptor table.
    bins
        One or two feature names, each mapped to at least two finite, strictly
        increasing edges. Cells are ``[e_i, e_{i+1})``; the last is closed.
    by
        Stratum columns, a subset of ``source_id`` and
        :data:`~pymc_generator.descriptors.METADATA_COLUMNS`.

    Returns
    -------
    BinCounts
        Use :meth:`BinCounts.locate` to place any compatible table on the same
        grid.
    """
    _require_table(descriptors, "descriptors")
    if not isinstance(bins, Mapping):
        raise TypeError(f"bins must map feature names to edges, got {type(bins).__name__}")
    if not 1 <= len(bins) <= 2:
        raise ValueError(f"bins must name one or two features, got {len(bins)}")
    names = _features(descriptors, tuple(bins))
    edges = tuple(_edges(name, bins[name]) for name in names)
    columns = _columns(by, _STRATUM_COLUMNS, "by")
    strata, index = _strata(descriptors, columns)
    n_strata = len(strata)
    grid = tuple(e.size - 1 for e in edges)
    n_cells = int(np.prod(grid))
    n_slots = n_strata * n_cells

    located, reasons = _bin_rows(descriptors, names, edges)
    in_grid = (located >= 0).all(axis=1)
    slot = index[in_grid] * n_cells + np.ravel_multi_index(tuple(located[in_grid].T), grid)
    counts = np.bincount(slot, minlength=n_slots).astype(np.int64)
    group_counts = np.zeros(n_slots, dtype=np.int64)
    groups = _group_codes(descriptors)[in_grid]
    known = groups >= 0
    if known.any():
        # One row per distinct (slot, group) pair, then count the pairs per slot.
        first, _ = _key_codes([slot[known], groups[known]], int(known.sum()))
        group_counts = np.bincount(slot[known][first], minlength=n_slots).astype(np.int64)
    outside = {
        name: {
            reason: np.bincount(index[reasons[:, j] == code], minlength=n_strata).astype(np.int64)
            for code, reason in enumerate(_OUTSIDE_REASONS)
        }
        for j, name in enumerate(names)
    }
    return BinCounts(
        features=names,
        definitions=tuple(descriptors.definition(name) for name in names),
        edges=edges,
        by=columns,
        strata=tuple(strata),
        counts=counts.reshape(n_strata, *grid),
        group_counts=group_counts.reshape(n_strata, *grid),
        n_stratum_worlds=np.bincount(index, minlength=n_strata).astype(np.int64),
        n_outside=np.bincount(index[~in_grid], minlength=n_strata).astype(np.int64),
        outside=outside,
    )


# ---------------------------------------------------------------------------
# Standardization
# ---------------------------------------------------------------------------


def _real_vector(value: Any, what: str, n: int) -> np.ndarray:
    arr = np.asarray(value)
    if arr.dtype.kind not in "fiu":
        raise TypeError(f"{what} must be real numbers")
    if arr.shape != (n,):
        raise ValueError(f"{what} must have shape ({n},), got {arr.shape}")
    return arr.astype(np.float64)


@dataclass(frozen=True)
class DescriptorScale:
    """Per-feature center and scale: ``z = (value - center) / scale``.

    Fit it on one table with :meth:`fit` and apply it unchanged to any
    compatible table with :meth:`transform`; or build it explicitly. Every
    scale must be positive and finite, every center finite, and
    ``definitions`` must describe ``features`` one to one, in order.
    """

    features: tuple[str, ...]
    center: np.ndarray
    scale: np.ndarray
    definitions: tuple[FeatureDefinition, ...]
    method: str

    def __post_init__(self) -> None:
        if isinstance(self.features, str) or not isinstance(self.features, Sequence):
            raise TypeError("features must be a sequence of names")
        features = tuple(self.features)
        if not features:
            raise ValueError("features is empty")
        if not all(isinstance(name, str) and name for name in features):
            raise TypeError("feature names must be non-empty strings")
        if len(set(features)) != len(features):
            raise ValueError(f"features repeats an entry: {list(features)}")
        center = _real_vector(self.center, "center", len(features))
        scale = _real_vector(self.scale, "scale", len(features))
        if not np.isfinite(center).all():
            raise ValueError("center must be finite")
        if not (np.isfinite(scale) & (scale > 0)).all():
            raise ValueError("scale must be positive and finite")
        definitions = tuple(self.definitions)
        if not all(isinstance(d, FeatureDefinition) for d in definitions):
            raise TypeError("definitions must be FeatureDefinition instances")
        if tuple(d.name for d in definitions) != features:
            raise ValueError("definitions must describe features one to one, in order")
        if not isinstance(self.method, str) or not self.method:
            raise ValueError("method must be a non-empty string")
        object.__setattr__(self, "features", features)
        object.__setattr__(self, "center", center)
        object.__setattr__(self, "scale", scale)
        object.__setattr__(self, "definitions", definitions)

    @classmethod
    def fit(
        cls, descriptors: WorldDescriptors, features: Sequence[str], *, method: str = "robust"
    ) -> DescriptorScale:
        """Fit on the rows of ``descriptors`` where every feature is finite.

        ``robust`` uses the median and the interquartile range, falling back to
        the population std for a feature whose IQR is zero; ``standard`` uses
        the mean and the population std. A feature that is constant over those
        rows has no scale and raises: drop it, or build the scale explicitly.
        """
        _require_table(descriptors, "descriptors")
        if features is None:
            raise TypeError("features is required: name the columns to standardize")
        names = _features(descriptors, features)
        if method not in ("robust", "standard"):
            raise ValueError(f"method must be 'robust' or 'standard', got {method!r}")
        values = descriptors.values[:, [descriptors.feature_index(name) for name in names]]
        complete = np.isfinite(values).all(axis=1)
        if not complete.any():
            raise ValueError(
                f"no row has every feature of {list(names)} finite; a scale needs at least "
                "one complete row"
            )
        rows = values[complete]
        lowest, highest = rows.min(axis=0), rows.max(axis=0)
        constant = np.flatnonzero(lowest == highest)
        if constant.size:
            found = {names[j]: float(lowest[j]) for j in constant}
            raise ValueError(
                f"features {list(found)} are constant over the {rows.shape[0]} complete rows "
                f"(values {found}) and cannot be standardized; drop them from features or "
                "build DescriptorScale(...) with an explicit scale"
            )
        with np.errstate(over="ignore", invalid="ignore"):
            if method == "robust":
                q25, center, q75 = np.quantile(rows, [0.25, 0.5, 0.75], axis=0)
                scale = q75 - q25
                flat = scale == 0
                scale[flat] = rows[:, flat].std(axis=0)
            else:
                center, scale = rows.mean(axis=0), rows.std(axis=0)
        unusable = ~(np.isfinite(center) & np.isfinite(scale) & (scale > 0))
        if unusable.any():
            bad = [names[j] for j in np.flatnonzero(unusable)]
            raise ValueError(
                f"features {bad} have no finite positive {method} scale over the complete "
                "rows (magnitudes near the float64 limits); rescale them or build "
                "DescriptorScale(...) explicitly"
            )
        return cls(
            features=names,
            center=center,
            scale=scale,
            definitions=tuple(descriptors.definition(name) for name in names),
            method=method,
        )

    def transform(self, descriptors: WorldDescriptors) -> tuple[np.ndarray, np.ndarray]:
        """``(z, complete)``: standardized ``(n, F)`` values and the rows with every feature finite.

        ``z`` is NaN on every row that is not complete.
        """
        _require_table(descriptors, "descriptors")
        _check_compatible(
            self.definitions,
            descriptors.definitions,
            self.features,
            labels=("scale", "descriptors"),
            hint="a scale applies only to tables carrying every feature it was fitted on",
        )
        values = descriptors.values[:, [descriptors.feature_index(name) for name in self.features]]
        with np.errstate(over="ignore", invalid="ignore"):
            z = (values - self.center) / self.scale
        # A finite value can still standardize to +/-inf; such a row has no distance.
        complete = np.isfinite(values).all(axis=1) & np.isfinite(z).all(axis=1)
        z[~complete] = np.nan
        return z, complete


# ---------------------------------------------------------------------------
# Nearest reference worlds
# ---------------------------------------------------------------------------


def _k_smallest(
    distances: np.ndarray, positions: np.ndarray, k: int
) -> tuple[np.ndarray, np.ndarray]:
    """Per row, the ``k`` smallest ``(distance, position)`` pairs in lexicographic order."""
    if distances.shape[1] > k:
        pick = np.argpartition(distances, k - 1, axis=1)[:, :k]
        kth = np.take_along_axis(distances, pick, axis=1).max(axis=1, keepdims=True)
        # argpartition splits a tie at the k-th distance arbitrarily; settle those
        # rows exactly, the lower reference position first.
        tied = (distances <= kth).sum(axis=1) > k
        if tied.any():
            pick[tied] = np.lexsort((positions[tied], distances[tied]), axis=1)[:, :k]
        distances = np.take_along_axis(distances, pick, axis=1)
        positions = np.take_along_axis(positions, pick, axis=1)
    order = np.lexsort((positions, distances), axis=1)
    return (
        np.take_along_axis(distances, order, axis=1),
        np.take_along_axis(positions, order, axis=1),
    )


def _nearest_search(
    z_query: np.ndarray,
    query_keys: tuple[np.ndarray, np.ndarray, np.ndarray],
    z_candidates: np.ndarray,
    candidate_keys: tuple[np.ndarray, np.ndarray, np.ndarray],
    candidate_positions: np.ndarray,
    match_pairs: list[tuple[np.ndarray, np.ndarray]],
    k: int,
    exclude_same_group: bool,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Exact blocked search: k nearest positions and distances, admissible counts, unknown groups.

    ``*_keys`` are ``(source code, world id, group id)`` per row. Differences
    are formed directly, never as more than ``_BLOCK_ELEMENTS`` floats at once.
    """
    n_query, n_features = z_query.shape
    n_candidates = z_candidates.shape[0]
    positions = np.full((n_query, k), -1, dtype=np.int64)
    # NaN marks "no admissible candidate": it sorts after every real distance.
    distances = np.full((n_query, k), np.nan)
    n_admissible = np.zeros(n_query, dtype=np.int64)
    group_unknown = np.zeros(n_query, dtype=bool)
    if n_query == 0 or n_candidates == 0:
        return positions, distances, n_admissible, group_unknown
    # A (query, candidate) pair costs its F differences plus the per-pair distance,
    # merge and selection arrays, so every block's working set is ~_BLOCK_ELEMENTS
    # eight-byte elements whatever the number of features.
    per_pair = n_features + _PAIR_OVERHEAD
    per_row = n_candidates * per_pair
    if per_row <= _BLOCK_ELEMENTS:
        query_step, candidate_step = _BLOCK_ELEMENTS // per_row, n_candidates
    else:
        query_step, candidate_step = 1, max(1, _BLOCK_ELEMENTS // per_pair)
    q_source, q_world, q_group = query_keys
    c_source, c_world, c_group = candidate_keys
    half_query, half_candidates = 0.5 * z_query, 0.5 * z_candidates  # exact; never overflow
    for q0 in range(0, n_query, query_step):
        q = slice(q0, min(q0 + query_step, n_query))
        qs, qw, qg = q_source[q, None], q_world[q, None], q_group[q, None]
        for c0 in range(0, n_candidates, candidate_step):
            c = slice(c0, min(c0 + candidate_step, n_candidates))
            same_source = qs == c_source[None, c]
            admissible = ~(same_source & (qw == c_world[None, c]))
            for query_column, candidate_column in match_pairs:
                admissible &= query_column[q, None] == candidate_column[None, c]
            if exclude_same_group:
                # A same-source pair with an unknown group on EITHER side cannot be
                # told apart from a same-group pair: report it, never guess.
                undecidable = same_source & ((qg < 0) | (c_group[None, c] < 0))
                group_unknown[q] |= (admissible & undecidable).any(axis=1)
                admissible &= ~(same_source & (qg >= 0) & (qg == c_group[None, c]))
            n_admissible[q] += admissible.sum(axis=1)
            with np.errstate(over="ignore", invalid="ignore", under="ignore"):
                # The documented formula, computed plainly: sqrt(mean_f(d ** 2)).
                diff = z_query[q, None, :] - z_candidates[None, c, :]
                np.square(diff, out=diff)
                block = np.sqrt(diff.mean(axis=2))
                # Squares overflow near |d| ~ 1e154 and underflow below ~1e-154. Only
                # those pairs are recomputed, at an exact power-of-two scale of the
                # half-differences, which is the same formula bit for bit wherever it
                # is representable; a true distance beyond float64 becomes +inf, a
                # real value that sorts before NaN (inadmissible).
                redo = ~np.isfinite(block) | (block < _RMS_UNDERFLOW)
                if redo.any():
                    rq, rc = np.nonzero(redo)
                    d = half_query[q][rq] - half_candidates[c][rc]
                    _, exponent = np.frexp(np.abs(d).max(axis=1))
                    scaled = np.ldexp(d, -exponent[:, None])
                    block[rq, rc] = np.ldexp(np.sqrt((scaled * scaled).mean(axis=1)), exponent + 1)
            block[~admissible] = np.nan
            merged_positions = np.broadcast_to(candidate_positions[c], block.shape)
            distances[q], positions[q] = _k_smallest(
                np.concatenate([distances[q], block], axis=1),
                np.concatenate([positions[q], merged_positions], axis=1),
                k,
            )
    return positions, distances, n_admissible, group_unknown


@dataclass(frozen=True)
class NearestWorlds:
    """The nearest admissible reference rows of every query row.

    ``status`` codes index :data:`NEIGHBOR_STATUSES`. Only ``ok`` rows carry
    neighbours: ``neighbor_positions`` (reference row positions, -1 none),
    ``distances`` (RMS standardized difference, NaN none) and ``differences``
    (signed standardized query-minus-nearest per feature, NaN none).
    ``reference_positions`` are the reference rows searched (every complete
    row unless ``max_reference`` subsampled them) out of
    ``n_reference_candidates`` complete rows.
    """

    features: tuple[str, ...]
    scale: DescriptorScale
    k: int
    match: tuple[str, ...]
    exclude_same_group: bool
    seed: int
    source_ids: np.ndarray
    world_ids: np.ndarray
    group_ids: np.ndarray
    status: np.ndarray
    neighbor_positions: np.ndarray
    distances: np.ndarray
    differences: np.ndarray
    reference_positions: np.ndarray
    n_reference_candidates: int
    query_sampled: bool
    reference_source_ids: np.ndarray
    reference_world_ids: np.ndarray
    reference_group_ids: np.ndarray

    @property
    def n_query(self) -> int:
        return int(self.status.size)

    def neighbor_ids(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """``(n_query, k)`` reference source ids (None), world ids and group ids (-1) where none."""
        found = self.neighbor_positions >= 0
        rows = self.neighbor_positions[found]
        shape = self.neighbor_positions.shape
        sources = np.full(shape, None, dtype=object)
        worlds = np.full(shape, -1, dtype=np.int64)
        groups = np.full(shape, -1, dtype=np.int64)
        sources[found] = self.reference_source_ids[rows]
        worlds[found] = self.reference_world_ids[rows]
        groups[found] = self.reference_group_ids[rows]
        return sources, worlds, groups

    def to_frame(self) -> pd.DataFrame:
        """One row per query row: identity, status, distances, neighbour ids, feature deltas."""
        import pandas as pd

        sources, worlds, _ = self.neighbor_ids()
        data: dict[str, Any] = {
            "source_id": self.source_ids,
            "world_id": self.world_ids,
            "group_id": self.group_ids,
            "status": np.asarray(NEIGHBOR_STATUSES, dtype=object)[self.status],
        }
        for i in range(self.k):
            data[f"distance_{i + 1}"] = self.distances[:, i]
        for i in range(self.k):
            data[f"neighbor_source_id_{i + 1}"] = sources[:, i]
            data[f"neighbor_world_id_{i + 1}"] = worlds[:, i]
        for j, name in enumerate(self.features):
            data[f"delta:{name}"] = self.differences[:, j]
        return pd.DataFrame(data)

    def _distance_stats(self) -> dict[str, float | None]:
        return _finite_stats(self.distances[:, 0], self.status == _OK, DEFAULT_QUANTILES)

    def table(self) -> str:
        """Query rows per status, then the nearest distance over ``ok`` rows."""
        title = (
            f"nearest worlds — {self.n_query} query rows, k={self.k}, {len(self.features)} "
            f"features, {self.scale.method} scale; {self.reference_positions.size} of "
            f"{self.n_reference_candidates} complete reference rows searched"
        )
        counts = np.bincount(self.status, minlength=len(NEIGHBOR_STATUSES))
        status_rows = [
            (label, str(int(count)), [_cell(count / self.n_query if self.n_query else None)])
            for label, count in zip(NEIGHBOR_STATUSES, counts)
        ]
        stats = self._distance_stats()
        columns = ["mean", *(_q_key(level) for level in DEFAULT_QUANTILES), "max"]
        distance_row = [("distance_1", str(stats["n"]), [_cell(stats[c]) for c in columns])]
        return "\n".join(
            [
                _render_table(title, ["fraction"], status_rows, count_header="rows"),
                "",
                _render_table(
                    "nearest distance over ok rows (RMS standardized difference)",
                    columns,
                    distance_row,
                    count_header="ok",
                ),
            ]
        )

    def summary(self) -> dict[str, Any]:
        """Strictly JSON-safe report (``json.dumps(..., allow_nan=False)``)."""
        counts = np.bincount(self.status, minlength=len(NEIGHBOR_STATUSES))
        payload: dict[str, Any] = {
            "features": list(self.features),
            "k": self.k,
            "match": list(self.match),
            "exclude_same_group": self.exclude_same_group,
            "seed": self.seed,
            "scale": {
                "method": self.scale.method,
                "center": dict(zip(self.features, self.scale.center)),
                "scale": dict(zip(self.features, self.scale.scale)),
            },
            "n_query": self.n_query,
            "query_sampled": self.query_sampled,
            "n_reference": int(self.reference_world_ids.size),
            "n_reference_candidates": self.n_reference_candidates,
            "n_reference_searched": int(self.reference_positions.size),
            "status_counts": dict(zip(NEIGHBOR_STATUSES, counts)),
            "distance_1": self._distance_stats(),
        }
        sanitized: dict[str, Any] = _strict_json_value(payload)
        return sanitized


def nearest_worlds(
    reference: WorldDescriptors,
    query: WorldDescriptors | None = None,
    *,
    features: Sequence[str],
    scale: DescriptorScale | None = None,
    k: int = 1,
    match: Sequence[str] = (),
    exclude_same_group: bool = False,
    max_reference: int | None = None,
    max_query: int | None = None,
    seed: int = 0,
) -> NearestWorlds:
    """Find the ``k`` nearest admissible reference rows of every query row.

    Distance is the root mean square of the standardized differences,
    ``sqrt(mean_f((z_query - z_reference) ** 2))``, computed exactly in
    bounded blocks. Equal distances go to the lower reference position.

    Parameters
    ----------
    reference
        Rows searched for neighbours; also the only rows a default scale is
        fitted on.
    query
        Rows whose neighbours are wanted; None to use ``reference`` itself.
    features
        Distance features, compatible between both tables.
    scale
        A :class:`DescriptorScale` over exactly ``features``. None fits
        ``DescriptorScale.fit(reference, features)`` on the full reference,
        before any subsampling; the query never influences it.
    k
        Neighbours per query row.
    match
        Columns (``source_id`` or metadata) whose value a neighbour must share
        with its query row.
    exclude_same_group
        Refuse neighbours from the query row's own generation group (same
        ``source_id`` and the same known ``group_id``). A query row with any
        otherwise admissible same-source candidate whose group — or its own —
        is unknown is marked ``group_unknown`` instead of guessed.
    max_reference, max_query
        Optional caps: a uniform sample without replacement of the complete
        reference rows, then of the complete query rows (``not_sampled``
        otherwise), both drawn from ``numpy.random.default_rng(seed)``.
    seed
        Seed of the subsamples.

    Returns
    -------
    NearestWorlds
        A row identical to the query row (same ``source_id`` and
        ``world_id``) is never its neighbour. Query rows without every feature
        finite are ``missing_feature``; with fewer than ``k`` admissible
        candidates, ``no_candidates``.
    """
    _require_table(reference, "reference")
    query = reference if query is None else _require_table(query, "query")
    if features is None:
        raise TypeError("features is required: name the descriptor columns that define distance")
    names = _features(reference, features)
    if query is not reference:
        _check_compatible(reference.definitions, query.definitions, names)
    k = _positive_int(k, "k")
    if k > reference.n_worlds:
        raise ValueError(
            f"k={k} exceeds the {reference.n_worlds} reference rows; no query row could "
            "have that many neighbours"
        )
    match_columns = _columns(match, _STRATUM_COLUMNS, "match")
    if not isinstance(exclude_same_group, (bool, np.bool_)):
        raise TypeError("exclude_same_group must be a bool")
    if max_reference is not None:
        max_reference = _positive_int(max_reference, "max_reference")
    if max_query is not None:
        max_query = _positive_int(max_query, "max_query")
    if isinstance(seed, bool) or not isinstance(seed, (int, np.integer)):
        raise TypeError(f"seed must be an integer, got {type(seed).__name__}")
    if seed < 0:
        raise ValueError(f"seed must be a non-negative integer, got {seed}")
    if scale is None:
        scale = DescriptorScale.fit(reference, names)
    else:
        if not isinstance(scale, DescriptorScale):
            raise TypeError(f"scale must be a DescriptorScale, got {type(scale).__name__}")
        if scale.features != names:
            raise ValueError(
                f"scale covers {list(scale.features)}, not features {list(names)}; fit or "
                "build a scale over exactly these features, in this order"
            )
        _check_compatible(
            scale.definitions,
            reference.definitions,
            names,
            labels=("scale", "reference"),
            hint="the scale must cover exactly the distance features",
        )

    z_reference, reference_complete = scale.transform(reference)
    z_query, query_complete = (
        (z_reference, reference_complete) if query is reference else scale.transform(query)
    )
    rng = np.random.default_rng(int(seed))
    candidates = np.flatnonzero(reference_complete)
    n_candidates = int(candidates.size)
    if max_reference is not None and max_reference < n_candidates:
        candidates = np.sort(rng.choice(candidates, size=max_reference, replace=False))
    status = np.full(query.n_worlds, _MISSING_FEATURE, dtype=np.uint8)
    assessed = np.flatnonzero(query_complete)
    query_sampled = max_query is not None and max_query < assessed.size
    if query_sampled:
        status[assessed] = _NOT_SAMPLED
        assessed = np.sort(rng.choice(assessed, size=max_query, replace=False))

    n_reference = reference.n_worlds
    _, source_codes = np.unique(
        np.concatenate([reference.source_ids, query.source_ids]), return_inverse=True
    )
    reference_source = source_codes.reshape(-1)[:n_reference]
    query_source = source_codes.reshape(-1)[n_reference:]
    match_pairs = [
        (query_source[assessed], reference_source[candidates])
        if column == "source_id"
        else (query.metadata[column][assessed], reference.metadata[column][candidates])
        for column in match_columns
    ]
    found_positions, found_distances, n_admissible, group_unknown = _nearest_search(
        z_query[assessed],
        (query_source[assessed], query.world_ids[assessed], query.group_ids[assessed]),
        z_reference[candidates],
        (
            reference_source[candidates],
            reference.world_ids[candidates],
            reference.group_ids[candidates],
        ),
        candidates,
        match_pairs,
        k,
        bool(exclude_same_group),
    )
    # ok needs k real neighbours: no empty slot, never a pick that was inadmissible.
    filled = (found_positions >= 0).all(axis=1) & ~np.isnan(found_distances).any(axis=1)
    ok = ~group_unknown & (n_admissible >= k) & filled
    status[assessed] = np.where(group_unknown, _GROUP_UNKNOWN, np.where(ok, _OK, _NO_CANDIDATES))
    neighbor_positions = np.full((query.n_worlds, k), -1, dtype=np.int64)
    distances = np.full((query.n_worlds, k), np.nan)
    differences = np.full((query.n_worlds, len(names)), np.nan)
    rows = assessed[ok]
    neighbor_positions[rows] = found_positions[ok]
    distances[rows] = found_distances[ok]
    # Near the float64 limit a difference is +/-inf: the correct extended-real value.
    with np.errstate(over="ignore"):
        differences[rows] = z_query[rows] - z_reference[neighbor_positions[rows, 0]]
    return NearestWorlds(
        features=names,
        scale=scale,
        k=k,
        match=match_columns,
        exclude_same_group=bool(exclude_same_group),
        seed=int(seed),
        source_ids=query.source_ids,
        world_ids=query.world_ids,
        group_ids=query.group_ids,
        status=status,
        neighbor_positions=neighbor_positions,
        distances=distances,
        differences=differences,
        reference_positions=candidates,
        n_reference_candidates=n_candidates,
        query_sampled=bool(query_sampled),
        reference_source_ids=reference.source_ids,
        reference_world_ids=reference.world_ids,
        reference_group_ids=reference.group_ids,
    )


# ---------------------------------------------------------------------------
# Reference-query comparison
# ---------------------------------------------------------------------------


def _shared_rows(reference: WorldDescriptors, query: WorldDescriptors) -> np.ndarray:
    """For each query row, the reference row with the same identity, or -1."""
    n_reference = reference.n_worlds
    if n_reference == 0 or query.n_worlds == 0:
        return np.full(query.n_worlds, -1, dtype=np.int64)
    _, identity = _key_codes(
        [
            np.concatenate([reference.source_ids, query.source_ids]),
            np.concatenate([reference.world_ids, query.world_ids]),
        ],
        n_reference + query.n_worlds,
    )
    owner = np.full(int(identity.max()) + 1, -1, dtype=np.int64)
    owner[identity[:n_reference]] = np.arange(n_reference)
    shared: np.ndarray = owner[identity[n_reference:]]
    return shared


def _midranks(
    ranked: np.ndarray,
    values: np.ndarray,
    valid: np.ndarray,
    own: np.ndarray,
    own_values: np.ndarray,
) -> np.ndarray:
    """Reference midrank of each value; rows ``own`` are left out of their own rank.

    ``(count(ranked < x) + count(ranked == x) / 2) / n`` over the sorted valid
    reference values ``ranked``. ``own`` indexes the values whose own reference
    row, holding ``own_values``, is among them.
    """
    below = np.searchsorted(ranked, values, side="left")
    through = np.searchsorted(ranked, values, side="right")
    numerator = 0.5 * (below + through)
    denominator = np.full(values.shape, float(ranked.size))
    x = values[own]
    numerator[own] -= np.where(own_values < x, 1.0, np.where(own_values == x, 0.5, 0.0))
    denominator[own] -= 1.0
    out = np.full(values.shape, np.nan)
    keep = valid & (denominator > 0)
    out[keep] = numerator[keep] / denominator[keep]
    return out


def _ks_distance(ranked_a: np.ndarray, ranked_b: np.ndarray) -> float:
    """``sup_t |F_a(t) - F_b(t)|`` of two empirical CDFs over the extended reals."""
    points = np.union1d(ranked_a, ranked_b)
    cdf_a = np.searchsorted(ranked_a, points, side="right") / ranked_a.size
    cdf_b = np.searchsorted(ranked_b, points, side="right") / ranked_b.size
    return float(np.abs(cdf_a - cdf_b).max())


def _contrast(
    ranked: np.ndarray,
    reference_values: np.ndarray,
    reference_codes: np.ndarray,
    query_values: np.ndarray,
    query_codes: np.ndarray,
    levels: tuple[float, ...],
) -> dict[str, Any]:
    """Both sides' counts and statistics, and the descriptive contrasts between them."""
    query_ranked = np.sort(query_values[query_codes == VALID])
    reference_finite = ranked[np.isfinite(ranked)]
    query_finite = query_ranked[np.isfinite(query_ranked)]
    both = ranked.size > 0 and query_ranked.size > 0
    return {
        "reference": {
            "n_worlds": int(reference_codes.size),
            **_column_stats(reference_values, reference_codes, levels),
        },
        "query": {
            "n_worlds": int(query_codes.size),
            **_column_stats(query_values, query_codes, levels),
        },
        "median_shift": (
            float(np.median(query_finite) - np.median(reference_finite))
            if reference_finite.size and query_finite.size
            else None
        ),
        "ks_distance": _ks_distance(ranked, query_ranked) if both else None,
        "fraction_below_range": float(np.mean(query_ranked < ranked[0])) if both else None,
        "fraction_above_range": float(np.mean(query_ranked > ranked[-1])) if both else None,
    }


@dataclass(frozen=True)
class DescriptorComparison:
    """Query rows placed in a reference distribution, per stratum and feature.

    ``percentiles[i, j]`` is the reference midrank of query row ``i`` on
    feature ``j`` within its stratum, ``(count(ref < x) + count(ref == x) / 2)
    / n`` over the valid reference values there, ``+inf``/``-inf`` included as
    ordered values. A query row that is itself a valid reference row there
    (same ``source_id`` and ``world_id``) is left out of its own rank. NaN when
    the query value is not valid or no reference value remains.

    Per stratum and feature, :meth:`stats` holds both sides' counts and
    finite-value statistics plus ``median_shift`` (query minus reference finite
    median), ``ks_distance`` (largest gap between the two empirical CDFs over
    valid values) and the fractions of valid query values strictly below the
    reference minimum or above its maximum; None when a side has no values.
    """

    features: tuple[str, ...]
    by: tuple[str, ...]
    strata: tuple[tuple[Any, ...], ...]
    quantile_levels: tuple[float, ...]
    source_ids: np.ndarray
    world_ids: np.ndarray
    group_ids: np.ndarray
    query_strata: np.ndarray
    percentiles: np.ndarray
    records: tuple[tuple[dict[str, Any], ...], ...]

    def stats(self, stratum_index: int, feature: str) -> dict[str, Any]:
        """Both sides and their contrasts for ``feature`` in one stratum."""
        s = _stratum_position(self.strata, stratum_index)
        record = self.records[s][_feature_position(self.features, feature)]
        return {**record, "reference": dict(record["reference"]), "query": dict(record["query"])}

    def query_frame(self) -> pd.DataFrame:
        """One row per query row: identity, stratum columns, ``percentile:<feature>``."""
        import pandas as pd

        data: dict[str, Any] = {
            "source_id": self.source_ids,
            "world_id": self.world_ids,
            "group_id": self.group_ids,
        }
        strata = self.query_strata.tolist()
        for j, column in enumerate(self.by):
            data[column] = np.asarray([self.strata[s][j] for s in strata], dtype=np.int64)
        for j, name in enumerate(self.features):
            data[f"percentile:{name}"] = self.percentiles[:, j]
        return pd.DataFrame(data)

    def to_frame(self) -> pd.DataFrame:
        """One row per stratum x feature: ``reference_*`` and ``query_*`` statistics, contrasts."""
        import pandas as pd

        keys = ("n_worlds", *_stat_keys(self.quantile_levels))
        rows = []
        for s, key in enumerate(self.strata):
            for name, record in zip(self.features, self.records[s]):
                rows.append(
                    [
                        *key,
                        name,
                        *(_frame_value(record["reference"][k]) for k in keys),
                        *(_frame_value(record["query"][k]) for k in keys),
                        *(_frame_value(record[k]) for k in _CONTRAST_KEYS),
                    ]
                )
        columns = [
            *self.by,
            "feature",
            *(f"reference_{k}" for k in keys),
            *(f"query_{k}" for k in keys),
            *_CONTRAST_KEYS,
        ]
        return pd.DataFrame(rows, columns=columns)

    def table(self, feature: str | None = None) -> str:
        """Every feature of the single stratum, or one ``feature`` across strata."""
        columns = ["reference", "median_shift", "ks_distance", "below_range", "above_range"]
        subtitle, layout = _table_layout(self.by, self.strata, self.features, feature)
        rows = []
        for label, s, name in layout:
            record = self.records[s][self.features.index(name)]
            cells = [str(record["reference"]["n_valid"])]
            cells += [_cell(record[k]) for k in _CONTRAST_KEYS]
            rows.append((label, str(record["query"]["n_valid"]), cells))
        return _render_table(
            f"descriptor comparison — {subtitle} — valid query vs reference values",
            columns,
            rows,
            count_header="query",
        )

    def summary(self) -> dict[str, Any]:
        """Strictly JSON-safe report (``json.dumps(..., allow_nan=False)``)."""
        payload: dict[str, Any] = {
            "features": list(self.features),
            "by": list(self.by),
            "quantiles": list(self.quantile_levels),
            "n_query": int(self.world_ids.size),
            "strata": [
                {"key": dict(zip(self.by, key)), "features": dict(zip(self.features, records))}
                for key, records in zip(self.strata, self.records)
            ],
        }
        sanitized: dict[str, Any] = _strict_json_value(payload)
        return sanitized


def compare_descriptors(
    reference: WorldDescriptors,
    query: WorldDescriptors,
    *,
    features: Sequence[str] | None = None,
    by: Sequence[str] = (),
    quantiles: Sequence[float] = DEFAULT_QUANTILES,
) -> DescriptorComparison:
    """Place every query row in the reference distribution of its stratum.

    Parameters
    ----------
    reference
        Rows that define ranks and ranges. The roles are directional: swapping
        the tables answers a different question.
    query
        Rows being placed.
    features
        Features to compare; None for every reference feature, each of which
        must then exist with an equal definition in ``query``.
    by
        Metadata strata (:data:`~pymc_generator.descriptors.METADATA_COLUMNS`),
        taken from the union of both tables. A query row is compared only with
        reference rows of equal ``by`` values.
    quantiles
        Quantile levels of each side's finite valid values.

    Returns
    -------
    DescriptorComparison
    """
    _require_table(reference, "reference")
    _require_table(query, "query")
    names = _features(reference, features)
    _check_compatible(reference.definitions, query.definitions, names)
    if isinstance(by, Sequence) and not isinstance(by, str) and "source_id" in by:
        raise ValueError(
            "by cannot hold 'source_id': a comparison stratum must exist in both tables, and "
            "source ids name the collections being compared; select rows with "
            "WorldDescriptors.take instead"
        )
    columns = _columns(by, METADATA_COLUMNS, "by")
    levels = _validate_quantiles(quantiles)
    n_reference = reference.n_worlds
    keys, index = _stratify(
        [np.concatenate([reference.grouping(c), query.grouping(c)]) for c in columns],
        n_reference + query.n_worlds,
    )
    reference_index, query_index = index[:n_reference], index[n_reference:]
    own = _shared_rows(reference, query)
    reference_columns = [reference.feature_index(name) for name in names]
    query_columns = [query.feature_index(name) for name in names]
    percentiles = np.full((query.n_worlds, len(names)), np.nan)
    records = []
    for s, (r_rows, q_rows) in enumerate(
        zip(_stratum_rows(reference_index, len(keys)), _stratum_rows(query_index, len(keys)))
    ):
        q_own = own[q_rows]
        mine = np.flatnonzero(q_own >= 0)
        mine = mine[reference_index[q_own[mine]] == s]
        stratum = []
        for j, (rc, qc) in enumerate(zip(reference_columns, query_columns)):
            r_values, r_codes = reference.values[r_rows, rc], reference.status[r_rows, rc]
            q_values, q_codes = query.values[q_rows, qc], query.status[q_rows, qc]
            ranked = np.sort(r_values[r_codes == VALID])
            left_out = mine[reference.status[q_own[mine], rc] == VALID]
            percentiles[q_rows, j] = _midranks(
                ranked, q_values, q_codes == VALID, left_out, reference.values[q_own[left_out], rc]
            )
            stratum.append(_contrast(ranked, r_values, r_codes, q_values, q_codes, levels))
        records.append(tuple(stratum))
    return DescriptorComparison(
        features=names,
        by=columns,
        strata=tuple(keys),
        quantile_levels=levels,
        source_ids=query.source_ids,
        world_ids=query.world_ids,
        group_ids=query.group_ids,
        query_strata=query_index,
        percentiles=percentiles,
        records=tuple(records),
    )
