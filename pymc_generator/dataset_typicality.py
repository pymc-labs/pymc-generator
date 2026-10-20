"""How typical one observed dataset is of a reference corpus or generator.

:func:`typicality` places datasets — rows of a
:class:`~pymc_generator.descriptors.WorldDescriptors` table, typically one
observed dataset (treatments, covariates and outcome over time) read with
:meth:`~pymc_generator.descriptors.ObservedWorlds.from_csv` — against a
reference table of generated worlds: an existing corpus, or worlds generated
for the purpose from a configuration narrowed by :func:`matched_prior`. Every
query row gets two answers:

* a typicality p-value (split conformal). The reference rows of the dataset's
  shape are split by generation group into ruler and calibration worlds; every
  calibration world and the dataset are scored against the ruler, and
  ``p = (1 + #{calibration worlds scoring at least as atypical}) /
  (n_calibration + 1)``;
* a profile: per descriptor feature, the dataset's value and state, its
  midrank percentile among the reference worlds of its shape, the reference
  low/median/high, whether it lies inside, and whether it breaks the support
  of the ruler.

Contract. If the dataset and the reference worlds are independent draws from
one generator, given the dataset's shape, and ``features``, ``k``,
``calibration_fraction``, ``central`` and ``seed`` are fixed before looking at
``p``, then ``P(p <= alpha) <= alpha`` for every ``alpha``. The guarantee is
marginal over references: for one fixed reference the realized rate scatters
around ``alpha`` by about ``sqrt(alpha * (1 - alpha) / (n_calibration + 2))``,
and the p-values of a panel scored against one reference are dependent. One
row per calibration cell makes the calibration worlds independent. A small
``p`` says the dataset is unusual for this reference; a large ``p`` says these
features do not tell them apart, not that the dataset came from the
generator. Raw levels only: describe the dataset on the scale it was recorded,
as the generator produces levels. The reference and the dataset must not share
generation seeds. Self-queries (``typicality(reference, reference)``) cost up
to one scoring of the calibration worlds per excluded group.

This module is numpy-only at import. It draws no random numbers — the split is
a hash of ``seed`` and each group's identity — and writes nothing.
"""

from __future__ import annotations

import hashlib
import math
from collections.abc import Sequence
from dataclasses import dataclass, replace
from fractions import Fraction
from typing import TYPE_CHECKING, Any

import numpy as np

from .descriptor_analysis import _OK as _NEIGHBOR_OK
from .descriptor_analysis import (
    DescriptorScale,
    _check_compatible,
    _features,
    _group_codes,
    _key_codes,
    _midranks,
    _positive_int,
    _shared_rows,
    nearest_worlds,
)
from .descriptors import (
    DESCRIPTOR_STATUSES,
    INELIGIBLE,
    METADATA_COLUMNS,
    UNDEFINED,
    VALID,
    FeatureDefinition,
    ObservedWorlds,
    WorldDescriptors,
)
from .diagnostics import _cell, _render_table, _strict_json_value

if TYPE_CHECKING:  # pragma: no cover - typing only
    import pandas as pd

    from .sampler import SCMPrior

__all__ = [
    "FEATURE_ROLES",
    "TYPICALITY_STATUSES",
    "Typicality",
    "matched_prior",
    "typicality",
]

#: Per-query-row outcome of :func:`typicality`, by status code.
TYPICALITY_STATUSES: tuple[str, ...] = ("ok", "no_reference", "too_few", "duplicate")
_OK_ROW, _NO_REFERENCE, _TOO_FEW, _DUPLICATE = 0, 1, 2, 3

#: How a selected feature enters the score of a query row, by role code.
FEATURE_ROLES: tuple[str, ...] = ("distance", "constant", "state", "duplicate")
_DISTANCE, _CONSTANT, _STATE, _REDUNDANT = 0, 1, 2, 3

#: Value states a support check tells apart: finite, +inf, -inf, undefined, ineligible.
_FINITE, _POSINF, _NEGINF, _UNDEFINED_STATE, _INELIGIBLE_STATE = 0, 1, 2, 3, 4
_N_STATES = 5

#: Two valid finite values are equal for the duplicate check when
#: ``|x - r| <= _DUPLICATE_TOLERANCE * max(|x|, |r|, 1)``: far above CSV and
#: float32 round-off, far below what separates two distinct worlds on every
#: shared feature at once.
_DUPLICATE_TOLERANCE = 1e-4

#: Statistics that take a handful of levels (a count over the time steps or over
#: the active series): distinct worlds often share them, so equal values are no
#: evidence of one world and the copy and duplicate checks leave them out.
_COUNT_STATISTICS: tuple[str, ...] = ("zero_fraction", "constant_fraction")

_NO_ROWS = np.zeros(0, dtype=np.int64)
_NO_VALUES = np.zeros(0)


# ---------------------------------------------------------------------------
# Result
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Typicality:
    """Typicality p-values and profiles of query rows against a reference.

    Per query row ``i``: ``status`` codes index :data:`TYPICALITY_STATUSES`;
    ``p_value``, ``p_min`` (the smallest p-value ``1 / (n_calibration + 1)``
    this row could get), ``violations`` and ``distance`` are NaN unless the
    row is ``ok``. ``n_ruler``, ``n_calibration`` and ``n_reference`` count the
    ruler worlds, calibration worlds and every reference row of the row's
    shape left after its own group (for a reference row) is removed.
    ``n_distance_features`` and ``n_constant_features`` count the selected
    features in those roles. ``n_profiled`` counts the non-duplicate features
    with a percentile, ``n_inside`` those inside the central interval, and
    ``expected_outside`` the number expected outside for an exchangeable row.

    Per query row and feature ``(i, j)``: ``value``/``value_status`` (codes of
    :data:`~pymc_generator.descriptors.DESCRIPTOR_STATUSES`), ``percentile``
    (midrank among the valid reference values of its shape), the
    ``reference_low``/``reference_median``/``reference_high`` order statistics,
    ``inside`` the inclusive interval, ``violation`` (``ok`` rows only) and
    ``role`` (codes of :data:`FEATURE_ROLES`).
    """

    features: tuple[str, ...]
    k: int
    calibration_fraction: float
    central: float
    seed: int
    source_ids: np.ndarray
    world_ids: np.ndarray
    group_ids: np.ndarray
    status: np.ndarray
    p_value: np.ndarray
    p_min: np.ndarray
    violations: np.ndarray
    distance: np.ndarray
    n_ruler: np.ndarray
    n_calibration: np.ndarray
    n_reference: np.ndarray
    n_distance_features: np.ndarray
    n_constant_features: np.ndarray
    n_profiled: np.ndarray
    n_inside: np.ndarray
    expected_outside: np.ndarray
    value: np.ndarray
    value_status: np.ndarray
    percentile: np.ndarray
    reference_low: np.ndarray
    reference_median: np.ndarray
    reference_high: np.ndarray
    inside: np.ndarray
    violation: np.ndarray
    role: np.ndarray

    @property
    def n_query(self) -> int:
        return int(self.status.size)

    def _row(self, row: Any) -> int:
        if isinstance(row, bool) or not isinstance(row, (int, np.integer)):
            raise TypeError(f"row must be an integer, got {type(row).__name__}")
        if not 0 <= row < self.n_query:
            raise IndexError(f"row {row} is out of range for {self.n_query} query rows")
        return int(row)

    def _label(self, row: int) -> str:
        return f"{self.source_ids[row]}:{int(self.world_ids[row])}"

    def _violating(self, row: int) -> list[str]:
        return [name for name, flag in zip(self.features, self.violation[row]) if flag]

    def table(self) -> str:
        """One line per query row, then the violating features of each row that has any."""
        labels = np.asarray(TYPICALITY_STATUSES, dtype=object)
        title = (
            f"dataset typicality — {self.n_query} query rows, {len(self.features)} features, "
            f"k={self.k}, calibration_fraction={self.calibration_fraction}, "
            f"central={self.central}, seed={self.seed}"
        )
        columns = [
            "status",
            "p_value",
            "p_min",
            "violations",
            "distance",
            "ruler",
            "calibration",
            "inside",
            "exp_outside",
        ]
        rows = []
        for i in range(self.n_query):
            ok = self.status[i] == _OK_ROW
            rows.append(
                (
                    self._label(i),
                    str(int(self.n_reference[i])),
                    [
                        str(labels[self.status[i]]),
                        _cell(float(self.p_value[i]) if ok else None),
                        _cell(float(self.p_min[i]) if ok else None),
                        str(int(self.violations[i])) if ok else "n/a",
                        _cell(float(self.distance[i]) if ok else None),
                        str(int(self.n_ruler[i])),
                        str(int(self.n_calibration[i])),
                        f"{int(self.n_inside[i])}/{int(self.n_profiled[i])}",
                        _cell(float(self.expected_outside[i])),
                    ],
                )
            )
        lines = [_render_table(title, columns, rows, count_header="reference")]
        notes = [
            f"{self._label(i)}: {', '.join(names)}"
            for i in range(self.n_query)
            if (names := self._violating(i))
        ]
        if notes:
            lines += [
                "",
                "violating features (a state or a constant the ruler never shows):",
                *notes,
            ]
        return "\n".join(lines)

    def to_frame(self) -> pd.DataFrame:
        """One row per query row: identity, status, p-value, score and counts."""
        import pandas as pd

        data: dict[str, Any] = {
            "source_id": self.source_ids,
            "world_id": self.world_ids,
            "group_id": self.group_ids,
            "status": np.asarray(TYPICALITY_STATUSES, dtype=object)[self.status],
            "p_value": self.p_value,
            "p_min": self.p_min,
            "violations": self.violations,
            "distance": self.distance,
            "n_ruler": self.n_ruler,
            "n_calibration": self.n_calibration,
            "n_reference": self.n_reference,
            "n_distance_features": self.n_distance_features,
            "n_constant_features": self.n_constant_features,
            "n_profiled": self.n_profiled,
            "n_inside": self.n_inside,
            "expected_outside": self.expected_outside,
        }
        return pd.DataFrame(data)

    def profile(self, row: int = 0) -> pd.DataFrame:
        """One row per feature of query row ``row``: value, state, reference placement, role."""
        import pandas as pd

        i = self._row(row)
        return pd.DataFrame(
            {
                "feature": list(self.features),
                "role": np.asarray(FEATURE_ROLES, dtype=object)[self.role[i]],
                "value": self.value[i],
                "status": np.asarray(DESCRIPTOR_STATUSES, dtype=object)[self.value_status[i]],
                "percentile": self.percentile[i],
                "reference_low": self.reference_low[i],
                "reference_median": self.reference_median[i],
                "reference_high": self.reference_high[i],
                "inside": self.inside[i],
                "violation": self.violation[i],
            }
        )

    def summary(self) -> dict[str, Any]:
        """Strictly JSON-safe report (``json.dumps(..., allow_nan=False)``).

        Non-finite numbers become null: the p-values and scores of rows that
        are not ``ok``, and an infinite distance (a query row that is not
        finite on a distance feature, which is also listed as a violation).
        """
        counts = np.bincount(self.status, minlength=len(TYPICALITY_STATUSES))
        rows = []
        for i in range(self.n_query):
            ok = self.status[i] == _OK_ROW
            rows.append(
                {
                    "source_id": str(self.source_ids[i]),
                    "world_id": self.world_ids[i],
                    "group_id": self.group_ids[i],
                    "status": TYPICALITY_STATUSES[self.status[i]],
                    "p_value": self.p_value[i],
                    "p_min": self.p_min[i],
                    "violations": int(self.violations[i]) if ok else None,
                    "distance": self.distance[i],
                    "n_ruler": self.n_ruler[i],
                    "n_calibration": self.n_calibration[i],
                    "n_reference": self.n_reference[i],
                    "n_distance_features": self.n_distance_features[i],
                    "n_constant_features": self.n_constant_features[i],
                    "n_profiled": self.n_profiled[i],
                    "n_inside": self.n_inside[i],
                    "expected_outside": self.expected_outside[i],
                    "violating_features": self._violating(i),
                }
            )
        payload: dict[str, Any] = {
            "features": list(self.features),
            "k": self.k,
            "calibration_fraction": self.calibration_fraction,
            "central": self.central,
            "seed": self.seed,
            "n_query": self.n_query,
            "status_counts": dict(zip(TYPICALITY_STATUSES, counts)),
            "rows": rows,
        }
        sanitized: dict[str, Any] = _strict_json_value(payload)
        return sanitized


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


def _identity(descriptors: WorldDescriptors, row: int) -> str:
    return f"({str(descriptors.source_ids[row])!r}, {int(descriptors.world_ids[row])})"


def _require_descriptors(value: Any, what: str) -> WorldDescriptors:
    if not isinstance(value, WorldDescriptors):
        raise TypeError(
            f"{what} must be WorldDescriptors, got {type(value).__name__}; describe it first "
            "with world_descriptors(..., source_id=...), with the same statistics and lags for "
            "the reference and the query"
        )
    return value


def _tells_apart(definition: FeatureDefinition) -> bool:
    """An observable feature whose equal values can be evidence of one world."""
    return definition.observable and definition.statistic not in _COUNT_STATISTICS


def _open_unit(value: Any, what: str) -> float:
    if isinstance(value, (bool, np.bool_)) or not isinstance(
        value, (int, float, np.integer, np.floating)
    ):
        raise TypeError(f"{what} must be a real number, got {type(value).__name__}")
    if not 0.0 < float(value) < 1.0:
        raise ValueError(f"{what} must lie strictly between 0 and 1, got {value}")
    return float(value)


def _seed(value: Any) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise TypeError(f"seed must be an integer, got {type(value).__name__}")
    if value < 0:
        raise ValueError(f"seed must be a non-negative integer, got {value}")
    return int(value)


def _shared_observable(reference: WorldDescriptors, query: WorldDescriptors) -> tuple[str, ...]:
    """Observable features of both tables with equal definitions, in reference order."""
    theirs = {d.name: d for d in query.definitions}
    return tuple(d.name for d in reference.definitions if d.observable and theirs.get(d.name) == d)


def _selected_features(
    reference: WorldDescriptors, query: WorldDescriptors, features: Sequence[str] | None
) -> tuple[str, ...]:
    if features is None:
        names = _shared_observable(reference, query)
        if not names:
            raise ValueError(
                "the reference and the query share no observable feature with an equal "
                "definition; describe both with the same statistics, lags and reducers"
            )
        return names
    names = _features(reference, features)
    hidden = [name for name in names if not reference.definition(name).observable]
    if hidden:
        raise ValueError(
            f"features {hidden[:5]} are not observable (generating-process truth); typicality "
            "compares observable features only, which an observed dataset also carries"
        )
    _check_compatible(reference.definitions, query.definitions, names)
    return names


def _check_query_rows(query: WorldDescriptors, names: tuple[str, ...]) -> None:
    """Every query row is a dataset: active treatments, active covariates and an outcome."""
    needs = "a dataset needs at least one treatment, one covariate and an observed outcome"
    for column, role in (
        ("n_treatments_active", "treatment"),
        ("n_covariates_active", "covariate"),
    ):
        missing = np.flatnonzero(query.metadata[column] < 1)
        if missing.size:
            raise ValueError(
                f"query row {_identity(query, int(missing[0]))} has no active {role}; {needs}"
            )
    outcome = [
        query.feature_index(name)
        for name in names
        if query.definition(name).role in ("outcome", "treatment_outcome")
    ]
    if outcome:
        missing = np.flatnonzero((query.status[:, outcome] == INELIGIBLE).all(axis=1))
        if missing.size:
            raise ValueError(
                f"query row {_identity(query, int(missing[0]))} has no observed outcome (every "
                f"selected outcome feature is ineligible); {needs}: describe it with its outcome "
                "series (ObservedWorlds(..., outcome_raw=...))"
            )


def _reference_rows(
    reference: WorldDescriptors,
    query: WorldDescriptors,
    shared: tuple[str, ...],
    reference_columns: np.ndarray,
    query_columns: np.ndarray,
) -> np.ndarray:
    """Per query row, the reference row it is, or -1 for an external row.

    A query row from a reference source must be one of its rows, unchanged.
    """
    own = _shared_rows(reference, query)
    sources = set(reference.sources)
    from_reference = np.asarray([s in sources for s in query.source_ids.tolist()], dtype=bool)
    stray = np.flatnonzero(from_reference & (own < 0))
    unchanged = "a query row from a reference source must be that reference row, unchanged"
    if stray.size:
        i = int(stray[0])
        raise ValueError(
            f"query row {_identity(query, i)} shares source_id {str(query.source_ids[i])!r} "
            "with the reference but is not one of its rows; give an external dataset its own "
            "source_id"
        )
    rows = np.flatnonzero(own >= 0)
    matched = own[rows]
    q_values = query.values[np.ix_(rows, query_columns)]
    r_values = reference.values[np.ix_(matched, reference_columns)]
    same = (
        query.status[np.ix_(rows, query_columns)]
        == reference.status[np.ix_(matched, reference_columns)]
    ) & ((q_values == r_values) | (np.isnan(q_values) & np.isnan(r_values)))
    moved = np.flatnonzero(query.group_ids[rows] != reference.group_ids[matched])
    if moved.size:
        m = int(moved[0])
        i, r = int(rows[m]), int(matched[m])
        group = int(reference.group_ids[r])
        if query.group_ids[i] >= 0:
            raise ValueError(
                f"query row {_identity(query, i)} is a reference row but carries group_id "
                f"{int(query.group_ids[i])} where the reference has {group}; {unchanged}"
            )
        if same[m].all():
            raise ValueError(
                f"query row {_identity(query, i)} equals the reference row with its identity on "
                "every shared feature but has no generation group (group_id -1, the reference "
                f"has {group}): it is that reference row described without its generation "
                "group. Pass the reference's group_ids= to world_descriptors, or take the rows "
                "from the reference table (e.g. reference.take(...))"
            )
        feature = shared[int(np.flatnonzero(~same[m])[0])]
        raise ValueError(
            f"query row {_identity(query, i)} has no generation group (group_id -1) and differs "
            f"on {feature!r} from the reference row with its identity (group_id {group}): two "
            f"collections share source_id {str(query.source_ids[i])!r}; give an external "
            "dataset its own source_id"
        )
    changed = np.flatnonzero(~same.all(axis=1))
    if changed.size:
        i = int(changed[0])
        feature = shared[int(np.flatnonzero(~same[i])[0])]
        raise ValueError(
            f"query row {_identity(query, int(rows[i]))} has the identity of a reference row but "
            f"differs from it on {feature!r}; {unchanged} — give an external dataset its own "
            "source_id"
        )
    return own


# ---------------------------------------------------------------------------
# Strata, split and ruler
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class _Stratum:
    """The reference rows of one shape and their ruler / calibration split."""

    rows: np.ndarray  # reference positions, (source_id, world_id) order
    groups: np.ndarray  # group code per row
    ruler_rows: np.ndarray  # every row of the ruler groups, identity order
    ruler_groups: np.ndarray  # group code per ruler row
    calibration_rows: np.ndarray  # lowest-world_id row of each calibration group
    calibration_groups: np.ndarray  # group code per calibration row


@dataclass(frozen=True)
class _Ruler:
    """What a ruler R fixes: roles, constants, seen states, scale and calibration scores."""

    rows: np.ndarray
    role: np.ndarray  # (F,) role codes
    level: np.ndarray  # (F,) the constant c_f, NaN for other roles
    seen: np.ndarray  # (F, _N_STATES) states present among the ruler rows
    table: WorldDescriptors | None  # set only when the ruler can score: |R| >= max(k, 2)
    scale: DescriptorScale | None
    distance_features: tuple[str, ...]
    calibration_violations: np.ndarray  # per stratum calibration row
    calibration_distance: np.ndarray


def _strata(reference: WorldDescriptors) -> dict[tuple[int, int, int], np.ndarray]:
    """Reference rows per shape, each in ``(source_id, world_id)`` order."""
    shapes = np.stack([reference.metadata[c] for c in METADATA_COLUMNS], axis=1).tolist()
    _, identity = _key_codes([reference.source_ids, reference.world_ids], reference.n_worlds)
    rows: dict[tuple[int, int, int], list[int]] = {}
    for position in np.argsort(identity).tolist():
        n_treat, n_cov, n_time = shapes[position]
        rows.setdefault((n_treat, n_cov, n_time), []).append(position)
    return {shape: np.asarray(members, dtype=np.int64) for shape, members in rows.items()}


def _takes_two_values(values: np.ndarray, status: np.ndarray) -> bool:
    """Whether some column holds two distinct valid values (infinities included)."""
    valid = status == VALID
    lowest = np.where(valid, values, np.inf).min(axis=0, initial=np.inf)
    highest = np.where(valid, values, -np.inf).max(axis=0, initial=-np.inf)
    return bool((lowest < highest).any())


def _refuse_copies(
    reference: WorldDescriptors, rows: np.ndarray, columns: np.ndarray, groups: np.ndarray
) -> None:
    """Rows of different groups bit-identical on every ``columns`` feature are copies: refuse.

    ``columns`` are the observable features that can tell worlds apart (count
    statistics left out). Judged only when one of them takes two distinct
    valid values among the rows: otherwise distinct worlds look identical too.
    """
    if rows.size < 2:
        return
    values = reference.values[np.ix_(rows, columns)]
    status = reference.status[np.ix_(rows, columns)]
    if not _takes_two_values(values, status):
        return
    # NaN marks every non-valid cell and the status tells those apart; + 0.0 maps -0.0 to 0.0.
    cells = np.concatenate(
        [np.where(status == VALID, values, 0.0) + 0.0, status.astype(np.float64)], axis=1
    )
    _, first, inverse = np.unique(cells, axis=0, return_index=True, return_inverse=True)
    canonical = rows[first[inverse.reshape(-1)]]
    clash = np.flatnonzero(groups[rows] != groups[canonical])
    if clash.size:
        a, b = int(canonical[clash[0]]), int(rows[clash[0]])
        raise ValueError(
            f"reference rows {_identity(reference, a)} (group_id {int(reference.group_ids[a])}) "
            f"and {_identity(reference, b)} (group_id {int(reference.group_ids[b])}) are "
            "bit-identical on every observable feature that can tell worlds apart but belong "
            "to different generation groups: the reference holds the same world twice (corpora "
            "generated with the same seed and concatenated?). Drop the copies or regenerate one "
            "part with another seed"
        )


def _split_key(seed: int, source_id: str, group_id: int) -> int:
    # surrogatepass: plain UTF-8 bytes for every encodable string, and still a key
    # for a source_id holding a lone surrogate (e.g. an undecodable file name).
    text = f"{seed}|{source_id}|{group_id}".encode("utf-8", "surrogatepass")
    return int.from_bytes(hashlib.blake2b(text, digest_size=8).digest(), "big")


def _split(
    reference: WorldDescriptors,
    rows: np.ndarray,
    group_codes: np.ndarray,
    seed: int,
    calibration_fraction: float,
) -> _Stratum:
    """Calibration and ruler groups of one stratum: a hash of the seed and group identity."""
    groups = group_codes[rows]
    # Rows are in identity order and a group has one source, so a group's first
    # row is its lowest world_id.
    codes, first = np.unique(groups, return_index=True)
    keyed = []
    for code, position in zip(codes.tolist(), first.tolist()):
        source = str(reference.source_ids[rows[position]])
        group = int(reference.group_ids[rows[position]])
        keyed.append((_split_key(seed, source, group), source, group, code, position))
    keyed.sort()
    n_groups = len(keyed)
    n_calibration = (
        min(max(round(calibration_fraction * n_groups), 1), n_groups - 1) if n_groups >= 2 else 0
    )
    positions = np.sort(np.asarray([item[4] for item in keyed[:n_calibration]], dtype=np.int64))
    ruler = np.isin(groups, [item[3] for item in keyed[n_calibration:]])
    return _Stratum(
        rows=rows,
        groups=groups,
        ruler_rows=rows[ruler],
        ruler_groups=groups[ruler],
        calibration_rows=rows[positions],
        calibration_groups=groups[positions],
    )


def _reduced_set_size(role: str, n_treatments: int, n_covariates: int) -> int | None:
    """Elements of the node or pair set a reduced role runs over at shape (K, M)."""
    sizes = {
        "treatment": n_treatments,
        "covariate": n_covariates,
        "treatment_pair": n_treatments * (n_treatments - 1) // 2,
        "covariate_pair": n_covariates * (n_covariates - 1) // 2,
        "treatment_covariate": n_treatments * n_covariates,
        "treatment_outcome": n_treatments,
    }
    return sizes.get(role)


def _structural_duplicates(
    definitions: Sequence[FeatureDefinition], n_treatments: int, n_covariates: int
) -> np.ndarray:
    """Reducers that repeat an earlier selected one because their set has a single element."""
    redundant = np.zeros(len(definitions), dtype=bool)
    first_seen: set[tuple[str, str, str, int | None]] = set()
    for j, d in enumerate(definitions):
        if d.reducer is None or _reduced_set_size(d.role, n_treatments, n_covariates) != 1:
            continue
        key = (d.role, d.statistic, d.view, d.lag)
        if key in first_seen:
            redundant[j] = True
        else:
            first_seen.add(key)
    return redundant


def _value_states(values: np.ndarray, status: np.ndarray) -> np.ndarray:
    """Support state of every cell: finite, +inf, -inf, undefined or ineligible."""
    states = np.full(status.shape, _INELIGIBLE_STATE, dtype=np.int64)
    states[status == UNDEFINED] = _UNDEFINED_STATE
    valid = status == VALID
    states[valid & np.isposinf(values)] = _POSINF
    states[valid & np.isneginf(values)] = _NEGINF
    states[valid & np.isfinite(values)] = _FINITE
    return states


def _score(
    ruler: _Ruler, k: int, values: np.ndarray, status: np.ndarray, table: WorldDescriptors
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """``(violations, distance, per-feature violation flags)`` of rows against a ruler."""
    states = _value_states(values, status)
    flags = ~ruler.seen[np.arange(states.shape[1]), states]
    constant = ruler.role == _CONSTANT
    flags |= constant & (states == _FINITE) & (values != ruler.level)
    flags[:, ruler.role == _REDUNDANT] = False
    violations = flags.sum(axis=1).astype(np.float64)
    if not ruler.distance_features:
        return violations, np.zeros(table.n_worlds), flags
    if ruler.table is None:  # pragma: no cover - only a ruler of max(k, 2) rows scores
        raise AssertionError("a ruler too small to score was asked for distances")
    found = nearest_worlds(
        ruler.table, table, features=ruler.distance_features, scale=ruler.scale, k=k
    )
    distance = np.full(table.n_worlds, np.inf)
    ok = found.status == _NEIGHBOR_OK
    distance[ok] = found.distances[ok].mean(axis=1)
    return violations, distance, flags


def _ruler(
    reference: WorldDescriptors,
    stratum: _Stratum,
    excluded: int,
    columns: np.ndarray,
    names: tuple[str, ...],
    redundant: np.ndarray,
    k: int,
) -> _Ruler:
    """Roles from the ruler rows left after removing group ``excluded``; scores if it can."""
    rows = stratum.ruler_rows[stratum.ruler_groups != excluded]
    values = reference.values[np.ix_(rows, columns)]
    status = reference.status[np.ix_(rows, columns)]
    states = _value_states(values, status)
    seen = np.stack([(states == s).any(axis=0) for s in range(_N_STATES)], axis=1)
    finite = (states == _FINITE).all(axis=0)
    equal = (values == values[:1]).all(axis=0)
    constant = finite & equal & ~redundant & (rows.size >= 2)
    distance = finite & ~equal & ~redundant
    role = np.full(len(names), _STATE, dtype=np.uint8)
    role[distance] = _DISTANCE
    role[constant] = _CONSTANT
    role[redundant] = _REDUNDANT
    level = np.where(constant, values[0], np.nan) if rows.size else np.full(len(names), np.nan)
    distance_features = tuple(names[j] for j in np.flatnonzero(distance))
    ruler = _Ruler(
        rows=rows,
        role=role,
        level=level,
        seen=seen,
        table=None,
        scale=None,
        distance_features=distance_features,
        calibration_violations=_NO_VALUES,
        calibration_distance=_NO_VALUES,
    )
    if rows.size < max(k, 2):
        return ruler
    table = reference.take(rows)
    scale = DescriptorScale.fit(table, distance_features) if distance_features else None
    ruler = replace(ruler, table=table, scale=scale)
    calibration = stratum.calibration_rows
    if calibration.size:
        violations, distances, _ = _score(
            ruler,
            k,
            reference.values[np.ix_(calibration, columns)],
            reference.status[np.ix_(calibration, columns)],
            reference.take(calibration),
        )
        ruler = replace(ruler, calibration_violations=violations, calibration_distance=distances)
    return ruler


def _duplicates(
    query_values: np.ndarray,
    query_status: np.ndarray,
    reference_values: np.ndarray,
    reference_status: np.ndarray,
) -> np.ndarray:
    """Per query row: does a reference row equal it on every column?

    The columns are the shared observable features that can tell worlds apart
    (count statistics left out). Equal means the same status everywhere and,
    on valid cells, equal infinities or finite values within the duplicate
    tolerance. Judged only when one column takes two distinct valid values
    among the reference rows.
    """
    out = np.zeros(query_values.shape[0], dtype=bool)
    if not _takes_two_values(reference_values, reference_status):
        return out
    finite = np.isfinite(reference_values)
    for i in range(query_values.shape[0]):
        x, codes = query_values[i], query_status[i]
        with np.errstate(over="ignore", invalid="ignore"):
            near = np.abs(reference_values - x) <= _DUPLICATE_TOLERANCE * np.maximum(
                np.maximum(np.abs(reference_values), np.abs(x)), 1.0
            )
        close = (reference_values == x) | (finite & np.isfinite(x) & near)
        match = (reference_status == codes) & (close | (codes != VALID))
        out[i] = bool(match.all(axis=1).any())
    return out


def _order_indices(n: int, central: Fraction) -> tuple[int, int, int]:
    """1-based ``inverted_cdf`` positions ``max(1, ceil(n q))`` of low, median and high."""
    levels = ((1 - central) / 2, Fraction(1, 2), (1 + central) / 2)
    low, median, high = (max(1, math.ceil(n * q)) for q in levels)
    return low, median, high


# ---------------------------------------------------------------------------
# Entry points
# ---------------------------------------------------------------------------


def typicality(
    reference: WorldDescriptors,
    query: WorldDescriptors,
    *,
    features: Sequence[str] | None = None,
    k: int = 5,
    calibration_fraction: float = 0.5,
    central: float = 0.9,
    seed: int = 0,
) -> Typicality:
    """Score how typical each query row is of the reference worlds of its shape.

    Parameters
    ----------
    reference
        Generated worlds with known generation groups (``group_id >= 0``): a
        described corpus (its ``cell_id``), or any table built with
        ``world_descriptors(..., group_ids=...)``. The number of groups (cells)
        sets the calibration size, the smallest p-value and the generation
        time: calibration takes one world per calibration group, its lowest
        ``world_id``. Extra draws per cell are nearly free to generate; they
        join the ruler (ruler cells) or only the profile and the duplicate
        check (calibration cells), and can add power against subtle
        differences, at a scoring cost that grows with the ruler.
    query
        Datasets to score, e.g. ``world_descriptors(ObservedWorlds.from_csv(...),
        source_id=...)``. Each row needs at least one active treatment, one
        active covariate and, when an outcome feature is selected, an observed
        outcome. A row whose ``source_id`` occurs in the reference must be one
        of its rows, unchanged (same ``group_id``, values and statuses, e.g.
        taken with ``reference.take``); its own group is then left out of
        every reference set (a self-query).
    features
        Features scored and profiled; None for every observable feature both
        tables carry with an equal definition, in reference order. Truth
        features are refused.
    k
        Ruler neighbours averaged into the distance score.
    calibration_fraction
        Share of each stratum's generation groups used as calibration worlds,
        strictly between 0 and 1.
    central
        Coverage of the profile's central interval, strictly between 0 and 1.
    seed
        Seed of the ruler / calibration split.

    Returns
    -------
    Typicality

    Notes
    -----
    A query row is compared only with the reference rows of its shape (equal
    ``n_treatments_active``, ``n_covariates_active`` and ``n_time_steps``).
    Within a shape some query row uses, reference rows of different groups
    that are bit-identical on every observable feature that can tell worlds
    apart raise: the reference holds copies of one world. The count
    statistics ``zero_fraction`` and ``constant_fraction`` take a handful of
    levels that distinct worlds share, so they are no evidence of one world:
    this check and the duplicate check below leave them out, and each runs
    only when one remaining feature takes two distinct valid values among the
    reference rows it compares.

    **Split.** Per shape, each generation group ``(source_id, group_id)``
    gets the key ``int.from_bytes(hashlib.blake2b(f"{seed}|{source_id}|{group_id}"
    .encode("utf-8", "surrogatepass"), digest_size=8).digest(), "big")``
    (plain UTF-8; ``surrogatepass`` only admits a lone surrogate); groups are
    sorted by ``(key, source_id, group_id)`` and the first ``min(max(round(
    calibration_fraction * n), 1), n - 1)`` of the ``n`` groups (Python
    ``round``; none when ``n == 1``) are calibration groups, the rest ruler
    groups. A calibration group contributes its lowest-``world_id`` row, a
    ruler group all its rows. The split ignores values, other shapes and row
    order. A query row that is a reference row then loses its own group on
    both sides.

    **Status.** ``no_reference`` when no reference row of the shape remains;
    ``duplicate`` when some remaining row equals the query row on every
    observable feature both tables share, count statistics aside (same
    status; equal infinities or ``|x - r| <= 1e-4 * max(|x|, |r|, 1)``),
    provided one of those features takes two distinct valid values among
    them; ``too_few`` with fewer than ``max(k, 2)`` ruler rows or no
    calibration row; otherwise ``ok``. Rows that are not ``ok`` get NaN
    p-values and scores, no violation flags, and still their counts and
    profile.

    **Roles**, from the ruler rows R and the shape only: a reducer over a set
    with a single element at this shape (K treatments, M covariates,
    ``K(K-1)/2`` / ``M(M-1)/2`` / ``K*M`` pairs, K treatment-outcome pairs)
    repeats the first selected reducer of its statistic and is a
    ``duplicate``, never scored nor counted in the profile; a feature valid,
    finite and equal on every R row (with at least two R rows) is a
    ``constant``; valid, finite and not all equal on R, a ``distance``
    feature; anything else a ``state``.

    **Score**, lexicographic, of every calibration row and query row ``x``:
    ``violations(x)`` counts the scored features whose state (finite, +inf,
    -inf, undefined, ineligible) no R row shows, plus the constants ``x``
    holds a different finite value of; ``distance(x)`` is the mean of the
    ``k`` nearest RMS distances to R over the distance features standardized
    by ``DescriptorScale.fit(R, ...)``, ``inf`` when ``x`` is not finite on
    one of them and 0 when there are none. Then ``p = (1 + #{c in C :
    (violations_c, distance_c) >= (violations_x, distance_x)}) / (|C| + 1)``:
    ties count against the query row, and ``p_min = 1 / (|C| + 1)``.

    **Profile**, over every remaining reference row of the shape: midrank
    percentiles, and the ``inverted_cdf`` order statistics
    ``v[max(1, ceil(n q))]`` at ``q = (1 - central) / 2``, ``1/2`` and
    ``(1 + central) / 2`` of the ``n`` valid values (exact rational levels;
    infinities allowed). ``expected_outside`` sums ``(j_low + n + 1 - j_high)
    / (n + 1)`` over the profiled features: the chance an exchangeable value
    falls outside the interval (an upper bound under ties).

    Contract: if the dataset and the reference worlds are independent draws
    from one generator, given the dataset's shape, and every argument above
    is fixed before looking at ``p``, then ``P(p <= alpha) <= alpha`` for
    every ``alpha`` — marginally over references (for one fixed reference the
    realized rate scatters around ``alpha`` by about
    ``sqrt(alpha * (1 - alpha) / (|C| + 2))``); the p-values of a panel
    scored against one reference are dependent. One row per calibration cell
    makes the calibration worlds independent. A small ``p`` says the dataset
    is unusual for this reference; a large ``p`` says these features do not
    tell them apart. Raw levels only. The reference and the dataset must not
    share generation seeds (:func:`matched_prior` takes its own).

    **Cost.** Every call scores each calibration world against the ruler,
    ``O(|C| |R| F)`` for ``F`` selected features, once per shape; a
    self-query repeats that for every excluded ruler group. Measured with the
    default features: one external query took about 2 s against 10,000
    reference rows and 28 s against 40,000; a self-query about 15 s at 1,000
    rows and 100 s at 2,000. Score a panel in one call, so it shares the
    calibration scores. For a very large corpus, subsample whole generation
    groups before scoring (``WorldDescriptors.take``): groups are
    exchangeable, so the guarantee is kept.
    """
    _require_descriptors(reference, "reference")
    _require_descriptors(query, "query")
    unknown = np.flatnonzero(reference.group_ids < 0)
    if unknown.size:
        raise ValueError(
            f"{unknown.size} reference rows have no known generation group (group_id -1, "
            f"first {_identity(reference, int(unknown[0]))}); ruler and calibration worlds "
            "are split by group, so describe a corpus (its cell_id is the group) or pass "
            "world_descriptors(..., group_ids=...)"
        )
    names = _selected_features(reference, query, features)
    k = _positive_int(k, "k")
    calibration_fraction = _open_unit(calibration_fraction, "calibration_fraction")
    central = _open_unit(central, "central")
    seed = _seed(seed)
    _check_query_rows(query, names)
    shared = _shared_observable(reference, query)
    shared_reference = np.asarray([reference.feature_index(n) for n in shared], dtype=np.int64)
    shared_query = np.asarray([query.feature_index(n) for n in shared], dtype=np.int64)
    own = _reference_rows(reference, query, shared, shared_reference, shared_query)
    # Copies and duplicates are judged on the features that can tell worlds apart.
    evidence = [j for j, name in enumerate(shared) if _tells_apart(reference.definition(name))]
    evidence_reference, evidence_query = shared_reference[evidence], shared_query[evidence]

    group_codes = _group_codes(reference)
    excluded = np.full(query.n_worlds, -1, dtype=np.int64)
    excluded[own >= 0] = group_codes[own[own >= 0]]
    query_shapes = [
        (n_treat, n_cov, n_time)
        for n_treat, n_cov, n_time in np.stack(
            [query.metadata[c] for c in METADATA_COLUMNS], axis=1
        ).tolist()
    ]
    copy_columns = np.asarray(
        [j for j, d in enumerate(reference.definitions) if _tells_apart(d)], dtype=np.int64
    )
    by_shape = _strata(reference)
    strata: dict[tuple[int, int, int], _Stratum] = {}
    for shape in dict.fromkeys(query_shapes):
        if shape in by_shape:
            _refuse_copies(reference, by_shape[shape], copy_columns, group_codes)
            strata[shape] = _split(
                reference, by_shape[shape], group_codes, seed, calibration_fraction
            )

    n_query, n_features = query.n_worlds, len(names)
    columns = np.asarray([reference.feature_index(n) for n in names], dtype=np.int64)
    query_columns = np.asarray([query.feature_index(n) for n in names], dtype=np.int64)
    definitions = tuple(reference.definition(n) for n in names)
    value = query.values[:, query_columns]
    value_status = query.status[:, query_columns]
    status = np.full(n_query, _NO_REFERENCE, dtype=np.uint8)
    p_value = np.full(n_query, np.nan)
    p_min = np.full(n_query, np.nan)
    violations = np.full(n_query, np.nan)
    distance = np.full(n_query, np.nan)
    n_ruler = np.zeros(n_query, dtype=np.int64)
    n_calibration = np.zeros(n_query, dtype=np.int64)
    n_reference = np.zeros(n_query, dtype=np.int64)
    percentile = np.full((n_query, n_features), np.nan)
    low = np.full((n_query, n_features), np.nan)
    median = np.full((n_query, n_features), np.nan)
    high = np.full((n_query, n_features), np.nan)
    outside_chance = np.full((n_query, n_features), np.nan)
    inside = np.zeros((n_query, n_features), dtype=bool)
    violation = np.zeros((n_query, n_features), dtype=bool)
    role = np.full((n_query, n_features), _STATE, dtype=np.uint8)

    batches: dict[tuple[tuple[int, int, int], int], list[int]] = {}
    for i, shape in enumerate(query_shapes):
        batches.setdefault((shape, int(excluded[i])), []).append(i)
    level = Fraction(str(central))
    order_at: dict[int, tuple[int, int, int]] = {}  # exact order positions per sample size
    redundant_at: dict[tuple[int, int, int], np.ndarray] = {}
    rulers: dict[tuple[tuple[int, int, int], int], _Ruler] = {}
    for (shape, group), members in batches.items():
        rows = np.asarray(members, dtype=np.int64)
        if shape not in redundant_at:
            redundant_at[shape] = _structural_duplicates(definitions, shape[0], shape[1])
        redundant = redundant_at[shape]
        stratum = strata.get(shape)
        if stratum is None:
            role[rows] = np.where(redundant, _REDUNDANT, _STATE)
            continue
        # Calibration scores depend on the ruler only: removing a calibration
        # group leaves the ruler, and every other calibration score, unchanged.
        ruler_key = (shape, group if (stratum.ruler_groups == group).any() else -1)
        if ruler_key not in rulers:
            rulers[ruler_key] = _ruler(
                reference, stratum, ruler_key[1], columns, names, redundant, k
            )
        ruler = rulers[ruler_key]
        kept = stratum.calibration_groups != group
        population = stratum.rows[stratum.groups != group]
        role[rows] = ruler.role
        n_ruler[rows] = ruler.rows.size
        n_calibration[rows] = int(kept.sum())
        n_reference[rows] = population.size
        if population.size == 0:
            continue

        pop_values = reference.values[np.ix_(population, columns)]
        pop_valid = reference.status[np.ix_(population, columns)] == VALID
        # One sort per batch: NaN (every non-valid cell) sorts after +inf.
        ordered = np.sort(np.where(pop_valid, pop_values, np.nan), axis=0)
        n_valid = pop_valid.sum(axis=0)
        for j in range(n_features):
            ranked = ordered[: n_valid[j], j]
            x = value[rows, j]
            x_valid = value_status[rows, j] == VALID
            percentile[rows, j] = _midranks(ranked, x, x_valid, _NO_ROWS, _NO_VALUES)
            if ranked.size == 0:
                continue
            if ranked.size not in order_at:
                order_at[ranked.size] = _order_indices(ranked.size, level)
            j_low, j_median, j_high = order_at[ranked.size]
            low[rows, j] = ranked[j_low - 1]
            median[rows, j] = ranked[j_median - 1]
            high[rows, j] = ranked[j_high - 1]
            inside[rows, j] = x_valid & (ranked[j_low - 1] <= x) & (x <= ranked[j_high - 1])
            outside_chance[rows, j] = (j_low + ranked.size + 1 - j_high) / (ranked.size + 1)

        duplicate = _duplicates(
            query.values[np.ix_(rows, evidence_query)],
            query.status[np.ix_(rows, evidence_query)],
            reference.values[np.ix_(population, evidence_reference)],
            reference.status[np.ix_(population, evidence_reference)],
        )
        too_few = ruler.rows.size < max(k, 2) or not kept.any()
        status[rows] = np.where(duplicate, _DUPLICATE, _TOO_FEW if too_few else _OK_ROW)
        scored = rows[status[rows] == _OK_ROW]
        if scored.size == 0:
            continue
        v_query, d_query, flags = _score(
            ruler,
            k,
            query.values[np.ix_(scored, query_columns)],
            query.status[np.ix_(scored, query_columns)],
            query.take(scored),
        )
        v_cal = ruler.calibration_violations[kept]
        d_cal = ruler.calibration_distance[kept]
        at_least = (v_cal[None, :] > v_query[:, None]) | (
            (v_cal[None, :] == v_query[:, None]) & (d_cal[None, :] >= d_query[:, None])
        )
        p_value[scored] = (1 + at_least.sum(axis=1)) / (v_cal.size + 1)
        p_min[scored] = 1 / (v_cal.size + 1)
        violations[scored] = v_query
        distance[scored] = d_query
        violation[scored] = flags

    counted = role != _REDUNDANT
    profiled = counted & ~np.isnan(percentile)
    return Typicality(
        features=names,
        k=k,
        calibration_fraction=calibration_fraction,
        central=central,
        seed=seed,
        source_ids=query.source_ids,
        world_ids=query.world_ids,
        group_ids=query.group_ids,
        status=status,
        p_value=p_value,
        p_min=p_min,
        violations=violations,
        distance=distance,
        n_ruler=n_ruler,
        n_calibration=n_calibration,
        n_reference=n_reference,
        n_distance_features=(role == _DISTANCE).sum(axis=1).astype(np.int64),
        n_constant_features=(role == _CONSTANT).sum(axis=1).astype(np.int64),
        n_profiled=profiled.sum(axis=1).astype(np.int64),
        n_inside=(counted & inside).sum(axis=1).astype(np.int64),
        # An exactly rounded sum: the same whatever other features are selected.
        expected_outside=np.asarray(
            [math.fsum(chance[keep]) for chance, keep in zip(outside_chance, profiled)],
            dtype=np.float64,
        ),
        value=value,
        value_status=value_status,
        percentile=percentile,
        reference_low=low,
        reference_median=median,
        reference_high=high,
        inside=inside,
        violation=violation,
        role=role,
    )


def _dataset_shape(dataset: ObservedWorlds | WorldDescriptors) -> tuple[int, int, int]:
    """The one ``(n_treatments_active, n_covariates_active, n_time_steps)`` of every world."""
    if isinstance(dataset, ObservedWorlds):
        shapes = np.stack(
            [
                dataset.treatment_active_mask.sum(axis=1),
                dataset.covariate_active_mask.sum(axis=1),
                np.full(dataset.n_worlds, dataset.n_time_steps),
            ],
            axis=1,
        )
    elif isinstance(dataset, WorldDescriptors):
        if dataset.n_worlds == 0:
            raise ValueError("dataset has no rows; describe at least one world")
        shapes = np.stack([dataset.metadata[c] for c in METADATA_COLUMNS], axis=1)
    else:
        raise TypeError(
            f"dataset must be ObservedWorlds or WorldDescriptors, got {type(dataset).__name__}"
        )
    distinct = sorted({(int(a), int(b), int(c)) for a, b, c in shapes.tolist()})
    if len(distinct) > 1:
        raise ValueError(
            "dataset worlds differ in (n_treatments_active, n_covariates_active, n_time_steps): "
            f"{distinct}; match one shape at a time"
        )
    return distinct[0]


def matched_prior(
    prior: SCMPrior,
    dataset: ObservedWorlds | WorldDescriptors,
    *,
    n_cells: int,
    seed: int,
    draws_per_cell: int = 1,
) -> SCMPrior:
    """The generator ``prior`` restricted to the dataset's shape, for a typicality reference.

    Every generated cell then has the dataset's active treatment and covariate
    counts; everything else is drawn as ``prior`` draws it given those counts
    (active counts are drawn first, so pinning them conditions on them).

    Parameters
    ----------
    prior
        The generator configuration to compare the dataset with, built at the
        dataset's horizon (``make_scm_prior(..., n_time_steps=T)``).
    dataset
        :class:`~pymc_generator.descriptors.ObservedWorlds` (active counts from
        its masks) or :class:`~pymc_generator.descriptors.WorldDescriptors`
        (from its metadata); every world must share one shape.
    n_cells, draws_per_cell
        Size of the reference. The number of cells sets the calibration size
        of :func:`typicality` (one world per calibration cell), its smallest
        p-value and the generation time. Extra draws per cell are nearly free
        to generate; they join the ruler (ruler cells) or only the profile
        and the duplicate check (calibration cells), and can add power
        against subtle differences, at a scoring cost that grows with the
        ruler. An ``ok`` row needs at least ``max(k, 2)`` ruler worlds left
        after the ``round(calibration_fraction * n_cells)`` calibration cells:
        at the :func:`typicality` defaults (``k=5``,
        ``calibration_fraction=0.5``) and ``draws_per_cell=1`` that is
        ``n_cells >= 9``.
    seed
        Generation seed of the reference. Required, and it must differ from
        the dataset's generation seed: a same-seed reference can contain the
        dataset itself, which :func:`typicality` then reports as a
        ``duplicate``.

    Returns
    -------
    SCMPrior
        ``prior`` with ``n_treatments_active_range=(K, K)``,
        ``n_covariates_active_range=(M, M)``, ``active_count_weights=None``
        and the given ``n_cells``, ``draws_per_cell`` and ``seed``, validated.
        Latent ranges are untouched.

    Raises
    ------
    ValueError
        When the horizons differ, when the dataset's counts lie outside the
        prior's active-count grid, or when stratified allocation gives the
        dataset's counts zero weight (independent allocation ignores weights).
    """
    from .sampler import SCMPrior

    if not isinstance(prior, SCMPrior):
        raise TypeError(
            f"prior must be an SCMPrior (see make_scm_prior), got {type(prior).__name__}"
        )
    n_treat, n_cov, n_time = _dataset_shape(dataset)
    if prior.n_time_steps != n_time:
        raise ValueError(
            f"prior.n_time_steps is {prior.n_time_steps} but the dataset has {n_time} time "
            "steps; build the prior at the dataset's horizon with make_scm_prior(..., "
            f"n_time_steps={n_time}) (the minimum horizon is 4; trajectory presets may need more)"
        )
    treatment_counts, covariate_counts = prior.active_count_grid
    if n_treat not in treatment_counts or n_cov not in covariate_counts:
        raise ValueError(
            f"the dataset has {n_treat} active treatments and {n_cov} active covariates, outside "
            f"the prior's active-count grid (n_treatments_active {list(treatment_counts)} of "
            f"n_treatments={prior.n_treatments} slots, n_covariates_active "
            f"{list(covariate_counts)} of n_covariates={prior.n_covariates} slots); build the "
            "prior with slots and active ranges that include them"
        )
    if prior.active_count_allocation == "stratified":
        weights = prior.active_count_weight_matrix()
        if weights[treatment_counts.index(n_treat), covariate_counts.index(n_cov)] == 0:
            raise ValueError(
                "stratified active_count_weights give the dataset's counts "
                f"(n_treatments_active={n_treat}, n_covariates_active={n_cov}) zero weight, so "
                "the prior never generates that shape; give it a positive weight"
            )
    matched = replace(
        prior,
        n_treatments_active_range=(n_treat, n_treat),
        n_covariates_active_range=(n_cov, n_cov),
        active_count_weights=None,
        n_cells=n_cells,
        draws_per_cell=draws_per_cell,
        seed=seed,
    )
    matched.validate()
    return matched
