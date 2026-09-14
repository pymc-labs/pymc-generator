"""Generated-data diagnostics: interrogate a corpus after it was generated.

``outcome_distributions`` answers "how big is each quantity"; this module
answers the questions that need the series themselves and the relations
BETWEEN them:

* **shape/spread** of every node series (observed C/Z/Y, retained latent
  truth D/B, and — on request — the decomposition and single-world
  counterfactual series),
* **dependence** between every pair: signed Pearson, Spearman, directional
  Chatterjee xi (general dependence, linear or not) and its symmetric max,
* **collinearity**: per-predictor VIF over the observed design (active C+Z)
  and over the oracle design (active C+Z+D),
* **dynamics**: autocorrelation, forward lag-xi, roughness and a robust
  spike ratio,
* **contributions to sales**: an explicit hierarchy of net/gross totals,
  per-period means and shares, with closure residuals.

Everything here is strictly post-hoc over retained arrays. Nothing in this
module runs during graph construction, sampling, corpus finalization,
persistence or migration, so generator RNG streams, corpus bytes and the
schema are untouched by importing or calling it.

Three rules shape every number reported here:

1. **World first.** Every statistic is computed inside one world and one
   view, then summarized across worlds. Pooling rows from different worlds
   manufactures dependence out of between-world level differences.
2. **Levels and differences.** Weekly series are random-walk-like, so level
   correlations are large even for independent series. Both views are
   reported side by side; differencing is a companion view, not a proof of
   stationarity.
3. **Description, not inference.** No p-values, no confidence bands, no
   significance, no causal or forecast claims. Overlapping lag pairs and
   correlated worlds make iid inference invalid here, and the latent truth
   (D, B) is retained ground truth, never a model input.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from typing import TYPE_CHECKING, Any, Literal

import numpy as np

from .outcomes import (
    DEFAULT_QUANTILES,
    _fmt,
    _q_key,
    _position_indices,
    _validate_quantiles,
    outcome_distributions,
)
from .signal_diagnostics import _hf_ratio

if TYPE_CHECKING:  # pragma: no cover - typing only
    import pandas as pd

    from .outcomes import OutcomeDistributions
    from .worlds import SCM

__all__ = [
    "CONTRIBUTION_BASES",
    "CONTRIBUTION_FIELDS",
    "DEFAULT_MAX_LAG",
    "DEPENDENCE_METRICS",
    "SCOPES",
    "SERIES_SLOTS",
    "VIEWS",
    "VIF_SCOPES",
    "ContributionBudget",
    "ContributionClosure",
    "ContributionDescriptor",
    "CoverageLedger",
    "DataDiagnostics",
    "DependenceDiagnostics",
    "DiagnosticView",
    "SeriesDescriptor",
    "SeriesDiagnostics",
    "TemporalDiagnostics",
    "VIFDiagnostics",
    "data_diagnostics",
]

Scope = Literal["nodes", "decomposition", "counterfactual"]
View = Literal["levels", "differences"]

#: Longest lag on the default axis (one year of weekly data).
DEFAULT_MAX_LAG = 52

#: Series scopes, in report order. ``nodes`` is always computed.
SCOPES: tuple[str, ...] = ("nodes", "decomposition", "counterfactual")

#: The two views every series is reported in.
VIEWS: tuple[str, ...] = ("levels", "differences")

#: The eight per-world descriptive slots, in stored order.
SERIES_SLOTS: tuple[str, ...] = (
    "mean",
    "std",
    "median",
    "iqr",
    "min",
    "max",
    "roughness",
    "spike",
)

#: Pairwise dependence metrics, in report order.
DEPENDENCE_METRICS: tuple[str, ...] = ("pearson", "spearman", "xi", "xi_max")

#: Design scopes VIF is reported for.
VIF_SCOPES: tuple[str, ...] = ("observed", "oracle")

#: The six contribution measures, in report order.
CONTRIBUTION_FIELDS: tuple[str, ...] = (
    "net_total",
    "net_mean_per_period",
    "net_share",
    "gross_total",
    "gross_mean_per_period",
    "gross_over_net_sales",
)

#: Contribution hierarchies. The two bases are alternative readings of the
#: same media effect and are never additive alongside each other.
CONTRIBUTION_BASES: tuple[str, ...] = ("base_direct_plus_indirect", "observed_path_media")

_MEASURE_FIELD = {
    ("net", "total"): "net_total",
    ("net", "mean_per_period"): "net_mean_per_period",
    ("net", "share"): "net_share",
    ("gross", "total"): "gross_total",
    ("gross", "mean_per_period"): "gross_mean_per_period",
    ("gross", "share"): "gross_over_net_sales",
}

# Minimum observations a reported dependence / lag statistic is computed
# from. This is a conservative REPORT-VALIDITY policy, not the mathematical
# definition boundary of the coefficients: two points always correlate
# perfectly, and a single lag pair carries no usable spread.
_MIN_PAIRS = 3

# Canonical internal array names -> corpus / SCM keys.
_CORPUS_ARRAYS: dict[str, str] = {
    "spend": "spend_raw",
    "controls": "controls",
    "demand": "demand",
    "baseline": "baseline_raw",
    "sales": "sales_raw",
    "baseline_intrinsic": "baseline_intrinsic",
    "sales_noise": "sales_noise",
    "control_contribution": "control_contribution",
    "confounder_contribution": "confounder_contribution",
    "channel_contribution": "contributions_raw",
    "indirect_by_source": "indirect_effects_by_source",
    "indirect_effects": "indirect_effects",
}

_WORLD_ARRAYS: dict[str, str] = {
    "spend": "channels",
    "controls": "controls",
    "demand": "demand",
    "baseline": "baseline",
    "sales": "sales",
    "baseline_intrinsic": "baseline_intrinsic",
    "sales_noise": "sales_noise",
    "control_contribution": "control_contribution",
    "confounder_contribution": "confounder_contribution",
    "channel_contribution": "contributions",
    "indirect_by_source": "indirect_effects_by_source",
    "indirect_effects": "indirect_effects",
}

# SCM-only audit paths. ``channels_base`` / ``contributions_observed`` are
# always drawn; the unshocked paths exist only when shocks were enabled.
_WORLD_COUNTERFACTUAL: dict[str, str] = {
    "channels_base": "channels_base",
    "contributions_observed": "contributions_observed",
    "channels_unshocked": "channels_unshocked",
    "sales_unshocked": "sales_unshocked",
}

# Column kind of every canonical array ("" = one series per world).
_ARRAY_KIND: dict[str, str] = {
    "spend": "channel",
    "controls": "control",
    "demand": "latent",
    "baseline": "",
    "sales": "",
    "baseline_intrinsic": "",
    "sales_noise": "",
    "control_contribution": "control",
    "confounder_contribution": "latent",
    "channel_contribution": "channel",
    "indirect_by_source": "source",
    "indirect_effects": "",
    "channels_base": "channel",
    "contributions_observed": "channel",
    "channels_unshocked": "channel",
    "sales_unshocked": "",
}

_MASK_KEYS: dict[str, str] = {
    "channel": "treatment_active_mask",
    "control": "covariate_active_mask",
    "latent": "latent_active_mask",
}

_WIDTH_KINDS: tuple[str, ...] = ("channel", "control", "latent")


# ---------------------------------------------------------------------------
# Series vocabulary
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SeriesDescriptor:
    """One diagnosable series: what it is and where it comes from.

    Attributes
    ----------
    key
        Globally unique ASCII token (``[A-Za-z][A-Za-z0-9_]*``). Keys are
        derived from the FULL source before any world selection, so they do
        not move when the analysed set shrinks, and they are the only
        selector accepted by raw access and plots — display labels are not
        unique across scopes (``C1``, ``C1_direct_y``, ``C1_base`` all read
        as "channel 1" to a human).
    display_label
        Human wording for tables and plots.
    scope
        ``nodes`` | ``decomposition`` | ``counterfactual``.
    role
        ``observed`` (a model could see it), ``latent`` (retained ground
        truth) or ``derived`` (a decomposition/audit series).
    source
        Canonical source array name.
    column_kind, column_index
        Trailing-axis kind ("" for a per-world series) and its index, used
        to resolve the per-world active mask.
    available
        False when the source never carried this series (an optional SCM
        audit path): the key stays known and selectable, with zero eligible
        worlds, instead of silently reading as zero.
    vif_scopes
        Design scopes this series is a predictor in.
    """

    key: str
    display_label: str
    scope: str
    role: str
    source: str
    column_kind: str
    column_index: int
    available: bool
    vif_scopes: tuple[str, ...]


def _node_descriptors(widths: Mapping[str, int]) -> list[SeriesDescriptor]:
    out: list[SeriesDescriptor] = []
    for k in range(widths["channel"]):
        out.append(
            SeriesDescriptor(
                key=f"C{k + 1}",
                display_label=f"C{k + 1} spend",
                scope="nodes",
                role="observed",
                source="spend",
                column_kind="channel",
                column_index=k,
                available=True,
                vif_scopes=("observed", "oracle"),
            )
        )
    for m in range(widths["control"]):
        out.append(
            SeriesDescriptor(
                key=f"Z{m + 1}",
                display_label=f"Z{m + 1} control",
                scope="nodes",
                role="observed",
                source="controls",
                column_kind="control",
                column_index=m,
                available=True,
                vif_scopes=("observed", "oracle"),
            )
        )
    for j in range(widths["latent"]):
        out.append(
            SeriesDescriptor(
                key=f"D{j + 1}",
                display_label=f"D{j + 1} latent demand",
                scope="nodes",
                role="latent",
                source="demand",
                column_kind="latent",
                column_index=j,
                available=True,
                vif_scopes=("oracle",),
            )
        )
    out.append(
        SeriesDescriptor(
            key="B",
            display_label="B retained baseline",
            scope="nodes",
            role="latent",
            source="baseline",
            column_kind="",
            column_index=-1,
            available=True,
            vif_scopes=(),
        )
    )
    out.append(
        SeriesDescriptor(
            key="Y",
            display_label="Y sales",
            scope="nodes",
            role="observed",
            source="sales",
            column_kind="",
            column_index=-1,
            available=True,
            vif_scopes=(),
        )
    )
    return out


def _decomposition_descriptors(widths: Mapping[str, int]) -> list[SeriesDescriptor]:
    out = [
        SeriesDescriptor(
            key="B_intrinsic",
            display_label="B intrinsic walk",
            scope="decomposition",
            role="latent",
            source="baseline_intrinsic",
            column_kind="",
            column_index=-1,
            available=True,
            vif_scopes=(),
        ),
        SeriesDescriptor(
            key="Y_noise",
            display_label="Y observation noise",
            scope="decomposition",
            role="latent",
            source="sales_noise",
            column_kind="",
            column_index=-1,
            available=True,
            vif_scopes=(),
        ),
    ]
    for m in range(widths["control"]):
        out.append(
            SeriesDescriptor(
                key=f"Z{m + 1}_baseline_alloc",
                display_label=f"Z{m + 1} retained baseline allocation",
                scope="decomposition",
                role="derived",
                source="control_contribution",
                column_kind="control",
                column_index=m,
                available=True,
                vif_scopes=(),
            )
        )
    for j in range(widths["latent"]):
        out.append(
            SeriesDescriptor(
                key=f"D{j + 1}_baseline_alloc",
                display_label=f"D{j + 1} retained baseline allocation",
                scope="decomposition",
                role="derived",
                source="confounder_contribution",
                column_kind="latent",
                column_index=j,
                available=True,
                vif_scopes=(),
            )
        )
    for k in range(widths["channel"]):
        out.append(
            SeriesDescriptor(
                key=f"C{k + 1}_direct_y",
                display_label=f"C{k + 1} direct C->Y contribution",
                scope="decomposition",
                role="derived",
                source="channel_contribution",
                column_kind="channel",
                column_index=k,
                available=True,
                vif_scopes=(),
            )
        )
    for index, label in enumerate(("cc", "zc", "dc")):
        out.append(
            SeriesDescriptor(
                key=f"indirect_{label}",
                display_label=f"indirect media effect via {label} edges",
                scope="decomposition",
                role="derived",
                source="indirect_by_source",
                column_kind="source",
                column_index=index,
                available=True,
                vif_scopes=(),
            )
        )
    out.append(
        SeriesDescriptor(
            key="indirect_total",
            display_label="total indirect media effect",
            scope="decomposition",
            role="derived",
            source="indirect_effects",
            column_kind="",
            column_index=-1,
            available=True,
            vif_scopes=(),
        )
    )
    out.append(
        SeriesDescriptor(
            key="media_total",
            display_label="total media effect (direct + indirect)",
            scope="decomposition",
            role="derived",
            source="media_total",
            column_kind="",
            column_index=-1,
            available=True,
            vif_scopes=(),
        )
    )
    return out


def _counterfactual_descriptors(
    widths: Mapping[str, int], available: frozenset[str]
) -> list[SeriesDescriptor]:
    out: list[SeriesDescriptor] = []
    for k in range(widths["channel"]):
        out.append(
            SeriesDescriptor(
                key=f"C{k + 1}_base",
                display_label=f"C{k + 1} base spend (no C/Z/D inputs)",
                scope="counterfactual",
                role="derived",
                source="channels_base",
                column_kind="channel",
                column_index=k,
                available="channels_base" in available,
                vif_scopes=(),
            )
        )
    for k in range(widths["channel"]):
        out.append(
            SeriesDescriptor(
                key=f"C{k + 1}_observed_y",
                display_label=f"C{k + 1} observed-path contribution",
                scope="counterfactual",
                role="derived",
                source="contributions_observed",
                column_kind="channel",
                column_index=k,
                available="contributions_observed" in available,
                vif_scopes=(),
            )
        )
    for k in range(widths["channel"]):
        out.append(
            SeriesDescriptor(
                key=f"C{k + 1}_unshocked",
                display_label=f"C{k + 1} spend without the shock schedule",
                scope="counterfactual",
                role="derived",
                source="channels_unshocked",
                column_kind="channel",
                column_index=k,
                available="channels_unshocked" in available,
                vif_scopes=(),
            )
        )
    out.append(
        SeriesDescriptor(
            key="Y_unshocked",
            display_label="Y sales without the shock schedule",
            scope="counterfactual",
            role="derived",
            source="sales_unshocked",
            column_kind="",
            column_index=-1,
            available="sales_unshocked" in available,
            vif_scopes=(),
        )
    )
    return out


# ---------------------------------------------------------------------------
# Selector validation
# ---------------------------------------------------------------------------


def _validate_world_selector(worlds: Any, n_available: int) -> np.ndarray:
    """Normalize a world selector into an int64 position array.

    Stricter than the outcome-space selector on purpose: a diagnostics report
    keys every world-axis array by position, so a duplicated or lossy
    selection would silently double-count a world in every quantile.
    """
    if n_available <= 0:
        raise ValueError("source has no worlds to diagnose")
    if worlds is None:
        return np.arange(n_available, dtype=np.int64)
    if isinstance(worlds, slice):
        idx = np.arange(n_available, dtype=np.int64)[worlds]
    elif isinstance(worlds, (bool, np.bool_)):
        raise TypeError(
            "worlds must be None, a slice, an integer position, an integer "
            "sequence or a boolean mask; a bare bool selects nothing"
        )
    elif np.ma.isMaskedArray(worlds):
        # np.asarray hands back .data, so the mask — the whole point of the
        # array — would silently select the worlds it excludes.
        raise TypeError(
            "masked world selectors are rejected because the mask is lost on "
            "conversion; pass worlds=selector.compressed() or "
            "np.flatnonzero(~selector.mask & selector.data.astype(bool))"
        )
    else:
        idx = _position_indices(worlds, n_available, "world")
        unique, counts = np.unique(idx, return_counts=True)
        if unique.size != idx.size:
            repeated = sorted(int(v) for v in unique[counts > 1])
            raise ValueError(
                f"world selection repeats {repeated}; a repeated world would be "
                "counted twice in every across-world summary"
            )
    if idx.size == 0:
        raise ValueError("world selection is empty")
    return np.asarray(idx, dtype=np.int64)


def _validate_names(
    values: Any, allowed: Sequence[str], what: str, *, required: Sequence[str] = ()
) -> tuple[str, ...]:
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
    missing = [r for r in required if r not in names]
    if missing:
        raise ValueError(f"{what} must include {missing}")
    return names




def _validate_lags(lags: Any, n_time_steps: int) -> tuple[int, ...]:
    if lags is None:
        return tuple(range(1, min(DEFAULT_MAX_LAG, n_time_steps // 2) + 1))
    if isinstance(lags, str) or not isinstance(lags, Sequence):
        raise TypeError("lags must be None or a non-string sequence of positive integers")
    out: list[int] = []
    for lag in lags:
        if isinstance(lag, (bool, np.bool_)) or not isinstance(lag, (int, np.integer)):
            raise TypeError(f"lags must be plain integers, got {lag!r}")
        if int(lag) < 1:
            raise ValueError(f"lags must be >= 1, got {int(lag)}")
        out.append(int(lag))
    if not out:
        raise ValueError("lags is empty; pass None for the default axis")
    if len(set(out)) != len(out):
        raise ValueError(f"lags repeats an entry: {out}")
    return tuple(out)


def _validate_keys(keys: Any, known: Sequence[str], what: str = "keys") -> tuple[str, ...]:
    if isinstance(keys, str):
        raise TypeError(f"{what} must be a sequence of keys, not a bare string {keys!r}")
    if not isinstance(keys, Sequence):
        raise TypeError(f"{what} must be a non-string sequence, got {type(keys).__name__}")
    chosen = tuple(str(k) for k in keys)
    if not chosen:
        raise ValueError(f"{what} is empty; pass None for every key")
    if len(set(chosen)) != len(chosen):
        raise ValueError(f"{what} repeats an entry: {list(chosen)}")
    unknown = [k for k in chosen if k not in known]
    if unknown:
        lookalike = {k: [n for n in known if n.lower() == k.lower()] for k in unknown}
        hints = {k: v for k, v in lookalike.items() if v}
        suffix = f"; did you mean {hints}" if hints else ""
        raise KeyError(f"unknown {what} {unknown}{suffix}")
    return chosen


# ---------------------------------------------------------------------------
# Source extraction + current-schema preflight
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class _DiagnosticSource:
    """Float64 arrays for the analysed worlds plus the full-source universe."""

    kind: str
    world_ids: np.ndarray
    n_worlds_total: int
    n_time_steps: int
    widths: dict[str, int]
    masks: dict[str, np.ndarray]
    arrays: dict[str, np.ndarray]
    available: frozenset[str]


def _numeric_float64(value: Any, name: str) -> np.ndarray:
    arr = np.asarray(value)
    if arr.dtype.kind not in "fiu":
        raise TypeError(f"{name} must be numeric, got dtype {arr.dtype}")
    return arr.astype(np.float64, copy=False)


def _require_finite(arr: np.ndarray, name: str) -> np.ndarray:
    if not np.isfinite(arr).all():
        raise ValueError(f"{name} holds non-finite values; diagnostics refuse to guess their scale")
    return arr


def _as_float64(value: Any, name: str) -> np.ndarray:
    return _require_finite(_numeric_float64(value, name), name)


def _check_zero_padding(arr: np.ndarray, mask: np.ndarray, name: str) -> None:
    """Inactive padded columns must be exactly zero, not merely small.

    ``outcome_distributions`` derives ``media_contribution`` by summing the
    padded channel block BEFORE masking, so a single NaN or junk value in a
    padded slot silently poisons a reported total. Reject it here, pointing
    at the exact world and column, instead of sanitizing a corpus the rest
    of the library treats as valid.
    """
    inactive = ~mask
    if not inactive.any():
        return
    block = arr.transpose(0, 2, 1)[inactive]
    bad = ~np.all(block == 0.0, axis=1)
    if bad.any():
        worlds, columns = np.nonzero(inactive)
        first = int(np.flatnonzero(bad)[0])
        raise ValueError(
            f"{name} has non-zero inactive padding at world {int(worlds[first])}, "
            f"column {int(columns[first])}: diagnostic-relevant padded slots must be "
            "exact zero under the current schema"
        )


def _corpus_source(
    corpus: Mapping[str, Any], worlds: Any, *, need_counterfactual: bool
) -> _DiagnosticSource:
    if need_counterfactual:
        raise ValueError(
            "scope 'counterfactual' needs the single-world audit paths "
            "(channels_base / contributions_observed), which a corpus does not retain; "
            "pass a sequence of SCM worlds from sample_scm"
        )
    required = [*_CORPUS_ARRAYS.values(), *_MASK_KEYS.values()]
    missing = sorted({key for key in required if key not in corpus})
    if missing:
        raise KeyError(f"corpus is missing keys required for data diagnostics: {missing}")

    sales = np.asarray(corpus["sales_raw"])
    if sales.ndim != 2:
        raise ValueError(f"sales_raw must be (n_tasks, n_time_steps), got shape {sales.shape}")
    n_total, n_time_steps = (int(sales.shape[0]), int(sales.shape[1]))
    if n_total == 0 or n_time_steps == 0:
        raise ValueError(
            f"corpus has no data to diagnose: {n_total} worlds x {n_time_steps} time steps"
        )

    masks: dict[str, np.ndarray] = {}
    widths: dict[str, int] = {}
    for kind, key in _MASK_KEYS.items():
        raw = np.asarray(corpus[key])
        if raw.ndim != 2 or raw.shape[0] != n_total:
            raise ValueError(f"{key} must be (n_tasks, width), got shape {raw.shape}")
        if raw.dtype.kind not in "buif" or not np.all((raw == 0) | (raw == 1)):
            raise ValueError(f"{key} must be a binary 0/1 mask")
        masks[kind] = raw.astype(bool)
        widths[kind] = int(raw.shape[1])

    idx = _validate_world_selector(worlds, n_total)

    arrays: dict[str, np.ndarray] = {}
    for name, key in _CORPUS_ARRAYS.items():
        # Padding is checked BEFORE finiteness so a NaN parked in an inactive
        # slot is named by world and column, not by the whole array.
        arr = _numeric_float64(corpus[key], key)
        kind = _ARRAY_KIND[name]
        if kind == "":
            expected: tuple[int, ...] = (n_total, n_time_steps)
        elif kind == "source":
            expected = (n_total, n_time_steps, 3)
        else:
            expected = (n_total, n_time_steps, widths[kind])
        if arr.shape != expected:
            raise ValueError(f"{key} must have shape {expected}, got {arr.shape}")
        if kind in _WIDTH_KINDS:
            _check_zero_padding(arr, masks[kind], key)
        _require_finite(arr, key)
        arrays[name] = arr[idx]

    return _DiagnosticSource(
        kind="corpus",
        world_ids=idx,
        n_worlds_total=n_total,
        n_time_steps=n_time_steps,
        widths=widths,
        masks={kind: mask[idx] for kind, mask in masks.items()},
        arrays=arrays,
        available=frozenset(_CORPUS_ARRAYS),
    )


def _worlds_source(
    scms: Sequence[SCM], worlds: Any, *, need_counterfactual: bool
) -> _DiagnosticSource:
    n_total = len(scms)
    if n_total == 0:
        raise ValueError("no worlds to diagnose")
    horizons = {int(np.asarray(w.data["sales"]).shape[0]) for w in scms}
    if len(horizons) > 1:
        raise ValueError(f"worlds must share n_time_steps to be pooled; got {sorted(horizons)}")
    n_time_steps = horizons.pop()
    if n_time_steps == 0:
        raise ValueError("worlds have no time steps to diagnose")

    widths = {
        "channel": max(int(w.n_treatments) for w in scms),
        "control": max(int(w.n_covariates) for w in scms),
        "latent": max(int(w.n_latent) for w in scms),
    }
    per_world = {
        "channel": np.array([int(w.n_treatments) for w in scms]),
        "control": np.array([int(w.n_covariates) for w in scms]),
        "latent": np.array([int(w.n_latent) for w in scms]),
    }
    # The optional audit paths are a property of the whole sequence: a key
    # present in some worlds and missing in others is ambiguous, not optional.
    optional = {}
    for name, key in _WORLD_COUNTERFACTUAL.items():
        present = [key in w.data for w in scms]
        if all(present):
            optional[name] = key
        elif any(present):
            raise ValueError(
                f"SCM key {key!r} is present in some worlds and missing in others; "
                "diagnose a homogeneous sequence"
            )
    if need_counterfactual:
        for name in ("channels_base", "contributions_observed"):
            if name not in optional:
                raise ValueError(
                    f"scope 'counterfactual' needs {_WORLD_COUNTERFACTUAL[name]!r} in every world"
                )

    idx = _validate_world_selector(worlds, n_total)
    chosen = [scms[int(i)] for i in idx]
    n_worlds = int(idx.size)

    arrays: dict[str, np.ndarray] = {}
    for name, key in {**_WORLD_ARRAYS, **optional}.items():
        kind = _ARRAY_KIND[name]
        if kind == "":
            block = np.stack(
                [_as_float64(w.data[key], f"{key} (world {int(i)})") for w, i in zip(chosen, idx)]
            )
            if block.shape != (n_worlds, n_time_steps):
                raise ValueError(f"{key} must be (n_time_steps,) per world, got {block.shape[1:]}")
            arrays[name] = block
            continue
        width = 3 if kind == "source" else widths[kind]
        dense = np.zeros((n_worlds, n_time_steps, width), dtype=np.float64)
        for row, (world, i) in enumerate(zip(chosen, idx)):
            active = _as_float64(world.data[key], f"{key} (world {int(i)})")
            if active.ndim != 2 or active.shape[0] != n_time_steps:
                raise ValueError(
                    f"{key} must be (n_time_steps, width) in world {int(i)}, got {active.shape}"
                )
            if active.shape[1] > width:
                raise ValueError(
                    f"{key} in world {int(i)} is wider ({active.shape[1]}) than the sequence "
                    f"maximum ({width})"
                )
            dense[row, :, : active.shape[1]] = active
        arrays[name] = dense

    masks = {
        kind: (np.arange(widths[kind])[None, :] < per_world[kind][idx][:, None])
        for kind in _WIDTH_KINDS
    }
    return _DiagnosticSource(
        kind="worlds",
        world_ids=idx,
        n_worlds_total=n_total,
        n_time_steps=n_time_steps,
        widths=widths,
        masks=masks,
        arrays=arrays,
        available=frozenset(_WORLD_ARRAYS) | frozenset(optional),
    )


def _extract_diagnostic_source(
    source: Mapping[str, Any] | Sequence[SCM], worlds: Any, *, need_counterfactual: bool
) -> _DiagnosticSource:
    """Validate the source against the current schema and pull float64 rows.

    Never reuses ``QuantityDistribution.series``: those values are cast to
    float32, which collapses sub-1e-7 relative variation into a constant and
    would turn a real correlation into a division by zero.
    """
    if isinstance(source, Mapping):
        return _corpus_source(source, worlds, need_counterfactual=need_counterfactual)
    if isinstance(source, Sequence):
        if not source:
            raise ValueError("no worlds to diagnose")
        if not hasattr(source[0], "data"):
            raise TypeError("sequence source must contain SCM worlds")
        return _worlds_source(source, worlds, need_counterfactual=need_counterfactual)
    raise TypeError(f"source must be a corpus mapping or a sequence of SCM, got {type(source)}")


def _companion_selector(world_ids: np.ndarray, n_total: int) -> Any:
    """A selector ``outcome_distributions`` reads exactly as we do.

    The companion must land on the same worlds IN THE SAME ORDER, or every
    row-wise comparison against the report silently pairs the wrong worlds.
    Positions carry the order, but a 0/1-valued integer position array is
    rejected there as ambiguous (mask or positions?) — and that case is
    precisely a permutation of ``{0}`` or ``{0, 1}``, because duplicates are
    already rejected and the selection is in range. Both permutations are
    expressible as slices, which carry order without the ambiguity, so the
    two branches below cover every input positions cannot.
    """
    order = np.arange(n_total, dtype=np.int64)
    if np.array_equal(world_ids, order):
        return slice(None)
    if np.array_equal(world_ids, order[::-1]):
        return slice(None, None, -1)
    return world_ids.astype(np.int64, copy=True)


def _series_block(source: _DiagnosticSource, descriptors: Sequence[SeriesDescriptor]) -> np.ndarray:
    """``(n_worlds, n_series, n_time_steps)`` float64, zero where ineligible."""
    n_worlds = int(source.world_ids.size)
    out = np.zeros((n_worlds, len(descriptors), source.n_time_steps), dtype=np.float64)
    for p, desc in enumerate(descriptors):
        if not desc.available:
            continue
        if desc.source == "media_total":
            block = (
                source.arrays["channel_contribution"].sum(axis=2)
                + source.arrays["indirect_effects"]
            )
        else:
            block = source.arrays[desc.source]
        out[:, p, :] = block if desc.column_kind == "" else block[:, :, desc.column_index]
    return out


def _eligibility(source: _DiagnosticSource, descriptors: Sequence[SeriesDescriptor]) -> np.ndarray:
    n_worlds = int(source.world_ids.size)
    out = np.zeros((n_worlds, len(descriptors)), dtype=bool)
    for p, desc in enumerate(descriptors):
        if not desc.available:
            continue
        if desc.column_kind in _WIDTH_KINDS:
            out[:, p] = source.masks[desc.column_kind][:, desc.column_index]
        else:
            out[:, p] = True
    return out


# ---------------------------------------------------------------------------
# Coverage + finite-only aggregation
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class CoverageLedger:
    """How much data stands behind one reported cell.

    ``valid = finite + positive_infinite + negative_infinite`` and
    ``eligible = valid + invalid`` always hold, and ``selected >= eligible``.
    Infinities are counted, never quietly reclassified as missing: an
    infinite VIF or spike ratio is a real, reportable fact about the data,
    while ``np.quantile([1.0, inf], 0.5)`` is NaN and would be
    indistinguishable from "nothing to report".
    """

    selected: int
    eligible: int
    valid: int
    finite: int
    positive_infinite: int
    negative_infinite: int
    invalid: int
    observations: int = 0
    pairs: int = 0

    def as_dict(self) -> dict[str, int]:
        return {
            "selected": self.selected,
            "eligible": self.eligible,
            "valid": self.valid,
            "finite": self.finite,
            "positive_infinite": self.positive_infinite,
            "negative_infinite": self.negative_infinite,
            "invalid": self.invalid,
            "observations": self.observations,
            "pairs": self.pairs,
        }


def _ledger(
    values: np.ndarray,
    valid: np.ndarray,
    *,
    eligible: np.ndarray | None = None,
    selected: int | None = None,
    observations: int = 0,
    pairs: int = 0,
) -> CoverageLedger:
    valid = np.asarray(valid, dtype=bool)
    values = np.asarray(values, dtype=np.float64)
    eligible_mask = valid if eligible is None else np.asarray(eligible, dtype=bool)
    finite = int(np.count_nonzero(valid & np.isfinite(values)))
    pos_inf = int(np.count_nonzero(valid & np.isposinf(values)))
    neg_inf = int(np.count_nonzero(valid & np.isneginf(values)))
    n_valid = finite + pos_inf + neg_inf
    n_eligible = int(np.count_nonzero(eligible_mask))
    n_eligible = max(n_eligible, n_valid)
    return CoverageLedger(
        selected=max(int(valid.size if selected is None else selected), n_eligible),
        eligible=n_eligible,
        valid=n_valid,
        finite=finite,
        positive_infinite=pos_inf,
        negative_infinite=neg_inf,
        invalid=n_eligible - n_valid,
        observations=int(observations),
        pairs=int(pairs),
    )


def _finite_stats(
    values: np.ndarray, valid: np.ndarray, levels: Sequence[float]
) -> dict[str, float | None]:
    """mean/std/min/max/quantiles over the FINITE valid values only."""
    values = np.asarray(values, dtype=np.float64)
    valid = np.asarray(valid, dtype=bool)
    finite = values[valid & np.isfinite(values)]
    if finite.size == 0:
        out: dict[str, float | None] = {
            "n": 0,
            "mean": None,
            "std": None,
            "min": None,
            "max": None,
        }
        out.update({_q_key(level): None for level in levels})
        return out
    out = {
        "n": int(finite.size),
        "mean": float(finite.mean()),
        "std": float(finite.std()),
        "min": float(finite.min()),
        "max": float(finite.max()),
    }
    out.update(
        {_q_key(level): float(q) for level, q in zip(levels, np.quantile(finite, list(levels)))}
    )
    return out


def _finite_quantile(values: np.ndarray, valid: np.ndarray, level: float) -> float:
    finite = np.asarray(values, dtype=np.float64)[
        np.asarray(valid, dtype=bool) & np.isfinite(values)
    ]
    if finite.size == 0:
        return float("nan")
    return float(np.quantile(finite, level))


def _strict_json_value(value: Any) -> Any:
    """Recursively make a summary strictly JSON-safe.

    Non-finite floats become ``None`` only AFTER the ledgers have counted
    them, so ``json.dumps(..., allow_nan=False)`` succeeds without hiding an
    infinity behind a missing value.
    """
    if isinstance(value, Mapping):
        return {str(k): _strict_json_value(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_strict_json_value(v) for v in value]
    if isinstance(value, np.ndarray):
        return [_strict_json_value(v) for v in value.tolist()]
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if isinstance(value, (int, np.integer)):
        return int(value)
    if isinstance(value, (float, np.floating)):
        return float(value) if np.isfinite(value) else None
    if isinstance(value, CoverageLedger):
        return value.as_dict()
    return value


# ---------------------------------------------------------------------------
# Descriptive slots
# ---------------------------------------------------------------------------


def _compute_slots(block: np.ndarray, eligible: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """The eight per-world descriptive slots for ``(W, P, T_view)`` data."""
    n_worlds, n_series, n = block.shape
    shape = (n_worlds, n_series, len(SERIES_SLOTS))
    slots = np.full(shape, np.nan, dtype=np.float64)
    valid = np.zeros(shape, dtype=bool)
    if n == 0 or n_series == 0 or n_worlds == 0:
        return slots, valid
    rows = eligible & np.isfinite(block).all(axis=2)
    quartiles = np.quantile(block, (0.25, 0.5, 0.75), axis=2)
    slots[..., 0] = block.mean(axis=2)
    slots[..., 1] = block.std(axis=2)
    slots[..., 2] = quartiles[1]
    slots[..., 3] = quartiles[2] - quartiles[0]
    slots[..., 4] = block.min(axis=2)
    slots[..., 5] = block.max(axis=2)
    valid[..., :6] = rows[..., None]
    if n >= 2:
        slots[..., 6] = _hf_ratio(block)
        valid[..., 6] = rows
    if n >= 3:
        steps = np.diff(block, axis=2)
        centred = np.abs(steps - np.median(steps, axis=2, keepdims=True)).max(axis=2)
        step_q = np.quantile(steps, (0.25, 0.75), axis=2)
        spread = step_q[1] - step_q[0]
        with np.errstate(divide="ignore", invalid="ignore"):
            ratio = np.where(spread > 0.0, centred / np.where(spread > 0.0, spread, 1.0), np.inf)
        # No movement at all is a spike ratio of zero, not an infinity: a
        # perfectly linear ramp has identical steps and no outlying week.
        slots[..., 7] = np.where(centred > 0.0, ratio, 0.0)
        valid[..., 7] = rows
    slots[~valid] = np.nan
    return slots, valid


# ---------------------------------------------------------------------------
# Dependence
# ---------------------------------------------------------------------------


def _constant_rows(block: np.ndarray, centred: np.ndarray, norms: np.ndarray) -> np.ndarray:
    """Rows whose variation is at or below the rounding floor of their scale.

    Exact-zero variance is not the only degenerate case: a series that is
    constant up to one ULP unit-normalizes into an arbitrary rounding-noise
    direction, which would then be correlated and regressed against as if it
    were signal.
    """
    eps = float(np.finfo(np.float64).eps)
    scale = np.abs(block).max(axis=1) if block.size else np.zeros(block.shape[0])
    return np.asarray(norms <= eps * np.sqrt(centred.shape[1]) * scale, dtype=bool)


def _pearson_matrix(block: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Signed Pearson correlation between every pair of rows."""
    n_series = block.shape[0]
    centred = block - block.mean(axis=1, keepdims=True)
    norms = np.sqrt((centred**2).sum(axis=1))
    nonconstant = ~_constant_rows(block, centred, norms)
    scaled = centred / np.where(nonconstant, norms, 1.0)[:, None]
    # Unit vectors: a projection outside [-1, 1] is rounding, so clipping is
    # exact repair rather than the suppression of a real out-of-range value.
    corr = np.clip(scaled @ scaled.T, -1.0, 1.0)
    corr[~nonconstant, :] = np.nan
    corr[:, ~nonconstant] = np.nan
    if n_series:
        np.fill_diagonal(corr, np.nan)
    return corr, nonconstant


def _average_ranks(block: np.ndarray) -> np.ndarray:
    from scipy.stats import rankdata  # lazy: keep the module import numpy-light

    return np.asarray(rankdata(block, method="average", axis=-1), dtype=np.float64)


def _rank_counts(y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """``(#{j: y_j <= y_i}, #{j: y_j >= y_i})`` for every ``i``."""
    n = y.size
    ordered = np.sort(y)
    below = np.searchsorted(ordered, y, side="right").astype(np.float64)
    above = (n - np.searchsorted(ordered, y, side="left")).astype(np.float64)
    return below, above


def _mean_abs_cross(left: np.ndarray, right: np.ndarray) -> float:
    """Mean ``|a - b|`` over every cross pair of two sorted arrays."""
    cumulative = np.concatenate(([0.0], np.cumsum(left)))
    position = np.searchsorted(left, right, side="right")
    below_sum = cumulative[position]
    above_sum = cumulative[-1] - below_sum
    count = position.astype(np.float64)
    total = float(
        np.sum(count * right - below_sum) + np.sum(above_sum - (left.size - count) * right)
    )
    return total / float(left.size * right.size)


def _chatterjee_xi_tie_average(x: np.ndarray, y: np.ndarray) -> float:
    """Chatterjee's xi with an exact average over predictor-tie orderings.

    ``xi(X -> Y) = 1 - n * E[sum_i |r_{i+1} - r_i|] / (2 * sum_i l_i (n - l_i))``
    with the data ordered by X, ``r_i`` counting ``Y_j <= Y_i`` and ``l_i``
    counting ``Y_j >= Y_i``.

    Ties in X leave the order inside a tied block undefined, and the
    statistic is not invariant to how it is broken — the same data reordered
    within ties scores differently. Rather than inherit whichever order the
    sort produced, this averages exactly over every ordering a tie block
    admits: inside a block of size ``m`` the expected adjacent distance is
    ``(2/m) * sum_{a<b} |r_a - r_b|``, and the junction between two adjacent
    blocks contributes the mean cross-block distance. Both come from sorted
    ranks and prefix sums, so the cost stays ``O(n log n)`` and the answer is
    deterministic — no RNG, no input order dependence.

    Returns NaN for a constant target: the denominator is zero and the
    population 0/1 characterization does not apply. The sample statistic can
    be negative and is NOT clipped. With distinct target values it cannot
    exceed ``(n - 2) / (n + 1)``; tied targets can.
    """
    x = np.asarray(x, dtype=np.float64).ravel()
    y = np.asarray(y, dtype=np.float64).ravel()
    if x.size != y.size:
        raise ValueError(f"xi needs paired samples, got {x.size} and {y.size}")
    n = int(x.size)
    if n < 2:
        return float("nan")
    below, above = _rank_counts(y)
    denominator = float((above * (n - above)).sum())
    if denominator == 0.0:
        return float("nan")
    order = np.argsort(x, kind="stable")
    ordered_x = x[order]
    ordered_rank = below[order]
    starts = np.flatnonzero(np.concatenate(([True], ordered_x[1:] != ordered_x[:-1])))
    ends = np.concatenate((starts[1:], [n]))
    expected = 0.0
    blocks = []
    for start, end in zip(starts, ends):
        values = np.sort(ordered_rank[start:end])
        blocks.append(values)
        size = values.size
        if size > 1:
            index = np.arange(size, dtype=np.float64)
            expected += (2.0 / size) * float(np.dot(values, 2.0 * index - (size - 1.0)))
    for left, right in zip(blocks[:-1], blocks[1:]):
        expected += _mean_abs_cross(left, right)
    return 1.0 - n * expected / (2.0 * denominator)


def _xi_batch(x_rows: np.ndarray, y_rows: np.ndarray) -> np.ndarray:
    """Row-wise ``xi(x_i -> y_i)``, identical to the scalar tie-average."""
    n_rows, n = x_rows.shape
    if n < 2 or n_rows == 0:
        return np.full(n_rows, np.nan, dtype=np.float64)
    below = np.empty((n_rows, n), dtype=np.float64)
    above = np.empty((n_rows, n), dtype=np.float64)
    for row in range(n_rows):
        below[row], above[row] = _rank_counts(y_rows[row])
    denominator = (above * (n - above)).sum(axis=1)
    order = np.argsort(x_rows, axis=1, kind="stable")
    ordered_x = np.take_along_axis(x_rows, order, axis=1)
    tied = (ordered_x[:, 1:] == ordered_x[:, :-1]).any(axis=1)
    steps = np.abs(np.diff(np.take_along_axis(below, order, axis=1), axis=1)).sum(axis=1)
    with np.errstate(divide="ignore", invalid="ignore"):
        out = np.where(denominator > 0.0, 1.0 - n * steps / (2.0 * denominator), np.nan)
    for tied_row in np.flatnonzero(tied):
        index = int(tied_row)
        out[index] = (
            _chatterjee_xi_tie_average(x_rows[index], y_rows[index])
            if denominator[index] > 0.0
            else np.nan
        )
    return np.asarray(out, dtype=np.float64)


def _xi_matrix(block: np.ndarray) -> np.ndarray:
    """Directional xi: row = predictor X, column = target Y."""
    n_series, n = block.shape
    out = np.full((n_series, n_series), np.nan, dtype=np.float64)
    if n_series == 0 or n < 2:
        return out
    below = np.empty((n_series, n), dtype=np.float64)
    above = np.empty((n_series, n), dtype=np.float64)
    for row in range(n_series):
        below[row], above[row] = _rank_counts(block[row])
    denominator = (above * (n - above)).sum(axis=1)
    for predictor in range(n_series):
        x = block[predictor]
        order = np.argsort(x, kind="stable")
        ordered_x = x[order]
        if np.any(ordered_x[1:] == ordered_x[:-1]):
            for target in range(n_series):
                if denominator[target] > 0.0:
                    out[predictor, target] = _chatterjee_xi_tie_average(x, block[target])
            continue
        steps = np.abs(np.diff(below[:, order], axis=1)).sum(axis=1)
        with np.errstate(divide="ignore", invalid="ignore"):
            row_values = 1.0 - n * steps / (2.0 * denominator)
        out[predictor] = np.where(denominator > 0.0, row_values, np.nan)
    np.fill_diagonal(out, np.nan)
    return out


def _dependence_world(
    block: np.ndarray, eligible: np.ndarray
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    """Every pairwise metric for one world and one view."""
    n_series, n = block.shape
    matrices = {m: np.full((n_series, n_series), np.nan) for m in DEPENDENCE_METRICS}
    valid = {m: np.zeros((n_series, n_series), dtype=bool) for m in DEPENDENCE_METRICS}
    if n_series == 0 or n < _MIN_PAIRS or not eligible.any():
        return matrices, valid
    rows = eligible & np.isfinite(block).all(axis=1)
    usable = np.outer(rows, rows) & ~np.eye(n_series, dtype=bool)

    pearson, nonconstant = _pearson_matrix(block)
    both_vary = usable & np.outer(nonconstant, nonconstant)
    matrices["pearson"] = np.where(both_vary, pearson, np.nan)
    valid["pearson"] = both_vary

    spearman, rank_varies = _pearson_matrix(_average_ranks(block))
    rank_pairs = usable & np.outer(rank_varies, rank_varies)
    matrices["spearman"] = np.where(rank_pairs, spearman, np.nan)
    valid["spearman"] = rank_pairs

    # xi needs a non-constant TARGET only: a constant predictor is a valid
    # question whose answer is exactly zero.
    xi = _xi_matrix(block)
    xi_valid = usable & nonconstant[None, :]
    matrices["xi"] = np.where(xi_valid, xi, np.nan)
    valid["xi"] = xi_valid

    # Symmetrize per world, after each direction has been tie-averaged: the
    # maximum of two averages is not the average of two maxima.
    pair_valid = xi_valid & xi_valid.T
    with np.errstate(invalid="ignore"):
        xi_max = np.maximum(xi, xi.T)
    matrices["xi_max"] = np.where(pair_valid, xi_max, np.nan)
    valid["xi_max"] = pair_valid
    return matrices, valid


# ---------------------------------------------------------------------------
# Variance inflation
# ---------------------------------------------------------------------------


def _vif_world(block: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, int, float]:
    """Per-predictor VIF, constant flags, design rank and condition number.

    Every target is projected onto the SVD-truncated span of the other
    predictors and the residual sum of squares is read directly. Inferring
    an infinite VIF from "the augmented design has the same numerical rank as
    the nuisance design" is wrong: a target can sit far outside a numerically
    rank-deficient nuisance span and still have a perfectly finite VIF.
    """
    n_predictors, n = block.shape
    vif = np.full(n_predictors, np.nan, dtype=np.float64)
    valid = np.zeros(n_predictors, dtype=bool)
    constant = np.zeros(n_predictors, dtype=bool)
    if n_predictors == 0 or n == 0:
        # No predictors, or a view with no observations at all (the difference
        # view of a one-step window): nothing is constant, nothing is ranked.
        return vif, valid, constant, 0, float("nan")

    eps = float(np.finfo(np.float64).eps)
    design = block.T
    centred = design - design.mean(axis=0, keepdims=True)
    norms = np.linalg.norm(centred, axis=0)
    constant = _constant_rows(block, centred.T, norms)
    unit = centred / np.where(constant, 1.0, norms)[None, :]

    varying = np.flatnonzero(~constant)
    if varying.size == 0:
        return vif, valid, constant, 0, float("inf")
    singular_full = np.linalg.svd(unit[:, varying], compute_uv=False)
    tol_full = max(eps, eps * max(n, int(varying.size)) * float(singular_full.max()))
    rank = int(np.count_nonzero(singular_full > tol_full))
    if constant.any() or rank < n_predictors:
        condition = float("inf")
    else:
        smallest = float(singular_full.min())
        condition = float(singular_full.max() / smallest) if smallest > 0.0 else float("inf")
    if n < _MIN_PAIRS:
        return vif, valid, constant, rank, condition

    for target in varying:
        z = unit[:, target]
        nuisance = unit[:, varying[varying != target]]
        q = int(nuisance.shape[1])
        if q == 0:
            vif[target] = 1.0
            valid[target] = True
            continue
        left, singular, _ = np.linalg.svd(nuisance, full_matrices=False)
        svd_tol = max(eps, eps * max(n, q) * float(singular.max()))
        basis = left[:, singular > svd_tol]
        residual = z - basis @ (basis.T @ z) if basis.shape[1] else z
        residual_fraction = float(residual @ residual)
        residual_tol = max(eps**2, (eps * max(n, q + 1)) ** 2)
        vif[target] = float("inf") if residual_fraction <= residual_tol else 1.0 / residual_fraction
        valid[target] = True
    return vif, valid, constant, rank, condition


# ---------------------------------------------------------------------------
# Temporal
# ---------------------------------------------------------------------------


def _acf_world(block: np.ndarray, lags: Sequence[int]) -> tuple[np.ndarray, np.ndarray]:
    """Global-mean, full-denominator sample ACF at each lag."""
    n_series, n = block.shape
    acf = np.full((n_series, len(lags)), np.nan, dtype=np.float64)
    valid = np.zeros((n_series, len(lags)), dtype=bool)
    if n_series == 0 or n == 0:
        return acf, valid
    centred = block - block.mean(axis=1, keepdims=True)
    denominator = (centred**2).sum(axis=1)
    varies = denominator > 0.0
    safe = np.where(varies, denominator, 1.0)
    for column, lag in enumerate(lags):
        if n - lag < _MIN_PAIRS:
            continue
        numerator = (centred[:, : n - lag] * centred[:, lag:]).sum(axis=1)
        acf[:, column] = np.where(varies, numerator / safe, np.nan)
        valid[:, column] = varies
    return acf, valid


def _lag_xi_world(block: np.ndarray, lags: Sequence[int]) -> tuple[np.ndarray, np.ndarray]:
    """Forward ``xi(z_t -> z_{t+h})``: lag association, never impact."""
    n_series, n = block.shape
    values = np.full((n_series, len(lags)), np.nan, dtype=np.float64)
    valid = np.zeros((n_series, len(lags)), dtype=bool)
    if n_series == 0 or n == 0:
        return values, valid
    for column, lag in enumerate(lags):
        if n - lag < _MIN_PAIRS:
            continue
        scores = _xi_batch(block[:, : n - lag], block[:, lag:])
        values[:, column] = scores
        valid[:, column] = ~np.isnan(scores)
    return values, valid


# ---------------------------------------------------------------------------
# Contributions to sales
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ContributionDescriptor:
    """One row of the sales contribution hierarchy."""

    key: str
    display_label: str
    parent: str
    path: tuple[str, ...]
    depth: int
    sibling_set: str
    basis: str
    atomic: bool
    role: str
    column_kind: str
    column_index: int


#: Parent of each complete sibling cut. ``top`` closes against sales itself.
_SIBLING_PARENT: dict[str, dict[str, str]] = {
    "base_direct_plus_indirect": {
        "top": "Y",
        "media_children": "media_total",
        "channel_direct_children": "channels_direct_y_total",
        "control_children": "controls_baseline_alloc_total",
        "demand_children": "demand_baseline_alloc_total",
    },
    "observed_path_media": {"observed_path_children": "media_total_observed_path"},
}


def _contribution_descriptors(
    widths: Mapping[str, int], basis: str
) -> list[ContributionDescriptor]:
    if basis == "observed_path_media":
        out = [
            ContributionDescriptor(
                key="media_total_observed_path",
                display_label="total observed-path media effect",
                parent="",
                path=("media_total_observed_path",),
                depth=0,
                sibling_set="root",
                basis=basis,
                atomic=False,
                role="media",
                column_kind="",
                column_index=-1,
            )
        ]
        out.extend(
            ContributionDescriptor(
                key=f"C{k + 1}_observed_y",
                display_label=f"C{k + 1} observed-path contribution",
                parent="media_total_observed_path",
                path=("media_total_observed_path", f"C{k + 1}_observed_y"),
                depth=1,
                sibling_set="observed_path_children",
                basis=basis,
                atomic=True,
                role="media",
                column_kind="channel",
                column_index=k,
            )
            for k in range(widths["channel"])
        )
        return out

    out = [
        ContributionDescriptor(
            key="media_total",
            display_label="media (direct + indirect)",
            parent="",
            path=("media_total",),
            depth=0,
            sibling_set="top",
            basis=basis,
            atomic=False,
            role="media",
            column_kind="",
            column_index=-1,
        ),
        ContributionDescriptor(
            key="channels_direct_y_total",
            display_label="direct C->Y contributions",
            parent="media_total",
            path=("media_total", "channels_direct_y_total"),
            depth=1,
            sibling_set="media_children",
            basis=basis,
            atomic=False,
            role="media",
            column_kind="",
            column_index=-1,
        ),
    ]
    out.extend(
        ContributionDescriptor(
            key=f"C{k + 1}_direct_y",
            display_label=f"C{k + 1} direct contribution",
            parent="channels_direct_y_total",
            path=("media_total", "channels_direct_y_total", f"C{k + 1}_direct_y"),
            depth=2,
            sibling_set="channel_direct_children",
            basis=basis,
            atomic=True,
            role="media",
            column_kind="channel",
            column_index=k,
        )
        for k in range(widths["channel"])
    )
    out.extend(
        ContributionDescriptor(
            key=f"indirect_{label}",
            display_label=f"indirect media effect via {label} edges",
            parent="media_total",
            path=("media_total", f"indirect_{label}"),
            depth=1,
            sibling_set="media_children",
            basis=basis,
            atomic=True,
            role="media",
            column_kind="source",
            column_index=index,
        )
        for index, label in enumerate(("cc", "zc", "dc"))
    )
    out.append(
        ContributionDescriptor(
            key="controls_baseline_alloc_total",
            display_label="controls, retained baseline allocation",
            parent="",
            path=("controls_baseline_alloc_total",),
            depth=0,
            sibling_set="top",
            basis=basis,
            atomic=False,
            role="control",
            column_kind="",
            column_index=-1,
        )
    )
    out.extend(
        ContributionDescriptor(
            key=f"Z{m + 1}_baseline_alloc",
            display_label=f"Z{m + 1} retained baseline allocation",
            parent="controls_baseline_alloc_total",
            path=("controls_baseline_alloc_total", f"Z{m + 1}_baseline_alloc"),
            depth=1,
            sibling_set="control_children",
            basis=basis,
            atomic=True,
            role="control",
            column_kind="control",
            column_index=m,
        )
        for m in range(widths["control"])
    )
    out.append(
        ContributionDescriptor(
            key="demand_baseline_alloc_total",
            display_label="latent demand, retained baseline allocation",
            parent="",
            path=("demand_baseline_alloc_total",),
            depth=0,
            sibling_set="top",
            basis=basis,
            atomic=False,
            role="latent",
            column_kind="",
            column_index=-1,
        )
    )
    out.extend(
        ContributionDescriptor(
            key=f"D{j + 1}_baseline_alloc",
            display_label=f"D{j + 1} retained baseline allocation",
            parent="demand_baseline_alloc_total",
            path=("demand_baseline_alloc_total", f"D{j + 1}_baseline_alloc"),
            depth=1,
            sibling_set="demand_children",
            basis=basis,
            atomic=True,
            role="latent",
            column_kind="latent",
            column_index=j,
        )
        for j in range(widths["latent"])
    )
    out.append(
        ContributionDescriptor(
            key="B_intrinsic",
            display_label="baseline intrinsic walk",
            parent="",
            path=("B_intrinsic",),
            depth=0,
            sibling_set="top",
            basis=basis,
            atomic=True,
            role="baseline",
            column_kind="",
            column_index=-1,
        )
    )
    out.append(
        ContributionDescriptor(
            key="Y_noise",
            display_label="observation noise",
            parent="",
            path=("Y_noise",),
            depth=0,
            sibling_set="top",
            basis=basis,
            atomic=True,
            role="noise",
            column_kind="",
            column_index=-1,
        )
    )
    return out


def _atomic_component(source: _DiagnosticSource, desc: ContributionDescriptor) -> np.ndarray:
    key = desc.key
    if key == "B_intrinsic":
        return source.arrays["baseline_intrinsic"]
    if key == "Y_noise":
        return source.arrays["sales_noise"]
    if key.startswith("indirect_"):
        return source.arrays["indirect_by_source"][:, :, desc.column_index]
    if key.endswith("_direct_y"):
        return source.arrays["channel_contribution"][:, :, desc.column_index]
    if key.endswith("_observed_y"):
        return source.arrays["contributions_observed"][:, :, desc.column_index]
    if key.endswith("_baseline_alloc") and desc.column_kind == "control":
        return source.arrays["control_contribution"][:, :, desc.column_index]
    if key.endswith("_baseline_alloc") and desc.column_kind == "latent":
        return source.arrays["confounder_contribution"][:, :, desc.column_index]
    raise KeyError(f"no component array for contribution {key!r}")  # pragma: no cover


@dataclass(frozen=True)
class ContributionClosure:
    """Whether one sibling cut adds up, and by how much it misses.

    Closure is a property of a CUT, not of the table: a complete set of
    siblings within one world closes exactly, and so do common-population
    linear rollups. Marginal quantiles, populations conditioned on activity,
    partial projections and mixed bases do not — those are marked incomplete
    and make no closure claim.
    """

    basis: str
    sibling_set: str
    parent: str
    children: tuple[str, ...]
    complete: bool
    common_population: bool
    is_partial_projection: bool
    omitted_keys: tuple[str, ...]
    residuals: dict[str, np.ndarray | None]

    def max_abs_residual(self, field: str = "net_total") -> float | None:
        residual = self.residuals.get(field)
        if residual is None or residual.size == 0:
            return None
        finite = residual[np.isfinite(residual)]
        return float(np.abs(finite).max()) if finite.size else None

    def summary(self) -> dict[str, Any]:
        return {
            "basis": self.basis,
            "sibling_set": self.sibling_set,
            "parent": self.parent,
            "children": list(self.children),
            "complete": self.complete,
            "common_population": self.common_population,
            "is_partial_projection": self.is_partial_projection,
            "omitted_keys": list(self.omitted_keys),
            "max_abs_residual": {
                field: self.max_abs_residual(field) for field in CONTRIBUTION_FIELDS
            },
        }


@dataclass(frozen=True)
class ContributionBudget:
    """Every contribution row's amount and share of sales, per world.

    Six measures are always carried. ``net_*`` keeps the sign, so a complete
    top cut sums to sales exactly. ``gross_*`` sums absolute atomic
    magnitudes, so a parent's gross is the sum of its atomic descendants'
    gross, never ``abs`` of their sum — cancellation between two children is
    activity, not absence. ``gross_over_net_sales`` for a complete valid top
    cut is therefore at least 1, with equality exactly when nothing cancels;
    an arbitrary subset has no such bound.
    """

    basis: str
    descriptors: tuple[ContributionDescriptor, ...]
    world_ids: np.ndarray
    n_time_steps: int
    sales_total: np.ndarray
    measures: dict[str, np.ndarray]
    active: np.ndarray
    denominator_valid: np.ndarray
    quantile_levels: tuple[float, ...]
    #: Column widths of the SOURCE, so a projection that drops a trailing
    #: column still knows the sibling cut it came from is incomplete.
    source_widths: dict[str, int]
    is_partial_projection: bool = False
    omitted_keys: tuple[str, ...] = ()

    # -- basics -------------------------------------------------------------
    @property
    def keys(self) -> tuple[str, ...]:
        return tuple(d.key for d in self.descriptors)

    @property
    def n_worlds(self) -> int:
        return int(self.world_ids.size)

    def __getitem__(self, key: str) -> ContributionDescriptor:
        for desc in self.descriptors:
            if desc.key == key:
                return desc
        raise KeyError(f"unknown contribution key {key!r}; available: {list(self.keys)}")

    def _index(self, key: str) -> int:
        for index, desc in enumerate(self.descriptors):
            if desc.key == key:
                return index
        raise KeyError(f"unknown contribution key {key!r}; available: {list(self.keys)}")

    @staticmethod
    def _field(measure: str, mode: str) -> str:
        try:
            return _MEASURE_FIELD[(mode, measure)]
        except KeyError:
            raise ValueError(
                f"unknown measure/mode {measure!r}/{mode!r}; measure is one of "
                "'total', 'mean_per_period', 'share' and mode is 'net' or 'gross'"
            ) from None

    def values(self, key: str, *, measure: str = "share", mode: str = "net") -> np.ndarray:
        """The per-world value of one row (``(n_worlds,)``)."""
        return self.measures[self._field(measure, mode)][:, self._index(key)]

    # -- aggregation --------------------------------------------------------
    def _population(self, index: int, population: str) -> np.ndarray:
        if population == "unconditional":
            return np.ones(self.n_worlds, dtype=bool)
        if population == "conditional_on_active":
            return self.active[:, index]
        raise ValueError(
            f"unknown population {population!r}; expected 'unconditional' or "
            "'conditional_on_active'"
        )

    def stats(
        self,
        key: str,
        *,
        measure: str = "share",
        mode: str = "net",
        weighting: str = "macro",
        population: str = "unconditional",
    ) -> tuple[dict[str, Any], CoverageLedger]:
        """One row's across-world report plus its coverage ledger."""
        field = self._field(measure, mode)
        index = self._index(key)
        included = self._population(index, population)
        share = measure == "share"
        valid = included & (self.denominator_valid if share else np.ones(self.n_worlds, bool))
        per_world = self.measures[field][:, index]
        ledger = _ledger(
            per_world,
            valid,
            eligible=included,
            selected=self.n_worlds,
            observations=int(included.sum()) * self.n_time_steps,
        )
        active_count = int(np.count_nonzero(self.active[:, index] & included))
        if weighting == "macro":
            stats = _finite_stats(per_world, valid, self.quantile_levels)
        elif weighting == "micro":
            stats = {"n": int(np.count_nonzero(valid)), "value": self._micro(field, index, valid)}
        else:
            raise ValueError(f"unknown weighting {weighting!r}; expected 'macro' or 'micro'")
        stats = dict(stats)
        stats["active_worlds"] = active_count
        stats["active_fraction"] = (
            float(active_count / self.n_worlds) if self.n_worlds else float("nan")
        )
        return stats, ledger

    def _micro(self, field: str, index: int, valid: np.ndarray) -> float | None:
        """Pooled (sales-weighted) value: totals over totals, never a mean of ratios."""
        if not valid.any():
            return None
        net = self.measures["net_total"][valid, index]
        gross = self.measures["gross_total"][valid, index]
        sales = self.sales_total[valid]
        periods = float(int(valid.sum()) * self.n_time_steps)
        if field == "net_total":
            return float(net.sum())
        if field == "gross_total":
            return float(gross.sum())
        if field == "net_mean_per_period":
            return float(net.sum() / periods) if periods else None
        if field == "gross_mean_per_period":
            return float(gross.sum() / periods) if periods else None
        denominator = float(sales.sum())
        if denominator == 0.0:
            return None
        if field == "net_share":
            return float(net.sum() / denominator)
        return float(gross.sum() / abs(denominator))

    # -- projections and closure -------------------------------------------
    def select(
        self, keys: Sequence[str] | None = None, *, basis: str | None = None
    ) -> ContributionBudget:
        """A presentation projection onto ``keys``, in caller order.

        A projection is not a smaller budget: the omitted siblings still
        exist, so the result records ``is_partial_projection`` and
        ``omitted_keys`` and stops making closure claims. Bases are
        alternative readings of the same media effect and never mix — ask
        ``DataDiagnostics.contribution_bases`` for the other one.
        """
        if basis is not None and basis != self.basis:
            raise ValueError(
                f"this budget is basis {self.basis!r}; {basis!r} is an alternative reading "
                "of the same media effect and is not additive alongside it — read it from "
                "DataDiagnostics.contribution_bases"
            )
        if keys is None:
            return self
        chosen = _validate_keys(keys, self.keys, "contribution keys")
        index = [self._index(key) for key in chosen]
        # Accumulate: projecting an already-partial budget onto everything it
        # still holds omits nothing NEW, but the earlier omissions are still
        # missing, and the result must not claim a complete cut.
        omitted = tuple(
            dict.fromkeys((*self.omitted_keys, *(k for k in self.keys if k not in set(chosen))))
        )
        return replace(
            self,
            descriptors=tuple(self.descriptors[i] for i in index),
            measures={field: values[:, index] for field, values in self.measures.items()},
            active=self.active[:, index],
            is_partial_projection=self.is_partial_projection or len(chosen) < len(self.keys),
            omitted_keys=omitted,
        )

    def sibling_sets(self) -> tuple[str, ...]:
        seen = [d.sibling_set for d in self.descriptors if d.sibling_set != "root"]
        return tuple(dict.fromkeys(seen))

    def closure(self, sibling_set: str = "top") -> ContributionClosure:
        """Per-world residuals of one sibling cut against its parent."""
        parents = _SIBLING_PARENT[self.basis]
        if sibling_set not in parents:
            raise ValueError(
                f"unknown sibling set {sibling_set!r} for basis {self.basis!r}; "
                f"expected one of {list(parents)}"
            )
        parent = parents[sibling_set]
        present = tuple(d.key for d in self.descriptors if d.sibling_set == sibling_set)
        complete_children = tuple(
            d.key
            for d in _contribution_descriptors(self.source_widths, self.basis)
            if d.sibling_set == sibling_set
        )
        omitted = tuple(k for k in complete_children if k not in set(present))
        complete = not omitted and not self.is_partial_projection
        residuals: dict[str, np.ndarray | None] = {}
        for field in CONTRIBUTION_FIELDS:
            child_sum = (
                np.zeros(self.n_worlds)
                if not present
                else sum(self.measures[field][:, self._index(k)] for k in present)
            )
            parent_values = self._parent_values(parent, field)
            residuals[field] = None if parent_values is None else parent_values - child_sum
        return ContributionClosure(
            basis=self.basis,
            sibling_set=sibling_set,
            parent=parent,
            children=present,
            complete=complete,
            common_population=True,
            is_partial_projection=self.is_partial_projection,
            omitted_keys=omitted,
            residuals=residuals,
        )

    def _parent_values(self, parent: str, field: str) -> np.ndarray | None:
        if parent != "Y":
            try:
                return self.measures[field][:, self._index(parent)]
            except KeyError:
                return None
        # The top cut closes against sales itself, which has no gross
        # decomposition of its own.
        if field == "net_total":
            return self.sales_total
        if field == "net_mean_per_period":
            return self.sales_total / self.n_time_steps
        if field == "net_share":
            return np.where(self.denominator_valid, 1.0, np.nan)
        return None

    # -- reports ------------------------------------------------------------
    def table(
        self,
        *,
        measure: str = "share",
        mode: str = "net",
        weighting: str = "macro",
        population: str = "unconditional",
        sibling_set: str | None = None,
    ) -> str:
        field = self._field(measure, mode)
        rows = [
            desc
            for desc in self.descriptors
            if sibling_set is None or desc.sibling_set == sibling_set
        ]
        if not rows:
            return f"contributions — {field} — no rows in sibling set {sibling_set!r}"
        levels = self.quantile_levels
        columns = (
            ["mean", *(_q_key(level) for level in levels)] if weighting == "macro" else ["value"]
        )
        body: list[tuple[str, str, list[str]]] = []
        for desc in rows:
            stats, ledger = self.stats(
                desc.key,
                measure=measure,
                mode=mode,
                weighting=weighting,
                population=population,
            )
            cells = [
                _fmt(stats[column]) if stats.get(column) is not None else "n/a"
                for column in columns
            ]
            body.append((("  " * desc.depth) + desc.key, str(ledger.valid), cells))
        name_w = max(len("row"), *(len(r[0]) for r in body))
        count_w = max(len("worlds"), *(len(r[1]) for r in body))
        value_w = max(8, *(len(v) for r in body for v in r[2]), *(len(c) for c in columns))
        header = f"{'row':<{name_w}}  {'worlds':>{count_w}}  " + "  ".join(
            f"{c:>{value_w}}" for c in columns
        )
        title = (
            f"contributions to sales — {field} — {weighting} — {population} — "
            f"basis {self.basis} — {self.n_worlds} worlds x {self.n_time_steps} steps"
        )
        lines = [title, header, "-" * len(header)]
        for name, count, cells in body:
            lines.append(
                f"{name:<{name_w}}  {count:>{count_w}}  "
                + "  ".join(f"{v:>{value_w}}" for v in cells)
            )
        if self.is_partial_projection:
            lines.append(
                f"partial projection: {len(self.omitted_keys)} sibling rows omitted, "
                "so these rows do not add up to their parent"
            )
        return "\n".join(lines)

    def summary(self) -> dict[str, Any]:
        rows: dict[str, Any] = {}
        for desc in self.descriptors:
            entry: dict[str, Any] = {
                "display_label": desc.display_label,
                "parent": desc.parent or None,
                "path": list(desc.path),
                "depth": desc.depth,
                "sibling_set": desc.sibling_set,
                "atomic": desc.atomic,
                "role": desc.role,
            }
            for field in CONTRIBUTION_FIELDS:
                measure, mode = next(
                    (m, md) for (md, m), name in _MEASURE_FIELD.items() if name == field
                )
                macro, ledger = self.stats(desc.key, measure=measure, mode=mode, weighting="macro")
                micro, _ = self.stats(desc.key, measure=measure, mode=mode, weighting="micro")
                entry[field] = {
                    "macro": macro,
                    "micro": micro["value"],
                    "coverage": ledger.as_dict(),
                }
            rows[desc.key] = entry
        return {
            "basis": self.basis,
            "n_worlds": self.n_worlds,
            "n_time_steps": self.n_time_steps,
            "is_partial_projection": self.is_partial_projection,
            "omitted_keys": list(self.omitted_keys),
            "closure": {
                name: self.closure(name).summary()
                for name in self.sibling_sets()
                if name in _SIBLING_PARENT[self.basis]
            },
            "rows": rows,
        }

    def to_frame(self) -> pd.DataFrame:
        """Long-form per-(world, row) table with all six measures."""
        import pandas as pd

        n_rows = len(self.descriptors)
        data = {
            "world": np.repeat(self.world_ids, n_rows),
            "key": np.tile(np.array(self.keys, dtype=object), self.n_worlds),
            "parent": np.tile(
                np.array([d.parent for d in self.descriptors], dtype=object), self.n_worlds
            ),
            "sibling_set": np.tile(
                np.array([d.sibling_set for d in self.descriptors], dtype=object), self.n_worlds
            ),
            "basis": self.basis,
            "atomic": np.tile(np.array([d.atomic for d in self.descriptors]), self.n_worlds),
            "active": self.active.reshape(-1),
        }
        for field in CONTRIBUTION_FIELDS:
            data[field] = self.measures[field].reshape(-1)
        return pd.DataFrame(data)


def _contribution_budget(
    source: _DiagnosticSource, basis: str, levels: tuple[float, ...]
) -> ContributionBudget:
    descriptors = tuple(_contribution_descriptors(source.widths, basis))
    n_worlds = int(source.world_ids.size)
    n_time_steps = source.n_time_steps
    sales_total = source.arrays["sales"].sum(axis=1)

    net = np.zeros((n_worlds, len(descriptors)), dtype=np.float64)
    gross = np.zeros((n_worlds, len(descriptors)), dtype=np.float64)
    active = np.ones((n_worlds, len(descriptors)), dtype=bool)
    index_of = {desc.key: i for i, desc in enumerate(descriptors)}

    for i, desc in enumerate(descriptors):
        if not desc.atomic:
            continue
        component = _atomic_component(source, desc)
        net[:, i] = component.sum(axis=1)
        gross[:, i] = np.abs(component).sum(axis=1)
        if desc.column_kind in _WIDTH_KINDS:
            active[:, i] = source.masks[desc.column_kind][:, desc.column_index]

    # Parents roll up from their ATOMIC descendants: abs(sum) would erase the
    # cancellation that gross exists to show.
    for i, desc in enumerate(descriptors):
        if desc.atomic:
            continue
        for other in descriptors:
            if other.atomic and desc.key in other.path:
                net[:, i] += net[:, index_of[other.key]]
                gross[:, i] += gross[:, index_of[other.key]]

    denominator_valid = sales_total != 0.0
    safe_sales = np.where(denominator_valid, sales_total, 1.0)
    share = np.where(denominator_valid[:, None], net / safe_sales[:, None], np.nan)
    gross_over = np.where(denominator_valid[:, None], gross / np.abs(safe_sales)[:, None], np.nan)
    measures = {
        "net_total": net,
        "net_mean_per_period": net / n_time_steps,
        "net_share": share,
        "gross_total": gross,
        "gross_mean_per_period": gross / n_time_steps,
        "gross_over_net_sales": gross_over,
    }
    return ContributionBudget(
        basis=basis,
        descriptors=descriptors,
        world_ids=source.world_ids,
        n_time_steps=n_time_steps,
        sales_total=sales_total,
        measures=measures,
        active=active,
        denominator_valid=denominator_valid,
        quantile_levels=levels,
        source_widths=dict(source.widths),
    )


# ---------------------------------------------------------------------------
# Per-view results
# ---------------------------------------------------------------------------


def _key_index(descriptors: Sequence[SeriesDescriptor], key: str) -> int:
    for index, desc in enumerate(descriptors):
        if desc.key == key:
            return index
    raise KeyError(f"unknown series key {key!r}")


def _render_table(
    title: str,
    columns: Sequence[str],
    rows: Sequence[tuple[str, str, list[str]]],
    count_header: str = "n",
) -> str:
    if not rows:
        return f"{title}\n(no rows)"
    if not columns:
        # A legitimately empty axis (a one-step window has no lags at all):
        # say so instead of rendering a table with no cells in it.
        return f"{title}\n(no columns to report)"
    name_w = max(len("key"), *(len(r[0]) for r in rows))
    count_w = max(len(count_header), *(len(r[1]) for r in rows))
    value_w = max([8, *(len(v) for r in rows for v in r[2]), *(len(c) for c in columns)])
    header = f"{'key':<{name_w}}  {count_header:>{count_w}}  " + "  ".join(
        f"{c:>{value_w}}" for c in columns
    )
    lines = [title, header, "-" * len(header)]
    for name, count, cells in rows:
        lines.append(
            f"{name:<{name_w}}  {count:>{count_w}}  " + "  ".join(f"{v:>{value_w}}" for v in cells)
        )
    return "\n".join(lines)


def _cell(value: float | None) -> str:
    if value is None:
        return "n/a"
    if np.isposinf(value):
        return "inf"
    if np.isneginf(value):
        return "-inf"
    return _fmt(float(value))


@dataclass(frozen=True)
class SeriesDiagnostics:
    """Raw values and the eight descriptive slots, per world and series."""

    view: str
    descriptors: tuple[SeriesDescriptor, ...]
    world_ids: np.ndarray
    n_time_steps: int
    series: np.ndarray | None
    eligible: np.ndarray
    slots: np.ndarray
    slot_valid: np.ndarray
    quantile_levels: tuple[float, ...]

    @property
    def keys(self) -> tuple[str, ...]:
        return tuple(d.key for d in self.descriptors)

    @property
    def n_worlds(self) -> int:
        return int(self.world_ids.size)

    def values(self, key: str) -> np.ndarray:
        """Every retained value of one series, over the worlds it is active in.

        ``n_eligible_worlds * T_view`` float64 values. This is a descriptive
        pool of realized values, not an iid sample: consecutive weeks are
        dependent and worlds have different scales.
        """
        if self.series is None:
            raise ValueError(
                f"{key}: raw values were dropped (keep_series=False); the eight per-world "
                "slots, dependence, VIF and temporal results are unaffected"
            )
        index = _key_index(self.descriptors, key)
        return self.series[self.eligible[:, index], index, :].reshape(-1)

    def slot(self, name: str) -> np.ndarray:
        """``(n_worlds, n_series)`` values of one descriptive slot."""
        if name not in SERIES_SLOTS:
            raise ValueError(f"unknown slot {name!r}; expected one of {list(SERIES_SLOTS)}")
        return self.slots[:, :, SERIES_SLOTS.index(name)]

    def stats(self, key: str, slot: str) -> tuple[dict[str, Any], CoverageLedger]:
        index = _key_index(self.descriptors, key)
        position = SERIES_SLOTS.index(slot) if slot in SERIES_SLOTS else -1
        if position < 0:
            raise ValueError(f"unknown slot {slot!r}; expected one of {list(SERIES_SLOTS)}")
        values = self.slots[:, index, position]
        valid = self.slot_valid[:, index, position]
        ledger = _ledger(
            values,
            valid,
            eligible=self.eligible[:, index],
            selected=self.n_worlds,
            observations=int(self.eligible[:, index].sum()) * self.n_time_steps,
        )
        return _finite_stats(values, valid, self.quantile_levels), ledger

    def table(self, slot: str = "std") -> str:
        columns = ["mean", *(_q_key(level) for level in self.quantile_levels)]
        rows = []
        for desc in self.descriptors:
            stats, ledger = self.stats(desc.key, slot)
            rows.append((desc.key, str(ledger.valid), [_cell(stats[column]) for column in columns]))
        title = f"series {slot} — {self.view} — {self.n_worlds} worlds x {self.n_time_steps} steps"
        return _render_table(title, columns, rows, count_header="worlds")

    def summary(self) -> dict[str, Any]:
        out: dict[str, Any] = {}
        for desc in self.descriptors:
            entry: dict[str, Any] = {
                "display_label": desc.display_label,
                "scope": desc.scope,
                "role": desc.role,
                "available": desc.available,
                "eligible_worlds": int(
                    self.eligible[:, _key_index(self.descriptors, desc.key)].sum()
                ),
            }
            for slot in SERIES_SLOTS:
                stats, ledger = self.stats(desc.key, slot)
                entry[slot] = {**stats, "coverage": ledger.as_dict()}
            out[desc.key] = entry
        return {
            "view": self.view,
            "n_worlds": self.n_worlds,
            "n_time_steps": self.n_time_steps,
            "keep_series": self.series is not None,
            "series": out,
        }

    def to_frame(self) -> pd.DataFrame:
        import pandas as pd

        n_series = len(self.descriptors)
        data: dict[str, Any] = {
            "view": self.view,
            "world": np.repeat(self.world_ids, n_series),
            "key": np.tile(np.array(self.keys, dtype=object), self.n_worlds),
            "scope": np.tile(
                np.array([d.scope for d in self.descriptors], dtype=object), self.n_worlds
            ),
            "role": np.tile(
                np.array([d.role for d in self.descriptors], dtype=object), self.n_worlds
            ),
            "eligible": self.eligible.reshape(-1),
        }
        for position, slot in enumerate(SERIES_SLOTS):
            data[slot] = self.slots[:, :, position].reshape(-1)
        return pd.DataFrame(data)


@dataclass(frozen=True)
class DependenceDiagnostics:
    """Per-world pairwise dependence matrices and their coverage."""

    view: str
    descriptors: tuple[SeriesDescriptor, ...]
    world_ids: np.ndarray
    n_time_steps: int
    matrices: dict[str, np.ndarray]
    valid: dict[str, np.ndarray]
    eligible: np.ndarray
    quantile_levels: tuple[float, ...]

    @property
    def keys(self) -> tuple[str, ...]:
        return tuple(d.key for d in self.descriptors)

    @property
    def n_worlds(self) -> int:
        return int(self.world_ids.size)

    def _metric(self, metric: str) -> str:
        if metric not in DEPENDENCE_METRICS:
            raise ValueError(
                f"unknown metric {metric!r}; expected one of {list(DEPENDENCE_METRICS)}"
            )
        return metric

    def matrix(self, metric: str = "xi_max", *, quantile: float = 0.5) -> np.ndarray:
        """Across-world quantile of one metric: ``(n_series, n_series)``.

        For ``xi`` the row is the predictor X and the column the target Y;
        the matrix is deliberately asymmetric. Aggregation is over the stored
        per-world values, and ``xi_max`` was symmetrized inside each world
        before it got here.
        """
        self._metric(metric)
        values = self.matrices[metric]
        valid = self.valid[metric]
        n_series = len(self.descriptors)
        out = np.full((n_series, n_series), np.nan)
        for i in range(n_series):
            for j in range(n_series):
                out[i, j] = _finite_quantile(values[:, i, j], valid[:, i, j], quantile)
        return out

    def pair(
        self, x_key: str, y_key: str, metric: str = "xi"
    ) -> tuple[dict[str, Any], CoverageLedger]:
        self._metric(metric)
        i = _key_index(self.descriptors, x_key)
        j = _key_index(self.descriptors, y_key)
        values = self.matrices[metric][:, i, j]
        valid = self.valid[metric][:, i, j]
        # A pair is eligible where BOTH series exist in the world; a world
        # where both exist but the statistic is undefined (a constant target)
        # is eligible-but-invalid, not "never asked".
        eligible = self.eligible[:, i] & self.eligible[:, j]
        ledger = _ledger(
            values,
            valid,
            eligible=eligible,
            selected=self.n_worlds,
            observations=int(eligible.sum()) * self.n_time_steps,
            pairs=int(valid.sum()),
        )
        return _finite_stats(values, valid, self.quantile_levels), ledger

    def table(self, metric: str = "xi_max", *, quantile: float = 0.5, limit: int = 20) -> str:
        """The strongest pairs by across-world quantile of ``metric``."""
        self._metric(metric)
        matrix = self.matrix(metric, quantile=quantile)
        keys = self.keys
        pairs: list[tuple[float, str, str, float]] = []
        for i, x_key in enumerate(keys):
            for j, y_key in enumerate(keys):
                if i == j or not np.isfinite(matrix[i, j]):
                    continue
                if metric in ("xi_max", "pearson", "spearman") and j < i:
                    continue
                pairs.append((abs(float(matrix[i, j])), x_key, y_key, float(matrix[i, j])))
        pairs.sort(reverse=True)
        rows = [
            (
                f"{x_key} -> {y_key}" if metric == "xi" else f"{x_key} ~ {y_key}",
                str(
                    int(
                        self.valid[metric][
                            :,
                            _key_index(self.descriptors, x_key),
                            _key_index(self.descriptors, y_key),
                        ].sum()
                    )
                ),
                [_cell(value)],
            )
            for _, x_key, y_key, value in pairs[:limit]
        ]
        title = (
            f"dependence — {metric} — {self.view} — {_q_key(quantile)} over "
            f"{self.n_worlds} worlds (top {min(limit, len(pairs))} of {len(pairs)} pairs)"
        )
        return _render_table(title, [_q_key(quantile)], rows, count_header="worlds")

    def summary(self) -> dict[str, Any]:
        out: dict[str, Any] = {"view": self.view, "n_worlds": self.n_worlds, "metrics": {}}
        for metric in DEPENDENCE_METRICS:
            values = self.matrices[metric]
            valid = self.valid[metric]
            pair_eligible = (
                self.eligible[:, :, None]
                & self.eligible[:, None, :]
                & ~np.eye(len(self.descriptors), dtype=bool)[None, :, :]
            )
            ledger = _ledger(values, valid, eligible=pair_eligible, selected=int(values.size))
            out["metrics"][metric] = {
                "keys": list(self.keys),
                "median_matrix": self.matrix(metric, quantile=0.5),
                "coverage": ledger.as_dict(),
            }
        return out

    def to_frame(self, metric: str = "xi") -> pd.DataFrame:
        import pandas as pd

        self._metric(metric)
        keys = self.keys
        n_series = len(keys)
        rows = []
        for i in range(n_series):
            for j in range(n_series):
                if i == j:
                    continue
                stats, ledger = self.pair(keys[i], keys[j], metric)
                rows.append(
                    {
                        "view": self.view,
                        "metric": metric,
                        "x": keys[i],
                        "y": keys[j],
                        "worlds": ledger.valid,
                        **{k: v for k, v in stats.items() if k != "n"},
                    }
                )
        return pd.DataFrame(rows)


@dataclass(frozen=True)
class VIFDiagnostics:
    """Variance inflation for one design scope, per world."""

    view: str
    scope: str
    descriptors: tuple[SeriesDescriptor, ...]
    world_ids: np.ndarray
    n_time_steps: int
    vif: np.ndarray
    valid: np.ndarray
    constant: np.ndarray
    eligible: np.ndarray
    rank: np.ndarray
    condition: np.ndarray
    quantile_levels: tuple[float, ...]

    @property
    def keys(self) -> tuple[str, ...]:
        return tuple(d.key for d in self.descriptors)

    @property
    def n_worlds(self) -> int:
        return int(self.world_ids.size)

    def values(self, key: str) -> np.ndarray:
        return self.vif[:, _key_index(self.descriptors, key)]

    def stats(self, key: str) -> tuple[dict[str, Any], CoverageLedger]:
        index = _key_index(self.descriptors, key)
        values = self.vif[:, index]
        valid = self.valid[:, index]
        ledger = _ledger(
            values,
            valid,
            eligible=self.eligible[:, index],
            selected=self.n_worlds,
            observations=int(self.eligible[:, index].sum()) * self.n_time_steps,
        )
        return _finite_stats(values, valid, self.quantile_levels), ledger

    def table(self) -> str:
        columns = ["mean", *(_q_key(level) for level in self.quantile_levels), "inf"]
        rows = []
        for desc in self.descriptors:
            stats, ledger = self.stats(desc.key)
            cells = [_cell(stats[column]) for column in columns[:-1]]
            cells.append(str(ledger.positive_infinite))
            rows.append((desc.key, str(ledger.valid), cells))
        title = (
            f"VIF — {self.scope} design — {self.view} — {self.n_worlds} worlds; "
            f"median rank {int(np.median(self.rank)) if self.n_worlds else 0}"
        )
        return _render_table(title, columns, rows, count_header="worlds")

    def summary(self) -> dict[str, Any]:
        rows = {}
        for desc in self.descriptors:
            stats, ledger = self.stats(desc.key)
            rows[desc.key] = {
                "role": desc.role,
                **stats,
                "constant_worlds": int(
                    self.constant[:, _key_index(self.descriptors, desc.key)].sum()
                ),
                "coverage": ledger.as_dict(),
            }
        condition_valid = np.isfinite(self.condition) | np.isinf(self.condition)
        return {
            "view": self.view,
            "scope": self.scope,
            "n_worlds": self.n_worlds,
            "predictors": list(self.keys),
            "rank": _finite_stats(
                self.rank.astype(np.float64), np.ones(self.n_worlds, bool), self.quantile_levels
            ),
            "condition": _finite_stats(self.condition, condition_valid, self.quantile_levels),
            "condition_infinite_worlds": int(np.count_nonzero(np.isposinf(self.condition))),
            "rows": rows,
        }

    def to_frame(self) -> pd.DataFrame:
        import pandas as pd

        n_series = len(self.descriptors)
        return pd.DataFrame(
            {
                "view": self.view,
                "scope": self.scope,
                "world": np.repeat(self.world_ids, n_series),
                "key": np.tile(np.array(self.keys, dtype=object), self.n_worlds),
                "vif": self.vif.reshape(-1),
                "valid": self.valid.reshape(-1),
                "constant": self.constant.reshape(-1),
                "eligible": self.eligible.reshape(-1),
                "rank": np.repeat(self.rank, n_series),
                "condition": np.repeat(self.condition, n_series),
            }
        )


@dataclass(frozen=True)
class TemporalDiagnostics:
    """Autocorrelation and forward lag-xi, per world and series."""

    view: str
    descriptors: tuple[SeriesDescriptor, ...]
    world_ids: np.ndarray
    n_time_steps: int
    lags: tuple[int, ...]
    acf: np.ndarray
    acf_valid: np.ndarray
    lag_xi: np.ndarray
    lag_xi_valid: np.ndarray
    eligible: np.ndarray
    quantile_levels: tuple[float, ...]

    @property
    def keys(self) -> tuple[str, ...]:
        return tuple(d.key for d in self.descriptors)

    @property
    def n_worlds(self) -> int:
        return int(self.world_ids.size)

    def _arrays(self, metric: str) -> tuple[np.ndarray, np.ndarray]:
        if metric == "acf":
            return self.acf, self.acf_valid
        if metric == "lag_xi":
            return self.lag_xi, self.lag_xi_valid
        raise ValueError(f"unknown temporal metric {metric!r}; expected 'acf' or 'lag_xi'")

    def matrix(self, metric: str = "acf", *, quantile: float = 0.5) -> np.ndarray:
        """Across-world quantile per (series, lag): ``(n_series, n_lags)``."""
        values, valid = self._arrays(metric)
        out = np.full((len(self.descriptors), len(self.lags)), np.nan)
        for i in range(out.shape[0]):
            for j in range(out.shape[1]):
                out[i, j] = _finite_quantile(values[:, i, j], valid[:, i, j], quantile)
        return out

    def stats(
        self, key: str, lag: int, metric: str = "acf"
    ) -> tuple[dict[str, Any], CoverageLedger]:
        values, valid = self._arrays(metric)
        if lag not in self.lags:
            raise ValueError(f"lag {lag} is not on the reported axis {list(self.lags)}")
        i = _key_index(self.descriptors, key)
        j = self.lags.index(lag)
        usable = max(self.n_time_steps - lag, 0)
        ledger = _ledger(
            values[:, i, j],
            valid[:, i, j],
            eligible=self.eligible[:, i],
            selected=self.n_worlds,
            observations=int(self.eligible[:, i].sum()) * self.n_time_steps,
            pairs=usable,
        )
        return _finite_stats(values[:, i, j], valid[:, i, j], self.quantile_levels), ledger

    def table(
        self, metric: str = "acf", *, quantile: float = 0.5, lags: Sequence[int] | None = None
    ) -> str:
        shown = tuple(self.lags) if lags is None else tuple(int(lag) for lag in lags)
        missing = [lag for lag in shown if lag not in self.lags]
        if missing:
            raise ValueError(f"lags {missing} are not on the reported axis {list(self.lags)}")
        shown = shown[:12]
        matrix = self.matrix(metric, quantile=quantile)
        columns = [f"h{lag}" for lag in shown]
        rows = []
        for i, desc in enumerate(self.descriptors):
            _, valid = self._arrays(metric)
            worlds = int(valid[:, i, :].any(axis=1).sum())
            cells = [_cell(matrix[i, self.lags.index(lag)]) for lag in shown]
            rows.append((desc.key, str(worlds), cells))
        title = (
            f"temporal {metric} — {self.view} — {_q_key(quantile)} over {self.n_worlds} worlds "
            f"({len(self.lags)} lags on the axis)"
        )
        return _render_table(title, columns, rows, count_header="worlds")

    def summary(self) -> dict[str, Any]:
        out: dict[str, Any] = {
            "view": self.view,
            "n_worlds": self.n_worlds,
            "lags": list(self.lags),
            "series": {},
        }
        for desc in self.descriptors:
            entry: dict[str, Any] = {}
            for metric in ("acf", "lag_xi"):
                values, valid = self._arrays(metric)
                i = _key_index(self.descriptors, desc.key)
                entry[metric] = {
                    "by_lag": [
                        _finite_quantile(values[:, i, j], valid[:, i, j], 0.5)
                        for j in range(len(self.lags))
                    ],
                    "coverage": _ledger(
                        values[:, i, :],
                        valid[:, i, :],
                        eligible=np.repeat(self.eligible[:, i, None], len(self.lags), axis=1),
                        selected=int(valid[:, i, :].size),
                    ).as_dict(),
                }
            out["series"][desc.key] = entry
        return out

    def to_frame(self) -> pd.DataFrame:
        import pandas as pd

        n_series, n_lags = len(self.descriptors), len(self.lags)
        return pd.DataFrame(
            {
                "view": self.view,
                "world": np.repeat(self.world_ids, n_series * n_lags),
                "key": np.tile(np.repeat(np.array(self.keys, dtype=object), n_lags), self.n_worlds),
                "lag": np.tile(np.array(self.lags), self.n_worlds * n_series),
                "acf": self.acf.reshape(-1),
                "acf_valid": self.acf_valid.reshape(-1),
                "lag_xi": self.lag_xi.reshape(-1),
                "lag_xi_valid": self.lag_xi_valid.reshape(-1),
            }
        )


@dataclass(frozen=True)
class DiagnosticView:
    """Everything computed in one view (levels or first differences)."""

    view: str
    series: SeriesDiagnostics
    dependence: DependenceDiagnostics
    temporal: TemporalDiagnostics
    vif: dict[str, VIFDiagnostics]

    def summary(self) -> dict[str, Any]:
        return {
            "view": self.view,
            "series": self.series.summary(),
            "dependence": self.dependence.summary(),
            "temporal": self.temporal.summary(),
            "vif": {scope: result.summary() for scope, result in self.vif.items()},
        }


@dataclass(frozen=True)
class DataDiagnostics:
    """The full diagnostic report for one analysed set of worlds."""

    source_kind: str
    world_ids: np.ndarray
    n_time_steps: int
    descriptors: tuple[SeriesDescriptor, ...]
    scopes: tuple[str, ...]
    lags: tuple[int, ...]
    quantile_levels: tuple[float, ...]
    keep_series: bool
    outcomes: OutcomeDistributions
    contributions: ContributionBudget
    contribution_bases: dict[str, ContributionBudget]
    views: dict[str, DiagnosticView]

    @property
    def keys(self) -> tuple[str, ...]:
        return tuple(d.key for d in self.descriptors)

    @property
    def n_worlds(self) -> int:
        return int(self.world_ids.size)

    def __getitem__(self, view: str) -> DiagnosticView:
        try:
            return self.views[view]
        except KeyError:
            raise KeyError(
                f"view {view!r} was not computed; available: {sorted(self.views)}"
            ) from None

    def descriptor(self, key: str) -> SeriesDescriptor:
        return self.descriptors[_key_index(self.descriptors, key)]

    def descriptor_table(self) -> str:
        rows = [
            (
                desc.key,
                desc.scope,
                [desc.role, desc.display_label, "yes" if desc.available else "absent"],
            )
            for desc in self.descriptors
        ]
        return _render_table(
            f"series keys — {len(self.descriptors)} in scopes {list(self.scopes)}",
            ["role", "label", "present"],
            rows,
            count_header="scope",
        )

    def table(self) -> str:
        lines = [
            f"data diagnostics — {self.source_kind} — {self.n_worlds} worlds x "
            f"{self.n_time_steps} steps",
            f"scopes: {list(self.scopes)}   views: {list(self.views)}   "
            f"series: {len(self.descriptors)}   lags: {len(self.lags)}",
            f"contribution basis: {self.contributions.basis} "
            f"({len(self.contributions.descriptors)} rows)",
            "",
            self.contributions.table(sibling_set="top"),
        ]
        return "\n".join(lines)

    def summary(self) -> dict[str, Any]:
        """Strictly JSON-safe report (``json.dumps(..., allow_nan=False)``)."""
        payload: dict[str, Any] = {
            "source_kind": self.source_kind,
            "world_ids": self.world_ids,
            "n_worlds": self.n_worlds,
            "n_time_steps": self.n_time_steps,
            "scopes": list(self.scopes),
            "views": list(self.views),
            "lags": list(self.lags),
            "quantiles": list(self.quantile_levels),
            "keep_series": self.keep_series,
            "keys": list(self.keys),
            "contributions": self.contributions.summary(),
            "contribution_bases": list(self.contribution_bases),
            "by_view": {name: view.summary() for name, view in self.views.items()},
        }
        sanitized: dict[str, Any] = _strict_json_value(payload)
        return sanitized

    def to_frame(self) -> pd.DataFrame:
        """Per-(view, world, series) descriptive slots for every view."""
        import pandas as pd

        return pd.concat(
            [view.series.to_frame() for view in self.views.values()], ignore_index=True
        )


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def data_diagnostics(
    source: Mapping[str, Any] | Sequence[SCM],
    *,
    worlds: Any = None,
    scopes: Sequence[str] = ("nodes",),
    views: Sequence[str] = ("levels", "differences"),
    lags: Sequence[int] | None = None,
    quantiles: Sequence[float] = DEFAULT_QUANTILES,
    keep_series: bool = True,
) -> DataDiagnostics:
    """Interrogate the data a generator run actually produced.

    Parameters
    ----------
    source
        A corpus mapping (``sample_prior_predictive`` /
        ``DataGenerator.generate`` / ``load_corpus``) or a sequence of
        :class:`~prior_generator.worlds.SCM` worlds sharing one horizon.
    worlds
        World selector: None, a slice, an integer position, an integer
        position array or a boolean row mask. Duplicated positions are
        rejected — a repeated world would be counted twice in every
        across-world summary. A single world gives the same report shape as
        the full corpus, with one row per world axis.
    scopes
        ``nodes`` (always required) covers C/Z/D/B/Y. ``decomposition`` adds
        the exact additive pieces of sales; ``counterfactual`` adds the
        SCM-only audit paths (base spend, observed-path contributions and,
        when shocks were drawn, the unshocked series).
    views
        ``levels`` and/or ``differences``. Both by default: weekly series are
        random-walk-like, so level dependence is large even between
        independent series, and the difference view is the companion that
        shows whether anything survives detrending. Differencing is not a
        stationarity proof.
    lags
        Lag axis for ACF and forward lag-xi. Default is the contiguous
        ``1..min(52, T // 2)``: a sparse axis silently misses MA(2)-style
        structure at lag 2 and annual recurrence at lag 52.
    quantiles
        Across-world quantile levels for every report.
    keep_series
        Retain the float64 values behind the report so raw distributions can
        be plotted and re-quantiled. Set False for very large corpora: every
        non-raw result is identical, and only ``SeriesDiagnostics.values``
        (and the value plots) then raise.

    Returns
    -------
    DataDiagnostics

    Notes
    -----
    Purely post-hoc: no generation, no RNG draw, no mutation of ``source``,
    no file written. Every statistic is computed inside one world and then
    summarized across worlds; nothing is pooled across worlds before it is
    computed, because between-world level differences would manufacture
    dependence that is not in any world.

    The retained latent series (D, B) are ground truth kept for auditing,
    never model inputs, and no reported number is a p-value, a significance
    claim or a causal effect.
    """
    scope_names = _validate_names(scopes, SCOPES, "scopes", required=("nodes",))
    view_names = _validate_names(views, VIEWS, "views")
    levels = _validate_quantiles(quantiles)

    source_data = _extract_diagnostic_source(
        source, worlds, need_counterfactual="counterfactual" in scope_names
    )
    lag_axis = _validate_lags(lags, source_data.n_time_steps)

    descriptors: list[SeriesDescriptor] = list(_node_descriptors(source_data.widths))
    if "decomposition" in scope_names:
        descriptors.extend(_decomposition_descriptors(source_data.widths))
    if "counterfactual" in scope_names:
        descriptors.extend(_counterfactual_descriptors(source_data.widths, source_data.available))
    descriptor_tuple = tuple(descriptors)
    duplicates = sorted(
        {
            key
            for key in (d.key for d in descriptor_tuple)
            if [d.key for d in descriptor_tuple].count(key) > 1
        }
    )
    if duplicates:  # pragma: no cover - guards the key vocabulary itself
        raise AssertionError(f"series keys must be globally unique; duplicated {duplicates}")

    level_block = _series_block(source_data, descriptor_tuple)
    eligible = _eligibility(source_data, descriptor_tuple)
    n_worlds = int(source_data.world_ids.size)

    view_results: dict[str, DiagnosticView] = {}
    for view in view_names:
        block = level_block if view == "levels" else np.diff(level_block, axis=2)
        n_view = int(block.shape[2])
        slots, slot_valid = _compute_slots(block, eligible)

        n_series = len(descriptor_tuple)
        matrices = {
            metric: np.full((n_worlds, n_series, n_series), np.nan) for metric in DEPENDENCE_METRICS
        }
        matrix_valid = {
            metric: np.zeros((n_worlds, n_series, n_series), dtype=bool)
            for metric in DEPENDENCE_METRICS
        }
        acf = np.full((n_worlds, n_series, len(lag_axis)), np.nan)
        acf_valid = np.zeros((n_worlds, n_series, len(lag_axis)), dtype=bool)
        lag_xi = np.full((n_worlds, n_series, len(lag_axis)), np.nan)
        lag_xi_valid = np.zeros((n_worlds, n_series, len(lag_axis)), dtype=bool)
        for w in range(n_worlds):
            world_matrices, world_valid = _dependence_world(block[w], eligible[w])
            for metric in DEPENDENCE_METRICS:
                matrices[metric][w] = world_matrices[metric]
                matrix_valid[metric][w] = world_valid[metric]
            acf[w], acf_valid[w] = _acf_world(block[w], lag_axis)
            lag_xi[w], lag_xi_valid[w] = _lag_xi_world(block[w], lag_axis)
        acf_valid &= eligible[:, :, None]
        lag_xi_valid &= eligible[:, :, None]
        acf = np.where(acf_valid, acf, np.nan)
        lag_xi = np.where(lag_xi_valid, lag_xi, np.nan)

        vif_results: dict[str, VIFDiagnostics] = {}
        for scope in VIF_SCOPES:
            positions = np.array(
                [i for i, d in enumerate(descriptor_tuple) if scope in d.vif_scopes], dtype=np.int64
            )
            n_predictors = int(positions.size)
            vif = np.full((n_worlds, n_predictors), np.nan)
            valid = np.zeros((n_worlds, n_predictors), dtype=bool)
            constant = np.zeros((n_worlds, n_predictors), dtype=bool)
            rank = np.zeros(n_worlds, dtype=np.int64)
            condition = np.full(n_worlds, np.nan)
            scope_eligible = (
                eligible[:, positions] if n_predictors else np.zeros((n_worlds, 0), dtype=bool)
            )
            for w in range(n_worlds):
                rows = scope_eligible[w]
                world_vif, world_valid_row, world_constant, world_rank, world_condition = (
                    _vif_world(block[w][positions[rows]])
                )
                vif[w, rows] = world_vif
                valid[w, rows] = world_valid_row
                constant[w, rows] = world_constant
                rank[w] = world_rank
                condition[w] = world_condition
            vif_results[scope] = VIFDiagnostics(
                view=view,
                scope=scope,
                descriptors=tuple(descriptor_tuple[i] for i in positions),
                world_ids=source_data.world_ids,
                n_time_steps=n_view,
                vif=vif,
                valid=valid,
                constant=constant,
                eligible=scope_eligible,
                rank=rank,
                condition=condition,
                quantile_levels=levels,
            )

        view_results[view] = DiagnosticView(
            view=view,
            series=SeriesDiagnostics(
                view=view,
                descriptors=descriptor_tuple,
                world_ids=source_data.world_ids,
                n_time_steps=n_view,
                series=block if keep_series else None,
                eligible=eligible,
                slots=slots,
                slot_valid=slot_valid,
                quantile_levels=levels,
            ),
            dependence=DependenceDiagnostics(
                view=view,
                descriptors=descriptor_tuple,
                world_ids=source_data.world_ids,
                n_time_steps=n_view,
                matrices=matrices,
                valid=matrix_valid,
                eligible=eligible,
                quantile_levels=levels,
            ),
            temporal=TemporalDiagnostics(
                view=view,
                descriptors=descriptor_tuple,
                world_ids=source_data.world_ids,
                n_time_steps=n_view,
                lags=lag_axis,
                acf=acf,
                acf_valid=acf_valid,
                lag_xi=lag_xi,
                lag_xi_valid=lag_xi_valid,
                eligible=eligible,
                quantile_levels=levels,
            ),
            vif=vif_results,
        )

    bases = {
        "base_direct_plus_indirect": _contribution_budget(
            source_data, "base_direct_plus_indirect", levels
        )
    }
    if "contributions_observed" in source_data.available:
        bases["observed_path_media"] = _contribution_budget(
            source_data, "observed_path_media", levels
        )

    companion = outcome_distributions(
        source,
        worlds=_companion_selector(source_data.world_ids, source_data.n_worlds_total),
        keep_series=False,
    )
    return DataDiagnostics(
        source_kind=source_data.kind,
        world_ids=source_data.world_ids,
        n_time_steps=source_data.n_time_steps,
        descriptors=descriptor_tuple,
        scopes=scope_names,
        lags=lag_axis,
        quantile_levels=levels,
        keep_series=keep_series,
        outcomes=companion,
        contributions=bases["base_direct_plus_indirect"],
        contribution_bases=bases,
        views=view_results,
    )
