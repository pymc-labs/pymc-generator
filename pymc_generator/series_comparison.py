"""Alignment-sensitive series comparisons against an existing reference bank.

Distances, signed associations and empirical ranks are descriptive, not origin
probabilities or calibrated tests. No model or generator is run here.
"""

from __future__ import annotations

from collections.abc import Hashable, Sequence
from dataclasses import dataclass
from operator import index
from typing import TYPE_CHECKING, Any

import numpy as np

from .descriptor_analysis import _midranks
from .descriptors import _resolution
from .diagnostics import _centered_rows, _numeric_float64, _require_finite, temporal_statistics

if TYPE_CHECKING:
    import pandas as pd

__all__ = [
    "SERIES_METRICS",
    "QuantileNormalization",
    "SeriesComparison",
    "NearestSeries",
    "SimilarityAssessment",
    "normalize_series_quantile",
    "compare_series",
    "nearest_series",
    "assess_series_similarity",
]

SERIES_METRICS = ("mse", "level_r", "diff_r", "overlap_pearson_rmse")
_MINIMIZE = frozenset(("mse", "overlap_pearson_rmse"))


def _unmasked(value: Any, name: str) -> None:
    if np.ma.isMaskedArray(value):
        raise TypeError(f"{name} cannot be a masked array; provide explicit complete values")


def _precision(value: Any, dtype: np.dtype, shape: tuple[int, ...]) -> np.ndarray:
    if value is None:
        return np.full(shape, _resolution(dtype))
    _unmasked(value, "resolution")
    raw = np.asarray(value)
    if raw.dtype.kind not in "iuf":
        raise TypeError("resolution must be real-valued machine epsilon")
    try:
        result = np.broadcast_to(raw.astype(np.float64, copy=False), shape)
    except ValueError as exc:
        raise ValueError(f"resolution must broadcast to {shape}") from exc
    if not np.isfinite(result).all() or ((result <= 0) | (result >= 1)).any():
        raise ValueError("resolution must be finite and strictly between 0 and 1")
    return np.maximum(result, np.finfo(np.float64).eps)


def _quantile(values: np.ndarray, q: float, axis: int) -> np.ndarray:
    """Partition in original units; protect cancellation and subnormal endpoints."""
    position = q * (values.shape[axis] - 1)
    low, high = int(np.floor(position)), int(np.ceil(position))
    partition = np.partition(values, (low, high), axis=axis)
    left = np.asarray(np.take(partition, low, axis=axis))
    if low == high:
        return left
    right = np.asarray(np.take(partition, high, axis=axis))
    fraction = position - low
    with np.errstate(over="ignore", invalid="ignore"):
        difference = right - left
        out = np.asarray(
            left + fraction * difference
            if fraction < 0.5
            else right - (1.0 - fraction) * difference
        )
    unstable = (left != right) & (
        ~np.isfinite(difference)
        | (np.signbit(left) != np.signbit(right))
        | (np.abs(out) < np.finfo(np.float64).tiny)
    )
    if np.any(unstable):
        # Rare exact fallback: neither rounding two cancelling weighted terms
        # nor scaling an entire wide-range row can preserve all finite endpoints.
        from fractions import Fraction

        weight = Fraction.from_float(fraction)
        flat, low_values, high_values = out.reshape(-1), left.reshape(-1), right.reshape(-1)
        for position in np.flatnonzero(unstable):
            a = Fraction.from_float(float(low_values[position]))
            b = Fraction.from_float(float(high_values[position]))
            flat[position] = float(a + weight * (b - a))
    return out


@dataclass(frozen=True)
class QuantileNormalization:
    """Explicit quantile scaling, retaining source precision and availability.

    ``values`` has the input shape; other arrays omit its time ``axis``.
    A zero divisor leaves that row in original units with ``scale_available``
    false. Such rows remain useful for shape statistics, but not scaled MSE.
    ``mode='signed'`` divides by the signed quantile and a negative divisor
    reverses signs/order; it is not a rank transform or distribution matching.
    """

    values: np.ndarray
    divisors: np.ndarray
    scale_available: np.ndarray
    constant: np.ndarray
    resolution: np.ndarray
    quantile: float
    mode: str
    axis: int


def normalize_series_quantile(
    values: Any,
    *,
    quantile: float,
    mode: str,
    axis: int = -1,
    resolution: Any = None,
) -> QuantileNormalization:
    """Divide each series by an explicitly chosen quantile.

    ``absolute`` uses ``quantile(abs(values))`` and preserves signs; ``signed``
    uses ``quantile(values)`` and may reverse them. Both choices are required
    arguments; no normalization is silently applied by any dataset adapter.
    Inputs must be finite, real and have a nonempty time axis. Unrepresentable
    normalized values raise an error instead of masquerading as scaled data.
    """
    _unmasked(values, "values")
    raw = np.asarray(values)
    data = _require_finite(_numeric_float64(raw, "values"), "values")
    if data.ndim == 0:
        raise ValueError("values needs a time axis")
    if isinstance(axis, (bool, np.bool_)):
        raise TypeError("axis must be an integer")
    try:
        axis = index(axis)
    except TypeError as exc:
        raise TypeError("axis must be an integer") from exc
    if not -data.ndim <= axis < data.ndim:
        raise ValueError("axis is outside values dimensions")
    axis %= data.ndim
    if data.shape[axis] == 0:
        raise ValueError("the time axis cannot be empty")
    if isinstance(quantile, (bool, np.bool_)) or not isinstance(
        quantile, (int, float, np.integer, np.floating)
    ):
        raise TypeError("quantile must be a real number")
    q = float(quantile)
    if not np.isfinite(q) or not 0 <= q <= 1:
        raise ValueError("quantile must be finite and within [0, 1]")
    if mode not in ("absolute", "signed"):
        raise ValueError("mode must be 'absolute' or 'signed'")
    shape = data.shape[:axis] + data.shape[axis + 1 :]
    eps = _precision(resolution, raw.dtype, shape)
    divisors = _quantile(np.abs(data) if mode == "absolute" else data, q, axis)
    available = divisors != 0
    normalized = data.copy()
    with np.errstate(over="ignore", divide="ignore", invalid="ignore"):
        np.divide(
            data,
            np.expand_dims(divisors, axis),
            out=normalized,
            where=np.expand_dims(available, axis),
        )
    if not np.isfinite(normalized).all() or np.any((data != 0) & (normalized == 0)):
        raise ValueError("quantile-normalized values cannot be represented in float64")
    np.copyto(normalized, data, where=data == 0)
    magnitude = np.max(np.abs(data), axis=axis)
    _, exponent = np.frexp(magnitude)
    scaled = np.ldexp(data, -np.expand_dims(exponent, axis))
    spread = np.ptp(scaled, axis=axis)
    constant = spread <= 4 * eps * np.max(np.abs(scaled), axis=axis)
    return QuantileNormalization(normalized, divisors, available, constant, eps, q, mode, axis)


@dataclass(frozen=True)
class _SeriesInput:
    values: np.ndarray
    resolution: np.ndarray
    scale_available: np.ndarray
    normalization: tuple[str, float] | None
    constant: np.ndarray | None


def _series_input(value: Any, resolution: Any, name: str) -> _SeriesInput:
    constant: np.ndarray | None = None
    if isinstance(value, QuantileNormalization):
        if resolution is not None:
            raise ValueError(f"{name} already carries source resolution")
        if value.values.ndim != 2 or value.axis != 1:
            raise ValueError(f"{name} normalization must have shape (series, time), time axis 1")
        raw = value.values
        eps = value.resolution
        available = value.scale_available
        _unmasked(value.constant, f"{name} source-constant mask")
        constant = np.array(value.constant, dtype=bool, copy=True)
        normalization = (value.mode, value.quantile)
    else:
        _unmasked(value, name)
        raw = np.asarray(value)
        eps = resolution
        available = np.ones(raw.shape[0], dtype=bool) if raw.ndim else np.zeros(0, dtype=bool)
        normalization = None
    _unmasked(raw, name)
    data = _require_finite(_numeric_float64(raw, name), name)
    if data.ndim != 2 or data.shape[1] == 0:
        raise ValueError(f"{name} must have shape (series, time) with nonempty time")
    snapshot = np.array(data, copy=True, order="C")
    precision = _precision(eps, raw.dtype, (data.shape[0],))
    scale_available = np.array(available, dtype=bool, copy=True)
    for array in (snapshot, precision, scale_available):
        array.setflags(write=False)
    if constant is not None:
        constant.setflags(write=False)
    return _SeriesInput(snapshot, precision, scale_available, normalization, constant)


def _names(metrics: Sequence[str], lags: Sequence[int]) -> tuple[tuple[str, ...], tuple[int, ...]]:
    if isinstance(metrics, str) or isinstance(lags, str):
        raise TypeError("metrics and lags must be non-string sequences")
    names = tuple(metrics)
    if not names or any(not isinstance(m, str) or m not in SERIES_METRICS for m in names):
        raise ValueError(f"metrics must select from {SERIES_METRICS}")
    if len(set(names)) != len(names):
        raise ValueError("metrics must not repeat")
    selected_lags = []
    for lag in lags:
        if isinstance(lag, (bool, np.bool_)) or not isinstance(lag, (int, np.integer)):
            raise TypeError("lags must contain positive integers")
        if lag <= 0:
            raise ValueError("lags must contain positive integers")
        selected_lags.append(int(lag))
    if len(set(selected_lags)) != len(selected_lags):
        raise ValueError("lags must not repeat")
    if "overlap_pearson_rmse" in names and not selected_lags:
        raise ValueError("overlap_pearson_rmse requires at least one explicit lag")
    return names, tuple(selected_lags)


def _safe_mse(left: np.ndarray, right: np.ndarray, *, root: bool = False) -> np.ndarray:
    """Direct MSE/RMSE, restoring scale only after the requested reduction."""
    from scipy.spatial.distance import cdist

    result = np.asarray(cdist(left, right, metric="sqeuclidean"))
    result /= left.shape[1]
    rows, columns = np.nonzero(~np.isfinite(result) | (result < np.finfo(np.float64).tiny))
    if root:
        np.sqrt(result, out=result)
    block_size = max(1, (1 << 18) // left.shape[1])
    for start in range(0, rows.size, block_size):
        i, j = rows[start : start + block_size], columns[start : start + block_size]
        a, b = left[i], right[j]
        with np.errstate(over="ignore", invalid="ignore"):
            difference = a - b
        wide = ~np.isfinite(difference).all(axis=1)
        _, exponent = np.frexp(np.max(np.abs(np.where(wide[:, None], 0.0, difference)), axis=1))
        scaled = np.ldexp(difference, -exponent[:, None])
        if wide.any():
            _, wide_exponent = np.frexp(
                np.maximum(np.max(np.abs(a[wide]), axis=1), np.max(np.abs(b[wide]), axis=1))
            )
            scaled[wide] = np.ldexp(a[wide], -wide_exponent[:, None]) - np.ldexp(
                b[wide], -wide_exponent[:, None]
            )
            exponent[wide] = wide_exponent
        mean = np.mean(scaled * scaled, axis=1)
        with np.errstate(over="ignore", under="ignore"):
            repaired = np.ldexp(np.sqrt(mean), exponent) if root else np.ldexp(mean, 2 * exponent)
        unavailable = ~np.isfinite(repaired) | ((repaired == 0) & (mean > 0))
        repaired[unavailable] = np.nan
        result[i, j] = repaired
    return result


def _correlations(left: _SeriesInput, right: _SeriesInput, *, differences: bool) -> np.ndarray:
    units, valid = [], []
    for source in (left, right):
        data = source.values
        floor = None
        if differences:
            _, exponent = np.frexp(np.max(np.abs(data), axis=1))
            scaled = np.ldexp(data, -exponent[:, None])
            floor = 8 * source.resolution * np.max(np.abs(scaled), axis=1)
            data = np.diff(scaled, axis=1)
        centered, available = _centered_rows(data, source.resolution, constant_floor=floor)
        if source.constant is not None:
            available &= ~source.constant
        units.append(centered)
        valid.append(available)
    # Every pair uses the same contiguous time-axis reduction, including 1x1.
    # Reuse a bounded workspace rather than a query x reference x time tensor.
    n_time = units[0].shape[1]
    block_size = max(1, (1 << 18) // max(1, n_time))
    out = np.empty((units[0].shape[0], units[1].shape[0]))
    workspace = np.empty((min(block_size, units[1].shape[0]), n_time))
    for row, left_row in enumerate(units[0]):
        for start in range(0, units[1].shape[0], block_size):
            stop = min(start + block_size, units[1].shape[0])
            block = workspace[: stop - start]
            np.multiply(left_row, units[1][start:stop], out=block)
            np.sum(block, axis=1, out=out[row, start:stop])
    np.clip(out, -1, 1, out=out)
    out[~valid[0], :] = np.nan
    out[:, ~valid[1]] = np.nan
    return out


def _profile(source: _SeriesInput, lags: tuple[int, ...]) -> np.ndarray:
    if source.values.shape[0] == 0:
        return np.empty((0, len(lags)))
    result = temporal_statistics(
        source.values[:, :, None], lags=lags, resolution=source.resolution[:, None]
    )
    values = result.values["overlap_pearson"][:, 0, :]
    if source.constant is not None:
        values[source.constant] = np.nan
    return values


@dataclass(frozen=True)
class SeriesComparison:
    """Per-metric pair matrices, with their exact query and reference inputs.

    Matrices have shape ``(n_query, n_reference)``. Nonfinite entries cannot
    produce a match. ``overlap_pearson_rmse`` requires every requested profile
    coordinate to be defined; it is NOT the library's sample-ACF estimator.
    Raw MSE assumes common units and aligned time coordinates, supplied by the
    caller. Normalized zero-divisor rows are MSE-ineligible, not silently mixed
    with normalized units. Nonzero MSE outside float64 range is unavailable.
    """

    query: np.ndarray
    reference: np.ndarray
    metrics: tuple[str, ...]
    values: dict[str, np.ndarray]
    lags: tuple[int, ...]
    normalization: tuple[str, float] | None
    query_scale_available: np.ndarray
    reference_scale_available: np.ndarray

    def to_frame(self) -> pd.DataFrame:
        """One row per query/reference pair with metric values and availability."""
        import pandas as pd

        i, j = np.indices((self.query.shape[0], self.reference.shape[0]))
        data = {"query_index": i.ravel(), "reference_index": j.ravel()}
        for metric in self.metrics:
            data[metric] = self.values[metric].ravel()
            data[f"valid:{metric}"] = np.isfinite(self.values[metric]).ravel()
        return pd.DataFrame(data)


def _compare(
    query: _SeriesInput,
    reference: _SeriesInput,
    metrics: tuple[str, ...],
    lags: tuple[int, ...],
) -> SeriesComparison:
    if query.values.shape[1] != reference.values.shape[1]:
        raise ValueError("query and reference must have identical aligned time lengths")
    if query.normalization != reference.normalization:
        raise ValueError("query and reference must share raw units or the same quantile/mode")
    shape = (query.values.shape[0], reference.values.shape[0])
    values = {}
    for metric in metrics:
        if not shape[0] or not shape[1]:
            values[metric] = np.full(shape, np.nan)
        elif metric == "mse":
            out = _safe_mse(query.values, reference.values)
            out[~query.scale_available, :] = np.nan
            out[:, ~reference.scale_available] = np.nan
            values[metric] = out
        elif metric in ("level_r", "diff_r"):
            values[metric] = _correlations(query, reference, differences=metric == "diff_r")
        else:
            q_profile, r_profile = _profile(query, lags), _profile(reference, lags)
            q_valid = np.isfinite(q_profile).all(axis=1)
            r_valid = np.isfinite(r_profile).all(axis=1)
            out = np.full(shape, np.nan)
            if q_valid.any() and r_valid.any():
                out[np.ix_(q_valid, r_valid)] = _safe_mse(
                    q_profile[q_valid], r_profile[r_valid], root=True
                )
            values[metric] = out
    return SeriesComparison(
        query.values,
        reference.values,
        metrics,
        values,
        lags,
        query.normalization,
        query.scale_available,
        reference.scale_available,
    )


def compare_series(
    query: Any,
    reference: Any,
    *,
    lags: Sequence[int],
    metrics: Sequence[str] = SERIES_METRICS,
    query_resolution: Any = None,
    reference_resolution: Any = None,
) -> SeriesComparison:
    """Compare aligned ``(series,time)`` arrays or explicit normalization results.

    Correlations are signed. Level and difference constants use the original
    source precision, not a global absolute epsilon. ``lags`` specifies the
    overlap-Pearson profile; use an empty tuple when that metric is omitted.
    Neither this function nor :func:`nearest_series` infers roles or alignment.
    """
    names, selected_lags = _names(metrics, lags)
    return _compare(
        _series_input(query, query_resolution, "query"),
        _series_input(reference, reference_resolution, "reference"),
        names,
        selected_lags,
    )


@dataclass(frozen=True)
class NearestSeries:
    """Best finite score per metric, retaining its producing comparison.

    Minimize MSE/profile RMSE; maximize SIGNED correlation, not its absolute
    value. Unavailable results are NaN, index -1, candidate count 0. Exact ties
    choose the first reference position deterministically.
    """

    comparison: SeriesComparison
    scores: dict[str, np.ndarray]
    indices: dict[str, np.ndarray]
    candidate_counts: dict[str, np.ndarray]

    @property
    def metrics(self) -> tuple[str, ...]:
        return self.comparison.metrics

    def to_frame(self) -> pd.DataFrame:
        """One row per query series and metric, including finite candidate count."""
        import pandas as pd

        return pd.DataFrame(
            [
                (
                    i,
                    metric,
                    self.scores[metric][i],
                    self.indices[metric][i],
                    self.candidate_counts[metric][i],
                )
                for i in range(self.comparison.query.shape[0])
                for metric in self.metrics
            ],
            columns=("query_index", "metric", "score", "reference_index", "n_candidates"),
        )


def nearest_series(comparison: SeriesComparison) -> NearestSeries:
    """Select the best eligible candidate in an already computed comparison."""
    if not isinstance(comparison, SeriesComparison):
        raise TypeError("comparison must be SeriesComparison")
    n_query, n_reference = comparison.query.shape[0], comparison.reference.shape[0]
    scores, indices, counts = {}, {}, {}
    for metric in comparison.metrics:
        values = comparison.values[metric]
        finite = np.isfinite(values)
        count = finite.sum(axis=1)
        score, chosen = np.full(n_query, np.nan), np.full(n_query, -1, dtype=np.int64)
        if n_reference:
            minimize = metric in _MINIMIZE
            filled = np.where(finite, values, np.inf if minimize else -np.inf)
            positions = filled.argmin(axis=1) if minimize else filled.argmax(axis=1)
            available = count > 0
            chosen[available] = positions[available]
            score[available] = values[np.flatnonzero(available), positions[available]]
        scores[metric], indices[metric], counts[metric] = score, chosen, count
    return NearestSeries(comparison, scores, indices, counts)


def _unknown_group(value: Hashable) -> bool:
    if value is None:
        return True
    # timedelta64 is also an np.integer, but -1 is a valid duration, not an ID sentinel.
    if isinstance(value, (np.datetime64, np.timedelta64)):
        return bool(np.isnat(value))
    # Nullable metadata can be non-reflexive or have indeterminate equality.
    # Detect those scalars without importing their dataframe library.
    try:
        if not bool(value == value):
            return True
    except (TypeError, ValueError):
        return True
    if isinstance(value, (int, np.integer)):
        return bool(value == -1)
    if isinstance(value, (float, np.floating)):
        return not bool(np.isfinite(value)) or value == -1
    if isinstance(value, tuple):
        return any(_unknown_group(part) for part in value)
    return False


def _group_labels(groups: Any, count: int, name: str, *, required: bool) -> set[Hashable]:
    if groups is None:
        if required:
            raise ValueError(f"{name} is required to establish group-disjoint pools")
        return set()
    _unmasked(groups, name)
    if isinstance(groups, (str, bytes)):
        raise TypeError(f"{name} must contain one known group label per series")
    labels = list(groups)
    if len(labels) != count:
        raise ValueError(f"{name} must contain {count} group labels")
    for value in labels:
        try:
            hash(value)
        except TypeError as exc:
            raise TypeError(f"{name} must contain hashable group labels") from exc
        if required and _unknown_group(value):
            raise ValueError(f"{name} contains an unknown group; independence cannot be inferred")
    return set(labels)


@dataclass(frozen=True)
class SimilarityAssessment:
    """Query matching scores relative to held-out scores against the SAME bank.

    Percentiles and lower/upper tie bounds are fractions in [0,1], oriented as
    raw-score CDFs (a high correlation percentile means high correlation; a high
    MSE percentile means high error). They are not tail/origin probabilities or
    conformal p-values. Finite reference counts are metric-specific numbers of
    held-out SERIES, not claims of independent datasets.
    """

    query_matches: NearestSeries
    calibration_matches: NearestSeries
    percentiles: dict[str, np.ndarray]
    lower: dict[str, np.ndarray]
    upper: dict[str, np.ndarray]
    tie_counts: dict[str, np.ndarray]
    finite_reference_counts: dict[str, int]
    group_disjoint: bool

    @property
    def metrics(self) -> tuple[str, ...]:
        return self.query_matches.metrics

    def to_frame(self) -> pd.DataFrame:
        """One query/metric row with score, candidates, empirical rank and ties."""
        frame = self.query_matches.to_frame()
        for name, values in (
            ("percentile", self.percentiles),
            ("percentile_lower", self.lower),
            ("percentile_upper", self.upper),
            ("n_ties", self.tie_counts),
        ):
            frame[name] = [values[m][i] for i, m in zip(frame.query_index, frame.metric)]
        frame["n_reference"] = [self.finite_reference_counts[m] for m in frame.metric]
        return frame


def assess_series_similarity(
    query: Any,
    calibration: Any,
    bank: Any,
    *,
    lags: Sequence[int],
    metrics: Sequence[str] = SERIES_METRICS,
    query_groups: Any = None,
    calibration_groups: Any = None,
    bank_groups: Any = None,
    require_disjoint_groups: bool = True,
    query_resolution: Any = None,
    calibration_resolution: Any = None,
    bank_resolution: Any = None,
) -> SimilarityAssessment:
    """Place query best-match scores in a held-out matching-score distribution.

    Group labels must be in a common namespace across all pools; use tuples
    such as ``(source, cell)`` when source-local IDs would collide. By default,
    all three pools must have known, pairwise-disjoint group labels. Repeated
    labels WITHIN a pool are retained, never invented as independent singleton
    groups. Explicitly disabling this check makes no disjointness assurance.
    All pools share metric, normalization, lag and candidate eligibility rules.
    """
    if not isinstance(require_disjoint_groups, (bool, np.bool_)):
        raise TypeError("require_disjoint_groups must be boolean")
    names, selected_lags = _names(metrics, lags)
    q = _series_input(query, query_resolution, "query")
    c = _series_input(calibration, calibration_resolution, "calibration")
    b = _series_input(bank, bank_resolution, "bank")
    pools = [
        _group_labels(labels, data.values.shape[0], name, required=bool(require_disjoint_groups))
        for labels, data, name in (
            (query_groups, q, "query_groups"),
            (calibration_groups, c, "calibration_groups"),
            (bank_groups, b, "bank_groups"),
        )
    ]
    if require_disjoint_groups and any(pools[i] & pools[j] for i, j in ((0, 1), (0, 2), (1, 2))):
        raise ValueError("query, calibration and bank must be group-disjoint")
    query_matches = nearest_series(_compare(q, b, names, selected_lags))
    calibration_matches = nearest_series(_compare(c, b, names, selected_lags))
    percentiles, lower, upper, ties, counts = {}, {}, {}, {}, {}
    empty = np.empty(0, dtype=np.int64)
    for metric in names:
        values = query_matches.scores[metric]
        scores = calibration_matches.scores[metric]
        ranked = np.sort(scores[np.isfinite(scores)])
        counts[metric] = int(ranked.size)
        valid = np.isfinite(values)
        percentiles[metric] = _midranks(ranked, values, valid, empty, np.empty(0))
        lo, hi = np.full(values.shape, np.nan), np.full(values.shape, np.nan)
        tied = np.zeros(values.shape, dtype=np.int64)
        if ranked.size:
            below = np.searchsorted(ranked, values[valid], side="left")
            through = np.searchsorted(ranked, values[valid], side="right")
            lo[valid], hi[valid] = below / ranked.size, through / ranked.size
            tied[valid] = through - below
        lower[metric], upper[metric], ties[metric] = lo, hi, tied
    return SimilarityAssessment(
        query_matches,
        calibration_matches,
        percentiles,
        lower,
        upper,
        ties,
        counts,
        bool(require_disjoint_groups),
    )
