"""Signal-strength diagnostics for generated corpora (plan doc 05 fix).

A corpus can satisfy every schema contract and still be unlearnable: if the
true per-channel contribution barely varies over the task window, there is no
signal for the model to attribute, and R²/interval metrics on those channels
measure noise. This module quantifies, per direct (C->Y) channel, how much
signal the generator actually produced — so a weak-signal prior is caught at
GENERATION time (the summary is embedded in every corpus' ``diagnostics``)
instead of after a training run.

Metrics per (task, direct channel) pair
---------------------------------------
* ``spend_cv``      — CV of the observed channel (the model's input).
* ``spend_hf``      — high-frequency ratio ``std(diff x) / (sqrt(2)·std x)``:
  1 for white noise, -> 0 for a smooth drift.
* ``contrib_cv``    — CV of the true contribution target (flatness).
* ``contrib_hf``    — high-frequency ratio of the target.
* ``contrib_rel_std`` — target std / per-task sales scale (amplitude of the
  target in training-loss units).
* ``spearman``      — |rank correlation| between true-family-adstocked spend
  and the contribution (how much of the target is visible from the input).
* ``warmup_ratio``  — (max-min over the first ``l_max`` weeks) / (std of the
  rest): >> 1 flags the adstock zero-padding warmup artifact dominating the
  target (fixed by ``adstock_burn_in``).
* ``contrib_r2_explained_by_rest`` — contribution R² against an intercept,
  baseline, and the other direct-channel contributions.
* ``contrib_corr_baseline`` — signed contribution/baseline correlation.

``signal_summary`` reduces these to quantiles plus degenerate-target
fractions, and adds the per-task normalized sales-level magnitude
``abs(mean(sales))/sales_scale``. Targets are trained
sales_scale-normalized without centering, so only the magnitude of the level
matters.
"""

from __future__ import annotations

import numpy as np

__all__ = [
    "DEFAULT_GATE",
    "FRAC_KEYS",
    "METRIC_KEYS",
    "SIGNAL_METRIC_LAYOUT",
    "SIGNAL_METRIC_VERSION",
    "check_signal_gate",
    "contemporaneous_weight",
    "dense_signal_metrics",
    "per_channel_signal",
    "summarize_signal_metrics",
    "signal_summary",
]

_QS = (0.1, 0.5, 0.9)
SIGNAL_METRIC_VERSION = 3

#: Per-pair metrics emitted by :func:`per_channel_signal` and summarized (as
#: ``<key>_quantiles``) by :func:`signal_summary`. Single source of truth —
#: report/gate tooling should import this instead of re-listing names.
METRIC_KEYS: tuple[str, ...] = (
    "spend_cv",
    "spend_hf",
    "contrib_cv",
    "contrib_hf",
    "contrib_rel_std",
    "spearman",
    "warmup_ratio",
    "contrib_r2_explained_by_rest",
    "contrib_corr_baseline",
)

# This is a persisted feature layout: append only.
SIGNAL_METRIC_LAYOUT: tuple[str, ...] = METRIC_KEYS

#: Degenerate-target fractions emitted by :func:`signal_summary`.
FRAC_KEYS: tuple[str, ...] = (
    "frac_contrib_cv_lt_005",
    "frac_contrib_cv_lt_010",
    "frac_contrib_hf_lt_015",
    "frac_contrib_rel_std_lt_001",
    "frac_contrib_r2_gt_095",
    "frac_spearman_lt_03",
    "frac_warmup_gt_3",
)

#: Minimum-signal thresholds (fraction <= value) for a healthy training
#: corpus; see :func:`check_signal_gate`. Tuned on the plan-06 pool sizes
#: (L1 reference ~3%/1.5% on the cv/hf checks).
DEFAULT_GATE: dict[str, float] = {
    "frac_contrib_cv_lt_010": 0.15,  # at most 15% near-flat targets
    "frac_contrib_hf_lt_015": 0.15,  # at most 15% targets without weekly variation
    # with burn-in the adstock artifact is gone, but a REAL pulse in the first
    # l_max weeks legitimately trips the 3x ratio — allow a modest tail
    "frac_warmup_gt_3": 0.10,
    "frac_spearman_lt_03": 0.15,  # target visible from the observed spend
    "frac_contrib_rel_std_lt_001": 0.10,  # targets too small to matter in loss units
    "frac_contrib_r2_gt_095": 0.10,  # targets a linear combination of baseline + other channels
}


def _validated_integer(value: object, name: str, minimum: int) -> int:
    """Return an integer value meeting its inclusive lower bound."""
    if (
        isinstance(value, (bool, np.bool_))
        or not isinstance(value, (int, np.integer))
        or value < minimum
    ):
        raise ValueError(f"{name} must be an integer >= {minimum}")
    return int(value)


def _validated_adstock_family(value: object, name: str = "family") -> int:
    """Return one supported scalar adstock-family identifier."""
    family = _validated_integer(value, name, 0)
    if family not in (0, 1, 2):
        raise ValueError(f"{name} must be an integer in {{0, 1, 2}}")
    return family


def _is_finite_real(value: object) -> bool:
    """Whether ``value`` is a finite, non-boolean scalar number."""
    return (
        not isinstance(value, (bool, np.bool_))
        and isinstance(value, (int, float, np.integer, np.floating))
        and bool(np.isfinite(value))
    )


def _metadata_float_array(value: object, name: str, shape: tuple[int, int]) -> np.ndarray:
    """Return float64 metadata with the required task/channel shape."""
    try:
        array = np.asarray(value, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be numeric with shape {shape}") from exc
    if array.shape != shape:
        raise ValueError(f"{name} must have shape {shape}")
    return array


def _validated_adstock_family_array(value: object, shape: tuple[int, int]) -> np.ndarray:
    """Return validated integer adstock-family metadata with the required shape."""
    try:
        family_raw = np.asarray(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"adstock_family must have shape {shape}") from exc
    if family_raw.shape != shape:
        raise ValueError(f"adstock_family must have shape {shape}")
    if np.issubdtype(family_raw.dtype, np.bool_):
        raise ValueError("adstock_family entries must be integers in {0, 1, 2}")
    try:
        family_float = family_raw.astype(np.float64, copy=False)
    except (TypeError, ValueError) as exc:
        raise ValueError("adstock_family entries must be integers in {0, 1, 2}") from exc
    if (
        not np.isfinite(family_float).all()
        or not np.equal(family_float, np.floor(family_float)).all()
        or not np.isin(family_float, (0.0, 1.0, 2.0)).all()
    ):
        raise ValueError("adstock_family entries must be integers in {0, 1, 2}")
    return family_float.astype(np.int8)


def _validated_adstock_metadata(
    adstock_family: object,
    adstock_alpha: object,
    weibull_lam: object,
    weibull_k: object,
    shape: tuple[int, int],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Validate adstock metadata and return arrays suitable for recomputation."""
    family = _validated_adstock_family_array(adstock_family, shape)

    alpha = _metadata_float_array(adstock_alpha, "adstock_alpha", shape)
    lam = _metadata_float_array(weibull_lam, "weibull_lam", shape)
    k = _metadata_float_array(weibull_k, "weibull_k", shape)
    geometric = family == 1
    weibull = family == 2
    if geometric.any() and (
        not np.isfinite(alpha[geometric]).all()
        or (alpha[geometric] < 0.0).any()
        or (alpha[geometric] > 1.0).any()
    ):
        raise ValueError("adstock_alpha must be finite in [0, 1] where adstock_family == 1")
    if weibull.any() and (not np.isfinite(lam[weibull]).all() or (lam[weibull] <= 0.0).any()):
        raise ValueError("weibull_lam must be finite and > 0 where adstock_family == 2")
    if weibull.any() and (not np.isfinite(k[weibull]).all() or (k[weibull] <= 0.0).any()):
        raise ValueError("weibull_k must be finite and > 0 where adstock_family == 2")
    return family, alpha, lam, k


def _validated_sales_scale(sales: np.ndarray, sales_scale: np.ndarray | None) -> np.ndarray:
    """Return a finite, strictly positive per-task sales scale."""
    try:
        scale = (
            sales.std(axis=1) if sales_scale is None else np.asarray(sales_scale, dtype=np.float64)
        )
    except (TypeError, ValueError) as exc:
        raise ValueError(
            "sales_scale must be finite and strictly positive with shape (N,)"
        ) from exc
    invalid_scale = (
        scale.shape != (sales.shape[0],) or not np.isfinite(scale).all() or (scale <= 0.0).any()
    )
    if invalid_scale:
        raise ValueError("sales_scale must be finite and strictly positive with shape (N,)")
    return scale


def _coefficient_of_variation(x: np.ndarray, sd: float) -> tuple[float, bool]:
    """Return a finite CV value and whether it is defined."""
    mean_abs = abs(float(x.mean()))
    if mean_abs != 0.0:
        return float(sd / mean_abs), True
    return (0.0, True) if np.all(x == 0.0) else (0.0, False)


def _hf_ratio(x: np.ndarray) -> np.ndarray:
    """High-frequency ratio along the last axis: 1 = white noise, ->0 smooth."""
    sd = x.std(axis=-1)
    diff_sd = np.diff(x, axis=-1).std(axis=-1)
    out = np.zeros_like(sd)
    np.divide(diff_sd, np.sqrt(2.0) * sd, out=out, where=sd != 0.0)
    return np.asarray(out)


def _spearman_abs(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """|Spearman rank correlation| along the last axis, vectorized.

    Average ranks for ties (scipy convention) — ordinal ranks would give a
    constant series ranks ``0..T-1`` and score |rho| ~ 1 against any trending
    series, hiding exactly the flat targets this module exists to flag. A
    (near-)constant series has zero rank variance and returns 0.0: no signal
    is visible from it (scipy returns NaN there; 0.0 is the gate-friendly
    encoding of the same fact).
    """
    from scipy.stats import rankdata  # lazy: keep module import numpy-light

    ra = rankdata(a, method="average", axis=-1)
    rb = rankdata(b, method="average", axis=-1)
    ra = ra - ra.mean(axis=-1, keepdims=True)
    rb = rb - rb.mean(axis=-1, keepdims=True)
    denom = np.sqrt((ra**2).sum(axis=-1) * (rb**2).sum(axis=-1))
    num = (ra * rb).sum(axis=-1)
    out = np.zeros_like(denom)
    np.divide(np.abs(num), denom, out=out, where=denom != 0.0)
    return np.asarray(out)


def _adstock_weights(
    family: int, alpha: float, lam: float, shape: float, l_max: int
) -> np.ndarray | None:
    """Return normalized causal weights, or ``None`` for identity/degenerate kernels."""
    family = _validated_adstock_family(family)
    l_max = _validated_integer(l_max, "l_max", 1)
    if family == 0 or l_max == 1:
        return None
    lag = np.arange(l_max, dtype=np.float64)
    if family == 1:
        if not _is_finite_real(alpha) or not 0.0 <= float(alpha) <= 1.0:
            raise ValueError("alpha must be finite in [0, 1]")
        weights = np.power(float(alpha), lag)
    else:
        if not _is_finite_real(lam) or float(lam) <= 0.0:
            raise ValueError("lam must be finite and > 0")
        if not _is_finite_real(shape) or float(shape) <= 0.0:
            raise ValueError("shape must be finite and > 0")
        t = lag + 1.0
        with np.errstate(over="ignore", under="ignore", invalid="ignore"):
            weights = (
                (shape / lam) * np.power(t / lam, shape - 1.0) * np.exp(-np.power(t / lam, shape))
            )
        span = weights.max() - weights.min()
        if not np.isfinite(span) or span == 0.0:
            return None
        weights = (weights - weights.min()) / span
    total = weights.sum()
    if not np.isfinite(total) or total == 0.0:
        return None
    return np.asarray(weights / total, dtype=np.float64)


def contemporaneous_weight(family: int, alpha: float, lam: float, k: float, l_max: int) -> float:
    """Normalized adstock weight on the current week (zero lag).

    Returns
    -------
    float
        ``weights[0]`` for a nondegenerate nonidentity kernel, 1.0 for an
        identity kernel, and 0.0 for a degenerate nonidentity kernel.
    """
    family = _validated_adstock_family(family)
    l_max = _validated_integer(l_max, "l_max", 1)
    weights = _adstock_weights(family, alpha, lam, k, l_max)
    if family == 0 or l_max == 1:
        return 1.0
    return 0.0 if weights is None else float(weights[0])


def _adstock_numpy(
    x: np.ndarray, family: int, alpha: float, lam: float, shape: float, l_max: int
) -> np.ndarray:
    """The generator's normalized, causal adstock for one reported series.

    ``pymc_marketing.weibull_adstock(type="PDF")`` min-max rescales the sampled
    density before sum-normalizing it. The resulting kernel is therefore not a
    Weibull pdf and has ``min(weights) == 0`` exactly. Under the default prior
    (``lam ~ U(2, 8)``, ``k ~ U(1.5, 4)``, ``l_max = 8``), 45.0% of Weibull
    channels have zero current-week weight (200k draws), mean lag-1 weight is
    0.065, and 38.5% peak at lag >= 5.
    """
    family = _validated_adstock_family(family)
    l_max = _validated_integer(l_max, "l_max", 1)
    weights = _adstock_weights(family, alpha, lam, shape, l_max)
    if family == 0 or l_max == 1:
        return x.copy()
    if weights is None:
        return np.zeros_like(x, dtype=np.float64)
    return np.convolve(x, weights, mode="full")[: x.size]


def dense_signal_metrics(
    spend: np.ndarray,
    contributions: np.ndarray,
    sales: np.ndarray,
    baseline: np.ndarray,
    cy_mask: np.ndarray,
    *,
    sales_scale: np.ndarray | None = None,
    adstock_family: np.ndarray | None = None,
    adstock_alpha: np.ndarray | None = None,
    weibull_lam: np.ndarray | None = None,
    weibull_k: np.ndarray | None = None,
    l_max: int = 8,
    adstock_burn_in: int = 0,
) -> tuple[np.ndarray, np.ndarray]:
    """Return persisted dense ``(N, K, L)`` metrics and uint8 validity.

    Inputs are deliberately the final persisted arrays; callers must not pass
    pre-cast draw values. Ineligible/padded cells are exactly zero and invalid.
    An all-zero series has valid CV 0.0. A nonconstant zero-mean series has an
    undefined/infinite CV and is invalid; generated spend and contribution
    targets are non-negative, so only the all-zero convention is reachable in
    a corpus.
    """
    spend = np.asarray(spend, dtype=np.float64)
    contributions = np.asarray(contributions, dtype=np.float64)
    sales = np.asarray(sales, dtype=np.float64)
    baseline = np.asarray(baseline, dtype=np.float64)
    mask = np.asarray(cy_mask, dtype=bool)
    if spend.ndim != 3:
        raise ValueError("spend must have shape (N, T, K)")
    N, T, K = spend.shape
    if contributions.shape != (N, T, K) or sales.shape != (N, T) or baseline.shape != (N, T):
        raise ValueError("signal source arrays have incompatible shapes")
    if mask.shape != (N, K):
        raise ValueError(f"cy_mask shape {mask.shape} != {(N, K)}")
    if not all(np.isfinite(a).all() for a in (spend, contributions, sales, baseline)):
        raise ValueError("signal source arrays must be finite")
    l_max = _validated_integer(l_max, "l_max", 1)
    adstock_burn_in = _validated_integer(adstock_burn_in, "adstock_burn_in", 0)
    scale = _validated_sales_scale(sales, sales_scale)
    metadata_shape = (N, K)
    family_input = (
        np.zeros(metadata_shape, dtype=np.int8) if adstock_family is None else adstock_family
    )
    alpha_input = np.zeros(metadata_shape) if adstock_alpha is None else adstock_alpha
    lam_input = np.ones(metadata_shape) if weibull_lam is None else weibull_lam
    k_input = np.ones(metadata_shape) if weibull_k is None else weibull_k
    fam, alpha, wlam, wk = _validated_adstock_metadata(
        family_input, alpha_input, lam_input, k_input, metadata_shape
    )
    metrics = np.zeros((N, K, len(SIGNAL_METRIC_LAYOUT)), dtype=np.float32)
    valid = np.zeros_like(metrics, dtype=np.uint8)
    metric_index = {name: i for i, name in enumerate(SIGNAL_METRIC_LAYOUT)}
    for n, k in zip(*np.nonzero(mask)):
        x, y = spend[n, :, k], contributions[n, :, k]
        std_x, std_y = x.std(), y.std()
        spend_cv, spend_cv_valid = _coefficient_of_variation(x, std_x)
        contrib_cv, contrib_cv_valid = _coefficient_of_variation(y, std_y)
        values = {
            "spend_cv": (spend_cv, spend_cv_valid),
            "contrib_cv": (contrib_cv, contrib_cv_valid),
            "contrib_rel_std": (std_y / scale[n], True),
        }
        for name, (value, is_valid) in values.items():
            metrics[n, k, metric_index[name]] = value
            valid[n, k, metric_index[name]] = is_valid
        if T >= 2:
            metrics[n, k, metric_index["spend_hf"]] = _hf_ratio(x[None])[0]
            metrics[n, k, metric_index["contrib_hf"]] = _hf_ratio(y[None])[0]
            valid[n, k, [metric_index["spend_hf"], metric_index["contrib_hf"]]] = 1
        ad_x = _adstock_numpy(x, fam[n, k], alpha[n, k], wlam[n, k], wk[n, k], l_max)
        # Visibility intentionally compares once-adstocked observed spend; do
        # not adstock a contribution that already contains the response.
        if T - l_max >= 3:
            metrics[n, k, metric_index["spearman"]] = _spearman_abs(
                ad_x[l_max:][None], y[l_max:][None]
            )[0]
            valid[n, k, metric_index["spearman"]] = 1
        if adstock_burn_in < l_max and T >= l_max + 3:
            warm_range = y[:l_max].max() - y[:l_max].min()
            suffix_std = y[l_max:].std()
            if suffix_std != 0.0:
                metrics[n, k, metric_index["warmup_ratio"]] = warm_range / suffix_std
                valid[n, k, metric_index["warmup_ratio"]] = 1
        if T >= 2:
            # Explained by the full reported window: intercept, baseline, and
            # all *other* active direct contributions.
            others = [j for j in np.flatnonzero(mask[n]) if j != k]
            centered_y, centered_b = y - y.mean(), baseline[n] - baseline[n].mean()
            centered_predictors = [centered_b] + [
                contributions[n, :, j] - contributions[n, :, j].mean() for j in others
            ]
            design = np.column_stack(centered_predictors)
            sst = float(centered_y @ centered_y)
            if sst == 0.0:
                # Persisted constants are exactly reproducible by the intercept.
                metrics[n, k, metric_index["contrib_r2_explained_by_rest"]] = 1.0
                valid[n, k, metric_index["contrib_r2_explained_by_rest"]] = 1
            else:
                design_rank = 1 + np.linalg.matrix_rank(design)
                if T > design_rank:
                    fitted = design @ np.linalg.lstsq(design, centered_y, rcond=None)[0]
                    ss_res = float(np.sum((centered_y - fitted) ** 2))
                    r2 = float(np.clip(1.0 - ss_res / sst, 0.0, 1.0))
                    metrics[n, k, metric_index["contrib_r2_explained_by_rest"]] = r2
                    valid[n, k, metric_index["contrib_r2_explained_by_rest"]] = 1
            denom = np.sqrt(np.sum(centered_y**2) * np.sum(centered_b**2))
            metrics[n, k, metric_index["contrib_corr_baseline"]] = (
                centered_y @ centered_b / denom if denom != 0.0 else 0.0
            )
            valid[n, k, metric_index["contrib_corr_baseline"]] = 1
    return metrics, valid


def per_channel_signal(
    spend: np.ndarray,
    contributions: np.ndarray,
    sales: np.ndarray,
    cy_mask: np.ndarray,
    *,
    sales_scale: np.ndarray | None = None,
    l_max: int = 8,
    baseline: np.ndarray | None = None,
    adstock_family: np.ndarray | None = None,
    adstock_alpha: np.ndarray | None = None,
    weibull_lam: np.ndarray | None = None,
    weibull_k: np.ndarray | None = None,
    adstock_burn_in: int = 0,
) -> dict[str, np.ndarray]:
    """Per-(task, direct channel) signal metrics.

    Parameters
    ----------
    spend : (N, T, K) observed channels (model input).
    contributions : (N, T, K) true contribution targets.
    sales : (N, T) sales series.
    cy_mask : (N, K) bool/0-1 — which channels are direct (C->Y) AND active.
    sales_scale : (N,) optional — per-task target normalizer; defaults to the
        full-series sales std.
    l_max : adstock length, defines the warmup window for ``warmup_ratio``.

    Returns
    -------
    dict of 1-D float arrays, one entry per (task, direct channel) pair, plus
    ``task_idx`` locating each pair.
    """
    spend = np.asarray(spend)
    N, T, K = spend.shape
    mask = np.asarray(cy_mask, dtype=bool)
    dense, valid = dense_signal_metrics(
        spend,
        contributions,
        sales,
        np.zeros((N, T), dtype=spend.dtype) if baseline is None else baseline,
        mask,
        sales_scale=sales_scale,
        l_max=l_max,
        adstock_family=adstock_family,
        adstock_alpha=adstock_alpha,
        weibull_lam=weibull_lam,
        weibull_k=weibull_k,
        adstock_burn_in=adstock_burn_in,
    )
    if baseline is None:
        for name in ("contrib_r2_explained_by_rest", "contrib_corr_baseline"):
            index = SIGNAL_METRIC_LAYOUT.index(name)
            dense[..., index] = 0.0
            valid[..., index] = 0
    task_idx = np.broadcast_to(np.arange(N)[:, None], (N, K))
    out = {"task_idx": task_idx[mask].astype(np.float64)}
    for i, key in enumerate(SIGNAL_METRIC_LAYOUT):
        # Select dense result rows for the legacy flattened API; this is not a
        # second metric computation.
        out[key] = dense[..., i][mask]
        out[f"{key}_valid"] = valid[..., i][mask]
    return out


def summarize_signal_metrics(
    metrics: np.ndarray,
    valid: np.ndarray,
    sales: np.ndarray,
    cy_mask: np.ndarray,
    *,
    sales_scale: np.ndarray | None = None,
    l_max: int = 8,
    adstock_burn_in: int = 0,
    adstock_family: np.ndarray | None = None,
    adstock_alpha: np.ndarray | None = None,
    weibull_lam: np.ndarray | None = None,
    weibull_k: np.ndarray | None = None,
) -> dict:
    """Summarize persisted dense metrics using their per-metric validity.

    This is deliberately separate from metric calculation so corpus diagnostics
    can be derived from exactly the float32 arrays written to disk.

    ``response_warmup_weeks`` identifies reported weeks whose response depends
    on unpersisted pre-window spend. With burn-in, reported week ``t`` for a
    non-identity adstock kernel reaches before the persisted window exactly
    when ``t < l_max - 1``. The count is therefore ``l_max - 1`` only when
    burn-in is present and at least one ``cy_mask``-eligible direct channel has
    a non-identity kernel; otherwise it is zero. If ``adstock_family`` is
    omitted, the summary conservatively reports ``l_max - 1`` with burn-in
    because it cannot establish that every eligible direct kernel is identity.
    ``support_mask`` is the temporal train/query split, not this
    response-warmup rule.

    Parameters
    ----------
    metrics, valid : np.ndarray
        Persisted dense metric values and their binary per-metric validity
        masks, each shaped ``(N, K, len(SIGNAL_METRIC_LAYOUT))``.
    sales : np.ndarray
        Persisted sales array with shape ``(N, T)``.
    cy_mask : np.ndarray
        Boolean ``(N, K)`` mask selecting eligible direct-channel pairs.
    sales_scale : np.ndarray, optional
        Positive persisted per-task target scales. When omitted, scales are
        computed from ``sales``.
    l_max : int
        Adstock kernel length.
    adstock_burn_in : int
        Number of generated leading burn-in weeks.
    adstock_family, adstock_alpha, weibull_lam, weibull_k : np.ndarray, optional
        Per-(task, channel) adstock metadata. All four must be supplied to
        calculate ``frac_zero_contemporaneous_weight``; omitting any sets that
        diagnostic to ``None``. ``adstock_family`` also determines whether
        active kernels require response warmup; when it is omitted, the
        summary conservatively reports ``l_max - 1`` with burn-in.
    """
    metrics = np.asarray(metrics)
    valid = np.asarray(valid)
    sales = np.asarray(sales, dtype=np.float64)
    mask = np.asarray(cy_mask, dtype=bool)
    expected = mask.shape + (len(SIGNAL_METRIC_LAYOUT),)
    if metrics.shape != expected or valid.shape != expected:
        raise ValueError(f"metrics and validity must have shape {expected}")
    if sales.ndim != 2 or sales.shape[0] != mask.shape[0]:
        raise ValueError("sales must have shape (N, T)")
    if not np.isfinite(sales).all():
        raise ValueError("sales must be finite")
    if not np.isfinite(metrics).all() or not np.isin(valid, (0, 1)).all():
        raise ValueError("metrics must be finite and validity must be binary")

    l_max = _validated_integer(l_max, "l_max", 1)
    adstock_burn_in = _validated_integer(adstock_burn_in, "adstock_burn_in", 0)
    scale = _validated_sales_scale(sales, sales_scale)
    level_ratio = np.abs(sales.mean(axis=1)) / scale
    n_pairs = int(mask.sum())
    adstock_metadata = (adstock_family, adstock_alpha, weibull_lam, weibull_k)
    fam: np.ndarray | None = None
    frac_zero_contemporaneous_weight: float | None = None
    if n_pairs and all(value is not None for value in adstock_metadata):
        fam, alpha, wlam, wk = _validated_adstock_metadata(
            adstock_family, adstock_alpha, weibull_lam, weibull_k, mask.shape
        )
        frac_zero_contemporaneous_weight = float(
            sum(
                contemporaneous_weight(fam[n, k], alpha[n, k], wlam[n, k], wk[n, k], l_max) < 1e-9
                for n, k in zip(*np.nonzero(mask))
            )
            / n_pairs
        )
    elif n_pairs and adstock_family is not None:
        fam = _validated_adstock_family_array(adstock_family, mask.shape)
    response_warmup_weeks = (
        l_max - 1
        if adstock_burn_in > 0 and n_pairs > 0 and (fam is None or np.any((fam != 0) & mask))
        else 0
    )
    out: dict = {
        "n_direct_channels": n_pairs,
        "response_warmup_weeks": response_warmup_weeks,
        "frac_zero_contemporaneous_weight": frac_zero_contemporaneous_weight,
    }
    metric_index = {name: i for i, name in enumerate(SIGNAL_METRIC_LAYOUT)}
    for key, i in metric_index.items():
        vals = metrics[..., i][mask & valid[..., i].astype(bool)]
        out[f"{key}_quantiles"] = {
            f"q{int(q * 100)}": (float(np.quantile(vals, q)) if vals.size else None) for q in _QS
        }

    fraction_specs = {
        "frac_contrib_cv_lt_005": ("contrib_cv", lambda x: x < 0.05),
        "frac_contrib_cv_lt_010": ("contrib_cv", lambda x: x < 0.10),
        "frac_contrib_hf_lt_015": ("contrib_hf", lambda x: x < 0.15),
        "frac_contrib_rel_std_lt_001": ("contrib_rel_std", lambda x: x < 0.01),
        "frac_contrib_r2_gt_095": ("contrib_r2_explained_by_rest", lambda x: x > 0.95),
        "frac_spearman_lt_03": ("spearman", lambda x: x < 0.3),
        "frac_warmup_gt_3": ("warmup_ratio", lambda x: x > 3.0),
    }
    for key, (metric, predicate) in fraction_specs.items():
        i = metric_index[metric]
        vals = metrics[..., i][mask & valid[..., i].astype(bool)]
        # A full burn-in makes warmup explicitly N/A, rather than absent due
        # to too few observations.  Its aggregate artifact fraction is 0.
        out[key] = (
            0.0
            if metric == "warmup_ratio" and adstock_burn_in >= l_max
            else (float(predicate(vals).mean()) if vals.size else None)
        )
    out["sales_level_ratio_quantiles"] = {
        f"q{int(q * 100)}": (float(np.quantile(level_ratio, q)) if level_ratio.size else None)
        for q in _QS
    }
    return out


def signal_summary(
    spend: np.ndarray,
    contributions: np.ndarray,
    sales: np.ndarray,
    cy_mask: np.ndarray,
    *,
    sales_scale: np.ndarray | None = None,
    l_max: int = 8,
    baseline: np.ndarray | None = None,
    adstock_family: np.ndarray | None = None,
    adstock_alpha: np.ndarray | None = None,
    weibull_lam: np.ndarray | None = None,
    weibull_k: np.ndarray | None = None,
    adstock_burn_in: int = 0,
) -> dict:
    """Corpus-level signal report: metric quantiles + degenerate fractions.

    JSON-serializable; embedded under ``diagnostics["signal"]`` by
    ``sample_prior_predictive`` so weak-signal priors are visible in every corpus /
    shard manifest. Interpretation guide:

    * ``frac_contrib_cv_lt_010`` — share of direct channels whose true
      contribution is near-flat (CV < 0.10). Some flat channels are realistic;
      a majority means the prior generates unlearnable attribution tasks.
    * ``frac_contrib_hf_lt_015`` — share with essentially no week-to-week
      variation (smooth drift only).
    * ``frac_spearman_lt_03`` — share whose target is invisible from the
      observed spend.
    * ``frac_warmup_gt_3`` — share whose largest feature is the adstock
      zero-padding warmup (generator artifact, fixed by ``adstock_burn_in``).
    * ``frac_contrib_rel_std_lt_001`` — share whose contribution amplitude is
      too small to matter in training-loss units.
    * ``frac_contrib_r2_gt_095`` — share whose target is a linear combination
      of baseline and the other channels, so only the sum is identified.
    """
    spend = np.asarray(spend)
    baseline_array = np.zeros(spend.shape[:2], dtype=spend.dtype) if baseline is None else baseline
    metrics, valid = dense_signal_metrics(
        spend,
        contributions,
        sales,
        baseline_array,
        cy_mask,
        sales_scale=sales_scale,
        l_max=l_max,
        adstock_family=adstock_family,
        adstock_alpha=adstock_alpha,
        weibull_lam=weibull_lam,
        weibull_k=weibull_k,
        adstock_burn_in=adstock_burn_in,
    )
    if baseline is None:
        for name in ("contrib_r2_explained_by_rest", "contrib_corr_baseline"):
            index = SIGNAL_METRIC_LAYOUT.index(name)
            metrics[..., index] = 0.0
            valid[..., index] = 0
    return summarize_signal_metrics(
        metrics,
        valid,
        sales,
        cy_mask,
        sales_scale=sales_scale,
        l_max=l_max,
        adstock_burn_in=adstock_burn_in,
        adstock_family=adstock_family,
        adstock_alpha=adstock_alpha,
        weibull_lam=weibull_lam,
        weibull_k=weibull_k,
    )


def check_signal_gate(signal: dict, gate: dict[str, float] | None = None) -> tuple[bool, list[str]]:
    """Check a ``signal_summary`` block against minimum-signal thresholds.

    Returns ``(ok, lines)`` where ``lines`` are human-readable PASS/FAIL rows.
    A missing or ``None`` metric (e.g. a corpus with zero direct channels)
    FAILS loudly — no signal measured is not a pass. Generation pipelines can
    call this on each shard's ``diagnostics["signal"]`` to reject weak-signal
    priors at generation time.
    """
    thresholds = DEFAULT_GATE if gate is None else gate
    ok = True
    lines: list[str] = []
    for key, thresh in thresholds.items():
        val = signal.get(key)
        if val is None:
            ok = False
            lines.append(f"[FAIL] {key} missing (no direct channels measured)")
            continue
        passed = val <= thresh
        ok &= passed
        lines.append(f"[{'PASS' if passed else 'FAIL'}] {key} = {val:.1%} (<= {thresh:.0%})")
    return ok, lines
