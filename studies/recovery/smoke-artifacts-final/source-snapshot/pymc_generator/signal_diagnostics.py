"""Signal-strength diagnostics for generated corpora.

A valid corpus can still contain weak or redundant signals. These metrics
quantify variation, relative amplitude, and dependence for direct C->Y
treatments, before a consumer trains on the corpus. Summaries are embedded in
``diagnostics``; screening with :func:`check_signal_gate` is optional.
Neither the summaries nor a passing gate establish causal identification.

Metrics per (task, direct treatment) pair
---------------------------------------
* ``treatment_cv``      — CV of the observed treatment (the model's input).
* ``treatment_hf``      — high-frequency ratio ``std(diff x) / (sqrt(2)·std x)``:
  1 for white noise, -> 0 for a smooth drift.
* ``contrib_cv``    — CV of the true contribution target (flatness).
* ``contrib_hf``    — high-frequency ratio of the target.
* ``contrib_rel_std`` — target std / per-task outcome scale (amplitude of the
  target in training-loss units).
* ``spearman``      — |rank correlation| between true-family-carryovered treatment
  and the contribution (how much of the target is visible from the input).
* ``warmup_ratio``  — (max-min over the first ``l_max`` weeks) / (std of the
  rest): >> 1 flags the carryover zero-padding warmup artifact dominating the
  target (fixed by ``carryover_burn_in``).
* ``contrib_r2_explained_by_rest`` — contribution R² against an intercept,
  baseline, and the other direct-treatment contributions.
* ``contrib_corr_baseline`` — signed contribution/baseline correlation.

``signal_summary`` reduces these to quantiles plus degenerate-target
fractions, and adds the per-task normalized outcome-level magnitude
``abs(mean(outcome))/outcome_scale``. Targets are trained
outcome_scale-normalized without centering, so only the magnitude of the level
matters.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable

import numpy as np

__all__ = [
    "DEFAULT_GATE",
    "FRAC_KEYS",
    "METRIC_KEYS",
    "SIGNAL_METRIC_LAYOUT",
    "SIGNAL_METRIC_VERSION",
    "admitted_response_support_weeks",
    "check_signal_gate",
    "contemporaneous_weight",
    "dense_signal_metrics",
    "per_treatment_signal",
    "response_support_weeks",
    "summarize_signal_metrics",
    "signal_summary",
]

_QS = (0.1, 0.5, 0.9)
SIGNAL_METRIC_VERSION = 3

#: Per-pair metrics emitted by :func:`per_treatment_signal` and summarized (as
#: ``<key>_quantiles``) by :func:`signal_summary`. Single source of truth —
#: report/gate tooling should import this instead of re-listing names.
METRIC_KEYS: tuple[str, ...] = (
    "treatment_cv",
    "treatment_hf",
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

_FRACTION_SPECS: dict[str, tuple[str, Callable[[np.ndarray], np.ndarray]]] = {
    "frac_contrib_cv_lt_005": ("contrib_cv", lambda x: x < 0.05),
    "frac_contrib_cv_lt_010": ("contrib_cv", lambda x: x < 0.10),
    "frac_contrib_hf_lt_015": ("contrib_hf", lambda x: x < 0.15),
    "frac_contrib_rel_std_lt_001": ("contrib_rel_std", lambda x: x < 0.01),
    "frac_contrib_r2_gt_095": ("contrib_r2_explained_by_rest", lambda x: x > 0.95),
    "frac_spearman_lt_03": ("spearman", lambda x: x < 0.3),
    "frac_warmup_gt_3": ("warmup_ratio", lambda x: x > 3.0),
}

#: Degenerate-target fractions emitted by :func:`signal_summary`.
FRAC_KEYS: tuple[str, ...] = tuple(_FRACTION_SPECS)

#: Default opt-in screening thresholds (fraction <= value).
#: See :func:`check_signal_gate`; these are not universal learnability bounds.
DEFAULT_GATE: dict[str, float] = {
    "frac_contrib_cv_lt_010": 0.15,  # at most 15% near-flat targets
    "frac_contrib_hf_lt_015": 0.15,  # at most 15% targets without weekly variation
    # with burn-in the carryover artifact is gone, but a REAL pulse in the first
    # l_max weeks legitimately trips the 3x ratio — allow a modest tail
    "frac_warmup_gt_3": 0.10,
    "frac_spearman_lt_03": 0.15,  # target visible from the observed treatment
    "frac_contrib_rel_std_lt_001": 0.10,  # targets too small to matter in loss units
    "frac_contrib_r2_gt_095": 0.10,  # targets a linear combination of baseline + other treatments
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


def _validated_carryover_family(value: object, name: str = "family") -> int:
    """Return one supported scalar carryover-family identifier."""
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
    """Return float64 metadata with the required task/treatment shape."""
    try:
        array = np.asarray(value, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be numeric with shape {shape}") from exc
    if array.shape != shape:
        raise ValueError(f"{name} must have shape {shape}")
    return array


def _validated_carryover_family_array(value: object, shape: tuple[int, int]) -> np.ndarray:
    """Return validated integer carryover-family metadata with the required shape."""
    try:
        family_raw = np.asarray(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"carryover_family must have shape {shape}") from exc
    if family_raw.shape != shape:
        raise ValueError(f"carryover_family must have shape {shape}")
    if np.issubdtype(family_raw.dtype, np.bool_):
        raise ValueError("carryover_family entries must be integers in {0, 1, 2}")
    try:
        family_float = family_raw.astype(np.float64, copy=False)
    except (TypeError, ValueError) as exc:
        raise ValueError("carryover_family entries must be integers in {0, 1, 2}") from exc
    if (
        not np.isfinite(family_float).all()
        or not np.equal(family_float, np.floor(family_float)).all()
        or not np.isin(family_float, (0.0, 1.0, 2.0)).all()
    ):
        raise ValueError("carryover_family entries must be integers in {0, 1, 2}")
    return family_float.astype(np.int8)


def _validated_carryover_metadata(
    carryover_family: object,
    carryover_alpha: object,
    weibull_lam: object,
    weibull_k: object,
    shape: tuple[int, int],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Validate carryover metadata and return arrays suitable for recomputation."""
    family = _validated_carryover_family_array(carryover_family, shape)

    alpha = _metadata_float_array(carryover_alpha, "carryover_alpha", shape)
    lam = _metadata_float_array(weibull_lam, "weibull_lam", shape)
    k = _metadata_float_array(weibull_k, "weibull_k", shape)
    geometric = family == 1
    weibull = family == 2
    if geometric.any() and (
        not np.isfinite(alpha[geometric]).all()
        or (alpha[geometric] < 0.0).any()
        or (alpha[geometric] > 1.0).any()
    ):
        raise ValueError("carryover_alpha must be finite in [0, 1] where carryover_family == 1")
    if weibull.any() and (not np.isfinite(lam[weibull]).all() or (lam[weibull] <= 0.0).any()):
        raise ValueError("weibull_lam must be finite and > 0 where carryover_family == 2")
    if weibull.any() and (not np.isfinite(k[weibull]).all() or (k[weibull] <= 0.0).any()):
        raise ValueError("weibull_k must be finite and > 0 where carryover_family == 2")
    return family, alpha, lam, k


def _validated_outcome_scale(outcome: np.ndarray, outcome_scale: np.ndarray | None) -> np.ndarray:
    """Return a finite, strictly positive per-task outcome scale."""
    try:
        scale = (
            outcome.std(axis=1)
            if outcome_scale is None
            else np.asarray(outcome_scale, dtype=np.float64)
        )
    except (TypeError, ValueError) as exc:
        raise ValueError(
            "outcome_scale must be finite and strictly positive with shape (n_tasks,)"
        ) from exc
    invalid_scale = (
        scale.shape != (outcome.shape[0],) or not np.isfinite(scale).all() or (scale <= 0.0).any()
    )
    if invalid_scale:
        raise ValueError("outcome_scale must be finite and strictly positive with shape (n_tasks,)")
    return scale


def _coefficient_of_variation(x: np.ndarray, sd: float) -> tuple[float, bool]:
    """Return a finite CV value and whether it is defined."""
    mean_abs = abs(float(x.mean()))
    if mean_abs != 0.0:
        return float(sd / mean_abs), True
    return (0.0, True) if np.all(x == 0.0) else (0.0, False)


def _hf_ratio(x: np.ndarray, sd: np.ndarray | float | None = None) -> np.ndarray:
    """High-frequency ratio along the last axis, optionally reusing its std."""
    if sd is None:
        sd = x.std(axis=-1)
    diff_sd = np.diff(x, axis=-1).std(axis=-1)
    out = np.zeros_like(sd)
    np.divide(diff_sd, np.sqrt(2.0) * sd, out=out, where=sd != 0.0)
    return np.asarray(out)


def _spearman_abs(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """|Spearman rank correlation| along the last axis, vectorized.

    Average ranks for ties (scipy convention) — ordinal ranks would give a
    constant series ranks ``0..n_time_steps-1`` and score |rho| ~ 1 against any
    trending series, hiding exactly the flat targets this module exists to
    flag. A (near-)constant series has zero rank variance and returns 0.0: no
    signal is visible from it (scipy returns NaN there; 0.0 is the
    gate-friendly encoding of the same fact).
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


def _carryover_weights(
    family: int, alpha: float, lam: float, shape: float, l_max: int
) -> np.ndarray | None:
    """Return normalized causal weights, or ``None`` for identity/degenerate kernels."""
    family = _validated_carryover_family(family)
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
        raw_weight_max = weights.max()
        weight_min = weights.min()
        span = raw_weight_max - weight_min
        # This guard MUST track mechanisms.apply_weibull_pdf_carryover. The oracle passes
        # symbolic value variables; a backend-dependent library-output guard would make
        # FAST_COMPILE generation and the FAST_RUN oracle disagree whether a treatment responds.
        # 1e-300 is above the float64 denormal cliff (~5e-324), yet below any
        # normal-magnitude density, so the analytic replica detects only underflow.
        if not (raw_weight_max > 1e-300) or not np.isfinite(span) or span == 0.0:
            return None
        weights = (weights - weight_min) / span
    total = weights.sum()
    if not np.isfinite(total) or total == 0.0:
        return None
    return np.asarray(weights / total, dtype=np.float64)


def contemporaneous_weight(family: int, alpha: float, lam: float, k: float, l_max: int) -> float:
    """Normalized carryover weight on the current week (zero lag).

    Returns
    -------
    float
        ``weights[0]`` for a nondegenerate nonidentity kernel, 1.0 for an
        identity kernel, and 0.0 for a degenerate nonidentity kernel.
    """
    family = _validated_carryover_family(family)
    l_max = _validated_integer(l_max, "l_max", 1)
    weights = _carryover_weights(family, alpha, lam, k, l_max)
    if family == 0 or l_max == 1:
        return 1.0
    return 0.0 if weights is None else float(weights[0])


def response_support_weeks(family: int, alpha: float, lam: float, k: float, l_max: int) -> int:
    """Largest positive lag the REALIZED kernel actually puts weight on.

    This is the response's true temporal reach, not its family label: a
    geometric kernel with ``alpha == 0`` and a min-max Weibull kernel whose
    trailing taps are annihilated both reach fewer weeks back than
    ``l_max - 1``. ``0`` means the response is contemporaneous only — an
    identity kernel (``family == 0`` or ``l_max == 1``), a kernel confined to
    the current week, or a degenerate kernel that annihilates the treatment.

    Returns
    -------
    int
        Maximum lag index with nonzero normalized weight, in ``[0, l_max - 1]``.
    """
    family = _validated_carryover_family(family)
    l_max = _validated_integer(l_max, "l_max", 1)
    weights = _carryover_weights(family, alpha, lam, k, l_max)
    if weights is None:
        return 0
    positive = np.flatnonzero(weights > 0.0)
    return int(positive[-1]) if positive.size else 0


def admitted_response_support_weeks(
    families: Iterable[int],
    l_max: int,
    *,
    carryover_alpha_range: tuple[float, float],
) -> int:
    """Largest positive lag ANY kernel these families and priors can produce.

    The pre-draw counterpart of :func:`response_support_weeks`: the shape
    parameters are still free, so the answer must hold for every value the
    priors admit. Per family:

    * ``none`` (0) reaches 0 — the transform is the identity.
    * ``geometric`` (1) reaches ``l_max - 1`` whenever the decay prior admits
      ``alpha > 0``, because ``alpha ** lag`` is then positive at every lag,
      and 0 when the prior pins ``alpha == 0``.
    * ``weibull`` (2) reaches ``l_max - 1``. The min-max rescaling annihilates
      the kernel's own minimum, but which lag that is depends on ``(lam, k)``,
      and the usual prior boxes contain kernels whose final tap survives, so
      this bound is deliberately not refined per box: over-reporting the reach
      keeps pre-draw validation and oracle slicing conservative.

    Parameters
    ----------
    families : iterable of int
        Carryover family ids that can occur (``0``/``1``/``2``).
    l_max : int
        Carryover kernel length.
    carryover_alpha_range : (float, float)
        Geometric-decay prior support, ``(lo, hi)`` with ``0 <= lo <= hi <= 1``.
        Only its upper end is consulted; a Weibull family needs no range
        because its bound does not depend on one.
    """
    l_max = _validated_integer(l_max, "l_max", 1)
    try:
        alpha_lo, alpha_hi = (float(value) for value in carryover_alpha_range)
    except (TypeError, ValueError) as exc:
        raise ValueError("carryover_alpha_range must be a finite (lo, hi) pair") from exc
    if not (np.isfinite(alpha_lo) and np.isfinite(alpha_hi)) or not 0.0 <= alpha_lo <= alpha_hi:
        raise ValueError("carryover_alpha_range must satisfy finite 0 <= lo <= hi")
    support = 0
    for value in families:
        family = _validated_carryover_family(value)
        if family == 1 and alpha_hi > 0.0:
            support = max(support, l_max - 1)
        elif family == 2:
            support = max(support, l_max - 1)
    return support


def _carryover_numpy(
    x: np.ndarray, family: int, alpha: float, lam: float, shape: float, l_max: int
) -> np.ndarray:
    """The generator's normalized, causal carryover for one reported series.

    ``pymc_marketing.weibull_adstock(type="PDF")`` min-max rescales the sampled
    density before sum-normalizing it. The resulting kernel is therefore not a
    Weibull pdf and has ``min(weights) == 0`` exactly. Under the default prior
    (``lam ~ U(2, 8)``, ``k ~ U(1.5, 4)``, ``l_max = 8``), 45.0% of Weibull
    treatments have zero current-week weight (200k draws), mean lag-1 weight is
    0.065, and 38.5% peak at lag >= 5.
    """
    family = _validated_carryover_family(family)
    l_max = _validated_integer(l_max, "l_max", 1)
    weights = _carryover_weights(family, alpha, lam, shape, l_max)
    if family == 0 or l_max == 1:
        return x.copy()
    if weights is None:
        return np.zeros_like(x, dtype=np.float64)
    return np.convolve(x, weights, mode="full")[: x.size]


def dense_signal_metrics(
    treatment: np.ndarray,
    contributions: np.ndarray,
    outcome: np.ndarray,
    baseline: np.ndarray,
    cy_mask: np.ndarray,
    *,
    outcome_scale: np.ndarray | None = None,
    carryover_family: np.ndarray | None = None,
    carryover_alpha: np.ndarray | None = None,
    weibull_lam: np.ndarray | None = None,
    weibull_k: np.ndarray | None = None,
    l_max: int = 8,
    carryover_burn_in: int = 0,
) -> tuple[np.ndarray, np.ndarray]:
    """Return persisted dense metrics and uint8 validity, both shaped
    ``(n_tasks, n_treatments, len(SIGNAL_METRIC_LAYOUT))``.

    Inputs are deliberately the final persisted arrays; callers must not pass
    pre-cast draw values. Ineligible/padded cells are exactly zero and invalid.
    An all-zero series has valid CV 0.0. A nonconstant zero-mean series has an
    undefined/infinite CV and is invalid; generated treatment and contribution
    targets are non-negative, so only the all-zero convention is reachable in
    a corpus.
    """
    treatment = np.asarray(treatment, dtype=np.float64)
    contributions = np.asarray(contributions, dtype=np.float64)
    outcome = np.asarray(outcome, dtype=np.float64)
    baseline = np.asarray(baseline, dtype=np.float64)
    mask = np.asarray(cy_mask, dtype=bool)
    if treatment.ndim != 3:
        raise ValueError("treatment must have shape (n_tasks, n_time_steps, n_treatments)")
    n_tasks, n_time_steps, n_treatments = treatment.shape
    if (
        contributions.shape != (n_tasks, n_time_steps, n_treatments)
        or outcome.shape != (n_tasks, n_time_steps)
        or baseline.shape != (n_tasks, n_time_steps)
    ):
        raise ValueError("signal source arrays have incompatible shapes")
    if mask.shape != (n_tasks, n_treatments):
        raise ValueError(f"cy_mask shape {mask.shape} != {(n_tasks, n_treatments)}")
    if not all(np.isfinite(a).all() for a in (treatment, contributions, outcome, baseline)):
        raise ValueError("signal source arrays must be finite")
    l_max = _validated_integer(l_max, "l_max", 1)
    carryover_burn_in = _validated_integer(carryover_burn_in, "carryover_burn_in", 0)
    scale = _validated_outcome_scale(outcome, outcome_scale)
    metadata_shape = (n_tasks, n_treatments)
    family_input = (
        np.zeros(metadata_shape, dtype=np.int8) if carryover_family is None else carryover_family
    )
    alpha_input = np.zeros(metadata_shape) if carryover_alpha is None else carryover_alpha
    lam_input = np.ones(metadata_shape) if weibull_lam is None else weibull_lam
    k_input = np.ones(metadata_shape) if weibull_k is None else weibull_k
    fam, alpha, wlam, wk = _validated_carryover_metadata(
        family_input, alpha_input, lam_input, k_input, metadata_shape
    )
    metrics = np.zeros((n_tasks, n_treatments, len(SIGNAL_METRIC_LAYOUT)), dtype=np.float32)
    valid = np.zeros_like(metrics, dtype=np.uint8)
    metric_index = {name: i for i, name in enumerate(SIGNAL_METRIC_LAYOUT)}
    for n in range(n_tasks):
        active_treatments = np.flatnonzero(mask[n])
        if active_treatments.size == 0:
            continue
        if n_time_steps >= 2:
            centered_b = baseline[n] - baseline[n].mean()
            centered_contributions = {
                k: contributions[n, :, k] - contributions[n, :, k].mean() for k in active_treatments
            }
        for k in active_treatments:
            x, y = treatment[n, :, k], contributions[n, :, k]
            std_x, std_y = x.std(), y.std()
            treatment_cv, treatment_cv_valid = _coefficient_of_variation(x, std_x)
            contrib_cv, contrib_cv_valid = _coefficient_of_variation(y, std_y)
            values = {
                "treatment_cv": (treatment_cv, treatment_cv_valid),
                "contrib_cv": (contrib_cv, contrib_cv_valid),
                "contrib_rel_std": (std_y / scale[n], True),
            }
            for name, (value, is_valid) in values.items():
                metrics[n, k, metric_index[name]] = value
                valid[n, k, metric_index[name]] = is_valid
            if n_time_steps >= 2:
                metrics[n, k, metric_index["treatment_hf"]] = _hf_ratio(x, std_x)
                metrics[n, k, metric_index["contrib_hf"]] = _hf_ratio(y, std_y)
                valid[n, k, [metric_index["treatment_hf"], metric_index["contrib_hf"]]] = 1
            ad_x = _carryover_numpy(x, fam[n, k], alpha[n, k], wlam[n, k], wk[n, k], l_max)
            # Compare once-carryovered observed treatment, not an already-transformed target.
            if n_time_steps - l_max >= 3:
                metrics[n, k, metric_index["spearman"]] = _spearman_abs(
                    ad_x[l_max:][None], y[l_max:][None]
                )[0]
                valid[n, k, metric_index["spearman"]] = 1
            if carryover_burn_in < l_max and n_time_steps >= l_max + 3:
                warm_range = y[:l_max].max() - y[:l_max].min()
                suffix_std = y[l_max:].std()
                if suffix_std != 0.0:
                    metrics[n, k, metric_index["warmup_ratio"]] = warm_range / suffix_std
                    valid[n, k, metric_index["warmup_ratio"]] = 1
            if n_time_steps >= 2:
                centered_y = centered_contributions[k]
                design = np.column_stack(
                    [
                        centered_b,
                        *(centered_contributions[j] for j in active_treatments if j != k),
                    ]
                )
                sst = float(centered_y @ centered_y)
                if sst == 0.0:
                    # Persisted constants are exactly reproducible by the intercept.
                    metrics[n, k, metric_index["contrib_r2_explained_by_rest"]] = 1.0
                    valid[n, k, metric_index["contrib_r2_explained_by_rest"]] = 1
                else:
                    coefficients, _, rank, _ = np.linalg.lstsq(design, centered_y, rcond=None)
                    # Use the fitted solver's rank for residual degrees of freedom.
                    if n_time_steps > 1 + rank:
                        fitted = design @ coefficients
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


def per_treatment_signal(
    treatment: np.ndarray,
    contributions: np.ndarray,
    outcome: np.ndarray,
    cy_mask: np.ndarray,
    *,
    outcome_scale: np.ndarray | None = None,
    l_max: int = 8,
    baseline: np.ndarray | None = None,
    carryover_family: np.ndarray | None = None,
    carryover_alpha: np.ndarray | None = None,
    weibull_lam: np.ndarray | None = None,
    weibull_k: np.ndarray | None = None,
    carryover_burn_in: int = 0,
) -> dict[str, np.ndarray]:
    """Per-(task, direct treatment) signal metrics.

    Parameters
    ----------
    treatment : (n_tasks, n_time_steps, n_treatments) observed treatments (model input).
    contributions : (n_tasks, n_time_steps, n_treatments) true contribution targets.
    outcome : (n_tasks, n_time_steps) outcome series.
    cy_mask : (n_tasks, n_treatments) bool/0-1 — which treatments are direct (C->Y) AND active.
    outcome_scale : (n_tasks,) optional — per-task target normalizer; defaults to the
        full-series outcome std.
    l_max : carryover length, defines the warmup window for ``warmup_ratio``.

    Returns
    -------
    dict of 1-D float arrays, one entry per (task, direct treatment) pair, plus
    ``task_idx`` locating each pair.
    """
    treatment = np.asarray(treatment)
    n_tasks, n_time_steps, n_treatments = treatment.shape
    mask = np.asarray(cy_mask, dtype=bool)
    dense, valid = dense_signal_metrics(
        treatment,
        contributions,
        outcome,
        np.zeros((n_tasks, n_time_steps), dtype=treatment.dtype) if baseline is None else baseline,
        mask,
        outcome_scale=outcome_scale,
        l_max=l_max,
        carryover_family=carryover_family,
        carryover_alpha=carryover_alpha,
        weibull_lam=weibull_lam,
        weibull_k=weibull_k,
        carryover_burn_in=carryover_burn_in,
    )
    if baseline is None:
        for name in ("contrib_r2_explained_by_rest", "contrib_corr_baseline"):
            index = SIGNAL_METRIC_LAYOUT.index(name)
            dense[..., index] = 0.0
            valid[..., index] = 0
    task_idx = np.broadcast_to(np.arange(n_tasks)[:, None], (n_tasks, n_treatments))
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
    outcome: np.ndarray,
    cy_mask: np.ndarray,
    *,
    outcome_scale: np.ndarray | None = None,
    l_max: int = 8,
    carryover_burn_in: int = 0,
    carryover_family: np.ndarray | None = None,
    carryover_alpha: np.ndarray | None = None,
    weibull_lam: np.ndarray | None = None,
    weibull_k: np.ndarray | None = None,
) -> dict:
    """Summarize persisted dense metrics using their per-metric validity.

    This is deliberately separate from metric calculation so corpus diagnostics
    can be derived from exactly the float32 arrays written to disk.

    ``response_warmup_weeks`` identifies reported weeks whose response depends
    on unpersisted pre-window treatment. With burn-in, reported week ``t`` reaches
    before the persisted window exactly when ``t < S``, where ``S`` is the
    largest positive lag any ``cy_mask``-eligible direct kernel actually
    weights (:func:`response_support_weeks`). The count is therefore ``S`` —
    zero without burn-in, and zero when every eligible kernel is
    contemporaneous, which includes an identity family, a geometric kernel with
    ``alpha == 0``, and a degenerate annihilated kernel. ``S`` is below
    ``l_max - 1`` whenever the realized min-max Weibull kernel has no weight on
    its trailing taps. Establishing ``S`` needs all four carryover metadata
    arrays; with only ``carryover_family`` the summary reports ``l_max - 1`` for
    any non-identity eligible kernel, and with no metadata at all it reports
    ``l_max - 1`` under burn-in because it cannot rule out a full-reach kernel.
    ``support_mask`` is the temporal train/query split, not this
    response-warmup rule.

    Parameters
    ----------
    metrics, valid : np.ndarray
        Persisted dense metric values and their binary per-metric validity
        masks, each shaped ``(n_tasks, n_treatments, len(SIGNAL_METRIC_LAYOUT))``.
    outcome : np.ndarray
        Persisted outcome array with shape ``(n_tasks, n_time_steps)``.
    cy_mask : np.ndarray
        Boolean ``(n_tasks, n_treatments)`` mask selecting eligible direct-treatment pairs.
    outcome_scale : np.ndarray, optional
        Positive persisted per-task target scales. When omitted, scales are
        computed from ``outcome``.
    l_max : int
        Carryover kernel length.
    carryover_burn_in : int
        Number of generated leading burn-in weeks.
    carryover_family, carryover_alpha, weibull_lam, weibull_k : np.ndarray, optional
        Per-(task, treatment) carryover metadata. All four must be supplied to
        calculate ``frac_zero_contemporaneous_weight`` and the realized
        ``response_warmup_weeks``; omitting any sets that fraction to ``None``
        and falls back to the family-only (or, with no metadata, the
        ``l_max - 1``) warmup rule described above.
    """
    metrics = np.asarray(metrics)
    valid = np.asarray(valid)
    outcome = np.asarray(outcome, dtype=np.float64)
    mask = np.asarray(cy_mask, dtype=bool)
    expected = mask.shape + (len(SIGNAL_METRIC_LAYOUT),)
    if metrics.shape != expected or valid.shape != expected:
        raise ValueError(f"metrics and validity must have shape {expected}")
    if outcome.ndim != 2 or outcome.shape[0] != mask.shape[0]:
        raise ValueError("outcome must have shape (n_tasks, n_time_steps)")
    if not np.isfinite(outcome).all():
        raise ValueError("outcome must be finite")
    if not np.isfinite(metrics).all() or not np.isin(valid, (0, 1)).all():
        raise ValueError("metrics must be finite and validity must be binary")

    l_max = _validated_integer(l_max, "l_max", 1)
    carryover_burn_in = _validated_integer(carryover_burn_in, "carryover_burn_in", 0)
    scale = _validated_outcome_scale(outcome, outcome_scale)
    level_ratio = np.abs(outcome.mean(axis=1)) / scale
    n_pairs = int(mask.sum())
    carryover_metadata = (carryover_family, carryover_alpha, weibull_lam, weibull_k)
    fam: np.ndarray | None = None
    frac_zero_contemporaneous_weight: float | None = None
    realized_support: int | None = None
    if n_pairs and all(value is not None for value in carryover_metadata):
        fam, alpha, wlam, wk = _validated_carryover_metadata(
            carryover_family, carryover_alpha, weibull_lam, weibull_k, mask.shape
        )
        zero_count = 0
        realized_support = 0
        for n, k in zip(*np.nonzero(mask)):
            weights = _carryover_weights(fam[n, k], alpha[n, k], wlam[n, k], wk[n, k], l_max)
            if weights is None:
                first_weight = 1.0 if fam[n, k] == 0 or l_max == 1 else 0.0
            else:
                first_weight = float(weights[0])
                positive = np.flatnonzero(weights > 0.0)
                if positive.size:
                    realized_support = max(realized_support, int(positive[-1]))
            zero_count += int(first_weight < 1e-9)
        frac_zero_contemporaneous_weight = zero_count / n_pairs
    elif n_pairs and carryover_family is not None:
        fam = _validated_carryover_family_array(carryover_family, mask.shape)
    if carryover_burn_in == 0 or n_pairs == 0:
        response_warmup_weeks = 0
    elif realized_support is not None:
        response_warmup_weeks = realized_support
    elif fam is None or np.any((fam != 0) & mask):
        response_warmup_weeks = l_max - 1
    else:
        response_warmup_weeks = 0
    out: dict = {
        "n_direct_treatments": n_pairs,
        "response_warmup_weeks": response_warmup_weeks,
        "frac_zero_contemporaneous_weight": frac_zero_contemporaneous_weight,
    }
    metric_index = {name: i for i, name in enumerate(SIGNAL_METRIC_LAYOUT)}
    for key, i in metric_index.items():
        vals = metrics[..., i][mask & valid[..., i].astype(bool)]
        out[f"{key}_quantiles"] = {
            f"q{int(q * 100)}": (float(np.quantile(vals, q)) if vals.size else None) for q in _QS
        }

    for key, (metric, predicate) in _FRACTION_SPECS.items():
        i = metric_index[metric]
        vals = metrics[..., i][mask & valid[..., i].astype(bool)]
        # A full burn-in makes warmup explicitly N/A, rather than absent due
        # to too few observations.  Its aggregate artifact fraction is 0.
        out[key] = (
            0.0
            if metric == "warmup_ratio" and carryover_burn_in >= l_max
            else (float(predicate(vals).mean()) if vals.size else None)
        )
    out["outcome_level_ratio_quantiles"] = {
        f"q{int(q * 100)}": (float(np.quantile(level_ratio, q)) if level_ratio.size else None)
        for q in _QS
    }
    return out


def signal_summary(
    treatment: np.ndarray,
    contributions: np.ndarray,
    outcome: np.ndarray,
    cy_mask: np.ndarray,
    *,
    outcome_scale: np.ndarray | None = None,
    l_max: int = 8,
    baseline: np.ndarray | None = None,
    carryover_family: np.ndarray | None = None,
    carryover_alpha: np.ndarray | None = None,
    weibull_lam: np.ndarray | None = None,
    weibull_k: np.ndarray | None = None,
    carryover_burn_in: int = 0,
) -> dict:
    """Corpus-level signal report: metric quantiles + degenerate fractions.

    JSON-serializable; embedded under ``diagnostics["signal"]`` by
    ``sample_prior_predictive`` so weak-signal priors are visible in every corpus /
    shard manifest. Interpretation guide:

    * ``frac_contrib_cv_lt_010`` — share of direct treatments whose true
      contribution is near-flat (CV < 0.10). Some flat treatments are realistic;
      a majority means the prior generates unlearnable attribution tasks.
    * ``frac_contrib_hf_lt_015`` — share with essentially no week-to-week
      variation (smooth drift only).
    * ``frac_spearman_lt_03`` — share whose target is invisible from the
      observed treatment.
    * ``frac_warmup_gt_3`` — share whose largest feature is the carryover
      zero-padding warmup (generator artifact, fixed by ``carryover_burn_in``).
    * ``frac_contrib_rel_std_lt_001`` — share whose contribution amplitude is
      too small to matter in training-loss units.
    * ``frac_contrib_r2_gt_095`` — share whose target is a linear combination
      of baseline and the other treatments, so only the sum is identified.
    """
    treatment = np.asarray(treatment)
    baseline_array = (
        np.zeros(treatment.shape[:2], dtype=treatment.dtype) if baseline is None else baseline
    )
    metrics, valid = dense_signal_metrics(
        treatment,
        contributions,
        outcome,
        baseline_array,
        cy_mask,
        outcome_scale=outcome_scale,
        l_max=l_max,
        carryover_family=carryover_family,
        carryover_alpha=carryover_alpha,
        weibull_lam=weibull_lam,
        weibull_k=weibull_k,
        carryover_burn_in=carryover_burn_in,
    )
    if baseline is None:
        for name in ("contrib_r2_explained_by_rest", "contrib_corr_baseline"):
            index = SIGNAL_METRIC_LAYOUT.index(name)
            metrics[..., index] = 0.0
            valid[..., index] = 0
    return summarize_signal_metrics(
        metrics,
        valid,
        outcome,
        cy_mask,
        outcome_scale=outcome_scale,
        l_max=l_max,
        carryover_burn_in=carryover_burn_in,
        carryover_family=carryover_family,
        carryover_alpha=carryover_alpha,
        weibull_lam=weibull_lam,
        weibull_k=weibull_k,
    )


def check_signal_gate(signal: dict, gate: dict[str, float] | None = None) -> tuple[bool, list[str]]:
    """Check a ``signal_summary`` block against minimum-signal thresholds.

    Returns ``(ok, lines)`` where ``lines`` are human-readable PASS/FAIL rows.
    A missing or ``None`` metric FAILS loudly — no signal measured is not a
    pass — and its row names WHY it is missing, which is not always the same
    thing: a corpus can have no direct treatments to measure at all, or it can
    have them and still leave a metric unmeasured because every eligible pair's
    value was invalid (a horizon too short for the metric's window, a constant
    series, ...). Only ``signal["n_direct_treatments"]`` separates the two, so a
    summary that omits the count gets an agnostic row rather than a guess.
    Generation pipelines can call this on each shard's ``diagnostics["signal"]``
    to reject weak-signal priors at generation time.
    """
    thresholds = DEFAULT_GATE if gate is None else gate
    n_direct = signal.get("n_direct_treatments")
    if n_direct is None:
        missing_reason = "n_direct_treatments not reported"
    elif int(n_direct) == 0:
        missing_reason = "no direct treatments measured"
    else:
        missing_reason = f"no valid observations across {int(n_direct)} direct treatments"
    ok = True
    lines: list[str] = []
    for key, thresh in thresholds.items():
        val = signal.get(key)
        if val is None:
            ok = False
            lines.append(f"[FAIL] {key} missing ({missing_reason})")
            continue
        passed = val <= thresh
        ok &= passed
        lines.append(f"[{'PASS' if passed else 'FAIL'}] {key} = {val:.1%} (<= {thresh:.0%})")
    return ok, lines
