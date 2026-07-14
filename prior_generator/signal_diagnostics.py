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
* ``spearman``      — |rank correlation| between observed spend and the true
  contribution (how much of the target is visible from the input).
* ``warmup_ratio``  — (max-min over the first ``l_max`` weeks) / (std of the
  rest): >> 1 flags the adstock zero-padding warmup artifact dominating the
  target (fixed by ``adstock_burn_in``).

``signal_summary`` reduces these to quantiles plus degenerate-target
fractions, and adds the per-task normalized sales level
``mean(sales)/sales_scale`` (targets are trained sales_scale-normalized
WITHOUT centering, so extreme levels make every head's job harder).
"""

from __future__ import annotations

import numpy as np

__all__ = [
    "DEFAULT_GATE",
    "FRAC_KEYS",
    "METRIC_KEYS",
    "check_signal_gate",
    "per_channel_signal",
    "signal_summary",
]

_QS = (0.1, 0.5, 0.9)

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
)

#: Degenerate-target fractions emitted by :func:`signal_summary`.
FRAC_KEYS: tuple[str, ...] = (
    "frac_contrib_cv_lt_005",
    "frac_contrib_cv_lt_010",
    "frac_contrib_hf_lt_015",
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
}


def _hf_ratio(x: np.ndarray) -> np.ndarray:
    """High-frequency ratio along the last axis: 1 = white noise, ->0 smooth."""
    s = x.std(axis=-1)
    d = np.diff(x, axis=-1).std(axis=-1)
    return np.asarray(np.where(s > 1e-12, d / (np.sqrt(2.0) * np.maximum(s, 1e-12)), 0.0))


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
    return np.asarray(np.where(denom > 1e-12, np.abs(num) / np.maximum(denom, 1e-12), 0.0))


def per_channel_signal(
    spend: np.ndarray,
    contributions: np.ndarray,
    sales: np.ndarray,
    cy_mask: np.ndarray,
    *,
    sales_scale: np.ndarray | None = None,
    l_max: int = 8,
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
    spend = np.asarray(spend, dtype=np.float64)
    contributions = np.asarray(contributions, dtype=np.float64)
    sales = np.asarray(sales, dtype=np.float64)
    mask = np.asarray(cy_mask).astype(bool)
    N, T, K = spend.shape
    if contributions.shape != (N, T, K):
        raise ValueError(f"contributions shape {contributions.shape} != {(N, T, K)}")
    if mask.shape != (N, K):
        raise ValueError(f"cy_mask shape {mask.shape} != {(N, K)}")
    if sales_scale is None:
        scale = sales.std(axis=1)
    else:
        scale = np.asarray(sales_scale, dtype=np.float64)
    scale = np.maximum(scale, 1e-12)

    s = np.swapaxes(spend, 1, 2)  # (N, K, T)
    c = np.swapaxes(contributions, 1, 2)  # (N, K, T)

    spend_cv = s.std(axis=-1) / (np.abs(s.mean(axis=-1)) + 1e-12)  # (N, K)
    spend_hf = _hf_ratio(s)
    contrib_cv = c.std(axis=-1) / (np.abs(c.mean(axis=-1)) + 1e-12)
    contrib_hf = _hf_ratio(c)
    contrib_rel_std = c.std(axis=-1) / scale[:, None]
    spearman = _spearman_abs(s, c)
    # Warmup window needs >= 3 post-warmup samples for a stable "rest" std;
    # for shorter series the ratio is unmeasurable — report 0 rather than a
    # 1e12 explosion that would spuriously fail every warmup gate.
    warm = min(max(int(l_max), 1), T - 3)
    if warm >= 1:
        warm_range = c[..., :warm].max(axis=-1) - c[..., :warm].min(axis=-1)
        rest_std = np.maximum(c[..., warm:].std(axis=-1), 1e-12)
        warmup_ratio = warm_range / rest_std
    else:
        warmup_ratio = np.zeros((N, K))

    task_idx = np.broadcast_to(np.arange(N)[:, None], (N, K))
    return {
        "task_idx": task_idx[mask].astype(np.float64),
        "spend_cv": spend_cv[mask],
        "spend_hf": spend_hf[mask],
        "contrib_cv": contrib_cv[mask],
        "contrib_hf": contrib_hf[mask],
        "contrib_rel_std": contrib_rel_std[mask],
        "spearman": spearman[mask],
        "warmup_ratio": warmup_ratio[mask],
    }


def signal_summary(
    spend: np.ndarray,
    contributions: np.ndarray,
    sales: np.ndarray,
    cy_mask: np.ndarray,
    *,
    sales_scale: np.ndarray | None = None,
    l_max: int = 8,
) -> dict:
    """Corpus-level signal report: metric quantiles + degenerate fractions.

    JSON-serializable; embedded under ``diagnostics["signal"]`` by
    ``generate_corpus`` so weak-signal priors are visible in every corpus /
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
    """
    per = per_channel_signal(
        spend, contributions, sales, cy_mask, sales_scale=sales_scale, l_max=l_max
    )
    sales = np.asarray(sales, dtype=np.float64)
    if sales_scale is None:
        scale = sales.std(axis=1)
    else:
        scale = np.asarray(sales_scale, dtype=np.float64)
    level_ratio = np.abs(sales.mean(axis=1)) / np.maximum(scale, 1e-12)

    # Empty-mask corpora emit None (never NaN — json.dumps would produce a
    # bare `NaN` literal that strict JSON parsers reject) and every frac key
    # is always present so the manifest schema is data-independent.
    n_pairs = int(per["spend_cv"].shape[0])
    out: dict = {"n_direct_channels": n_pairs}
    for key in METRIC_KEYS:
        vals = per[key]
        out[f"{key}_quantiles"] = {
            f"q{int(q * 100)}": (float(np.quantile(vals, q)) if n_pairs else None) for q in _QS
        }
    _fracs = {
        "frac_contrib_cv_lt_005": (per["contrib_cv"] < 0.05),
        "frac_contrib_cv_lt_010": (per["contrib_cv"] < 0.10),
        "frac_contrib_hf_lt_015": (per["contrib_hf"] < 0.15),
        "frac_spearman_lt_03": (per["spearman"] < 0.3),
        "frac_warmup_gt_3": (per["warmup_ratio"] > 3.0),
    }
    for key in FRAC_KEYS:
        out[key] = float(_fracs[key].mean()) if n_pairs else None
    out["sales_level_ratio_quantiles"] = {
        f"q{int(q * 100)}": float(np.quantile(level_ratio, q)) for q in _QS
    }
    return out


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
