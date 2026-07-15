"""Stratified corpus sampler for the additive causal SCM.

Each world (task) is an additive structural causal model over latent demand
factors D, observed controls Z, media channels C, a baseline B and sales Y.
The DAG is drawn per cell by :func:`sample_g_additive` (per-edge-type
Bernoulli rates or "pot" budgets over the extended 8-block layout). Each world
is then a PyMC model (:mod:`prior_generator.world_model`) whose continuous
priors are pm distributions and whose noise is pm RVs; drawing it yields the
series with exact interventional decomposition targets — direct contributions,
per-control / per-confounder contributions, ``baseline_intrinsic`` and the
telescoping 3-source indirect split ``(cc, zc, dc)``.

The corpus schema is the dict-of-ndarrays documented in
:func:`sample_prior_predictive` — the same format the structural-pfn training
pipeline consumes (persist with ``prior_generator.save_corpus``).

Extraction note: the legacy L0/L1 PyMC-model rungs from structural-pfn were
deprecated there (plan-05) and deliberately NOT migrated; the one supported
world prior is ``make_scm_prior(texture="diverse")``.
"""

from __future__ import annotations

import time
import warnings
from dataclasses import dataclass, replace

import numpy as np

from .signal_diagnostics import signal_summary
from .slots import (
    EDGE_BASE_RATES,
    EDGE_TYPES_EXTENDED,
    J_DEMO,
    K_DEMO,
    M_DEMO,
    T_DEMO,
    SlotLayout,
)

#: Maximum draw rounds per cell before giving up (post-filter top-up loop).
MAX_TOPUPS_PER_CELL = 8


@dataclass
class SCMPrior:
    """Corpus generation knobs + prior-range constants for the additive SCM.

    Supports variable-size DAGs via padding to max sizes. Each cell draws
    random active counts (K_active, M_active, J_active) from configured
    ranges. Inactive nodes are zero-padded and masked via active_channel_mask.

    Key fields:
        K, M, J: Default sizes for backward compatibility (used as K_max etc. if *_max not set)
        K_max, M_max, J_max: Maximum sizes for variable-size DAGs
        K_active_range, M_active_range, J_active_range: Ranges for random active counts per cell

    Prefer building configs through
    :func:`prior_generator.presets.make_scm_prior`, which pins the
    layout and enables the supported "diverse" channel texture.
    """

    T: int = T_DEMO
    K: int = K_DEMO
    M: int = M_DEMO
    J: int = J_DEMO
    n_cells: int = 50
    draws_per_cell: int = 20
    val_cell_frac: float = 0.2
    query_frac: float = 0.25
    p_long_horizon: float = 0.5  # fraction of tasks with 50% horizon (vs 25%)
    spend_cv_floor: float = 0.08
    l_max: int = 8
    seed: int = 0

    # -- variable-size DAG support ----------------------------------------
    # Maximum sizes for padding (K_max >= K, etc.)
    K_max: int = 0  # 0 means "use K" (backward compat)
    M_max: int = 0  # 0 means "use M"
    J_max: int = 0  # 0 means "use J"
    # Ranges for random active counts per cell (inclusive)
    K_active_range: tuple[int, int] = (4, 20)
    M_active_range: tuple[int, int] = (2, 10)
    J_active_range: tuple[int, int] = (1, 5)

    # -- prior-range constants -------------------------------------------
    # Per-channel media-response mechanism priors (realized as PyMC
    # distributions in prior_generator.world_model.build_world_model):
    adstock_alpha_range: tuple[float, float] = (0.2, 0.8)
    # Adstock families: 0=none, 1=geometric, 2=weibull
    adstock_family_probs: tuple[float, ...] = (0.15, 0.425, 0.425)
    # Saturation families: 0=none(linear), 1=hill, 2=logistic, 3=michaelis_menten, 4=tanh, 5=root
    saturation_family_probs: tuple[float, ...] = (0.15, 0.17, 0.17, 0.17, 0.17, 0.17)
    # Weibull adstock prior ranges
    weibull_lam_range: tuple[float, float] = (2.0, 8.0)
    weibull_k_range: tuple[float, float] = (1.5, 4.0)

    # -- additive causal graph priors ----------------------------------------
    # New edge-type base rates (extended layout: dz, zc, cc, zz)
    dz_base_rate: float = 0.3  # D->Z base rate
    zc_base_rate: float = 0.3  # Z->C base rate
    cc_base_rate: float = 0.15  # C->C base rate (sparse)
    zz_base_rate: float = 0.1  # Z->Z base rate (sparse)
    # Linear coefficient ranges for the additive structural equations
    dc_coeff_range: tuple[float, float] = (0.1, 0.5)  # D->C loadings
    dz_coeff_range: tuple[float, float] = (0.1, 0.5)  # D->Z loadings
    zc_coeff_range: tuple[float, float] = (0.05, 0.3)  # Z->C loadings
    cc_coeff_range: tuple[float, float] = (0.05, 0.3)  # C->C (positive-only)
    zz_coeff_range: tuple[float, float] = (-0.2, 0.2)  # Z->Z (signed)
    db_coeff_range: tuple[float, float] = (0.15, 0.45)  # D->B loadings
    zb_coeff_range: tuple[float, float] = (0.1, 0.4)  # Z->B loadings
    beta_additive_range: tuple[float, float] = (0.5, 2.0)  # channel effects
    # Random-walk noise priors (plan doc D1a)
    rw_mean_range: tuple[float, float] = (-1.0, 1.0)  # signed nodes (D, Z)
    rw_positive_mean_range: tuple[float, float] = (0.5, 3.0)  # channels
    rw_baseline_mean_range: tuple[float, float] = (3.0, 8.0)  # baseline level
    rw_std_sigma: float = 1.0  # HalfNormal prior for walk std
    rw_channel_std_sigma: float = 0.6  # HalfNormal for channel walk std
    rw_sales_std_sigma: float = 0.25  # HalfNormal for sales-noise walk std
    rw_smoothness_alpha: float = 2.0  # Beta prior alpha for smoothness
    rw_smoothness_beta: float = 2.0  # Beta prior beta for smoothness

    # -- Channel texture (plan doc 05 signal fix) --------------------------
    # High-frequency exogenous drive on the channel's own pre-softplus input:
    # iid weekly noise (sigma ~ U(range)) and campaign pulses (per-week
    # probability ~ U(prob_range), amplitude ~ U(amp_range)). Optional uniform
    # walk-std range replacing the HalfNormal (which piles mass at 0 and
    # produces flat contribution targets). The sigma/amp/std factors are
    # RELATIVE to each channel's own level (softplus of its walk mean), so the
    # texture is scale-free across small and large channels — like L1's
    # log-space spend noise. adstock_burn_in simulates extra leading weeks and
    # drops them so the adstock zero-padding warmup never reaches the reported
    # window. Defaults disable the texture (flat targets); `make_scm_prior`
    # enables the diverse texture, which is the supported prior.
    rw_channel_std_range: tuple[float, float] | None = None
    channel_hf_sigma_range: tuple[float, float] = (0.0, 0.0)
    channel_pulse_prob_range: tuple[float, float] = (0.0, 0.0)
    channel_pulse_amp_range: tuple[float, float] = (0.5, 1.5)
    adstock_burn_in: int = 0

    # Prior-shift eval support: optional overrides for the LEGACY edge base
    # rates ("cy", "dc", "db", "zb") which otherwise come from the slots.py
    # module constants. None (default) => byte-identical legacy behaviour.
    # The Phase-4 rates (dz/zc/cc/zz) have their own config fields above.
    edge_rate_overrides: dict[str, float] | None = None

    # Per-edge-type arrow budget ("pot"). Maps edge type ->
    # "up to N" cap (int) or (lo, hi) inclusive range; the per-task count is
    # drawn uniformly (int N => {0..N}; use (N, N) for exactly N) and capped at
    # the number of eligible pairs. When a type is present, its arrows are
    # scattered uniformly over eligible node pairs instead of drawn per-pair
    # Bernoulli; each type's pot is independent. Types absent from the dict keep
    # their Bernoulli base rate. None (default) => byte-identical legacy path.
    # Example: {"zc": 5} places up to 5 control->channel arrows however they
    # land (one control fanning out, or spread across controls — capped at 5),
    # while "zb" (controls' effect on the outcome) is untouched.
    # Note: "cy" keeps its >=1 floor from the degenerate C->Y guard, so a cy
    # budget that draws 0 still yields exactly one C->Y edge.
    edge_budget: dict[str, int | tuple[int, int]] | None = None

    @property
    def layout(self) -> SlotLayout:
        return SlotLayout(
            K=self.K_max_effective,
            M=self.M_max_effective,
            J=self.J_max_effective,
            edge_types=EDGE_TYPES_EXTENDED,
        )

    @property
    def K_max_effective(self) -> int:
        """K_max if set, else K (backward compat)."""
        return self.K_max if self.K_max > 0 else self.K

    @property
    def M_max_effective(self) -> int:
        """M_max if set, else M (backward compat)."""
        return self.M_max if self.M_max > 0 else self.M

    @property
    def J_max_effective(self) -> int:
        """J_max if set, else J (backward compat)."""
        return self.J_max if self.J_max > 0 else self.J

    @property
    def K_active_range_effective(self) -> tuple[int, int]:
        """K_active_range clamped to K_max_effective."""
        lo = min(self.K_active_range[0], self.K_max_effective)
        hi = min(self.K_active_range[1], self.K_max_effective)
        return (lo, hi)

    @property
    def M_active_range_effective(self) -> tuple[int, int]:
        """M_active_range clamped to M_max_effective."""
        lo = min(self.M_active_range[0], self.M_max_effective)
        hi = min(self.M_active_range[1], self.M_max_effective)
        return (lo, hi)

    @property
    def J_active_range_effective(self) -> tuple[int, int]:
        """J_active_range clamped to J_max_effective."""
        lo = min(self.J_active_range[0], self.J_max_effective)
        hi = min(self.J_active_range[1], self.J_max_effective)
        return (lo, hi)

    @property
    def n_query(self) -> int:
        return int(round(self.query_frac * self.T))

    def validate(self) -> None:
        if self.K < 1:
            raise ValueError(f"K must be >= 1, got {self.K}")
        if self.M < 1:
            raise ValueError(f"M must be >= 1, got {self.M}")
        if self.J < 1:
            raise ValueError(f"J must be >= 1, got {self.J}")
        if not 0 < self.n_query < self.T:
            raise ValueError(
                f"query_frac={self.query_frac} gives {self.n_query} query weeks "
                f"for T={self.T}; need 0 < n_query < T"
            )
        if self.n_query > self.T - 2:
            raise ValueError(
                f"query_frac={self.query_frac} gives {self.n_query} query weeks "
                f"for T={self.T}, leaving only {self.T - self.n_query} support weeks. "
                f"Need at least 2 support weeks for meaningful statistics."
            )
        if self.n_cells < 2:
            raise ValueError(f"n_cells must be >= 2, got {self.n_cells}")
        if self.draws_per_cell < 1:
            raise ValueError(f"draws_per_cell must be >= 1, got {self.draws_per_cell}")
        if self.T < 4:
            raise ValueError(f"T must be >= 4, got {self.T}")
        if not 0.0 <= self.p_long_horizon <= 1.0:
            raise ValueError(f"p_long_horizon must be in [0, 1], got {self.p_long_horizon}")
        n_val_cells = max(1, int(round(self.val_cell_frac * self.n_cells)))
        if n_val_cells >= self.n_cells:
            n_val_cells = self.n_cells - 1
        if n_val_cells < 1:
            raise ValueError(
                f"val_cell_frac={self.val_cell_frac} gives {n_val_cells} val cells "
                f"out of {self.n_cells}; need at least 1 train and 1 val cell"
            )
        # Validate mechanism diversity probabilities
        if len(self.adstock_family_probs) != 3:
            raise ValueError(
                f"adstock_family_probs must have 3 entries (none/geometric/weibull), "
                f"got {len(self.adstock_family_probs)}"
            )
        if not np.isclose(sum(self.adstock_family_probs), 1.0, atol=1e-6):
            raise ValueError(
                f"adstock_family_probs must sum to 1.0, got {sum(self.adstock_family_probs)}"
            )
        if len(self.saturation_family_probs) != 6:
            raise ValueError(
                f"saturation_family_probs must have 6 entries "
                f"(none/hill/logistic/michaelis_menten/tanh/root), "
                f"got {len(self.saturation_family_probs)}"
            )
        if not np.isclose(sum(self.saturation_family_probs), 1.0, atol=1e-6):
            raise ValueError(
                f"saturation_family_probs must sum to 1.0, got {sum(self.saturation_family_probs)}"
            )
        # Prior-shift eval: legacy edge-rate overrides
        if self.edge_rate_overrides is not None:
            unknown = sorted(set(self.edge_rate_overrides) - {"cy", "dc", "db", "zb"})
            if unknown:
                raise ValueError(
                    f"edge_rate_overrides only accepts legacy edge types "
                    f"('cy', 'dc', 'db', 'zb'), got {unknown}. The dz/zc/cc/zz "
                    f"rates have dedicated config fields."
                )
            for et, rate in self.edge_rate_overrides.items():
                if not 0.0 <= rate <= 1.0:
                    raise ValueError(f"edge_rate_overrides[{et!r}] must be in [0, 1], got {rate}")
        # Per-edge-type arrow budgets (pots)
        if self.edge_budget is not None:
            unknown = sorted(set(self.edge_budget) - set(EDGE_TYPES_EXTENDED))
            if unknown:
                raise ValueError(
                    f"edge_budget keys must be edge types {EDGE_TYPES_EXTENDED}, got {unknown}"
                )
            for et, spec in self.edge_budget.items():
                if isinstance(spec, bool):
                    raise ValueError(f"edge_budget[{et!r}] must be an int or (lo, hi), got bool")
                if isinstance(spec, (tuple, list)):
                    if len(spec) != 2:
                        raise ValueError(
                            f"edge_budget[{et!r}] range must be (lo, hi), got {spec!r}"
                        )
                    lo, hi = spec
                    if (
                        isinstance(lo, bool)
                        or isinstance(hi, bool)
                        or not (
                            isinstance(lo, (int, np.integer))
                            and isinstance(hi, (int, np.integer))
                            and 0 <= lo <= hi
                        )
                    ):
                        raise ValueError(
                            f"edge_budget[{et!r}] range must be ints with 0 <= lo <= hi, "
                            f"got {spec!r}"
                        )
                elif isinstance(spec, (int, np.integer)):
                    if spec < 0:
                        raise ValueError(f"edge_budget[{et!r}] must be >= 0, got {spec}")
                else:
                    raise ValueError(
                        f"edge_budget[{et!r}] must be an int or (lo, hi) tuple, got {type(spec)}"
                    )
        # Phase 4: additive-SCM priors
        for name in ("dz_base_rate", "zc_base_rate", "cc_base_rate", "zz_base_rate"):
            rate = getattr(self, name)
            if not 0.0 <= rate <= 1.0:
                raise ValueError(f"{name} must be in [0, 1], got {rate}")
        if self.cc_coeff_range[0] < 0:
            raise ValueError(
                f"cc_coeff_range must be positive-only (C->C amplify, not cannibalize), "
                f"got {self.cc_coeff_range}"
            )
        for name in ("rw_std_sigma", "rw_channel_std_sigma", "rw_sales_std_sigma"):
            sigma = getattr(self, name)
            if sigma <= 0:
                raise ValueError(f"{name} must be > 0, got {sigma}")
        if self.rw_positive_mean_range[0] <= 0:
            raise ValueError(
                f"rw_positive_mean_range must be positive (channel walks stay positive "
                f"after softplus), got {self.rw_positive_mean_range}"
            )
        # Channel texture
        if self.adstock_burn_in < 0:
            raise ValueError(f"adstock_burn_in must be >= 0, got {self.adstock_burn_in}")
        if 0 < self.adstock_burn_in < self.l_max:
            # A partial burn-in still convolves the first reported weeks into
            # zero padding — the warmup artifact the knob exists to remove.
            # Also catches dataclasses.replace(cfg, l_max=...) desyncing a
            # preset-built config (presets pin burn_in = l_max).
            raise ValueError(
                f"adstock_burn_in={self.adstock_burn_in} must be 0 (off) or >= "
                f"l_max ({self.l_max}) — a partial burn-in leaves adstock warmup "
                f"in the reported window"
            )
        if self.rw_channel_std_range is not None and not (
            0.0 <= self.rw_channel_std_range[0] <= self.rw_channel_std_range[1]
        ):
            raise ValueError(
                f"rw_channel_std_range must satisfy 0 <= lo <= hi, got {self.rw_channel_std_range}"
            )
        for name in ("channel_hf_sigma_range", "channel_pulse_amp_range"):
            lo, hi = getattr(self, name)
            if not 0.0 <= lo <= hi:
                raise ValueError(f"{name} must satisfy 0 <= lo <= hi, got {(lo, hi)}")
        p_lo, p_hi = self.channel_pulse_prob_range
        if not 0.0 <= p_lo <= p_hi <= 0.5:
            raise ValueError(
                f"channel_pulse_prob_range must satisfy 0 <= lo <= hi <= 0.5, got {(p_lo, p_hi)}"
            )
        if float(p_hi) > 0.0 and float(self.channel_pulse_amp_range[1]) <= 0.0:
            raise ValueError(
                "channel_pulse_prob_range enables pulses but channel_pulse_amp_range "
                "has zero amplitude — disable pulses via the prob range instead"
            )
        # Validate variable-size DAG ranges
        K_max_eff = self.K_max_effective
        M_max_eff = self.M_max_effective
        J_max_eff = self.J_max_effective
        # Clamp active ranges to effective max (backward compat: if K_max not set, use K)
        k_act_max = min(self.K_active_range[1], K_max_eff)
        m_act_max = min(self.M_active_range[1], M_max_eff)
        j_act_max = min(self.J_active_range[1], J_max_eff)
        if self.K_active_range[0] < 1:
            raise ValueError(f"K_active_range[0] must be >= 1, got {self.K_active_range[0]}")
        if self.M_active_range[0] < 1:
            raise ValueError(f"M_active_range[0] must be >= 1, got {self.M_active_range[0]}")
        if self.J_active_range[0] < 1:
            raise ValueError(f"J_active_range[0] must be >= 1, got {self.J_active_range[0]}")
        if k_act_max < self.K_active_range[0]:
            raise ValueError(
                f"K_max ({K_max_eff}) must be >= K_active_range[0] ({self.K_active_range[0]})"
            )
        if m_act_max < self.M_active_range[0]:
            raise ValueError(
                f"M_max ({M_max_eff}) must be >= M_active_range[0] ({self.M_active_range[0]})"
            )
        if j_act_max < self.J_active_range[0]:
            raise ValueError(
                f"J_max ({J_max_eff}) must be >= J_active_range[0] ({self.J_active_range[0]})"
            )


def _resolve_budget(rng: np.random.Generator, spec: int | tuple[int, int], n_eligible: int) -> int:
    """Resolve an edge-budget spec to a concrete arrow count, clamped to eligible pairs.

    An ``int`` N is an "up to N" cap: the per-task count is drawn uniformly in
    ``{0, ..., N}`` (never more than N). A ``(lo, hi)`` tuple draws uniformly in
    the inclusive range ``{lo, ..., hi}`` — use ``(N, N)`` for exactly N. Both
    forms are capped at ``n_eligible`` (the number of eligible source->dest
    pairs for the edge type).
    """
    if isinstance(spec, (tuple, list)):
        lo = max(0, min(int(spec[0]), n_eligible))
        hi = max(lo, min(int(spec[1]), n_eligible))
    else:
        lo = 0
        hi = max(0, min(int(spec), n_eligible))
    return int(rng.integers(lo, hi + 1))


def _scatter(rng: np.random.Generator, count: int, n_eligible: int) -> np.ndarray:
    """0/1 vector of length ``n_eligible`` with ``count`` ones at uniform-random slots.

    The "pot" mechanic: ``count`` arrows scattered over the eligible pairs.
    How they clump (one source fanning out vs. one arrow each) is emergent from
    the uniform draw.
    """
    out = np.zeros(n_eligible, dtype="float64")
    if count > 0 and n_eligible > 0:
        out[rng.choice(n_eligible, size=count, replace=False)] = 1.0
    return out


def _scatter_triu(rng: np.random.Generator, n_act: int, count: int, out_full: np.ndarray) -> None:
    """Place ``count`` arrows in the strict upper triangle of ``out_full[:n_act, :n_act]``.

    Strict upper triangle (src index < dst index) guarantees acyclicity for the
    C->C and Z->Z edge types.
    """
    if n_act < 2 or count <= 0:
        return
    iu = np.triu_indices(n_act, k=1)
    count = min(count, iu[0].size)
    sel = rng.choice(iu[0].size, size=count, replace=False)
    out_full[iu[0][sel], iu[1][sel]] = 1.0


def sample_g(
    rng: np.random.Generator,
    layout: SlotLayout,
    K_active: int | None = None,
    M_active: int | None = None,
    J_active: int | None = None,
    rates: dict[str, float] | None = None,
    budget: dict[str, int | tuple[int, int]] | None = None,
) -> dict[str, np.ndarray]:
    """Draw one DAG cell from the slot base rates (0/1 numpy arrays).

    For variable-size DAGs, pass K_active, M_active, J_active to restrict
    edges to active nodes only. Inactive nodes get zero-padded g-vectors
    and active masks.

    Parameters
    ----------
    rng : numpy random generator
    layout : SlotLayout with max sizes
    K_active, M_active, J_active : optional int
        Number of active nodes. If None, uses layout.K/M/J (all active).
    rates : optional dict
        Per-edge-type base-rate overrides for the legacy types
        ("cy", "dc", "db", "zb"); missing keys fall back to
        ``EDGE_BASE_RATES``. None (default) is byte-identical to the
        module constants (prior-shift eval support).
    budget : optional dict
        Per-edge-type arrow budget ("pot") for the legacy types: an "up to N"
        cap. When a type is present, its (uniformly drawn) count of arrows is
        scattered uniformly over eligible node pairs instead of drawn per-pair
        Bernoulli (see ``_resolve_budget`` / ``_scatter``). Types absent from
        the dict keep their Bernoulli rate; with ``budget=None`` (or ``{}``)
        the RNG stream is byte-identical to the legacy path.

    Returns
    -------
    dict with g-vectors, active masks, and active counts.
    """
    _rates = {**EDGE_BASE_RATES, **(rates or {})}
    _budget = budget or {}
    K_max = layout.K
    M_max = layout.M
    J_max = layout.J

    # Default: all nodes active (backward compat)
    if K_active is None:
        K_active = K_max
    if M_active is None:
        M_active = M_max
    if J_active is None:
        J_active = J_max

    # Clamp to valid range (ensure non-negative)
    K_active = max(0, min(K_active, K_max))
    M_active = max(0, min(M_active, M_max))
    J_active = max(0, min(J_active, J_max))

    # Generate edges only for active nodes; pad rest with zeros
    g_cy = np.zeros(K_max)
    if K_active > 0:
        if _budget.get("cy") is not None:
            n = _resolve_budget(rng, _budget["cy"], K_active)
            g_cy[:K_active] = _scatter(rng, n, K_active)
        else:
            g_cy[:K_active] = rng.binomial(1, _rates["cy"], size=K_active)
        # Degenerate guard: ensure at least one C->Y edge in active range
        if not g_cy[:K_active].any():
            g_cy[rng.integers(0, K_active)] = 1.0

    g_dc = np.zeros((J_max, K_max))
    if J_active > 0 and K_active > 0:
        if _budget.get("dc") is not None:
            n = _resolve_budget(rng, _budget["dc"], J_active * K_active)
            g_dc[:J_active, :K_active] = _scatter(rng, n, J_active * K_active).reshape(
                J_active, K_active
            )
        else:
            g_dc[:J_active, :K_active] = rng.binomial(1, _rates["dc"], size=(J_active, K_active))

    g_db = np.zeros(J_max)
    if J_active > 0:
        if _budget.get("db") is not None:
            n = _resolve_budget(rng, _budget["db"], J_active)
            g_db[:J_active] = _scatter(rng, n, J_active)
        else:
            g_db[:J_active] = rng.binomial(1, _rates["db"], size=J_active)

    g_zb = np.zeros(M_max)
    if M_active > 0:
        if _budget.get("zb") is not None:
            n = _resolve_budget(rng, _budget["zb"], M_active)
            g_zb[:M_active] = _scatter(rng, n, M_active)
        else:
            g_zb[:M_active] = rng.binomial(1, _rates["zb"], size=M_active)

    # Active-node masks (1 = node exists, 0 = padding)
    active_c = np.zeros(K_max)
    active_c[:K_active] = 1.0
    active_m = np.zeros(M_max)
    active_m[:M_active] = 1.0
    active_j = np.zeros(J_max)
    active_j[:J_active] = 1.0

    return {
        "g_cy": g_cy,
        "g_dc": g_dc,
        "g_db": g_db,
        "g_zb": g_zb,
        "active_c": active_c,
        "active_m": active_m,
        "active_j": active_j,
        "K_active": K_active,
        "M_active": M_active,
        "J_active": J_active,
    }


def sample_g_additive(
    rng: np.random.Generator,
    cfg: SCMPrior,
    layout: SlotLayout,
    K_active: int | None = None,
    M_active: int | None = None,
    J_active: int | None = None,
) -> dict[str, np.ndarray]:
    """Draw one extended DAG cell for the additive SCM (Phase 4).

    Extends :func:`sample_g` with the four new edge types. C->C and Z->Z
    edges are restricted to the strict upper triangle (src index < dst
    index) which guarantees acyclicity. Base rates for the new types come
    from ``cfg`` (dz/zc/cc/zz_base_rate); legacy types keep the slots.py
    base rates. When ``cfg.edge_budget`` names a type, that type's arrows are
    placed by budget (uniform scatter over eligible pairs) instead of by
    Bernoulli rate — see :func:`_resolve_budget` and :func:`_scatter`.

    Channel-activation rule (plan doc D1b): a channel is *causally active*
    iff it has a direct C->Y edge OR an outgoing C->C edge. The rule is
    reported in ``channel_active`` (all channels remain observed; the
    degenerate guard only forces at least one C->Y edge).

    Returns
    -------
    dict with the 8 g-blocks at max (padded) sizes — ``g_cc``/``g_zz`` as
    FULL square matrices with zero diagonals — plus active masks, active
    counts, and ``channel_active``.
    """
    base = sample_g(
        rng,
        layout,
        K_active=K_active,
        M_active=M_active,
        J_active=J_active,
        rates=cfg.edge_rate_overrides,
        budget=cfg.edge_budget,
    )
    K_max, M_max, J_max = layout.K, layout.M, layout.J
    K_act = base["K_active"]
    M_act = base["M_active"]
    J_act = base["J_active"]
    _budget = cfg.edge_budget or {}

    g_dz = np.zeros((J_max, M_max))
    if J_act > 0 and M_act > 0:
        if _budget.get("dz") is not None:
            n = _resolve_budget(rng, _budget["dz"], J_act * M_act)
            g_dz[:J_act, :M_act] = _scatter(rng, n, J_act * M_act).reshape(J_act, M_act)
        else:
            g_dz[:J_act, :M_act] = rng.binomial(1, cfg.dz_base_rate, size=(J_act, M_act))

    g_zc = np.zeros((M_max, K_max))
    if M_act > 0 and K_act > 0:
        if _budget.get("zc") is not None:
            n = _resolve_budget(rng, _budget["zc"], M_act * K_act)
            g_zc[:M_act, :K_act] = _scatter(rng, n, M_act * K_act).reshape(M_act, K_act)
        else:
            g_zc[:M_act, :K_act] = rng.binomial(1, cfg.zc_base_rate, size=(M_act, K_act))

    g_cc = np.zeros((K_max, K_max))
    if K_act > 1:
        if _budget.get("cc") is not None:
            n = _resolve_budget(rng, _budget["cc"], K_act * (K_act - 1) // 2)
            _scatter_triu(rng, K_act, n, g_cc)  # strict upper: src < dst
        else:
            draws = rng.binomial(1, cfg.cc_base_rate, size=(K_act, K_act))
            g_cc[:K_act, :K_act] = np.triu(draws, k=1)  # strict upper: src < dst

    g_zz = np.zeros((M_max, M_max))
    if M_act > 1:
        if _budget.get("zz") is not None:
            n = _resolve_budget(rng, _budget["zz"], M_act * (M_act - 1) // 2)
            _scatter_triu(rng, M_act, n, g_zz)
        else:
            draws = rng.binomial(1, cfg.zz_base_rate, size=(M_act, M_act))
            g_zz[:M_act, :M_act] = np.triu(draws, k=1)

    # Channel-activation rule (D1b): direct C->Y OR outgoing C->C
    channel_active = ((base["g_cy"] == 1) | (g_cc.sum(axis=1) > 0)).astype("float64")
    channel_active *= base["active_c"]  # padding nodes are never active

    return {
        **base,
        "g_dz": g_dz,
        "g_zc": g_zc,
        "g_cc": g_cc,
        "g_zz": g_zz,
        "channel_active": channel_active,
    }


def _signal_block(
    cfg: SCMPrior,
    layout: SlotLayout,
    spend_raw: np.ndarray,
    contributions_raw: np.ndarray,
    sales_raw: np.ndarray,
    g_tasks: np.ndarray,
    active_c_mask: np.ndarray,
    sales_scale: np.ndarray,
) -> dict:
    """``diagnostics["signal"]`` for a corpus.

    "Direct active channel" = cy edge present AND channel not padding.
    """
    cy_mask = (g_tasks[:, layout.slices["cy"]] == 1) & (active_c_mask == 1)
    return signal_summary(
        spend_raw,
        contributions_raw,
        sales_raw,
        cy_mask,
        sales_scale=sales_scale,
        l_max=cfg.l_max,
    )


def _make_support_mask(
    rng: np.random.Generator, T: int, n_query: int, p_long_horizon: float
) -> tuple[np.ndarray, int]:
    """Per-task support mask (u8, 1=support) and the split type.

    All splits are temporal (contiguous suffix) — no random week masking,
    which would leak future information in time series.

    Split types:
        0 = short_horizon: last n_query weeks are query (25% default)
        1 = long_horizon: last T//2 weeks are query (50%)
    """
    split_type = int(rng.random() < p_long_horizon)
    support = np.ones(T, dtype=np.uint8)
    if split_type == 1:
        # Long horizon: predict the second half
        query_start = T // 2
    else:
        # Short horizon: predict the last n_query weeks
        query_start = T - n_query
    support[query_start:] = 0
    return support, split_type


def _warn_flat_texture(cfg: SCMPrior) -> None:
    """Steer every caller to the ONE supported world prior.

    The blessed path is ``make_scm_prior(texture="diverse")`` — the
    additive SCM with high-frequency channel texture and adstock burn-in.
    A config with the flat (smooth-walk-only) channel prior still generates
    but warns: its contribution targets degenerate to near-flat lines
    (measured on the reference config: ~49% of direct-channel targets without
    week-to-week variation; see ``signal_diagnostics``).
    """
    if (
        float(cfg.channel_hf_sigma_range[1]) == 0.0
        and float(cfg.channel_pulse_prob_range[1]) == 0.0
    ):
        warnings.warn(
            "The flat (smooth-walk-only) channel texture is "
            "deprecated: it produces near-flat contribution targets the model cannot "
            "learn attribution from. Build configs with "
            "make_scm_prior(texture='diverse').",
            FutureWarning,
            stacklevel=3,
        )


def sample_prior_predictive(prior: SCMPrior, n: int | None = None) -> dict:
    """Draw a prior-predictive corpus: N SCMs and their data (dict of ndarrays).

    Each world routes through :func:`_generate_corpus_additive` — the additive
    causal SCM with the extended g-vector layout and exact interventional
    decomposition targets (``indirect_effects``, ``indirect_effects_by_source``,
    ``control_contribution``, ``confounder_contribution``, ``baseline_intrinsic``)
    plus ``channel_active``.

    Parameters
    ----------
    prior : SCMPrior
        The prior over SCMs (see :func:`prior_generator.make_scm_prior`).
    n : int, optional
        Number of worlds to return. If ``None`` (default), returns
        ``prior.n_cells * prior.draws_per_cell`` worlds. Otherwise the cell
        count is raised to cover ``n`` and the corpus is truncated to exactly
        ``n``.

    Notes
    -----
    Priors with the flat (texture-free) channel prior emit a ``FutureWarning``
    — build with ``make_scm_prior(texture="diverse")`` instead.
    """
    _warn_flat_texture(prior)
    prior.validate()
    if n is not None:
        if n <= 0:
            raise ValueError(f"n must be positive, got {n}")
        dpc = prior.draws_per_cell
        prior = replace(prior, n_cells=max(2, (n + dpc - 1) // dpc))
    corpus = _generate_corpus_additive(prior)
    if n is not None and corpus["spend_raw"].shape[0] > n:
        actual = corpus["spend_raw"].shape[0]
        for key, val in list(corpus.items()):
            if isinstance(val, np.ndarray) and val.ndim > 0 and val.shape[0] == actual:
                corpus[key] = val[:n]
        if "is_val" in corpus and corpus["is_val"].sum() == 0:
            corpus["is_val"][0] = 1  # keep at least one val world after truncation
    return corpus


# --------------------------------------------------------------------------
# Additive causal graph corpus
# --------------------------------------------------------------------------

_ADDITIVE_OUT_NAMES = (
    "demand",
    "controls",
    "channels",
    "baseline",
    "baseline_intrinsic",
    "control_contribution",
    "confounder_contribution",
    "contributions",
    "indirect_effects",
    "indirect_effects_by_source",
    "sales",
)


def _slice_g_active(
    g: dict[str, np.ndarray], K_act: int, M_act: int, J_act: int
) -> dict[str, np.ndarray]:
    """Restrict padded g-blocks to the active node ranges."""
    return {
        "g_cy": g["g_cy"][:K_act],
        "g_dc": g["g_dc"][:J_act, :K_act],
        "g_dz": g["g_dz"][:J_act, :M_act],
        "g_db": g["g_db"][:J_act],
        "g_zb": g["g_zb"][:M_act],
        "g_zc": g["g_zc"][:M_act, :K_act],
        "g_cc": g["g_cc"][:K_act, :K_act],
        "g_zz": g["g_zz"][:M_act, :M_act],
    }


def _additive_task_ok(
    spend: np.ndarray,
    sales: np.ndarray,
    arrays: dict[str, np.ndarray],
    g_cy_active: np.ndarray,
    cv_floor: float,
    sales_spike_ratio: float = 8.0,
    spend_spike_ratio: float = 50.0,
) -> bool:
    """Single-task realism filter for the additive SCM.

    Realism checks on the additive scale: finite
    arrays, non-negative sales, spend-CV floor on direct (C->Y) channels,
    and spike-ratio guards.
    """
    for a in arrays.values():
        if not np.isfinite(a).all():
            return False
    if (sales < 0).any():
        return False
    active = np.asarray(g_cy_active) == 1
    if active.any():
        cv = spend.std(axis=0) / (spend.mean(axis=0) + 1e-12)  # (K,)
        if cv[active].min() < cv_floor:
            return False
    sales_med = np.median(sales)
    safe_med = sales_med if sales_med > 0 else 1.0
    if sales.max() / safe_med >= sales_spike_ratio:
        return False
    spend_med = np.median(spend, axis=0)  # (K,)
    safe_spend_med = np.where(spend_med > 0, spend_med, 1.0)
    if (spend.max(axis=0) / safe_spend_med).max() >= spend_spike_ratio:
        return False
    return True


def _generate_corpus_additive(cfg: SCMPrior) -> dict:
    """Generate the Phase-4 additive-SCM corpus (plan doc 03, task 4.4).

    Same schema as :func:`sample_prior_predictive` (CONTRACTS §1) with the extended
    g-vector layout plus two new keys:

    * ``indirect_effects`` (N, T) float32 — total indirect effect on sales
      from all upstream influences flowing through channels.
    * ``channel_active`` (N, K) uint8 — the D1b activation rule
      (C->Y or outgoing C->C).

    Phase-5 per-node decomposition targets (float32, zero-padded to max sizes),
    emitted unconditionally. They are deterministic given the same eps inputs,
    so no extra randomness is consumed; note the acceptance filter now also
    checks finiteness of these arrays, so a task whose new targets were
    non-finite while the old outputs were finite would be rejected where it
    was previously accepted (not observed in practice — a non-finite variant
    implies a non-finite observed path):

    * ``control_contribution`` (N, T, M) — direct Z->B effect per control
      (``g_zb[m]·ρ[m]·Z[:, m]``). Name matches the legacy Phase-2 opt-in key so
      the existing per-control batch plumbing picks it up unchanged.
    * ``confounder_contribution`` (N, T, J) — direct D->B effect per confounder
      (``g_db[j]·δ[j]·D[:, j]``).
    * ``baseline_intrinsic`` (N, T) — ``RW_B + RW_Y`` (baseline minus all parent
      terms).
    * ``indirect_effects_by_source`` (N, T, 3) — telescoping 3-way indirect
      split in the LOCKED order ``(cc, zc, dc)``; the three columns sum exactly
      to ``indirect_effects``.

    The full additive invariant holds exactly (float64 pre-storage):
    ``baseline_intrinsic + Σ_j confounder_contribution + Σ_m control_contribution
    + Σ_k contributions + indirect_effects_by_source.sum(-1) == sales``.

    The additive decomposition invariant holds exactly:
    ``baseline_raw + contributions_raw.sum(-1) + indirect_effects == sales_raw``
    (up to float32 storage rounding).

    Each task samples fresh SCM parameters (matching the legacy θ-per-draw
    semantics) and builds/evaluates its own PyTensor graph at the cell's
    active sizes; outputs are zero-padded to max sizes.
    """
    from .world_model import build_world_model, draw_worlds, sample_structure

    t_start = time.perf_counter()
    rng = np.random.default_rng(cfg.seed)
    layout = cfg.layout  # extended edge types
    n_query = cfg.n_query
    K_max, M_max, J_max = layout.K, layout.M, layout.J
    T = cfg.T

    tasks: list[dict] = []
    cell_gs: list[dict[str, np.ndarray]] = []
    n_evaluated = 0
    n_rejected = 0

    for cell in range(cfg.n_cells):
        K_active = int(
            rng.integers(cfg.K_active_range_effective[0], cfg.K_active_range_effective[1] + 1)
        )
        M_active = int(
            rng.integers(cfg.M_active_range_effective[0], cfg.M_active_range_effective[1] + 1)
        )
        J_active = int(
            rng.integers(cfg.J_active_range_effective[0], cfg.J_active_range_effective[1] + 1)
        )
        g = sample_g_additive(
            rng, cfg, layout, K_active=K_active, M_active=M_active, J_active=J_active
        )
        cell_gs.append(g)
        g_act = _slice_g_active(g, K_active, M_active, J_active)

        # One pm.Model per cell (fixed structure: DAG + families + smoothness);
        # the continuous params and noise are the model's RVs, drawn in batches
        # and filtered by the realism gate. Building/compiling the graph
        # dominates at large (K, M), so a model per cell (not per draw) is key;
        # FAST_COMPILE (the draw_worlds default) keeps the one-off compile cheap.
        structural = sample_structure(g_act, cfg, rng)
        model, out_names, _param_names = build_world_model(g_act, cfg, structural, T)
        accepted: list[dict] = []
        last_draw_error: str | None = None
        for _round in range(MAX_TOPUPS_PER_CELL):
            if len(accepted) == cfg.draws_per_cell:
                break
            n_missing = cfg.draws_per_cell - len(accepted)
            n_req = n_missing + max(2, int(np.ceil(0.5 * n_missing)))
            draw_seed = int(rng.integers(2**31 - 1))
            try:
                drawn_b = draw_worlds(model, _ADDITIVE_OUT_NAMES, draw_seed, draws=n_req)
            except Exception as exc:
                # A sporadic pytensor py-linker evaluation crash on large graphs
                # (or any draw failure) — count the whole batch as rejected and
                # retry with a new seed; keep the error so a genuine, repeatable
                # failure surfaces in the RuntimeError below instead of being
                # disguised as filter strictness.
                last_draw_error = repr(exc)
                n_rejected += n_req
                n_evaluated += n_req
                continue

            for b in range(n_req):
                if len(accepted) == cfg.draws_per_cell:
                    break
                drawn = {name: drawn_b[name][b] for name in _ADDITIVE_OUT_NAMES}
                n_evaluated += 1

                if not _additive_task_ok(
                    spend=drawn["channels"],
                    sales=drawn["sales"],
                    arrays=drawn,
                    g_cy_active=g_act["g_cy"],
                    cv_floor=cfg.spend_cv_floor,
                ):
                    n_rejected += 1
                    continue

                support, split_type = _make_support_mask(rng, T, n_query, cfg.p_long_horizon)
                sales_scale = float(np.std(drawn["sales"][support == 1]))
                if not (np.isfinite(sales_scale) and sales_scale > 0.0):
                    n_rejected += 1
                    continue

                # Zero-pad active-size outputs to max sizes
                spend_pad = np.zeros((T, K_max))
                spend_pad[:, :K_active] = drawn["channels"]
                controls_pad = np.zeros((T, M_max))
                controls_pad[:, :M_active] = drawn["controls"]
                demand_pad = np.zeros((T, J_max))
                demand_pad[:, :J_active] = drawn["demand"]
                contrib_pad = np.zeros((T, K_max))
                contrib_pad[:, :K_active] = drawn["contributions"]
                # Phase 5 decomposition targets (zero-padded to max sizes)
                control_contrib_pad = np.zeros((T, M_max))
                control_contrib_pad[:, :M_active] = drawn["control_contribution"]
                confounder_contrib_pad = np.zeros((T, J_max))
                confounder_contrib_pad[:, :J_active] = drawn["confounder_contribution"]

                accepted.append(
                    {
                        "spend": spend_pad,
                        "controls": controls_pad,
                        "sales": drawn["sales"],
                        "contribution": contrib_pad,
                        "baseline": drawn["baseline"],
                        "demand": demand_pad,
                        "indirect_effects": drawn["indirect_effects"],
                        "baseline_intrinsic": drawn["baseline_intrinsic"],
                        "control_contribution": control_contrib_pad,
                        "confounder_contribution": confounder_contrib_pad,
                        "indirect_effects_by_source": drawn["indirect_effects_by_source"],
                        "support": support,
                        "split_type": split_type,
                        "sales_scale": sales_scale,
                        "cell": cell,
                    }
                )
        if len(accepted) < cfg.draws_per_cell:
            raise RuntimeError(
                f"cell {cell}: only {len(accepted)}/{cfg.draws_per_cell} tasks "
                f"accepted after {MAX_TOPUPS_PER_CELL} rounds — the realism filter "
                f"rejected the rest for this G-cell"
                + (f"; last draw error: {last_draw_error}" if last_draw_error else "")
            )
        tasks.extend(accepted)

    # -- assemble arrays (float64 math, float32 storage) --------------------
    n_tasks = len(tasks)
    spend_raw = np.stack([tk["spend"] for tk in tasks])  # (N,T,K)
    controls = np.stack([tk["controls"] for tk in tasks])  # (N,T,M)
    sales_raw = np.stack([tk["sales"] for tk in tasks])  # (N,T)
    contributions_raw = np.stack([tk["contribution"] for tk in tasks])
    baseline_raw = np.stack([tk["baseline"] for tk in tasks])  # (N,T)
    demand = np.stack([tk["demand"] for tk in tasks])  # (N,T,J)
    indirect_effects = np.stack([tk["indirect_effects"] for tk in tasks])  # (N,T)
    baseline_intrinsic = np.stack([tk["baseline_intrinsic"] for tk in tasks])  # (N,T)
    control_contribution = np.stack([tk["control_contribution"] for tk in tasks])  # (N,T,M)
    confounder_contribution = np.stack([tk["confounder_contribution"] for tk in tasks])  # (N,T,J)
    indirect_effects_by_source = np.stack(
        [tk["indirect_effects_by_source"] for tk in tasks]
    )  # (N,T,3)
    support_mask = np.stack([tk["support"] for tk in tasks]).astype(np.uint8)
    split_type = np.array([tk["split_type"] for tk in tasks], dtype=np.uint8)
    cell_id = np.array([tk["cell"] for tk in tasks], dtype=np.int32)

    active_c_mask = np.stack([cell_gs[c]["active_c"] for c in cell_id])
    active_m_mask = np.stack([cell_gs[c]["active_m"] for c in cell_id])
    active_j_mask = np.stack([cell_gs[c]["active_j"] for c in cell_id])
    channel_active = np.stack([cell_gs[c]["channel_active"] for c in cell_id])
    K_active_arr = np.array([cell_gs[c]["K_active"] for c in cell_id], dtype=np.int32)
    M_active_arr = np.array([cell_gs[c]["M_active"] for c in cell_id], dtype=np.int32)
    J_active_arr = np.array([cell_gs[c]["J_active"] for c in cell_id], dtype=np.int32)

    spend_means = spend_raw.mean(axis=1)  # (N,K)
    spend_norm = spend_raw / (spend_means[:, None, :] + 1e-8)
    active_spend_sum = (spend_raw * active_c_mask[:, None, :]).sum(axis=-1, keepdims=True)
    spend_share = (spend_raw / (active_spend_sum + 1e-8)) * active_c_mask[:, None, :]

    g_cells = np.stack(
        [
            layout.pack(
                g_cy=g["g_cy"],
                g_dc=g["g_dc"],
                g_db=g["g_db"],
                g_zb=g["g_zb"],
                g_dz=g["g_dz"],
                g_zc=g["g_zc"],
                g_cc=g["g_cc"],
                g_zz=g["g_zz"],
            )
            for g in cell_gs
        ]
    )  # (n_cells, S)
    g_tasks = g_cells[cell_id].astype(np.uint8)  # (N, S)

    # -- cell-level validation split (same logic as the legacy path) --------
    n_val_cells = max(1, int(round(cfg.val_cell_frac * cfg.n_cells)))
    if n_val_cells >= cfg.n_cells:
        n_val_cells = cfg.n_cells - 1
    val_cells = rng.permutation(cfg.n_cells)[:n_val_cells]
    is_val = np.isin(cell_id, val_cells).astype(np.uint8)

    val_mask = is_val == 1
    val_split_types = split_type[val_mask]
    if val_mask.sum() >= 2:
        if val_split_types.sum() == 0:
            idx_flip = np.flatnonzero(val_mask)[0]
            new_support, new_split = _make_support_mask(rng, T, n_query, 1.0)
            support_mask[idx_flip] = new_support
            split_type[idx_flip] = new_split
        elif val_split_types.sum() == val_mask.sum():
            idx_flip = np.flatnonzero(val_mask)[0]
            new_support, new_split = _make_support_mask(rng, T, n_query, 0.0)
            support_mask[idx_flip] = new_support
            split_type[idx_flip] = new_split

    sales_scale = np.array(
        [float(np.std(sales_raw[i][support_mask[i] == 1])) for i in range(n_tasks)],
        dtype=np.float64,
    )
    bad_scale = ~(np.isfinite(sales_scale) & (sales_scale > 0.0))
    if bad_scale.any():
        full_std = np.std(sales_raw[bad_scale], axis=1)
        sales_scale[bad_scale] = np.where(np.isfinite(full_std) & (full_std > 0.0), full_std, 1.0)
    sales_norm = sales_raw / sales_scale[:, None]

    # -- diagnostics ---------------------------------------------------------
    elapsed = time.perf_counter() - t_start
    qs = (0.1, 0.5, 0.9)
    cv_all = (spend_raw.std(axis=1) / (spend_raw.mean(axis=1) + 1e-12)).ravel()
    contrib_tot = contributions_raw.sum(axis=(1, 2))
    media_share = contrib_tot / (contrib_tot + baseline_raw.sum(axis=1) + 1e-12)
    decomp_err = np.abs(
        baseline_raw + contributions_raw.sum(axis=-1) + indirect_effects - sales_raw
    ).max()
    # Phase 5 invariants (float64 pre-storage error):
    #  (1) telescoping split sums to the total indirect series
    telescoping_err = np.abs(indirect_effects_by_source.sum(axis=-1) - indirect_effects).max()
    #  (2) full per-node additivity to sales
    full_decomp_err = np.abs(
        baseline_intrinsic
        + confounder_contribution.sum(axis=-1)
        + control_contribution.sum(axis=-1)
        + contributions_raw.sum(axis=-1)
        + indirect_effects_by_source.sum(axis=-1)
        - sales_raw
    ).max()
    #  (3) per-node baseline terms reconstruct the aggregate baseline
    baseline_decomp_err = np.abs(
        baseline_intrinsic
        + confounder_contribution.sum(axis=-1)
        + control_contribution.sum(axis=-1)
        - baseline_raw
    ).max()
    diagnostics = {
        "edge_types": list(layout.edge_types),
        "n_tasks": int(n_tasks),
        "n_cells": int(cfg.n_cells),
        "draws_per_cell": int(cfg.draws_per_cell),
        "elapsed_s": float(elapsed),
        "tasks_per_sec": float(n_tasks / elapsed),
        "n_draws_evaluated": int(n_evaluated),
        "rejection_rate": float(n_rejected / max(n_evaluated, 1)),
        "edge_marginals": {
            et: float(g_cells[:, layout.slices[et]].mean()) for et in layout.edge_types
        },
        "edge_base_rates": {
            "cy": float(EDGE_BASE_RATES["cy"]),
            "dc": float(EDGE_BASE_RATES["dc"]),
            "dz": float(cfg.dz_base_rate),
            "db": float(EDGE_BASE_RATES["db"]),
            "zb": float(EDGE_BASE_RATES["zb"]),
            "zc": float(cfg.zc_base_rate),
            "cc": float(cfg.cc_base_rate),
            "zz": float(cfg.zz_base_rate),
        },
        "edge_budget": (
            {
                et: list(v) if isinstance(v, (tuple, list)) else int(v)
                for et, v in cfg.edge_budget.items()
            }
            if cfg.edge_budget
            else None
        ),
        "decomposition_max_abs_error": float(decomp_err),
        "telescoping_split_max_abs_error": float(telescoping_err),
        "full_decomposition_max_abs_error": float(full_decomp_err),
        "baseline_decomposition_max_abs_error": float(baseline_decomp_err),
        "media_share_quantiles": {
            f"q{int(q * 100)}": float(np.quantile(media_share, q)) for q in qs
        },
        "spend_cv_quantiles": {f"q{int(q * 100)}": float(np.quantile(cv_all, q)) for q in qs},
        "signal": _signal_block(
            cfg,
            layout,
            spend_raw,
            contributions_raw,
            sales_raw,
            g_tasks,
            active_c_mask,
            sales_scale,
        ),
    }

    return {
        "spend_raw": spend_raw.astype(np.float32),
        "spend_norm": spend_norm.astype(np.float32),
        "spend_share": spend_share.astype(np.float32),
        "controls": controls.astype(np.float32),
        "sales_raw": sales_raw.astype(np.float32),
        "sales_norm": sales_norm.astype(np.float32),
        "support_mask": support_mask,
        "is_future": split_type,
        "g": g_tasks,
        "contributions_raw": contributions_raw.astype(np.float32),
        "baseline_raw": baseline_raw.astype(np.float32),
        "demand": demand.astype(np.float32),
        "spend_means": spend_means.astype(np.float32),
        "sales_scale": sales_scale.astype(np.float32),
        "is_val": is_val,
        "cell_id": cell_id,
        "active_c_mask": active_c_mask.astype(np.uint8),
        "active_m_mask": active_m_mask.astype(np.uint8),
        "active_j_mask": active_j_mask.astype(np.uint8),
        "K_active": K_active_arr,
        "M_active": M_active_arr,
        "J_active": J_active_arr,
        # Phase 4 additions
        "indirect_effects": indirect_effects.astype(np.float32),
        "channel_active": channel_active.astype(np.uint8),
        # Phase 5 decomposition targets
        "control_contribution": control_contribution.astype(np.float32),
        "confounder_contribution": confounder_contribution.astype(np.float32),
        "baseline_intrinsic": baseline_intrinsic.astype(np.float32),
        "indirect_effects_by_source": indirect_effects_by_source.astype(np.float32),
        "diagnostics": diagnostics,
    }
