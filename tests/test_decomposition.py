"""Decomposition invariants — the correctness contract of the generator.

These assert behaviour at the public level (``sample_prior_predictive`` /
``sample_scm``), so they hold regardless of how the SCM graph is built or
drawn internally. They are the regression net for any engine refactor.

The one exception is the texture fixture at the bottom: there is no public knob
that mutes ONE texture term of an already-sampled world, so it replays that
world's own recorded parameters and innovations through
``symbolic_graph.build_symbolic_graph`` with a single term muted. The world is
still drawn by ``sample_scm``; the replay only supplies the paired reference,
the same way ``tests/test_scm_audit.py`` does.
"""

from __future__ import annotations

import numpy as np
import pytest

import prior_generator as pg
from prior_generator.scenarios import SCENARIOS
from prior_generator.symbolic_graph import build_symbolic_graph


def _float32_storage_budget(*terms: np.ndarray) -> np.ndarray:
    """Elementwise worst-case residual of an identity over float32-persisted terms.

    ``sample_prior_predictive`` evaluates every decomposition component in
    float64 and stores it as float32, so reading one back gives
    ``stored = exact + delta`` with ``|delta| <= ulp32(exact) / 2``. A
    reconstruction identity re-adds several such terms, nothing forces their
    roundings to cancel, and the float64 re-addition itself contributes ~1e-16
    relative — eight orders below the float32 term — so the residual is bounded
    ELEMENTWISE by the sum of the individual half-ulps returned here.

    This is a derivation, not a fitted constant: it cannot be tightened without
    assuming the roundings cancel, and it leaves no room for a real decomposition
    error to hide. Measured over five corpus seeds the worst element used
    61-82% of the budget, while the ``1e-3 * max|sales|`` magic numbers this
    replaced were 12756x / 17496x / 8948x the observed residual — loose enough
    that a +0.05% corruption of ``contributions_raw`` still passed.

    ``np.spacing`` IS the float32 ulp at each magnitude, so exact zeros and
    subnormals need no special case. Terms carrying a trailing component axis
    (``contributions_raw``, ``*_contribution``, ``indirect_effects_by_source``)
    are reduced over it, matching the ``.sum(-1)`` in the identity itself.
    """
    budget = np.zeros(())
    for term in terms:
        half_ulp = 0.5 * np.spacing(np.abs(np.asarray(term, dtype=np.float32)))
        budget = budget + half_ulp.reshape(*term.shape[:2], -1).sum(-1, dtype=np.float64)
    return budget


@pytest.fixture(scope="module")
def corpus():
    cfg = pg.make_scm_prior(
        n_treatments=4,
        n_covariates=2,
        n_latent=1,
        n_time_steps=48,
        n_cells=2,
        draws_per_cell=4,
        seed=11,
        edge_budget={"cy": (3, 4), "cc": (1, 2), "zc": (1, 2), "dc": (1, 2)},
    )
    return pg.sample_prior_predictive(cfg)


def test_additive_identity(corpus):
    """sales_raw == baseline_raw + Σ_k contributions_raw + indirect_effects."""
    residual = np.abs(
        corpus["baseline_raw"].astype(np.float64)
        + corpus["contributions_raw"].astype(np.float64).sum(-1)
        + corpus["indirect_effects"].astype(np.float64)
        - corpus["sales_raw"].astype(np.float64)
    )
    budget = _float32_storage_budget(
        corpus["baseline_raw"],
        corpus["contributions_raw"],
        corpus["indirect_effects"],
        corpus["sales_raw"],
    )
    assert (residual <= budget).all(), (
        f"max|residual| {residual.max():.4g} exceeds the float32 storage budget "
        f"(worst element {(residual - budget).max():.4g} over)"
    )


def test_telescoping_split_sums_to_indirect(corpus):
    ibs = corpus["indirect_effects_by_source"].astype(np.float64)
    assert ibs.shape[-1] == 3  # locked order (cc, zc, dc)
    residual = np.abs(ibs.sum(-1) - corpus["indirect_effects"].astype(np.float64))
    budget = _float32_storage_budget(
        corpus["indirect_effects_by_source"], corpus["indirect_effects"]
    )
    assert (residual <= budget).all(), (
        f"max|residual| {residual.max():.4g} exceeds the float32 storage budget "
        f"(worst element {(residual - budget).max():.4g} over)"
    )


def test_full_per_node_additivity(corpus):
    """baseline_intrinsic + sales_noise + Σ confounder + Σ control + Σ direct + Σ by_source == sales."""
    f = lambda k: corpus[k].astype(np.float64)  # noqa: E731
    terms = (
        "baseline_intrinsic",
        "sales_noise",
        "confounder_contribution",
        "control_contribution",
        "contributions_raw",
        "indirect_effects_by_source",
    )
    recon = sum(f(name).sum(-1) if corpus[name].ndim == 3 else f(name) for name in terms)
    residual = np.abs(recon - f("sales_raw"))
    budget = _float32_storage_budget(*(corpus[name] for name in terms), corpus["sales_raw"])
    assert (residual <= budget).all(), (
        f"max|residual| {residual.max():.4g} exceeds the float32 storage budget "
        f"(worst element {(residual - budget).max():.4g} over)"
    )


def test_diagnostics_report_persisted_identity_error(corpus):
    """Diagnostics quantify reconstruction error from the stored float32 arrays."""
    expected_decomposition = np.abs(
        corpus["baseline_raw"]
        + corpus["contributions_raw"].sum(axis=-1)
        + corpus["indirect_effects"]
        - corpus["sales_raw"]
    ).max()
    expected_telescoping = np.abs(
        corpus["indirect_effects_by_source"].sum(axis=-1) - corpus["indirect_effects"]
    ).max()
    expected_full = np.abs(
        corpus["baseline_intrinsic"]
        + corpus["sales_noise"]
        + corpus["confounder_contribution"].sum(axis=-1)
        + corpus["control_contribution"].sum(axis=-1)
        + corpus["contributions_raw"].sum(axis=-1)
        + corpus["indirect_effects_by_source"].sum(axis=-1)
        - corpus["sales_raw"]
    ).max()
    expected_baseline = np.abs(
        corpus["baseline_intrinsic"]
        + corpus["sales_noise"]
        + corpus["confounder_contribution"].sum(axis=-1)
        + corpus["control_contribution"].sum(axis=-1)
        - corpus["baseline_raw"]
    ).max()

    diagnostics = corpus["diagnostics"]
    assert diagnostics["decomposition_max_abs_error"] == float(expected_decomposition)
    assert diagnostics["telescoping_split_max_abs_error"] == float(expected_telescoping)
    assert diagnostics["full_decomposition_max_abs_error"] == float(expected_full)
    assert diagnostics["baseline_decomposition_max_abs_error"] == float(expected_baseline)

    # Equality locks the persisted-float32 measurement contract; the ceiling
    # independently bounds the reconstruction residual itself.
    for name in (
        "decomposition_max_abs_error",
        "telescoping_split_max_abs_error",
        "full_decomposition_max_abs_error",
        "baseline_decomposition_max_abs_error",
    ):
        assert diagnostics[name] < 1e-5


def test_channels_positive_and_finite(corpus):
    assert np.isfinite(corpus["spend_raw"]).all()
    assert (corpus["spend_raw"] >= 0).all()
    assert np.isfinite(corpus["sales_raw"]).all()
    assert (corpus["sales_raw"] >= 0).all()


def test_inactive_channels_zero_padded():
    """Channels beyond the active count contribute nothing and carry no spend."""
    cfg = pg.make_scm_prior(
        n_treatments=6,
        n_covariates=3,
        n_latent=2,
        n_treatments_active_range=(3, 3),
        n_covariates_active_range=(2, 2),
        n_latent_active_range=(1, 1),
        n_time_steps=40,
        n_cells=2,
        draws_per_cell=2,
        seed=5,
    )
    corpus = pg.sample_prior_predictive(cfg)
    acm = corpus["treatment_active_mask"]
    # padded (inactive) channel slots are exactly zero in spend and contribution
    pad = acm == 0
    assert (corpus["spend_raw"][pad[:, None, :].repeat(corpus["spend_raw"].shape[1], 1)] == 0).all()
    contrib = corpus["contributions_raw"]
    assert (contrib[pad[:, None, :].repeat(contrib.shape[1], 1)] == 0).all()


def test_determinism_same_seed():
    kw = {
        "n_treatments": 4,
        "n_covariates": 2,
        "n_latent": 1,
        "n_time_steps": 40,
        "n_cells": 2,
        "draws_per_cell": 3,
        "seed": 99,
    }
    a = pg.sample_prior_predictive(pg.make_scm_prior(**kw))
    b = pg.sample_prior_predictive(pg.make_scm_prior(**kw))
    for key in ("spend_raw", "sales_raw", "g", "contributions_raw", "indirect_effects"):
        assert np.array_equal(a[key], b[key]), f"{key} not reproducible"


def test_different_seed_differs():
    a = pg.sample_prior_predictive(
        pg.make_scm_prior(n_treatments=4, n_covariates=2, n_latent=1, n_time_steps=40, seed=1)
    )
    b = pg.sample_prior_predictive(
        pg.make_scm_prior(n_treatments=4, n_covariates=2, n_latent=1, n_time_steps=40, seed=2)
    )
    assert not np.array_equal(a["sales_raw"], b["sales_raw"])


#: ``indirect_effects_by_source`` column order, locked by the corpus schema.
INDIRECT_SOURCES = ("cc", "zc", "dc")


def _budget_bounds(allowance: int | tuple[int, int]) -> tuple[int, int]:
    """Normalize an ``edge_budget`` entry to inclusive ``(low, high)`` arrow counts.

    A bare int is an "up to" cap (the count is drawn uniformly in ``{0..cap}``);
    a tuple is an explicit range. See :class:`~prior_generator.sampler.SCMPrior`.
    """
    return (0, allowance) if isinstance(allowance, int) else allowance


@pytest.mark.parametrize("scenario", SCENARIOS, ids=lambda sc: sc.name)
def test_scenario_isolates_the_pathway_it_names(scenario):
    """Every scenario honours its arrow budget and lights up only its own pathway.

    Each :class:`~prior_generator.scenarios.Scenario` exists so a decomposition
    failure can be traced to ONE causal route, which only works if the route it
    claims to isolate is the only live one. Two directions are asserted:

    * FORBIDDEN — a zero budget removes the arrow type outright, so the matching
      ``indirect_effects_by_source`` column is EXACTLY zero. This is structural:
      no arrow, no telescoping difference, at any seed.
    * REQUIRED — a budget with a positive lower bound puts the arrows in the
      graph, and the column carries the effect. This one additionally needs the
      arrows to REACH Y (a ``zc`` arrow into a channel with no ``C->Y`` edge and
      no outgoing ``C->C`` contributes nothing), so the seed is pinned.

    ``cfg.edge_budget`` is read back off the built config rather than off the
    scenario, because ``Scenario.prior`` substitutes ``connect_all_edge_budget``
    when connectivity is forced — reading the effective budget keeps this test
    from duplicating that policy.
    """
    cfg = scenario.prior(n_time_steps=52, seed=0)
    world = pg.sample_scm(
        cfg, seed=0, connect_all=scenario.connect_all, name=scenario.name, purpose=scenario.purpose
    )

    for edge, allowance in cfg.edge_budget.items():
        low, high = _budget_bounds(allowance)
        count = int(np.asarray(world.g[f"g_{edge}"]).sum())
        assert low <= count <= high, f"{edge} count {count} outside its budget {allowance}"

    by_source = world.data["indirect_effects_by_source"]
    assert by_source.shape[-1] == len(INDIRECT_SOURCES)
    for column, edge in enumerate(INDIRECT_SOURCES):
        low, high = _budget_bounds(cfg.edge_budget[edge])  # every scenario budgets all three
        peak = float(np.abs(by_source[:, column]).max())
        # A budget spanning zero (low == 0 < high) would decide neither way, and
        # silently dropping the claim is how this coverage rotted in the first
        # place — so demand a decisive budget instead.
        assert high == 0 or low >= 1, f"{edge} budget {cfg.edge_budget[edge]} decides nothing"
        if high == 0:
            assert peak == 0.0, f"{edge} is budgeted out yet its column is live ({peak:.3g})"
        else:
            assert peak > 0.0, f"{edge} is budgeted in yet its column is dead"

    # The scenario is only a usable diagnostic if its own decomposition closes.
    assert world.identity_error() < 1e-9


def _weekly_jaggedness(series: np.ndarray) -> np.ndarray:
    """Median |second difference| per column, in units of the column's own spread.

    ``x[t] - (x[t-1] + x[t+1]) / 2`` annihilates any locally-linear path, so a
    smoothed random walk scores ~0 however wide its swings are — which is
    exactly what a variance-based statistic like ``std / |mean|`` cannot tell
    apart. Taking the MEDIAN makes it specific to texture that touches EVERY
    week (the iid weekly term) rather than a sparse one (the pulse), so the two
    mechanisms below need different statistics and cannot cover for each other.
    """
    second_difference = np.abs(series[1:-1] - 0.5 * (series[:-2] + series[2:]))
    return np.median(second_difference, axis=0) / series.std(0)


def _causal_reach(fired: np.ndarray, l_max: int) -> np.ndarray:
    """Weeks an adstock kernel of length ``l_max`` can carry a fire into.

    Adstock is applied in ``ConvMode.After``: a week-``s`` fire reaches weeks
    ``s .. s + l_max - 1`` and no others. Everything downstream of the
    convolution (κ-relative saturation, the ``g_cy * beta`` gate) is pointwise,
    so this mask is the exact support of a pulse's effect on the contribution
    target.
    """
    reach = np.zeros_like(fired, dtype=bool)
    for lag in range(l_max):
        reach[lag:] |= fired[: fired.shape[0] - lag]
    return reach


@pytest.fixture(scope="module")
def texture_arms():
    """One sampled world, replayed with its channel texture selectively muted.

    The previous version of these tests asserted ``std / |mean| > 0.05`` on the
    contribution targets, which the texture-FREE arm below also satisfies
    (0.22-0.54 over six seeds) — it measured total variation, which a smooth
    random walk supplies on its own. The texture's actual job is the WEEKLY
    variation that sweeps the response curve, so each arm here holds the graph,
    the drawn parameters and the innovations fixed and mutes one mechanism:

    * ``quiet``   — no weekly jitter, no fires: the smooth-walk reference.
    * ``jittery`` — ``use_hf`` back on, still no fires.
    * ``pulsed``  — the world's own fires back on, still no jitter.

    The high-frequency term is toggled through ``use_hf`` because it is
    mean-zero and so never enters the parameter-only saturation anchor. The
    pulse is toggled through its FIRE indicator instead of ``use_pulse``:
    ``_expected_levels`` folds ``pulse_amp * pulse_prob`` into that anchor, so
    dropping the flag would also move every channel's saturation scale and stop
    being a per-week comparison. All three arms therefore share one anchor,
    asserted below.

    Adstock is pinned to geometric so no channel draws the min-max-rescaled
    Weibull kernel, which can annihilate individual lags (including the current
    week) and would make the per-channel weekly signal a family lottery rather
    than a texture measurement. ``cc`` is budgeted out so a fire on one channel
    cannot smear into another's column.
    """
    cfg = pg.make_scm_prior(
        n_treatments=3,
        n_covariates=1,
        n_latent=1,
        n_time_steps=52,
        seed=7,
        edge_budget={"cy": (3, 3), "cc": 0},
        adstock_family_probs={"none": 0.0, "geometric": 1.0, "weibull": 0.0},
        channel_hf_sigma_range=(0.4, 0.4),
        channel_pulse_prob_range=(0.15, 0.15),
        channel_pulse_amp_range=(1.5, 1.5),
    )
    world = pg.sample_scm(cfg, seed=7, connect_all=True)
    fired = world.exogenous["eps_c_pulse"] > 0

    def arm(*, use_hf: bool, fires: np.ndarray) -> dict[str, np.ndarray]:
        params = dict(world.params)
        params["use_hf"] = np.full(world.n_treatments, use_hf)
        params["use_pulse"] = np.full(world.n_treatments, True)
        eps = world.exogenous
        eps["eps_c_pulse"] = fires.astype(np.float64)
        outputs = build_symbolic_graph(
            world.g,
            params,
            world.n_time_steps,
            world.n_treatments,
            world.n_covariates,
            world.n_latent,
            burn_in=cfg.adstock_burn_in,
            eps=eps,
        )["outputs"]
        # Only the two outputs these tests read: evaluating the whole output
        # dict compiles ~30 graphs per arm and costs 15x as long.
        return {
            name: np.asarray(outputs[name].eval()) for name in ("contributions", "saturation_scale")
        }

    no_fires = np.zeros_like(fired)
    return (
        cfg,
        fired,
        {
            "quiet": arm(use_hf=False, fires=no_fires),
            "jittery": arm(use_hf=True, fires=no_fires),
            "pulsed": arm(use_hf=False, fires=fired),
        },
    )


def test_weekly_jitter_survives_the_response_mechanism(texture_arms):
    """The iid weekly term reaches the contribution target, not just the spend.

    Adstock is a low-pass filter and κ-relative saturation compresses the knee,
    so texture that moves spend does not automatically move the TARGET — which
    is the whole reason the texture prior exists. Measured 8.4-91x over six
    seeds and eighteen channels; the bound is half the observed floor. All three
    arms share one saturation anchor, so this is a per-week comparison and not a
    rescaling.
    """
    _cfg, _fired, arms = texture_arms
    np.testing.assert_array_equal(
        arms["jittery"]["saturation_scale"], arms["quiet"]["saturation_scale"]
    )
    smooth = _weekly_jaggedness(arms["quiet"]["contributions"])
    jittery = _weekly_jaggedness(arms["jittery"]["contributions"])
    assert (jittery > 4.0 * smooth).all(), (
        f"weekly jitter barely reached the target: jaggedness {jittery} vs {smooth}"
    )


def test_a_pulse_lifts_exactly_the_weeks_its_adstock_kernel_reaches(texture_arms):
    """A campaign pulse is an isolated, causal, bounded-support lift.

    The mechanism-specific signature, stated exactly: the pulse only ADDS spend,
    and softplus, the non-negative adstock kernel, saturation and the ``beta``
    gate are all monotone increasing, so the target can only rise. It rises on
    precisely the weeks a week-``s`` fire can reach through an ``l_max`` causal
    kernel — bit-identical everywhere else, including every week BEFORE a fire.
    Smooth walk texture cannot produce that support pattern, and neither can a
    pulse wired non-causally or leaked through the saturation anchor.
    """
    cfg, fired, arms = texture_arms
    np.testing.assert_array_equal(
        arms["pulsed"]["saturation_scale"], arms["quiet"]["saturation_scale"]
    )
    lift = arms["pulsed"]["contributions"] - arms["quiet"]["contributions"]
    reach = _causal_reach(fired, cfg.l_max)[cfg.adstock_burn_in :]

    # both classes present, or one of the two claims below is vacuous
    assert reach.any(axis=0).all() and (~reach).any(axis=0).all(), (
        f"every channel needs both reached and unreached weeks, got "
        f"{reach.sum(axis=0)} of {reach.shape[0]}"
    )
    assert (lift[reach] > 0.0).all(), "a reached week must be lifted"
    assert (lift[~reach] == 0.0).all(), "an unreached week must be untouched"
