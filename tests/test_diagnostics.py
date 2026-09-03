"""Generated-data diagnostics: schema preflight, estimators, coverage, closure.

Every numerical expectation here is an INDEPENDENT oracle — a literal, a
closed-form value, or a second implementation written from the definition —
never the production helper the test is checking.
"""

from __future__ import annotations

import json
import math
import random
from copy import deepcopy

import numpy as np
import pytest

import prior_generator as pg
from prior_generator import diagnostics as dg
from prior_generator.diagnostics import (
    CONTRIBUTION_FIELDS,
    DEFAULT_MAX_LAG,
    SERIES_SLOTS,
    data_diagnostics,
)

# ---------------------------------------------------------------------------
# Fixtures: a hand-built corpus with an exact decomposition
# ---------------------------------------------------------------------------


def make_corpus(
    *,
    spend: np.ndarray,
    controls: np.ndarray,
    demand: np.ndarray,
    contributions: np.ndarray,
    control_contribution: np.ndarray,
    confounder_contribution: np.ndarray,
    indirect_by_source: np.ndarray,
    baseline_intrinsic: np.ndarray,
    sales_noise: np.ndarray,
    channel_mask: np.ndarray,
    control_mask: np.ndarray,
    latent_mask: np.ndarray,
) -> dict[str, np.ndarray]:
    """A minimal current-schema corpus whose sales identity holds exactly."""
    baseline = (
        baseline_intrinsic + confounder_contribution.sum(axis=2) + control_contribution.sum(axis=2)
    )
    sales = baseline + sales_noise + contributions.sum(axis=2) + indirect_by_source.sum(axis=2)
    return {
        "spend_raw": spend,
        "controls": controls,
        "demand": demand,
        "sales_raw": sales,
        "baseline_raw": baseline,
        "baseline_intrinsic": baseline_intrinsic,
        "sales_noise": sales_noise,
        "control_contribution": control_contribution,
        "confounder_contribution": confounder_contribution,
        "contributions_raw": contributions,
        "indirect_effects_by_source": indirect_by_source,
        "indirect_effects": indirect_by_source.sum(axis=2),
        "sales_scale": np.maximum(sales.std(axis=1), 1e-3),
        "treatment_active_mask": channel_mask.astype(np.uint8),
        "covariate_active_mask": control_mask.astype(np.uint8),
        "latent_active_mask": latent_mask.astype(np.uint8),
    }


def toy_corpus(
    *, n_worlds: int = 3, n_time: int = 12, seed: int = 7, inactive_last_channel: bool = True
) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(seed)
    shape2 = (n_worlds, n_time)
    spend = rng.normal(size=(*shape2, 3))
    controls = rng.normal(size=(*shape2, 2))
    demand = rng.normal(size=(*shape2, 2))
    contributions = rng.normal(size=(*shape2, 3))
    control_contribution = rng.normal(size=(*shape2, 2))
    confounder_contribution = rng.normal(size=(*shape2, 2))
    indirect = rng.normal(size=(*shape2, 3))
    channel_mask = np.ones((n_worlds, 3), dtype=bool)
    control_mask = np.ones((n_worlds, 2), dtype=bool)
    latent_mask = np.ones((n_worlds, 2), dtype=bool)
    if inactive_last_channel:
        channel_mask[:, 2] = False
        spend[:, :, 2] = 0.0
        contributions[:, :, 2] = 0.0
        control_mask[-1, 1] = False
        controls[-1, :, 1] = 0.0
        control_contribution[-1, :, 1] = 0.0
    return make_corpus(
        spend=spend,
        controls=controls,
        demand=demand,
        contributions=contributions,
        control_contribution=control_contribution,
        confounder_contribution=confounder_contribution,
        indirect_by_source=indirect,
        baseline_intrinsic=rng.normal(size=shape2),
        sales_noise=rng.normal(size=shape2),
        channel_mask=channel_mask,
        control_mask=control_mask,
        latent_mask=latent_mask,
    )


@pytest.fixture(scope="module")
def toy() -> dict[str, np.ndarray]:
    return toy_corpus()


@pytest.fixture(scope="module")
def report(toy):
    return data_diagnostics(toy, scopes=("nodes", "decomposition"))


@pytest.fixture(scope="module")
def generated_corpus():
    cfg = pg.make_scm_prior(
        n_treatments=3,
        n_covariates=2,
        n_latent=2,
        n_time_steps=48,
        n_cells=2,
        draws_per_cell=2,
        seed=20260903,
    )
    return pg.sample_prior_predictive(cfg)


# ---------------------------------------------------------------------------
# 1-2. Preflight: padding, keys, shapes, degenerate sources
# ---------------------------------------------------------------------------


def test_exact_zero_padding_is_accepted(toy):
    report = data_diagnostics(toy)
    assert report.n_worlds == 3
    # C3 is padded off everywhere; the key stays known and simply has no worlds.
    assert "C3" in report.keys
    assert not report["levels"].series.eligible[:, report.keys.index("C3")].any()


@pytest.mark.parametrize("poison", [7.5, np.nan, np.inf])
def test_junk_inactive_padding_is_rejected_before_the_companion(toy, monkeypatch, poison):
    corpus = deepcopy(toy)
    corpus["spend_raw"][0, 3, 2] = poison

    def explode(*args, **kwargs):  # pragma: no cover - must never run
        raise AssertionError("outcome_distributions was called before preflight finished")

    monkeypatch.setattr(dg, "outcome_distributions", explode)
    # every poison — junk, NaN or inf — is localised to its world and column,
    # not reported as "this whole array is unusable"
    with pytest.raises(ValueError, match=r"non-zero inactive padding at world 0, column 2"):
        data_diagnostics(corpus)
    # rejection, never sanitation
    assert corpus["spend_raw"][0, 3, 2] == poison or math.isnan(corpus["spend_raw"][0, 3, 2])


def test_missing_key_and_bad_shapes_are_named(toy):
    corpus = deepcopy(toy)
    del corpus["demand"]
    with pytest.raises(KeyError, match="demand"):
        data_diagnostics(corpus)

    corpus = deepcopy(toy)
    corpus["controls"] = corpus["controls"][:, :, :1]
    with pytest.raises(ValueError, match="controls must have shape"):
        data_diagnostics(corpus)

    corpus = deepcopy(toy)
    corpus["treatment_active_mask"] = np.full((3, 3), 2, dtype=np.uint8)
    with pytest.raises(ValueError, match="binary 0/1 mask"):
        data_diagnostics(corpus)


def test_empty_sources_error_early(toy):
    corpus = deepcopy(toy)
    for key, value in corpus.items():
        corpus[key] = value[:0] if isinstance(value, np.ndarray) else value
    with pytest.raises(ValueError, match="no worlds to diagnose|no data to diagnose"):
        data_diagnostics(corpus)
    with pytest.raises(ValueError, match="no worlds to diagnose"):
        data_diagnostics([])


def test_counterfactual_scope_needs_scm_worlds(toy):
    with pytest.raises(ValueError, match="counterfactual"):
        data_diagnostics(toy, scopes=("nodes", "counterfactual"))


def test_single_time_step_gives_empty_differences():
    corpus = toy_corpus(n_time=1, inactive_last_channel=False)
    report = data_diagnostics(corpus)
    assert report["levels"].series.n_time_steps == 1
    assert report["differences"].series.n_time_steps == 0
    assert report.lags == ()
    # level descriptive slots still exist; roughness/spike need 2 and 3 points.
    levels = report["levels"].series
    assert levels.slot_valid[:, :, SERIES_SLOTS.index("mean")].all()
    assert not levels.slot_valid[:, :, SERIES_SLOTS.index("roughness")].any()
    # a one-step window has no lags at all: every report must SAY so rather
    # than crash on an empty column axis
    assert "no columns to report" in report["levels"].temporal.table()
    assert report["levels"].temporal.matrix("acf").shape == (len(report.keys), 0)


# ---------------------------------------------------------------------------
# 3. Selectors
# ---------------------------------------------------------------------------


def test_world_selector_forms_agree(toy):
    full = data_diagnostics(toy)
    assert np.array_equal(full.world_ids, [0, 1, 2])
    assert np.array_equal(data_diagnostics(toy, worlds=slice(1, None)).world_ids, [1, 2])
    assert np.array_equal(data_diagnostics(toy, worlds=[2, 0]).world_ids, [2, 0])
    assert np.array_equal(data_diagnostics(toy, worlds=1).world_ids, [1])
    mask = np.array([False, True, True])
    assert np.array_equal(data_diagnostics(toy, worlds=mask).world_ids, [1, 2])


@pytest.mark.parametrize(
    ("selector", "error"),
    [
        ([1, 1], ValueError),
        ([0.0, 1.0], TypeError),
        (["0"], TypeError),
        (True, TypeError),
        (np.zeros((2, 2), dtype=np.int64), ValueError),
        (np.array([True, False]), ValueError),
        ([5], IndexError),
        (slice(3, None), ValueError),
        (np.ma.array([0, 1, 2], mask=[1, 0, 0]), TypeError),
    ],
)
def test_lossy_world_selectors_are_rejected(toy, selector, error):
    with pytest.raises(error):
        data_diagnostics(toy, worlds=selector)


def test_scope_view_lag_quantile_key_selectors(toy):
    with pytest.raises(ValueError, match="must include"):
        data_diagnostics(toy, scopes=("decomposition",))
    with pytest.raises(ValueError, match="unknown scopes"):
        data_diagnostics(toy, scopes=("nodes", "nope"))
    with pytest.raises(TypeError, match="non-string sequence"):
        data_diagnostics(toy, scopes="nodes")
    with pytest.raises(ValueError, match="views is empty"):
        data_diagnostics(toy, views=())
    with pytest.raises(ValueError, match="repeats"):
        data_diagnostics(toy, views=("levels", "levels"))
    with pytest.raises(ValueError, match="lags must be >= 1"):
        data_diagnostics(toy, lags=(0,))
    with pytest.raises(TypeError, match="plain integers"):
        data_diagnostics(toy, lags=(1.0,))
    with pytest.raises(ValueError, match="quantile levels"):
        data_diagnostics(toy, quantiles=(0.5, 1.5))
    with pytest.raises(ValueError, match="quantiles is empty"):
        data_diagnostics(toy, quantiles=())


def test_overlarge_lags_are_kept_but_invalid(toy):
    report = data_diagnostics(toy, lags=(1, 99))
    temporal = report["levels"].temporal
    assert temporal.lags == (1, 99)
    assert temporal.acf_valid[:, :, 1].sum() == 0
    assert np.isnan(temporal.acf[:, :, 1]).all()


# ---------------------------------------------------------------------------
# 4. Key vocabulary
# ---------------------------------------------------------------------------


def test_keys_are_unique_ordered_and_separate_from_labels(toy):
    report = data_diagnostics(toy, scopes=("nodes", "decomposition"))
    keys = report.keys
    assert len(set(keys)) == len(keys)
    assert keys[:7] == ("C1", "C2", "C3", "Z1", "Z2", "D1", "D2")
    assert keys[7:9] == ("B", "Y")
    assert "C1_direct_y" in keys and "Z1_baseline_alloc" in keys and "media_total" in keys
    # display labels are NOT unique selectors: two keys, one human reading.
    assert report.descriptor("C1").display_label != report.descriptor("C1_direct_y").display_label
    assert report.descriptor("Z1_baseline_alloc").display_label.endswith(
        "retained baseline allocation"
    )


def test_key_universe_is_derived_before_world_selection(toy):
    full = data_diagnostics(toy, scopes=("nodes", "decomposition"))
    one = data_diagnostics(toy, worlds=[2], scopes=("nodes", "decomposition"))
    assert one.keys == full.keys
    assert one.world_ids.tolist() == [2]
    # the world-2 row of the full report equals the one-world report
    full_slots = full["levels"].series.slots[2]
    assert np.allclose(one["levels"].series.slots[0], full_slots, equal_nan=True)


# ---------------------------------------------------------------------------
# 5-7. Extraction, companion, raw access
# ---------------------------------------------------------------------------


def test_float64_variation_below_float32_resolution_survives():
    n_time = 8
    base = 1.0 + np.arange(n_time) * 1e-9
    corpus = toy_corpus(n_worlds=1, n_time=n_time, inactive_last_channel=False)
    corpus["spend_raw"][0, :, 0] = base
    assert np.float32(base).std() == 0.0  # float32 would erase the whole signal
    report = data_diagnostics(corpus, views=("levels",))
    values = report["levels"].series.values("C1")
    assert values.dtype == np.float64
    assert np.array_equal(values, base)
    assert report["levels"].series.slot("std")[0, 0] > 0.0


def test_companion_is_called_once_without_series(toy, monkeypatch):
    calls: list[dict] = []
    real = dg.outcome_distributions

    def spy(source, **kwargs):
        calls.append(kwargs)
        return real(source, **kwargs)

    monkeypatch.setattr(dg, "outcome_distributions", spy)
    report = data_diagnostics(toy, worlds=[1, 2])
    assert len(calls) == 1
    assert calls[0]["keep_series"] is False
    assert report.outcomes.n_worlds == 2
    assert report.outcomes["sales"].series is None


@pytest.mark.parametrize("selector", [None, [1, 2], [2, 0], [2, 1, 0], [0]])
def test_companion_rows_line_up_with_the_report(toy, selector):
    """A row-wise comparison must not silently pair different worlds.

    The companion takes a selector of its own, so a permuted selection could
    come back in corpus order while the report stays in caller order.
    """
    report = data_diagnostics(toy, worlds=selector, scopes=("nodes", "decomposition"))
    assert np.array_equal(report.outcomes.world_ids, report.world_ids)
    assert np.allclose(
        report.outcomes["sales"].unit_total,
        report.contributions.sales_total,
        rtol=1e-9,
        atol=1e-9,
    )


def test_raw_values_are_the_eligible_rows(toy):
    report = data_diagnostics(toy, views=("levels", "differences"))
    index = report.keys.index("Z2")
    eligible = np.array([True, True, False])
    assert np.array_equal(report["levels"].series.eligible[:, index], eligible)
    expected = toy["controls"][eligible, :, 1].reshape(-1)
    assert np.array_equal(report["levels"].series.values("Z2"), expected)
    diffs = np.diff(toy["controls"][eligible, :, 1], axis=1).reshape(-1)
    assert np.array_equal(report["differences"].series.values("Z2"), diffs)
    # differences never cross a world boundary
    assert report["differences"].series.values("Z2").size == 2 * (toy["controls"].shape[1] - 1)


def test_lean_mode_changes_only_raw_access(toy):
    full = data_diagnostics(toy, scopes=("nodes", "decomposition"))
    lean = data_diagnostics(toy, scopes=("nodes", "decomposition"), keep_series=False)
    for view in ("levels", "differences"):
        assert np.allclose(full[view].series.slots, lean[view].series.slots, equal_nan=True)
        for metric in ("pearson", "spearman", "xi", "xi_max"):
            assert np.allclose(
                full[view].dependence.matrices[metric],
                lean[view].dependence.matrices[metric],
                equal_nan=True,
            )
        assert np.allclose(full[view].temporal.acf, lean[view].temporal.acf, equal_nan=True)
        assert np.allclose(
            full[view].vif["oracle"].vif, lean[view].vif["oracle"].vif, equal_nan=True
        )
    # ... everything except the flag that truthfully records what was kept
    full_summary, lean_summary = full.summary(), lean.summary()
    assert full_summary.pop("keep_series") is True
    assert lean_summary.pop("keep_series") is False
    for payload in (full_summary, lean_summary):
        for view in payload["by_view"].values():
            view["series"].pop("keep_series")
    assert json.dumps(full_summary, sort_keys=True) == json.dumps(lean_summary, sort_keys=True)
    with pytest.raises(ValueError, match="keep_series=False"):
        lean["levels"].series.values("C1")


# ---------------------------------------------------------------------------
# 8-9. Descriptive slots and finite-only aggregation
# ---------------------------------------------------------------------------


def test_eight_slots_against_literal_oracles():
    series = np.array([0.0, 2.0, 1.0, 5.0, 4.0, 4.0, 3.0, 20.0])
    corpus = toy_corpus(n_worlds=1, n_time=series.size, inactive_last_channel=False)
    corpus["spend_raw"][0, :, 0] = series
    slots = data_diagnostics(corpus, views=("levels",))["levels"].series.slots[0, 0]

    assert slots[SERIES_SLOTS.index("mean")] == pytest.approx(39.0 / 8.0)
    assert slots[SERIES_SLOTS.index("std")] == pytest.approx(
        math.sqrt(sum((v - 39.0 / 8.0) ** 2 for v in series) / 8.0)
    )
    assert slots[SERIES_SLOTS.index("median")] == pytest.approx(3.5)
    # NumPy's default linear quantile: q25 = 1.75, q75 = 4.25
    assert slots[SERIES_SLOTS.index("iqr")] == pytest.approx(4.25 - 1.75)
    assert slots[SERIES_SLOTS.index("min")] == 0.0
    assert slots[SERIES_SLOTS.index("max")] == 20.0

    steps = np.diff(series)
    roughness = steps.std() / (math.sqrt(2.0) * series.std())
    assert slots[SERIES_SLOTS.index("roughness")] == pytest.approx(roughness)

    centred = np.abs(steps - np.median(steps)).max()
    spread = np.quantile(steps, 0.75) - np.quantile(steps, 0.25)
    assert slots[SERIES_SLOTS.index("spike")] == pytest.approx(centred / spread)


@pytest.mark.parametrize(
    ("series", "roughness", "spike"),
    [
        (np.arange(10, dtype=float), 0.0, 0.0),  # linear ramp: no roughness, no spike
        (np.zeros(10), 0.0, 0.0),  # structural zero is real data, not missing
        (np.array([0.0] * 5 + [1.0] + [0.0] * 4), None, float("inf")),  # single pulse
    ],
)
def test_roughness_and_spike_edge_cases(series, roughness, spike):
    corpus = toy_corpus(n_worlds=1, n_time=series.size, inactive_last_channel=False)
    corpus["spend_raw"][0, :, 0] = series
    slots = data_diagnostics(corpus, views=("levels",))["levels"].series.slots[0, 0]
    if roughness is not None:
        assert slots[SERIES_SLOTS.index("roughness")] == pytest.approx(roughness)
    assert slots[SERIES_SLOTS.index("spike")] == spike


def test_finite_summaries_exclude_infinities_and_ledgers_balance():
    values = np.array([1.0, 3.0, np.inf, -np.inf, np.nan])
    valid = np.array([True, True, True, True, False])
    stats = dg._finite_stats(values, valid, (0.5,))
    ledger = dg._ledger(values, valid, selected=6)
    assert stats["n"] == 2
    assert stats["mean"] == pytest.approx(2.0)
    assert stats["q50"] == pytest.approx(2.0)
    assert (ledger.valid, ledger.finite, ledger.positive_infinite, ledger.negative_infinite) == (
        4,
        2,
        1,
        1,
    )
    assert ledger.valid == ledger.finite + ledger.positive_infinite + ledger.negative_infinite
    assert ledger.eligible == ledger.valid + ledger.invalid
    assert ledger.selected >= ledger.eligible

    all_infinite = dg._finite_stats(np.array([np.inf, np.inf]), np.array([True, True]), (0.5,))
    assert all_infinite["mean"] is None and all_infinite["n"] == 0
    nothing = dg._finite_stats(np.array([1.0]), np.array([False]), (0.5,))
    assert nothing["mean"] is None


# ---------------------------------------------------------------------------
# 10-12. Dependence
# ---------------------------------------------------------------------------


def _dependence_for(rows: np.ndarray):
    n_time = rows.shape[1]
    corpus = toy_corpus(n_worlds=1, n_time=n_time, inactive_last_channel=False)
    corpus["spend_raw"][0, :, 0] = rows[0]
    corpus["spend_raw"][0, :, 1] = rows[1]
    report = data_diagnostics(corpus, views=("levels",))
    return report["levels"].dependence, report.keys


def test_dependence_needs_three_observations():
    two = _dependence_for(np.array([[0.0, 1.0], [0.0, 2.0]]))[0]
    for metric in ("pearson", "spearman", "xi", "xi_max"):
        assert not two.valid[metric].any()
    three = _dependence_for(np.array([[0.0, 1.0, 2.0], [0.0, 2.0, 5.0]]))[0]
    assert three.valid["pearson"][0, 0, 1]
    # by hand: cov = 5, sd_x = sqrt(2), sd_y = sqrt(114)/3 -> 15 / sqrt(228)
    assert three.matrices["pearson"][0, 0, 1] == pytest.approx(15.0 / math.sqrt(228.0))


def test_constant_series_and_diagonal_are_not_available():
    dep, keys = _dependence_for(np.array([[1.0] * 6, [0.0, 1.0, 2.0, 3.0, 4.0, 5.0]]))
    i, j = keys.index("C1"), keys.index("C2")
    assert not dep.valid["pearson"][0, i, j]
    assert not dep.valid["spearman"][0, i, j]
    # xi tolerates a constant PREDICTOR and answers exactly zero
    assert dep.valid["xi"][0, i, j]
    assert dep.matrices["xi"][0, i, j] == pytest.approx(0.0, abs=1e-12)
    assert not dep.valid["xi"][0, j, i]  # constant TARGET has no answer
    assert not dep.valid["xi_max"][0, i, j]
    assert np.isnan(dep.matrices["pearson"][0, i, i])
    assert not dep.valid["xi"][0, i, i]


def test_eligible_but_undefined_is_invalid_not_absent():
    """A constant active series was ASKED and had no answer; say that."""
    corpus = toy_corpus(inactive_last_channel=False)
    corpus["spend_raw"][:, :, 0] = 3.0
    report = data_diagnostics(corpus, views=("levels",))
    levels = report["levels"]
    _, dependence = levels.dependence.pair("C1", "C2", "pearson")
    _, temporal = levels.temporal.stats("C1", 1, "acf")
    _, vif = levels.vif["observed"].stats("C1")
    for ledger in (dependence, temporal, vif):
        assert ledger.eligible == 3, ledger
        assert ledger.valid == 0 and ledger.invalid == 3, ledger
        assert ledger.selected >= ledger.eligible
    # and a pair whose partner never exists reports no observations at all
    padded = data_diagnostics(toy_corpus(), views=("levels",))
    _, absent = padded["levels"].dependence.pair("C1", "C3", "pearson")
    assert absent.eligible == 0 and absent.observations == 0


def test_correlations_stay_inside_their_range_and_near_constants_are_flagged():
    exact = np.arange(8.0)
    dep, keys = _dependence_for(np.stack([exact, 2.0 * exact + 1.0]))
    value = dep.matrices["pearson"][0, keys.index("C1"), keys.index("C2")]
    assert value <= 1.0 and value == pytest.approx(1.0)

    # a column that is constant up to one ULP is noise, not a predictor
    near_constant = np.full(8, 1.0)
    near_constant[3] += 1e-17
    _, valid, constant, rank, condition = dg._vif_world(np.stack([near_constant, exact]))
    assert constant[0] and not valid[0]
    assert rank == 1 and np.isposinf(condition)


def test_pearson_is_signed_and_spearman_uses_average_ranks():
    x = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
    dep, keys = _dependence_for(np.stack([x, -x]))
    i, j = keys.index("C1"), keys.index("C2")
    assert dep.matrices["pearson"][0, i, j] == pytest.approx(-1.0)

    tied = np.array([1.0, 1.0, 2.0, 3.0, 4.0, 5.0])
    monotone = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
    dep, keys = _dependence_for(np.stack([tied, monotone]))
    ranks_x = np.array([1.5, 1.5, 3.0, 4.0, 5.0, 6.0])
    expected = np.corrcoef(ranks_x, np.arange(1.0, 7.0))[0, 1]
    assert dep.matrices["spearman"][0, keys.index("C1"), keys.index("C2")] == pytest.approx(
        expected
    )


def _xi_bruteforce(x: np.ndarray, y: np.ndarray) -> float:
    """xi averaged over EVERY ordering the ties in x admit (exhaustive)."""
    from itertools import permutations

    n = x.size
    below = np.array([np.count_nonzero(y <= value) for value in y], dtype=float)
    above = np.array([np.count_nonzero(y >= value) for value in y], dtype=float)
    denominator = float((above * (n - above)).sum())
    groups: dict[float, list[int]] = {}
    for index, value in enumerate(x):
        groups.setdefault(float(value), []).append(index)
    blocks = [groups[key] for key in sorted(groups)]

    def orderings(remaining):
        if not remaining:
            yield []
            return
        head, *tail = remaining
        for permutation in permutations(head):
            for rest in orderings(tail):
                yield [*permutation, *rest]

    totals = []
    for order in orderings(blocks):
        ranks = below[list(order)]
        totals.append(float(np.abs(np.diff(ranks)).sum()))
    expected = sum(totals) / len(totals)
    return 1.0 - n * expected / (2.0 * denominator)


def test_xi_tie_average_matches_exhaustive_enumeration():
    x = np.array([0.0, 0.0, 1.0, 1.0, 1.0, 2.0, 2.0])
    y = np.array([3.0, 1.0, 4.0, 1.0, 5.0, 9.0, 2.0])
    assert dg._chatterjee_xi_tie_average(x, y) == pytest.approx(_xi_bruteforce(x, y), abs=1e-12)


def test_xi_is_invariant_to_order_inside_ties_and_to_rng(monkeypatch):
    x = np.array([0.0, 0.0, 1.0, 1.0, 1.0, 2.0, 2.0])
    y = np.array([3.0, 1.0, 4.0, 1.0, 5.0, 9.0, 2.0])
    reference = dg._chatterjee_xi_tie_average(x, y)

    def poisoned(*args, **kwargs):  # pragma: no cover - must never run
        raise AssertionError("xi must not consult any RNG")

    monkeypatch.setattr(np.random, "shuffle", poisoned)
    monkeypatch.setattr(np.random, "permutation", poisoned)
    monkeypatch.setattr(random, "shuffle", poisoned)

    rng = np.random.default_rng(3)
    for _ in range(20):
        order = np.arange(x.size)
        for value in np.unique(x):
            block = np.flatnonzero(x == value)
            order[block] = rng.permutation(block)
        assert dg._chatterjee_xi_tie_average(x[order], y[order]) == pytest.approx(
            reference, abs=1e-12
        )


def test_xi_is_directional_and_finds_nonlinear_structure():
    rng = np.random.default_rng(11)
    x = rng.normal(size=200)
    y = x**2
    forward = dg._chatterjee_xi_tie_average(x, y)
    backward = dg._chatterjee_xi_tie_average(y, x)
    assert forward > 0.7
    assert backward < forward
    assert abs(np.corrcoef(x, y)[0, 1]) < 0.2


def test_xi_ceiling_and_negative_values():
    n = 12
    x = np.arange(float(n))
    # a perfectly monotone target cannot exceed (n - 2) / (n + 1)
    assert dg._chatterjee_xi_tie_average(x, x) == pytest.approx((n - 2) / (n + 1))
    # the sample statistic can be negative and must not be clipped
    zigzag = np.array([0.0, 5.0, 1.0, 6.0, 2.0, 7.0, 3.0, 8.0])
    assert dg._chatterjee_xi_tie_average(np.arange(8.0), zigzag) < 0.0
    assert dg._chatterjee_xi_tie_average(np.arange(3.0), np.ones(3)) != (
        dg._chatterjee_xi_tie_average(np.arange(3.0), np.ones(3))
    )  # NaN for a constant target


def test_xi_max_is_taken_per_world_before_aggregation():
    # max of averages != average of maxima; the report must use the former.
    x = np.array([0.0, 0.0, 1.0])
    y = np.array([0.0, 1.0, 0.0])
    forward = dg._chatterjee_xi_tie_average(x, y)
    backward = dg._chatterjee_xi_tie_average(y, x)
    assert max(forward, backward) == pytest.approx(-0.125)

    rows = np.stack([np.tile(x, 3), np.tile(y, 3)])
    dep, keys = _dependence_for(rows)
    i, j = keys.index("C1"), keys.index("C2")
    stored = dep.matrices["xi_max"][0, i, j]
    assert stored == pytest.approx(max(dep.matrices["xi"][0, i, j], dep.matrices["xi"][0, j, i]))
    assert dep.matrices["xi_max"][0, i, j] == dep.matrices["xi_max"][0, j, i]


def test_batched_xi_matches_the_scalar_reference():
    rng = np.random.default_rng(5)
    x = rng.normal(size=(6, 40))
    y = np.round(rng.normal(size=(6, 40)), 1)  # ties on purpose
    x[3] = np.round(x[3], 1)
    batch = dg._xi_batch(x, y)
    for row in range(x.shape[0]):
        assert batch[row] == pytest.approx(dg._chatterjee_xi_tie_average(x[row], y[row]))


def test_xi_matrix_orientation_is_predictor_row_target_column():
    rng = np.random.default_rng(13)
    x = rng.normal(size=60)
    rows = np.stack([x, x**2])
    dep, keys = _dependence_for(rows)
    i, j = keys.index("C1"), keys.index("C2")
    assert dep.matrices["xi"][0, i, j] == pytest.approx(dg._chatterjee_xi_tie_average(x, x**2))
    assert dep.matrices["xi"][0, j, i] == pytest.approx(dg._chatterjee_xi_tie_average(x**2, x))


# ---------------------------------------------------------------------------
# 13. Temporal
# ---------------------------------------------------------------------------


def test_default_lag_axis_is_contiguous_and_capped(toy):
    assert data_diagnostics(toy).lags == tuple(range(1, toy["sales_raw"].shape[1] // 2 + 1))
    long = toy_corpus(n_worlds=1, n_time=200, inactive_last_channel=False)
    assert data_diagnostics(long, views=("levels",)).lags == tuple(range(1, DEFAULT_MAX_LAG + 1))


def test_acf_matches_a_literal_oracle():
    series = np.array([0.0, 1.0, 4.0, 9.0, 16.0])
    corpus = toy_corpus(n_worlds=1, n_time=series.size, inactive_last_channel=False)
    corpus["spend_raw"][0, :, 0] = series
    report = data_diagnostics(corpus, views=("levels",), lags=(1,))
    assert report["levels"].temporal.acf[0, 0, 0] == pytest.approx(32.0 / 87.0)


def test_contiguous_axis_sees_lag_two_and_annual_structure():
    rng = np.random.default_rng(19)
    noise = rng.normal(size=406)
    ma2 = noise[2:] + noise[:-2]  # ACF(1) = 0, ACF(2) = 0.5 in the population
    corpus = toy_corpus(n_worlds=1, n_time=ma2.size, inactive_last_channel=False)
    corpus["spend_raw"][0, :, 0] = ma2
    temporal = data_diagnostics(corpus, views=("levels",))["levels"].temporal
    assert temporal.acf[0, 0, temporal.lags.index(2)] > 0.35
    assert abs(temporal.acf[0, 0, temporal.lags.index(1)]) < 0.15

    block = rng.normal(size=52)
    annual = np.tile(block, 4)
    corpus = toy_corpus(n_worlds=1, n_time=annual.size, inactive_last_channel=False)
    corpus["spend_raw"][0, :, 0] = annual
    temporal = data_diagnostics(corpus, views=("levels",))["levels"].temporal
    assert temporal.acf[0, 0, temporal.lags.index(52)] > 0.6
    assert 52 in temporal.lags  # a sparse (1, 4, 13, 26) axis would miss it


def test_lag_xi_finds_nonlinear_serial_structure():
    x = np.empty(300)
    x[0] = 0.4
    for t in range(1, x.size):
        x[t] = 4.0 * x[t - 1] * (1.0 - x[t - 1])  # logistic map: ACF ~ 0, xi ~ 1
    corpus = toy_corpus(n_worlds=1, n_time=x.size, inactive_last_channel=False)
    corpus["spend_raw"][0, :, 0] = x
    temporal = data_diagnostics(corpus, views=("levels",), lags=(1,))["levels"].temporal
    assert abs(temporal.acf[0, 0, 0]) < 0.2
    assert temporal.lag_xi[0, 0, 0] > 0.8


def test_lag_validity_follows_the_three_pair_report_policy():
    corpus = toy_corpus(n_worlds=1, n_time=4, inactive_last_channel=False)
    report = data_diagnostics(corpus, views=("levels", "differences"), lags=(1, 2))
    levels = report["levels"].temporal
    assert levels.acf_valid[0, 0, 0]  # T=4, lag 1 -> 3 pairs
    assert not levels.acf_valid[0, 0, 1]  # lag 2 -> 2 pairs
    differences = report["differences"].temporal
    assert not differences.acf_valid[0, 0, :].any()  # T_view = 3


# ---------------------------------------------------------------------------
# 14-17. VIF
# ---------------------------------------------------------------------------


def _vif_of(rows: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, int, float]:
    return dg._vif_world(np.asarray(rows, dtype=np.float64))


def test_vif_matches_a_hand_computed_non_orthogonal_oracle():
    target = np.array([-2.0, -1.0, 0.0, 3.0])
    nuisance = np.array([-3.0, -1.0, 1.0, 3.0])
    vif, valid, constant, rank, condition = _vif_of([target, nuisance])
    # r^2 = 32/35 by hand, so VIF = 1 / (1 - 32/35) = 35/3
    assert vif[0] == pytest.approx(35.0 / 3.0)
    assert valid.all() and not constant.any() and rank == 2 and np.isfinite(condition)


def test_orthogonal_predictors_have_vif_one():
    vif, valid, _, rank, condition = _vif_of([[1.0, -1.0, 1.0, -1.0], [1.0, 1.0, -1.0, -1.0]])
    assert vif == pytest.approx([1.0, 1.0])
    assert valid.all() and rank == 2 and condition == pytest.approx(1.0)


def test_exact_collinearity_is_infinite_and_partial_overlap_is_finite():
    a = np.array([1.0, 2.0, 3.0, 4.0, 6.0])
    b = np.array([0.0, 1.0, 0.0, 2.0, 1.0])
    exact = _vif_of([a + b, a, b])[0]
    assert np.isposinf(exact).all()

    rng = np.random.default_rng(21)
    noisy = np.stack([a + b + 0.3 * rng.normal(size=a.size), a, b])
    vif, valid, _, _, _ = _vif_of(noisy)
    assert np.isfinite(vif).all() and valid.all()


def test_vif_is_not_inferred_from_rank_equality():
    """A numerically singular nuisance set still admits a finite target VIF."""
    n = 8
    basis = np.eye(n)
    duplicated = np.stack([basis[0], basis[0], basis[1]])  # rank 2, three columns
    target = 0.6 * basis[0] + 0.8 * basis[2]
    block = np.vstack([target[None, :], duplicated])
    vif, valid, _, rank, condition = _vif_of(block)

    design = np.stack([duplicated[i] - duplicated[i].mean() for i in range(3)], axis=1)
    centred_target = target - target.mean()
    coefficients, *_ = np.linalg.lstsq(design, centred_target, rcond=None)
    residual = centred_target - design @ coefficients
    oracle = float((centred_target @ centred_target) / (residual @ residual))

    assert np.isfinite(vif[0])
    assert vif[0] == pytest.approx(oracle, rel=1e-8)
    assert valid[0]
    assert rank < block.shape[0]  # the design is rank deficient ...
    assert np.isposinf(condition)  # ... and its condition number is infinite
    assert np.isfinite(vif[0])  # ... yet this target's VIF is not


def test_vif_is_stable_under_rotation_and_permutation_of_the_nuisance_span():
    rng = np.random.default_rng(23)
    n = 24
    nuisance = rng.normal(size=(3, n))
    nuisance[2] = nuisance[0] + nuisance[1]  # singular span
    target = rng.normal(size=n)
    baseline = _vif_of(np.vstack([target[None, :], nuisance]))[0][0]

    rotation = np.linalg.qr(rng.normal(size=(3, 3)))[0]
    rotated = rotation @ nuisance
    assert _vif_of(np.vstack([target[None, :], rotated]))[0][0] == pytest.approx(baseline, rel=1e-8)
    permuted = nuisance[[2, 0, 1]]
    assert _vif_of(np.vstack([target[None, :], permuted]))[0][0] == pytest.approx(
        baseline, rel=1e-8
    )


def test_vif_degenerate_designs():
    empty = dg._vif_world(np.zeros((0, 10)))
    assert empty[0].size == 0 and empty[3] == 0 and math.isnan(empty[4])

    single = _vif_of([[1.0, 2.0, 3.0, 5.0]])
    assert single[0] == pytest.approx([1.0]) and single[3] == 1
    assert single[4] == pytest.approx(1.0)

    with_constant = _vif_of([[1.0, 1.0, 1.0, 1.0], [1.0, 2.0, 3.0, 5.0]])
    assert with_constant[2][0]  # flagged constant
    assert not with_constant[1][0]  # and therefore invalid, not zero
    assert np.isposinf(with_constant[4])  # the full design is degenerate
    assert with_constant[0][1] == pytest.approx(1.0)  # the other target is unaffected


def test_vif_is_invariant_to_scale_and_offset():
    rng = np.random.default_rng(29)
    rows = rng.normal(size=(3, 30))
    baseline = _vif_of(rows)[0]
    shifted = rows * np.array([[1e6], [1e-6], [3.0]]) + np.array([[500.0], [-2.0], [0.0]])
    assert _vif_of(shifted)[0] == pytest.approx(baseline, rel=1e-6)


def test_vif_scopes_hold_exactly_the_right_predictors(report):
    observed = report["levels"].vif["observed"]
    oracle = report["levels"].vif["oracle"]
    assert observed.keys == ("C1", "C2", "C3", "Z1", "Z2")
    assert oracle.keys == ("C1", "C2", "C3", "Z1", "Z2", "D1", "D2")
    for scope in (observed, oracle):
        assert "B" not in scope.keys and "Y" not in scope.keys
        assert not any(key.endswith("_direct_y") for key in scope.keys)
    # C3 is inactive in every world: never a predictor, never counted valid
    assert not observed.eligible[:, observed.keys.index("C3")].any()
    assert not observed.valid[:, observed.keys.index("C3")].any()


def test_each_vif_scope_is_verified_independently(toy, report):
    """No cross-scope numerical monotonicity is asserted: truncation is not nested."""
    levels = report["levels"]
    for scope_name, scope in levels.vif.items():
        for world in range(report.n_worlds):
            active = scope.eligible[world]
            rows = np.array(
                [toy_key_series(toy, key)[report.world_ids[world]] for key in scope.keys]
            )
            expected = dg._vif_world(rows[active])[0]
            assert np.allclose(scope.vif[world, active], expected, equal_nan=True), (
                f"{scope_name} world {world}"
            )


def toy_key_series(corpus, key: str) -> np.ndarray:
    source = {
        "C": ("spend_raw", 0),
        "Z": ("controls", 0),
        "D": ("demand", 0),
    }[key[0]]
    return corpus[source[0]][:, :, int(key[1:]) - 1]


# ---------------------------------------------------------------------------
# 18-20. Contributions
# ---------------------------------------------------------------------------


def test_all_six_contribution_fields_from_unique_atomic_paths(toy):
    budget = data_diagnostics(toy).contributions
    component = toy["contributions_raw"][:, :, 0]
    sales_total = toy["sales_raw"].sum(axis=1)
    n_time = toy["sales_raw"].shape[1]

    assert budget.values("C1_direct_y", measure="total") == pytest.approx(component.sum(axis=1))
    assert budget.values("C1_direct_y", measure="mean_per_period") == pytest.approx(
        component.sum(axis=1) / n_time
    )
    assert budget.values("C1_direct_y", measure="share") == pytest.approx(
        component.sum(axis=1) / sales_total
    )
    assert budget.values("C1_direct_y", measure="total", mode="gross") == pytest.approx(
        np.abs(component).sum(axis=1)
    )
    assert budget.values("C1_direct_y", measure="mean_per_period", mode="gross") == pytest.approx(
        np.abs(component).sum(axis=1) / n_time
    )
    assert budget.values("C1_direct_y", measure="share", mode="gross") == pytest.approx(
        np.abs(component).sum(axis=1) / np.abs(sales_total)
    )


def test_top_cut_closes_against_sales_and_media_rolls_up(toy):
    budget = data_diagnostics(toy).contributions
    closure = budget.closure("top")
    assert closure.complete and closure.children == (
        "media_total",
        "controls_baseline_alloc_total",
        "demand_baseline_alloc_total",
        "B_intrinsic",
        "Y_noise",
    )
    assert closure.max_abs_residual("net_total") == pytest.approx(0.0, abs=1e-9)
    assert closure.max_abs_residual("net_share") == pytest.approx(0.0, abs=1e-12)
    assert closure.residuals["gross_total"] is None  # sales has no gross decomposition

    media = budget.closure("media_children")
    assert media.max_abs_residual("net_total") == pytest.approx(0.0, abs=1e-9)
    assert media.max_abs_residual("gross_total") == pytest.approx(0.0, abs=1e-9)


def test_parent_gross_sums_atomic_children_and_keeps_cancellation():
    n_time = 4
    corpus = toy_corpus(n_worlds=1, n_time=n_time, inactive_last_channel=False)
    corpus["contributions_raw"][0] = 0.0
    corpus["contributions_raw"][0, :, 0] = np.array([10.0, 0.0, 0.0, 0.0])
    corpus["contributions_raw"][0, :, 1] = np.array([-10.0, 0.0, 0.0, 0.0])
    corpus["indirect_effects_by_source"][0] = 0.0
    corpus["indirect_effects"] = corpus["indirect_effects_by_source"].sum(axis=2)
    corpus = make_corpus(
        spend=corpus["spend_raw"],
        controls=corpus["controls"],
        demand=corpus["demand"],
        contributions=corpus["contributions_raw"],
        control_contribution=corpus["control_contribution"],
        confounder_contribution=corpus["confounder_contribution"],
        indirect_by_source=corpus["indirect_effects_by_source"],
        baseline_intrinsic=corpus["baseline_intrinsic"],
        sales_noise=corpus["sales_noise"],
        channel_mask=corpus["treatment_active_mask"].astype(bool),
        control_mask=corpus["covariate_active_mask"].astype(bool),
        latent_mask=corpus["latent_active_mask"].astype(bool),
    )
    budget = data_diagnostics(corpus).contributions
    assert budget.values("channels_direct_y_total", measure="total")[0] == pytest.approx(0.0)
    # abs(sum) would report 0 activity; the sum of atomic gross keeps both legs
    assert budget.values("channels_direct_y_total", measure="total", mode="gross")[
        0
    ] == pytest.approx(20.0)


def test_complete_gross_over_net_sales_is_at_least_one(toy):
    budget = data_diagnostics(toy).contributions
    top = [d.key for d in budget.descriptors if d.sibling_set == "top"]
    total = sum(budget.values(key, measure="share", mode="gross") for key in top)
    assert (total >= 1.0 - 1e-9).all()

    # equality when nothing cancels: one positive atomic row carries all of sales
    n_time = 5
    zeros = np.zeros((1, n_time, 2))
    corpus = make_corpus(
        spend=np.zeros((1, n_time, 1)),
        controls=np.zeros((1, n_time, 1)),
        demand=np.zeros((1, n_time, 1)),
        contributions=np.zeros((1, n_time, 1)),
        control_contribution=np.zeros((1, n_time, 1)),
        confounder_contribution=np.zeros((1, n_time, 1)),
        indirect_by_source=np.zeros((1, n_time, 3)),
        baseline_intrinsic=np.full((1, n_time), 2.0),
        sales_noise=np.zeros((1, n_time)),
        channel_mask=np.ones((1, 1), dtype=bool),
        control_mask=np.ones((1, 1), dtype=bool),
        latent_mask=np.ones((1, 1), dtype=bool),
    )
    del zeros
    budget = data_diagnostics(corpus).contributions
    top = [d.key for d in budget.descriptors if d.sibling_set == "top"]
    assert sum(budget.values(key, measure="share", mode="gross")[0] for key in top) == (
        pytest.approx(1.0)
    )


def test_macro_micro_and_activity_populations(toy):
    budget = data_diagnostics(toy).contributions
    key = "Z2_baseline_alloc"
    active = budget.active[:, budget.keys.index(key)]
    assert active.tolist() == [True, True, False]

    macro, ledger = budget.stats(key, measure="share", weighting="macro")
    per_world = budget.values(key, measure="share")
    assert macro["mean"] == pytest.approx(per_world.mean())
    assert ledger.eligible == 3 and ledger.valid == 3

    conditional, cond_ledger = budget.stats(
        key, measure="share", weighting="macro", population="conditional_on_active"
    )
    assert conditional["mean"] == pytest.approx(per_world[active].mean())
    assert cond_ledger.eligible == 2

    micro, _ = budget.stats(key, measure="share", weighting="micro")
    totals = budget.values(key, measure="total")
    assert micro["value"] == pytest.approx(totals.sum() / budget.sales_total.sum())
    assert micro["value"] != pytest.approx(per_world.mean())  # pooled != mean of ratios


def test_projection_is_partial_and_suppresses_closure(toy):
    budget = data_diagnostics(toy).contributions
    projected = budget.select(("C2_direct_y", "C1_direct_y"))
    assert projected.keys == ("C2_direct_y", "C1_direct_y")  # caller order
    assert projected.is_partial_projection
    assert "media_total" in projected.omitted_keys
    assert not projected.closure("channel_direct_children").complete
    assert np.array_equal(
        projected.values("C1_direct_y", measure="total"),
        budget.values("C1_direct_y", measure="total"),
    )
    with pytest.raises(KeyError, match="unknown contribution keys"):
        budget.select(("c1_direct_y",))
    with pytest.raises(ValueError, match="is empty"):
        budget.select(())
    with pytest.raises(TypeError, match="bare string"):
        budget.select("C1_direct_y")
    with pytest.raises(ValueError, match="alternative reading"):
        budget.select(("C1_direct_y",), basis="observed_path_media")


def test_reprojection_keeps_saying_it_is_partial():
    """Projecting a projection omits nothing NEW — the old omissions remain."""
    corpus = toy_corpus(inactive_last_channel=False)
    budget = data_diagnostics(corpus).contributions
    kept = ("channels_direct_y_total", "C1_direct_y", "C2_direct_y")
    once = budget.select(kept)
    twice = once.select(kept)
    assert twice.is_partial_projection
    assert set(once.omitted_keys) <= set(twice.omitted_keys)

    dropped = twice.closure("channel_direct_children")
    assert not dropped.complete
    # the dropped TRAILING channel is still named, and the residual is real
    assert "C3_direct_y" in dropped.omitted_keys
    residual = dropped.residuals["net_total"]
    assert residual is not None
    assert np.allclose(residual, budget.values("C3_direct_y", measure="total"))
    assert dropped.max_abs_residual("net_total") > 0.0
    assert "partial projection" in twice.table(sibling_set="channel_direct_children")


def test_zero_sales_invalidates_shares_only():
    n_time = 4
    corpus = make_corpus(
        spend=np.ones((1, n_time, 1)),
        controls=np.ones((1, n_time, 1)),
        demand=np.ones((1, n_time, 1)),
        contributions=np.tile(np.array([1.0, -1.0, 1.0, -1.0])[None, :, None], (1, 1, 1)),
        control_contribution=np.zeros((1, n_time, 1)),
        confounder_contribution=np.zeros((1, n_time, 1)),
        indirect_by_source=np.zeros((1, n_time, 3)),
        baseline_intrinsic=np.zeros((1, n_time)),
        sales_noise=np.zeros((1, n_time)),
        channel_mask=np.ones((1, 1), dtype=bool),
        control_mask=np.ones((1, 1), dtype=bool),
        latent_mask=np.ones((1, 1), dtype=bool),
    )
    assert corpus["sales_raw"].sum() == 0.0
    budget = data_diagnostics(corpus).contributions
    assert budget.values("C1_direct_y", measure="total")[0] == pytest.approx(0.0)
    assert budget.values("C1_direct_y", measure="total", mode="gross")[0] == pytest.approx(4.0)
    assert np.isnan(budget.values("C1_direct_y", measure="share")[0])
    stats, ledger = budget.stats("C1_direct_y", measure="share")
    assert stats["mean"] is None and ledger.valid == 0


def test_contribution_table_and_frame(toy):
    budget = data_diagnostics(toy).contributions
    text = budget.table(sibling_set="top")
    assert "contributions to sales" in text and "media_total" in text
    frame = budget.to_frame()
    assert set(CONTRIBUTION_FIELDS).issubset(frame.columns)
    assert len(frame) == 3 * len(budget.descriptors)


# ---------------------------------------------------------------------------
# 21. Strict JSON and no inference
# ---------------------------------------------------------------------------


def test_summary_is_strictly_json_serializable(report):
    payload = json.dumps(report.summary(), allow_nan=False)
    assert "NaN" not in payload and "Infinity" not in payload
    text = payload.lower()
    for banned in ("p_value", "pvalue", "significan", "confidence", "causal", "forecast"):
        assert banned not in text


def test_infinite_values_are_counted_not_hidden():
    series = np.array([0.0] * 5 + [1.0] + [0.0] * 4)
    corpus = toy_corpus(n_worlds=1, n_time=series.size, inactive_last_channel=False)
    corpus["spend_raw"][0, :, 0] = series
    report = data_diagnostics(corpus, views=("levels",))
    stats, ledger = report["levels"].series.stats("C1", "spike")
    assert stats["mean"] is None  # no FINITE value to average
    assert ledger.positive_infinite == 1 and ledger.valid == 1 and ledger.invalid == 0
    assert json.dumps(report.summary(), allow_nan=False)


# ---------------------------------------------------------------------------
# 26. Purity and determinism
# ---------------------------------------------------------------------------


def test_report_is_pure(toy, monkeypatch, tmp_path):
    snapshot = {key: np.array(value, copy=True) for key, value in toy.items()}
    state_before = np.random.get_state()

    def no_write(*args, **kwargs):  # pragma: no cover - must never run
        raise AssertionError("diagnostics must not write anything")

    monkeypatch.setattr(np, "save", no_write)
    monkeypatch.setattr(np, "savez", no_write)
    monkeypatch.setattr(np, "savez_compressed", no_write)
    monkeypatch.setattr(pg, "sample_prior_predictive", no_write)
    monkeypatch.chdir(tmp_path)

    first = data_diagnostics(toy, scopes=("nodes", "decomposition"))
    second = data_diagnostics(toy, scopes=("nodes", "decomposition"))

    for key, value in snapshot.items():
        assert np.array_equal(np.asarray(toy[key]), value), key
    assert np.array_equal(np.random.get_state()[1], state_before[1])
    assert list(tmp_path.iterdir()) == []
    assert json.dumps(first.summary(), sort_keys=True) == json.dumps(
        second.summary(), sort_keys=True
    )


# ---------------------------------------------------------------------------
# Real generated corpora and SCM worlds
# ---------------------------------------------------------------------------


def test_generated_corpus_report_is_consistent(generated_corpus):
    report = data_diagnostics(generated_corpus, scopes=("nodes", "decomposition"))
    assert report.n_worlds == generated_corpus["sales_raw"].shape[0]
    closure = report.contributions.closure("top")
    scale = float(np.abs(report.contributions.sales_total).max())
    assert closure.max_abs_residual("net_total") == pytest.approx(0.0, abs=1e-4 * scale)
    assert closure.complete

    # the additive share budget agrees with the outcome-space companion
    assert np.allclose(report.outcomes.additive_share_total(), 1.0, atol=1e-5)

    # every reported quantile is finite where the ledger says it is
    for view in ("levels", "differences"):
        stats, ledger = report[view].series.stats("Y", "std")
        assert ledger.valid == report.n_worlds
        assert stats["mean"] is not None


def test_scm_worlds_add_the_counterfactual_scope():
    cfg = pg.make_scm_prior(n_treatments=2, n_covariates=1, n_latent=1, n_time_steps=32, seed=4242)
    worlds = [pg.sample_scm(cfg, seed=seed) for seed in (1, 2)]
    report = data_diagnostics(worlds, scopes=("nodes", "decomposition", "counterfactual"))
    assert report.source_kind == "worlds"
    assert "C1_base" in report.keys and "C1_observed_y" in report.keys
    assert report.descriptor("C1_base").scope == "counterfactual"
    assert "observed_path_media" in report.contribution_bases

    observed = report.contribution_bases["observed_path_media"]
    closure = observed.closure("observed_path_children")
    assert closure.parent == "media_total_observed_path"
    assert closure.max_abs_residual("net_total") == pytest.approx(0.0, abs=1e-6)

    # the two bases are alternative readings, never additive alongside
    assert observed.basis != report.contributions.basis
    with pytest.raises(ValueError, match="alternative reading"):
        report.contributions.select(("C1_direct_y",), basis="observed_path_media")


def test_scm_optional_paths_absent_stay_known_and_ineligible():
    cfg = pg.make_scm_prior(n_treatments=2, n_covariates=1, n_latent=1, n_time_steps=32, seed=99)
    world = pg.sample_scm(cfg, seed=5)
    assert "channels_unshocked" not in world.data  # no shock schedule in this config
    report = data_diagnostics([world], scopes=("nodes", "counterfactual"))
    assert "C1_unshocked" in report.keys
    assert not report.descriptor("C1_unshocked").available
    index = report.keys.index("C1_unshocked")
    assert not report["levels"].series.eligible[:, index].any()
    assert report["levels"].series.values("C1_unshocked").size == 0
