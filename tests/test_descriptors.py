"""Per-world descriptors: definitions, boundaries, identities and invariances.

Expected values are independent oracles (literals or numpy written from the
definition), never the production helper under test.
"""

from __future__ import annotations

import codecs
import io
import re
from copy import deepcopy

import numpy as np
import pandas as pd
import pytest

import pymc_generator as pg
from pymc_generator.descriptors import (
    INELIGIBLE,
    TRUTH_FEATURES,
    ObservedWorlds,
    WorldDescriptors,
    world_descriptors,
)


def _observed(treatment, covariates=None, outcome=None, **masks):
    return ObservedWorlds(
        treatment_raw=np.asarray(treatment, dtype=np.float64),
        covariates=None if covariates is None else np.asarray(covariates, dtype=np.float64),
        outcome_raw=None if outcome is None else np.asarray(outcome, dtype=np.float64),
        **masks,
    )


def _describe(source, **kwargs) -> WorldDescriptors:
    kwargs.setdefault("source_id", "S")
    return world_descriptors(source, **kwargs)


def _value(desc: WorldDescriptors, name: str, row: int = 0) -> tuple[float, str]:
    return float(desc.column(name)[row]), str(desc.column_status(name)[row])


def _r(a: np.ndarray, b: np.ndarray) -> float:
    return float(abs(np.corrcoef(a, b)[0, 1]))


def _acf(x: np.ndarray, lag: int) -> float:
    d = x - x.mean()
    return float((d[:-lag] * d[lag:]).sum() / (d**2).sum())


@pytest.fixture(scope="module")
def generated():
    cfg = pg.make_scm_prior(
        n_treatments=4,
        n_covariates=3,
        n_latent=1,
        n_time_steps=40,
        n_cells=3,
        draws_per_cell=2,
        seed=20260929,
        n_treatments_active_range=(2, 4),
        n_covariates_active_range=(1, 3),
    )
    return pg.sample_prior_predictive(cfg)


@pytest.fixture(scope="module")
def scm_worlds():
    cfg = pg.make_scm_prior(n_treatments=3, n_covariates=2, n_latent=1, n_time_steps=32, seed=4242)
    return [pg.sample_scm(cfg, seed=s) for s in (1, 2)]


# ---------------------------------------------------------------------------
# Definitions
# ---------------------------------------------------------------------------


def test_ratio_boundaries_per_series():
    t = 12
    series = np.zeros((3, t, 1))
    series[0, :, 0] = 0.1  # constant, not exactly representable
    series[2, :, 0] = np.tile([1.0, -1.0], t // 2)  # varying, exactly zero mean
    desc = _describe(_observed(series), reducers=("median",))
    ltv = [_value(desc, "treatment_level_to_variation_median", w) for w in range(3)]
    cv = [_value(desc, "treatment_cv_median", w) for w in range(3)]
    zero = [_value(desc, "treatment_zero_fraction_median", w)[0] for w in range(3)]
    # |mean|/std: constant -> +inf, all-zero -> undefined (0/0), zero mean -> 0.
    assert ltv[0] == (np.inf, "valid") and ltv[2] == (0.0, "valid") and ltv[1][1] == "undefined"
    # std/|mean|: constant and all-zero -> 0; the zero-mean series has no CV.
    assert cv[0] == (0.0, "valid") and cv[1] == (0.0, "valid") and cv[2][1] == "undefined"
    assert zero == [0.0, 1.0, 0.0]


def test_ratio_values_match_the_definition_over_active_nodes_only():
    rng = np.random.default_rng(3)
    treatment = rng.normal(5.0, 1.5, size=(1, 30, 3))
    treatment[0, :, 2] = 0.0  # padded
    desc = _describe(_observed(treatment, treatment_active_mask=np.array([[1, 1, 0]])))
    active = treatment[0, :, :2]
    ratios = np.abs(active.mean(axis=0)) / active.std(axis=0)
    assert _value(desc, "treatment_level_to_variation_median")[0] == pytest.approx(
        float(np.median(ratios))
    )
    assert _value(desc, "treatment_cv_max")[0] == pytest.approx(float((1 / ratios).max()))
    assert _value(desc, "treatment_zero_fraction_max") == (0.0, "valid")


def test_outcome_statistics_describe_the_outcome_series():
    rng = np.random.default_rng(14)
    treatment = rng.normal(3.0, 1.0, size=(1, 30, 2))
    covariates = rng.normal(-2.0, 0.5, size=(1, 30, 2))
    outcome = rng.normal(10.0, 2.0, size=(1, 30))
    desc = _describe(_observed(treatment, covariates, outcome))
    y = outcome[0]
    assert _value(desc, "outcome_level_to_variation")[0] == pytest.approx(abs(y.mean()) / y.std())
    assert _value(desc, "outcome_cv")[0] == pytest.approx(y.std() / abs(y.mean()))


def test_every_pair_role_runs_over_its_own_node_sets():
    rng = np.random.default_rng(13)
    treatment = rng.normal(size=(1, 40, 2)).cumsum(axis=1)
    covariates = rng.normal(size=(1, 40, 3)).cumsum(axis=1)
    outcome = rng.normal(size=(1, 40)).cumsum(axis=1)
    reducers = ("min", "median", "max")
    desc = _describe(_observed(treatment, covariates, outcome), reducers=reducers)
    t, c, y = treatment[0].T, covariates[0].T, outcome[0]
    expected = {
        "treatment_pair": [_r(t[0], t[1])],
        "covariate_pair": [_r(c[0], c[1]), _r(c[0], c[2]), _r(c[1], c[2])],
        "treatment_covariate": [_r(t[i], c[j]) for i in range(2) for j in range(3)],
        "treatment_outcome": [_r(t[0], y), _r(t[1], y)],
    }
    for role, values in expected.items():
        assert _value(desc, f"{role}_abs_pearson_min")[0] == pytest.approx(min(values)), role
        assert _value(desc, f"{role}_abs_pearson_max")[0] == pytest.approx(max(values)), role
        # 3 and 6 pairs: the median is not the mean (np.median, not np.mean).
        assert _value(desc, f"{role}_abs_pearson_median")[0] == pytest.approx(
            float(np.median(values))
        ), role
    d = np.diff(t, axis=1)
    assert _value(desc, "treatment_pair_diff_abs_pearson_max")[0] == pytest.approx(_r(d[0], d[1]))


def test_acf_features_are_signed_per_lag_in_levels_and_differences():
    rng = np.random.default_rng(15)
    e = rng.normal(size=60)
    x = np.zeros(60)
    for i in range(1, 60):
        x[i] = -0.7 * x[i - 1] + e[i]
    x = x + np.linspace(0.0, 5.0, 60)
    desc = _describe(_observed(x[None, :, None]), lags=(1, 2), reducers=("median",))
    for lag in (1, 2):
        assert _value(desc, f"treatment_acf_lag_{lag}_median")[0] == pytest.approx(_acf(x, lag))
        assert _value(desc, f"treatment_diff_acf_lag_{lag}_median")[0] == pytest.approx(
            _acf(np.diff(x), lag)
        )


def test_roughness_and_spike_follow_their_definitions():
    rng = np.random.default_rng(16)
    x = rng.normal(size=50).cumsum()
    x[30:] += 8.0
    desc = _describe(_observed(x[None, :, None]), reducers=("median",))
    steps = np.diff(x)
    q25, q75 = np.quantile(steps, [0.25, 0.75])
    assert _value(desc, "treatment_roughness_median")[0] == pytest.approx(
        steps.std() / (np.sqrt(2.0) * x.std())
    )
    assert _value(desc, "treatment_spike_median")[0] == pytest.approx(
        np.abs(steps - np.median(steps)).max() / (q75 - q25)
    )


# ---------------------------------------------------------------------------
# Applicability: ineligible versus undefined
# ---------------------------------------------------------------------------


def test_missing_roles_are_ineligible_not_undefined():
    rng = np.random.default_rng(4)
    desc = _describe(_observed(rng.normal(size=(1, 20, 1))))  # no covariates, no outcome
    assert desc.column_status("covariate_cv_median")[0] == "ineligible"
    assert desc.column_status("covariate_constant_fraction")[0] == "ineligible"
    assert desc.column_status("outcome_cv")[0] == "ineligible"
    assert desc.column_status("treatment_pair_abs_pearson_max")[0] == "ineligible"
    assert desc.column_status("treatment_outcome_abs_pearson_max")[0] == "ineligible"
    assert desc.column_status("treatment_cv_median")[0] == "valid"


def test_padded_nodes_never_count_as_active():
    treatment = np.zeros((1, 20, 3))
    treatment[0, :, 0] = 5.0
    treatment[0, :, 1] = np.random.default_rng(17).normal(size=20)
    desc = _describe(
        _observed(
            treatment,
            np.zeros((1, 20, 2)),
            treatment_active_mask=np.array([[1, 1, 0]]),
            covariate_active_mask=np.array([[0, 0]]),
        )
    )
    assert _value(desc, "treatment_constant_fraction") == (0.5, "valid")
    assert desc.column_status("treatment_pair_abs_pearson_max")[0] == "undefined"
    assert desc.column_status("covariate_cv_median")[0] == "ineligible"
    assert desc.column_status("treatment_covariate_abs_pearson_max")[0] == "ineligible"
    assert desc.metadata["n_treatments_active"].tolist() == [2]


def test_short_horizons_leave_arithmetic_artifacts_undefined():
    rng = np.random.default_rng(6)
    desc = _describe(_observed(rng.normal(size=(1, 6, 2))), lags=(1, 4))
    assert desc.column_status("treatment_acf_lag_1_median")[0] == "valid"
    # 6 - 4 = 2 lag pairs < the 3-pair reporting floor used by data_diagnostics.
    assert desc.column_status("treatment_acf_lag_4_median")[0] == "undefined"
    three = _describe(_observed(rng.normal(size=(1, 3, 2))))
    # Level pairs have 3 points; differenced pairs, roughness and spike only 2 steps
    # (spike would be 1 and roughness 0 by arithmetic alone).
    assert three.column_status("treatment_pair_abs_pearson_max")[0] == "valid"
    for name in (
        "treatment_pair_diff_abs_pearson_max",
        "treatment_roughness_max",
        "treatment_spike_max",
    ):
        assert three.column_status(name)[0] == "undefined", name
    with pytest.raises(ValueError, match="no features"):
        _describe(_observed(rng.normal(size=(1, 1, 1))), statistics=("acf",), lags=None)


# ---------------------------------------------------------------------------
# Rounding: degenerate series are recognized, never mistaken for structure
# ---------------------------------------------------------------------------


def test_rounding_residue_never_passes_for_structure():
    rng = np.random.default_rng(13)
    t = 52  # where a std-based floor misses ~25% of exact constants
    treatment = np.zeros((1, t, 3))
    treatment[0, :, 0] = 0.1
    treatment[0, :, 1] = 0.7
    treatment[0, :, 2] = rng.normal(5.0, 1.0, t)
    desc = _describe(_observed(treatment), reducers=("min", "max"))
    assert _value(desc, "treatment_level_to_variation_max") == (np.inf, "valid")
    assert _value(desc, "treatment_constant_fraction")[0] == pytest.approx(2 / 3)
    # Every pair has a constant member, so there is no correlation to report.
    assert desc.column_status("treatment_pair_abs_pearson_max")[0] == "undefined"
    # Only the varying series has an autocorrelation; constants have no texture.
    assert _value(desc, "treatment_acf_lag_1_min")[0] == pytest.approx(_acf(treatment[0, :, 2], 1))
    assert _value(desc, "treatment_roughness_min") == (0.0, "valid")


def _one_treatment_corpus(x: np.ndarray, y: np.ndarray) -> dict[str, np.ndarray]:
    return {
        "treatment_raw": x[None, :, None],
        "covariates": np.zeros((1, x.size, 0), dtype=x.dtype),
        "outcome_raw": y[None],
        "treatment_active_mask": np.ones((1, 1)),
        "covariate_active_mask": np.ones((1, 0)),
    }


@pytest.mark.parametrize("kind", ["observed", "corpus"])
@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_constant_floor_is_a_few_ulps_of_the_input_dtype(dtype, kind):
    eps = float(np.finfo(dtype).eps)
    k = np.arange(52) % 2
    y = np.random.default_rng(1).normal(10.0, 1.0, 52).astype(dtype)
    for relative, constant in ((eps, True), (64 * eps, False)):
        x = (0.7 * (1.0 + relative * k)).astype(dtype)
        if kind == "observed":
            source = ObservedWorlds(treatment_raw=x[None, :, None], outcome_raw=y[None])
        else:
            source = _one_treatment_corpus(x, y)
        desc = _describe(source, reducers=("max",))
        ltv = desc.column("treatment_level_to_variation_max")[0]
        assert bool(ltv == np.inf) == constant, (relative / eps, constant)


def test_a_small_but_real_mean_is_not_zero():
    z = np.random.default_rng(21).normal(10.0, 1.0, 52)
    z = (z - z.mean()) / z.std()
    shifted = z + 1e-13 * np.abs(z).max()  # far above T * eps, far below an absolute 1e-12
    desc = _describe(
        ObservedWorlds(treatment_raw=np.ones((1, 52, 1)), covariates=shifted[None, :, None]),
        reducers=("max",),
    )
    assert desc.column_status("covariate_cv_max")[0] == "valid"


def test_a_mixed_dtype_scm_sequence_describes_identically_in_any_chunking(scm_worlds):
    first, second = deepcopy(scm_worlds[0]), deepcopy(scm_worlds[1])
    n_time = first.data["outcome"].shape[0]
    # A 1e-9 wiggle is real at float64 resolution and invisible at float32's.
    first.data["treatments"] = first.data["treatments"].copy()
    first.data["treatments"][:, 0] = 1.0 + 1e-9 * np.random.default_rng(24).normal(size=n_time)
    for key in ("treatments", "covariates", "outcome"):
        second.data[key] = second.data[key].astype(np.float32)
    together = _describe([first, second], chunk_size=2)
    apart = _describe([first, second], chunk_size=1)
    np.testing.assert_array_equal(together.status, apart.status)
    np.testing.assert_allclose(together.values, apart.values, rtol=1e-12, atol=1e-15)
    assert _value(together, "treatment_constant_fraction", 0)[0] == 0.0


def test_each_series_is_judged_at_its_own_input_resolution():
    rng = np.random.default_rng(23)
    # A float64 treatment varying by ~2e-8 relative: real variation at float64
    # resolution, but within one float32 ULP (~1.2e-7) once stored as float32.
    wiggle = 1.0 + 2e-8 * rng.normal(size=(1, 40, 1))
    # A float16 outcome must not loosen the treatment's floor to float16's.
    coarse_outcome = rng.normal(10.0, 1.0, size=(1, 40)).astype(np.float16)
    desc = _describe(ObservedWorlds(treatment_raw=wiggle, outcome_raw=coarse_outcome))
    assert _value(desc, "treatment_constant_fraction") == (0.0, "valid")
    stored32 = _describe(ObservedWorlds(treatment_raw=wiggle.astype(np.float32)))
    assert _value(stored32, "treatment_constant_fraction") == (1.0, "valid")
    # Long float16 horizons: the zero-mean floor must not grow into max|x| itself.
    y16 = rng.normal(100.0, 20.0, size=(1, 2000, 1)).astype(np.float16)
    long16 = _describe(ObservedWorlds(treatment_raw=y16), reducers=("max",))
    y = y16[0, :, 0].astype(np.float64)
    assert _value(long16, "treatment_level_to_variation_max")[0] == pytest.approx(
        abs(y.mean()) / y.std()
    )
    assert _value(long16, "treatment_cv_max")[0] == pytest.approx(y.std() / abs(y.mean()))
    # A sparse non-negative series is never zero-mean, even when |mean| is tiny.
    spike16 = np.zeros((1, 52, 1), dtype=np.float16)
    spike16[0, 7, 0] = 3.0
    sparse = _describe(ObservedWorlds(treatment_raw=spike16), reducers=("max",))
    s = spike16[0, :, 0].astype(np.float64)
    assert _value(sparse, "treatment_cv_max")[0] == pytest.approx(s.std() / s.mean())


def test_a_ramp_keeps_its_level_statistics_and_loses_its_differenced_ones():
    steps = np.arange(52, dtype=np.float64)
    ramps = np.stack([0.1 * steps, 0.3 * steps + 1.0], axis=1)[None]
    ramp = _describe(_observed(ramps), reducers=("max",))
    assert _value(ramp, "treatment_acf_lag_1_max")[0] == pytest.approx(_acf(0.1 * steps, 1))
    assert _value(ramp, "treatment_pair_abs_pearson_max") == (pytest.approx(1.0), "valid")
    assert _value(ramp, "treatment_spike_max") == (0.0, "valid")
    assert ramp.column_status("treatment_diff_acf_lag_1_max")[0] == "undefined"
    assert ramp.column_status("treatment_pair_diff_abs_pearson_max")[0] == "undefined"


@pytest.mark.parametrize("factor", [1e-15, 1.0, 1e9])
def test_degeneracy_floors_are_relative_to_each_series_magnitude(factor):
    t = 52
    k = np.arange(t, dtype=np.float64)
    z = np.random.default_rng(21).normal(10.0, 1.0, t)
    z = (z - z.mean()) / z.std()  # mean ~1e-15: zero within summation error
    ramp = (factor / 3.0) * k  # one rounding per point: steps agree to 2 eps max|x|
    desc = _describe(
        _observed(ramp[None, :, None], covariates=(z * factor)[None, :, None]), reducers=("max",)
    )
    assert desc.column_status("treatment_diff_acf_lag_1_max")[0] == "undefined"
    assert _value(desc, "treatment_spike_max") == (0.0, "valid")
    assert desc.column_status("covariate_cv_max")[0] == "undefined"
    assert _value(desc, "covariate_level_to_variation_max") == (0.0, "valid")


@pytest.mark.parametrize("magnitude", [1e-250, 1e250])
def test_extreme_magnitudes_describe_like_ordinary_ones(magnitude):
    rng = np.random.default_rng(22)
    treatment = rng.gamma(2.0, 1.0, size=(2, 40, 3))
    covariates = rng.normal(1.0, 1.0, size=(2, 40, 2))
    outcome = rng.normal(10.0, 2.0, size=(2, 40))
    base = _describe(_observed(treatment, covariates, outcome), lags=(1, 2))
    with np.errstate(all="raise"):  # squared sums must neither overflow nor underflow
        far = _describe(
            _observed(treatment * magnitude, covariates * magnitude, outcome * magnitude),
            lags=(1, 2),
        )
    np.testing.assert_array_equal(far.status, base.status)
    np.testing.assert_allclose(far.values, base.values, rtol=1e-9, atol=1e-12)


# ---------------------------------------------------------------------------
# Invariances and source equivalence
# ---------------------------------------------------------------------------


def test_positive_rescaling_leaves_every_observable_feature_unchanged():
    rng = np.random.default_rng(7)
    treatment = np.abs(rng.normal(3.0, 1.0, size=(4, 30, 3)))
    covariates = rng.normal(2.0, 1.0, size=(4, 30, 2))
    outcome = rng.normal(10.0, 2.0, size=(4, 30))
    base = _describe(_observed(treatment, covariates, outcome))
    scaled = _describe(_observed(treatment * 1e3, covariates * 7.0, outcome * 0.01))
    np.testing.assert_array_equal(base.status, scaled.status)
    np.testing.assert_allclose(scaled.values, base.values, rtol=1e-9, atol=1e-12)


def test_shifting_a_covariate_changes_its_level_to_variation():
    rng = np.random.default_rng(8)
    covariates = rng.normal(0.0, 1.0, size=(1, 30, 1))
    treatment = rng.normal(3.0, 1.0, size=(1, 30, 1))
    name = "covariate_level_to_variation_median"
    near_zero = _describe(_observed(treatment, covariates))
    shifted = _describe(_observed(treatment, covariates + 100.0))
    x = covariates[0, :, 0]
    assert _value(shifted, name)[0] == pytest.approx(abs(x.mean() + 100.0) / x.std())
    assert _value(shifted, name)[0] > 50 * _value(near_zero, name)[0]


def test_observed_adapter_and_corpus_agree(generated):
    corpus_desc = _describe(generated, lags=(1, 3))
    observed = ObservedWorlds(
        treatment_raw=generated["treatment_raw"],
        covariates=generated["covariates"],
        outcome_raw=generated["outcome_raw"],
        treatment_active_mask=generated["treatment_active_mask"],
        covariate_active_mask=generated["covariate_active_mask"],
    )
    obs_desc = _describe(observed, lags=(1, 3))
    np.testing.assert_array_equal(obs_desc.status, corpus_desc.status)
    np.testing.assert_array_equal(obs_desc.values, corpus_desc.values)
    np.testing.assert_array_equal(
        corpus_desc.metadata["n_covariates_active"], generated["covariate_active_mask"].sum(axis=1)
    )


def test_scm_worlds_describe_like_their_observed_series(scm_worlds):
    observed = ObservedWorlds(
        treatment_raw=np.stack([w.data["treatments"] for w in scm_worlds]),
        covariates=np.stack([w.data["covariates"] for w in scm_worlds]),
        outcome_raw=np.stack([w.data["outcome"] for w in scm_worlds]),
    )
    a = _describe(scm_worlds, lags=(1, 2))
    b = _describe(observed, lags=(1, 2))
    np.testing.assert_array_equal(a.status, b.status)
    np.testing.assert_array_equal(a.values, b.values)


@pytest.mark.parametrize("chunk_size", [1, 4])
def test_a_selection_describes_exactly_the_selected_worlds_in_any_chunking(generated, chunk_size):
    full = _describe(generated, include_truth=True, lags=(1, 3))
    part = _describe(
        generated, include_truth=True, lags=(1, 3), worlds=[4, 1], chunk_size=chunk_size
    )
    np.testing.assert_array_equal(part.status, full.status[[4, 1]])
    np.testing.assert_allclose(part.values, full.values[[4, 1]], rtol=1e-12, atol=1e-15)
    chunked = _describe(generated, include_truth=True, lags=(1, 3), chunk_size=chunk_size)
    np.testing.assert_array_equal(chunked.status, full.status)
    np.testing.assert_allclose(chunked.values, full.values, rtol=1e-12, atol=1e-15)


# ---------------------------------------------------------------------------
# Identity, groups and truth
# ---------------------------------------------------------------------------


def test_selection_keeps_original_world_and_cell_ids(generated):
    desc = _describe(generated, worlds=[4, 1])
    assert desc.world_ids.tolist() == [4, 1]
    assert desc.group_ids.tolist() == generated["cell_id"][[4, 1]].tolist()
    override = _describe(generated, worlds=[4, 1], world_ids=[40, 10], group_ids=[-1, 7])
    assert override.world_ids.tolist() == [40, 10]
    assert override.group_ids.tolist() == [-1, 7]
    with pytest.raises(ValueError, match="one entry per selected world"):
        _describe(generated, worlds=[4, 1], world_ids=[1, 2, 3])
    with pytest.raises(ValueError, match="int64"):
        _describe(generated, worlds=[4, 1], world_ids=np.array([2**63, 1], dtype=np.uint64))


def test_truth_shares_are_net_block_totals_over_the_outcome_total(generated):
    desc = _describe(generated, include_truth=True, statistics=("cv",))
    y = generated["outcome_raw"].astype(np.float64).sum(axis=1)

    def share(key):
        block = generated[key].astype(np.float64)
        return block.reshape(block.shape[0], -1).sum(axis=1) / y

    expected = {
        "truth_treatment_share": share("treatment_contribution_raw")
        + share("indirect_effects_by_source"),
        "truth_covariate_share": share("covariate_contribution"),
        "truth_latent_share": share("latent_unobserved_contribution"),
        "truth_baseline_share": share("baseline_intrinsic"),
        "truth_noise_share": share("outcome_noise"),
    }
    for name, values in expected.items():
        np.testing.assert_allclose(desc.column(name), values, rtol=1e-9, atol=1e-12, err_msg=name)
    assert not any(desc.definition(name).observable for name in TRUTH_FEATURES)
    assert desc.definition("treatment_cv_median").observable
    with pytest.raises(ValueError, match="include_truth"):
        _describe(_observed(np.ones((1, 10, 1))), include_truth=True)


def test_concat_keeps_rows_aligns_columns_and_refuses_collisions():
    rng = np.random.default_rng(10)
    a = _describe(_observed(rng.normal(size=(2, 10, 2))), source_id="A")
    b = _describe(_observed(rng.normal(size=(2, 10, 2))), source_id="B")
    both = WorldDescriptors.concat([a, b])
    assert both.sources == ("A", "B")
    np.testing.assert_array_equal(both.values, np.vstack([a.values, b.values]))
    # Equal definitions in a different column order are aligned, not refused.
    shuffled = WorldDescriptors.concat([a, b.select(tuple(reversed(b.features)))])
    np.testing.assert_array_equal(shuffled.values, both.values)
    # Lags are canonical: (3, 1) and (1, 3) describe the same columns, in one order.
    x = rng.normal(size=(1, 10, 2))
    left = _describe(_observed(x), lags=(1, 3), source_id="L")
    right = _describe(_observed(x), lags=(3, 1), source_id="R")
    assert left.features == right.features
    # Wording is not part of a definition.
    reworded = tuple(type(d)(**{**d.__dict__, "description": "reworded"}) for d in b.definitions)
    relabeled = WorldDescriptors(**{**b.__dict__, "definitions": reworded})
    assert WorldDescriptors.concat([a, relabeled]).n_worlds == 4
    with pytest.raises(ValueError, match="unique"):
        WorldDescriptors.concat([a, a])
    other_lags = _describe(_observed(rng.normal(size=(2, 10, 2))), lags=(2,), source_id="C")
    with pytest.raises(ValueError, match="definitions differ"):
        WorldDescriptors.concat([a, other_lags])


def test_select_keeps_each_column_with_its_definition():
    rng = np.random.default_rng(18)
    desc = _describe(_observed(rng.normal(size=(2, 10, 3))), statistics=("cv",))
    names = ["treatment_cv_max", "treatment_cv_median"]
    picked = desc.select(names)
    assert picked.features == tuple(names)
    for name in names:
        assert picked.definition(name) == desc.definition(name)
        np.testing.assert_array_equal(picked.column(name), desc.column(name))


def test_take_follows_numpy_selection_semantics():
    rng = np.random.default_rng(12)
    desc = _describe(_observed(rng.normal(size=(3, 10, 2))), statistics=("cv",))
    assert desc.take([2, 0]).world_ids.tolist() == [2, 0]
    assert desc.take(-1).world_ids.tolist() == [2]
    assert desc.take(np.array([True, False, True])).world_ids.tolist() == [0, 2]
    assert desc.take(np.zeros(3, dtype=bool)).n_worlds == 0
    assert desc.take([]).n_worlds == 0
    one = desc.take([0])
    assert one.take([0]).n_worlds == 1  # a 0/1 position list is positions, not a mask
    with pytest.raises(ValueError, match="repeat"):
        desc.take([0, 0])
    with pytest.raises(ValueError, match="shape"):
        desc.take(np.array([True, False]))


def test_table_rejects_a_bare_string_of_features():
    desc = _describe(
        _observed(np.random.default_rng(12).normal(size=(3, 10, 2))), statistics=("cv",)
    )
    with pytest.raises(TypeError, match="bare string"):
        desc.table("treatment_cv_median")


def test_table_invariants_are_enforced_by_the_constructor():
    rng = np.random.default_rng(11)
    desc = _describe(_observed(rng.normal(size=(2, 10, 2))), statistics=("cv",))
    fields = dict(desc.__dict__)
    bad_status = desc.status.copy()
    bad_status[0, 0] = INELIGIBLE  # value still present -> NaN/status mismatch
    with pytest.raises(ValueError, match="NaN exactly"):
        WorldDescriptors(**{**fields, "status": bad_status})
    with pytest.raises(ValueError, match="unique"):
        WorldDescriptors(**{**fields, "world_ids": np.array([0, 0])})
    with pytest.raises(ValueError, match=">= 0"):
        WorldDescriptors(**{**fields, "world_ids": np.array([-1, 0])})


def test_observed_padding_must_be_exactly_zero():
    with pytest.raises(ValueError, match="non-zero inactive padding at world 1"):
        ObservedWorlds(
            treatment_raw=np.ones((2, 8, 2)),
            treatment_active_mask=np.array([[1, 1], [1, 0]]),
        )


def test_corpus_padding_errors_name_the_source_world():
    rng = np.random.default_rng(25)
    treatment = rng.gamma(2.0, 1.0, size=(6, 10, 2))
    mask = np.ones((6, 2))
    mask[5, 1] = 0  # world 5's second treatment is padding, but holds data
    corpus = {
        "treatment_raw": treatment,
        "covariates": np.zeros((6, 10, 0)),
        "outcome_raw": rng.normal(10.0, 1.0, size=(6, 10)),
        "treatment_active_mask": mask,
        "covariate_active_mask": np.ones((6, 0)),
    }
    # chunk_size=2 puts world 5 second in its chunk: the error must still say 5.
    with pytest.raises(ValueError, match="padding at world 5,"):
        _describe(corpus, chunk_size=2)


# ---------------------------------------------------------------------------
# Dataset format: from_frame, from_csv, to_frame
# ---------------------------------------------------------------------------

EPS16, EPS32, EPS64 = (float(np.finfo(t).eps) for t in (np.float16, np.float32, np.float64))
_ROLES = {
    "treatments": ["treatment_C1", "treatment_C2"],
    "covariates": ["covariate_Z1"],
    "outcome": "outcome_Y",
}


def _dataset(t: int = 24, **columns) -> pd.DataFrame:
    """A bundle-layout dataset table: two treatments, one covariate and the outcome."""
    rng = np.random.default_rng(t)
    frame = pd.DataFrame(
        {
            "week": np.arange(t),
            "treatment_C1": rng.gamma(2.0, 1.0, t),
            "treatment_C2": rng.gamma(2.0, 1.0, t),
            "covariate_Z1": rng.normal(10.0, 1.0, t),
            "outcome_Y": rng.normal(100.0, 5.0, t),
        }
    )
    for label, values in columns.items():
        frame[label] = values
    return frame


def _assert_same_series(a: ObservedWorlds, b: ObservedWorlds) -> None:
    for name in ("treatment_raw", "covariates", "outcome_raw"):
        np.testing.assert_array_equal(getattr(a, name), getattr(b, name))


def test_a_bundle_dataset_reads_back_exactly_and_describes_like_its_world(scm_worlds, tmp_path):
    world = scm_worlds[0]
    path = pg.write_scm_bundle(world, tmp_path / "bundle", plots=False) / "dataset.csv"
    observed = ObservedWorlds.from_csv(path)
    np.testing.assert_array_equal(observed.treatment_raw[0], world.data["treatments"])
    np.testing.assert_array_equal(observed.covariates[0], world.data["covariates"])
    np.testing.assert_array_equal(observed.outcome_raw[0], world.data["outcome"])
    read = _describe(observed, lags=(1, 2))
    direct = _describe([world], lags=(1, 2))
    np.testing.assert_array_equal(read.status, direct.status)
    np.testing.assert_array_equal(read.values, direct.values)
    # The same series under other names and in another column order: roles set the order.
    x, z, y = (world.data[key] for key in ("treatments", "covariates", "outcome"))
    renamed = pd.DataFrame(
        {
            "sales": y,
            "search": x[:, 2],
            "temperature": z[:, 1],
            "tv": x[:, 0],
            "price": z[:, 0],
            "radio": x[:, 1],
        }
    )
    explicit = ObservedWorlds.from_frame(
        renamed,
        treatments=["tv", "radio", "search"],
        covariates=["price", "temperature"],
        outcome="sales",
    )
    _assert_same_series(explicit, observed)


def test_to_frame_round_trips_the_active_columns_of_a_padded_float32_world(generated):
    corpus = ObservedWorlds(
        treatment_raw=generated["treatment_raw"],
        covariates=generated["covariates"],
        outcome_raw=generated["outcome_raw"],
        treatment_active_mask=generated["treatment_active_mask"],
        covariate_active_mask=generated["covariate_active_mask"],
    )
    tmask = generated["treatment_active_mask"].astype(bool)
    cmask = generated["covariate_active_mask"].astype(bool)
    world = int(np.flatnonzero(~tmask.all(axis=1) & ~cmask.all(axis=1))[0])
    frame = corpus.to_frame(world)
    n_treat, n_cov = int(tmask[world].sum()), int(cmask[world].sum())
    assert list(frame.columns) == [
        "week",
        *(f"treatment_C{k}" for k in range(1, n_treat + 1)),
        *(f"covariate_Z{m}" for m in range(1, n_cov + 1)),
        "outcome_Y",
    ]
    assert frame["week"].dtype == np.int64
    assert frame["week"].tolist() == list(range(corpus.n_time_steps))
    assert set(frame.dtypes.iloc[1:]) == {np.dtype(np.float32)}
    back = ObservedWorlds.from_frame(frame)
    np.testing.assert_array_equal(
        back.treatment_raw[0], generated["treatment_raw"][world][:, tmask[world]]
    )
    np.testing.assert_array_equal(
        back.covariates[0], generated["covariates"][world][:, cmask[world]]
    )
    np.testing.assert_array_equal(back.outcome_raw[0], generated["outcome_raw"][world])
    assert back.resolution == corpus.resolution == (EPS32, EPS32, EPS32)


def test_a_float32_column_flat_within_an_ulp_keeps_an_infinite_level_to_variation():
    flat = (0.7 * (1.0 + EPS32 * (np.arange(40) % 2))).astype(np.float32)
    assert np.unique(flat).size == 2  # a one-ULP wiggle, not a literal constant
    frame = _dataset(40).astype({"treatment_C2": np.float32})
    frame["treatment_C1"] = flat
    desc = _describe(ObservedWorlds.from_frame(frame), reducers=("max",))
    assert desc.column("treatment_level_to_variation_max")[0] == np.inf


def test_each_role_keeps_its_columns_resolution_through_to_frame():
    frame = _dataset(40).astype(
        {"treatment_C1": np.float16, "treatment_C2": np.float16, "covariate_Z1": np.float32}
    )
    observed = ObservedWorlds.from_frame(frame)
    assert observed.resolution == (EPS16, EPS32, EPS64)
    written = observed.to_frame()
    assert written.dtypes.tolist() == [np.int64, np.float16, np.float16, np.float32, np.float64]
    back = ObservedWorlds.from_frame(written)
    assert back.resolution == observed.resolution
    _assert_same_series(back, observed)
    alone = ObservedWorlds.from_frame(
        frame.astype({"treatment_C1": np.float32, "treatment_C2": np.float32})
    )
    mixed = ObservedWorlds.from_frame(
        frame.astype({"treatment_C1": np.float32, "treatment_C2": np.float64})
    )
    assert (alone.resolution[0], mixed.resolution[0]) == (EPS32, EPS64)


def test_nullable_columns_are_read_at_the_resolution_of_their_values():
    frame = _dataset(30)
    frame["treatment_C2"] = np.round(100.0 * frame["treatment_C2"])
    nullable = frame.astype(
        {"treatment_C1": "Float64", "treatment_C2": "Int64", "covariate_Z1": "Float32"}
    )
    observed = ObservedWorlds.from_frame(nullable)
    _assert_same_series(
        observed, ObservedWorlds.from_frame(frame.astype({"covariate_Z1": np.float32}))
    )
    assert observed.resolution == (EPS64, EPS32, EPS64)


def test_arrow_columns_read_like_numpy_ones_and_arrow_nulls_and_flags_are_refused(tmp_path):
    path = tmp_path / "dataset.csv"
    _dataset(30).to_csv(path, index=False)
    arrow = ObservedWorlds.from_csv(path, dtype_backend="pyarrow")
    _assert_same_series(arrow, ObservedWorlds.from_csv(path))
    assert arrow.resolution == (EPS64, EPS64, EPS64)
    gap = _dataset(30)
    gap.loc[4, "covariate_Z1"] = np.nan
    gap.to_csv(path, index=False)
    with pytest.raises(
        ValueError,
        match=r"'covariate_Z1' \(covariates\) has 1 missing value; the first is at row 4",
    ):
        ObservedWorlds.from_csv(path, dtype_backend="pyarrow")
    _dataset(30, covariate_Z1=np.arange(30) % 3 == 0).to_csv(path, index=False)
    assert str(pd.read_csv(path, dtype_backend="pyarrow")["covariate_Z1"].dtype) == "bool[pyarrow]"
    with pytest.raises(TypeError, match=r"'covariate_Z1' \(covariates\) is boolean"):
        ObservedWorlds.from_csv(path, dtype_backend="pyarrow")


def test_missing_and_non_finite_values_are_refused_naming_column_count_and_row():
    frame = _dataset(12).set_axis(range(100, 112)).astype({"treatment_C2": "Float64"})
    frame.loc[[102, 107], "treatment_C2"] = pd.NA
    with pytest.raises(
        ValueError,
        match=r"'treatment_C2' \(treatments\) has 2 missing values; the first is at row 102",
    ):
        ObservedWorlds.from_frame(frame)
    frame = _dataset(12).set_axis(range(100, 112))
    frame.loc[105, "outcome_Y"] = np.inf
    with pytest.raises(
        ValueError, match=r"'outcome_Y' \(outcome\) has 1 non-finite value; the first is at row 105"
    ):
        ObservedWorlds.from_frame(frame)


@pytest.mark.parametrize(
    "values",
    [
        pd.Series([str(v) for v in range(1, 13)]),
        pd.Series([float(v) for v in range(1, 13)], dtype=object),
        pd.Series(np.arange(12) % 2 == 0),
        pd.Series(np.arange(12) % 2 == 0, dtype="boolean"),
    ],
    ids=["str", "object", "bool", "boolean"],
)
def test_text_object_and_flag_columns_are_refused_naming_the_column(values):
    with pytest.raises(TypeError, match=r"column 'covariate_Z1' \(covariates\)"):
        ObservedWorlds.from_frame(_dataset(12, covariate_Z1=values))


def test_text_time_is_parsed_as_iso_8601_with_offsets_converted():
    frame = _dataset(4)
    plain = ObservedWorlds.from_frame(frame)
    for stamps in (
        ["2024-01-01", "2024-01-08", "2024-01-15", "2024-01-22"],
        # Increasing in time although the second stamp sorts first as text.
        [
            "2024-03-31T00:15:00+01:00",
            "2024-03-30T23:30:00+00:00",
            "2024-03-31T01:00:00+01:00",
            "2024-03-31T03:00:00+02:00",
        ],
    ):
        _assert_same_series(
            ObservedWorlds.from_frame(frame.assign(date=stamps), time="date"), plain
        )
    # Increasing as text, but 00:15+01:00 is 15 minutes before 23:30+00:00.
    backwards = [
        "2024-03-30T23:30:00+00:00",
        "2024-03-31T00:15:00+01:00",
        "2024-03-31T01:00:00+01:00",
        "2024-03-31T03:00:00+02:00",
    ]
    with pytest.raises(ValueError, match="time column 'date' does not strictly increase: row 1 "):
        ObservedWorlds.from_frame(frame.assign(date=backwards), time="date")
    # Text that is not ISO 8601 names the first value that does not parse.
    mixed = ["2024-01-01", "2024-01-08", "15/01/2024", "2024-01-22"]
    with pytest.raises(
        ValueError,
        match=r"not ISO 8601 date text: row 102 holds '15/01/2024'; read it with parse_dates=",
    ) as raised:
        ObservedWorlds.from_frame(frame.set_axis(range(100, 104)).assign(date=mixed), time="date")
    assert "You might want" not in str(raised.value)
    # A missing stamp before it is not the value that failed to parse.
    gap = ["2024-01-01", None, "15/01/2024", "2024-01-22"]
    with pytest.raises(ValueError, match=r"not ISO 8601 date text: row 2 holds '15/01/2024'"):
        ObservedWorlds.from_frame(frame.assign(date=gap), time="date")


@pytest.mark.parametrize(
    "stamps",
    [
        pd.period_range("2024-01-01", periods=5, freq="W"),
        pd.to_timedelta(np.arange(5), unit="W"),
    ],
    ids=["period", "timedelta"],
)
def test_period_and_timedelta_time_count_in_order_and_must_increase(stamps):
    frame = _dataset(5)
    observed = ObservedWorlds.from_frame(frame.assign(t=stamps), time="t")
    _assert_same_series(observed, ObservedWorlds.from_frame(frame))
    with pytest.raises(ValueError, match="row 3 is not later than the row before it"):
        ObservedWorlds.from_frame(frame.assign(t=stamps[[0, 1, 2, 1, 4]]), time="t")


def test_out_of_order_missing_or_infinite_time_names_the_first_offending_row():
    frame = _dataset(6).set_axis(range(100, 106))
    with pytest.raises(ValueError, match=r"row 103 is not later than the row before it.*sort"):
        ObservedWorlds.from_frame(frame.assign(t=[0, 1, 2, 2, 4, 5]), time="t")
    for stamps, row in (
        ([0.0, 1.0, 2.0, 3.0, np.nan, 5.0], 104),
        ([0.0, 1.0, 2.0, 3.0, 4.0, np.inf], 105),
    ):
        with pytest.raises(
            ValueError, match=f"time column 't' is missing or non-finite at row {row}"
        ):
            ObservedWorlds.from_frame(frame.assign(t=stamps), time="t")


def test_week_is_never_read_as_time_unless_named():
    frame = _dataset(10)
    frame["week"] = np.random.default_rng(1).permutation(10)
    frame[0] = np.arange(10.0)  # other columns, non-str labels included, are ignored
    frame["notes"] = "promo"
    expected = frame[["treatment_C1", "treatment_C2"]].to_numpy()
    for observed in (ObservedWorlds.from_frame(frame), ObservedWorlds.from_frame(frame, **_ROLES)):
        np.testing.assert_array_equal(observed.treatment_raw[0], expected)
    with pytest.raises(ValueError, match="time column 'week' does not strictly increase"):
        ObservedWorlds.from_frame(frame, time="week")


def test_default_labels_keep_table_order_and_multi_digit_numbers():
    rng = np.random.default_rng(5)
    names = ("treatment_C2", "treatment_C10", "treatment_C1", "covariate_Z1", "outcome_Y")
    frame = pd.DataFrame({name: rng.gamma(2.0, 1.0, 10) for name in names})
    observed = ObservedWorlds.from_frame(frame)
    np.testing.assert_array_equal(observed.treatment_raw[0], frame[list(names[:3])].to_numpy())


@pytest.mark.parametrize(
    ("column", "near_miss"),
    [
        ("treatment_C1", "treatment_C1 "),
        ("treatment_C2", "Treatment_C2"),
        ("treatment_C1", "treatment_\u200bC1"),
        ("covariate_Z1", "\ufeffcovariate_Z1"),
        ("outcome_Y", "OUTCOME_Y"),
    ],
    ids=["space", "case", "zero-width", "byte-order-mark", "outcome-case"],
)
def test_labels_matching_the_bundle_only_after_normalization_are_refused(column, near_miss):
    with pytest.raises(ValueError, match=re.escape(repr(near_miss))):
        ObservedWorlds.from_frame(_dataset(8).rename(columns={column: near_miss}))


def test_a_label_must_name_one_column_and_play_one_role():
    frame = _dataset(8)
    with pytest.raises(ValueError, match="'treatment_C2' appears 2 times"):
        ObservedWorlds.from_frame(pd.concat([frame, frame[["treatment_C2"]]], axis=1))
    repeated = {**_ROLES, "treatments": ["treatment_C1", "treatment_C1"]}
    with pytest.raises(ValueError, match="'treatment_C1' is used twice in treatments"):
        ObservedWorlds.from_frame(frame, **repeated)
    with pytest.raises(ValueError, match="'covariate_Z1' is used as covariates and time"):
        ObservedWorlds.from_frame(frame, **_ROLES, time="covariate_Z1")
    with pytest.raises(KeyError, match="treatment column 'tv' not found"):
        ObservedWorlds.from_frame(frame, **{**_ROLES, "treatments": ["tv"]})


def test_role_labels_may_be_any_ordered_collection_but_never_a_set():
    frame = _dataset(8)
    reordered = frame[["treatment_C2", "treatment_C1"]].to_numpy()
    for treatments in (frame.columns[[2, 1]], np.array(["treatment_C2", "treatment_C1"])):
        observed = ObservedWorlds.from_frame(
            frame, treatments=treatments, covariates=frame.columns[3:4], outcome="outcome_Y"
        )
        np.testing.assert_array_equal(observed.treatment_raw[0], reordered)
    with pytest.raises(KeyError, match="treatment column 'tv' not found"):
        ObservedWorlds.from_frame(frame, **{**_ROLES, "treatments": np.array(["tv"])})
    unordered = {"treatment_C1", "treatment_C2"}
    with pytest.raises(TypeError, match="ordered collection of column labels .* got set"):
        ObservedWorlds.from_frame(frame, **{**_ROLES, "treatments": unordered})


def test_multiindex_columns_are_named_by_their_full_tuples():
    frame = _dataset(8)
    wide = pd.concat(
        {
            "media": frame[["treatment_C1", "treatment_C2"]],
            "context": frame[["covariate_Z1"]],
            "result": frame[["outcome_Y"]],
            "calendar": frame[["week"]],
        },
        axis=1,
    )
    roles = {
        "treatments": [("media", "treatment_C1"), ("media", "treatment_C2")],
        "covariates": [("context", "covariate_Z1")],
        "outcome": ("result", "outcome_Y"),
    }
    observed = ObservedWorlds.from_frame(wide, **roles, time=("calendar", "week"))
    _assert_same_series(observed, ObservedWorlds.from_frame(frame))
    # A partial key names a block of columns, never one column.
    with pytest.raises(KeyError, match="outcome column 'result' not found"):
        ObservedWorlds.from_frame(wide, **{**roles, "outcome": "result"})
    with pytest.raises(TypeError, match="a dataset has one outcome column; .*flatten"):
        ObservedWorlds.from_frame(wide, **{**roles, "outcome": ("result",)})
    wide[("calendar", "week")] = frame["week"].to_numpy()[::-1]
    with pytest.raises(ValueError, match=r"time column \('calendar', 'week'\) does not strictly"):
        ObservedWorlds.from_frame(wide, **roles, time=("calendar", "week"))


def test_roles_are_label_sequences_and_one_outcome_given_together():
    frame = _dataset(8)
    with pytest.raises(TypeError, match="treatments must be a sequence .* not a bare string"):
        ObservedWorlds.from_frame(frame, **{**_ROLES, "treatments": "treatment_C1"})
    with pytest.raises(TypeError, match="covariates must be a sequence .* not a bare string"):
        ObservedWorlds.from_frame(frame, **{**_ROLES, "covariates": "covariate_Z1"})
    with pytest.raises(TypeError, match="outcome must be one column label, got list"):
        ObservedWorlds.from_frame(frame, **{**_ROLES, "outcome": ["outcome_Y"]})
    two = ("outcome_Y", "covariate_Z1")
    with pytest.raises(TypeError, match="got tuple .*; a dataset has one outcome column"):
        ObservedWorlds.from_frame(frame, **{**_ROLES, "outcome": two})
    with pytest.raises(ValueError, match="covariates and outcome are missing"):
        ObservedWorlds.from_frame(frame, treatments=["treatment_C1"])


def test_an_all_zero_column_is_refused_as_an_inactive_series():
    with pytest.raises(
        ValueError,
        match=r"'treatment_C2' \(treatments\) is zero at every time step; drop the inactive",
    ):
        ObservedWorlds.from_frame(_dataset(8, treatment_C2=np.zeros(8)))


def test_a_mapping_is_read_by_position_from_aligned_one_dimensional_values():
    index = pd.Index([7, 3, 9, 1])
    columns = {
        "treatment_C1": pd.Series([1.0, 2.0, 3.0, 4.0], index=index),
        "covariate_Z1": pd.Series([5.0, 6.0, 7.0, 8.0], index=index),
        "outcome_Y": [9.0, 10.0, 11.0, 12.0],
    }
    observed = ObservedWorlds.from_frame(columns)
    np.testing.assert_array_equal(observed.treatment_raw[0, :, 0], [1.0, 2.0, 3.0, 4.0])
    np.testing.assert_array_equal(observed.outcome_raw[0], [9.0, 10.0, 11.0, 12.0])
    gap = {**columns, "covariate_Z1": pd.Series([5.0, 6.0, np.nan, 8.0], index=index)}
    with pytest.raises(ValueError, match="the first is at row 2"):
        ObservedWorlds.from_frame(gap)
    with pytest.raises(ValueError, match="'treatment_C1' must be one-dimensional.*a scalar"):
        ObservedWorlds.from_frame({**columns, "treatment_C1": 1.0})
    with pytest.raises(ValueError, match="differ in length"):
        ObservedWorlds.from_frame({**columns, "outcome_Y": [9.0, 10.0, 11.0]})
    shuffled = pd.Series([5.0, 6.0, 7.0, 8.0], index=[1, 3, 7, 9])
    with pytest.raises(ValueError, match="different indexes.*align them first"):
        ObservedWorlds.from_frame({**columns, "covariate_Z1": shuffled})


def test_to_frame_refuses_worlds_that_from_frame_could_not_read():
    rng = np.random.default_rng(31)
    x = rng.gamma(2.0, 1.0, size=(2, 10, 2))
    z = rng.normal(10.0, 1.0, size=(2, 10, 1))
    y = rng.normal(100.0, 5.0, size=(2, 10))
    with pytest.raises(ValueError, match=r"world must be an integer in \[0, 2\), got 2"):
        ObservedWorlds(treatment_raw=x, covariates=z, outcome_raw=y).to_frame(2)
    with pytest.raises(ValueError, match="no outcome"):
        ObservedWorlds(treatment_raw=x, covariates=z).to_frame(0)
    with pytest.raises(ValueError, match="world 0 has no active covariate"):
        ObservedWorlds(treatment_raw=x, outcome_raw=y).to_frame(0)
    silent = x.copy()
    silent[1, :, 1] = 0.0
    with pytest.raises(ValueError, match="world 1: active treatment column 1 is zero"):
        ObservedWorlds(treatment_raw=silent, covariates=z, outcome_raw=y).to_frame(1)
    with pytest.raises(ValueError, match="world 1: the outcome is zero at every time step"):
        ObservedWorlds(treatment_raw=x, covariates=z, outcome_raw=y * [[1.0], [0.0]]).to_frame(1)


def test_from_csv_reads_full_precision_float64_exactly_from_a_decimal_comma_file(tmp_path):
    frame = _dataset(200)
    path = tmp_path / "dataset.csv"
    frame.to_csv(path, index=False, sep=";", decimal=",")
    observed = ObservedWorlds.from_csv(path, sep=";", decimal=",")
    _assert_same_series(observed, ObservedWorlds.from_frame(frame))
    with pytest.raises(TypeError, match="check sep=, decimal= and thousands="):
        ObservedWorlds.from_csv(path, sep=";")


@pytest.mark.parametrize("source", ["path", "text buffer", "text handle"])
def test_from_csv_reads_with_the_python_engine_and_a_sniffed_separator(tmp_path, source):
    frame = _dataset(12).round(2)  # short decimals: every parser engine reads them exactly
    path = tmp_path / "dataset.csv"
    frame.to_csv(path, index=False, sep=";")
    expected = ObservedWorlds.from_frame(frame)
    for options in ({"engine": "python", "sep": ";"}, {"sep": None}):
        if source == "text handle":  # already decoded: from_csv must not impose an encoding
            with path.open(encoding="utf-8") as handle:
                observed = ObservedWorlds.from_csv(handle, **options)
        else:
            target = path if source == "path" else io.StringIO(path.read_text())
            observed = ObservedWorlds.from_csv(target, **options)
        _assert_same_series(observed, expected)


@pytest.mark.parametrize("source", ["path", "bytes"])
def test_from_csv_drops_a_byte_order_mark_when_sniffing_the_separator(tmp_path, source):
    rng = np.random.default_rng(9)
    names = ("tv", "radio", "price", "sales")
    frame = pd.DataFrame({name: rng.gamma(2.0, 10.0, 12).round(2) for name in names})
    path = tmp_path / "dataset.csv"
    frame.to_csv(path, index=False, sep=";", decimal=",", encoding="utf-8-sig")
    assert path.read_bytes().startswith(codecs.BOM_UTF8)
    roles = {"treatments": ["tv", "radio"], "covariates": ["price"], "outcome": "sales"}
    target = path if source == "path" else io.BytesIO(path.read_bytes())
    observed = ObservedWorlds.from_csv(target, sep=None, decimal=",", **roles)
    _assert_same_series(observed, ObservedWorlds.from_frame(frame, **roles))
