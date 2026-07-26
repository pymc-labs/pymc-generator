"""SCMPrior validation, the preset factory, and the deprecation policy."""

from __future__ import annotations

import warnings

import numpy as np
import pytest

import prior_generator as pg
from prior_generator.sampler import SCMPrior
from prior_generator.slots import EDGE_TYPES_EXTENDED


def test_factory_pins_additive_schema():
    cfg = pg.make_scm_prior(n_treatments=8, n_covariates=4, n_latent=3)
    assert (cfg.n_treatments, cfg.n_covariates, cfg.n_latent) == (8, 4, 3)
    assert cfg.layout.edge_types == EDGE_TYPES_EXTENDED
    # default active ranges pin every node active
    assert cfg.n_treatments_active_range == (8, 8)
    assert cfg.n_covariates_active_range == (4, 4)
    assert cfg.n_latent_active_range == (3, 3)


def test_factory_diverse_texture_defaults():
    cfg = pg.make_scm_prior(n_treatments=4, n_covariates=2, n_latent=1, l_max=8)
    assert cfg.channel_hf_sigma_range[1] > 0
    assert cfg.channel_pulse_prob_range[1] > 0
    assert cfg.rw_channel_std_range is not None
    assert cfg.adstock_burn_in == cfg.l_max


def test_factory_linear_nonlinearity():
    cfg = pg.make_scm_prior(n_treatments=4, n_covariates=2, n_latent=1, nonlinearity="linear")
    assert cfg.adstock_family_probs == (1.0, 0.0, 0.0)
    assert cfg.saturation_family_probs == (1.0, 0.0, 0.0, 0.0, 0.0, 0.0)


def test_factory_overrides_win():
    cfg = pg.make_scm_prior(n_treatments=4, n_covariates=2, n_latent=1, T=200, spend_cv_floor=0.2)
    assert cfg.T == 200
    assert cfg.spend_cv_floor == 0.2


def test_legacy_texture_rejected():
    with pytest.raises(ValueError, match="texture"):
        pg.make_scm_prior(n_treatments=4, n_covariates=2, n_latent=1, texture="legacy")


def test_bad_nonlinearity_rejected():
    with pytest.raises(ValueError, match="nonlinearity"):
        pg.make_scm_prior(n_treatments=4, n_covariates=2, n_latent=1, nonlinearity="quadratic")


def test_edge_budget_unknown_key_rejected():
    with pytest.raises(ValueError, match="edge_budget"):
        pg.make_scm_prior(n_treatments=4, n_covariates=2, n_latent=1, edge_budget={"xy": 3})


@pytest.mark.parametrize("spec", [-1, (2, 1), (1, 2, 3), 1.5, True])
def test_edge_budget_bad_spec_rejected(spec):
    with pytest.raises(ValueError):
        pg.make_scm_prior(n_treatments=4, n_covariates=2, n_latent=1, edge_budget={"cc": spec})


@pytest.mark.parametrize("l_max", (0, -1, True, 1.5))
def test_lmax_must_be_a_positive_integer(l_max):
    with pytest.raises(ValueError, match="l_max"):
        SCMPrior(l_max=l_max).validate()


@pytest.mark.parametrize("name", ("rw_std_sigma", "rw_channel_std_sigma", "rw_sales_std_sigma"))
@pytest.mark.parametrize("value", (np.nan, np.inf, 0.0, -1.0))
def test_random_walk_sigmas_must_be_finite_and_positive(name, value):
    with pytest.raises(ValueError, match=name):
        SCMPrior(**{name: value}).validate()


@pytest.mark.parametrize(
    ("name", "value"),
    (
        ("adstock_alpha_range", (-0.1, 0.5)),
        ("adstock_alpha_range", (0.5, 1.1)),
        ("weibull_lam_range", (0.0, 1.0)),
        ("weibull_k_range", (2.0, 1.0)),
        ("beta_additive_range", (-0.1, 1.0)),
        ("cc_coeff_range", (-0.1, 0.2)),
        ("rw_positive_mean_range", (0.0, 1.0)),
        ("rw_mean_range", (np.nan, 1.0)),
        ("rw_baseline_mean_range", (2.0, 1.0)),
    ),
)
def test_prior_ranges_fail_fast_on_invalid_bounds(name, value):
    with pytest.raises(ValueError, match=name):
        SCMPrior(**{name: value}).validate()


@pytest.mark.parametrize("value", (np.nan, np.inf, 8.0, 8.5, True))
def test_adstock_burn_in_must_be_an_integer(value):
    with pytest.raises(ValueError, match="adstock_burn_in"):
        SCMPrior(adstock_burn_in=value).validate()


@pytest.mark.parametrize("value", (True, "bad", [1.0], np.nan, np.inf, 0.0))
def test_baseline_walk_sigma_must_be_a_finite_positive_scalar(value):
    with pytest.raises(ValueError, match="rw_baseline_std_sigma"):
        SCMPrior(rw_baseline_std_sigma=value).validate()


@pytest.mark.parametrize(
    ("name", "value"),
    (
        ("spend_cv_floor", np.nan),
        ("spend_cv_floor", np.inf),
        ("spend_cv_floor", -0.1),
        ("spend_cv_floor", True),
        ("rw_smoothness_alpha", np.nan),
        ("rw_smoothness_alpha", 0.0),
        ("rw_smoothness_beta", np.inf),
        ("rw_smoothness_beta", -1.0),
        ("val_cell_frac", np.nan),
        ("val_cell_frac", 0.0),
        ("val_cell_frac", 1.0),
        ("val_cell_frac", True),
        ("query_frac", np.nan),
        ("query_frac", np.inf),
        ("query_frac", 0.0),
        ("query_frac", True),
        ("p_long_horizon", np.nan),
        ("p_long_horizon", np.inf),
        ("p_long_horizon", -0.1),
        ("p_long_horizon", 1.1),
    ),
)
def test_scalar_domain_parameters_fail_fast(name, value):
    with pytest.raises(ValueError, match=name):
        SCMPrior(**{name: value}).validate()


@pytest.mark.parametrize(
    ("name", "value"),
    (
        ("adstock_family_probs", None),
        ("adstock_family_probs", 1),
        ("adstock_family_probs", (np.nan, 0.5, 0.5)),
        ("adstock_family_probs", (True, 0.0, 0.0)),
        ("adstock_family_probs", (-0.1, 0.5, 0.6)),
        ("adstock_family_probs", (0.3, 0.3, 0.40001)),
        ("saturation_family_probs", None),
        ("saturation_family_probs", 1),
        ("saturation_family_probs", (np.inf, 0.0, 0.0, 0.0, 0.0, 0.0)),
        ("saturation_family_probs", (False, 0.2, 0.2, 0.2, 0.2, 0.2)),
        ("saturation_family_probs", (1.1, 0.0, 0.0, 0.0, 0.0, -0.1)),
        ("saturation_family_probs", (0.15, 0.17, 0.17, 0.17, 0.17, 0.0)),
    ),
)
def test_family_probabilities_fail_fast(name, value):
    with pytest.raises(ValueError, match=name):
        SCMPrior(**{name: value}).validate()


@pytest.mark.parametrize(
    "name",
    ("n_treatments", "n_covariates", "n_latent", "n_cells", "draws_per_cell", "T", "seed"),
)
@pytest.mark.parametrize("value", (True, np.nan, np.inf, 1.5))
def test_integer_configuration_fields_reject_nonintegers(name, value):
    with pytest.raises(ValueError, match=name):
        SCMPrior(**{name: value}).validate()


@pytest.mark.parametrize(
    ("name", "value"),
    (
        ("n_treatments", 0),
        ("n_covariates", 0),
        ("n_latent", 0),
        ("n_cells", 1),
        ("draws_per_cell", 0),
        ("T", 3),
        ("seed", -1),
    ),
)
def test_integer_configuration_fields_enforce_minima(name, value):
    with pytest.raises(ValueError, match=name):
        SCMPrior(**{name: value}).validate()


@pytest.mark.parametrize(
    ("name", "value"),
    (
        ("n_treatments_active_range", (1.5, 2)),
        ("n_covariates_active_range", (True, 2)),
        ("n_latent_active_range", (2, 1)),
        ("n_treatments_active_range", (1, 2, 3)),
        ("n_covariates_active_range", None),
    ),
)
def test_active_count_ranges_require_ordered_integer_pairs(name, value):
    with pytest.raises(ValueError, match=name):
        SCMPrior(**{name: value}).validate()


# --- deprecation / steering policy -----------------------------------------


def test_diverse_texture_does_not_warn():
    cfg = pg.make_scm_prior(
        n_treatments=4, n_covariates=2, n_latent=1, T=32, n_cells=2, draws_per_cell=1
    )
    with warnings.catch_warnings():
        warnings.simplefilter("error", FutureWarning)
        pg.sample_prior_predictive(cfg)  # must not raise


def test_flat_texture_warns():
    # A hand-built config with the texture disabled (hf & pulse ranges zero) is
    # the deprecated smooth-walk-only prior and must warn.
    cfg = SCMPrior(
        T=32,
        n_treatments=4,
        n_covariates=2,
        n_latent=1,
        n_treatments_active_range=(4, 4),
        n_covariates_active_range=(2, 2),
        n_latent_active_range=(1, 1),
        n_cells=2,
        draws_per_cell=1,
    )
    with pytest.warns(FutureWarning, match="texture"):
        pg.sample_prior_predictive(cfg)
