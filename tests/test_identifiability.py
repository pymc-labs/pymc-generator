"""Legacy deterministic contract before identifiability features are enabled."""

from __future__ import annotations

import hashlib
from dataclasses import replace

import numpy as np
import pytest

import prior_generator as pg
from prior_generator.sampler import (
    _ADDITIVE_OUT_NAMES,
    _CORPUS_PARAM_NAMES,
    _CORPUS_SHOCK_NAMES,
    _slice_g_active,
    sample_g_additive,
)
from prior_generator.signal_diagnostics import (
    SIGNAL_METRIC_LAYOUT,
    check_signal_gate,
    dense_signal_metrics,
)
from prior_generator.world_model import build_world_model, draw_worlds, sample_structure

EXPECTED_CORPUS_HASHES = {
    "spend_raw": "c2bbfc474bd5a7b8c9a70eb518e5a082b6f699d8c9806554de87fdc877df5279",
    # Moved when the persisted normalizers dropped their raw-unit `+ 1e-8`
    # denominator epsilon for exact-zero-guarded division: a 1.17e-7 relative
    # shift, i.e. the last float32 bit of 38 of 96 cells. Every other array in
    # this contract is unchanged.
    "spend_norm": "3ae8d9289fd25095d3cf26db1a575e42480587ecce8d97d67ab97dcba9a78bba",
    "spend_share": "674ba766b14c141d578b1e1825a0937d077a23344c3b2dfa455f3d4c608acc8a",
    "controls": "3626a20d0a1e6f15b84cd589b22fe110ddfac34853e92f4369889bf55eccbaf4",
    "sales_raw": "e1c86549bc89889185d421a537e98bba0635e671e686c0bd5d3c830312077bcf",
    "sales_norm": "d02208d64adc6ff3c0039941b6e0fce93a65781855c096eb2dd55419d1e47fb2",
    "support_mask": "95ee8b55a20094102512582a4d0021d303214a842b292998ea7a97b7dcd7b0c3",
    "is_future": "bd541332240f592301f7e720c94c416fea54a2ebd26ee22cf46c4dca06c4bdb0",
    "g": "e74d8d2a53ee7a978fe94b066510147be374f197504cff24e02e24cb00478e5c",
    "contributions_raw": "d04b8e617949c1e112c31cf114c2c523dc333aac9dd079d3e7606d2cf647393a",
    "baseline_raw": "70c9efa8f8638132aecc04ac5b5884d259581d3e85382140a324a78bb036bb05",
    "demand": "6e35699b0076b745ca7211ccb0ec0031f77e8c0733692cd62af46756408cba71",
    "spend_means": "54ed564fbe95f8940e354d81d2897fe33878b96ca9bab869206905e4bdaaf270",
    "sales_scale": "a676e6e3b3a867ccc2da124fc035d82051d5e3548efb23aab281e65aa9699924",
    "is_val": "3ba2b8d7203a2ac328478a2be7cb0e3ea6e28af5e648841fbf1a41fbbb2e09f7",
    "cell_id": "85bf0249350a8edb436598ba927c2298bca0e058665eb864b071e65830562096",
    "active_c_mask": "a9affb5f52630150ac24b5fb37a2b9e88bc54312f549319752cced7601ebeac1",
    "active_m_mask": "9f40977b73dea1782a868d23dfccb7b0264d20bb07b3e07ad73de81dc5797e91",
    "active_j_mask": "06afea6ca346b374357ad90556c91c315cc6d429ada4eb2a1900fc30a9fdaff7",
    "K_active": "a9073e6b59ee724764dc6cafa783aee1aae5fda1329f86b3b634578b0c8687f1",
    "M_active": "9aed02a9be4f658a038cf3eb48d2effffb9d8e44125a6afe60bdb69cee9cf1b8",
    "J_active": "a9bb0c6117f6fc1cb4a0e889122a322c8c515ce1b82ed2564d127109bb066713",
    "indirect_effects": "8b6bd08524473069fc4dcf6c7840c0e219424e0b6c84a6f0ec09e5124213d346",
    "channel_active": "e6a619d2852bed3ed0f8890b700bec0062e3f6f7352eec5a769792bd8c191671",
    "control_contribution": "3fc589c3a0210236442dce0e0da1b5821f451763a2715a281aacc9cf79458904",
    "confounder_contribution": "9aa4453b1646906ac2bccdeebb732632c3f4ab800246e3c93c8ff19f9e3d4ddb",
    "baseline_intrinsic": "7d35cdd3938daa5077de631364109a3f2df7440d6a2f1e46512a94a2a7c3cd7d",
    "indirect_effects_by_source": "db4eff94a8b16ce084c0c9080e1978e2e07335968a1392aabd4e82986209b254",
    "confounding_strength": "1ebfc5942a66b91e14dd2667a96be1e65275c531c1f4de99471e8982f6367421",
    "channel_shock_mask": "a2c17b7c3b20dff7ef9b8ef7316a3ef323f705522c3fef1744df2a33a3af832b",
    "channel_shock_channel": "7f956e232d961d634c20094542cf11ae23525f62749f31010e76d0cbcacc7e82",
    "channel_shock_start": "52d81fd29e1721a2b63114638580692d9faf97a1ef36eb6d5129adf758cd1f35",
    "channel_shock_length": "fc492e4b1613b6e1cf17271739cd12251266644e410cb1c14180e21c6b07c59e",
    "channel_shock_level_multiplier": "ab57f647547011d917a044e853a1d5796e715c0ccacb505d2c51442e4da8c341",
    "channel_shock_level": "1550bbf2bded16bb609b5f52669f0b29e6fdf0d7a0898bfb82c5c51a9db3124a",
    "channel_level": "22dbdf23eea79026eab5eff302fc06407a8cff7f7cfd391f11012b1cbf72646c",
    "saturation_scale": "67151595c5d2e7e02020b87cd1fc655dd640e531eb782cc8c46dd015c4b13a99",
    "adstock_family": "232a4fa82e296516ddeca77e254d8ff5a83d73c3ae89f8a3fe4477a4b6344380",
    "adstock_alpha": "ec14426e030159bdd87ce9eb96c954ce3f226149923e6a2b77c8739964131bf8",
    "weibull_lam": "e9d430f3f8679b6fe2569cb5a09825214231d0050db69fc91704c60d4faf30d7",
    "weibull_k": "0a48c8e646f1b4f9dc70f0e0e03be9b59503b19db4c71a2a5d44f869efcf8c44",
    "signal_metrics": "65967ac869a5da05d4eb29f7cb2195bb38d859bd6e5062d77840db33e76d82b3",
    "signal_metric_valid": "f1ecbab717272a25f5802cb87be6fd3b1bdd88cede7e75a141f74ebacf212b16",
}

EXPECTED_DEFAULT_FREE_RVS = (
    "rw_z_mean",
    "rw_z_std",
    "rw_c_mean",
    "rw_c_std",
    "rw_b_mean",
    "rw_b_std",
    "rw_y_std",
    "pulse_prob",
    "w_dc",
    "u_dz",
    "v_zc",
    "alpha_cc",
    "gamma_zz",
    "delta_db",
    "rho_zb",
    "beta",
    "adstock_alpha",
    "weibull_lam",
    "weibull_k",
    "hill_slope",
    "hill_kappa_mult",
    "logistic_lam",
    "mm_kappa_mult",
    "tanh_c",
    "root_alpha",
    "hf_sigma",
    "pulse_amp",
    "eps_d",
    "eps_z",
    "eps_c",
    "eps_b",
    "eps_y",
    "eps_c_hf",
    "eps_c_pulse",
)


def _hash_array(key: str, value: np.ndarray) -> str:
    hasher = hashlib.sha256()
    hasher.update(key.encode())
    hasher.update(str(value.dtype).encode())
    hasher.update(str(value.shape).encode())
    hasher.update(np.ascontiguousarray(value).tobytes(order="C"))
    return hasher.hexdigest()


def test_additive_corpus_matches_legacy_hashes():
    """Disabled/default features must preserve every serialized array byte-for-byte."""
    cfg = pg.make_scm_prior(
        n_treatments=2,
        n_covariates=2,
        n_latent=1,
        T=24,
        n_cells=2,
        draws_per_cell=1,
        seed=17,
        edge_budget={
            "cy": (2, 2),
            "dc": (2, 2),
            "dz": (2, 2),
            "db": (1, 1),
            "zb": (2, 2),
            "zc": (4, 4),
            "cc": (1, 1),
            "zz": (1, 1),
        },
    )
    corpus = pg.sample_prior_predictive(cfg)

    # Diagnostics include elapsed timing, so only the serialized top-level arrays
    # participate in this deterministic legacy contract.
    hashes = {
        key: _hash_array(key, value)
        for key, value in corpus.items()
        if isinstance(value, np.ndarray)
    }
    hashes.update(
        {key: _hash_array(key, value) for key, value in corpus["identifiability"].items()}
    )

    assert hashes == EXPECTED_CORPUS_HASHES


def test_default_world_model_free_rvs_match_legacy_order():
    """The default path must not register or consume extra random variables."""
    cfg = pg.make_scm_prior(n_treatments=2, n_covariates=2, n_latent=1, T=24)
    rng = np.random.default_rng(23)
    g = sample_g_additive(rng, cfg, cfg.layout, K_active=2, M_active=2, J_active=1)
    g_active = _slice_g_active(g, 2, 2, 1)
    structural = sample_structure(g_active, cfg, rng)
    model, _out_names, _param_names = build_world_model(g_active, cfg, structural, cfg.T)

    assert tuple(rv.name for rv in model.free_RVs) == EXPECTED_DEFAULT_FREE_RVS


def test_identifiability_labels_are_optional_metadata_not_features():
    cfg = pg.make_scm_prior(
        n_treatments=2,
        n_covariates=2,
        n_latent=1,
        T=16,
        n_cells=2,
        draws_per_cell=1,
        seed=47,
        confounding_strength_range=(0.6, 0.6),
        n_channel_shocks=1,
        channel_shock_length_range=(2, 2),
        channel_shock_level_range=(0.0, 0.5),
    )
    labelled = pg.sample_prior_predictive(cfg)
    feature_only = pg.sample_prior_predictive(replace(cfg, include_identifiability_labels=False))
    assert set(labelled["identifiability"]) == {"signal_metrics", "signal_metric_valid"}
    assert "identifiability" not in feature_only
    for key, value in feature_only.items():
        if isinstance(value, np.ndarray):
            assert np.array_equal(value, labelled[key]), key
    assert feature_only["diagnostics"]["signal"] == labelled["diagnostics"]["signal"]
    assert pg.DataGenerator.validate_corpus(feature_only) == []


def test_metadata_first_draw_preserves_legacy_rng_order():
    cfg = pg.make_scm_prior(n_treatments=2, n_covariates=2, n_latent=1, T=24)
    rng = np.random.default_rng(23)
    g = sample_g_additive(rng, cfg, cfg.layout, K_active=2, M_active=2, J_active=1)
    g_active = _slice_g_active(g, 2, 2, 1)
    model, _out_names, _param_names = build_world_model(
        g_active, cfg, sample_structure(g_active, cfg, rng), cfg.T
    )

    legacy = draw_worlds(model, _ADDITIVE_OUT_NAMES, seed=29, draws=3)
    combined = draw_worlds(
        model,
        _CORPUS_PARAM_NAMES + _CORPUS_SHOCK_NAMES + _ADDITIVE_OUT_NAMES,
        seed=29,
        draws=3,
    )

    for name in _ADDITIVE_OUT_NAMES:
        assert np.array_equal(combined[name], legacy[name]), name


def test_baseline_walk_sigma_default_tracks_shared_sigma_after_replace():
    """The default baseline scale remains dynamic rather than being copied at construction."""
    cfg = pg.SCMPrior(rw_std_sigma=0.8)
    assert cfg.rw_baseline_std_sigma is None
    assert cfg.rw_baseline_std_sigma_effective == 0.8
    assert replace(cfg, rw_std_sigma=0.35).rw_baseline_std_sigma_effective == 0.35
    assert replace(cfg, rw_baseline_std_sigma=0.6).rw_baseline_std_sigma_effective == 0.6


@pytest.mark.parametrize("sigma", (0.0, -0.1, np.nan, np.inf, -np.inf))
def test_baseline_walk_sigma_override_must_be_finite_and_positive(sigma):
    with pytest.raises(ValueError, match="rw_baseline_std_sigma must be finite and > 0"):
        pg.SCMPrior(rw_baseline_std_sigma=sigma).validate()


def test_confounding_strength_monotonically_increases_dense_r2_without_flattening():
    """The calibrated rho fixture moves label confounding without weakening targets."""
    model_seed = 20260725
    draw_seed = 8675309
    g = {
        "g_cy": np.ones(2, dtype=int),
        "g_dc": np.zeros((1, 2), dtype=int),
        "g_dz": np.zeros((1, 1), dtype=int),
        "g_db": np.zeros(1, dtype=int),
        "g_zb": np.zeros(1, dtype=int),
        "g_zc": np.zeros((1, 2), dtype=int),
        "g_cc": np.zeros((2, 2), dtype=int),
        "g_zz": np.zeros((1, 1), dtype=int),
    }
    fixture = {
        "T": 72,
        "n_treatments": 2,
        "n_covariates": 1,
        "n_latent": 1,
        "nonlinearity": "linear",
        "adstock_family_probs": {"none": 1.0, "geometric": 0.0, "weibull": 0.0},
        "saturation_family_probs": {
            "linear": 1.0,
            "hill": 0.0,
            "logistic": 0.0,
            "michaelis_menten": 0.0,
            "tanh": 0.0,
            "root": 0.0,
        },
        "beta_additive_range": (1.0, 1.0),
        "rw_channel_std_range": (0.25, 0.25),
        "rw_baseline_std_sigma": 0.75,
        "rw_sales_std_sigma": 1e-4,
        "channel_hf_sigma_range": (0.08, 0.08),
        "channel_pulse_prob_range": (0.0, 0.0),
        "channel_pulse_amp_range": (0.0, 0.0),
        "rw_mean_range": (3.0, 3.0),
        "rw_positive_mean_range": (3.0, 3.0),
        "rw_baseline_mean_range": (3.0, 3.0),
        "adstock_burn_in": 0,
    }
    medians = []
    target_cv_medians = []
    target_std_medians = []
    free_rv_orders = []
    r2_index = SIGNAL_METRIC_LAYOUT.index("contrib_r2_explained_by_rest")
    cv_index = SIGNAL_METRIC_LAYOUT.index("contrib_cv")

    # rho enters the mixture as sqrt(1 - rho**2) * eps_c + rho * eps_b, so R2's
    # response to rho is convex: a 0.45 midpoint still leaves 89% of the channel
    # innovation independent and lands within noise of rho = 0 (measured first
    # step 0.07-0.14 across draw seeds). 0.6 separates the three levels well
    # clear of the 0.10 margin asserted below (measured 0.15-0.21).
    for rho in (0.0, 0.6, 0.9):
        cfg = pg.make_scm_prior(
            **fixture,
            confounding_strength_range=(rho, rho),
        )
        structural = sample_structure(g, cfg, np.random.default_rng(model_seed))
        structural["smoothness_c"][:] = 0.0
        structural["smoothness_b"][:] = 0.0
        model, out_names, _ = build_world_model(g, cfg, structural, cfg.T)
        drawn = draw_worlds(model, out_names, seed=draw_seed, draws=48)
        metrics, valid = dense_signal_metrics(
            drawn["channels"],
            drawn["contributions"],
            drawn["sales"],
            drawn["baseline"],
            np.ones((48, 2), dtype=bool),
            sales_scale=np.ones(48),
            adstock_family=np.zeros((48, 2), dtype=np.uint8),
            adstock_alpha=np.zeros((48, 2)),
            weibull_lam=np.zeros((48, 2)),
            weibull_k=np.zeros((48, 2)),
            l_max=cfg.l_max,
            adstock_burn_in=cfg.adstock_burn_in,
        )
        r2 = metrics[..., r2_index][valid[..., r2_index].astype(bool)]
        target_cv = metrics[..., cv_index][valid[..., cv_index].astype(bool)]
        medians.append(float(np.median(r2)))
        target_cv_medians.append(float(np.median(target_cv)))
        target_std = drawn["contributions"].std(axis=1).ravel()
        target_std_medians.append(float(np.median(target_std)))
        free_rv_orders.append(tuple(rv.name for rv in model.free_RVs))

    # The realized walk amplitude is now random (the walk is normalized by a
    # constant, not by its own realized sd). At these pinned seeds, the
    # per-draw minima are 0.106 (CV) and 0.107 (std), but those are tail
    # statistics. The median-ratio assertions below carry the "targets do not
    # flatten" property.
    assert np.diff(medians).min() > 0.10
    assert medians[-1] - medians[0] > 0.40
    assert target_cv_medians[-1] >= 0.90 * target_cv_medians[0]
    assert target_std_medians[-1] >= 0.90 * target_std_medians[0]
    assert free_rv_orders[0] == free_rv_orders[1] == free_rv_orders[2]
    assert all("confounding_strength" not in order for order in free_rv_orders)


@pytest.mark.slow
def test_supported_corpus_signal_gate():
    """The supported corpus clears the calibrated identifiability signal gate."""
    corpus = pg.sample_prior_predictive(
        pg.make_scm_prior(
            n_treatments=6,
            n_covariates=4,
            n_latent=2,
            T=104,
            n_cells=4,
            draws_per_cell=6,
            seed=314159,
        )
    )
    signal = corpus["diagnostics"]["signal"]
    passed, _ = check_signal_gate(signal)

    assert passed
    assert signal["frac_spearman_lt_03"] <= 0.15
    assert signal["frac_warmup_gt_3"] <= 0.10
