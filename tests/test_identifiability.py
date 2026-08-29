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
    summarize_signal_metrics,
)
from prior_generator.world_model import build_world_model, draw_worlds, sample_structure

#: Byte-level contract for every serialized corpus array. When a hash moves,
#: record WHY here — a silent regeneration is how a real regression hides.
#:
#: Last moved by CONTROL TEXTURE: ``make_scm_prior(texture="diverse")`` now
#: gives controls high-frequency own drive (iid weekly noise + a centred
#: calendar pulse, both relative to the control's own walk std), so a control
#: is no longer a smooth walk drawn from the same function space as the smooth
#: baseline walk. The new draws re-partition the seeded RNG list (``reseed_rngs``
#: assigns streams by the compiled graph's traversal order), so EVERY
#: non-structural array moves here, including ones with no control ancestor
#: (``demand``, ``adstock_alpha``, ``weibull_*``). Unchanged: the
#: structure/mask/split arrays, ``adstock_family``, and ``channel_level``.
#: ``EXPECTED_UNTEXTURED_CONTROL_HASHES`` below pins the pre-texture bytes.
EXPECTED_CORPUS_HASHES = {
    "spend_raw": "50ee92c932a13ba4db21c7e60db0acf1e082458a80a677c1c9db0b79ae56b680",
    "spend_norm": "c6d2f305497aa46f7957e826512dec5af595a1a8e4d59d8b2a883d752ac5b152",
    "spend_share": "974ec3a99ab5305b390ff67108b9f9ba0736aea41384685044e70c33bd211f59",
    "controls": "2b1e89f1dfcb06918121cfacb1f445ec2a20f56a83a51c76092a7fd9bb03acb5",
    "sales_raw": "931690c12039de059c3ef16406fb84edef0944168f11a203480b53211e054c23",
    "sales_norm": "2487da7fc2420256fa40dd3dcee2a83cd0fbaa1d9f395aacf0baddd9134a16f9",
    "support_mask": "003f9ad7fd18ab65a0ca13820be76c5f1637b7e15c7d94fa1b5e327f148c7508",
    "is_future": "7801dc551deac4014f0121c44146e6565844cf603d6ad9e3e61ad1007743afe9",
    "g": "e74d8d2a53ee7a978fe94b066510147be374f197504cff24e02e24cb00478e5c",
    "contributions_raw": "0cdc73b55bbbbd539efb578356327ee81d24ce99856052ed9720a6b316319aec",
    "baseline_raw": "b79c215f68878bb2ae77c01802bd555b4d9e9a91fd92c939a356bd3fe1ea3e24",
    "demand": "383c26cf53401e8b4637cefa3c5de504e1cc226d6cf9d72b5c838b57217cc2fd",
    "spend_means": "6e0cf461894d812b53b70f69a76769f61859eaa7f64d54c2c896dcd5a2c709ff",
    "sales_scale": "78cb8fcc79b3428bc12f439759f32d5849b82bab075d4f23c1ec4d15a439d655",
    "is_val": "eb4b02f10d5660691db02cfd97e0fee99ab9f2461e8e802369f07300b5285ad4",
    "cell_id": "85bf0249350a8edb436598ba927c2298bca0e058665eb864b071e65830562096",
    "treatment_active_mask": "3a7cd171c96fc8fb27a1dadbd085253acf5fb34ff5976402507504ddadd92889",
    "covariate_active_mask": "28a1a7f96dc3b37a56bc3bed913b9f7d3433e765ca26ec6adaedaa272b87baa2",
    "latent_active_mask": "c5d57bca6001a56485f5bb07b383f17e38716d894f10a46c648521ce1d7a677f",
    "n_treatments_active": "c1198a3bdca08a0ae0112be2f753ffc046cece6c10a96c3b4edd974798b21530",
    "n_covariates_active": "9ae6b0b1e6912e108c443a1e35a8fd522c1e0455c79b1152d8d44744d080504f",
    "n_latent_active": "33462003b0a367a219d05faffcae2fa50ef2f8bdac6cb881e694c7c46335f46e",
    "indirect_effects": "3d20abcfbd838bdb23b2888d8542794d35eb1d2da9f4f6893b4ffcbc53e03847",
    "channel_active": "e6a619d2852bed3ed0f8890b700bec0062e3f6f7352eec5a769792bd8c191671",
    "control_contribution": "a8ea808e9304f892e7bd43616a9a9f1f879c18dfdd370b7f1a47a7b5babf321c",
    "confounder_contribution": "ac7d0ad77be2a513bdb2ec46ea024d8c7ee7d2d63961f38ed109c495be0766f2",
    "baseline_intrinsic": "731555b9dd9953eab9288079850faae6a6d4dac2a93dc2c244c45660b1535ef3",
    "indirect_effects_by_source": "773c263e398cd6aaafc5f4b3b741deffd0ac9f5f78f8d75562b00219dd5fe5ad",
    "confounding_strength": "1ebfc5942a66b91e14dd2667a96be1e65275c531c1f4de99471e8982f6367421",
    "channel_shock_mask": "a2c17b7c3b20dff7ef9b8ef7316a3ef323f705522c3fef1744df2a33a3af832b",
    "channel_shock_channel": "7f956e232d961d634c20094542cf11ae23525f62749f31010e76d0cbcacc7e82",
    "channel_shock_start": "52d81fd29e1721a2b63114638580692d9faf97a1ef36eb6d5129adf758cd1f35",
    "channel_shock_length": "fc492e4b1613b6e1cf17271739cd12251266644e410cb1c14180e21c6b07c59e",
    "channel_shock_level_multiplier": "ab57f647547011d917a044e853a1d5796e715c0ccacb505d2c51442e4da8c341",
    "channel_shock_level": "1550bbf2bded16bb609b5f52669f0b29e6fdf0d7a0898bfb82c5c51a9db3124a",
    "channel_level": "dc4751c12f53123084e1d827851c8fbedf4d214f0513f4c046852697889e3cb4",
    "saturation_scale": "3aaa9bb34f12c5a9844bfd51112158912e787f2c77480c062098622e25c6fad4",
    "adstock_family": "b8b595535043fbd9bee4855aa2adc923c2a7cf6c5d91fd9b538f1779223d66c8",
    "adstock_alpha": "47c6002f084de409224518f5f5f2dcd769065bc800e492b346309f27a7c08df2",
    "weibull_lam": "64ee54fc37b421f641f6da185965522fb12996987717d4d76f552002be4c2228",
    "weibull_k": "607ee7504d909286e200ac704d8ac4715cc97635ef8fd6cb226733bda0430e13",
    "signal_metrics": "e881193b08a34921a9cd742bd2c69e5ab95b49016da20fe8836f67988048f1be",
    "signal_metric_valid": "f1ecbab717272a25f5802cb87be6fd3b1bdd88cede7e75a141f74ebacf212b16",
}

#: The SAME corpus with the control texture explicitly disabled, pinned to the
#: hashes generated BEFORE the mechanism existed. This is the RNG-isolation
#: contract: the new parameters degenerate to constants and the new noise RVs
#: reach no output, so they collect no random stream and every legacy byte
#: survives. It moves only when something outside the control texture changes.
EXPECTED_UNTEXTURED_CONTROL_HASHES = {
    "spend_raw": "c0bab46c1873c91e56d05fb383b1eef66e02002ffa080de0ca814a6b7b6d4adf",
    "spend_norm": "e0f991be830a85ca0ab20917d1e0bf60848ae167426c98835f07bff7f88cb8ef",
    "spend_share": "5b58d30f1281b1101132ff820e33f91a3158ee41fd63737941aa731c03ac1254",
    "controls": "2fc7043959d83b4475c71dfddbe1a297a6b8a4226cc74c9dbc8335bee40f0ad5",
    "sales_raw": "d31551fcb8b4a444e6ebd44eaea81c90bd0e965b94555880bf544bb56f9fb6f6",
    "sales_norm": "43bff67e36ae5cc305d2dab3264790c018f311eeef80e5f11e15e2464ff16260",
    "support_mask": "003f9ad7fd18ab65a0ca13820be76c5f1637b7e15c7d94fa1b5e327f148c7508",
    "is_future": "7801dc551deac4014f0121c44146e6565844cf603d6ad9e3e61ad1007743afe9",
    "g": "e74d8d2a53ee7a978fe94b066510147be374f197504cff24e02e24cb00478e5c",
    "contributions_raw": "9d0ae9fff70ee95831fedb5e2c441d2b9fca0a0d9e43beef94753371e344b55f",
    "baseline_raw": "9b35e472341b7f7355f058529572b8de2ccbdbc8321ee3dfe4fb45c925584030",
    "demand": "ddd9004bf39dd0e790cf550ea99ebac2454868815bc6591f9ee321326a1f758e",
    "spend_means": "8f0598524beaec26652a7809811b44a587420f0ea11ca4130ff04461d267dcdc",
    "sales_scale": "6e611bd5cd659ec8f6959dd5e759c30ef0c3402ec8d414d01b740c3138696108",
    "is_val": "eb4b02f10d5660691db02cfd97e0fee99ab9f2461e8e802369f07300b5285ad4",
    "cell_id": "85bf0249350a8edb436598ba927c2298bca0e058665eb864b071e65830562096",
    "treatment_active_mask": "3a7cd171c96fc8fb27a1dadbd085253acf5fb34ff5976402507504ddadd92889",
    "covariate_active_mask": "28a1a7f96dc3b37a56bc3bed913b9f7d3433e765ca26ec6adaedaa272b87baa2",
    "latent_active_mask": "c5d57bca6001a56485f5bb07b383f17e38716d894f10a46c648521ce1d7a677f",
    "n_treatments_active": "c1198a3bdca08a0ae0112be2f753ffc046cece6c10a96c3b4edd974798b21530",
    "n_covariates_active": "9ae6b0b1e6912e108c443a1e35a8fd522c1e0455c79b1152d8d44744d080504f",
    "n_latent_active": "33462003b0a367a219d05faffcae2fa50ef2f8bdac6cb881e694c7c46335f46e",
    "indirect_effects": "7f79fb8eb08e2c904dea27c78f8a24d3f181c25cdd0dac0d2e166265a0c968e9",
    "channel_active": "e6a619d2852bed3ed0f8890b700bec0062e3f6f7352eec5a769792bd8c191671",
    "control_contribution": "27c49ad126d6a7e67033d5fd17785f66e68a0f0141c5bc5104c9c74ed5b3a55b",
    "confounder_contribution": "a87dee854c1b45c48847c4d7508b2033ce955fa799f94b702de1f503cd849ca7",
    "baseline_intrinsic": "36aefc7f3c797f2eabe023525b324de46475b8a0e7e4e1eede7b267720eb1725",
    "indirect_effects_by_source": "0b8fde4981e3422bfccc1524db3e4e962316a8e7fb778f5c88882bbf84744309",
    "confounding_strength": "1ebfc5942a66b91e14dd2667a96be1e65275c531c1f4de99471e8982f6367421",
    "channel_shock_mask": "a2c17b7c3b20dff7ef9b8ef7316a3ef323f705522c3fef1744df2a33a3af832b",
    "channel_shock_channel": "7f956e232d961d634c20094542cf11ae23525f62749f31010e76d0cbcacc7e82",
    "channel_shock_start": "52d81fd29e1721a2b63114638580692d9faf97a1ef36eb6d5129adf758cd1f35",
    "channel_shock_length": "fc492e4b1613b6e1cf17271739cd12251266644e410cb1c14180e21c6b07c59e",
    "channel_shock_level_multiplier": "ab57f647547011d917a044e853a1d5796e715c0ccacb505d2c51442e4da8c341",
    "channel_shock_level": "1550bbf2bded16bb609b5f52669f0b29e6fdf0d7a0898bfb82c5c51a9db3124a",
    "channel_level": "dc4751c12f53123084e1d827851c8fbedf4d214f0513f4c046852697889e3cb4",
    "saturation_scale": "70287d636760848f2f81e503695a23449e0c4611601ef4863eba74ec3b4ba155",
    "adstock_family": "b8b595535043fbd9bee4855aa2adc923c2a7cf6c5d91fd9b538f1779223d66c8",
    "adstock_alpha": "7f6a515b0ec2e49cff4ef09be52324f3b4209b81888f3b9f0231b71c2ae07ec2",
    "weibull_lam": "24767f169da9b06c638e12aedbcf929e0cb7e8515fdf032ac8b0604796d4bf08",
    "weibull_k": "d799a934d9ac7e843eb624fef3ca42e8b48e0cef0dd2987a838de9cd6bb4ce7a",
    "signal_metrics": "97283a4f1011aca9a6306edae61093907f7481574d375b0d55fbb08164373d8e",
    "signal_metric_valid": "f1ecbab717272a25f5802cb87be6fd3b1bdd88cede7e75a141f74ebacf212b16",
}

EXPECTED_DEFAULT_FREE_RVS = (
    "rw_z_mean",
    "rw_z_std",
    "rw_c_mean",
    "rw_c_std",
    "rw_b_mean",
    "rw_b_std_rel",
    "rw_y_std_rel",
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
    # Control texture. These are creation-order pins, not stream positions:
    # reseed_rngs walks the compiled graph's traversal order. A disabled config
    # keeps its bytes because the magnitudes degenerate to constants and the two
    # noise RVs reach no output (see the byte contract below).
    "control_hf_sigma",
    "control_pulse_amp",
    "control_pulse_prob",
    "eps_d",
    "eps_z",
    "eps_c",
    "eps_b",
    "eps_y",
    "eps_c_hf",
    "eps_c_pulse",
    "eps_z_hf",
    "eps_z_pulse",
)


def _hash_array(key: str, value: np.ndarray) -> str:
    hasher = hashlib.sha256()
    hasher.update(key.encode())
    hasher.update(str(value.dtype).encode())
    hasher.update(str(value.shape).encode())
    hasher.update(np.ascontiguousarray(value).tobytes(order="C"))
    return hasher.hexdigest()


def _corpus_hashes(**overrides) -> dict[str, str]:
    cfg = pg.make_scm_prior(
        n_treatments=2,
        n_covariates=2,
        n_latent=1,
        n_time_steps=24,
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
        **overrides,
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
    return hashes


def test_additive_corpus_matches_default_outcome_noise_hashes():
    """Default relative outcome-noise corpora retain an explicit byte contract."""
    assert _corpus_hashes() == EXPECTED_CORPUS_HASHES


def test_disabling_control_texture_reproduces_the_pre_texture_corpus_bytes():
    """Control texture must be RNG-inert when switched off.

    Its parameters degenerate to constants and its noise RVs reach no output,
    so they collect no random stream. Any drift here means a new draw slipped
    into an existing stream and silently changed every downstream world.
    """
    hashes = _corpus_hashes(
        control_hf_sigma_range=(0.0, 0.0),
        control_pulse_prob_range=(0.0, 0.0),
        control_pulse_amp_range=(0.0, 0.0),
    )
    assert hashes == EXPECTED_UNTEXTURED_CONTROL_HASHES


def test_default_world_model_free_rvs_match_outcome_noise_contract():
    """Relative mode registers dimensionless outcome scales before ``beta``."""
    cfg = pg.make_scm_prior(n_treatments=2, n_covariates=2, n_latent=1, n_time_steps=24)
    rng = np.random.default_rng(23)
    g = sample_g_additive(
        rng, cfg, cfg.layout, n_treatments_active=2, n_covariates_active=2, n_latent_active=1
    )
    g_active = _slice_g_active(g, 2, 2, 1)
    structural = sample_structure(g_active, cfg, rng)
    model, _out_names, _param_names = build_world_model(g_active, cfg, structural, cfg.n_time_steps)

    assert tuple(rv.name for rv in model.free_RVs) == EXPECTED_DEFAULT_FREE_RVS


def test_identifiability_labels_are_optional_metadata_not_features():
    cfg = pg.make_scm_prior(
        n_treatments=2,
        n_covariates=2,
        n_latent=1,
        n_time_steps=16,
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
    cfg = pg.make_scm_prior(n_treatments=2, n_covariates=2, n_latent=1, n_time_steps=24)
    rng = np.random.default_rng(23)
    g = sample_g_additive(
        rng, cfg, cfg.layout, n_treatments_active=2, n_covariates_active=2, n_latent_active=1
    )
    g_active = _slice_g_active(g, 2, 2, 1)
    model, _out_names, _param_names = build_world_model(
        g_active, cfg, sample_structure(g_active, cfg, rng), cfg.n_time_steps
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
        "n_time_steps": 72,
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
        "outcome_std_mode": "absolute",
        "channel_hf_sigma_range": (0.08, 0.08),
        "channel_pulse_prob_range": (0.0, 0.0),
        "channel_pulse_amp_range": (0.0, 0.0),
        # This fixture isolates rho, so every other texture axis is pinned —
        # the controls here are edge-free anyway, and leaving their texture on
        # would only move the calibrated constants below.
        "control_hf_sigma_range": (0.0, 0.0),
        "control_pulse_prob_range": (0.0, 0.0),
        "control_pulse_amp_range": (0.0, 0.0),
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
        model, out_names, _ = build_world_model(g, cfg, structural, cfg.n_time_steps)
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
    """The supported prior clears the calibrated identifiability signal gate."""
    cfg = pg.make_scm_prior(
        n_treatments=6,
        n_covariates=4,
        n_latent=2,
        n_time_steps=104,
        n_cells=4,
        draws_per_cell=6,
        seed=314159,
    )
    seed_pool = (314159, 1, 2, 3, 4, 5, 6, 7)
    corpora = [pg.sample_prior_predictive(replace(cfg, seed=seed)) for seed in seed_pool]
    signal = summarize_signal_metrics(
        np.concatenate([corpus["identifiability"]["signal_metrics"] for corpus in corpora]),
        np.concatenate([corpus["identifiability"]["signal_metric_valid"] for corpus in corpora]),
        np.concatenate([corpus["sales_raw"] for corpus in corpora]),
        np.concatenate(
            [
                (corpus["g"][:, cfg.layout.slices["cy"]] == 1)
                & (corpus["treatment_active_mask"] == 1)
                for corpus in corpora
            ]
        ),
        l_max=cfg.l_max,
        adstock_burn_in=cfg.adstock_burn_in,
        sales_scale=np.concatenate([corpus["sales_scale"] for corpus in corpora]),
        adstock_family=np.concatenate([corpus["adstock_family"] for corpus in corpora]),
        adstock_alpha=np.concatenate([corpus["adstock_alpha"] for corpus in corpora]),
        weibull_lam=np.concatenate([corpus["weibull_lam"] for corpus in corpora]),
        weibull_k=np.concatenate([corpus["weibull_k"] for corpus in corpora]),
    )
    passed, _ = check_signal_gate(signal)

    # Gated values are Bernoulli fractions over direct channels. At the old
    # 100-channel size, their population means sat about one sigma below the
    # thresholds, so one draw failed about a quarter of the time (6/8 passed).
    # Pooling eight independent corpora yields roughly 800 direct channels and
    # measures the prior's gate compliance rather than one noisy realization.
    assert passed
    assert signal["frac_spearman_lt_03"] <= 0.15
    assert signal["frac_warmup_gt_3"] <= 0.10
