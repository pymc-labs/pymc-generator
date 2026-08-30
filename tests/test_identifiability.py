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
#: Last moved by the INTERCEPT SPLIT: ``B`` is now the intercept alone (D and Z
#: attach directly to ``Y``), ``baseline_intrinsic`` reports that intercept
#: WITHOUT the iid sales noise, and the noise became its own ``sales_noise``
#: column. Dropping ``RW_Y`` from ``baseline_intrinsic`` means ``eps_y`` is
#: first reached later in the graph, and ``reseed_rngs`` assigns streams by the
#: compiled graph's traversal order, so every non-structural array moves.
#: Unchanged: the structure/mask/split arrays (``g``, ``*_active_mask``,
#: ``support_mask``, ``is_val``, ``cell_id``) and ``adstock_family``.
EXPECTED_CORPUS_HASHES = {
    "spend_raw": "a28507d6843679e98d015a67d0da943c4d76df5991b82540f470045a2ed048f2",
    "spend_norm": "f45cf176672584e285706eff6563747e3fa4e802f3555aa9d9b5d1899a3d195a",
    "spend_share": "876d8f409dc9393189973f3b04125083c979be2c09f322fd07635b7c048cd603",
    "controls": "19289710c49ee30b61b6cd40cc330e98ffdc77382964edfe927ec9a976fb1962",
    "sales_raw": "e8b8f3d5b72d3f9d48096863b706cee76d0fe55c400abf586e3cd07492da2ec1",
    "sales_norm": "3b5718f8eab1e3d02f73908e7e4019760f4904b44424321b6f5b9dee21305f17",
    "support_mask": "003f9ad7fd18ab65a0ca13820be76c5f1637b7e15c7d94fa1b5e327f148c7508",
    "is_future": "7801dc551deac4014f0121c44146e6565844cf603d6ad9e3e61ad1007743afe9",
    "g": "e74d8d2a53ee7a978fe94b066510147be374f197504cff24e02e24cb00478e5c",
    "contributions_raw": "a68d569b571e38285f78c26278ce109041c676f53bfc531f541bbb5e6c6a8c57",
    "baseline_raw": "ffb5ee541626f221f37eb240320763981369bc697dadd2fd898435f18b0611aa",
    "demand": "9b885bdb0f6878ee4a5cd4f29a8a2226b583df76c4d8bafce4ef3277a07f5b3c",
    "spend_means": "0e2a1c38805785b53eba7026a2cd6c034dc7312e325db0d7bd74da985cf541da",
    "sales_scale": "d2c04c6028e121e67a1beb72e9c4aed1dbdd4b49fa112a5942000fe9c57caf50",
    "is_val": "eb4b02f10d5660691db02cfd97e0fee99ab9f2461e8e802369f07300b5285ad4",
    "cell_id": "85bf0249350a8edb436598ba927c2298bca0e058665eb864b071e65830562096",
    "treatment_active_mask": "3a7cd171c96fc8fb27a1dadbd085253acf5fb34ff5976402507504ddadd92889",
    "covariate_active_mask": "28a1a7f96dc3b37a56bc3bed913b9f7d3433e765ca26ec6adaedaa272b87baa2",
    "latent_active_mask": "c5d57bca6001a56485f5bb07b383f17e38716d894f10a46c648521ce1d7a677f",
    "n_treatments_active": "c1198a3bdca08a0ae0112be2f753ffc046cece6c10a96c3b4edd974798b21530",
    "n_covariates_active": "9ae6b0b1e6912e108c443a1e35a8fd522c1e0455c79b1152d8d44744d080504f",
    "n_latent_active": "33462003b0a367a219d05faffcae2fa50ef2f8bdac6cb881e694c7c46335f46e",
    "indirect_effects": "24f1fed8da9de9ee18e84132be4d6a772da2ca9eeb5af09c2b985e31b4a16c2e",
    "channel_active": "e6a619d2852bed3ed0f8890b700bec0062e3f6f7352eec5a769792bd8c191671",
    "control_contribution": "0368e70dc2d1e4bb90c022c17945860e672a7b7b938281f09f7d17c2d3a0d8a5",
    "confounder_contribution": "c053f2fd79ae81912551bbc0d1e71a91abf2de23671465cb0d31eb51199241d6",
    "baseline_intrinsic": "9034c76ba9be5ed3cd6fc2a3f47d52db69d092dd758c0ef0fa018df556fe4540",
    "indirect_effects_by_source": "ad69f3ec5fcdf3e26012ddd75af788005c7840c56899ee986c06d3584ab13a10",
    "confounding_strength": "1ebfc5942a66b91e14dd2667a96be1e65275c531c1f4de99471e8982f6367421",
    "channel_shock_mask": "a2c17b7c3b20dff7ef9b8ef7316a3ef323f705522c3fef1744df2a33a3af832b",
    "channel_shock_channel": "7f956e232d961d634c20094542cf11ae23525f62749f31010e76d0cbcacc7e82",
    "channel_shock_start": "52d81fd29e1721a2b63114638580692d9faf97a1ef36eb6d5129adf758cd1f35",
    "channel_shock_length": "fc492e4b1613b6e1cf17271739cd12251266644e410cb1c14180e21c6b07c59e",
    "channel_shock_level_multiplier": "ab57f647547011d917a044e853a1d5796e715c0ccacb505d2c51442e4da8c341",
    "channel_shock_level": "1550bbf2bded16bb609b5f52669f0b29e6fdf0d7a0898bfb82c5c51a9db3124a",
    "channel_level": "c8dc59f99e57ad30039969921e10ac444de23336f162f7143de84c5e9d055659",
    "saturation_scale": "ff5ba62a84aa40872f2343049ef67c871c5f5fce7a4fa73672cb18f8ff8c11f6",
    "adstock_family": "b8b595535043fbd9bee4855aa2adc923c2a7cf6c5d91fd9b538f1779223d66c8",
    "adstock_alpha": "964b21cdccd0420c8b4e173b9e805d7227315adfd97efbaec6cd0913742588c7",
    "weibull_lam": "a00faf5a38b43360cd8298be9c5f42a2bd390a7eb1e5a2691539d6be0bec8a43",
    "weibull_k": "d1e1ab9e38a9b1557f2383cddc56ae9e083f62bf131ec64e5b6db330efe0a25b",
    "signal_metrics": "2890d03dfacc9264a9d07dbfb48c379925a0c758ff06b0d3eea7069647fe5187",
    "signal_metric_valid": "f1ecbab717272a25f5802cb87be6fd3b1bdd88cede7e75a141f74ebacf212b16",
    "sales_noise": "6e95377b5e6f66ce17a3b621e72b0b27c7ecd5d7d4e3ada9b277ccef14e8413c",
}

#: The SAME corpus with the control texture explicitly disabled. This is the
#: RNG-isolation contract for that mechanism: its parameters degenerate to
#: constants and its noise RVs reach no output, so they collect no random
#: stream. It is a snapshot at this commit, not a pre-texture reproduction —
#: the intercept split above moved both dicts together.
EXPECTED_UNTEXTURED_CONTROL_HASHES = {
    "spend_raw": "dbdf729791679436315c883a009364f25f2e408ea31356402b966231cea67c77",
    "spend_norm": "953649ac8c065c41250786bab74114a941b16959e0c9860efaf7e3bc88a82a1b",
    "spend_share": "572bb7d88f47e606bfb0ed6030c8faf411168cccd4006ae197a46f318d53c6f7",
    "controls": "06471033e7eeef15a1801ead29801aa09fa0cc7308797c54d1b4dd4782338c78",
    "sales_raw": "3f973b17d9b2af66affd0ebb53ce13032577f1d77b55538ec8a68a7d731a9689",
    "sales_norm": "6d425ac1aeb6d3399423ec9172338fcec548f71a08f8ac5df4f704dd4346dee3",
    "support_mask": "003f9ad7fd18ab65a0ca13820be76c5f1637b7e15c7d94fa1b5e327f148c7508",
    "is_future": "7801dc551deac4014f0121c44146e6565844cf603d6ad9e3e61ad1007743afe9",
    "g": "e74d8d2a53ee7a978fe94b066510147be374f197504cff24e02e24cb00478e5c",
    "contributions_raw": "13b9f430aeae2287321e28e415625d189b16ba2fce8b83c22bfa5dc5ff5ec897",
    "baseline_raw": "d8cd100cabff6db38a850bce2eb028a057b0ad61adaa816b770375f5dd7a8602",
    "demand": "0235e997f7a33422a66327bcf454ad173aa7b3d00bce993f2c6668cfa4903ea0",
    "spend_means": "d8f7f278540aa6a0ac23cc5be7ace4b9c6f850dd12723743ef894febe699dc02",
    "sales_scale": "52474f5f9aa9c42ff54060377d4f9ff68bb4e276c4f54dd9f11602f40e31267b",
    "is_val": "eb4b02f10d5660691db02cfd97e0fee99ab9f2461e8e802369f07300b5285ad4",
    "cell_id": "85bf0249350a8edb436598ba927c2298bca0e058665eb864b071e65830562096",
    "treatment_active_mask": "3a7cd171c96fc8fb27a1dadbd085253acf5fb34ff5976402507504ddadd92889",
    "covariate_active_mask": "28a1a7f96dc3b37a56bc3bed913b9f7d3433e765ca26ec6adaedaa272b87baa2",
    "latent_active_mask": "c5d57bca6001a56485f5bb07b383f17e38716d894f10a46c648521ce1d7a677f",
    "n_treatments_active": "c1198a3bdca08a0ae0112be2f753ffc046cece6c10a96c3b4edd974798b21530",
    "n_covariates_active": "9ae6b0b1e6912e108c443a1e35a8fd522c1e0455c79b1152d8d44744d080504f",
    "n_latent_active": "33462003b0a367a219d05faffcae2fa50ef2f8bdac6cb881e694c7c46335f46e",
    "indirect_effects": "b09fa64911c7ca14a34c0988bfa64edeb0157329d76c5f11b45e7470bab151eb",
    "channel_active": "e6a619d2852bed3ed0f8890b700bec0062e3f6f7352eec5a769792bd8c191671",
    "control_contribution": "9ac67afb6432eec1d7a687b6f8e33089661c502acabdef3bad54a0a9051b676a",
    "confounder_contribution": "e1a62ff6a697acae060678dba203e1e557c735d4a6bc91947d9b99063c715900",
    "baseline_intrinsic": "0a88397d6aefc8db85b7d043264644905c2506598849d9a1f4137e495df4cbf1",
    "indirect_effects_by_source": "906c8f58ba38a9692fc897a2cea2e35339a5816d428980474a0d76b1a0ef706d",
    "confounding_strength": "1ebfc5942a66b91e14dd2667a96be1e65275c531c1f4de99471e8982f6367421",
    "channel_shock_mask": "a2c17b7c3b20dff7ef9b8ef7316a3ef323f705522c3fef1744df2a33a3af832b",
    "channel_shock_channel": "7f956e232d961d634c20094542cf11ae23525f62749f31010e76d0cbcacc7e82",
    "channel_shock_start": "52d81fd29e1721a2b63114638580692d9faf97a1ef36eb6d5129adf758cd1f35",
    "channel_shock_length": "fc492e4b1613b6e1cf17271739cd12251266644e410cb1c14180e21c6b07c59e",
    "channel_shock_level_multiplier": "ab57f647547011d917a044e853a1d5796e715c0ccacb505d2c51442e4da8c341",
    "channel_shock_level": "1550bbf2bded16bb609b5f52669f0b29e6fdf0d7a0898bfb82c5c51a9db3124a",
    "channel_level": "c8dc59f99e57ad30039969921e10ac444de23336f162f7143de84c5e9d055659",
    "saturation_scale": "11cb93d14c48fac9dba725594d7ab86e91b887b0dde918ffa4712ced084ad7b8",
    "adstock_family": "b8b595535043fbd9bee4855aa2adc923c2a7cf6c5d91fd9b538f1779223d66c8",
    "adstock_alpha": "87b6928e3d6732651243a012ea29d6d9d499aa3a245693382e6e567075089e4a",
    "weibull_lam": "52355c719e7345dee371e28a8273b58e0a2de9212785736c9b42a858c0eea0a9",
    "weibull_k": "b157e14238bdbcef6f87a1bcdb9ea22312dc4eca9a0c5be514274afd47050118",
    "signal_metrics": "5cd58dd8dcf6b1ec40ad5cf25b176e87e31cb04edb0f1894a6018aba0bef5814",
    "signal_metric_valid": "f1ecbab717272a25f5802cb87be6fd3b1bdd88cede7e75a141f74ebacf212b16",
    "sales_noise": "6e95377b5e6f66ce17a3b621e72b0b27c7ecd5d7d4e3ada9b277ccef14e8413c",
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


def test_disabling_control_texture_is_rng_inert():
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


@pytest.mark.parametrize("scope", ("intercept", "non_media"))
def test_the_intercept_floor_is_a_pure_clip_and_consumes_no_rng(scope):
    """A floor must clip, never re-seed.

    ``pt.maximum`` adds no random variable and no graph output, so a floor that
    never binds has to reproduce the unfloored corpus byte for byte — in either
    scope. If this drifts, the floor is silently changing worlds it was supposed
    to leave alone, which is what would make it unsafe to enable on an existing
    recipe. (Both scopes rearrange the float64 summation slightly, ~2e-15; that
    vanishes in the corpus's float32 storage, which is the persisted contract.)
    """
    floored = _corpus_hashes(baseline_floor=0.0, baseline_floor_scope=scope)
    assert floored == EXPECTED_CORPUS_HASHES


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
