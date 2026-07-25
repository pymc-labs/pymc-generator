"""Legacy deterministic contract before identifiability features are enabled."""

from __future__ import annotations

import hashlib
from dataclasses import replace

import numpy as np
import pytest

import prior_generator as pg
from prior_generator.sampler import _slice_g_active, sample_g_additive
from prior_generator.world_model import build_world_model, sample_structure

EXPECTED_CORPUS_HASHES = {
    "spend_raw": "d70d6d526c7105b7376d354f46735722c1fc998a18e1d0064c5a1c6389f661ee",
    "spend_norm": "bab299e570d050204bef7ebb2e9ee7cf0b336c1f7a881dc8112e8381b1c9287a",
    "spend_share": "f8275467e93bc8ba3005caff4e206c9ce03de323c75da725d916c153223ff955",
    "controls": "356c7d2c37a5f42aa6c4321b49944ba05f77dded77e85f64e2e0ecee955b7884",
    "sales_raw": "24f8fcfd7dd579f6a24b411556e85e92d350ae7bcd04bca30235dcf0e8ce2b6b",
    "sales_norm": "7f31ff8f209a0418bc997fd44600c434ca60b16b246ee9eee397396dde0dc599",
    "support_mask": "95ee8b55a20094102512582a4d0021d303214a842b292998ea7a97b7dcd7b0c3",
    "is_future": "bd541332240f592301f7e720c94c416fea54a2ebd26ee22cf46c4dca06c4bdb0",
    "g": "e74d8d2a53ee7a978fe94b066510147be374f197504cff24e02e24cb00478e5c",
    "contributions_raw": "c95b27dc0aa40e433a86df048d0a0a003449c26b4127789ec7966adc0bbba359",
    "baseline_raw": "d43bb30676bc31100dc0044fc82d39014d4e38acf51dbaffdd8a98441890b3a4",
    "demand": "44ed8c764d5519c24d4e1e6e017c4dc3b47c427cf74edf54e84be7bf9b38b8e4",
    "spend_means": "55caaca24efb6f422b6aebb9e1f4a1ce2413049a44fd0ddcb5472f13059139e1",
    "sales_scale": "2a8d92afd4cbdacb27fba81aa4cad17bc5bcc82e6747f0a7c148089e9bddf636",
    "is_val": "3ba2b8d7203a2ac328478a2be7cb0e3ea6e28af5e648841fbf1a41fbbb2e09f7",
    "cell_id": "85bf0249350a8edb436598ba927c2298bca0e058665eb864b071e65830562096",
    "active_c_mask": "a9affb5f52630150ac24b5fb37a2b9e88bc54312f549319752cced7601ebeac1",
    "active_m_mask": "9f40977b73dea1782a868d23dfccb7b0264d20bb07b3e07ad73de81dc5797e91",
    "active_j_mask": "06afea6ca346b374357ad90556c91c315cc6d429ada4eb2a1900fc30a9fdaff7",
    "K_active": "a9073e6b59ee724764dc6cafa783aee1aae5fda1329f86b3b634578b0c8687f1",
    "M_active": "9aed02a9be4f658a038cf3eb48d2effffb9d8e44125a6afe60bdb69cee9cf1b8",
    "J_active": "a9bb0c6117f6fc1cb4a0e889122a322c8c515ce1b82ed2564d127109bb066713",
    "indirect_effects": "d91c999bd459cb5be90bb6378cc173b035d2f062e23dead52e390c48b9cd4003",
    "channel_active": "e6a619d2852bed3ed0f8890b700bec0062e3f6f7352eec5a769792bd8c191671",
    "control_contribution": "8a097daff4652456595a02f853ab5bae50f512c282715e84add97c284037a397",
    "confounder_contribution": "7c8bbd686627bf048057afc0cd20c3fbdf8b95140281511a45aedfce4035d73f",
    "baseline_intrinsic": "15c1859dceb34a5c392aba4dcccd7042f5003bfb3600bc92f0288359e7f82f99",
    "indirect_effects_by_source": "7f5420e30590585e1102d670bf6e8c4fffec7c5991658807edf0f184c4c8b157",
    "confounding_strength": "1ebfc5942a66b91e14dd2667a96be1e65275c531c1f4de99471e8982f6367421",
}

EXPECTED_DEFAULT_FREE_RVS = (
    "rw_d_mean",
    "rw_d_std",
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
    "mm_alpha",
    "mm_kappa_mult",
    "tanh_b",
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
        key: _hash_array(key, value) for key, value in corpus.items() if isinstance(value, np.ndarray)
    }

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
