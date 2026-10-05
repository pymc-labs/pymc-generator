"""Checkpoint A: noiseless OLS recovery in a controlled generated world.

This is one deterministic world, not a prevalence survey.  The graph is
structure-known and deliberately excludes confounding, mediation, floors, and
latent paths so that the linear target is well specified.  Each regressor is
still computed independently from that treatment's generated carryover and
saturation mechanism.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pytensor
import pytensor.tensor as pt

from pymc_generator import mechanisms
from pymc_generator.sampler import CARRYOVER_FAMILY_KEYS, SATURATION_FAMILY_KEYS
from pymc_generator.symbolic_graph import build_symbolic_graph

SEED = 20261005
N_TIME_STEPS = 64
WARMUP = 4
L_MAX = 4
N_TREATMENTS = 3
N_COVARIATES = 2
N_LATENT = 1


@dataclass(frozen=True)
class ControlledWorld:
    """Generated arrays and the fixed parameters used to generate them."""

    outputs: dict[str, np.ndarray]
    params: dict[str, object]
    graph: dict[str, np.ndarray]


def _rw(mean, std, *, positive_only: bool) -> dict[str, object]:
    mean_array = np.asarray(mean, dtype=float)
    return {
        "mean": mean_array,
        "std": np.asarray(std, dtype=float),
        "smoothness": np.zeros(mean_array.size),
        "positive_only": positive_only,
        "rw_smoothness_max_weeks": 26,
    }


def _controlled_world() -> ControlledWorld:
    """Build one seeded world through the production symbolic generation graph.

    The first ``WARMUP`` generated rows are retained as carryover history and
    excluded from the fit.  Treatments use geometric, Weibull, and geometric
    carryover; their saturations are Hill, Michaelis--Menten, and root.  The
    two observed controls are direct, while the latent is disconnected.
    """
    n_full = N_TIME_STEPS + WARMUP
    rng = np.random.default_rng(SEED)
    eps = {
        "eps_d": np.zeros((n_full, N_LATENT)),
        "eps_z": rng.normal(size=(n_full, N_COVARIATES)),
        "eps_c": rng.normal(size=(n_full, N_TREATMENTS)),
        "eps_b": np.zeros(n_full),
        "eps_y": np.zeros(n_full),
        "eps_c_hf": np.zeros((n_full, N_TREATMENTS)),
        "eps_c_pulse": np.zeros((n_full, N_TREATMENTS)),
    }
    params: dict[str, object] = {
        "l_max": L_MAX,
        "carryover_family": np.array(
            [CARRYOVER_FAMILY_KEYS.index(name) for name in ("geometric", "weibull", "geometric")]
        ),
        "carryover_alpha": np.array([0.42, 0.61, 0.73]),
        "weibull_lam": np.array([2.4, 3.1, 3.7]),
        "weibull_k": np.array([1.9, 2.3, 2.7]),
        "sat_family": np.array(
            [SATURATION_FAMILY_KEYS.index(name) for name in ("hill", "michaelis_menten", "root")]
        ),
        "hill_slope": np.array([1.8, 2.0, 2.2]),
        "hill_kappa_mult": np.array([0.85, 1.0, 1.15]),
        "logistic_lam": np.ones(N_TREATMENTS),
        "mm_kappa_mult": np.array([0.9, 1.25, 1.0]),
        "tanh_c": np.ones(N_TREATMENTS),
        "root_alpha": np.array([0.65, 0.72, 0.83]),
        "rw_d": _rw([0.0], [0.0], positive_only=False),
        "rw_z": _rw([0.25, -0.45], [0.55, 0.45], positive_only=False),
        "rw_c": _rw([0.45, 0.95, 1.35], [0.55, 0.65, 0.5], positive_only=True),
        "rw_b": _rw([5.0], [0.0], positive_only=False),
        "rw_y": _rw([0.0], [0.0], positive_only=False),
        "u_dz": np.zeros((N_LATENT, N_COVARIATES)),
        "gamma_zz": np.zeros((N_COVARIATES, N_COVARIATES)),
        "w_dc": np.zeros((N_LATENT, N_TREATMENTS)),
        "v_zc": np.zeros((N_COVARIATES, N_TREATMENTS)),
        "alpha_cc": np.zeros((N_TREATMENTS, N_TREATMENTS)),
        "delta_dy": np.zeros(N_LATENT),
        "rho_zy": np.array([0.35, -0.28]),
        "beta": np.array([1.15, 0.82, 1.08]),
        "hf_sigma": np.zeros(N_TREATMENTS),
        "pulse_amp": np.zeros(N_TREATMENTS),
        "pulse_prob": np.zeros(N_TREATMENTS),
        "covariate_hf_sigma": np.zeros(N_COVARIATES),
        "covariate_pulse_amp": np.zeros(N_COVARIATES),
        "covariate_pulse_prob": np.zeros(N_COVARIATES),
        "baseline_floor": None,
    }
    graph = {
        "g_cy": np.ones(N_TREATMENTS, dtype=int),
        "g_dc": np.zeros((N_LATENT, N_TREATMENTS), dtype=int),
        "g_dz": np.zeros((N_LATENT, N_COVARIATES), dtype=int),
        "g_dy": np.zeros(N_LATENT, dtype=int),
        "g_zy": np.ones(N_COVARIATES, dtype=int),
        "g_zc": np.zeros((N_COVARIATES, N_TREATMENTS), dtype=int),
        "g_cc": np.zeros((N_TREATMENTS, N_TREATMENTS), dtype=int),
        "g_zz": np.zeros((N_COVARIATES, N_COVARIATES), dtype=int),
    }
    symbolic = build_symbolic_graph(
        graph,
        params,
        n_full,
        N_TREATMENTS,
        N_COVARIATES,
        N_LATENT,
        burn_in=0,
        eps=eps,
    )["outputs"]
    evaluate = pytensor.function([], list(symbolic.values()))
    outputs = {name: np.asarray(value) for name, value in zip(symbolic, evaluate(), strict=True)}
    return ControlledWorld(outputs=outputs, params=params, graph=graph)


def _evaluate(expression: pt.TensorVariable) -> np.ndarray:
    return np.asarray(pytensor.function([], expression)())


def _mechanism_features(
    treatments: np.ndarray, scales: np.ndarray, params: dict[str, object]
) -> np.ndarray:
    """Apply each channel's true carryover, then its true saturation."""
    features = []
    carryover_family = np.asarray(params["carryover_family"])
    saturation_family = np.asarray(params["sat_family"])
    for k in range(N_TREATMENTS):
        column = pt.as_tensor_variable(treatments[:, k, None])
        carry_id = int(carryover_family[k])
        if carry_id == CARRYOVER_FAMILY_KEYS.index("none"):
            carried = column[:, 0]
        elif carry_id == CARRYOVER_FAMILY_KEYS.index("geometric"):
            carried = mechanisms.apply_geometric_carryover(
                column, np.asarray(params["carryover_alpha"])[k], L_MAX
            )[:, 0]
        else:
            carried = mechanisms.apply_weibull_pdf_carryover(
                column,
                np.asarray(params["weibull_lam"])[k],
                np.asarray(params["weibull_k"])[k],
                L_MAX,
            )[:, 0]
        carried_values = _evaluate(carried)
        family = SATURATION_FAMILY_KEYS[int(saturation_family[k])]
        if family == "linear":
            saturated = pt.as_tensor_variable(carried_values / scales[k])
        elif family == "hill":
            saturated = mechanisms.SATURATION_FAMILIES[family](
                pt.as_tensor_variable(carried_values),
                scales[k],
                slope=np.asarray(params["hill_slope"])[k],
                kappa_mult=np.asarray(params["hill_kappa_mult"])[k],
            )
        elif family == "logistic":
            saturated = mechanisms.SATURATION_FAMILIES[family](
                pt.as_tensor_variable(carried_values),
                scales[k],
                lam=np.asarray(params["logistic_lam"])[k],
            )
        elif family == "michaelis_menten":
            saturated = mechanisms.SATURATION_FAMILIES[family](
                pt.as_tensor_variable(carried_values),
                scales[k],
                kappa_mult=np.asarray(params["mm_kappa_mult"])[k],
            )
        elif family == "tanh":
            saturated = mechanisms.SATURATION_FAMILIES[family](
                pt.as_tensor_variable(carried_values),
                scales[k],
                c=np.asarray(params["tanh_c"])[k],
            )
        else:
            saturated = mechanisms.SATURATION_FAMILIES[family](
                pt.as_tensor_variable(carried_values),
                scales[k],
                alpha=np.asarray(params["root_alpha"])[k],
            )
        features.append(_evaluate(saturated))
    return np.column_stack(features)


def _fit(world: ControlledWorld) -> tuple[np.ndarray, np.ndarray, int, float]:
    outputs = world.outputs
    params = world.params
    features_full = _mechanism_features(outputs["treatments"], outputs["saturation_scale"], params)
    window = slice(WARMUP, None)
    design = np.column_stack(
        [features_full[window], outputs["covariates"][window], np.ones(N_TIME_STEPS)]
    )
    coefficient, _residuals, rank, _singular_values = np.linalg.lstsq(
        design, outputs["outcome"][window], rcond=None
    )
    truth = np.concatenate([np.asarray(params["beta"]), np.asarray(params["rho_zy"]), [5.0]])
    return coefficient, truth, int(rank), float(np.linalg.cond(design))


def test_checkpoint_a_noiseless_ols_recovers_the_generated_coefficients(capsys):
    world = _controlled_world()
    coefficient, truth, rank, condition = _fit(world)
    error = coefficient - truth
    print(
        "seed=20261005 warmup=4 l_max=4 rank="
        f"{rank}/6 raw_condition={condition:.12g} max_abs_error={np.max(np.abs(error)):.12g}"
    )
    print("coefficient truth estimate error")
    for target, estimate, delta in zip(truth, coefficient, error, strict=True):
        print(f"{target:.16g} {estimate:.16g} {delta:.3g}")
    captured = capsys.readouterr().out
    assert "rank=6/6" in captured
    assert rank == 6
    assert condition < 100.0
    np.testing.assert_allclose(coefficient, truth, rtol=2e-12, atol=2e-12)


def test_checkpoint_a_features_match_generation_and_use_history():
    world = _controlled_world()
    outputs = world.outputs
    features_full = _mechanism_features(
        outputs["treatments"], outputs["saturation_scale"], world.params
    )
    window = slice(WARMUP, None)
    generated_contributions = outputs["contributions_observed"][window]
    np.testing.assert_allclose(
        generated_contributions,
        features_full[window] * np.asarray(world.params["beta"]),
        rtol=2e-12,
        atol=2e-12,
    )
    np.testing.assert_array_equal(outputs["outcome_noise"], 0.0)
    np.testing.assert_array_equal(outputs["baseline_intrinsic"], 5.0)
    np.testing.assert_allclose(
        outputs["baseline"],
        outputs["baseline_intrinsic"] + outputs["covariate_contribution"].sum(axis=1),
        rtol=0.0,
        atol=2e-15,
    )
    assert np.all(outputs["saturation_scale"] > 0.0)

    # Re-starting the carryover at the fit window is a silent feature error.
    reset_features = _mechanism_features(
        outputs["treatments"][window], outputs["saturation_scale"], world.params
    )
    assert np.max(np.abs(reset_features[0] - features_full[WARMUP])) > 1e-4


def test_checkpoint_a_graph_has_no_confounding_or_floor_paths():
    world = _controlled_world()
    graph = world.graph
    assert not graph["g_dc"].any()
    assert not graph["g_dz"].any()
    assert not graph["g_dy"].any()
    assert not graph["g_zc"].any()
    assert not graph["g_cc"].any()
    assert not graph["g_zz"].any()
    assert graph["g_cy"].all()
    assert graph["g_zy"].all()
    assert world.params["baseline_floor"] is None


__all__ = [
    "N_TIME_STEPS",
    "SEED",
    "WARMUP",
    "_controlled_world",
    "_fit",
    "_mechanism_features",
]
