"""Contract tests for the public oracle sampling/provenance API.

Sampling itself is mocked in this module.  The DataTree fixtures are deliberately
small, while still exercising the bracket-only PyMC 6 result shape used by the
receipt extractor.
"""

from __future__ import annotations

import ast
import copy
import json
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytensor
import pytest
import xarray as xr

import pymc_generator as pg
import pymc_generator.oracle_sampling as oracle
import pymc_generator.world_model as world_model

DEFAULT_REQUESTED = {
    "draws": 800,
    "tune": 500,
    "chains": 4,
    "cores": 4,
    "target_accept": 0.9,
    "random_seed": None,
    "discard_tuned_samples": True,
    "progressbar": False,
    "compute_convergence_checks": False,
    "nuts_sampler": "nutpie",
    "adaptation": "diag",
    "sampler_kwargs": {},
}


def _tree(
    *,
    chains: int = 2,
    draws: int = 4,
    diverging: Any = None,
    energy: Any = None,
    tree_depth: Any = None,
    reached_max_treedepth: Any = None,
    include_posterior: bool = True,
    include_stats: bool = True,
):
    """Build a minimal DataTree without relying on PyMC sampling."""
    coords = {"chain": np.arange(chains), "draw": np.arange(draws)}
    groups = {}
    if include_posterior:
        groups["/posterior"] = xr.Dataset(
            {"theta": (("chain", "draw"), np.ones((chains, draws)))},
            coords=coords,
        )
    if include_stats:
        values = {
            "diverging": diverging,
            "energy": energy,
            "tree_depth": tree_depth,
            "reached_max_treedepth": reached_max_treedepth,
        }
        data = {
            name: (("chain", "draw"), np.asarray(value))
            for name, value in values.items()
            if value is not None
        }
        groups["/sample_stats"] = xr.Dataset(data, coords=coords)
    return xr.DataTree.from_dict(groups)


def _valid_tree(**overrides):
    shape = (2, 4)
    values = {
        "diverging": np.zeros(shape, dtype=bool),
        "energy": np.arange(8, dtype=float).reshape(shape) + 1.0,
        "tree_depth": np.full(shape, 3, dtype=int),
        "reached_max_treedepth": np.zeros(shape, dtype=bool),
    }
    values.update(overrides)
    return _tree(**values)


def _world(model: object = "MODEL"):
    calls: list[str] = []

    def oracle_model(*, latent="marginal"):
        calls.append(latent)
        return model

    return SimpleNamespace(oracle_model=oracle_model), calls


def _sample(
    monkeypatch,
    tree,
    world=None,
    *,
    criteria=None,
    world_identity=None,
    source_identity=None,
    configuration_identity=None,
    **config_kwargs,
):
    world = world or _world()[0]
    calls = []

    def fake_sample(**kwargs):
        calls.append(kwargs)
        return tree

    monkeypatch.setattr(oracle.pm, "sample", fake_sample)
    result = pg.sample_oracle(
        world,
        pg.OracleSamplingConfig(**config_kwargs),
        criteria=criteria,
        world_identity=world_identity,
        source_identity=source_identity,
        configuration_identity=configuration_identity,
    )
    return result, calls


# Configuration -------------------------------------------------------------


def test_public_oracle_exports_are_lazy_and_advertised():
    assert {
        "OracleSamplingConfig",
        "OracleHealthCriteria",
        "OracleSamplingReceipt",
        "OracleSamplingResult",
        "sample_oracle",
    } <= set(pg.__all__)
    assert pg.OracleSamplingConfig is oracle.OracleSamplingConfig
    assert pg.OracleHealthCriteria is oracle.OracleHealthCriteria
    assert pg.sample_oracle is oracle.sample_oracle


def test_config_defaults_and_to_dict_are_literal_contract():
    config = pg.OracleSamplingConfig()
    assert config.to_dict() == DEFAULT_REQUESTED | {"latent": "marginal"}
    assert config.random_seed is None
    assert config.cores == 4
    assert config.to_dict() is not config.to_dict()


@pytest.mark.parametrize(
    "field,value",
    [
        ("draws", 0),
        ("tune", -1),
        ("chains", 0),
        ("cores", 0),
        ("target_accept", 0.0),
        ("target_accept", 1.0),
        ("target_accept", float("nan")),
        ("target_accept", float("inf")),
        ("random_seed", -1),
        ("latent", "other"),
        ("nuts_sampler", "other"),
    ],
)
def test_config_rejects_invalid_values(field, value):
    with pytest.raises((TypeError, ValueError)):
        pg.OracleSamplingConfig(**{field: value}).validate()


@pytest.mark.parametrize("field", ["draws", "tune", "chains", "cores"])
def test_config_rejects_boolean_integer_fields(field):
    with pytest.raises(TypeError):
        pg.OracleSamplingConfig(**{field: True}).validate()


@pytest.mark.parametrize(
    "field,value",
    [
        ("target_accept", True),
        ("random_seed", True),
        ("discard_tuned_samples", 1),
        ("progressbar", 0),
        ("compute_convergence_checks", "false"),
    ],
)
def test_config_rejects_boolean_coercion(field, value):
    with pytest.raises(TypeError):
        pg.OracleSamplingConfig(**{field: value}).validate()


def test_config_accepts_pymc_sampler_and_serializes_it():
    config = pg.OracleSamplingConfig(nuts_sampler="pymc")
    config.validate()
    assert config.to_dict()["nuts_sampler"] == "pymc"
    assert config.to_dict()["adaptation"] == "diag"


@pytest.mark.parametrize("adaptation", ["diag", "draw_diag", "low_rank", "flow"])
def test_config_accepts_nutpie_adaptation_values(adaptation):
    config = pg.OracleSamplingConfig(adaptation=adaptation)
    config.validate()
    assert config.to_dict()["adaptation"] == adaptation


@pytest.mark.parametrize("adaptation", [None, True, 1, "other"])
def test_config_rejects_invalid_adaptation_values(adaptation):
    with pytest.raises((TypeError, ValueError)):
        pg.OracleSamplingConfig(adaptation=adaptation).validate()


def test_config_rejects_nondefault_adaptation_for_pymc():
    with pytest.raises(ValueError, match="Nutpie-only"):
        pg.OracleSamplingConfig(nuts_sampler="pymc", adaptation="draw_diag")


@pytest.mark.parametrize(
    "key",
    [
        "draws",
        "tune",
        "chains",
        "cores",
        "target_accept",
        "random_seed",
        "seed",
        "save_warmup",
        "progress_bar",
        "blocking",
        "sampler",
        "adaptation",
        "nuts_sampler_kwargs",
        "return_raw_trace",
        "progressbar",
        "compute_convergence_checks",
        "discard_tuned_samples",
        "nuts_sampler",
        "model",
        "step",
        "trace",
        "backend",
        "return_inferencedata",
        "var_names",
    ],
)
def test_config_rejects_reserved_sampler_kwargs(key):
    with pytest.raises(ValueError):
        pg.OracleSamplingConfig(sampler_kwargs={key: 1}).validate()


@pytest.mark.parametrize("value", [float("nan"), float("inf"), {1: "not-string"}, object()])
def test_config_rejects_non_json_sampler_values(value):
    with pytest.raises((TypeError, ValueError)):
        pg.OracleSamplingConfig(sampler_kwargs={"bad": value}).validate()


def test_config_requires_sampler_mapping():
    with pytest.raises(TypeError):
        pg.OracleSamplingConfig(sampler_kwargs=[("x", 1)]).validate()


def test_invalid_config_is_rejected_before_model_or_sampling(monkeypatch):
    world, model_calls = _world()
    sample_calls = []
    monkeypatch.setattr(oracle.pm, "sample", lambda **kwargs: sample_calls.append(kwargs))
    with pytest.raises(ValueError):
        pg.sample_oracle(world, pg.OracleSamplingConfig(draws=0))
    assert model_calls == []
    assert sample_calls == []


def test_config_copies_nested_sampler_kwargs():
    nested = {"nuts": {"max_treedepth": 10}, "custom": [1, 2]}
    config = pg.OracleSamplingConfig(sampler_kwargs=nested)
    nested["nuts"]["max_treedepth"] = 99
    nested["custom"].append(3)
    assert config.to_dict()["sampler_kwargs"] == {
        "nuts": {"max_treedepth": 10},
        "custom": [1, 2],
    }
    with pytest.raises(TypeError):
        config.sampler_kwargs["new"] = 1
    with pytest.raises(TypeError):
        config.sampler_kwargs["nuts"]["max_treedepth"] = 1


@pytest.mark.parametrize(
    "field", ["draws", "tune", "chains", "cores", "target_accept", "random_seed", "latent"]
)
def test_config_to_dict_is_json_native(field):
    config = pg.OracleSamplingConfig()
    assert json.dumps(config.to_dict(), allow_nan=False)
    assert field in config.to_dict()


# Health criteria -----------------------------------------------------------


def test_health_criteria_defaults_and_to_dict():
    criteria = pg.OracleHealthCriteria()
    assert criteria.to_dict() == {
        "max_divergences": 0,
        "max_rhat": 1.01,
        "min_ess_bulk": 400.0,
        "min_ess_tail": 400.0,
        "max_tree_depth_saturation": 0,
        "min_bfmi": 0.3,
        "require_divergences": True,
        "require_rhat": True,
        "require_ess_bulk": True,
        "require_ess_tail": True,
        "require_tree_depth": False,
        "require_bfmi": False,
    }


@pytest.mark.parametrize(
    "field,value",
    [
        ("max_divergences", -1),
        ("max_rhat", 0.99),
        ("max_rhat", float("nan")),
        ("min_ess_bulk", -1),
        ("min_ess_tail", float("inf")),
        ("max_tree_depth_saturation", -1),
        ("min_bfmi", -1),
    ],
)
def test_health_criteria_rejects_invalid_thresholds(field, value):
    with pytest.raises((TypeError, ValueError)):
        pg.OracleHealthCriteria(**{field: value}).validate()


def test_health_criteria_allows_disabled_thresholds_and_custom_requirements():
    criteria = pg.OracleHealthCriteria(
        max_divergences=None,
        max_rhat=None,
        min_ess_bulk=None,
        min_ess_tail=None,
        max_tree_depth_saturation=None,
        min_bfmi=None,
        require_divergences=False,
        require_rhat=False,
        require_ess_bulk=False,
        require_ess_tail=False,
        require_tree_depth=True,
        require_bfmi=True,
    )
    criteria.validate()
    assert criteria.to_dict()["max_rhat"] is None
    assert criteria.to_dict()["require_bfmi"] is True


# Sampling boundary and receipt --------------------------------------------


@pytest.mark.parametrize("latent", ["marginal", "sampled"])
def test_sample_oracle_supports_both_latent_representations(monkeypatch, latent):
    world, model_calls = _world()
    _sample(monkeypatch, _valid_tree(), world, latent=latent)
    assert model_calls == [latent]


@pytest.mark.parametrize("nuts_sampler", ["nutpie", "pymc"])
def test_sample_oracle_uses_only_scm_oracle_model_and_one_nuts_call(monkeypatch, nuts_sampler):
    world, model_calls = _world()
    tree = _valid_tree()
    result, calls = _sample(
        monkeypatch,
        tree,
        world,
        latent="sampled",
        draws=7,
        tune=8,
        chains=2,
        cores=1,
        target_accept=0.91,
        random_seed=11,
        nuts_sampler=nuts_sampler,
        sampler_kwargs={"nuts": {"max_treedepth": 12}},
    )
    assert model_calls == ["sampled"]
    assert len(calls) == 1
    assert calls[0] == {
        "model": "MODEL",
        "draws": 7,
        "tune": 8,
        "chains": 2,
        "cores": 1,
        "target_accept": 0.91,
        "random_seed": 11,
        "discard_tuned_samples": True,
        "progressbar": False,
        "compute_convergence_checks": False,
        "nuts_sampler": nuts_sampler,
        **({"nuts_sampler_kwargs": {"adaptation": "diag"}} if nuts_sampler == "nutpie" else {}),
        "nuts": {"max_treedepth": 12},
    }
    assert result.idata is tree
    assert isinstance(result.idata, xr.DataTree)
    assert result.receipt.to_dict()["sampling"]["requested"]["nuts_sampler"] == nuts_sampler


def test_one_shot_nutpie_passes_exact_adaptation_through_supported_kwargs(monkeypatch):
    result, calls = _sample(
        monkeypatch, _valid_tree(), adaptation="draw_diag", nuts_sampler="nutpie"
    )
    assert calls[0]["nuts_sampler_kwargs"] == {"adaptation": "draw_diag"}
    assert result.receipt.to_dict()["sampling"]["requested"]["adaptation"] == "draw_diag"


def test_nondefault_nutpie_adaptation_rejects_before_sampling(monkeypatch):
    world, model_calls = _world()
    sample_calls = []
    monkeypatch.setattr(oracle.pm, "sample", lambda **kwargs: sample_calls.append(kwargs))
    with pytest.raises(ValueError, match="Nutpie-only"):
        pg.sample_oracle(
            world, pg.OracleSamplingConfig(nuts_sampler="pymc", adaptation="draw_diag")
        )
    assert model_calls == []
    assert sample_calls == []


def test_non_datatree_sampling_result_is_rejected(monkeypatch):
    monkeypatch.setattr(oracle.pm, "sample", lambda **kwargs: xr.Dataset())
    with pytest.raises(TypeError):
        pg.sample_oracle(_world()[0])


def test_sample_oracle_preserves_model_and_sampling_errors(monkeypatch):
    world, _ = _world()
    error = RuntimeError("sampling failed")

    def fail(**kwargs):
        raise error

    monkeypatch.setattr(oracle.pm, "sample", fail)
    with pytest.raises(RuntimeError, match="sampling failed"):
        pg.sample_oracle(world)


def test_model_construction_error_propagates_without_sampling(monkeypatch):
    error = ValueError("model construction failed")
    calls = []

    def fail_model(*, latent):
        raise error

    world = SimpleNamespace(oracle_model=fail_model)
    monkeypatch.setattr(oracle.pm, "sample", lambda **kwargs: calls.append(kwargs))
    with pytest.raises(ValueError, match="model construction failed"):
        pg.sample_oracle(world)
    assert calls == []


def test_receipt_contains_caller_identities_without_inference(monkeypatch):
    identity = {"world_id": "w-1", "nested": {"source": "fixture"}}
    result, _ = _sample(
        monkeypatch,
        _valid_tree(),
        world_identity=identity,
        source_identity={"source_id": "s-1"},
        configuration_identity={"config_id": "c-1"},
    )
    receipt = result.receipt.to_dict()
    assert receipt["identities"] == {
        "world": identity,
        "source": {"source_id": "s-1"},
        "configuration": {"config_id": "c-1"},
    }
    assert result.receipt.to_dict()["identities"] == receipt["identities"]


def test_receipt_unknown_identities_are_empty_and_identity_json_is_strict(monkeypatch):
    result, _ = _sample(monkeypatch, _valid_tree())
    assert result.receipt.to_dict()["identities"] == {
        "world": {},
        "source": {},
        "configuration": {},
    }
    with pytest.raises((TypeError, ValueError)):
        pg.sample_oracle(_world()[0], world_identity={"bad": float("nan")})
    with pytest.raises(TypeError):
        pg.sample_oracle(_world()[0], world_identity={1: "not-string"})


def test_invalid_world_and_identity_types_fail_before_model_construction(monkeypatch):
    sample_calls = []
    world, model_calls = _world()
    monkeypatch.setattr(oracle.pm, "sample", lambda **kwargs: sample_calls.append(kwargs))
    with pytest.raises(TypeError):
        pg.sample_oracle(object())
    for kwargs in (
        {"world_identity": []},
        {"source_identity": object()},
        {"configuration_identity": []},
    ):
        with pytest.raises(TypeError):
            pg.sample_oracle(world, **kwargs)
    assert model_calls == []
    assert sample_calls == []


def test_receipt_is_canonical_strict_json_and_excludes_posterior_content(monkeypatch):
    result, _ = _sample(monkeypatch, _valid_tree())
    data = result.receipt.to_dict()
    encoded = result.receipt.to_json()
    assert json.loads(encoded) == data
    assert encoded == json.dumps(data, allow_nan=False, sort_keys=True, separators=(",", ":"))
    assert data["posterior"] == {"present": True, "included_in_receipt": False}
    serialized = encoded.lower()
    for forbidden in ("theta", "coordinates", "initial_point", "model_graph"):
        assert forbidden not in serialized
    assert "health_is_not_a_convergence_guarantee" in data["limitations"]
    assert "step_method_and_mass_matrix_are_not_observed" in data["limitations"]


def test_receipt_has_all_required_top_level_fields(monkeypatch):
    result, _ = _sample(monkeypatch, _valid_tree())
    receipt = result.receipt.to_dict()
    assert set(receipt) == {
        "schema_version",
        "kind",
        "package_version",
        "environment",
        "identities",
        "oracle",
        "sampling",
        "timing",
        "posterior",
        "diagnostics",
        "health",
        "limitations",
    }
    assert receipt["schema_version"] == 1
    assert receipt["kind"] == "pymc_generator.oracle_sampling_receipt"
    assert receipt["oracle"] == {"builder": "SCM.oracle_model", "latent": "marginal"}


@pytest.mark.parametrize(
    "field,value",
    [
        ("schema_version", "1"),
        ("package_version", 1),
        ("environment", []),
        ("environment", {"python": 1}),
        ("identities", {"world": []}),
        ("limitations", [1]),
    ],
)
def test_receipt_post_init_rejects_bad_scalar_mapping_and_limitations(monkeypatch, field, value):
    result, _ = _sample(monkeypatch, _valid_tree())
    with pytest.raises((TypeError, ValueError)):
        replace(result.receipt, **{field: value})


def test_receipt_requested_and_effective_claims_are_separate(monkeypatch):
    result, _ = _sample(monkeypatch, _valid_tree(), draws=3, chains=2, cores=1)
    sampling = result.receipt.to_dict()["sampling"]
    assert sampling["requested"]["nuts_sampler"] == "nutpie"
    assert sampling["requested"]["draws"] == 3
    assert sampling["effective"]["groups"] == ["posterior", "sample_stats"]
    assert sampling["effective"]["sample_stat_names"] == [
        "diverging",
        "energy",
        "reached_max_treedepth",
        "tree_depth",
    ]
    assert sampling["effective"]["inference_library"] == "pymc"
    assert sampling["effective"]["step_methods"] == {
        "status": "unavailable",
        "reason": "not_exposed_by_sampling_result",
    }
    assert sampling["effective"]["mass_matrix"] == {
        "status": "unavailable",
        "reason": "not_exposed_by_sampling_result",
    }


def test_receipt_effective_groups_are_complete_and_sorted(monkeypatch):
    base = _valid_tree()
    extra = xr.Dataset(
        {"aux": (("chain", "draw"), np.zeros((2, 4)))},
        coords={"chain": np.arange(2), "draw": np.arange(4)},
    )
    tree = xr.DataTree.from_dict(
        {
            "/posterior": base["posterior"].to_dataset(),
            "/sample_stats": base["sample_stats"].to_dataset(),
            "/zeta": extra,
            "/alpha": extra,
        }
    )
    result, _ = _sample(monkeypatch, tree)
    assert result.receipt.to_dict()["sampling"]["effective"]["groups"] == [
        "alpha",
        "posterior",
        "sample_stats",
        "zeta",
    ]


def test_receipt_environment_has_versioned_runtime_fields(monkeypatch):
    result, _ = _sample(monkeypatch, _valid_tree())
    environment = result.receipt.to_dict()["environment"]
    assert set(environment) == {
        "python",
        "numpy",
        "scipy",
        "xarray",
        "arviz",
        "pymc",
        "pytensor",
        "pymc_marketing",
        "nutpie",
    }
    assert all(isinstance(value, str) and value for value in environment.values())


# Diagnostic extraction -----------------------------------------------------


def _metric_dataset(value, name="theta"):
    return xr.Dataset({name: xr.DataArray(value)})


def _patch_arviz(monkeypatch, *, rhat=1.003, bulk=612.4, tail=488.2, bfmi=0.84):
    monkeypatch.setattr(oracle.az, "rhat", lambda *args, **kwargs: _metric_dataset(rhat))
    monkeypatch.setattr(
        oracle.az,
        "ess",
        lambda *args, **kwargs: _metric_dataset(bulk if kwargs.get("method") == "bulk" else tail),
    )
    monkeypatch.setattr(oracle.az, "bfmi", lambda *args, **kwargs: np.asarray([bfmi, bfmi + 0.01]))


def test_diagnostics_use_bracket_groups_and_expected_reductions(monkeypatch):
    _patch_arviz(monkeypatch)
    result, _ = _sample(monkeypatch, _valid_tree())
    diagnostics = result.receipt.to_dict()["diagnostics"]
    assert diagnostics["posterior_present"] is True
    assert diagnostics["sample_stats_present"] is True
    assert diagnostics["n_chains"] == 2
    assert diagnostics["draws_per_chain"] == 4
    assert diagnostics["total_draws"] == 8
    assert diagnostics["divergences"] == {"status": "available", "value": 0}
    assert diagnostics["rhat"] == {
        "status": "available",
        "value": 1.003,
        "method": "rank",
        "reduction": "max",
    }
    assert diagnostics["ess_bulk"] == {"status": "available", "value": 612.4, "reduction": "min"}
    assert diagnostics["ess_tail"] == {"status": "available", "value": 488.2, "reduction": "min"}
    assert diagnostics["bfmi"] == {"status": "available", "value": 0.84, "reduction": "min"}
    assert diagnostics["tree_depth_max"] == {"status": "available", "value": 3}
    assert diagnostics["tree_depth_saturation"] == {
        "status": "available",
        "value": 0,
        "source": "reached_max_treedepth",
    }


def test_divergences_count_only_diverging_not_divergences_alias(monkeypatch):
    shape = (2, 4)
    tree = _tree(
        diverging=np.array([[True, False, False, False], [False] * 4]),
        reached_max_treedepth=np.zeros(shape, dtype=bool),
        tree_depth=np.ones(shape, dtype=int),
        energy=np.ones(shape),
    )
    tree["sample_stats"]["divergences"] = (("chain", "draw"), np.ones(shape, dtype=bool))
    _patch_arviz(monkeypatch)
    result, _ = _sample(monkeypatch, tree)
    assert result.receipt.to_dict()["diagnostics"]["divergences"]["value"] == 1


def test_tree_depth_max_does_not_infer_saturation(monkeypatch):
    tree = _valid_tree(reached_max_treedepth=None, tree_depth=np.full((2, 4), 12))
    _patch_arviz(monkeypatch)
    result, _ = _sample(monkeypatch, tree)
    diagnostics = result.receipt.to_dict()["diagnostics"]
    assert diagnostics["tree_depth_max"] == {"status": "available", "value": 12}
    assert diagnostics["tree_depth_saturation"]["status"] == "unavailable"
    assert "value" not in diagnostics["tree_depth_saturation"]


@pytest.mark.parametrize(
    "tree_kwargs,metric",
    [
        ({"include_stats": False}, "divergences"),
        ({"diverging": None}, "divergences"),
        ({"diverging": np.zeros((2, 4), dtype=bool), "energy": np.full((2, 4), np.nan)}, "bfmi"),
        (
            {"diverging": np.zeros((2, 4), dtype=bool), "tree_depth": np.full((2, 4), np.nan)},
            "tree_depth_max",
        ),
    ],
)
def test_missing_prerequisites_are_unavailable(monkeypatch, tree_kwargs, metric):
    tree = _valid_tree(**tree_kwargs)
    _patch_arviz(monkeypatch)
    result, _ = _sample(monkeypatch, tree)
    assert result.receipt.to_dict()["diagnostics"][metric]["status"] == "unavailable"
    assert "reason" in result.receipt.to_dict()["diagnostics"][metric]


def test_bfmi_requires_energy_sample_stat(monkeypatch):
    tree = _valid_tree(energy=None)
    _patch_arviz(monkeypatch)
    result, _ = _sample(monkeypatch, tree)
    bfmi = result.receipt.to_dict()["diagnostics"]["bfmi"]
    assert bfmi["status"] == "unavailable"
    assert "reason" in bfmi


def test_less_than_two_chains_makes_rhat_unavailable(monkeypatch):
    tree = _valid_tree()
    posterior = tree["posterior"].to_dataset().isel(chain=slice(0, 1))
    stats = tree["sample_stats"].to_dataset().isel(chain=slice(0, 1))
    tree = xr.DataTree.from_dict({"/posterior": posterior, "/sample_stats": stats})
    _patch_arviz(monkeypatch)
    result, _ = _sample(monkeypatch, tree)
    assert result.receipt.to_dict()["diagnostics"]["rhat"]["status"] == "unavailable"


@pytest.mark.parametrize("metric", ["rhat", "ess_bulk", "ess_tail", "bfmi"])
def test_no_finite_metric_is_invalid(monkeypatch, metric):
    _patch_arviz(monkeypatch, rhat=np.nan, bulk=np.nan, tail=np.nan, bfmi=np.nan)
    result, _ = _sample(monkeypatch, _valid_tree())
    assert result.receipt.to_dict()["diagnostics"][metric]["status"] == "invalid"
    assert "reason" in result.receipt.to_dict()["diagnostics"][metric]


def test_malformed_metric_shape_is_invalid(monkeypatch):
    tree = _valid_tree()
    stats = tree["sample_stats"].to_dataset()
    stats["tree_depth"] = (("chain",), np.array([1, 2]))
    tree = xr.DataTree.from_dict(
        {"/posterior": tree["posterior"].to_dataset(), "/sample_stats": stats}
    )
    _patch_arviz(monkeypatch)
    result, _ = _sample(monkeypatch, tree)
    diagnostic = result.receipt.to_dict()["diagnostics"]["tree_depth_max"]
    assert diagnostic["status"] == "invalid"
    assert "reason" in diagnostic


def test_empty_diagnostic_array_is_invalid(monkeypatch):
    base = _valid_tree()
    stats = xr.Dataset(
        {"diverging": (("empty",), np.array([], dtype=bool))},
        coords={"empty": np.array([], dtype=int)},
    )
    tree = xr.DataTree.from_dict(
        {"/posterior": base["posterior"].to_dataset(), "/sample_stats": stats}
    )
    result, _ = _sample(monkeypatch, tree)
    diagnostic = result.receipt.to_dict()["diagnostics"]["divergences"]
    assert diagnostic["status"] == "invalid"
    assert diagnostic["reason"]


def test_extra_dimensional_diagnostic_is_invalid(monkeypatch):
    base = _valid_tree()
    stats = base["sample_stats"].to_dataset()
    stats["diverging"] = (
        ("chain", "draw", "extra"),
        np.zeros((2, 4, 1), dtype=bool),
    )
    stats = stats.assign_coords(extra=[0])
    tree = xr.DataTree.from_dict(
        {"/posterior": base["posterior"].to_dataset(), "/sample_stats": stats}
    )
    result, _ = _sample(monkeypatch, tree)
    diagnostic = result.receipt.to_dict()["diagnostics"]["divergences"]
    assert diagnostic["status"] == "invalid"
    assert diagnostic["reason"]


def test_mismatched_diagnostic_dimensions_are_invalid(monkeypatch):
    base = _valid_tree()
    stats = base["sample_stats"].to_dataset()
    stats = stats.drop_vars("energy")
    stats["energy"] = (("chain", "other"), np.ones((2, 3)))
    stats = stats.assign_coords(other=[0, 1, 2])
    tree = xr.DataTree.from_dict(
        {"/posterior": base["posterior"].to_dataset(), "/sample_stats": stats}
    )
    result, _ = _sample(monkeypatch, tree)
    diagnostic = result.receipt.to_dict()["diagnostics"]["bfmi"]
    assert diagnostic["status"] == "invalid"
    assert diagnostic["reason"]


def test_arviz_errors_become_deterministic_invalid_diagnostics(monkeypatch):
    def explode(*args, **kwargs):
        raise RuntimeError("bad arviz")

    monkeypatch.setattr(oracle.az, "rhat", explode)
    monkeypatch.setattr(oracle.az, "ess", explode)
    monkeypatch.setattr(oracle.az, "bfmi", explode)
    result, _ = _sample(monkeypatch, _valid_tree())
    diagnostics = result.receipt.to_dict()["diagnostics"]
    for metric in ("rhat", "ess_bulk", "ess_tail", "bfmi"):
        assert diagnostics[metric]["status"] == "invalid"
        assert diagnostics[metric]["reason"]


# Reusable compiled Oracle lifecycle ----------------------------------------


def _shared_template_and_world(*, changed_sales=0.0, selected=(0, 1, 2)):
    values = {
        "channels_data": np.zeros((4, 1)),
        "controls_data": np.ones((4, 1)),
        "sales_data": np.arange(4, dtype=float) + changed_sales,
        "saturation_scale_data": np.ones(1),
        "g_cy_data": np.ones(1),
        "g_db_data": np.zeros(1),
        "g_zb_data": np.ones(1),
        "observed_indices_data": np.asarray(selected, dtype="int64"),
    }
    with oracle.pm.Model() as model:
        for name, value in values.items():
            oracle.pm.Data(name, value)
        # Every rebound value participates in one connected observed graph;
        # this is intentionally not a disconnected pm.Data placeholder.
        mu = (
            model["channels_data"][:, 0] * model["g_cy_data"][0]
            + model["controls_data"][:, 0] * model["g_zb_data"][0]
            + model["saturation_scale_data"][0]
            + model["g_db_data"][0]
            + model["g_zb_data"][0]
        )
        oracle.pm.Normal(
            "connected_likelihood",
            mu=mu[model["observed_indices_data"]],
            sigma=1.0,
            observed=model["sales_data"][model["observed_indices_data"]],
        )
    world = SimpleNamespace(
        data={
            "treatments": values["channels_data"],
            "covariates": values["controls_data"],
            "outcome": values["sales_data"],
            "saturation_scale": values["saturation_scale_data"],
        },
        g={"g_cy": values["g_cy_data"], "g_dy": values["g_db_data"], "g_zy": values["g_zb_data"]},
        cfg=SimpleNamespace(__dataclass_fields__={}),
        extras={"structural": {}},
    )
    signature = oracle._world_signature(world, latent="marginal", observed_count=len(selected))
    template = pg.OracleTemplate(
        model=model,
        signature=signature,
        latent="marginal",
        observed_indices=tuple(selected),
        reported_rows=3,
        data_contract=oracle.ORACLE_SHARED_DATA_NAMES,
    )
    return template, world


class _FakeCompiled:
    def __init__(self, tree, *, shared_var_keys=None):
        self.tree = tree
        self.payloads = []
        self.sample_calls = []
        if shared_var_keys is not None:
            self.shared_var_keys = shared_var_keys

    def with_data(self, **payload):
        self.payloads.append({name: np.array(value, copy=True) for name, value in payload.items()})
        return self

    def sample(self, **kwargs):
        self.sample_calls.append(kwargs)
        return self.tree


class _EvaluatingCompiled(_FakeCompiled):
    """Small compiled-model stand-in that evaluates the connected PyMC graph."""

    def __init__(self, tree, model):
        super().__init__(tree)
        self.model = model
        self.logps = []
        self._logp = model.compile_logp()

    def with_data(self, **payload):
        super().with_data(**payload)
        oracle.pm.set_data(payload, model=self.model)
        return self

    def sample(self, **kwargs):
        self.logps.append(float(self._logp(self.model.initial_point())))
        return super().sample(**kwargs)


def _compiled(template, fake):
    return pg.CompiledOracle(template, fake, sampler=lambda bound, **kwargs: bound.sample(**kwargs))


@pytest.mark.parametrize(
    ("surface", "message"),
    [
        ({}, "surface"),
        ({"first": "channels_data", "second": "channels_data"}, "mapping keys"),
        ({"first": "channels_data", "second": "not_contract"}, "mapping keys"),
        ({"first": 1}, "mapping keys"),
    ],
)
def test_compiled_oracle_rejects_invalid_shared_variable_surface(surface, message):
    template, _ = _shared_template_and_world()
    with pytest.raises((TypeError, ValueError), match=message):
        _compiled(template, _FakeCompiled(_valid_tree(), shared_var_keys=surface))


def test_compiled_oracle_accepts_nutpie_shared_variable_mapping():
    template, _ = _shared_template_and_world()
    variables = {name: template.model.named_vars[name] for name in pg.ORACLE_SHARED_DATA_NAMES}
    compiled = _compiled(
        template,
        _FakeCompiled(
            _valid_tree(),
            shared_var_keys={variable: f"internal-{name}" for name, variable in variables.items()},
        ),
    )
    assert compiled is not None


def test_compile_oracle_compiles_once_and_binds_each_fit(monkeypatch):
    template, world = _shared_template_and_world()
    _, changed = _shared_template_and_world(changed_sales=10.0)
    fake = _FakeCompiled(_valid_tree())
    compile_calls = []
    sample_calls = []
    monkeypatch.setitem(
        __import__("sys").modules,
        "nutpie",
        SimpleNamespace(
            compile_pymc_model=lambda model, **kwargs: (
                compile_calls.append((model, kwargs)) or fake
            ),
            sample=lambda bound, **kwargs: sample_calls.append(bound) or bound.sample(**kwargs),
        ),
    )
    compiled = pg.compile_oracle(template)
    compiled.fit(world)
    compiled.fit(changed)
    assert len(compile_calls) == 1
    assert compile_calls[0][1] == {"backend": "numba"}
    assert len(fake.payloads) == 2
    assert sample_calls == [fake, fake]
    assert not np.array_equal(fake.payloads[0]["sales_data"], fake.payloads[1]["sales_data"])
    assert set(fake.payloads[0]) == set(pg.ORACLE_SHARED_DATA_NAMES)
    assert np.array_equal(fake.payloads[0]["observed_indices_data"], [0, 1, 2])


def test_compiled_fit_reuses_one_connected_graph_without_stale_data():
    template, world = _shared_template_and_world()
    _, changed = _shared_template_and_world(changed_sales=10.0)
    fake = _EvaluatingCompiled(_valid_tree(), template.model)
    compiled = _compiled(template, fake)

    compiled.fit(world)
    compiled.fit(changed)
    compiled.fit(world)

    assert len(fake.logps) == 3
    assert fake.logps[0] == pytest.approx(fake.logps[2])
    assert fake.logps[0] != pytest.approx(fake.logps[1])


def _dynamic_width_world(smoothness_b, *, smoothness_d=None):
    structural = {"smoothness_b": np.asarray(smoothness_b)}
    if smoothness_d is not None:
        structural["smoothness_d"] = np.asarray(smoothness_d)
    return SimpleNamespace(
        data={
            "treatments": np.zeros((4, 1)),
            "covariates": np.ones((4, 1)),
            "outcome": np.arange(4, dtype=float),
            "saturation_scale": np.ones(1),
        },
        g={"g_cy": np.ones(1), "g_dy": np.zeros(1), "g_zy": np.ones(1)},
        cfg=SimpleNamespace(carryover_burn_in=0, rw_smoothness_max_weeks=26),
        extras={"structural": structural},
    )


def test_oracle_payload_normalises_historical_int64_walk_width_b():
    world = _dynamic_width_world(np.asarray([0, 1], dtype="int64"))
    payload = oracle._oracle_payload(
        world, np.asarray([0], dtype="int64"), dynamic_names={"walk_width_b_data"}
    )
    assert payload["walk_width_b_data"].dtype == np.dtype("int32")
    np.testing.assert_array_equal(payload["walk_width_b_data"], np.asarray([0, 3], dtype="int32"))


@pytest.mark.parametrize(
    ("values", "message"),
    [
        (np.asarray([1.5]), "integral"),
        (np.asarray([np.iinfo(np.int32).max + 1], dtype="int64"), "fit int32"),
        (np.asarray([np.nan]), "integral"),
    ],
)
def test_oracle_payload_rejects_lossy_walk_width_b_values(monkeypatch, values, message):
    monkeypatch.setattr(
        "pymc_generator.random_walk.walk_width_index",
        lambda *args, **kwargs: values,
    )
    world = _dynamic_width_world([0])
    with pytest.raises(ValueError, match=message):
        oracle._oracle_payload(
            world, np.asarray([0], dtype="int64"), dynamic_names={"walk_width_b_data"}
        )


def test_oracle_payload_normalises_historical_int64_walk_width_d():
    world = _dynamic_width_world([0], smoothness_d=np.asarray([0, 1], dtype="int64"))
    payload = oracle._oracle_payload(
        world, np.asarray([0], dtype="int64"), dynamic_names={"walk_width_d_data"}
    )
    assert payload["walk_width_d_data"].dtype == np.dtype("int32")
    np.testing.assert_array_equal(payload["walk_width_d_data"], np.asarray([0, 3], dtype="int32"))


@pytest.mark.parametrize(
    ("values", "message"),
    [
        (np.asarray([1.5]), "integral"),
        (np.asarray([np.nan]), "integral"),
        (np.asarray([np.iinfo(np.int32).max + 1], dtype="int64"), "fit int32"),
        (np.asarray([True]), "numeric integers"),
    ],
)
def test_oracle_payload_rejects_lossy_walk_width_d_values_before_binding(
    monkeypatch, values, message
):
    monkeypatch.setattr(
        "pymc_generator.random_walk.walk_width_index",
        lambda *args, **kwargs: values,
    )
    world = _dynamic_width_world([0], smoothness_d=[0])
    with pytest.raises(ValueError, match=message):
        oracle._oracle_payload(
            world, np.asarray([0], dtype="int64"), dynamic_names={"walk_width_d_data"}
        )


def test_zero_width_conditioning_rejected_before_runtime_binding():
    _, world = _shared_template_and_world()
    world.extras["prior_cond"] = {"carryover_alpha": (0.2, 0.0)}
    with pytest.raises(ValueError, match="zero-width.*unsupported"):
        oracle._dynamic_payload(world, {"prior_cond_carryover_alpha_data"})


def test_point_mass_and_interval_have_distinct_topologies():
    _, world = _shared_template_and_world()
    interval = oracle._world_signature(world, latent="marginal", observed_count=3)
    world.extras["prior_cond"] = {"carryover_alpha": (0.2, 0.0)}
    point = oracle._world_signature(world, latent="marginal", observed_count=3)
    assert point != interval


def test_changed_fixed_config_fails_before_binding():
    template, world = _shared_template_and_world()
    template = replace(template, config_contract={"rw_std_sigma": 1.0})
    world.cfg = SimpleNamespace(__dataclass_fields__={}, rw_std_sigma=2.0)
    fake = _FakeCompiled(_valid_tree())
    with pytest.raises(ValueError, match="rw_std_sigma.*unsupported"):
        _compiled(template, fake).fit(world)
    assert fake.payloads == []


@pytest.mark.parametrize("field", ["weibull_lam_range", "weibull_k_range"])
def test_changed_fixed_weibull_config_fails_before_binding(field):
    template, world = _shared_template_and_world()
    template = replace(template, config_contract={field: (2.0, 8.0)})
    setattr(world.cfg, field, (3.0, 8.0))
    fake = _FakeCompiled(_valid_tree())
    with pytest.raises(ValueError, match=f"{field}.*unsupported"):
        _compiled(template, fake).fit(world)
    assert fake.payloads == []


def _cfg_access_inventory(tree: ast.Module, seam: set[str]) -> dict[str, set[str]]:
    """Collect direct ``cfg.<field>`` accesses from each named source seam."""
    return {
        node.name: {
            child.attr
            for child in ast.walk(node)
            if isinstance(child, ast.Attribute)
            and isinstance(child.value, ast.Name)
            and child.value.id == "cfg"
        }
        for node in tree.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name in seam
    }


def _assert_cfg_accesses_classified(tree: ast.Module, seam: set[str], classified: set[str]) -> None:
    accessed = set().union(*_cfg_access_inventory(tree, seam).values())
    assert accessed, "the source seams must yield a nonempty cfg access inventory"
    assert accessed <= classified, f"unclassified cfg fields: {sorted(accessed - classified)}"


def test_oracle_config_classification_covers_builder_accesses_and_is_disjoint():
    """The inventory comes from the world-model builder seam, not its schema."""
    seam = {
        "_uniform_prior_specs",
        "_walk_priors",
        "build_oracle_model",
        "_validate_oracle_treatment_shocks",
    }
    tree = ast.parse(Path(world_model.__file__).read_text())
    inventory = _cfg_access_inventory(tree, seam)
    assert set(inventory) == seam
    assert inventory["_uniform_prior_specs"] >= {
        "carryover_alpha_range",
        "weibull_lam_range",
        "weibull_k_range",
    }
    assert inventory["_walk_priors"] >= {"rw_smoothness_max_weeks", "outcome_std_mode"}
    assert inventory["build_oracle_model"] >= {"l_max", "carryover_burn_in"}
    assert inventory["_validate_oracle_treatment_shocks"] >= {
        "n_treatment_shocks",
        "treatment_shock_level_range",
    }

    schema = oracle.ORACLE_CONFIG_SCHEMA
    topology = set(schema["topology"])
    fixed = set(schema["fixed_numeric"])
    runtime = set(schema["runtime_data"])
    validation_only = set(schema["validation_only"])
    classified = topology | fixed | validation_only
    assert topology.isdisjoint(fixed)
    assert topology.isdisjoint(runtime)
    assert fixed.isdisjoint(runtime)
    assert classified.isdisjoint(runtime)
    _assert_cfg_accesses_classified(tree, seam, classified)
    assert {"weibull_lam_range", "weibull_k_range"} <= fixed
    assert {"n_treatment_shocks", "treatment_shock_length_range"} <= validation_only

    synthetic = ast.parse("def build_oracle_model(cfg):\n    return cfg.synthetic_unclassified\n")
    with pytest.raises(AssertionError, match="synthetic_unclassified"):
        _assert_cfg_accesses_classified(synthetic, {"build_oracle_model"}, classified)


def test_compiled_fit_passes_adaptation_and_binds_requested_receipt():
    template, world = _shared_template_and_world()
    fake = _FakeCompiled(_valid_tree())
    result = _compiled(template, fake).fit(world, pg.OracleSamplingConfig(adaptation="draw_diag"))
    assert fake.sample_calls[0]["blocking"] is True
    assert fake.sample_calls[0]["adaptation"] == "draw_diag"
    receipt = result.receipt.to_dict()
    assert receipt["sampling"]["requested"]["adaptation"] == "draw_diag"
    assert receipt["sampling"]["effective"]["inference_library"] == "nutpie"
    assert (
        receipt["sampling"]["effective"]["inference_library_version"]
        == oracle._versions()["nutpie"]
    )
    assert receipt["identities"]["compile"]["owner_pid"] == __import__("os").getpid()


def test_compiled_fit_selector_uses_reported_row_coordinates():
    template, world = _shared_template_and_world(selected=(0, 2))
    fake = _FakeCompiled(_valid_tree())
    compiled = _compiled(template, fake)
    compiled.fit(world, observed_indices=[1, 2])
    assert np.array_equal(fake.payloads[0]["observed_indices_data"], [1, 2])
    with pytest.raises(ValueError, match="observed-index rank"):
        compiled.fit(world, observed_indices=[0])


def test_compiled_fit_rejects_latent_mismatch_before_binding():
    template, world = _shared_template_and_world()
    fake = _FakeCompiled(_valid_tree())
    compiled = _compiled(template, fake)
    with pytest.raises(ValueError, match="latent"):
        compiled.fit(world, pg.OracleSamplingConfig(latent="sampled"))
    assert fake.payloads == []


def test_template_rejects_missing_shared_variable():
    with oracle.pm.Model() as model:
        oracle.pm.Data("channels_data", np.zeros((4, 1)))
    with pytest.raises(ValueError, match="explicit data_contract"):
        pg.OracleTemplate(model, "sig", "marginal", (0,))


@pytest.mark.parametrize("shared_name", ["unexpected_runtime_scale", None])
def test_template_rejects_graph_connected_unexpected_shared_variable(shared_name):
    with oracle.pm.Model() as model:
        values = {
            "channels_data": np.zeros((4, 1)),
            "controls_data": np.ones((4, 1)),
            "sales_data": np.arange(4, dtype=float),
            "saturation_scale_data": np.ones(1),
            "g_cy_data": np.ones(1),
            "g_db_data": np.zeros(1),
            "g_zb_data": np.ones(1),
            "observed_indices_data": np.asarray([0, 1, 2], dtype="int64"),
        }
        for name, value in values.items():
            oracle.pm.Data(name, value)
        scale = pytensor.shared(np.ones(1), name=shared_name)
        oracle.pm.Normal(
            "connected_likelihood",
            mu=scale[0] * model["channels_data"][:, 0],
            sigma=1.0,
            observed=model["sales_data"],
        )
    with pytest.raises(ValueError, match="graph/unexpected extras|unnamed"):
        pg.OracleTemplate(
            model,
            "sig",
            "marginal",
            (0,),
            reported_rows=3,
            data_contract=oracle.ORACLE_SHARED_DATA_NAMES,
        )


def test_signature_excludes_compatible_prior_and_walk_values():
    _, world = _shared_template_and_world()
    world.extras["structural"] = {"carryover_family": np.asarray([1])}
    first = oracle._world_signature(world, latent="marginal", observed_count=3)
    world.extras["structural"].update(
        {"smoothness_d": np.asarray([0.1]), "smoothness_b": np.asarray([0.8])}
    )
    world.extras["prior_cond"] = {"carryover_alpha": (0.2, 0.3)}
    second = oracle._world_signature(world, latent="marginal", observed_count=3)
    assert first == second


def test_fixed_baseline_range_changes_signature_and_graph_parameterization():
    constant = _generated_world()
    smooth = copy.deepcopy(constant)
    constant.cfg.rw_baseline_std_range = (0.0, 0.0)
    smooth.cfg.rw_baseline_std_range = (0.1, 0.2)

    constant_signature = oracle._world_signature(constant, latent="marginal", observed_count=20)
    smooth_signature = oracle._world_signature(smooth, latent="marginal", observed_count=20)
    assert constant_signature != smooth_signature

    constant_model = constant.oracle_model(latent="marginal")
    smooth_model = smooth.oracle_model(latent="marginal")
    constant_free = {rv.name for rv in constant_model.free_RVs}
    smooth_free = {rv.name for rv in smooth_model.free_RVs}
    assert "rw_b_std_rel" not in constant_free
    assert "rw_b_std_rel" in smooth_free


@pytest.mark.parametrize("field", oracle.ORACLE_CONFIG_FIXED_NUMERIC_FIELDS)
def test_every_fixed_numeric_config_field_changes_signature(field):
    _, world = _shared_template_and_world()
    world.cfg = pg.SCMPrior()
    first = oracle._world_signature(world, latent="marginal", observed_count=3)
    changed = copy.deepcopy(world)
    value = getattr(changed.cfg, field)
    if field == "rw_baseline_std_sigma_effective":
        changed.cfg.rw_baseline_std_sigma = 0.1
    else:
        if isinstance(value, tuple):
            changed_value = tuple(float(item) + 0.1 for item in value)
        elif value is None:
            changed_value = 0.1
        else:
            changed_value = float(value) + 0.1
        setattr(changed.cfg, field, changed_value)
    assert oracle._world_signature(changed, latent="marginal", observed_count=3) != first


def test_fixed_signature_cache_compiles_constant_smooth_constant_twice():
    constant = _generated_world()
    smooth = copy.deepcopy(constant)
    constant.cfg.rw_baseline_std_range = (0.0, 0.0)
    smooth.cfg.rw_baseline_std_range = (0.1, 0.2)
    compiled_by_signature = {}
    compiled = []
    compile_calls = []
    for world in (constant, smooth, constant):
        signature = oracle._world_signature(world, latent="marginal", observed_count=20)
        if signature not in compiled_by_signature:
            compile_calls.append(signature)
            compiled_by_signature[signature] = object()
        compiled.append(compiled_by_signature[signature])
    assert len(compile_calls) == 2
    assert compiled[0] is compiled[2]
    assert compiled[0] is not compiled[1]


@pytest.mark.parametrize(
    "value",
    [
        True,
        np.bool_(True),
        float("nan"),
        float("inf"),
        np.asarray([np.nan]),
        np.asarray([object()], dtype=object),
        object(),
    ],
)
def test_invalid_fixed_numeric_config_values_fail_closed(value):
    _, world = _shared_template_and_world()
    world.cfg = SimpleNamespace(rw_std_sigma=value)
    with pytest.raises((TypeError, ValueError), match="fixed Oracle config field"):
        oracle._world_signature(world, latent="marginal", observed_count=3)


def test_signature_change_fails_closed_without_binding():
    template, world = _shared_template_and_world()
    _, changed = _shared_template_and_world()
    changed.g["g_cy"] = np.zeros(1)
    fake = _FakeCompiled(_valid_tree())
    compiled = _compiled(template, fake)
    with pytest.raises(ValueError, match="structural signature"):
        compiled.fit(changed)
    assert fake.payloads == []


def _generated_world(*, prior_conditioning=False):
    cfg = pg.make_scm_prior(
        n_time_steps=20,
        n_treatments=2,
        n_covariates=1,
        n_latent=1,
        n_cells=2,
        prior_conditioning=prior_conditioning,
    )
    return pg.sample_scm(cfg, seed=3, max_eps_draws=4)


def _evaluate_generated_template(template, worlds, *, point=None):
    logp = template.model.compile_logp()
    dynamic_names = set(template.data_contract or ()) - set(oracle.ORACLE_SHARED_DATA_NAMES)
    values = []
    for world in worlds:
        payload = oracle._oracle_payload(
            world,
            np.asarray(template.observed_indices, dtype="int64"),
            dynamic_names=dynamic_names,
        )
        oracle.pm.set_data(payload, model=template.model)
        values.append(float(logp(template.model.initial_point() if point is None else point)))
    return values


def test_generated_oracle_prior_interval_graph_restores_without_stale_data():
    world_a = _generated_world(prior_conditioning=True)
    world_b = copy.deepcopy(world_a)
    world_b.extras["prior_cond"] = {
        "carryover_alpha": (0.25, 0.10),
        "hill_shape": (2.1, 0.2),
    }
    for name in world_a.data:
        assert np.array_equal(world_a.data[name], world_b.data[name]), name

    template = pg.build_oracle_template(world_a, latent="marginal")
    dynamic_names = set(template.data_contract or ()) - set(oracle.ORACLE_SHARED_DATA_NAMES)
    point = template.model.initial_point()
    prior_logp = template.model.compile_logp(vars=[template.model["hill_slope"]], jacobian=False)
    values = []
    for world in (world_a, world_b, world_a):
        payload = oracle._oracle_payload(
            world,
            np.asarray(template.observed_indices, dtype="int64"),
            dynamic_names=dynamic_names,
        )
        oracle.pm.set_data(payload, model=template.model)
        values.append(float(prior_logp(point)))

    # The fixed point is graph-connected to the interval bounds, so this is a
    # prior-only proof that binding changes and is restored; sales_data never
    # changes in this A -> B -> A walk.
    assert values[1] != pytest.approx(values[0])
    assert values[2] == pytest.approx(values[0])


def test_generated_oracle_marginal_walk_graph_restores_without_stale_data():
    world_a = _generated_world()
    world_b = copy.deepcopy(world_a)
    world_b.extras["structural"]["smoothness_d"] = np.full_like(
        world_b.extras["structural"]["smoothness_d"], 0.95
    )
    world_b.extras["structural"]["smoothness_b"] = np.full_like(
        world_b.extras["structural"]["smoothness_b"], 0.95
    )
    template = pg.build_oracle_template(world_a, latent="marginal")
    first, changed, restored = _evaluate_generated_template(template, [world_a, world_b, world_a])
    assert changed != first
    assert restored == first


def test_generated_oracle_sampled_walk_graph_restores_without_stale_data():
    world_a = _generated_world()
    world_b = copy.deepcopy(world_a)
    world_b.extras["structural"]["smoothness_b"] = np.full_like(
        world_b.extras["structural"]["smoothness_b"], 0.95
    )
    template = pg.build_oracle_template(world_a, latent="sampled")
    point = template.model.initial_point()
    point["eps_b"] = np.ones_like(point["eps_b"])
    first, changed, restored = _evaluate_generated_template(
        template, [world_a, world_b, world_a], point=point
    )
    assert changed != first
    assert restored == first


def test_generated_oracle_point_mass_rejected_before_model_construction():
    world = _generated_world(prior_conditioning=False)
    world.extras["prior_cond"] = {"carryover_alpha": (0.3, 0.0)}
    world.oracle_model = lambda **kwargs: (_ for _ in ()).throw(AssertionError("built graph"))
    with pytest.raises(ValueError, match="zero-width.*point masses"):
        pg.build_oracle_template(world)


def test_generated_oracle_topology_change_splits_signature():
    interval = _generated_world(prior_conditioning=True)
    point = copy.deepcopy(interval)
    point.extras["prior_cond"]["carryover_alpha"] = (
        point.extras["prior_cond"]["carryover_alpha"][0],
        0.0,
    )
    assert oracle._world_signature(
        interval, latent="marginal", observed_count=20
    ) != oracle._world_signature(point, latent="marginal", observed_count=20)


def test_compiled_receipt_data_identities_are_stable():
    template, world = _shared_template_and_world()
    first = _compiled(template, _FakeCompiled(_valid_tree())).fit(world)
    second = _compiled(template, _FakeCompiled(_valid_tree())).fit(world)
    first_identities = first.receipt.to_dict()["identities"]
    second_identities = second.receipt.to_dict()["identities"]
    assert first_identities["compile"]["instance_id"] != second_identities["compile"]["instance_id"]
    assert first_identities["compile"]["owner_pid"] == second_identities["compile"]["owner_pid"]
    assert first.receipt.to_dict()["sampling"]["compile"] == first_identities["compile"]


def test_compile_and_sampling_exceptions_are_not_retried(monkeypatch):
    template, world = _shared_template_and_world()
    compile_calls = []

    def fail_compile(*args, **kwargs):
        compile_calls.append((args, kwargs))
        raise RuntimeError("compile failed")

    monkeypatch.setitem(
        __import__("sys").modules, "nutpie", SimpleNamespace(compile_pymc_model=fail_compile)
    )
    with pytest.raises(RuntimeError, match="compile failed"):
        pg.compile_oracle(template)
    assert len(compile_calls) == 1

    fake = _FakeCompiled(_valid_tree())
    fake.with_data = lambda **payload: (_ for _ in ()).throw(RuntimeError("bind failed"))
    with pytest.raises(RuntimeError, match="bind failed"):
        _compiled(template, fake).fit(world)


# Health tri-state semantics ------------------------------------------------


def test_health_available_threshold_violation_is_unhealthy(monkeypatch):
    _patch_arviz(monkeypatch, rhat=1.2, bulk=100, tail=100, bfmi=0.1)
    result, _ = _sample(monkeypatch, _valid_tree())
    health = result.receipt.to_dict()["health"]
    assert health["status"] == "unhealthy"
    assert [reason["code"] for reason in health["failure_reasons"]] == sorted(
        reason["code"] for reason in health["failure_reasons"]
    )


def test_health_available_failure_precedes_required_unavailable(monkeypatch):
    shape = (2, 4)
    tree = _tree(
        include_posterior=False,
        diverging=np.ones(shape, dtype=bool),
        energy=np.ones(shape),
        tree_depth=np.ones(shape, dtype=int),
        reached_max_treedepth=np.zeros(shape, dtype=bool),
    )
    result, _ = _sample(monkeypatch, tree)
    health = result.receipt.to_dict()["health"]
    assert health["status"] == "unhealthy"
    assert health["failure_reasons"]
    assert "rhat" in health["missing_diagnostics"]


def test_required_unavailable_stays_unknown_when_threshold_is_disabled(monkeypatch):
    tree = _tree(
        chains=1,
        draws=4,
        diverging=np.zeros((1, 4), dtype=bool),
        energy=np.ones((1, 4)),
        tree_depth=np.ones((1, 4), dtype=int),
        reached_max_treedepth=np.zeros((1, 4), dtype=bool),
    )
    criteria = pg.OracleHealthCriteria(max_rhat=None, require_rhat=True)
    _patch_arviz(monkeypatch)
    result, _ = _sample(monkeypatch, tree, criteria=criteria)
    health = result.receipt.to_dict()["health"]
    assert health["status"] == "unknown"
    assert "rhat" in health["missing_diagnostics"]


def test_required_invalid_stays_unknown_when_threshold_is_disabled(monkeypatch):
    criteria = pg.OracleHealthCriteria(max_rhat=None, require_rhat=True)
    _patch_arviz(monkeypatch, rhat=np.nan)
    result, _ = _sample(monkeypatch, _valid_tree(), criteria=criteria)
    health = result.receipt.to_dict()["health"]
    assert health["status"] == "unknown"
    assert "rhat" in health["missing_diagnostics"]


def test_health_failure_reasons_are_sorted_by_code_then_metric(monkeypatch):
    _patch_arviz(monkeypatch, rhat=2, bulk=1, tail=1)
    result, _ = _sample(monkeypatch, _valid_tree(), criteria=pg.OracleHealthCriteria(min_bfmi=None))
    reasons = result.receipt.to_dict()["health"]["failure_reasons"]
    assert [(item["code"], item["metric"]) for item in reasons] == sorted(
        (item["code"], item["metric"]) for item in reasons
    )
    assert all("observed" in item and "threshold" in item for item in reasons)


def test_health_required_missing_is_unknown_even_with_other_missing_metrics(monkeypatch):
    tree = _valid_tree(include_stats=False)
    result, _ = _sample(monkeypatch, tree)
    health = result.receipt.to_dict()["health"]
    assert health["status"] == "unknown"
    assert health["failure_reasons"] == []
    assert health["missing_diagnostics"] == sorted(health["missing_diagnostics"])


def test_health_unknown_when_no_failures_but_required_diagnostic_invalid(monkeypatch):
    _patch_arviz(monkeypatch, rhat=np.nan)
    result, _ = _sample(monkeypatch, _valid_tree())
    health = result.receipt.to_dict()["health"]
    assert health["status"] == "unknown"
    assert health["failure_reasons"] == []
    assert "rhat" in health["missing_diagnostics"]


def test_health_healthy_when_all_required_metrics_pass(monkeypatch):
    _patch_arviz(monkeypatch)
    result, _ = _sample(monkeypatch, _valid_tree())
    health = result.receipt.to_dict()["health"]
    assert health["status"] == "healthy"
    assert health["failure_reasons"] == []
    assert health["missing_diagnostics"] == []


def test_optional_missing_diagnostic_is_listed_but_stays_healthy(monkeypatch):
    criteria = pg.OracleHealthCriteria(require_tree_depth=False, require_bfmi=False)
    tree = _tree(diverging=np.zeros((2, 4), dtype=bool))
    _patch_arviz(monkeypatch)
    result, _ = _sample(monkeypatch, tree, criteria=criteria)
    health = result.receipt.to_dict()["health"]
    assert health["status"] == "healthy"
    assert "tree_depth_max" in health["missing_diagnostics"]
    assert "bfmi" in health["missing_diagnostics"]


def test_disabled_thresholds_do_not_create_failures(monkeypatch):
    criteria = pg.OracleHealthCriteria(
        max_divergences=None,
        max_rhat=None,
        min_ess_bulk=None,
        min_ess_tail=None,
        max_tree_depth_saturation=None,
        min_bfmi=None,
        require_divergences=False,
        require_rhat=False,
        require_ess_bulk=False,
        require_ess_tail=False,
    )
    _patch_arviz(monkeypatch, rhat=9, bulk=0, tail=0, bfmi=0)
    result, _ = _sample(monkeypatch, _valid_tree(), criteria=criteria)
    assert result.receipt.to_dict()["health"]["status"] == "healthy"


def test_custom_required_tree_depth_and_bfmi_are_enforced(monkeypatch):
    criteria = pg.OracleHealthCriteria(require_tree_depth=True, require_bfmi=True)
    tree = _tree(
        diverging=np.zeros((2, 4), dtype=bool),
        tree_depth=np.full((2, 4), 2),
        reached_max_treedepth=None,
        energy=None,
    )
    _patch_arviz(monkeypatch)
    result, _ = _sample(monkeypatch, tree, criteria=criteria)
    health = result.receipt.to_dict()["health"]
    assert health["status"] == "unknown"
    assert {"tree_depth_saturation", "bfmi"} <= set(health["missing_diagnostics"])
