"""Experimental max-size template model with reusable compilation.

Production corpus generation builds a graph per cell; this module is not its
drop-in replacement. It supports a narrower configuration set and has its own
random-number schedule. Benchmark both paths on concrete shared structures
before comparing performance; a shared seed alone does not align their draws.

:func:`build_world_model_template` instead builds ONE model at the layout's
maximum node counts, with every structural choice as a ``pm.Data`` input:

* the ``g`` edge blocks and the per-node activity masks,
* the carryover / saturation family ids,
* the random-walk kernel widths.

That model compiles once per shard and then draws every cell by swapping those
inputs. The cost is a denser graph — all candidate edges wired, all mechanism
families built behind ``pt.switch``, padded slots carried and zeroed rather than
omitted — traded against paying compilation once instead of once per cell.

Structure is passed as compiled-function arguments rather than via
``pm.set_data``, making the inputs explicit and allowing eager compilation.
The benchmark reports measured candidate-drawing costs for this implementation,
not end-to-end accepted-corpus throughput or projected production speedups.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import numpy as np
import pymc as pm
import pytensor.tensor as pt
from pymc.pytensorf import convert_data
from pytensor.graph import ancestors

from .random_walk import walk_width_index
from .sampler import SATURATION_FAMILY_KEYS, SCMPrior
from .symbolic_graph import build_symbolic_graph
from .world_model import (
    _apply_outcome_std_scale,
    _confounded_treatment_eps,
    _disabled_shock_outputs,
    _execute_draws,
    _get_cached_draw_fn,
    _register_param_reports,
    _scm_eps,
    _scm_params,
    _uniform_prior_specs,
    _walk_priors,
)

#: The structural ``pm.Data`` slots compiled as positional draw-function inputs,
#: in the order :func:`build_cell_inputs` emits them.
TEMPLATE_STRUCTURE_INPUT_NAMES: tuple[str, ...] = (
    "g_cy",
    "g_dc",
    "g_dz",
    "g_dy",
    "g_zy",
    "g_zc",
    "g_cc",
    "g_zz",
    "active_treatment",
    "active_covariate",
    "active_latent",
    "carryover_family",
    "sat_family",
    "rw_width_d",
    "rw_width_z",
    "rw_width_c",
    "rw_width_b",
)


def check_template_supported(cfg: SCMPrior) -> None:
    """Raise if ``cfg`` uses a feature the template path does not model yet.

    Treatment shocks and prior conditioning both add per-world structure that is
    still baked into the graph, and a non-degenerate confounding range would need
    its own input slot. Composable trajectory components and per-input hf/pulse
    inclusion are per-cell structure the template does not take as inputs yet
    (its texture flags are all-or-none). Recipes using them must stay on the
    per-world path.
    """
    if cfg.n_treatment_shocks:
        raise ValueError("template generation does not support treatment shocks yet")
    if cfg.prior_conditioning:
        raise ValueError("template generation does not support prior conditioning yet")
    if cfg.confounding_strength_range is not None:
        lo, hi = cfg.confounding_strength_range
        if float(lo) != float(hi):
            raise ValueError(
                "template generation requires a fixed confounding_strength; "
                f"got range {cfg.confounding_strength_range}"
            )
    if cfg.trajectory_components_enabled:
        raise ValueError(
            "template generation does not support composable trajectory components "
            "(onset, offset, flighting, level_jump, seasonal, trend) yet"
        )
    for input_type, probs in cfg.trajectory_inclusion_probs().items():
        for component, range_name in (("hf", "hf_sigma_range"), ("pulse", "pulse_prob_range")):
            # The template wires a texture term for every node whenever its
            # range is live (_texture_flag), i.e. it assumes inclusion 1.
            live = float(getattr(cfg, f"{input_type}_{range_name}")[1]) > 0.0
            if probs[component] != (1.0 if live else 0.0):
                raise ValueError(
                    "template generation requires all-or-none texture "
                    f"({input_type}_{component}_inclusion_prob=1.0), got "
                    f"{getattr(cfg, f'{input_type}_{component}_inclusion_prob')!r}"
                )


def _pad_to(values, width: int, dtype: str) -> np.ndarray:
    """Right-pad a per-node vector with zeros out to the template's max width."""
    out = np.zeros(width, dtype=dtype)
    src = np.asarray(values, dtype=dtype).ravel()
    out[: src.shape[0]] = src
    return out


def build_cell_inputs(
    cfg: SCMPrior,
    g: dict[str, np.ndarray],
    active: dict[str, np.ndarray],
    structural: dict,
) -> dict[str, np.ndarray]:
    """One cell's structure as the template's ``pm.Data`` payload.

    ``g`` must already be at the layout's MAXIMUM node counts (unsliced), since
    the template's tensors are that shape and inactive slots are switched off by
    the activity masks rather than removed.
    """
    layout = cfg.layout
    n_treatments = layout.n_treatments
    n_covariates = layout.n_covariates
    n_latent = layout.n_latent
    n_time_steps_full = cfg.n_time_steps + cfg.carryover_burn_in
    rw_max = cfg.rw_smoothness_max_weeks

    def _widths(key: str) -> np.ndarray:
        return walk_width_index(structural[key], n_time_steps_full, rw_smoothness_max_weeks=rw_max)

    payload = {
        "g_cy": np.asarray(g["g_cy"][:n_treatments], dtype="float64"),
        "g_dc": np.asarray(g["g_dc"][:n_latent, :n_treatments], dtype="float64"),
        "g_dz": np.asarray(g["g_dz"][:n_latent, :n_covariates], dtype="float64"),
        "g_dy": np.asarray(g["g_dy"][:n_latent], dtype="float64"),
        "g_zy": np.asarray(g["g_zy"][:n_covariates], dtype="float64"),
        "g_zc": np.asarray(g["g_zc"][:n_covariates, :n_treatments], dtype="float64"),
        "g_cc": np.asarray(g["g_cc"][:n_treatments, :n_treatments], dtype="float64"),
        "g_zz": np.asarray(g["g_zz"][:n_covariates, :n_covariates], dtype="float64"),
        "active_treatment": np.asarray(active["active_treatment"], dtype="float64"),
        "active_covariate": np.asarray(active["active_covariate"], dtype="float64"),
        "active_latent": np.asarray(active["active_latent"], dtype="float64"),
        "carryover_family": _pad_to(structural["carryover_family"], n_treatments, "int64"),
        "sat_family": _pad_to(structural["sat_family"], n_treatments, "int64"),
        "rw_width_d": _pad_to(_widths("smoothness_d"), n_latent, "int64"),
        "rw_width_z": _pad_to(_widths("smoothness_z"), n_covariates, "int64"),
        "rw_width_c": _pad_to(_widths("smoothness_c"), n_treatments, "int64"),
        "rw_width_b": _widths("smoothness_b"),
    }
    if cfg.treatment_reference_contribution_range is not None:
        # Synthetic slots must stay inside the validated coefficient support.
        padding_family = next(
            i
            for i, family in enumerate(SATURATION_FAMILY_KEYS)
            if cfg.saturation_family_probs[family] > 0.0
        )
        payload["sat_family"][len(structural["sat_family"]) :] = padding_family
    # ``pm.Data`` normalizes integer arrays (int64 -> int32), so route the payload
    # through the same conversion; otherwise the compiled function rejects these
    # arrays for risking a precision loss.
    return {name: np.asarray(convert_data(value)) for name, value in payload.items()}


def sample_cell_structures(cfg: SCMPrior, rng: np.random.Generator) -> list[dict[str, np.ndarray]]:
    """Draw all structures up front, using active node counts for prior shapes.

    Production instead interleaves each structure with draw-seed and support-mask
    sampling. After the first cell these RNG schedules differ, so equal initial
    seeds do not select the same sequence of production DAGs.
    """
    from .sampler import _slice_g_active, sample_g_additive
    from .world_model import sample_structure

    layout = cfg.layout
    cells: list[dict[str, np.ndarray]] = []
    for _ in range(cfg.n_cells):
        tr = cfg.n_treatments_active_range_effective
        cv = cfg.n_covariates_active_range_effective
        lt = cfg.n_latent_active_range_effective
        n_treatments_active = int(rng.integers(tr[0], tr[1] + 1))
        n_covariates_active = int(rng.integers(cv[0], cv[1] + 1))
        n_latent_active = int(rng.integers(lt[0], lt[1] + 1))
        g = sample_g_additive(
            rng,
            cfg,
            layout,
            n_treatments_active=n_treatments_active,
            n_covariates_active=n_covariates_active,
            n_latent_active=n_latent_active,
        )
        g_act = _slice_g_active(g, n_treatments_active, n_covariates_active, n_latent_active)
        structural = sample_structure(g_act, cfg, rng)
        active = {
            "active_treatment": g["active_treatment"],
            "active_covariate": g["active_covariate"],
            "active_latent": g["active_latent"],
        }
        cells.append(build_cell_inputs(cfg, g, active, structural))
    return cells


def build_world_model_template(
    cfg: SCMPrior,
    init_inputs: dict[str, np.ndarray],
    n_time_steps: int,
) -> tuple[pm.Model, tuple[str, ...], tuple[str, ...]]:
    """One max-size model whose structure arrives as ``pm.Data`` at draw time.

    ``init_inputs`` (from :func:`build_cell_inputs`) only sizes and seeds the data
    containers; every value is replaced per cell. Priors and noise use the same
    helper definitions as :func:`pymc_generator.world_model.build_world_model`,
    but padded RV shapes do not promise samplewise equivalence to that path.

    Returns the model, its graph output names, and its ``param_*`` names.
    """
    check_template_supported(cfg)
    layout = cfg.layout
    n_treatments = layout.n_treatments
    n_covariates = layout.n_covariates
    n_latent = layout.n_latent
    burn_in = cfg.carryover_burn_in
    n_time_steps_full = n_time_steps + burn_in
    specs = _uniform_prior_specs(cfg, n_treatments, n_covariates, n_latent, None)

    with pm.Model() as model:
        data = {name: pm.Data(name, init_inputs[name]) for name in TEMPLATE_STRUCTURE_INPUT_NAMES}
        g_data = {
            key: data[key]
            for key in ("g_cy", "g_dc", "g_dz", "g_dy", "g_zy", "g_zc", "g_cc", "g_zz")
        }
        active_data = {
            key: data[key] for key in ("active_treatment", "active_covariate", "active_latent")
        }

        # Smoothness reaches the graph as a kernel-width index rather than a
        # float, which is what keeps the walk operators out of the compiled
        # structure. Texture flags stay concrete: they gate whole terms, and the
        # supported configs enable them uniformly across treatments and covariates.
        rw = _walk_priors(
            cfg,
            {},
            n_treatments,
            n_covariates,
            n_latent,
            width_indices={
                "rw_d": data["rw_width_d"],
                "rw_z": data["rw_width_z"],
                "rw_c": data["rw_width_c"],
                "rw_b": data["rw_width_b"],
            },
        )
        c_level = pt.softplus(rw["rw_c"]["mean"])
        params = _scm_params(
            cfg,
            specs,
            rw,
            c_level,
            g=g_data,
            carryover_family=data["carryover_family"],
            sat_family=data["sat_family"],
            use_hf=_texture_flag(cfg.treatment_hf_sigma_range, n_treatments),
            use_pulse=_texture_flag(cfg.treatment_pulse_prob_range, n_treatments),
            use_covariate_hf=_texture_flag(cfg.covariate_hf_sigma_range, n_covariates),
            use_covariate_pulse=_texture_flag(cfg.covariate_pulse_prob_range, n_covariates),
            dynamic_family=True,
        )
        _apply_outcome_std_scale(cfg, rw, g_data["g_cy"], params["beta"])

        eps = _scm_eps(
            n_time_steps_full,
            n_treatments,
            n_covariates,
            n_latent,
            params["pulse_prob"],
            params["covariate_pulse_prob"],
        )
        confounding_strength = _confounded_treatment_eps(cfg, eps)

        graph = build_symbolic_graph(
            g_data,
            params,
            n_time_steps,
            n_treatments,
            n_covariates,
            n_latent,
            burn_in=burn_in,
            eps=eps,
            active=active_data,
            dynamic_g=True,
        )
        outputs = graph["outputs"]
        outputs["confounding_strength"] = confounding_strength
        outputs.update(_disabled_shock_outputs(n_time_steps, n_time_steps_full, n_treatments))
        out_names = tuple(outputs)
        for name in out_names:
            pm.Deterministic(name, outputs[name])
        param_names = _register_param_reports(
            params,
            rw,
            c_level,
            confounding_strength,
            cfg=cfg,
        )

    return model, out_names, param_names


def _texture_flag(value_range: tuple[float, float], n_nodes: int) -> np.ndarray:
    """Whether a texture term is enabled anywhere in this config.

    The flag gates a whole additive term in the graph, so it must be concrete.
    A config whose upper bound is zero can never produce the term.
    """
    return np.full(n_nodes, float(value_range[1]) > 0.0)


@dataclass(frozen=True)
class TemplateDrawFn:
    """A compiled template draw function bound to its output and input names.

    The names travel with the function deliberately. The compiled function
    returns a bare positional list, and the output order also fixes PyTensor's
    RNG traversal, so labelling the results with a different name tuple than the
    one compiled would mislabel every array without raising.
    """

    fn: Callable[..., Any]
    ordered_rngs: list
    out_names: tuple[str, ...]
    input_names: tuple[str, ...]

    def __call__(
        self, cell_inputs: dict[str, np.ndarray], seed: int, draws: int = 1
    ) -> dict[str, np.ndarray]:
        """Draw ``draws`` worlds for one cell, with a ``(draws, ...)`` contract."""
        input_args = tuple(np.asarray(cell_inputs[name]) for name in self.input_names)
        vals = _execute_draws(
            self.fn, self.ordered_rngs, seed=seed, draws=draws, input_args=input_args
        )
        return {
            name: (np.asarray(value)[None] if draws == 1 else np.asarray(value))
            for name, value in zip(self.out_names, vals)
        }


def compile_template_draw_fn(
    model: pm.Model,
    out_names: tuple[str, ...],
    *,
    input_names: tuple[str, ...] = TEMPLATE_STRUCTURE_INPUT_NAMES,
    mode: str = "FAST_COMPILE",
) -> TemplateDrawFn:
    """Compile with only structure inputs reached by the requested outputs."""
    with model:
        out_vars = [model[name] for name in out_names]
        input_vars = [model[name] for name in input_names]
    # One-slot graphs have no C->C / Z->Z recursion, and parameter-only draws
    # may read no structure at all. Do not pass absent dependencies to compile.
    reached = set(ancestors(out_vars))
    inputs = [(name, var) for name, var in zip(input_names, input_vars) if var in reached]
    input_names = tuple(name for name, _ in inputs)
    input_vars = [var for _, var in inputs]
    fn, ordered_rngs = _get_cached_draw_fn(
        model,
        out_vars,
        mode=mode,
        reference_vars=None,
        input_vars=input_vars,
    )
    return TemplateDrawFn(fn, ordered_rngs, tuple(out_names), tuple(input_names))
