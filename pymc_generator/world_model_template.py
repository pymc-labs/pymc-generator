"""Experimental max-size template model with reusable compilation.

Production corpus generation builds a graph per cell; this module is not its
drop-in replacement. It supports a narrower configuration set and has its own
random-number schedule. Benchmark both paths on concrete shared structures
before comparing performance; a shared seed alone does not align their draws.

:func:`build_world_model_template` instead builds ONE model at the layout's
maximum node counts, with every structural choice as a ``pm.Data`` input:

* the ``g`` edge blocks and the per-node activity masks,
* the carryover / saturation family ids,
* the random-walk kernel widths and opt-in trajectory inclusion flags.

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
from .slots import TRAJECTORY_COMPONENTS, TRAJECTORY_INPUTS
from .symbolic_graph import build_symbolic_graph, inactive_zero
from .trajectories import SCHEDULE_COMPONENTS, structural_key, trajectory_params
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

#: The base structural ``pm.Data`` slots compiled as positional draw inputs.
#: Config-admitted trajectory flags follow them in canonical component order.
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
    its own input slot. Composable trajectories and per-input texture inclusion
    are supported through optional runtime flags.
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


def _trajectory_input_names(cfg: SCMPrior) -> tuple[str, ...]:
    """Runtime flags only for admitted schedules and fractional texture."""
    probs = cfg.trajectory_inclusion_probs()
    sizes = {"treatment": cfg.layout.n_treatments, "covariate": cfg.layout.n_covariates}
    return tuple(
        structural_key(input_type, component)
        for input_type in TRAJECTORY_INPUTS
        for component in TRAJECTORY_COMPONENTS
        if sizes[input_type] > 0
        and (
            (component in SCHEDULE_COMPONENTS and probs[input_type][component] > 0.0)
            or 0.0 < probs[input_type][component] < 1.0
        )
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
    for name in _trajectory_input_names(cfg):
        input_type = "covariate" if name.startswith("use_covariate_") else "treatment"
        width = n_covariates if input_type == "covariate" else n_treatments
        payload[name] = _pad_to(structural[name], width, "float64")
    # ``pm.Data`` normalizes integer arrays (int64 -> int32), so route the payload
    # through the same conversion; otherwise the compiled function rejects these
    # arrays for risking a precision loss.
    return {name: np.asarray(convert_data(value)) for name, value in payload.items()}


def sample_cell_structures(cfg: SCMPrior, rng: np.random.Generator) -> list[dict[str, np.ndarray]]:
    """Draw all structures up front, using active node counts for prior shapes.

    Active counts follow ``cfg.active_count_allocation`` through the same
    helpers as production, so a stratified config covers the treatment ×
    covariate grid here exactly as it does in a corpus. Production instead
    interleaves each structure with draw-seed and support-mask sampling. After
    the first cell these RNG schedules differ, so equal initial seeds do not
    select the same sequence of production DAGs.
    """
    from .active_counts import draw_active_counts, plan_active_counts
    from .sampler import _slice_g_active, sample_g_additive
    from .world_model import sample_structure

    layout = cfg.layout
    plan = plan_active_counts(cfg, rng)
    cells: list[dict[str, np.ndarray]] = []
    for cell in range(cfg.n_cells):
        n_treatments_active, n_covariates_active, n_latent_active = draw_active_counts(
            cfg, rng, plan, cell
        )
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
        input_names = TEMPLATE_STRUCTURE_INPUT_NAMES + _trajectory_input_names(cfg)
        data = {name: pm.Data(name, init_inputs[name]) for name in input_names}
        g_data = {
            key: data[key]
            for key in ("g_cy", "g_dc", "g_dz", "g_dy", "g_zy", "g_zc", "g_cc", "g_zz")
        }
        active_data = {
            key: data[key] for key in ("active_treatment", "active_covariate", "active_latent")
        }

        # Smoothness reaches the graph as a kernel-width index rather than a
        # float. Constant texture inclusion stays concrete, so defaults build no
        # additional flag inputs or switches.
        probs = cfg.trajectory_inclusion_probs()
        sizes = {"treatment": n_treatments, "covariate": n_covariates}
        flags = {
            structural_key(input_type, component): data.get(
                structural_key(input_type, component),
                np.full(sizes[input_type], probs[input_type][component] == 1.0),
            )
            for input_type in TRAJECTORY_INPUTS
            for component in TRAJECTORY_COMPONENTS
        }
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
            use_hf=flags["use_hf"],
            use_pulse=flags["use_pulse"],
            use_covariate_hf=flags["use_covariate_hf"],
            use_covariate_pulse=flags["use_covariate_pulse"],
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
        trajectory = trajectory_params(
            cfg,
            flags,
            n_treatments,
            n_covariates,
            n_time_steps,
            dynamic_flags=True,
        )
        if trajectory is not None:
            params["trajectory"] = trajectory

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
        report_params = params
        if trajectory is not None:
            # Reports zero padded nodes without altering the parameter operands
            # used by the graph (e.g. periods must stay positive under switches).
            report_trajectory = {}
            for input_type, spec in trajectory.items():
                active_flag = active_data[f"active_{input_type}"]
                report_trajectory[input_type] = {
                    key: (
                        value
                        if key == "use"
                        else {
                            field: (
                                leaf
                                if field == "count"
                                else inactive_zero(active_flag, pt.as_tensor_variable(leaf))
                            )
                            for field, leaf in value.items()
                        }
                    )
                    for key, value in spec.items()
                }
            report_params = {**params, "trajectory": report_trajectory}
        param_names = _register_param_reports(
            report_params,
            rw,
            c_level,
            confounding_strength,
            cfg=cfg,
        )

    return model, out_names, param_names


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
    input_names: tuple[str, ...] | None = None,
    mode: str = "FAST_COMPILE",
) -> TemplateDrawFn:
    """Compile with only structure inputs reached by the requested outputs."""
    if input_names is None:
        input_names = TEMPLATE_STRUCTURE_INPUT_NAMES + tuple(
            name
            for input_type in TRAJECTORY_INPUTS
            for component in TRAJECTORY_COMPONENTS
            if (name := structural_key(input_type, component)) in model.named_vars
        )
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
