"""One-compile-per-shard world generation: a max-size template model.

:func:`prior_generator.world_model.build_world_model` builds and compiles a fresh
PyTensor graph for every world, because the DAG, the mechanism families, and the
random-walk kernel widths are all baked into the graph. Compilation then
dominates corpus generation — it is roughly 80% of the wall time on a production
shard, and it is repeated once per cell even though the cells differ only in
that structure.

:func:`build_world_model_template` instead builds ONE model at the layout's
maximum node counts, with every structural choice as a ``pm.Data`` input:

* the ``g`` edge blocks and the per-node activity masks,
* the adstock / saturation family ids,
* the random-walk kernel widths.

That model compiles once per shard and then draws every cell by swapping those
inputs. The cost is a denser graph — all candidate edges wired, all mechanism
families built behind ``pt.switch``, padded slots carried and zeroed rather than
omitted — traded against paying compilation once instead of once per cell.

Structure is passed as compiled-function arguments rather than via
``pm.set_data``. Both were measured at the same execute cost; explicit arguments
just make the contract visible and let the function be compiled eagerly.

Approaches that did not pay off, so they are deliberately absent (measured at 10
cells x 5 draws):

* One unified model over all cells at once: 11x SLOWER (482s vs 44s). The graph
  grows with the cell count, so compilation grows with it too.
* Vectorizing the draw axis inside the template: 0.22x net (114.2s vs 24.7s).
  Execute improved only 1.14x while compile cost 6x more — the serial draw loop
  was only ~60 ms/cell, so the SCM forward pass, not Python overhead, dominates.
* ``FAST_RUN`` instead of ``FAST_COMPILE``: compiles in 21s but the first
  execute never returned (3+ min, ~32 GiB resident).
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import numpy as np
import pymc as pm
import pytensor.tensor as pt
from pymc.pytensorf import convert_data

from .random_walk import walk_width_index
from .sampler import SCMPrior
from .symbolic_graph import build_symbolic_graph
from .world_model import (
    _apply_outcome_std_scale,
    _confounded_channel_eps,
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
    "g_db",
    "g_zb",
    "g_zc",
    "g_cc",
    "g_zz",
    "active_treatment",
    "active_covariate",
    "active_latent",
    "adstock_family",
    "sat_family",
    "rw_width_d",
    "rw_width_z",
    "rw_width_c",
    "rw_width_b",
)


def check_template_supported(cfg: SCMPrior) -> None:
    """Raise if ``cfg`` uses a feature the template path does not model yet.

    Channel shocks and prior conditioning both add per-world structure that is
    still baked into the graph, and a non-degenerate confounding range would need
    its own input slot. Recipes using them must stay on the per-world path.
    """
    if cfg.n_channel_shocks:
        raise ValueError("template generation does not support channel shocks yet")
    if cfg.prior_conditioning:
        raise ValueError("template generation does not support prior conditioning yet")
    if cfg.confounding_strength_range is not None:
        lo, hi = cfg.confounding_strength_range
        if float(lo) != float(hi):
            raise ValueError(
                "template generation requires a fixed confounding_strength; "
                f"got range {cfg.confounding_strength_range}"
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
    n_time_steps_full = cfg.n_time_steps + cfg.adstock_burn_in
    rw_max = cfg.rw_smoothness_max_weeks

    def _widths(key: str) -> np.ndarray:
        return walk_width_index(structural[key], n_time_steps_full, rw_smoothness_max_weeks=rw_max)

    payload = {
        "g_cy": np.asarray(g["g_cy"][:n_treatments], dtype="float64"),
        "g_dc": np.asarray(g["g_dc"][:n_latent, :n_treatments], dtype="float64"),
        "g_dz": np.asarray(g["g_dz"][:n_latent, :n_covariates], dtype="float64"),
        "g_db": np.asarray(g["g_db"][:n_latent], dtype="float64"),
        "g_zb": np.asarray(g["g_zb"][:n_covariates], dtype="float64"),
        "g_zc": np.asarray(g["g_zc"][:n_covariates, :n_treatments], dtype="float64"),
        "g_cc": np.asarray(g["g_cc"][:n_treatments, :n_treatments], dtype="float64"),
        "g_zz": np.asarray(g["g_zz"][:n_covariates, :n_covariates], dtype="float64"),
        "active_treatment": np.asarray(active["active_treatment"], dtype="float64"),
        "active_covariate": np.asarray(active["active_covariate"], dtype="float64"),
        "active_latent": np.asarray(active["active_latent"], dtype="float64"),
        "adstock_family": _pad_to(structural["adstock_family"], n_treatments, "int64"),
        "sat_family": _pad_to(structural["sat_family"], n_treatments, "int64"),
        "rw_width_d": _pad_to(_widths("smoothness_d"), n_latent, "int64"),
        "rw_width_z": _pad_to(_widths("smoothness_z"), n_covariates, "int64"),
        "rw_width_c": _pad_to(_widths("smoothness_c"), n_treatments, "int64"),
        "rw_width_b": _widths("smoothness_b"),
    }
    # ``pm.Data`` normalizes integer arrays (int64 -> int32), so route the payload
    # through the same conversion; otherwise the compiled function rejects these
    # arrays for risking a precision loss.
    return {name: np.asarray(convert_data(value)) for name, value in payload.items()}


def sample_cell_structures(cfg: SCMPrior, rng: np.random.Generator) -> list[dict[str, np.ndarray]]:
    """Draw every cell's structure, in the same RNG order as the per-world path.

    Mirrors the per-cell prologue of
    :func:`prior_generator.sampler._generate_corpus_additive`: active node counts,
    then the DAG, then the structural families and smoothness sampled against the
    ACTIVE-sliced DAG. Sampling against the active slice (not the padded one) is
    what keeps a template-generated cell's structure identical to the per-world
    path's for the same seed.
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
    containers; every value is replaced per cell. The priors, noise, and
    ``param_*`` reports are built by the same helpers
    :func:`prior_generator.world_model.build_world_model` uses, so the two paths
    draw the same parameters in the same order.

    Returns the model, its graph output names, and its ``param_*`` names.
    """
    check_template_supported(cfg)
    layout = cfg.layout
    n_treatments = layout.n_treatments
    n_covariates = layout.n_covariates
    n_latent = layout.n_latent
    burn_in = cfg.adstock_burn_in
    n_time_steps_full = n_time_steps + burn_in
    specs = _uniform_prior_specs(cfg, n_treatments, n_covariates, n_latent, None)

    with pm.Model() as model:
        data = {name: pm.Data(name, init_inputs[name]) for name in TEMPLATE_STRUCTURE_INPUT_NAMES}
        g_data = {
            key: data[key]
            for key in ("g_cy", "g_dc", "g_dz", "g_db", "g_zb", "g_zc", "g_cc", "g_zz")
        }
        active_data = {
            key: data[key] for key in ("active_treatment", "active_covariate", "active_latent")
        }

        # Smoothness reaches the graph as a kernel-width index rather than a
        # float, which is what keeps the walk operators out of the compiled
        # structure. Texture flags stay concrete: they gate whole terms, and the
        # supported configs enable them uniformly across channels.
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
            adstock_family=data["adstock_family"],
            sat_family=data["sat_family"],
            use_hf=_texture_flag(cfg.channel_hf_sigma_range, n_treatments),
            use_pulse=_texture_flag(cfg.channel_pulse_prob_range, n_treatments),
        )
        _apply_outcome_std_scale(cfg, rw, g_data["g_cy"], params["beta"])

        eps = _scm_eps(
            n_time_steps_full, n_treatments, n_covariates, n_latent, params["pulse_prob"]
        )
        confounding_strength = _confounded_channel_eps(cfg, eps)

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
        param_names = _register_param_reports(params, rw, c_level, confounding_strength)

    return model, out_names, param_names


def _texture_flag(value_range: tuple[float, float], n_treatments: int) -> np.ndarray:
    """Whether a channel-texture term is enabled anywhere in this config.

    The flag gates a whole additive term in the graph, so it must be concrete.
    A config whose upper bound is zero can never produce the term.
    """
    return np.full(n_treatments, float(value_range[1]) > 0.0)


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
    """Compile the template's draw function with structure as positional inputs."""
    with model:
        out_vars = [model[name] for name in out_names]
        input_vars = [model[name] for name in input_names]
    fn, ordered_rngs = _get_cached_draw_fn(
        model,
        out_vars,
        mode=mode,
        reference_vars=None,
        input_vars=input_vars,
    )
    return TemplateDrawFn(fn, ordered_rngs, tuple(out_names), tuple(input_names))
