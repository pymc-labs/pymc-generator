"""Compare per-cell compilation with a reusable template on identical structures.

Uses a representative direct-effect prior (geometric x michaelis-menten,
K=10/M=10/J=1). Both paths receive the same precomputed DAGs, mechanism families,
walk widths and draw seeds, and request the full corpus output set.

This measures candidate drawing, not accepted-corpus throughput: structure
sampling, realism filtering, retries, normalization and serialization are
excluded. Padded template RVs have different shapes; identical seeds do not
imply identical numerical draws across these two implementations.

Usage:
  python benchmarks/template_vs_per_world.py
  python benchmarks/template_vs_per_world.py --cells 25 50 --draws-per-cell 5
"""

from __future__ import annotations

import argparse
import time
from dataclasses import dataclass

import numpy as np

from pymc_generator import make_scm_prior
from pymc_generator.sampler import (
    _ADDITIVE_OUT_NAMES,
    _CORPUS_PARAM_NAMES,
    _CORPUS_SHOCK_NAMES,
    _slice_g_active,
    sample_g_additive,
)
from pymc_generator.world_model import (
    build_world_model,
    draw_worlds,
    reset_world_model_caches,
    sample_structure,
    set_compile_cache_enabled,
)
from pymc_generator.world_model_template import (
    build_cell_inputs,
    build_world_model_template,
    compile_template_draw_fn,
)

CORPUS_NAMES = _CORPUS_PARAM_NAMES + _CORPUS_SHOCK_NAMES + _ADDITIVE_OUT_NAMES


def dea_prior(*, n_cells: int, draws_per_cell: int = 5, seed: int = 999000):
    """Representative direct-effect workload with variable active node counts."""
    return make_scm_prior(
        n_treatments=10,
        n_covariates=10,
        n_latent=1,
        n_time_steps=104,
        l_max=4,
        carryover_burn_in=4,
        n_cells=n_cells,
        draws_per_cell=draws_per_cell,
        seed=seed,
        n_treatments_active_range=(3, 10),
        n_covariates_active_range=(1, 10),
        n_latent_active_range=(1, 1),
        carryover_alpha_range=(0.2, 0.8),
        weibull_lam_range=(2.0, 8.0),
        weibull_k_range=(1.5, 4.0),
        beta_additive_range=(0.5, 2.0),
        dc_coeff_range=(0.1, 0.5),
        dz_coeff_range=(0.1, 0.5),
        zc_coeff_range=(0.05, 0.3),
        cc_coeff_range=(0.05, 0.3),
        zz_coeff_range=(-0.2, 0.2),
        dy_coeff_range=(0.15, 0.45),
        zy_coeff_range=(0.1, 0.4),
        rw_covariate_mean_range=(-1.0, 1.0),
        rw_positive_mean_range=(0.3, 4.0),
        rw_baseline_mean_range=(3.0, 8.0),
        rw_baseline_std_range=(0.05, 0.2),
        rw_outcome_std_range=(0.01, 0.028),
        rw_treatment_std_range=(0.15, 0.8),
        treatment_hf_sigma_range=(0.08, 0.6),
        treatment_pulse_prob_range=(0.0, 0.25),
        treatment_pulse_amp_range=(0.4, 2.5),
        treatment_shock_length_range=(2, 2),
        treatment_shock_level_range=(0.0, 0.0),
        edge_budget={
            "cy": (1, 10),
            "zy": (1, 10),
            "dc": 0,
            "dz": 0,
            "dy": 0,
            "zc": 0,
            "cc": 0,
            "zz": 0,
        },
        confounding_strength_range=(0.0, 0.0),
        min_no_direct_effect_treatments=1,
        carryover_family_probs={"none": 0.0, "geometric": 1.0, "weibull": 0.0},
        saturation_family_probs={
            "linear": 0.0,
            "hill": 0.0,
            "logistic": 0.0,
            "michaelis_menten": 1.0,
            "tanh": 0.0,
            "root": 0.0,
        },
    )


@dataclass(frozen=True)
class PathTiming:
    label: str
    n_cells: int
    draws_per_cell: int
    build_s: float
    compile_s: float
    execute_s: float
    total_s: float
    n_worlds: int

    @property
    def worlds_per_min(self) -> float:
        return self.n_worlds / self.total_s * 60.0 if self.total_s else 0.0

    @property
    def per_world_ms(self) -> float:
        return self.total_s / self.n_worlds * 1000.0 if self.n_worlds else 0.0


@dataclass(frozen=True)
class CellWorkload:
    g: dict[str, np.ndarray]
    structure: dict
    template_inputs: dict[str, np.ndarray]
    seed: int


def make_workload(cfg) -> list[CellWorkload]:
    """Prepare concrete inputs once, outside both timed paths."""
    rng = np.random.default_rng(cfg.seed)
    cells = []
    for _ in range(cfg.n_cells):
        counts = [
            int(rng.integers(lo, hi + 1))
            for lo, hi in (
                cfg.n_treatments_active_range_effective,
                cfg.n_covariates_active_range_effective,
                cfg.n_latent_active_range_effective,
            )
        ]
        g = sample_g_additive(rng, cfg, cfg.layout, *counts)
        g_active = _slice_g_active(g, *counts)
        structure = sample_structure(g_active, cfg, rng)
        cells.append(
            CellWorkload(
                g_active,
                structure,
                build_cell_inputs(cfg, g, g, structure),
                int(rng.integers(2**31 - 1)),
            )
        )
    return cells


def run_production(cfg, cells: list[CellWorkload], *, draw_names: tuple[str, ...]) -> PathTiming:
    """Build and draw each precomputed cell independently, without acceptance."""
    reset_world_model_caches()
    set_compile_cache_enabled(True)

    t_build = 0.0
    t0_total = time.perf_counter()

    for cell in cells:
        t0 = time.perf_counter()
        model, _out, _param = build_world_model(cell.g, cfg, cell.structure, cfg.n_time_steps)
        t_build += time.perf_counter() - t0

        draw_worlds(model, draw_names, seed=cell.seed, draws=cfg.draws_per_cell)

    total = time.perf_counter() - t0_total
    draw_s = total - t_build
    return PathTiming(
        label="per-cell compile (pre-filter)",
        n_cells=cfg.n_cells,
        draws_per_cell=cfg.draws_per_cell,
        build_s=t_build,
        compile_s=draw_s,
        execute_s=0.0,
        total_s=total,
        n_worlds=cfg.n_cells * cfg.draws_per_cell,
    )


def run_template(cfg, cells: list[CellWorkload], *, draw_names: tuple[str, ...]) -> PathTiming:
    """One template build + compile, then per-cell draws."""
    reset_world_model_caches()
    set_compile_cache_enabled(True)

    t0 = time.perf_counter()
    model, _out, _param = build_world_model_template(
        cfg, cells[0].template_inputs, cfg.n_time_steps
    )
    t_build = time.perf_counter() - t0

    t0 = time.perf_counter()
    draw = compile_template_draw_fn(model, draw_names)
    t_compile = time.perf_counter() - t0

    t0 = time.perf_counter()
    for cell in cells:
        draw(cell.template_inputs, seed=cell.seed, draws=cfg.draws_per_cell)
    t_execute = time.perf_counter() - t0

    return PathTiming(
        label="template (one compile per shard)",
        n_cells=cfg.n_cells,
        draws_per_cell=cfg.draws_per_cell,
        build_s=t_build,
        compile_s=t_compile,
        execute_s=t_execute,
        total_s=t_build + t_compile + t_execute,
        n_worlds=cfg.n_cells * cfg.draws_per_cell,
    )


def print_timing(t: PathTiming) -> None:
    print(
        f"  {t.label}: total {t.total_s:.1f}s | {t.worlds_per_min:.1f} candidates/min | "
        f"{t.per_world_ms:.0f} ms/candidate | candidates={t.n_worlds}"
    )
    if t.label.startswith("template"):
        print(
            f"    build {t.build_s:.1f}s | compile {t.compile_s:.1f}s | execute {t.execute_s:.1f}s"
        )
    else:
        print(f"    model build {t.build_s:.1f}s | draw (compile+execute) {t.compile_s:.1f}s")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--cells",
        type=int,
        nargs="+",
        default=[25, 50],
        help="Cell counts to benchmark (default: 25 50)",
    )
    parser.add_argument("--draws-per-cell", type=int, default=5)
    parser.add_argument("--seed", type=int, default=999000)
    args = parser.parse_args()

    print("=== deA prior: template vs production ===\n")
    print(f"Output set: {len(CORPUS_NAMES)} corpus names\n")
    print("Pre-filter candidate drawing only; identical precomputed structures and seeds.\n")

    for n_cells in args.cells:
        cfg = dea_prior(
            n_cells=n_cells,
            draws_per_cell=args.draws_per_cell,
            seed=args.seed,
        )
        print(
            f"--- {n_cells} cells x {args.draws_per_cell} draws = {n_cells * args.draws_per_cell} worlds ---"
        )
        cells = make_workload(cfg)
        prod = run_production(cfg, cells, draw_names=CORPUS_NAMES)
        templ = run_template(cfg, cells, draw_names=CORPUS_NAMES)
        print_timing(prod)
        print_timing(templ)
        speedup = prod.total_s / templ.total_s if templ.total_s else 0.0
        print(f"  → speedup: {speedup:.2f}x\n")


if __name__ == "__main__":
    main()
