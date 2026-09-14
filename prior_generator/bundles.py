"""SCM bundles: one folder per world that a human can audit end-to-end.

Each bundle contains:

* ``dataset.csv``           — what a model eats: week, spend_C*, control_Z*, sales_Y
* ``true_components.csv``   — the full additive truth: baseline_intrinsic,
  sales_noise, per-confounder and per-control contributions, per-channel
  DIRECT contributions, indirect effects by source (cc/zc/dc), plus the
  latent demand and base-channel series. Every column except ``week``,
  ``sales_reconstructed`` and the ``demand_*``/``channel_base_*`` diagnostics
  is an ADDITIVE term of ``sales_Y``: they sum to it exactly (this is
  ``SCM.reconstruction()``, the quantity ``SCM.identity_error()`` scores).
* ``description.txt``       — DAG edges with drawn coefficients, per-channel
  mechanism/texture parameters, decomposition identity check, signal metrics
* ``dag.dot`` / ``dag.png`` — the causal graph (matplotlib render; no graphviz)
* ``timeseries.png``        — model-input series
* ``decomposition.png``     — every true effect on Y + reconstruction check
* ``channels.png``          — per-channel spend vs true contribution (indexed)

:func:`write_scenario_bundles` reproduces structural-pfn's five-scenario
inspection layout (one numbered folder per scenario + a root README).
"""

from __future__ import annotations

import json
import platform
from collections.abc import Sequence
from dataclasses import asdict
from importlib.metadata import version
from pathlib import Path

import numpy as np
import pandas as pd

from . import __version__
from .describe import describe_scm, world_to_dot
from .sampler import SCMPrior
from .scenarios import SCENARIOS, Scenario
from .worlds import SCM, draw_feasible_graph, sample_scm


def _require_empty_destination(path: Path) -> None:
    if path.exists() and (not path.is_dir() or any(path.iterdir())):
        raise FileExistsError(f"bundle destination must be absent or empty: {path}")


def write_scm_bundle(
    world: SCM,
    out_dir: str | Path,
    *,
    title: str | None = None,
    plots: bool = True,
) -> Path:
    """Write one world's full audit bundle into ``out_dir``.

    Parameters
    ----------
    world : SCM
        A sampled world (see :func:`prior_generator.sample_scm`).
    out_dir : path-like
        Empty target directory (created if missing). Nonempty targets are refused.
    title : str, optional
        Figure/description title; defaults to ``world.name``.
    plots : bool
        Also render the four PNG figures (requires matplotlib; set False for
        a text/CSV-only bundle).

    Returns
    -------
    Path
        The bundle directory.
    """
    d = world.data
    n_treatments, n_covariates, n_latent, n_time_steps = (
        world.n_treatments,
        world.n_covariates,
        world.n_latent,
        world.n_time_steps,
    )
    weeks = np.arange(n_time_steps)
    title = world.name if title is None else title

    out = Path(out_dir)
    _require_empty_destination(out)
    out.mkdir(parents=True, exist_ok=True)

    recon = world.reconstruction()

    model_cols = {"week": weeks}
    model_cols.update({f"spend_C{k + 1}": d["channels"][:, k] for k in range(n_treatments)})
    model_cols.update({f"control_Z{m + 1}": d["controls"][:, m] for m in range(n_covariates)})
    model_cols["sales_Y"] = d["sales"]
    pd.DataFrame(model_cols).to_csv(out / "dataset.csv", index=False)

    truth_cols = {"week": weeks, "baseline_intrinsic": d["baseline_intrinsic"]}
    # The observation noise is an additive term of sales like any other, so it
    # belongs in the exported truth: without it the additive columns sum to
    # sales MINUS the noise and an auditor reads a residual of max|sales_noise|
    # where SCM.identity_error() reports ~1e-15.
    truth_cols["sales_noise"] = d["sales_noise"]
    truth_cols.update(
        {
            f"confounder_contribution_D{j + 1}": d["confounder_contribution"][:, j]
            for j in range(n_latent)
        }
    )
    truth_cols.update(
        {
            f"control_contribution_Z{m + 1}": d["control_contribution"][:, m]
            for m in range(n_covariates)
        }
    )
    truth_cols.update(
        {f"contribution_C{k + 1}": d["contributions"][:, k] for k in range(n_treatments)}
    )
    for i, src in enumerate(("cc", "zc", "dc")):
        truth_cols[f"indirect_{src}"] = d["indirect_effects_by_source"][:, i]
    truth_cols["sales_reconstructed"] = recon
    truth_cols.update({f"demand_D{j + 1}": d["demand"][:, j] for j in range(n_latent)})
    truth_cols.update(
        {f"channel_base_C{k + 1}": d["channels_base"][:, k] for k in range(n_treatments)}
    )
    pd.DataFrame(truth_cols).to_csv(out / "true_components.csv", index=False)

    (out / "description.txt").write_text(describe_scm(world))
    (out / "dag.dot").write_text(world_to_dot(world))

    if plots:
        from . import viz

        viz.plot_dag(world, str(out / "dag.png"), title)
        viz.plot_timeseries(world, str(out / "timeseries.png"), title)
        viz.plot_decomposition(world, str(out / "decomposition.png"), title)
        viz.plot_channels(world, str(out / "channels.png"), title)

    return out


def write_scenario_bundles(
    out_root: str | Path,
    *,
    scenarios: Sequence[Scenario] = SCENARIOS,
    n_time_steps: int = 104,
    seed: int = 20260712,
    require_path_to_y: bool = False,
    plots: bool = True,
    verbose: bool = True,
) -> list[Path]:
    """Write one bundle per scenario (numbered folders) plus a root README.

    Scenario ``idx`` is sampled with ``seed + idx``. Reproduction requires the
    recorded effective priors, connectivity policy, and numerical environment.

    Parameters
    ----------
    out_root : path-like
        Root output directory.
    scenarios : sequence of Scenario
        Defaults to the five audit scenarios (:data:`SCENARIOS`).
    n_time_steps : int
        Weeks per world.
    seed : int
        Base seed; scenario ``idx`` uses ``seed + idx``.
    require_path_to_y : bool
        Force full connectivity in EVERY scenario (overrides each scenario's
        ``connect_all``; removes the deliberate isolated-null traps). Each
        scenario's ``connect_all_edge_budget`` then SUBSTITUTES its
        connectivity-feasible budget, because a budget tuned for isolated-null
        traps can put "every node reaches Y" out of reach — thinly budgeted,
        even permanently (see :class:`~prior_generator.scenarios.Scenario`).
    plots : bool
        Render PNG figures per bundle.
    verbose : bool
        Print one line per bundle.

    Returns
    -------
    list of Path
        The bundle directories, in scenario order.

    Raises
    ------
    RuntimeError
        When graph search exhausts its budget for a scenario. Graph search is
        checked before creating output directories; later sampling, rendering,
        or filesystem errors may still leave a partial output.
    FileExistsError
        If the root already contains files.
    """
    out_root = Path(out_root)
    _require_empty_destination(out_root)
    # Probe the same seeded graph search before writing any scenario.
    plans: list[tuple[Scenario, SCMPrior, bool]] = []
    infeasible: list[str] = []
    for idx, sc in enumerate(scenarios):
        connect_all = True if require_path_to_y else sc.connect_all
        cfg = sc.prior(n_time_steps=n_time_steps, seed=seed + idx, connect_all=connect_all)
        plans.append((sc, cfg, connect_all))
        probe = draw_feasible_graph(cfg, np.random.default_rng(seed + idx), connect_all=connect_all)
        if probe is None:
            infeasible.append(f"[{idx}] {sc.name!r}")
    if infeasible:
        raise RuntimeError(
            "no DAG satisfies the connectivity rule for "
            + ", ".join(infeasible)
            + f" (require_path_to_y={require_path_to_y}); nothing was written. "
            "Widen the scenario's edge_budget, or its connect_all_edge_budget "
            "when only forced connectivity is infeasible."
        )

    out_root.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    for idx, (sc, cfg, connect_all) in enumerate(plans):
        world = sample_scm(
            cfg,
            seed=seed + idx,
            connect_all=connect_all,
            name=sc.name,
            purpose=sc.purpose,
        )
        out = write_scm_bundle(world, out_root / str(idx), title=f"{idx}: {sc.name}", plots=plots)
        if verbose:
            print(f"[{idx}] {sc.name}: identity err {world.identity_error():.1e} -> {out}")
        written.append(out)

    manifest = {
        "schema_version": 1,
        "package_version": __version__,
        "environment": {
            "python": platform.python_version(),
            **{
                name: version(name)
                for name in ("numpy", "scipy", "pymc", "pytensor", "pymc-marketing")
            },
        },
        "generation": {
            "seed": seed,
            "n_time_steps": n_time_steps,
            "require_path_to_y": require_path_to_y,
            "plots": plots,
        },
        "scenarios": [
            {
                "folder": str(idx),
                "name": sc.name,
                "purpose": sc.purpose,
                "seed": seed + idx,
                "connect_all": connect_all,
                "prior": asdict(cfg),
            }
            for idx, (sc, cfg, connect_all) in enumerate(plans)
        ],
    }
    (out_root / "recipe.json").write_text(json.dumps(manifest, indent=2) + "\n")
    lines = [
        "# Inspection datasets",
        "",
        f"Generated with prior-generator {__version__}. Full effective priors and",
        "numerical-library versions are recorded in `recipe.json`.",
        "Reproduce with the same package revision and numerical environment.",
        "",
    ]
    if tuple(scenarios) == SCENARIOS:
        command = (
            f"prior-generator --out reproduced-inspection-datasets --seed {seed} --t {n_time_steps}"
        )
        if require_path_to_y:
            command += " --require-path-to-y"
        if not plots:
            command += " --no-plots"
        lines.extend(["```bash", command, "```", ""])
    lines.extend(
        [
            "For any recipe, including custom scenarios, run from this directory:",
            "",
            "```python",
            "import json",
            "from pathlib import Path",
            "import prior_generator as pg",
            "",
            'recipe = json.loads(Path("recipe.json").read_text())',
            'for entry in recipe["scenarios"]:',
            '    world = pg.sample_scm(pg.SCMPrior(**entry["prior"]), seed=entry["seed"],',
            '                          connect_all=entry["connect_all"],',
            '                          name=entry["name"], purpose=entry["purpose"])',
            '    pg.write_scm_bundle(world, Path("reproduced") / entry["folder"],',
            '                        title=entry["folder"] + ": " + entry["name"],',
            '                        plots=recipe["generation"]["plots"])',
            "```",
            "",
            "| # | scenario | isolates |",
            "|---|---|---|",
            *[f"| {idx} | {sc.name} | {sc.purpose} |" for idx, sc in enumerate(scenarios)],
            "",
            "Per folder: `dataset.csv`, `true_components.csv`,",
            "`description.txt`, and `dag.dot`.",
        ]
    )
    if plots:
        lines.append(
            "Figures: `dag.png`, `timeseries.png`, `decomposition.png`, and `channels.png`."
        )
    (out_root / "README.md").write_text("\n".join(lines) + "\n")
    return written
