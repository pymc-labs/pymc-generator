"""SCM bundles: one folder per world that a human can audit end-to-end.

Each bundle contains:

* ``dataset.csv``           — what a model eats: week, spend_C*, control_Z*, sales_Y
* ``true_components.csv``   — the full additive truth: baseline_intrinsic,
  per-confounder and per-control contributions, per-channel DIRECT
  contributions, indirect effects by source (cc/zc/dc), plus the latent
  demand and base-channel series
* ``true_contribution.csv`` — legacy-compatible view (contribution_C*, baseline_B)
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

import os
from collections.abc import Sequence
from pathlib import Path

import numpy as np
import pandas as pd

from .describe import describe_scm, world_to_dot
from .scenarios import SCENARIOS, Scenario
from .worlds import SCM, sample_scm


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
        Target directory (created if missing).
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
    out.mkdir(parents=True, exist_ok=True)

    recon = world.reconstruction()

    model_cols = {"week": weeks}
    model_cols.update({f"spend_C{k + 1}": d["channels"][:, k] for k in range(n_treatments)})
    model_cols.update({f"control_Z{m + 1}": d["controls"][:, m] for m in range(n_covariates)})
    model_cols["sales_Y"] = d["sales"]
    pd.DataFrame(model_cols).to_csv(out / "dataset.csv", index=False)

    truth_cols = {"week": weeks, "baseline_intrinsic": d["baseline_intrinsic"]}
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

    legacy = {"week": weeks}
    legacy.update({f"contribution_C{k + 1}": d["contributions"][:, k] for k in range(n_treatments)})
    legacy["baseline_B"] = d["baseline"]
    pd.DataFrame(legacy).to_csv(out / "true_contribution.csv", index=False)

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

    Scenario ``idx`` is sampled with ``seed + idx`` — fully deterministic
    given ``(scenarios, n_time_steps, seed)``.

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
        ``connect_all``; removes the deliberate isolated-null traps).
    plots : bool
        Render PNG figures per bundle.
    verbose : bool
        Print one line per bundle.

    Returns
    -------
    list of Path
        The bundle directories, in scenario order.
    """
    out_root = Path(out_root)
    out_root.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    for idx, sc in enumerate(scenarios):
        cfg = sc.prior(n_time_steps=n_time_steps, seed=seed + idx)
        world = sample_scm(
            cfg,
            seed=seed + idx,
            connect_all=True if require_path_to_y else sc.connect_all,
            name=sc.name,
            purpose=sc.purpose,
        )
        out = write_scm_bundle(world, out_root / str(idx), title=f"{idx}: {sc.name}", plots=plots)
        if verbose:
            print(f"[{idx}] {sc.name}: identity err {world.identity_error():.1e} -> {out}")
        written.append(out)

    with open(os.path.join(out_root, "README.md"), "w") as f:
        f.write("# Inspection datasets (additive SCM, diverse texture)\n\n")
        f.write(f"Generated by `prior-generator --seed {seed}`.\n\n")
        f.write("| # | scenario | isolates |\n|---|---|---|\n")
        for idx, sc in enumerate(scenarios):
            f.write(f"| {idx} | {sc.name} | {sc.purpose.split('.')[0]}. |\n")
        f.write(
            "\nPer folder: `dataset.csv` (model inputs), `true_components.csv` (full "
            "decomposition truth), `true_contribution.csv` (legacy view), "
            "`description.txt` (DAG + coefficients + mechanisms + signal metrics), "
            "`dag.dot`/`dag.png`, `timeseries.png`, `decomposition.png`, `channels.png`.\n"
        )
    return written
