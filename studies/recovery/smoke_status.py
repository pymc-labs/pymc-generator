"""Read-only computational-status summary for the FOUR authorized smoke worlds."""

from __future__ import annotations

import json
from pathlib import Path

import arviz as az
import numpy as np
import xarray as xr

from .artifacts import checkpoint, identity
from .config import StudyConfig
from .runner import _json, _sources, _world_identity


def summarize(study: Path) -> dict:
    output = {}
    for label in ("constant", "varying"):
        config = StudyConfig.load(study / f"smoke_{label}.toml")
        root = study / "smoke-artifacts-final" / label
        regimes = []
        for world in (0, 1):
            generation = root / "generate" / f"world_{world:02d}"
            state = checkpoint(generation, _world_identity(config, world)).state
            methods = {}
            for name in config.resolved["methods"]["enabled"]:
                for replicate in range(2 if name == "advi" else 1):
                    method = f"{name}_{replicate + 1}" if name == "advi" else name
                    path = root / "fit" / f"world_{world:02d}" / method
                    budget = config.resolved["methods"][name]
                    seed = budget["seeds"][replicate] if name == "advi" else budget["seed"] + world
                    files = {
                        "dataset.npz": generation / "dataset.npz",
                        "dataset-manifest.json": generation / "manifest.json",
                        "dataset-completion.json": generation / "completion.json",
                        "dataset-metadata.json": generation / "metadata.json",
                    }
                    expected = {
                        "report.json",
                        "components.npz" if name == "ols" else "posterior.nc",
                    }
                    ident = identity(
                        "fit",
                        config,
                        source_files=_sources(
                            "config.py",
                            "dgp.py",
                            "models.py",
                            "methods.py",
                            "runner.py",
                            "artifacts.py",
                        ),
                        dataset_files=files,
                        expected_outputs=expected,
                    )
                    ident.update(
                        world_id=world,
                        method=name,
                        replicate=replicate,
                        method_seed=seed + world if name == "advi" else seed,
                    )
                    fit_state = checkpoint(path, ident).state
                    report = json.loads((path / "report.json").read_text())
                    info = {"seconds": report["seconds"], "completed": fit_state == "complete"}
                    if name == "ols":
                        info.update(
                            optimizer_success=report["success"],
                            note="bounded nonlinear LS, not unconstrained OLS",
                        )
                    elif name == "advi":
                        info.update(
                            loss_finite=report["loss_finite"],
                            posterior_finite=report["posterior_finite"],
                            warning=report["optimization_warning"],
                            no_mcmc_diagnostics=True,
                        )
                    else:
                        with xr.open_datatree(path / "posterior.nc", engine="h5netcdf") as tree:
                            stats = tree["sample_stats"]
                            free = [
                                "intercept",
                                "alpha",
                                "log_kappa",
                                "amplitude",
                                "control_reference_contribution",
                                "sigma_relative",
                            ] + (["baseline_sd_relative"] if label == "varying" else [])
                            rhat = az.rhat(tree, var_names=free)
                            bulk = az.ess(tree, var_names=free, method="bulk")
                            tail = az.ess(tree, var_names=free, method="tail")
                            all_rhat = np.concatenate(
                                [v.values.ravel() for v in rhat.data_vars.values()]
                            )
                            all_bulk = np.concatenate(
                                [v.values.ravel() for v in bulk.data_vars.values()]
                            )
                            all_tail = np.concatenate(
                                [v.values.ravel() for v in tail.data_vars.values()]
                            )
                            bfmi = np.asarray(az.bfmi(tree)["energy"])
                            info.update(
                                divergences=int(stats["diverging"].sum()),
                                maxdepth_hits=int(stats["maxdepth_reached"].sum()),
                                max_rhat=float(np.max(all_rhat)),
                                min_bulk_ess=float(np.min(all_bulk)),
                                min_tail_ess=float(np.min(all_tail)),
                                min_bfmi=float(np.min(bfmi)),
                                diagnostic_pass=bool(
                                    np.all(np.isfinite(all_rhat))
                                    and np.max(all_rhat) < 1.01
                                    and np.min(all_bulk) >= 400
                                    and np.min(all_tail) >= 400
                                    and np.min(bfmi) >= 0.3
                                    and int(stats["diverging"].sum()) == 0
                                    and int(stats["maxdepth_reached"].sum()) == 0
                                ),
                                warning="100 retained draws/chain cannot satisfy ESS >= 400; smoke is not recovery evidence",
                            )
                    methods[method] = info
            regimes.append({"world_id": world, "generation_checkpoint": state, "methods": methods})
        output[label] = regimes
    _json(study / "smoke-artifacts-final", "diagnostics.json", output)
    return output


if __name__ == "__main__":
    print(json.dumps(summarize(Path(__file__).parent), indent=2))
