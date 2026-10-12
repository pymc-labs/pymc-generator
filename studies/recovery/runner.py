"""Explicit, hash-qualified per-world generation and method checkpoint orchestration."""

from __future__ import annotations

import json
import os
from pathlib import Path
from time import perf_counter

import pymc_generator as pg

from .artifacts import checkpoint, complete_checkpoint, identity, write_manifest
from .config import ContractError, StudyConfig
from .dgp import SHOWCASE, generate_pair, prior, validate_world, world_data
from .methods import bayes_fit, bounded_ols


def _sources(*names: str) -> dict[str, Path]:
    files = {name: Path(__file__).with_name(name) for name in names}
    files["generator-reference-config"] = SHOWCASE
    package = Path(pg.__file__).parent
    files.update(
        {f"generator/{path.relative_to(package)}": path for path in sorted(package.rglob("*.py"))}
    )
    return files


def _atomic(directory: Path, name: str, writer) -> None:
    temporary = directory / f".completion-{name}-{os.getpid()}{Path(name).suffix}"
    try:
        writer(temporary)
        os.replace(temporary, directory / name)
    finally:
        temporary.unlink(missing_ok=True)


def _json(directory: Path, name: str, payload: dict) -> None:
    _atomic(
        directory,
        name,
        lambda path: path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n"),
    )


def _stage(directory: Path, ident: dict) -> bool:
    state = checkpoint(directory, ident)
    if state.complete:
        return True
    write_manifest(directory, ident)
    # Partial outputs are unsealed; rewrite them atomically before creating the receipt.
    unknown = {
        path
        for path in directory.iterdir()
        if path.name not in {"manifest.json", "completion.json", *ident["expected_outputs"]}
        and not path.name.startswith(".completion-")
    }
    if unknown:
        raise ContractError(f"unexpected partial stage files: {sorted(map(str, unknown))}")
    return False


def _world_identity(config: StudyConfig, world: int) -> dict:
    ident = identity(
        "generate",
        config,
        source_files=_sources("config.py", "dgp.py", "runner.py", "artifacts.py"),
        expected_outputs={"dataset.npz", "metadata.json"},
    )
    ident.update(
        world_id=world,
        generation_seed=config.resolved["study"]["seed"] + world // 2,
        local_world=world % 2,
    )
    return ident


def _dataset(root: Path, config: StudyConfig, world: int):
    directory = root / "generate" / f"world_{world:02d}"
    ident = _world_identity(config, world)
    if not checkpoint(directory, ident).complete:
        raise ContractError(f"world {world} is not a completed generation checkpoint")
    meta = json.loads((directory / "metadata.json").read_text())
    if (
        meta["world_id"] != world
        or meta["generation_seed"] != ident["generation_seed"]
        or meta["local_world"] != world % 2
    ):
        raise ContractError("generation metadata does not match its world identity")
    corpus = pg.load_corpus(directory / "dataset.npz")
    cfg = prior(config, seed=ident["generation_seed"])
    validate_world(corpus, world % 2, cfg)
    return directory, corpus, cfg


def generate(config: StudyConfig, root: Path) -> list[dict]:
    count = config.resolved["study"]["worlds"]
    if count % 2:
        raise ContractError("cell-split generator requires even world count (two cells per batch)")
    results = []
    for batch in range(count // 2):
        seed = config.resolved["study"]["seed"] + batch
        folders = [root / "generate" / f"world_{batch * 2 + local:02d}" for local in range(2)]
        identities = [_world_identity(config, batch * 2 + local) for local in range(2)]
        states = [checkpoint(folder, ident) for folder, ident in zip(folders, identities)]
        if all(state.complete for state in states):
            for local in range(2):
                _dataset(root, config, batch * 2 + local)
                results.append({"world_id": batch * 2 + local, "state": "reused"})
            continue
        prior_completed = next((i for i, state in enumerate(states) if state.complete), None)
        if prior_completed is None:
            began = perf_counter()
            corpus = generate_pair(config, seed=seed)
            seconds = perf_counter() - began
        else:
            _, corpus, _ = _dataset(root, config, batch * 2 + prior_completed)
            seconds = 0.0
        cfg = prior(config, seed=seed)
        for local, (folder, ident, state) in enumerate(zip(folders, identities, states)):
            if state.complete:
                results.append({"world_id": batch * 2 + local, "state": "reused"})
                continue
            validate_world(corpus, local, cfg)
            _stage(folder, ident)
            _atomic(folder, "dataset.npz", lambda path, batch=corpus: pg.save_corpus(batch, path))
            _json(
                folder,
                "metadata.json",
                {
                    "world_id": batch * 2 + local,
                    "local_world": local,
                    "generation_seed": seed,
                    "cell_id": int(corpus["cell_id"][local]),
                    "seconds": seconds,
                    "baseline": config.resolved["study"]["baseline"],
                },
            )
            complete_checkpoint(folder, ident, set(ident["expected_outputs"]))
            results.append(
                {"world_id": batch * 2 + local, "state": "generated", "seconds": seconds}
            )
    return results


def fit(config: StudyConfig, root: Path, method: str | None = None) -> list[dict]:
    enabled = config.resolved["methods"]["enabled"]
    if method is not None and method not in enabled:
        raise ContractError("requested fit method is not enabled")
    results = []
    for world in range(config.resolved["study"]["worlds"]):
        folder, corpus, cfg = _dataset(root, config, world)
        data = world_data(corpus, world % 2, cfg, config.resolved["study"]["score_weeks"][0])
        for name in enabled if method is None else [method]:
            budget = config.resolved["methods"][name]
            seeds = budget["seeds"] if name == "advi" else [budget["seed"] + world]
            for replicate, seed in enumerate(seeds):
                label = f"{name}_{replicate + 1}" if name == "advi" else name
                directory = root / "fit" / f"world_{world:02d}" / label
                outputs = {"report.json", "posterior.nc" if name != "ols" else "components.npz"}
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
                    dataset_files={
                        "dataset.npz": folder / "dataset.npz",
                        "dataset-manifest.json": folder / "manifest.json",
                        "dataset-completion.json": folder / "completion.json",
                        "dataset-metadata.json": folder / "metadata.json",
                    },
                    expected_outputs=outputs,
                )
                actual_seed = seed + world if name == "advi" else seed
                ident.update(
                    world_id=world, method=name, replicate=replicate, method_seed=actual_seed
                )
                if _stage(directory, ident):
                    results.append({"world_id": world, "method": label, "state": "reused"})
                    continue
                if name == "ols":
                    import numpy as np

                    arrays, report = bounded_ols(
                        data,
                        cfg,
                        start=config.resolved["study"]["score_weeks"][0],
                        starts=budget["starts"],
                        max_nfev=budget["max_nfev"],
                        seed=seed,
                    )
                    _atomic(
                        directory,
                        "components.npz",
                        lambda path, values=arrays: np.savez_compressed(path.open("wb"), **values),
                    )
                else:
                    idata, report = bayes_fit(
                        data,
                        cfg,
                        start=config.resolved["study"]["score_weeks"][0],
                        varying=config.resolved["study"]["baseline"] == "varying",
                        method=name,
                        budget=budget,
                        seed=actual_seed,
                    )
                    _atomic(
                        directory,
                        "posterior.nc",
                        lambda path, posterior=idata: posterior.to_netcdf(path, engine="h5netcdf"),
                    )
                report.update(
                    world_id=world,
                    replicate=replicate,
                    configured_seed=seed,
                    effective_seed=actual_seed,
                    status="completed_smoke_not_scientifically_validated",
                )
                _json(directory, "report.json", report)
                complete_checkpoint(directory, ident, outputs)
                results.append(
                    {"world_id": world, "method": label, "state": "fitted", "report": report}
                )
    return results
