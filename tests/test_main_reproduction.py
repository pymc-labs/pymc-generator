"""With every feature off, the current package reproduces main (ce068d5) byte for byte.

``tests/_golden_probe.py`` records seeded corpora with their raw candidate
batches, ``sample_scm`` worlds, oracle densities, ``DataGenerator`` batches and
template draws. The test runs it twice with this interpreter and these
dependencies, importing main's package tree once and the current tree once, and
compares every recorded array by dtype, shape and bytes.

Main's tree comes from ``PYMC_GENERATOR_BASELINE_TREE`` (a checkout of ce068d5;
an unusable value fails) or, when that is unset, from this repository's git
objects. Either way its ``pymc_generator/`` files must be ce068d5's blobs. Without
either source, e.g. in an sdist or a clone lacking the commit, the test skips.
"""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
import time
from collections import Counter, defaultdict
from collections.abc import Iterable
from contextlib import ExitStack
from pathlib import Path

import numpy as np
import pytest

BASELINE_SHA = "ce068d598fc0438b511c265ef09d7ba33354cb48"
BASELINE_ENV = "PYMC_GENERATOR_BASELINE_TREE"
#: Git blob ids of ce068d5's tracked ``pymc_generator/`` files.
MAIN_PACKAGE_BLOBS = {
    "pymc_generator/__init__.py": "9c982e325491aeb4cd6f5859590235785e94ee98",
    "pymc_generator/_version.py": "af8b15aa72945cbb687dcaf71aa7499f64971c57",
    "pymc_generator/bundles.py": "6223de7d996c68a828e24d475c9622230fef740c",
    "pymc_generator/cli.py": "20f090b938afbc0d979a5e28fccaf24230185f48",
    "pymc_generator/data_generator.py": "014d8cd24d6b91bfe5c229809f627dc65a2b1a62",
    "pymc_generator/describe.py": "dd7a4326ae7a145c904be3b214e1bedad6c21ab4",
    "pymc_generator/diagnostics.py": "3aeffc745f3ce1a31a4f5218d12cccdfce9cb93c",
    "pymc_generator/mechanisms.py": "959126c20638bb7b171afc9a05c0b6d7c9230355",
    "pymc_generator/outcomes.py": "0f98eb1498d630c19d4e4c165cb6932c605a102e",
    "pymc_generator/presets.py": "8bc63d1a32deb2361b99dcee79497f8a693a8ba3",
    "pymc_generator/py.typed": "e69de29bb2d1d6434b8b29ae775ad8c2e48c5391",
    "pymc_generator/random_walk.py": "aec7af1a8f4e645bb8c4bf997d5037e6e6b5baa5",
    "pymc_generator/sampler.py": "729cd9b52b6093add86cc943f8d0c59b23d2f03d",
    "pymc_generator/scenarios.py": "d9638561986282c8341684085e749603af667ad0",
    "pymc_generator/signal_diagnostics.py": "ebbaf89b14bb86e5c75e76c043fe5155807d3285",
    "pymc_generator/slots.py": "7a50282d8f5d5ee8c0404b3e019d0b494519ca56",
    "pymc_generator/symbolic_graph.py": "8c5eede3829ab0138f1a1d7c4d93c2490c062b64",
    "pymc_generator/viz.py": "54e515b5290dd0cc8cd47129e48cb2b1b3124fb6",
    "pymc_generator/world_model.py": "7b314ded6b0c76e8aeb5340591ef2651cd944946",
    "pymc_generator/world_model_template.py": "38af691659085e22836cc92bff766cd1cbf8a4be",
    "pymc_generator/worlds.py": "c4fe937eaea64a8c3edc87f6f04878b4b449ed25",
}
REPO = Path(__file__).resolve().parents[1]
PROBE = REPO / "tests" / "_golden_probe.py"
SECTIONS = ("ordinary", "scm", "oracle", "datagen", "template")
PROBE_TIMEOUT_S = 1800
STDERR_TAIL = 4000
_SOURCE_HINT = (
    f"set {BASELINE_ENV} to a checkout of {BASELINE_SHA} or run `git fetch origin {BASELINE_SHA}`"
)


def _blob_id(path: Path) -> str:
    data = path.read_bytes()
    return hashlib.sha1(b"blob %d\0" % len(data) + data, usedforsecurity=False).hexdigest()


def _package_blobs(tree: Path) -> dict[str, str]:
    package = tree / "pymc_generator"
    return {
        path.relative_to(tree).as_posix(): _blob_id(path)
        for path in sorted(package.rglob("*"))
        if path.is_file() and "__pycache__" not in path.relative_to(package).parts
    }


def _check_main_package(tree: Path, source: str) -> None:
    actual = _package_blobs(tree)
    problems = [f"missing {path}" for path in sorted(MAIN_PACKAGE_BLOBS.keys() - actual.keys())]
    problems += [f"extra {path}" for path in sorted(actual.keys() - MAIN_PACKAGE_BLOBS.keys())]
    problems += [
        f"changed {path}"
        for path in sorted(MAIN_PACKAGE_BLOBS.keys() & actual.keys())
        if actual[path] != MAIN_PACKAGE_BLOBS[path]
    ]
    if problems:
        pytest.fail(f"{source} is not {BASELINE_SHA}'s package:\n  " + "\n  ".join(problems))


def _git(*args: str, stdin: bytes | None = None) -> subprocess.CompletedProcess[bytes]:
    return subprocess.run(
        ["git", "-C", str(REPO), *args], input=stdin, capture_output=True, check=False
    )


def _extract_main_package(dest: Path) -> Path:
    """Write ce068d5's ``pymc_generator/`` blobs from this clone's object store.

    Raw blobs, not ``git archive``: export attributes and eol settings rewrite
    archive output.
    """
    try:
        toplevel = _git("rev-parse", "--show-toplevel")
    except FileNotFoundError:
        pytest.skip(f"git is not available; {_SOURCE_HINT}")
    if toplevel.returncode != 0 or Path(toplevel.stdout.decode().strip()).resolve() != REPO:
        pytest.skip(f"{REPO} is not the top level of a git checkout; {_SOURCE_HINT}")
    if _git("cat-file", "-e", f"{BASELINE_SHA}^{{commit}}").returncode != 0:
        pytest.skip(f"this clone lacks commit {BASELINE_SHA}; {_SOURCE_HINT}")
    listing = _git("ls-tree", "-r", "-z", BASELINE_SHA, "--", "pymc_generator")
    if listing.returncode != 0:
        pytest.skip(f"git ls-tree failed: {listing.stderr.decode(errors='replace')}")
    entries = []
    for entry in filter(None, listing.stdout.split(b"\0")):
        meta, _, path = entry.partition(b"\t")
        _mode, kind, object_id = meta.split()
        if kind == b"blob":
            entries.append((path.decode(), object_id))
    batch = _git("cat-file", "--batch", stdin=b"".join(oid + b"\n" for _, oid in entries))
    if batch.returncode != 0:
        pytest.skip(f"git cat-file failed: {batch.stderr.decode(errors='replace')}")
    # Each object is "<oid> <type> <size>\n<content>\n".
    output, offset = batch.stdout, 0
    for path, _object_id in entries:
        header_end = output.index(b"\n", offset)
        header = output[offset:header_end].split()
        if len(header) != 3:
            pytest.skip(f"git cat-file could not read {path}: {header!r}")
        size = int(header[2])
        target = dest / path
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(output[header_end + 1 : header_end + 1 + size])
        offset = header_end + 1 + size + 1
    _check_main_package(dest, f"{BASELINE_SHA} extracted from {REPO}")
    return dest


def _baseline_tree(tmp_path: Path) -> Path:
    value = os.environ.get(BASELINE_ENV, "")
    if not value:
        return _extract_main_package(tmp_path / "baseline")
    # Resolved once and reused for PYTHONPATH and the module check.
    tree = Path(value).expanduser().resolve()
    source = f"{BASELINE_ENV}={value!r}"
    if not (tree / "pymc_generator" / "__init__.py").is_file():
        pytest.fail(f"{source}: {tree} has no pymc_generator/__init__.py")
    if tree == REPO:
        pytest.fail(f"{source} is this checkout, not main")
    _check_main_package(tree, source)
    return tree


def _env_for(tree: Path) -> dict[str, str]:
    env = {key: value for key, value in os.environ.items() if key != "PYTHONPATH"}
    # A Numba function loaded from PyTensor's on-disk cache can round differently
    # from a fresh compile, and both probes share that cache; compile fresh.
    flags = [flag for flag in env.get("PYTENSOR_FLAGS", "").split(",") if flag]
    env.update(
        PYTHONPATH=str(tree),
        PYTENSOR_FLAGS=",".join([*flags, "numba__cache=False"]),
        PYTHONHASHSEED="0",
        PYTHONDONTWRITEBYTECODE="1",
        OMP_NUM_THREADS="1",
        OPENBLAS_NUM_THREADS="1",
        MKL_NUM_THREADS="1",
        VECLIB_MAXIMUM_THREADS="1",
        NUMBA_NUM_THREADS="1",
    )
    return env


def _stderr_tails(tmp_path: Path, sides: Iterable[str]) -> str:
    return "\n".join(
        f"--- {side} stderr (tail) ---\n"
        + (tmp_path / side / "stderr.txt").read_text(errors="replace")[-STDERR_TAIL:]
        for side in sides
    )


def _run_probes(trees: dict[str, Path], tmp_path: Path) -> dict[str, Path]:
    """Run the probe once per tree, concurrently, and return the ``.npz`` paths.

    Each probe works in a fresh, empty directory, never inside a package tree, so
    ``PYTHONPATH`` alone selects which ``pymc_generator`` it imports.
    """
    procs: dict[str, subprocess.Popen[bytes]] = {}
    outputs = {side: tmp_path / side / "probe.npz" for side in trees}
    with ExitStack() as files:
        try:
            for side, tree in trees.items():
                work = outputs[side].parent
                work.mkdir()
                procs[side] = subprocess.Popen(
                    [sys.executable, str(PROBE), str(outputs[side])],
                    cwd=work,
                    env=_env_for(tree),
                    # Files, not pipes: a full pipe would block the probe.
                    stdout=files.enter_context((work / "stdout.txt").open("wb")),
                    stderr=files.enter_context((work / "stderr.txt").open("wb")),
                )
            deadline = time.monotonic() + PROBE_TIMEOUT_S
            for proc in procs.values():
                proc.wait(timeout=max(0.0, deadline - time.monotonic()))
        except subprocess.TimeoutExpired:
            timed_out = True
        else:
            timed_out = False
        finally:
            for proc in procs.values():
                if proc.poll() is None:
                    proc.kill()
                    proc.wait()
    if timed_out:
        pytest.fail(
            f"probes exceeded {PROBE_TIMEOUT_S} s and were killed\n"
            + _stderr_tails(tmp_path, trees)
        )
    failed = {side: proc.returncode for side, proc in procs.items() if proc.returncode}
    if failed:
        pytest.fail(f"probe exited non-zero: {failed}\n" + _stderr_tails(tmp_path, trees))
    return outputs


def _check_modules(side: str, tree: Path, modules: dict[str, str | None]) -> None:
    """Every loaded package module must come from ``tree``.

    A submodule absent from main's tree would otherwise resolve, unnoticed,
    through the editable install of the current checkout.
    """
    package = (tree / "pymc_generator").resolve()
    leaked = {
        name: file
        for name, file in modules.items()
        if file is None or not Path(file).is_relative_to(package)
    }
    if modules.get("pymc_generator") != str(package / "__init__.py") or leaked:
        pytest.fail(
            f"{side} probe imported pymc_generator modules outside {package}:\n"
            + json.dumps(leaked or modules, indent=1)
        )


def _array_kind(a: np.ndarray, b: np.ndarray) -> str | None:
    if a.dtype != b.dtype:
        return f"dtype {a.dtype} != {b.dtype}"
    if a.shape != b.shape:
        return f"shape {a.shape} != {b.shape}"
    if np.ascontiguousarray(a).tobytes() == np.ascontiguousarray(b).tobytes():
        return None
    a_items, b_items = (
        np.ascontiguousarray(x).reshape(-1).view(np.uint8).reshape(x.size, x.dtype.itemsize)
        for x in (a, b)
    )
    differing = int(np.any(a_items != b_items, axis=1).sum())
    return f"bytes differ in {differing} of {a.size} elements"


def _calls_kind(a: list[dict], b: list[dict]) -> str:
    if len(a) != len(b):
        return f"{len(a)} draw_worlds calls != {len(b)}"
    index, (x, y) = next((i, pair) for i, pair in enumerate(zip(a, b)) if pair[0] != pair[1])
    fields = sorted(key for key in x.keys() | y.keys() if x.get(key) != y.get(key))
    return f"draw_worlds call {index} differs in {', '.join(fields)}"


def _differences(outputs: dict[str, Path], sidecars: dict[str, dict]) -> dict[str, str]:
    """``{key: kind}`` for every array or sidecar entry that differs from main."""
    found: dict[str, str] = {}
    with np.load(outputs["main"]) as main, np.load(outputs["current"]) as current:
        main_keys, current_keys = set(main.files), set(current.files)
        for key in sorted(main_keys | current_keys):
            if key not in current_keys:
                found[key] = "only in main"
            elif key not in main_keys:
                found[key] = "only in current"
            elif kind := _array_kind(main[key], current[key]):
                found[key] = kind
        for side, keys in (("main", main_keys), ("current", current_keys)):
            counts = Counter(key.split("/", 1)[0] for key in keys)
            for section in SECTIONS:
                if not counts[section]:
                    found[f"{section} ({side})"] = "no arrays recorded"
    a, b = sidecars["main"], sidecars["current"]
    for prefix in sorted(a["calls"].keys() | b["calls"].keys()):
        calls_a, calls_b = a["calls"].get(prefix), b["calls"].get(prefix)
        if calls_a != calls_b:
            found[f"{prefix} (calls)"] = (
                _calls_kind(calls_a, calls_b)
                if calls_a is not None and calls_b is not None
                else f"draw_worlds calls only in {'main' if calls_b is None else 'current'}"
            )
    missing = object()
    for key in sorted(a["scalars"].keys() | b["scalars"].keys()):
        value_a, value_b = a["scalars"].get(key, missing), b["scalars"].get(key, missing)
        if value_a is missing or value_b is missing:
            found[f"{key} (scalar)"] = f"only in {'main' if value_b is missing else 'current'}"
        elif value_a != value_b or type(value_a) is not type(value_b):
            found[f"{key} (scalar)"] = f"{value_a!r} != {value_b!r}"
    if a["features_off"] != b["features_off"]:
        found["features_off"] = f"{a['features_off']} != {b['features_off']}"
    for side, sidecar in sidecars.items():
        if sidecar["skipped"]:
            found[f"skipped ({side})"] = json.dumps(sidecar["skipped"])
    return found


@pytest.mark.slow
def test_features_off_outputs_reproduce_main_byte_for_byte(tmp_path):
    trees = {"main": _baseline_tree(tmp_path), "current": REPO}
    outputs = _run_probes(trees, tmp_path)
    sidecars = {
        side: json.loads(path.with_suffix(".json").read_text()) for side, path in outputs.items()
    }
    for side, tree in trees.items():
        _check_modules(side, tree, sidecars[side]["modules"])

    found = _differences(outputs, sidecars)
    if found:
        by_section: dict[str, list[str]] = defaultdict(list)
        for key, kind in found.items():
            by_section[key.split("/", 1)[0].split(" ", 1)[0]].append(f"    {key}: {kind}")
        report = [f"{len(found)} entries differ from main {BASELINE_SHA}:"]
        for section, lines in sorted(by_section.items()):
            report += [f"  {section}: {len(lines)}", *lines]
        report += [f"  main npz: {outputs['main']}", f"  current npz: {outputs['current']}"]
        pytest.fail("\n".join(report), pytrace=False)
