"""Four once-only cold native MAP attempts. Engineering preflight is not a fit release.

The launcher guards a supervisor, which guards four fresh case process groups. Only
--execute-released permits generation; independent acceptance and a separate owner
release are procedural prerequisites, not implied by this switch.
"""
from __future__ import annotations

import argparse
import contextlib
import ctypes
import dataclasses
import datetime as dt
import fcntl
import hashlib
import importlib
import json
import logging
import math
import os
import platform
import resource
import select
import signal
import stat
import subprocess
import sys
import tempfile
import time
import traceback
from pathlib import Path
from typing import Any

BASE = "0e43c6b708b1584c94fd2271124142a312409a65"
ROOT = Path("/home/teemu/pymc-labs/prior-generator/.worktrees/issue-30-map-cold-smoke-report")
BRANCH = "feature/issue-30-map-cold-smoke-report"
PYTHON = "/home/teemu/pymc-labs/prior-generator/.worktrees/issue-30-identifiable-linear-recovery/.venv/bin/python3"
CONFIG_DEFAULT = "docs/examples/data/map-recovery-pilot-config.json"
CONFIG_SHA256 = "bd31e76743a8c51e0906dab5f5ecd8e2d9f196219039629c7cbce2d92d9ebbd0"
THREAD_ENV = ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS")
EXPECTED_VERSIONS = {"pymc": "6.2.0", "pytensor": "3.2.4", "scipy": "1.18.0"}
MODULES = ("pymc_generator", "pymc_generator.worlds", "pymc_generator.world_model",
           "pymc_generator.presets", "pymc_generator.sampler")
OPTIONS = {"maxiter": 1000, "maxfun": 1000, "ftol": 1e-12, "gtol": 1e-6, "maxls": 20}
LIMITATIONS = [
    "four illustrative cases, not MAP20 prevalence or a paired experiment",
    "Issue42 startup/warmup limitation: native all-104 likelihood, B=0/discard=0",
    "optimizer convergence is not identifiability or posterior calibration",
    "native MAP no-Jacobian free-primitive-density search in transformed coordinates",
    "compile/solve split unavailable; inclusive native API timing only",
    "each case has a fresh private PyTensor cache; OS/library caches may still warm across processes",
    "no measured20-world costs, MCMC, intervals or frequency claims",
]


def digest(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def plain(value: Any) -> Any:
    """Lossless JSON for normal arrays; explicitly label nonfinite failure evidence."""
    if isinstance(value, Path):
        return str(value)
    if dataclasses.is_dataclass(value):
        return plain(dataclasses.asdict(value))
    if isinstance(value, dict):
        return {str(k): plain(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [plain(v) for v in value]
    if hasattr(value, "tolist"):
        return plain(value.tolist())
    if isinstance(value, float) and not math.isfinite(value):
        return {"nonfinite": repr(value)}
    if value is None or isinstance(value, (str, bool, int, float)):
        return value
    raise TypeError(f"unsupported report value: {type(value).__name__}")


def canonical(value: Any) -> bytes:
    return json.dumps(plain(value), sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def utc() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat()


def _git(*args: str) -> str:
    return subprocess.check_output(["git", "-C", str(ROOT), *args], text=True,
                                   stderr=subprocess.STDOUT, timeout=15).strip()


def load_config(path: Path) -> dict[str, Any]:
    cfg = json.loads(path.read_text())
    if cfg.get("schema_version") != "map-recovery-smoke/v1" or len(cfg.get("cases", [])) != 4:
        raise ValueError("expected exactly four fixed smoke cases")
    if digest(canonical(cfg)) != CONFIG_SHA256:
        raise ValueError("approved full config/schema/regimes/seeds/resources freeze mismatch")
    return cfg


def build_config(regime: str) -> Any:
    from pymc_generator import make_scm_prior
    cfg = load_config(ROOT / CONFIG_DEFAULT)
    if regime not in cfg["regimes"]:
        raise ValueError(f"unknown regime: {regime}")
    params = dict(cfg["generator"])
    params["rw_outcome_std_range"] = tuple(params["rw_outcome_std_range"])
    params["rw_baseline_std_range"] = tuple(cfg["regimes"][regime])
    return make_scm_prior(**params)


def _under(path: str | Path, root: Path) -> bool:
    return Path(path).resolve().is_relative_to(root.resolve())


def source_proof() -> dict[str, Any]:
    if Path(__file__).resolve() != ROOT / "scripts/map_recovery_pilot.py" or Path.cwd().resolve() != ROOT:
        raise RuntimeError("runner/cwd is not the accepted cold lane")
    if _git("rev-parse", "HEAD") != BASE or _git("branch", "--show-current") != BRANCH:
        raise RuntimeError("HEAD/branch differs from accepted baseline")
    if _git("diff", "--cached", "--name-only") or _git("diff", "--name-only"):
        raise RuntimeError("tracked source/index is dirty")
    # NUL records preserve exact paths/statuses; ignored runtime metadata stays ignored.
    status = subprocess.check_output(
        ["git", "-C", str(ROOT), "status", "--porcelain=v1", "--untracked-files=all", "-z"],
        stderr=subprocess.STDOUT, timeout=15)
    allowed = ("scripts/map_recovery_pilot.py", "tests/test_map_recovery_pilot.py",
               CONFIG_DEFAULT, "docs/examples/map-recovery-pilot.md")
    if sorted(status.split(b"\0")) != sorted([b""] + [f"?? {name}".encode() for name in allowed]):
        raise RuntimeError("source status must contain exactly the four authorized untracked report paths")
    entries = _git("ls-tree", "-r", BASE).splitlines()
    files = {}
    for entry in entries:
        meta, name = entry.split("\t", 1)
        mode, kind, oid = meta.split()
        path = ROOT / name
        if kind != "blob" or path.is_symlink() or not path.is_file():
            raise RuntimeError(f"unsupported or missing tracked source: {name}")
        content = path.read_bytes()
        actual_oid = hashlib.sha1(b"blob " + str(len(content)).encode() + b"\0" + content).hexdigest()
        actual_mode = "100755" if path.stat().st_mode & 0o111 else "100644"
        if oid != actual_oid or mode != actual_mode:
            raise RuntimeError(f"tracked byte/mode mismatch: {name}")
        files[name] = {"blob_oid": oid, "sha256": digest(content), "mode": mode}
    if len(files) != 122:
        raise RuntimeError("accepted 122-file source manifest differs")
    return {"head": BASE, "branch": BRANCH, "index_empty": True,
            "index_entries_sha256": digest(_git("ls-files", "--stage").encode()),
            "tracked_files": files, "manifest_sha256": digest(canonical(files))}


def cache_path() -> Path:
    flags = os.environ.get("PYTENSOR_FLAGS", "")
    if not flags.startswith("base_compiledir=/") or "," in flags:
        raise RuntimeError("PYTENSOR_FLAGS must be exactly base_compiledir=<absolute private cache>")
    path = Path(flags.split("=", 1)[1])
    if path != path.resolve() or not path.is_dir():
        raise RuntimeError("cache must be an existing canonical directory (no symlinks)")
    info = path.stat()
    if info.st_uid != os.getuid() or stat.S_IMODE(info.st_mode) != 0o700:
        raise RuntimeError("cache must be owned by this uid with mode 0700")
    return path


@contextlib.contextmanager
def cache_lock(cache: Path):
    fd = os.open(cache / ".map-smoke-lock", os.O_RDWR | os.O_CREAT | os.O_NOFOLLOW, 0o600)
    try:
        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        yield fd
    finally:
        os.close(fd)


def _verify_lock(cache: Path, fd: int) -> None:
    info = os.fstat(fd)
    if info.st_ino != (cache / ".map-smoke-lock").stat().st_ino or info.st_uid != os.getuid():
        raise RuntimeError("private cache lock fd mismatch")
    # Inherited descriptors share the launcher's held exclusive flock.
    fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)


def runtime_bindings() -> dict[str, Any]:
    versions, paths = {}, {}
    for name, expected in EXPECTED_VERSIONS.items():
        module = importlib.import_module(name)
        versions[name] = module.__version__
        paths[name] = str(Path(module.__file__).resolve())
        if versions[name] != expected or not _under(paths[name], Path(PYTHON).parents[1]):
            raise RuntimeError(f"actual imported dependency version/path mismatch: {name}: {versions[name]} {paths[name]}")
    for name in MODULES:
        module = importlib.import_module(name)
        paths[name] = str(Path(module.__file__).resolve())
        if not _under(paths[name], ROOT / "pymc_generator"):
            raise RuntimeError(f"actual imported source escaped cold lane: {name}")
    import pytensor
    if not _under(pytensor.config.compiledir, cache_path()):
        raise RuntimeError("actual PyTensor compiledir escaped private cache")
    return {"versions": versions, "module_paths": paths, "compiledir": str(pytensor.config.compiledir),
            "pytensor_floatX": pytensor.config.floatX}


def script_context() -> dict[str, Any]:
    """Check the real script launcher, not an imported/python-c approximation."""
    script = str(ROOT / "scripts/map_recovery_pilot.py")
    if sys.argv[0] != script or sys.orig_argv != [PYTHON, *sys.argv]:
        raise RuntimeError("expected exact absolute-script interpreter/argv context")
    if sys.path[:2] != [str(ROOT / "scripts"), str(ROOT)]:
        raise RuntimeError("expected script sys.path prefix, not imported/python-c context")
    return {"orig_argv": list(sys.orig_argv), "sys_path": list(sys.path)}


def launcher_command(config: Path, output: Path, mode: str) -> list[str]:
    return [PYTHON, str(ROOT / "scripts/map_recovery_pilot.py"), "--config", str(config.resolve()),
            "--output", str(output.resolve()), mode]


def expected_launch(argv: list[str], env: dict[str, str], cache_root: Path) -> dict[str, Any]:
    """Prospective contract: dynamic fields are exact per launch, not cross-run hashes."""
    cfg = load_config(ROOT / CONFIG_DEFAULT)
    return {"stable": {"python": PYTHON, "script": str(ROOT / "scripts/map_recovery_pilot.py"),
                "cwd": str(ROOT), "sys_path_prefix": [str(ROOT / "scripts"), str(ROOT)],
                "source_oid": BASE, "versions": EXPECTED_VERSIONS,
                "threads": dict.fromkeys(THREAD_ENV, "1"),
                "cache_policy": "one fresh case directory beneath the locked run root; no case reuse"},
            "per_launch": {"argv": list(argv), "environment": dict(env), "cache_root": str(cache_root),
                "case_cache_directories": {c["case_id"]: str(cache_root / c["case_id"]) for c in cfg["cases"]}},
            "observational_not_cross_run_identity": ["pid", "timestamps", "resources", "compiledir",
                "sys_path_after_prefix", "threadpool_metadata"],
            "full_runtime_metadata_digest_claimed": False}


def verify_launch(config: Path, output: Path | None, cache: Path, lock_fd: int | None) -> dict[str, Any]:
    context = script_context()  # Before any scientific import.
    if output is None:
        output = Path(sys.argv[sys.argv.index("--output") + 1])
    cache_root = cache
    if "--case-index" in sys.argv:
        index = int(sys.argv[sys.argv.index("--case-index") + 1])
        case = load_config(config)["cases"][index]
        cache_root = cache.parent
        if cache.name != case["case_id"]:
            raise RuntimeError("case cache does not match its declared private directory")
        argv = child_command(config, output, lock_fd, "--case-index", str(index))
        label = case["case_id"]
    elif "--supervisor" in sys.argv:
        started = sys.argv[sys.argv.index("--started") + 1]
        argv = child_command(config, output, lock_fd, "--supervisor", "--started", started)
        label = "supervisor"
    else:
        argv = launcher_command(config, output, "--preflight")
        label = None
    if [sys.executable, *sys.argv] != argv:
        raise RuntimeError("actual argv differs from the prospective launch contract")
    expected = expected_launch(argv, clean_environment(), cache_root)
    if label is not None:
        planned = json.loads((output / f"{label}.launch.json").read_text())
        if planned != expected:
            raise RuntimeError("actual launch environment/argv/cache differs from the pre-execution contract")
    return {"expected": expected, "actual_import_context": context, "cache_root": str(cache_root)}


def preflight(config_path: Path, output: Path | None = None, lock_fd: int | None = None) -> dict[str, Any]:
    """Import/config only: never samples, constructs an oracle, or runs an optimizer."""
    result: dict[str, Any] = {"ok": False, "errors": [], "timestamp": utc(),
        "python": sys.executable, "python_version": platform.python_version(),
        "argv": sys.argv, "environment": clean_environment()}
    try:
        extra = set(os.environ) - set(clean_environment()) - {"LC_CTYPE"}
        if extra:
            raise RuntimeError("use the documented env -i invocation; unexpected environment keys: " + ", ".join(sorted(extra)))
        cfg = load_config(config_path)
        result["config_sha256"] = digest(canonical(cfg))
        result["config_file_sha256"] = digest(config_path.read_bytes())
        result["source"] = source_proof()
        if sys.executable != PYTHON or sys.prefix != str(Path(PYTHON).parents[1]):
            raise RuntimeError("interpreter/prefix is not the accepted C3 executable")
        if os.environ.get("PYTHONPATH") != str(ROOT):
            raise RuntimeError("PYTHONPATH must be exactly the accepted cold lane")
        if any(os.environ.get(name) != "1" for name in THREAD_ENV):
            raise RuntimeError("all four thread variables must be exactly 1")
        limits = resource.getrlimit(resource.RLIMIT_AS)
        result["actual_address_space_limits"] = list(limits)
        if limits != (cfg["resources"]["address_space_bytes"],) * 2:
            raise RuntimeError("actual soft/hard address-space limits must both equal 16 GiB")
        cache = cache_path()
        result["launch_contract"] = verify_launch(config_path, output, cache, lock_fd)
        cache_root = Path(result["launch_contract"]["cache_root"])
        if output is not None and (_under(output, cache_root) or _under(cache_root, output)):
            raise RuntimeError("cache and artifacts must be disjoint")
        if lock_fd is None:
            with cache_lock(cache_root) as fd:
                return preflight(config_path, output, fd)
        _verify_lock(cache_root, lock_fd)
        result["cache"] = {"path": str(cache), "run_root": str(cache_root), "mode": "0700",
                           "uid": os.getuid(), "exclusive_lock": True}
        result["resources"] = _resource_snapshot(cache_root, output)
        result.update(runtime_bindings())
        result["resolved_factory"] = {}
        for regime, expected in cfg["resolved_factory"].items():
            actual = dataclasses.asdict(build_config(regime))
            if canonical(actual) != canonical(expected):
                raise RuntimeError(f"full resolved public factory configuration differs: {regime}")
            result["resolved_factory"][regime] = {"config": plain(actual), "sha256": digest(canonical(actual))}
        result["freeze"] = {name: digest((ROOT / name).read_bytes()) for name in (
            "scripts/map_recovery_pilot.py", "tests/test_map_recovery_pilot.py", CONFIG_DEFAULT,
            "docs/examples/map-recovery-pilot.md")}
        result["ok"] = True
    except Exception as exc:
        result["errors"].append(f"{type(exc).__name__}: {exc}")
        result["traceback"] = traceback.format_exc()
    return result


def array_record(value: Any) -> dict[str, Any]:
    import numpy as np
    a = np.asarray(value)
    # Canonical values also make string/object arrays independent of pointer bytes.
    return {"shape": list(a.shape), "dtype": str(a.dtype),
            "sha256": digest(canonical({"shape": a.shape, "dtype": str(a.dtype), "values": a}))}


def data_contract(data: dict[str, Any]) -> dict[str, Any]:
    return {key: array_record(value) for key, value in sorted(data.items())}


def layout_contract(model: Any, start: dict[str, Any]) -> dict[str, Any]:
    import numpy as np
    shapes = model.eval_rv_shapes()
    coords = plain(dict(model.coords))
    mapping, offset = [], 0
    for rv in model.free_RVs:
        value = model.rvs_to_values[rv]
        a = np.asarray(start[value.name])
        transform = model.rvs_to_transforms.get(rv)
        dims = list(model.named_vars_to_dims.get(rv.name, ()))
        mapping.append({"rv": rv.name, "value": value.name, "rv_shape": list(shapes[rv.name]),
            "rv_dtype": rv.dtype, "value_shape": list(a.shape), "value_dtype": str(a.dtype),
            "transform": None if transform is None else {"name": transform.name,
                "class": f"{type(transform).__module__}.{type(transform).__qualname__}"},
            "dims": dims, "coordinates": {d: coords[d] for d in dims},
            "ravel_order": "C", "flat_slice": [offset, offset + a.size]})
        offset += a.size
    # PyMC iterates start order when forming its DictToArrayBijection.
    by_value = {entry["value"]: entry for entry in mapping}
    ordered, offset = [], 0
    for name in start:
        if name in by_value:
            entry = by_value[name]
            size = int(np.prod(entry["value_shape"], dtype=int))
            entry["flat_slice"] = [offset, offset + size]
            offset += size
            ordered.append(entry)
    if len(ordered) != len(mapping):
        raise RuntimeError("initial point/free-value mapping is incomplete")
    return {"free_rvs": [rv.name for rv in model.free_RVs],
            "value_vars": [v.name for v in model.value_vars], "optimizer_value_order": [v["value"] for v in ordered],
            "mapping": ordered, "coords": coords, "flat_size": offset}


def compatibility_contract(world: Any, config: dict[str, Any], layout: dict[str, Any]) -> dict[str, Any]:
    # Saturation scale and prior conditioning are baked into the oracle, not bound
    # inputs in an implemented helper. Including their values is conservative.
    return {"graph": plain(world.g), "families_and_structure": plain(world.extras["structural"]),
            "full_config": plain(config), "prior_conditioning": plain(world.extras.get("prior_cond")),
            "saturation_scale": plain(world.data["saturation_scale"]),
            "latent": "marginal", "rows": list(range(104)), "B": 0, "discard": 0,
            "data_layout": {k: {f: v[f] for f in ("shape", "dtype")}
                            for k, v in data_contract({name: world.data[name] for name in
                                ("treatments", "covariates", "outcome", "saturation_scale")}).items()},
            "free_value_layout": layout}


def validate_optimizer(point: Any, raw: Any) -> tuple[dict[str, Any], bool]:
    import numpy as np
    fields = ("success", "status", "message", "nfev", "njev", "nit", "fun", "x", "jac")
    evidence = {key: plain(getattr(raw, key)) for key in fields if hasattr(raw, key)} if raw is not None else {}
    reasons = []
    required = ("success", "status", "message", "nfev", "njev", "nit", "fun", "x", "jac")
    if raw is None:
        reasons.append("raw optimizer result is None (including native evaluation-cap interruption)")
    else:
        missing = [key for key in required if not hasattr(raw, key)]
        if missing:
            reasons.append("missing raw exit fields: " + ", ".join(missing))
        for key in ("fun", "x", "jac"):
            try:
                if not np.asarray(getattr(raw, key)).size or not np.isfinite(getattr(raw, key)).all():
                    reasons.append(f"nonfinite/empty raw {key}")
            except (AttributeError, TypeError, ValueError):
                reasons.append(f"invalid raw {key}")
        success, status = getattr(raw, "success", None), getattr(raw, "status", None)
        if not isinstance(success, (bool, np.bool_)) or not success or not isinstance(status, (int, np.integer)) or status != 0:
            reasons.append("optimizer did not report a successful exit")
        if not isinstance(getattr(raw, "message", None), str) or not raw.message:
            reasons.append("missing optimizer exit message")
        if np.asarray(getattr(raw, "fun", None)).shape != ():
            reasons.append("raw objective is not scalar")
        for key in ("nfev", "njev", "nit"):
            value = getattr(raw, key, None)
            if isinstance(value, bool) or not isinstance(value, (int, np.integer)) or value < 0:
                reasons.append(f"invalid raw count {key}")
        if any(isinstance(getattr(raw, name, None), (int, np.integer)) and
               getattr(raw, name) >= 1000 for name in ("nfev", "nit")):
            reasons.append("optimizer cap reached")
    try:
        finite = bool(point) and all(np.isfinite(v).all() for v in point.values())
    except (AttributeError, TypeError, ValueError):
        finite = False
    if not finite:
        reasons.append("missing/nonfinite returned point")
    evidence.update(valid_exit=not reasons, failure_reasons=reasons, point_finite=finite)
    return evidence, not reasons


def recovery_diagnostics(world: Any, point: dict[str, Any], layout: dict[str, Any]) -> dict[str, Any]:
    """Only called after MAP. Exact public names; no inferred coordinate aliases."""
    import numpy as np
    primitive = world.primitive_parameters
    params = world.params
    candidates: dict[str, tuple[Any, str]] = {k: (v, "primitive_parameters") for k, v in primitive.items()}
    unmapped = []
    def visit(prefix: str, value: Any) -> None:
        if isinstance(value, dict):
            for k, v in value.items():
                visit(f"{prefix}_{k}" if prefix else k, v)
        else:
            candidates.setdefault("param_" + prefix, (value, "params." + prefix))
    for key, value in params.items():
        visit(key, value)
    mapped = {}
    dims_by_name = {m["rv"]: m for m in layout["mapping"]}
    # Native oracle names may directly match public derived parameter leaves.
    # No transform inversion or guessed renaming of unrelated coordinates.
    for name, (truth, origin) in list(candidates.items()):
        if name.startswith("param_") and name not in point and name[6:] in point:
            candidates.setdefault(name[6:], (truth, origin))
    for name, (truth, origin) in candidates.items():
        if name not in point:
            unmapped.append({"name": name, "source": origin, "reason": "not returned by marginal native MAP"})
            continue
        target, estimate = np.asarray(truth), np.asarray(point[name])
        if target.shape != estimate.shape or target.dtype.kind not in "fiub" or estimate.dtype.kind not in "fiub":
            unmapped.append({"name": name, "source": origin, "reason": "shape/type mismatch"})
            continue
        if not np.isfinite(target).all() or not np.isfinite(estimate).all() or not target.size:
            unmapped.append({"name": name, "source": origin, "reason": "nonfinite/empty values"})
            continue
        error = estimate.astype(float) - target.astype(float)
        mapped[name] = {"source": origin, "shape": list(target.shape), "truth": plain(target),
            "estimate": plain(estimate), "bias": float(error.mean()),
            "mae": float(np.abs(error).mean()), "rmse": float(np.sqrt(np.mean(error ** 2))),
            "max_abs_error": float(np.abs(error).max()), "coordinate_mapping": dims_by_name.get(name, "named public param deterministic")}
    return {"truth_digest": digest(canonical({"params": params, "primitive_parameters": primitive})),
            "truth_arrays": {"params": plain(params), "primitive_parameters": plain(primitive)},
            "mapped": mapped, "unmapped": unmapped,
            "interpretation": "descriptive point recovery only; no identifiability/calibration claim"}


@contextlib.contextmanager
def forbid_native_fallback():
    """Abort PyMC's documented warning branch before it dispatches Powell.

    This is a scoped logging observer, not optimizer/compiler instrumentation or
    monkeypatching. The selected native implementation warns before changing method.
    """
    class NoFallback(logging.Handler):
        def emit(self, record):
            if "Defaulting to non-gradient minimization" in record.getMessage():
                raise RuntimeError("native gradient failure: solver fallback forbidden")
    logger = logging.getLogger("pymc")
    handler = NoFallback()
    logger.addHandler(handler)
    try:
        yield
    finally:
        logger.removeHandler(handler)


def _case(case: dict[str, Any], regime: str, root: Path, checkpoint=None) -> dict[str, Any]:
    import pymc as pm
    from pymc_generator import sample_scm
    begin = time.monotonic()
    result: dict[str, Any] = {**case, "status": "started", "started_at": utc(), "timers": {"serialization": 0.0},
        "pid": os.getpid(), "argv": sys.argv, "environment": dict(os.environ), "phase": "generation",
        "generation_attempted": False, "fit_attempted": False}
    def save() -> None:
        t = time.monotonic()
        if checkpoint:
            checkpoint(result)
        else:
            canonical(result)
        result["timers"]["serialization"] += time.monotonic() - t
    @contextlib.contextmanager
    def phase(name: str):
        result["phase"] = name
        save()
        t = time.monotonic()
        try:
            yield
        finally:
            result["timers"][name] = time.monotonic() - t
    try:
        with phase("config_generation"):
            cfg = build_config(regime)
            result["generation_attempted"] = True
            save()
            world = sample_scm(cfg, seed=case["world_seed"], connect_all=False,
                               name=case["case_id"], purpose="issue-30-cold-smoke-report")
            result["resolved_config"] = dataclasses.asdict(cfg)
        with phase("model_build"):
            model = world.oracle_model(latent="marginal")
        with phase("support_initial_point"):
            start = model.initial_point(random_seed=case["support_seed"])
        with phase("pre_fit_contract_diagnostics"):
            observed = {name: world.data[name] for name in ("treatments", "covariates", "outcome", "saturation_scale")}
            result["data"] = {"arrays": data_contract(observed), "sha256": digest(canonical(observed)),
                              "likelihood_rows": list(range(104)), "generation_B": 0, "discard": 0}
            result["graph"] = plain(world.g)
            result["families_and_structure"] = plain(world.extras["structural"])
            result["free_value_layout"] = layout_contract(model, start)
            contract = compatibility_contract(world, result["resolved_config"], result["free_value_layout"])
            result["compatibility"] = {"contract": contract, "sha256": digest(canonical(contract))}
            # Public symbolic gradient check: fail rather than request a fallback.
            model.dlogp(vars=list(model.free_RVs), jacobian=False)
        result["fit_attempted"] = True
        with phase("find_MAP_solve_inclusive_compilation"):
            with model, forbid_native_fallback():
                point, raw = pm.find_MAP(start=start, vars=list(model.free_RVs), method="L-BFGS-B",
                    return_raw=True, include_transformed=True, progressbar=False, maxeval=998, options=dict(OPTIONS))
        with phase("diagnostics"):
            result["optimizer"], good = validate_optimizer(point, raw)
            result["point_finite"] = result["optimizer"]["point_finite"]
            result["point"] = plain(point)
            save()
            result["recovery"] = recovery_diagnostics(world, point or {}, result["free_value_layout"])
            result["full_data"] = {"arrays": data_contract(world.data), "sha256": digest(canonical(world.data))}
            result["status"] = "success" if good else "optimizer_failure"
    except Exception as exc:
        result.update(status="failure", exception_type=type(exc).__name__, exception=str(exc), traceback=traceback.format_exc())
        # Truth is still diagnostic only, including a failed MAP attempt.
        if "world" in locals() and result["phase"] in ("find_MAP_solve_inclusive_compilation", "diagnostics"):
            t = time.monotonic()
            try:
                result["recovery"] = recovery_diagnostics(world, {}, result["free_value_layout"])
                result["full_data"] = {"arrays": data_contract(world.data), "sha256": digest(canonical(world.data))}
            except Exception as error:
                result["recovery_error"] = f"{type(error).__name__}: {error}"
            result["timers"]["diagnostics"] = time.monotonic() - t
    result["total_seconds"] = time.monotonic() - begin
    result["ended_at"] = utc()
    result["peak_memory_bytes"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024
    save()
    result["total_seconds"] = time.monotonic() - begin
    return plain(result)


def _tree_bytes(path: Path | None) -> int:
    if path is None or not path.exists():
        return 0
    total = 0
    def unreadable(error):
        raise error
    for directory, dirs, files in os.walk(path, followlinks=False, onerror=unreadable):
        for name in dirs + files:
            item = Path(directory) / name
            try:
                info = item.lstat()
            except FileNotFoundError:  # compiler temporary removed between list/stat
                continue
            if stat.S_ISLNK(info.st_mode) or info.st_uid != os.getuid():
                raise RuntimeError(f"unowned/symlink resource entry: {item}")
            if stat.S_ISREG(info.st_mode):
                total += max(info.st_size, info.st_blocks * 512)
    return total


def _resource_snapshot(cache_dir: Path, output: Path | None, limit: int = 10737418240) -> dict[str, Any]:
    if not cache_dir.is_dir():
        raise RuntimeError("private cache disappeared; resource measurement unavailable")
    cache_bytes, artifact_bytes = _tree_bytes(cache_dir), _tree_bytes(output)
    result = {"timestamp": utc(), "cache_bytes": cache_bytes, "artifact_bytes": artifact_bytes,
              "peak_rss_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024}
    if cache_bytes + artifact_bytes >= limit - 134217728:
        raise RuntimeError("cache/artifact budget reached 128 MiB finalization reserve")
    for path in (cache_dir, output if output and output.exists() else cache_dir):
        fs = os.statvfs(path)
        if fs.f_bavail * fs.f_frsize < 134217728:
            raise RuntimeError("filesystem free space reached finalization reserve")
    return result


def _atomic(path: Path, content: bytes, *, replace: bool = False) -> None:
    """fsync payload, publish atomically, fsync directory. Case records never replace."""
    fd, temporary = tempfile.mkstemp(prefix=".pending-", dir=path.parent)
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
        if replace:
            os.replace(temporary, path)
        else:
            os.link(temporary, path)  # fails atomically if the destination exists
        directory = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def write_json(path: Path, value: Any, *, replace: bool = False) -> None:
    _atomic(path, json.dumps(plain(value), sort_keys=True, indent=2, allow_nan=False).encode() + b"\n", replace=replace)


def _checkpoint(output: Path, report: dict[str, Any]) -> None:
    # JSON is authoritative; Markdown is independently atomic and can be regenerated.
    write_json(output / "map-recovery-smoke.json", report, replace=True)
    _atomic(output / "map-recovery-smoke.md", render_markdown(report).encode(), replace=True)


def remaining_allows_case(elapsed: float) -> bool:
    return elapsed < 5220 and 5400 - elapsed >= 930 + 180


def process_table() -> dict[int, dict[str, Any]]:
    table = {}
    for path in Path("/proc").glob("[0-9]*/stat"):
        try:
            text = path.read_text()
            fields = text[text.rfind(")") + 2:].split()
            pid = int(path.parent.name)
            table[pid] = {"pid": pid, "state": fields[0], "ppid": int(fields[1]), "pgid": int(fields[2]),
                          "start_ticks": int(fields[19]), "vsize_bytes": int(fields[20]),
                          "rss_bytes": int(fields[21]) * os.sysconf("SC_PAGE_SIZE")}
        except (FileNotFoundError, ProcessLookupError):
            pass
    return table


def descendants(table: dict[int, dict[str, Any]], root: int) -> dict[int, dict[str, Any]]:
    selected = {root}
    while True:
        more = {pid for pid, info in table.items() if info["ppid"] in selected}
        if more <= selected:
            break
        selected |= more
    return {pid: table[pid] for pid in selected if pid in table and table[pid]["state"] != "Z"}


def subreaper() -> None:
    # Linux-only guard feasibility is mandatory, not a best-effort resource claim.
    libc = ctypes.CDLL(None, use_errno=True)
    if libc.prctl(36, 1, 0, 0, 0) != 0:  # PR_SET_CHILD_SUBREAPER
        raise OSError(ctypes.get_errno(), "cannot establish descendant subreaper")


def reap_adopted(exclude_pids: tuple[int, ...] = ()) -> None:
    while True:
        reaped = False
        for pid, info in process_table().items():
            if info["ppid"] != os.getpid() or pid in exclude_pids:
                continue
            try:
                waited, _ = os.waitpid(pid, os.WNOHANG)
                reaped |= bool(waited)
            except ChildProcessError:
                pass
        if not reaped:
            return


def signal_tree(process: subprocess.Popen, known: dict[int, dict[str, Any]], sig: int) -> None:
    table = process_table()
    live = {pid: info for pid, info in known.items()
            if pid in table and table[pid]["start_ticks"] == info["start_ticks"] and table[pid]["state"] != "Z"}
    groups = {info["pgid"] for info in live.values() if info["pgid"] != os.getpgrp()}
    for group in groups:
        try:
            os.killpg(group, sig)
        except ProcessLookupError:
            pass
    # Include any descendant that created a new group after the previous poll.
    for pid in live:
        try:
            os.kill(pid, sig)
        except ProcessLookupError:
            pass


def clean_environment() -> dict[str, str]:
    # The exact child environment is serialized, with no inherited credentials.
    names = (*THREAD_ENV, "PYTHONPATH", "PYTENSOR_FLAGS", "PATH", "HOME", "LANG", "LC_ALL", "LC_CTYPE", "TMPDIR")
    env = {name: os.environ[name] for name in names if name in os.environ}
    env.update(PYTHONDONTWRITEBYTECODE="1", PYTHONNOUSERSITE="1")
    return env


def supervise(argv: list[str], output: Path, label: str, cache: Path,
              timeout: float, *, pass_fds: tuple[int, ...] = (), stop=None,
              aggregate_start: float | None = None, poll_seconds: float = 1,
              kill_grace: float = 30, exclude_pids: tuple[int, ...] = ()) -> dict[str, Any]:
    """Real OS process-group guard. Test-only parameters shorten simulated sleepers."""
    begin = time.monotonic()
    env = clean_environment()
    if argv[:2] == [PYTHON, str(ROOT / "scripts/map_recovery_pilot.py")]:
        if "--case-index" in argv:
            index = int(argv[argv.index("--case-index") + 1])
            case = load_config(Path(argv[argv.index("--config") + 1]))["cases"][index]
            case_cache = cache / case["case_id"]
            case_cache.mkdir(mode=0o700)  # Exclusive: never warm or reuse a case cache.
            env["PYTENSOR_FLAGS"] = f"base_compiledir={case_cache}"
        write_json(output / f"{label}.launch.json", expected_launch(argv, env, cache))
    record: dict[str, Any] = {"argv": argv, "environment": env, "started_at": utc(),
        "term_after_seconds": timeout, "kill_after_seconds": kill_grace, "poll_seconds": poll_seconds,
        "samples": [], "signals": [], "exit_code": None, "reason": None,
        "stdout": f"{label}.stdout", "stderr": f"{label}.stderr"}
    known: dict[int, dict[str, Any]] = {}
    term_at = None
    work_stop_sent = False
    with (output / record["stdout"]).open("xb") as stdout, (output / record["stderr"]).open("xb") as stderr:
        process = subprocess.Popen(argv, cwd=ROOT, env=env, stdout=stdout, stderr=stderr,
                                   start_new_session=True, pass_fds=pass_fds)
        record.update(pid=process.pid, pgid=process.pid)
        try:
            while True:
                now = time.monotonic()
                table = process_table()
                known.update(descendants(table, process.pid))
                # Subreaper adopts daemonized/orphaned grandchildren, not unrelated processes.
                known.update({pid: info for pid, info in descendants(table, os.getpid()).items()
                              if pid != os.getpid() and pid not in exclude_pids})
                live = {pid: info for pid, info in known.items() if pid in table
                        and table[pid]["start_ticks"] == info["start_ticks"] and table[pid]["state"] != "Z"}
                status = process.poll()
                if status is not None:
                    # Avoid treating a direct child which exited between /proc
                    # sampling and poll() as a surviving descendant.
                    table = process_table()
                    live = {pid: info for pid, info in live.items() if pid != process.pid
                            and pid in table and table[pid]["start_ticks"] == info["start_ticks"]
                            and table[pid]["state"] != "Z"}
                try:
                    if not output.is_dir():
                        raise RuntimeError("artifact directory disappeared; resource measurement unavailable")
                    resources = _resource_snapshot(cache, output)
                    resources["processes"] = [table[pid] for pid in sorted(live)]
                    resources["elapsed_seconds"] = now - begin
                    record["samples"].append(resources)
                    if any(info["vsize_bytes"] > 17179869184 for info in resources["processes"]):
                        raise RuntimeError("observed process exceeds 16 GiB address space")
                except Exception as exc:
                    record["reason"] = record["reason"] or f"resource_monitor: {type(exc).__name__}: {exc}"
                if aggregate_start is not None:
                    elapsed = now - aggregate_start
                    if elapsed >= 5220 and not work_stop_sent and status is None:
                        os.kill(process.pid, signal.SIGUSR1)
                        work_stop_sent = True
                        record["signals"].append({"signal": "USR1", "elapsed_seconds": now - begin, "reason": "work_stop_5220"})
                    if elapsed >= 5370:
                        record["reason"] = record["reason"] or "finalization_boundary_5370"
                if now - begin >= timeout:
                    record["reason"] = record["reason"] or "timeout"
                if stop and stop():
                    record["reason"] = record["reason"] or stop()
                if status is not None and not live:
                    break
                if status is not None and live:
                    record["reason"] = record["reason"] or "descendant_cleanup_after_exit"
                if record["reason"] and term_at is None:
                    signal_tree(process, known, signal.SIGTERM)
                    term_at = now
                    record["signals"].append({"signal": "TERM", "elapsed_seconds": now - begin})
                if term_at is not None and now - term_at >= kill_grace:
                    signal_tree(process, known, signal.SIGKILL)
                    record["signals"].append({"signal": "KILL", "elapsed_seconds": now - begin})
                    process.wait(timeout=5)
                    break
                time.sleep(max(0, poll_seconds - (time.monotonic() - now)))
        finally:
            # Even monitor/serialization errors must not leak a running fit.
            table = process_table()
            known.update(descendants(table, process.pid))
            # Kill all known surviving groups even when the direct child exited.
            signal_tree(process, known, signal.SIGKILL)
            process.wait(timeout=5)
            record["exit_code"] = process.returncode
            reap_adopted(exclude_pids)
            stdout.flush(); stderr.flush()
            os.fsync(stdout.fileno()); os.fsync(stderr.fileno())
    record["ended_at"] = utc()
    record["total_seconds"] = time.monotonic() - begin
    record["observed_processes"] = list(known.values())
    table = process_table()
    survivors = [pid for pid, info in known.items() if pid in table
                 and table[pid]["start_ticks"] == info["start_ticks"] and table[pid]["state"] != "Z"]
    record["survivors"] = survivors
    if survivors:
        raise RuntimeError(f"descendant cleanup failed: {survivors}")
    for stream in ("stdout", "stderr"):
        path = output / record[stream]
        with path.open("rb") as handle:
            record[stream + "_sha256"] = hashlib.file_digest(handle, "sha256").hexdigest()
        record[stream + "_bytes"] = path.stat().st_size
    return record


def reuse_estimate(cases: list[dict[str, Any]]) -> dict[str, Any]:
    attempted = [c for c in cases if c.get("status") != "unattempted"]
    known = [c for c in attempted if "compatibility" in c]
    groups: dict[str, list[str]] = {}
    for c in known:
        # Recompute from actual serialized contracts, not case labels or supplied hashes.
        key = digest(canonical(c["compatibility"]["contract"]))
        groups.setdefault(key, []).append(c["case_id"])
    n, k = len(known), len(groups)
    timing_names = ("model_build", "support_initial_point", "find_MAP_solve_inclusive_compilation")
    timed = [c for c in known if all(name in c.get("timers", {}) for name in timing_names)]
    inclusive = [c["timers"]["find_MAP_solve_inclusive_compilation"] for c in timed]
    setup = [c["timers"]["model_build"] + c["timers"]["support_initial_point"] for c in timed]
    high_compile, high_setup = max(inclusive, default=0), max(setup, default=0)
    low_setup = min(setup, default=0)
    scenarios = []
    if known and len(timed) == len(known):
        for fraction in (0, 0.01, 0.1, 1):
            bind_high = fraction * (high_setup + high_compile)
            scenarios.append({"label": f"assumed bind in [0,{fraction} * max cold setup+inclusive API]",
                "compile_seconds_range": [0, high_compile], "bind_seconds_range": [0, bind_high],
                "setup_plus_compile_seconds_range": [low_setup, high_setup + high_compile],
                "avoided_seconds_range": [(n-k)*low_setup-n*bind_high, (n-k)*(high_setup+high_compile)]})
    elif known:
        scenarios.append({"label": "symbolic only: interrupted/missing inclusive API timings",
            "compile_seconds_range": [0, "C_compile_hi (unknown)"],
            "bind_seconds_range": [0, "B_hi (unknown)"],
            "avoided_seconds_range": ["(N-K)*C_setup - N*B_hi", "(N-K)*(C_setup+C_compile_hi)"],
            "units": "symbolic seconds, not measured"})
    return {"N": len(attempted), "N_compatible_evidence": n,
            "K": k if len(known) == len(attempted) and known else "unavailable for cases without complete graph/layout evidence",
            "known_K": k, "N_complete_timing_evidence": len(timed), "groups": groups, "unknown_cases": [c["case_id"] for c in attempted if "compatibility" not in c],
            "measured_compile_cost": "unavailable", "formula": "(N-K)*C - N*bind",
            "counterfactual_only": True, "solve_work_avoided": 0, "scenarios": scenarios,
            "projection20": {"label": "symbolic projection only, not measured20-world cost",
                "K_range": [1, 20], "formula": "(20-K20)*C - 20*bind", "K20": "unknown"},
            "caveat": "scenario endpoints are assumptions bounded by inclusive timing, not measured compilation/binding or speedup; solve work unchanged"}


def render_markdown(report: dict[str, Any]) -> str:
    rows = ["# Cold MAP recovery smoke report", "", f"Source: `{report.get('source_commit', BASE)}`", "",
            *[f"- {s}" for s in LIMITATIONS], "", "## Case ledger", "",
            "| Case | Status | seconds |", "|---|---|---:|"]
    rows += [f"| {c.get('case_id')} | {c.get('status')} | {c.get('total_seconds', 'unavailable')} |" for c in report.get("cases", [])]
    rows += ["", "## Counterfactual reuse", "", "```json", json.dumps(plain(report.get("reuse_counterfactual", {})), indent=2), "```",
             "", "Authoritative JSON and immutable per-case checkpoints retain full provenance, recovery, timers and process records. No compilation/solve split is measured."]
    return "\n".join(rows) + "\n"


def fresh_report(cfg: dict[str, Any]) -> dict[str, Any]:
    return {"schema_version": "map-recovery-smoke-report/v2", "source_commit": BASE, "config": cfg,
            "config_sha256": digest(canonical(cfg)), "started_at": utc(), "limitations": LIMITATIONS,
            "cases": [{**c, "status": "unattempted", "generation_attempted": False, "fit_attempted": False,
                       "reason": "not yet launched"} for c in cfg["cases"]]}


def child_command(config: Path, output: Path, lock_fd: int, *args: str) -> list[str]:
    return [PYTHON, str(ROOT / "scripts/map_recovery_pilot.py"), "--config", str(config.resolve()),
            "--output", str(output.resolve()), "--lock-fd", str(lock_fd), *args]


def _case_child(args: Any) -> int:
    cfg = load_config(args.config)
    case = cfg["cases"][args.case_index]
    gate = preflight(args.config, args.output, args.lock_fd)
    write_json(args.output / f"{case['case_id']}.preflight.json", gate)
    if not gate["ok"]:
        write_json(args.output / f"{case['case_id']}.result.json", {**case, "status": "failure", "generation_attempted": False, "fit_attempted": False,
            "reason": "preflight blocked", "preflight": gate})
        return 2
    sequence = 0
    def checkpoint(record):
        nonlocal sequence
        write_json(args.output / f"{case['case_id']}.phase-{sequence:02d}.json", record)
        sequence += 1
    record = _case(case, case["regime"], ROOT, checkpoint)
    record["preflight"] = gate
    write_json(args.output / f"{case['case_id']}.result.json", record)
    return 0 if record["status"] == "success" else 1


def recover_entry(output: Path, case: dict[str, Any], process: dict[str, Any] | None, reason: str) -> dict[str, Any]:
    path = output / f"{case['case_id']}.result.json"
    phases = sorted(output.glob(f"{case['case_id']}.phase-*.json"))
    record = json.loads(path.read_text()) if path.exists() else json.loads(phases[-1].read_text()) if phases else {**case, "generation_attempted": False, "fit_attempted": False}
    if process is None or process["exit_code"] != 0 or process.get("reason") or not path.exists():
        record["status"] = "timeout" if process and process.get("reason") == "timeout" else "failure"
        record["reason"] = reason
    if process:
        record["process"] = process
    return record


def _supervisor(args: Any) -> int:
    subreaper()
    started = float(args.started)
    stop_reason = [None]
    def request_stop(signum, frame):
        stop_reason[0] = f"supervisor_signal_{signal.Signals(signum).name}"
    for sig in (signal.SIGTERM, signal.SIGINT, signal.SIGUSR1):
        signal.signal(sig, request_stop)
    cfg = load_config(args.config)
    report = fresh_report(cfg)
    gate = preflight(args.config, args.output, args.lock_fd)
    report["preflight"] = gate
    _checkpoint(args.output, report)
    if not gate["ok"]:
        return 2
    for index, case in enumerate(cfg["cases"]):
        elapsed = time.monotonic() - started
        if stop_reason[0] or not remaining_allows_case(elapsed):
            for entry in report["cases"][index:]:
                entry["reason"] = stop_reason[0] or "remaining window below 930+180 seconds"
            break
        write_json(args.output / f"{case['case_id']}.attempt.json", {**case, "started_at": utc()})
        process = supervise(child_command(args.config, args.output, args.lock_fd, "--case-index", str(index)),
                            args.output, case["case_id"], cache_path(), 900,
                            pass_fds=(args.lock_fd,), stop=lambda: stop_reason[0] or (
                                "work_stop_5220" if time.monotonic()-started >= 5220 else None))
        write_json(args.output / f"{case['case_id']}.process.json", process)
        entry = recover_entry(args.output, case, process, process["reason"] or f"child_exit_{process['exit_code']}")
        write_json(args.output / f"{case['case_id']}.checkpoint.json", entry)
        report["cases"][index] = entry
        report["reuse_counterfactual"] = reuse_estimate(report["cases"])
        _checkpoint(args.output, report)
        if process["reason"] and process["reason"].startswith("resource_monitor"):
            stop_reason[0] = process["reason"]
    t = time.monotonic()
    report["ended_at"] = utc()
    report["total_seconds"] = time.monotonic() - started
    report["finalization_deadline_met"] = report["total_seconds"] < 5370
    for entry in report["cases"]:
        path = args.output / f"{entry['case_id']}.checkpoint.json"
        if not path.exists():
            write_json(path, entry)
    report["reuse_counterfactual"] = reuse_estimate(report["cases"])
    _checkpoint(args.output, report)
    report["finalization_seconds"] = time.monotonic() - t
    _checkpoint(args.output, report)
    return 0 if all(c["status"] == "success" for c in report["cases"]) else 1


def _watchdog(args: Any) -> int:
    """Independent hard guard: no scientific imports, even if parent I/O stalls."""
    parent = args.watchdog_parent
    started = float(args.started)
    table = process_table()
    if os.getppid() != parent or parent not in table:
        raise RuntimeError("watchdog requires its live launcher parent")
    parent_ticks, parent_group = table[parent]["start_ticks"], table[parent]["pgid"]
    known = {}
    term_at = None
    reason = None
    write_json(args.output / "watchdog.ready.json", {"pid": os.getpid(), "parent": parent,
        "parent_start_ticks": parent_ticks, "started_monotonic": started,
        "aggregate_term_seconds": 5400, "kill_grace_seconds": 30, "poll_seconds": 1,
        "argv": sys.argv, "environment": dict(os.environ),
        "actual_address_space_limits": list(resource.getrlimit(resource.RLIMIT_AS))})
    while True:
        now = time.monotonic()
        table = process_table()
        known.update({pid: info for pid, info in descendants(table, parent).items() if pid != os.getpid()})
        live = {pid: info for pid, info in known.items() if pid in table
                and table[pid]["start_ticks"] == info["start_ticks"] and table[pid]["state"] != "Z"}
        if parent not in table or table[parent]["start_ticks"] != parent_ticks:
            reason = reason or "launcher exited without completion handshake"
        try:
            sample = _resource_snapshot(cache_path(), args.output)
            sample.update(elapsed_seconds=now-started, processes=[table[pid] for pid in live])
            print(json.dumps(plain(sample)), flush=True)
        except Exception as exc:
            reason = reason or f"watchdog resource monitor: {type(exc).__name__}: {exc}"
        if now - started >= 5400:
            reason = reason or "aggregate_TERM5400"
        ready, _, _ = select.select([args.control_fd], [], [], 0)
        if ready:
            message = os.read(args.control_fd, 16)
            if message == b"DONE" and not reason and not (set(live) - {parent}):
                print(json.dumps({"status": "completed", "timestamp": utc()}), flush=True)
                os.fsync(sys.stdout.fileno())
                return 0
            reason = reason or "launcher completion/pipe failed with surviving descendants"
        # Never signal the launcher's inherited group (which may contain a shell).
        # Fresh supervisor/case groups are safe; launcher itself is signaled by PID.
        safe = {pid: {**info, "pgid": os.getpgrp() if info["pgid"] == parent_group else info["pgid"]}
                for pid, info in known.items()}
        if reason and term_at is None:
            signal_tree(None, safe, signal.SIGTERM)
            term_at = now
            print(json.dumps({"signal": "TERM", "reason": reason, "elapsed_seconds": now-started}), flush=True)
        if term_at is not None and now-term_at >= 30:
            signal_tree(None, safe, signal.SIGKILL)
            print(json.dumps({"signal": "KILL", "reason": reason, "elapsed_seconds": now-started}), flush=True)
            os.fsync(sys.stdout.fileno())
            return 1
        if reason and not live:
            return 1
        time.sleep(1)


@contextlib.contextmanager
def hard_watchdog(config: Path, output: Path, started: float):
    read_fd, write_fd = os.pipe()
    argv = [PYTHON, str(ROOT / "scripts/map_recovery_pilot.py"), "--config", str(config),
            "--output", str(output), "--watchdog-parent", str(os.getpid()),
            "--started", str(started), "--control-fd", str(read_fd)]
    env = clean_environment()
    write_json(output / "watchdog.launch.json", expected_launch(argv, env, cache_path()))
    with (output / "watchdog.stdout").open("xb") as stdout, (output / "watchdog.stderr").open("xb") as stderr:
        guard = subprocess.Popen(argv, cwd=ROOT, env=env, stdout=stdout, stderr=stderr,
                                 pass_fds=(read_fd,), start_new_session=True)
        os.close(read_fd)
        try:
            deadline = time.monotonic() + 10
            while not (output / "watchdog.ready.json").exists():
                if guard.poll() is not None or time.monotonic() >= deadline:
                    raise RuntimeError("independent hard watchdog failed before supervisor launch")
                time.sleep(.05)
            yield guard.pid
        finally:
            try:
                os.write(write_fd, b"DONE")
            except BrokenPipeError:
                pass
            os.close(write_fd)
            try:
                code = guard.wait(timeout=32)
            except subprocess.TimeoutExpired:
                guard.kill()
                guard.wait()
                raise RuntimeError("independent hard watchdog failed to finish")
            write_json(output / "watchdog.process.json", {"exit_code": code, "ended_at": utc()})
            if code:
                raise RuntimeError(f"independent hard watchdog exited {code}")


def run(config_path: Path, output: Path, preflight_only: bool = False, released: bool = False) -> int:
    cfg = load_config(config_path)
    if preflight_only:
        gate = preflight(config_path, output)
        print(json.dumps(plain(gate), indent=2))
        return 0 if gate["ok"] else 2
    if not released:
        raise RuntimeError("no execution: separate owner release after independent acceptance is required")
    if output.exists():
        raise FileExistsError(f"refusing to clobber {output}")
    subreaper()
    # Cheap checks here; full imports/provenance are inside the guarded supervisor.
    cache = cache_path()
    stop_reason = [None]
    def request_stop(signum, frame):
        stop_reason[0] = f"launcher_signal_{signal.Signals(signum).name}"
    for sig in (signal.SIGTERM, signal.SIGINT):
        signal.signal(sig, request_stop)
    with cache_lock(cache) as fd:
        output.mkdir(mode=0o700)  # atomic exclusive claim, never resume an old run
        write_json(output / "launcher.launch.json",
                   expected_launch([sys.executable, *sys.argv], clean_environment(), cache))
        started = time.monotonic()
        with hard_watchdog(config_path, output, started) as watchdog_pid:
            process = supervise(child_command(config_path, output, fd, "--supervisor", "--started", str(started)),
                                output, "supervisor", cache, 5400, pass_fds=(fd,),
                                stop=lambda: stop_reason[0], aggregate_start=started, exclude_pids=(watchdog_pid,))
            write_json(output / "supervisor.process.json", process)
            # Launcher repairs the ledger from immutable checkpoints after interruptions.
            report_path = output / "map-recovery-smoke.json"
            report = json.loads(report_path.read_text()) if report_path.exists() else fresh_report(cfg)
            for index, case in enumerate(cfg["cases"]):
                checkpoint = output / f"{case['case_id']}.checkpoint.json"
                if checkpoint.exists():
                    entry = json.loads(checkpoint.read_text())
                elif (output / f"{case['case_id']}.attempt.json").exists():
                    entry = recover_entry(output, case, None, "aggregate interrupted before durable child exit")
                    write_json(checkpoint, entry)
                else:
                    entry = {**case, "status": "unattempted", "generation_attempted": False, "fit_attempted": False,
                             "reason": process["reason"] or "supervisor stopped before launch"}
                    write_json(checkpoint, entry)
                report["cases"][index] = entry
            report["supervisor_process"] = process
            report["reuse_counterfactual"] = reuse_estimate(report["cases"])
            report["ended_at"] = utc()
            report["total_seconds"] = time.monotonic() - started
            _checkpoint(output, report)
    return 0 if process["exit_code"] == 0 and not process["reason"] else 1


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / CONFIG_DEFAULT)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--preflight", action="store_true")
    parser.add_argument("--execute-released", action="store_true")
    parser.add_argument("--supervisor", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--case-index", type=int, choices=range(4), help=argparse.SUPPRESS)
    parser.add_argument("--lock-fd", type=int, help=argparse.SUPPRESS)
    parser.add_argument("--started", help=argparse.SUPPRESS)
    parser.add_argument("--watchdog-parent", type=int, help=argparse.SUPPRESS)
    parser.add_argument("--control-fd", type=int, help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    script_context()
    if not (args.watchdog_parent is not None or args.supervisor or args.case_index is not None):
        mode = "--preflight" if args.preflight else "--execute-released"
        if [sys.executable, *sys.argv] != launcher_command(args.config, args.output, mode):
            raise RuntimeError("use the exact documented absolute-script config/output/mode argv")
    if args.watchdog_parent is not None:
        return _watchdog(args)
    if args.supervisor or args.case_index is not None:
        if args.lock_fd is None or os.getppid() == 1:
            raise RuntimeError("internal children require live parent and inherited cache lock")
        return _supervisor(args) if args.supervisor else _case_child(args)
    return run(args.config, args.output, args.preflight, args.execute_released)


if __name__ == "__main__":
    raise SystemExit(main())
