"""Record or compare seeded generation across supported model features.

Run from the repository with ``uv run python -m benchmarks.generation_baseline
record /tmp/generation-baseline`` before a refactor, then use ``compare`` with
that directory afterwards. Artifacts are deliberately external to the source
tree: equality is a same-environment refactoring check, not a cross-version
promise about numerical libraries.
"""

from __future__ import annotations

import argparse
import importlib.metadata
import json
from pathlib import Path

import numpy as np

import pymc_generator as pg


def cases():
    """Exercise graph padding, nonlinearities, shocks, floors and conditioning."""
    base = {
        "n_treatments": 4,
        "n_covariates": 2,
        "n_latent": 1,
        "n_time_steps": 32,
        "n_cells": 3,
        "draws_per_cell": 2,
        "seed": 731,
    }
    variants = {
        "diverse": {},
        "linear": {"nonlinearity": "linear"},
        "padded": {
            "n_treatments_active_range": (2, 4),
            "n_covariates_active_range": (1, 2),
        },
        "shocks": {
            "n_treatment_shocks": 1,
            "treatment_shock_length_range": (3, 5),
        },
        "intercept_floor": {"baseline_floor": 0.0},
        "non_treatment_floor": {"baseline_floor": 0.0, "baseline_floor_scope": "non_treatment"},
        "conditioned": {"prior_conditioning": True},
        "smooth_covariates": {
            "covariate_hf_sigma_range": (0.0, 0.0),
            "covariate_pulse_prob_range": (0.0, 0.0),
            "covariate_pulse_amp_range": (0.0, 0.0),
        },
    }
    for name, overrides in variants.items():
        yield name, pg.make_scm_prior(**(base | overrides))


def numeric_arrays(corpus):
    """Keep data and labels, excluding descriptive and wall-clock diagnostics."""
    arrays = {key: value for key, value in corpus.items() if isinstance(value, np.ndarray)}
    arrays.update(
        {f"identifiability/{key}": value for key, value in corpus["identifiability"].items()}
    )
    return arrays


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("record", "compare"))
    parser.add_argument("directory", type=Path)
    args = parser.parse_args()
    if args.mode == "record":
        args.directory.mkdir(parents=True, exist_ok=False)
        versions = {
            name: importlib.metadata.version(name)
            for name in ("pymc-generator", "numpy", "scipy", "pymc", "pytensor", "pymc-marketing")
        }
        (args.directory / "environment.json").write_text(json.dumps(versions, indent=2) + "\n")
    for name, prior in cases():
        corpus = pg.sample_prior_predictive(prior)
        errors = pg.DataGenerator.validate_corpus(corpus)
        if errors:
            raise AssertionError(f"{name}: {errors}")
        arrays = numeric_arrays(corpus)
        path = args.directory / f"{name}.npz"
        if args.mode == "record":
            np.savez_compressed(path, **arrays)
        else:
            with np.load(path, allow_pickle=False) as expected:
                if set(expected.files) != set(arrays):
                    raise AssertionError(f"{name}: numeric field inventory changed")
                for key, actual in arrays.items():
                    np.testing.assert_array_equal(actual, expected[key], err_msg=f"{name}/{key}")
        print(f"{args.mode}: {name}: {len(arrays)} arrays", flush=True)


if __name__ == "__main__":
    main()
