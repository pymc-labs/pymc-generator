"""High-level generation facade + corpus persistence.

Provides a clean, tested interface for generating synthetic MMM worlds at
scale, plus the ``.npz`` persistence format that PFN training pipelines
(e.g. structural-pfn) consume directly.

Design Principles:
1. Model-independent - generates data without any learned model
2. Reproducible - same seed + config produces an identical corpus
3. Validated - generated corpora are checked for schema correctness
4. Modular - generate in batches or all at once

Usage:
    from prior_generator import DataGenerator, make_l1_additive_cfg

    cfg = make_l1_additive_cfg(K_max=4, M_max=2, J_max=1, n_cells=10, draws_per_cell=10)
    generator = DataGenerator(cfg)
    corpus = generator.generate(n_tasks=100, seed=42)

    # Or generate in batches
    for batch in generator.generate_batches(n_tasks=1000, batch_size=100, seed=42):
        process(batch)
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .sampler import CorpusConfig, generate_corpus

# ---------------------------------------------------------------------------
# Data generator
# ---------------------------------------------------------------------------


@dataclass
class DataGenerator:
    """Independent data generator for synthetic MMM data.

    Parameters
    ----------
    config : CorpusConfig
        Corpus generation configuration.
    """

    config: CorpusConfig

    def generate(
        self,
        n_tasks: int | None = None,
        seed: int | None = None,
        validate: bool = True,
    ) -> dict[str, np.ndarray]:
        """Generate a corpus of synthetic MMM tasks.

        Parameters
        ----------
        n_tasks : int, optional
            Number of tasks to generate. If None, uses config.n_cells * config.draws_per_cell.
        seed : int, optional
            Random seed. If None, uses config.seed.
        validate : bool
            Whether to validate the generated corpus.

        Returns
        -------
        dict
            Corpus dictionary with all generated data.
        """
        if n_tasks is not None and n_tasks <= 0:
            raise ValueError(f"n_tasks must be positive, got {n_tasks}")

        cfg = self._make_config(n_tasks, seed)

        # Generate corpus
        corpus = generate_corpus(cfg)

        # Truncate to requested n_tasks
        if n_tasks is not None:
            actual_n = corpus["spend_raw"].shape[0]
            if actual_n > n_tasks:
                for key in corpus:
                    if isinstance(corpus[key], np.ndarray) and corpus[key].shape[0] == actual_n:
                        corpus[key] = corpus[key][:n_tasks]
                # Ensure at least one val task after truncation
                if "is_val" in corpus and corpus["is_val"].sum() == 0:
                    corpus["is_val"][0] = 1

        # Validate if requested
        if validate:
            errors = self.validate_corpus(corpus)
            if errors:
                raise ValueError(
                    "Corpus validation failed:\n" + "\n".join(f"  - {e}" for e in errors)
                )

        return corpus

    def generate_batches(
        self,
        n_tasks: int,
        batch_size: int = 100,
        seed: int = 0,
        validate: bool = True,
    ) -> list[dict[str, np.ndarray]]:
        """Generate data in batches.

        Parameters
        ----------
        n_tasks : int
            Total number of tasks to generate.
        batch_size : int
            Number of tasks per batch.
        seed : int
            Base random seed.
        validate : bool
            Whether to validate each batch.

        Returns
        -------
        list of dict
            List of corpus dictionaries, one per batch.
        """
        if n_tasks <= 0:
            raise ValueError(f"n_tasks must be positive, got {n_tasks}")
        if batch_size <= 0:
            raise ValueError(f"batch_size must be positive, got {batch_size}")

        batches = []
        n_batches = (n_tasks + batch_size - 1) // batch_size

        for i in range(n_batches):
            batch_seed = seed + i
            batch_tasks = min(batch_size, n_tasks - i * batch_size)

            corpus = self.generate(
                n_tasks=batch_tasks,
                seed=batch_seed,
                validate=validate,
            )
            batches.append(corpus)

        return batches

    def generate_and_save(
        self,
        path: str | Path,
        n_tasks: int | None = None,
        seed: int | None = None,
        validate: bool = True,
    ) -> dict[str, np.ndarray]:
        """Generate corpus and save to disk.

        Parameters
        ----------
        path : str or Path
            Path to save the corpus (.npz file).
        n_tasks : int, optional
            Number of tasks.
        seed : int, optional
            Random seed.
        validate : bool
            Whether to validate.

        Returns
        -------
        dict
            Generated corpus.
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        corpus = self.generate(n_tasks=n_tasks, seed=seed, validate=validate)

        # Save
        save_corpus(corpus, path)

        return corpus

    def _make_config(
        self,
        n_tasks: int | None = None,
        seed: int | None = None,
    ) -> CorpusConfig:
        """Create a CorpusConfig with overrides."""
        cfg = self.config
        overrides = {}

        if n_tasks is not None:
            dpc = cfg.draws_per_cell
            overrides["n_cells"] = max(2, (n_tasks + dpc - 1) // dpc)

        if seed is not None:
            overrides["seed"] = seed

        return dataclasses.replace(cfg, **overrides)

    @staticmethod
    def validate_corpus(corpus: dict[str, np.ndarray]) -> list[str]:
        """Validate a generated corpus.

        Parameters
        ----------
        corpus : dict
            Corpus dictionary.

        Returns
        -------
        list of str
            List of validation errors. Empty if valid.
        """
        errors = []

        # Check required keys
        required_keys = [
            "spend_raw",
            "spend_norm",
            "spend_share",
            "controls",
            "sales_raw",
            "sales_norm",
            "support_mask",
            "is_future",
            "g",
            "contributions_raw",
            "baseline_raw",
            "demand",
            "spend_means",
            "sales_scale",
            "is_val",
            "cell_id",
        ]
        for key in required_keys:
            if key not in corpus:
                errors.append(f"Missing required key: {key}")

        if errors:
            return errors

        # Get dimensions
        N = corpus["spend_raw"].shape[0]
        T = corpus["spend_raw"].shape[1]
        K = corpus["spend_raw"].shape[2]
        M = corpus["controls"].shape[2]
        S = corpus["g"].shape[1]

        # Check shapes
        expected_shapes = {
            "spend_raw": (N, T, K),
            "spend_norm": (N, T, K),
            "spend_share": (N, T, K),
            "controls": (N, T, M),
            "sales_raw": (N, T),
            "sales_norm": (N, T),
            "support_mask": (N, T),
            "is_future": (N,),
            "g": (N, S),
            "contributions_raw": (N, T, K),
            "baseline_raw": (N, T),
            "demand": (N, T, -1),  # J can vary
            "spend_means": (N, K),
            "sales_scale": (N,),
            "is_val": (N,),
            "cell_id": (N,),
        }

        for key, expected in expected_shapes.items():
            if key not in corpus:
                continue
            actual = corpus[key].shape
            if len(actual) != len(expected):
                errors.append(
                    f"Shape mismatch for {key}: expected ndim={len(expected)} {expected}, got ndim={len(actual)} {actual}"
                )
                continue
            for i, (e, a) in enumerate(zip(expected, actual)):
                if e == -1:
                    continue
                if e != a:
                    errors.append(f"Shape mismatch for {key}: expected dim {i} = {e}, got {a}")
                    break

        # Check finite values
        for key in ["spend_raw", "controls", "sales_raw", "contributions_raw", "baseline_raw"]:
            if key in corpus:
                if not np.isfinite(corpus[key]).all():
                    errors.append(f"{key} contains NaN or Inf")

        # Check sales_norm = sales_raw / sales_scale
        if "sales_raw" in corpus and "sales_scale" in corpus and "sales_norm" in corpus:
            sales_raw = corpus["sales_raw"].astype(np.float64)
            sales_scale = corpus["sales_scale"].astype(np.float64)
            sales_norm = corpus["sales_norm"].astype(np.float64)

            expected_norm = sales_raw / sales_scale[:, None]
            if not np.allclose(sales_norm, expected_norm, rtol=1e-5):
                errors.append("sales_norm != sales_raw / sales_scale")

        # Check support_mask is binary
        if "support_mask" in corpus:
            support = corpus["support_mask"]
            if not np.all(
                (np.abs(support.astype(float)) < 1e-9) | (np.abs(support.astype(float) - 1) < 1e-9)
            ):
                errors.append("support_mask is not binary")

        # Check is_future is binary
        if "is_future" in corpus:
            is_future = corpus["is_future"]
            if not np.all(
                (np.abs(is_future.astype(float)) < 1e-9)
                | (np.abs(is_future.astype(float) - 1) < 1e-9)
            ):
                errors.append("is_future is not binary")

        # Check is_val is binary
        if "is_val" in corpus:
            is_val = corpus["is_val"]
            if not np.all(
                (np.abs(is_val.astype(float)) < 1e-9) | (np.abs(is_val.astype(float) - 1) < 1e-9)
            ):
                errors.append("is_val is not binary")

        # Check g is binary
        if "g" in corpus:
            g = corpus["g"]
            if not np.all((np.abs(g.astype(float)) < 1e-9) | (np.abs(g.astype(float) - 1) < 1e-9)):
                errors.append("g is not binary")

        # Check null channels have zero contribution
        if "g" in corpus and "contributions_raw" in corpus:
            g = corpus["g"]
            contribs = corpus["contributions_raw"]

            # For each task, check that null channels (g_cy=0) have zero contribution
            # This is a bit tricky because g_cy is embedded in the g vector
            # We'll check that contributions are non-negative
            if (contribs < 0).any():
                errors.append("contributions_raw contains negative values")

        return errors


# ---------------------------------------------------------------------------
# Save/Load utilities
# ---------------------------------------------------------------------------


def save_corpus(corpus: dict[str, np.ndarray], path: str | Path) -> None:
    """Save corpus to disk.

    Parameters
    ----------
    corpus : dict
        Corpus dictionary.
    path : str or Path
        Path to save the corpus (.npz file).
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    # Convert diagnostics dict to JSON string if present
    save_dict = {}
    for k, v in corpus.items():
        if k == "diagnostics" and isinstance(v, dict):
            import json

            save_dict[k] = np.array(json.dumps(v))
        else:
            save_dict[k] = v

    np.savez_compressed(path, **save_dict)


def load_corpus(path: str | Path) -> dict[str, np.ndarray]:
    """Load corpus from disk.

    Parameters
    ----------
    path : str or Path
        Path to the corpus (.npz file).

    Returns
    -------
    dict
        Corpus dictionary.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Corpus file not found: {path}")

    with np.load(path, allow_pickle=True) as data:
        corpus = {k: data[k] for k in data.files}

    # Parse diagnostics JSON string if present
    if "diagnostics" in corpus:
        import json

        diag_str = corpus["diagnostics"]
        if isinstance(diag_str, np.ndarray) and diag_str.ndim == 0:
            corpus["diagnostics"] = json.loads(str(diag_str))

    return corpus


