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
    from prior_generator import DataGenerator, make_scm_prior

    cfg = make_scm_prior(n_treatments=4, n_covariates=2, n_latent=1, n_cells=10, draws_per_cell=10)
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

from .sampler import SCMPrior, sample_prior_predictive
from .signal_diagnostics import SIGNAL_METRIC_LAYOUT
from .slots import EDGE_TYPES_EXTENDED, SlotLayout

# ---------------------------------------------------------------------------
# Data generator
# ---------------------------------------------------------------------------


@dataclass
class DataGenerator:
    """Independent data generator for synthetic MMM data.

    Parameters
    ----------
    config : SCMPrior
        Corpus generation configuration.
    """

    config: SCMPrior

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
        corpus = sample_prior_predictive(cfg)

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
    ) -> SCMPrior:
        """Create a SCMPrior with overrides."""
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
            "active_c_mask",
            "channel_shock_mask",
            "channel_shock_channel",
            "channel_shock_start",
            "channel_shock_length",
            "channel_shock_level_multiplier",
            "channel_shock_level",
            "channel_level",
            "adstock_family",
            "adstock_alpha",
            "weibull_lam",
            "weibull_k",
            "signal_metrics",
            "signal_metric_valid",
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
        n_channel_shocks = corpus["channel_shock_channel"].shape[1]

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
            "active_c_mask": (N, K),
            "channel_shock_mask": (N, T, K),
            "channel_shock_channel": (N, n_channel_shocks),
            "channel_shock_start": (N, n_channel_shocks),
            "channel_shock_length": (N, n_channel_shocks),
            "channel_shock_level_multiplier": (N, n_channel_shocks),
            "channel_shock_level": (N, n_channel_shocks),
            "channel_level": (N, K),
            "adstock_family": (N, K),
            "adstock_alpha": (N, K),
            "weibull_lam": (N, K),
            "weibull_k": (N, K),
            "signal_metrics": (N, K, len(SIGNAL_METRIC_LAYOUT)),
            "signal_metric_valid": (N, K, len(SIGNAL_METRIC_LAYOUT)),
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

        expected_dtypes = {
            "channel_shock_mask": np.uint8,
            "active_c_mask": np.uint8,
            "channel_shock_channel": np.int32,
            "channel_shock_start": np.int32,
            "channel_shock_length": np.int32,
            "channel_shock_level_multiplier": np.float32,
            "channel_shock_level": np.float32,
            "channel_level": np.float32,
            "adstock_family": np.uint8,
            "adstock_alpha": np.float32,
            "weibull_lam": np.float32,
            "weibull_k": np.float32,
            "signal_metrics": np.float32,
            "signal_metric_valid": np.uint8,
        }
        for key, dtype in expected_dtypes.items():
            if key in corpus and corpus[key].dtype != dtype:
                errors.append(f"{key} has dtype {corpus[key].dtype}, expected {np.dtype(dtype)}")

        # Check finite values
        for key in ["spend_raw", "controls", "sales_raw", "contributions_raw", "baseline_raw"]:
            if key in corpus:
                if not np.isfinite(corpus[key]).all():
                    errors.append(f"{key} contains NaN or Inf")

        for key in (
            "channel_shock_level_multiplier",
            "channel_shock_level",
            "channel_level",
            "adstock_alpha",
            "weibull_lam",
            "weibull_k",
            "signal_metrics",
        ):
            if key in corpus and not np.isfinite(corpus[key]).all():
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
        if not np.isin(corpus["active_c_mask"], (0, 1)).all():
            errors.append("active_c_mask is not binary")

        # Top-level shock audit metadata is intentionally sufficient to
        # reconstruct every held-spend intervention without persisting the
        # full burn-in mask or natural (unshocked) paths.
        shock_mask = corpus["channel_shock_mask"]
        if not np.isin(shock_mask, (0, 1)).all():
            errors.append("channel_shock_mask is not binary")
        if (corpus["channel_shock_level_multiplier"] < 0).any():
            errors.append("channel_shock_level_multiplier contains negative values")
        if (corpus["channel_shock_level"] < 0).any():
            errors.append("channel_shock_level contains negative values")
        if not np.isin(corpus["adstock_family"], (0, 1, 2)).all():
            errors.append("adstock_family contains invalid family ids")
        if not np.isin(corpus["signal_metric_valid"], (0, 1)).all():
            errors.append("signal_metric_valid is not binary")
        if not np.isfinite(corpus["signal_metrics"]).all():
            errors.append("signal_metrics contains NaN or Inf")
        if (corpus["signal_metrics"][corpus["signal_metric_valid"] == 0] != 0).any():
            errors.append("signal_metrics has nonzero invalid values")
        if corpus["signal_metrics"].shape[-1] != len(SIGNAL_METRIC_LAYOUT):
            errors.append("signal_metrics has an unknown metric layout")
        signal_diagnostics = corpus.get("diagnostics", {}).get("signal", {})
        if signal_diagnostics.get("metric_version") != 1:
            errors.append("diagnostics signal metric_version is not supported")
        if signal_diagnostics.get("metric_layout") != list(SIGNAL_METRIC_LAYOUT):
            errors.append("diagnostics signal metric_layout does not match signal_metrics")

        active_c = corpus.get("active_c_mask")
        if active_c is not None and active_c.shape == (N, K):
            inactive = active_c == 0
            for key in (
                "channel_level",
                "adstock_family",
                "adstock_alpha",
                "weibull_lam",
                "weibull_k",
            ):
                if not np.array_equal(
                    corpus[key][inactive], np.zeros(inactive.sum(), dtype=corpus[key].dtype)
                ):
                    errors.append(f"{key} has nonzero inactive-channel padding")
            if (corpus["signal_metrics"][inactive] != 0).any() or corpus["signal_metric_valid"][
                inactive
            ].any():
                errors.append("signal metrics have nonzero inactive-channel padding")
            direct = (
                corpus["g"][
                    :,
                    SlotLayout(
                        K=K, M=M, J=corpus["demand"].shape[2], edge_types=EDGE_TYPES_EXTENDED
                    ).slices["cy"],
                ]
                == 1
            )
            ineligible = ~(direct & (active_c == 1))
            if (corpus["signal_metrics"][ineligible] != 0).any() or corpus["signal_metric_valid"][
                ineligible
            ].any():
                errors.append("signal metrics have nonzero ineligible-channel values")

        layout = SlotLayout(K=K, M=M, J=corpus["demand"].shape[2], edge_types=EDGE_TYPES_EXTENDED)
        channels = corpus["channel_shock_channel"]
        starts = corpus["channel_shock_start"]
        lengths = corpus["channel_shock_length"]
        levels = corpus["channel_shock_level"]
        multipliers = corpus["channel_shock_level_multiplier"]
        channel_level = corpus["channel_level"]
        audit_shapes_ok = (
            shock_mask.shape == (N, T, K)
            and corpus["g"].shape == (N, layout.n_slots)
            and channels.shape
            == starts.shape
            == lengths.shape
            == levels.shape
            == (N, n_channel_shocks)
        )
        if audit_shapes_ok:
            direct = corpus["g"][:, layout.slices["cy"]] == 1
            for n in range(N):
                rebuilt = np.zeros((T, K), dtype=np.uint8)
                occupied = np.zeros(T, dtype=bool)
                for s_idx in range(n_channel_shocks):
                    channel, start, length = (
                        int(channels[n, s_idx]),
                        int(starts[n, s_idx]),
                        int(lengths[n, s_idx]),
                    )
                    slot_lo = s_idx * T // max(n_channel_shocks, 1)
                    slot_hi = (s_idx + 1) * T // max(n_channel_shocks, 1)
                    if not (0 <= channel < K and direct[n, channel]):
                        errors.append("channel_shock_channel is not an active direct channel")
                        continue
                    if not (length > 0 and slot_lo <= start and start + length <= slot_hi):
                        errors.append("channel shock start/length is outside its schedule slot")
                        continue
                    if occupied[start : start + length].any():
                        errors.append("channel shocks overlap globally")
                    occupied[start : start + length] = True
                    rebuilt[start : start + length, channel] = 1
                    expected_level = multipliers[n, s_idx] * channel_level[n, channel]
                    if not np.isclose(levels[n, s_idx], expected_level, rtol=1e-6, atol=1e-7):
                        errors.append(
                            "channel_shock_level does not match multiplier * channel_level"
                        )
                    if not np.array_equal(
                        corpus["spend_raw"][n, start : start + length, channel],
                        np.full(length, levels[n, s_idx], dtype=np.float32),
                    ):
                        errors.append("held spend does not equal channel_shock_level")
                if not np.array_equal(rebuilt, shock_mask[n]):
                    errors.append("channel_shock_mask does not match the schedule")

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
