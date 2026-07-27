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
from .signal_diagnostics import (
    SIGNAL_METRIC_LAYOUT,
    SIGNAL_METRIC_VERSION,
    dense_signal_metrics,
    summarize_signal_metrics,
)
from .slots import EDGE_TYPES_EXTENDED, PRIOR_COND_LAYOUT, PRIOR_COND_QUANTITIES, SlotLayout

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

        cfg = self._make_config(seed=seed)

        # sample_prior_predictive truncates before deriving retained-corpus
        # diagnostics and signal labels. Do not independently slice a finalized
        # corpus here, or those task-level summaries would become stale.
        corpus = sample_prior_predictive(cfg, n=n_tasks)

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
        """Generate data in batches of at least two worlds.

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

        Batches are at least two worlds because every corpus carries its own
        cell-level train/validation split.

        Returns
        -------
        list of dict
            List of corpus dictionaries, one per batch.
        """
        if n_tasks <= 0:
            raise ValueError(f"n_tasks must be positive, got {n_tasks}")
        if n_tasks == 1:
            raise ValueError("n=1 cannot form a cell-level train/validation split; n must be >= 2")
        if batch_size <= 0:
            raise ValueError(f"batch_size must be positive, got {batch_size}")
        if batch_size == 1:
            raise ValueError(
                "batch_size must be at least 2 for a cell-level train/validation split"
            )

        n_full_batches, remainder = divmod(n_tasks, batch_size)
        batch_sizes = [batch_size] * n_full_batches
        if remainder == 1 and batch_sizes:
            batch_sizes[-1] += 1
        elif remainder:
            batch_sizes.append(remainder)

        batches = []
        for i, batch_tasks in enumerate(batch_sizes):
            corpus = self.generate(
                n_tasks=batch_tasks,
                seed=seed + i,
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
        seed: int | None = None,
    ) -> SCMPrior:
        """Create a SCMPrior with overrides."""
        cfg = self.config
        overrides = {}

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
            "active_m_mask",
            "active_j_mask",
            "K_active",
            "M_active",
            "J_active",
            "confounding_strength",
            "indirect_effects",
            "channel_active",
            "control_contribution",
            "confounder_contribution",
            "baseline_intrinsic",
            "indirect_effects_by_source",
            "channel_shock_mask",
            "channel_shock_channel",
            "channel_shock_start",
            "channel_shock_length",
            "channel_shock_level_multiplier",
            "channel_shock_level",
            "channel_level",
            "saturation_scale",
            "adstock_family",
            "adstock_alpha",
            "weibull_lam",
            "weibull_k",
        ]
        for key in required_keys:
            if key not in corpus:
                errors.append(f"Missing required key: {key}")

        signal_label_keys = ("signal_metrics", "signal_metric_valid")
        if any(key in corpus for key in signal_label_keys):
            errors.append("signal labels must live under the identifiability metadata block")
        identifiability = corpus.get("identifiability")
        if "identifiability" in corpus and not isinstance(identifiability, dict):
            errors.append("identifiability must be a mapping")
        has_signal_labels = isinstance(identifiability, dict) and all(
            key in identifiability for key in signal_label_keys
        )
        if isinstance(identifiability, dict) and not has_signal_labels:
            errors.append(
                "identifiability signal_metrics and signal_metric_valid must be present together"
            )

        if errors:
            return errors

        for key, value in corpus.items():
            if key in {"diagnostics", "identifiability"}:
                continue
            if not isinstance(value, np.ndarray):
                errors.append(f"{key} must be an ndarray")
        if errors:
            return errors

        core_ndims = {
            "spend_raw": 3,
            "controls": 3,
            "demand": 3,
            "g": 2,
            "channel_shock_channel": 2,
        }
        for key, ndim in core_ndims.items():
            value = corpus[key]
            if not isinstance(value, np.ndarray) or value.ndim != ndim:
                actual = getattr(value, "ndim", type(value).__name__)
                errors.append(f"{key} must have ndim={ndim}, got {actual}")
        if errors:
            return errors

        # Get dimensions
        N = corpus["spend_raw"].shape[0]
        T = corpus["spend_raw"].shape[1]
        K = corpus["spend_raw"].shape[2]
        M = corpus["controls"].shape[2]
        J = corpus["demand"].shape[2]
        layout = SlotLayout(K=K, M=M, J=J, edge_types=EDGE_TYPES_EXTENDED)
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
            "g": (N, layout.n_slots),
            "contributions_raw": (N, T, K),
            "baseline_raw": (N, T),
            "demand": (N, T, J),
            "spend_means": (N, K),
            "sales_scale": (N,),
            "is_val": (N,),
            "cell_id": (N,),
            "active_c_mask": (N, K),
            "active_m_mask": (N, M),
            "active_j_mask": (N, J),
            "K_active": (N,),
            "M_active": (N,),
            "J_active": (N,),
            "confounding_strength": (N,),
            "indirect_effects": (N, T),
            "channel_active": (N, K),
            "control_contribution": (N, T, M),
            "confounder_contribution": (N, T, J),
            "baseline_intrinsic": (N, T),
            "indirect_effects_by_source": (N, T, 3),
            "channel_shock_mask": (N, T, K),
            "channel_shock_channel": (N, n_channel_shocks),
            "channel_shock_start": (N, n_channel_shocks),
            "channel_shock_length": (N, n_channel_shocks),
            "channel_shock_level_multiplier": (N, n_channel_shocks),
            "channel_shock_level": (N, n_channel_shocks),
            "channel_level": (N, K),
            "saturation_scale": (N, K),
            "adstock_family": (N, K),
            "adstock_alpha": (N, K),
            "weibull_lam": (N, K),
            "weibull_k": (N, K),
        }
        if "prior_cond" in corpus:
            expected_shapes["prior_cond"] = (N, len(PRIOR_COND_LAYOUT))
            if not isinstance(corpus["prior_cond"], np.ndarray):
                errors.append("prior_cond must be an ndarray")

        if errors:
            return errors

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
        if errors:
            return errors

        if has_signal_labels:
            expected_label_shape = (N, K, len(SIGNAL_METRIC_LAYOUT))
            for key in signal_label_keys:
                value = identifiability[key]
                if not isinstance(value, np.ndarray) or value.shape != expected_label_shape:
                    errors.append(
                        f"Shape mismatch for identifiability {key}: "
                        f"expected {expected_label_shape}, got {getattr(value, 'shape', None)}"
                    )
            if errors:
                return errors

        expected_dtypes = {
            "spend_raw": np.float32,
            "spend_norm": np.float32,
            "spend_share": np.float32,
            "controls": np.float32,
            "sales_raw": np.float32,
            "sales_norm": np.float32,
            "support_mask": np.uint8,
            "is_future": np.uint8,
            "g": np.uint8,
            "contributions_raw": np.float32,
            "baseline_raw": np.float32,
            "demand": np.float32,
            "spend_means": np.float32,
            "sales_scale": np.float32,
            "is_val": np.uint8,
            "cell_id": np.int32,
            "channel_shock_mask": np.uint8,
            "active_c_mask": np.uint8,
            "active_m_mask": np.uint8,
            "active_j_mask": np.uint8,
            "K_active": np.int32,
            "M_active": np.int32,
            "J_active": np.int32,
            "indirect_effects": np.float32,
            "channel_active": np.uint8,
            "control_contribution": np.float32,
            "confounder_contribution": np.float32,
            "baseline_intrinsic": np.float32,
            "indirect_effects_by_source": np.float32,
            "channel_shock_channel": np.int32,
            "channel_shock_start": np.int32,
            "channel_shock_length": np.int32,
            "channel_shock_level_multiplier": np.float32,
            "channel_shock_level": np.float32,
            "channel_level": np.float32,
            "confounding_strength": np.float32,
            "saturation_scale": np.float32,
            "adstock_family": np.uint8,
            "adstock_alpha": np.float32,
            "weibull_lam": np.float32,
            "weibull_k": np.float32,
        }
        if "prior_cond" in corpus:
            expected_dtypes["prior_cond"] = np.float32
        for key, dtype in expected_dtypes.items():
            if key in corpus and corpus[key].dtype != dtype:
                errors.append(f"{key} has dtype {corpus[key].dtype}, expected {np.dtype(dtype)}")
        if has_signal_labels:
            for key, dtype in (
                ("signal_metrics", np.float32),
                ("signal_metric_valid", np.uint8),
            ):
                if identifiability[key].dtype != dtype:
                    errors.append(
                        f"identifiability {key} has dtype {identifiability[key].dtype}, "
                        f"expected {np.dtype(dtype)}"
                    )
        if errors:
            return errors

        # Every serialized numeric array must be finite, including normalized
        # model inputs rather than only their raw source arrays.
        for key, value in corpus.items():
            if not isinstance(value, np.ndarray):
                continue
            if value.dtype.hasobject or not (
                np.issubdtype(value.dtype, np.integer)
                or np.issubdtype(value.dtype, np.floating)
                or np.issubdtype(value.dtype, np.bool_)
            ):
                errors.append(f"{key} has unsupported dtype {value.dtype}")
            elif not np.isfinite(value).all():
                errors.append(f"{key} contains NaN or Inf")
        if has_signal_labels:
            for key in signal_label_keys:
                if not np.isfinite(identifiability[key]).all():
                    errors.append(f"{key} contains NaN or Inf")

        positive_sales_scale = (corpus["sales_scale"] > 0.0).all()
        if not positive_sales_scale:
            errors.append("sales_scale must be positive")
        expected_spend_means = corpus["spend_raw"].astype(np.float64).mean(axis=1)
        if not np.allclose(corpus["spend_means"], expected_spend_means, rtol=1e-6, atol=1e-7):
            errors.append("spend_means != mean(spend_raw, axis=1)")
        spend_raw = corpus["spend_raw"].astype(np.float64)
        spend_means = corpus["spend_means"].astype(np.float64)
        expected_spend_norm = np.divide(
            spend_raw,
            spend_means[:, None, :],
            out=np.zeros_like(spend_raw),
            where=spend_means[:, None, :] != 0.0,
        )
        if not np.allclose(corpus["spend_norm"], expected_spend_norm, rtol=1e-5, atol=1e-7):
            errors.append("spend_norm does not match spend_raw / spend_means")
        active_spend_sum = (spend_raw * corpus["active_c_mask"].astype(np.float64)[:, None, :]).sum(
            axis=-1, keepdims=True
        )
        expected_spend_share = (
            np.divide(
                spend_raw,
                active_spend_sum,
                out=np.zeros_like(spend_raw),
                where=active_spend_sum != 0.0,
            )
            * corpus["active_c_mask"].astype(np.float64)[:, None, :]
        )
        if not np.allclose(corpus["spend_share"], expected_spend_share, rtol=1e-5, atol=1e-7):
            errors.append("spend_share does not match active-channel spend shares")

        if positive_sales_scale:
            expected_norm = (
                corpus["sales_raw"].astype(np.float64)
                / corpus["sales_scale"].astype(np.float64)[:, None]
            )
            if not np.allclose(corpus["sales_norm"], expected_norm, rtol=1e-5):
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

        short_n_query = None
        diagnostics = corpus.get("diagnostics")
        if isinstance(diagnostics, dict) and isinstance(diagnostics.get("signal"), dict):
            candidate = diagnostics.get("short_horizon_n_query")
            if (
                isinstance(candidate, (int, np.integer))
                and not isinstance(candidate, (bool, np.bool_))
                and 0 < candidate < T
            ):
                short_n_query = int(candidate)
            else:
                errors.append("diagnostics short_horizon_n_query must be an integer in (0, T)")
        if short_n_query is not None:
            expected_support_count = np.where(
                corpus["is_future"] == 1,
                T // 2,
                T - short_n_query,
            )
            expected_support = (np.arange(T)[None, :] < expected_support_count[:, None]).astype(
                np.uint8
            )
            if not np.array_equal(corpus["support_mask"], expected_support):
                errors.append("support_mask does not match the recorded temporal split")

            sales = corpus["sales_raw"].astype(np.float64)
            expected_sales_scale = np.asarray(
                [sales[i, expected_support[i] == 1].std() for i in range(N)],
                dtype=np.float64,
            )
            bad_scale = ~(np.isfinite(expected_sales_scale) & (expected_sales_scale > 0.0))
            if bad_scale.any():
                full_std = sales[bad_scale].std(axis=1)
                expected_sales_scale[bad_scale] = np.where(
                    np.isfinite(full_std) & (full_std > 0.0), full_std, 1.0
                )
            if not np.allclose(corpus["sales_scale"], expected_sales_scale, rtol=1e-5, atol=1e-7):
                errors.append("sales_scale does not match supported sales observations")

        # Check is_val is binary and contains both sides of the corpus split.
        if "is_val" in corpus:
            is_val = corpus["is_val"]
            if not np.all(
                (np.abs(is_val.astype(float)) < 1e-9) | (np.abs(is_val.astype(float) - 1) < 1e-9)
            ):
                errors.append("is_val is not binary")
            elif not 0 < is_val.sum() < N:
                errors.append("is_val must contain at least one training and one validation world")

        # Check g is binary
        if "g" in corpus:
            g = corpus["g"]
            if not np.all((np.abs(g.astype(float)) < 1e-9) | (np.abs(g.astype(float) - 1) < 1e-9)):
                errors.append("g is not binary")
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
        if not (
            (corpus["confounding_strength"] >= 0.0) & (corpus["confounding_strength"] <= 0.95)
        ).all():
            errors.append("confounding_strength must be in [0, 0.95]")
        if not np.isin(corpus["adstock_family"], (0, 1, 2)).all():
            errors.append("adstock_family contains invalid family ids")
        if has_signal_labels:
            signal_metrics = identifiability["signal_metrics"]
            signal_metric_valid = identifiability["signal_metric_valid"]
            if not np.isin(signal_metric_valid, (0, 1)).all():
                errors.append("signal_metric_valid is not binary")
            if not np.isfinite(signal_metrics).all():
                errors.append("signal_metrics contains NaN or Inf")
            if (signal_metrics[signal_metric_valid == 0] != 0).any():
                errors.append("signal_metrics has nonzero invalid values")
            metric_index = {name: i for i, name in enumerate(SIGNAL_METRIC_LAYOUT)}
            for name, low, high in (
                ("spearman", 0.0, 1.0),
                ("contrib_r2_explained_by_rest", 0.0, 1.0),
                ("contrib_corr_baseline", -1.0, 1.0),
            ):
                index = metric_index[name]
                values = signal_metrics[..., index][signal_metric_valid[..., index].astype(bool)]
                if ((values < low - 1e-6) | (values > high + 1e-6)).any():
                    errors.append(f"signal metric {name} is outside [{low}, {high}]")
        if not isinstance(diagnostics, dict):
            errors.append("diagnostics must be a mapping")
            return errors
        signal_diagnostics = diagnostics.get("signal")
        if not isinstance(signal_diagnostics, dict):
            errors.append("diagnostics signal must be a mapping")
            return errors
        try:
            normalized_edge_types = list(diagnostics.get("edge_types"))
        except TypeError:
            normalized_edge_types = None
        if normalized_edge_types != list(EDGE_TYPES_EXTENDED):
            errors.append("diagnostics edge_types does not match the canonical layout")

        metric_version = signal_diagnostics.get("metric_version")

        def _is_integer(value):
            return isinstance(value, (int, np.integer)) and not isinstance(value, (bool, np.bool_))

        def _diagnostic_equal(actual, expected) -> bool:
            if isinstance(actual, np.ndarray):
                actual = actual.item() if actual.ndim == 0 else actual.tolist()
            elif isinstance(actual, np.generic):
                actual = actual.item()
            if isinstance(expected, np.ndarray):
                expected = expected.item() if expected.ndim == 0 else expected.tolist()
            elif isinstance(expected, np.generic):
                expected = expected.item()
            if isinstance(actual, dict) and isinstance(expected, dict):
                return actual.keys() == expected.keys() and all(
                    _diagnostic_equal(actual[key], expected[key]) for key in actual
                )
            if isinstance(actual, (list, tuple)) and isinstance(expected, (list, tuple)):
                return len(actual) == len(expected) and all(
                    _diagnostic_equal(left, right) for left, right in zip(actual, expected)
                )
            if isinstance(actual, (dict, list, tuple)) or isinstance(expected, (dict, list, tuple)):
                return False
            try:
                result = actual == expected
            except (TypeError, ValueError):
                return False
            return isinstance(result, (bool, np.bool_)) and bool(result)

        version_supported = (
            _is_integer(metric_version) and int(metric_version) == SIGNAL_METRIC_VERSION
        )
        if not version_supported:
            errors.append("diagnostics signal metric_version is not supported")
        metric_layout = signal_diagnostics.get("metric_layout")
        try:
            normalized_layout = list(metric_layout)
        except TypeError:
            normalized_layout = None
        if normalized_layout != list(SIGNAL_METRIC_LAYOUT):
            errors.append("diagnostics signal metric_layout does not match signal_metrics")
        if version_supported:
            if not _is_integer(signal_diagnostics.get("l_max")) or signal_diagnostics["l_max"] < 1:
                errors.append("diagnostics signal l_max must be a positive integer")
            if (
                not _is_integer(signal_diagnostics.get("adstock_burn_in"))
                or signal_diagnostics["adstock_burn_in"] < 0
            ):
                errors.append("diagnostics signal adstock_burn_in must be a nonnegative integer")
            response_warmup_weeks = signal_diagnostics.get("response_warmup_weeks")
            if not _is_integer(response_warmup_weeks) or not 0 <= response_warmup_weeks < T:
                errors.append(
                    "diagnostics signal response_warmup_weeks must be a nonnegative integer below T"
                )
            frac_zero_contemporaneous_weight = signal_diagnostics.get(
                "frac_zero_contemporaneous_weight"
            )
            if frac_zero_contemporaneous_weight is not None and (
                not isinstance(frac_zero_contemporaneous_weight, (float, np.floating))
                or not np.isfinite(frac_zero_contemporaneous_weight)
                or not 0.0 <= frac_zero_contemporaneous_weight <= 1.0
            ):
                errors.append(
                    "diagnostics signal frac_zero_contemporaneous_weight must be None or a float "
                    "in [0, 1]"
                )
            if (
                signal_diagnostics.get("adstock_kernel_semantics")
                != "normalized-causal-minmax-weibull-density"
                or signal_diagnostics.get("adstock_kernel_version") != 3
            ):
                errors.append("diagnostics signal adstock kernel semantics are not supported")

        cell_ids = np.unique(corpus["cell_id"])
        if not _is_integer(diagnostics.get("n_tasks")) or diagnostics["n_tasks"] != N:
            errors.append("diagnostics n_tasks does not match the corpus")
        if not _is_integer(diagnostics.get("n_cells")) or diagnostics["n_cells"] != len(cell_ids):
            errors.append("diagnostics n_cells does not match cell_id")
        if (corpus["cell_id"] < 0).any() or not np.array_equal(
            cell_ids, np.arange(len(cell_ids), dtype=cell_ids.dtype)
        ):
            errors.append("cell_id must contain contiguous nonnegative ids")
        cell_level_keys = (
            "g",
            "active_c_mask",
            "active_m_mask",
            "active_j_mask",
            "K_active",
            "M_active",
            "J_active",
            "channel_active",
            "is_val",
        )
        for cell in cell_ids:
            in_cell = corpus["cell_id"] == cell
            for key in cell_level_keys:
                rows = corpus[key][in_cell]
                if len(rows) > 1 and not np.equal(rows, rows[0]).all():
                    errors.append(f"{key} differs within a cell")

        has_prior_cond = "prior_cond" in corpus
        has_prior_cond_diagnostics = "prior_cond" in diagnostics
        if has_prior_cond != has_prior_cond_diagnostics:
            errors.append("prior_cond and diagnostics prior_cond must be present together")
        elif has_prior_cond:
            prior_diagnostics = diagnostics["prior_cond"]
            if not isinstance(prior_diagnostics, dict):
                errors.append("diagnostics prior_cond must be a mapping")
            else:
                prior_layout = prior_diagnostics.get("layout")
                try:
                    normalized_prior_layout = list(prior_layout)
                except TypeError:
                    normalized_prior_layout = None
                if normalized_prior_layout != list(PRIOR_COND_LAYOUT):
                    errors.append("diagnostics prior_cond layout does not match prior_cond")
                supports = prior_diagnostics.get("supports")
                width_ranges = prior_diagnostics.get("width_ranges")
                if not isinstance(supports, dict) or not isinstance(width_ranges, dict):
                    errors.append(
                        "diagnostics prior_cond supports and width_ranges must be mappings"
                    )
                elif set(supports) != set(PRIOR_COND_QUANTITIES) or set(width_ranges) != set(
                    PRIOR_COND_QUANTITIES
                ):
                    errors.append("diagnostics prior_cond quantities do not match the layout")
                else:
                    prior_cond = corpus["prior_cond"].astype(np.float64)
                    for q_idx, quantity in enumerate(PRIOR_COND_QUANTITIES):
                        try:
                            support = np.asarray(supports[quantity], dtype=np.float64)
                            width_range = np.asarray(width_ranges[quantity], dtype=np.float64)
                        except (TypeError, ValueError):
                            errors.append(f"diagnostics prior_cond {quantity} bounds are invalid")
                            continue
                        if (
                            support.shape != (2,)
                            or width_range.shape != (2,)
                            or not np.isfinite(support).all()
                            or not np.isfinite(width_range).all()
                            or support[0] >= support[1]
                            or not 0.0 < width_range[0] <= width_range[1] <= support[1] - support[0]
                        ):
                            errors.append(f"diagnostics prior_cond {quantity} bounds are invalid")
                            continue
                        low = prior_cond[:, 2 * q_idx]
                        width = prior_cond[:, 2 * q_idx + 1]
                        tolerance = (
                            8
                            * np.finfo(np.float32).eps
                            * max(
                                float(np.abs(support).max()), float(np.abs(width_range).max()), 1.0
                            )
                        )
                        if (
                            (width < width_range[0] - tolerance).any()
                            or (width > width_range[1] + tolerance).any()
                            or (low < support[0] - tolerance).any()
                            or (low + width > support[1] + tolerance).any()
                        ):
                            errors.append(
                                f"prior_cond {quantity} intervals are outside diagnostics bounds"
                            )
                    for cell in np.unique(corpus["cell_id"]):
                        rows = corpus["prior_cond"][corpus["cell_id"] == cell]
                        if len(rows) > 1 and not np.equal(rows, rows[0]).all():
                            errors.append("prior_cond rows differ within a cell")
                            break

        active_c = corpus["active_c_mask"]
        active_m = corpus["active_m_mask"]
        active_j = corpus["active_j_mask"]
        for key, mask in (
            ("active_c_mask", active_c),
            ("active_m_mask", active_m),
            ("active_j_mask", active_j),
            ("channel_active", corpus["channel_active"]),
        ):
            if not np.isin(mask, (0, 1)).all():
                errors.append(f"{key} is not binary")
        for key, count, width, mask in (
            ("K_active", corpus["K_active"], K, active_c),
            ("M_active", corpus["M_active"], M, active_m),
            ("J_active", corpus["J_active"], J, active_j),
        ):
            if ((count < 1) | (count > width)).any() or not np.array_equal(
                mask, (np.arange(width)[None, :] < count[:, None]).astype(np.uint8)
            ):
                errors.append(f"{key} does not match its active prefix mask")

        inactive_c, inactive_m, inactive_j = active_c == 0, active_m == 0, active_j == 0

        def _padded_nonzero(array: np.ndarray, inactive: np.ndarray) -> bool:
            expanded = np.broadcast_to(inactive[:, None, :], array.shape)
            return bool((array[expanded] != 0).any())

        for key in ("spend_raw", "spend_norm", "spend_share", "contributions_raw"):
            if _padded_nonzero(corpus[key], inactive_c):
                errors.append(f"{key} has nonzero inactive-channel padding")
        if _padded_nonzero(corpus["channel_shock_mask"], inactive_c):
            errors.append("channel_shock_mask has nonzero inactive-channel padding")
        for key in (
            "spend_means",
            "channel_active",
            "channel_level",
            "saturation_scale",
            "adstock_family",
            "adstock_alpha",
            "weibull_lam",
            "weibull_k",
        ):
            if (corpus[key][inactive_c] != 0).any():
                errors.append(f"{key} has nonzero inactive-channel padding")
        for key in ("controls", "control_contribution"):
            if _padded_nonzero(corpus[key], inactive_m):
                errors.append(f"{key} has nonzero inactive-control padding")
        for key in ("demand", "confounder_contribution"):
            if _padded_nonzero(corpus[key], inactive_j):
                errors.append(f"{key} has nonzero inactive-demand padding")
        if (corpus["saturation_scale"][active_c == 1] <= 0.0).any():
            errors.append("saturation_scale must be positive for active channels")

        graph = layout.unpack(corpus["g"])
        graph_masks = {
            "cy": active_c.astype(bool),
            "dc": active_j.astype(bool)[:, :, None] & active_c.astype(bool)[:, None, :],
            "dz": active_j.astype(bool)[:, :, None] & active_m.astype(bool)[:, None, :],
            "db": active_j.astype(bool),
            "zb": active_m.astype(bool),
            "zc": active_m.astype(bool)[:, :, None] & active_c.astype(bool)[:, None, :],
            "cc": active_c.astype(bool)[:, :, None] & active_c.astype(bool)[:, None, :],
            "zz": active_m.astype(bool)[:, :, None] & active_m.astype(bool)[:, None, :],
        }
        for edge_type, edge_mask in graph_masks.items():
            if (graph[edge_type][~edge_mask] != 0).any():
                errors.append(f"g_{edge_type} has edges incident to inactive nodes")
        for edge_type in ("cc", "zz"):
            if np.tril(graph[edge_type], k=-1).any():
                errors.append(f"g_{edge_type} must be strictly upper triangular")
        direct = graph["cy"] == 1
        expected_channel_active = (
            (direct | (graph["cc"].sum(axis=2) > 0)) & active_c.astype(bool)
        ).astype(np.uint8)
        if not np.array_equal(corpus["channel_active"], expected_channel_active):
            errors.append("channel_active does not match graph reachability rule")
        if _padded_nonzero(corpus["contributions_raw"], ~direct):
            errors.append("contributions_raw is nonzero for channels without C->Y edges")
        if _padded_nonzero(corpus["control_contribution"], graph["zb"] != 1):
            errors.append("control_contribution is nonzero without a Z->B edge")
        if _padded_nonzero(corpus["confounder_contribution"], graph["db"] != 1):
            errors.append("confounder_contribution is nonzero without a D->B edge")
        for source_index, edge_type in enumerate(("cc", "zc", "dc")):
            source_present = graph[edge_type].reshape(N, -1).any(axis=1)
            if (corpus["indirect_effects_by_source"][~source_present, :, source_index] != 0).any():
                errors.append(
                    f"indirect_effects_by_source {edge_type} column is nonzero without an edge"
                )

        if has_signal_labels and (
            (signal_metrics[inactive_c] != 0).any() or signal_metric_valid[inactive_c].any()
        ):
            errors.append("signal metrics have nonzero inactive-channel padding")
        ineligible = ~(direct & (active_c == 1))
        if has_signal_labels and (
            (signal_metrics[ineligible] != 0).any() or signal_metric_valid[ineligible].any()
        ):
            errors.append("signal metrics have nonzero ineligible-channel values")

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

        sales = corpus["sales_raw"].astype(np.float64)
        baseline = corpus["baseline_raw"].astype(np.float64)
        contributions = corpus["contributions_raw"].astype(np.float64)
        indirect = corpus["indirect_effects"].astype(np.float64)
        indirect_by_source = corpus["indirect_effects_by_source"].astype(np.float64)
        intrinsic = corpus["baseline_intrinsic"].astype(np.float64)
        control_contribution = corpus["control_contribution"].astype(np.float64)
        confounder_contribution = corpus["confounder_contribution"].astype(np.float64)
        tolerance_factor = 32 * np.finfo(corpus["sales_raw"].dtype).eps
        decomposition_errors = {
            "additive decomposition": (
                np.abs(baseline + contributions.sum(axis=2) + indirect - sales),
                sales,
            ),
            "telescoping decomposition": (
                np.abs(indirect_by_source.sum(axis=2) - indirect),
                indirect,
            ),
            "baseline decomposition": (
                np.abs(
                    intrinsic
                    + confounder_contribution.sum(axis=2)
                    + control_contribution.sum(axis=2)
                    - baseline
                ),
                baseline,
            ),
            "full decomposition": (
                np.abs(
                    intrinsic
                    + confounder_contribution.sum(axis=2)
                    + control_contribution.sum(axis=2)
                    + contributions.sum(axis=2)
                    + indirect_by_source.sum(axis=2)
                    - sales
                ),
                sales,
            ),
        }
        for name, (residual, reference) in decomposition_errors.items():
            tolerance = tolerance_factor * np.maximum(np.abs(reference), 1.0)
            if (residual > tolerance).any():
                errors.append(f"{name} exceeds float32 storage tolerance")

        if not errors and version_supported:
            eligible = direct & (corpus["active_c_mask"] == 1)
            expected_metrics, expected_valid = dense_signal_metrics(
                corpus["spend_raw"],
                corpus["contributions_raw"],
                corpus["sales_raw"],
                corpus["baseline_raw"],
                eligible,
                sales_scale=corpus["sales_scale"],
                adstock_family=corpus["adstock_family"],
                adstock_alpha=corpus["adstock_alpha"],
                weibull_lam=corpus["weibull_lam"],
                weibull_k=corpus["weibull_k"],
                l_max=int(signal_diagnostics["l_max"]),
                adstock_burn_in=int(signal_diagnostics["adstock_burn_in"]),
            )
            if has_signal_labels:
                if not np.array_equal(signal_metrics, expected_metrics):
                    errors.append("identifiability signal_metrics do not match recomputation")
                if not np.array_equal(signal_metric_valid, expected_valid):
                    errors.append(
                        "identifiability signal_metric_valid does not match recomputation"
                    )
            expected_signal = summarize_signal_metrics(
                expected_metrics,
                expected_valid,
                corpus["sales_raw"],
                eligible,
                sales_scale=corpus["sales_scale"],
                l_max=int(signal_diagnostics["l_max"]),
                adstock_burn_in=int(signal_diagnostics["adstock_burn_in"]),
                adstock_family=corpus["adstock_family"],
                adstock_alpha=corpus["adstock_alpha"],
                weibull_lam=corpus["weibull_lam"],
                weibull_k=corpus["weibull_k"],
            )
            expected_signal.update(
                {
                    "metric_version": SIGNAL_METRIC_VERSION,
                    "metric_layout": list(SIGNAL_METRIC_LAYOUT),
                    "l_max": int(signal_diagnostics["l_max"]),
                    "adstock_burn_in": int(signal_diagnostics["adstock_burn_in"]),
                    "adstock_kernel_semantics": "normalized-causal-minmax-weibull-density",
                    "adstock_kernel_version": 3,
                }
            )
            actual_signal = dict(signal_diagnostics)
            actual_signal["metric_layout"] = normalized_layout
            if not _diagnostic_equal(actual_signal, expected_signal):
                errors.append("diagnostics signal summary does not match recomputation")

        # This corpus-level invariant enforces beta_additive_range >= 0 downstream.
        if (corpus["contributions_raw"] < 0).any():
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
    if path.suffix != ".npz":
        raise ValueError(f"corpus path must use the .npz suffix, got {path}")
    if any(key.startswith("identifiability__") for key in corpus):
        raise ValueError("top-level corpus keys may not use the reserved identifiability__ prefix")
    identifiability = corpus.get("identifiability")
    if isinstance(identifiability, dict) and not identifiability:
        raise ValueError("identifiability metadata must be omitted rather than empty")

    def _is_real_numeric(array: np.ndarray) -> bool:
        return bool(
            np.issubdtype(array.dtype, np.integer)
            or np.issubdtype(array.dtype, np.floating)
            or np.issubdtype(array.dtype, np.bool_)
        )

    # Convert diagnostics dict to JSON string if present
    save_dict = {}
    for k, v in corpus.items():
        if k == "diagnostics":
            if not isinstance(v, dict):
                raise TypeError("diagnostics must be a mapping")
            import json

            def _json_default(value):
                if isinstance(value, np.generic):
                    return value.item()
                if isinstance(value, np.ndarray):
                    if value.dtype.hasobject:
                        raise ValueError("diagnostics arrays may not have object dtype")
                    return value.tolist()
                raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")

            save_dict[k] = np.array(json.dumps(v, default=_json_default))
        elif k == "identifiability":
            if not isinstance(v, dict):
                raise TypeError("identifiability metadata must be a mapping")
            for label, value in v.items():
                if not isinstance(value, np.ndarray):
                    raise TypeError(f"identifiability.{label} must be an ndarray")
                if value.dtype.hasobject:
                    raise ValueError(f"identifiability.{label} may not have object dtype")
                if np.issubdtype(value.dtype, np.complexfloating):
                    raise ValueError(f"identifiability.{label} may not have complex dtype")
                if not _is_real_numeric(value):
                    raise ValueError(f"identifiability.{label} must have a real numeric dtype")
                save_dict[f"identifiability__{label}"] = value
        else:
            if not isinstance(v, np.ndarray):
                raise TypeError(f"{k} must be an ndarray")
            if v.dtype.hasobject:
                raise ValueError(f"{k} may not have object dtype")
            if np.issubdtype(v.dtype, np.complexfloating):
                raise ValueError(f"{k} may not have complex dtype")
            if not _is_real_numeric(v):
                raise ValueError(f"{k} must have a real numeric dtype")
            save_dict[k] = v

    path.parent.mkdir(parents=True, exist_ok=True)
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

    with np.load(path, allow_pickle=False) as data:
        corpus = {k: data[k] for k in data.files}

    identifiability = {
        key.removeprefix("identifiability__"): corpus.pop(key)
        for key in tuple(corpus)
        if key.startswith("identifiability__")
    }
    if identifiability:
        corpus["identifiability"] = identifiability

    # Parse diagnostics JSON string if present
    if "diagnostics" in corpus:
        import json

        diag_str = corpus["diagnostics"]
        if not isinstance(diag_str, np.ndarray) or diag_str.ndim != 0:
            raise ValueError(f"corpus {path} has a non-scalar diagnostics field")
        try:
            diagnostics = json.loads(str(diag_str.item()))
        except (json.JSONDecodeError, TypeError) as exc:
            raise ValueError(f"corpus {path} has invalid diagnostics JSON") from exc
        if not isinstance(diagnostics, dict):
            raise ValueError(f"corpus {path} diagnostics JSON must contain an object")
        corpus["diagnostics"] = diagnostics

    return corpus
