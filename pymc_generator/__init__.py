"""pymc-generator: structural causal model simulation with exact decompositions.

Generate thousands of "worlds" — additive structural causal models over latent
confounders, observed covariates, treatments, a baseline intercept and one
outcome — with exact interventional decomposition targets, human-readable
dataset bundles, and stacked padded tensor corpora. The default vocabulary is
marketing-mix-model-shaped (treatments, treatment, outcome); the machinery is not.

Public symbols are lazy-loaded so ``import pymc_generator`` stays light
(no pytensor / scipy / pandas / matplotlib until actually used).
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from ._version import __version__

if TYPE_CHECKING:
    from .bundles import write_scenario_bundles, write_scm_bundle
    from .data_generator import DataGenerator, load_corpus, save_corpus
    from .describe import describe_scm
    from .diagnostics import DataDiagnostics, data_diagnostics
    from .oracle_sampling import (
        OracleHealthCriteria,
        OracleSamplingConfig,
        OracleSamplingReceipt,
        OracleSamplingResult,
        sample_oracle,
    )
    from .outcomes import (
        OUTCOME_QUANTITIES,
        OutcomeDistributions,
        QuantityDistribution,
        outcome_distributions,
    )
    from .presets import make_scm_prior
    from .sampler import SCMPrior, sample_prior_predictive
    from .scenarios import SCENARIOS
    from .signal_diagnostics import SIGNAL_METRIC_LAYOUT, SIGNAL_METRIC_VERSION
    from .world_model import (
        build_oracle_model,
        build_world_model,
        draw_worlds,
        sample_prior_cond,
        sample_structure,
    )
    from .worlds import SCM, sample_scm


_LAZY_IMPORTS = {
    # config + generation
    "SCMPrior": (".sampler", "SCMPrior"),
    "sample_prior_predictive": (".sampler", "sample_prior_predictive"),
    "make_scm_prior": (".presets", "make_scm_prior"),
    # high-level facade + persistence (the stacked .npz corpus format)
    "DataGenerator": (".data_generator", "DataGenerator"),
    "save_corpus": (".data_generator", "save_corpus"),
    "load_corpus": (".data_generator", "load_corpus"),
    # single-world API + human-readable bundles
    "sample_scm": (".worlds", "sample_scm"),
    "SCM": (".worlds", "SCM"),
    # the world as a pm.Model: generative build, structure draw, seeded draws,
    # and the observed-data (NUTS oracle) variant
    "build_world_model": (".world_model", "build_world_model"),
    "build_oracle_model": (".world_model", "build_oracle_model"),
    "sample_structure": (".world_model", "sample_structure"),
    "sample_prior_cond": (".world_model", "sample_prior_cond"),
    "draw_worlds": (".world_model", "draw_worlds"),
    # outcome-space (quantity axis) distributions across worlds
    "outcome_distributions": (".outcomes", "outcome_distributions"),
    "OutcomeDistributions": (".outcomes", "OutcomeDistributions"),
    "QuantityDistribution": (".outcomes", "QuantityDistribution"),
    "OUTCOME_QUANTITIES": (".outcomes", "OUTCOME_QUANTITIES"),
    # post-hoc diagnostics over the generated series (shape, dependence,
    # collinearity, dynamics, contributions to outcome)
    "data_diagnostics": (".diagnostics", "data_diagnostics"),
    "DataDiagnostics": (".diagnostics", "DataDiagnostics"),
    "describe_scm": (".describe", "describe_scm"),
    "write_scm_bundle": (".bundles", "write_scm_bundle"),
    "write_scenario_bundles": (".bundles", "write_scenario_bundles"),
    "OracleSamplingConfig": (".oracle_sampling", "OracleSamplingConfig"),
    "OracleHealthCriteria": (".oracle_sampling", "OracleHealthCriteria"),
    "OracleSamplingReceipt": (".oracle_sampling", "OracleSamplingReceipt"),
    "OracleSamplingResult": (".oracle_sampling", "OracleSamplingResult"),
    "sample_oracle": (".oracle_sampling", "sample_oracle"),
    "SCENARIOS": (".scenarios", "SCENARIOS"),
    "SIGNAL_METRIC_VERSION": (".signal_diagnostics", "SIGNAL_METRIC_VERSION"),
    "SIGNAL_METRIC_LAYOUT": (".signal_diagnostics", "SIGNAL_METRIC_LAYOUT"),
}


def __getattr__(name: str) -> Any:
    if name in _LAZY_IMPORTS:
        import importlib

        module_path, attr_name = _LAZY_IMPORTS[name]
        module = importlib.import_module(module_path, __name__)
        return getattr(module, attr_name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "OracleHealthCriteria",
    "OracleSamplingConfig",
    "OracleSamplingReceipt",
    "OracleSamplingResult",
    "OUTCOME_QUANTITIES",
    "SCENARIOS",
    "SCMPrior",
    "DataDiagnostics",
    "DataGenerator",
    "SCM",
    "OutcomeDistributions",
    "QuantityDistribution",
    "SIGNAL_METRIC_LAYOUT",
    "SIGNAL_METRIC_VERSION",
    "__version__",
    "build_oracle_model",
    "build_world_model",
    "data_diagnostics",
    "describe_scm",
    "draw_worlds",
    "outcome_distributions",
    "sample_oracle",
    "sample_prior_predictive",
    "load_corpus",
    "make_scm_prior",
    "sample_prior_cond",
    "sample_scm",
    "sample_structure",
    "save_corpus",
    "write_scenario_bundles",
    "write_scm_bundle",
]
