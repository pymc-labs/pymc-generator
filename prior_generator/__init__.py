"""prior-generator: synthetic MMM world simulation with exact causal decompositions.

Simulate thousands of marketing-mix-model "worlds" — additive structural
causal models over demand, controls, media channels, baseline and sales —
with exact interventional decomposition targets, human-readable dataset
bundles, and corpora in the format PFN training pipelines consume.

Public symbols are lazy-loaded so ``import prior_generator`` stays light
(no pytensor / scipy / pandas / matplotlib until actually used).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .data_generator import DataGenerator
    from .sampler import SCMPrior

__version__ = "0.0.1"

_LAZY_IMPORTS = {
    # config + generation
    "SCMPrior": (".sampler", "SCMPrior"),
    "sample_prior_predictive": (".sampler", "sample_prior_predictive"),
    "make_scm_prior": (".presets", "make_scm_prior"),
    # high-level facade + persistence (the PFN-consumable .npz format)
    "DataGenerator": (".data_generator", "DataGenerator"),
    "save_corpus": (".data_generator", "save_corpus"),
    "load_corpus": (".data_generator", "load_corpus"),
    # single-world API + human-readable bundles
    "sample_scm": (".worlds", "sample_scm"),
    "SCM": (".worlds", "SCM"),
    "describe_scm": (".describe", "describe_scm"),
    "write_scm_bundle": (".bundles", "write_scm_bundle"),
    "write_scenario_bundles": (".bundles", "write_scenario_bundles"),
    "SCENARIOS": (".scenarios", "SCENARIOS"),
}


def __getattr__(name: str):
    if name in _LAZY_IMPORTS:
        import importlib

        module_path, attr_name = _LAZY_IMPORTS[name]
        module = importlib.import_module(module_path, __name__)
        return getattr(module, attr_name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "SCENARIOS",
    "SCMPrior",
    "DataGenerator",
    "SCM",
    "__version__",
    "describe_scm",
    "sample_prior_predictive",
    "load_corpus",
    "make_scm_prior",
    "sample_scm",
    "save_corpus",
    "write_scenario_bundles",
    "write_scm_bundle",
]
