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
    from .sampler import CorpusConfig

__version__ = "0.0.1"

_LAZY_IMPORTS = {
    # config + generation
    "CorpusConfig": (".sampler", "CorpusConfig"),
    "generate_corpus": (".sampler", "generate_corpus"),
    "make_world_config": (".presets", "make_world_config"),
    # high-level facade + persistence (the PFN-consumable .npz format)
    "DataGenerator": (".data_generator", "DataGenerator"),
    "save_corpus": (".data_generator", "save_corpus"),
    "load_corpus": (".data_generator", "load_corpus"),
    # single-world API + human-readable bundles
    "sample_world": (".worlds", "sample_world"),
    "World": (".worlds", "World"),
    "describe_world": (".describe", "describe_world"),
    "write_world_bundle": (".bundles", "write_world_bundle"),
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
    "CorpusConfig",
    "DataGenerator",
    "World",
    "__version__",
    "describe_world",
    "generate_corpus",
    "load_corpus",
    "make_world_config",
    "sample_world",
    "save_corpus",
    "write_world_bundle",
]
