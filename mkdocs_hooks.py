"""MkDocs build hook.

Adds the repo-root ``docs_gen`` directory to ``sys.path`` so executable Markdown
blocks can share its cached world sampler. Selects the headless Matplotlib
backend before any figure is drawn.

The path is (re)asserted in ``on_page_markdown`` — right before each page's
markdown is converted (which is when markdown-exec runs the code blocks) —
because other plugins (mkdocstrings/griffe) save and restore ``sys.path`` around
their own work and can drop a one-time insertion done at import.

Kept OUTSIDE ``docs/`` on purpose: files under ``docs_dir`` are copied into the
built site and scanned by mkdocs-jupyter — the helper is build machinery, not
content.
"""

from __future__ import annotations

import os
import sys

import matplotlib

matplotlib.use("Agg")

_GEN = os.path.join(os.path.dirname(os.path.abspath(__file__)), "docs_gen")


def _ensure_path() -> None:
    if _GEN not in sys.path:
        sys.path.insert(0, _GEN)


_ensure_path()


def on_page_markdown(markdown: str, **kwargs: object) -> str:
    """Guarantee ``docs_gen`` is importable while this page's code blocks run."""
    _ensure_path()
    return markdown
