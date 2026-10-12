"""Saved-component recovery gates; pure computation with no generation or inference."""

from __future__ import annotations

import numpy as np

from .config import ContractError


def score_components(
    truth: np.ndarray,
    estimate: np.ndarray,
    terms: list[str],
    *,
    relative: float = 0.1,
    absolute: float = 0.01,
) -> dict:
    """Per-channel and aggregate L2 gates; joint excludes the baseline."""
    if (
        truth.ndim != 2
        or estimate.shape != truth.shape
        or truth.shape[1] != len(terms)
        or not terms
        or terms[0] != "Baseline"
    ):
        raise ContractError("component shape/terms mismatch")
    if not np.isfinite(truth).all() or not np.isfinite(estimate).all():
        raise ContractError("nonfinite contribution")
    noiseless_norm = np.linalg.norm(truth.sum(axis=1))

    def gate(actual: np.ndarray, predicted: np.ndarray) -> dict:
        allowance = max(relative * np.linalg.norm(actual), absolute * noiseless_norm)
        error = float(np.linalg.norm(actual - predicted))
        return {"error": error, "allowance": float(allowance), "pass": bool(error <= allowance)}

    channels = {name: gate(truth[:, i], estimate[:, i]) for i, name in enumerate(terms) if i > 0}
    media = [name for name in channels if name.startswith("C")]
    controls = [name for name in channels if name.startswith("Z")]
    if len(media) + len(controls) != len(channels):
        raise ContractError("unexpected term vocabulary")
    return {
        "channels": channels,
        "media_pass": all(channels[name]["pass"] for name in media),
        "controls_pass": all(channels[name]["pass"] for name in controls),
        "joint_pass": all(value["pass"] for value in channels.values()),
        "total_media": gate(
            truth[:, [terms.index(name) for name in media]].sum(axis=1),
            estimate[:, [terms.index(name) for name in media]].sum(axis=1),
        ),
        "baseline": gate(truth[:, 0], estimate[:, 0]),
        "media_count": len(media),
        "control_count": len(controls),
        "nonzero_media": [
            name for name in media if np.linalg.norm(truth[:, terms.index(name)]) > 0
        ],
        "nonzero_controls": [
            name for name in controls if np.linalg.norm(truth[:, terms.index(name)]) > 0
        ],
    }
