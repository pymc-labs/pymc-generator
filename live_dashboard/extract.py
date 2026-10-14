"""Dashboard metrics of one published payload: a pure function of its bytes.

``extract_payload`` turns a ``save_corpus`` archive into identity-free numbers: every
world's composition (the trajectory-showcase notebook's channel / control / baseline
split of total outcome), every active channel's direct contribution share, and the
adstock and saturation families every world's active channels use. The
collector attaches run, task and world identities afterwards, so equal bytes published
by different attempts share one cached result keyed by digest and ``EXTRACTION_VERSION``.

Every problem with the content raises ``ExtractionError``. Those failures are
deterministic (same bytes, same code, same error), so the collector caches them instead
of re-reading a bad payload on every refresh; any other exception (no space for the
temporary file, memory) is environmental and must not be cached.
"""

from __future__ import annotations

import tempfile
import zipfile
import zlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from pymc_generator.data_generator import DataGenerator, load_corpus
from pymc_generator.outcomes import outcome_distributions
from pymc_generator.sampler import CARRYOVER_FAMILY_KEYS, SATURATION_FAMILY_KEYS

# Part of the collector's cache key: bump whenever the same bytes would yield other metrics.
EXTRACTION_VERSION = 3

# Every adstock (carryover) and saturation family of the library, in its own order: the
# corpus stores a channel's family as an index into these.
ADSTOCK_FAMILIES: tuple[str, ...] = tuple(CARRYOVER_FAMILY_KEYS)
SATURATION_FAMILIES: tuple[str, ...] = tuple(SATURATION_FAMILY_KEYS)

# Unit roundoff of the float32 storage: the notebook's per-world closure tolerance.
_STORAGE_ROUNDOFF = 2.0**-24
# The channel share and the library's treatment_total_contribution share are computed
# with the same reductions; anything beyond rounding means the formulas diverged.
_CHANNEL_RTOL = 1e-12
# What np.load / load_corpus / validate_corpus raise for bytes that are not a usable
# corpus (a non-archive is read as a refused pickle: ValueError; a bare .npy array has
# no context manager: TypeError; a damaged archive: BadZipFile, zlib.error or EOFError).
_CORPUS_ERRORS = (zipfile.BadZipFile, zlib.error, EOFError, ValueError, KeyError, TypeError)


class ExtractionError(ValueError):
    """The payload's content cannot produce dashboard metrics (same bytes, same error)."""


@dataclass(frozen=True)
class PayloadMetrics:
    """Metrics of one payload, free of run/task/world identities.

    ``channel``, ``control`` and ``baseline`` hold one signed share of total outcome per
    corpus row. The ``impact_*`` tuples describe one unit per active treatment slot,
    world-major then slot order: its corpus row, slot label ("C1".."CK") and direct
    contribution share. ``adstock_forms`` and ``saturation_forms`` hold, per corpus row,
    the families its active channels use (in the library's order); ``saturation_forms``
    is ``None`` when the payload does not record saturation families (the library writes
    them only for configurations with mechanism priors). ``active_channels`` and
    ``active_controls`` count each row's active channels and controls, out of the
    payload's ``channel_slots`` and ``control_slots``.
    """

    n_worlds: int
    channel: tuple[float, ...]
    control: tuple[float, ...]
    baseline: tuple[float, ...]
    impact_rows: tuple[int, ...]
    impact_labels: tuple[str, ...]
    impact_shares: tuple[float, ...]
    adstock_forms: tuple[tuple[str, ...], ...]
    saturation_forms: tuple[tuple[str, ...], ...] | None
    active_channels: tuple[int, ...]
    active_controls: tuple[int, ...]
    channel_slots: int
    control_slots: int


def extract_payload(data: bytes) -> PayloadMetrics:
    """Validate a corpus archive and compute its metrics; ``ExtractionError`` if unusable."""
    corpus = _load_valid_corpus(data)
    channel, control, baseline = _composition(corpus)
    distributions = outcome_distributions(
        corpus,
        quantities=["treatment_contribution", "treatment_total_contribution"],
        keep_series=False,
    )
    library_channel = distributions.quantities["treatment_total_contribution"].unit_share
    disagreeing = np.flatnonzero(
        ~(np.abs(channel - library_channel) <= _CHANNEL_RTOL * np.abs(library_channel))
    )
    if disagreeing.size:
        raise ExtractionError(
            "channel shares disagree with the library's treatment_total_contribution share "
            f"in worlds {disagreeing.tolist()}"
        )
    impact = distributions["treatment_contribution"]
    active = np.asarray(corpus["treatment_active_mask"]) == 1
    controls = np.asarray(corpus["covariate_active_mask"]) == 1
    adstock = _families(corpus, "carryover_family", active, ADSTOCK_FAMILIES)
    saturation = (
        _families(corpus, "sat_family", active, SATURATION_FAMILIES)
        if "sat_family" in corpus
        else None
    )
    return PayloadMetrics(
        n_worlds=int(channel.size),
        channel=tuple(channel.tolist()),
        control=tuple(control.tolist()),
        baseline=tuple(baseline.tolist()),
        impact_rows=tuple(impact.world_index.tolist()),
        impact_labels=impact.labels(),
        impact_shares=tuple(impact.unit_share.tolist()),
        adstock_forms=adstock,
        saturation_forms=saturation,
        active_channels=tuple(int(count) for count in active.sum(axis=1)),
        active_controls=tuple(int(count) for count in controls.sum(axis=1)),
        channel_slots=int(active.shape[1]),
        control_slots=int(controls.shape[1]),
    )


def _families(
    corpus: dict[str, Any], key: str, active: np.ndarray, names: tuple[str, ...]
) -> tuple[tuple[str, ...], ...]:
    """Per corpus row, the families its active channels use, in the library's order.

    Inactive (padded) slots never count: their zero fill is also a real family's id.
    """
    codes = np.asarray(corpus[key])
    if codes.shape != active.shape:
        raise ExtractionError(f"{key} has shape {codes.shape}, expected {active.shape}")
    forms = []
    for row_codes, row_active in zip(codes, active, strict=True):
        used = {int(code) for code in row_codes[row_active]}
        unknown = sorted(code for code in used if not 0 <= code < len(names))
        if unknown:
            raise ExtractionError(f"{key} holds unknown family ids {unknown}")
        forms.append(tuple(name for code, name in enumerate(names) if code in used))
    return tuple(forms)


def _load_valid_corpus(data: bytes) -> dict[str, Any]:
    # load_corpus reads a path. A private temporary directory keeps scratch files out of
    # the store, which the dashboard only reads (and may mount read-only).
    with tempfile.TemporaryDirectory(prefix="live-dashboard-extract-") as scratch:
        path = Path(scratch) / "payload.npz"
        path.write_bytes(data)
        try:
            corpus = load_corpus(path)
            errors = DataGenerator.validate_corpus(corpus)
        except _CORPUS_ERRORS as error:
            # Messages may embed the random scratch path; cached errors must not.
            message = str(error).replace(str(path), "<payload>")
            raise ExtractionError(
                f"payload is not a readable corpus: {type(error).__name__}: {message}"
            ) from error
    if errors:
        raise ExtractionError(
            f"payload fails corpus validation ({len(errors)} problems), first: {errors[0]}"
        )
    return corpus


def _composition(corpus: dict[str, Any]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Per-world channel, control and baseline shares (the notebook's realized_shares).

    channel = direct treatment contributions + indirect effects, control = covariate
    contributions, baseline = intrinsic baseline + latent contributions + outcome noise
    (not ``baseline_raw``, which already contains the controls), each summed over time
    in float64 and divided by the world's total outcome.
    """

    def load(key: str) -> np.ndarray:
        return np.asarray(corpus[key], dtype=np.float64)

    outcome = load("outcome_raw")
    treatment, indirect = load("treatment_contribution_raw"), load("indirect_effects")
    control, intrinsic = load("covariate_contribution"), load("baseline_intrinsic")
    latent, noise = load("latent_unobserved_contribution"), load("outcome_noise")
    outcome_total = outcome.sum(axis=1)
    not_positive = np.flatnonzero(~(outcome_total > 0.0))
    if not_positive.size:
        raise ExtractionError(
            f"shares need a positive total outcome; worlds {not_positive.tolist()} have none"
        )
    # Same reduction order as the library's treatment_total_contribution, so the
    # cross-check below can be exact to rounding.
    channel_total = (treatment.sum(axis=2) + indirect).sum(axis=1)
    control_total = control.sum(axis=2).sum(axis=1)
    baseline_total = intrinsic.sum(axis=1) + latent.sum(axis=2).sum(axis=1) + noise.sum(axis=1)
    residual = np.abs(channel_total + control_total + baseline_total - outcome_total)
    magnitude = (
        np.abs(outcome)
        + np.abs(treatment).sum(axis=2)
        + np.abs(indirect)
        + np.abs(control).sum(axis=2)
        + np.abs(intrinsic)
        + np.abs(latent).sum(axis=2)
        + np.abs(noise)
    ).sum(axis=1)
    open_worlds = np.flatnonzero(~(residual <= _STORAGE_ROUNDOFF * magnitude))
    if open_worlds.size:
        raise ExtractionError(
            "channel + control + baseline differ from the total outcome by more than the "
            f"float32 storage bound in worlds {open_worlds.tolist()}"
        )
    return (
        channel_total / outcome_total,
        control_total / outcome_total,
        baseline_total / outcome_total,
    )
