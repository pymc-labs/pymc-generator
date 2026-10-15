"""Dashboard metrics of one published payload: a pure function of its bytes.

``extract_payload`` turns a ``save_corpus`` archive into identity-free numbers: every
world's composition (the trajectory-showcase notebook's channel / control / baseline
split of total outcome), its channels and controls with an effect (a direct edge to the
target) and their direct contribution shares, the adstock and saturation families (and
their pairs) its active channels use, and the weeks at exactly zero of each active channel
and control per third of the series. The collector attaches run, task and world
identities afterwards, so equal bytes published by different attempts share one cached
result keyed by digest and ``EXTRACTION_VERSION``.

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
from pymc_generator.outcomes import QuantityDistribution, outcome_distributions
from pymc_generator.sampler import CARRYOVER_FAMILY_KEYS, SATURATION_FAMILY_KEYS
from pymc_generator.slots import EDGE_TYPES_EXTENDED, SlotLayout

# Part of the collector's cache key: bump whenever the same bytes would yield other metrics.
EXTRACTION_VERSION = 6

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
    corpus row. A row's columns are its active channel and control slots; an input with
    an effect is a column with a direct edge to the target (``C->Y`` for a channel,
    ``Z->Y`` for a control). The ``impact_*`` tuples describe one unit per channel with
    an effect, world-major then slot order: its corpus row, slot label ("C1".."CK") and
    direct contribution share; the ``control_impact_*`` tuples do the same for every
    control with an effect ("Z1".."ZM"). ``adstock_forms`` and ``saturation_forms`` hold,
    per corpus row, the families its active channels use (in the library's order), and
    ``structural_pairs`` the distinct (saturation, adstock) family pairs of its active
    channels, sorted; ``saturation_forms`` and ``structural_pairs`` are ``None`` when the
    payload does not record saturation families (the library writes them only for
    configurations with mechanism priors). ``active_channels`` and ``active_controls``
    count each row's columns, out of the payload's ``channel_slots`` and
    ``control_slots``; ``effect_channels`` and ``effect_controls`` count its inputs with
    an effect. ``third_weeks`` splits the reported weeks into a first, middle and last
    third (at ``weeks // 3`` and ``2 * weeks // 3``); ``channel_zero_weeks`` and
    ``control_zero_weeks`` hold, per corpus row and active column in slot order, the
    weeks its series (``treatment_raw`` or ``covariates``) is exactly 0 in each third.
    """

    n_worlds: int
    channel: tuple[float, ...]
    control: tuple[float, ...]
    baseline: tuple[float, ...]
    impact_rows: tuple[int, ...]
    impact_labels: tuple[str, ...]
    impact_shares: tuple[float, ...]
    control_impact_rows: tuple[int, ...]
    control_impact_labels: tuple[str, ...]
    control_impact_shares: tuple[float, ...]
    adstock_forms: tuple[tuple[str, ...], ...]
    saturation_forms: tuple[tuple[str, ...], ...] | None
    structural_pairs: tuple[tuple[tuple[str, str], ...], ...] | None
    active_channels: tuple[int, ...]
    active_controls: tuple[int, ...]
    effect_channels: tuple[int, ...]
    effect_controls: tuple[int, ...]
    channel_slots: int
    control_slots: int
    third_weeks: tuple[int, int, int]
    channel_zero_weeks: tuple[tuple[tuple[int, int, int], ...], ...]
    control_zero_weeks: tuple[tuple[tuple[int, int, int], ...], ...]


def extract_payload(data: bytes) -> PayloadMetrics:
    """Validate a corpus archive and compute its metrics; ``ExtractionError`` if unusable."""
    corpus = _load_valid_corpus(data)
    channel, control, baseline = _composition(corpus)
    distributions = outcome_distributions(
        corpus,
        quantities=[
            "treatment_contribution",
            "treatment_total_contribution",
            "covariate_contribution",
        ],
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
    active = np.asarray(corpus["treatment_active_mask"]) == 1
    controls = np.asarray(corpus["covariate_active_mask"]) == 1
    channel_edges, control_edges = _direct_edges(corpus, active, controls)
    impact_rows, impact_labels, impact_shares = _units_with_effect(
        distributions["treatment_contribution"], channel_edges
    )
    control_rows, control_labels, control_shares = _units_with_effect(
        distributions["covariate_contribution"], control_edges
    )
    adstock = _channel_families(corpus, "carryover_family", active, ADSTOCK_FAMILIES)
    saturation = (
        _channel_families(corpus, "sat_family", active, SATURATION_FAMILIES)
        if "sat_family" in corpus
        else None
    )
    weeks = int(np.asarray(corpus["treatment_raw"]).shape[1])
    bounds = (0, weeks // 3, 2 * weeks // 3, weeks)
    return PayloadMetrics(
        n_worlds=int(channel.size),
        channel=tuple(channel.tolist()),
        control=tuple(control.tolist()),
        baseline=tuple(baseline.tolist()),
        impact_rows=impact_rows,
        impact_labels=impact_labels,
        impact_shares=impact_shares,
        control_impact_rows=control_rows,
        control_impact_labels=control_labels,
        control_impact_shares=control_shares,
        adstock_forms=_forms(adstock, ADSTOCK_FAMILIES),
        saturation_forms=None if saturation is None else _forms(saturation, SATURATION_FAMILIES),
        structural_pairs=None if saturation is None else _pairs(saturation, adstock),
        active_channels=tuple(int(count) for count in active.sum(axis=1)),
        active_controls=tuple(int(count) for count in controls.sum(axis=1)),
        effect_channels=tuple(int(count) for count in channel_edges.sum(axis=1)),
        effect_controls=tuple(int(count) for count in control_edges.sum(axis=1)),
        channel_slots=int(active.shape[1]),
        control_slots=int(controls.shape[1]),
        third_weeks=(bounds[1] - bounds[0], bounds[2] - bounds[1], bounds[3] - bounds[2]),
        channel_zero_weeks=_zero_weeks(corpus["treatment_raw"], active, bounds),
        control_zero_weeks=_zero_weeks(corpus["covariates"], controls, bounds),
    )


def _direct_edges(
    corpus: dict[str, Any], channels: np.ndarray, controls: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Per corpus row and slot, the inputs with an effect: active channels with a direct
    ``C->Y`` edge and active controls with a direct ``Z->Y`` edge.

    The edges are the ``cy`` and ``zy`` blocks of the corpus edge vector ``g``, located
    by the library's canonical layout of the payload's channel, control and latent slots.
    """
    layout = SlotLayout(
        n_treatments=channels.shape[1],
        n_covariates=controls.shape[1],
        n_latent=np.asarray(corpus["latent_active_mask"]).shape[1],
        edge_types=EDGE_TYPES_EXTENDED,
    )
    g = np.asarray(corpus["g"])
    expected = (channels.shape[0], layout.n_slots)
    if g.shape != expected:
        raise ExtractionError(f"g has shape {g.shape}, expected {expected}")
    return (
        (g[:, layout.slices["cy"]] == 1) & channels,
        (g[:, layout.slices["zy"]] == 1) & controls,
    )


def _units_with_effect(
    units: QuantityDistribution, edges: np.ndarray
) -> tuple[tuple[int, ...], tuple[str, ...], tuple[float, ...]]:
    """Corpus rows, slot labels and direct shares of the units whose slot has an effect.

    ``outcome_distributions`` makes one unit per active slot, world-major then slot
    order: ``world_index`` is its corpus row (every row is analysed) and
    ``column_index`` its slot. The kept units stay in that order.
    """
    keep = edges[units.world_index, units.column_index]
    return (
        tuple(units.world_index[keep].tolist()),
        tuple(label for label, kept in zip(units.labels(), keep.tolist(), strict=True) if kept),
        tuple(units.unit_share[keep].tolist()),
    )


def _channel_families(
    corpus: dict[str, Any], key: str, active: np.ndarray, names: tuple[str, ...]
) -> tuple[tuple[str, ...], ...]:
    """Per corpus row, the family of each active channel, in slot order.

    Inactive (padded) slots never count: their zero fill is also a real family's id.
    """
    codes = np.asarray(corpus[key])
    if codes.shape != active.shape:
        raise ExtractionError(f"{key} has shape {codes.shape}, expected {active.shape}")
    families = []
    for row_codes, row_active in zip(codes, active, strict=True):
        used = [int(code) for code in row_codes[row_active]]
        unknown = sorted({code for code in used if not 0 <= code < len(names)})
        if unknown:
            raise ExtractionError(f"{key} holds unknown family ids {unknown}")
        families.append(tuple(names[code] for code in used))
    return tuple(families)


def _forms(
    families: tuple[tuple[str, ...], ...], names: tuple[str, ...]
) -> tuple[tuple[str, ...], ...]:
    """Per corpus row, the families its active channels use, once each, in ``names`` order."""
    return tuple(tuple(name for name in names if name in row) for row in families)


def _pairs(
    saturation: tuple[tuple[str, ...], ...], adstock: tuple[tuple[str, ...], ...]
) -> tuple[tuple[tuple[str, str], ...], ...]:
    """Per corpus row, the distinct (saturation, adstock) pairs of its active channels, sorted."""
    return tuple(
        tuple(sorted(set(zip(row_saturation, row_adstock, strict=True))))
        for row_saturation, row_adstock in zip(saturation, adstock, strict=True)
    )


def _zero_weeks(
    series: Any, active: np.ndarray, bounds: tuple[int, int, int, int]
) -> tuple[tuple[tuple[int, int, int], ...], ...]:
    """Per corpus row and active column (slot order), its weeks at exactly 0 per third.

    ``series`` is (rows, weeks, slots); padded slots never count.
    """
    zero = np.asarray(series) == 0
    if zero.shape[::2] != active.shape:
        raise ExtractionError(f"series of shape {zero.shape} do not match slots {active.shape}")
    per_third = np.stack(
        [zero[:, start:stop, :].sum(axis=1) for start, stop in zip(bounds, bounds[1:])], axis=-1
    )
    return tuple(
        tuple(
            (int(first), int(middle), int(last))
            for first, middle, last in per_third[row][active[row]].tolist()
        )
        for row in range(active.shape[0])
    )


def _load_valid_corpus(data: bytes) -> dict[str, Any]:
    # load_corpus reads a path. A private temporary directory keeps scratch files out of
    # the store, which the dashboard only reads (and may mount read-only).
    with tempfile.TemporaryDirectory(prefix="live-dashboard-extract-") as scratch:
        path = Path(scratch) / "payload.npz"
        path.write_bytes(data)
        try:
            corpus = load_corpus(path)
            # Completions carry deferred (unassigned, is_val=2) train/validation membership.
            errors = DataGenerator.validate_corpus(corpus, allow_deferred_split=True)
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
