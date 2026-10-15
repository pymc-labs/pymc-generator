"""extract_payload: the notebook's composition shares, the inputs with an effect and their
impact, and the functional forms of a payload.

Payloads are the real corpora of the session's golden run (``small`` preset, two worlds
per task), so every check runs against genuine float32 decompositions; expected values
are recomputed here from the raw corpus arrays, independently of the implementation.
"""

from __future__ import annotations

import dataclasses
import importlib.util
import io
import json
import re
import sys
import tempfile
from pathlib import Path
from types import ModuleType
from typing import Any

import numpy as np
import pytest

import live_dashboard.extract as extract_module
from live_dashboard import publication as pub
from live_dashboard.extract import ExtractionError, extract_payload
from live_dashboard.storage import DirectoryStore
from pymc_generator.data_generator import DataGenerator, load_corpus, save_corpus
from pymc_generator.slots import SlotLayout


def _load(name: str) -> ModuleType:
    """Load a plain helper module of ``tests/`` by path, once per session."""
    loaded = sys.modules.get(name)
    if loaded is not None:
        return loaded
    spec = importlib.util.spec_from_file_location(name, Path(__file__).with_name(f"{name}.py"))
    assert spec is not None and spec.loader is not None
    module = sys.modules.setdefault(name, importlib.util.module_from_spec(spec))
    spec.loader.exec_module(module)
    return module


support = _load("_live_dashboard_support")


@pytest.fixture(autouse=True)
def _dashboard_environment(monkeypatch):
    for name in (
        "DASHBOARD_USERNAME",
        "DASHBOARD_PASSWORD_HASH",
        "DASHBOARD_POLL_SECONDS",
        "LIVE_DASHBOARD_CODE_REVISION",
    ):
        monkeypatch.delenv(name, raising=False)


@pytest.fixture(scope="module")
def payloads(live_golden) -> tuple[bytes, ...]:
    """The golden run's payload bytes in task order."""
    store = DirectoryStore(live_golden.root)
    run_id = live_golden.request.run_id
    return tuple(
        store.read_bytes(
            pub.completion_payload_path(run_id, pub.task_id_for(index), live_golden.attempt_id)
        )
        for index in range(live_golden.request.n_tasks)
    )


def _corpus(data: bytes, directory: Path) -> dict[str, Any]:
    path = directory / "corpus.npz"
    path.write_bytes(data)
    return load_corpus(path)


def _saved(corpus: dict[str, Any], directory: Path) -> bytes:
    path = directory / "edited.npz"
    save_corpus(corpus, path)
    return path.read_bytes()


def _f64(corpus: dict[str, Any], key: str) -> np.ndarray:
    return np.asarray(corpus[key], dtype=np.float64)


def _layout(corpus: dict[str, Any]) -> SlotLayout:
    """The library's edge layout of the corpus's channel, control and latent slots."""
    return SlotLayout(
        n_treatments=corpus["treatment_active_mask"].shape[1],
        n_covariates=corpus["covariate_active_mask"].shape[1],
        n_latent=corpus["latent_active_mask"].shape[1],
    )


def test_composition_matches_the_raw_array_formula_within_the_closure_bound(payloads, tmp_path):
    largest_indirect_share = 0.0
    for data in payloads:
        metrics = extract_payload(data)
        corpus = _corpus(data, tmp_path)
        outcome = _f64(corpus, "outcome_raw")
        treatment = _f64(corpus, "treatment_contribution_raw")
        indirect = _f64(corpus, "indirect_effects")
        control = _f64(corpus, "covariate_contribution")
        intrinsic = _f64(corpus, "baseline_intrinsic")
        latent = _f64(corpus, "latent_unobserved_contribution")
        noise = _f64(corpus, "outcome_noise")
        total = outcome.sum(axis=1)
        baseline = intrinsic.sum(axis=1) + latent.sum(axis=(1, 2)) + noise.sum(axis=1)
        expected = {
            "channel": (treatment.sum(axis=(1, 2)) + indirect.sum(axis=1)) / total,
            "control": control.sum(axis=(1, 2)) / total,
            "baseline": baseline / total,
        }
        assert metrics.n_worlds == outcome.shape[0] == 2
        for part, values in expected.items():
            assert np.max(np.abs(np.asarray(getattr(metrics, part)) - values)) <= 1e-7, part
        magnitude = (
            np.abs(outcome).sum(axis=1)
            + np.abs(treatment).sum(axis=(1, 2))
            + np.abs(indirect).sum(axis=1)
            + np.abs(control).sum(axis=(1, 2))
            + np.abs(intrinsic).sum(axis=1)
            + np.abs(latent).sum(axis=(1, 2))
            + np.abs(noise).sum(axis=1)
        )
        closure = np.sum([metrics.channel, metrics.control, metrics.baseline], axis=0)
        assert np.all(np.abs(closure - 1.0) * total <= 2.0**-24 * magnitude)
        largest_indirect_share = max(
            largest_indirect_share, float(np.max(np.abs(indirect.sum(axis=1) / total)))
        )
    # Precondition: dropping the indirect effects would miss the tolerance by far.
    assert largest_indirect_share > 1e-3


@pytest.mark.parametrize(
    ("field", "mask", "edge", "contribution", "letter"),
    [
        ("impact", "treatment_active_mask", "cy", "treatment_contribution_raw", "C"),
        ("control_impact", "covariate_active_mask", "zy", "covariate_contribution", "Z"),
    ],
)
def test_impact_units_are_the_active_slots_with_an_edge_and_their_direct_share(
    payloads, tmp_path, field, mask, edge, contribution, letter
):
    saw_inactive_slot = False
    for data in payloads:
        metrics = extract_payload(data)
        corpus = _corpus(data, tmp_path)
        active = np.asarray(corpus[mask]) == 1
        saw_inactive_slot |= not active.all()
        rows, slots = np.nonzero(active & (_layout(corpus).unpack(corpus["g"])[edge] == 1))
        assert getattr(metrics, f"{field}_rows") == tuple(rows.tolist())
        labels = tuple(f"{letter}{slot + 1}" for slot in slots.tolist())
        assert getattr(metrics, f"{field}_labels") == labels
        direct = _f64(corpus, contribution).sum(axis=1)
        total = _f64(corpus, "outcome_raw").sum(axis=1)
        expected = direct[rows, slots] / total[rows]
        shares = np.asarray(getattr(metrics, f"{field}_shares"))
        assert np.all(np.abs(shares - expected) <= 1e-7)
    # Precondition: padding slots exist, so including them would change the units.
    assert saw_inactive_slot


def _names(codes: np.ndarray, keys: tuple[str, ...]) -> tuple[str, ...]:
    """Family names of ``codes``, once each, in the library's order."""
    return tuple(name for code, name in enumerate(keys) if code in set(codes.tolist()))


def test_functional_forms_are_the_families_of_the_active_channels(payloads, tmp_path):
    from pymc_generator.sampler import CARRYOVER_FAMILY_KEYS

    saw_inactive_zero = False
    for data in payloads:
        metrics = extract_payload(data)
        corpus = _corpus(data, tmp_path)
        active = np.asarray(corpus["treatment_active_mask"]) == 1
        codes = np.asarray(corpus["carryover_family"])
        saw_inactive_zero |= bool((codes[~active] == 0).any())
        expected = tuple(
            _names(row[mask], CARRYOVER_FAMILY_KEYS) for row, mask in zip(codes, active)
        )
        assert metrics.adstock_forms == expected
        # The small preset has no mechanism priors, so its payloads do not store sat_family.
        assert "sat_family" not in corpus
        assert (metrics.saturation_forms, metrics.structural_pairs) == (None, None)
    # Precondition: padding holds code 0 (also "none"), so counting it would add "none".
    assert saw_inactive_zero


def test_column_and_effect_counts_are_row_sums_of_the_masks_and_their_edges(payloads, tmp_path):
    saw_padding = False
    for data in payloads:
        metrics = extract_payload(data)
        corpus = _corpus(data, tmp_path)
        channels = np.asarray(corpus["treatment_active_mask"]) == 1
        controls = np.asarray(corpus["covariate_active_mask"]) == 1
        graph = _layout(corpus).unpack(corpus["g"])
        saw_padding |= not (channels.all() and controls.all())
        assert metrics.active_channels == tuple(channels.sum(axis=1).tolist())
        assert metrics.active_controls == tuple(controls.sum(axis=1).tolist())
        channel_edges = (graph["cy"] == 1) & channels
        control_edges = (graph["zy"] == 1) & controls
        assert metrics.effect_channels == tuple(channel_edges.sum(axis=1).tolist())
        assert metrics.effect_controls == tuple(control_edges.sum(axis=1).tolist())
        assert (metrics.channel_slots, metrics.control_slots) == (
            channels.shape[1],
            controls.shape[1],
        )
    # Precondition: some worlds leave slots inactive, so counting slots would differ.
    assert saw_padding


def test_zero_weeks_count_each_active_column_s_exact_zeros_per_third(
    payloads, tmp_path, lenient_validation
):
    corpus = _corpus(payloads[0], tmp_path)
    channels = np.asarray(corpus["treatment_active_mask"]) == 1
    controls = np.asarray(corpus["covariate_active_mask"]) == 1
    assert not channels.all()  # precondition: a padded slot, zero-filled, must not count
    weeks = corpus["treatment_raw"].shape[1]
    bounds = (0, weeks // 3, 2 * weeks // 3, weeks)
    channel = int(np.flatnonzero(channels[0])[0])
    control = int(np.flatnonzero(controls[0])[0])
    corpus["treatment_raw"][0, : bounds[1], channel] = 0.0  # off for the whole first third
    corpus["treatment_raw"][0, bounds[2] : bounds[2] + 3, channel] = 0.0
    corpus["covariates"][0, bounds[1] : bounds[1] + 5, control] = 0.0
    metrics = extract_payload(_saved(corpus, tmp_path))
    assert metrics.third_weeks == (bounds[1], bounds[2] - bounds[1], weeks - bounds[2])
    for key, active, found in (
        ("treatment_raw", channels, metrics.channel_zero_weeks),
        ("covariates", controls, metrics.control_zero_weeks),
    ):
        zero = np.asarray(corpus[key]) == 0
        expected = tuple(
            tuple(
                tuple(
                    int(zero[row, start:stop, slot].sum())
                    for start, stop in zip(bounds, bounds[1:])
                )
                for slot in np.flatnonzero(active[row])
            )
            for row in range(active.shape[0])
        )
        assert found == expected
    first, _, last = metrics.channel_zero_weeks[0][0]
    assert first == bounds[1] and last >= 3
    assert metrics.control_zero_weeks[0][0][1] >= 5


def test_a_column_without_a_direct_edge_has_no_effect_and_no_impact_unit(
    payloads, tmp_path, lenient_validation
):
    corpus = _corpus(payloads[0], tmp_path)
    channels = np.asarray(corpus["treatment_active_mask"]) == 1
    controls = np.asarray(corpus["covariate_active_mask"]) == 1
    # Preconditions: C1 and Z1 are columns of both worlds, and a padded channel slot exists.
    assert channels[:, 0].all() and controls[:, 0].all() and not channels.all()
    layout = _layout(corpus)
    graph = layout.unpack(corpus["g"])
    # Every column points to the target except world 0's C1 and world 1's Z1. A padded
    # slot gets an edge too: it is no column, so it has no effect either.
    graph["cy"] = channels.astype(np.uint8)
    graph["zy"] = controls.astype(np.uint8)
    graph["cy"][0, 0] = 0
    graph["zy"][1, 0] = 0
    graph["cy"][tuple(np.argwhere(~channels)[0])] = 1
    corpus["g"] = layout.pack(**{f"g_{edge}": block for edge, block in graph.items()})
    metrics = extract_payload(_saved(corpus, tmp_path))
    channel_effects, control_effects = channels.copy(), controls.copy()
    channel_effects[0, 0] = False
    control_effects[1, 0] = False
    assert metrics.active_channels == tuple(channels.sum(axis=1).tolist())
    assert metrics.effect_channels == tuple(channel_effects.sum(axis=1).tolist())
    assert metrics.active_controls == tuple(controls.sum(axis=1).tolist())
    assert metrics.effect_controls == tuple(control_effects.sum(axis=1).tolist())
    total = _f64(corpus, "outcome_raw").sum(axis=1)
    for field, effects, contribution, letter in (
        ("impact", channel_effects, "treatment_contribution_raw", "C"),
        ("control_impact", control_effects, "covariate_contribution", "Z"),
    ):
        rows, slots = np.nonzero(effects)
        assert getattr(metrics, f"{field}_rows") == tuple(rows.tolist())
        labels = tuple(f"{letter}{slot + 1}" for slot in slots.tolist())
        assert getattr(metrics, f"{field}_labels") == labels
        expected = _f64(corpus, contribution).sum(axis=1)[rows, slots] / total[rows]
        shares = np.asarray(getattr(metrics, f"{field}_shares"))
        assert np.all(np.abs(shares - expected) <= 1e-7)


def test_an_edge_vector_of_another_layout_is_refused(payloads, tmp_path, lenient_validation):
    corpus = _corpus(payloads[0], tmp_path)
    worlds, width = corpus["g"].shape
    corpus["g"] = corpus["g"][:, :-1]
    expected = f"g has shape ({worlds}, {width - 1}), expected ({worlds}, {width})"
    with pytest.raises(ExtractionError, match=re.escape(expected)):
        extract_payload(_saved(corpus, tmp_path))


def test_recorded_saturation_families_skip_padded_slots(payloads, tmp_path, lenient_validation):
    corpus = _corpus(payloads[0], tmp_path)
    active = np.asarray(corpus["treatment_active_mask"]) == 1
    assert not active.all()  # precondition: a padded slot exists
    codes = np.full(active.shape, 4, dtype=np.uint8)  # tanh, left only in padded slots
    codes[0][active[0]] = 2  # logistic
    codes[1][active[1]] = 3  # michaelis_menten
    codes[1][np.flatnonzero(active[1])[0]] = 5  # root, so world 1 mixes two families
    corpus["sat_family"] = codes
    metrics = extract_payload(_saved(corpus, tmp_path))
    assert metrics.saturation_forms == (("logistic",), ("michaelis_menten", "root"))


def test_structural_pairs_are_the_distinct_families_of_each_active_channel(
    payloads, tmp_path, lenient_validation
):
    corpus = _corpus(payloads[0], tmp_path)
    active = np.asarray(corpus["treatment_active_mask"]) == 1
    # Preconditions: C1 and another channel are active in world 1; a padded slot exists.
    assert active[:, 0].all() and active[1].sum() >= 2 and not active.all()
    saturation = np.full(active.shape, 4, dtype=np.uint8)  # tanh, left only in padded slots
    adstock = np.full(active.shape, 2, dtype=np.uint8)  # weibull, likewise
    saturation[active] = 3  # michaelis_menten
    adstock[active] = 1  # geometric
    saturation[0][active[0]] = 2  # world 0: every channel logistic-geometric, one pair
    saturation[1, 0] = 5  # world 1: C1 root-none, the others michaelis_menten-geometric
    adstock[1, 0] = 0
    corpus["sat_family"] = saturation
    corpus["carryover_family"] = adstock
    metrics = extract_payload(_saved(corpus, tmp_path))
    assert metrics.structural_pairs == (
        (("logistic", "geometric"),),
        (("michaelis_menten", "geometric"), ("root", "none")),
    )
    # Pairs, not every combination of the world's forms: no channel is root-geometric.
    assert (metrics.saturation_forms[1], metrics.adstock_forms[1]) == (
        ("michaelis_menten", "root"),
        ("none", "geometric"),
    )


@pytest.mark.parametrize("key", ["carryover_family", "sat_family"])
def test_unknown_family_ids_are_refused(payloads, tmp_path, lenient_validation, key):
    corpus = _corpus(payloads[0], tmp_path)
    codes = np.asarray(corpus["carryover_family"]).copy()
    codes[0, 0] = 9  # slot C1 is active in every world
    corpus[key] = codes
    with pytest.raises(ExtractionError, match=rf"{key} holds unknown family ids \[9\]"):
        extract_payload(_saved(corpus, tmp_path))


def test_metrics_are_deterministic_plain_python_values(payloads):
    data = payloads[0]
    metrics = extract_payload(data)
    assert extract_payload(data) == metrics  # the collector caches by digest
    fields = dataclasses.asdict(metrics)
    shares = metrics.channel + metrics.impact_shares + metrics.control_impact_shares
    assert {type(value) for value in shares} == {float}
    counts = (
        metrics.impact_rows
        + metrics.control_impact_rows
        + metrics.effect_channels
        + metrics.effect_controls
    )
    assert {type(value) for value in counts} == {int}
    json.dumps(fields, allow_nan=False)  # figures serialise these without numpy


def _npy(_: bytes, __: Path) -> bytes:
    buffer = io.BytesIO()
    np.save(buffer, np.arange(3))
    return buffer.getvalue()


def _foreign_npz(_: bytes, __: Path) -> bytes:
    buffer = io.BytesIO()
    np.savez(buffer, outcome_raw=np.ones((2, 3)))
    return buffer.getvalue()


def _flipped(data: bytes, _: Path) -> bytes:
    middle = len(data) // 2
    return data[:middle] + bytes([data[middle] ^ 0xFF]) + data[middle + 1 :]


def _missing_key(data: bytes, directory: Path) -> bytes:
    corpus = _corpus(data, directory)
    del corpus["baseline_intrinsic"]
    return _saved(corpus, directory)


DAMAGE = {
    "not-an-archive": lambda data, directory: b"definitely not a corpus",
    "truncated": lambda data, directory: data[: len(data) // 2],
    "flipped-byte": _flipped,
    "bare-npy": _npy,
    "foreign-npz": _foreign_npz,
    "fails-validation": _missing_key,
}


@pytest.mark.parametrize("damage", sorted(DAMAGE))
def test_unusable_payloads_raise_the_same_extraction_error_and_leave_no_scratch(
    payloads, tmp_path, monkeypatch, damage
):
    data = DAMAGE[damage](payloads[0], tmp_path)
    scratch = tmp_path / "scratch"
    scratch.mkdir()
    monkeypatch.setattr(tempfile, "tempdir", str(scratch))
    with pytest.raises(ExtractionError) as first:
        extract_payload(data)
    with pytest.raises(ExtractionError) as second:
        extract_payload(data)
    # Errors are cached and shown by digest: they must not depend on the call.
    assert str(first.value) == str(second.value)
    assert list(scratch.iterdir()) == []


def test_successful_extraction_leaves_no_scratch(payloads, tmp_path, monkeypatch):
    monkeypatch.setattr(tempfile, "tempdir", str(tmp_path))
    extract_payload(payloads[0])
    assert list(tmp_path.iterdir()) == []


@pytest.fixture
def lenient_validation(monkeypatch):
    """Let corpora through validation, to reach the extraction's own guards."""
    monkeypatch.setattr(DataGenerator, "validate_corpus", staticmethod(lambda corpus, **_: []))


def test_worlds_without_positive_total_outcome_are_refused(payloads, tmp_path, lenient_validation):
    corpus = _corpus(payloads[0], tmp_path)
    corpus["outcome_raw"][1] = 0.0
    with pytest.raises(ExtractionError, match=r"positive total outcome.*\[1\]"):
        extract_payload(_saved(corpus, tmp_path))


def test_parts_must_close_on_the_outcome_within_the_storage_bound(
    payloads, tmp_path, lenient_validation
):
    corpus = _corpus(payloads[0], tmp_path)
    total = float(_f64(corpus, "outcome_raw")[0].sum())
    corpus["baseline_intrinsic"][0, 0] += np.float32(1e-3 * total)
    with pytest.raises(ExtractionError, match=r"storage bound in worlds \[0\]"):
        extract_payload(_saved(corpus, tmp_path))


@pytest.mark.parametrize(("relative_error", "accepted"), [(1e-14, True), (1e-9, False)])
def test_channel_shares_must_agree_with_the_library_to_1e_12(
    payloads, monkeypatch, relative_error, accepted
):
    library = extract_module.outcome_distributions

    def perturbed(*args, **kwargs):
        distributions = library(*args, **kwargs)
        total = distributions.quantities["treatment_total_contribution"]
        distributions.quantities["treatment_total_contribution"] = dataclasses.replace(
            total, unit_share=total.unit_share * (1.0 + relative_error)
        )
        return distributions

    monkeypatch.setattr(extract_module, "outcome_distributions", perturbed)
    if accepted:
        extract_payload(payloads[0])
    else:
        with pytest.raises(ExtractionError, match="treatment_total_contribution"):
            extract_payload(payloads[0])
