"""``validate_corpus`` rules for the optional per-input trajectory block.

The fixture is an ordinary corpus with a synthetic trajectory block attached
that satisfies every rule, so these tests pin the validator independently of
the generation code that emits the block.
"""

from __future__ import annotations

import copy

import numpy as np
import pytest

import pymc_generator as pg
from pymc_generator import DataGenerator
from pymc_generator.slots import (
    EDGE_TYPES_EXTENDED,
    TRAJECTORY_ARRAY_FIELDS,
    TRAJECTORY_COMPONENTS,
    TRAJECTORY_INPUTS,
    SlotLayout,
)
from pymc_generator.trajectories import GATE_COMPONENTS, LEVEL_COMPONENTS

#: Components per (cell, input slot), varying across cells and inputs. Seed 21
#: pads treatment 2 in cells 0 and 2 and covariate 1 in cells 1 and 2; those
#: entries are masked out, so padding stays zero by construction. Treatment 2 is
#: the only treatment without a C->Y edge; seed 21 shocks treatment 0 in both
#: tasks of cell 2, the one direct treatment given a gate form.
TREATMENT_SETS = (
    (("hf", "seasonal"), ("hf", "pulse"), ("trend",)),
    (("hf", "level_jump"), ("hf", "pulse", "trend"), ("onset", "offset", "flighting", "seasonal")),
    (("flighting", "seasonal", "trend"), (), ("onset",)),
)
COVARIATE_SETS = (
    (("pulse", "flighting", "level_jump"), ("hf",)),
    (("hf", "offset"), ("seasonal",)),
    (("onset", "seasonal", "trend"), ("flighting",)),
)

OFF_WEEK_TREATMENT = "treatment_raw must be 0 on treatment_activity off-weeks outside shocks"
LAYOUT = "diagnostics trajectory components do not match the canonical layout"
INCLUSION = (
    "diagnostics trajectory inclusion_probs must map every input and component to a float in [0, 1]"
)
PREVALENCE = "diagnostics trajectory prevalence does not match recomputation"
CONTRADICTION = "diagnostics trajectory inclusion_probs contradict the stored {}_components for {}"
_DELETE = object()


def _active(corpus, input_type):
    """(task, input) mask of active slots."""
    return corpus[f"{input_type}_active_mask"] == 1


def _carries(corpus, input_type, components):
    """(task, input) mask of slots whose stored flags include any of ``components``."""
    columns = [TRAJECTORY_COMPONENTS.index(name) for name in components]
    return corpus[f"{input_type}_components"][..., columns].any(axis=-1)


def _direct(corpus):
    """(task, treatment) mask of treatments with a C->Y edge."""
    layout = SlotLayout(
        n_treatments=corpus["treatment_raw"].shape[2],
        n_covariates=corpus["covariates"].shape[2],
        n_latent=corpus["latent_unobserved"].shape[2],
        edge_types=EDGE_TYPES_EXTENDED,
    )
    return corpus["g"][:, layout.slices["cy"]] == 1


def _on_weeks(flags, n_time):
    """(task, time, input) gate schedule: launch at T/4, stop at 3T/4, 4-on/4-off flighting."""
    week = np.arange(n_time)[None, :, None]

    def carries(name):
        return flags[:, None, :, TRAJECTORY_COMPONENTS.index(name)] == 1

    return (
        (~carries("onset") | (week >= n_time // 4))
        & (~carries("offset") | (week < 3 * n_time // 4))
        & (~carries("flighting") | (week // 4 % 2 == 0))
    )


def _renormalized(corpus):
    """Copy of ``corpus`` with the treatment normalizations recomputed from ``treatment_raw``."""
    raw = corpus["treatment_raw"].astype(np.float64)
    mask = corpus["treatment_active_mask"][:, None, :]
    means = raw.mean(axis=1)
    total = (raw * mask).sum(axis=-1, keepdims=True)
    return {
        **corpus,
        "treatment_means": means.astype(np.float32),
        "treatment_norm": np.divide(
            raw, means[:, None, :], out=np.zeros_like(raw), where=means[:, None, :] != 0.0
        ).astype(np.float32),
        "treatment_share": (
            np.divide(raw, total, out=np.zeros_like(raw), where=total != 0.0) * mask
        ).astype(np.float32),
    }


def _attach_trajectory_block(legacy):
    """``legacy`` plus a synthetic trajectory block that satisfies every rule.

    A direct treatment's ``treatment_raw`` feeds the recomputed signal labels, so
    it stays untouched: a gated direct treatment is switched off exactly under
    its held shock, which overrides the gate. The other inputs follow
    :func:`_on_weeks`; their series are zeroed on off-weeks and the treatment
    normalizations recomputed, and nothing else the validator checks reads
    those entries.
    """
    corpus = dict(legacy)
    n_time = legacy["treatment_raw"].shape[1]
    for input_type, table in (("treatment", TREATMENT_SETS), ("covariate", COVARIATE_SETS)):
        active = _active(legacy, input_type)
        flags = np.zeros((*active.shape, len(TRAJECTORY_COMPONENTS)), dtype=np.uint8)
        for n, cell in enumerate(legacy["cell_id"]):
            for i, names in enumerate(table[cell]):
                for name in names:
                    flags[n, i, TRAJECTORY_COMPONENTS.index(name)] = 1
        flags[~active] = 0
        corpus[f"{input_type}_components"] = flags

    shocked = legacy["treatment_shock_mask"] == 1
    gated = _carries(corpus, "treatment", GATE_COMPONENTS)[:, None, :]
    on_weeks = {
        "treatment": np.where(
            _direct(legacy)[:, None, :],
            ~(gated & shocked),
            _on_weeks(corpus["treatment_components"], n_time),
        ),
        "covariate": _on_weeks(corpus["covariate_components"], n_time),
    }
    held = {"treatment": shocked, "covariate": False}
    level = (0.5 * np.sin(2.0 * np.pi * np.arange(n_time) / 13.0) + 0.25).astype(np.float32)
    for input_type, series_key, shift_key in (
        ("treatment", "treatment_raw", "treatment_log_level_shift"),
        ("covariate", "covariates", "covariate_level_shift"),
    ):
        active = _active(legacy, input_type)[:, None, :]
        activity = on_weeks[input_type] & active
        # The off-week rules must accept a real off-week of an active input.
        assert (~activity & active).any(), f"the fixture needs an active {input_type} off-week"
        corpus[f"{input_type}_activity"] = activity.astype(np.uint8)
        corpus[series_key] = np.where(
            activity | held[input_type], legacy[series_key], np.float32(0.0)
        )
        levelled = _carries(corpus, input_type, LEVEL_COMPONENTS)
        corpus[shift_key] = np.where(levelled[:, None, :], level[None, :, None], np.float32(0.0))

    corpus = _renormalized(corpus)

    corpus["diagnostics"] = dict(legacy["diagnostics"])
    corpus["diagnostics"]["trajectory"] = {
        "components": list(TRAJECTORY_COMPONENTS),
        "inclusion_probs": {
            input_type: dict.fromkeys(TRAJECTORY_COMPONENTS, 0.5)
            for input_type in TRAJECTORY_INPUTS
        },
        **_recomputed_prevalence(corpus),
    }
    return corpus


def _recomputed_prevalence(corpus):
    """Prevalence and input counts recomputed here: flag means over active slots."""
    prevalence, n_inputs = {}, {}
    for input_type in TRAJECTORY_INPUTS:
        active = _active(corpus, input_type)
        flags = corpus[f"{input_type}_components"][active]
        n_inputs[input_type] = int(active.sum())
        prevalence[input_type] = dict(
            zip(TRAJECTORY_COMPONENTS, (flags.sum(axis=0) / active.sum()).tolist(), strict=True)
        )
    return {"prevalence": prevalence, "n_inputs": n_inputs}


@pytest.fixture(scope="module")
def legacy():
    """Padded treatments and covariates, and one held (nonzero) shock per task."""
    corpus = pg.sample_prior_predictive(
        pg.make_scm_prior(
            n_treatments=3,
            n_covariates=2,
            n_latent=1,
            n_time_steps=32,
            n_cells=3,
            draws_per_cell=2,
            seed=21,
            n_treatments_active_range=(2, 3),
            n_covariates_active_range=(1, 2),
            n_treatment_shocks=1,
            treatment_shock_level_range=(0.5, 1.0),
        )
    )
    assert DataGenerator.validate_corpus(corpus) == []
    return corpus


@pytest.fixture(scope="module")
def trajectory_corpus(legacy):
    return _attach_trajectory_block(legacy)


def _first(mask):
    """Index of the first True entry of ``mask``; the fixture must provide one."""
    hits = np.argwhere(mask)
    assert len(hits), "the fixture lacks the slot this case needs"
    return tuple(int(i) for i in hits[0])


def _replace(corpus, key, index, value):
    """Copy of ``corpus`` whose ``key`` array, itself copied, holds ``value`` at ``index``."""
    broken = dict(corpus)
    broken[key] = corpus[key].copy()
    broken[key][index] = value
    return broken


def _edit_diagnostics(corpus, path, edit):
    """Copy of ``corpus`` with ``edit(old)`` (or a deletion) at ``path`` of copied diagnostics."""
    broken = dict(corpus)
    broken["diagnostics"] = copy.deepcopy(corpus["diagnostics"])
    parent = broken["diagnostics"]
    for key in path[:-1]:
        parent = parent[key]
    if edit is _DELETE:
        del parent[path[-1]]
    else:
        parent[path[-1]] = edit(parent.get(path[-1]))
    return broken


def _input(input_type, *, active=True, carrying=(), lacking=()):
    """Finder of the first (task, input) slot with this activity and these stored flags."""

    def find(corpus):
        mask = _active(corpus, input_type) == active
        if carrying:
            mask &= _carries(corpus, input_type, carrying)
        if lacking:
            mask &= ~_carries(corpus, input_type, lacking)
        return _first(mask)

    return find


def _flag(find, component="hf"):
    """Index of ``component``'s flag at the slot ``find`` selects."""
    return lambda corpus: (*find(corpus), TRAJECTORY_COMPONENTS.index(component))


def _cell_flag(find, component="hf"):
    """Index of ``component``'s flag at the selected input in every task of its cell."""

    def index(corpus):
        task, slot = find(corpus)
        tasks = np.flatnonzero(corpus["cell_id"] == corpus["cell_id"][task])
        assert len(tasks) > 1
        return tasks, slot, TRAJECTORY_COMPONENTS.index(component)

    return index


def _week(find, week=0):
    """Index of ``week`` of the slot ``find`` selects in a (task, time, input) array."""

    def index(corpus):
        task, slot = find(corpus)
        return task, week, slot

    return index


GATED_TREATMENT = _input("treatment", carrying=GATE_COMPONENTS)
GATED_COVARIATE = _input("covariate", carrying=GATE_COMPONENTS)
PADDED_TREATMENT = _input("treatment", active=False)
PADDED_COVARIATE = _input("covariate", active=False)


def _covariate_on_week(corpus):
    """A nonzero on-week of the first gated active covariate."""
    task, slot = GATED_COVARIATE(corpus)
    on = (corpus["covariate_activity"][task, :, slot] == 1) & (
        corpus["covariates"][task, :, slot] != 0
    )
    return task, int(np.flatnonzero(on)[0]), slot


def _shocked_gated_treatment(corpus):
    """First active direct treatment that carries a gate form and a held shock."""
    return _first(
        _active(corpus, "treatment")
        & _direct(corpus)
        & _carries(corpus, "treatment", GATE_COMPONENTS)
        & corpus["treatment_shock_mask"].any(axis=1)
    )


def _zeroable_gated(corpus, input_type):
    """First gated active input whose series no other rule reads (no C->Y edge for treatments)."""
    mask = _active(corpus, input_type) & _carries(corpus, input_type, GATE_COMPONENTS)
    if input_type == "treatment":
        mask &= ~_direct(corpus)
    return _first(mask)


def _switch_off(corpus, input_type, index):
    """Copy of ``corpus`` with an input switched off at ``index`` and its series zeroed there."""
    series_key = "treatment_raw" if input_type == "treatment" else "covariates"
    edited = _replace(corpus, f"{input_type}_activity", index, 0)
    return _renormalized(_replace(edited, series_key, index, 0.0))


REJECTED_ARRAYS = (
    pytest.param(
        "treatment_components",
        _flag(_input("treatment", carrying=("hf",))),
        2,
        "treatment_components is not binary",
        id="treatment-flag-not-binary",
    ),
    pytest.param(
        "covariate_components",
        _flag(_input("covariate", carrying=("hf",))),
        2,
        "covariate_components is not binary",
        id="covariate-flag-not-binary",
    ),
    pytest.param(
        "treatment_activity",
        _week(GATED_TREATMENT),
        2,
        "treatment_activity is not binary",
        id="treatment-activity-not-binary",
    ),
    pytest.param(
        "covariate_activity",
        _week(GATED_COVARIATE),
        2,
        "covariate_activity is not binary",
        id="covariate-activity-not-binary",
    ),
    pytest.param(
        "treatment_components",
        _flag(PADDED_TREATMENT),
        1,
        "treatment_components has nonzero inactive-treatment padding",
        id="treatment-flag-padding",
    ),
    pytest.param(
        "covariate_components",
        _flag(PADDED_COVARIATE),
        1,
        "covariate_components has nonzero inactive-covariate padding",
        id="covariate-flag-padding",
    ),
    pytest.param(
        "treatment_activity",
        _week(PADDED_TREATMENT),
        1,
        "treatment_activity has nonzero inactive-treatment padding",
        id="treatment-activity-padding",
    ),
    pytest.param(
        "covariate_activity",
        _week(PADDED_COVARIATE),
        1,
        "covariate_activity has nonzero inactive-covariate padding",
        id="covariate-activity-padding",
    ),
    pytest.param(
        "treatment_log_level_shift",
        _week(PADDED_TREATMENT),
        0.5,
        "treatment_log_level_shift has nonzero inactive-treatment padding",
        id="treatment-shift-padding",
    ),
    pytest.param(
        "covariate_level_shift",
        _week(PADDED_COVARIATE),
        0.5,
        "covariate_level_shift has nonzero inactive-covariate padding",
        id="covariate-shift-padding",
    ),
    pytest.param(
        "treatment_components",
        _flag(_input("treatment", lacking=("hf",))),
        1,
        "treatment_components differs within a cell",
        id="treatment-flags-vary-within-cell",
    ),
    pytest.param(
        "covariate_components",
        _flag(_input("covariate", lacking=("hf",))),
        1,
        "covariate_components differs within a cell",
        id="covariate-flags-vary-within-cell",
    ),
    pytest.param(
        "treatment_activity",
        _week(_input("treatment", lacking=GATE_COMPONENTS), week=5),
        0,
        "treatment_activity must be 1 for active treatments without a gate component",
        id="ungated-treatment-off",
    ),
    pytest.param(
        "covariate_activity",
        _week(_input("covariate", lacking=GATE_COMPONENTS), week=5),
        0,
        "covariate_activity must be 1 for active covariates without a gate component",
        id="ungated-covariate-off",
    ),
    pytest.param(
        "treatment_log_level_shift",
        _week(_input("treatment", lacking=LEVEL_COMPONENTS), week=5),
        0.5,
        "treatment_log_level_shift must be 0 for active treatments without a level component",
        id="unlevelled-treatment-shifted",
    ),
    pytest.param(
        "covariate_level_shift",
        _week(_input("covariate", lacking=LEVEL_COMPONENTS), week=5),
        0.5,
        "covariate_level_shift must be 0 for active covariates without a level component",
        id="unlevelled-covariate-shifted",
    ),
    pytest.param(
        "covariate_activity",
        _covariate_on_week,
        0,
        "covariates must be 0 on covariate_activity off-weeks",
        id="covariate-nonzero-off-week",
    ),
    pytest.param(
        "treatment_activity",
        _week(_input("treatment", carrying=("onset",))),
        1,
        "treatment_activity must be 0 at week 0 for an active treatment with an onset component",
        id="treatment-onset-on-at-week-0",
    ),
    pytest.param(
        "covariate_activity",
        _week(_input("covariate", carrying=("onset",))),
        1,
        "covariate_activity must be 0 at week 0 for an active covariate with an onset component",
        id="covariate-onset-on-at-week-0",
    ),
    pytest.param(
        "treatment_activity",
        _week(_input("treatment", carrying=("offset",)), week=-1),
        1,
        "treatment_activity must be 0 at the last week for an active treatment "
        "with an offset component",
        id="treatment-offset-on-at-last-week",
    ),
    pytest.param(
        "covariate_activity",
        _week(_input("covariate", carrying=("offset",)), week=-1),
        1,
        "covariate_activity must be 0 at the last week for an active covariate "
        "with an offset component",
        id="covariate-offset-on-at-last-week",
    ),
    # hf has no array-level consequence, so the flip leaves only the stored
    # prevalence stale; flipping the whole cell keeps the flags cell-constant.
    pytest.param(
        "treatment_components",
        _cell_flag(_input("treatment", lacking=("hf",))),
        1,
        PREVALENCE,
        id="prevalence-stale-after-flag-flip",
    ),
)

REJECTED_DIAGNOSTICS = (
    pytest.param(
        ("trajectory",),
        lambda _: [],
        "diagnostics trajectory must be a mapping",
        id="block-not-a-mapping",
    ),
    pytest.param(("trajectory", "components"), lambda old: old[::-1], LAYOUT, id="layout-reversed"),
    pytest.param(("trajectory", "components"), lambda old: old[:-1], LAYOUT, id="layout-short"),
    pytest.param(("trajectory", "components"), _DELETE, LAYOUT, id="layout-missing"),
    pytest.param(
        ("trajectory", "inclusion_probs", "treatment", "hf"),
        lambda _: 1.5,
        INCLUSION,
        id="prob-above-one",
    ),
    pytest.param(
        ("trajectory", "inclusion_probs", "covariate", "trend"),
        lambda _: -0.25,
        INCLUSION,
        id="prob-negative",
    ),
    pytest.param(
        ("trajectory", "inclusion_probs", "treatment", "onset"),
        lambda _: float("nan"),
        INCLUSION,
        id="prob-nan",
    ),
    pytest.param(
        ("trajectory", "inclusion_probs", "treatment", "pulse"),
        lambda _: True,
        INCLUSION,
        id="prob-bool",
    ),
    pytest.param(
        ("trajectory", "inclusion_probs", "covariate", "hf"),
        lambda _: "0.5",
        INCLUSION,
        id="prob-string",
    ),
    pytest.param(
        ("trajectory", "inclusion_probs", "covariate"), _DELETE, INCLUSION, id="input-missing"
    ),
    pytest.param(
        ("trajectory", "inclusion_probs", "latent"),
        lambda _: dict.fromkeys(TRAJECTORY_COMPONENTS, 0.5),
        INCLUSION,
        id="input-extra",
    ),
    pytest.param(
        ("trajectory", "inclusion_probs", "treatment", "trend"),
        _DELETE,
        INCLUSION,
        id="component-missing",
    ),
    pytest.param(
        ("trajectory", "inclusion_probs", "treatment", "spike"),
        lambda _: 0.5,
        INCLUSION,
        id="component-extra",
    ),
    pytest.param(
        ("trajectory", "inclusion_probs"),
        lambda old: list(old.values()),
        INCLUSION,
        id="probs-not-a-mapping",
    ),
    pytest.param(
        ("trajectory", "n_inputs", "covariate"),
        lambda old: old + 1,
        "diagnostics trajectory n_inputs does not match recomputation",
        id="n_inputs-stale",
    ),
    # The fixture carries every component on some, but not all, active inputs.
    pytest.param(
        ("trajectory", "inclusion_probs", "treatment", "onset"),
        lambda _: 0.0,
        CONTRADICTION.format("treatment", "onset"),
        id="treatment-zero-prob-but-carried",
    ),
    pytest.param(
        ("trajectory", "inclusion_probs", "covariate", "trend"),
        lambda _: 0.0,
        CONTRADICTION.format("covariate", "trend"),
        id="covariate-zero-prob-but-carried",
    ),
    pytest.param(
        ("trajectory", "inclusion_probs", "treatment", "hf"),
        lambda _: 1.0,
        CONTRADICTION.format("treatment", "hf"),
        id="treatment-unit-prob-but-not-carried-by-all",
    ),
    pytest.param(
        ("trajectory", "inclusion_probs", "covariate", "hf"),
        lambda _: 1.0,
        CONTRADICTION.format("covariate", "hf"),
        id="covariate-unit-prob-but-not-carried-by-all",
    ),
    pytest.param(("trajectory", "prevalence"), _DELETE, PREVALENCE, id="prevalence-missing"),
)


def test_consistent_trajectory_block_validates(trajectory_corpus):
    assert DataGenerator.validate_corpus(trajectory_corpus) == []


def test_trajectory_block_validates_after_save_and_load(tmp_path, trajectory_corpus):
    path = tmp_path / "trajectory.npz"
    pg.save_corpus(trajectory_corpus, path)
    loaded = pg.load_corpus(path)
    assert loaded["diagnostics"]["trajectory"] == trajectory_corpus["diagnostics"]["trajectory"]
    assert DataGenerator.validate_corpus(loaded) == []


@pytest.mark.parametrize("part", (*TRAJECTORY_ARRAY_FIELDS, "diagnostics trajectory"))
def test_trajectory_block_is_all_or_nothing(trajectory_corpus, part):
    if part in TRAJECTORY_ARRAY_FIELDS:
        broken = {key: value for key, value in trajectory_corpus.items() if key != part}
    else:
        broken = _edit_diagnostics(trajectory_corpus, ("trajectory",), _DELETE)
    assert DataGenerator.validate_corpus(broken) == [
        f"trajectory arrays and diagnostics trajectory must be present together; missing {part}"
    ]


@pytest.mark.parametrize("key", TRAJECTORY_ARRAY_FIELDS)
def test_trajectory_arrays_follow_their_schema(trajectory_corpus, key):
    stored = trajectory_corpus[key]
    errors = DataGenerator.validate_corpus({**trajectory_corpus, key: stored[..., :-1]})
    assert any(error.startswith(f"Shape mismatch for {key}:") for error in errors), errors

    wrong = np.dtype(np.int8 if stored.dtype == np.uint8 else np.float64)
    errors = DataGenerator.validate_corpus({**trajectory_corpus, key: stored.astype(wrong)})
    assert f"{key} has dtype {wrong}, expected {stored.dtype}" in errors


@pytest.mark.parametrize(("key", "index", "value", "expected"), REJECTED_ARRAYS)
def test_validate_corpus_rejects_inconsistent_trajectory_arrays(
    trajectory_corpus, key, index, value, expected
):
    broken = _replace(trajectory_corpus, key, index(trajectory_corpus), value)
    assert expected in DataGenerator.validate_corpus(broken)


@pytest.mark.parametrize(("path", "edit", "expected"), REJECTED_DIAGNOSTICS)
def test_validate_corpus_rejects_malformed_trajectory_diagnostics(
    trajectory_corpus, path, edit, expected
):
    broken = _edit_diagnostics(trajectory_corpus, path, edit)
    assert expected in DataGenerator.validate_corpus(broken)


def test_held_shock_overrides_a_treatment_off_week(trajectory_corpus):
    """The accepted fixture holds a gated direct treatment off exactly under its shock."""
    task, slot = _shocked_gated_treatment(trajectory_corpus)
    shocked = trajectory_corpus["treatment_shock_mask"][task, :, slot] == 1
    off = trajectory_corpus["treatment_activity"][task, :, slot] == 0
    held = trajectory_corpus["treatment_raw"][task, :, slot]
    assert np.array_equal(off, shocked) and (held[shocked] > 0).all()

    week = int(np.flatnonzero(~shocked & (held > 0))[0])
    off_outside_shock = _replace(trajectory_corpus, "treatment_activity", (task, week, slot), 0)
    assert DataGenerator.validate_corpus(off_outside_shock) == [OFF_WEEK_TREATMENT]


@pytest.mark.parametrize("input_type", TRAJECTORY_INPUTS)
def test_gate_component_must_switch_its_input_off(trajectory_corpus, input_type):
    index = _cell_flag(_input(input_type, lacking=GATE_COMPONENTS), "flighting")(trajectory_corpus)
    broken = _replace(trajectory_corpus, f"{input_type}_components", index, 1)
    tasks, slot, _ = index
    assert (broken[f"{input_type}_activity"][tasks, :, slot] == 1).all()
    broken["diagnostics"] = copy.deepcopy(trajectory_corpus["diagnostics"])
    broken["diagnostics"]["trajectory"].update(_recomputed_prevalence(broken))
    assert DataGenerator.validate_corpus(broken) == [
        f"{input_type}_activity never switches off for an active {input_type} with a gate component"
    ]


@pytest.mark.parametrize("keep", (2, 1))
@pytest.mark.parametrize("input_type", TRAJECTORY_INPUTS)
def test_gated_inputs_keep_two_on_weeks_in_the_support_window(trajectory_corpus, input_type, keep):
    task, slot = _zeroable_gated(trajectory_corpus, input_type)
    activity_key = f"{input_type}_activity"
    on = np.flatnonzero(
        (trajectory_corpus[activity_key][task, :, slot] == 1)
        & (trajectory_corpus["support_mask"][task] == 1)
    )
    assert len(on) > keep
    edited = _switch_off(trajectory_corpus, input_type, (task, on[keep:], slot))
    expected = (
        []
        if keep == 2
        else [
            f"{activity_key} keeps fewer than 2 on-weeks inside the support window "
            f"for an active {input_type} with a gate component"
        ]
    )
    assert DataGenerator.validate_corpus(edited) == expected


@pytest.mark.parametrize("probability", (0.0, 1.0))
@pytest.mark.parametrize("input_type", TRAJECTORY_INPUTS)
def test_exact_inclusion_probabilities_accept_matching_flags(
    trajectory_corpus, input_type, probability
):
    """An echoed 0 or 1 validates when no or every active input carries the component."""
    key = f"{input_type}_components"
    edited = dict(trajectory_corpus)
    edited[key] = trajectory_corpus[key].copy()
    # pulse has no array-level consequence, so only the diagnostics follow it.
    pulse = TRAJECTORY_COMPONENTS.index("pulse")
    edited[key][..., pulse] = _active(trajectory_corpus, input_type) & bool(probability)
    edited["diagnostics"] = copy.deepcopy(trajectory_corpus["diagnostics"])
    trajectory = edited["diagnostics"]["trajectory"]
    trajectory.update(_recomputed_prevalence(edited))
    trajectory["inclusion_probs"][input_type]["pulse"] = probability
    assert DataGenerator.validate_corpus(edited) == []


def test_validate_corpus_requires_nonnegative_treatments_without_the_block(legacy):
    broken = _replace(legacy, "treatment_raw", _week(_input("treatment"))(legacy), -1.0)
    assert "treatment_raw must be nonnegative" in DataGenerator.validate_corpus(broken)
