"""Structural-split designs: how a run's tasks spread over splits and how each is seeded.

The reader re-derives every task's seed from its run's request, so these seeds are part
of the run contract: a change to the derivation would make every published design run
unreadable.
"""

from __future__ import annotations

import json
from collections import Counter
from datetime import UTC, datetime, timedelta, timezone

import pytest

from live_dashboard import publication as pub
from live_dashboard import structural
from live_dashboard.deployment import check_demo_bounds
from pymc_generator.sampler import CARRYOVER_FAMILY_KEYS, SATURATION_FAMILY_KEYS

DESIGN = structural.DESIGNS["structural-splits"]
BASE_SEED = 20261015121429
NOW = datetime(2026, 10, 15, 12, 0, tzinfo=UTC)


def _request(worlds: int = 120, seed: int = BASE_SEED) -> pub.RunRequest:
    return pub.RunRequest("splits", "structural-splits", worlds, 2, seed)


def _manifest(cycle: list[str] | None = None) -> pub.RunManifest:
    request = _request(worlds=20)
    generator = pub.GeneratorInfo(
        preset=request.preset,
        config={"cycle": list(DESIGN.cycle) if cycle is None else cycle, "presets": {}},
        pymc_generator_version="0.0.2",
        corpus_schema_version=5,
        code_sha256="ab" * 32,
        code_revision=None,
        environment={"python": "3.13.9"},
    )
    return pub.RunManifest(
        run_id=request.run_id, created_at=NOW, request=request, generator=generator
    )


def test_the_design_covers_every_saturation_but_root_with_both_adstocks():
    pairs = {(split.saturation, split.adstock) for split in DESIGN.splits}
    assert pairs == {
        (saturation, adstock)
        for saturation in SATURATION_FAMILY_KEYS
        if saturation != "root"
        for adstock in CARRYOVER_FAMILY_KEYS
        if adstock != "weibull"
    }
    assert len(pairs) == len(DESIGN.splits) == 10


def test_the_design_order_is_pinned():
    # The reader maps task i to split i % 10 and regime (i // 10) % 5 from this order:
    # reordering either tuple would make every published design run unreadable.
    assert [split.name for split in DESIGN.splits] == [
        "linear-none",
        "linear-geometric",
        "hill-none",
        "hill-geometric",
        "logistic-none",
        "logistic-geometric",
        "michaelis-menten-none",
        "michaelis-menten-geometric",
        "tanh-none",
        "tanh-geometric",
    ]
    assert DESIGN.regimes == ("calm", "hectic", "mixed", "spiky-inputs", "hectic-baseline")
    # Readers match the recorded rule word for word and cap the recorded redraws.
    assert structural.SEED_RULE == (
        "split seed + the task's index within its split, split seed = first 31 bits of "
        "sha256('<base seed>:<saturation>:<adstock>'); the k-th redraw adds k * 2**32"
    )
    assert structural.MAX_REDRAWS == 5


def test_a_design_needs_two_or_more_members():
    # A run records a design as a mix of its members; a single member has no cycle.
    with pytest.raises(ValueError, match="two or more members"):
        structural.SplitDesign(splits=(DESIGN.splits[0],), regimes=("calm",))
    structural.SplitDesign(splits=(DESIGN.splits[0],), regimes=("calm", "hectic"))


def test_task_seeds_are_pinned_to_the_base_seed_and_families():
    # Published runs re-derive their task seeds from these values.
    request = _request(worlds=40)
    seeds = [request.task_seed(index) for index in (0, 1, 9, 10, 13, 19)]
    assert seeds == [57402002, 303994010, 99099249, 57402003, 2075023208, 99099250]
    split_seeds = [split.seed(BASE_SEED) for split in DESIGN.splits]
    assert len(set(split_seeds)) == 10 and all(0 <= seed < 2**31 for seed in split_seeds)
    later = {split.seed(BASE_SEED + 1) for split in DESIGN.splits}
    assert not set(split_seeds) & later


def test_the_launched_run_gives_every_split_and_regime_the_same_worlds():
    request = _request(worlds=1000)  # the launch: 500 tasks of 2 worlds
    check_demo_bounds(request)
    contexts = [request.seed_context(index) for index in range(request.n_tasks)]
    per_split = Counter((c["split"]["saturation"], c["split"]["adstock"]) for c in contexts)
    assert len(per_split) == 10 and set(per_split.values()) == {50}  # 100 worlds each
    per_member = Counter(
        (c["split"]["saturation"], c["split"]["adstock"], c["regime"]) for c in contexts
    )
    assert len(per_member) == 50 and set(per_member.values()) == {10}  # 20 worlds each
    # Any 10 consecutive tasks visit every split once, so the splits fill evenly.
    for start in range(request.n_tasks - 9):
        window = {
            (c["split"]["saturation"], c["split"]["adstock"]) for c in contexts[start : start + 10]
        }
        assert len(window) == 10
    # The member preset of a task (the cycle the worker runs) is its split and regime.
    cycle = DESIGN.cycle
    for index, context in enumerate(contexts):
        split = structural.StructuralSplit(
            context["split"]["saturation"], context["split"]["adstock"]
        )
        assert cycle[index % len(cycle)] == structural.member_name(split, context["regime"])
    seeds = [request.task_seed(index) for index in range(request.n_tasks)]
    assert len(set(seeds)) == len(seeds)


def test_redraw_seeds_never_repeat_a_seed_of_the_run():
    request = _request(worlds=2000)
    first = [request.task_seed(index) for index in range(request.n_tasks)]
    redraws = [
        structural.redraw_seed(seed, redraw)
        for seed in first
        for redraw in range(1, structural.MAX_REDRAWS + 1)
    ]
    assert len(set(first) | set(redraws)) == len(first) + len(redraws)
    assert max(redraws) <= pub.MAX_JSON_INT


@pytest.mark.parametrize("worlds", [2, 22, 110])
def test_a_design_run_whose_tasks_do_not_divide_over_its_splits_is_refused(worlds):
    with pytest.raises(ValueError, match="divide evenly"):
        _request(worlds=worlds)


def test_splits_whose_seed_ranges_overlap_are_refused():
    # Base seed 1357 puts tanh-none and tanh-geometric 573 seeds apart: 573 tasks per split
    # fit, 574 would give two tasks the same seed.
    accepted = pub.RunRequest("splits", "structural-splits", 2 * 5730, 2, 1357)
    assert len({accepted.task_seed(index) for index in range(accepted.n_tasks)}) == 5730
    with pytest.raises(ValueError, match="overlap"):
        pub.RunRequest("splits", "structural-splits", 2 * 5740, 2, 1357)


def test_other_presets_keep_seed_plus_task_index():
    request = pub.RunRequest("demo", "small", 6, 2, 41)
    assert [request.task_seed(index) for index in range(3)] == [41, 42, 43]
    assert request.seed_context(2) == {
        "rule": pub.SEED_RULE,
        "base_seed": 41,
        "task_index": 2,
        "task_seed": 43,
    }


def test_a_design_run_records_its_seed_rule_and_every_split_seed():
    manifest = _manifest()
    data = manifest.encode()
    body = json.loads(data)
    assert body["seed_rule"] == structural.SEED_RULE
    assert body["structural_splits"] == [
        {"saturation": s.saturation, "adstock": s.adstock, "seed": s.seed(BASE_SEED)}
        for s in DESIGN.splits
    ]
    decoded = pub.decode_run_manifest(data, run_id=manifest.run_id)
    assert decoded.request.task_seed(3) == manifest.request.task_seed(3)


def _edited(data: bytes, edit) -> bytes:
    body = json.loads(data)
    edit(body)
    return json.dumps(body).encode()


@pytest.mark.parametrize(
    "edit",
    [
        lambda body: body["structural_splits"][0].update(seed=12345),
        lambda body: body["structural_splits"].pop(),
        lambda body: body.pop("structural_splits"),
        lambda body: body.update(seed_rule=pub.SEED_RULE),
    ],
    ids=["split-seed", "split-missing", "no-splits", "seed-rule"],
)
def test_a_design_run_whose_recorded_seeds_differ_from_the_design_is_refused(edit):
    manifest = _manifest()
    with pytest.raises(pub.RecordError):
        pub.decode_run_manifest(_edited(manifest.encode(), edit), run_id=manifest.run_id)


def test_a_design_run_recorded_with_another_member_cycle_is_refused():
    manifest = _manifest(cycle=list(reversed(DESIGN.cycle)))
    with pytest.raises(pub.RecordError, match="cycle"):
        pub.decode_run_manifest(manifest.encode(), run_id=manifest.run_id)


def test_the_launch_seed_is_the_utc_launch_time_in_digits():
    moment = datetime(2026, 10, 15, 12, 14, 29, 999999, tzinfo=UTC)
    assert structural.launch_seed(moment) == 20261015121429
    elsewhere = moment.astimezone(timezone(timedelta(hours=-5)))
    assert structural.launch_seed(elsewhere) == 20261015121429
    assert _request(seed=structural.launch_seed(moment)).seed == 20261015121429
