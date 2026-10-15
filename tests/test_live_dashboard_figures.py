"""Cards, figures and status of one run, checked on hand-built snapshots."""

from __future__ import annotations

import importlib.util
import json
import math
import sys
from datetime import timedelta
from pathlib import Path
from types import ModuleType
from typing import Any

import numpy as np
import pytest

from live_dashboard import figures
from live_dashboard.snapshot import (
    AttemptView,
    ControlUnit,
    ImpactUnit,
    RunSnapshot,
    TaskState,
    WorldMetrics,
)
from pymc_generator.sampler import CARRYOVER_FAMILY_KEYS, SATURATION_FAMILY_KEYS


def _load(name: str) -> ModuleType:
    module = sys.modules.get(name)
    if module is None:
        spec = importlib.util.spec_from_file_location(name, Path(__file__).with_name(f"{name}.py"))
        assert spec is not None and spec.loader is not None
        module = sys.modules.setdefault(name, importlib.util.module_from_spec(spec))
        spec.loader.exec_module(module)
    return module


support = _load("_live_dashboard_support")

RUN_ID = "demo-0123456789ab"
ATTEMPT = "20260101T000000Z-0000abcd"
PANELS = (
    "speed",
    "cumulative",
    "impact",
    "control_impact",
    "baseline",
    "composition",
    "saturation",
    "adstock",
    "combinations",
    "active_counts",
)


@pytest.fixture(autouse=True)
def _no_dashboard_environment(monkeypatch):
    for name in (
        "DASHBOARD_USERNAME",
        "DASHBOARD_PASSWORD_HASH",
        "DASHBOARD_POLL_SECONDS",
        "LIVE_DASHBOARD_CODE_REVISION",
    ):
        monkeypatch.delenv(name, raising=False)


def at(seconds: float):
    return support.EPOCH + timedelta(seconds=seconds)


def iso(seconds: int) -> str:
    return f"{at(seconds):%Y-%m-%dT%H:%M:%S}.000Z"


def task(
    index: int,
    status: str,
    span: tuple[float, float] | None = None,
    *,
    accepted: int = 0,
    attempt: str | None = None,
    **fields: Any,
) -> TaskState:
    return TaskState(
        task_id=f"task-{index:05d}",
        task_index=index,
        status=status,  # type: ignore[arg-type]
        target_worlds=2,
        accepted_worlds=accepted,
        started_at=None if span is None else at(span[0]),
        finished_at=None if span is None else at(span[1]),
        attempt_id=attempt,
        **fields,
    )


def published(index: int, span: tuple[float, float], *, accepted: int = 2, **fields) -> TaskState:
    fields = {"rejected_candidates": 0, "draw_failures": 0, **fields}
    return task(index, "published", span, accepted=accepted, attempt=ATTEMPT, **fields)


def zero(index: int, span: tuple[float, float], **fields: Any) -> TaskState:
    return task(index, "zero", span, attempt=ATTEMPT, **fields)


def attempt(status: str = "running", **fields: Any) -> AttemptView:
    values = {
        "deadline_at": None,
        "current_task_index": None,
        "error": None,
        "gpu_requested": None,
        "gpu_name": None,
        **fields,
    }
    return AttemptView(
        attempt_id=ATTEMPT,
        status=status,  # type: ignore[arg-type]
        started_at=support.EPOCH,
        updated_at=support.EPOCH,
        **values,
    )


def world(
    row: int,
    channel: float,
    control: float,
    baseline: float,
    adstock: tuple[str, ...] = ("geometric",),
    saturation: tuple[str, ...] | None = ("michaelis_menten",),
    active: tuple[int, int] = (3, 2),
    slots: tuple[int, int] = (10, 10),
) -> WorldMetrics:
    return WorldMetrics(
        f"{RUN_ID}/task-00000/w{row}",
        "task-00000",
        channel,
        control,
        baseline,
        adstock,
        saturation,
        *active,
        *slots,
    )


def impact(row: int, channel: str, share: float) -> ImpactUnit:
    return ImpactUnit(f"{RUN_ID}/task-00000/w{row}", "task-00000", channel, share)


def run(
    tasks: list[TaskState],
    *,
    state: str = "running",
    worlds: tuple[WorldMetrics, ...] = (),
    impacts: tuple[ImpactUnit, ...] = (),
    control_impacts: tuple[ControlUnit, ...] = (),
    attempts: tuple[AttemptView, ...] = (),
    issues: tuple[str, ...] = (),
) -> RunSnapshot:
    return RunSnapshot(
        run_id=RUN_ID,
        label="demo",
        preset="small",
        created_at=support.EPOCH,
        state=state,  # type: ignore[arg-type]
        n_tasks=len(tasks),
        worlds_per_task=2,
        target_worlds=2 * len(tasks),
        seed=1,
        code_revision=None,
        tasks=tuple(tasks),
        worlds=worlds,
        impacts=impacts,
        attempts=attempts,
        issues=issues,
        control_impacts=control_impacts,
    )


def card(snapshot: RunSnapshot, card_id: str) -> dict[str, Any]:
    (found,) = [item for item in figures.cards(snapshot) if item["id"] == card_id]
    return found


def panel(snapshot: RunSnapshot, panel_id: str) -> dict[str, Any]:
    (found,) = [item for item in figures.run_body(snapshot)["figures"] if item["id"] == panel_id]
    return found


def trace(found_panel: dict[str, Any], name: str) -> dict[str, Any]:
    (found,) = [item for item in found_panel["figure"]["data"] if item.get("name") == name]
    return found


# A run with an idle gap (90 s -> 300 s), overlapping tasks (0-60 and 30-90), a
# zero-length task and two tasks finishing at the same instant (360 s). Invalid and
# unavailable tasks spent generation time but add no worlds; conflicts are not timed.
# Active time: [0, 90] + [300, 360] + [400, 420] + [480, 510] = 200 s.
GAPPED = [
    published(0, (0, 60), generation_seconds=50.0),
    published(1, (30, 90)),
    zero(2, (300, 360), generation_seconds=60.0),
    published(3, (360, 360)),
    task(4, "invalid", (400, 420), accepted=2, attempt=ATTEMPT),
    task(5, "conflict", (0, 1000)),
    task(6, "unavailable", (480, 510), accepted=2, attempt=ATTEMPT, generation_seconds=25.0),
]


def test_cards_show_published_worlds_and_speed_over_active_minutes():
    snapshot = run(
        [
            published(0, (0, 60), generation_seconds=30.0),
            published(1, (60, 120), generation_seconds=40.0),
            task(2, "pending"),
        ]
    )
    assert card(snapshot, "worlds") == {
        "id": "worlds",
        "title": "Worlds generated",
        "value": 4,
        "text": "4 / 6",
        "subtitle": "2/3 tasks recorded",
    }
    speed = card(snapshot, "speed")
    # 4 worlds in 2 active minutes; the latest task reports 40 s for its 2 worlds.
    assert speed["value"] == pytest.approx(2.0)
    assert speed["text"] == "2.0 worlds/min"
    assert speed["subtitle"] == "latest task: 3.0 worlds/min"


def test_speed_card_is_a_dash_without_positive_active_time():
    nothing_yet = card(run([task(0, "pending")], state="waiting"), "speed")
    assert (nothing_yet["value"], nothing_yet["text"]) == (None, "—")
    assert nothing_yet["subtitle"] == "no task published yet"

    instantaneous = run([published(0, (30, 30))])
    speed = card(instantaneous, "speed")
    assert (speed["value"], speed["text"], speed["subtitle"]) == (None, "—", "latest task: —")
    assert card(instantaneous, "worlds")["text"] == "2 / 2"


def test_speed_figure_divides_by_the_union_of_task_intervals():
    snapshot = run(GAPPED)
    speed = panel(snapshot, "speed")
    assert speed["empty"] is None
    line = trace(speed, "cumulative (active time)")
    assert line["x"] == [iso(60), iso(90), iso(360), iso(420), iso(510)]
    assert line["y"] == pytest.approx([2 / 1.0, 4 / 1.5, 6 / 2.5, 6 / (170 / 60), 6 / (200 / 60)])
    markers = trace(speed, "per task")
    # Rates use reported generation seconds, else wall time; zero-length tasks are skipped
    # and invalid or unavailable tasks have no rate of their own.
    assert markers["x"] == [iso(60), iso(90), iso(360)]
    assert markers["y"] == pytest.approx([2 / 50 * 60, 2.0, 0.0])
    assert card(snapshot, "speed")["value"] == pytest.approx(1.8)  # 6 worlds / (200 s)
    assert card(snapshot, "worlds")["text"] == "6 / 14"
    worlds = trace(panel(snapshot, "cumulative"), "published worlds")
    assert worlds["x"] == [iso(0), iso(60), iso(90), iso(360)]
    assert worlds["y"] == [0, 2, 4, 6]
    assert trace(panel(snapshot, "cumulative"), "target (14)")["x"] == [iso(0), iso(510)]


def test_active_time_merges_intervals_bridged_by_a_later_task():
    snapshot = run(
        [published(0, (0, 10)), published(1, (20, 30)), published(2, (5, 40))],
    )
    line = trace(panel(snapshot, "speed"), "cumulative (active time)")
    assert line["y"] == pytest.approx([2 / (10 / 60), 4 / (20 / 60), 6 / (40 / 60)])
    assert card(snapshot, "speed")["value"] == pytest.approx(9.0)


def test_cumulative_rate_is_null_until_active_time_is_positive():
    snapshot = run([published(0, (10, 10)), published(1, (10, 70))])
    line = trace(panel(snapshot, "speed"), "cumulative (active time)")
    assert line["x"] == [iso(10), iso(70)]
    assert line["y"][0] is None
    assert line["y"][1] == pytest.approx(4.0)


def test_cumulative_figure_steps_through_published_completions():
    snapshot = run(
        [
            published(0, (0, 60)),
            published(1, (30, 60)),
            zero(2, (60, 120)),
            published(3, (200, 300), accepted=1),
        ]
    )
    cumulative = panel(snapshot, "cumulative")
    worlds = trace(cumulative, "published worlds")
    assert worlds["x"] == [iso(0), iso(60), iso(300)]
    assert worlds["y"] == [0, 4, 5]
    assert worlds["line"]["shape"] == "hv"
    target = trace(cumulative, "target (8)")
    assert target["x"] == [iso(0), iso(300)]
    assert target["y"] == [8, 8]
    assert target["line"]["dash"] == "dash"


def test_impact_figure_shows_channel_units_and_world_totals_as_percentages():
    snapshot = run(
        [published(0, (0, 60))],
        worlds=(world(0, 0.5, 0.3, 0.2), world(1, 0.25, 0.5, 0.25)),
        impacts=(
            impact(0, "C1", 0.3),
            impact(0, "C2", 0.15),
            impact(1, "C1", 0.2),
            impact(1, "C3", math.nan),
        ),
    )
    found = panel(snapshot, "impact")
    units = trace(found, "Each active channel")
    assert units["y"] == [0.3, 0.15, 0.2]
    assert units["text"] == [
        f"{RUN_ID}/task-00000/w0 C1",
        f"{RUN_ID}/task-00000/w0 C2",
        f"{RUN_ID}/task-00000/w1 C1",
    ]
    totals = trace(found, "All channels per world")
    assert totals["y"] == [0.5, 0.25]
    assert totals["text"] == [f"{RUN_ID}/task-00000/w0", f"{RUN_ID}/task-00000/w1"]
    for violin in (units, totals):
        assert violin["type"] == "violin"
        assert violin["points"] == "all"
        assert violin["marker"]["opacity"] == pytest.approx(0.4)
        assert violin["box"]["visible"] is True
    assert found["figure"]["layout"]["yaxis"]["tickformat"] == ".0%"
    # Every finite share lies in [0, 1] (the NaN is dropped), so the axis stays 0-100 %.
    assert found["figure"]["layout"]["yaxis"]["range"] == pytest.approx([-0.02, 1.02])
    assert found["note"] == "1 non-finite share not shown"


def control(row: int, label: str, share: float) -> ControlUnit:
    return ControlUnit(f"{RUN_ID}/task-00000/w{row}", "task-00000", label, share)


def test_control_and_baseline_panels_show_the_same_picture_on_a_shared_axis():
    snapshot = run(
        [published(0, (0, 60))],
        worlds=(world(0, 0.5, 0.3, 0.2), world(1, 0.25, -0.1, 0.85)),
        control_impacts=(control(0, "Z1", 0.2), control(0, "Z2", 0.1), control(1, "Z1", -0.1)),
    )
    controls = panel(snapshot, "control_impact")
    units = trace(controls, "Each active control")
    assert units["y"] == [0.2, 0.1, -0.1]
    assert units["text"][:2] == [f"{RUN_ID}/task-00000/w0 Z1", f"{RUN_ID}/task-00000/w0 Z2"]
    assert trace(controls, "All controls per world")["y"] == [0.3, -0.1]
    baseline = panel(snapshot, "baseline")
    assert trace(baseline, "Baseline per world")["y"] == [0.2, 0.85]
    # Channels, controls and baseline keep one 0-100 % axis: the negative control share
    # widens all three panels, not only the control panel.
    ranges = {
        tuple(panel(snapshot, panel_id)["figure"]["layout"]["yaxis"]["range"])
        for panel_id in ("impact", "control_impact", "baseline")
    }
    assert len(ranges) == 1
    ((low, high),) = ranges
    assert low < -0.1 and high > 1.0


def test_composition_triangle_places_corners_and_crosses_negative_parts():
    height = math.sqrt(3) / 2
    snapshot = run(
        [published(0, (0, 60), accepted=2)],
        worlds=(
            world(0, 1.0, 0.0, 0.0),
            world(1, 0.0, 1.0, 0.0),
            world(2, 0.0, 0.0, 1.0),
            world(3, 0.2, 0.3, 0.5),
            world(4, 1.2, -0.3, 0.1),
            world(5, math.nan, 0.5, 0.5),
        ),
    )
    found = panel(snapshot, "composition")
    circles = trace(found, "worlds (4)")
    assert circles["marker"]["symbol"] == "circle"
    assert circles["marker"]["opacity"] == pytest.approx(0.35)
    assert circles["x"] == pytest.approx([0.0, 1.0, 0.5, 0.55])
    assert circles["y"] == pytest.approx([0.0, 0.0, height, 0.5 * height])
    assert circles["customdata"][3] == pytest.approx([0.2, 0.3, 0.5])
    crosses = trace(found, "worlds with a negative part (1)")
    assert crosses["marker"]["symbol"] == "x"
    assert crosses["x"] == pytest.approx([-0.25])
    assert crosses["y"] == pytest.approx([0.1 * height])
    assert crosses["text"] == [f"{RUN_ID}/task-00000/w4"]
    assert found["note"] == "1 world with non-finite shares not shown"
    layout = found["figure"]["layout"]
    assert layout["xaxis"]["visible"] is False
    assert layout["yaxis"]["visible"] is False
    assert layout["yaxis"]["scaleanchor"] == "x"
    corners = {
        (round(note["x"], 6), round(note["y"], 6)): note["text"] for note in layout["annotations"]
    }
    assert corners == {
        (0.0, 0.0): "Channels (direct + indirect)",
        (1.0, 0.0): "Controls",
        (0.5, round(height, 6)): "Baseline",
    }


def test_composition_grid_lines_stay_on_the_triangle():
    snapshot = run([published(0, (0, 60))], worlds=(world(0, 0.4, 0.3, 0.3),))
    data = panel(snapshot, "composition")["figure"]["data"]
    grid = data[0]
    points = [(x, y) for x, y in zip(grid["x"], grid["y"]) if x is not None]
    assert len(points) == 2 * 3 * 4  # two ends per line, three parts, four levels
    for x, y in points:
        baseline = y / (math.sqrt(3) / 2)
        control = x - 0.5 * baseline
        channel = 1.0 - control - baseline
        assert min(channel, control, baseline) >= -1e-12
        assert max(channel, control, baseline) <= 0.8 + 1e-12


def _bars(snapshot: RunSnapshot, panel_id: str) -> dict[str, int]:
    """A form panel's bars, keyed by the library family each one counts."""
    (bars,) = panel(snapshot, panel_id)["figure"]["data"]
    return dict(zip(bars["customdata"], bars["y"], strict=True))


def test_form_panels_count_worlds_per_family_with_every_library_family_on_x():
    worlds = (
        world(0, 0.4, 0.3, 0.3, ("geometric",), ("michaelis_menten",)),
        world(1, 0.4, 0.3, 0.3, ("geometric",), ("logistic",)),
        # A world whose channels mix families counts once in each of them.
        world(2, 0.4, 0.3, 0.3, ("none", "weibull"), ("logistic", "michaelis_menten")),
    )
    snapshot = run([published(0, (0, 60)), published(1, (60, 120), accepted=1)], worlds=worlds)
    saturation = _bars(snapshot, "saturation")
    assert list(saturation) == list(SATURATION_FAMILY_KEYS)
    assert saturation == {
        **dict.fromkeys(SATURATION_FAMILY_KEYS, 0),
        "logistic": 2,
        "michaelis_menten": 2,
    }
    adstock = _bars(snapshot, "adstock")
    assert list(adstock) == list(CARRYOVER_FAMILY_KEYS)
    assert adstock == {"none": 1, "geometric": 2, "weibull": 1}
    # The bars add up to more than the 3 worlds; the panel says why.
    for panel_id in ("saturation", "adstock"):
        assert panel(snapshot, panel_id)["note"] == (
            "1 world mixing families, counted once per family"
        )


@pytest.mark.parametrize("worlds", [1, 2, 7, 160, 400])
def test_form_panel_ticks_are_whole_worlds(worlds):
    snapshot = run(
        [published(0, (0, 60), accepted=worlds)],
        worlds=tuple(world(row, 0.4, 0.3, 0.3) for row in range(worlds)),
    )
    axis = panel(snapshot, "saturation")["figure"]["layout"]["yaxis"]
    # Integer ticks from 0, at most 8 steps up to the tallest bar.
    assert axis["tick0"] == 0 and isinstance(axis["dtick"], int) and axis["dtick"] >= 1
    assert worlds <= 8 * axis["dtick"]


def test_saturation_panel_counts_only_worlds_whose_payload_records_families():
    unrecorded = world(0, 0.4, 0.3, 0.3, saturation=None)
    snapshot = run([published(0, (0, 60), accepted=1)], worlds=(unrecorded,))
    found = panel(snapshot, "saturation")
    assert (found["figure"], found["empty"]) == (
        None,
        "this run's payloads do not record these families",
    )
    assert _bars(snapshot, "adstock")["geometric"] == 1  # the adstock family is always stored
    mixed = run([published(0, (0, 60))], worlds=(unrecorded, world(1, 0.4, 0.3, 0.3)))
    assert panel(mixed, "saturation")["note"] == "1 world without recorded families not counted"
    assert _bars(mixed, "saturation")["michaelis_menten"] == 1


def _matrix(snapshot: RunSnapshot) -> dict[tuple[int, int], int]:
    """The heatmap's counts keyed by (active channels, active controls)."""
    (cells,) = panel(snapshot, "combinations")["figure"]["data"]
    return {
        (channels, controls): count
        for channels, row in zip(cells["y"], cells["z"], strict=True)
        for controls, count in zip(cells["x"], row, strict=True)
    }


def test_the_combination_matrix_has_a_cell_for_every_allowed_pair():
    worlds = (
        world(0, 0.4, 0.3, 0.3, active=(2, 5), slots=(4, 6)),
        world(1, 0.4, 0.3, 0.3, active=(2, 5), slots=(4, 6)),
        world(2, 0.4, 0.3, 0.3, active=(4, 1), slots=(4, 6)),
    )
    snapshot = run([published(0, (0, 60)), published(1, (60, 120), accepted=1)], worlds=worlds)
    matrix = _matrix(snapshot)
    # Every pair from 1 up to the layout's slots has a cell; absent pairs count zero.
    assert sorted(matrix) == [(c, m) for c in range(1, 5) for m in range(1, 7)]
    assert {pair: n for pair, n in matrix.items() if n} == {(2, 5): 2, (4, 1): 1}
    found = panel(snapshot, "combinations")
    assert found["note"] == "2 of 24 combinations present"
    (cells,) = found["figure"]["data"]
    layout = found["figure"]["layout"]
    # Channels run up the rows and controls across the columns, in the axes and the hover.
    assert "channels" in layout["yaxis"]["title"]["text"]
    assert "controls" in layout["xaxis"]["title"]["text"]
    assert "%{y} channels" in cells["hovertemplate"]
    assert "%{x} controls" in cells["hovertemplate"]
    # Present combinations show their count; absent ones stay blank.
    for text_row, count_row in zip(cells["text"], cells["z"], strict=True):
        assert text_row == [str(count) if count else "" for count in count_row]
    colorbar = cells["colorbar"]
    assert colorbar["tick0"] == 0 and isinstance(colorbar["dtick"], int)
    assert max(matrix.values()) <= 8 * colorbar["dtick"]


def test_a_world_without_active_controls_adds_a_zero_column():
    lone = world(0, 0.6, 0.0, 0.4, active=(1, 0), slots=(2, 2))
    snapshot = run([published(0, (0, 60), accepted=1)], worlds=(lone,))
    assert sorted(_matrix(snapshot)) == [(c, m) for c in (1, 2) for m in (0, 1, 2)]


def test_active_count_bars_count_worlds_per_number_of_channels_and_controls():
    worlds = (
        world(0, 0.4, 0.3, 0.3, active=(2, 5), slots=(3, 6)),
        world(1, 0.4, 0.3, 0.3, active=(2, 1), slots=(3, 6)),
        world(2, 0.4, 0.3, 0.3, active=(3, 5), slots=(3, 6)),
    )
    snapshot = run([published(0, (0, 60)), published(1, (60, 120), accepted=1)], worlds=worlds)
    found = panel(snapshot, "active_counts")["figure"]
    bars = {
        trace["name"]: dict(zip(trace["x"], trace["y"], strict=True)) for trace in found["data"]
    }
    assert bars == {
        "channels (treatments)": {1: 0, 2: 2, 3: 1, 4: 0, 5: 0, 6: 0},
        "controls (covariates)": {1: 1, 2: 0, 3: 0, 4: 0, 5: 2, 6: 0},
    }
    for trace in found["data"]:  # each bar's hover names its own input
        assert trace["name"].split()[0] in trace["hovertemplate"]
    axis = found["layout"]["yaxis"]
    assert axis["tick0"] == 0 and isinstance(axis["dtick"], int)


WAITING = "waiting for the first publication"


@pytest.mark.parametrize(
    ("snapshot", "expected"),
    [
        pytest.param(
            run([task(0, "pending"), task(1, "pending")], state="waiting"),
            dict.fromkeys(PANELS, WAITING),
            id="waiting",
        ),
        pytest.param(
            run([task(0, "pending")], attempts=(attempt(),)),
            dict.fromkeys(PANELS, WAITING),
            id="running-before-first-record",
        ),
        pytest.param(
            run(
                [task(0, "pending")],
                state="failed",
                attempts=(attempt("failed", error="TypeError: bad config"),),
            ),
            dict.fromkeys(PANELS, "run failed: TypeError: bad config"),
            id="failed",
        ),
        pytest.param(
            run(
                [task(0, "pending")],
                state="lost",
                attempts=(attempt("lost", deadline_at=at(1200)),),
            ),
            dict.fromkeys(
                PANELS,
                f"run lost: attempt {ATTEMPT} passed its deadline {iso(1200)} while running",
            ),
            id="lost",
        ),
        pytest.param(
            run([zero(0, (0, 60)), zero(1, (60, 120))], state="complete"),
            {
                "speed": None,
                "cumulative": None,
                **dict.fromkeys(PANELS[2:], "no consumable worlds: 2 zero-world tasks"),
            },
            id="only-zero-world-tasks",
        ),
        pytest.param(
            run(
                [
                    task(0, "unavailable", accepted=2, attempt=ATTEMPT),
                    task(1, "unreadable"),
                    task(2, "pending"),
                ],
                state="stopped",
            ),
            dict.fromkeys(PANELS, "no consumable worlds: 1 unavailable, 1 unreadable"),
            id="unusable-records",
        ),
    ],
)
def test_empty_states_explain_why_a_figure_is_empty(snapshot, expected):
    body = figures.run_body(snapshot)
    shown = {item["id"]: item["empty"] for item in body["figures"]}
    assert shown == expected
    for item in body["figures"]:
        assert (item["figure"] is None) == (item["empty"] is not None)


def test_status_line_reports_every_category():
    snapshot = run(
        [
            published(0, (0, 60), rejected_candidates=2, draw_failures=0, duplicate_records=1),
            published(1, (60, 120), accepted=1, rejected_candidates=3, draw_failures=1),
            zero(2, (120, 180), generation_failure="RuntimeError: tasks accepted after 8 rounds"),
            task(
                3,
                "unavailable",
                (180, 240),
                accepted=2,
                attempt=ATTEMPT,
                rejected_candidates=1,
                draw_failures=0,
                detail="payload missing",
            ),
            task(4, "conflict", detail="attempts disagree"),
            task(5, "invalid", detail="record fails validation"),
            task(6, "unsupported", detail="contract 2.0"),
            task(7, "unreadable"),
            task(8, "pending"),
        ],
        attempts=(attempt(gpu_requested="A100", gpu_name="NVIDIA A100-SXM4-40GB"),),
        issues=("attempt record x is not JSON",),
    )
    status = figures.status(snapshot)
    assert status["line"] == [
        "state: running",
        "tasks recorded: 8/9",
        "zero-world tasks: 1",
        "shortfall: 3 worlds",
        "rejected candidates: ≥ 6; unknown for 1 task",
        "draw failures: ≥ 1; unknown for 1 task",
        "unavailable: 1",
        "conflict: 1",
        "invalid: 1",
        "unsupported: 1",
        "unreadable: 1",
        "duplicate records: 1",
        "latest attempt: running",
        "GPU: A100 requested, NVIDIA A100-SXM4-40GB detected",
        "issues: attempt record x is not JSON",
    ]
    assert status["problems"] == [
        {
            "task_id": "task-00002",
            "status": "zero",
            "detail": "RuntimeError: tasks accepted after 8 rounds",
        },
        {"task_id": "task-00003", "status": "unavailable", "detail": "payload missing"},
        {"task_id": "task-00004", "status": "conflict", "detail": "attempts disagree"},
        {"task_id": "task-00005", "status": "invalid", "detail": "record fails validation"},
        {"task_id": "task-00006", "status": "unsupported", "detail": "contract 2.0"},
        {"task_id": "task-00007", "status": "unreadable", "detail": "no detail recorded"},
    ]


def test_status_line_with_complete_accounting_and_a_failed_attempt():
    snapshot = run(
        [published(0, (0, 60), rejected_candidates=1), published(1, (60, 90))],
        state="failed",
        attempts=(attempt("failed", error="MemoryError"),),
    )
    assert figures.status(snapshot) == {
        "line": [
            "state: failed",
            "tasks recorded: 2/2",
            "zero-world tasks: 0",
            "shortfall: 0 worlds",
            "rejected candidates: 1",
            "draw failures: 0",
            "latest attempt: failed (MemoryError)",
        ],
        "problems": [],
    }
    assert figures.status(run([task(0, "pending")], state="waiting"))["line"][-1] == (
        "latest attempt: none recorded"
    )


def _plain_values(value: Any) -> None:
    """Fail on anything but JSON's own Python types, and on non-finite floats."""
    if isinstance(value, dict):
        assert all(type(key) is str for key in value)
        for item in value.values():
            _plain_values(item)
    elif isinstance(value, list):
        for item in value:
            _plain_values(item)
    else:
        assert type(value) in (str, int, float, bool, type(None)), type(value)
        if type(value) is float:
            assert math.isfinite(value)


def test_run_body_is_plain_strict_json_even_from_numpy_and_nan_inputs():
    snapshot = run(
        [
            published(0, (0, 60), generation_seconds=np.float64(math.nan)),
            published(1, (60, 120), generation_seconds=np.float64(30.0)),
        ],
        worlds=(
            world(0, np.float64(0.5), np.float64(0.25), np.float64(0.25)),
            world(1, np.float64(math.inf), 0.5, 0.5),
            world(2, np.float64(0.1), np.float64(-0.2), np.float64(1.1)),
        ),
        impacts=(impact(0, "C1", np.float64(0.5)), impact(1, "C1", np.float64(-math.inf))),
    )
    body = figures.run_body(snapshot)
    _plain_values(body)
    text = json.dumps(body, allow_nan=False)
    assert "bdata" not in text
    assert body["run"] == {
        "run_id": RUN_ID,
        "label": "demo",
        "preset": "small",
        "state": "running",
        "created_at": "2026-01-01T00:00:00.000Z",
        "n_tasks": 2,
        "worlds_per_task": 2,
        "target_worlds": 4,
        "seed": 1,
        "code_revision": None,
        "published_worlds": 4,
        "recorded_tasks": 2,
    }
    assert [item["id"] for item in body["figures"]] == list(PANELS)
