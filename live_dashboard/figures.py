"""Score cards, figures and the status line of one run, as plain JSON-ready values.

The page draws exactly what this module returns, so everything a viewer learns about a
run is decided here, in pure functions of a ``RunSnapshot``:

- values are Python lists, strings, ints and finite floats. Never numpy arrays: plotly
  serialises those as binary ``bdata``, and the API promises plain, NaN-free JSON. A
  non-finite number becomes ``None`` (rates) or is left out and counted (points);
- times are UTC ISO-8601 strings, which plotly.js reads as dates;
- speeds divide by *active* generation time, the union of the timed tasks'
  ``[started_at, finished_at]`` intervals: idle gaps between attempts do not dilute a
  rate and overlapping attempts are not counted twice.

Only ``published`` tasks contribute worlds; every other outcome is reported in the
status line, never mixed into the distributions of the consumable dataset.
"""

from __future__ import annotations

import bisect
import itertools
import math
from collections import Counter
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any, NamedTuple

import plotly
import plotly.graph_objects as go
from plotly.offline import get_plotlyjs

from live_dashboard.extract import ADSTOCK_FAMILIES, SATURATION_FAMILIES
from live_dashboard.publication import format_timestamp
from live_dashboard.snapshot import RunSnapshot, TaskState, WorldMetrics

PLOTLY_VERSION: str = plotly.__version__

# Statuses whose authoritative record carries the generation interval: their time counts
# as active generation time even when the output is not consumable.
TIMED_STATUSES = frozenset({"published", "zero", "unavailable", "invalid"})
# Statuses that measured how fast the generator produced consumable worlds.
RATE_STATUSES = frozenset({"published", "zero"})
PROBLEM_STATUSES = ("unavailable", "conflict", "invalid", "unsupported", "unreadable")
WAITING_TEXT = "waiting for the first publication"

_TRIANGLE_HEIGHT = math.sqrt(3.0) / 2.0
_GRID_LEVELS = (0.2, 0.4, 0.6, 0.8)
_TEMPLATE = "plotly_white"  # explicit: the global plotly default is mutable process state
_WORLD_COLOR = "#306998"
_CONTROL_COLOR = "#e07b39"
_NEGATIVE_COLOR = "#c0392b"
_TARGET_COLOR = "#7f7f7f"
_GRID_COLOR = "#d9d9d9"
_OUTLINE_COLOR = "#444444"
# Axis labels of the library's family keys (the hover text shows the key itself).
_FAMILY_LABELS = {"michaelis_menten": "Michaelis-Menten", "hill": "Hill", "weibull": "Weibull"}


def plotly_js() -> bytes:
    """The plotly.js bundle of the installed plotly package.

    The app serves it itself, so the page needs no third-party CDN and always runs the
    plotly.js version the Python figure schema was written for.
    """
    return get_plotlyjs().encode("utf-8")


# -- helpers ----------------------------------------------------------------------------


class _Timed(NamedTuple):
    task: TaskState
    started_at: datetime
    finished_at: datetime


def _iso(moment: datetime) -> str:
    return format_timestamp(moment)


def _finite(value: float | None) -> float | None:
    """A JSON-safe Python float, or ``None`` for missing and non-finite values."""
    if value is None:
        return None
    number = float(value)
    return number if math.isfinite(number) else None


def _plural(count: int, noun: str) -> str:
    return f"{count} {noun}" if count == 1 else f"{count} {noun}s"


def _rate_text(rate: float | None) -> str:
    return "—" if rate is None else f"{rate:.1f} worlds/min"


def _timed_tasks(run: RunSnapshot) -> list[_Timed]:
    """Tasks with a generation interval, in completion order (ties by task index)."""
    timed = [
        _Timed(task, task.started_at, task.finished_at)
        for task in run.tasks
        if task.status in TIMED_STATUSES
        and task.started_at is not None
        and task.finished_at is not None
    ]
    return sorted(timed, key=lambda item: (item.finished_at, item.task.task_index))


def _task_rate(item: _Timed) -> float | None:
    """Consumable worlds per minute of one task, from its reported generation seconds
    (wall time when unreported); ``None`` unless that duration is positive."""
    seconds = item.task.generation_seconds
    if seconds is None:
        seconds = (item.finished_at - item.started_at).total_seconds()
    return _rate(item.task.consumable_worlds, float(seconds))


def _rate(worlds: int, seconds: float) -> float | None:
    if not (math.isfinite(seconds) and seconds > 0):
        return None
    return _finite(worlds * 60.0 / seconds)


class _ActiveTime:
    """Total length of a growing union of closed time intervals.

    Intervals are kept disjoint and sorted, so adding one costs a bisection plus the
    merge of the intervals it touches: appending in completion order (the usual case
    for one writer) is cheap. An interval ending before it starts counts as empty.
    """

    def __init__(self) -> None:
        self._starts: list[datetime] = []
        self._ends: list[datetime] = []
        self._total = timedelta(0)

    def add(self, start: datetime, end: datetime) -> None:
        end = max(start, end)
        first = bisect.bisect_left(self._ends, start)
        stop = bisect.bisect_right(self._starts, end)
        if first < stop:
            for index in range(first, stop):
                self._total -= self._ends[index] - self._starts[index]
            start = min(start, self._starts[first])
            end = max(end, self._ends[stop - 1])
        self._starts[first:stop] = [start]
        self._ends[first:stop] = [end]
        self._total += end - start

    @property
    def seconds(self) -> float:
        return self._total.total_seconds()


def _active_seconds(timed: Iterable[_Timed]) -> float:
    active = _ActiveTime()
    for item in timed:
        active.add(item.started_at, item.finished_at)
    return active.seconds


def _layout(**properties: Any) -> dict[str, Any]:
    layout: dict[str, Any] = {
        "template": _TEMPLATE,
        "autosize": True,
        "margin": {"l": 56, "r": 16, "t": 32, "b": 48},
        "legend": {"orientation": "h", "x": 0.0, "y": 1.02, "yanchor": "bottom"},
        "hovermode": "closest",
    }
    layout.update(properties)
    return layout


def _panel(
    panel_id: str, title: str, figure: go.Figure | None, empty: str | None, note: str | None
) -> dict[str, Any]:
    """One figure slot: either a plotly figure or the reason it is empty."""
    return {
        "id": panel_id,
        "title": title,
        "figure": None if figure is None else figure.to_dict(),
        "empty": empty,
        "note": note,
    }


# -- run summary, cards and empty states ----------------------------------------------


def run_summary(run: RunSnapshot) -> dict[str, Any]:
    """What identifies a run in the run list and above its figures."""
    return {
        "run_id": run.run_id,
        "label": run.label,
        "preset": run.preset,
        "state": run.state,
        "created_at": _iso(run.created_at),
        "n_tasks": run.n_tasks,
        "worlds_per_task": run.worlds_per_task,
        "target_worlds": run.target_worlds,
        "seed": run.seed,
        "code_revision": run.code_revision,
        "published_worlds": run.published_worlds,
        "recorded_tasks": run.recorded_tasks,
    }


def worlds_card(run: RunSnapshot) -> dict[str, Any]:
    return {
        "id": "worlds",
        "title": "Worlds generated",
        "value": run.published_worlds,
        "text": f"{run.published_worlds} / {run.target_worlds}",
        "subtitle": f"{run.recorded_tasks}/{run.n_tasks} tasks recorded",
    }


def speed_card(run: RunSnapshot) -> dict[str, Any]:
    """Consumable worlds per minute of active generation, plus the latest task's rate."""
    timed = _timed_tasks(run)
    rate = _rate(run.published_worlds, _active_seconds(timed))
    published = [item for item in timed if item.task.status == "published"]
    if published:
        subtitle = f"latest task: {_rate_text(_task_rate(published[-1]))}"
    else:
        subtitle = "no task published yet"
    return {
        "id": "speed",
        "title": "Generation speed",
        "value": rate,
        "text": _rate_text(rate),
        "subtitle": subtitle,
    }


def cost_card(run: RunSnapshot) -> dict[str, Any]:
    """Estimated cost of the generation worker: each attempt's recorded time (its start to
    its latest update, which the worker rewrites before every task) at the hourly list
    price its worker recorded. A lower bound of the bill: container start-up, scale-down
    and the time after an attempt's last update are not counted. Attempts without a price
    that spent time make the total a lower bound too ("≥")."""
    title = "Estimated generation cost"
    timed = [
        (max(0.0, (attempt.updated_at - attempt.started_at).total_seconds()), attempt.usd_per_hour)
        for attempt in run.attempts
    ]
    priced = [(seconds, price) for seconds, price in timed if price is not None]
    cost = _finite(sum(seconds / 3600 * price for seconds, price in priced))
    if not priced or cost is None:
        if not run.attempts:
            subtitle = "no attempt yet"
        elif not priced:
            subtitle = "no worker price recorded"
        else:
            subtitle = "recorded prices give no finite cost"
        return {"id": "cost", "title": title, "value": None, "text": "—", "subtitle": subtitle}
    minutes = sum(seconds for seconds, _ in priced) / 60
    prices = {price for _, price in priced}
    rate = f"${prices.pop():.3f}/h list price" if len(prices) == 1 else "their list prices"
    subtitle = f"{minutes:.1f} min of worker time at {rate}"
    unpriced = [seconds for seconds, price in timed if price is None]
    if unpriced:
        subtitle += f"; {_plural(len(unpriced), 'attempt')} without a price"
    partial = any(seconds > 0 for seconds in unpriced)
    return {
        "id": "cost",
        "title": title,
        "value": cost,
        "text": f"{'≥ ' if partial else ''}${cost:.3f}",
        "subtitle": subtitle,
    }


def cards(run: RunSnapshot) -> list[dict[str, Any]]:
    return [worlds_card(run), speed_card(run), cost_card(run)]


def empty_state(run: RunSnapshot) -> str:
    """Why a figure has nothing to draw, from the most to the least specific cause."""
    newest = run.attempts[0] if run.attempts else None
    if run.state == "failed":
        error = newest.error if newest is not None and newest.error else "no error recorded"
        return f"run failed: {error}"
    if run.state == "lost":
        if newest is not None and newest.deadline_at is not None:
            deadline = _iso(newest.deadline_at)
            return f"run lost: attempt {newest.attempt_id} passed its deadline {deadline} while running"
        return "run lost: the generation attempt stopped reporting"
    if run.recorded_tasks == 0 and run.state in ("waiting", "running"):
        return WAITING_TEXT
    reasons = []
    zero = sum(task.status == "zero" for task in run.tasks)
    if zero:
        reasons.append(_plural(zero, "zero-world task"))
    for problem in PROBLEM_STATUSES:
        count = sum(task.status == problem for task in run.tasks)
        if count:
            reasons.append(f"{count} {problem}")
    return "no consumable worlds: " + (", ".join(reasons) if reasons else "no task recorded")


# -- figures -----------------------------------------------------------------------------


def speed_figure(run: RunSnapshot) -> dict[str, Any]:
    """Fig 1: per-task rates and the cumulative rate over active time."""
    title = "Generation speed (worlds/min)"
    timed = _timed_tasks(run)
    if not timed:
        return _panel("speed", title, None, empty_state(run), None)
    task_x: list[str] = []
    task_y: list[float] = []
    task_text: list[str] = []
    for item in timed:
        rate = _task_rate(item) if item.task.status in RATE_STATUSES else None
        if rate is not None:
            task_x.append(_iso(item.finished_at))
            task_y.append(rate)
            task_text.append(
                f"{item.task.task_id}: {_plural(item.task.consumable_worlds, 'world')}"
            )
    line_x: list[str] = []
    line_y: list[float | None] = []
    active = _ActiveTime()
    consumable = 0
    for finished_at, group in itertools.groupby(timed, key=lambda item: item.finished_at):
        for item in group:
            active.add(item.started_at, item.finished_at)
            consumable += item.task.consumable_worlds
        line_x.append(_iso(finished_at))
        line_y.append(_rate(consumable, active.seconds))
    figure = go.Figure(
        data=[
            go.Scatter(
                x=line_x,
                y=line_y,
                mode="lines+markers",
                name="cumulative (active time)",
                hovertemplate="%{y:.1f} worlds/min<extra>cumulative</extra>",
            ),
            go.Scatter(
                x=task_x,
                y=task_y,
                mode="markers",
                name="per task",
                text=task_text,
                marker={"size": 9, "opacity": 0.7},
                hovertemplate="%{text}<br>%{y:.1f} worlds/min<extra></extra>",
            ),
        ],
        layout=_layout(
            xaxis={"type": "date", "title": {"text": "task finished (UTC)"}},
            yaxis={"title": {"text": "worlds / min"}, "rangemode": "tozero"},
        ),
    )
    return _panel("speed", title, figure, None, None)


def cumulative_figure(run: RunSnapshot) -> dict[str, Any]:
    """Fig 2: consumable worlds over time as a step line, against the run target."""
    title = "Worlds generated"
    timed = _timed_tasks(run)
    if not timed:
        return _panel("cumulative", title, None, empty_state(run), None)
    start = min(item.started_at for item in timed)
    end = max(start, timed[-1].finished_at)
    x = [_iso(start)]
    y = [0]
    total = 0
    published = [item for item in timed if item.task.status == "published"]
    for finished_at, group in itertools.groupby(published, key=lambda item: item.finished_at):
        total += sum(item.task.consumable_worlds for item in group)
        x.append(_iso(finished_at))
        y.append(total)
    figure = go.Figure(
        data=[
            go.Scatter(
                x=x,
                y=y,
                mode="lines+markers",
                line_shape="hv",
                name="published worlds",
                hovertemplate="%{y} worlds<extra></extra>",
            ),
            go.Scatter(
                x=[_iso(start), _iso(end)],
                y=[run.target_worlds, run.target_worlds],
                mode="lines",
                line={"dash": "dash", "color": _TARGET_COLOR},
                name=f"target ({run.target_worlds})",
                hoverinfo="skip",
            ),
        ],
        layout=_layout(
            xaxis={"type": "date", "title": {"text": "time (UTC)"}},
            yaxis={"title": {"text": "worlds"}, "rangemode": "tozero"},
        ),
    )
    return _panel("cumulative", title, figure, None, None)


def _violin(name: str, values: list[float], text: list[str]) -> go.Violin:
    return go.Violin(
        y=values,
        text=text,
        name=name,
        points="all",
        jitter=0.4,
        pointpos=0.0,
        spanmode="hard",
        box={"visible": True},
        marker={"opacity": 0.4, "size": 5},
        hoveron="points",
        hovertemplate="%{text}<br>%{y:.2%}<extra></extra>",
    )


def _share_axis(run: RunSnapshot) -> dict[str, Any]:
    """The y axis of the three share panels: 0-100 %, widened only to fit a finite share
    outside it in any of them (a negative control share), so all three keep one scale."""
    shares = itertools.chain(
        (unit.share for unit in run.impacts),
        (unit.share for unit in run.control_impacts),
        itertools.chain.from_iterable(
            (world.channel_share, world.control_share, world.baseline_share) for world in run.worlds
        ),
    )
    values = [value for value in map(_finite, shares) if value is not None]
    low, high = min([0.0, *values]), max([1.0, *values])
    pad = 0.02 * (high - low)
    return {
        "title": {"text": "share of outcome"},
        "tickformat": ".0%",
        "range": [low - pad, high + pad],
    }


def _shares_figure(
    run: RunSnapshot,
    panel_id: str,
    title: str,
    violins: list[tuple[str, Iterable[tuple[float, str]]]],
) -> dict[str, Any]:
    """One violin per ``(name, (share, hover text) points)`` on the run's share axis;
    non-finite shares are left out and counted in the note."""
    traces: list[go.Violin] = []
    dropped = 0
    for name, points in violins:
        values: list[float] = []
        text: list[str] = []
        for share, label in points:
            value = _finite(share)
            if value is None:
                dropped += 1
            else:
                values.append(value)
                text.append(label)
        traces.append(_violin(name, values, text))
    figure = go.Figure(data=traces, layout=_layout(showlegend=False, yaxis=_share_axis(run)))
    note = f"{_plural(dropped, 'non-finite share')} not shown" if dropped else None
    return _panel(panel_id, title, figure, None, note)


def impact_figure(run: RunSnapshot) -> dict[str, Any]:
    """Fig 3: direct share of each active channel, and all channels' share per world."""
    title = "Channel impact (share of outcome)"
    if not run.worlds:
        return _panel("impact", title, None, empty_state(run), None)
    return _shares_figure(
        run,
        "impact",
        title,
        [
            ("Each active channel", ((u.share, f"{u.world_id} {u.channel}") for u in run.impacts)),
            ("All channels per world", ((w.channel_share, w.world_id) for w in run.worlds)),
        ],
    )


def control_impact_figure(run: RunSnapshot) -> dict[str, Any]:
    """Fig 4: direct share of each active control, and all controls' share per world."""
    title = "Control impact (share of outcome)"
    if not run.worlds:
        return _panel("control_impact", title, None, empty_state(run), None)
    return _shares_figure(
        run,
        "control_impact",
        title,
        [
            (
                "Each active control",
                ((u.share, f"{u.world_id} {u.control}") for u in run.control_impacts),
            ),
            ("All controls per world", ((w.control_share, w.world_id) for w in run.worlds)),
        ],
    )


def baseline_figure(run: RunSnapshot) -> dict[str, Any]:
    """Fig 5: the baseline's share per world (intrinsic walk, latent and noise)."""
    title = "Baseline (share of outcome)"
    if not run.worlds:
        return _panel("baseline", title, None, empty_state(run), None)
    return _shares_figure(
        run,
        "baseline",
        title,
        [("Baseline per world", ((w.baseline_share, w.world_id) for w in run.worlds))],
    )


def _triangle_point(control: float, baseline: float) -> tuple[float, float]:
    """Ternary position: channels at (0, 0), controls at (1, 0), baseline at the apex."""
    return control + 0.5 * baseline, _TRIANGLE_HEIGHT * baseline


def _grid_lines() -> tuple[list[float | None], list[float | None]]:
    """Lines of constant share every 20 % for each part, as one gapped polyline."""
    x: list[float | None] = []
    y: list[float | None] = []
    for level in _GRID_LEVELS:
        rest = 1.0 - level
        for ends in (  # (control, baseline) end points
            ((0.0, level), (rest, level)),  # baseline == level
            ((level, 0.0), (level, rest)),  # control == level
            ((rest, 0.0), (0.0, rest)),  # channels == level
        ):
            for control, baseline in ends:
                point_x, point_y = _triangle_point(control, baseline)
                x.append(point_x)
                y.append(point_y)
            x.append(None)
            y.append(None)
    return x, y


@dataclass
class _Points:
    name: str
    symbol: str
    opacity: float
    color: str
    x: list[float] = field(default_factory=list)
    y: list[float] = field(default_factory=list)
    text: list[str] = field(default_factory=list)
    shares: list[list[float]] = field(default_factory=list)

    def trace(self) -> go.Scatter:
        return go.Scatter(
            x=self.x,
            y=self.y,
            text=self.text,
            customdata=self.shares,
            mode="markers",
            name=f"{self.name} ({len(self.x)})",
            marker={"symbol": self.symbol, "opacity": self.opacity, "color": self.color, "size": 9},
            hovertemplate=(
                "%{text}<br>channels %{customdata[0]:.1%}<br>controls %{customdata[1]:.1%}"
                "<br>baseline %{customdata[2]:.1%}<extra></extra>"
            ),
        )


def composition_figure(run: RunSnapshot) -> dict[str, Any]:
    """Fig 6: each world's channel / control / baseline split of its outcome.

    Worlds with a negative part lie outside the triangle and are drawn as crosses;
    worlds with a missing or non-finite share cannot be placed and are only counted.
    """
    title = "Outcome composition per world"
    if not run.worlds:
        return _panel("composition", title, None, empty_state(run), None)
    inside = _Points("worlds", "circle", 0.35, _WORLD_COLOR)
    outside = _Points("worlds with a negative part", "x", 0.8, _NEGATIVE_COLOR)
    excluded = 0
    for world in run.worlds:
        channel = _finite(world.channel_share)
        control = _finite(world.control_share)
        baseline = _finite(world.baseline_share)
        if channel is None or control is None or baseline is None:
            excluded += 1
            continue
        points = outside if min(channel, control, baseline) < 0 else inside
        point_x, point_y = _triangle_point(control, baseline)
        points.x.append(point_x)
        points.y.append(point_y)
        points.text.append(world.world_id)
        points.shares.append([channel, control, baseline])
    grid_x, grid_y = _grid_lines()
    corners = (
        (0.0, 0.0, "Channels (direct + indirect)", "top"),
        (1.0, 0.0, "Controls", "top"),
        (0.5, _TRIANGLE_HEIGHT, "Baseline", "bottom"),
    )
    figure = go.Figure(
        data=[
            go.Scatter(
                x=grid_x,
                y=grid_y,
                mode="lines",
                line={"color": _GRID_COLOR, "width": 1},
                hoverinfo="skip",
                showlegend=False,
            ),
            go.Scatter(
                x=[0.0, 1.0, 0.5, 0.0],
                y=[0.0, 0.0, _TRIANGLE_HEIGHT, 0.0],
                mode="lines",
                line={"color": _OUTLINE_COLOR, "width": 1.5},
                hoverinfo="skip",
                showlegend=False,
            ),
            inside.trace(),
            outside.trace(),
        ],
        layout=_layout(
            xaxis={"visible": False},
            yaxis={"visible": False, "scaleanchor": "x", "scaleratio": 1},
            annotations=[
                {"x": x, "y": y, "text": text, "showarrow": False, "yanchor": anchor}
                for x, y, text, anchor in corners
            ],
        ),
    )
    note = f"{_plural(excluded, 'world')} with non-finite shares not shown" if excluded else None
    return _panel("composition", title, figure, None, note)


# Tick steps of a count axis: the first that keeps at most 8 steps up to the largest bar,
# so ticks stay integers (plotly's automatic steps go fractional for small counts).
_COUNT_STEPS = (1, 2, 5, 10, 20, 50, 100, 200, 500, 1000, 2000, 5000)


def _count_step(top: int) -> int:
    return next((step for step in _COUNT_STEPS if top <= 8 * step), _COUNT_STEPS[-1])


def _forms_figure(
    panel_id: str,
    title: str,
    run: RunSnapshot,
    families: tuple[str, ...],
    forms_of: Callable[[WorldMetrics], tuple[str, ...] | None],
) -> dict[str, Any]:
    """Consumable worlds per family, with every family of the library on the x axis.

    A world counts once for each family its active channels use, so a world whose
    channels mix families counts in each of them.
    """
    if not run.worlds:
        return _panel(panel_id, title, None, empty_state(run), None)
    recorded = [forms for forms in map(forms_of, run.worlds) if forms is not None]
    if not recorded:
        return _panel(
            panel_id, title, None, "this run's payloads do not record these families", None
        )
    counts = [sum(family in forms for forms in recorded) for family in families]
    figure = go.Figure(
        data=[
            go.Bar(
                x=[_FAMILY_LABELS.get(family, family) for family in families],
                y=counts,
                customdata=list(families),
                text=counts,
                textposition="outside",
                cliponaxis=False,
                marker={"color": _WORLD_COLOR},
                hovertemplate="%{customdata}: %{y} worlds<extra></extra>",
            )
        ],
        layout=_layout(
            showlegend=False,
            yaxis={
                "title": {"text": "worlds"},
                "rangemode": "tozero",
                "tick0": 0,
                "dtick": _count_step(max(counts)),
                "tickformat": "d",
            },
        ),
    )
    notes = []
    unrecorded = len(run.worlds) - len(recorded)
    if unrecorded:
        notes.append(f"{_plural(unrecorded, 'world')} without recorded families not counted")
    mixed = sum(len(forms) > 1 for forms in recorded)
    if mixed:
        notes.append(f"{_plural(mixed, 'world')} mixing families, counted once per family")
    return _panel(panel_id, title, figure, None, "; ".join(notes) or None)


def saturation_figure(run: RunSnapshot) -> dict[str, Any]:
    """Fig 7: consumable worlds per saturation family of their active channels."""
    return _forms_figure(
        "saturation",
        "Worlds per saturation form",
        run,
        SATURATION_FAMILIES,
        lambda world: world.saturation_forms,
    )


def adstock_figure(run: RunSnapshot) -> dict[str, Any]:
    """Fig 8: consumable worlds per adstock (carryover) family of their active channels."""
    return _forms_figure(
        "adstock",
        "Worlds per adstock form",
        run,
        ADSTOCK_FAMILIES,
        lambda world: world.adstock_forms,
    )


def _active_axes(run: RunSnapshot) -> tuple[list[int], list[int]]:
    """Every active-channel count and every active-control count the layouts allow.

    Counts run from 1 (from 0 when some world has none active) up to the most slots.
    """

    def axis(active: list[int], slots: list[int]) -> list[int]:
        return list(range(0 if min(active) == 0 else 1, max(*active, *slots) + 1))

    worlds = run.worlds
    return (
        axis([w.active_channels for w in worlds], [w.channel_slots for w in worlds]),
        axis([w.active_controls for w in worlds], [w.control_slots for w in worlds]),
    )


def combinations_figure(run: RunSnapshot) -> dict[str, Any]:
    """Fig 9: consumable worlds per (active channels, active controls) combination.

    Every combination the layout allows has a cell, so absent combinations show as empty.
    """
    title = "Worlds per active channels × active controls"
    if not run.worlds:
        return _panel("combinations", title, None, empty_state(run), None)
    channels, controls = _active_axes(run)
    counts = Counter((w.active_channels, w.active_controls) for w in run.worlds)
    z = [[counts.get((c, m), 0) for m in controls] for c in channels]
    figure = go.Figure(
        data=[
            go.Heatmap(
                x=controls,
                y=channels,
                z=z,
                text=[[str(value) if value else "" for value in row] for row in z],
                texttemplate="%{text}",
                colorscale="Blues",
                zmin=0,
                xgap=2,
                ygap=2,
                colorbar={
                    "title": {"text": "worlds"},
                    "tick0": 0,
                    "dtick": _count_step(max(counts.values())),
                    "tickformat": "d",
                },
                hovertemplate="%{y} channels, %{x} controls: %{z} worlds<extra></extra>",
            )
        ],
        layout=_layout(
            xaxis={
                "title": {"text": "active controls (covariates)"},
                "dtick": 1,
                "showgrid": False,
            },
            yaxis={
                "title": {"text": "active channels (treatments)"},
                "dtick": 1,
                "showgrid": False,
            },
        ),
    )
    note = f"{len(counts)} of {len(channels) * len(controls)} combinations present"
    return _panel("combinations", title, figure, None, note)


def active_counts_figure(run: RunSnapshot) -> dict[str, Any]:
    """Fig 10: consumable worlds per number of active channels and of active controls."""
    title = "Worlds per number of active inputs"
    if not run.worlds:
        return _panel("active_counts", title, None, empty_state(run), None)
    channel_axis, control_axis = _active_axes(run)
    axis = sorted({*channel_axis, *control_axis})
    channels = Counter(w.active_channels for w in run.worlds)
    controls = Counter(w.active_controls for w in run.worlds)
    traces = [
        ("channels (treatments)", channels, _WORLD_COLOR),
        ("controls (covariates)", controls, _CONTROL_COLOR),
    ]
    top = max(max(channels.values()), max(controls.values()))
    figure = go.Figure(
        data=[
            go.Bar(
                name=name,
                x=axis,
                y=[counts.get(n, 0) for n in axis],
                marker={"color": color},
                hovertemplate="%{x} active " + name.split()[0] + ": %{y} worlds<extra></extra>",
            )
            for name, counts, color in traces
        ],
        layout=_layout(
            barmode="group",
            xaxis={"title": {"text": "active inputs"}, "dtick": 1},
            yaxis={
                "title": {"text": "worlds"},
                "rangemode": "tozero",
                "tick0": 0,
                "dtick": _count_step(top),
                "tickformat": "d",
            },
        ),
    )
    return _panel("active_counts", title, figure, None, None)


# -- status ------------------------------------------------------------------------------


def _count_text(values: list[int | None]) -> str:
    """A sum that admits when some tasks did not report the count."""
    known = [value for value in values if value is not None]
    unknown = len(values) - len(known)
    if unknown:
        return f"≥ {sum(known)}; unknown for {_plural(unknown, 'task')}"
    return str(sum(known))


def status(run: RunSnapshot) -> dict[str, Any]:
    """The status line (one segment per fact) and the tasks that need an explanation."""
    tasks = run.tasks
    outcomes = [task for task in tasks if task.status in RATE_STATUSES]
    # Tasks with an authoritative record: only these report candidate accounting.
    accounted = [task for task in tasks if task.attempt_id is not None]
    shortfall = sum(max(task.target_worlds - task.accepted_worlds, 0) for task in outcomes)
    line = [
        f"state: {run.state}",
        f"tasks recorded: {run.recorded_tasks}/{run.n_tasks}",
        f"zero-world tasks: {sum(task.status == 'zero' for task in tasks)}",
        f"shortfall: {_plural(shortfall, 'world')}",
        f"rejected candidates: {_count_text([task.rejected_candidates for task in accounted])}",
        f"draw failures: {_count_text([task.draw_failures for task in accounted])}",
    ]
    for problem in PROBLEM_STATUSES:
        count = sum(task.status == problem for task in tasks)
        if count:
            line.append(f"{problem}: {count}")
    duplicates = sum(task.duplicate_records for task in tasks)
    if duplicates:
        line.append(f"duplicate records: {duplicates}")
    if run.attempts:
        newest = run.attempts[0]
        attempt = f"latest attempt: {newest.status}"
        line.append(f"{attempt} ({newest.error})" if newest.error else attempt)
        if newest.gpu_requested or newest.gpu_name:  # e.g. the early A100 runs
            requested = newest.gpu_requested or "none"
            line.append(f"GPU: {requested} requested, {newest.gpu_name or 'none'} detected")
    else:
        line.append("latest attempt: none recorded")
    if run.issues:
        line.append("issues: " + "; ".join(run.issues))
    problems = [
        {
            "task_id": task.task_id,
            "status": task.status,
            "detail": task.detail or task.generation_failure or "no detail recorded",
        }
        for task in tasks
        if task.status in PROBLEM_STATUSES or (task.status == "zero" and task.generation_failure)
    ]
    return {"line": line, "problems": problems}


_FIGURES: tuple[Callable[[RunSnapshot], dict[str, Any]], ...] = (
    speed_figure,
    cumulative_figure,
    impact_figure,
    control_impact_figure,
    baseline_figure,
    composition_figure,
    saturation_figure,
    adstock_figure,
    combinations_figure,
    active_counts_figure,
)


def run_body(run: RunSnapshot) -> dict[str, Any]:
    """Everything the page shows for one run; ``json.dumps(allow_nan=False)`` safe."""
    return {
        "run": run_summary(run),
        "cards": cards(run),
        "figures": [build(run) for build in _FIGURES],
        "status": status(run),
    }
