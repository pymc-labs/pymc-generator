"""Archive-free checks for compact historical study analysis, not new inference."""

import shutil
from pathlib import Path

import numpy as np
import pytest

from studies.recovery.results.tv_baseline.analysis import load_snapshot, summarize, trajectory

ROOT = Path(__file__).parent


def test_compact_histories_recompute_approved_denominators_and_descriptors():
    meta, arrays = load_snapshot(ROOT)
    report = summarize(meta, arrays)
    tv, bounded = report["counts"]
    assert (
        tv["valid"],
        tv["media_pass"],
        tv["joint_pass"],
        tv["total_media_pass"],
        tv["baseline_pass"],
    ) == (50, 11, 9, 36, 36)
    assert (
        bounded["valid"],
        bounded["unresolved_ids"],
        bounded["media_pass"],
        bounded["joint_pass"],
        bounded["total_media_pass"],
        bounded["baseline_pass"],
    ) == (48, [6, 9], 13, 13, 28, 26)
    assert (
        tv["media_eligible"],
        tv["media_failed"],
        tv["control_eligible"],
        tv["control_failed"],
    ) == (236, 144, 148, 19)
    assert (
        bounded["media_evaluable"],
        bounded["media_failed"],
        bounded["media_unresolved"],
        bounded["control_evaluable"],
        bounded["control_failed"],
        bounded["control_unresolved"],
    ) == (211, 122, 25, 135, 15, 13)
    assert report["tv_diagnostics"]["all_pass"] and report["tv_diagnostics"][
        "max_rhat"
    ] == pytest.approx(1.0069055735834518)
    assert report["tv_noise_fraction_quartiles"] == pytest.approx(
        [0.010858795225, 0.018585616915, 0.03479104749]
    )
    passed, failed = report["tv_descriptor_medians"]
    assert (passed["count"], failed["count"]) == (92, 144)
    assert (
        passed["contribution_sd_per_noise_sd"],
        failed["contribution_sd_per_noise_sd"],
    ) == pytest.approx((1.573056702109, 0.781619424339))
    for regime, evaluable, failures, unresolved in (("tv", 35, 19, 0), ("bounded", 32, 17, 3)):
        row = next(
            r
            for r in report["pattern_rows"]
            if r["regime"] == regime and r["kind"] == "media" and r["pattern"] == "trend_up"
        )
        assert (
            row["total"],
            row["eligible"],
            row["known_zero"],
            row["evaluable"],
            row["failed"],
            row["unresolved"],
        ) == (72, 35, 37, evaluable, failures, unresolved)
    assert len(report["pattern_rows"]) == 40
    assert len(report["world_rows"]) == 100 and len(report["channel_rows"]) == 1598
    assert all(
        trajectory(meta, arrays, label, world)["truth"].shape[0] == 97
        for label in ("tv", "bounded")
        for world in range(50)
    )
    assert np.array_equal(
        trajectory(meta, arrays, "tv", 0)["raw_spend"],
        trajectory(meta, arrays, "bounded", 0)["raw_spend"],
    )


def test_generated_scatter_legend_uses_derived_group_sizes():
    """Exercise the actual generated cell with nonhistorical counts (1 pass, 2 fail)."""
    import matplotlib.pyplot as plt
    import pandas as pd

    from studies.recovery.results.tv_baseline.build_notebook import notebook

    book = notebook()
    scatter = next(
        cell.source
        for cell in book.cells
        if cell.cell_type == "code" and "TV channel-level descriptors" in cell.source
    )
    assert not any("92 passing versus 144 failing" in cell.source for cell in book.cells)
    figures = []
    fixtures = [
        {
            "point_pass": passed,
            "contribution_cv": float(index + 1),
            "contribution_sd_per_noise_sd": float(index + 1),
        }
        for index, passed in enumerate((True, False, False))
    ]
    exec(
        scatter,
        {
            "pd": pd,
            "plt": plt,
            "display": lambda value: figures.append(value),
            "report": {
                "tv_diagnostics": {},
                "tv_descriptor_medians": [],
                "tv_noise_fraction_quartiles": [0.1, 0.2, 0.3],
                "tv_descriptors": fixtures,
            },
        },
    )
    legend = figures[-1].axes[0].get_legend()
    assert [text.get_text() for text in legend.get_texts()] == [
        "TV channel passes (1)",
        "TV channel fails (2)",
    ]


def test_compact_hash_fails_closed_before_scoring(tmp_path):
    for name in (
        "snapshot.json",
        "snapshot.npz",
        "historical-config.json",
        "provenance.json",
        "tv50.md",
        "bounded50.md",
    ):
        shutil.copyfile(ROOT / name, tmp_path / name)
    meta, arrays = load_snapshot(tmp_path)
    assert len(summarize(meta, arrays)["world_rows"]) == 100
    target = tmp_path / "tv50.md"
    target.write_bytes(target.read_bytes() + b"\nUNREVIEWED\n")
    with pytest.raises(ValueError, match="hash mismatch"):
        load_snapshot(tmp_path)
