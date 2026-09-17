"""Diagnostics figures: artist-level semantics, bounds, and purity.

Every assertion here inspects the ARTISTS a plot function produced — the
image arrays, bar patches, markers and annotations — never a rendered pixel.
``savefig`` is intercepted so a figure can be interrogated after the function
returned; the one smoke test at the end writes a real file.
"""

from __future__ import annotations

import hashlib

import numpy as np
import pytest

from pymc_generator import viz
from pymc_generator.diagnostics import DEPENDENCE_METRICS, data_diagnostics

# ---------------------------------------------------------------------------
# Fixtures: hand-built corpora with the exact structures under test
# ---------------------------------------------------------------------------


def make_corpus(
    *,
    treatment: np.ndarray,
    covariates: np.ndarray,
    latent_unobserved: np.ndarray,
    contributions: np.ndarray,
    covariate_contribution: np.ndarray,
    latent_unobserved_contribution: np.ndarray,
    indirect_by_source: np.ndarray,
    baseline_intrinsic: np.ndarray,
    outcome_noise: np.ndarray,
    treatment_mask: np.ndarray,
    covariate_mask: np.ndarray,
    latent_mask: np.ndarray,
) -> dict[str, np.ndarray]:
    """A minimal current-schema corpus whose outcome identity holds exactly."""
    baseline = (
        baseline_intrinsic
        + latent_unobserved_contribution.sum(axis=2)
        + covariate_contribution.sum(axis=2)
    )
    outcome = baseline + outcome_noise + contributions.sum(axis=2) + indirect_by_source.sum(axis=2)
    return {
        "treatment_raw": treatment,
        "covariates": covariates,
        "latent_unobserved": latent_unobserved,
        "outcome_raw": outcome,
        "baseline_raw": baseline,
        "baseline_intrinsic": baseline_intrinsic,
        "outcome_noise": outcome_noise,
        "covariate_contribution": covariate_contribution,
        "latent_unobserved_contribution": latent_unobserved_contribution,
        "treatment_contribution_raw": contributions,
        "indirect_effects_by_source": indirect_by_source,
        "indirect_effects": indirect_by_source.sum(axis=2),
        "outcome_scale": np.maximum(outcome.std(axis=1), 1e-3),
        "treatment_active_mask": treatment_mask.astype(np.uint8),
        "covariate_active_mask": covariate_mask.astype(np.uint8),
        "latent_active_mask": latent_mask.astype(np.uint8),
    }


def toy_corpus(
    *,
    n_worlds: int = 3,
    n_time: int = 16,
    n_treatments: int = 3,
    n_covariates: int = 2,
    seed: int = 7,
    dead_last_treatment: bool = True,
    quadratic_second_treatment: bool = False,
    collinear_first_covariate: bool = False,
    zero_last_covariate: bool = False,
) -> dict[str, np.ndarray]:
    """A fast synthetic corpus with switchable diagnosable structures.

    * ``dead_last_treatment`` pads the last treatment off (no eligible world),
    * ``quadratic_second_treatment`` makes C2 an exact function of C1 (so xi is
      strongly asymmetric between them),
    * ``collinear_first_covariate`` sets Z1 = C1 (so the observed design is
      exactly collinear and VIF is a valid ``+inf``),
    * ``zero_last_covariate`` keeps the last covariate ACTIVE but identically
      zero (a real structural zero, and a constant predictor for VIF).
    """
    rng = np.random.default_rng(seed)
    shape2 = (n_worlds, n_time)
    treatment = rng.normal(size=(*shape2, n_treatments))
    covariates = rng.normal(size=(*shape2, n_covariates))
    latent_unobserved = rng.normal(size=(*shape2, 2))
    contributions = rng.normal(size=(*shape2, n_treatments))
    covariate_contribution = rng.normal(size=(*shape2, n_covariates))
    latent_unobserved_contribution = rng.normal(size=(*shape2, 2))
    indirect = rng.normal(size=(*shape2, 3))
    treatment_mask = np.ones((n_worlds, n_treatments), dtype=bool)
    covariate_mask = np.ones((n_worlds, n_covariates), dtype=bool)
    latent_mask = np.ones((n_worlds, 2), dtype=bool)
    if quadratic_second_treatment:
        treatment[:, :, 1] = treatment[:, :, 0] ** 2
    if collinear_first_covariate:
        covariates[:, :, 0] = treatment[:, :, 0]
    if zero_last_covariate:
        covariates[:, :, -1] = 0.0
        covariate_contribution[:, :, -1] = 0.0
    if dead_last_treatment:
        treatment_mask[:, -1] = False
        treatment[:, :, -1] = 0.0
        contributions[:, :, -1] = 0.0
    return make_corpus(
        treatment=treatment,
        covariates=covariates,
        latent_unobserved=latent_unobserved,
        contributions=contributions,
        covariate_contribution=covariate_contribution,
        latent_unobserved_contribution=latent_unobserved_contribution,
        indirect_by_source=indirect,
        baseline_intrinsic=rng.normal(size=shape2),
        outcome_noise=rng.normal(size=shape2),
        treatment_mask=treatment_mask,
        covariate_mask=covariate_mask,
        latent_mask=latent_mask,
    )


@pytest.fixture(scope="module", autouse=True)
def agg_backend():
    """Headless rendering, exactly as the module docstring instructs."""
    import matplotlib

    matplotlib.use("Agg")


@pytest.fixture(scope="module")
def report():
    """Padded C3, an exactly-collinear Z1, and an identically-zero Z2."""
    return data_diagnostics(
        toy_corpus(
            quadratic_second_treatment=True,
            collinear_first_covariate=True,
            zero_last_covariate=True,
        )
    )


@pytest.fixture(scope="module")
def levels_report():
    return data_diagnostics(toy_corpus(), views=("levels",))


@pytest.fixture(scope="module")
def wide_report():
    """Exactly ``MAX_HEATMAP_KEYS`` series: 40 C + 20 Z + 2 D + B + Y."""
    return data_diagnostics(
        toy_corpus(
            n_worlds=1, n_time=10, n_treatments=40, n_covariates=20, dead_last_treatment=False
        ),
        views=("levels",),
    )


# ---------------------------------------------------------------------------
# Harness: intercept savefig, keep the figure, inspect the artists
# ---------------------------------------------------------------------------


@pytest.fixture
def capture(monkeypatch):
    """Replace ``Figure.savefig`` with a recorder; write nothing."""
    from matplotlib.figure import Figure

    calls: list[tuple[Figure, object]] = []

    def spy(self, fname, *args, **kwargs):
        calls.append((self, fname))

    monkeypatch.setattr(Figure, "savefig", spy)
    return calls


def only_figure(calls):
    assert len(calls) == 1, f"expected exactly one savefig, got {len(calls)}"
    return calls[0][0]


def image_axes(fig):
    return [ax for ax in fig.axes if ax.images]


def title_of(ax) -> str:
    """The panel title, wherever the figure anchored it."""
    return ax.get_title(loc="left") or ax.get_title()


def all_texts(fig) -> list[str]:
    out = [t.get_text() for t in fig.texts]
    for ax in fig.axes:
        out.append(title_of(ax))
        out.append(ax.get_xlabel())
        out.append(ax.get_ylabel())
        out.extend(t.get_text() for t in ax.texts)
    return out


def axis_texts(ax) -> list[str]:
    return [t.get_text() for t in ax.texts]


def n_bars(ax) -> int:
    return len(ax.patches)


def fingerprint(rep) -> str:
    """Hash of every array a diagnostics figure reads."""
    parts: list[bytes] = []
    for name in sorted(rep.views):
        view = rep[name]
        series = view.series
        parts += [series.slots.tobytes(), series.slot_valid.tobytes(), series.eligible.tobytes()]
        if series.series is not None:
            parts.append(series.series.tobytes())
        for metric in DEPENDENCE_METRICS:
            parts += [
                view.dependence.matrices[metric].tobytes(),
                view.dependence.valid[metric].tobytes(),
            ]
        temporal = view.temporal
        parts += [
            temporal.acf.tobytes(),
            temporal.acf_valid.tobytes(),
            temporal.lag_xi.tobytes(),
            temporal.lag_xi_valid.tobytes(),
        ]
        for scope in sorted(view.vif):
            vif = view.vif[scope]
            parts += [
                vif.vif.tobytes(),
                vif.valid.tobytes(),
                vif.constant.tobytes(),
                vif.eligible.tobytes(),
            ]
    budget = rep.contributions
    parts += [budget.active.tobytes(), budget.outcome_total.tobytes()]
    parts += [budget.measures[field].tobytes() for field in sorted(budget.measures)]
    return hashlib.sha256(b"".join(parts)).hexdigest()


# ---------------------------------------------------------------------------
# 1. Dependence: orientation, asymmetry, masking, colour limits
# ---------------------------------------------------------------------------


def test_xi_panel_rows_are_predictors_and_columns_are_targets(report, capture, tmp_path):
    """Row = predictor X, column = target Y — and the panel says so."""
    viz.plot_dependence_matrices(
        report, str(tmp_path / "xi.png"), views=("levels",), metrics=("xi",)
    )
    fig = only_figure(capture)
    ax = image_axes(fig)[0]
    drawn = ax.images[0].get_array()
    expected = report["levels"].dependence.matrix("xi", quantile=0.5)

    keys = list(report.keys)
    for x_key, y_key in (("C1", "C2"), ("C2", "C1"), ("Z1", "Y")):
        i, j = keys.index(x_key), keys.index(y_key)
        assert drawn[i, j] == pytest.approx(expected[i, j])

    title = title_of(ax)
    assert "row = predictor X" in title and "column = target Y" in title
    assert ax.get_ylabel() == "predictor X (row)"
    assert ax.get_xlabel() == "target Y (column)"


def test_xi_panel_keeps_a_known_asymmetric_pair_asymmetric(report, capture, tmp_path):
    """C2 = C1^2, so C2 is a function of C1 but not the other way round."""
    viz.plot_dependence_matrices(
        report, str(tmp_path / "xi.png"), views=("levels",), metrics=("xi",)
    )
    ax = image_axes(only_figure(capture))[0]
    drawn = ax.images[0].get_array()
    keys = list(report.keys)
    forward = float(drawn[keys.index("C1"), keys.index("C2")])
    backward = float(drawn[keys.index("C2"), keys.index("C1")])
    assert forward > backward + 0.1


def test_xi_limits_keep_negative_values_visible(report, capture, tmp_path):
    viz.plot_dependence_matrices(
        report, str(tmp_path / "xi.png"), views=("levels",), metrics=("xi", "xi_max")
    )
    fig = only_figure(capture)
    for ax in image_axes(fig):
        drawn = ax.images[0].get_array()
        finite = np.asarray(drawn.compressed(), dtype=np.float64)
        low, high = ax.images[0].get_clim()
        assert finite.min() < 0.0, "the toy corpus has small negative xi values"
        assert low <= finite.min(), "negatives must not be clamped out of the map"
        assert high >= finite.max()


def test_correlation_limits_are_symmetric_about_zero(report, capture, tmp_path):
    viz.plot_dependence_matrices(
        report, str(tmp_path / "corr.png"), views=("levels",), metrics=("pearson", "spearman")
    )
    for ax in image_axes(only_figure(capture)):
        low, high = ax.images[0].get_clim()
        assert low == pytest.approx(-high)
        assert high > 0.0


def test_dependence_masks_the_diagonal_and_annotates_missing_pairs(report, capture, tmp_path):
    """C3 is padded off everywhere: its pairs are N/A, never a zero."""
    viz.plot_dependence_matrices(
        report, str(tmp_path / "dep.png"), views=("levels",), metrics=("pearson",)
    )
    fig = only_figure(capture)
    ax = image_axes(fig)[0]
    drawn = ax.images[0].get_array()
    n = len(report.keys)
    diagonal = np.eye(n, dtype=bool)

    assert np.all(np.ma.getmaskarray(drawn)[diagonal]), "the diagonal is never reported"
    expected = report["levels"].dependence.matrix("pearson", quantile=0.5)
    missing = ~np.isfinite(expected) & ~diagonal
    assert missing.any(), "the padded treatment must leave unreported pairs"
    assert np.all(np.ma.getmaskarray(drawn)[missing])
    assert axis_texts(ax).count("N/A") == int(missing.sum())

    dead = list(report.keys).index("C3")
    assert np.all(np.ma.getmaskarray(drawn)[dead, :]), "a padded series is masked, not zeroed"


# ---------------------------------------------------------------------------
# 2. Paired views, the levels-only warning, and missing views
# ---------------------------------------------------------------------------


def test_dependence_defaults_to_both_views(report, capture, tmp_path):
    viz.plot_dependence_matrices(report, str(tmp_path / "dep.png"), metrics=("pearson", "xi"))
    fig = only_figure(capture)
    titles = [title_of(ax) for ax in image_axes(fig)]
    assert sum("levels" in t for t in titles) == 2
    assert sum("differences" in t for t in titles) == 2
    assert not any(viz.LEVELS_ONLY_NOTE in t for t in all_texts(fig))


def test_temporal_and_vif_figures_carry_both_views(report, capture, tmp_path):
    viz.plot_temporal_diagnostics(report, str(tmp_path / "temporal.png"), keys=("C1", "Z1"))
    temporal_titles = [title_of(ax) for ax in only_figure(capture).axes]
    assert "acf — levels — q50 over 3 worlds" in temporal_titles
    assert "acf — differences — q50 over 3 worlds" in temporal_titles
    assert "roughness / spike — levels" in temporal_titles
    assert "roughness / spike — differences" in temporal_titles

    capture.clear()
    viz.plot_vif_diagnostics(report, str(tmp_path / "vif.png"), keys=("C1", "C2"))
    vif_titles = [title_of(ax) for ax in only_figure(capture).axes]
    for view in ("levels", "differences"):
        for scope in ("observed", "oracle"):
            assert any(f"VIF — {scope} design — {view}" in t for t in vif_titles)


def test_levels_only_figure_carries_the_random_walk_warning(report, capture, tmp_path):
    viz.plot_dependence_matrices(
        report, str(tmp_path / "dep.png"), views=("levels",), metrics=("pearson",)
    )
    fig = only_figure(capture)
    assert report.n_worlds > 1
    assert viz.LEVELS_ONLY_NOTE in [t.get_text() for t in fig.texts]
    assert "random-walk-like" in viz.LEVELS_ONLY_NOTE
    assert "differences view is" in viz.LEVELS_ONLY_NOTE
    assert not any("— differences" in title_of(ax) for ax in fig.axes)


def test_levels_only_vif_figure_carries_the_warning_too(report, capture, tmp_path):
    viz.plot_vif_diagnostics(report, str(tmp_path / "vif.png"), views=("levels",))
    assert viz.LEVELS_ONLY_NOTE in [t.get_text() for t in only_figure(capture).texts]


def test_differences_only_figure_is_labelled(report, capture, tmp_path):
    viz.plot_dependence_matrices(
        report, str(tmp_path / "dep.png"), views=("differences",), metrics=("pearson",)
    )
    fig = only_figure(capture)
    assert viz.DIFFERENCES_ONLY_NOTE in [t.get_text() for t in fig.texts]
    assert "differences" in viz.DIFFERENCES_ONLY_NOTE


def test_a_view_the_report_lacks_is_named(levels_report, tmp_path):
    with pytest.raises(ValueError, match=r"\['differences'\] were not computed"):
        viz.plot_vif_diagnostics(levels_report, str(tmp_path / "vif.png"))
    with pytest.raises(ValueError, match=r"\['differences'\] were not computed"):
        viz.plot_dependence_matrices(levels_report, str(tmp_path / "dep.png"))


# ---------------------------------------------------------------------------
# 3. VIF: finite bars vs valid infinity vs no valid world
# ---------------------------------------------------------------------------


def test_vif_finite_infinite_and_unavailable_are_distinguishable(report, capture, tmp_path):
    """Z1 = C1 (valid +inf), Z2 constant (N/A), C3 padded off (N/A)."""
    viz.plot_vif_diagnostics(report, str(tmp_path / "vif.png"), views=("levels",))
    fig = only_figure(capture)
    observed = fig.axes[0]
    assert "observed design" in title_of(observed)

    vif = report["levels"].vif["observed"]
    finite_keys = [
        key
        for key in vif.keys
        if (vif.valid[:, vif.keys.index(key)] & np.isfinite(vif.values(key))).any()
    ]
    infinite_keys = [
        key
        for key in vif.keys
        if (vif.valid[:, vif.keys.index(key)] & np.isposinf(vif.values(key))).any()
    ]
    assert set(infinite_keys) == {"C1", "Z1"}
    assert finite_keys == ["C2"]

    # a finite quantile is an ordinary bar
    assert len(observed.patches) == len(finite_keys)
    heights = [patch.get_height() for patch in observed.patches]
    assert all(np.isfinite(heights)) and all(h > 0 for h in heights)

    # a valid +inf is a labelled top-edge marker, never a bar
    markers = [line for line in observed.lines if line.get_marker() == "^"]
    assert len(markers) == len(infinite_keys)
    top = observed.get_ylim()[1]
    assert np.isfinite(observed.get_ylim()).all(), "an inf bar would autoscale the axis away"
    for marker in markers:
        assert marker.get_ydata()[0] == pytest.approx(top)
    assert axis_texts(observed).count("∞ in 3 worlds") == len(infinite_keys)

    # an invalid entry is annotated, and says WHY where the report knows
    annotations = axis_texts(observed)
    assert annotations.count("N/A (constant)") == 1  # Z2 is constant
    assert annotations.count("N/A (no valid world)") == 1  # C3 is padded off
    assert "0" not in annotations


def test_all_infinite_vif_is_not_no_data(report, capture, tmp_path):
    viz.plot_vif_diagnostics(
        report, str(tmp_path / "vif.png"), views=("levels",), keys=("C1", "Z1")
    )
    observed = only_figure(capture).axes[0]
    assert n_bars(observed) == 0
    assert len([line for line in observed.lines if line.get_marker() == "^"]) == 2
    texts = axis_texts(observed)
    assert "no finite VIF: every valid world is ∞" in texts
    assert "No predictors in scope" not in texts


def test_scope_without_predictors_is_annotated(report, capture, tmp_path):
    """Y is in no design scope: the panel says so instead of drawing zeros."""
    viz.plot_vif_diagnostics(report, str(tmp_path / "vif.png"), views=("levels",), keys=("Y",))
    fig = only_figure(capture)
    for ax in fig.axes:
        assert "No predictors in scope" in axis_texts(ax)
        assert n_bars(ax) == 0


def test_vif_bar_is_the_requested_across_world_quantile(report, capture, tmp_path):
    viz.plot_vif_diagnostics(
        report, str(tmp_path / "vif.png"), views=("levels",), keys=("C2",), quantile=0.5
    )
    observed = only_figure(capture).axes[0]
    vif = report["levels"].vif["observed"]
    values = vif.values("C2")
    valid = vif.valid[:, vif.keys.index("C2")] & np.isfinite(values)
    assert observed.patches[0].get_height() == pytest.approx(np.quantile(values[valid], 0.5))
    assert "q50" in observed.get_ylabel()


# ---------------------------------------------------------------------------
# 4. Series distributions: raw values, slots, real zeros, empty panels
# ---------------------------------------------------------------------------


def test_single_world_raw_value_panel_renders(capture, tmp_path):
    single = data_diagnostics(toy_corpus(), worlds=[0], views=("levels",))
    assert single.n_worlds == 1
    viz.plot_series_distributions(
        single, str(tmp_path / "series.png"), views=("levels",), keys=("C1",), of="value"
    )
    fig = only_figure(capture)
    ax = fig.axes[0]
    assert len(ax.patches) > 0, "one world still has T retained values"
    assert "n=16 values (1 worlds)" in title_of(ax)
    assert axis_texts(ax) == []


def test_value_plot_requires_the_retained_series(tmp_path):
    dropped = data_diagnostics(toy_corpus(), views=("levels",), keep_series=False)
    with pytest.raises(ValueError, match="keep_series=True"):
        viz.plot_series_distributions(
            dropped, str(tmp_path / "series.png"), views=("levels",), keys=("C1",)
        )


def test_slot_distribution_works_without_the_retained_series(capture, tmp_path):
    dropped = data_diagnostics(toy_corpus(), views=("levels",), keep_series=False)
    viz.plot_series_distributions(
        dropped, str(tmp_path / "series.png"), views=("levels",), keys=("C1",), of="std"
    )
    ax = only_figure(capture).axes[0]
    assert len(ax.patches) > 0
    assert "std — n=3 worlds — median" in title_of(ax)


def test_unknown_of_lists_the_slots(report, tmp_path):
    with pytest.raises(ValueError, match="expected 'value' or one of"):
        viz.plot_series_distributions(
            report, str(tmp_path / "s.png"), views=("levels",), keys=("C1",), of="variance"
        )


def test_unknown_kind_is_rejected(report, tmp_path):
    with pytest.raises(ValueError, match="expected 'hist' or 'ecdf'"):
        viz.plot_series_distributions(
            report, str(tmp_path / "s.png"), views=("levels",), keys=("C1",), kind="violin"
        )


def test_ecdf_kind_draws_a_step_to_one(report, capture, tmp_path):
    viz.plot_series_distributions(
        report,
        str(tmp_path / "series.png"),
        views=("levels",),
        keys=("C1",),
        of="value",
        kind="ecdf",
    )
    ax = only_figure(capture).axes[0]
    assert n_bars(ax) == 0
    steps = [line for line in ax.lines if len(line.get_ydata()) > 2]
    assert steps, "the ecdf itself is a step line"
    assert float(np.asarray(steps[0].get_ydata())[-1]) == pytest.approx(1.0)
    assert float(np.asarray(steps[0].get_ydata())[0]) == pytest.approx(1.0 / 48.0)


def test_real_active_zero_renders_as_a_zero_and_missing_data_is_annotated(
    report, capture, tmp_path
):
    """Z2 is active but identically zero; C3 has no eligible world at all."""
    viz.plot_series_distributions(
        report, str(tmp_path / "series.png"), views=("levels",), keys=("Z2", "C3"), of="value"
    )
    fig = only_figure(capture)
    zero_panel, empty_panel = fig.axes[0], fig.axes[1]

    assert "Z2" in title_of(zero_panel)
    assert len(zero_panel.patches) > 0, "an active structural zero is a real value"
    assert axis_texts(zero_panel) == []
    values = report["levels"].series.values("Z2")
    assert values.size == 48 and np.all(values == 0.0)
    assert f"n={values.size:,} values" in title_of(zero_panel)

    assert "C3" in title_of(empty_panel)
    assert axis_texts(empty_panel) == ["No eligible observations"]
    assert n_bars(empty_panel) == 0


def test_a_view_with_no_difference_to_take_is_annotated_as_such(capture, tmp_path):
    """One time step: the worlds are eligible, the difference view is empty."""
    one_step = data_diagnostics(toy_corpus(n_time=1, dead_last_treatment=False))
    assert one_step["differences"].series.n_time_steps == 0

    viz.plot_series_distributions(
        one_step, str(tmp_path / "diff.png"), views=("differences",), keys=("C1",)
    )
    ax = only_figure(capture).axes[0]
    assert axis_texts(ax) == ["No difference observations"]
    assert n_bars(ax) == 0

    capture.clear()
    viz.plot_series_distributions(
        one_step, str(tmp_path / "levels.png"), views=("levels",), keys=("C1",)
    )
    ax = only_figure(capture).axes[0]
    assert axis_texts(ax) == [], "the same series is fine in levels"
    assert n_bars(ax) > 0


def test_empty_lag_axis_reports_no_valid_lags_without_claiming_ineligibility(capture, tmp_path):
    one_step = data_diagnostics(toy_corpus(n_time=1, dead_last_treatment=False))
    assert one_step.lags == ()

    viz.plot_temporal_diagnostics(
        one_step, str(tmp_path / "temporal.png"), views=("levels",), keys=("C1",)
    )
    fig = only_figure(capture)
    assert image_axes(fig) == []
    assert sum("No valid lags" in axis_texts(ax) for ax in fig.axes) == 2
    roughness = fig.axes[2]
    assert "No valid observations" in axis_texts(roughness)
    assert "No eligible observations" not in axis_texts(roughness)


def test_latent_series_are_labelled_as_retained_truth(report, capture, tmp_path):
    viz.plot_series_distributions(
        report, str(tmp_path / "series.png"), views=("levels",), keys=("D1", "C1")
    )
    titles = [title_of(ax) for ax in only_figure(capture).axes]
    assert any(title.startswith("D1 (retained latent truth)") for title in titles)
    assert any(title.startswith("C1 —") for title in titles)


# ---------------------------------------------------------------------------
# 5. Temporal: heatmaps, missing lags, real zero roughness
# ---------------------------------------------------------------------------


def test_temporal_heatmaps_are_series_by_lag(report, capture, tmp_path):
    viz.plot_temporal_diagnostics(
        report, str(tmp_path / "temporal.png"), views=("levels",), keys=("C1", "Z1", "Y")
    )
    fig = only_figure(capture)
    panels = image_axes(fig)
    assert len(panels) == 2
    temporal = report["levels"].temporal
    keys = list(report.keys)
    index = [keys.index(key) for key in ("C1", "Z1", "Y")]
    for ax, metric in zip(panels, ("acf", "lag_xi")):
        drawn = ax.images[0].get_array()
        assert drawn.shape == (3, len(temporal.lags))
        expected = temporal.matrix(metric, quantile=0.5)[index, :]
        assert np.allclose(np.ma.filled(drawn, np.nan), expected, equal_nan=True)
    assert [line.get_text() for line in panels[0].get_xticklabels()][0] == "h1"


def test_temporal_panel_without_a_valid_lag_says_so(report, capture, tmp_path):
    viz.plot_temporal_diagnostics(
        report, str(tmp_path / "temporal.png"), views=("levels",), keys=("C3",)
    )
    fig = only_figure(capture)
    assert image_axes(fig) == [], "no heatmap is drawn when nothing is valid"
    assert [ax for ax in fig.axes if "No valid lags" in axis_texts(ax)]
    roughness = fig.axes[2]
    assert n_bars(roughness) == 0
    assert "No eligible observations" in axis_texts(roughness)
    assert axis_texts(roughness).count("N/A") == 2  # roughness and spike


def test_flat_series_has_a_real_zero_roughness_bar(report, capture, tmp_path):
    """Z2 is identically zero: roughness and spike are 0, not missing."""
    viz.plot_temporal_diagnostics(
        report, str(tmp_path / "temporal.png"), views=("levels",), keys=("Z2",)
    )
    roughness = only_figure(capture).axes[2]
    assert len(roughness.patches) == 2
    assert [patch.get_height() for patch in roughness.patches] == [0.0, 0.0]
    assert "N/A" not in axis_texts(roughness)


def test_roughness_bars_are_medians_over_valid_worlds(report, capture, tmp_path):
    viz.plot_temporal_diagnostics(
        report, str(tmp_path / "temporal.png"), views=("levels",), keys=("C1",)
    )
    roughness = only_figure(capture).axes[2]
    series = report["levels"].series
    column = series.keys.index("C1")
    for patch, slot in zip(roughness.patches, ("roughness", "spike")):
        values = series.slot(slot)[:, column]
        valid = series.slot_valid[:, column, list(viz.SERIES_SLOTS).index(slot)]
        assert patch.get_height() == pytest.approx(np.median(values[valid]))


# ---------------------------------------------------------------------------
# 6. Contributions
# ---------------------------------------------------------------------------


def test_contribution_bars_are_the_selected_cut_with_coverage(report, capture, tmp_path):
    viz.plot_contribution_diagnostics(report, str(tmp_path / "contrib.png"))
    fig = only_figure(capture)
    ax = fig.axes[0]
    budget = report.contributions
    cut = [d.key for d in budget.descriptors if d.sibling_set == "top"]
    assert [label.get_text() for label in ax.get_yticklabels()] == cut
    assert len(ax.patches) == len(cut)
    for patch, key in zip(ax.patches, cut):
        stats, _ = budget.stats(key, measure="share", mode="net")
        assert patch.get_width() == pytest.approx(stats["mean"])
    assert all("(3 worlds)" in text for text in axis_texts(ax))
    assert "complete 'top' cut against Y" in [t.get_text() for t in fig.texts]


def test_micro_weighting_plots_the_pooled_value(report, capture, tmp_path):
    viz.plot_contribution_diagnostics(
        report,
        str(tmp_path / "contrib.png"),
        sibling_set="treatment_children",
        weighting="micro",
        measure="total",
    )
    ax = only_figure(capture).axes[0]
    budget = report.contributions
    cut = [d.key for d in budget.descriptors if d.sibling_set == "treatment_children"]
    for patch, key in zip(ax.patches, cut):
        stats, _ = budget.stats(key, measure="total", mode="net", weighting="micro")
        assert patch.get_width() == pytest.approx(stats["value"])
    assert "net total (micro, unconditional)" == ax.get_xlabel()


def test_partial_projection_reports_the_omitted_siblings(report, capture, tmp_path):
    viz.plot_contribution_diagnostics(
        report, str(tmp_path / "contrib.png"), keys=("treatment_total",)
    )
    fig = only_figure(capture)
    notes = [t.get_text() for t in fig.texts]
    assert any(
        "partial projection: 4 sibling row(s) omitted" in note and "do not add up to Y" in note
        for note in notes
    ), notes


def test_inactive_row_is_a_real_zero_unconditionally_and_na_when_conditioned(
    report, capture, tmp_path
):
    """C3 is padded off: its direct contribution is exactly zero everywhere."""
    viz.plot_contribution_diagnostics(
        report,
        str(tmp_path / "contrib.png"),
        sibling_set="treatment_direct_children",
        keys=("C3_direct_y",),
    )
    ax = only_figure(capture).axes[0]
    assert len(ax.patches) == 1
    assert ax.patches[0].get_width() == 0.0
    assert axis_texts(ax) == ["0 (3 worlds)", "No active worlds for selected key(s)"]

    capture.clear()
    viz.plot_contribution_diagnostics(
        report,
        str(tmp_path / "contrib2.png"),
        sibling_set="treatment_direct_children",
        keys=("C3_direct_y",),
        population="conditional_on_active",
    )
    ax = only_figure(capture).axes[0]
    assert n_bars(ax) == 0
    assert axis_texts(ax) == ["N/A (0 worlds)", "No active worlds for selected key(s)"]


def test_unavailable_basis_is_named(report, tmp_path):
    assert "observed_path_treatment" not in report.contribution_bases
    with pytest.raises(ValueError, match="observed_path_treatment' is not available"):
        viz.plot_contribution_diagnostics(
            report, str(tmp_path / "c.png"), basis="observed_path_treatment"
        )


def test_unknown_sibling_set_lists_the_cuts(report, tmp_path):
    with pytest.raises(ValueError, match="unknown sibling_set 'treatments'"):
        viz.plot_contribution_diagnostics(report, str(tmp_path / "c.png"), sibling_set="treatments")


def test_unknown_measure_is_rejected_by_the_budget(report, tmp_path):
    with pytest.raises(ValueError, match="unknown measure/mode"):
        viz.plot_contribution_diagnostics(report, str(tmp_path / "c.png"), measure="fraction")


# ---------------------------------------------------------------------------
# 7. Selector validation
# ---------------------------------------------------------------------------


def test_caller_passed_empty_selectors_are_rejected(report, tmp_path):
    path = str(tmp_path / "x.png")
    with pytest.raises(ValueError, match="views is empty"):
        viz.plot_dependence_matrices(report, path, views=())
    with pytest.raises(ValueError, match="metrics is empty"):
        viz.plot_dependence_matrices(report, path, metrics=())
    with pytest.raises(ValueError, match="keys is empty"):
        viz.plot_dependence_matrices(report, path, keys=())
    with pytest.raises(ValueError, match="keys is empty"):
        viz.plot_series_distributions(report, path, keys=())
    with pytest.raises(ValueError, match="keys is empty"):
        viz.plot_temporal_diagnostics(report, path, keys=())
    with pytest.raises(ValueError, match="keys is empty"):
        viz.plot_vif_diagnostics(report, path, keys=())
    with pytest.raises(ValueError, match="keys is empty"):
        viz.plot_contribution_diagnostics(report, path, keys=())


def test_bare_strings_are_rejected(report, tmp_path):
    path = str(tmp_path / "x.png")
    with pytest.raises(TypeError, match="not the bare string"):
        viz.plot_dependence_matrices(report, path, views="levels")
    with pytest.raises(TypeError, match="not the bare string"):
        viz.plot_dependence_matrices(report, path, metrics="xi")
    with pytest.raises(TypeError, match="not the bare string"):
        viz.plot_dependence_matrices(report, path, keys="C1")
    with pytest.raises(TypeError, match="not the bare string"):
        viz.plot_vif_diagnostics(report, path, keys="C1")


def test_unknown_names_list_the_valid_ones(report, tmp_path):
    path = str(tmp_path / "x.png")
    with pytest.raises(ValueError, match=r"unknown metrics \['kendall'\]; expected"):
        viz.plot_dependence_matrices(report, path, metrics=("kendall",))
    with pytest.raises(ValueError, match=r"unknown views \['level'\]; expected"):
        viz.plot_dependence_matrices(report, path, views=("level",))
    with pytest.raises(ValueError, match=r"unknown keys \['C9'\]; expected"):
        viz.plot_dependence_matrices(report, path, keys=("C9",))
    with pytest.raises(ValueError, match="repeats an entry"):
        viz.plot_dependence_matrices(report, path, keys=("C1", "C1"))


def test_quantile_must_be_a_level(report, tmp_path):
    with pytest.raises(ValueError, match=r"within \[0, 1\]"):
        viz.plot_dependence_matrices(report, str(tmp_path / "x.png"), quantile=1.5)


# ---------------------------------------------------------------------------
# 8. Bounds
# ---------------------------------------------------------------------------


def test_heatmap_key_bound_is_enforced_and_the_figure_stays_capped(wide_report, capture, tmp_path):
    assert len(wide_report.keys) == viz.MAX_HEATMAP_KEYS
    viz.plot_dependence_matrices(
        wide_report, str(tmp_path / "dep.png"), views=("levels",), metrics=("pearson",)
    )
    fig = only_figure(capture)
    width, height = fig.get_size_inches()
    assert width <= 24.0 and height <= 24.0
    ax = image_axes(fig)[0]
    # deterministic tick thinning above 32 keys
    assert list(ax.get_xticks()) == list(range(0, 64, 2))
    assert [label.get_text() for label in ax.get_yticklabels()][:2] == ["C1", "C3"]


def test_over_the_heatmap_key_bound_raises(tmp_path):
    over = data_diagnostics(
        toy_corpus(
            n_worlds=1, n_time=10, n_treatments=41, n_covariates=20, dead_last_treatment=False
        ),
        views=("levels",),
    )
    assert len(over.keys) == viz.MAX_HEATMAP_KEYS + 1
    for plot in (
        viz.plot_dependence_matrices,
        viz.plot_temporal_diagnostics,
        viz.plot_vif_diagnostics,
    ):
        with pytest.raises(ValueError, match="exceed this figure's bound of 64"):
            plot(over, str(tmp_path / "x.png"), views=("levels",))


def test_panel_bound_is_enforced_and_the_grid_stays_capped(capture, tmp_path):
    full = data_diagnostics(toy_corpus(), scopes=("nodes", "decomposition"))
    assert len(full.keys) == 23
    with pytest.raises(ValueError, match="panels exceed this figure's bound of 24"):
        viz.plot_series_distributions(full, str(tmp_path / "s.png"), keys=full.keys[:13])

    viz.plot_series_distributions(full, str(tmp_path / "s.png"), keys=full.keys[:12], of="std")
    fig = only_figure(capture)
    width, height = fig.get_size_inches()
    assert width <= 18.0 and height <= 18.0
    assert len(fig.axes) == 24
    rows = {round(ax.get_position().y0, 4) for ax in fig.axes}
    columns = {round(ax.get_position().x0, 4) for ax in fig.axes}
    assert len(columns) <= 4 and len(rows) <= 6


def test_over_the_contribution_row_bound_raises(tmp_path):
    wide = data_diagnostics(
        toy_corpus(n_worlds=1, n_time=10, n_treatments=25, dead_last_treatment=False),
        views=("levels",),
    )
    with pytest.raises(ValueError, match="exceed this figure's bound of 24"):
        viz.plot_contribution_diagnostics(
            wide, str(tmp_path / "c.png"), sibling_set="treatment_direct_children"
        )


# ---------------------------------------------------------------------------
# 9. Purity: one savefig, no stray files, no open figures, no mutation
# ---------------------------------------------------------------------------


PLOT_CALLS = (
    ("plot_dependence_matrices", {"metrics": ("pearson",)}),
    ("plot_series_distributions", {"keys": ("C1", "Z1")}),
    ("plot_temporal_diagnostics", {"keys": ("C1", "Z1")}),
    ("plot_vif_diagnostics", {"keys": ("C1", "Z1")}),
    ("plot_contribution_diagnostics", {}),
)


@pytest.mark.parametrize(("name", "kwargs"), PLOT_CALLS)
def test_plot_is_pure_and_saves_exactly_once(report, capture, tmp_path, name, kwargs):
    import matplotlib.pyplot as plt

    before = fingerprint(report)
    path = tmp_path / f"{name}.png"
    assert getattr(viz, name)(report, str(path), **kwargs) is None

    assert [call[1] for call in capture] == [str(path)]
    assert list(tmp_path.iterdir()) == [], "savefig was intercepted: nothing else is written"
    assert plt.get_fignums() == [], "the figure must be closed"
    assert fingerprint(report) == before


def test_figures_write_real_non_empty_files(report, tmp_path):
    written = []
    for name, kwargs in PLOT_CALLS:
        path = tmp_path / f"{name}.png"
        getattr(viz, name)(report, str(path), **kwargs)
        assert path.stat().st_size > 0
        written.append(path.name)
    assert sorted(p.name for p in tmp_path.iterdir()) == sorted(written)
