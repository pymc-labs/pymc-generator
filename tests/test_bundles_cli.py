"""Audit bundles and the CLI — the contract as an auditor reads it off disk.

A bundle is this package's human-facing product, so the assertions here are
on-disk ones: ``true_components.csv`` must be the FULL additive truth (its
component columns sum to ``dataset.csv``'s ``sales_Y``, the very identity
``SCM.identity_error()`` scores), the file set must be complete,
``description.txt`` must name the formula it reports the error of, and
``--require-path-to-y`` must be satisfiable for every shipped scenario.

Worlds are deliberately tiny (24-40 weeks, ``plots=False`` unless the figures
are the thing under test): none of these properties depends on the horizon.
"""

from __future__ import annotations

import subprocess
import sys

import numpy as np
import pandas as pd
import pytest

import prior_generator as pg
from prior_generator.bundles import write_scenario_bundles, write_scm_bundle
from prior_generator.describe import describe_scm
from prior_generator.sampler import SCMPrior
from prior_generator.scenarios import SCENARIOS, Scenario

#: ``true_components.csv`` columns that are NOT additive terms of sales: the
#: index, the reconstruction total itself, and the latent-input diagnostics
#: (demand reaches sales through the confounder/indirect columns, base channels
#: through the direct ones — adding them would double-count).
NON_ADDITIVE_COLUMNS = ("week", "sales_reconstructed")
NON_ADDITIVE_PREFIXES = ("demand_", "channel_base_")

TEXT_FILES = frozenset(
    {
        "dataset.csv",
        "true_components.csv",
        "true_contribution.csv",
        "description.txt",
        "dag.dot",
    }
)
PLOT_FILES = frozenset({"dag.png", "timeseries.png", "decomposition.png", "channels.png"})


def _world(
    scenario: int, *, n_time_steps: int = 24, seed: int = 3, connect_all: bool | None = None
):
    """Sample one scenario world exactly the way the bundle writer does."""
    sc = SCENARIOS[scenario]
    forced = sc.connect_all if connect_all is None else connect_all
    return pg.sample_scm(
        sc.prior(n_time_steps=n_time_steps, seed=seed, connect_all=forced),
        seed=seed,
        connect_all=forced,
        name=sc.name,
        purpose=sc.purpose,
    )


def test_exported_dag_keeps_the_intercept_parentless(tmp_path):
    from prior_generator.worlds import edges_with_coeffs

    world = _world(4)
    edges = edges_with_coeffs(world.g, world.params)
    outcome_edges = [edge for edge in edges if edge[0] in {"db", "zb"}]
    assert outcome_edges
    assert all(target == "Y" for _, _, target, _ in outcome_edges)
    assert not any(target == "B" for _, _, target, _ in edges)
    out = write_scm_bundle(world, tmp_path / "dag", plots=False)
    dot = (out / "dag.dot").read_text()
    for _, source, _, _ in outcome_edges:
        assert f"{source} -> Y" in dot.replace('"', "")


def _additive_columns(frame: pd.DataFrame) -> list[str]:
    return [
        column
        for column in frame.columns
        if column not in NON_ADDITIVE_COLUMNS and not column.startswith(NON_ADDITIVE_PREFIXES)
    ]


def _node_statuses(description: str) -> dict[str, str]:
    """Parse the ``Node connectivity`` census out of a ``description.txt``."""
    found = dict(
        pair.split("=")
        for line in description.splitlines()
        for pair in line.split()
        if "=" in pair and pair.split("=")[1] in ("connected", "isolated", "dead-end")
    )
    assert found, "description.txt carries no node connectivity census"
    return found


@pytest.fixture(scope="module")
def kitchen_sink_bundle(tmp_path_factory):
    """kitchen_sink (every edge type live) written once, text/CSV only."""
    world = _world(4)
    out = write_scm_bundle(world, tmp_path_factory.mktemp("bundle") / "kitchen_sink", plots=False)
    return world, out


def test_true_components_columns_reconstruct_sales(kitchen_sink_bundle):
    """The exported additive columns sum to sales_Y — sales_noise included.

    This is the whole point of ``true_components.csv``: a reader who adds up
    the component columns must land on the observed outcome. Omitting the
    observation noise leaves a residual of ``max|sales_noise|`` — invisible in
    ``sales_reconstructed`` (which is computed, not summed) and contradicted by
    the ``max |error| = ~1e-15`` printed next to it in ``description.txt``.
    """
    world, out = kitchen_sink_bundle
    dataset = pd.read_csv(out / "dataset.csv")
    truth = pd.read_csv(out / "true_components.csv")

    additive = _additive_columns(truth)
    assert "sales_noise" in additive
    assert "baseline_intrinsic" in additive

    sales = dataset["sales_Y"].to_numpy()
    total = truth[additive].to_numpy().sum(axis=1)
    # float64 CSV round-trip on O(sales) magnitudes; the in-memory identity is ~1e-15.
    np.testing.assert_allclose(total, sales, atol=1e-9)
    np.testing.assert_allclose(truth["sales_reconstructed"].to_numpy(), sales, atol=1e-9)

    # Keep the test honest: a zero-noise world would satisfy the sum with or
    # without the column, and would prove nothing.
    noise = truth["sales_noise"].to_numpy()
    assert np.abs(noise).max() > 1e-6
    without_noise = np.abs(total - noise - sales).max()
    assert without_noise == pytest.approx(np.abs(noise).max(), rel=1e-6)

    # And it is the same identity the world reports on itself.
    assert world.identity_error() < 1e-9


def test_true_components_covers_every_reconstruction_term(kitchen_sink_bundle):
    """Every term of ``SCM.reconstruction()`` is exported as its own column."""
    world, out = kitchen_sink_bundle
    truth = pd.read_csv(out / "true_components.csv")
    expected = {"baseline_intrinsic", "sales_noise", "indirect_cc", "indirect_zc", "indirect_dc"}
    expected |= {f"confounder_contribution_D{j + 1}" for j in range(world.n_latent)}
    expected |= {f"control_contribution_Z{m + 1}" for m in range(world.n_covariates)}
    expected |= {f"contribution_C{k + 1}" for k in range(world.n_treatments)}
    assert set(_additive_columns(truth)) == expected


def test_bundle_writes_the_text_file_set(kitchen_sink_bundle):
    _world_, out = kitchen_sink_bundle
    assert {path.name for path in out.iterdir()} == set(TEXT_FILES)


def test_bundle_writes_the_figures_when_plots_are_enabled(tmp_path):
    import matplotlib

    matplotlib.use("Agg")

    out = write_scm_bundle(_world(0), tmp_path / "direct_only", plots=True)
    assert {path.name for path in out.iterdir()} == set(TEXT_FILES | PLOT_FILES)


def test_forced_connectivity_connects_every_node_in_every_scenario(tmp_path):
    """``require_path_to_y=True`` must be attainable for all five scenarios.

    channel_halo budgets ``zb=(1, 1)`` with ``zc=zz=dz=0``, so one of its two
    controls has no route to Y at all: forcing connectivity on that budget is
    unsatisfiable at any draw count, not merely unlikely. The scenarios carry
    a ``connect_all_edge_budget`` that substitutes a feasible budget, and this
    asserts the auditor's view of the result — every node reported connected.
    """
    written = write_scenario_bundles(
        tmp_path / "forced",
        n_time_steps=24,
        seed=3,
        require_path_to_y=True,
        plots=False,
        verbose=False,
    )
    assert len(written) == len(SCENARIOS)
    for out, sc in zip(written, SCENARIOS, strict=True):
        assert TEXT_FILES <= {path.name for path in out.iterdir()}
        status = _node_statuses((out / "description.txt").read_text())
        assert len(status) == sc.n_treatments + sc.n_covariates + sc.n_latent
        assert set(status.values()) == {"connected"}, (sc.name, status)


def test_scenario_prior_substitutes_the_budget_only_when_forced():
    """``connect_all_edge_budget`` overrides ``edge_budget``, and only then."""
    halo = SCENARIOS[3]
    assert halo.name == "channel_halo" and not halo.connect_all
    assert halo.connect_all_edge_budget == {"zb": (2, 2)}

    # The default follows the scenario's own policy: traps intact.
    assert halo.prior(n_time_steps=24, seed=0).edge_budget["zb"] == (1, 1)
    assert halo.prior(n_time_steps=24, seed=0, connect_all=False).edge_budget["zb"] == (1, 1)
    forced = halo.prior(n_time_steps=24, seed=0, connect_all=True).edge_budget
    assert forced["zb"] == (2, 2)
    # Only the named type moves; everything else is the scenario's own budget.
    assert {et: v for et, v in forced.items() if et != "zb"} == {
        et: v for et, v in halo.edge_budget.items() if et != "zb"
    }

    # A scenario that needs no substitution is unaffected by either policy.
    direct = SCENARIOS[0]
    assert direct.connect_all_edge_budget == {}
    assert direct.prior(n_time_steps=24, seed=0, connect_all=True).edge_budget == dict(
        direct.edge_budget
    )


def test_infeasible_forced_scenario_raises_before_writing_anything(tmp_path):
    """The pre-flight is what makes the writer atomic.

    Feasibility depends only on ``(edge_budget, connect_all)``, so it is known
    before a single byte is written — and discovering it mid-loop would strand
    a partial inspection set on disk. The infeasible scenario is placed LAST
    precisely so an un-preflighted writer would have written the first three.
    """
    impossible = Scenario(
        name="impossible_control",
        purpose="A control with no zb/zc/zz/dz route to Y under forced connectivity.",
        n_treatments=2,
        n_covariates=2,
        n_latent=1,
        connect_all=False,
        edge_budget={
            "cy": (2, 2),
            "db": (1, 1),
            "zb": (1, 1),
            "dc": 0,
            "zc": 0,
            "dz": 0,
            "zz": 0,
            "cc": 0,
        },
    )
    out_root = tmp_path / "partial"
    with pytest.raises(RuntimeError, match="impossible_control"):
        write_scenario_bundles(
            out_root,
            scenarios=(*SCENARIOS[:3], impossible),
            n_time_steps=24,
            seed=3,
            require_path_to_y=True,
            plots=False,
            verbose=False,
        )
    assert not out_root.exists()

    # Unforced, that same scenario is legal — the isolated control is the trap.
    written = write_scenario_bundles(
        tmp_path / "unforced",
        scenarios=(impossible,),
        n_time_steps=24,
        seed=3,
        require_path_to_y=False,
        plots=False,
        verbose=False,
    )
    assert _node_statuses((written[0] / "description.txt").read_text())["Z2"] == "isolated"


def test_description_identity_names_the_formula_it_scores(kitchen_sink_bundle):
    """The printed equation must be the one ``identity_error()`` evaluates."""
    _world_, out = kitchen_sink_bundle
    text = (out / "description.txt").read_text()
    identity = text.split("Decomposition identity ")[1].split("max |error|")[0]
    for term in (
        "baseline_intrinsic",
        "sales_noise",
        "confounder",
        "control",
        "direct contributions",
        "indirect_by_source",
    ):
        assert term in identity, (term, identity)


def test_description_texture_reports_the_drawn_flags():
    """No hardcoded preset label: a zero-texture world must not claim one.

    A bare ``SCMPrior`` defaults every high-frequency range to ``(0.0, 0.0)``,
    so every ``use_*`` flag is off. The texture section must describe THAT
    world, not the preset the scenarios happen to use.
    """
    cfg = SCMPrior(
        n_treatments=3,
        n_covariates=2,
        n_latent=1,
        n_treatments_active_range=(3, 3),
        n_covariates_active_range=(2, 2),
        n_latent_active_range=(1, 1),
        n_time_steps=24,
        n_cells=2,
        draws_per_cell=1,
        seed=0,
    )
    flat = pg.sample_scm(cfg, seed=5, connect_all=True, name="zero_texture")
    assert not np.asarray(flat.params["use_hf"]).any()
    assert not np.asarray(flat.params["use_pulse"]).any()

    flat_text = describe_scm(flat)
    assert "diverse" not in flat_text
    assert "Texture prior (drawn: channel hf 0/3, channel pulse 0/3" in flat_text
    assert "control hf 0/2, control pulse 0/2):" in flat_text

    # A textured world reports its live flags instead.
    textured = describe_scm(_world(0))
    assert "Texture prior (drawn: channel hf 4/4, channel pulse 4/4" in textured


def _cli(*argv: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, "-m", "prior_generator.cli", *argv],
        capture_output=True,
        text=True,
    )


def test_cli_help_documents_the_connectivity_flag():
    result = _cli("--help")
    assert result.returncode == 0, result.stderr
    assert "--require-path-to-y" in result.stdout
    # The flag substitutes a budget; the help must say so.
    assert "edge budget" in result.stdout


def test_cli_rejects_an_unknown_argument():
    result = _cli("--not-an-option")
    assert result.returncode != 0
    assert "unrecognized arguments" in result.stderr


def test_cli_writes_the_full_inspection_set(tmp_path):
    out_root = tmp_path / "datasets"
    result = _cli("--out", str(out_root), "--seed", "3", "--t", "24", "--no-plots")
    assert result.returncode == 0, result.stderr
    assert (out_root / "README.md").is_file()
    assert {path.name for path in out_root.iterdir()} == {"README.md"} | {
        str(idx) for idx in range(len(SCENARIOS))
    }
    for idx in range(len(SCENARIOS)):
        assert {path.name for path in (out_root / str(idx)).iterdir()} == set(TEXT_FILES)
