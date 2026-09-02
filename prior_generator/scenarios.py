"""Named audit scenarios — each isolates one causal pathway.

A :class:`Scenario` fixes graph sizes and per-edge-type arrow budgets so a
decomposition failure can be traced to the pathway that broke. The five
default scenarios (ported from structural-pfn's inspection-dataset tooling)
cover: pure direct effects, demand-confounded spend, promo-driven spend,
channel halo, and everything at once.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from .presets import make_scm_prior
from .sampler import SCMPrior


@dataclass(frozen=True)
class Scenario:
    """A named world recipe: sizes + edge budgets + connectivity policy.

    Attributes
    ----------
    name : str
        Short identifier (used as the bundle folder title).
    purpose : str
        What the scenario isolates — written into ``description.txt``.
    n_treatments, n_covariates, n_latent : int
        Channels / controls / demand factors (all active).
    connect_all : bool
        When True, every node must have a directed path to Y (no isolated
        nulls). When False, fully-isolated null nodes are allowed as
        deliberate zero-attribution traps; dead-ends are never allowed.
    edge_budget : dict
        Per-edge-type arrow budgets (see ``SCMPrior.edge_budget``).
    connect_all_edge_budget : dict
        Per-edge-type budget entries that OVERRIDE ``edge_budget`` when
        connectivity is forced (``prior(connect_all=True)``). A scenario tuned
        for isolated-null traps can budget an edge type so thinly that "every
        node reaches Y" is unsatisfiable — a control with no ``zb``/``zc``/
        ``zz``/``dz`` arrow available cannot reach Y at any draw count — so
        forcing connectivity has to substitute a budget that admits it.
        Entries are chosen empirically as the smallest change reaching a
        per-draw feasible fraction well above what ``sample_scm``'s 2000-round
        graph search needs; see the inline notes on each scenario.
    """

    name: str
    purpose: str
    n_treatments: int
    n_covariates: int
    n_latent: int
    connect_all: bool
    edge_budget: dict[str, int | tuple[int, int]] = field(default_factory=dict)
    connect_all_edge_budget: dict[str, int | tuple[int, int]] = field(default_factory=dict)

    def prior(
        self, n_time_steps: int = 104, seed: int = 0, *, connect_all: bool | None = None
    ) -> SCMPrior:
        """Build the scenario's validated ``SCMPrior``.

        ``n_cells``/``draws_per_cell`` are placebo values — single-SCM sampling
        goes through :func:`prior_generator.worlds.sample_scm`, which
        bypasses the corpus loop — but they keep ``validate()`` happy.

        Parameters
        ----------
        n_time_steps : int
            Weeks per world.
        seed : int
            Config seed (the WORLD's seed is ``sample_scm``'s own argument).
        connect_all : bool, optional
            The connectivity policy the world will be sampled under; defaults
            to the scenario's own ``connect_all``. When it is True,
            ``connect_all_edge_budget`` overrides ``edge_budget``, so the
            returned config is one under which "every node reaches Y" is
            actually attainable. Pass the SAME value here and to
            ``sample_scm(connect_all=...)``: a forced-connectivity world drawn
            from an unforced config is exactly the impossible search this
            field exists to prevent.
        """
        forced = self.connect_all if connect_all is None else connect_all
        edge_budget = dict(self.edge_budget)
        if forced:
            edge_budget.update(self.connect_all_edge_budget)
        return make_scm_prior(
            n_treatments=self.n_treatments,
            n_covariates=self.n_covariates,
            n_latent=self.n_latent,
            edge_budget=edge_budget,
            n_time_steps=n_time_steps,
            n_cells=2,
            draws_per_cell=1,
            seed=seed,
        )


SCENARIOS: tuple[Scenario, ...] = (
    Scenario(
        name="direct_only",
        purpose=(
            "Pure C→Y channels; NO channel-input interactions (dc=zc=cc=dz=zz=0). "
            "Controls/demand act on the baseline only. The model should attribute "
            "everything to direct contributions + baseline; indirect effects are "
            "exactly zero."
        ),
        n_treatments=4,
        n_covariates=2,
        n_latent=1,
        connect_all=True,
        edge_budget={
            "cy": (4, 4),
            "db": (1, 1),
            "zb": (2, 2),
            "dc": 0,
            "zc": 0,
            "cc": 0,
            "dz": 0,
            "zz": 0,
        },
    ),
    Scenario(
        name="confounded_spend",
        purpose=(
            "Classic MMM confounding: latent demand drives BOTH spend (D→C) and the "
            "baseline (D→B). The dc indirect column carries the demand-through-spend "
            "effect; naive attribution overcredits channels."
        ),
        n_treatments=4,
        n_covariates=2,
        n_latent=2,
        connect_all=True,
        edge_budget={
            "cy": (4, 4),
            "dc": (3, 3),
            "db": (2, 2),
            "zb": (2, 2),
            "zc": 0,
            "cc": 0,
            "dz": 0,
            "zz": 0,
        },
    ),
    Scenario(
        name="promo_drives_spend",
        purpose=(
            "Observed controls push spend (Z→C, e.g. promo calendar triggers media) "
            "AND the baseline (Z→B), with demand also moving the controls (D→Z). "
            "The zc indirect column carries the control-through-spend effect."
        ),
        n_treatments=4,
        n_covariates=3,
        n_latent=1,
        connect_all=True,
        edge_budget={
            "cy": (4, 4),
            "zc": (3, 3),
            "zb": (3, 3),
            "db": (1, 1),
            "dz": (2, 2),
            "dc": 0,
            "cc": 0,
            "zz": 0,
        },
    ),
    Scenario(
        name="channel_halo",
        purpose=(
            "Channel-to-channel halo (C→C): upstream channels amplify downstream "
            "spend. Channels without a direct C→Y edge are feeders (their outgoing "
            "C→C reaches Y through the cascade) — their DIRECT attribution must "
            "still be zero. The cc indirect column carries the halo effect. May "
            "also keep fully-ISOLATED null nodes (no edges at all) as "
            "zero-attribution traps; dead-ends are never generated — see 'Node "
            "connectivity' in description.txt."
        ),
        n_treatments=5,
        n_covariates=2,
        n_latent=1,
        connect_all=False,
        edge_budget={
            "cy": (3, 3),
            "cc": (3, 3),
            "db": (1, 1),
            "zb": (1, 1),
            "dc": 0,
            "zc": 0,
            "dz": 0,
            "zz": 0,
        },
        # Forced connectivity (``prior(connect_all=True)``): with zc=zz=dz=0 a
        # control's ONLY route to Y is its own Z->B arrow, so zb=(1,1) leaves
        # the second control edgeless — permanently isolated, at any draw
        # count. Budgeting both arrows is the whole fix; nothing else needs to
        # move. The channels already cope: g_cc is strict upper triangular
        # (src index < dst index), so C5 has no outgoing halo arrow and must
        # take one of the three cy arrows, which leaves the two feeders to be
        # covered by three cc arrows. Measured per-draw feasible fraction
        # (sample_g_additive + worlds.node_status, 4000 draws): 0.00% as
        # budgeted above -> 17.07% with zb=(2,2), i.e. ~1e-163 odds of
        # exhausting sample_scm's 2000 graph rounds.
        connect_all_edge_budget={"zb": (2, 2)},
    ),
    Scenario(
        name="kitchen_sink",
        purpose=(
            "Everything at once at sparse budgets: confounded spend, "
            "control-driven spend, channel halo, control chains. The hardest "
            "decomposition; all three indirect sources are live. May keep "
            "fully-ISOLATED null channels/controls (no edges at all) as "
            "zero-attribution traps; every non-isolated node has a real path to "
            "Y and dead-ends are never generated. See 'Node connectivity' in "
            "description.txt."
        ),
        n_treatments=6,
        n_covariates=4,
        n_latent=2,
        connect_all=False,
        edge_budget={
            "cy": (4, 4),
            "dc": (2, 2),
            "zc": (1, 3),
            "cc": (1, 2),
            "zb": (2, 2),
            "zz": (1, 2),
        },
        # Forced connectivity: nothing here is structurally impossible, only
        # rare. The binding constraint is cc: cy=(4,4) of 6 channels leaves 2
        # feeders, each of which needs an OUTGOING halo arrow into a channel
        # that reaches Y, and a (1, 2) budget can draw a single arrow — which
        # cannot cover two feeders at all. Widening cc keeps the scenario's
        # character (sparse cy, so the zero-direct-attribution feeders stay)
        # and is the smallest single-key change that clears the bar; raising
        # cy instead would buy feasibility by DELETING a feeder. Measured
        # per-draw feasible fraction (sample_g_additive + worlds.node_status,
        # 20000 draws): 0.85% as budgeted above -> 1.94% at cc=(2,2), 3.25% at
        # cc=(2,3), 4.46% at cc=(3,3), 5.89% at cc=(3,4), i.e. ~1e-53 odds of
        # exhausting sample_scm's 2000 graph rounds.
        connect_all_edge_budget={"cc": (3, 4)},
    ),
)
