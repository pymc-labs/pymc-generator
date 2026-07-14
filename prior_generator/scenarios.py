"""Named audit scenarios — each isolates one causal pathway.

A :class:`Scenario` fixes graph sizes and per-edge-type arrow budgets so a
decomposition failure can be traced to the pathway that broke. The five
default scenarios (ported from structural-pfn's inspection-dataset tooling)
cover: pure direct effects, demand-confounded spend, promo-driven spend,
channel halo, and everything at once.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from .presets import make_world_config
from .sampler import CorpusConfig


@dataclass(frozen=True)
class Scenario:
    """A named world recipe: sizes + edge budgets + connectivity policy.

    Attributes
    ----------
    name : str
        Short identifier (used as the bundle folder title).
    purpose : str
        What the scenario isolates — written into ``description.txt``.
    K, M, J : int
        Channels / controls / demand factors (all active).
    connect_all : bool
        When True, every node must have a directed path to Y (no isolated
        nulls). When False, fully-isolated null nodes are allowed as
        deliberate zero-attribution traps; dead-ends are never allowed.
    edge_budget : dict
        Per-edge-type arrow budgets (see ``CorpusConfig.edge_budget``).
    """

    name: str
    purpose: str
    K: int
    M: int
    J: int
    connect_all: bool
    edge_budget: dict[str, int | tuple[int, int]] = field(default_factory=dict)

    def cfg(self, T: int = 104, seed: int = 0) -> CorpusConfig:
        """Build the scenario's validated ``CorpusConfig``.

        ``n_cells``/``draws_per_cell`` are placebo values — world sampling
        goes through :func:`prior_generator.worlds.sample_world`, which
        bypasses the corpus loop — but they keep ``validate()`` happy.
        """
        return make_world_config(
            K_max=self.K,
            M_max=self.M,
            J_max=self.J,
            edge_budget=dict(self.edge_budget),
            T=T,
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
        K=4,
        M=2,
        J=1,
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
        K=4,
        M=2,
        J=2,
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
        K=4,
        M=3,
        J=1,
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
        K=5,
        M=2,
        J=1,
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
        K=6,
        M=4,
        J=2,
        connect_all=False,
        edge_budget={
            "cy": (4, 4),
            "dc": (2, 2),
            "zc": (1, 3),
            "cc": (1, 2),
            "zb": (2, 2),
            "zz": (1, 2),
        },
    ),
)
