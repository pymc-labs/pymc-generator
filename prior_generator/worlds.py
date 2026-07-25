"""Single-world sampling: one accepted task, with its full ground truth.

Where :func:`prior_generator.sample_prior_predictive` produces a padded N-task corpus
for training, :func:`sample_scm` draws ONE accepted world at the config's
max sizes and keeps everything a human (or exporter) needs: the active-size
DAG blocks, the drawn SCM parameters, and up to 23 named output series,
including the interventional decomposition truth and shock-schedule audit
outputs when enabled.

The sampling path mirrors ``sampler._generate_corpus_additive``: draw the DAG
+ structure, build the world's PyMC model (``world_model.build_world_model``),
``pm.draw`` candidate worlds, and keep the first that passes the realism
filter. Deterministic given (cfg, seed): one numpy RNG drives the DAG +
structure draws and the per-round pm.draw seeds.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .sampler import SCMPrior, _additive_task_ok, _slice_g_active, sample_g_additive
from .signal_diagnostics import SIGNAL_METRIC_LAYOUT, SIGNAL_METRIC_VERSION, per_channel_signal

#: Adstock family names, indexed by ``params["adstock_family"]``.
ADSTOCK_NAMES = ("none", "geometric", "weibull")

#: Saturation family names, indexed by ``params["sat_family"]``.
SATURATION_NAMES = ("linear", "hill", "logistic", "michaelis_menten", "tanh", "root")

#: Graph outputs kept for every world, including decomposition and optional
#: shock audit paths. Order must match ``build_symbolic_graph``'s outputs.
SCM_OUT_NAMES = (
    "demand",
    "controls",
    "channels",
    "channels_base",
    "baseline",
    "baseline_intrinsic",
    "control_contribution",
    "confounder_contribution",
    "contributions",
    "contributions_observed",
    "indirect_effects",
    "indirect_effects_by_source",
    "sales",
    "channels_unshocked",
    "sales_unshocked",
    "confounding_strength",
    "channel_shock_mask",
    "channel_shock_mask_full",
    "channel_shock_channel",
    "channel_shock_start",
    "channel_shock_length",
    "channel_shock_level_multiplier",
    "channel_shock_level",
)


@dataclass
class SCM:
    """One simulated world: outputs + DAG + drawn parameters + config.

    Attributes
    ----------
    data : dict
        The :data:`SCM_OUT_NAMES` series at active sizes — e.g.
        ``channels (T, K)``, ``sales (T,)``, ``contributions (T, K)``,
        ``indirect_effects_by_source (T, 3)`` in the locked (cc, zc, dc)
        order.
    g : dict
        Active-size DAG blocks (``g_cy``, ``g_dc``, ``g_dz``, ``g_db``,
        ``g_zb``, ``g_zc``, ``g_cc``, ``g_zz``).
    params : dict
        Drawn SCM parameters (edge coefficients, per-node random-walk
        params, per-channel mechanism families and texture).
    cfg : SCMPrior
        The config the world was drawn from.
    name, purpose : str
        Optional labels (set from a :class:`~prior_generator.scenarios.Scenario`)
        used by ``describe_scm`` and the bundle writer.
    extras : dict
        Extra per-world records. With ``cfg.prior_conditioning`` enabled,
        ``extras["prior_cond"]`` holds the drawn ACE intervals
        ``{quantity: (low, width)}`` the mechanism shape priors were narrowed
        to (printed by ``describe_scm``).
    """

    data: dict[str, np.ndarray]
    g: dict[str, np.ndarray]
    params: dict
    cfg: SCMPrior
    name: str = "scm"
    purpose: str = ""
    seed: int | None = None
    extras: dict = field(default_factory=dict)

    @property
    def T(self) -> int:
        return int(self.data["sales"].shape[0])

    @property
    def K(self) -> int:
        return int(self.data["channels"].shape[1])

    @property
    def M(self) -> int:
        return int(self.data["controls"].shape[1])

    @property
    def J(self) -> int:
        return int(self.data["demand"].shape[1])

    def reconstruction(self) -> np.ndarray:
        """Σ of all true components — equals ``sales`` up to float error."""
        d = self.data
        return np.asarray(
            d["baseline_intrinsic"]
            + d["confounder_contribution"].sum(1)
            + d["control_contribution"].sum(1)
            + d["contributions"].sum(1)
            + d["indirect_effects_by_source"].sum(1)
        )

    def identity_error(self) -> float:
        """Max |Σ true components − sales| (float64; ~1e-15 in practice)."""
        return float(np.abs(self.reconstruction() - self.data["sales"]).max())

    def oracle_model(self):
        """The observed-data (NUTS oracle) ``pm.Model`` for THIS world.

        Rebuilds :func:`prior_generator.world_model.build_oracle_model` from
        the world's own structure, config, observables and (when present)
        prior-conditioning intervals, so ``pm.sample(model=world.oracle_model())``
        yields the structure-known posterior on the world's dataset. Requires
        an SCM produced by ``sample_scm`` (which records the structural draw
        in ``extras``). See the oracle guide for caveats.
        """
        from .world_model import build_oracle_model

        if "structural" not in self.extras:
            raise ValueError(
                "this SCM does not carry its structural draw (extras['structural']); "
                "oracle_model() needs a world produced by sample_scm"
            )
        data = {
            "channels": self.data["channels"],
            "controls": self.data["controls"],
            "sales": self.data["sales"],
        }
        if self.cfg.n_channel_shocks:
            data.update(
                {
                    "channel_shock_channel": self.data["channel_shock_channel"],
                    "channel_shock_start": self.data["channel_shock_start"],
                    "channel_shock_length": self.data["channel_shock_length"],
                    "channel_shock_level_multiplier": self.data["channel_shock_level_multiplier"],
                }
            )
        return build_oracle_model(
            self.g,
            self.cfg,
            self.extras["structural"],
            data=data,
            prior_cond=self.extras.get("prior_cond"),
        )

    def signal(self) -> dict[str, np.ndarray]:
        """Per-direct-channel signal metrics (see ``signal_diagnostics``).

        Adds a ``"channel"`` key with the 0-based indices of the direct
        (C→Y) channels the rows refer to.
        """
        d = self.data
        cy_mask = (np.asarray(self.g["g_cy"]) == 1)[None, :]
        per = per_channel_signal(
            d["channels"][None],
            d["contributions"][None],
            d["sales"][None],
            cy_mask,
            l_max=self.cfg.l_max,
            baseline=d["baseline"][None],
            adstock_family=np.asarray(self.params["adstock_family"])[None],
            adstock_alpha=np.asarray(self.params["adstock_alpha"])[None],
            weibull_lam=np.asarray(self.params["weibull_lam"])[None],
            weibull_k=np.asarray(self.params["weibull_k"])[None],
            channel_shock_channel=np.asarray(d["channel_shock_channel"])[None],
            channel_shock_start=np.asarray(d["channel_shock_start"])[None],
            adstock_burn_in=self.cfg.adstock_burn_in,
        )
        per["channel"] = np.nonzero(cy_mask[0])[0].astype(float)
        per["metric_version"] = SIGNAL_METRIC_VERSION
        per["metric_layout"] = SIGNAL_METRIC_LAYOUT
        return per


def path_to_y(g: dict) -> dict[str, bool]:
    """Directed reachability to Y for every node.

    A channel reaches Y via its own C->Y edge or a C->C chain into one; a
    control via Z->B, Z->C into a reaching channel, or a Z->Z chain into a
    reaching control; a demand via D->B, D->C, or D->Z into reaching nodes.
    """
    g_cy = np.asarray(g["g_cy"])
    g_cc = np.asarray(g["g_cc"])
    g_zb = np.asarray(g["g_zb"])
    g_zc = np.asarray(g["g_zc"])
    g_zz = np.asarray(g["g_zz"])
    g_db = np.asarray(g["g_db"])
    g_dc = np.asarray(g["g_dc"])
    g_dz = np.asarray(g["g_dz"])
    K, M, J = len(g_cy), len(g_zb), len(g_db)

    c_ok = g_cy == 1
    for _ in range(K):  # propagate through C->C chains
        c_ok = c_ok | ((g_cc @ c_ok) > 0)
    z_ok = (g_zb == 1) | ((g_zc @ c_ok) > 0)
    for _ in range(M):  # propagate through Z->Z chains
        z_ok = z_ok | ((g_zz @ z_ok) > 0)
    d_ok = (g_db == 1) | ((g_dc @ c_ok) > 0) | ((g_dz @ z_ok) > 0)

    out: dict[str, bool] = {}
    out.update({f"C{k + 1}": bool(c_ok[k]) for k in range(K)})
    out.update({f"Z{m + 1}": bool(z_ok[m]) for m in range(M)})
    out.update({f"D{j + 1}": bool(d_ok[j]) for j in range(J)})
    return out


def node_status(g: dict) -> dict[str, str]:
    """Per-node status: connected (path to Y) | isolated (no edges at all) |
    dead-end (touches edges but none of them lead to Y).

    Dead-ends are forbidden by default: an arrow into (or out of) a node that
    never reaches Y is causal structure that does nothing — the world should
    contain either real pathways or FULLY disconnected nulls (which teach the
    model zero-attribution), never wasted arrows.
    """
    reach = path_to_y(g)
    g_cy = np.asarray(g["g_cy"])
    g_cc = np.asarray(g["g_cc"])
    g_zb = np.asarray(g["g_zb"])
    g_zc = np.asarray(g["g_zc"])
    g_zz = np.asarray(g["g_zz"])
    g_db = np.asarray(g["g_db"])
    g_dc = np.asarray(g["g_dc"])
    g_dz = np.asarray(g["g_dz"])
    K, M, J = len(g_cy), len(g_zb), len(g_db)

    touched: dict[str, bool] = {}
    for k in range(K):
        touched[f"C{k + 1}"] = bool(
            g_cy[k] or g_cc[k, :].any() or g_cc[:, k].any() or g_dc[:, k].any() or g_zc[:, k].any()
        )
    for m in range(M):
        touched[f"Z{m + 1}"] = bool(
            g_zb[m] or g_zc[m, :].any() or g_zz[m, :].any() or g_zz[:, m].any() or g_dz[:, m].any()
        )
    for j in range(J):
        touched[f"D{j + 1}"] = bool(g_db[j] or g_dc[j, :].any() or g_dz[j, :].any())

    return {
        n: ("connected" if reach[n] else ("isolated" if not touched[n] else "dead-end"))
        for n in reach
    }


def edges_with_coeffs(g: dict, params: dict) -> list[tuple[str, str, str, float]]:
    """(edge_type, src, dst, coefficient) for every active edge."""
    out: list[tuple[str, str, str, float]] = []
    J, K = np.asarray(g["g_dc"]).shape
    M = np.asarray(g["g_zb"]).shape[0]
    for k in range(K):
        if g["g_cy"][k]:
            out.append(("cy", f"C{k + 1}", "Y", float(params["beta"][k])))
    for j in range(J):
        for k in range(K):
            if g["g_dc"][j, k]:
                out.append(("dc", f"D{j + 1}", f"C{k + 1}", float(params["w_dc"][j, k])))
        for m in range(M):
            if g["g_dz"][j, m]:
                out.append(("dz", f"D{j + 1}", f"Z{m + 1}", float(params["u_dz"][j, m])))
        if g["g_db"][j]:
            out.append(("db", f"D{j + 1}", "B", float(params["delta_db"][j])))
    for m in range(M):
        for k in range(K):
            if g["g_zc"][m, k]:
                out.append(("zc", f"Z{m + 1}", f"C{k + 1}", float(params["v_zc"][m, k])))
        if g["g_zb"][m]:
            out.append(("zb", f"Z{m + 1}", "B", float(params["rho_zb"][m])))
        for m2 in range(M):
            if g["g_zz"][m, m2]:
                out.append(("zz", f"Z{m + 1}", f"Z{m2 + 1}", float(params["gamma_zz"][m, m2])))
    for k1 in range(K):
        for k2 in range(K):
            if g["g_cc"][k1, k2]:
                out.append(("cc", f"C{k1 + 1}", f"C{k2 + 1}", float(params["alpha_cc"][k1, k2])))
    return out


def channel_role(g: dict, k: int) -> str:
    """direct | feeder (no C->Y but feeds other channels) | null (no effect on Y)."""
    if g["g_cy"][k]:
        return "direct"
    if np.asarray(g["g_cc"])[k, :].any():
        return "feeder"
    return "null"


def mechanism_label(params: dict, k: int) -> str:
    """'saturation·adstock' label for channel ``k`` (e.g. ``hill·geometric``)."""
    sat = SATURATION_NAMES[int(params["sat_family"][k])]
    ad = ADSTOCK_NAMES[int(params["adstock_family"][k])]
    return f"{sat}·{ad}"


def sample_scm(
    cfg: SCMPrior,
    seed: int = 0,
    *,
    connect_all: bool = False,
    name: str = "scm",
    purpose: str = "",
    max_graph_rounds: int = 2000,
    max_param_rounds: int = 6,
    max_eps_draws: int = 40,
) -> SCM:
    """Draw ONE accepted world at the config's max sizes.

    Resamples the DAG until it satisfies the connectivity rule — dead-end
    nodes (edges that never reach Y) are NEVER allowed; fully-isolated null
    nodes are allowed only when ``connect_all=False`` — then builds the world's
    PyMC model and ``pm.draw``s candidates until one passes the realism filter
    (finite arrays, non-negative sales, spend-CV floor, spike guards).

    Fully deterministic given ``(cfg, seed)``: one numpy RNG drives the DAG +
    structure draws and the per-round pm.draw seeds.

    Parameters
    ----------
    cfg : SCMPrior
        An additive-SCM config (see ``make_scm_prior`` or
        ``Scenario.prior``). All ``n_treatments/n_covariates/n_latent`` nodes are active.
    seed : int
        Seed for the world's RNG stream.
    connect_all : bool
        Require every node to have a directed path to Y.
    name, purpose : str
        Labels carried into descriptions and bundles.

    Returns
    -------
    SCM
    """
    from .world_model import build_world_model, draw_worlds, sample_prior_cond, sample_structure

    cfg.validate()
    rng = np.random.default_rng(seed)
    layout = cfg.layout
    K, M, J, T = cfg.n_treatments, cfg.n_covariates, cfg.n_latent, cfg.T

    for _g_round in range(max_graph_rounds):
        g = sample_g_additive(rng, cfg, layout, K_active=K, M_active=M, J_active=J)
        g_act = _slice_g_active(g, K, M, J)
        status = node_status(g_act)
        if any(s == "dead-end" for s in status.values()):
            continue
        if connect_all and any(s == "isolated" for s in status.values()):
            continue
        break
    else:
        raise RuntimeError(
            f"world {name!r}: no DAG satisfying the connectivity rule in {max_graph_rounds} draws"
        )

    # Structure (families / smoothness / texture flags) is drawn once; the
    # continuous priors and noise are the pm.Model's RVs, drawn per candidate
    # and filtered by the realism gate. build the model once, draw in batches.
    structural = sample_structure(g_act, cfg, rng)
    # Prior-conditioning intervals (same draw as the corpus path; None — and
    # no RNG consumed — when cfg.prior_conditioning is False).
    prior_cond = sample_prior_cond(cfg, rng)
    model, out_names, param_names = build_world_model(
        g_act, cfg, structural, T, prior_cond=prior_cond
    )
    # structural is recorded so the world's oracle model (SCM.oracle_model)
    # can be rebuilt from the SCM alone.
    extras: dict = {"structural": structural}
    if prior_cond is not None:
        extras["prior_cond"] = prior_cond

    for _round in range(max_param_rounds):
        draw_seed = int(rng.integers(2**31 - 1))
        drawn = draw_worlds(model, out_names + param_names, draw_seed, draws=max_eps_draws)
        for b in range(max_eps_draws):
            d = {nm: drawn[nm][b] for nm in out_names}
            check = {
                k: v for k, v in d.items() if k not in ("channels_base", "contributions_observed")
            }
            if _additive_task_ok(
                spend=d["channels"],
                sales=d["sales"],
                arrays=check,
                g_cy_active=g_act["g_cy"],
                cv_floor=cfg.spend_cv_floor,
                realism_spend=d.get("channels_unshocked"),
                realism_sales=d.get("sales_unshocked"),
            ):
                return SCM(
                    data=d,
                    g=g_act,
                    params=_assemble_params(drawn, b, structural),
                    cfg=cfg,
                    name=name,
                    purpose=purpose,
                    seed=seed,
                    extras=extras,
                )
    raise RuntimeError(f"world {name!r}: no accepted draw in {max_param_rounds} rounds")


def _assemble_params(drawn: dict, b: int, structural: dict) -> dict:
    """Assemble the per-world reported params (candidate ``b``) into the shape
    descriptions / bundles expect: drawn ``param_*`` values plus the concrete
    structural families and per-channel walk smoothness."""
    keys = (
        "beta",
        "w_dc",
        "u_dz",
        "v_zc",
        "alpha_cc",
        "gamma_zz",
        "delta_db",
        "rho_zb",
        "adstock_alpha",
        "weibull_lam",
        "weibull_k",
        "hf_sigma",
        "pulse_amp",
        "pulse_prob",
        "confounding_strength",
        "channel_level",
    )
    params = {k: drawn[f"param_{k}"][b] for k in keys}
    params["adstock_family"] = structural["adstock_family"]
    params["sat_family"] = structural["sat_family"]
    params["rw_c"] = {
        "mean": drawn["param_rw_c_mean"][b],
        "std": drawn["param_rw_c_std"][b],
        "smoothness": structural["smoothness_c"],
    }
    return params
