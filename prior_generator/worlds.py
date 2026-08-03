"""Single-world sampling: one accepted task, with its full ground truth.

Where :func:`prior_generator.sample_prior_predictive` produces a padded ``n_tasks``-task corpus
for training, :func:`sample_scm` draws ONE accepted world at the config's
max sizes and keeps everything a human (or exporter) needs: the active-size
DAG blocks, the drawn SCM parameters, and named output series and metadata,
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
from typing import Any, Literal, cast

import numpy as np

from .random_walk import _centred_walk_scale, _kernel_width
from .sampler import (
    ADSTOCK_FAMILY_KEYS,
    SATURATION_FAMILY_KEYS,
    SCMPrior,
    _additive_task_ok,
    _slice_g_active,
    sample_g_additive,
)
from .signal_diagnostics import SIGNAL_METRIC_LAYOUT, SIGNAL_METRIC_VERSION, per_channel_signal

#: Adstock family names, indexed by ``params["adstock_family"]``.
ADSTOCK_NAMES = ADSTOCK_FAMILY_KEYS

#: Saturation family names, indexed by ``params["sat_family"]``.
SATURATION_NAMES = SATURATION_FAMILY_KEYS


_EXOGENOUS_NAMES = (
    "eps_d",
    "eps_z",
    "eps_c",
    "eps_b",
    "eps_y",
    "eps_c_hf",
    "eps_c_pulse",
)

_LEGACY_WORLD_PARAM_NAMES = tuple(
    f"param_{name}"
    for name in (
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
        "rw_c_mean",
        "channel_level",
        "rw_c_std",
        "confounding_strength",
    )
)


def _copy_audit_value(value: Any) -> Any:
    """Recursively copy audit data without exposing a private array backing."""
    if isinstance(value, np.ndarray):
        return value.copy()
    if isinstance(value, dict):
        return {key: _copy_audit_value(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_copy_audit_value(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_copy_audit_value(item) for item in value)
    return value


@dataclass
class SCM:
    """One simulated world: outputs + DAG + drawn parameters + config.

    Attributes
    ----------
    data : dict
        Active-size graph outputs — e.g. ``channels (n_time_steps, n_treatments)``,
        ``sales (n_time_steps,)``, ``contributions (n_time_steps, n_treatments)``,
        ``saturation_scale (n_treatments,)``, and
        ``indirect_effects_by_source (n_time_steps, 3)`` in the locked (cc, zc, dc) order.
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
    _exogenous: dict[str, np.ndarray] = field(default_factory=dict, repr=False)

    @property
    def n_time_steps(self) -> int:
        return int(self.data["sales"].shape[0])

    @property
    def n_treatments(self) -> int:
        return int(self.data["channels"].shape[1])

    @property
    def n_covariates(self) -> int:
        return int(self.data["controls"].shape[1])

    @property
    def n_latent(self) -> int:
        return int(self.data["demand"].shape[1])

    @property
    def exogenous(self) -> dict[str, np.ndarray]:
        """Raw full-horizon innovation draws, returned as defensive copies.

        ``eps_c`` is the independent, pre-mixture channel innovation. The
        graph uses ``sqrt(1-rho**2) * eps_c + rho * eps_b[:, None]`` when
        confounding is enabled. ``eps_c_pulse`` is a 0/1 Bernoulli fire.
        """
        return cast(dict[str, np.ndarray], _copy_audit_value(self._exogenous))

    @property
    def equations(self) -> dict[str, str]:
        """Readable vector-valued structural assignments for this exact world."""
        return cast(dict[str, str], _copy_audit_value(_build_equations(self)))

    @property
    def equation_parameters(self) -> dict[str, Any]:
        """Executed equation inputs, returned as recursively defensive copies."""
        return cast(
            dict[str, Any],
            _copy_audit_value(_build_equation_parameters(self)),
        )

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

    def oracle_model(self, *, latent: Literal["marginal", "sampled"] = "marginal"):
        """The observed-data posterior ``pm.Model`` for THIS world.

        Rebuilds :func:`prior_generator.world_model.build_oracle_model` from
        the world's own structure, config, observables and (when present)
        prior-conditioning intervals, so ``pm.sample(model=world.oracle_model())``
        yields the structure-known posterior on the world's dataset. The
        default ``latent="marginal"`` analytically integrates the outcome-side
        Gaussian walks and is the recommended reference posterior.
        ``latent="sampled"`` reproduces the previous representation and is
        required for posterior ``demand`` / ``baseline`` series. Requires an
        SCM produced by ``sample_scm`` (which records the structural draw in
        ``extras``). A Weibull-adstock channel downgrades ``weibull_lam`` and
        ``weibull_k`` from NUTS to Metropolis: pymc-marketing's min-max
        Weibull normalization has an upstream ``Min`` with no pullback.
        Identity and geometric adstock remain NUTS-differentiable. Treat the
        Metropolis parameters' ESS with suspicion and prefer geometric-adstock
        worlds when using this as a reference posterior. See the oracle guide
        for the other caveats.
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
            "saturation_scale": self.data["saturation_scale"],
        }
        if self.cfg.n_channel_shocks:
            data.update(
                {
                    "channel_shock_channel": self.data["channel_shock_channel"],
                    "channel_shock_start": self.data["channel_shock_start"],
                    "channel_shock_length": self.data["channel_shock_length"],
                    "channel_shock_level_multiplier": self.data["channel_shock_level_multiplier"],
                    "channel_shock_level": self.data["channel_shock_level"],
                    "channel_level": self.params["channel_level"],
                }
            )
        return build_oracle_model(
            self.g,
            self.cfg,
            self.extras["structural"],
            data=data,
            prior_cond=self.extras.get("prior_cond"),
            latent=latent,
        )

    def signal(self) -> dict[str, Any]:
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
            adstock_burn_in=self.cfg.adstock_burn_in,
        )
        out: dict[str, Any] = dict(per)
        out["channel"] = np.nonzero(cy_mask[0])[0].astype(float)
        out["metric_version"] = SIGNAL_METRIC_VERSION
        out["metric_layout"] = SIGNAL_METRIC_LAYOUT
        return out


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
    n_treatments, n_covariates, n_latent = len(g_cy), len(g_zb), len(g_db)

    c_ok = g_cy == 1
    for _ in range(n_treatments):  # propagate through C->C chains
        c_ok = c_ok | ((g_cc @ c_ok) > 0)
    z_ok = (g_zb == 1) | ((g_zc @ c_ok) > 0)
    for _ in range(n_covariates):  # propagate through Z->Z chains
        z_ok = z_ok | ((g_zz @ z_ok) > 0)
    d_ok = (g_db == 1) | ((g_dc @ c_ok) > 0) | ((g_dz @ z_ok) > 0)

    out: dict[str, bool] = {}
    out.update({f"C{k + 1}": bool(c_ok[k]) for k in range(n_treatments)})
    out.update({f"Z{m + 1}": bool(z_ok[m]) for m in range(n_covariates)})
    out.update({f"D{j + 1}": bool(d_ok[j]) for j in range(n_latent)})
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
    n_treatments, n_covariates, n_latent = len(g_cy), len(g_zb), len(g_db)

    touched: dict[str, bool] = {}
    for k in range(n_treatments):
        touched[f"C{k + 1}"] = bool(
            g_cy[k] or g_cc[k, :].any() or g_cc[:, k].any() or g_dc[:, k].any() or g_zc[:, k].any()
        )
    for m in range(n_covariates):
        touched[f"Z{m + 1}"] = bool(
            g_zb[m] or g_zc[m, :].any() or g_zz[m, :].any() or g_zz[:, m].any() or g_dz[:, m].any()
        )
    for j in range(n_latent):
        touched[f"D{j + 1}"] = bool(g_db[j] or g_dc[j, :].any() or g_dz[j, :].any())

    return {
        n: ("connected" if reach[n] else ("isolated" if not touched[n] else "dead-end"))
        for n in reach
    }


def edges_with_coeffs(g: dict, params: dict) -> list[tuple[str, str, str, float]]:
    """(edge_type, src, dst, coefficient) for every active edge."""
    out: list[tuple[str, str, str, float]] = []
    n_latent, n_treatments = np.asarray(g["g_dc"]).shape
    n_covariates = np.asarray(g["g_zb"]).shape[0]
    for k in range(n_treatments):
        if g["g_cy"][k]:
            out.append(("cy", f"C{k + 1}", "Y", float(params["beta"][k])))
    for j in range(n_latent):
        for k in range(n_treatments):
            if g["g_dc"][j, k]:
                out.append(("dc", f"D{j + 1}", f"C{k + 1}", float(params["w_dc"][j, k])))
        for m in range(n_covariates):
            if g["g_dz"][j, m]:
                out.append(("dz", f"D{j + 1}", f"Z{m + 1}", float(params["u_dz"][j, m])))
        if g["g_db"][j]:
            out.append(("db", f"D{j + 1}", "B", float(params["delta_db"][j])))
    for m in range(n_covariates):
        for k in range(n_treatments):
            if g["g_zc"][m, k]:
                out.append(("zc", f"Z{m + 1}", f"C{k + 1}", float(params["v_zc"][m, k])))
        if g["g_zb"][m]:
            out.append(("zb", f"Z{m + 1}", "B", float(params["rho_zb"][m])))
        for m2 in range(n_covariates):
            if g["g_zz"][m, m2]:
                out.append(("zz", f"Z{m + 1}", f"Z{m2 + 1}", float(params["gamma_zz"][m, m2])))
    for k1 in range(n_treatments):
        for k2 in range(n_treatments):
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


def _rw_parameters(params: dict, group: str, index: int) -> dict[str, float | bool]:
    """Concrete inputs for one executed random-walk or iid-noise column."""
    values = params[group]
    out: dict[str, float | bool] = {
        "mean": float(np.asarray(values["mean"])[index]),
        "std": float(np.asarray(values["std"])[index]),
        "positive_only": bool(values["positive_only"]),
    }
    if "smoothness" in values:
        out["smoothness"] = float(np.asarray(values["smoothness"])[index])
        out["rw_smoothness_max_weeks"] = int(values["rw_smoothness_max_weeks"])
    return out


def _channel_response_parameters(world: SCM, k: int) -> dict[str, Any]:
    """The family-specific response inputs actually consumed for channel ``k``."""
    params = world.params
    ad_name = ADSTOCK_NAMES[int(params["adstock_family"][k])]
    sat_name = SATURATION_NAMES[int(params["sat_family"][k])]
    adstock: dict[str, Any] = {"family": ad_name, "l_max": int(params["l_max"])}
    if ad_name == "geometric":
        adstock["alpha"] = float(np.asarray(params["adstock_alpha"])[k])
    elif ad_name == "weibull":
        adstock["lam"] = float(np.asarray(params["weibull_lam"])[k])
        adstock["k"] = float(np.asarray(params["weibull_k"])[k])

    saturation: dict[str, Any] = {
        "family": sat_name,
        "scale": float(np.asarray(world.data["saturation_scale"])[k]),
    }
    if sat_name == "hill":
        saturation["slope"] = float(np.asarray(params["hill_slope"])[k])
        saturation["kappa_mult"] = float(np.asarray(params["hill_kappa_mult"])[k])
    elif sat_name == "logistic":
        saturation["lam"] = float(np.asarray(params["logistic_lam"])[k])
    elif sat_name == "michaelis_menten":
        saturation["kappa_mult"] = float(np.asarray(params["mm_kappa_mult"])[k])
    elif sat_name == "tanh":
        saturation["c"] = float(np.asarray(params["tanh_c"])[k])
    elif sat_name == "root":
        saturation["alpha"] = float(np.asarray(params["root_alpha"])[k])

    g_cy = int(np.asarray(world.g["g_cy"])[k])
    gate: dict[str, float | int] = {"g_cy": g_cy, "value": 0.0}
    if g_cy:
        beta = float(np.asarray(params["beta"])[k])
        gate.update({"beta": beta, "value": beta})
    return {
        "adstock": adstock,
        "saturation": saturation,
        "gate": gate,
    }


def _build_equation_parameters(world: SCM) -> dict[str, Any]:
    """Build the concrete, sparse parameter audit for the executed SCM."""
    g, params = world.g, world.params
    n_treatments, n_covariates, n_latent = world.n_treatments, world.n_covariates, world.n_latent
    values: dict[str, Any] = {
        "innovations": {
            "rho": float(np.asarray(params["confounding_strength"])),
            "eps_c": "raw independent pre-mixture innovation",
            "eps_c_pulse": "0/1 Bernoulli fire",
        }
    }
    for j in range(n_latent):
        values[f"D{j + 1}"] = {"random_walk": _rw_parameters(params, "rw_d", j)}
    for m in range(n_covariates):
        parents: dict[str, float] = {}
        for j in range(n_latent):
            if g["g_dz"][j, m]:
                parents[f"D{j + 1}"] = float(np.asarray(params["u_dz"])[j, m])
        for m_parent in range(m):
            if g["g_zz"][m_parent, m]:
                parents[f"Z{m_parent + 1}"] = float(np.asarray(params["gamma_zz"])[m_parent, m])
        values[f"Z{m + 1}"] = {"random_walk": _rw_parameters(params, "rw_z", m)}
        if parents:
            values[f"Z{m + 1}"]["parents"] = parents
    for k in range(n_treatments):
        parents = {}
        for j in range(n_latent):
            if g["g_dc"][j, k]:
                parents[f"D{j + 1}"] = float(np.asarray(params["w_dc"])[j, k])
        for m in range(n_covariates):
            if g["g_zc"][m, k]:
                parents[f"Z{m + 1}"] = float(np.asarray(params["v_zc"])[m, k])
        for k_parent in range(k):
            if g["g_cc"][k_parent, k]:
                parents[f"C{k_parent + 1}"] = float(np.asarray(params["alpha_cc"])[k_parent, k])
        values[f"C{k + 1}"] = {
            "random_walk": _rw_parameters(params, "rw_c", k),
            "texture": {
                "use_hf": bool(np.asarray(params["use_hf"])[k]),
                "hf_sigma": float(np.asarray(params["hf_sigma"])[k]),
                "use_pulse": bool(np.asarray(params["use_pulse"])[k]),
                "pulse_amp": float(np.asarray(params["pulse_amp"])[k]),
                "pulse_prob": float(np.asarray(params["pulse_prob"])[k]),
            },
            "response": _channel_response_parameters(world, k),
        }
        if parents:
            values[f"C{k + 1}"]["parents"] = parents
    b_parents: dict[str, float] = {}
    for j in range(n_latent):
        if g["g_db"][j]:
            b_parents[f"D{j + 1}"] = float(np.asarray(params["delta_db"])[j])
    for m in range(n_covariates):
        if g["g_zb"][m]:
            b_parents[f"Z{m + 1}"] = float(np.asarray(params["rho_zb"])[m])
    values["B"] = {"random_walk": _rw_parameters(params, "rw_b", 0)}
    if b_parents:
        values["B"]["parents"] = b_parents
    values["Y"] = {"iid_noise": _rw_parameters(params, "rw_y", 0)}
    if "channel_shock" in params:
        schedule = params["channel_shock"]
        values["channel_shocks"] = {
            "n_shocks": int(schedule["n_shocks"]),
            "channel": schedule["channel"],
            "start_full": schedule["start_full"],
            "mask_full": schedule["mask_full"],
            "level_full": schedule["level_full"],
        }
    return values


def _join_terms(base: str, terms: list[str]) -> str:
    return " + ".join([base, *terms]) if terms else base


def _build_equations(world: SCM) -> dict[str, str]:
    """Readable vector assignments mirroring ``build_symbolic_graph`` exactly."""
    g, params = world.g, world.params
    n_treatments, n_covariates, n_latent = world.n_treatments, world.n_covariates, world.n_latent
    shocks_enabled = "channel_shock" in params
    n_time_steps_full = world.n_time_steps + world.cfg.adstock_burn_in

    def rw_scale(label: str, group: str, index: int) -> str:
        smoothness = float(np.asarray(params[group]["smoothness"])[index])
        width = _kernel_width(
            smoothness,
            n_time_steps_full,
            rw_smoothness_max_weeks=int(params[group]["rw_smoothness_max_weeks"]),
        )
        scale = _centred_walk_scale(n_time_steps_full, width)
        return (
            f"{label}: centred_walk_scale(n_time_steps_full={n_time_steps_full}, "
            f"width={width})={scale:.17g}"
        )

    rw_scales = [
        *(rw_scale(f"D{j + 1}", "rw_d", j) for j in range(n_latent)),
        *(rw_scale(f"Z{m + 1}", "rw_z", m) for m in range(n_covariates)),
        *(rw_scale(f"C{k + 1}", "rw_c", k) for k in range(n_treatments)),
        rw_scale("B", "rw_b", 0),
    ]
    equations: dict[str, str] = {
        "innovations": (
            "eps_c_eff = sqrt(1 - rho**2) * eps_c + rho * eps_b[:, None]; "
            "eps_c is the raw independent pre-mixture channel innovation and "
            "eps_c_pulse is a 0/1 Bernoulli fire. rho=0 gives mutually independent "
            "exogenous vectors; rho!=0 makes eps_c_eff and eps_b dependent."
        ),
        "RW": (
            f"n_time_steps_full = n_time_steps + adstock_burn_in = {n_time_steps_full}. "
            "For each random-walk innovation column, "
            "q = edge_padded_MA(cumsum(eps), "
            "width=kernel_width(smoothness, rw_smoothness_max_weeks), "
            "capped at n_time_steps_full); "
            "centred_walk_scale(n_time_steps_full, width) = "
            "sqrt(tr(A A^T) / n_time_steps_full), where "
            "A = centre . movavg(width) . cumsum is fixed; "
            "RW_full = mean + std * (q - mean(q)) / centred_walk_scale(n_time_steps_full, width); "
            f"world constants: {'; '.join(rw_scales)}. "
            "Apply softplus(RW_full) only when positive_only=True; RW = RW_full[burn_in:]. "
            "Y instead uses iid rw_y[0].std * eps_y[burn_in:]."
        ),
    }
    if shocks_enabled:
        equations["channel_shocks"] = (
            "The exact full-horizon channel_shock mask and held levels clamp each "
            "channel before adstock; the clamped path then adstocks with the "
            "ordinary normalized causal kernel, so carryover from pre-shock spend "
            "decays across a held window instead of being discarded. C_base, "
            "C_no_cc, and C_no_cc_zc share the same clamp."
        )

    for j in range(n_latent):
        equations[f"D{j + 1}"] = (
            f"D{j + 1}_full = RW_full(eps_d[:, {j}], rw_d[{j}]); D{j + 1} = D{j + 1}_full[burn_in:]"
        )
    for m in range(n_covariates):
        terms: list[str] = []
        for j in range(n_latent):
            if g["g_dz"][j, m]:
                terms.append(f"u_dz[{j}, {m}] * D{j + 1}_full")
        for m_parent in range(m):
            if g["g_zz"][m_parent, m]:
                terms.append(f"gamma_zz[{m_parent}, {m}] * Z{m_parent + 1}_full")
        equations[f"Z{m + 1}"] = (
            f"Z{m + 1}_full = {_join_terms(f'RW_full(eps_z[:, {m}], rw_z[{m}])', terms)}; "
            f"Z{m + 1} = Z{m + 1}_full[burn_in:]"
        )

    def clamp(expression: str, k: int) -> str:
        return f"clamp_shock_{k + 1}({expression})" if shocks_enabled else expression

    for k in range(n_treatments):
        d_terms = [f"w_dc[{j}, {k}] * D{j + 1}_full" for j in range(n_latent) if g["g_dc"][j, k]]
        z_terms = [
            f"v_zc[{m}, {k}] * Z{m + 1}_full" for m in range(n_covariates) if g["g_zc"][m, k]
        ]
        c_terms = [
            f"alpha_cc[{k_parent}, {k}] * C{k_parent + 1}_full"
            for k_parent in range(k)
            if g["g_cc"][k_parent, k]
        ]
        natural_c_terms = [
            f"alpha_cc[{k_parent}, {k}] * C{k_parent + 1}_unshocked_full"
            for k_parent in range(k)
            if g["g_cc"][k_parent, k]
        ]
        own = f"RW_full(eps_c_eff[:, {k}], rw_c[{k}])"
        hf_on = bool(np.asarray(params["use_hf"])[k])
        pulse_on = bool(np.asarray(params["use_pulse"])[k])
        if hf_on:
            own += f" + hf_sigma[{k}] * eps_c_hf[:, {k}]"
        if pulse_on:
            own += f" + pulse_amp[{k}] * eps_c_pulse[:, {k}]"
        observed_inner = _join_terms(f"own_C{k + 1}_full", [*d_terms, *z_terms, *c_terms])
        natural_inner = _join_terms(f"own_C{k + 1}_full", [*d_terms, *z_terms, *natural_c_terms])
        equations[f"C{k + 1}"] = (
            f"use_hf[{k}]={hf_on}; use_pulse[{k}]={pulse_on}; "
            f"own_C{k + 1}_full = {own}; "
            f"C{k + 1}_unshocked_full = softplus({natural_inner}); "
            f"C{k + 1}_full = {clamp(f'softplus({observed_inner})', k)}; "
            f"C{k + 1} = C{k + 1}_full[burn_in:]"
        )
        equations[f"C_base{k + 1}"] = (
            f"C_base{k + 1}_full = {clamp(f'softplus(own_C{k + 1}_full)', k)}; "
            f"C_base{k + 1} = C_base{k + 1}_full[burn_in:]"
        )
        no_cc_inner = _join_terms(f"own_C{k + 1}_full", [*d_terms, *z_terms])
        equations[f"C_no_cc{k + 1}"] = (
            f"C_no_cc{k + 1}_full = {clamp(f'softplus({no_cc_inner})', k)}; "
            f"C_no_cc{k + 1} = C_no_cc{k + 1}_full[burn_in:]"
        )
        no_cc_zc_inner = _join_terms(f"own_C{k + 1}_full", d_terms)
        equations[f"C_no_cc_zc{k + 1}"] = (
            f"C_no_cc_zc{k + 1}_full = {clamp(f'softplus({no_cc_zc_inner})', k)}; "
            f"C_no_cc_zc{k + 1} = C_no_cc_zc{k + 1}_full[burn_in:]"
        )
        ad_name = ADSTOCK_NAMES[int(params["adstock_family"][k])]
        sat_name = SATURATION_NAMES[int(params["sat_family"][k])]
        equations[f"f{k + 1}"] = (
            f"ad_C{k + 1} = adstock[{ad_name}](C{k + 1}_full); "
            f"saturation_scale[{k}] = max(E[C{k + 1}] from parameters, 1e-8); "
            f"f{k + 1}(X_full) = {sat_name}(adstock[{ad_name}](X_full)"
            f"[burn_in:], saturation_scale[{k}]); "
            f"gate[{k}] = g_cy[{k}] * beta[{k}]"
        )

    b_terms = [f"delta_db[{j}] * D{j + 1}_full" for j in range(n_latent) if g["g_db"][j]] + [
        f"rho_zb[{m}] * Z{m + 1}_full" for m in range(n_covariates) if g["g_zb"][m]
    ]
    equations["B"] = (
        f"B_full = {_join_terms('RW_full(eps_b, rw_b[0])', b_terms)}; B = B_full[burn_in:]"
    )
    equations["contributions"] = (
        "contributions[:, k] = gate[k] * f{k}(C_base{k}_full); "
        "contributions_observed[:, k] = gate[k] * f{k}(C{k}_full), for k=1..n_treatments."
    )
    equations["indirect_effects"] = (
        "IE_cc = sum_k gate[k] * (f{k}(C{k}_full) - f{k}(C_no_cc{k}_full)); "
        "IE_zc = sum_k gate[k] * (f{k}(C_no_cc{k}_full) - f{k}(C_no_cc_zc{k}_full)); "
        "IE_dc = sum_k gate[k] * (f{k}(C_no_cc_zc{k}_full) - f{k}(C_base{k}_full)); "
        "indirect_effects_by_source = [IE_cc, IE_zc, IE_dc]; "
        "indirect_effects = sum_k(contributions_observed[:, k] - contributions[:, k])."
    )
    equations["Y"] = (
        "baseline = B_full[burn_in:] + rw_y[0].std * eps_y[burn_in:]; "
        "baseline_intrinsic = RW_full(eps_b, rw_b[0])[burn_in:] + "
        "rw_y[0].std * eps_y[burn_in:]; "
        "Y (sales) = baseline + sum_k contributions_observed[:, k]."
    )
    return equations


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
    n_treatments, n_covariates, n_latent, n_time_steps = (
        cfg.n_treatments,
        cfg.n_covariates,
        cfg.n_latent,
        cfg.n_time_steps,
    )

    for _g_round in range(max_graph_rounds):
        g = sample_g_additive(
            rng,
            cfg,
            layout,
            n_treatments_active=n_treatments,
            n_covariates_active=n_covariates,
            n_latent_active=n_latent,
        )
        g_act = _slice_g_active(g, n_treatments, n_covariates, n_latent)
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
        g_act, cfg, structural, n_time_steps, prior_cond=prior_cond
    )
    # structural is recorded so the world's oracle model (SCM.oracle_model)
    # can be rebuilt from the SCM alone.
    extras: dict = {"structural": structural}
    if prior_cond is not None:
        extras["prior_cond"] = prior_cond

    for _round in range(max_param_rounds):
        draw_seed = int(rng.integers(2**31 - 1))
        # Keep the raw accepted-candidate innovations in this same draw as the
        # outputs and parameters. In a confounded world model["eps_c"] remains
        # the independent pre-mixture Normal RV; the graph receives eps_c_eff.
        drawn = draw_worlds(
            model,
            out_names + param_names + _EXOGENOUS_NAMES,
            draw_seed,
            draws=max_eps_draws,
            rng_reference_names=out_names + _LEGACY_WORLD_PARAM_NAMES,
        )
        for b in range(max_eps_draws):
            d = {nm: np.array(drawn[nm][b], copy=True) for nm in out_names}
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
                    params=_assemble_params(drawn, b, structural, param_names, cfg),
                    cfg=cfg,
                    name=name,
                    purpose=purpose,
                    seed=seed,
                    extras=extras,
                    _exogenous={
                        exogenous_name: np.array(drawn[exogenous_name][b], copy=True)
                        for exogenous_name in _EXOGENOUS_NAMES
                    },
                )
    raise RuntimeError(f"world {name!r}: no accepted draw in {max_param_rounds} rounds")


def _assemble_channel_shock_schedule(
    drawn: dict[str, np.ndarray], b: int, cfg: SCMPrior
) -> dict[str, Any]:
    """Reconstruct the accepted full-horizon shock inputs for graph replay."""
    channel = np.array(drawn["channel_shock_channel"][b], dtype="int64", copy=True)
    start_full = np.array(drawn["channel_shock_start"][b], dtype="int64", copy=True)
    start_full += cfg.adstock_burn_in
    length = np.array(drawn["channel_shock_length"][b], dtype="int64", copy=True)
    level = np.array(drawn["channel_shock_level"][b], dtype="float64", copy=True)
    mask_full = np.array(drawn["channel_shock_mask_full"][b], copy=True)
    level_full = np.zeros(mask_full.shape, dtype="float64")
    occupied = np.zeros(mask_full.shape[0], dtype=bool)
    for selected, start, duration, held_level in zip(channel, start_full, length, level):
        end = int(start + duration)
        if occupied[int(start) : end].any():
            raise AssertionError("channel shock windows must not overlap")
        occupied[int(start) : end] = True
        # This concrete replay schedule must agree with world_model's symbolic
        # sum. Stratified slots make windows disjoint, so assignment is equivalent.
        level_full[int(start) : end, int(selected)] = held_level
    return {
        "n_shocks": int(cfg.n_channel_shocks),
        "channel": channel,
        "start_full": start_full,
        "mask_full": mask_full,
        "level_full": level_full,
    }


def _assemble_params(
    drawn: dict[str, np.ndarray],
    b: int,
    structural: dict,
    param_names: tuple[str, ...],
    cfg: SCMPrior,
) -> dict[str, Any]:
    """Build the exact concrete input dictionary consumed by ``build_symbolic_graph``."""
    params: dict[str, Any] = {
        param_name.removeprefix("param_"): np.array(drawn[param_name][b], copy=True)
        for param_name in param_names
    }
    params["l_max"] = cfg.l_max
    params["adstock_family"] = np.array(structural["adstock_family"], copy=True)
    params["sat_family"] = np.array(structural["sat_family"], copy=True)
    params["use_hf"] = np.array(structural["use_hf"], copy=True)
    params["use_pulse"] = np.array(structural["use_pulse"], copy=True)
    for suffix, positive_only in (
        ("d", False),
        ("z", False),
        ("c", True),
        ("b", False),
        ("y", False),
    ):
        group = f"rw_{suffix}"
        group_params: dict[str, Any] = {
            "mean": params.pop(f"{group}_mean"),
            "std": params.pop(f"{group}_std"),
            "positive_only": positive_only,
        }
        if suffix != "y":
            group_params["smoothness"] = np.array(structural[f"smoothness_{suffix}"], copy=True)
            group_params["rw_smoothness_max_weeks"] = cfg.rw_smoothness_max_weeks
        params[group] = group_params
    if cfg.n_channel_shocks:
        params["channel_shock"] = _assemble_channel_shock_schedule(drawn, b, cfg)
    return params
