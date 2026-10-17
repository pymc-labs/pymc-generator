"""Family-specialized, active-only outcome oracle with inferred walk widths.

This opt-in consumer models outcomes conditional on observed inputs and the
known saturation reference anchor. It does not fit the input-generating process.
It never consumes query outcomes, realized coefficients, or baseline/latent truth.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any
from uuid import uuid4

import numpy as np
import pymc as pm
import pytensor.tensor as pt
from numpy.typing import NDArray
from pytensor.graph.rewriting.utils import rewrite_graph

from .oracle_dynamic import BoundOracleKernel, DynamicOracleKernel
from .oracle_inferred import add_inferred_walk_likelihood
from .oracle_smoothness import walk_width_prior
from .sampler import CARRYOVER_FAMILY_KEYS, SATURATION_FAMILY_KEYS, SCMPrior
from .symbolic_graph import _carryover_col, _saturate_family
from .world_model import (
    _apply_outcome_std_scale,
    _checked_reference_coefficient,
    _uniform_prior_specs,
)

if TYPE_CHECKING:
    import xarray as xr

    from .oracle_sampling import OracleSamplingConfig


@dataclass(frozen=True)
class OracleInputs:
    """Only allowed oracle observations/design metadata; no numerical truth."""

    channels: NDArray[np.float64]
    controls: NDArray[np.float64]
    context_indices: NDArray[np.int64]
    context_outcome: NDArray[np.float64]
    saturation_scale: NDArray[np.float64]
    channel_ids: tuple[int, ...]
    control_ids: tuple[int, ...]
    channel_gates: NDArray[np.float64]
    control_gates: NDArray[np.float64]
    latent_gates: NDArray[np.float64] = field(default_factory=lambda: np.zeros(0))

    def validate(self) -> None:
        t, p = self.channels.shape
        if t < 2 or p < 1 or self.controls.ndim != 2 or self.controls.shape[0] != t:
            raise ValueError("oracle inputs require aligned time and at least one active channel")
        if len(self.channel_ids) != p or len(self.control_ids) != self.controls.shape[1]:
            raise ValueError("input identities do not match active column counts")
        if len(set(self.channel_ids)) != p or len(set(self.control_ids)) != len(self.control_ids):
            raise ValueError("active input identities must be unique")
        if self.saturation_scale.shape != (p,) or np.any(self.saturation_scale <= 0):
            raise ValueError("saturation_scale must be positive and match active channels")
        if self.channel_gates.shape != (p,) or self.control_gates.shape != (
            self.controls.shape[1],
        ):
            raise ValueError("direct gates must match active input columns")
        if (
            not np.isin(self.channel_gates, [0, 1]).all()
            or not np.isin(self.control_gates, [0, 1]).all()
        ):
            raise ValueError("direct gates must be binary")
        if self.latent_gates.ndim != 1 or not np.isin(self.latent_gates, [0, 1]).all():
            raise ValueError("latent gates must be a binary vector")
        indices = self.context_indices
        if indices.ndim != 1 or not np.issubdtype(indices.dtype, np.integer) or not len(indices):
            raise ValueError("context indices must be a nonempty integer vector")
        if np.any(indices < 0) or np.any(indices >= t) or len(np.unique(indices)) != len(indices):
            raise ValueError("context indices must be unique reported-window indices")
        if self.context_outcome.shape != indices.shape:
            raise ValueError("context outcomes must match selected rows")
        if np.any(self.channels < 0):
            raise ValueError("channel inputs must be nonnegative")
        for array in (
            self.channels,
            self.controls,
            self.context_outcome,
            self.saturation_scale,
            self.channel_gates,
            self.control_gates,
        ):
            if not np.isfinite(array).all():
                raise ValueError("oracle inputs must be finite")


def _response_parameters(saturation: str, carryover: str) -> tuple[str, ...]:
    saturation_parameters = {
        "linear": (),
        "hill": ("hill_slope", "hill_kappa_mult"),
        "logistic": ("logistic_lam",),
        "michaelis_menten": ("mm_kappa_mult",),
        "tanh": ("tanh_c",),
        "root": ("root_alpha",),
    }
    carryover_parameters = {
        "none": (),
        "geometric": ("carryover_alpha",),
        "weibull": ("weibull_lam", "weibull_k"),
    }
    return carryover_parameters[carryover] + saturation_parameters[saturation]


def family_response(
    x: Any,
    scale: Any,
    parameters: dict[str, Any],
    *,
    saturation: str,
    carryover: str,
    l_max: int,
    stable: bool = False,
    apply_carryover: bool = True,
) -> Any:
    """Vectorize the SAME scalar-column response definitions as generation."""
    names = _response_parameters(saturation, carryover)

    def column(values, anchor, *shapes):
        params = {name: value[None] for name, value in zip(names, shapes, strict=True)}
        params.update(
            l_max=l_max,
            carryover_family=[CARRYOVER_FAMILY_KEYS.index(carryover)],
            mechanism_priors_enabled=stable,
        )
        adstock = _carryover_col(values, params, 0) if apply_carryover else values
        response = _saturate_family(saturation, adstock, anchor, params, 0)
        # The generator's transformers use named xtensors internally. Lower
        # those labels before positional active-channel vectorization.
        return rewrite_graph(response, include=("lower_xtensor",))

    signature = ",".join(["(t)", "()", *(["()"] * len(names))]) + "->(t)"
    return pt.vectorize(column, signature=signature)(
        pt.as_tensor_variable(x).T,
        pt.as_tensor_variable(scale),
        *[pt.as_tensor_variable(parameters[name]) for name in names],
    ).T


def _family_contract(cfg: SCMPrior, saturation: str, carryover: str) -> tuple:
    """Only inference-relevant fixed choices; trajectory recipes are not topology."""
    names = [*_response_parameters(saturation, carryover), "delta_dy"]
    names += [
        "beta"
        if cfg.treatment_reference_contribution_range is None
        else "treatment_reference_contribution"
    ]
    names += [
        "rho_zy"
        if cfg.covariate_reference_contribution_range is None
        else "covariate_reference_contribution"
    ]
    specs = _uniform_prior_specs(cfg, 1, 1, cfg.n_latent)
    return (
        cfg.n_time_steps,
        cfg.n_latent,
        cfg.carryover_burn_in,
        cfg.l_max,
        cfg.rw_smoothness_max_weeks,
        cfg.outcome_std_mode,
        cfg.baseline_floor,
        cfg.baseline_floor_scope,
        cfg.mm_scale_prior,
        cfg.mechanism_priors_enabled,
        cfg.prior_conditioning,
        cfg.treatment_reference_multiplier,
        cfg.covariate_reference_scale,
        tuple((name, specs[name][1], specs[name][2]) for name in names),
        # A point prior has different RV topology, but its value can still rebind.
        tuple(
            lo == hi
            for lo, hi in (
                cfg.rw_baseline_mean_range,
                cfg.rw_baseline_std_range,
                cfg.rw_outcome_std_range,
            )
        ),
    )


def _runtime_priors(cfg: SCMPrior) -> NDArray[np.float64]:
    result = np.array(
        [
            *cfg.rw_baseline_mean_range,
            cfg.rw_baseline_std_sigma_effective,
            cfg.rw_outcome_std_sigma,
            *cfg.rw_baseline_std_range,
            *cfg.rw_outcome_std_range,
            cfg.rw_smoothness_alpha,
            cfg.rw_smoothness_beta,
        ],
        dtype="float64",
    )
    if not np.isfinite(result).all() or np.any(result[[2, 3, 8, 9]] <= 0):
        raise ValueError("walk prior scales and Beta parameters must be positive and finite")
    if any(result[i] > result[i + 1] for i in (0, 4, 6)):
        raise ValueError("walk prior bounds must be ordered")
    return result


class FamilyOracle:
    """One active-only numerical compilation per saturation/carryover worker.

    ``bind(inputs, cfg=...)`` may change active counts, gates, context rows and
    baseline/noise/smoothness priors across regimes without recompilation.
    Horizon, latent count, mechanism priors and other topology remain fixed;
    incompatible bindings fail explicitly. No dataset selection or job policy
    lives here. Inactive columns must be removed by the caller, preserving IDs.
    """

    def __init__(self, cfg: SCMPrior, *, saturation: str, carryover: str) -> None:
        import copy

        if saturation not in SATURATION_FAMILY_KEYS or carryover not in CARRYOVER_FAMILY_KEYS:
            raise ValueError("unknown response family")
        if cfg.prior_conditioning:
            raise ValueError(
                "world-specific prior conditioning is not yet supported by FamilyOracle"
            )
        self.cfg = cfg = copy.deepcopy(cfg)
        self.saturation, self.carryover = saturation, carryover
        self.contract = _family_contract(cfg, saturation, carryover)
        self.compile_instance_id = uuid4().hex
        self.axes: dict[str, tuple[str, ...]] = {}
        t = cfg.n_time_steps
        self.width_prior = walk_width_prior(
            t + cfg.carryover_burn_in,
            cfg.rw_smoothness_max_weeks,
            cfg.rw_smoothness_alpha,
            cfg.rw_smoothness_beta,
        )
        with pm.Model() as model:
            x = pm.Data("channels", np.ones((t, 2)))
            z = pm.Data("controls", np.ones((t, 2)))
            scale = pm.Data("anchor", np.ones(2))
            cy = pm.Data("channel_gates", np.ones(2))
            zy = pm.Data("control_gates", np.ones(2))
            dy = pm.Data("latent_gates", np.ones(cfg.n_latent))
            indices = pm.Data("context_indices", np.arange(max(1, t // 2), dtype="int64"))
            # Context values only; scatter into zeros solely for the shared builder.
            y = pm.Data("context_outcome", np.zeros(max(1, t // 2)))
            outcome = pt.set_subtensor(pt.zeros((t,))[indices], y)
            priors = pm.Data("walk_priors", _runtime_priors(cfg))
            width_probs = pm.Data("width_probs", self.width_prior.probabilities.copy())
            sizes = {"channel": x.shape[1], "control": z.shape[1], "latent": cfg.n_latent}

            def uniform(name, lo, hi, axis=None, log=False, point=False):
                shape = () if axis is None else (sizes[axis],)
                axes = () if axis is None else (axis,)
                if point or (isinstance(lo, (int, float)) and lo == hi):
                    value = pm.Deterministic(name, pt.full(shape, lo))
                elif log:
                    raw_name = name + "_log"
                    self.axes[raw_name] = axes
                    raw = pm.Uniform(raw_name, np.log(lo), np.log(hi), shape=shape)
                    value = pm.Deterministic(name, pt.clip(pt.exp(raw), lo, hi))
                else:
                    value = pm.Uniform(name, lo, hi, shape=shape)
                self.axes[name] = axes
                return value

            specs = _uniform_prior_specs(cfg, 2, 2, cfg.n_latent)

            def parameter(name, axis):
                _, lo, hi, _ = specs[name]
                return uniform(
                    name,
                    lo,
                    hi,
                    axis,
                    log=name == "mm_kappa_mult" and cfg.mm_scale_prior == "log_uniform",
                )

            parameters = {
                name: parameter(name, "channel")
                for name in _response_parameters(saturation, carryover)
            }
            response = family_response(
                x,
                scale,
                parameters,
                saturation=saturation,
                carryover=carryover,
                l_max=cfg.l_max,
                stable=cfg.mechanism_priors_enabled,
            )
            if cfg.treatment_reference_contribution_range is None:
                beta = parameter("beta", "channel")
            else:
                target = parameter("treatment_reference_contribution", "channel")
                reference = (cfg.treatment_reference_multiplier * scale)[None, :]
                reference_response = family_response(
                    reference,
                    scale,
                    parameters,
                    saturation=saturation,
                    carryover=carryover,
                    l_max=cfg.l_max,
                    stable=cfg.mechanism_priors_enabled,
                    apply_carryover=False,
                )[0]
                beta = pm.Deterministic(
                    "beta",
                    _checked_reference_coefficient(
                        target / reference_response,
                        target,
                        "treatment_reference_contribution_range",
                    ),
                )
                self.axes["beta"] = ("channel",)
            if cfg.covariate_reference_contribution_range is None:
                rho = parameter("rho_zy", "control")
            else:
                target = parameter("covariate_reference_contribution", "control")
                rho = pm.Deterministic(
                    "rho_zy",
                    _checked_reference_coefficient(
                        target * (1.0 / float(cfg.covariate_reference_scale)),
                        target,
                        "covariate_reference_contribution_range",
                    ),
                )
                self.axes["rho_zy"] = ("control",)
            delta = parameter("delta_dy", "latent") if cfg.n_latent else pt.zeros((0,))
            mean = uniform(
                "rw_b_mean",
                priors[0],
                priors[1],
                point=cfg.rw_baseline_mean_range[0] == cfg.rw_baseline_mean_range[1],
            )
            if cfg.outcome_std_mode == "relative":
                rw = {
                    "rw_b": {
                        "std": uniform(
                            "rw_b_std_rel",
                            priors[4],
                            priors[5],
                            point=cfg.rw_baseline_std_range[0] == cfg.rw_baseline_std_range[1],
                        )[None]
                    },
                    "rw_y": {
                        "std": uniform(
                            "rw_y_std_rel",
                            priors[6],
                            priors[7],
                            point=cfg.rw_outcome_std_range[0] == cfg.rw_outcome_std_range[1],
                        )[None]
                    },
                }
                _apply_outcome_std_scale(cfg, rw, cy, beta)
                self.axes.update(rw_b_std=("singleton",), rw_y_std=("singleton",))
            elif cfg.outcome_std_mode == "absolute":
                rw = {
                    "rw_b": {"std": pm.HalfNormal("rw_b_std", priors[2])[None]},
                    "rw_y": {"std": pm.HalfNormal("rw_y_std", priors[3])[None]},
                }
                self.axes.update(rw_b_std=(), rw_y_std=())
            else:
                raise ValueError("unsupported outcome_std_mode")
            rw["rw_b"]["mean"] = mean[None]
            contributions = pm.Deterministic("contributions", response * cy * beta)
            self.axes["contributions"] = ("time", "channel")
            add_inferred_walk_likelihood(
                model=model,
                cfg=cfg,
                rw=rw,
                n_time_steps=t,
                n_latent=cfg.n_latent,
                n_covariates=2,
                contributions=contributions,
                controls=z,
                control_loadings=zy * rho,
                latent_loadings=dy * delta,
                outcome=outcome,
                observed_indices=indices,
                width_probabilities=width_probs,
            )
            self.axes.update(
                eps_b=("innovation",),
                baseline=("time",),
                outcome_mu=("time",),
                latent_unobserved=("time", "latent"),
                walk_width_probabilities=("walk_combination",),
            )
            if cfg.n_latent:
                self.axes["eps_d"] = ("innovation", "latent")
            components = model._inferred_walk_components
            internal_names = (
                "_probabilities",
                "_baseline",
                "_means",
                "_latent",
                "_noise",
                "_channels",
                "_controls",
            )
            for name, expr in zip(internal_names, components["outputs"], strict=True):
                pm.Deterministic(name, expr)
            pm.Deterministic("_state", pt.concatenate([v.reshape((-1,)) for v in model.value_vars]))
        self.model = model
        self.parameter_names = tuple(
            dict.fromkeys([rv.name for rv in model.free_RVs] + list(self.axes))
        )
        self.kernel = DynamicOracleKernel(
            model, output_names=[*self.parameter_names, *internal_names, "_state"]
        )

    def bind(self, inputs: OracleInputs, *, cfg: SCMPrior | None = None) -> BoundOracleKernel:
        """Bind one world's observations/priors without model construction or compilation."""
        inputs.validate()
        cfg = self.cfg if cfg is None else cfg
        if _family_contract(cfg, self.saturation, self.carryover) != self.contract:
            raise ValueError("world inference configuration is incompatible with family worker")
        t, p = inputs.channels.shape
        if t != cfg.n_time_steps or inputs.latent_gates.shape != (cfg.n_latent,):
            raise ValueError("world horizon or latent count differs from worker configuration")
        prior = walk_width_prior(
            t + cfg.carryover_burn_in,
            cfg.rw_smoothness_max_weeks,
            cfg.rw_smoothness_alpha,
            cfg.rw_smoothness_beta,
        )
        combinations = self.model._inferred_walk_components["combinations"]
        sizes = {
            "channel": p,
            "control": inputs.controls.shape[1],
            "latent": cfg.n_latent,
            "time": t,
            "innovation": t + cfg.carryover_burn_in,
            "walk_combination": len(combinations),
            "singleton": 1,
        }
        value_shapes = {
            self.model.rvs_to_values[rv].name: tuple(sizes[a] for a in self.axes[rv.name])
            for rv in self.model.free_RVs
        }
        n = sum(int(np.prod(shape)) for shape in value_shapes.values())
        output_shapes = {
            name: tuple(sizes[a] for a in self.axes[name]) for name in self.parameter_names
        }
        m = len(combinations)
        output_shapes.update(
            _probabilities=(m,),
            _baseline=(m, t),
            _means=(m, t),
            _latent=(m, t, cfg.n_latent),
            _noise=(),
            _channels=(t, p),
            _controls=(t, sizes["control"]),
            _state=(n,),
        )
        data = {
            "channels": inputs.channels,
            "controls": inputs.controls,
            "anchor": inputs.saturation_scale,
            "channel_gates": inputs.channel_gates,
            "control_gates": inputs.control_gates,
            "latent_gates": inputs.latent_gates,
            "context_indices": inputs.context_indices,
            "context_outcome": inputs.context_outcome,
            "walk_priors": _runtime_priors(cfg),
            "width_probs": prior.probabilities,
        }
        data = {
            name: np.asarray(value, dtype=self.model[name].dtype) for name, value in data.items()
        }
        coords = {axis: tuple(range(size)) for axis, size in sizes.items()}
        coords.update(
            channel=inputs.channel_ids,
            control=inputs.control_ids,
        )
        bound = self.kernel.bind(
            value_shapes=value_shapes,
            data=data,
            output_shapes=output_shapes,
            coords=coords,
            dims={name: self.axes[name] for name in self.parameter_names},
            initial_point=np.full(n, 0.01),
        )
        if len(self.kernel.signatures) != 1:
            raise RuntimeError("family worker compiled more than one numerical signature")
        return bound

    def binding_id(self, bound: BoundOracleKernel) -> str:
        """Fingerprint observations, runtime priors, active labels and model contract."""
        import hashlib

        if bound.kernel is not self.kernel:
            raise ValueError("binding belongs to a different family worker")
        digest = hashlib.sha256(
            repr(
                (
                    "family-inferred-v1",
                    self.saturation,
                    self.carryover,
                    self.contract,
                    bound.layout,
                    bound.coords,
                )
            ).encode()
        )
        for name, value in zip(self.kernel.data_names, bound.data, strict=True):
            digest.update(repr((name, value.shape, value.dtype.str)).encode())
            digest.update(value.tobytes())
        return digest.hexdigest()

    def sample(self, bound: BoundOracleKernel, config: OracleSamplingConfig) -> xr.DataTree:
        """Sample an immutable binding with fresh adaptation; retain all posterior draws.

        This is a local library call, not a scheduler. Selection, retention policy,
        diagnostics/qualification and persistence belong to the consumer.
        """
        import nutpie  # type: ignore[import-untyped]
        import xarray as xr

        from .oracle_sampling import OracleSamplingConfig

        if not isinstance(config, OracleSamplingConfig):
            raise TypeError("config must be an OracleSamplingConfig")
        config.validate()
        if (
            config.latent != "sampled"
            or config.walk_smoothness != "inferred"
            or config.nuts_sampler != "nutpie"
        ):
            raise ValueError("FamilyOracle requires sampled latent, inferred smoothness and nutpie")
        if config.adaptation == "flow" or config.compute_convergence_checks:
            raise ValueError("flow adaptation and automatic convergence analysis are unsupported")
        if bound.kernel is not self.kernel:
            raise ValueError("binding belongs to a different family worker")
        trace = nutpie.sample(
            bound.sampler_model([*self.parameter_names, "_state"]),
            draws=config.draws,
            tune=config.tune,
            chains=config.chains,
            cores=config.cores,
            seed=config.random_seed,
            target_accept=config.target_accept,
            adaptation=config.adaptation,
            progress_bar=config.progressbar,
            save_warmup=not config.discard_tuned_samples,
            blocking=True,
            **dict(config.sampler_kwargs),
        )
        if not isinstance(trace, xr.DataTree):
            raise TypeError("Nutpie must return an xarray.DataTree")
        trace["posterior"] = (
            trace["posterior"]
            .to_dataset()
            .assign_coords(walk_combination=list(self.model.coords["walk_combination"]))
        )
        trace["posterior"].attrs["oracle_binding_sha256"] = self.binding_id(bound)
        trace.attrs.update(
            walk_smoothness="inferred",
            compiled_instance_id=self.compile_instance_id,
            compiled_signature_count=len(self.kernel.signatures),
            boundary="zero-initialized adstock; all context rows retained",
        )
        return trace

    def predict_posterior(
        self, bound: BoundOracleKernel, posterior: xr.Dataset | xr.DataTree, *, random_seed: int
    ) -> xr.Dataset:
        """Return paired full-window trajectories for every supplied posterior draw.

        Input is a posterior Dataset or DataTree group returned by ``sample``;
        its private ``_state`` variable is required. This does not thin or score.
        Use the same immutable binding as the fit, not the worker's initial data.
        """
        import xarray as xr

        if isinstance(random_seed, bool) or not isinstance(random_seed, int) or random_seed < 0:
            raise ValueError("random_seed must be a nonnegative integer")
        if bound.kernel is not self.kernel:
            raise ValueError("binding belongs to a different family worker")
        posterior = posterior.to_dataset() if isinstance(posterior, xr.DataTree) else posterior
        if posterior.attrs.get("oracle_binding_sha256") != self.binding_id(bound):
            raise ValueError("posterior does not match this world's immutable binding")
        if "_state" not in posterior or not {"chain", "draw"}.issubset(posterior.dims):
            raise ValueError("posterior requires chain/draw dimensions and _state")
        if min(posterior.sizes["chain"], posterior.sizes["draw"]) < 1:
            raise ValueError("posterior must contain draws")
        rng = np.random.default_rng(random_seed)
        rows: dict[str, list] = {}
        for chain in range(posterior.sizes["chain"]):
            for draw in range(posterior.sizes["draw"]):
                result = self.predict(
                    bound, posterior["_state"].isel(chain=chain, draw=draw).values, rng
                )
                for name, value in result.items():
                    rows.setdefault(name, []).append(value)
        dims = {
            "baseline": ("time",),
            "outcome_mu": ("time",),
            "outcome": ("time",),
            "latent_unobserved": ("time", "latent"),
            "contributions": ("time", "channel"),
            "control_contributions": ("time", "control"),
            "walk_width": ("walk",),
            "walk_smoothness": ("walk",),
        }
        sizes = (posterior.sizes["chain"], posterior.sizes["draw"])
        predictions = xr.Dataset(
            {
                name: (
                    ("chain", "draw", *dims[name]),
                    np.asarray(value).reshape((*sizes, *np.asarray(value).shape[1:])),
                )
                for name, value in rows.items()
            },
            coords={
                "chain": posterior.chain,
                "draw": posterior.draw,
                "channel": np.asarray(bound.coords["channel"]),
                "control": np.asarray(bound.coords["control"]),
                "walk": ["baseline", *[f"latent_{j}" for j in range(self.cfg.n_latent)]],
            },
        )
        predictions.attrs.update(
            random_seed=random_seed,
            walk_smoothness="inferred",
            compiled_instance_id=self.compile_instance_id,
            boundary="zero-initialized adstock; all context rows retained",
        )
        return predictions

    def predict(
        self, bound: BoundOracleKernel, state: NDArray[np.float64], rng: np.random.Generator
    ) -> dict[str, NDArray[Any]]:
        """Joint paths for one draw, using this binding's priors (not the first world's)."""
        if bound.kernel is not self.kernel:
            raise ValueError("binding belongs to a different family worker")
        _, _, outputs = bound.evaluate(state)
        probs = outputs["_probabilities"]
        index = rng.choice(len(probs), p=probs / probs.sum())
        combinations = self.model._inferred_walk_components["combinations"]
        priors = bound.data[self.kernel.data_names.index("walk_priors")]
        prior = walk_width_prior(
            self.cfg.n_time_steps + self.cfg.carryover_burn_in,
            self.cfg.rw_smoothness_max_weeks,
            priors[8],
            priors[9],
        )
        indices = combinations[index]
        return {
            "baseline": outputs["_baseline"][index],
            "outcome_mu": outputs["_means"][index],
            "outcome": rng.normal(outputs["_means"][index], outputs["_noise"]),
            "latent_unobserved": outputs["_latent"][index],
            "contributions": outputs["_channels"],
            "control_contributions": outputs["_controls"],
            "walk_width": prior.widths[indices],
            "walk_smoothness": np.array([prior.sample_smoothness(int(i), rng) for i in indices]),
        }
