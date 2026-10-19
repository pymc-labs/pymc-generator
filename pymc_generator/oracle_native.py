"""Fixed-capacity native Oracle templates with inferred walk smoothness.

Inactive input parameters retain proper priors. Data masks remove them from the
likelihood. Their slots are removed from public posterior and prediction arrays.
"""

from __future__ import annotations

import hashlib
import os
import time
from dataclasses import dataclass, replace
from typing import TYPE_CHECKING, Literal

import numpy as np
import pymc as pm
import xarray as xr

from .oracle_family import FamilyOracle, OracleInputs, _family_contract, _runtime_priors
from .oracle_smoothness import walk_width_prior
from .sampler import SCMPrior

if TYPE_CHECKING:
    from .oracle_sampling import CompiledOracle, OracleSamplingConfig, OracleSamplingResult


@dataclass
class NativeFamilyTemplate:
    """One fixed horizon, latent count, response family, and input capacity."""

    family: FamilyOracle
    max_channels: int
    max_controls: int
    latent: Literal["sampled"] = "sampled"

    @property
    def model(self) -> pm.Model:
        return self.family.model

    @property
    def data_contract(self) -> tuple[str, ...]:
        from .oracle_sampling import _graph_shared_variables

        names = [v.name for v in _graph_shared_variables(self.model)]
        if any(not isinstance(name, str) for name in names):
            raise ValueError("native family shared variables must have names")
        return tuple(sorted(str(name) for name in names))

    @property
    def signature(self) -> str:
        return hashlib.sha256(
            repr(
                ("native-family-v1", self.family.contract, self.max_channels, self.max_controls)
            ).encode()
        ).hexdigest()

    @classmethod
    def build(
        cls, cfg: SCMPrior, *, saturation: str, carryover: str, max_channels: int, max_controls: int
    ) -> NativeFamilyTemplate:
        if type(max_channels) is not int or max_channels < 1:
            raise ValueError("max_channels must be a positive integer")
        if type(max_controls) is not int or max_controls < 0:
            raise ValueError("max_controls must be a nonnegative integer")
        family = FamilyOracle(
            cfg, saturation=saturation, carryover=carryover, _capacity=(max_channels, max_controls)
        )
        sizes = {
            "channel": max_channels,
            "control": max_controls,
            "latent": cfg.n_latent,
            "time": cfg.n_time_steps,
            "innovation": cfg.n_time_steps + cfg.carryover_burn_in,
            "singleton": 1,
        }
        for axis, size in sizes.items():
            family.model.add_coord(axis, np.arange(size))
        family.model.named_vars_to_dims.update(family.axes)
        return cls(family, max_channels, max_controls)

    def payload(self, inputs: OracleInputs, cfg: SCMPrior | None = None):
        """Copy finite context observations. Query outcomes are not accepted."""
        if not isinstance(inputs, OracleInputs):
            raise TypeError("native family fits require OracleInputs")
        inputs.validate()
        cfg = self.family.cfg if cfg is None else cfg
        if (
            _family_contract(cfg, self.family.saturation, self.family.carryover)
            != self.family.contract
        ):
            raise ValueError("configuration is incompatible with native family template")
        t, p = inputs.channels.shape
        c = inputs.controls.shape[1]
        if t != cfg.n_time_steps or inputs.latent_gates.shape != (cfg.n_latent,):
            raise ValueError("horizon or latent count differs from template")
        if p > self.max_channels or c > self.max_controls:
            raise ValueError("active input count exceeds compiled capacity")
        # Safe placeholders keep nonlinear response calculations finite before masking.
        x = np.ones((t, self.max_channels))
        z = np.zeros((t, self.max_controls))
        x[:, :p], z[:, :c] = inputs.channels, inputs.controls
        anchors = np.ones(self.max_channels)
        anchors[:p] = inputs.saturation_scale
        cy, zy = np.zeros(self.max_channels), np.zeros(self.max_controls)
        cy[:p], zy[:c] = inputs.channel_gates, inputs.control_gates
        y, mask = np.zeros(t), np.zeros(t)
        y[inputs.context_indices], mask[inputs.context_indices] = inputs.context_outcome, 1.0
        prior = walk_width_prior(
            t + cfg.carryover_burn_in,
            cfg.rw_smoothness_max_weeks,
            cfg.rw_smoothness_alpha,
            cfg.rw_smoothness_beta,
        )
        data = {
            "channels": x,
            "controls": z,
            "anchor": anchors,
            "channel_gates": cy,
            "control_gates": zy,
            "latent_gates": np.array(inputs.latent_gates, copy=True),
            "context_indices": np.arange(t, dtype="int64"),
            "context_outcome": y,
            "context_mask": mask,
            "walk_priors": _runtime_priors(cfg),
            "width_probs": prior.probabilities.copy(),
        }
        data = {k: np.array(v, dtype=self.model[k].dtype, copy=True) for k, v in data.items()}
        assert set(self.data_contract) <= set(data)
        digest = hashlib.sha256(self.signature.encode())
        digest.update(repr((inputs.channel_ids, inputs.control_ids)).encode())
        for name, value in sorted(data.items()):
            digest.update(repr((name, value.shape, value.dtype.str)).encode())
            digest.update(value.tobytes())
        return data, digest.hexdigest()

    def fit(
        self,
        owner: CompiledOracle,
        inputs: OracleInputs,
        config: OracleSamplingConfig | None,
        *,
        cfg: SCMPrior | None = None,
        criteria=None,
        world_identity=None,
        source_identity=None,
        configuration_identity=None,
    ) -> OracleSamplingResult:
        from .oracle_sampling import (
            OracleHealthCriteria,
            OracleSamplingConfig,
            _immutable,
            _receipt_for_compiled_fit,
        )

        config = (
            OracleSamplingConfig(latent="sampled", walk_smoothness="inferred")
            if config is None
            else config
        )
        criteria = OracleHealthCriteria() if criteria is None else criteria
        if not isinstance(config, OracleSamplingConfig) or not isinstance(
            criteria, OracleHealthCriteria
        ):
            raise TypeError("config and criteria must use the Oracle sampling types")
        for identity in (world_identity, source_identity, configuration_identity):
            if identity is not None:
                from collections.abc import Mapping

                if not isinstance(identity, Mapping):
                    raise TypeError("identities must be mappings")
                _immutable(identity)
        config.validate()
        criteria.validate()
        if config.latent != "sampled" or config.walk_smoothness != "inferred":
            raise ValueError("native family fits require sampled latent and inferred smoothness")
        if config.nuts_sampler != "nutpie":
            raise ValueError("native family fits require nutpie")
        if os.getpid() != owner._owner_pid:
            raise RuntimeError("CompiledOracle is process-local")
        data, binding = self.payload(inputs, cfg)
        started = time.monotonic()
        if not callable(owner.sampler):
            raise TypeError("CompiledOracle requires a sampler")
        with owner._lock:
            bound = owner.compiled.with_data(**{k: data[k] for k in self.data_contract})
            trace = owner.sampler(
                bound,
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
            raise TypeError("nutpie must return an xarray.DataTree")
        # Keep full unconstrained state separately for exact joint reconstruction.
        # Public posterior and diagnostics must not include unused input slots.
        for group in ("posterior", "warmup_posterior"):
            if group not in trace:
                continue
            ds = trace[group].to_dataset()
            if group == "posterior":
                trace["oracle_state"] = ds[["_state"]]
            ds = ds.drop_vars("_state")
            ds = ds.isel(
                channel=slice(0, len(inputs.channel_ids)), control=slice(0, len(inputs.control_ids))
            )
            ds = ds.assign_coords(
                channel=list(inputs.channel_ids), control=list(inputs.control_ids)
            )
            ds.attrs["oracle_binding_sha256"] = binding
            trace[group] = ds
        trace["oracle_state"].attrs["oracle_binding_sha256"] = binding
        result = _receipt_for_compiled_fit(
            trace,
            config=config,
            criteria=criteria,
            started=started,
            template_signature=self.signature,
            compile_identity={
                "engine": "nutpie",
                "backend": "numba",
                "instance_id": owner._instance_id,
                "owner_pid": owner._owner_pid,
                "capacity": {"channels": self.max_channels, "controls": self.max_controls},
            },
            observed_indices=inputs.context_indices,
            payload=data,
            world_identity=world_identity,
            source_identity=source_identity,
            configuration_identity=configuration_identity,
        )
        receipt = replace(
            result.receipt,
            oracle=_immutable(
                {
                    "builder": "CompiledOracle.for_family",
                    "latent": "sampled",
                    "walk_smoothness": "inferred",
                    "binding_sha256": binding,
                    "observed_indices": "full_reported_window",
                }
            ),
            limitations=result.receipt.limitations
            + (
                "inactive_capacity_parameters_keep_priors_but_are_not_reported",
                "use_predict_posterior_for_joint_width_draws",
                "zero_initialized_adstock_all_context_rows_retained",
            ),
        )
        return replace(result, receipt=receipt)

    def predict(
        self,
        owner: CompiledOracle,
        inputs: OracleInputs,
        trace: xr.DataTree,
        *,
        random_seed: int,
        cfg: SCMPrior | None = None,
    ) -> xr.Dataset:
        """Reconstruct joint trajectories, not width-averaged conditional means."""
        if os.getpid() != owner._owner_pid:
            raise RuntimeError("CompiledOracle is process-local")
        data, binding = self.payload(inputs, cfg)
        state = trace["oracle_state"].to_dataset()
        if state.attrs.get("oracle_binding_sha256") != binding:
            raise ValueError("posterior belongs to a different immutable binding")
        padded = OracleInputs(
            channels=data["channels"],
            controls=data["controls"],
            context_indices=data["context_indices"],
            context_outcome=data["context_outcome"],
            saturation_scale=data["anchor"],
            channel_ids=tuple(range(self.max_channels)),
            control_ids=tuple(range(self.max_controls)),
            channel_gates=data["channel_gates"],
            control_gates=data["control_gates"],
            latent_gates=data["latent_gates"],
        )
        with owner._lock:
            # A separate, lazy prediction kernel. It is not a sampling callback.
            bound = self.family.bind(padded, cfg=cfg, _context_mask=data["context_mask"])
            state.attrs["oracle_binding_sha256"] = self.family.binding_id(bound)
            predicted = self.family.predict_posterior(bound, state, random_seed=random_seed)
        predicted = predicted.isel(
            channel=slice(0, len(inputs.channel_ids)), control=slice(0, len(inputs.control_ids))
        )
        predicted = predicted.assign_coords(
            channel=list(inputs.channel_ids), control=list(inputs.control_ids)
        )
        predicted.attrs.update(
            oracle_binding_sha256=binding, compiled_instance_id=owner._instance_id
        )
        return predicted
