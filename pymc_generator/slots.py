"""Canonical edge-slot layout (design-freeze constants).

This module is the design-freeze constants module: every other module imports
slot ordering, sizes, and base rates from here — no magic numbers elsewhere.

The additive SCM uses the extended 8-block edge layout
(`EDGE_TYPES_EXTENDED`):

    {C->Y, D->C, D->Z, D->Y, Z->Y, Z->C, C->C, Z->Z}.

Canonical g-vector ordering (LOCKED):

    [ g_cy (n_treatments) | g_dc (n_latent * n_treatments)
    | g_dz (n_latent * n_covariates) | g_dy (n_latent) | g_zy (n_covariates)
    | g_zc (n_covariates * n_treatments)
    | g_cc (n_treatments**2 - n_treatments)
    | g_zz (n_covariates**2 - n_covariates) ]

The cc and zz blocks EXCLUDE self-edges: the full (n_treatments, n_treatments)
/ (n_covariates, n_covariates) matrices carry a structurally-zero diagonal
which is dropped on `pack` (row-major over ordered pairs (i, k) with i != k)
and restored on `unpack`.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import cached_property

import numpy as np

# --------------------------------------------------------------------------
# Demo sizes — M0 milestone scale (KANBAN P1.1: fixed sizes, no padding)
# --------------------------------------------------------------------------
N_TREATMENTS_DEMO = 4  # treatments (treatment treatments)
N_COVARIATES_DEMO = 2  # covariates (observed covariates)
N_LATENT_DEMO = 1  # latent factors (latent latent_unobserved)
N_TIME_STEPS_DEMO = 104  # time steps (weeks) per task (design doc §6.1: 104–156)

# --------------------------------------------------------------------------
# Edge-slot Bernoulli base rates (KANBAN P0.5 / design doc §5)
# --------------------------------------------------------------------------
P_CY = 0.8  # C_k -> Y   (null treatments possible: 1 - 0.8)
P_DC = 0.5  # D_j -> C_k (endogenous treatment)
P_DY = 0.5  # D_j -> Y   (confounding path)
P_ZY = 0.4  # Z_m -> Y
P_ZC = 0.3  # Z_m -> C_k (extended layout, unused in demo)
P_DZ = 0.3  # D_j -> Z_m (extended layout)
P_CC = 0.15  # C_i -> C_k, i != k (extended layout)
P_ZZ = 0.1  # Z_i -> Z_m, i != m (extended layout)

# Edge types in the locked canonical block order.
EDGE_TYPES_EXTENDED: tuple[str, ...] = ("cy", "dc", "dz", "dy", "zy", "zc", "cc", "zz")

EDGE_BASE_RATES: dict[str, float] = {
    "cy": P_CY,
    "dc": P_DC,
    "dy": P_DY,
    "zy": P_ZY,
    "dz": P_DZ,
    "zc": P_ZC,
    "cc": P_CC,
    "zz": P_ZZ,
}

# Edge types whose block is packed from a full square matrix with the
# (structurally zero) diagonal dropped: type -> which size attr is the side.
_SQUARE_TYPES: dict[str, str] = {"cc": "n_treatments", "zz": "n_covariates"}

# --------------------------------------------------------------------------
# Persisted corpus schema version
# --------------------------------------------------------------------------
# 1: symbolic dimension keys (``K_active``, ``M_active``, ``J_active``,
#    ``active_c_mask``, ``active_m_mask``, ``active_j_mask``).
# 2: descriptive dimension keys, with legacy db/zb outcome-edge names.
# 3: dy/zy outcome-edge names and an explicitly direct-null treatment floor.
# 4: domain-neutral names — spend/sales/channel/control/demand/adstock become
#    treatment/outcome/covariate/latent_unobserved/carryover.
# 5: exact realised trajectory leaves and opt-in mechanism truth. Archives from
#    earlier versions cannot recover these draws and are rejected, not migrated.
CORPUS_SCHEMA_VERSION: int = 5

#: Historical symbolic dimension keys, retained only to reject stale vocabulary.
LEGACY_CORPUS_KEYS_V1: dict[str, str] = {
    "K_active": "n_treatments_active",
    "M_active": "n_covariates_active",
    "J_active": "n_latent_active",
    "active_c_mask": "treatment_active_mask",
    "active_m_mask": "covariate_active_mask",
    "active_j_mask": "latent_active_mask",
}

#: Historical outcome-edge metadata keys, retained only for rejection.
LEGACY_EDGE_KEYS_V2: dict[str, str] = {"db": "dy", "zb": "zy"}

# --------------------------------------------------------------------------
# Prior-conditioning (ACE) layout — design-freeze constants (to-do 01)
# --------------------------------------------------------------------------
# Conditioned quantities, canonical order. Each contributes a packed
# ``(low, width)`` pair to the corpus ``prior_cond`` key. APPEND-ONLY: later
# quantities (e.g. Weibull lam/k, other saturation families) extend the tail;
# consumers index columns by name via PRIOR_COND_LAYOUT, never by position
# literals.
PRIOR_COND_QUANTITIES: tuple[str, ...] = ("carryover_alpha", "hill_shape")

#: Column names of the corpus ``prior_cond`` key, shape
#: (n_tasks, len(PRIOR_COND_LAYOUT)) — the packed
#: ``(low, width)`` pairs per conditioned quantity, canonical order (LOCKED,
#: append-only).
PRIOR_COND_LAYOUT: tuple[str, ...] = (
    "carryover_alpha_low",
    "carryover_alpha_width",
    "hill_shape_low",
    "hill_shape_width",
)

#: Required corpus arrays: named axes and storage dtype. Optional metadata
#: blocks are validated separately; ``prior_cond`` follows PRIOR_COND_LAYOUT.
CORPUS_ARRAY_FIELDS: dict[str, tuple[tuple[str, ...], type[np.generic]]] = {
    "treatment_raw": (("task", "time", "treatment"), np.float32),
    "treatment_norm": (("task", "time", "treatment"), np.float32),
    "treatment_share": (("task", "time", "treatment"), np.float32),
    "covariates": (("task", "time", "covariate"), np.float32),
    "outcome_raw": (("task", "time"), np.float32),
    "outcome_norm": (("task", "time"), np.float32),
    "support_mask": (("task", "time"), np.uint8),
    "is_future": (("task",), np.uint8),
    "g": (("task", "edge"), np.uint8),
    "treatment_contribution_raw": (("task", "time", "treatment"), np.float32),
    "baseline_raw": (("task", "time"), np.float32),
    "latent_unobserved": (("task", "time", "latent"), np.float32),
    "treatment_means": (("task", "treatment"), np.float32),
    "outcome_scale": (("task",), np.float32),
    "is_val": (("task",), np.uint8),
    "cell_id": (("task",), np.int32),
    "treatment_active_mask": (("task", "treatment"), np.uint8),
    "covariate_active_mask": (("task", "covariate"), np.uint8),
    "latent_active_mask": (("task", "latent"), np.uint8),
    "n_treatments_active": (("task",), np.int32),
    "n_covariates_active": (("task",), np.int32),
    "n_latent_active": (("task",), np.int32),
    "confounding_strength": (("task",), np.float32),
    "indirect_effects": (("task", "time"), np.float32),
    "treatment_active": (("task", "treatment"), np.uint8),
    "covariate_contribution": (("task", "time", "covariate"), np.float32),
    "latent_unobserved_contribution": (("task", "time", "latent"), np.float32),
    "baseline_intrinsic": (("task", "time"), np.float32),
    "outcome_noise": (("task", "time"), np.float32),
    "indirect_effects_by_source": (("task", "time", "indirect_source"), np.float32),
    "treatment_shock_mask": (("task", "time", "treatment"), np.uint8),
    "treatment_shock_index": (("task", "shock"), np.int32),
    "treatment_shock_start": (("task", "shock"), np.int32),
    "treatment_shock_length": (("task", "shock"), np.int32),
    "treatment_shock_level_multiplier": (("task", "shock"), np.float32),
    "treatment_shock_level": (("task", "shock"), np.float32),
    "treatment_level": (("task", "treatment"), np.float32),
    "saturation_scale": (("task", "treatment"), np.float32),
    "carryover_family": (("task", "treatment"), np.uint8),
    "carryover_alpha": (("task", "treatment"), np.float32),
    "weibull_lam": (("task", "treatment"), np.float32),
    "weibull_k": (("task", "treatment"), np.float32),
}

# --------------------------------------------------------------------------
# Composable per-input trajectories (optional corpus block)
# --------------------------------------------------------------------------
#: Input series that carry trajectory components, in canonical order.
TRAJECTORY_INPUTS: tuple[str, ...] = ("treatment", "covariate")

#: Per-input trajectory components, in the canonical order of the stored
#: ``component`` axis and of ``diagnostics["trajectory"]``. ``hf`` and ``pulse``
#: are the existing texture terms; ``onset``/``offset``/``flighting`` are the
#: gate forms; ``level_jump``/``seasonal``/``trend`` are the level forms.
TRAJECTORY_COMPONENTS: tuple[str, ...] = (
    "hf",
    "pulse",
    "onset",
    "offset",
    "flighting",
    "level_jump",
    "seasonal",
    "trend",
)

#: Largest admitted treatment log-level swing ``A_hi + |B|_max + K·max|log f|``
#: (seasonal amplitude, trend change, jump count times jump size), i.e. a level
#: multiplier within ``[e^-3, e^3]`` (about x20). It keeps scheduled treatments at a
#: realism-filter scale and their decomposition representable in float32.
TRAJECTORY_MAX_LOG_SHIFT: float = 3.0

#: Optional trajectory arrays, present together (with ``diagnostics["trajectory"]``)
#: iff the config sets any trajectory inclusion knob. ``*_components`` are the
#: per-input 0/1 component flags; ``*_activity`` is the gate schedule (1 = on);
#: ``*_level_shift`` is the summed level component (treatments: log-level).
#: The pre-v5 schedule-output dtypes remain unchanged.
TRAJECTORY_ARRAY_FIELDS: dict[str, tuple[tuple[str, ...], type[np.generic]]] = {
    "treatment_components": (("task", "treatment", "component"), np.uint8),
    "covariate_components": (("task", "covariate", "component"), np.uint8),
    "treatment_activity": (("task", "time", "treatment"), np.uint8),
    "covariate_activity": (("task", "time", "covariate"), np.uint8),
    "treatment_log_level_shift": (("task", "time", "treatment"), np.float32),
    "covariate_level_shift": (("task", "time", "covariate"), np.float32),
}

#: Exact realised component leaves. Unselected components and inactive inputs
#: have zero padding, not an invented draw. Jump axes are role-specific because
#: treatment and covariate recipes may configure different event counts.
TRAJECTORY_PARAM_FIELDS: dict[str, tuple[tuple[str, ...], type[np.generic]]] = {}
for _input in TRAJECTORY_INPUTS:
    for _component, _leaves, _dtype in (
        ("hf", ("sigma",), np.float64),
        ("pulse", ("amp", "prob"), np.float64),
        ("onset", ("start",), np.int64),
        ("offset", ("stop",), np.int64),
        ("flighting", ("period", "on_weeks", "phase"), np.int64),
        ("level_jump", ("week",), np.int64),
        (
            "level_jump",
            ("factor", "log_factor") if _input == "treatment" else ("size",),
            np.float64,
        ),
        ("seasonal", ("amplitude", "period", "phase"), np.float64),
        ("trend", ("change",), np.float64),
    ):
        _axes = (
            ("task", f"{_input}_jump", _input) if _component == "level_jump" else ("task", _input)
        )
        for _leaf in _leaves:
            TRAJECTORY_PARAM_FIELDS[f"trajectory_{_input}_{_component}_{_leaf}"] = (
                _axes,
                _dtype,
            )
TRAJECTORY_ARRAY_FIELDS.update(TRAJECTORY_PARAM_FIELDS)

#: Shared world-model report for each persisted trajectory leaf. Texture
#: magnitudes already have reports outside the nested trajectory specification.
TRAJECTORY_PARAM_REPORTS: dict[str, str] = {}
for _key in TRAJECTORY_PARAM_FIELDS:
    _suffix = _key.removeprefix("trajectory_")
    _input, _component_leaf = _suffix.split("_", 1)
    if _component_leaf in ("hf_sigma", "pulse_amp", "pulse_prob"):
        _report = _component_leaf if _input == "treatment" else f"covariate_{_component_leaf}"
    else:
        _report = _key
    TRAJECTORY_PARAM_REPORTS[_key] = f"param_{_report}"

#: Opt-in response-mechanism truth. The exact saturation anchor is retained in
#: addition to the historical float32 ``saturation_scale`` feature. Shape draws
#: are reported for every active treatment, including unused response families.
MECHANISM_ARRAY_FIELDS: dict[str, tuple[tuple[str, ...], type[np.generic]]] = {
    "sat_family": (("task", "treatment"), np.uint8),
    "mechanism_saturation_scale": (("task", "treatment"), np.float64),
    "beta": (("task", "treatment"), np.float64),
    "rho_zy": (("task", "covariate"), np.float64),
    "hill_slope": (("task", "treatment"), np.float64),
    "hill_kappa_mult": (("task", "treatment"), np.float64),
    "logistic_lam": (("task", "treatment"), np.float64),
    "mm_kappa_mult": (("task", "treatment"), np.float64),
    "tanh_c": (("task", "treatment"), np.float64),
    "root_alpha": (("task", "treatment"), np.float64),
}

#: Recipe support corresponding to each exact saturation-shape field.
MECHANISM_PRIOR_FIELDS: dict[str, tuple[str, str]] = {
    "hill_slope": ("hill", "slope"),
    "hill_kappa_mult": ("hill", "kappa_mult"),
    "logistic_lam": ("logistic", "lam"),
    "mm_kappa_mult": ("michaelis_menten", "kappa_mult"),
    "tanh_c": ("tanh", "c"),
    "root_alpha": ("root", "alpha"),
}

#: Targets, actual reference inputs, and the treatment response at its reference
#: are present only for roles whose contribution prior is enabled. Retaining
#: the realised response preserves the generator's compiler-sensitive arithmetic.
MECHANISM_REFERENCE_FIELDS: dict[str, tuple[tuple[str, ...], type[np.generic]]] = {
    "treatment_reference_contribution": (("task", "treatment"), np.float64),
    "treatment_reference_input": (("task", "treatment"), np.float64),
    "treatment_reference_response": (("task", "treatment"), np.float64),
    "covariate_reference_contribution": (("task", "covariate"), np.float64),
    "covariate_reference_input": (("task", "covariate"), np.float64),
}


@dataclass(frozen=True)
class SlotLayout:
    """Canonical slot bookkeeping for one treatment/covariate/latent configuration.

    `edge_types` is the locked canonical extended block order.
    """

    n_treatments: int = N_TREATMENTS_DEMO
    n_covariates: int = N_COVARIATES_DEMO
    n_latent: int = N_LATENT_DEMO
    edge_types: tuple[str, ...] = EDGE_TYPES_EXTENDED

    def __post_init__(self) -> None:
        if self.edge_types != EDGE_TYPES_EXTENDED:
            raise ValueError(
                f"edge_types must match the locked canonical order {EDGE_TYPES_EXTENDED}"
            )

    @cached_property
    def block_sizes(self) -> dict[str, int]:
        all_sizes = {
            "cy": self.n_treatments,
            "dc": self.n_latent * self.n_treatments,
            "dz": self.n_latent * self.n_covariates,
            "dy": self.n_latent,
            "zy": self.n_covariates,
            "zc": self.n_covariates * self.n_treatments,
            "cc": self.n_treatments * (self.n_treatments - 1),
            "zz": self.n_covariates * (self.n_covariates - 1),
        }
        return {et: all_sizes[et] for et in self.edge_types}

    @cached_property
    def n_slots(self) -> int:
        return sum(self.block_sizes.values())

    @cached_property
    def slices(self) -> dict[str, slice]:
        """Slice of each edge-type block inside the canonical g-vector."""
        out, start = {}, 0
        for et in self.edge_types:
            size = self.block_sizes[et]
            out[et] = slice(start, start + size)
            start += size
        return out

    @cached_property
    def _block_shapes(self) -> dict[str, tuple[int, ...]]:
        """Trailing (unbatched) shape of each block as passed to `pack`."""
        return {
            "cy": (self.n_treatments,),
            "dc": (self.n_latent, self.n_treatments),
            "dz": (self.n_latent, self.n_covariates),
            "dy": (self.n_latent,),
            "zy": (self.n_covariates,),
            "zc": (self.n_covariates, self.n_treatments),
            "cc": (self.n_treatments, self.n_treatments),
            "zz": (self.n_covariates, self.n_covariates),
        }

    # -- helpers ------------------------------------------------------------
    def _square_mask(self, n: int) -> np.ndarray:
        """(n, n) boolean off-diagonal mask (row-major True positions)."""
        return ~np.eye(n, dtype=bool)

    def pack(
        self,
        g_cy=None,
        g_dc=None,
        g_dy=None,
        g_zy=None,
        *,
        g_dz=None,
        g_zc=None,
        g_cc=None,
        g_zz=None,
    ) -> np.ndarray:
        """Pack per-type arrays into the canonical flat g-vector.

        g_dc is (n_latent, n_treatments) and is raveled row-major (j, k) — the
        locked order. Extended blocks: g_dz is (n_latent, n_covariates), g_zc
        is (n_covariates, n_treatments), g_cc is the FULL
        (n_treatments, n_treatments) matrix with an all-zero diagonal (raises
        ValueError otherwise) packed by dropping the diagonal row-major over
        (i, k) with i != k; g_zz is (n_covariates, n_covariates), handled the
        same way.
        Works on a single task or a leading batch axis.
        """
        provided = {
            "cy": g_cy,
            "dc": g_dc,
            "dy": g_dy,
            "zy": g_zy,
            "dz": g_dz,
            "zc": g_zc,
            "cc": g_cc,
            "zz": g_zz,
        }
        missing = [et for et in self.edge_types if provided[et] is None]
        if missing:
            raise ValueError(
                f"pack() missing required blocks {missing} for edge_types={self.edge_types}"
            )
        arrays = {et: np.asarray(provided[et]) for et in self.edge_types}

        # Batch shape from the first block (vector or matrix trailing dims).
        first = self.edge_types[0]
        first_ndim = len(self._block_shapes[first])
        batch = arrays[first].shape[:-first_ndim]

        flats: list[np.ndarray] = []
        for et in self.edge_types:
            arr = arrays[et]
            shape = self._block_shapes[et]
            size = self.block_sizes[et]
            if len(shape) == 1:
                if arr.shape[-1] != shape[0]:
                    raise ValueError(f"g_{et} last dim must be {shape[0]}, got {arr.shape[-1]}")
                flats.append(arr.reshape(*batch, size))
            elif et in _SQUARE_TYPES:
                n = shape[0]
                if arr.ndim < 2 or arr.shape[-2:] != shape:
                    raise ValueError(
                        f"g_{et} trailing dims must be the full {shape} matrix, "
                        f"got shape {arr.shape}"
                    )
                diag = np.diagonal(arr, axis1=-2, axis2=-1)
                if np.any(diag != 0):
                    raise ValueError(
                        f"g_{et} diagonal must be all zeros (no self-edges), got diagonal {diag}"
                    )
                flats.append(arr[..., self._square_mask(n)].reshape(*batch, size))
            else:
                if arr.ndim >= 2:
                    if arr.shape[-2:] != shape:
                        raise ValueError(
                            f"g_{et} trailing dims must be {shape}, got {arr.shape[-2:]}"
                        )
                elif arr.size != size:
                    raise ValueError(
                        f"g_{et} must have shape (..., {shape[0]}, {shape[1]}) "
                        f"or total size {size}, got shape {arr.shape}"
                    )
                flats.append(arr.reshape(*batch, size))
        return np.concatenate(flats, axis=-1)

    def unpack(self, g_vec: np.ndarray) -> dict[str, np.ndarray]:
        """Inverse of `pack`; returns dict with shaped per-type arrays.

        Square blocks ("cc", "zz") come back as FULL matrices with a zero
        diagonal.
        """
        g_vec = np.asarray(g_vec)
        if g_vec.shape[-1] != self.n_slots:
            raise ValueError(
                f"g_vec last dim must be n_slots={self.n_slots}, got {g_vec.shape[-1]}"
            )
        batch = g_vec.shape[:-1]
        s = self.slices
        out: dict[str, np.ndarray] = {}
        for et in self.edge_types:
            block = g_vec[..., s[et]]
            shape = self._block_shapes[et]
            if et in _SQUARE_TYPES:
                n = shape[0]
                full = np.zeros((*batch, n, n), dtype=g_vec.dtype)
                full[..., self._square_mask(n)] = block
                out[et] = full
            else:
                out[et] = block.reshape(*batch, *shape)
        return out
