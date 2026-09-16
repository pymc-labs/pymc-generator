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
# Packed graph positions and all numeric arrays are unchanged.
CORPUS_SCHEMA_VERSION: int = 4

#: v1 corpus key -> v2 canonical key, applied by ``load_corpus``.
LEGACY_CORPUS_KEYS_V1: dict[str, str] = {
    "K_active": "n_treatments_active",
    "M_active": "n_covariates_active",
    "J_active": "n_latent_active",
    "active_c_mask": "treatment_active_mask",
    "active_m_mask": "covariate_active_mask",
    "active_j_mask": "latent_active_mask",
}

#: Historical v2 order and metadata keys; only the disk reader accepts these.
LEGACY_EDGE_TYPES_V2: tuple[str, ...] = ("cy", "dc", "dz", "db", "zb", "zc", "cc", "zz")
LEGACY_EDGE_KEYS_V2: dict[str, str] = {"db": "dy", "zb": "zy"}

#: v3 marketing-vocabulary corpus key -> v4 domain-neutral key, applied by
#: ``load_corpus``. Array contents, dtypes and shapes are untouched: this is a
#: pure renaming of the persisted vocabulary.
LEGACY_CORPUS_KEYS_V3: dict[str, str] = {
    "spend_raw": "treatment_raw",
    "spend_norm": "treatment_norm",
    "spend_share": "treatment_share",
    "spend_means": "treatment_means",
    "controls": "covariates",
    "sales_raw": "outcome_raw",
    "sales_norm": "outcome_norm",
    "sales_scale": "outcome_scale",
    "sales_noise": "outcome_noise",
    "demand": "latent_unobserved",
    "contributions_raw": "treatment_contribution_raw",
    "control_contribution": "covariate_contribution",
    "confounder_contribution": "latent_unobserved_contribution",
    "channel_active": "treatment_active",
    "channel_level": "treatment_level",
    "channel_shock_mask": "treatment_shock_mask",
    "channel_shock_channel": "treatment_shock_index",
    "channel_shock_start": "treatment_shock_start",
    "channel_shock_length": "treatment_shock_length",
    "channel_shock_level": "treatment_shock_level",
    "channel_shock_level_multiplier": "treatment_shock_level_multiplier",
    "adstock_family": "carryover_family",
    "adstock_alpha": "carryover_alpha",
}

#: v3 diagnostics metadata key -> v4 key, applied alongside the arrays.
LEGACY_DIAGNOSTIC_KEYS_V3: dict[str, str] = {
    "min_no_direct_effect_channels": "min_no_direct_effect_treatments",
    "adstock_kernel_version": "carryover_kernel_version",
    "adstock_kernel_semantics": "carryover_kernel_semantics",
}

#: v3 ``prior_cond`` column name -> v4 column name.
LEGACY_PRIOR_COND_COLUMNS_V3: dict[str, str] = {
    "adstock_alpha_low": "carryover_alpha_low",
    "adstock_alpha_width": "carryover_alpha_width",
}

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
