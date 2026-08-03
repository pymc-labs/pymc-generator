"""Canonical edge-slot layout (design-freeze constants).

This module is the design-freeze constants module: every other module imports
slot ordering, sizes, and base rates from here — no magic numbers elsewhere.

The additive SCM uses the extended 8-block edge layout
(`EDGE_TYPES_EXTENDED`):

    {C->Y, D->C, D->Z, D->B, Z->B, Z->C, C->C, Z->Z}.

Canonical g-vector ordering (LOCKED):

    [ g_cy (n_treatments) | g_dc (n_latent * n_treatments)
    | g_dz (n_latent * n_covariates) | g_db (n_latent) | g_zb (n_covariates)
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
N_TREATMENTS_DEMO = 4  # treatments (media channels)
N_COVARIATES_DEMO = 2  # covariates (observed controls)
N_LATENT_DEMO = 1  # latent factors (latent demand)
N_TIME_STEPS_DEMO = 104  # time steps (weeks) per task (design doc §6.1: 104–156)

# --------------------------------------------------------------------------
# Edge-slot Bernoulli base rates (KANBAN P0.5 / design doc §5)
# --------------------------------------------------------------------------
P_CY = 0.8  # C_k -> Y   (null channels possible: 1 - 0.8)
P_DC = 0.5  # D_j -> C_k (endogenous spend)
P_DB = 0.5  # D_j -> B   (confounding path)
P_ZB = 0.4  # Z_m -> B
P_ZC = 0.3  # Z_m -> C_k (extended layout, unused in demo)
P_DZ = 0.3  # D_j -> Z_m (extended layout)
P_CC = 0.15  # C_i -> C_k, i != k (extended layout)
P_ZZ = 0.1  # Z_i -> Z_m, i != m (extended layout)

# Edge types in the locked canonical block order.
EDGE_TYPES_EXTENDED: tuple[str, ...] = ("cy", "dc", "dz", "db", "zb", "zc", "cc", "zz")

EDGE_BASE_RATES: dict[str, float] = {
    "cy": P_CY,
    "dc": P_DC,
    "db": P_DB,
    "zb": P_ZB,
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
# 2: canonical descriptive names (see LEGACY_CORPUS_KEYS_V1). Written into
#    ``diagnostics["schema_version"]``; ``load_corpus`` migrates v1 shards on
#    read, ``save_corpus`` refuses to write v1 names.
CORPUS_SCHEMA_VERSION: int = 2

#: v1 corpus key -> v2 canonical key, applied by ``load_corpus``.
LEGACY_CORPUS_KEYS_V1: dict[str, str] = {
    "K_active": "n_treatments_active",
    "M_active": "n_covariates_active",
    "J_active": "n_latent_active",
    "active_c_mask": "treatment_active_mask",
    "active_m_mask": "covariate_active_mask",
    "active_j_mask": "latent_active_mask",
}

# --------------------------------------------------------------------------
# Prior-conditioning (ACE) layout — design-freeze constants (to-do 01)
# --------------------------------------------------------------------------
# Conditioned quantities, canonical order. Each contributes a packed
# ``(low, width)`` pair to the corpus ``prior_cond`` key. APPEND-ONLY: later
# quantities (e.g. Weibull lam/k, other saturation families) extend the tail;
# consumers index columns by name via PRIOR_COND_LAYOUT, never by position
# literals.
PRIOR_COND_QUANTITIES: tuple[str, ...] = ("adstock_alpha", "hill_shape")

#: Column names of the corpus ``prior_cond`` key, shape
#: (n_tasks, len(PRIOR_COND_LAYOUT)) — the packed
#: ``(low, width)`` pairs per conditioned quantity, canonical order (LOCKED,
#: append-only).
PRIOR_COND_LAYOUT: tuple[str, ...] = (
    "adstock_alpha_low",
    "adstock_alpha_width",
    "hill_shape_low",
    "hill_shape_width",
)


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
            "db": self.n_latent,
            "zb": self.n_covariates,
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
            "db": (self.n_latent,),
            "zb": (self.n_covariates,),
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
        g_db=None,
        g_zb=None,
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
            "db": g_db,
            "zb": g_zb,
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
