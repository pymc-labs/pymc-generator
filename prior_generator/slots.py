"""Canonical edge-slot layout (design-freeze constants).

This module is the design-freeze constants module: every other module imports
slot ordering, sizes, and base rates from here — no magic numbers elsewhere.

Node types:
    C_1..C_K  media channels        (spend observed)
    Z_1..Z_M  observed controls
    D_1..D_J  latent demand factors (never observed)
    B         baseline              (always present, always -> Y)
    Y         sales/revenue         (sink)

Base edge set = {C->Y, D->C, D->B, Z->B} (`EDGE_TYPES`, the legacy 4-block
layout); the additive SCM uses the extended 8-block layout
(`EDGE_TYPES_EXTENDED`), which adds {D->Z, Z->C, C->C, Z->Z}.

Canonical g-vector ordering (LOCKED):

    [ g_cy (K) | g_dc (J*K, row-major over (j, k)) | g_db (J) | g_zb (M) ]

This matches the `pt.concatenate` order used in `exploration/01–03` restricted
to the L0 edge set. Slots are defined relative to *node identities*: permuting
channels permutes the cy block and the k-axis of the dc block identically —
the permutation-equivariance tests (P1.4) rely on exactly this.

Extended g-vector ordering (Phase 4, LOCKED once adopted):

    [ g_cy (K) | g_dc (J*K) | g_dz (J*M) | g_db (J) | g_zb (M)
    | g_zc (M*K) | g_cc (K*K - K) | g_zz (M*M - M) ]

The cc and zz blocks EXCLUDE self-edges: the full (K, K) / (M, M) matrices
carry a structurally-zero diagonal which is dropped on `pack` (row-major over
ordered pairs (i, k) with i != k) and restored on `unpack`. Layouts opt into
the extension via `SlotLayout(edge_types=EDGE_TYPES_EXTENDED)`; the default
`edge_types` keeps the legacy 4-block layout byte-identical.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from functools import cached_property

import numpy as np

# --------------------------------------------------------------------------
# Demo sizes — M0 milestone scale (KANBAN P1.1: fixed sizes, no padding)
# --------------------------------------------------------------------------
K_DEMO = 4  # media channels
M_DEMO = 2  # observed controls
J_DEMO = 1  # latent demand factors
T_DEMO = 104  # weeks per task (design doc §6.1: T = 104–156)

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

# Node-type ids for type embeddings (network + edge head)
NODE_CHANNEL, NODE_CONTROL, NODE_DEMAND, NODE_BASELINE, NODE_SALES = 0, 1, 2, 3, 4
N_NODE_TYPES = 5

# Edge types in canonical block order, with (src_type, dst_type)
EDGE_TYPES: tuple[str, ...] = ("cy", "dc", "db", "zb")
# Extended block order (Phase 4): superset of EDGE_TYPES, new canonical order
EDGE_TYPES_EXTENDED: tuple[str, ...] = ("cy", "dc", "dz", "db", "zb", "zc", "cc", "zz")
EDGE_TYPE_NODES: dict[str, tuple[int, int]] = {
    "cy": (NODE_CHANNEL, NODE_SALES),
    "dc": (NODE_DEMAND, NODE_CHANNEL),
    "db": (NODE_DEMAND, NODE_BASELINE),
    "zb": (NODE_CONTROL, NODE_BASELINE),
    "dz": (NODE_DEMAND, NODE_CONTROL),
    "zc": (NODE_CONTROL, NODE_CHANNEL),
    "cc": (NODE_CHANNEL, NODE_CHANNEL),
    "zz": (NODE_CONTROL, NODE_CONTROL),
}
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
_SQUARE_TYPES: dict[str, str] = {"cc": "K", "zz": "M"}

# --------------------------------------------------------------------------
# Prior-conditioning (ACE) layout — design-freeze constants (to-do 01)
# --------------------------------------------------------------------------
# Conditioned quantities, canonical order. Each contributes a packed
# ``(low, width)`` pair to the corpus ``prior_cond`` key. APPEND-ONLY: later
# quantities (e.g. Weibull lam/k, other saturation families) extend the tail;
# consumers index columns by name via PRIOR_COND_LAYOUT, never by position
# literals.
PRIOR_COND_QUANTITIES: tuple[str, ...] = ("adstock_alpha", "hill_shape")

#: Column names of the corpus ``prior_cond`` key, shape (N, P) — the packed
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
    """Canonical slot bookkeeping for one (K, M, J) configuration.

    `edge_types` selects the blocks (and their order) in the flat g-vector.
    The default is the legacy 4-block demo layout; pass
    `edge_types=EDGE_TYPES_EXTENDED` for the Phase-4 extended layout.
    """

    K: int = K_DEMO
    M: int = M_DEMO
    J: int = J_DEMO
    edge_types: tuple[str, ...] = field(default=EDGE_TYPES)

    def __post_init__(self) -> None:
        unknown = [et for et in self.edge_types if et not in EDGE_TYPES_EXTENDED]
        if unknown:
            raise ValueError(f"Unknown edge types {unknown}; known types: {EDGE_TYPES_EXTENDED}")
        if len(set(self.edge_types)) != len(self.edge_types):
            raise ValueError(f"Duplicate edge types in {self.edge_types}")

    @cached_property
    def block_sizes(self) -> dict[str, int]:
        all_sizes = {
            "cy": self.K,
            "dc": self.J * self.K,
            "dz": self.J * self.M,
            "db": self.J,
            "zb": self.M,
            "zc": self.M * self.K,
            "cc": self.K * (self.K - 1),
            "zz": self.M * (self.M - 1),
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
            "cy": (self.K,),
            "dc": (self.J, self.K),
            "dz": (self.J, self.M),
            "db": (self.J,),
            "zb": (self.M,),
            "zc": (self.M, self.K),
            "cc": (self.K, self.K),
            "zz": (self.M, self.M),
        }

    @cached_property
    def names(self) -> list[str]:
        """Human-readable slot names, canonical order (1-based node ids)."""
        K, M, J = self.K, self.M, self.J
        per_type: dict[str, list[str]] = {
            "cy": [f"C{k + 1}->Y" for k in range(K)],
            "dc": [f"D{j + 1}->C{k + 1}" for j in range(J) for k in range(K)],
            "dz": [f"D{j + 1}->Z{m + 1}" for j in range(J) for m in range(M)],
            "db": [f"D{j + 1}->B" for j in range(J)],
            "zb": [f"Z{m + 1}->B" for m in range(M)],
            "zc": [f"Z{m + 1}->C{k + 1}" for m in range(M) for k in range(K)],
            "cc": [f"C{i + 1}->C{k + 1}" for i in range(K) for k in range(K) if i != k],
            "zz": [f"Z{i + 1}->Z{m + 1}" for i in range(M) for m in range(M) if i != m],
        }
        names: list[str] = []
        for et in self.edge_types:
            names += per_type[et]
        return names

    @cached_property
    def base_rates(self) -> np.ndarray:
        """(n_slots,) Bernoulli base rate per slot, canonical order."""
        out = np.empty(self.n_slots, dtype="float64")
        for et in self.edge_types:
            out[self.slices[et]] = EDGE_BASE_RATES[et]
        return out

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

        g_dc is (J, K) and is raveled row-major (j, k) — the locked order.
        Extended blocks (only for layouts that include them): g_dz is (J, M),
        g_zc is (M, K), g_cc is the FULL (K, K) matrix with an all-zero
        diagonal (raises ValueError otherwise) packed by dropping the diagonal
        row-major over (i, k) with i != k; g_zz is (M, M), same treatment.
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
        diagonal. Only blocks present in `self.edge_types` are returned.
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

    def _repack(self, parts: dict[str, np.ndarray]) -> np.ndarray:
        """Pack from an unpack-style dict (only keys in `self.edge_types`)."""
        return self.pack(
            g_cy=parts.get("cy"),
            g_dc=parts.get("dc"),
            g_db=parts.get("db"),
            g_zb=parts.get("zb"),
            g_dz=parts.get("dz"),
            g_zc=parts.get("zc"),
            g_cc=parts.get("cc"),
            g_zz=parts.get("zz"),
        )

    def permute_channels(self, g_vec: np.ndarray, perm: np.ndarray) -> np.ndarray:
        """g-vector after relabeling channels by `perm` (slot k -> perm[k]).

        `perm` maps NEW position -> OLD index, i.e. new_cy = old_cy[perm].
        Extended layouts also permute the k-axis of zc and BOTH axes of the
        full cc matrix. Used by the equivariance tests (P1.4).
        """
        parts = self.unpack(g_vec)
        if "cy" in parts:
            parts["cy"] = parts["cy"][..., perm]
        if "dc" in parts:
            parts["dc"] = parts["dc"][..., :, perm]
        if "zc" in parts:
            parts["zc"] = parts["zc"][..., :, perm]
        if "cc" in parts:
            parts["cc"] = parts["cc"][..., perm, :][..., :, perm]
        return self._repack(parts)

    def permute_controls(self, g_vec: np.ndarray, perm: np.ndarray) -> np.ndarray:
        """g-vector after relabeling controls by `perm` (slot m -> perm[m])."""
        parts = self.unpack(g_vec)
        if "zb" in parts:
            parts["zb"] = parts["zb"][..., perm]
        if "dz" in parts:
            parts["dz"] = parts["dz"][..., :, perm]
        if "zc" in parts:
            parts["zc"] = parts["zc"][..., perm, :]
        if "zz" in parts:
            parts["zz"] = parts["zz"][..., perm, :][..., :, perm]
        return self._repack(parts)


DEMO_LAYOUT = SlotLayout(K=K_DEMO, M=M_DEMO, J=J_DEMO)  # n_slots = 11
