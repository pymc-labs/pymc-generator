"""Internal helpers extracted from research d61f98f; not a public API."""

import hashlib
from collections import defaultdict
from types import FunctionType


def clone_function(fn, **bindings):
    result = FunctionType(
        fn.__code__, {**fn.__globals__, **bindings}, fn.__name__, fn.__defaults__, fn.__closure__
    )
    result.__kwdefaults__ = fn.__kwdefaults__
    return result


def widths(counts):
    k, m, j = counts
    return {
        **dict.fromkeys(
            (
                "treatments",
                "contributions",
                "saturation_scale",
                "param_treatment_level",
                "param_carryover_alpha",
                "param_weibull_lam",
                "param_weibull_k",
                "treatment_shock_mask",
            ),
            k,
        ),
        "covariates": m,
        "covariate_contribution": m,
        "latent_unobserved": j,
        "latent_unobserved_contribution": j,
    }


def active_result(result, counts):
    sizes = widths(counts)
    return {
        name: (value[..., : sizes[name]] if name in sizes else value)
        for name, value in result.items()
    }


def digest(result):
    h = hashlib.sha256()
    for name, value in sorted(result.items()):
        h.update(name.encode())
        h.update(str(value.dtype).encode())
        h.update(str(value.shape).encode())
        h.update(value.tobytes())
    return h.hexdigest()


class Adapter:
    def __init__(self, method, cfg, cells, event):
        self.method, self.cfg, self.cells, self.event = method, cfg, cells, event
        self.models = {}
        self.draws = {}
        self.current_cell = 0
        self.times = defaultdict(float)
        self.accepted_ids = []
        self.retained_cell_ids = []
        self.evaluated = 0
        self.rejected = 0
        self.draw_failures = 0
        self.last_draw_error = None
        self.primary_hashes = {}
        self.executions = []

    def counts(self, cell):
        g = self.cells[cell].g
        return len(g["g_cy"]), len(g["g_zy"]), len(g["g_dy"])

    def padded_g(self, cell):
        data = self.cells[cell].template_inputs
        k, m, j = self.counts(cell)
        return {
            **{
                key: data[key].copy()
                for key in (
                    "g_cy",
                    "g_dc",
                    "g_dz",
                    "g_dy",
                    "g_zy",
                    "g_zc",
                    "g_cc",
                    "g_zz",
                    "active_treatment",
                    "active_covariate",
                    "active_latent",
                )
            },
            "treatment_active": ((data["g_cy"] != 0) | (data["g_cc"] != 0).any(axis=1)).astype(
                "uint8"
            ),
            "n_treatments_active": k,
            "n_covariates_active": m,
            "n_latent_active": j,
        }
