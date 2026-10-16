"""Active-dimension numerical kernel for family-specialized oracle workers.

Unlike Nutpie's ordinary PyMC bridge, flattening and output metadata are bound
per world, outside compiled code. No padded free parameters are introduced.
This is an opt-in integration primitive, not a replacement for CompiledOracle.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, cast

import numpy as np
import pymc as pm
import pytensor.tensor as pt
from numpy.typing import NDArray
from pytensor.graph.replace import graph_replace
from pytensor.tensor.type import TensorType


@dataclass(frozen=True)
class ValueLayout:
    """Concrete per-world shapes of a fixed ordered list of value variables."""

    names: tuple[str, ...]
    shapes: tuple[tuple[int, ...], ...]

    def __post_init__(self) -> None:
        if len(self.names) != len(self.shapes) or len(set(self.names)) != len(self.names):
            raise ValueError("value names must be unique and match shapes")
        if any(
            isinstance(n, bool) or not isinstance(n, (int, np.integer)) or n < 0
            for shape in self.shapes
            for n in shape
        ):
            raise ValueError("value dimensions must be nonnegative integers")

    @property
    def size(self) -> int:
        return sum(int(np.prod(shape, dtype=np.int64)) for shape in self.shapes)

    def unpack(self, position: NDArray[np.float64]) -> list[NDArray[np.float64]]:
        if position.shape != (self.size,) or position.dtype != np.dtype("float64"):
            raise ValueError("position must be a float64 vector matching the value layout")
        arrays = []
        offset = 0
        for shape in self.shapes:
            size = int(np.prod(shape, dtype=np.int64))
            # Own buffers at the lowered-function boundary. Callbacks share no
            # PyTensor storage maps and can run concurrently without mutation.
            arrays.append(np.array(position[offset : offset + size].reshape(shape), copy=True))
            offset += size
        return arrays


class DynamicOracleKernel:
    """Compile a dynamic PyMC graph once; bind only metadata/data for each world.

    The builder must derive RV dimensions from runtime data and vectorize over
    active inputs. This class cannot make a statically unrolled model dynamic.
    ``output_names`` identifies constrained parameters/deterministics to retain.
    All data containers must be explicitly supplied on every bind; omitted data,
    changed rank/dtype, and inconsistent outputs fail closed.
    """

    def __init__(self, model: pm.Model, *, output_names: Sequence[str]) -> None:
        if not output_names or len(set(output_names)) != len(output_names):
            raise ValueError("output_names must be nonempty and unique")
        self.value_names = tuple(v.name for v in model.value_vars)
        self.value_ranks = tuple(v.ndim for v in model.value_vars)
        if not self.value_names or any(v.dtype != "float64" for v in model.value_vars):
            raise ValueError("kernel requires continuous float64 value variables")
        self.output_names = tuple(output_names)
        self.data_names = tuple(sorted(v.name for v in model.data_vars))
        shared = [model[name] for name in self.data_names]
        self.data_types = tuple((v.dtype, v.ndim) for v in shared)
        inputs = [
            TensorType(dtype=v.dtype, shape=(None,) * v.ndim)(name=v.name + "_input")
            for v in shared
        ]
        outputs = model.replace_rvs_by_values([model[name] for name in output_names])
        logp = model.logp()
        gradients = cast(list[Any], pt.grad(logp, model.value_vars))
        expressions = graph_replace(
            [logp, *gradients, *outputs],
            replace=dict(zip(shared, inputs, strict=True)),
            strict=False,
        )
        function = pm.pytensorf.compile(
            [*model.value_vars, *inputs], expressions, mode="NUMBA", on_unused_input="ignore"
        )
        # Retain the owner as well as the stateless lowered function.
        self._function = function
        self._kernel = function.vm.jit_fn

    @property
    def signatures(self) -> tuple[Any, ...]:
        """Numba signatures for execution evidence (not a cache key)."""
        return tuple(self._kernel.signatures)

    def bind(
        self,
        *,
        value_shapes: Mapping[str, tuple[int, ...]],
        data: Mapping[str, NDArray[Any]],
        output_shapes: Mapping[str, tuple[int, ...]],
        coords: Mapping[str, Sequence[Any]],
        dims: Mapping[str, tuple[str, ...]],
        initial_point: NDArray[np.float64],
    ) -> BoundOracleKernel:
        """Make a world-specific immutable binding without compiling anything."""
        if set(value_shapes) != set(self.value_names):
            raise ValueError("value_shapes must describe every compiled value variable")
        layout = ValueLayout(self.value_names, tuple(value_shapes[n] for n in self.value_names))
        if tuple(map(len, layout.shapes)) != self.value_ranks:
            raise ValueError("value rank differs from compiled graph")
        if set(data) != set(self.data_names):
            raise ValueError("data must supply every compiled data container exactly once")
        if set(output_shapes) != set(self.output_names):
            raise ValueError("output_shapes must describe every requested output")
        buffers = []
        for name, (dtype, rank) in zip(self.data_names, self.data_types, strict=True):
            value = np.asarray(data[name])
            if value.dtype != np.dtype(dtype) or value.ndim != rank:
                raise ValueError(f"data {name!r} has incompatible dtype or rank")
            if not np.isfinite(value).all():
                raise ValueError(f"data {name!r} contains nonfinite values")
            buffer = np.array(value, order="C", copy=True)
            buffer.flags.writeable = False
            buffers.append(buffer)
        initial = np.array(initial_point, dtype="float64", copy=True)
        layout.unpack(initial)
        if not np.isfinite(initial).all():
            raise ValueError("initial point must be finite")
        initial.flags.writeable = False
        shapes = {name: tuple(output_shapes[name]) for name in self.output_names}
        for name, axes in dims.items():
            if name not in shapes or len(axes) != len(shapes[name]):
                raise ValueError(f"invalid output dimensions for {name!r}")
            if tuple(len(coords[axis]) for axis in axes) != shapes[name]:
                raise ValueError(f"coordinate lengths disagree with {name!r}")
        bound = BoundOracleKernel(
            self,
            layout,
            tuple(buffers),
            shapes,
            {k: tuple(v) for k, v in coords.items()},
            dict(dims),
            initial,
        )
        lp, gradient, _ = bound.evaluate(initial)
        if not np.isfinite(lp) or not np.isfinite(gradient).all():
            raise ValueError("initial point has nonfinite density or gradient")
        return bound


@dataclass(frozen=True)
class BoundOracleKernel:
    """One world's active layout, read-only data buffers, and fresh sampler factory."""

    kernel: DynamicOracleKernel
    layout: ValueLayout
    data: tuple[NDArray[Any], ...]
    output_shapes: dict[str, tuple[int, ...]]
    coords: dict[str, tuple[Any, ...]]
    dims: dict[str, tuple[str, ...]]
    initial_point: NDArray[np.float64]

    def evaluate(
        self, position: NDArray[np.float64]
    ) -> tuple[float, NDArray[np.float64], dict[str, NDArray[np.float64]]]:
        values = self.kernel._kernel(
            *self.layout.unpack(position), *(np.array(v, copy=True) for v in self.data)
        )
        n = len(self.layout.names)
        grad_parts = values[1 : n + 1]
        for gradient, shape in zip(grad_parts, self.layout.shapes, strict=True):
            if np.shape(gradient) != shape:
                raise ValueError("gradient shape disagrees with active value layout")
        gradient = np.concatenate([np.asarray(g).ravel() for g in grad_parts])
        outputs = {}
        for name, raw in zip(self.kernel.output_names, values[n + 1 :], strict=True):
            value = np.array(raw, dtype="float64", copy=True)
            if value.shape != self.output_shapes[name]:
                raise ValueError(f"output {name!r} shape disagrees with active layout")
            outputs[name] = value
        return float(values[0]), gradient, outputs

    def sampler_model(self, output_names: Sequence[str] | None = None) -> Any:
        """Create fresh Nutpie metadata; this is NOT a numerical compilation.

        Large predictive intermediates can remain kernel outputs without being
        saved for every unthinned sampling draw.
        """
        from nutpie.compiled_pyfunc import from_pyfunc  # type: ignore[import-untyped]

        names = list(self.kernel.output_names if output_names is None else output_names)
        if not names or len(set(names)) != len(names) or set(names) - set(self.output_shapes):
            raise ValueError("sampler outputs must be a nonempty unique subset of kernel outputs")

        def make_logp():
            def logp(position):
                value, gradient, _ = self.evaluate(position)
                return value, gradient

            return logp

        def make_expand(_seed1, _seed2, _chain):
            def expand(position):
                outputs = self.evaluate(position)[2]
                return {name: outputs[name] for name in names}

            return expand

        return from_pyfunc(
            ndim=self.layout.size,
            make_logp_fn=make_logp,
            make_expand_fn=make_expand,
            expanded_dtypes=[np.dtype("float64")] * len(names),
            expanded_shapes=[self.output_shapes[n] for n in names],
            expanded_names=names,
            coords={name: np.asarray(values) for name, values in self.coords.items()},
            dims={name: axes for name, axes in self.dims.items() if name in names},
            make_initial_point_fn=lambda _seed: self.initial_point.copy(),
        )
