"""The pointer/scalar ABI shared by CuTe compilation and generated launchers."""

import inspect
from collections.abc import Callable
from dataclasses import dataclass
from typing import get_type_hints

import cutlass
from cuda.bindings import (
    driver as cuda,  # ty: ignore[unresolved-import]  # Binary extension, no stubs.
)
from cutlass import cute
from qsutil.config import CuteSpec, DtypeConstant


@dataclass(frozen=True)
class Dtype:
    cute: type[cutlass.Numeric]
    marker: str | None
    rust: str | None


TYPES = {
    "bf16": Dtype(cutlass.BFloat16, "BF16", None),
    "f16": Dtype(cutlass.Float16, "F16", None),
    "f32": Dtype(cutlass.Float32, "F32", "f32"),
    "i8": Dtype(cutlass.Int8, "I8", "i8"),
    "i32": Dtype(cutlass.Int32, "I32", "i32"),
    "u8": Dtype(cutlass.Uint8, "U8", "u8"),
    "u32": Dtype(cutlass.Uint32, "U32", "u32"),
    "i64": Dtype(cutlass.Int64, None, "i64"),
    "u64": Dtype(cutlass.Uint64, None, "u64"),
    "fp8_e4m3": Dtype(cutlass.Float8E4M3FN, "Fp8E4M3", None),
}


@dataclass
class Argument:
    name: str
    kind: str
    dtype: str | None
    alignment: int | None
    rust: str


def signature(
    kernel: Callable[..., object], spec: CuteSpec
) -> tuple[list[object], list[Argument]]:
    function = inspect.unwrap(kernel)
    annotations = get_type_hints(function)
    values = []
    arguments = []
    constants = set()
    pointers = set()
    for name, param in inspect.signature(function).parameters.items():
        if param.kind not in (param.POSITIONAL_ONLY, param.POSITIONAL_OR_KEYWORD):
            raise ValueError(f"{name}: only positional kernel parameters are supported")
        annotation = annotations.get(name)
        if annotation is cutlass.Constexpr:
            value = spec.constants[name]
            values.append(
                TYPES[value.dtype].cute if isinstance(value, DtypeConstant) else value
            )
            constants.add(name)
        elif annotation is cute.Pointer:
            dtype = spec.precision[name]
            ty = TYPES[dtype]
            if ty.marker is None:
                raise ValueError(f"{name}: unsupported pointer dtype {dtype}")
            alignment = spec.alignments[name]
            if (
                type(alignment) is not int
                or alignment < 1
                or alignment & (alignment - 1)
            ):
                raise ValueError(f"{name}: alignment must be a positive power of two")
            values.append(
                cute.runtime.make_ptr(
                    ty.cute, alignment, cute.AddressSpace.gmem, assumed_align=alignment
                )
            )
            arguments.append(
                Argument(
                    name,
                    "pointer",
                    dtype,
                    alignment,
                    f"DevicePtr<{ty.marker}>",
                )
            )
            pointers.add(name)
        elif annotation is cuda.CUstream:
            values.append(cute.runtime.make_fake_stream())
            arguments.append(Argument(name, "stream", None, None, "*mut c_void"))
        else:
            selected = next(
                (
                    (dtype, ty)
                    for dtype, ty in TYPES.items()
                    if ty.cute is annotation and ty.rust is not None
                ),
                None,
            )
            if selected is None:
                raise ValueError(
                    f"{name}: expected cute.Pointer, cutlass scalar/Constexpr, or CUstream"
                )
            dtype, ty = selected
            assert ty.rust is not None
            values.append(ty.cute(0))
            arguments.append(Argument(name, "scalar", dtype, None, ty.rust))
    if constants != set(spec.constants):
        raise ValueError("constants must exactly match the Constexpr parameters")
    if pointers != set(spec.precision) or pointers != set(spec.alignments):
        raise ValueError(
            "precision and alignments must exactly match the pointer parameters"
        )
    if sum(a.kind == "stream" for a in arguments) != 1:
        raise ValueError("kernel must declare exactly one CUstream parameter")
    return values, arguments
