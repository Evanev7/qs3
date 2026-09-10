from __future__ import annotations

import json
import struct
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

from .schema import MANIFEST_FILE, TensorSpec, VectorManifest

STRUCT_FORMATS: dict[str, str] = {
    "f32": "f",
    "f64": "d",
    "i8": "b",
    "u8": "B",
    "i16": "h",
    "u16": "H",
    "i32": "i",
    "u32": "I",
    "i64": "q",
    "u64": "Q",
    "bf16": "H",
}


def write_manifest(root: str | Path, manifest: VectorManifest) -> Path:
    root_path = Path(root)
    root_path.mkdir(parents=True, exist_ok=True)
    path = root_path / MANIFEST_FILE
    payload = json.dumps(manifest.to_json(), indent=2, sort_keys=True)
    path.write_text(f"{payload}\n", encoding="utf-8")
    return path


def read_manifest(root: str | Path) -> VectorManifest:
    path = Path(root) / MANIFEST_FILE
    data = json.loads(path.read_text(encoding="utf-8"))
    return VectorManifest.from_json(data)


def write_tensor(
    root: str | Path,
    spec: TensorSpec,
    values: Iterable[Any],
    *,
    encode_bf16: bool = False,
) -> Path:
    path = _tensor_path(root, spec)
    path.parent.mkdir(parents=True, exist_ok=True)
    if spec.dtype == "bf16" and encode_bf16:
        materialized = float32_to_bf16_words(values)
    else:
        materialized = list(values)
    if len(materialized) != spec.element_count:
        raise ValueError(
            f"{spec.name}: expected {spec.element_count} values, got {len(materialized)}"
        )
    path.write_bytes(_pack_values(spec.dtype, materialized))
    return path


def read_tensor(
    root: str | Path,
    spec: TensorSpec,
    *,
    decode_bf16: bool = False,
) -> list[Any]:
    path = _tensor_path(root, spec)
    raw = path.read_bytes()
    if len(raw) != spec.byte_count:
        raise ValueError(
            f"{spec.name}: expected {spec.byte_count} bytes, got {len(raw)} in {path}"
        )
    values = _unpack_values(spec.dtype, raw, spec.element_count)
    if spec.dtype == "bf16" and decode_bf16:
        return bf16_words_to_float32(values)
    return values


def write_artifact(
    root: str | Path,
    manifest: VectorManifest,
    tensors: Mapping[str, Iterable[Any]],
    *,
    encode_bf16: bool = False,
) -> Path:
    tensor_names = {tensor.name for tensor in manifest.tensors}
    provided_names = set(tensors)
    missing = sorted(tensor_names - provided_names)
    unexpected = sorted(provided_names - tensor_names)
    if missing:
        raise ValueError(f"missing tensor values for: {', '.join(missing)}")
    if unexpected:
        raise ValueError(f"unexpected tensor values for: {', '.join(unexpected)}")
    for spec in manifest.tensors:
        write_tensor(root, spec, tensors[spec.name], encode_bf16=encode_bf16)
    return write_manifest(root, manifest)


def read_artifact(
    root: str | Path,
    *,
    decode_bf16: bool = False,
) -> tuple[VectorManifest, dict[str, list[Any]]]:
    manifest = read_manifest(root)
    tensors = {
        spec.name: read_tensor(root, spec, decode_bf16=decode_bf16)
        for spec in manifest.tensors
    }
    return manifest, tensors


def float32_to_bf16_bits(value: float) -> int:
    bits = struct.unpack("<I", struct.pack("<f", float(value)))[0]
    if (bits & 0x7FFFFFFF) > 0x7F800000:
        return ((bits >> 16) | 0x0040) & 0xFFFF
    rounding_bias = 0x7FFF + ((bits >> 16) & 1)
    return ((bits + rounding_bias) >> 16) & 0xFFFF


def bf16_bits_to_float32(bits: int) -> float:
    bits = _validate_u16(bits)
    return struct.unpack("<f", struct.pack("<I", bits << 16))[0]


def float32_to_bf16_words(values: Iterable[Any]) -> list[int]:
    return [float32_to_bf16_bits(float(value)) for value in values]


def bf16_words_to_float32(values: Iterable[int]) -> list[float]:
    return [bf16_bits_to_float32(value) for value in values]


def _pack_values(dtype: str, values: list[Any]) -> bytes:
    if not values:
        return b""
    fmt = STRUCT_FORMATS[dtype]
    if dtype == "bf16":
        values = [_validate_u16(value) for value in values]
    return struct.pack(f"<{len(values)}{fmt}", *values)


def _unpack_values(dtype: str, raw: bytes, count: int) -> list[Any]:
    if count == 0:
        return []
    return list(struct.unpack(f"<{count}{STRUCT_FORMATS[dtype]}", raw))


def _tensor_path(root: str | Path, spec: TensorSpec) -> Path:
    return Path(root) / spec.file


def _validate_u16(value: Any) -> int:
    if not isinstance(value, int):
        raise TypeError(f"expected raw u16 BF16 word, got {type(value).__name__}")
    if value < 0 or value > 0xFFFF:
        raise ValueError(f"raw u16 BF16 word out of range: {value}")
    return value
