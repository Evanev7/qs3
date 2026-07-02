from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import PurePosixPath
from typing import Any


SCHEMA_VERSION = 1
MANIFEST_FILE = "manifest.json"
BYTE_ORDER = "little"

DTYPE_EXTENSIONS: dict[str, str] = {
    "bf16": ".bf16",
    "f32": ".f32",
    "f64": ".f64",
    "i8": ".i8",
    "u8": ".u8",
    "i16": ".i16",
    "u16": ".u16",
    "i32": ".i32",
    "u32": ".u32",
    "i64": ".i64",
    "u64": ".u64",
}

DTYPE_BYTE_WIDTHS: dict[str, int] = {
    "bf16": 2,
    "f32": 4,
    "f64": 8,
    "i8": 1,
    "u8": 1,
    "i16": 2,
    "u16": 2,
    "i32": 4,
    "u32": 4,
    "i64": 8,
    "u64": 8,
}


@dataclass(frozen=True)
class TensorSpec:
    name: str
    dtype: str
    shape: tuple[int, ...]
    file: str | None = None
    role: str = ""
    description: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        name = _validate_name(self.name, "tensor name")
        dtype = _validate_dtype(self.dtype)
        shape = _validate_shape(self.shape)
        file_name = self.file or default_tensor_file(name, dtype)
        object.__setattr__(self, "name", name)
        object.__setattr__(self, "dtype", dtype)
        object.__setattr__(self, "shape", shape)
        object.__setattr__(self, "file", _validate_flat_file(file_name))
        object.__setattr__(self, "metadata", dict(self.metadata))

    @property
    def element_count(self) -> int:
        count = 1
        for dim in self.shape:
            count *= dim
        return count

    @property
    def byte_count(self) -> int:
        return self.element_count * DTYPE_BYTE_WIDTHS[self.dtype]

    def to_json(self) -> dict[str, Any]:
        data: dict[str, Any] = {
            "name": self.name,
            "dtype": self.dtype,
            "shape": list(self.shape),
            "file": self.file,
        }
        if self.role:
            data["role"] = self.role
        if self.description:
            data["description"] = self.description
        if self.metadata:
            data["metadata"] = self.metadata
        return data

    @classmethod
    def from_json(cls, data: dict[str, Any]) -> TensorSpec:
        return cls(
            name=data["name"],
            dtype=data["dtype"],
            shape=tuple(data["shape"]),
            file=data.get("file"),
            role=data.get("role", ""),
            description=data.get("description", ""),
            metadata=dict(data.get("metadata", {})),
        )


@dataclass(frozen=True)
class VectorManifest:
    name: str
    tensors: tuple[TensorSpec, ...] = ()
    schema_version: int = SCHEMA_VERSION
    byte_order: str = BYTE_ORDER
    groups: tuple[str, ...] = ()
    description: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.schema_version != SCHEMA_VERSION:
            raise ValueError(
                f"unsupported vector manifest schema_version {self.schema_version}"
            )
        if self.byte_order != BYTE_ORDER:
            raise ValueError(f"unsupported byte_order {self.byte_order!r}")
        object.__setattr__(self, "name", _validate_name(self.name, "manifest name"))
        object.__setattr__(self, "tensors", tuple(self.tensors))
        groups = tuple(_validate_name(group, "group") for group in self.groups)
        object.__setattr__(self, "groups", groups)
        object.__setattr__(self, "metadata", dict(self.metadata))
        _validate_unique_tensors(self.tensors)

    def to_json(self) -> dict[str, Any]:
        data: dict[str, Any] = {
            "schema_version": self.schema_version,
            "byte_order": self.byte_order,
            "name": self.name,
            "groups": list(self.groups),
            "tensors": [tensor.to_json() for tensor in self.tensors],
        }
        if self.description:
            data["description"] = self.description
        if self.metadata:
            data["metadata"] = self.metadata
        return data

    @classmethod
    def from_json(cls, data: dict[str, Any]) -> VectorManifest:
        return cls(
            schema_version=data["schema_version"],
            byte_order=data.get("byte_order", BYTE_ORDER),
            name=data["name"],
            groups=tuple(data.get("groups", ())),
            tensors=tuple(TensorSpec.from_json(item) for item in data.get("tensors", ())),
            description=data.get("description", ""),
            metadata=dict(data.get("metadata", {})),
        )

    def tensor(self, name: str) -> TensorSpec:
        for tensor in self.tensors:
            if tensor.name == name:
                return tensor
        raise KeyError(name)


def default_tensor_file(name: str, dtype: str) -> str:
    suffix = DTYPE_EXTENSIONS[_validate_dtype(dtype)]
    stem = "".join(
        ch if ch.isascii() and (ch.isalnum() or ch in "._-") else "_"
        for ch in name
    )
    stem = stem.strip("._-") or "tensor"
    return f"{stem}{suffix}"


def _validate_name(value: str, label: str) -> str:
    if not isinstance(value, str):
        raise TypeError(f"{label} must be a string")
    value = value.strip()
    if not value:
        raise ValueError(f"{label} must not be empty")
    return value


def _validate_dtype(dtype: str) -> str:
    if dtype not in DTYPE_EXTENSIONS:
        known = ", ".join(sorted(DTYPE_EXTENSIONS))
        raise ValueError(f"unsupported dtype {dtype!r}; expected one of: {known}")
    return dtype


def _validate_shape(shape: tuple[int, ...]) -> tuple[int, ...]:
    if isinstance(shape, list):
        shape = tuple(shape)
    if not isinstance(shape, tuple):
        raise TypeError("shape must be a tuple or list of integers")
    for dim in shape:
        if not isinstance(dim, int):
            raise TypeError("shape dimensions must be integers")
        if dim < 0:
            raise ValueError("shape dimensions must be non-negative")
    return shape


def _validate_flat_file(file_name: str) -> str:
    if not isinstance(file_name, str):
        raise TypeError("tensor file must be a string")
    if not file_name or file_name in {".", "..", MANIFEST_FILE}:
        raise ValueError(f"invalid tensor file {file_name!r}")
    path = PurePosixPath(file_name)
    if path.name != file_name or "\\" in file_name:
        raise ValueError(f"tensor file must be a flat relative path: {file_name!r}")
    return file_name


def _validate_unique_tensors(tensors: tuple[TensorSpec, ...]) -> None:
    names: set[str] = set()
    files: set[str] = set()
    for tensor in tensors:
        if tensor.name in names:
            raise ValueError(f"duplicate tensor name {tensor.name!r}")
        if tensor.file in files:
            raise ValueError(f"duplicate tensor file {tensor.file!r}")
        names.add(tensor.name)
        files.add(tensor.file)
