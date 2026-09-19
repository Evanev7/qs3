"""Typed compiler inputs; Nix owns model facts and build planning."""

import json
from dataclasses import dataclass
from typing import Literal

import dacite


@dataclass
class ComputeCapability:
    major: int
    minor: int


@dataclass
class CudaTarget:
    backend: Literal["cuda"]
    computeCapability: ComputeCapability
    warpSize: int


@dataclass
class DtypeConstant:
    dtype: str


@dataclass
class TritonSpec:
    precision: dict[str, str]
    # todo: type these
    constants: dict[str, bool | int | float | str | None | DtypeConstant]
    grid: list[int] | None
    # todo: type these
    options: dict[str, bool | int | float | str]


def parse[T](source: str, ty: type[T]) -> T:
    return dacite.from_dict(ty, json.loads(source), config=dacite.Config(strict=True))


@dataclass
class CuteSpec:
    precision: dict[str, str]
    alignments: dict[str, int]
    constants: dict[str, bool | int | float | str | None | DtypeConstant]
    # CuTe owns launch geometry in its host entrypoint. These are compiler options.
    options: dict[str, bool | int | str]
