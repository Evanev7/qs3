"""Compile an explicit CuTe host entrypoint and export its native AOT launcher."""

import argparse
import hashlib
import importlib.metadata
import importlib.util
import json
import re
import shlex
import sys
from collections.abc import Callable
from dataclasses import asdict
from os.path import relpath
from pathlib import Path

import qsutil
from cutlass import cute
from cutlass.cutlass_dsl import CuTeDSL
from qsutil.config import CudaTarget, CuteSpec, parse

from qscute.render import rust_source
from qscute.signature import signature


def load_kernel(source: Path) -> Callable[..., object]:
    source = source.resolve(strict=True)
    name = "_qscute_source_" + hashlib.sha256(str(source).encode()).hexdigest()
    loader = importlib.util.spec_from_file_location(name, source)
    if loader is None or loader.loader is None:
        raise ValueError(f"{source}: expected a Python source file")
    module = importlib.util.module_from_spec(loader)
    sys.modules[name] = module
    try:
        loader.loader.exec_module(module)
        kernel = getattr(module, "kernel", None)
        if (
            not callable(kernel)
            or not hasattr(kernel, "__wrapped__")
            or getattr(kernel, "_dsl_cls", None) is not CuTeDSL
        ):
            raise ValueError(f"{source}: expected @cute.jit entrypoint 'kernel'")
        return kernel
    finally:
        del sys.modules[name]


def compile_options(spec: CuteSpec, target: CudaTarget) -> str:
    if target.backend != "cuda" or target.warpSize != 32:
        raise ValueError("qscute requires a CUDA target with 32-thread warps")
    cap = target.computeCapability
    arch = f"sm_{cap.major}{cap.minor}"
    options = dict(spec.options)
    selected = options.pop("gpu-arch", arch)
    if selected not in (arch, arch + "a", arch + "f"):
        raise ValueError("gpu-arch must match the configured compute capability")
    supported = {
        "opt-level",
        "enable-assertions",
        "generate-line-info",
        "ptxas-options",
        "host-target",
    }
    if options.keys() - supported:
        raise ValueError(
            f"unsupported CuTe compiler options: {sorted(options.keys() - supported)}"
        )
    words = ["--gpu-arch", str(selected)]
    for key, value in sorted(options.items()):
        if value is True:
            words.append("--" + key)
        elif value is not False:
            words.extend(["--" + key, str(value)])
    return shlex.join(words)


def source_dependencies(source: Path) -> list[Path]:
    roots = [
        Path(__file__).parent,
        Path(qsutil.__file__).parent,
        source.resolve().parent,
    ]
    return sorted({p.resolve() for root in roots for p in root.rglob("*.py")})


def compile_source(
    source: Path, spec: CuteSpec, target: CudaTarget, prefix: str
) -> None:
    path = Path(prefix)
    if re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", path.name) is None:
        raise ValueError("specialization name must be a C/Rust identifier")
    symbol = "qscute_" + path.name
    options = compile_options(spec, target)
    kernel = load_kernel(source)
    values, arguments = signature(kernel, spec)
    dependencies = source_dependencies(source)
    for name in [prefix, *(relpath(p) for p in dependencies)]:
        if any(char in name for char in "$# :\\\t\r\n"):
            raise ValueError(f"unsupported depfile path: {name}")
    compiled = cute.compile(kernel, *values, options=options)
    # Use the same symbol construction as CuTe's exporter, without generating
    # or parsing its C header. The pinned compiler owns the packed MLIR ABI.
    entrypoint = f"_mlir_{symbol}__mlir_ciface_{compiled.function_name}"
    object_bytes = compiled.dump_to_object(symbol)
    if not object_bytes.startswith(b"\x7fELF"):
        raise ValueError("CuTe did not export an ELF object")
    artifacts = {
        ".o": object_bytes,
        ".rs": rust_source(symbol, entrypoint, arguments, spec.constants).encode(),
    }
    manifest = {
        "specialization": path.name,
        "symbol": symbol,
        "entrypoint": entrypoint,
        "cutlass_version": importlib.metadata.version("nvidia-cutlass-dsl"),
        "target": asdict(target),
        "spec": asdict(spec),
        "compile_options": options,
        "arguments": [asdict(a) for a in arguments],
        "sources": {
            str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in dependencies
        },
        "artifacts": {
            path.name + suffix: hashlib.sha256(data).hexdigest()
            for suffix, data in artifacts.items()
        },
        "link": {
            "static_runtime": "libcuda_dialect_runtime_static.a",
            "libraries": ["cuda", "cudart", "stdc++"],
            "note": "Link the runtime archive built for the exported object's host architecture.",
        },
    }
    artifacts[".json"] = (json.dumps(manifest, indent=2) + "\n").encode()
    artifacts[".d"] = (
        prefix + ".rs: " + " ".join(relpath(p) for p in dependencies) + "\n"
    ).encode()
    path.parent.mkdir(parents=True, exist_ok=True)
    for suffix, data in artifacts.items():
        Path(prefix + suffix).write_bytes(data)
    print(f"{prefix}: CuTe AOT object and direct Rust launcher")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--spec", required=True)
    parser.add_argument("--target", required=True)
    parser.add_argument("--prefix", required=True)
    args = parser.parse_args()
    compile_source(
        args.source,
        parse(args.spec, CuteSpec),
        parse(args.target, CudaTarget),
        args.prefix,
    )
