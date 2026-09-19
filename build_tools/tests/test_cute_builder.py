"""GPU-free qualification of real CuTe AOT exports and their launch contracts."""

import copy
import hashlib
import json
import re
import shlex
import shutil
import subprocess
from dataclasses import asdict
from pathlib import Path

import pytest
from qscute.builder import compile_options, compile_source, load_kernel
from qscute.signature import TYPES, Dtype, signature
from qsutil.config import CudaTarget, CuteSpec, parse
from test_triton_builder import ROOT, evaluate


@pytest.fixture
def spec() -> CuteSpec:
    return CuteSpec(
        precision={"x": "f32", "y": "f32", "output": "f32"},
        alignments={"x": 16, "y": 16, "output": 16},
        constants={"BLOCK": 128},
        options={"gpu-arch": "sm_121a", "host-target": "linux-aarch64"},
    )


@pytest.fixture
def target() -> CudaTarget:
    return parse(
        json.dumps(json.loads(evaluate("models/config.nix", "--json"))["target"]),
        CudaTarget,
    )


def test_signature(spec: CuteSpec) -> None:
    values, args = signature(load_kernel(ROOT / "cute_kernels/saxpy.py"), spec)
    assert len(values) == 7
    assert [(a.name, a.rust) for a in args] == [
        ("x", "DevicePtr<F32>"),
        ("y", "DevicePtr<F32>"),
        ("output", "DevicePtr<F32>"),
        ("n", "i32"),
        ("alpha", "f32"),
        ("stream", "*mut c_void"),
    ]
    assert all(isinstance(ty, Dtype) for ty in TYPES.values())
    assert TYPES["fp8_e4m3"].marker == "Fp8E4M3"


@pytest.mark.parametrize("dtype,block", [("f32", 128), ("bf16", 64)])
def test_real_aot_export(
    tmp_path: Path, spec: CuteSpec, target: CudaTarget, dtype: str, block: int
) -> None:
    spec.precision = {name: dtype for name in spec.precision}
    spec.constants["BLOCK"] = block
    prefix = tmp_path / "saxpy"
    compile_source(ROOT / "cute_kernels/saxpy.py", spec, target, str(prefix))
    obj = prefix.with_suffix(".o").read_bytes()
    assert obj[:4] == b"\x7fELF"
    assert int.from_bytes(obj[18:20], "little") == 183  # EM_AARCH64, no GPU required.
    assert not prefix.with_suffix(".h").exists()
    assert not prefix.with_suffix(".cu").exists()
    manifest = json.loads(prefix.with_suffix(".json").read_text())
    symbols = subprocess.check_output(["nm", str(prefix.with_suffix(".o"))], text=True)
    assert manifest["entrypoint"] in symbols
    assert (
        f'link_name = "{manifest["entrypoint"]}"'
        in prefix.with_suffix(".rs").read_text()
    )
    assert [a["name"] for a in manifest["arguments"]] == [
        "x",
        "y",
        "output",
        "n",
        "alpha",
        "stream",
    ]
    assert manifest["spec"]["constants"] == {"BLOCK": block}
    assert manifest["artifacts"]["saxpy.o"] == hashlib.sha256(obj).hexdigest()
    depfile = prefix.with_suffix(".d").read_text()
    assert "cute_kernels/saxpy.py" in depfile and "qscute/signature.py" in depfile
    # Type-check the emitted Rust with the real storage marker and DevicePtr sources.
    raw_dtypes = re.findall(
        r"QSFI_(DTYPE_\w+) = (\d+)", (ROOT / "qs_tensor.h").read_text()
    )
    constants = "\n".join(
        f"pub const {name}: DTypeRaw = {value};" for name, value in raw_dtypes
    )
    harness = tmp_path / "check.rs"
    harness.write_text(f'''
#![allow(dead_code)]
pub enum Status {{ InvalidArgument }}
#[path = "{ROOT / "src/dtype.rs"}"] mod dtype;
mod ffi {{
    pub type DTypeRaw = u32;
    {constants}
    pub mod sys {{ pub const QSFI_DTYPE_INVALID: u32 = 0; }}
    #[path = "{ROOT / "src/ffi/device_ptr.rs"}"] mod pointer;
    pub use pointer::DevicePtr;
}}
include!("saxpy.rs");
''')
    subprocess.run(
        [
            "rustc",
            "--edition=2024",
            "--crate-type=lib",
            str(harness),
            "--emit=metadata",
            "-o",
            str(tmp_path / "check.rmeta"),
        ],
        check=True,
    )


@pytest.mark.parametrize("text", ["kernel = 1", "def kernel(): pass"])
def test_invalid_source_has_no_artifacts(
    tmp_path: Path, spec: CuteSpec, target: CudaTarget, text: str
) -> None:
    source = tmp_path / "invalid.py"
    source.write_text(text)
    prefix = tmp_path / "out/kernel"
    with pytest.raises(ValueError, match="expected @cute.jit"):
        compile_source(source, spec, target, str(prefix))
    assert not prefix.parent.exists()


def test_invalid_contract(spec: CuteSpec, target: CudaTarget) -> None:
    kernel = load_kernel(ROOT / "cute_kernels/saxpy.py")
    invalid = copy.deepcopy(spec)
    invalid.alignments["x"] = 3
    with pytest.raises(ValueError, match="power of two"):
        signature(kernel, invalid)
    invalid = copy.deepcopy(spec)
    invalid.constants["NOT_A_PARAMETER"] = 1
    with pytest.raises(ValueError, match="exactly match"):
        signature(kernel, invalid)
    invalid = copy.deepcopy(spec)
    invalid.options["gpu-arch"] = "sm_100a"
    with pytest.raises(ValueError, match="compute capability"):
        compile_options(invalid, target)
    invalid = copy.deepcopy(spec)
    invalid.options["enable-tvm-ffi"] = True
    with pytest.raises(ValueError, match="unsupported CuTe compiler options"):
        compile_options(invalid, target)


def test_ninja_recipe(tmp_path: Path, spec: CuteSpec) -> None:
    generators = tmp_path / "build_tools/nixsrc"
    shutil.copytree(ROOT / "build_tools/nixsrc", generators)
    models = tmp_path / "models"
    models.mkdir()

    entry = json.dumps(
        {"provider": "cute", "source": "cute_kernels/saxpy.py", "spec": asdict(spec)}
    )
    (models / "config.nix").write_text(
        f"let base = import {ROOT / 'models/config.nix'}; in base // {{ kernels = {{ saxpy = builtins.fromJSON {json.dumps(entry)}; }}; }}"
    )
    graph = tmp_path / "cute.ninja"
    graph.write_text(
        "qscute = /compiler/qscute\n"
        + subprocess.check_output(
            [
                "nix",
                "eval",
                "--offline",
                "--raw",
                "--file",
                str(generators / "cute.nix"),
            ],
            text=True,
        )
    )
    command = subprocess.check_output(
        ["ninja", "-f", str(graph), "-t", "commands", "cute/kernels"], text=True
    )
    args = shlex.split(command)
    assert args[0] == "/compiler/qscute"
    assert args[args.index("--source") + 1] == "../cute_kernels/saxpy.py"
    assert parse(args[args.index("--spec") + 1], CuteSpec) == spec
