import json
import shlex
import subprocess
from pathlib import Path

import pytest
from qstriton.builder import (
    compile_source,
    load_kernel,
    rust_type,
    signature,
    source_dependencies,
)
from qsutil.config import CudaTarget, TritonSpec, parse

ROOT = Path(__file__).resolve().parents[2]


def evaluate(path: str, flag: str) -> str:
    return subprocess.check_output(
        ["nix", "eval", flag, "--file", str(ROOT / path)], text=True
    )


def test_same_source_distinct_outputs() -> None:
    config = json.loads(evaluate("models/config.nix", "--json"))
    lm_head = config["kernels"]["lm_head"]
    qkv = config["kernels"]["gdn_qkv"]
    assert lm_head["source"] == qkv["source"] == "triton_kernels/gemv.py"
    kernel = load_kernel(ROOT / lm_head["source"])
    lm_types, lm_constants, _ = signature(
        kernel, parse(json.dumps(lm_head["spec"]), TritonSpec)
    )
    qkv_types, qkv_constants, _ = signature(
        kernel, parse(json.dumps(qkv["spec"]), TritonSpec)
    )
    assert lm_types["output"] == "*fp32"
    assert qkv_types["output"] == "*bf16"
    assert lm_constants == qkv_constants
    dependencies = source_dependencies(ROOT / lm_head["source"])
    assert (ROOT / lm_head["source"]) in dependencies
    assert (ROOT / "build_tools/pysrc/qsutil/config.py") in dependencies


def test_ninja_source_and_json_arguments(tmp_path: Path) -> None:
    graph = tmp_path / "kernels.ninja"
    graph.write_text(
        "qstriton = /compiler/qstriton\n"
        + evaluate("build_tools/nixsrc/ninja.nix", "--raw")
    )
    commands = subprocess.check_output(
        ["ninja", "-f", str(graph), "-t", "commands", "triton/kernels"], text=True
    ).splitlines()
    config = json.loads(evaluate("models/config.nix", "--json"))
    selected = {
        name: entry
        for name, entry in config["kernels"].items()
        if entry["provider"] == "triton"
    }
    assert len(commands) == len(selected)
    outputs = set()
    for command in commands:
        args = shlex.split(command)
        assert "--kernel" not in args
        name = Path(args[args.index("--prefix") + 1]).name
        assert args[args.index("--source") + 1] == "../" + selected[name]["source"]
        parse(args[args.index("--spec") + 1], TritonSpec)
        parse(args[args.index("--target") + 1], CudaTarget)
        outputs.add(args[args.index("--prefix") + 1])
    assert outputs == {"triton/" + name for name in selected}


def test_sampler_integer_pointer_and_scalar_abi() -> None:
    config = json.loads(evaluate("models/config.nix", "--json"))
    entry = config["kernels"]["sampling_gumbel"]
    types, _, args = signature(
        load_kernel(ROOT / entry["source"]),
        parse(json.dumps(entry["spec"]), TritonSpec),
    )
    assert types["position"] == "*i32"
    assert types["seed"] == "u64"
    assert {a.name: a.rust for a in args}["seed"] == "u64"
    assert rust_type("*i32") == "*mut i32"
    assert rust_type("*ku32") == "*const u32"


@pytest.mark.parametrize("text", ["", "kernel = 1", "def kernel(): pass"])
def test_invalid_entrypoint_fails_before_artifacts(tmp_path: Path, text: str) -> None:
    source = tmp_path / "invalid.py"
    source.write_text(text)
    config = json.loads(evaluate("models/config.nix", "--json"))
    output = tmp_path / "output"
    with pytest.raises(ValueError, match="expected @triton.jit entrypoint 'kernel'"):
        compile_source(
            source,
            parse(json.dumps(config["kernels"]["lm_head"]["spec"]), TritonSpec),
            parse(json.dumps(config["target"]), CudaTarget),
            str(output / "invalid"),
        )
    assert not output.exists()
