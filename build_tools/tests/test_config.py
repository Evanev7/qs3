"""Validate the compiler inputs selected by Nix."""

import json
import subprocess
from dataclasses import asdict
from pathlib import Path

import pytest
from qsutil.config import CudaTarget, DtypeConstant, TritonSpec, parse


@pytest.fixture(scope="module")
def config() -> dict:
    path = Path(__file__).resolve().parents[2] / "models/config.nix"
    return json.loads(
        subprocess.check_output(
            ["nix", "eval", "--json", "--file", str(path)], text=True
        )
    )


def test_current_nix_inputs(config: dict) -> None:
    target = parse(json.dumps(config["target"]), CudaTarget)
    assert asdict(target) == config["target"]
    entry = config["kernels"]["lm_head"]
    spec = parse(json.dumps(entry["spec"]), TritonSpec)
    assert spec.constants["ACC"] == DtypeConstant("f32")
    assert spec.grid == [248320, 1, 1]
    assert asdict(spec) == entry["spec"]
