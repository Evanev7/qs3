"""Validate the compiler inputs selected by Nix."""

import json
import shutil
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


@pytest.mark.parametrize(
    "tiles, schedulers, expected",
    [
        ([64], [True], ["TILE128X64_STREAM_K"]),
        ([32], [False, True], ["TILE128X32_DP", "TILE128X32_STREAM_K"]),
        ([32, 64], [False], ["TILE128X32_DP", "TILE128X64_DP"]),
        (
            [32, 64],
            [False, True],
            [
                "TILE128X32_DP",
                "TILE128X32_STREAM_K",
                "TILE128X64_DP",
                "TILE128X64_STREAM_K",
            ],
        ),
        (["32"], [False], None),
        ([32], ["false"], None),
    ],
)
def test_nvfp4_tactic_selection(
    tmp_path: Path,
    tiles: list[int | str],
    schedulers: list[bool | str],
    expected: list[str] | None,
) -> None:
    root = Path(__file__).resolve().parents[2]
    # Evaluate the real generator against an isolated configuration override.
    # This exercises selection without editing the checkout's build inputs.
    generators = tmp_path / "build_tools/nixsrc"
    shutil.copytree(root / "build_tools/nixsrc", generators)
    models = tmp_path / "models"
    models.mkdir()
    (models / "config.nix").write_text(
        f"let base = import {root / 'models/config.nix'}; in base // {{ "
        "kernels = base.kernels // { nvfp4 = base.kernels.nvfp4 // { "
        f"tileN = builtins.fromJSON {json.dumps(json.dumps(tiles))}; "
        f"streamK = builtins.fromJSON {json.dumps(json.dumps(schedulers))}; "
        "}; }; }"
    )
    result = subprocess.run(
        ["nix", "eval", "--offline", "--raw", "--file", str(generators / "c.nix")],
        check=False,
        text=True,
        capture_output=True,
    )
    if expected is None:
        assert result.returncode != 0
        assert "assertion" in result.stderr
        return
    assert result.returncode == 0, result.stderr
    rows = [line.strip().removesuffix(" \\") for line in result.stdout.splitlines()]
    enabled = [
        line.split(",", 1)[0].removeprefix("X(QSFI_NVFP4_")
        for line in rows
        if line.startswith("X(QSFI_NVFP4_")
    ]
    assert enabled == expected
    # Scheduler choices do not duplicate tile instantiations.
    for tile in tiles:
        assert rows.count(f"X({tile})") == 1
    assert all(f"X({other})" not in rows for other in {32, 64} - set(tiles))
