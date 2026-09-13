"""Build full-width and masked-tail kernels, then exercise their Rust launchers."""

import copy
import json
import os
import subprocess
import sys
from pathlib import Path

from qsutil.config import CudaTarget, TritonSpec, parse

from qstriton.builder import compile_source


def main() -> None:
    config_path = Path(sys.argv[1]).resolve()
    config = json.loads(config_path.read_text())
    target = parse(json.dumps(config["target"]), CudaTarget)
    output = config_path.parent
    source = Path(__file__).resolve().parent
    for name in ("full", "tail"):
        selected = copy.deepcopy(config["kernels"])
        if name == "tail":
            spec = selected["lm_head"]["spec"]
            spec["constants"].update(K=93, BLOCK_K=128)
            spec["precision"]["output"] = "bf16"
            spec["grid"] = [37, 1, 1]
        for specialization, entry in selected.items():
            if entry["provider"] == "triton":
                compile_source(
                    source.parents[2] / entry["source"],
                    parse(json.dumps(entry["spec"]), TritonSpec),
                    target,
                    str(output / name / specialization),
                )
    env = dict(
        os.environ,
        QS3_TRITON_OUTPUT=str(output),
        QS3_TRITON_K=str(config["model"]["config"]["hidden_size"]),
    )
    subprocess.run(
        [
            "rustc",
            "--edition=2024",
            "-O",
            "--test",
            str(source / "test.rs"),
            "-o",
            str(output / "test"),
        ],
        env=env,
        check=True,
    )
    subprocess.run(
        [str(output / "test"), "--nocapture", "--test-threads=1"], check=True
    )


if __name__ == "__main__":
    main()
