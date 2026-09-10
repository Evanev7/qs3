"""Build full-width and masked-tail kernels, then exercise their Rust launchers."""

import copy
import json
import os
import subprocess
import sys
from pathlib import Path

from qstriton.builder import BuildConfig, build


def main() -> None:
    config_path = Path(sys.argv[1]).resolve()
    config: BuildConfig = json.loads(config_path.read_text())
    output = config_path.parent
    source = Path(__file__).resolve().parent
    for name in ("full", "tail"):
        selected = copy.deepcopy(config)
        if name == "tail":
            kernel = selected["kernels"]["lm_head"]
            kernel["constants"].update(K=93, BLOCK_K=128)
            kernel["precision"]["output"] = "bf16"
            kernel["grid"] = [37, 1, 1]
        build(source / "kernels/lm_head.py", selected, output / name)
    env = dict(
        os.environ,
        QS3_TRITON_OUTPUT=str(output),
        QS3_TRITON_K=str(config["kernels"]["lm_head"]["constants"]["K"]),
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
