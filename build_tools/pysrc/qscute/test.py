"""Link CuTe AOT exports and exercise the generated Rust launchers on CUDA."""

import json
import os
import subprocess
import sys
from pathlib import Path

from qsutil.config import CudaTarget, CuteSpec, parse

from qscute.builder import compile_source
from qscute.runtime import runtime_archive


def main() -> None:
    config_path = Path(sys.argv[1]).resolve()
    config = json.loads(config_path.read_text())
    target = parse(json.dumps(config["target"]), CudaTarget)
    output = config_path.parent
    source = Path(__file__).resolve().parent
    root = source.parents[2]
    prefix = output / "fp8_decode_test"
    spec = parse(json.dumps(config["kernels"]["fp8_decode"]["spec"]), CuteSpec)
    compile_source(root / "cute_kernels/fp8_decode.py", spec, target, str(prefix))
    archive = output / "libqscute_test.a"
    archive.unlink(missing_ok=True)
    subprocess.run(["ar", "rcs", str(archive), str(prefix) + ".o"], check=True)
    runtime = runtime_archive()
    subprocess.run(["just", "--justfile", str(root / "justfile"), "build"], check=True)
    subprocess.run(
        [
            "rustc",
            "--edition=2024",
            "-O",
            "--test",
            str(source / "test.rs"),
            "--extern",
            f"qs3={root / 'target/debug/libqs3.rlib'}",
            "-L",
            f"dependency={root / 'target/debug/deps'}",
            "-L",
            f"native={root / 'build'}",
            "-L",
            f"native={output}",
            "-L",
            f"native={runtime.parent}",
            "-l",
            "static=qscute_test",
            "-l",
            "static=cuda_dialect_runtime_static",
            "-l",
            "cuda",
            "-l",
            "cudart",
            "-l",
            "stdc++",
            "-o",
            str(output / "test"),
        ],
        env=dict(os.environ, QS3_CUTE_OUTPUT=str(output)),
        check=True,
    )
    subprocess.run(
        [str(output / "test"), "--nocapture", "--test-threads=1"], check=True
    )


if __name__ == "__main__":
    main()
