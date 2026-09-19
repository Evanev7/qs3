"""Link CuTe AOT exports and exercise the generated Rust launchers on CUDA."""

import importlib.metadata
import json
import os
import subprocess
import sys
from pathlib import Path

from qsutil.config import CudaTarget, CuteSpec, parse

from qscute.builder import compile_source


def main() -> None:
    config_path = Path(sys.argv[1]).resolve()
    config = json.loads(config_path.read_text())
    target = parse(json.dumps(config["target"]), CudaTarget)
    output = config_path.parent
    source = Path(__file__).resolve().parent
    root = source.parents[2]
    objects = []
    for dtype, block in [("f32", 128), ("bf16", 64)]:
        prefix = output / ("saxpy_" + dtype)
        spec = CuteSpec(
            precision={name: dtype for name in ("x", "y", "output")},
            alignments={name: 16 for name in ("x", "y", "output")},
            constants={"BLOCK": block},
            options={"gpu-arch": "sm_121a"},
        )
        compile_source(root / "cute_kernels/saxpy.py", spec, target, str(prefix))
        objects.append(str(prefix) + ".o")
    archive = output / "libqscute_test.a"
    archive.unlink(missing_ok=True)
    subprocess.run(["ar", "rcs", str(archive), *objects], check=True)
    package = importlib.metadata.distribution("nvidia-cutlass-dsl-libs-cu13")
    runtime = Path(
        str(
            package.locate_file(
                "nvidia_cutlass_dsl/cu13/lib/libcuda_dialect_runtime_static.a"
            )
        )
    ).resolve(strict=True)
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
