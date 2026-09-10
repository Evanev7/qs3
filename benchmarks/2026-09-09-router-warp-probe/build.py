import subprocess
from pathlib import Path

root = Path.home() / "qs3"
base = [
    "/usr/local/cuda/bin/nvcc",
    "-std=c++17",
    "-arch=sm_121",
    "--expt-relaxed-constexpr",
    "-DQSFI_ENABLE_CHECKED_VALIDATION=0",
    "-I..",
    "-I.",
]
subprocess.run(
    base
    + [
        "-c",
        "../.prototypes/router_parallel/qscu.cu",
        "-o",
        "../.prototypes/router_parallel/qscu.o",
    ],
    cwd=root / "build",
    check=True,
)
for label, obj in [
    ("serial", "qscu.o"),
    ("warp", "../.prototypes/router_parallel/qscu.o"),
]:
    subprocess.run(
        base
        + [
            "../.prototypes/router_parallel/bench.cu",
            "qsfi.o",
            "qscu_gdn.o",
            obj,
            "qscb.o",
            "-o",
            "../.prototypes/router_parallel/" + label,
            "-lcuda",
            "-lcublas",
            "-lcublasLt",
        ],
        cwd=root / "build",
        check=True,
    )
