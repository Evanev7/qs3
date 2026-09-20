"""Stage the pinned wheel's native runtime archive for the final Rust link."""

import argparse
import importlib.metadata
import shutil
from pathlib import Path


def runtime_archive() -> Path:
    package = importlib.metadata.distribution("nvidia-cutlass-dsl-libs-cu13")
    return Path(
        str(
            package.locate_file(
                "nvidia_cutlass_dsl/cu13/lib/libcuda_dialect_runtime_static.a"
            )
        )
    ).resolve(strict=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(runtime_archive(), args.output)
