from __future__ import annotations

import argparse
from collections.abc import Sequence
from pathlib import Path

from .moe import (
    CASES,
    DEFAULT_OUTPUT,
    NUM_EXPERTS,
    TOP_K,
    build_moe_artifact,
    write_moe_artifact,
)


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Generate Qwen3.6 MoE/shared-expert semantic vectors."
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help=f"artifact directory to write (default: {DEFAULT_OUTPUT})",
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="build the artifact in memory and print the summary without writing files",
    )
    args = parser.parse_args(argv)

    manifest, _ = build_moe_artifact()
    if args.check:
        output = args.output
    else:
        output = write_moe_artifact(args.output).parent

    print(
        "generated "
        f"{manifest.name}: cases={len(CASES)} experts={NUM_EXPERTS} top_k={TOP_K} "
        f"tensors={len(manifest.tensors)} output={output}"
    )


if __name__ == "__main__":
    main()
