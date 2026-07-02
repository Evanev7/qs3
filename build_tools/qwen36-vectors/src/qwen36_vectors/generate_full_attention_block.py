from __future__ import annotations

import argparse
from collections.abc import Sequence
from pathlib import Path

from .full_attention_block import (
    DEFAULT_OUTPUT,
    POSITIONS,
    ROWS,
    build_full_attention_block_artifact,
    write_full_attention_block_artifact,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate deterministic Qwen3.6 full-attention block vectors."
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
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    manifest, _ = build_full_attention_block_artifact()
    output = args.output
    if not args.check:
        write_full_attention_block_artifact(output)

    print(
        "generated "
        f"{manifest.name}: rows={ROWS} positions={list(POSITIONS)} "
        f"tensors={len(manifest.tensors)} output={output}"
    )


if __name__ == "__main__":
    main()
