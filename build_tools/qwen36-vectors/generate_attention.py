from __future__ import annotations

import argparse
from collections.abc import Sequence
from pathlib import Path

from .attention import DEFAULT_OUTPUT, NUM_TOKENS, POSITIONS, build_attention_artifact, write_bundle


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate deterministic Qwen3.6 full-attention primitive vectors."
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
    parser.add_argument(
        "--force",
        action="store_true",
        help="overwrite existing generated attention artifacts in the output directory",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    manifest, _ = build_attention_artifact()
    if args.check:
        output = args.output
    else:
        output = args.output
        write_bundle(output, force=args.force)

    print(
        "generated "
        f"{manifest.name}: tokens={NUM_TOKENS} positions={list(POSITIONS)} "
        f"tensors={len(manifest.tensors)} output={output}"
    )


if __name__ == "__main__":
    main()
