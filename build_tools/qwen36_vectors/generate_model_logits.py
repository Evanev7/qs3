from __future__ import annotations

import argparse
from collections.abc import Sequence
from pathlib import Path

from .model_logits import (
    DEFAULT_OUTPUT,
    PAGE_SIZE,
    PROMPT_TOKENS,
    TOTAL_ROWS,
    VOCAB_SIZE,
    build_model_logits_artifact,
    write_model_logits_artifact,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate deterministic Qwen3.6 ModelRunner logits vectors."
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
    manifest, metadata = build_model_logits_artifact()
    output = args.output
    if not args.check:
        write_model_logits_artifact(output)

    tokens = metadata["tokens"]
    print(
        "generated "
        f"{manifest.name}: prompt={list(PROMPT_TOKENS)} "
        f"decode_input={tokens['decode_input']} rows={TOTAL_ROWS} "
        f"vocab={VOCAB_SIZE} page_size={PAGE_SIZE} "
        f"tensors={len(manifest.tensors)} output={output}"
    )


if __name__ == "__main__":
    main()
