from __future__ import annotations

import argparse
from collections.abc import Sequence
from pathlib import Path

from .norms import NormIndex, generate_norm_artifacts
from .paths import DEFAULT_VECTOR_ROOT

DEFAULT_NORMS_OUT = DEFAULT_VECTOR_ROOT / "norms"


def generate_norms(out_dir: str | Path = DEFAULT_NORMS_OUT) -> NormIndex:
    return generate_norm_artifacts(out_dir)


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Generate Qwen3.6 Gemma RMSNorm semantic vectors."
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_NORMS_OUT,
        help=f"artifact output directory (default: {DEFAULT_NORMS_OUT})",
    )
    args = parser.parse_args(argv)
    index = generate_norms(args.output)
    print(f"generated norms: cases={len(index['cases'])} output={args.output}")


if __name__ == "__main__":
    main()
