from __future__ import annotations

import argparse
from collections.abc import Sequence
from pathlib import Path

from .gdn import DEFAULT_OUTPUT, SEQ_LENS, build_gdn_artifact, write_gdn_artifact


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Generate Qwen3.6 GDN post-conv/prep semantic vectors."
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

    manifest, _ = build_gdn_artifact()
    if args.check:
        output = args.output
    else:
        output = write_gdn_artifact(args.output).parent

    print(
        "generated "
        f"{manifest.name}: cases={list(SEQ_LENS)} tensors={len(manifest.tensors)} "
        f"output={output}"
    )


if __name__ == "__main__":
    main()
