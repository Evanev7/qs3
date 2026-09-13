from __future__ import annotations

import argparse
from collections.abc import Sequence
from pathlib import Path

from .gdn_decoder_layer import (
    DEFAULT_OUTPUT,
    ROWS,
    build_gdn_decoder_layer_artifact,
    write_gdn_decoder_layer_artifact,
)


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Generate Qwen3.6 GDN decoder-layer semantic vectors."
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

    manifest, _ = build_gdn_decoder_layer_artifact()
    if args.check:
        output = args.output
    else:
        output = write_gdn_decoder_layer_artifact(args.output).parent

    print(
        f"generated {manifest.name}: rows={ROWS} tensors={len(manifest.tensors)} output={output}"
    )


if __name__ == "__main__":
    main()
