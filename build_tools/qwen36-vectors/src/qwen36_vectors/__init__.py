from __future__ import annotations

import argparse
import os
from collections.abc import Sequence
from pathlib import Path

from . import generate_attention
from .full_attention_block import DEFAULT_OUTPUT as DEFAULT_FULL_ATTENTION_BLOCK_OUTPUT
from .gdn import DEFAULT_OUTPUT as DEFAULT_GDN_OUTPUT
from .gdn_decoder_layer import DEFAULT_OUTPUT as DEFAULT_GDN_DECODER_LAYER_OUTPUT
from .model_logits import DEFAULT_OUTPUT as DEFAULT_MODEL_LOGITS_OUTPUT
from .generate_norms import DEFAULT_NORMS_OUT, generate_norms
from .io import (
    bf16_bits_to_float32,
    bf16_words_to_float32,
    float32_to_bf16_bits,
    float32_to_bf16_words,
    read_artifact,
    read_manifest,
    read_tensor,
    write_artifact,
    write_manifest,
    write_tensor,
)
from .oracle import check_oracles, generate_all, write_oracles
from .paths import DEFAULT_ORACLE_ROOT, DEFAULT_VECTOR_ROOT, VECTOR_GROUPS
from .schema import (
    BYTE_ORDER,
    DTYPE_BYTE_WIDTHS,
    DTYPE_EXTENSIONS,
    MANIFEST_FILE,
    SCHEMA_VERSION,
    TensorSpec,
    VectorManifest,
    default_tensor_file,
)

__all__ = [
    "BYTE_ORDER",
    "DEFAULT_ORACLE_ROOT",
    "DEFAULT_VECTOR_ROOT",
    "DTYPE_BYTE_WIDTHS",
    "DTYPE_EXTENSIONS",
    "MANIFEST_FILE",
    "SCHEMA_VERSION",
    "TensorSpec",
    "VECTOR_GROUPS",
    "VectorManifest",
    "bf16_bits_to_float32",
    "bf16_words_to_float32",
    "check_oracles",
    "default_tensor_file",
    "float32_to_bf16_bits",
    "float32_to_bf16_words",
    "generate_attention",
    "generate_all",
    "generate_norms",
    "main",
    "read_artifact",
    "read_manifest",
    "read_tensor",
    "write_artifact",
    "write_manifest",
    "write_oracles",
    "write_tensor",
]


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        prog="qwen36-vectors",
        description="Qwen3.6 correctness vector generator scaffolding.",
    )
    subparsers = parser.add_subparsers(dest="command")
    subparsers.add_parser("list-groups", help="list implemented vector groups")
    generate_all_parser = subparsers.add_parser(
        "generate-all",
        help="generate every vector group under one root",
    )
    generate_all_parser.add_argument(
        "--output-root",
        default=str(DEFAULT_VECTOR_ROOT),
        help=f"vector output root (default: {DEFAULT_VECTOR_ROOT})",
    )
    generate_all_parser.add_argument(
        "--clean",
        action="store_true",
        help="remove existing generated group directories before writing",
    )
    write_oracles_parser = subparsers.add_parser(
        "write-oracles",
        help="write byte-level oracle hashes for generated vector groups",
    )
    write_oracles_parser.add_argument(
        "--input-root",
        default=str(DEFAULT_VECTOR_ROOT),
        help=f"generated vector root (default: {DEFAULT_VECTOR_ROOT})",
    )
    write_oracles_parser.add_argument(
        "--oracle-root",
        default=str(DEFAULT_ORACLE_ROOT),
        help=f"oracle output root (default: {DEFAULT_ORACLE_ROOT})",
    )
    check_oracles_parser = subparsers.add_parser(
        "check-oracles",
        help="validate generated vector groups against committed oracle hashes",
    )
    check_oracles_parser.add_argument(
        "--input-root",
        default=str(DEFAULT_VECTOR_ROOT),
        help=f"generated vector root (default: {DEFAULT_VECTOR_ROOT})",
    )
    check_oracles_parser.add_argument(
        "--oracle-root",
        default=str(DEFAULT_ORACLE_ROOT),
        help=f"oracle input root (default: {DEFAULT_ORACLE_ROOT})",
    )
    check_oracles_parser.add_argument(
        "--depfile",
        help="write Make-style source/oracle dependencies after successful verification",
    )
    check_oracles_parser.add_argument(
        "--depfile-target",
        help="build output named by the depfile (for example, the .oracle-ok stamp)",
    )
    generate = subparsers.add_parser(
        "generate",
        help="generate one vector group",
    )
    generate.add_argument("group", help="vector group to generate")
    generate.add_argument("output", help="artifact output directory")
    generate_moe = subparsers.add_parser(
        "generate-moe",
        help="generate MoE/shared-expert vectors",
    )
    generate_moe.add_argument("output", nargs="?", help="artifact output directory")
    generate_moe.add_argument("--output", dest="output_flag", help="artifact output directory")
    generate_norms_parser = subparsers.add_parser(
        "generate-norms",
        help="generate Gemma RMSNorm semantics vectors",
    )
    generate_norms_parser.add_argument(
        "--output",
        default=str(DEFAULT_NORMS_OUT),
        help=f"artifact output directory (default: {DEFAULT_NORMS_OUT})",
    )
    generate_moe.add_argument(
        "--check",
        action="store_true",
        help="build the artifact in memory without writing files",
    )
    generate_gdn_parser = subparsers.add_parser(
        "generate-gdn",
        help="generate GDN post-conv/prep vectors",
    )
    generate_gdn_parser.add_argument(
        "--output",
        default=str(DEFAULT_GDN_OUTPUT),
        help=f"artifact output directory (default: {DEFAULT_GDN_OUTPUT})",
    )
    generate_gdn_parser.add_argument(
        "--check",
        action="store_true",
        help="build the artifact in memory without writing files",
    )
    generate_gdn_decoder_layer_parser = subparsers.add_parser(
        "generate-gdn-decoder-layer",
        help="generate GDN decoder-layer vectors",
    )
    generate_gdn_decoder_layer_parser.add_argument(
        "--output",
        default=str(DEFAULT_GDN_DECODER_LAYER_OUTPUT),
        help=(
            "artifact output directory "
            f"(default: {DEFAULT_GDN_DECODER_LAYER_OUTPUT})"
        ),
    )
    generate_gdn_decoder_layer_parser.add_argument(
        "--check",
        action="store_true",
        help="build the artifact in memory without writing files",
    )
    generate_attention_parser = subparsers.add_parser(
        "generate-attention",
        help="generate full-attention primitive vectors",
    )
    generate_attention_parser.add_argument(
        "--output",
        default=str(generate_attention.DEFAULT_OUTPUT),
        help=(
            "artifact output directory "
            f"(default: {generate_attention.DEFAULT_OUTPUT})"
        ),
    )
    generate_attention_parser.add_argument(
        "--check",
        action="store_true",
        help="build the artifact in memory without writing files",
    )
    generate_attention_parser.add_argument(
        "--force",
        action="store_true",
        help="overwrite existing generated attention artifacts",
    )
    generate_full_attention_block_parser = subparsers.add_parser(
        "generate-full-attention-block",
        help="generate full-attention block vectors",
    )
    generate_full_attention_block_parser.add_argument(
        "--output",
        default=str(DEFAULT_FULL_ATTENTION_BLOCK_OUTPUT),
        help=(
            "artifact output directory "
            f"(default: {DEFAULT_FULL_ATTENTION_BLOCK_OUTPUT})"
        ),
    )
    generate_full_attention_block_parser.add_argument(
        "--check",
        action="store_true",
        help="build the artifact in memory without writing files",
    )
    generate_model_logits_parser = subparsers.add_parser(
        "generate-model-logits",
        help="generate ModelRunner logits vectors",
    )
    generate_model_logits_parser.add_argument(
        "--output",
        default=str(DEFAULT_MODEL_LOGITS_OUTPUT),
        help=(
            "artifact output directory "
            f"(default: {DEFAULT_MODEL_LOGITS_OUTPUT})"
        ),
    )
    generate_model_logits_parser.add_argument(
        "--check",
        action="store_true",
        help="build the artifact in memory without writing files",
    )

    args = parser.parse_args(argv)
    if args.command == "check-oracles" and bool(args.depfile) != bool(args.depfile_target):
        parser.error("--depfile and --depfile-target must be supplied together")
    if args.command == "list-groups":
        if VECTOR_GROUPS:
            for group in VECTOR_GROUPS:
                print(group)
        else:
            print("no vector groups registered")
        return
    if args.command == "generate-all":
        generate_all(args.output_root, clean=args.clean)
        print(f"generated all vector groups: output={args.output_root}")
        return
    if args.command == "write-oracles":
        write_oracles(args.input_root, args.oracle_root)
        print(f"wrote oracle hashes: input={args.input_root} output={args.oracle_root}")
        return
    if args.command == "check-oracles":
        check_oracles(args.input_root, args.oracle_root)
        if args.depfile:
            _write_depfile(args.depfile, args.depfile_target, args.oracle_root)
        print(f"oracle hashes verified: input={args.input_root}")
        return
    if args.command == "generate":
        if args.group == "norms":
            output = args.output or DEFAULT_NORMS_OUT
            index = generate_norms(output)
            _print_norm_index(index, output)
            return
        if args.group in {"attention", "full_attention_primitives"}:
            generate_attention.main(["--output", args.output])
            return
        if args.group == "full_attention_block":
            from .generate_full_attention_block import main as generate_block_main

            generate_block_main(["--output", args.output])
            return
        if args.group == "model_logits":
            from .generate_model_logits import main as generate_model_logits_main

            generate_model_logits_main(["--output", args.output])
            return
        if args.group in {"gdn", "gdn_post_conv_prep"}:
            from .generate_gdn import main as generate_gdn_main

            generate_gdn_main(["--output", args.output])
            return
        if args.group == "gdn_decoder_layer":
            from .generate_gdn_decoder_layer import main as generate_gdn_decoder_layer_main

            generate_gdn_decoder_layer_main(["--output", args.output])
            return
        if args.group not in {"moe", "moe_shared_expert"}:
            parser.error(f"unknown vector group {args.group!r}")
        from .moe import write_moe_artifact

        manifest_path = write_moe_artifact(args.output)
        print(f"generated moe_shared_expert: {manifest_path.parent}")
        return
    if args.command == "generate-moe":
        from .generate_moe import main as generate_moe_main

        generate_args = []
        output = args.output_flag or args.output
        if output is not None:
            generate_args.extend(["--output", output])
        if args.check:
            generate_args.append("--check")
        generate_moe_main(generate_args)
        return
    if args.command == "generate-norms":
        index = generate_norms(args.output)
        _print_norm_index(index, args.output)
        return
    if args.command == "generate-gdn":
        from .generate_gdn import main as generate_gdn_main

        generate_args = ["--output", args.output]
        if args.check:
            generate_args.append("--check")
        generate_gdn_main(generate_args)
        return
    if args.command == "generate-gdn-decoder-layer":
        from .generate_gdn_decoder_layer import main as generate_gdn_decoder_layer_main

        generate_args = ["--output", args.output]
        if args.check:
            generate_args.append("--check")
        generate_gdn_decoder_layer_main(generate_args)
        return
    if args.command == "generate-attention":
        generate_args = ["--output", args.output]
        if args.check:
            generate_args.append("--check")
        if args.force:
            generate_args.append("--force")
        generate_attention.main(generate_args)
        return
    if args.command == "generate-full-attention-block":
        from .generate_full_attention_block import main as generate_block_main

        generate_args = ["--output", args.output]
        if args.check:
            generate_args.append("--check")
        generate_block_main(generate_args)
        return
    if args.command == "generate-model-logits":
        from .generate_model_logits import main as generate_model_logits_main

        generate_args = ["--output", args.output]
        if args.check:
            generate_args.append("--check")
        generate_model_logits_main(generate_args)
        return
    parser.print_help()


def _write_depfile(output: str, target: str, oracle_root: str) -> None:
    dependencies: set[Path] = set()
    for root, suffix in (
        (Path(__file__).resolve().parent, ".py"),
        (Path(oracle_root).resolve(), ".json"),
    ):
        for directory, subdirs, files in os.walk(root):
            subdirs[:] = [name for name in subdirs if name != "__pycache__"]
            # Directory mtimes catch additions/removals; file mtimes catch edits.
            dependencies.add(Path(directory))
            dependencies.update(
                Path(directory) / name for name in files if name.endswith(suffix)
            )

    def escape(path: str) -> str:
        if "\n" in path or "\r" in path:
            raise ValueError("depfile paths cannot contain newlines")
        return (
            path.replace("\\", "\\\\")
            .replace("$", "$$")
            .replace("#", "\\#")
            .replace(" ", "\\ ")
            .replace("\t", "\\\t")
            .replace(":", "\\:")
        )

    inputs = " ".join(escape(str(path)) for path in sorted(dependencies))
    depfile = Path(output)
    depfile.parent.mkdir(parents=True, exist_ok=True)
    depfile.write_text(f"{escape(target)}: {inputs}\n", encoding="utf-8")


def _print_norm_index(index: dict[str, object], output: object) -> None:
    cases = index["cases"]
    print(f"generated norms: cases={len(cases)} output={output}")
    for case in cases:
        print(
            f"- {case['name']}: h={case['hidden_size']} rows={case['rows']} "
            f"standard_bf16_mismatches={case['standard_rmsnorm_bf16_mismatch_count']}"
        )
