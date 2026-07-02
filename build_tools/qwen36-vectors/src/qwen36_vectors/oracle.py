from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path
from typing import Any

from .paths import DEFAULT_ORACLE_ROOT, DEFAULT_VECTOR_ROOT, VECTOR_GROUPS


SCHEMA_VERSION = 1


class OracleError(RuntimeError):
    pass


def generate_all(
    output_root: str | Path = DEFAULT_VECTOR_ROOT,
    *,
    clean: bool = False,
) -> None:
    root = Path(output_root)
    if clean:
        for group in VECTOR_GROUPS:
            group_root = root / group
            if group_root.exists():
                shutil.rmtree(group_root)

    from .attention import write_bundle
    from .full_attention_block import write_full_attention_block_artifact
    from .gdn import write_gdn_artifact
    from .gdn_decoder_layer import write_gdn_decoder_layer_artifact
    from .generate_norms import generate_norms
    from .model_logits import write_model_logits_artifact
    from .moe import write_moe_artifact

    write_full_attention_block_artifact(root / "full_attention_block")
    write_bundle(root / "full_attention_primitives", force=True)
    write_gdn_decoder_layer_artifact(root / "gdn_decoder_layer")
    write_gdn_artifact(root / "gdn_post_conv_prep")
    write_model_logits_artifact(root / "model_logits")
    write_moe_artifact(root / "moe_shared_expert")
    generate_norms(root / "norms")


def write_oracles(
    input_root: str | Path = DEFAULT_VECTOR_ROOT,
    oracle_root: str | Path = DEFAULT_ORACLE_ROOT,
    *,
    groups: tuple[str, ...] = VECTOR_GROUPS,
) -> None:
    oracle_dir = Path(oracle_root)
    oracle_dir.mkdir(parents=True, exist_ok=True)
    for group in groups:
        payload = _oracle_payload(group, Path(input_root) / group)
        path = oracle_dir / f"{group}.oracle.json"
        path.write_text(
            json.dumps(payload, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )


def check_oracles(
    input_root: str | Path = DEFAULT_VECTOR_ROOT,
    oracle_root: str | Path = DEFAULT_ORACLE_ROOT,
    *,
    groups: tuple[str, ...] = VECTOR_GROUPS,
) -> None:
    errors: list[str] = []
    for group in groups:
        oracle_path = Path(oracle_root) / f"{group}.oracle.json"
        try:
            expected = _read_oracle(oracle_path, group)
            actual = _file_entries(Path(input_root) / group)
            _compare_group(group, expected, actual)
        except OracleError as err:
            errors.append(str(err))

    if errors:
        raise OracleError("\n".join(errors))


def _oracle_payload(group: str, root: Path) -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "group": group,
        "root": group,
        "hash": "sha256",
        "files": _file_entries(root),
    }


def _read_oracle(path: Path, group: str) -> list[dict[str, Any]]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as err:
        raise OracleError(f"{group}: missing oracle {path}") from err

    if payload.get("schema_version") != SCHEMA_VERSION:
        raise OracleError(f"{group}: unsupported oracle schema {payload.get('schema_version')!r}")
    if payload.get("group") != group:
        raise OracleError(f"{group}: oracle group is {payload.get('group')!r}")
    files = payload.get("files")
    if not isinstance(files, list):
        raise OracleError(f"{group}: oracle files must be a list")
    return files


def _compare_group(
    group: str,
    expected: list[dict[str, Any]],
    actual: list[dict[str, Any]],
) -> None:
    expected_by_path = {entry["path"]: entry for entry in expected}
    actual_by_path = {entry["path"]: entry for entry in actual}
    expected_paths = set(expected_by_path)
    actual_paths = set(actual_by_path)

    missing = sorted(expected_paths - actual_paths)
    unexpected = sorted(actual_paths - expected_paths)
    mismatched = sorted(
        path
        for path in expected_paths & actual_paths
        if expected_by_path[path]["bytes"] != actual_by_path[path]["bytes"]
        or expected_by_path[path]["sha256"] != actual_by_path[path]["sha256"]
    )

    if not missing and not unexpected and not mismatched:
        return

    lines = [f"{group}: generated files do not match oracle"]
    if missing:
        lines.append(f"  missing: {', '.join(missing[:8])}")
    if unexpected:
        lines.append(f"  unexpected: {', '.join(unexpected[:8])}")
    if mismatched:
        lines.append(f"  sha256/size mismatch: {', '.join(mismatched[:8])}")
    raise OracleError("\n".join(lines))


def _file_entries(root: Path) -> list[dict[str, Any]]:
    if not root.exists():
        raise OracleError(f"missing generated vector group {root}")
    if not root.is_dir():
        raise OracleError(f"generated vector group is not a directory: {root}")

    entries = []
    for path in sorted(item for item in root.rglob("*") if item.is_file()):
        raw = path.read_bytes()
        entries.append(
            {
                "path": path.relative_to(root).as_posix(),
                "bytes": len(raw),
                "sha256": hashlib.sha256(raw).hexdigest(),
            }
        )
    if not entries:
        raise OracleError(f"generated vector group is empty: {root}")
    return entries
