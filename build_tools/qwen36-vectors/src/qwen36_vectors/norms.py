from __future__ import annotations

import bisect
import hashlib
import json
import math
import struct
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .io import bf16_bits_to_float32, float32_to_bf16_bits, write_artifact
from .schema import TensorSpec, VectorManifest


DEFAULT_EPS = 1.0e-6
QWEN36_HIDDEN_SIZE = 2048


@dataclass(frozen=True)
class NormCase:
    name: str
    op: str
    hidden_size: int
    rows: int
    primitive_only: bool
    x: tuple[tuple[float, ...], ...]
    raw_weight: tuple[float, ...]
    residual: tuple[tuple[float, ...], ...] | None = None
    eps: float = DEFAULT_EPS


@dataclass(frozen=True)
class MaterializedNormCase:
    manifest: VectorManifest
    tensors: dict[str, list[int] | list[float]]
    metadata: dict[str, Any]


def generate_norm_artifacts(out_dir: str | Path) -> dict[str, Any]:
    root = Path(out_dir)
    root.mkdir(parents=True, exist_ok=True)

    cases: list[dict[str, Any]] = []
    for case in norm_cases():
        materialized = materialize_norm_case(case)
        case_dir = root / case.name
        write_artifact(case_dir, materialized.manifest, materialized.tensors)
        case_hash = _artifact_hash(case_dir)
        cases.append(
            {
                "name": case.name,
                "path": case.name,
                "manifest": f"{case.name}/manifest.json",
                "sha256": case_hash,
                "op": case.op,
                "dtype": "bf16",
                "rows": case.rows,
                "hidden_size": case.hidden_size,
                "primitive_only": case.primitive_only,
                "standard_rmsnorm_expected_fail": True,
                "standard_rmsnorm_max_abs_delta_f32": materialized.metadata[
                    "standard_rmsnorm_comparison"
                ]["max_abs_delta_f32"],
                "standard_rmsnorm_bf16_mismatch_count": materialized.metadata[
                    "standard_rmsnorm_comparison"
                ]["bf16_mismatch_count"],
            }
        )

    index = {
        "schema_version": 1,
        "name": "qwen36_norms",
        "group": "norms",
        "description": (
            "Gemma RMSNorm and Gemma fused add RMSNorm vectors. vLLM/FlashInfer "
            "uses effective_weight = raw_weight + 1.0."
        ),
        "bf16_rounding": (
            "BF16 tensors are raw little-endian u16 files. Expected BF16 output "
            "files are round-to-nearest-even conversions of the f32 oracle, and "
            "each case includes output_bf16_margin_to_midpoint.f32."
        ),
        "cases": cases,
    }
    _write_json(root / "index.json", index)
    _write_readme(root)
    return index


def materialize_norm_case(case: NormCase) -> MaterializedNormCase:
    _validate_case(case)
    x_words = _bf16_words_2d(case.x)
    x_f32 = _decode_bf16_2d(x_words, case.rows, case.hidden_size)
    weight_words = _bf16_words_1d(case.raw_weight)
    weight_f32 = _decode_bf16_1d(weight_words)

    tensors: dict[str, list[int] | list[float]] = {
        "x": _flatten_int(x_words),
        "raw_weight": weight_words,
        "gemma_effective_weight_f32": [_f32(value + 1.0) for value in weight_f32],
    }

    if case.op == "gemma_rmsnorm":
        norm_input = x_f32
    elif case.op == "gemma_fused_add_rmsnorm":
        if case.residual is None:
            raise ValueError(f"{case.name}: residual is required for fused case")
        residual_words = _bf16_words_2d(case.residual)
        residual_f32 = _decode_bf16_2d(residual_words, case.rows, case.hidden_size)
        residual_out_f32 = _add_rows_f32(x_f32, residual_f32)
        tensors["residual"] = _flatten_int(residual_words)
        tensors["expected_residual_out_f32"] = _flatten_float(residual_out_f32)
        tensors["expected_residual_out_bf16"] = _bf16_words_1d(
            _flatten_float(residual_out_f32)
        )
        norm_input = residual_out_f32
    else:
        raise ValueError(f"{case.name}: unsupported op {case.op!r}")

    expected_output_f32 = gemma_rmsnorm_f32(norm_input, weight_f32, case.eps)
    standard_output_f32 = standard_rmsnorm_f32(norm_input, weight_f32, case.eps)
    expected_output_words = _bf16_words_1d(_flatten_float(expected_output_f32))
    standard_output_words = _bf16_words_1d(_flatten_float(standard_output_f32))
    rounding_margins, rounding_summary = _bf16_rounding_margins(expected_output_f32)
    comparison = _comparison_summary(
        expected_output_f32,
        standard_output_f32,
        expected_output_words,
        standard_output_words,
        weight_f32,
    )

    tensors.update(
        {
            "expected_output_f32": _flatten_float(expected_output_f32),
            "expected_output_bf16": expected_output_words,
            "standard_output_f32": _flatten_float(standard_output_f32),
            "standard_output_bf16": standard_output_words,
            "output_bf16_margin_to_midpoint_f32": rounding_margins,
        }
    )

    metadata = {
        "semantic_reference": "vLLM/FlashInfer Gemma RMSNorm",
        "effective_weight": "raw_weight + 1.0",
        "eps": case.eps,
        "primitive_only": case.primitive_only,
        "input_pattern": "deterministic signed table pattern; not random",
        "raw_weight_pattern": (
            "deterministic nonuniform signed pattern containing negative, zero, "
            "and positive raw weights"
        ),
        "formula": (
            "inv_rms = rsqrt(mean(norm_input^2) + eps); "
            "expected_output = norm_input * inv_rms * (raw_weight + 1.0)"
        ),
        "fused_residual_order": (
            "gemma_fused_add_rmsnorm computes residual_out = x + residual first, "
            "then normalizes residual_out"
        ),
        "bf16_rounding": rounding_summary,
        "standard_rmsnorm_comparison": comparison,
    }
    tensorspecs = _tensor_specs(case)
    manifest = VectorManifest(
        name=case.name,
        groups=("qwen36_semantics", "norms"),
        description=f"{case.op} correctness vector for Gemma raw_weight + 1 semantics",
        tensors=tensorspecs,
        metadata=metadata,
    )
    return MaterializedNormCase(manifest=manifest, tensors=tensors, metadata=metadata)


def gemma_rmsnorm_f32(
    rows: list[list[float]], raw_weight: list[float], eps: float
) -> list[list[float]]:
    return _rmsnorm_f32(rows, raw_weight, eps, weight_bias=1.0)


def standard_rmsnorm_f32(
    rows: list[list[float]], raw_weight: list[float], eps: float
) -> list[list[float]]:
    return _rmsnorm_f32(rows, raw_weight, eps, weight_bias=0.0)


def norm_cases() -> tuple[NormCase, ...]:
    debug_weight = _debug_weight()
    hidden_weight = _hidden2048_weight()
    return (
        NormCase(
            name="gemma_rmsnorm_dim8_debug",
            op="gemma_rmsnorm",
            hidden_size=8,
            rows=2,
            primitive_only=True,
            x=_debug_x(),
            raw_weight=debug_weight,
        ),
        NormCase(
            name="gemma_fused_add_rmsnorm_dim8_debug",
            op="gemma_fused_add_rmsnorm",
            hidden_size=8,
            rows=2,
            primitive_only=True,
            x=_debug_x(),
            raw_weight=debug_weight,
            residual=_debug_residual(),
        ),
        NormCase(
            name="gemma_rmsnorm_hidden2048",
            op="gemma_rmsnorm",
            hidden_size=QWEN36_HIDDEN_SIZE,
            rows=3,
            primitive_only=False,
            x=_hidden2048_x(3),
            raw_weight=hidden_weight,
        ),
        NormCase(
            name="gemma_fused_add_rmsnorm_hidden2048",
            op="gemma_fused_add_rmsnorm",
            hidden_size=QWEN36_HIDDEN_SIZE,
            rows=2,
            primitive_only=False,
            x=_hidden2048_x(2),
            raw_weight=hidden_weight,
            residual=_hidden2048_residual(2),
        ),
    )


def _tensor_specs(case: NormCase) -> tuple[TensorSpec, ...]:
    common = [
        TensorSpec(
            "x",
            "bf16",
            (case.rows, case.hidden_size),
            role="input",
            description="BF16 input hidden states as raw u16 bit patterns.",
        ),
        TensorSpec(
            "raw_weight",
            "bf16",
            (case.hidden_size,),
            role="input",
            description="Raw Gemma RMSNorm weight. Effective weight is raw_weight + 1.0.",
        ),
        TensorSpec(
            "gemma_effective_weight_f32",
            "f32",
            (case.hidden_size,),
            role="derived",
            description="Decoded raw_weight plus 1.0, included to make the Gemma bias explicit.",
        ),
    ]
    if case.op == "gemma_fused_add_rmsnorm":
        common.extend(
            [
                TensorSpec(
                    "residual",
                    "bf16",
                    (case.rows, case.hidden_size),
                    role="input",
                    description="BF16 residual input as raw u16 bit patterns.",
                ),
                TensorSpec(
                    "expected_residual_out_f32",
                    "f32",
                    (case.rows, case.hidden_size),
                    role="expected",
                    description="f32 oracle for residual_out = x + residual.",
                ),
                TensorSpec(
                    "expected_residual_out_bf16",
                    "bf16",
                    (case.rows, case.hidden_size),
                    role="expected",
                    description="BF16-rounded residual_out = x + residual.",
                ),
            ]
        )
    common.extend(
        [
            TensorSpec(
                "expected_output_f32",
                "f32",
                (case.rows, case.hidden_size),
                role="expected",
                description="f32 Gemma RMSNorm oracle using raw_weight + 1.0.",
            ),
            TensorSpec(
                "expected_output_bf16",
                "bf16",
                (case.rows, case.hidden_size),
                role="expected",
                description="BF16-rounded Gemma RMSNorm expected output.",
            ),
            TensorSpec(
                "standard_output_f32",
                "f32",
                (case.rows, case.hidden_size),
                role="comparison",
                description="f32 output from incorrect standard RMSNorm using raw_weight.",
            ),
            TensorSpec(
                "standard_output_bf16",
                "bf16",
                (case.rows, case.hidden_size),
                role="comparison",
                description="BF16-rounded output from incorrect standard RMSNorm using raw_weight.",
            ),
            TensorSpec(
                "output_bf16_margin_to_midpoint_f32",
                "f32",
                (case.rows, case.hidden_size),
                role="diagnostic",
                description=(
                    "Distance from f32 oracle output to the nearest BF16 rounding midpoint."
                ),
            ),
        ]
    )
    return tuple(common)


def _rmsnorm_f32(
    rows: list[list[float]],
    raw_weight: list[float],
    eps: float,
    *,
    weight_bias: float,
) -> list[list[float]]:
    hidden_size = len(raw_weight)
    out: list[list[float]] = []
    for row in rows:
        square_mean = math.fsum(float(value) * float(value) for value in row) / hidden_size
        inv_rms = 1.0 / math.sqrt(square_mean + eps)
        out.append(
            [
                _f32(float(value) * inv_rms * (float(weight) + weight_bias))
                for value, weight in zip(row, raw_weight)
            ]
        )
    return out


def _comparison_summary(
    gemma_f32: list[list[float]],
    standard_f32: list[list[float]],
    gemma_bf16_words: list[int],
    standard_bf16_words: list[int],
    raw_weight_f32: list[float],
) -> dict[str, Any]:
    max_abs_delta = -1.0
    max_location = (0, 0)
    nonzero_delta_count = 0
    for row_index, (gemma_row, standard_row) in enumerate(zip(gemma_f32, standard_f32)):
        for col_index, (gemma_value, standard_value) in enumerate(zip(gemma_row, standard_row)):
            delta = abs(float(gemma_value) - float(standard_value))
            if delta != 0.0:
                nonzero_delta_count += 1
            if delta > max_abs_delta:
                max_abs_delta = delta
                max_location = (row_index, col_index)

    row, col = max_location
    bf16_mismatch_count = sum(
        1 for gemma, standard in zip(gemma_bf16_words, standard_bf16_words) if gemma != standard
    )
    return {
        "standard_formula": "output = norm_input * inv_rms * raw_weight",
        "why_it_fails": (
            "The vectors are generated for vLLM/FlashInfer Gemma semantics, "
            "where output = norm_input * inv_rms * (raw_weight + 1.0). "
            "Raw weights include negative, zero, and positive values; a standard "
            "RMSNorm path drops the +1.0 term, so output lanes diverge. "
            "For fused add, residual_out still matches because the add is the same."
        ),
        "nonzero_delta_count": nonzero_delta_count,
        "bf16_mismatch_count": bf16_mismatch_count,
        "max_abs_delta_f32": _f32(max_abs_delta),
        "max_abs_delta_location": [row, col],
        "at_max_delta": {
            "gemma_f32": gemma_f32[row][col],
            "standard_f32": standard_f32[row][col],
            "raw_weight_f32": raw_weight_f32[col],
            "gemma_effective_weight_f32": _f32(raw_weight_f32[col] + 1.0),
        },
    }


def _bf16_rounding_margins(matrix: list[list[float]]) -> tuple[list[float], dict[str, Any]]:
    margins: list[float] = []
    min_margin = math.inf
    min_location = (0, 0)
    min_interval: dict[str, Any] = {}
    within_1e_7 = 0
    within_1e_6 = 0

    for row_index, row in enumerate(matrix):
        for col_index, value in enumerate(row):
            rounded_word = float32_to_bf16_bits(value)
            rounded_value = bf16_bits_to_float32(rounded_word)
            idx = bisect.bisect_left(_FINITE_BF16_VALUES, rounded_value)
            if idx <= 0 or idx >= len(_FINITE_BF16_VALUES) - 1:
                margin = math.inf
                lower_midpoint = -math.inf
                upper_midpoint = math.inf
            else:
                lower_midpoint = (rounded_value + _FINITE_BF16_VALUES[idx - 1]) / 2.0
                upper_midpoint = (rounded_value + _FINITE_BF16_VALUES[idx + 1]) / 2.0
                margin = min(abs(float(value) - lower_midpoint), abs(upper_midpoint - float(value)))
            margin_f32 = _f32(margin)
            margins.append(margin_f32)
            if margin < 1.0e-7:
                within_1e_7 += 1
            if margin < 1.0e-6:
                within_1e_6 += 1
            if margin < min_margin:
                min_margin = margin
                min_location = (row_index, col_index)
                min_interval = {
                    "oracle_f32": _f32(value),
                    "rounded_bf16_u16": rounded_word,
                    "rounded_bf16_f32": rounded_value,
                    "lower_midpoint_f32": _f32(lower_midpoint),
                    "upper_midpoint_f32": _f32(upper_midpoint),
                }

    return margins, {
        "mode": "IEEE round-to-nearest-even",
        "boundary_definition": (
            "For each f32 oracle value, the nearest BF16 rounding boundaries are "
            "midpoints between the rounded BF16 value and its adjacent finite "
            "representable BF16 values."
        ),
        "margin_tensor": "output_bf16_margin_to_midpoint_f32",
        "min_abs_distance_to_output_bf16_midpoint_f32": _f32(min_margin),
        "min_margin_location": [min_location[0], min_location[1]],
        "min_margin_interval": min_interval,
        "values_within_1e-7_of_midpoint": within_1e_7,
        "values_within_1e-6_of_midpoint": within_1e_6,
    }


def _finite_bf16_values() -> list[float]:
    values: set[float] = set()
    for word in range(0x10000):
        value = bf16_bits_to_float32(word)
        if math.isfinite(value):
            values.add(value)
    return sorted(values)


_FINITE_BF16_VALUES = _finite_bf16_values()


def _validate_case(case: NormCase) -> None:
    if case.hidden_size <= 0:
        raise ValueError(f"{case.name}: hidden_size must be positive")
    if case.rows <= 0:
        raise ValueError(f"{case.name}: rows must be positive")
    if len(case.raw_weight) != case.hidden_size:
        raise ValueError(f"{case.name}: raw_weight length mismatch")
    if len(case.x) != case.rows:
        raise ValueError(f"{case.name}: x row count mismatch")
    for row in case.x:
        if len(row) != case.hidden_size:
            raise ValueError(f"{case.name}: x hidden dimension mismatch")
    if case.residual is not None:
        if len(case.residual) != case.rows:
            raise ValueError(f"{case.name}: residual row count mismatch")
        for row in case.residual:
            if len(row) != case.hidden_size:
                raise ValueError(f"{case.name}: residual hidden dimension mismatch")
    if not any(value < 0 for value in case.raw_weight):
        raise ValueError(f"{case.name}: raw weights must include negative values")
    if not any(value == 0 for value in case.raw_weight):
        raise ValueError(f"{case.name}: raw weights must include zero values")
    if not any(value > 0 for value in case.raw_weight):
        raise ValueError(f"{case.name}: raw weights must include positive values")
    if case.hidden_size != QWEN36_HIDDEN_SIZE and not case.primitive_only:
        raise ValueError(f"{case.name}: non-2048 hidden dim cases must be primitive_only")


def _bf16_words_2d(rows: tuple[tuple[float, ...], ...]) -> list[list[int]]:
    return [[float32_to_bf16_bits(value) for value in row] for row in rows]


def _bf16_words_1d(values: list[float] | tuple[float, ...]) -> list[int]:
    return [float32_to_bf16_bits(value) for value in values]


def _decode_bf16_2d(words: list[list[int]], rows: int, hidden_size: int) -> list[list[float]]:
    return [
        [bf16_bits_to_float32(words[row][col]) for col in range(hidden_size)]
        for row in range(rows)
    ]


def _decode_bf16_1d(words: list[int]) -> list[float]:
    return [bf16_bits_to_float32(word) for word in words]


def _add_rows_f32(lhs: list[list[float]], rhs: list[list[float]]) -> list[list[float]]:
    return [[_f32(left + right) for left, right in zip(lhs_row, rhs_row)] for lhs_row, rhs_row in zip(lhs, rhs)]


def _flatten_float(rows: list[list[float]]) -> list[float]:
    return [value for row in rows for value in row]


def _flatten_int(rows: list[list[int]]) -> list[int]:
    return [value for row in rows for value in row]


def _f32(value: float) -> float:
    return struct.unpack("<f", struct.pack("<f", float(value)))[0]


def _artifact_hash(case_dir: Path) -> str:
    digest = hashlib.sha256()
    for path in sorted(case_dir.iterdir()):
        if not path.is_file():
            continue
        digest.update(path.name.encode("utf-8"))
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_readme(root: Path) -> None:
    text = """# Qwen3.6 Norm Semantics Vectors

These generated artifacts pin vLLM/FlashInfer Gemma RMSNorm semantics:
`effective_weight = raw_weight + 1.0`.

For `gemma_fused_add_rmsnorm`, the fused residual step is:
`residual_out = x + residual`, then `output = gemma_rmsnorm(residual_out)`.
The residual output is unchanged by the Gemma weight bias; the normalized output
is what fails against standard RMSNorm.

Each case directory is a raw-tensor artifact with `manifest.json`. BF16 tensors
are little-endian raw `u16` bit patterns. The `dim8_debug` cases are marked
`primitive_only`; the `hidden2048` cases use the Qwen3.6 hidden dimension.
"""
    (root / "README.md").write_text(text, encoding="utf-8")


def _debug_x() -> tuple[tuple[float, ...], ...]:
    return (
        (1.0, -2.0, 0.5, -0.25, 4.0, -1.5, 0.125, -0.75),
        (-0.5, 0.25, -3.0, 2.5, -0.125, 1.75, -2.25, 0.625),
    )


def _debug_residual() -> tuple[tuple[float, ...], ...]:
    return (
        (-0.25, 0.5, 1.0, -1.5, 0.75, 0.125, -0.375, 2.0),
        (1.5, -0.75, 0.25, -0.5, 2.0, -1.25, 0.875, -0.125),
    )


def _debug_weight() -> tuple[float, ...]:
    return (-0.75, -0.25, 0.0, 0.25, 0.5, -0.5, 1.0, -0.125)


def _hidden2048_weight() -> tuple[float, ...]:
    pattern = (
        -0.75,
        -0.625,
        -0.5,
        -0.375,
        -0.25,
        -0.125,
        0.0,
        0.0625,
        0.125,
        0.25,
        0.375,
        0.5,
        0.75,
        1.0,
        1.25,
        1.5,
    )
    return tuple(pattern[index % len(pattern)] for index in range(QWEN36_HIDDEN_SIZE))


def _hidden2048_x(rows: int) -> tuple[tuple[float, ...], ...]:
    base = (
        -2.0,
        -1.25,
        -0.75,
        -0.5,
        -0.125,
        0.1875,
        0.5,
        0.875,
        1.25,
        1.75,
        2.5,
        3.0,
        -3.5,
        0.03125,
        -0.03125,
        4.0,
    )
    out: list[tuple[float, ...]] = []
    for row in range(rows):
        row_values: list[float] = []
        row_offset = (row - 1) * 0.0625
        sign = -1.0 if row % 2 else 1.0
        for col in range(QWEN36_HIDDEN_SIZE):
            lane = base[(col + row * 3) % len(base)]
            slow = ((col // len(base)) % 7 - 3) * 0.015625
            row_values.append(_f32(sign * lane + row_offset + slow))
        out.append(tuple(row_values))
    return tuple(out)


def _hidden2048_residual(rows: int) -> tuple[tuple[float, ...], ...]:
    base = (
        0.5,
        -0.875,
        1.5,
        -1.75,
        0.25,
        -0.0625,
        2.25,
        -2.5,
        0.125,
        0.75,
        -1.25,
        3.5,
        -3.0,
        0.375,
        -0.5,
        1.0,
    )
    out: list[tuple[float, ...]] = []
    for row in range(rows):
        row_values: list[float] = []
        row_offset = (1 - row) * 0.03125
        for col in range(QWEN36_HIDDEN_SIZE):
            lane = base[(col * 5 + row) % len(base)]
            slow = ((col // len(base)) % 5 - 2) * 0.0078125
            row_values.append(_f32(lane + row_offset + slow))
        out.append(tuple(row_values))
    return tuple(out)
