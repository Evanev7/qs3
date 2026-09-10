"""Deterministic Qwen3.6 full-attention block vector.

This vector covers the block composition above the primitive attention prep:
decoder Gemma RMSNorm, dense Q/K/V projections, packed Q/output-gate split,
q/k Gemma RMSNorm, partial RoPE, causal GQA attention, sigmoid output gate,
O projection, residual add, and post-attention Gemma RMSNorm.
"""

from __future__ import annotations

import hashlib
import json
import math
import struct
import sys
from array import array
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .io import bf16_bits_to_float32, float32_to_bf16_bits
from .paths import DEFAULT_VECTOR_ROOT
from .schema import MANIFEST_FILE, TensorSpec, VectorManifest

ROWS = 6
HIDDEN_SIZE = 2048
Q_HEADS = 16
KV_HEADS = 2
HEAD_DIM = 256
ROTARY_DIM = 64
Q_HIDDEN = Q_HEADS * HEAD_DIM
KV_HIDDEN = KV_HEADS * HEAD_DIM
Q_PROJ_OUT = 2 * Q_HIDDEN
GROUP_SIZE = Q_HEADS // KV_HEADS
POSITIONS = (0, 1, 7, 64, 65, 511)
RMS_EPS = 1.0e-6
ROPE_THETA = 10_000.0
ROPE_SCALE = 1.0
ATTENTION_SCALE = 1.0 / math.sqrt(HEAD_DIM)
DEFAULT_OUTPUT = DEFAULT_VECTOR_ROOT / "full_attention_block"
MOE_NUM_EXPERTS = 256
MOE_TOP_K = 8
MOE_INTERMEDIATE = 8
MOE_ROUTED_SCALING_FACTOR = 1.0
MOE_ROUTER_TERMS = 5
MOE_GATE_UP_TERMS = 5
MOE_SHARED_TERMS = 5
INLINE_INPUT_TENSORS = 7


@dataclass(frozen=True)
class TensorWrite:
    name: str
    dtype: str
    shape: tuple[int, ...]
    values: tuple[int, ...] | tuple[float, ...]
    role: str
    op: str
    description: str

    @property
    def file_name(self) -> str:
        return f"block_{self.name}.{self.dtype}"

    @property
    def element_count(self) -> int:
        total = 1
        for extent in self.shape:
            total *= extent
        return total


@dataclass(frozen=True)
class WrittenTensor:
    spec: TensorSpec
    sha256: str
    byte_count: int


def _f32(value: float) -> float:
    return struct.unpack("<f", struct.pack("<f", float(value)))[0]


def _bf16_words(values: Iterable[float]) -> tuple[int, ...]:
    return tuple(float32_to_bf16_bits(value) for value in values)


def _bf16_to_f32_rows(
    values: tuple[int, ...], rows: int, cols: int
) -> list[list[float]]:
    return [
        [bf16_bits_to_float32(values[row * cols + col]) for col in range(cols)]
        for row in range(rows)
    ]


def _contiguous_strides(shape: tuple[int, ...]) -> list[int]:
    stride = 1
    out: list[int] = []
    for extent in reversed(shape):
        out.append(stride)
        stride *= extent
    return list(reversed(out))


def _input_value(row: int, col: int) -> float:
    centered = ((row * 101 + col * 17 + (col // 16) * 5) % 127) - 63
    slow = (((col // 64) % 9) - 4) * 0.01171875
    row_bias = (row - 2.5) * 0.01953125
    marker = 0.03515625 if col in (0, 1, 63, 64, 255, 511, 1024, 2047) else 0.0
    return centered * 0.0078125 + slow + row_bias + marker


def _hidden_norm_weight_value(col: int) -> float:
    base = ((col * 7 + (col // 31) * 3) % 41 - 20) * 0.00390625
    return base + (0.01171875 if col % 5 == 0 else -0.005859375)


def _post_norm_weight_value(col: int) -> float:
    base = ((col * 11 + (col // 17) * 5) % 43 - 21) * 0.003662109375
    return base + (-0.009765625 if col % 7 == 0 else 0.0068359375)


def _next_norm_weight_value(col: int) -> float:
    base = ((col * 13 + (col // 19) * 7) % 47 - 23) * 0.003173828125
    return base + (0.0107421875 if col % 11 == 0 else -0.0048828125)


def _q_norm_weight_value(lane: int) -> float:
    base = ((lane * 7) % 31 - 15) * 0.0078125
    return base + (0.015625 if lane % 2 == 0 else -0.01171875)


def _k_norm_weight_value(lane: int) -> float:
    base = ((lane * 5 + 3) % 29 - 14) * 0.0087890625
    return base + (-0.013671875 if lane % 3 == 0 else 0.017578125)


def _projection_scale(kind: str, out_feature: int) -> float:
    if kind == "q_proj":
        lane_in_pair = out_feature % (2 * HEAD_DIM)
        return 0.24 if lane_in_pair >= HEAD_DIM else 0.020
    if kind == "k_proj":
        return 0.020
    if kind == "v_proj":
        return 0.090
    if kind == "o_proj":
        return 0.23
    raise ValueError(f"unknown projection kind {kind!r}")


def _projection_term_count(kind: str) -> int:
    return 4 if kind == "o_proj" else 3


def _projection_terms(
    kind: str, out_feature: int, in_features: int
) -> tuple[tuple[int, int], ...]:
    scale = _projection_scale(kind, out_feature)
    seed = {
        "q_proj": 13,
        "k_proj": 29,
        "v_proj": 47,
        "o_proj": 61,
    }[kind]
    terms: dict[int, float] = {}
    for term in range(_projection_term_count(kind)):
        col = (
            out_feature * (37 + 12 * term)
            + seed * (term + 1)
            + (out_feature // 17) * (term + 3)
        ) % in_features
        centered = ((out_feature * (19 + term * 6) + term * 23 + seed) % 31) - 15
        sign = -1.0 if ((out_feature >> (term % 5)) + term + seed) & 1 else 1.0
        magnitude = 0.55 + abs(centered) / 30.0
        terms[col] = terms.get(col, 0.0) + sign * scale * magnitude
    out: list[tuple[int, int]] = []
    for col, value in sorted(terms.items()):
        bits = float32_to_bf16_bits(value)
        if bits == 0:
            bits = float32_to_bf16_bits(math.copysign(scale * 0.5, value or 1.0))
        out.append((col, bits))
    return tuple(out)


def _sparse_terms(
    *,
    seed: int,
    owner: int,
    row: int,
    in_features: int,
    term_count: int,
    scale: float,
) -> tuple[tuple[int, int], ...]:
    terms: dict[int, float] = {}
    for term in range(term_count):
        col = (
            owner * (41 + 10 * term)
            + row * (73 + 6 * term)
            + seed * (term + 1)
            + (owner // 11) * (term + 5)
            + (row // 3) * (term + 7)
        ) % in_features
        centered = (
            (owner * (17 + 2 * term) + row * (23 + term) + seed + term * 29) % 37
        ) - 18
        sign = -1.0 if ((owner >> (term % 7)) + row + seed + term) & 1 else 1.0
        magnitude = 0.45 + abs(centered) / 42.0
        terms[col] = terms.get(col, 0.0) + sign * scale * magnitude
    out: list[tuple[int, int]] = []
    for col, value in sorted(terms.items()):
        bits = float32_to_bf16_bits(value)
        if bits == 0:
            bits = float32_to_bf16_bits(math.copysign(scale * 0.5, value or 1.0))
        out.append((col, bits))
    return tuple(out)


def _moe_router_terms(expert: int) -> tuple[tuple[int, int], ...]:
    return _sparse_terms(
        seed=101,
        owner=expert,
        row=expert % 29,
        in_features=HIDDEN_SIZE,
        term_count=MOE_ROUTER_TERMS,
        scale=0.075,
    )


def _moe_gate_up_terms(expert: int, row: int) -> tuple[tuple[int, int], ...]:
    is_up = row >= MOE_INTERMEDIATE
    local_row = row - MOE_INTERMEDIATE if is_up else row
    return _sparse_terms(
        seed=149 if is_up else 137,
        owner=expert,
        row=local_row,
        in_features=HIDDEN_SIZE,
        term_count=MOE_GATE_UP_TERMS,
        scale=0.0625 if is_up else 0.0546875,
    )


def _shared_proj_terms(kind: str, row: int) -> tuple[tuple[int, int], ...]:
    seed = {
        "gate": 173,
        "up": 181,
        "shared_gate": 193,
    }[kind]
    return _sparse_terms(
        seed=seed,
        owner=seed + row * 3,
        row=row,
        in_features=HIDDEN_SIZE,
        term_count=MOE_SHARED_TERMS,
        scale=0.05078125 if kind != "shared_gate" else 0.01953125,
    )


def _moe_down_weight_value(expert: int, hidden: int, intermediate: int) -> float:
    centered = ((expert * 5 + hidden * 3 + intermediate * 17 + hidden // 37) % 41) - 20
    sign = -1.0 if ((expert + hidden + 3 * intermediate) & 1) else 1.0
    value = sign * (0.018 + abs(centered) * 0.00073) + ((expert % 7) - 3) * 0.00041
    return bf16_bits_to_float32(float32_to_bf16_bits(value))


def _shared_down_weight_value(hidden: int, intermediate: int) -> float:
    centered = ((hidden * 5 + intermediate * 19 + hidden // 29) % 43) - 21
    sign = -1.0 if ((hidden + intermediate) & 1) else 1.0
    value = sign * (0.016 + abs(centered) * 0.00061) + (
        0.0011 if hidden % 13 == 0 else -0.0007
    )
    return bf16_bits_to_float32(float32_to_bf16_bits(value))


def _gemma_rmsnorm_rows(
    x_rows: list[list[float]],
    raw_weight_bf16: tuple[int, ...],
) -> tuple[tuple[float, ...], tuple[int, ...], list[list[float]]]:
    weights = [bf16_bits_to_float32(value) + 1.0 for value in raw_weight_bf16]
    out_f32: list[float] = []
    out_bf16: list[int] = []
    out_rows: list[list[float]] = []
    for row in x_rows:
        variance = _f32(sum(_f32(value * value) for value in row) / len(row))
        inv_rms = _f32(1.0 / math.sqrt(_f32(variance + RMS_EPS)))
        out_row: list[float] = []
        for lane, value in enumerate(row):
            y = _f32(_f32(value * inv_rms) * weights[lane])
            out_f32.append(y)
            out_bf16.append(float32_to_bf16_bits(y))
            out_row.append(bf16_bits_to_float32(out_bf16[-1]))
        out_rows.append(out_row)
    return tuple(out_f32), tuple(out_bf16), out_rows


def _project_sparse(
    x_rows: list[list[float]],
    kind: str,
    out_features: int,
    in_features: int,
) -> tuple[tuple[float, ...], tuple[int, ...], list[list[float]]]:
    out_f32: list[float] = []
    out_bf16: list[int] = []
    out_rows: list[list[float]] = []
    decoded_terms = [
        tuple(
            (col, bf16_bits_to_float32(bits))
            for col, bits in _projection_terms(kind, out, in_features)
        )
        for out in range(out_features)
    ]
    for row in x_rows:
        out_row: list[float] = []
        for out in range(out_features):
            acc = 0.0
            for col, weight in decoded_terms[out]:
                acc = _f32(acc + _f32(row[col] * weight))
            out_f32.append(acc)
            bits = float32_to_bf16_bits(acc)
            out_bf16.append(bits)
            out_row.append(bf16_bits_to_float32(bits))
        out_rows.append(out_row)
    return tuple(out_f32), tuple(out_bf16), out_rows


def _project_sparse_terms_f32(
    x_rows: list[list[float]],
    out_features: int,
    term_fn,
) -> tuple[tuple[float, ...], list[list[float]]]:
    decoded_terms = [
        tuple((col, bf16_bits_to_float32(bits)) for col, bits in term_fn(out))
        for out in range(out_features)
    ]
    out_f32: list[float] = []
    out_rows: list[list[float]] = []
    for row in x_rows:
        out_row: list[float] = []
        for out in range(out_features):
            acc = 0.0
            for col, weight in decoded_terms[out]:
                acc = _f32(acc + _f32(row[col] * weight))
            out_f32.append(acc)
            out_row.append(acc)
        out_rows.append(out_row)
    return tuple(out_f32), out_rows


def _project_sparse_terms_bf16(
    x_rows: list[list[float]],
    out_features: int,
    term_fn,
) -> tuple[tuple[float, ...], tuple[int, ...], list[list[float]]]:
    out_f32, _f32_rows = _project_sparse_terms_f32(x_rows, out_features, term_fn)
    out_bf16: list[int] = []
    out_rows: list[list[float]] = []
    idx = 0
    for _ in x_rows:
        row: list[float] = []
        for _out in range(out_features):
            bits = float32_to_bf16_bits(out_f32[idx])
            out_bf16.append(bits)
            row.append(bf16_bits_to_float32(bits))
            idx += 1
        out_rows.append(row)
    return out_f32, tuple(out_bf16), out_rows


def _extract_q_gate(
    q_proj_rows: list[list[float]],
    q_proj_bf16: tuple[int, ...],
) -> tuple[tuple[int, ...], tuple[int, ...], list[list[float]], list[list[float]]]:
    q: list[int] = []
    gate: list[int] = []
    q_rows: list[list[float]] = []
    gate_rows: list[list[float]] = []
    for row in range(ROWS):
        q_row: list[float] = []
        gate_row: list[float] = []
        for head in range(Q_HEADS):
            base = row * Q_PROJ_OUT + head * 2 * HEAD_DIM
            q.extend(q_proj_bf16[base : base + HEAD_DIM])
            gate.extend(q_proj_bf16[base + HEAD_DIM : base + 2 * HEAD_DIM])
            q_row.extend(
                q_proj_rows[row][head * 2 * HEAD_DIM : head * 2 * HEAD_DIM + HEAD_DIM]
            )
            gate_row.extend(
                q_proj_rows[row][
                    head * 2 * HEAD_DIM + HEAD_DIM : (head + 1) * 2 * HEAD_DIM
                ]
            )
        q_rows.append(q_row)
        gate_rows.append(gate_row)
    return tuple(q), tuple(gate), q_rows, gate_rows


def _reshape_heads(rows: list[list[float]], heads: int) -> list[list[list[float]]]:
    return [
        [row[head * HEAD_DIM : (head + 1) * HEAD_DIM] for head in range(heads)]
        for row in rows
    ]


def _flatten_heads_bf16(rows: list[list[list[float]]]) -> tuple[int, ...]:
    return tuple(
        float32_to_bf16_bits(value) for row in rows for head in row for value in head
    )


def _flatten_heads_f32(rows: list[list[list[float]]]) -> tuple[float, ...]:
    return tuple(value for row in rows for head in row for value in head)


def _gemma_rmsnorm_heads(
    heads: list[list[list[float]]],
    raw_weight_bf16: tuple[int, ...],
) -> tuple[tuple[float, ...], tuple[int, ...], list[list[list[float]]]]:
    flat_rows = [head for row in heads for head in row]
    out_f32, out_bf16, out_rows = _gemma_rmsnorm_rows(flat_rows, raw_weight_bf16)
    regrouped: list[list[list[float]]] = []
    idx = 0
    for row in heads:
        out_row = []
        for _ in row:
            out_row.append(out_rows[idx])
            idx += 1
        regrouped.append(out_row)
    return out_f32, out_bf16, regrouped


def _cos_sin_for_position(pos: int) -> tuple[list[float], list[float]]:
    half = ROTARY_DIM // 2
    cos_values: list[float] = []
    sin_values: list[float] = []
    for lane in range(half):
        inv_freq = 1.0 / (ROPE_THETA ** (float(lane * 2) / ROTARY_DIM))
        angle = (pos / ROPE_SCALE) * inv_freq
        cos_values.append(bf16_bits_to_float32(float32_to_bf16_bits(math.cos(angle))))
        sin_values.append(bf16_bits_to_float32(float32_to_bf16_bits(math.sin(angle))))
    return cos_values, sin_values


def _apply_partial_rope(
    heads: list[list[list[float]]],
) -> tuple[tuple[float, ...], tuple[int, ...], list[list[list[float]]]]:
    half = ROTARY_DIM // 2
    out_rows: list[list[list[float]]] = []
    out_f32: list[float] = []
    out_bf16: list[int] = []
    for row_idx, row in enumerate(heads):
        cos_values, sin_values = _cos_sin_for_position(POSITIONS[row_idx])
        out_row: list[list[float]] = []
        for head in row:
            y = list(head)
            for lane in range(half):
                x1 = head[lane]
                x2 = head[half + lane]
                cos = cos_values[lane]
                sin = sin_values[lane]
                y[lane] = _f32(_f32(x1 * cos) - _f32(x2 * sin))
                y[half + lane] = _f32(_f32(x2 * cos) + _f32(x1 * sin))
            rounded = [bf16_bits_to_float32(float32_to_bf16_bits(value)) for value in y]
            out_row.append(rounded)
            out_f32.extend(y)
            out_bf16.extend(float32_to_bf16_bits(value) for value in y)
        out_rows.append(out_row)
    return tuple(out_f32), tuple(out_bf16), out_rows


def _causal_gqa_attention(
    q: list[list[list[float]]],
    k: list[list[list[float]]],
    v: list[list[list[float]]],
) -> tuple[tuple[float, ...], tuple[int, ...], list[list[list[float]]]]:
    out_rows: list[list[list[float]]] = []
    out_f32: list[float] = []
    out_bf16: list[int] = []
    for row_idx in range(ROWS):
        out_row: list[list[float]] = []
        for q_head in range(Q_HEADS):
            kv_head = q_head // GROUP_SIZE
            scores: list[float] = []
            for key_row in range(row_idx + 1):
                dot = 0.0
                for lane in range(HEAD_DIM):
                    dot = _f32(
                        dot + _f32(q[row_idx][q_head][lane] * k[key_row][kv_head][lane])
                    )
                scores.append(_f32(dot * ATTENTION_SCALE))
            max_score = max(scores)
            exp_scores = [math.exp(score - max_score) for score in scores]
            denom = sum(exp_scores)
            probs = [_f32(score / denom) for score in exp_scores]
            head_out: list[float] = []
            for lane in range(HEAD_DIM):
                value = 0.0
                for key_row, prob in enumerate(probs):
                    value = _f32(value + _f32(prob * v[key_row][kv_head][lane]))
                head_out.append(value)
            out_row.append(head_out)
            out_f32.extend(head_out)
            out_bf16.extend(float32_to_bf16_bits(value) for value in head_out)
        out_rows.append(out_row)
    return tuple(out_f32), tuple(out_bf16), out_rows


def _sigmoid(value: float) -> float:
    if value >= 0.0:
        z = math.exp(-value)
        return 1.0 / (1.0 + z)
    z = math.exp(value)
    return z / (1.0 + z)


def _silu(value: float) -> float:
    return value * _sigmoid(value)


def _apply_output_gate(
    raw_attention_bf16: tuple[int, ...],
    gate_bf16: tuple[int, ...],
) -> tuple[tuple[float, ...], tuple[int, ...], list[list[float]]]:
    out_f32: list[float] = []
    out_bf16: list[int] = []
    rows: list[list[float]] = []
    row_width = Q_HIDDEN
    for row_idx in range(ROWS):
        row: list[float] = []
        for col in range(row_width):
            idx = row_idx * row_width + col
            value = _f32(
                bf16_bits_to_float32(raw_attention_bf16[idx])
                * _sigmoid(bf16_bits_to_float32(gate_bf16[idx]))
            )
            out_f32.append(value)
            bits = float32_to_bf16_bits(value)
            out_bf16.append(bits)
            row.append(bf16_bits_to_float32(bits))
        rows.append(row)
    return tuple(out_f32), tuple(out_bf16), rows


def _route_moe_topk(
    logits: list[list[float]],
) -> tuple[list[list[int]], list[list[float]], list[list[float]]]:
    all_ids: list[list[int]] = []
    all_unrenorm: list[list[float]] = []
    all_weights: list[list[float]] = []
    for row in logits:
        best = sorted(
            range(MOE_NUM_EXPERTS), key=lambda expert: (-row[expert], expert)
        )[:MOE_TOP_K]
        max_logit = max(row)
        exp_scores = [math.exp(logit - max_logit) for logit in row]
        denom = sum(exp_scores)
        unrenorm = [_f32(exp_scores[expert] / denom) for expert in best]
        selected_sum = sum(unrenorm)
        weights = [
            _f32((weight / max(selected_sum, 1.0e-20)) * MOE_ROUTED_SCALING_FACTOR)
            for weight in unrenorm
        ]
        all_ids.append(best)
        all_unrenorm.append(unrenorm)
        all_weights.append(weights)
    return all_ids, all_unrenorm, all_weights


def _dot_sparse(row: list[float], terms: tuple[tuple[int, int], ...]) -> float:
    acc = 0.0
    for col, bits in terms:
        acc = _f32(acc + _f32(row[col] * bf16_bits_to_float32(bits)))
    return acc


def _moe_expert_output(row: list[float], expert: int) -> list[float]:
    activated: list[float] = []
    for idx in range(MOE_INTERMEDIATE):
        gate = _dot_sparse(row, _moe_gate_up_terms(expert, idx))
        up = _dot_sparse(row, _moe_gate_up_terms(expert, MOE_INTERMEDIATE + idx))
        activated.append(_f32(_silu(gate) * up))
    out: list[float] = []
    for hidden in range(HIDDEN_SIZE):
        acc = 0.0
        for idx, value in enumerate(activated):
            acc = _f32(acc + _f32(value * _moe_down_weight_value(expert, hidden, idx)))
        out.append(acc)
    return out


def _compute_routed_moe(
    hidden_rows: list[list[float]],
    topk_ids: list[list[int]],
    topk_weights: list[list[float]],
) -> tuple[tuple[float, ...], tuple[int, ...], list[list[float]]]:
    out_f32: list[float] = []
    out_bf16: list[int] = []
    out_rows: list[list[float]] = []
    for row_idx, row in enumerate(hidden_rows):
        routed = [0.0 for _ in range(HIDDEN_SIZE)]
        for route_idx, expert in enumerate(topk_ids[row_idx]):
            expert_out = _moe_expert_output(row, expert)
            scale = topk_weights[row_idx][route_idx]
            for hidden in range(HIDDEN_SIZE):
                routed[hidden] = _f32(routed[hidden] + _f32(scale * expert_out[hidden]))
        row_out: list[float] = []
        for value in routed:
            value = _f32(value)
            out_f32.append(value)
            bits = float32_to_bf16_bits(value)
            out_bf16.append(bits)
            row_out.append(bf16_bits_to_float32(bits))
        out_rows.append(row_out)
    return tuple(out_f32), tuple(out_bf16), out_rows


def _compute_shared_expert(
    hidden_rows: list[list[float]],
) -> tuple[tuple[float, ...], tuple[int, ...], list[list[float]], tuple[float, ...]]:
    _gate_f32, gate_bf16, gate_rows = _project_sparse_terms_bf16(
        hidden_rows,
        MOE_INTERMEDIATE,
        lambda out: _shared_proj_terms("gate", out),
    )
    _up_f32, up_bf16, up_rows = _project_sparse_terms_bf16(
        hidden_rows,
        MOE_INTERMEDIATE,
        lambda out: _shared_proj_terms("up", out),
    )

    activated_rows: list[list[float]] = []
    for row_idx in range(ROWS):
        row: list[float] = []
        for idx in range(MOE_INTERMEDIATE):
            flat = row_idx * MOE_INTERMEDIATE + idx
            value = _f32(
                _silu(bf16_bits_to_float32(gate_bf16[flat]))
                * bf16_bits_to_float32(up_bf16[flat])
            )
            row.append(bf16_bits_to_float32(float32_to_bf16_bits(value)))
        activated_rows.append(row)

    out_f32: list[float] = []
    out_bf16: list[int] = []
    out_rows: list[list[float]] = []
    for row in activated_rows:
        out_row: list[float] = []
        for hidden in range(HIDDEN_SIZE):
            acc = 0.0
            for idx, value in enumerate(row):
                acc = _f32(acc + _f32(value * _shared_down_weight_value(hidden, idx)))
            out_f32.append(acc)
            bits = float32_to_bf16_bits(acc)
            out_bf16.append(bits)
            out_row.append(bf16_bits_to_float32(bits))
        out_rows.append(out_row)

    shared_gate_logits, _ = _project_sparse_terms_f32(
        hidden_rows,
        1,
        lambda out: _shared_proj_terms("shared_gate", out),
    )
    _ = gate_rows, up_rows
    return tuple(out_f32), tuple(out_bf16), out_rows, shared_gate_logits


def _combine_moe_and_shared(
    routed_bf16: tuple[int, ...],
    shared_bf16: tuple[int, ...],
    shared_gate_logits: tuple[float, ...],
) -> tuple[tuple[float, ...], tuple[int, ...]]:
    out_f32: list[float] = []
    out_bf16: list[int] = []
    for row_idx in range(ROWS):
        gate = _f32(_sigmoid(shared_gate_logits[row_idx]))
        for hidden in range(HIDDEN_SIZE):
            idx = row_idx * HIDDEN_SIZE + hidden
            value = _f32(
                bf16_bits_to_float32(routed_bf16[idx])
                + _f32(gate * bf16_bits_to_float32(shared_bf16[idx]))
            )
            out_f32.append(value)
            out_bf16.append(float32_to_bf16_bits(value))
    return tuple(out_f32), tuple(out_bf16)


def _flatten2(
    values: list[list[float]] | list[list[int]],
) -> tuple[float, ...] | tuple[int, ...]:
    return tuple(item for row in values for item in row)


def _compute_one_layer_moe_tensors(
    post_attention_rows: list[list[float]],
    residual_after_attention_bf16: tuple[int, ...],
) -> tuple[list[TensorWrite], dict[str, Any]]:
    next_norm_weight = _bf16_words(
        _next_norm_weight_value(col) for col in range(HIDDEN_SIZE)
    )
    router_logits_f32, router_logits_bf16, router_logits_rows = (
        _project_sparse_terms_bf16(
            post_attention_rows,
            MOE_NUM_EXPERTS,
            _moe_router_terms,
        )
    )
    topk_ids, topk_unrenorm, topk_weights = _route_moe_topk(router_logits_rows)
    routed_f32, routed_bf16, _routed_rows = _compute_routed_moe(
        post_attention_rows,
        topk_ids,
        topk_weights,
    )
    shared_f32, shared_bf16, _shared_rows, shared_gate_logits = _compute_shared_expert(
        post_attention_rows,
    )
    combined_f32, combined_bf16 = _combine_moe_and_shared(
        routed_bf16,
        shared_bf16,
        shared_gate_logits,
    )
    final_residual_f32, final_residual_bf16, final_residual_rows = _add_rows(
        combined_bf16,
        residual_after_attention_bf16,
        HIDDEN_SIZE,
    )
    final_norm_f32, final_norm_bf16, _ = _gemma_rmsnorm_rows(
        final_residual_rows,
        next_norm_weight,
    )

    tensors = [
        TensorWrite(
            "next_layer_norm_raw_weight",
            "bf16",
            (HIDDEN_SIZE,),
            next_norm_weight,
            "input",
            "fused_add_gemma_rmsnorm",
            "Raw next-layer input RMSNorm weight; effective multiplier is raw + 1.",
        ),
        TensorWrite(
            "expected_moe_router_logits_f32",
            "f32",
            (ROWS, MOE_NUM_EXPERTS),
            router_logits_f32,
            "reference",
            "moe_router_projection",
            "FP32 accumulation before BF16 router-output rounding.",
        ),
        TensorWrite(
            "expected_moe_router_logits_bf16",
            "bf16",
            (ROWS, MOE_NUM_EXPERTS),
            router_logits_bf16,
            "expected",
            "moe_router_projection",
            "BF16 router logits consumed by FP32 softmax and top-k.",
        ),
        TensorWrite(
            "expected_moe_topk_ids",
            "i32",
            (ROWS, MOE_TOP_K),
            _flatten2(topk_ids),
            "expected",
            "moe_router_topk",
            "Selected top-8 expert ids after router softmax ranking.",
        ),
        TensorWrite(
            "expected_moe_topk_unrenormalized_weights_f32",
            "f32",
            (ROWS, MOE_TOP_K),
            _flatten2(topk_unrenorm),
            "reference",
            "moe_router_topk",
            "Selected global-softmax probabilities before top-k renormalization.",
        ),
        TensorWrite(
            "expected_moe_topk_weights_f32",
            "f32",
            (ROWS, MOE_TOP_K),
            _flatten2(topk_weights),
            "expected",
            "moe_router_topk",
            "Top-k-renormalized route weights.",
        ),
        TensorWrite(
            "expected_routed_moe_output_f32",
            "f32",
            (ROWS, HIDDEN_SIZE),
            routed_f32,
            "reference",
            "moe_execute",
            "f32 routed MoE output before BF16 storage and shared expert add.",
        ),
        TensorWrite(
            "expected_routed_moe_output_bf16",
            "bf16",
            (ROWS, HIDDEN_SIZE),
            routed_bf16,
            "reference",
            "moe_execute",
            "Routed MoE output rounded to BF16 before shared expert add.",
        ),
        TensorWrite(
            "expected_shared_expert_output_f32",
            "f32",
            (ROWS, HIDDEN_SIZE),
            shared_f32,
            "reference",
            "shared_expert",
            "f32 shared expert output before BF16 storage and sigmoid gate.",
        ),
        TensorWrite(
            "expected_shared_expert_output_bf16",
            "bf16",
            (ROWS, HIDDEN_SIZE),
            shared_bf16,
            "reference",
            "shared_expert",
            "Shared expert output rounded to BF16 before sigmoid gate.",
        ),
        TensorWrite(
            "expected_shared_gate_logits_f32",
            "f32",
            (ROWS, 1),
            shared_gate_logits,
            "reference",
            "shared_expert_gate",
            "f32 shared expert gate logits from the post-attention hidden state.",
        ),
        TensorWrite(
            "expected_moe_shared_output_f32",
            "f32",
            (ROWS, HIDDEN_SIZE),
            combined_f32,
            "reference",
            "shared_expert_gate_add",
            "f32 routed MoE plus sigmoid-gated shared expert output.",
        ),
        TensorWrite(
            "expected_moe_shared_output_bf16",
            "bf16",
            (ROWS, HIDDEN_SIZE),
            combined_bf16,
            "expected",
            "shared_expert_gate_add",
            "Combined MoE/shared expert output rounded to BF16.",
        ),
        TensorWrite(
            "expected_residual_after_mlp_f32",
            "f32",
            (ROWS, HIDDEN_SIZE),
            final_residual_f32,
            "reference",
            "mlp_residual_add",
            "f32 residual after adding the BF16 MoE/shared expert output.",
        ),
        TensorWrite(
            "expected_residual_after_mlp_bf16",
            "bf16",
            (ROWS, HIDDEN_SIZE),
            final_residual_bf16,
            "expected",
            "mlp_residual_add",
            "BF16 residual after the post-attention MoE/shared expert add.",
        ),
        TensorWrite(
            "expected_next_layer_norm_output_f32",
            "f32",
            (ROWS, HIDDEN_SIZE),
            final_norm_f32,
            "reference",
            "fused_add_gemma_rmsnorm",
            "f32 next-layer Gemma RMSNorm oracle after MLP residual add.",
        ),
        TensorWrite(
            "expected_next_layer_norm_output_bf16",
            "bf16",
            (ROWS, HIDDEN_SIZE),
            final_norm_bf16,
            "expected",
            "fused_add_gemma_rmsnorm",
            "BF16 next-layer Gemma RMSNorm output.",
        ),
    ]
    metadata = {
        "moe": {
            "num_experts": MOE_NUM_EXPERTS,
            "top_k": MOE_TOP_K,
            "intermediate_size": MOE_INTERMEDIATE,
            "shared_expert_intermediate_size": MOE_INTERMEDIATE,
            "router_score": "softmax",
            "topk_renormalize": True,
            "routed_scaling_factor": MOE_ROUTED_SCALING_FACTOR,
            "tie_break": "higher score first, then lower expert id",
        },
        "projection_layout": {
            "router_proj_weight": "[num_experts, hidden_size]",
            "gate_up_proj_weight": (
                "[num_experts, 2 * intermediate_size, hidden_size], "
                "gate rows first then up rows"
            ),
            "down_proj_weight": "[num_experts, hidden_size, intermediate_size]",
            "shared_gate_proj_weight": "[intermediate_size, hidden_size]",
            "shared_up_proj_weight": "[intermediate_size, hidden_size]",
            "shared_down_proj_weight": "[hidden_size, intermediate_size]",
            "shared_expert_gate_weight": "[1, hidden_size]",
        },
        "rounding": [
            "MoE consumes the BF16 post-attention norm output as hidden input.",
            "Router projection consumes BF16 hidden/weights and stores f32 logits.",
            "Routed MoE oracle accumulates selected expert SwiGLU/down paths in f32 before BF16 output.",
            "Shared expert oracle mirrors the staged local path: BF16 gate/up GEMMs, BF16 silu*up, BF16 down GEMM, then sigmoid-gated add.",
            "Final residual/norm reads BF16 residual-after-attention and BF16 MoE/shared output, sums in f32, stores BF16 residual, and normalizes the f32 sum.",
        ],
    }
    return tensors, metadata


def _add_rows(
    lhs_bf16: tuple[int, ...],
    rhs_bf16: tuple[int, ...],
    cols: int,
) -> tuple[tuple[float, ...], tuple[int, ...], list[list[float]]]:
    out_f32: list[float] = []
    out_bf16: list[int] = []
    rows: list[list[float]] = []
    for row_idx in range(ROWS):
        row: list[float] = []
        for col in range(cols):
            idx = row_idx * cols + col
            value = _f32(
                bf16_bits_to_float32(lhs_bf16[idx])
                + bf16_bits_to_float32(rhs_bf16[idx])
            )
            out_f32.append(value)
            bits = float32_to_bf16_bits(value)
            out_bf16.append(bits)
            row.append(value)
        rows.append(row)
    return tuple(out_f32), tuple(out_bf16), rows


def _pack_values(dtype: str, values: tuple[int, ...] | tuple[float, ...]) -> bytes:
    if not values:
        return b""
    if dtype == "bf16":
        return struct.pack(f"<{len(values)}H", *values)
    if dtype == "i32":
        return struct.pack(f"<{len(values)}i", *values)
    if dtype == "f32":
        return struct.pack(f"<{len(values)}f", *values)
    raise ValueError(f"unsupported dtype {dtype!r}")


def _write_tensor(root: Path, tensor: TensorWrite) -> WrittenTensor:
    blob = _pack_values(tensor.dtype, tensor.values)
    if len(tensor.values) != tensor.element_count:
        raise ValueError(
            f"{tensor.name}: expected {tensor.element_count} values, got {len(tensor.values)}"
        )
    path = root / tensor.file_name
    path.write_bytes(blob)
    sha256 = hashlib.sha256(blob).hexdigest()
    return WrittenTensor(
        spec=TensorSpec(
            name=tensor.name,
            dtype=tensor.dtype,
            shape=tensor.shape,
            file=tensor.file_name,
            role=tensor.role,
            description=tensor.description,
            metadata={
                "op": tensor.op,
                "strides": _contiguous_strides(tensor.shape),
                "elements": tensor.element_count,
                "bytes": len(blob),
                "sha256": sha256,
            },
        ),
        sha256=sha256,
        byte_count=len(blob),
    )


def _write_weight_tensor(
    root: Path,
    name: str,
    kind: str,
    out_features: int,
    in_features: int,
    description: str,
) -> WrittenTensor:
    file_name = f"block_{name}.bf16"
    path = root / file_name
    digest = hashlib.sha256()
    byte_count = 0
    with path.open("wb") as f:
        for out_feature in range(out_features):
            row = array("H", [0]) * in_features
            for col, bits in _projection_terms(kind, out_feature, in_features):
                row[col] = bits
            if sys.byteorder != "little":
                row.byteswap()
            blob = row.tobytes()
            f.write(blob)
            digest.update(blob)
            byte_count += len(blob)
    element_count = out_features * in_features
    sha256 = digest.hexdigest()
    return WrittenTensor(
        spec=TensorSpec(
            name=name,
            dtype="bf16",
            shape=(out_features, in_features),
            file=file_name,
            role="input",
            description=description,
            metadata={
                "op": "projection_weight",
                "strides": [in_features, 1],
                "elements": element_count,
                "bytes": byte_count,
                "sha256": sha256,
                "sparse_terms_per_row": _projection_term_count(kind),
                "storage": "dense row-major BF16; deterministic sparse rows",
            },
        ),
        sha256=sha256,
        byte_count=byte_count,
    )


def _write_sparse_bf16_weight_tensor(
    root: Path,
    name: str,
    shape: tuple[int, ...],
    in_features: int,
    row_count: int,
    term_fn,
    description: str,
    metadata: dict[str, Any],
) -> WrittenTensor:
    file_name = f"block_{name}.bf16"
    path = root / file_name
    digest = hashlib.sha256()
    byte_count = 0
    with path.open("wb") as f:
        for row_idx in range(row_count):
            row = array("H", [0]) * in_features
            for col, bits in term_fn(row_idx):
                row[col] = bits
            if sys.byteorder != "little":
                row.byteswap()
            blob = row.tobytes()
            f.write(blob)
            digest.update(blob)
            byte_count += len(blob)
    element_count = row_count * in_features
    sha256 = digest.hexdigest()
    return WrittenTensor(
        spec=TensorSpec(
            name=name,
            dtype="bf16",
            shape=shape,
            file=file_name,
            role="input",
            description=description,
            metadata={
                **metadata,
                "elements": element_count,
                "bytes": byte_count,
                "sha256": sha256,
                "storage": "dense row-major BF16; deterministic sparse rows",
            },
        ),
        sha256=sha256,
        byte_count=byte_count,
    )


def _write_moe_gate_up_weight_tensor(root: Path) -> WrittenTensor:
    return _write_sparse_bf16_weight_tensor(
        root,
        "moe_gate_up_proj_weight",
        (MOE_NUM_EXPERTS, 2 * MOE_INTERMEDIATE, HIDDEN_SIZE),
        HIDDEN_SIZE,
        MOE_NUM_EXPERTS * 2 * MOE_INTERMEDIATE,
        lambda row_idx: _moe_gate_up_terms(
            row_idx // (2 * MOE_INTERMEDIATE),
            row_idx % (2 * MOE_INTERMEDIATE),
        ),
        "Dense row-major BF16 fused MoE gate/up projection weight.",
        {
            "op": "moe_gate_up_projection_weight",
            "strides": [2 * MOE_INTERMEDIATE * HIDDEN_SIZE, HIDDEN_SIZE, 1],
            "sparse_terms_per_row": MOE_GATE_UP_TERMS,
        },
    )


def _write_moe_down_weight_tensor(root: Path) -> WrittenTensor:
    file_name = "block_moe_down_proj_weight.bf16"
    path = root / file_name
    digest = hashlib.sha256()
    byte_count = 0
    with path.open("wb") as f:
        for expert in range(MOE_NUM_EXPERTS):
            for hidden in range(HIDDEN_SIZE):
                row = array(
                    "H",
                    [
                        float32_to_bf16_bits(
                            _moe_down_weight_value(expert, hidden, idx)
                        )
                        for idx in range(MOE_INTERMEDIATE)
                    ],
                )
                if sys.byteorder != "little":
                    row.byteswap()
                blob = row.tobytes()
                f.write(blob)
                digest.update(blob)
                byte_count += len(blob)
    element_count = MOE_NUM_EXPERTS * HIDDEN_SIZE * MOE_INTERMEDIATE
    sha256 = digest.hexdigest()
    return WrittenTensor(
        spec=TensorSpec(
            name="moe_down_proj_weight",
            dtype="bf16",
            shape=(MOE_NUM_EXPERTS, HIDDEN_SIZE, MOE_INTERMEDIATE),
            file=file_name,
            role="input",
            description="Dense row-major BF16 MoE down projection weight.",
            metadata={
                "op": "moe_down_projection_weight",
                "strides": [HIDDEN_SIZE * MOE_INTERMEDIATE, MOE_INTERMEDIATE, 1],
                "elements": element_count,
                "bytes": byte_count,
                "sha256": sha256,
            },
        ),
        sha256=sha256,
        byte_count=byte_count,
    )


def _write_shared_down_weight_tensor(root: Path) -> WrittenTensor:
    file_name = "block_moe_shared_down_proj_weight.bf16"
    path = root / file_name
    digest = hashlib.sha256()
    byte_count = 0
    with path.open("wb") as f:
        for hidden in range(HIDDEN_SIZE):
            row = array(
                "H",
                [
                    float32_to_bf16_bits(_shared_down_weight_value(hidden, idx))
                    for idx in range(MOE_INTERMEDIATE)
                ],
            )
            if sys.byteorder != "little":
                row.byteswap()
            blob = row.tobytes()
            f.write(blob)
            digest.update(blob)
            byte_count += len(blob)
    element_count = HIDDEN_SIZE * MOE_INTERMEDIATE
    sha256 = digest.hexdigest()
    return WrittenTensor(
        spec=TensorSpec(
            name="moe_shared_down_proj_weight",
            dtype="bf16",
            shape=(HIDDEN_SIZE, MOE_INTERMEDIATE),
            file=file_name,
            role="input",
            description="Dense row-major BF16 shared expert down projection weight.",
            metadata={
                "op": "shared_expert_down_projection_weight",
                "strides": [MOE_INTERMEDIATE, 1],
                "elements": element_count,
                "bytes": byte_count,
                "sha256": sha256,
            },
        ),
        sha256=sha256,
        byte_count=byte_count,
    )


def _moe_weight_specs() -> list[TensorSpec]:
    return [
        TensorSpec(
            name="moe_router_proj_weight",
            dtype="bf16",
            shape=(MOE_NUM_EXPERTS, HIDDEN_SIZE),
            file="block_moe_router_proj_weight.bf16",
            role="input",
            description="Dense row-major BF16 MoE router projection weight.",
        ),
        TensorSpec(
            name="moe_gate_up_proj_weight",
            dtype="bf16",
            shape=(MOE_NUM_EXPERTS, 2 * MOE_INTERMEDIATE, HIDDEN_SIZE),
            file="block_moe_gate_up_proj_weight.bf16",
            role="input",
            description="Dense row-major BF16 fused MoE gate/up projection weight.",
        ),
        TensorSpec(
            name="moe_down_proj_weight",
            dtype="bf16",
            shape=(MOE_NUM_EXPERTS, HIDDEN_SIZE, MOE_INTERMEDIATE),
            file="block_moe_down_proj_weight.bf16",
            role="input",
            description="Dense row-major BF16 MoE down projection weight.",
        ),
        TensorSpec(
            name="moe_shared_gate_proj_weight",
            dtype="bf16",
            shape=(MOE_INTERMEDIATE, HIDDEN_SIZE),
            file="block_moe_shared_gate_proj_weight.bf16",
            role="input",
            description="Dense row-major BF16 shared expert gate projection weight.",
        ),
        TensorSpec(
            name="moe_shared_up_proj_weight",
            dtype="bf16",
            shape=(MOE_INTERMEDIATE, HIDDEN_SIZE),
            file="block_moe_shared_up_proj_weight.bf16",
            role="input",
            description="Dense row-major BF16 shared expert up projection weight.",
        ),
        TensorSpec(
            name="moe_shared_down_proj_weight",
            dtype="bf16",
            shape=(HIDDEN_SIZE, MOE_INTERMEDIATE),
            file="block_moe_shared_down_proj_weight.bf16",
            role="input",
            description="Dense row-major BF16 shared expert down projection weight.",
        ),
        TensorSpec(
            name="moe_shared_expert_gate_weight",
            dtype="bf16",
            shape=(1, HIDDEN_SIZE),
            file="block_moe_shared_expert_gate_weight.bf16",
            role="input",
            description="Dense row-major BF16 scalar shared expert gate projection weight.",
        ),
    ]


def build_full_attention_block_tensors() -> tuple[list[TensorWrite], dict[str, Any]]:
    input_residual = _bf16_words(
        _input_value(row, col) for row in range(ROWS) for col in range(HIDDEN_SIZE)
    )
    input_rows = _bf16_to_f32_rows(input_residual, ROWS, HIDDEN_SIZE)
    attn_norm_weight = _bf16_words(
        _hidden_norm_weight_value(col) for col in range(HIDDEN_SIZE)
    )
    q_norm_weight = _bf16_words(_q_norm_weight_value(lane) for lane in range(HEAD_DIM))
    k_norm_weight = _bf16_words(_k_norm_weight_value(lane) for lane in range(HEAD_DIM))
    post_norm_weight = _bf16_words(
        _post_norm_weight_value(col) for col in range(HIDDEN_SIZE)
    )

    attn_norm_f32, attn_norm_bf16, attn_norm_rows = _gemma_rmsnorm_rows(
        input_rows,
        attn_norm_weight,
    )
    q_proj_f32, q_proj_bf16, q_proj_rows = _project_sparse(
        attn_norm_rows,
        "q_proj",
        Q_PROJ_OUT,
        HIDDEN_SIZE,
    )
    k_proj_f32, k_proj_bf16, k_proj_rows = _project_sparse(
        attn_norm_rows,
        "k_proj",
        KV_HIDDEN,
        HIDDEN_SIZE,
    )
    v_proj_f32, v_proj_bf16, v_proj_rows = _project_sparse(
        attn_norm_rows,
        "v_proj",
        KV_HIDDEN,
        HIDDEN_SIZE,
    )

    q_bf16, gate_bf16, q_rows, _gate_rows = _extract_q_gate(q_proj_rows, q_proj_bf16)
    q_heads = _reshape_heads(q_rows, Q_HEADS)
    k_heads = _reshape_heads(k_proj_rows, KV_HEADS)
    v_heads = _reshape_heads(v_proj_rows, KV_HEADS)
    q_norm_f32, q_norm_bf16, q_norm_heads = _gemma_rmsnorm_heads(q_heads, q_norm_weight)
    k_norm_f32, k_norm_bf16, k_norm_heads = _gemma_rmsnorm_heads(k_heads, k_norm_weight)
    q_rope_f32, q_rope_bf16, q_rope_heads = _apply_partial_rope(q_norm_heads)
    k_rope_f32, k_rope_bf16, k_rope_heads = _apply_partial_rope(k_norm_heads)
    raw_attn_f32, raw_attn_bf16, _raw_attn_heads = _causal_gqa_attention(
        q_rope_heads,
        k_rope_heads,
        v_heads,
    )
    gated_f32, gated_bf16, gated_rows = _apply_output_gate(raw_attn_bf16, gate_bf16)
    o_proj_f32, o_proj_bf16, _o_proj_rows = _project_sparse(
        gated_rows,
        "o_proj",
        HIDDEN_SIZE,
        Q_HIDDEN,
    )
    residual_f32, residual_bf16, residual_rows = _add_rows(
        o_proj_bf16,
        input_residual,
        HIDDEN_SIZE,
    )
    post_norm_f32, post_norm_bf16, post_norm_rows = _gemma_rmsnorm_rows(
        residual_rows,
        post_norm_weight,
    )
    moe_tensors, moe_metadata = _compute_one_layer_moe_tensors(
        post_norm_rows, residual_bf16
    )

    tensors = [
        TensorWrite(
            "input_residual",
            "bf16",
            (ROWS, HIDDEN_SIZE),
            input_residual,
            "input",
            "block_input",
            "BF16 residual entering a Qwen3.6 full-attention block.",
        ),
        TensorWrite(
            "positions",
            "i32",
            (ROWS,),
            POSITIONS,
            "input",
            "partial_rope",
            "Non-contiguous token positions for partial RoPE.",
        ),
        TensorWrite(
            "attn_norm_raw_weight",
            "bf16",
            (HIDDEN_SIZE,),
            attn_norm_weight,
            "input",
            "gemma_rmsnorm",
            "Raw attention input RMSNorm weight; effective multiplier is raw + 1.",
        ),
        TensorWrite(
            "q_norm_raw_weight",
            "bf16",
            (HEAD_DIM,),
            q_norm_weight,
            "input",
            "qk_gemma_rmsnorm",
            "Raw per-head q_norm weight; effective multiplier is raw + 1.",
        ),
        TensorWrite(
            "k_norm_raw_weight",
            "bf16",
            (HEAD_DIM,),
            k_norm_weight,
            "input",
            "qk_gemma_rmsnorm",
            "Raw per-head k_norm weight; effective multiplier is raw + 1.",
        ),
        TensorWrite(
            "post_attn_norm_raw_weight",
            "bf16",
            (HIDDEN_SIZE,),
            post_norm_weight,
            "input",
            "fused_add_gemma_rmsnorm",
            "Raw post-attention RMSNorm weight; effective multiplier is raw + 1.",
        ),
        moe_tensors[0],
        TensorWrite(
            "expected_attn_norm_output_f32",
            "f32",
            (ROWS, HIDDEN_SIZE),
            attn_norm_f32,
            "reference",
            "gemma_rmsnorm",
            "f32 attention input Gemma RMSNorm oracle before BF16 storage.",
        ),
        TensorWrite(
            "expected_attn_norm_output_bf16",
            "bf16",
            (ROWS, HIDDEN_SIZE),
            attn_norm_bf16,
            "expected",
            "gemma_rmsnorm",
            "BF16-stored attention input Gemma RMSNorm output.",
        ),
        TensorWrite(
            "expected_q_proj_output_f32",
            "f32",
            (ROWS, Q_HEADS, 2, HEAD_DIM),
            q_proj_f32,
            "reference",
            "q_projection",
            "f32 packed q projection oracle laid out per head as [q, output_gate].",
        ),
        TensorWrite(
            "expected_q_proj_output_bf16",
            "bf16",
            (ROWS, Q_HEADS, 2, HEAD_DIM),
            q_proj_bf16,
            "expected",
            "q_projection",
            "BF16 packed q projection output laid out per head as [q, output_gate].",
        ),
        TensorWrite(
            "expected_q_extracted_bf16",
            "bf16",
            (ROWS, Q_HEADS, HEAD_DIM),
            q_bf16,
            "expected",
            "packed_q_gate_extract",
            "Contiguous Q heads extracted from the packed q projection output.",
        ),
        TensorWrite(
            "expected_gate_extracted_bf16",
            "bf16",
            (ROWS, Q_HEADS, HEAD_DIM),
            gate_bf16,
            "expected",
            "packed_q_gate_extract",
            "Contiguous output-gate heads extracted from packed q projection output.",
        ),
        TensorWrite(
            "expected_k_proj_output_f32",
            "f32",
            (ROWS, KV_HEADS, HEAD_DIM),
            k_proj_f32,
            "reference",
            "k_projection",
            "f32 key projection oracle.",
        ),
        TensorWrite(
            "expected_k_proj_output_bf16",
            "bf16",
            (ROWS, KV_HEADS, HEAD_DIM),
            k_proj_bf16,
            "expected",
            "k_projection",
            "BF16 key projection output.",
        ),
        TensorWrite(
            "expected_v_proj_output_f32",
            "f32",
            (ROWS, KV_HEADS, HEAD_DIM),
            v_proj_f32,
            "reference",
            "v_projection",
            "f32 value projection oracle.",
        ),
        TensorWrite(
            "expected_v_proj_output_bf16",
            "bf16",
            (ROWS, KV_HEADS, HEAD_DIM),
            v_proj_bf16,
            "expected",
            "v_projection",
            "BF16 value projection output.",
        ),
        TensorWrite(
            "expected_q_norm_output_f32",
            "f32",
            (ROWS, Q_HEADS, HEAD_DIM),
            q_norm_f32,
            "reference",
            "qk_gemma_rmsnorm",
            "f32 per-head q Gemma RMSNorm oracle.",
        ),
        TensorWrite(
            "expected_q_norm_output_bf16",
            "bf16",
            (ROWS, Q_HEADS, HEAD_DIM),
            q_norm_bf16,
            "expected",
            "qk_gemma_rmsnorm",
            "BF16 per-head q Gemma RMSNorm output.",
        ),
        TensorWrite(
            "expected_k_norm_output_f32",
            "f32",
            (ROWS, KV_HEADS, HEAD_DIM),
            k_norm_f32,
            "reference",
            "qk_gemma_rmsnorm",
            "f32 per-head k Gemma RMSNorm oracle.",
        ),
        TensorWrite(
            "expected_k_norm_output_bf16",
            "bf16",
            (ROWS, KV_HEADS, HEAD_DIM),
            k_norm_bf16,
            "expected",
            "qk_gemma_rmsnorm",
            "BF16 per-head k Gemma RMSNorm output.",
        ),
        TensorWrite(
            "expected_q_rope_output_f32",
            "f32",
            (ROWS, Q_HEADS, HEAD_DIM),
            q_rope_f32,
            "reference",
            "partial_rope",
            "f32 q partial RoPE oracle using BF16 cos/sin cache semantics.",
        ),
        TensorWrite(
            "expected_q_rope_output_bf16",
            "bf16",
            (ROWS, Q_HEADS, HEAD_DIM),
            q_rope_bf16,
            "expected",
            "partial_rope",
            "BF16 q output after partial RoPE over lanes [0, 64).",
        ),
        TensorWrite(
            "expected_k_rope_output_f32",
            "f32",
            (ROWS, KV_HEADS, HEAD_DIM),
            k_rope_f32,
            "reference",
            "partial_rope",
            "f32 k partial RoPE oracle using BF16 cos/sin cache semantics.",
        ),
        TensorWrite(
            "expected_k_rope_output_bf16",
            "bf16",
            (ROWS, KV_HEADS, HEAD_DIM),
            k_rope_bf16,
            "expected",
            "partial_rope",
            "BF16 k output after partial RoPE over lanes [0, 64).",
        ),
        TensorWrite(
            "expected_raw_attention_output_f32",
            "f32",
            (ROWS, Q_HEADS, HEAD_DIM),
            raw_attn_f32,
            "reference",
            "causal_gqa_attention",
            "f32 raw causal GQA attention output before output-gate sigmoid.",
        ),
        TensorWrite(
            "expected_raw_attention_output_bf16",
            "bf16",
            (ROWS, Q_HEADS, HEAD_DIM),
            raw_attn_bf16,
            "expected",
            "causal_gqa_attention",
            "BF16 raw causal GQA attention output before output-gate sigmoid.",
        ),
        TensorWrite(
            "expected_gated_attention_output_f32",
            "f32",
            (ROWS, Q_HEADS, HEAD_DIM),
            gated_f32,
            "reference",
            "output_gate",
            "f32 raw_attention * sigmoid(output_gate) oracle.",
        ),
        TensorWrite(
            "expected_gated_attention_output_bf16",
            "bf16",
            (ROWS, Q_HEADS, HEAD_DIM),
            gated_bf16,
            "expected",
            "output_gate",
            "BF16 gated attention output.",
        ),
        TensorWrite(
            "expected_o_proj_output_f32",
            "f32",
            (ROWS, HIDDEN_SIZE),
            o_proj_f32,
            "reference",
            "o_projection",
            "f32 output projection oracle.",
        ),
        TensorWrite(
            "expected_o_proj_output_bf16",
            "bf16",
            (ROWS, HIDDEN_SIZE),
            o_proj_bf16,
            "expected",
            "o_projection",
            "BF16 output projection result.",
        ),
        TensorWrite(
            "expected_residual_after_attention_f32",
            "f32",
            (ROWS, HIDDEN_SIZE),
            residual_f32,
            "reference",
            "attention_residual_add",
            "f32 residual + BF16 output projection sum.",
        ),
        TensorWrite(
            "expected_residual_after_attention_bf16",
            "bf16",
            (ROWS, HIDDEN_SIZE),
            residual_bf16,
            "expected",
            "attention_residual_add",
            "BF16-rounded residual after attention add.",
        ),
        TensorWrite(
            "expected_post_attn_norm_output_f32",
            "f32",
            (ROWS, HIDDEN_SIZE),
            post_norm_f32,
            "reference",
            "fused_add_gemma_rmsnorm",
            "f32 post-attention Gemma RMSNorm oracle.",
        ),
        TensorWrite(
            "expected_post_attn_norm_output_bf16",
            "bf16",
            (ROWS, HIDDEN_SIZE),
            post_norm_bf16,
            "expected",
            "fused_add_gemma_rmsnorm",
            "BF16 post-attention Gemma RMSNorm output.",
        ),
        *moe_tensors[1:],
    ]
    metadata = {
        "case": "full_attention_block_v1",
        "dimensions": {
            "rows": ROWS,
            "hidden_size": HIDDEN_SIZE,
            "q_heads": Q_HEADS,
            "kv_heads": KV_HEADS,
            "group_size": GROUP_SIZE,
            "head_dim": HEAD_DIM,
            "rotary_dim": ROTARY_DIM,
            "q_hidden": Q_HIDDEN,
            "kv_hidden": KV_HIDDEN,
            "q_proj_out": Q_PROJ_OUT,
            "positions": list(POSITIONS),
        },
        "params": {
            "rms_eps": RMS_EPS,
            "rope_theta": ROPE_THETA,
            "rope_scale": ROPE_SCALE,
            "attention_scale": ATTENTION_SCALE,
            "output_gate_activation": "sigmoid",
        },
        "rounding": [
            "All input and weight tensors are stored as raw little-endian BF16 words.",
            "Decoder and q/k Gemma RMSNorm use effective_weight = bf16(raw_weight) + 1.0.",
            "Projection GEMMs consume BF16 inputs and BF16 weights, accumulate in f32, and store BF16.",
            "RoPE reads BF16-stored q/k norm outputs and BF16 cos/sin cache values as f32.",
            "Causal GQA attention reads BF16 q/k/v as f32 and uses scale 1/sqrt(256).",
            "Output gate and post-attention residual/norm read BF16-stored prior outputs.",
        ],
        "source_semantics": [
            "vLLM/FlashInfer Gemma RMSNorm raw_weight + 1",
            "Qwen3.6 q_proj packed per head as [q, output_gate]",
            "vLLM NeoX partial RoPE over rotary_dim=64",
            "FlashInfer POS_ENCODING_NONE causal GQA attention after explicit RoPE",
            "Qwen3.6 sigmoid full-attention output gate",
        ],
        "one_layer_moe": moe_metadata,
    }
    return tensors, metadata


def write_full_attention_block_artifact(root: str | Path = DEFAULT_OUTPUT) -> Path:
    root_path = Path(root)
    root_path.mkdir(parents=True, exist_ok=True)

    tensors, metadata = build_full_attention_block_tensors()
    written: list[WrittenTensor] = []
    for tensor in tensors[:INLINE_INPUT_TENSORS]:
        written.append(_write_tensor(root_path, tensor))
    written.extend(
        [
            _write_weight_tensor(
                root_path,
                "q_proj_weight",
                "q_proj",
                Q_PROJ_OUT,
                HIDDEN_SIZE,
                "Dense row-major BF16 q projection weight [8192, 2048].",
            ),
            _write_weight_tensor(
                root_path,
                "k_proj_weight",
                "k_proj",
                KV_HIDDEN,
                HIDDEN_SIZE,
                "Dense row-major BF16 k projection weight [512, 2048].",
            ),
            _write_weight_tensor(
                root_path,
                "v_proj_weight",
                "v_proj",
                KV_HIDDEN,
                HIDDEN_SIZE,
                "Dense row-major BF16 v projection weight [512, 2048].",
            ),
            _write_weight_tensor(
                root_path,
                "o_proj_weight",
                "o_proj",
                HIDDEN_SIZE,
                Q_HIDDEN,
                "Dense row-major BF16 output projection weight [2048, 4096].",
            ),
        ]
    )
    written.extend(
        [
            _write_sparse_bf16_weight_tensor(
                root_path,
                "moe_router_proj_weight",
                (MOE_NUM_EXPERTS, HIDDEN_SIZE),
                HIDDEN_SIZE,
                MOE_NUM_EXPERTS,
                _moe_router_terms,
                "Dense row-major BF16 MoE router projection weight.",
                {
                    "op": "moe_router_projection_weight",
                    "strides": [HIDDEN_SIZE, 1],
                    "sparse_terms_per_row": MOE_ROUTER_TERMS,
                },
            ),
            _write_moe_gate_up_weight_tensor(root_path),
            _write_moe_down_weight_tensor(root_path),
            _write_sparse_bf16_weight_tensor(
                root_path,
                "moe_shared_gate_proj_weight",
                (MOE_INTERMEDIATE, HIDDEN_SIZE),
                HIDDEN_SIZE,
                MOE_INTERMEDIATE,
                lambda row_idx: _shared_proj_terms("gate", row_idx),
                "Dense row-major BF16 shared expert gate projection weight.",
                {
                    "op": "shared_expert_gate_projection_weight",
                    "strides": [HIDDEN_SIZE, 1],
                    "sparse_terms_per_row": MOE_SHARED_TERMS,
                },
            ),
            _write_sparse_bf16_weight_tensor(
                root_path,
                "moe_shared_up_proj_weight",
                (MOE_INTERMEDIATE, HIDDEN_SIZE),
                HIDDEN_SIZE,
                MOE_INTERMEDIATE,
                lambda row_idx: _shared_proj_terms("up", row_idx),
                "Dense row-major BF16 shared expert up projection weight.",
                {
                    "op": "shared_expert_up_projection_weight",
                    "strides": [HIDDEN_SIZE, 1],
                    "sparse_terms_per_row": MOE_SHARED_TERMS,
                },
            ),
            _write_shared_down_weight_tensor(root_path),
            _write_sparse_bf16_weight_tensor(
                root_path,
                "moe_shared_expert_gate_weight",
                (1, HIDDEN_SIZE),
                HIDDEN_SIZE,
                1,
                lambda row_idx: _shared_proj_terms("shared_gate", row_idx),
                "Dense row-major BF16 scalar shared expert gate projection weight.",
                {
                    "op": "shared_expert_gate_weight",
                    "strides": [HIDDEN_SIZE, 1],
                    "sparse_terms_per_row": MOE_SHARED_TERMS,
                },
            ),
        ]
    )
    for tensor in tensors[INLINE_INPUT_TENSORS:]:
        written.append(_write_tensor(root_path, tensor))

    manifest = VectorManifest(
        name="qwen36_full_attention_block",
        groups=("qwen36_semantics", "attention", "full_attention_block"),
        description=(
            "Deterministic Qwen3.6 full-attention block correctness vector with "
            "real hidden/head dimensions and six token rows."
        ),
        metadata=metadata,
        tensors=tuple(item.spec for item in written),
    )
    payload = json.dumps(manifest.to_json(), indent=2, sort_keys=True)
    manifest_path = root_path / MANIFEST_FILE
    manifest_path.write_text(f"{payload}\n", encoding="utf-8")
    return manifest_path


def build_full_attention_block_artifact() -> tuple[VectorManifest, dict[str, Any]]:
    tensors, metadata = build_full_attention_block_tensors()
    tensor_specs = [
        TensorSpec(
            name=tensor.name,
            dtype=tensor.dtype,
            shape=tensor.shape,
            file=tensor.file_name,
            role=tensor.role,
            description=tensor.description,
            metadata={
                "op": tensor.op,
                "strides": _contiguous_strides(tensor.shape),
                "elements": tensor.element_count,
                "bytes": tensor.element_count
                * {
                    "bf16": 2,
                    "i32": 4,
                    "f32": 4,
                }[tensor.dtype],
            },
        )
        for tensor in tensors
    ]
    tensor_specs[INLINE_INPUT_TENSORS:INLINE_INPUT_TENSORS] = [
        TensorSpec(
            name="q_proj_weight",
            dtype="bf16",
            shape=(Q_PROJ_OUT, HIDDEN_SIZE),
            file="block_q_proj_weight.bf16",
            role="input",
            description="Dense row-major BF16 q projection weight [8192, 2048].",
        ),
        TensorSpec(
            name="k_proj_weight",
            dtype="bf16",
            shape=(KV_HIDDEN, HIDDEN_SIZE),
            file="block_k_proj_weight.bf16",
            role="input",
            description="Dense row-major BF16 k projection weight [512, 2048].",
        ),
        TensorSpec(
            name="v_proj_weight",
            dtype="bf16",
            shape=(KV_HIDDEN, HIDDEN_SIZE),
            file="block_v_proj_weight.bf16",
            role="input",
            description="Dense row-major BF16 v projection weight [512, 2048].",
        ),
        TensorSpec(
            name="o_proj_weight",
            dtype="bf16",
            shape=(HIDDEN_SIZE, Q_HIDDEN),
            file="block_o_proj_weight.bf16",
            role="input",
            description="Dense row-major BF16 output projection weight [2048, 4096].",
        ),
        *_moe_weight_specs(),
    ]
    return (
        VectorManifest(
            name="qwen36_full_attention_block",
            groups=("qwen36_semantics", "attention", "full_attention_block"),
            description=(
                "Deterministic Qwen3.6 full-attention block correctness vector with "
                "real hidden/head dimensions and six token rows."
            ),
            metadata=metadata,
            tensors=tuple(tensor_specs),
        ),
        metadata,
    )
