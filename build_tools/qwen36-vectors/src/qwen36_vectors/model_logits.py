"""Deterministic Qwen3.6 ModelRunner logits vector.

The vector is intentionally loader-free: it writes the raw BF16 weight tensors
needed to build a private one-layer `QwenWeights`, plus f32 logits and greedy
top-token oracles for a short prefill followed by one decode step.
"""

from __future__ import annotations

from array import array
from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
import struct
import sys
from typing import Any

from .io import bf16_bits_to_float32, float32_to_bf16_bits
from .paths import DEFAULT_VECTOR_ROOT
from .schema import MANIFEST_FILE, TensorSpec, VectorManifest


PROMPT_TOKENS = (2, 6, 9, 12)
PROMPT_LEN = len(PROMPT_TOKENS)
TOTAL_ROWS = PROMPT_LEN + 1
HIDDEN_SIZE = 2048
Q_HEADS = 16
KV_HEADS = 2
HEAD_DIM = 256
ROTARY_DIM = 64
Q_HIDDEN = Q_HEADS * HEAD_DIM
KV_HIDDEN = KV_HEADS * HEAD_DIM
Q_PROJ_OUT = 2 * Q_HIDDEN
GROUP_SIZE = Q_HEADS // KV_HEADS
VOCAB_SIZE = 16
INTERMEDIATE_SIZE = 8
MOE_NUM_EXPERTS = 256
MOE_TOP_K = 8
MOE_INTERMEDIATE = INTERMEDIATE_SIZE
MOE_ROUTED_SCALING_FACTOR = 1.0
MOE_ROUTER_TERMS = 5
MOE_GATE_UP_TERMS = 5
MOE_SHARED_TERMS = 5
PAGE_SIZE = 4
MAX_PAGES = 2
RMS_EPS = 1.0e-6
ROPE_THETA = 10_000.0
ROPE_SCALE = 1.0
ATTENTION_SCALE = 1.0 / math.sqrt(HEAD_DIM)
DEFAULT_OUTPUT = DEFAULT_VECTOR_ROOT / "model_logits"


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
        return f"model_{self.name}.{self.dtype}"

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


def _bf16_words(values: Any) -> tuple[int, ...]:
    return tuple(float32_to_bf16_bits(float(value)) for value in values)


def _bf16_to_f32_rows(values: tuple[int, ...], rows: int, cols: int) -> list[list[float]]:
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


def _embedding_value(token: int, col: int) -> float:
    centered = ((token * 97 + col * 13 + (col // 29) * 7) % 127) - 63
    band = (((col // 128) % 9) - 4) * 0.0107421875
    token_bias = (token - 7.5) * 0.0048828125
    marker = 0.0546875 if col == ((token * 113 + 17) % HIDDEN_SIZE) else 0.0
    return centered * 0.0068359375 + band + token_bias + marker


def _attn_norm_weight_value(col: int) -> float:
    base = ((col * 7 + (col // 31) * 5) % 43 - 21) * 0.00390625
    return base + (0.009765625 if col % 5 == 0 else -0.0048828125)


def _final_norm_weight_value(col: int) -> float:
    base = ((col * 11 + (col // 23) * 3) % 47 - 23) * 0.00341796875
    return base + (-0.0078125 if col % 7 == 0 else 0.005859375)


def _mlp_norm_weight_value(col: int) -> float:
    base = ((col * 13 + (col // 19) * 5) % 53 - 26) * 0.0029296875
    return base + (0.0107421875 if col % 11 == 0 else -0.0048828125)


def _q_norm_weight_value(lane: int) -> float:
    base = ((lane * 7 + 3) % 31 - 15) * 0.0078125
    return base + (0.013671875 if lane % 2 == 0 else -0.009765625)


def _k_norm_weight_value(lane: int) -> float:
    base = ((lane * 5 + 11) % 29 - 14) * 0.00830078125
    return base + (-0.01171875 if lane % 3 == 0 else 0.015625)


def _projection_profile(kind: str, out_feature: int) -> tuple[int, float, int]:
    if kind == "q_proj":
        lane = out_feature % (2 * HEAD_DIM)
        return 37, 0.19 if lane >= HEAD_DIM else 0.030, 4
    if kind == "k_proj":
        return 53, 0.030, 4
    if kind == "v_proj":
        return 71, 0.082, 4
    if kind == "o_proj":
        return 89, 0.18, 5
    if kind == "lm_head":
        return 109, 1.60, 9
    raise ValueError(f"unknown projection kind {kind!r}")


def _projection_terms(
    kind: str,
    out_feature: int,
    in_features: int,
) -> tuple[tuple[int, int], ...]:
    seed, scale, term_count = _projection_profile(kind, out_feature)
    terms: dict[int, float] = {}
    for term in range(term_count):
        col = (
            out_feature * (37 + 10 * term)
            + seed * (term + 1)
            + (out_feature // 11) * (term + 5)
            + term * term * 17
        ) % in_features
        centered = ((out_feature * (19 + 4 * term) + seed + term * 23) % 41) - 20
        sign = -1.0 if ((out_feature >> (term % 6)) + seed + term) & 1 else 1.0
        magnitude = 0.50 + abs(centered) / 44.0
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
        centered = ((owner * (17 + 2 * term) + row * (23 + term) + seed + term * 29) % 37) - 18
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
    value = sign * (0.016 + abs(centered) * 0.00061) + (0.0011 if hidden % 13 == 0 else -0.0007)
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
            bits = float32_to_bf16_bits(y)
            out_bf16.append(bits)
            out_row.append(bf16_bits_to_float32(bits))
        out_rows.append(out_row)
    return tuple(out_f32), tuple(out_bf16), out_rows


def _project_sparse_bf16(
    x_rows: list[list[float]],
    kind: str,
    out_features: int,
    in_features: int,
) -> tuple[tuple[float, ...], tuple[int, ...], list[list[float]]]:
    decoded_terms = [
        tuple(
            (col, bf16_bits_to_float32(bits))
            for col, bits in _projection_terms(kind, out, in_features)
        )
        for out in range(out_features)
    ]
    out_f32: list[float] = []
    out_bf16: list[int] = []
    out_rows: list[list[float]] = []
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


def _project_sparse_f32(
    x_rows: list[list[float]],
    kind: str,
    out_features: int,
    in_features: int,
) -> tuple[tuple[float, ...], list[list[float]]]:
    decoded_terms = [
        tuple(
            (col, bf16_bits_to_float32(bits))
            for col, bits in _projection_terms(kind, out, in_features)
        )
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
    rows = len(q_proj_rows)
    q: list[int] = []
    gate: list[int] = []
    q_rows: list[list[float]] = []
    gate_rows: list[list[float]] = []
    for row in range(rows):
        q_row: list[float] = []
        gate_row: list[float] = []
        for head in range(Q_HEADS):
            base = row * Q_PROJ_OUT + head * 2 * HEAD_DIM
            q.extend(q_proj_bf16[base : base + HEAD_DIM])
            gate.extend(q_proj_bf16[base + HEAD_DIM : base + 2 * HEAD_DIM])
            q_row.extend(q_proj_rows[row][head * 2 * HEAD_DIM : head * 2 * HEAD_DIM + HEAD_DIM])
            gate_row.extend(
                q_proj_rows[row][head * 2 * HEAD_DIM + HEAD_DIM : (head + 1) * 2 * HEAD_DIM]
            )
        q_rows.append(q_row)
        gate_rows.append(gate_row)
    return tuple(q), tuple(gate), q_rows, gate_rows


def _reshape_heads(rows: list[list[float]], heads: int) -> list[list[list[float]]]:
    return [
        [row[head * HEAD_DIM : (head + 1) * HEAD_DIM] for head in range(heads)]
        for row in rows
    ]


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
    positions: tuple[int, ...],
) -> tuple[tuple[float, ...], tuple[int, ...], list[list[list[float]]]]:
    half = ROTARY_DIM // 2
    out_rows: list[list[list[float]]] = []
    out_f32: list[float] = []
    out_bf16: list[int] = []
    for row_idx, row in enumerate(heads):
        cos_values, sin_values = _cos_sin_for_position(positions[row_idx])
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
    rows = len(q)
    out_rows: list[list[list[float]]] = []
    out_f32: list[float] = []
    out_bf16: list[int] = []
    for row_idx in range(rows):
        out_row: list[list[float]] = []
        for q_head in range(Q_HEADS):
            kv_head = q_head // GROUP_SIZE
            scores: list[float] = []
            for key_row in range(row_idx + 1):
                dot = 0.0
                for lane in range(HEAD_DIM):
                    dot = _f32(dot + _f32(q[row_idx][q_head][lane] * k[key_row][kv_head][lane]))
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
    rows: int,
) -> tuple[tuple[float, ...], tuple[int, ...], list[list[float]]]:
    out_f32: list[float] = []
    out_bf16: list[int] = []
    out_rows: list[list[float]] = []
    for row_idx in range(rows):
        row: list[float] = []
        for col in range(Q_HIDDEN):
            idx = row_idx * Q_HIDDEN + col
            value = _f32(
                bf16_bits_to_float32(raw_attention_bf16[idx])
                * _sigmoid(bf16_bits_to_float32(gate_bf16[idx]))
            )
            out_f32.append(value)
            bits = float32_to_bf16_bits(value)
            out_bf16.append(bits)
            row.append(bf16_bits_to_float32(bits))
        out_rows.append(row)
    return tuple(out_f32), tuple(out_bf16), out_rows


def _add_rows(
    lhs_bf16: tuple[int, ...],
    rhs_bf16: tuple[int, ...],
    rows: int,
    cols: int,
) -> tuple[tuple[float, ...], tuple[int, ...], list[list[float]]]:
    out_f32: list[float] = []
    out_bf16: list[int] = []
    out_rows: list[list[float]] = []
    for row_idx in range(rows):
        row: list[float] = []
        for col in range(cols):
            idx = row_idx * cols + col
            value = _f32(bf16_bits_to_float32(lhs_bf16[idx]) + bf16_bits_to_float32(rhs_bf16[idx]))
            out_f32.append(value)
            bits = float32_to_bf16_bits(value)
            out_bf16.append(bits)
            row.append(value)
        out_rows.append(row)
    return tuple(out_f32), tuple(out_bf16), out_rows


def _route_moe_topk(
    logits: list[list[float]],
) -> tuple[list[list[int]], list[list[float]], list[list[float]]]:
    all_ids: list[list[int]] = []
    all_unrenorm: list[list[float]] = []
    all_weights: list[list[float]] = []
    for row in logits:
        best = sorted(range(MOE_NUM_EXPERTS), key=lambda expert: (-row[expert], expert))[:MOE_TOP_K]
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
    _gate_f32, gate_bf16, _gate_rows = _project_sparse_terms_bf16(
        hidden_rows,
        MOE_INTERMEDIATE,
        lambda out: _shared_proj_terms("gate", out),
    )
    _up_f32, up_bf16, _up_rows = _project_sparse_terms_bf16(
        hidden_rows,
        MOE_INTERMEDIATE,
        lambda out: _shared_proj_terms("up", out),
    )

    activated_rows: list[list[float]] = []
    for row_idx in range(len(hidden_rows)):
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
    return tuple(out_f32), tuple(out_bf16), out_rows, shared_gate_logits


def _combine_moe_and_shared(
    routed_bf16: tuple[int, ...],
    shared_bf16: tuple[int, ...],
    shared_gate_logits: tuple[float, ...],
    rows: int,
) -> tuple[tuple[float, ...], tuple[int, ...], list[list[float]]]:
    out_f32: list[float] = []
    out_bf16: list[int] = []
    out_rows: list[list[float]] = []
    for row_idx in range(rows):
        gate = _f32(_sigmoid(shared_gate_logits[row_idx]))
        row: list[float] = []
        for hidden in range(HIDDEN_SIZE):
            idx = row_idx * HIDDEN_SIZE + hidden
            value = _f32(
                bf16_bits_to_float32(routed_bf16[idx])
                + _f32(gate * bf16_bits_to_float32(shared_bf16[idx]))
            )
            out_f32.append(value)
            bits = float32_to_bf16_bits(value)
            out_bf16.append(bits)
            row.append(bf16_bits_to_float32(bits))
        out_rows.append(row)
    return tuple(out_f32), tuple(out_bf16), out_rows


def _compute_moe_shared_output(
    hidden_rows: list[list[float]],
) -> tuple[tuple[float, ...], tuple[int, ...], list[list[float]]]:
    _router_logits_f32, _router_logits_bf16, router_logits_rows = _project_sparse_terms_bf16(
        hidden_rows,
        MOE_NUM_EXPERTS,
        _moe_router_terms,
    )
    topk_ids, _topk_unrenorm, topk_weights = _route_moe_topk(router_logits_rows)
    _routed_f32, routed_bf16, _routed_rows = _compute_routed_moe(
        hidden_rows,
        topk_ids,
        topk_weights,
    )
    _shared_f32, shared_bf16, _shared_rows, shared_gate_logits = _compute_shared_expert(
        hidden_rows,
    )
    return _combine_moe_and_shared(
        routed_bf16,
        shared_bf16,
        shared_gate_logits,
        len(hidden_rows),
    )


def _argmax_rows(logits: list[list[float]]) -> tuple[tuple[int, ...], tuple[float, ...]]:
    ids: list[int] = []
    margins: list[float] = []
    for row in logits:
        ranked = sorted(range(len(row)), key=lambda idx: (-row[idx], idx))
        ids.append(ranked[0])
        margins.append(_f32(row[ranked[0]] - row[ranked[1]]))
    return tuple(ids), tuple(margins)


def _compute_logits_for_tokens(
    tokens: tuple[int, ...],
    *,
    token_embedding: tuple[int, ...],
    attn_norm_weight: tuple[int, ...],
    q_norm_weight: tuple[int, ...],
    k_norm_weight: tuple[int, ...],
    mlp_norm_weight: tuple[int, ...],
    final_norm_weight: tuple[int, ...],
) -> tuple[tuple[float, ...], list[list[float]]]:
    rows = len(tokens)
    positions = tuple(range(rows))
    residual_bf16 = tuple(
        token_embedding[token * HIDDEN_SIZE + col]
        for token in tokens
        for col in range(HIDDEN_SIZE)
    )
    residual_rows = _bf16_to_f32_rows(residual_bf16, rows, HIDDEN_SIZE)
    _attn_norm_f32, _attn_norm_bf16, attn_norm_rows = _gemma_rmsnorm_rows(
        residual_rows,
        attn_norm_weight,
    )

    _q_proj_f32, q_proj_bf16, q_proj_rows = _project_sparse_bf16(
        attn_norm_rows,
        "q_proj",
        Q_PROJ_OUT,
        HIDDEN_SIZE,
    )
    _k_proj_f32, _k_proj_bf16, k_proj_rows = _project_sparse_bf16(
        attn_norm_rows,
        "k_proj",
        KV_HIDDEN,
        HIDDEN_SIZE,
    )
    _v_proj_f32, _v_proj_bf16, v_proj_rows = _project_sparse_bf16(
        attn_norm_rows,
        "v_proj",
        KV_HIDDEN,
        HIDDEN_SIZE,
    )
    q_bf16, gate_bf16, q_rows, _gate_rows = _extract_q_gate(q_proj_rows, q_proj_bf16)
    q_heads = _reshape_heads(q_rows, Q_HEADS)
    k_heads = _reshape_heads(k_proj_rows, KV_HEADS)
    v_heads = _reshape_heads(v_proj_rows, KV_HEADS)
    _q_norm_f32, _q_norm_bf16, q_norm_heads = _gemma_rmsnorm_heads(q_heads, q_norm_weight)
    _k_norm_f32, _k_norm_bf16, k_norm_heads = _gemma_rmsnorm_heads(k_heads, k_norm_weight)
    _ = q_bf16
    _q_rope_f32, _q_rope_bf16, q_rope_heads = _apply_partial_rope(q_norm_heads, positions)
    _k_rope_f32, _k_rope_bf16, k_rope_heads = _apply_partial_rope(k_norm_heads, positions)
    _raw_attn_f32, raw_attn_bf16, _raw_attn_heads = _causal_gqa_attention(
        q_rope_heads,
        k_rope_heads,
        v_heads,
    )
    _gated_f32, gated_bf16, gated_rows = _apply_output_gate(raw_attn_bf16, gate_bf16, rows)
    _o_proj_f32, o_proj_bf16, _o_proj_rows = _project_sparse_bf16(
        gated_rows,
        "o_proj",
        HIDDEN_SIZE,
        Q_HIDDEN,
    )
    _residual_f32, _residual_bf16, residual_after_attn_rows = _add_rows(
        o_proj_bf16,
        residual_bf16,
        rows,
        HIDDEN_SIZE,
    )
    _post_norm_f32, _post_norm_bf16, post_norm_rows = _gemma_rmsnorm_rows(
        residual_after_attn_rows,
        mlp_norm_weight,
    )
    _moe_f32, moe_bf16, _moe_rows = _compute_moe_shared_output(post_norm_rows)
    _residual_after_mlp_f32, _residual_after_mlp_bf16, residual_after_mlp_rows = _add_rows(
        moe_bf16,
        _residual_bf16,
        rows,
        HIDDEN_SIZE,
    )
    _final_norm_f32, _final_norm_bf16, final_norm_rows = _gemma_rmsnorm_rows(
        residual_after_mlp_rows,
        final_norm_weight,
    )
    logits_f32, logits_rows = _project_sparse_f32(
        final_norm_rows,
        "lm_head",
        VOCAB_SIZE,
        HIDDEN_SIZE,
    )
    return logits_f32, logits_rows


def build_model_logits_tensors() -> tuple[list[TensorWrite], dict[str, Any]]:
    token_embedding = _bf16_words(
        _embedding_value(token, col)
        for token in range(VOCAB_SIZE)
        for col in range(HIDDEN_SIZE)
    )
    attn_norm_weight = _bf16_words(_attn_norm_weight_value(col) for col in range(HIDDEN_SIZE))
    q_norm_weight = _bf16_words(_q_norm_weight_value(lane) for lane in range(HEAD_DIM))
    k_norm_weight = _bf16_words(_k_norm_weight_value(lane) for lane in range(HEAD_DIM))
    final_norm_weight = _bf16_words(_final_norm_weight_value(col) for col in range(HIDDEN_SIZE))
    mlp_norm_weight = _bf16_words(_mlp_norm_weight_value(col) for col in range(HIDDEN_SIZE))

    prefill_logits, prefill_rows = _compute_logits_for_tokens(
        PROMPT_TOKENS,
        token_embedding=token_embedding,
        attn_norm_weight=attn_norm_weight,
        q_norm_weight=q_norm_weight,
        k_norm_weight=k_norm_weight,
        mlp_norm_weight=mlp_norm_weight,
        final_norm_weight=final_norm_weight,
    )
    prefill_top_ids, prefill_margins = _argmax_rows(prefill_rows)
    decode_input_token = prefill_top_ids[-1]
    full_tokens = (*PROMPT_TOKENS, decode_input_token)
    full_logits, full_rows = _compute_logits_for_tokens(
        full_tokens,
        token_embedding=token_embedding,
        attn_norm_weight=attn_norm_weight,
        q_norm_weight=q_norm_weight,
        k_norm_weight=k_norm_weight,
        mlp_norm_weight=mlp_norm_weight,
        final_norm_weight=final_norm_weight,
    )
    decode_logits = tuple(full_logits[PROMPT_LEN * VOCAB_SIZE : TOTAL_ROWS * VOCAB_SIZE])
    decode_top_ids, decode_margins = _argmax_rows([full_rows[-1]])

    tensors = [
        TensorWrite(
            "token_embedding_weight",
            "bf16",
            (VOCAB_SIZE, HIDDEN_SIZE),
            token_embedding,
            "input",
            "embedding",
            "Dense row-major BF16 token embedding table for the tiny public vocab.",
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
            "mlp_norm_raw_weight",
            "bf16",
            (HIDDEN_SIZE,),
            mlp_norm_weight,
            "input",
            "fused_add_gemma_rmsnorm",
            "Raw post-attention MLP RMSNorm weight before the MoE/shared-expert branch.",
        ),
        TensorWrite(
            "final_norm_raw_weight",
            "bf16",
            (HIDDEN_SIZE,),
            final_norm_weight,
            "input",
            "fused_add_gemma_rmsnorm",
            "Raw final RMSNorm weight before the lm head; effective multiplier is raw + 1.",
        ),
        TensorWrite(
            "prompt_tokens",
            "i32",
            (PROMPT_LEN,),
            PROMPT_TOKENS,
            "input",
            "runner_prefill",
            "Prompt tokens used for the prefill batch.",
        ),
        TensorWrite(
            "decode_input_token",
            "i32",
            (1,),
            (decode_input_token,),
            "expected",
            "greedy_argmax",
            "The greedy token sampled from the final prefill row and fed to decode.",
        ),
        TensorWrite(
            "expected_prefill_logits_f32",
            "f32",
            (PROMPT_LEN, VOCAB_SIZE),
            prefill_logits,
            "expected",
            "model_runner_prefill_logits",
            "f32 logits from the prompt prefill rows before greedy sampling.",
        ),
        TensorWrite(
            "expected_decode_logits_f32",
            "f32",
            (1, VOCAB_SIZE),
            decode_logits,
            "expected",
            "model_runner_decode_logits",
            "f32 logits from one decode step at position 4, crossing the first page boundary.",
        ),
        TensorWrite(
            "expected_prefill_top_ids",
            "i32",
            (PROMPT_LEN,),
            prefill_top_ids,
            "expected",
            "greedy_argmax",
            "Greedy top token for each prefill logits row.",
        ),
        TensorWrite(
            "expected_decode_top_ids",
            "i32",
            (1,),
            decode_top_ids,
            "expected",
            "greedy_argmax",
            "Greedy top token for the decode logits row.",
        ),
        TensorWrite(
            "expected_prefill_top_margin_f32",
            "f32",
            (PROMPT_LEN,),
            prefill_margins,
            "reference",
            "greedy_argmax",
            "Top-1 minus top-2 logit margins for the prefill rows.",
        ),
        TensorWrite(
            "expected_decode_top_margin_f32",
            "f32",
            (1,),
            decode_margins,
            "reference",
            "greedy_argmax",
            "Top-1 minus top-2 logit margin for the decode row.",
        ),
    ]
    metadata = {
        "case": "model_logits_prefill4_decode1_v1",
        "dimensions": {
            "num_layers": 1,
            "hidden_size": HIDDEN_SIZE,
            "q_heads": Q_HEADS,
            "kv_heads": KV_HEADS,
            "group_size": GROUP_SIZE,
            "head_dim": HEAD_DIM,
            "rotary_dim": ROTARY_DIM,
            "q_hidden": Q_HIDDEN,
            "kv_hidden": KV_HIDDEN,
            "q_proj_out": Q_PROJ_OUT,
            "vocab_size": VOCAB_SIZE,
            "intermediate_size": INTERMEDIATE_SIZE,
            "moe_num_experts": MOE_NUM_EXPERTS,
            "moe_top_k": MOE_TOP_K,
            "moe_intermediate_size": MOE_INTERMEDIATE,
            "shared_expert_intermediate_size": MOE_INTERMEDIATE,
            "prompt_len": PROMPT_LEN,
            "decode_len": 1,
            "page_size": PAGE_SIZE,
            "max_pages": MAX_PAGES,
        },
        "tokens": {
            "prompt": list(PROMPT_TOKENS),
            "decode_input": decode_input_token,
            "prefill_top_ids": list(prefill_top_ids),
            "decode_top_ids": list(decode_top_ids),
        },
        "params": {
            "rms_eps": RMS_EPS,
            "rope_theta": ROPE_THETA,
            "rope_scale": ROPE_SCALE,
            "attention_scale": ATTENTION_SCALE,
            "logits_soft_cap": 0.0,
        },
        "rounding": [
            "All model weights are raw little-endian BF16 words.",
            "Decoder, q/k, and final RMSNorm use Gemma raw_weight + 1 semantics.",
            "Projection GEMMs consume BF16 inputs and BF16 weights; intermediate projections store BF16, logits store f32.",
            "Post-attention MoE uses 256 experts, top-8 softmax routing, fused gate/up expert weights, and a sigmoid-gated shared expert branch.",
            "The MoE/shared branch uses synthetic intermediate width 8 while retaining the real hidden, GQA, and head dimensions.",
            "Decode row reference is computed as row 4 of the same causal sequence after the prefill top token is appended.",
        ],
    }
    return tensors, metadata


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


def _write_sparse_weight_tensor(
    root: Path,
    *,
    name: str,
    kind: str,
    out_features: int,
    in_features: int,
    description: str,
) -> WrittenTensor:
    file_name = f"model_{name}.bf16"
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
                "storage": "dense row-major BF16; deterministic sparse rows",
            },
        ),
        sha256=sha256,
        byte_count=byte_count,
    )


def _write_sparse_bf16_weight_tensor(
    root: Path,
    *,
    name: str,
    shape: tuple[int, ...],
    in_features: int,
    row_count: int,
    term_fn,
    description: str,
    metadata: dict[str, Any],
) -> WrittenTensor:
    file_name = f"model_{name}.bf16"
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
        name="moe_gate_up_proj_weight",
        shape=(MOE_NUM_EXPERTS, 2 * MOE_INTERMEDIATE, HIDDEN_SIZE),
        in_features=HIDDEN_SIZE,
        row_count=MOE_NUM_EXPERTS * 2 * MOE_INTERMEDIATE,
        term_fn=lambda row_idx: _moe_gate_up_terms(
            row_idx // (2 * MOE_INTERMEDIATE),
            row_idx % (2 * MOE_INTERMEDIATE),
        ),
        description="Dense row-major BF16 fused MoE gate/up projection weight.",
        metadata={
            "op": "moe_gate_up_projection_weight",
            "strides": [2 * MOE_INTERMEDIATE * HIDDEN_SIZE, HIDDEN_SIZE, 1],
            "sparse_terms_per_row": MOE_GATE_UP_TERMS,
        },
    )


def _write_moe_down_weight_tensor(root: Path) -> WrittenTensor:
    file_name = "model_moe_down_proj_weight.bf16"
    path = root / file_name
    digest = hashlib.sha256()
    byte_count = 0
    with path.open("wb") as f:
        for expert in range(MOE_NUM_EXPERTS):
            for hidden in range(HIDDEN_SIZE):
                row = array(
                    "H",
                    [
                        float32_to_bf16_bits(_moe_down_weight_value(expert, hidden, idx))
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
    file_name = "model_moe_shared_down_proj_weight.bf16"
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


def _write_zero_bf16_tensor(
    root: Path,
    *,
    name: str,
    shape: tuple[int, ...],
    description: str,
) -> WrittenTensor:
    file_name = f"model_{name}.bf16"
    path = root / file_name
    element_count = 1
    for extent in shape:
        element_count *= extent
    chunk = bytes(8192)
    byte_count = element_count * 2
    digest = hashlib.sha256()
    with path.open("wb") as f:
        remaining = byte_count
        while remaining:
            size = min(remaining, len(chunk))
            piece = chunk[:size]
            f.write(piece)
            digest.update(piece)
            remaining -= size
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
                "op": "zero_weight",
                "strides": _contiguous_strides(shape),
                "elements": element_count,
                "bytes": byte_count,
                "sha256": sha256,
                "storage": "dense row-major BF16 zeros",
            },
        ),
        sha256=sha256,
        byte_count=byte_count,
    )


def _weight_specs_without_hashes() -> list[TensorSpec]:
    return [
        TensorSpec(
            name="q_proj_weight",
            dtype="bf16",
            shape=(Q_PROJ_OUT, HIDDEN_SIZE),
            file="model_q_proj_weight.bf16",
            role="input",
            description="Dense row-major BF16 q projection weight [8192, 2048].",
        ),
        TensorSpec(
            name="k_proj_weight",
            dtype="bf16",
            shape=(KV_HIDDEN, HIDDEN_SIZE),
            file="model_k_proj_weight.bf16",
            role="input",
            description="Dense row-major BF16 k projection weight [512, 2048].",
        ),
        TensorSpec(
            name="v_proj_weight",
            dtype="bf16",
            shape=(KV_HIDDEN, HIDDEN_SIZE),
            file="model_v_proj_weight.bf16",
            role="input",
            description="Dense row-major BF16 v projection weight [512, 2048].",
        ),
        TensorSpec(
            name="o_proj_weight",
            dtype="bf16",
            shape=(HIDDEN_SIZE, Q_HIDDEN),
            file="model_o_proj_weight.bf16",
            role="input",
            description="Dense row-major BF16 output projection weight [2048, 4096].",
        ),
        TensorSpec(
            name="lm_head_weight",
            dtype="bf16",
            shape=(VOCAB_SIZE, HIDDEN_SIZE),
            file="model_lm_head_weight.bf16",
            role="input",
            description="Dense row-major BF16 lm head weight [16, 2048].",
        ),
        TensorSpec(
            name="moe_router_proj_weight",
            dtype="bf16",
            shape=(MOE_NUM_EXPERTS, HIDDEN_SIZE),
            file="model_moe_router_proj_weight.bf16",
            role="input",
            description="Dense row-major BF16 MoE router projection weight.",
        ),
        TensorSpec(
            name="moe_gate_up_proj_weight",
            dtype="bf16",
            shape=(MOE_NUM_EXPERTS, 2 * MOE_INTERMEDIATE, HIDDEN_SIZE),
            file="model_moe_gate_up_proj_weight.bf16",
            role="input",
            description="Dense row-major BF16 fused MoE gate/up projection weight.",
        ),
        TensorSpec(
            name="moe_down_proj_weight",
            dtype="bf16",
            shape=(MOE_NUM_EXPERTS, HIDDEN_SIZE, MOE_INTERMEDIATE),
            file="model_moe_down_proj_weight.bf16",
            role="input",
            description="Dense row-major BF16 MoE down projection weight.",
        ),
        TensorSpec(
            name="moe_shared_gate_proj_weight",
            dtype="bf16",
            shape=(MOE_INTERMEDIATE, HIDDEN_SIZE),
            file="model_moe_shared_gate_proj_weight.bf16",
            role="input",
            description="Dense row-major BF16 shared expert gate projection weight.",
        ),
        TensorSpec(
            name="moe_shared_up_proj_weight",
            dtype="bf16",
            shape=(MOE_INTERMEDIATE, HIDDEN_SIZE),
            file="model_moe_shared_up_proj_weight.bf16",
            role="input",
            description="Dense row-major BF16 shared expert up projection weight.",
        ),
        TensorSpec(
            name="moe_shared_down_proj_weight",
            dtype="bf16",
            shape=(HIDDEN_SIZE, MOE_INTERMEDIATE),
            file="model_moe_shared_down_proj_weight.bf16",
            role="input",
            description="Dense row-major BF16 shared expert down projection weight.",
        ),
        TensorSpec(
            name="moe_shared_expert_gate_weight",
            dtype="bf16",
            shape=(1, HIDDEN_SIZE),
            file="model_moe_shared_expert_gate_weight.bf16",
            role="input",
            description="Dense row-major BF16 scalar shared expert gate projection weight.",
        ),
    ]


def _write_model_weight_tensors(root: Path) -> list[WrittenTensor]:
    return [
        _write_sparse_weight_tensor(
            root,
            name="q_proj_weight",
            kind="q_proj",
            out_features=Q_PROJ_OUT,
            in_features=HIDDEN_SIZE,
            description="Dense row-major BF16 q projection weight [8192, 2048].",
        ),
        _write_sparse_weight_tensor(
            root,
            name="k_proj_weight",
            kind="k_proj",
            out_features=KV_HIDDEN,
            in_features=HIDDEN_SIZE,
            description="Dense row-major BF16 k projection weight [512, 2048].",
        ),
        _write_sparse_weight_tensor(
            root,
            name="v_proj_weight",
            kind="v_proj",
            out_features=KV_HIDDEN,
            in_features=HIDDEN_SIZE,
            description="Dense row-major BF16 v projection weight [512, 2048].",
        ),
        _write_sparse_weight_tensor(
            root,
            name="o_proj_weight",
            kind="o_proj",
            out_features=HIDDEN_SIZE,
            in_features=Q_HIDDEN,
            description="Dense row-major BF16 output projection weight [2048, 4096].",
        ),
        _write_sparse_weight_tensor(
            root,
            name="lm_head_weight",
            kind="lm_head",
            out_features=VOCAB_SIZE,
            in_features=HIDDEN_SIZE,
            description="Dense row-major BF16 lm head weight [16, 2048].",
        ),
        _write_sparse_bf16_weight_tensor(
            root,
            name="moe_router_proj_weight",
            shape=(MOE_NUM_EXPERTS, HIDDEN_SIZE),
            in_features=HIDDEN_SIZE,
            row_count=MOE_NUM_EXPERTS,
            term_fn=_moe_router_terms,
            description="Dense row-major BF16 MoE router projection weight.",
            metadata={
                "op": "moe_router_projection_weight",
                "strides": [HIDDEN_SIZE, 1],
                "sparse_terms_per_row": MOE_ROUTER_TERMS,
            },
        ),
        _write_moe_gate_up_weight_tensor(root),
        _write_moe_down_weight_tensor(root),
        _write_sparse_bf16_weight_tensor(
            root,
            name="moe_shared_gate_proj_weight",
            shape=(MOE_INTERMEDIATE, HIDDEN_SIZE),
            in_features=HIDDEN_SIZE,
            row_count=MOE_INTERMEDIATE,
            term_fn=lambda row_idx: _shared_proj_terms("gate", row_idx),
            description="Dense row-major BF16 shared expert gate projection weight.",
            metadata={
                "op": "shared_expert_gate_projection_weight",
                "strides": [HIDDEN_SIZE, 1],
                "sparse_terms_per_row": MOE_SHARED_TERMS,
            },
        ),
        _write_sparse_bf16_weight_tensor(
            root,
            name="moe_shared_up_proj_weight",
            shape=(MOE_INTERMEDIATE, HIDDEN_SIZE),
            in_features=HIDDEN_SIZE,
            row_count=MOE_INTERMEDIATE,
            term_fn=lambda row_idx: _shared_proj_terms("up", row_idx),
            description="Dense row-major BF16 shared expert up projection weight.",
            metadata={
                "op": "shared_expert_up_projection_weight",
                "strides": [HIDDEN_SIZE, 1],
                "sparse_terms_per_row": MOE_SHARED_TERMS,
            },
        ),
        _write_shared_down_weight_tensor(root),
        _write_sparse_bf16_weight_tensor(
            root,
            name="moe_shared_expert_gate_weight",
            shape=(1, HIDDEN_SIZE),
            in_features=HIDDEN_SIZE,
            row_count=1,
            term_fn=lambda row_idx: _shared_proj_terms("shared_gate", row_idx),
            description="Dense row-major BF16 scalar shared expert gate projection weight.",
            metadata={
                "op": "shared_expert_gate_weight",
                "strides": [HIDDEN_SIZE, 1],
                "sparse_terms_per_row": MOE_SHARED_TERMS,
            },
        ),
    ]


INLINE_INPUT_TENSORS = 6


def write_model_logits_artifact(root: str | Path = DEFAULT_OUTPUT) -> Path:
    root_path = Path(root)
    root_path.mkdir(parents=True, exist_ok=True)

    tensors, metadata = build_model_logits_tensors()
    written: list[WrittenTensor] = []
    for tensor in tensors[:INLINE_INPUT_TENSORS]:
        written.append(_write_tensor(root_path, tensor))
    written.extend(_write_model_weight_tensors(root_path))
    for tensor in tensors[INLINE_INPUT_TENSORS:]:
        written.append(_write_tensor(root_path, tensor))

    manifest = VectorManifest(
        name="qwen36_model_logits",
        groups=("qwen36_semantics", "model_logits"),
        description=(
            "Deterministic one-layer Qwen3.6 ModelRunner vector covering "
            "full-attention plus MoE/shared-expert prefill logits, greedy sampling, "
            "and one page-boundary decode logits row."
        ),
        metadata=metadata,
        tensors=tuple(item.spec for item in written),
    )
    payload = json.dumps(manifest.to_json(), indent=2, sort_keys=True)
    manifest_path = root_path / MANIFEST_FILE
    manifest_path.write_text(f"{payload}\n", encoding="utf-8")
    return manifest_path


def build_model_logits_artifact() -> tuple[VectorManifest, dict[str, Any]]:
    tensors, metadata = build_model_logits_tensors()
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
                "bytes": tensor.element_count * {"bf16": 2, "i32": 4, "f32": 4}[tensor.dtype],
            },
        )
        for tensor in tensors
    ]
    tensor_specs[INLINE_INPUT_TENSORS:INLINE_INPUT_TENSORS] = _weight_specs_without_hashes()
    return (
        VectorManifest(
            name="qwen36_model_logits",
            groups=("qwen36_semantics", "model_logits"),
            description=(
                "Deterministic one-layer Qwen3.6 ModelRunner vector covering "
                "full-attention plus MoE/shared-expert prefill logits, greedy sampling, "
                "and one page-boundary decode logits row."
            ),
            metadata=metadata,
            tensors=tuple(tensor_specs),
        ),
        metadata,
    )
