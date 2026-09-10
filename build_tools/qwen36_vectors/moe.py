from __future__ import annotations

import math
import struct
from dataclasses import dataclass
from pathlib import Path

from .io import bf16_bits_to_float32, float32_to_bf16_bits, write_artifact
from .paths import DEFAULT_VECTOR_ROOT
from .schema import TensorSpec, VectorManifest

NUM_EXPERTS = 256
TOP_K = 8
NUM_TOKENS = 5
HIDDEN_SIZE = 8
INTERMEDIATE_SIZE = 8
ROUTED_SCALING_FACTOR = 1.0
DEFAULT_OUTPUT = DEFAULT_VECTOR_ROOT / "moe_shared_expert"


@dataclass(frozen=True)
class CaseSpec:
    name: str
    description: str
    shared_gate_logit: float
    logit_overrides: tuple[tuple[int, float], ...]


CASES: tuple[CaseSpec, ...] = (
    CaseSpec(
        name="tie_topk_boundary",
        description=(
            "Exact ties inside the selected set and at the top-k boundary; "
            "lower expert id wins the boundary tie."
        ),
        shared_gate_logit=-2.0,
        logit_overrides=(
            (4, 2.40),
            (8, 1.70),
            (9, 1.70),
            (21, 1.20),
            (22, 1.10),
            (23, 1.00),
            (30, 0.95),
            (31, 0.90),
            (44, 0.90),
            (45, 0.89),
        ),
    ),
    CaseSpec(
        name="all_negative_logits",
        description=(
            "All selected router logits are negative, including ties before and "
            "at the top-k boundary."
        ),
        shared_gate_logit=0.0,
        logit_overrides=(
            (250, -0.125),
            (0, -0.25),
            (199, -0.25),
            (127, -0.50),
            (3, -0.75),
            (4, -0.75),
            (80, -1.00),
            (81, -1.25),
            (82, -1.25),
            (2, -1.50),
        ),
    ),
    CaseSpec(
        name="near_ties",
        description=(
            "Small logit deltas collapse to BF16 ties, including the eighth and "
            "ninth candidates; lower expert id wins."
        ),
        shared_gate_logit=2.0,
        logit_overrides=(
            (12, 0.499950),
            (13, 0.500020),
            (14, 0.500010),
            (15, 0.500000),
            (16, 0.499990),
            (17, 0.499980),
            (18, 0.499970),
            (19, 0.499960),
            (20, 0.500030),
        ),
    ),
    CaseSpec(
        name="shared_gate_strong_negative",
        description=(
            "Shared expert gate is strongly negative, so the combined output "
            "should be almost the routed MoE output."
        ),
        shared_gate_logit=-14.0,
        logit_overrides=(
            (33, 3.00),
            (34, 2.00),
            (35, 1.00),
            (36, 0.00),
            (37, -0.50),
            (38, -1.00),
            (39, -1.50),
            (40, -2.00),
            (41, -2.50),
        ),
    ),
    CaseSpec(
        name="shared_gate_strong_positive",
        description=(
            "Shared expert gate is strongly positive, so the combined output "
            "should include nearly all of the shared expert branch."
        ),
        shared_gate_logit=14.0,
        logit_overrides=(
            (255, 2.20),
            (254, 2.10),
            (128, 2.00),
            (0, 1.90),
            (1, 1.80),
            (2, 1.70),
            (3, 1.60),
            (4, 1.50),
            (5, 1.40),
        ),
    ),
)


def build_moe_artifact() -> tuple[VectorManifest, dict[str, list[float] | list[int]]]:
    hidden_words, hidden = _build_hidden()
    gate_up_words, gate_up = _build_gate_up_weight()
    down_words, down = _build_down_weight()
    shared_gate_up_words, shared_gate_up = _build_shared_gate_up_weight()
    shared_down_words, shared_down = _build_shared_down_weight()

    router_logits = _build_router_logits()
    topk_ids, topk_unrenorm, topk_weights = _route_topk(router_logits)
    shared_gate_logits = [[_f32(case.shared_gate_logit)] for case in CASES]

    (
        route_down_output,
        routed_moe_output,
        shared_expert_output,
        shared_gate_sigmoid,
        combined_output,
    ) = _compute_outputs(
        hidden=hidden,
        gate_up=gate_up,
        down=down,
        shared_gate_up=shared_gate_up,
        shared_down=shared_down,
        shared_gate_logits=shared_gate_logits,
        topk_ids=topk_ids,
        topk_weights=topk_weights,
    )

    routed_moe_output_bf16 = _bf16_words_from_flat(_flatten2(routed_moe_output))
    combined_output_bf16 = _bf16_words_from_flat(_flatten2(combined_output))

    tensors: dict[str, list[float] | list[int]] = {
        "hidden": hidden_words,
        "router_logits": [float32_to_bf16_bits(x) for x in _flatten2(router_logits)],
        "gate_up_weight": gate_up_words,
        "down_weight": down_words,
        "shared_gate_up_weight": shared_gate_up_words,
        "shared_down_weight": shared_down_words,
        "shared_gate_logits": _flatten2(shared_gate_logits),
        "topk_ids": _flatten2(topk_ids),
        "topk_unrenormalized_weights": _flatten2(topk_unrenorm),
        "topk_weights": _flatten2(topk_weights),
        "route_down_output": _flatten3(route_down_output),
        "routed_moe_output": _flatten2(routed_moe_output),
        "routed_moe_output_bf16": routed_moe_output_bf16,
        "shared_expert_output": _flatten2(shared_expert_output),
        "shared_gate_sigmoid": _flatten2(shared_gate_sigmoid),
        "combined_output": _flatten2(combined_output),
        "combined_output_bf16": combined_output_bf16,
    }
    manifest = _build_manifest(topk_ids)
    return manifest, tensors


def write_moe_artifact(root: str | Path = DEFAULT_OUTPUT) -> Path:
    manifest, tensors = build_moe_artifact()
    return write_artifact(root, manifest, tensors)


def _build_manifest(topk_ids: list[list[int]]) -> VectorManifest:
    return VectorManifest(
        name="qwen36_moe_shared_expert",
        groups=("qwen36_semantics", "moe", "shared_expert"),
        description=(
            "Deterministic Qwen3.6 MoE/shared-expert semantic vectors with "
            "softmax top-8 routing over 256 experts, top-k renormalization, "
            "fused gate/up SwiGLU experts, and sigmoid-gated shared expert add."
        ),
        metadata={
            "num_tokens": NUM_TOKENS,
            "num_experts": NUM_EXPERTS,
            "top_k": TOP_K,
            "hidden_size": HIDDEN_SIZE,
            "intermediate_size": INTERMEDIATE_SIZE,
            "router_score": "softmax",
            "topk_renormalize": True,
            "routed_scaling_factor": ROUTED_SCALING_FACTOR,
            "tie_break": "higher score first, then lower expert id",
            "projection_layout": {
                "gate_up_weight": (
                    "[num_experts, 2 * intermediate_size, hidden_size], "
                    "gate rows first then up rows"
                ),
                "down_weight": "[num_experts, hidden_size, intermediate_size]",
                "shared_gate_up_weight": (
                    "[2 * intermediate_size, hidden_size], gate rows first then up rows"
                ),
                "formula": "out_h = sum_i silu(gate_i) * up_i * down_weight[h, i]",
            },
            "precision": {
                "source_bf16_tensors": (
                    "hidden and weights are stored as raw little-endian BF16 "
                    "words and dequantized before oracle math"
                ),
                "oracle_math": (
                    "routing uses f32-stored logits; expert and shared projections "
                    "use deterministic host float arithmetic"
                ),
                "bf16_expected_outputs": (
                    "routed_moe_output_bf16 and combined_output_bf16 are final "
                    "outputs rounded to BF16"
                ),
            },
            "weight_formulas": {
                "hidden": (
                    "0.125 * (hidden_index - 3.5) + 0.0625 * (token - 2) "
                    "+ alternating 0.03125 * ((token % 3) + 1)"
                ),
                "gate_up_weight": (
                    "signed low-amplitude affine pattern over expert, row, hidden; "
                    "rows [0, intermediate) are gate and rows [intermediate, 2*intermediate) are up"
                ),
                "down_weight": "signed low-amplitude affine pattern over expert, hidden, intermediate",
            },
            "cases": [
                {
                    "row": row,
                    "name": case.name,
                    "description": case.description,
                    "shared_gate_logit": case.shared_gate_logit,
                    "expected_topk_ids": topk_ids[row],
                }
                for row, case in enumerate(CASES)
            ],
        },
        tensors=(
            TensorSpec(
                "hidden",
                "bf16",
                (NUM_TOKENS, HIDDEN_SIZE),
                role="input",
                description="BF16 token activations.",
            ),
            TensorSpec(
                "router_logits",
                "bf16",
                (NUM_TOKENS, NUM_EXPERTS),
                role="input",
                description="BF16 router logits consumed by FP32 softmax top-k routing.",
            ),
            TensorSpec(
                "gate_up_weight",
                "bf16",
                (NUM_EXPERTS, 2 * INTERMEDIATE_SIZE, HIDDEN_SIZE),
                role="input",
                description="Fused expert gate/up projection weights.",
            ),
            TensorSpec(
                "down_weight",
                "bf16",
                (NUM_EXPERTS, HIDDEN_SIZE, INTERMEDIATE_SIZE),
                role="input",
                description="Expert down projection weights.",
            ),
            TensorSpec(
                "shared_gate_up_weight",
                "bf16",
                (2 * INTERMEDIATE_SIZE, HIDDEN_SIZE),
                role="input",
                description="Fused shared expert gate/up projection weights.",
            ),
            TensorSpec(
                "shared_down_weight",
                "bf16",
                (HIDDEN_SIZE, INTERMEDIATE_SIZE),
                role="input",
                description="Shared expert down projection weights.",
            ),
            TensorSpec(
                "shared_gate_logits",
                "f32",
                (NUM_TOKENS, 1),
                role="input",
                description="Scalar logits for sigmoid-gating the shared expert branch.",
            ),
            TensorSpec(
                "topk_ids",
                "i32",
                (NUM_TOKENS, TOP_K),
                role="expected",
                description="Selected expert ids after top-k routing.",
            ),
            TensorSpec(
                "topk_unrenormalized_weights",
                "f32",
                (NUM_TOKENS, TOP_K),
                role="expected",
                description="Selected global-softmax probabilities before top-k renormalization.",
            ),
            TensorSpec(
                "topk_weights",
                "f32",
                (NUM_TOKENS, TOP_K),
                role="expected",
                description="Top-k-renormalized route weights.",
            ),
            TensorSpec(
                "route_down_output",
                "f32",
                (NUM_TOKENS, TOP_K, HIDDEN_SIZE),
                role="expected",
                description="Per-route expert output before route-weight accumulation.",
            ),
            TensorSpec(
                "routed_moe_output",
                "f32",
                (NUM_TOKENS, HIDDEN_SIZE),
                role="expected",
                description="Accumulated routed MoE output before shared expert add.",
            ),
            TensorSpec(
                "routed_moe_output_bf16",
                "bf16",
                (NUM_TOKENS, HIDDEN_SIZE),
                role="expected",
                description="Routed MoE output rounded to BF16.",
            ),
            TensorSpec(
                "shared_expert_output",
                "f32",
                (NUM_TOKENS, HIDDEN_SIZE),
                role="expected",
                description="Shared SwiGLU expert output before sigmoid gate.",
            ),
            TensorSpec(
                "shared_gate_sigmoid",
                "f32",
                (NUM_TOKENS, 1),
                role="expected",
                description="sigmoid(shared_gate_logits).",
            ),
            TensorSpec(
                "combined_output",
                "f32",
                (NUM_TOKENS, HIDDEN_SIZE),
                role="expected",
                description="routed_moe_output + sigmoid(shared_gate) * shared_expert_output.",
            ),
            TensorSpec(
                "combined_output_bf16",
                "bf16",
                (NUM_TOKENS, HIDDEN_SIZE),
                role="expected",
                description="Combined MoE/shared output rounded to BF16.",
            ),
        ),
    )


def _build_hidden() -> tuple[list[int], list[list[float]]]:
    words: list[int] = []
    values: list[list[float]] = []
    for token in range(NUM_TOKENS):
        row: list[float] = []
        for h in range(HIDDEN_SIZE):
            sign = 1.0 if (token + h) % 2 == 0 else -1.0
            value = (
                0.125 * (h - 3.5)
                + 0.0625 * (token - 2)
                + sign * 0.03125 * ((token % 3) + 1)
            )
            word = float32_to_bf16_bits(value)
            words.append(word)
            row.append(bf16_bits_to_float32(word))
        values.append(row)
    return words, values


def _build_gate_up_weight() -> tuple[list[int], list[list[list[float]]]]:
    words: list[int] = []
    values: list[list[list[float]]] = []
    for expert in range(NUM_EXPERTS):
        expert_rows: list[list[float]] = []
        expert_term = (expert % 17) - 8
        up_term = (expert % 7) - 3
        for row in range(2 * INTERMEDIATE_SIZE):
            is_up = row >= INTERMEDIATE_SIZE
            i = row - INTERMEDIATE_SIZE if is_up else row
            row_values: list[float] = []
            for h in range(HIDDEN_SIZE):
                if is_up:
                    value = (
                        -0.015 * (i + 1)
                        + 0.017 * (h + 1)
                        + 0.003 * up_term
                        + (0.005 if (expert * (h + 1) + i) % 3 == 0 else -0.004)
                    )
                else:
                    value = (
                        0.018 * (i + 1)
                        - 0.011 * (h + 1)
                        + 0.004 * expert_term
                        + (0.013 if (expert + i + h) % 2 == 0 else -0.009)
                    )
                word = float32_to_bf16_bits(value)
                words.append(word)
                row_values.append(bf16_bits_to_float32(word))
            expert_rows.append(row_values)
        values.append(expert_rows)
    return words, values


def _build_down_weight() -> tuple[list[int], list[list[list[float]]]]:
    words: list[int] = []
    values: list[list[list[float]]] = []
    for expert in range(NUM_EXPERTS):
        expert_rows: list[list[float]] = []
        expert_term = (expert % 13) - 6
        for h in range(HIDDEN_SIZE):
            row_values: list[float] = []
            for i in range(INTERMEDIATE_SIZE):
                value = (
                    0.014 * (h + 1)
                    - 0.012 * (i + 1)
                    + 0.002 * expert_term
                    + (0.006 if (expert + h + 2 * i) % 4 == 0 else -0.003)
                )
                word = float32_to_bf16_bits(value)
                words.append(word)
                row_values.append(bf16_bits_to_float32(word))
            expert_rows.append(row_values)
        values.append(expert_rows)
    return words, values


def _build_shared_gate_up_weight() -> tuple[list[int], list[list[float]]]:
    words: list[int] = []
    values: list[list[float]] = []
    for row in range(2 * INTERMEDIATE_SIZE):
        is_up = row >= INTERMEDIATE_SIZE
        i = row - INTERMEDIATE_SIZE if is_up else row
        row_values: list[float] = []
        for h in range(HIDDEN_SIZE):
            if is_up:
                value = (
                    -0.010 * (i + 1)
                    + 0.015 * (h + 1)
                    + (0.007 if (i + h) % 3 == 0 else -0.002)
                )
            else:
                value = (
                    0.016 * (i + 1)
                    - 0.009 * (h + 1)
                    + (0.006 if (i + 2 * h) % 4 == 0 else -0.005)
                )
            word = float32_to_bf16_bits(value)
            words.append(word)
            row_values.append(bf16_bits_to_float32(word))
        values.append(row_values)
    return words, values


def _build_shared_down_weight() -> tuple[list[int], list[list[float]]]:
    words: list[int] = []
    values: list[list[float]] = []
    for h in range(HIDDEN_SIZE):
        row_values: list[float] = []
        for i in range(INTERMEDIATE_SIZE):
            value = (
                0.011 * (h + 1)
                - 0.010 * (i + 1)
                + (0.004 if (h + i) % 2 == 0 else -0.006)
            )
            word = float32_to_bf16_bits(value)
            words.append(word)
            row_values.append(bf16_bits_to_float32(word))
        values.append(row_values)
    return words, values


def _build_router_logits() -> list[list[float]]:
    rows: list[list[float]] = []
    for token, case in enumerate(CASES):
        row = [_f32(_default_logit(token, expert)) for expert in range(NUM_EXPERTS)]
        for expert, logit in case.logit_overrides:
            if expert < 0 or expert >= NUM_EXPERTS:
                raise ValueError(f"{case.name}: expert id out of range: {expert}")
            row[expert] = _f32(logit)
        rows.append([bf16_bits_to_float32(float32_to_bf16_bits(x)) for x in row])
    return rows


def _default_logit(token: int, expert: int) -> float:
    return -8.0 - 0.03125 * ((expert * 37 + token * 13) % 29)


def _route_topk(
    logits: list[list[float]],
) -> tuple[list[list[int]], list[list[float]], list[list[float]]]:
    all_ids: list[list[int]] = []
    all_unrenorm: list[list[float]] = []
    all_weights: list[list[float]] = []
    for row in logits:
        best = sorted(range(NUM_EXPERTS), key=lambda expert: (-row[expert], expert))[
            :TOP_K
        ]
        max_logit = max(row)
        exp_scores = [math.exp(logit - max_logit) for logit in row]
        denom = sum(exp_scores)
        unrenorm = [_f32(exp_scores[expert] / denom) for expert in best]
        selected_sum = sum(unrenorm)
        weights = [
            _f32((weight / max(selected_sum, 1.0e-20)) * ROUTED_SCALING_FACTOR)
            for weight in unrenorm
        ]
        all_ids.append(best)
        all_unrenorm.append(unrenorm)
        all_weights.append(weights)
    return all_ids, all_unrenorm, all_weights


def _compute_outputs(
    *,
    hidden: list[list[float]],
    gate_up: list[list[list[float]]],
    down: list[list[list[float]]],
    shared_gate_up: list[list[float]],
    shared_down: list[list[float]],
    shared_gate_logits: list[list[float]],
    topk_ids: list[list[int]],
    topk_weights: list[list[float]],
) -> tuple[
    list[list[list[float]]],
    list[list[float]],
    list[list[float]],
    list[list[float]],
    list[list[float]],
]:
    route_down_output: list[list[list[float]]] = []
    routed_moe_output: list[list[float]] = []
    shared_expert_output: list[list[float]] = []
    shared_gate_sigmoid: list[list[float]] = []
    combined_output: list[list[float]] = []

    for token in range(NUM_TOKENS):
        token_routes: list[list[float]] = []
        routed = [0.0 for _ in range(HIDDEN_SIZE)]
        for pos, expert in enumerate(topk_ids[token]):
            expert_out = _expert_output(hidden[token], gate_up[expert], down[expert])
            token_routes.append([_f32(value) for value in expert_out])
            scale = topk_weights[token][pos]
            for h in range(HIDDEN_SIZE):
                routed[h] += scale * expert_out[h]
        routed = [_f32(value) for value in routed]

        shared = _expert_output(hidden[token], shared_gate_up, shared_down)
        shared = [_f32(value) for value in shared]
        gate = _f32(_sigmoid(shared_gate_logits[token][0]))
        combined = [_f32(routed[h] + gate * shared[h]) for h in range(HIDDEN_SIZE)]

        route_down_output.append(token_routes)
        routed_moe_output.append(routed)
        shared_expert_output.append(shared)
        shared_gate_sigmoid.append([gate])
        combined_output.append(combined)

    return (
        route_down_output,
        routed_moe_output,
        shared_expert_output,
        shared_gate_sigmoid,
        combined_output,
    )


def _expert_output(
    x: list[float],
    gate_up_weight: list[list[float]],
    down_weight: list[list[float]],
) -> list[float]:
    gate = [
        sum(x[h] * gate_up_weight[i][h] for h in range(HIDDEN_SIZE))
        for i in range(INTERMEDIATE_SIZE)
    ]
    up = [
        sum(x[h] * gate_up_weight[INTERMEDIATE_SIZE + i][h] for h in range(HIDDEN_SIZE))
        for i in range(INTERMEDIATE_SIZE)
    ]
    activated = [_silu(gate[i]) * up[i] for i in range(INTERMEDIATE_SIZE)]
    return [
        sum(activated[i] * down_weight[h][i] for i in range(INTERMEDIATE_SIZE))
        for h in range(HIDDEN_SIZE)
    ]


def _silu(value: float) -> float:
    return value * _sigmoid(value)


def _sigmoid(value: float) -> float:
    if value >= 0.0:
        z = math.exp(-value)
        return 1.0 / (1.0 + z)
    z = math.exp(value)
    return z / (1.0 + z)


def _f32(value: float) -> float:
    return struct.unpack("<f", struct.pack("<f", float(value)))[0]


def _bf16_words_from_flat(values: list[float]) -> list[int]:
    return [float32_to_bf16_bits(value) for value in values]


def _flatten2[T](values: list[list[T]]) -> list[T]:
    return [item for row in values for item in row]


def _flatten3(values: list[list[list[float]]]) -> list[float]:
    return [item for row in values for route in row for item in route]
