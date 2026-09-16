"""Deterministic Qwen3.6 GDN decoder-layer vector.

This vector covers the currently implemented qs3 model-layer surface for a
linear-attention block: GDN projections, causal conv, local recurrence, gated
RMSNorm, output projection into the shared post-attention scratch buffer, then
the common MoE/shared-expert MLP, residual add, and next Gemma RMSNorm.
"""

from __future__ import annotations

import hashlib
import json
import struct
import sys
from array import array
from collections.abc import Iterable
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from typing import Any

from .config import VectorConfig
from .full_attention_block import (
    MOE_GATE_UP_TERMS,
    MOE_INTERMEDIATE,
    MOE_NUM_EXPERTS,
    MOE_ROUTER_TERMS,
    MOE_SHARED_TERMS,
    MOE_TOP_K,
    ROWS,
    _add_rows,
    _compute_one_layer_moe_tensors,
    _contiguous_strides,
    _gemma_rmsnorm_rows,
    _moe_down_weight_value,
    _moe_gate_up_terms,
    _moe_router_terms,
    _post_norm_weight_value,
    _project_sparse,
    _projection_terms,
    _shared_down_weight_value,
    _shared_proj_terms,
)
from .gdn import (
    _a_value,
    _b_value,
    _bf16,
    _build_a_log,
    _build_conv_weight,
    _build_dt_bias,
    _build_gated_rmsnorm_weight,
    _causal_conv1d_silu,
    _f32,
    _gating,
    _gdn_prefill_recurrence,
    _k_value,
    _q_value,
    _recurrent_qk_values,
    _silu,
    _split_qkv,
    _v_value,
    _z_value,
)
from .io import bf16_bits_to_float32, float32_to_bf16_bits
from .paths import DEFAULT_VECTOR_ROOT
from .schema import MANIFEST_FILE, TensorSpec, VectorManifest

DEFAULT_OUTPUT = DEFAULT_VECTOR_ROOT / "gdn_decoder_layer"
INPUT_COLUMNS = tuple(range(ROWS))

INLINE_TENSORS_BEFORE_WEIGHTS = 7


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
        return f"gdn_decoder_{self.name}.{self.dtype}"

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


def build_gdn_decoder_layer_tensors(
    config: VectorConfig,
) -> tuple[list[TensorWrite], dict[str, Any]]:
    layer_input = _build_row_coded_input(config)
    residual = _bf16_words(
        _residual_value(row, col)
        for row in range(ROWS)
        for col in range(config.hidden_size)
    )
    mlp_norm_weight = _bf16_words(
        _post_norm_weight_value(col) for col in range(config.hidden_size)
    )

    mixed_qkv, gate, b, a = _build_gdn_projected_rows(config)
    _conv_weight_words, conv_weight = _build_conv_weight(config)
    conv_output_f32, conv_output_bf16 = _causal_conv1d_silu(
        config,
        mixed_qkv,
        conv_weight,
        [0, ROWS],
    )
    q_raw, k_raw, v_raw = _split_qkv(config, conv_output_bf16, token_count=ROWS)
    a_log_words = _bf16_words(_build_a_log(config))
    dt_bias_words = _bf16_words(_build_dt_bias(config))
    a_log = tuple(bf16_bits_to_float32(word) for word in a_log_words)
    dt_bias = tuple(bf16_bits_to_float32(word) for word in dt_bias_words)
    _g, decay_exp, beta = _gating(config, a, b, list(a_log), list(dt_bias))
    q_values, k_values = _recurrent_qk_values(
        config,
        q_raw,
        k_raw,
        normalize_inside=True,
        token_count=ROWS,
    )
    (
        recurrent_output_f32,
        recurrent_output_bf16,
        recurrent_final_state_f32,
        recurrent_final_state_bf16,
    ) = _gdn_prefill_recurrence(
        config,
        q_values,
        k_values,
        v_raw,
        decay_exp,
        beta,
        [0, ROWS],
    )
    rms_weight_words, rms_weight = _build_gated_rmsnorm_weight(config)
    gated_norm_f32, gated_norm_bf16, gated_norm_rows = _gated_rmsnorm_silu_rows(
        config,
        recurrent_output_bf16,
        gate,
        rms_weight,
    )
    out_proj_f32, out_proj_bf16, _out_proj_rows = _project_sparse(
        config,
        gated_norm_rows,
        "o_proj",
        config.hidden_size,
        config.gdn_output_dim,
    )
    residual_after_attention_f32, residual_after_attention_bf16, residual_rows = (
        _add_rows(
            out_proj_bf16,
            residual,
            config.hidden_size,
        )
    )
    post_norm_f32, post_norm_bf16, post_norm_rows = _gemma_rmsnorm_rows(
        config,
        residual_rows,
        mlp_norm_weight,
    )
    moe_tensors, moe_metadata = _compute_one_layer_moe_tensors(
        config,
        post_norm_rows,
        residual_after_attention_bf16,
    )
    next_norm_tensor = _convert_block_tensor(moe_tensors[0])
    moe_expected_tensors = [_convert_block_tensor(tensor) for tensor in moe_tensors[1:]]

    tensors = [
        TensorWrite(
            "layer_input",
            "bf16",
            (ROWS, config.hidden_size),
            layer_input,
            "input",
            "gdn_layer_input",
            "Row-coded BF16 hidden input consumed by the GDN projections.",
        ),
        TensorWrite(
            "input_residual",
            "bf16",
            (ROWS, config.hidden_size),
            residual,
            "input",
            "block_residual",
            "BF16 residual entering the GDN decoder-layer slice.",
        ),
        TensorWrite(
            "mlp_norm_raw_weight",
            "bf16",
            (config.hidden_size,),
            mlp_norm_weight,
            "input",
            "fused_add_gemma_rmsnorm",
            "Raw post-GDN RMSNorm weight; effective multiplier is raw + 1.",
        ),
        next_norm_tensor,
        TensorWrite(
            "conv_bias",
            "bf16",
            (config.qkv_dim,),
            tuple(_bf16(0.0) for _ in range(config.qkv_dim)),
            "input",
            "gdn_causal_conv",
            "Zero BF16 causal-conv bias used by execute_gdn_layer.",
        ),
        TensorWrite(
            "A_log",
            "bf16",
            (config.value_heads,),
            a_log_words,
            "input",
            "gdn_recurrence",
            "BF16 per-value-head A_log used for GDN decay materialization.",
        ),
        TensorWrite(
            "dt_bias",
            "bf16",
            (config.value_heads,),
            dt_bias_words,
            "input",
            "gdn_recurrence",
            "BF16 per-value-head dt_bias added to decay gate logits.",
        ),
        TensorWrite(
            "rms_weight",
            "bf16",
            (config.value_dim,),
            tuple(rms_weight_words),
            "input",
            "gdn_gated_rmsnorm",
            "Per-lane GDN gated RMSNorm weight.",
        ),
        TensorWrite(
            "projected_mixed_qkv",
            "bf16",
            (ROWS, config.qkv_dim),
            mixed_qkv,
            "reference",
            "gdn_in_projection",
            "Expected BF16 [q,k,v] projection output before causal conv.",
        ),
        TensorWrite(
            "projected_gate",
            "bf16",
            (ROWS, config.value_heads, config.value_dim),
            gate,
            "reference",
            "gdn_gate_projection",
            "Expected BF16 GDN output-gate projection.",
        ),
        TensorWrite(
            "projected_b",
            "bf16",
            (ROWS, config.value_heads),
            b,
            "reference",
            "gdn_b_projection",
            "Expected BF16 beta-gate projection.",
        ),
        TensorWrite(
            "projected_a",
            "bf16",
            (ROWS, config.value_heads),
            a,
            "reference",
            "gdn_a_projection",
            "Expected BF16 decay-gate projection.",
        ),
        TensorWrite(
            "expected_conv_output_f32",
            "f32",
            (ROWS, config.qkv_dim),
            tuple(conv_output_f32),
            "reference",
            "gdn_causal_conv",
            "Host-float causal conv + SiLU reference before BF16 storage.",
        ),
        TensorWrite(
            "expected_conv_output_bf16",
            "bf16",
            (ROWS, config.qkv_dim),
            tuple(conv_output_bf16),
            "expected",
            "gdn_causal_conv",
            "BF16-stored causal conv + SiLU output.",
        ),
        TensorWrite(
            "expected_recurrent_output_f32",
            "f32",
            (ROWS, config.value_heads, config.value_dim),
            tuple(recurrent_output_f32),
            "reference",
            "gdn_prefill_recurrence",
            "Host-float GDN prefill recurrent output before BF16 storage.",
        ),
        TensorWrite(
            "expected_recurrent_output_bf16",
            "bf16",
            (ROWS, config.value_heads, config.value_dim),
            tuple(recurrent_output_bf16),
            "expected",
            "gdn_prefill_recurrence",
            "BF16-stored GDN prefill recurrent output.",
        ),
        TensorWrite(
            "expected_recurrent_final_state_f32",
            "f32",
            (1, config.value_heads, config.value_dim, config.key_dim),
            tuple(recurrent_final_state_f32),
            "reference",
            "gdn_prefill_recurrence",
            "Host-float GDN prefill final recurrent state.",
        ),
        TensorWrite(
            "expected_recurrent_final_state_bf16",
            "bf16",
            (1, config.value_heads, config.value_dim, config.key_dim),
            tuple(recurrent_final_state_bf16),
            "expected",
            "gdn_prefill_recurrence",
            "BF16-stored GDN prefill final recurrent state.",
        ),
        TensorWrite(
            "expected_gated_norm_output_f32",
            "f32",
            (ROWS, config.value_heads, config.value_dim),
            gated_norm_f32,
            "reference",
            "gdn_gated_rmsnorm",
            "Host-float RMSNormGated(recurrent_out, gate) before BF16 storage.",
        ),
        TensorWrite(
            "expected_gated_norm_output_bf16",
            "bf16",
            (ROWS, config.value_heads, config.value_dim),
            gated_norm_bf16,
            "expected",
            "gdn_gated_rmsnorm",
            "BF16-stored RMSNormGated(recurrent_out, gate) output.",
        ),
        TensorWrite(
            "expected_output_proj_f32",
            "f32",
            (ROWS, config.hidden_size),
            out_proj_f32,
            "reference",
            "gdn_output_projection",
            "f32 GDN output projection before BF16 storage.",
        ),
        TensorWrite(
            "expected_output_proj_bf16",
            "bf16",
            (ROWS, config.hidden_size),
            out_proj_bf16,
            "expected",
            "gdn_output_projection",
            "BF16 GDN output projection stored in scratch.attn_proj.",
        ),
        TensorWrite(
            "expected_residual_after_attention_f32",
            "f32",
            (ROWS, config.hidden_size),
            residual_after_attention_f32,
            "reference",
            "post_gdn_residual_add",
            "f32 residual after adding the BF16 GDN output projection.",
        ),
        TensorWrite(
            "expected_residual_after_attention_bf16",
            "bf16",
            (ROWS, config.hidden_size),
            residual_after_attention_bf16,
            "expected",
            "post_gdn_residual_add",
            "BF16 residual after adding the GDN output projection.",
        ),
        TensorWrite(
            "expected_post_attn_norm_output_f32",
            "f32",
            (ROWS, config.hidden_size),
            post_norm_f32,
            "reference",
            "post_gdn_gemma_rmsnorm",
            "f32 post-GDN Gemma RMSNorm oracle before BF16 storage.",
        ),
        TensorWrite(
            "expected_post_attn_norm_output_bf16",
            "bf16",
            (ROWS, config.hidden_size),
            post_norm_bf16,
            "expected",
            "post_gdn_gemma_rmsnorm",
            "BF16 post-GDN Gemma RMSNorm output consumed by MoE.",
        ),
        *moe_expected_tensors,
    ]
    metadata = {
        "rows": ROWS,
        "hidden_size": config.hidden_size,
        "input_columns": list(INPUT_COLUMNS),
        "gdn": {
            "key_heads": config.key_heads,
            "value_heads": config.value_heads,
            "key_dim": config.key_dim,
            "value_dim": config.value_dim,
            "packed_qkv_dim": config.qkv_dim,
            "output_dim": config.gdn_output_dim,
            "conv_width": config.conv_width,
            "rms_eps": config.rms_eps,
        },
        "moe": {
            "num_experts": MOE_NUM_EXPERTS,
            "top_k": MOE_TOP_K,
            "intermediate_size": MOE_INTERMEDIATE,
            "shared_expert_intermediate_size": MOE_INTERMEDIATE,
        },
        "source_semantics": [
            "qwen36_vectors.gdn projected-row formulas and GDN recurrence oracle",
            "qwen36_vectors.full_attention_block Gemma RMSNorm and MoE/shared expert oracle",
        ],
        "rounding": [
            "Layer input and projection weights are BF16; row-coded input makes projection GEMMs materialize deterministic BF16 GDN rows.",
            "GDN recurrent output is BF16 before gated RMSNorm, matching execute_gdn_layer scratch storage.",
            "GDN output projection writes BF16 scratch.attn_proj before the shared post-attention MLP path.",
            *moe_metadata["rounding"],
        ],
    }
    return tensors, metadata


def build_gdn_decoder_layer_artifact(
    config: VectorConfig,
) -> tuple[VectorManifest, dict[str, Any]]:
    tensors, metadata = build_gdn_decoder_layer_tensors(config)
    specs = [
        _tensor_spec_without_hash(tensor)
        for tensor in tensors[:INLINE_TENSORS_BEFORE_WEIGHTS]
    ]
    specs.extend(_weight_specs_without_hash(config))
    specs.extend(
        _tensor_spec_without_hash(tensor)
        for tensor in tensors[INLINE_TENSORS_BEFORE_WEIGHTS:]
    )
    return (
        VectorManifest(
            name="qwen36_gdn_decoder_layer",
            groups=("qwen36_semantics", "gdn", "gdn_decoder_layer"),
            description=(
                "Deterministic Qwen3.6 GDN decoder-layer correctness vector with "
                "real hidden/GDN dimensions, GDN output projection, and the common "
                "MoE/shared-expert post-attention path."
            ),
            metadata=metadata,
            tensors=tuple(specs),
        ),
        metadata,
    )


def write_gdn_decoder_layer_artifact(
    config: VectorConfig, root: str | Path = DEFAULT_OUTPUT
) -> Path:
    root_path = Path(root)
    root_path.mkdir(parents=True, exist_ok=True)
    tensors, metadata = build_gdn_decoder_layer_tensors(config)
    written: list[WrittenTensor] = []
    for tensor in tensors[:INLINE_TENSORS_BEFORE_WEIGHTS]:
        written.append(_write_tensor(root_path, tensor))
    written.extend(_write_weight_tensors(config, root_path))
    for tensor in tensors[INLINE_TENSORS_BEFORE_WEIGHTS:]:
        written.append(_write_tensor(root_path, tensor))

    manifest = VectorManifest(
        name="qwen36_gdn_decoder_layer",
        groups=("qwen36_semantics", "gdn", "gdn_decoder_layer"),
        description=(
            "Deterministic Qwen3.6 GDN decoder-layer correctness vector with "
            "real hidden/GDN dimensions, GDN output projection, and the common "
            "MoE/shared-expert post-attention path."
        ),
        metadata=metadata,
        tensors=tuple(item.spec for item in written),
    )
    payload = json.dumps(manifest.to_json(), indent=2, sort_keys=True)
    manifest_path = root_path / MANIFEST_FILE
    manifest_path.write_text(f"{payload}\n", encoding="utf-8")
    return manifest_path


def _build_row_coded_input(config: VectorConfig) -> tuple[int, ...]:
    values = [float32_to_bf16_bits(0.0)] * (ROWS * config.hidden_size)
    one = float32_to_bf16_bits(1.0)
    for row, col in enumerate(INPUT_COLUMNS):
        values[row * config.hidden_size + col] = one
    return tuple(values)


def _build_gdn_projected_rows(
    config: VectorConfig,
) -> tuple[
    tuple[int, ...],
    tuple[int, ...],
    tuple[int, ...],
    tuple[int, ...],
]:
    mixed_qkv: list[int] = []
    gate: list[int] = []
    b: list[int] = []
    a: list[int] = []
    case_id = 0
    for pos in range(ROWS):
        for head in range(config.key_heads):
            for lane in range(config.key_dim):
                mixed_qkv.append(_bf16(_q_value(case_id, pos, head, lane)))
        for head in range(config.key_heads):
            for lane in range(config.key_dim):
                mixed_qkv.append(_bf16(_k_value(case_id, pos, head, lane)))
        for head in range(config.value_heads):
            for lane in range(config.value_dim):
                mixed_qkv.append(_bf16(_v_value(case_id, pos, head, lane)))
        for head in range(config.value_heads):
            for lane in range(config.value_dim):
                gate.append(_bf16(_z_value(case_id, pos, head, lane)))
        for head in range(config.value_heads):
            b.append(_bf16(_b_value(case_id, pos, head)))
        for head in range(config.value_heads):
            a.append(_bf16(_a_value(case_id, pos, head)))
    return tuple(mixed_qkv), tuple(gate), tuple(b), tuple(a)


def _gated_rmsnorm_silu_rows(
    config: VectorConfig,
    x_words: tuple[int, ...] | list[int],
    gate_words: tuple[int, ...] | list[int],
    weight: list[float],
) -> tuple[tuple[float, ...], tuple[int, ...], list[list[float]]]:
    out_f32: list[float] = []
    out_bf16: list[int] = []
    rows: list[list[float]] = []
    for row_idx in range(ROWS):
        row: list[float] = []
        for head in range(config.value_heads):
            base = (row_idx * config.value_heads + head) * config.value_dim
            x = [
                bf16_bits_to_float32(word)
                for word in x_words[base : base + config.value_dim]
            ]
            gate = [
                bf16_bits_to_float32(word)
                for word in gate_words[base : base + config.value_dim]
            ]
            variance = _f32(sum(_f32(value * value) for value in x) / config.value_dim)
            inv_rms = _f32(1.0 / (variance + config.rms_eps) ** 0.5)
            for lane, value in enumerate(x):
                y = _f32(_f32(value * inv_rms) * weight[lane] * _silu(gate[lane]))
                out_f32.append(y)
                bits = float32_to_bf16_bits(y)
                out_bf16.append(bits)
                row.append(bf16_bits_to_float32(bits))
        rows.append(row)
    return tuple(out_f32), tuple(out_bf16), rows


def _residual_value(row: int, col: int) -> float:
    centered = ((row * 89 + col * 23 + (col // 29) * 7) % 131) - 65
    slow = (((col // 97) % 11) - 5) * 0.0078125
    row_bias = (row - 2.5) * 0.015625
    marker = 0.025390625 if col in (0, 5, 127, 511, 1023, 2047) else 0.0
    return centered * 0.0068359375 + slow + row_bias + marker


def _bf16_words(values: Iterable[float]) -> tuple[int, ...]:
    return tuple(float32_to_bf16_bits(value) for value in values)


def _convert_block_tensor(tensor: Any) -> TensorWrite:
    return TensorWrite(
        tensor.name,
        tensor.dtype,
        tensor.shape,
        tensor.values,
        tensor.role,
        tensor.op,
        tensor.description,
    )


def _tensor_spec_without_hash(tensor: TensorWrite) -> TensorSpec:
    return TensorSpec(
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
            * {"bf16": 2, "i32": 4, "f32": 4}[tensor.dtype],
        },
    )


def _weight_specs_without_hash(config: VectorConfig) -> list[TensorSpec]:
    return [
        TensorSpec(
            name="in_proj_weight",
            dtype="bf16",
            shape=(config.qkv_dim, config.hidden_size),
            file="gdn_decoder_in_proj_weight.bf16",
            role="input",
            description="Dense row-major BF16 GDN [q,k,v] projection weight.",
        ),
        TensorSpec(
            name="gate_proj_weight",
            dtype="bf16",
            shape=(config.gdn_output_dim, config.hidden_size),
            file="gdn_decoder_gate_proj_weight.bf16",
            role="input",
            description="Dense row-major BF16 GDN output-gate projection weight.",
        ),
        TensorSpec(
            name="a_proj_weight",
            dtype="bf16",
            shape=(config.value_heads, config.hidden_size),
            file="gdn_decoder_a_proj_weight.bf16",
            role="input",
            description="Dense row-major BF16 GDN decay-gate projection weight.",
        ),
        TensorSpec(
            name="b_proj_weight",
            dtype="bf16",
            shape=(config.value_heads, config.hidden_size),
            file="gdn_decoder_b_proj_weight.bf16",
            role="input",
            description="Dense row-major BF16 GDN beta-gate projection weight.",
        ),
        TensorSpec(
            name="conv_weight",
            dtype="bf16",
            shape=(config.qkv_dim, config.conv_width),
            file="gdn_decoder_conv_weight.bf16",
            role="input",
            description="Dense row-major BF16 depthwise causal conv weight.",
        ),
        TensorSpec(
            name="out_proj_weight",
            dtype="bf16",
            shape=(config.hidden_size, config.gdn_output_dim),
            file="gdn_decoder_out_proj_weight.bf16",
            role="input",
            description="Dense row-major BF16 GDN output projection weight.",
        ),
        TensorSpec(
            name="moe_router_proj_weight",
            dtype="bf16",
            shape=(MOE_NUM_EXPERTS, config.hidden_size),
            file="gdn_decoder_moe_router_proj_weight.bf16",
            role="input",
            description="Dense row-major BF16 MoE router projection weight.",
        ),
        TensorSpec(
            name="moe_gate_up_proj_weight",
            dtype="bf16",
            shape=(MOE_NUM_EXPERTS, 2 * MOE_INTERMEDIATE, config.hidden_size),
            file="gdn_decoder_moe_gate_up_proj_weight.bf16",
            role="input",
            description="Dense row-major BF16 fused MoE gate/up projection weight.",
        ),
        TensorSpec(
            name="moe_down_proj_weight",
            dtype="bf16",
            shape=(MOE_NUM_EXPERTS, config.hidden_size, MOE_INTERMEDIATE),
            file="gdn_decoder_moe_down_proj_weight.bf16",
            role="input",
            description="Dense row-major BF16 MoE down projection weight.",
        ),
        TensorSpec(
            name="moe_shared_gate_proj_weight",
            dtype="bf16",
            shape=(MOE_INTERMEDIATE, config.hidden_size),
            file="gdn_decoder_moe_shared_gate_proj_weight.bf16",
            role="input",
            description="Dense row-major BF16 shared expert gate projection weight.",
        ),
        TensorSpec(
            name="moe_shared_up_proj_weight",
            dtype="bf16",
            shape=(MOE_INTERMEDIATE, config.hidden_size),
            file="gdn_decoder_moe_shared_up_proj_weight.bf16",
            role="input",
            description="Dense row-major BF16 shared expert up projection weight.",
        ),
        TensorSpec(
            name="moe_shared_down_proj_weight",
            dtype="bf16",
            shape=(config.hidden_size, MOE_INTERMEDIATE),
            file="gdn_decoder_moe_shared_down_proj_weight.bf16",
            role="input",
            description="Dense row-major BF16 shared expert down projection weight.",
        ),
        TensorSpec(
            name="moe_shared_expert_gate_weight",
            dtype="bf16",
            shape=(1, config.hidden_size),
            file="gdn_decoder_moe_shared_expert_gate_weight.bf16",
            role="input",
            description="Dense row-major BF16 scalar shared expert gate weight.",
        ),
    ]


def _write_weight_tensors(config: VectorConfig, root: Path) -> list[WrittenTensor]:
    conv_weight_words, _ = _build_conv_weight(config)
    return [
        _write_row_coded_projection_weight(
            config,
            root,
            "in_proj_weight",
            config.qkv_dim,
            _build_gdn_projected_rows(config)[0],
            "gdn_in_projection_weight",
            "Dense row-major BF16 GDN [q,k,v] projection weight.",
        ),
        _write_row_coded_projection_weight(
            config,
            root,
            "gate_proj_weight",
            config.gdn_output_dim,
            _build_gdn_projected_rows(config)[1],
            "gdn_gate_projection_weight",
            "Dense row-major BF16 GDN output-gate projection weight.",
        ),
        _write_row_coded_projection_weight(
            config,
            root,
            "a_proj_weight",
            config.value_heads,
            _build_gdn_projected_rows(config)[3],
            "gdn_a_projection_weight",
            "Dense row-major BF16 GDN decay-gate projection weight.",
        ),
        _write_row_coded_projection_weight(
            config,
            root,
            "b_proj_weight",
            config.value_heads,
            _build_gdn_projected_rows(config)[2],
            "gdn_b_projection_weight",
            "Dense row-major BF16 GDN beta-gate projection weight.",
        ),
        _write_weight_values_tensor(
            root,
            "conv_weight",
            (config.qkv_dim, config.conv_width),
            conv_weight_words,
            "gdn_causal_conv_weight",
            "Dense row-major BF16 depthwise causal conv weight.",
        ),
        _write_projection_weight_tensor(
            config,
            root,
            "out_proj_weight",
            "o_proj",
            config.hidden_size,
            config.gdn_output_dim,
            "Dense row-major BF16 GDN output projection weight.",
        ),
        _write_sparse_bf16_weight_tensor(
            root,
            "moe_router_proj_weight",
            (MOE_NUM_EXPERTS, config.hidden_size),
            config.hidden_size,
            MOE_NUM_EXPERTS,
            partial(_moe_router_terms, config),
            "Dense row-major BF16 MoE router projection weight.",
            {
                "op": "moe_router_projection_weight",
                "strides": [config.hidden_size, 1],
                "sparse_terms_per_row": MOE_ROUTER_TERMS,
            },
        ),
        _write_sparse_bf16_weight_tensor(
            root,
            "moe_gate_up_proj_weight",
            (MOE_NUM_EXPERTS, 2 * MOE_INTERMEDIATE, config.hidden_size),
            config.hidden_size,
            MOE_NUM_EXPERTS * 2 * MOE_INTERMEDIATE,
            lambda row_idx: _moe_gate_up_terms(
                config,
                row_idx // (2 * MOE_INTERMEDIATE),
                row_idx % (2 * MOE_INTERMEDIATE),
            ),
            "Dense row-major BF16 fused MoE gate/up projection weight.",
            {
                "op": "moe_gate_up_projection_weight",
                "strides": [
                    2 * MOE_INTERMEDIATE * config.hidden_size,
                    config.hidden_size,
                    1,
                ],
                "sparse_terms_per_row": MOE_GATE_UP_TERMS,
            },
        ),
        _write_moe_down_weight_tensor(config, root),
        _write_sparse_bf16_weight_tensor(
            root,
            "moe_shared_gate_proj_weight",
            (MOE_INTERMEDIATE, config.hidden_size),
            config.hidden_size,
            MOE_INTERMEDIATE,
            lambda row_idx: _shared_proj_terms(config, "gate", row_idx),
            "Dense row-major BF16 shared expert gate projection weight.",
            {
                "op": "shared_expert_gate_projection_weight",
                "strides": [config.hidden_size, 1],
                "sparse_terms_per_row": MOE_SHARED_TERMS,
            },
        ),
        _write_sparse_bf16_weight_tensor(
            root,
            "moe_shared_up_proj_weight",
            (MOE_INTERMEDIATE, config.hidden_size),
            config.hidden_size,
            MOE_INTERMEDIATE,
            lambda row_idx: _shared_proj_terms(config, "up", row_idx),
            "Dense row-major BF16 shared expert up projection weight.",
            {
                "op": "shared_expert_up_projection_weight",
                "strides": [config.hidden_size, 1],
                "sparse_terms_per_row": MOE_SHARED_TERMS,
            },
        ),
        _write_shared_down_weight_tensor(config, root),
        _write_sparse_bf16_weight_tensor(
            root,
            "moe_shared_expert_gate_weight",
            (1, config.hidden_size),
            config.hidden_size,
            1,
            lambda row_idx: _shared_proj_terms(config, "shared_gate", row_idx),
            "Dense row-major BF16 scalar shared expert gate weight.",
            {
                "op": "shared_expert_gate_weight",
                "strides": [config.hidden_size, 1],
                "sparse_terms_per_row": MOE_SHARED_TERMS,
            },
        ),
    ]


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


def _write_weight_values_tensor(
    root: Path,
    name: str,
    shape: tuple[int, ...],
    values: list[int],
    op: str,
    description: str,
) -> WrittenTensor:
    tensor = TensorWrite(name, "bf16", shape, tuple(values), "input", op, description)
    return _write_tensor(root, tensor)


def _write_row_coded_projection_weight(
    config: VectorConfig,
    root: Path,
    name: str,
    out_features: int,
    projected_rows: tuple[int, ...],
    op: str,
    description: str,
) -> WrittenTensor:
    file_name = f"gdn_decoder_{name}.bf16"
    path = root / file_name
    digest = hashlib.sha256()
    byte_count = 0
    with path.open("wb") as f:
        for out_feature in range(out_features):
            row = array("H", [0]) * config.hidden_size
            for token_row, input_col in enumerate(INPUT_COLUMNS):
                row[input_col] = projected_rows[token_row * out_features + out_feature]
            if sys.byteorder != "little":
                row.byteswap()
            blob = row.tobytes()
            f.write(blob)
            digest.update(blob)
            byte_count += len(blob)
    sha256 = digest.hexdigest()
    element_count = out_features * config.hidden_size
    return WrittenTensor(
        spec=TensorSpec(
            name=name,
            dtype="bf16",
            shape=(out_features, config.hidden_size),
            file=file_name,
            role="input",
            description=description,
            metadata={
                "op": op,
                "strides": [config.hidden_size, 1],
                "elements": element_count,
                "bytes": byte_count,
                "sha256": sha256,
                "input_columns": list(INPUT_COLUMNS),
                "storage": "dense row-major BF16; row-coded sparse columns",
            },
        ),
        sha256=sha256,
        byte_count=byte_count,
    )


def _write_projection_weight_tensor(
    config: VectorConfig,
    root: Path,
    name: str,
    kind: str,
    out_features: int,
    in_features: int,
    description: str,
) -> WrittenTensor:
    file_name = f"gdn_decoder_{name}.bf16"
    path = root / file_name
    digest = hashlib.sha256()
    byte_count = 0
    with path.open("wb") as f:
        for out_feature in range(out_features):
            row = array("H", [0]) * in_features
            for col, bits in _projection_terms(config, kind, out_feature, in_features):
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
                "op": "gdn_output_projection_weight",
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
    name: str,
    shape: tuple[int, ...],
    in_features: int,
    row_count: int,
    term_fn,
    description: str,
    metadata: dict[str, Any],
) -> WrittenTensor:
    file_name = f"gdn_decoder_{name}.bf16"
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


def _write_moe_down_weight_tensor(config: VectorConfig, root: Path) -> WrittenTensor:
    file_name = "gdn_decoder_moe_down_proj_weight.bf16"
    path = root / file_name
    digest = hashlib.sha256()
    byte_count = 0
    with path.open("wb") as f:
        for expert in range(MOE_NUM_EXPERTS):
            for hidden in range(config.hidden_size):
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
    element_count = MOE_NUM_EXPERTS * config.hidden_size * MOE_INTERMEDIATE
    sha256 = digest.hexdigest()
    return WrittenTensor(
        spec=TensorSpec(
            name="moe_down_proj_weight",
            dtype="bf16",
            shape=(MOE_NUM_EXPERTS, config.hidden_size, MOE_INTERMEDIATE),
            file=file_name,
            role="input",
            description="Dense row-major BF16 MoE down projection weight.",
            metadata={
                "op": "moe_down_projection_weight",
                "strides": [config.hidden_size * MOE_INTERMEDIATE, MOE_INTERMEDIATE, 1],
                "elements": element_count,
                "bytes": byte_count,
                "sha256": sha256,
            },
        ),
        sha256=sha256,
        byte_count=byte_count,
    )


def _write_shared_down_weight_tensor(config: VectorConfig, root: Path) -> WrittenTensor:
    file_name = "gdn_decoder_moe_shared_down_proj_weight.bf16"
    path = root / file_name
    digest = hashlib.sha256()
    byte_count = 0
    with path.open("wb") as f:
        for hidden in range(config.hidden_size):
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
    element_count = config.hidden_size * MOE_INTERMEDIATE
    sha256 = digest.hexdigest()
    return WrittenTensor(
        spec=TensorSpec(
            name="moe_shared_down_proj_weight",
            dtype="bf16",
            shape=(config.hidden_size, MOE_INTERMEDIATE),
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
