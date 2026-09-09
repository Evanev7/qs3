from __future__ import annotations

import math
import struct
from pathlib import Path

from .io import bf16_bits_to_float32, float32_to_bf16_bits, write_artifact
from .paths import DEFAULT_VECTOR_ROOT
from .schema import TensorSpec, VectorManifest


SEQ_LENS = (1, 3, 4, 5, 65)
KEY_HEADS = 16
VALUE_HEADS = 32
KEY_DIM = 128
VALUE_DIM = 128
CONV_WIDTH = 4
L2_EPS = 1.0e-6
RMS_EPS = 1.0e-6
SOFTPLUS_THRESHOLD = 20.0
RECURRENT_SCALE = 1.0 / math.sqrt(KEY_DIM)
DEFAULT_OUTPUT = DEFAULT_VECTOR_ROOT / "gdn_post_conv_prep"

Q_DIM = KEY_HEADS * KEY_DIM
K_DIM = KEY_HEADS * KEY_DIM
V_DIM = VALUE_HEADS * VALUE_DIM
Z_DIM = VALUE_HEADS * VALUE_DIM
QKV_DIM = Q_DIM + K_DIM + V_DIM
QKVZ_DIM = QKV_DIM + Z_DIM
BA_DIM = 2 * VALUE_HEADS
TOTAL_TOKENS = sum(SEQ_LENS)
DECODE_STEPS = 2
DECODE_CASES = len(SEQ_LENS)


def build_gdn_artifact() -> tuple[VectorManifest, dict[str, list[float] | list[int]]]:
    case_lengths = list(SEQ_LENS)
    case_offsets = _case_offsets(SEQ_LENS)
    token_case_ids, token_positions = _token_index_tensors(SEQ_LENS)

    (
        mixed_qkvz,
        mixed_ba,
        mixed_qkv,
        z,
        b,
        a,
        mixed_qkvz_interleaved_debug,
        mixed_ba_interleaved_debug,
    ) = _build_projected_inputs()

    conv_weight_words, conv_weight = _build_conv_weight()
    conv_output_f32, conv_output_bf16 = _causal_conv1d_silu(
        mixed_qkv,
        conv_weight,
        case_offsets,
    )
    conv_final_state = _conv_final_state(mixed_qkv, case_offsets)

    q_raw, k_raw, v_raw = _split_qkv(conv_output_bf16)
    q_l2norm_f32, q_l2norm_bf16 = _l2_normalize_heads(q_raw, KEY_HEADS, KEY_DIM)
    k_l2norm_f32, k_l2norm_bf16 = _l2_normalize_heads(k_raw, KEY_HEADS, KEY_DIM)

    a_log_words = [_bf16(value) for value in _build_a_log()]
    dt_bias_words = [_bf16(value) for value in _build_dt_bias()]
    a_log = [bf16_bits_to_float32(word) for word in a_log_words]
    dt_bias = [bf16_bits_to_float32(word) for word in dt_bias_words]
    g, decay_exp, beta = _gating(a, b, a_log, dt_bias)

    recurrent_q, recurrent_k = _recurrent_qk_values(
        q_raw,
        k_raw,
        normalize_inside=True,
    )
    (
        recurrent_output_f32,
        recurrent_output_bf16,
        recurrent_final_state_f32,
        recurrent_final_state_bf16,
    ) = _gdn_prefill_recurrence(
        recurrent_q,
        recurrent_k,
        v_raw,
        decay_exp,
        beta,
        case_offsets,
    )
    recurrence_debug = _vllm_prefill_recurrence_debug(
        q_l2norm_bf16,
        k_l2norm_bf16,
        v_raw,
        decay_exp,
        beta,
        case_offsets,
        recurrent_output_f32,
        recurrent_final_state_f32,
    )
    decode = _gdn_decode_continuation(
        mixed_qkv,
        conv_weight,
        a_log,
        dt_bias,
        recurrent_final_state_bf16,
        case_offsets,
    )

    gated_rmsnorm_weight_words, gated_rmsnorm_weight = _build_gated_rmsnorm_weight()
    gated_rmsnorm_output_f32, gated_rmsnorm_output_bf16 = _gated_rmsnorm_silu(
        v_raw,
        z,
        gated_rmsnorm_weight,
    )

    tensors: dict[str, list[float] | list[int]] = {
        "case_lengths": case_lengths,
        "case_offsets": case_offsets,
        "token_case_ids": token_case_ids,
        "token_positions": token_positions,
        "mixed_qkvz": mixed_qkvz,
        "mixed_ba": mixed_ba,
        "mixed_qkv": mixed_qkv,
        "z": z,
        "b": b,
        "a": a,
        "mixed_qkvz_interleaved_debug": mixed_qkvz_interleaved_debug,
        "mixed_ba_interleaved_debug": mixed_ba_interleaved_debug,
        "conv_weight": conv_weight_words,
        "conv_output_f32": conv_output_f32,
        "conv_output_bf16": conv_output_bf16,
        "conv_final_state": conv_final_state,
        "q_raw": q_raw,
        "k_raw": k_raw,
        "v_raw": v_raw,
        "q_l2norm_f32": q_l2norm_f32,
        "q_l2norm_bf16": q_l2norm_bf16,
        "k_l2norm_f32": k_l2norm_f32,
        "k_l2norm_bf16": k_l2norm_bf16,
        "A_log": a_log_words,
        "dt_bias": dt_bias_words,
        "g": g,
        "decay_exp": decay_exp,
        "beta": beta,
        "gated_rmsnorm_weight": gated_rmsnorm_weight_words,
        "gated_rmsnorm_output_f32": gated_rmsnorm_output_f32,
        "gated_rmsnorm_output_bf16": gated_rmsnorm_output_bf16,
        "recurrent_output_f32": recurrent_output_f32,
        "recurrent_output_bf16": recurrent_output_bf16,
        "recurrent_final_state_f32": recurrent_final_state_f32,
        "recurrent_final_state_bf16": recurrent_final_state_bf16,
    }
    for step in range(1, DECODE_STEPS + 1):
        step_tensors = decode[f"step{step}"]
        for name, values in step_tensors.items():
            tensors[f"decode_step{step}_{name}"] = values
    tensors["decode_case_ids"] = list(range(DECODE_CASES))
    tensors["decode_step1_positions"] = [length for length in SEQ_LENS]
    tensors["decode_step2_positions"] = [length + 1 for length in SEQ_LENS]
    return _build_manifest(recurrence_debug), tensors


def write_gdn_artifact(root: str | Path = DEFAULT_OUTPUT) -> Path:
    manifest, tensors = build_gdn_artifact()
    return write_artifact(root, manifest, tensors)


def _build_manifest(recurrence_debug: dict[str, object]) -> VectorManifest:
    return VectorManifest(
        name="qwen36_gdn_post_conv_prep",
        groups=("qwen36_semantics", "gdn", "post_conv_prep"),
        description=(
            "Deterministic Qwen3.6 GDN post-conv/prep vectors covering "
            "the production non-interleaved [q,k,v,z] and [b,a] projection "
            "layout, causal conv width 4, q/k/v split, q/k L2 normalization, "
            "beta/decay materialization, norm-before-gate RMSNormGated "
            "with SiLU z, and decode-side continued conv state evolution."
        ),
        metadata={
            "sequence_lengths": list(SEQ_LENS),
            "total_tokens": TOTAL_TOKENS,
            "production_layout": {
                "family": "qwen3.5_qwen3.6_non_interleaved",
                "gqa_interleaved_layout": False,
                "projection_order": {
                    "mixed_qkvz": ["q", "k", "v", "z"],
                    "mixed_ba": ["b", "a"],
                },
                "qs3_consumed_buffers": ["mixed_qkv", "z", "b", "a"],
            },
            "dimensions": {
                "linear_num_key_heads": KEY_HEADS,
                "linear_num_value_heads": VALUE_HEADS,
                "linear_key_head_dim": KEY_DIM,
                "linear_value_head_dim": VALUE_DIM,
                "q_dim": Q_DIM,
                "k_dim": K_DIM,
                "v_dim": V_DIM,
                "z_dim": Z_DIM,
                "packed_qkv_dim": QKV_DIM,
                "packed_qkvz_dim": QKVZ_DIM,
                "ba_dim": BA_DIM,
                "conv_width": CONV_WIDTH,
            },
            "normalization_eps": {
                "qk_l2norm": L2_EPS,
                "gated_rmsnorm": RMS_EPS,
            },
            "source_semantics": [
                (
                    "3pty/vllm/vllm/model_executor/layers/mamba/gdn/"
                    "qwen_gdn_linear_attn.py::prepare_gdn_attention_core_inputs"
                ),
                (
                    "3pty/vllm/vllm/model_executor/layers/mamba/gdn/"
                    "qwen_gdn_linear_attn.py::_forward_core"
                ),
                "3pty/vllm/tests/kernels/test_fused_gdn_post_conv.py::reference_post_conv",
                "3pty/vllm/vllm/model_executor/layers/layernorm.py::RMSNormGated.forward_static",
            ],
            "layout": {
                "mixed_qkvz": (
                    "[token, q_all[2048], k_all[2048], v_all[4096], "
                    "z_all[4096]]"
                ),
                "mixed_ba": "[token, b_all[32], a_all[32]]",
                "mixed_qkv": "[token, q_all[2048], k_all[2048], v_all[4096]]",
                "z": "[token, value_head, value_dim]",
                "b": "[token, value_head]",
                "a": "[token, value_head]",
                "post_conv_split": (
                    "conv_output_bf16 is split as q_raw[16,128], "
                    "k_raw[16,128], v_raw[32,128]"
                ),
            },
            "debug_layouts": {
                "mixed_qkvz_interleaved_debug": (
                    "Optional Qwen3-Next-only comparison layout: "
                    "[token, key_head_group, q[128], k[128], "
                    "v_group[2,128], z_group[2,128]]. Not production Qwen3.6."
                ),
                "mixed_ba_interleaved_debug": (
                    "Optional Qwen3-Next-only comparison layout: "
                    "[token, key_head_group, b_group[2], a_group[2]]. "
                    "Not production Qwen3.6."
                ),
            },
            "math": {
                "causal_conv": (
                    "depthwise width-4 convolution with zero initial state per "
                    "case: out[t,d] = silu(sum_i x[t-3+i,d] * w[d,i]); "
                    "invalid negative positions are zero"
                ),
                "qk_l2norm": (
                    f"after q/k split: x / max(sqrt(sum(x*x)), {L2_EPS:g}), "
                    "matching torch.nn.functional.normalize placement"
                ),
                "gating": (
                    "beta = sigmoid(b); g = -exp(A_log) * "
                    "softplus(a + dt_bias, beta=1, threshold=20); decay_exp = exp(g)"
                ),
                "gated_rmsnorm": (
                    "RMSNormGated norm_before_gate=True, activation=silu: "
                    f"rms_norm(v_raw, eps={RMS_EPS:g}) * weight * silu(z)."
                ),
                "gdn_prefill_recurrence": (
                    "zero initial recurrent state per case; q/k are raw post-conv "
                    f"BF16 inputs normalized inside recurrence with eps={L2_EPS:g}; "
                    "q/k heads are repeated by two to value heads; q is scaled by "
                    f"{RECURRENT_SCALE:.10g}; state layout is "
                    "[value_head, value_dim, key_dim]. Per token: "
                    "state *= exp(g); kv_mem = state @ k; "
                    "delta = (v - kv_mem) * beta; state += outer(delta, k); "
                    "out = state @ q."
                ),
                "gdn_decode_continuation": (
                    "continuation rows first run through continued causal conv "
                    "seeded from conv_final_state. Recurrence uses the same raw "
                    "post-conv q/k/v, inside q/k normalization, and VK state "
                    "layout as prefill. Step 1 is seeded from "
                    "recurrent_final_state_bf16; step 2 is seeded from "
                    "decode_step1_final_state_bf16. The two steps model qs3's "
                    "staged/live ping-pong by reading slot 0 and writing slot 1, "
                    "then reading slot 1 and writing slot 0."
                ),
            },
            "recurrence_debug": recurrence_debug,
            "decode_continuation": {
                "case_count": DECODE_CASES,
                "steps": DECODE_STEPS,
                "case_ids": list(range(DECODE_CASES)),
                "step_positions": [
                    [length + step for length in SEQ_LENS]
                    for step in range(DECODE_STEPS)
                ],
                "conv_state_seed": "conv_final_state",
                "state_seed": "recurrent_final_state_bf16",
                "state_slots_per_case": 2,
                "slot_schedule": [
                    {"step": 1, "read_slot": 0, "write_slot": 1},
                    {"step": 2, "read_slot": 1, "write_slot": 0},
                ],
                "conv_state_pool_layout": (
                    "[case * 2 + slot, packed_qkv_dim, conv_width - 1] in tests; "
                    "oracle tensors are compact [case, packed_qkv_dim, conv_width - 1]."
                ),
                "state_pool_layout": (
                    "[case * 2 + slot, value_head, value_dim, key_dim] in tests; "
                    "oracle tensors are compact [case, value_head, value_dim, key_dim]."
                ),
            },
            "precision": {
                "bf16_storage": "BF16 tensors store raw little-endian u16 words.",
                "oracle_math": (
                    "Inputs and weights are rounded to BF16 before reference math; "
                    "conv, L2 norm, gating, GDN prefill recurrence, GDN decode "
                    "continuation, and gated RMSNorm references use host float "
                    "arithmetic and f32 tensor storage where named *_f32."
                ),
                "bf16_expected_outputs": (
                    "conv_output_bf16, q_l2norm_bf16, k_l2norm_bf16, and "
                    "gated_rmsnorm_output_bf16 are rounded reference outputs. "
                    "decode_step*_conv_output_bf16 are rounded continued-conv "
                    "reference outputs. recurrent_output_bf16, "
                    "recurrent_final_state_bf16, decode_step*_output_bf16, and "
                    "decode_step*_final_state_bf16 are BF16-rounded views of the "
                    "recurrence f32 oracle."
                ),
            },
        },
        tensors=(
            TensorSpec(
                "case_lengths",
                "i32",
                (len(SEQ_LENS),),
                role="metadata",
                description="Per-case sequence lengths.",
            ),
            TensorSpec(
                "case_offsets",
                "i32",
                (len(SEQ_LENS) + 1,),
                role="metadata",
                description="Prefix sums over case_lengths, suitable as cu_seqlens.",
            ),
            TensorSpec(
                "token_case_ids",
                "i32",
                (TOTAL_TOKENS,),
                role="metadata",
                description="Case id for each flattened token row.",
            ),
            TensorSpec(
                "token_positions",
                "i32",
                (TOTAL_TOKENS,),
                role="metadata",
                description="Position within the case for each flattened token row.",
            ),
            TensorSpec(
                "mixed_qkvz",
                "bf16",
                (TOTAL_TOKENS, QKVZ_DIM),
                role="input",
                description=(
                    "Production non-interleaved Qwen3.6/Qwen3.5 [q,k,v,z] "
                    "projection output."
                ),
                metadata={
                    "layout": "qwen35_qwen36_non_interleaved_qkvz",
                    "gqa_interleaved_layout": False,
                },
            ),
            TensorSpec(
                "mixed_ba",
                "bf16",
                (TOTAL_TOKENS, BA_DIM),
                role="input",
                description=(
                    "Production non-interleaved Qwen3.6/Qwen3.5 [b,a] "
                    "projection output."
                ),
                metadata={
                    "layout": "qwen35_qwen36_non_interleaved_ba",
                    "gqa_interleaved_layout": False,
                },
            ),
            TensorSpec(
                "mixed_qkv",
                "bf16",
                (TOTAL_TOKENS, QKV_DIM),
                role="input",
                description=(
                    "Contiguous [q,k,v] conv input, split from mixed_qkvz and "
                    "also matching qs3's directly projected packed buffer."
                ),
            ),
            TensorSpec(
                "z",
                "bf16",
                (TOTAL_TOKENS, VALUE_HEADS, VALUE_DIM),
                role="input",
                description=(
                    "Output gate z split from non-interleaved mixed_qkvz and "
                    "matching qs3's gate projection buffer."
                ),
            ),
            TensorSpec(
                "b",
                "bf16",
                (TOTAL_TOKENS, VALUE_HEADS),
                role="input",
                description="Beta gate logits split from non-interleaved mixed_ba.",
            ),
            TensorSpec(
                "a",
                "bf16",
                (TOTAL_TOKENS, VALUE_HEADS),
                role="input",
                description="Decay gate logits split from non-interleaved mixed_ba.",
            ),
            TensorSpec(
                "mixed_qkvz_interleaved_debug",
                "bf16",
                (TOTAL_TOKENS, QKVZ_DIM),
                file="mixed_qkvz_interleaved.bf16",
                role="debug",
                description=(
                    "Optional Qwen3-Next interleaved qkvz comparison tensor; "
                    "not production Qwen3.6."
                ),
                metadata={
                    "layout": "qwen3_next_gqa_interleaved_qkvz",
                    "do_not_compare": True,
                    "optional": True,
                    "production_qwen36": False,
                },
            ),
            TensorSpec(
                "mixed_ba_interleaved_debug",
                "bf16",
                (TOTAL_TOKENS, BA_DIM),
                file="mixed_ba_interleaved.bf16",
                role="debug",
                description=(
                    "Optional Qwen3-Next interleaved b/a comparison tensor; "
                    "not production Qwen3.6."
                ),
                metadata={
                    "layout": "qwen3_next_gqa_interleaved_ba",
                    "do_not_compare": True,
                    "optional": True,
                    "production_qwen36": False,
                },
            ),
            TensorSpec(
                "conv_weight",
                "bf16",
                (QKV_DIM, CONV_WIDTH),
                role="input",
                description="Depthwise causal conv weights with width 4.",
            ),
            TensorSpec(
                "conv_output_f32",
                "f32",
                (TOTAL_TOKENS, QKV_DIM),
                role="reference",
                description="Host-float causal conv + SiLU reference before BF16 store rounding.",
            ),
            TensorSpec(
                "conv_output_bf16",
                "bf16",
                (TOTAL_TOKENS, QKV_DIM),
                role="expected",
                description="BF16-stored causal conv + SiLU output.",
            ),
            TensorSpec(
                "conv_final_state",
                "bf16",
                (len(SEQ_LENS), QKV_DIM, CONV_WIDTH - 1),
                role="expected",
                description="Per-case causal conv final state: last width-1 mixed_qkv rows.",
            ),
            TensorSpec(
                "q_raw",
                "bf16",
                (TOTAL_TOKENS, KEY_HEADS, KEY_DIM),
                role="expected",
                description="q split from conv_output_bf16 before L2 normalization.",
            ),
            TensorSpec(
                "k_raw",
                "bf16",
                (TOTAL_TOKENS, KEY_HEADS, KEY_DIM),
                role="expected",
                description="k split from conv_output_bf16 before L2 normalization.",
            ),
            TensorSpec(
                "v_raw",
                "bf16",
                (TOTAL_TOKENS, VALUE_HEADS, VALUE_DIM),
                role="expected",
                description="v split from conv_output_bf16.",
            ),
            TensorSpec(
                "q_l2norm_f32",
                "f32",
                (TOTAL_TOKENS, KEY_HEADS, KEY_DIM),
                role="reference",
                description="q L2 normalization reference before BF16 store rounding.",
            ),
            TensorSpec(
                "q_l2norm_bf16",
                "bf16",
                (TOTAL_TOKENS, KEY_HEADS, KEY_DIM),
                role="expected",
                description="BF16-stored q after post-conv L2 normalization.",
            ),
            TensorSpec(
                "k_l2norm_f32",
                "f32",
                (TOTAL_TOKENS, KEY_HEADS, KEY_DIM),
                role="reference",
                description="k L2 normalization reference before BF16 store rounding.",
            ),
            TensorSpec(
                "k_l2norm_bf16",
                "bf16",
                (TOTAL_TOKENS, KEY_HEADS, KEY_DIM),
                role="expected",
                description="BF16-stored k after post-conv L2 normalization.",
            ),
            TensorSpec(
                "A_log",
                "bf16",
                (VALUE_HEADS,),
                role="input",
                description="BF16 per-value-head A_log used for GDN decay materialization.",
            ),
            TensorSpec(
                "dt_bias",
                "bf16",
                (VALUE_HEADS,),
                role="input",
                description="BF16 per-value-head dt_bias added to a before softplus.",
            ),
            TensorSpec(
                "g",
                "f32",
                (TOTAL_TOKENS, VALUE_HEADS),
                role="expected",
                description="-exp(A_log) * softplus(a + dt_bias).",
            ),
            TensorSpec(
                "decay_exp",
                "f32",
                (TOTAL_TOKENS, VALUE_HEADS),
                role="expected",
                description="exp(g), the multiplicative recurrent decay factor.",
            ),
            TensorSpec(
                "beta",
                "f32",
                (TOTAL_TOKENS, VALUE_HEADS),
                role="expected",
                description="sigmoid(b) beta gate.",
            ),
            TensorSpec(
                "gated_rmsnorm_weight",
                "bf16",
                (VALUE_DIM,),
                role="input",
                description="Nonuniform RMSNormGated weight for each value-head lane.",
            ),
            TensorSpec(
                "gated_rmsnorm_output_f32",
                "f32",
                (TOTAL_TOKENS, VALUE_HEADS, VALUE_DIM),
                role="reference",
                description=(
                    "norm_before_gate RMSNormGated(v_raw, z) reference before "
                    "BF16 rounding."
                ),
            ),
            TensorSpec(
                "gated_rmsnorm_output_bf16",
                "bf16",
                (TOTAL_TOKENS, VALUE_HEADS, VALUE_DIM),
                role="expected",
                description="BF16-stored RMSNormGated(v_raw, z) output.",
            ),
            TensorSpec(
                "recurrent_output_f32",
                "f32",
                (TOTAL_TOKENS, VALUE_HEADS, VALUE_DIM),
                role="reference",
                description=(
                    "Host-float GDN prefill recurrent output before BF16 store rounding."
                ),
            ),
            TensorSpec(
                "recurrent_output_bf16",
                "bf16",
                (TOTAL_TOKENS, VALUE_HEADS, VALUE_DIM),
                role="expected",
                description="BF16-stored GDN prefill recurrent output.",
            ),
            TensorSpec(
                "recurrent_final_state_f32",
                "f32",
                (len(SEQ_LENS), VALUE_HEADS, VALUE_DIM, KEY_DIM),
                role="reference",
                description=(
                    "Host-float GDN prefill final recurrent state in "
                    "[case, value_head, value_dim, key_dim] layout."
                ),
            ),
            TensorSpec(
                "recurrent_final_state_bf16",
                "bf16",
                (len(SEQ_LENS), VALUE_HEADS, VALUE_DIM, KEY_DIM),
                role="expected",
                description=(
                    "BF16-stored GDN prefill final recurrent state in "
                    "[case, value_head, value_dim, key_dim] layout."
                ),
            ),
            TensorSpec(
                "decode_case_ids",
                "i32",
                (DECODE_CASES,),
                role="metadata",
                description="Case id for each compact decode continuation row.",
            ),
            TensorSpec(
                "decode_step1_positions",
                "i32",
                (DECODE_CASES,),
                role="metadata",
                description="Absolute sequence position consumed by decode step 1.",
            ),
            TensorSpec(
                "decode_step2_positions",
                "i32",
                (DECODE_CASES,),
                role="metadata",
                description="Absolute sequence position consumed by decode step 2.",
            ),
            *(
                spec
                for step in range(1, DECODE_STEPS + 1)
                for spec in _decode_step_tensor_specs(step)
            ),
        ),
    )


def _decode_step_tensor_specs(step: int) -> tuple[TensorSpec, ...]:
    prefix = f"decode_step{step}"
    return (
        TensorSpec(
            f"{prefix}_mixed_qkv",
            "bf16",
            (DECODE_CASES, QKV_DIM),
            role="input",
            description=f"Decode step {step} projected [q,k,v] row before continued conv.",
        ),
        TensorSpec(
            f"{prefix}_conv_output_f32",
            "f32",
            (DECODE_CASES, QKV_DIM),
            role="reference",
            description=(
                f"Host-float decode step {step} continued causal conv + SiLU "
                "reference before BF16 store rounding."
            ),
        ),
        TensorSpec(
            f"{prefix}_conv_output_bf16",
            "bf16",
            (DECODE_CASES, QKV_DIM),
            role="expected",
            description=f"BF16-stored decode step {step} continued causal conv + SiLU output.",
        ),
        TensorSpec(
            f"{prefix}_conv_final_state",
            "bf16",
            (DECODE_CASES, QKV_DIM, CONV_WIDTH - 1),
            role="expected",
            description=(
                f"Decode step {step} per-case causal conv final state after "
                "appending the decode row."
            ),
        ),
        TensorSpec(
            f"{prefix}_q_raw",
            "bf16",
            (DECODE_CASES, KEY_HEADS, KEY_DIM),
            role="input",
            description=f"Decode step {step} q rows after continued causal conv.",
        ),
        TensorSpec(
            f"{prefix}_k_raw",
            "bf16",
            (DECODE_CASES, KEY_HEADS, KEY_DIM),
            role="input",
            description=f"Decode step {step} k rows after continued causal conv.",
        ),
        TensorSpec(
            f"{prefix}_v_raw",
            "bf16",
            (DECODE_CASES, VALUE_HEADS, VALUE_DIM),
            role="input",
            description=f"Decode step {step} v rows after continued causal conv.",
        ),
        TensorSpec(
            f"{prefix}_a",
            "bf16",
            (DECODE_CASES, VALUE_HEADS),
            role="input",
            description=f"Decode step {step} decay gate logits.",
        ),
        TensorSpec(
            f"{prefix}_b",
            "bf16",
            (DECODE_CASES, VALUE_HEADS),
            role="input",
            description=f"Decode step {step} beta gate logits.",
        ),
        TensorSpec(
            f"{prefix}_g",
            "f32",
            (DECODE_CASES, VALUE_HEADS),
            role="expected",
            description=f"Decode step {step} g = -exp(A_log) * softplus(a + dt_bias).",
        ),
        TensorSpec(
            f"{prefix}_decay_exp",
            "f32",
            (DECODE_CASES, VALUE_HEADS),
            role="expected",
            description=f"Decode step {step} exp(g) recurrent decay factor.",
        ),
        TensorSpec(
            f"{prefix}_beta",
            "f32",
            (DECODE_CASES, VALUE_HEADS),
            role="expected",
            description=f"Decode step {step} sigmoid(b) beta gate.",
        ),
        TensorSpec(
            f"{prefix}_output_f32",
            "f32",
            (DECODE_CASES, VALUE_HEADS, VALUE_DIM),
            role="reference",
            description=(
                f"Host-float GDN decode step {step} output before BF16 rounding."
            ),
        ),
        TensorSpec(
            f"{prefix}_output_bf16",
            "bf16",
            (DECODE_CASES, VALUE_HEADS, VALUE_DIM),
            role="expected",
            description=f"BF16-stored GDN decode step {step} output.",
        ),
        TensorSpec(
            f"{prefix}_final_state_f32",
            "f32",
            (DECODE_CASES, VALUE_HEADS, VALUE_DIM, KEY_DIM),
            role="reference",
            description=(
                f"Host-float GDN decode step {step} final state in compact "
                "[case, value_head, value_dim, key_dim] layout."
            ),
        ),
        TensorSpec(
            f"{prefix}_final_state_bf16",
            "bf16",
            (DECODE_CASES, VALUE_HEADS, VALUE_DIM, KEY_DIM),
            role="expected",
            description=(
                f"BF16-stored GDN decode step {step} final state in compact "
                "[case, value_head, value_dim, key_dim] layout."
            ),
        ),
    )


def _case_offsets(lengths: tuple[int, ...]) -> list[int]:
    offsets = [0]
    total = 0
    for length in lengths:
        total += length
        offsets.append(total)
    return offsets


def _token_index_tensors(lengths: tuple[int, ...]) -> tuple[list[int], list[int]]:
    case_ids: list[int] = []
    positions: list[int] = []
    for case_id, length in enumerate(lengths):
        for pos in range(length):
            case_ids.append(case_id)
            positions.append(pos)
    return case_ids, positions


def _build_projected_inputs() -> tuple[
    list[int],
    list[int],
    list[int],
    list[int],
    list[int],
    list[int],
    list[int],
    list[int],
]:
    mixed_qkvz: list[int] = []
    mixed_ba: list[int] = []
    mixed_qkv: list[int] = []
    z_out: list[int] = []
    b_out: list[int] = []
    a_out: list[int] = []
    qkvz_interleaved_debug: list[int] = []
    ba_interleaved_debug: list[int] = []

    token_index = 0
    for case_id, length in enumerate(SEQ_LENS):
        for pos in range(length):
            q_heads = [
                [_bf16(_q_value(case_id, pos, head, lane)) for lane in range(KEY_DIM)]
                for head in range(KEY_HEADS)
            ]
            k_heads = [
                [_bf16(_k_value(case_id, pos, head, lane)) for lane in range(KEY_DIM)]
                for head in range(KEY_HEADS)
            ]
            v_heads = [
                [_bf16(_v_value(case_id, pos, head, lane)) for lane in range(VALUE_DIM)]
                for head in range(VALUE_HEADS)
            ]
            z_heads = [
                [_bf16(_z_value(case_id, pos, head, lane)) for lane in range(VALUE_DIM)]
                for head in range(VALUE_HEADS)
            ]
            b_heads = [_bf16(_b_value(case_id, pos, head)) for head in range(VALUE_HEADS)]
            a_heads = [_bf16(_a_value(case_id, pos, head)) for head in range(VALUE_HEADS)]

            token_qkv: list[int] = []
            for heads in (q_heads, k_heads, v_heads):
                for head in heads:
                    token_qkv.extend(head)
            mixed_qkv.extend(token_qkv)
            mixed_qkvz.extend(token_qkv)
            for head in z_heads:
                mixed_qkvz.extend(head)
                z_out.extend(head)
            mixed_ba.extend(b_heads)
            mixed_ba.extend(a_heads)
            b_out.extend(b_heads)
            a_out.extend(a_heads)

            for key_head in range(KEY_HEADS):
                qkvz_interleaved_debug.extend(q_heads[key_head])
                qkvz_interleaved_debug.extend(k_heads[key_head])
                first_value_head = key_head * (VALUE_HEADS // KEY_HEADS)
                for value_head in range(first_value_head, first_value_head + 2):
                    qkvz_interleaved_debug.extend(v_heads[value_head])
                for value_head in range(first_value_head, first_value_head + 2):
                    qkvz_interleaved_debug.extend(z_heads[value_head])
                ba_interleaved_debug.extend(
                    b_heads[first_value_head : first_value_head + 2]
                )
                ba_interleaved_debug.extend(
                    a_heads[first_value_head : first_value_head + 2]
                )
            token_index += 1

    if token_index != TOTAL_TOKENS:
        raise AssertionError("token construction mismatch")
    return (
        mixed_qkvz,
        mixed_ba,
        mixed_qkv,
        z_out,
        b_out,
        a_out,
        qkvz_interleaved_debug,
        ba_interleaved_debug,
    )


def _build_conv_weight() -> tuple[list[int], list[float]]:
    words: list[int] = []
    values: list[float] = []
    for channel in range(QKV_DIM):
        for tap in range(CONV_WIDTH):
            value = _conv_weight_value(channel, tap)
            word = _bf16(value)
            words.append(word)
            values.append(bf16_bits_to_float32(word))
    return words, values


def _causal_conv1d_silu(
    mixed_qkv_words: list[int],
    conv_weight: list[float],
    case_offsets: list[int],
) -> tuple[list[float], list[int]]:
    x = [bf16_bits_to_float32(word) for word in mixed_qkv_words]
    out_f32: list[float] = []
    out_bf16: list[int] = []

    for begin, end in zip(case_offsets, case_offsets[1:]):
        for token in range(begin, end):
            local_pos = token - begin
            for channel in range(QKV_DIM):
                acc = 0.0
                weight_base = channel * CONV_WIDTH
                for tap in range(CONV_WIDTH):
                    src_local = local_pos - (CONV_WIDTH - 1) + tap
                    if src_local < 0:
                        continue
                    src_token = begin + src_local
                    acc += x[src_token * QKV_DIM + channel] * conv_weight[weight_base + tap]
                value = _f32(_silu(acc))
                out_f32.append(value)
                out_bf16.append(_bf16(value))
    return out_f32, out_bf16


def _conv_final_state(mixed_qkv_words: list[int], case_offsets: list[int]) -> list[int]:
    zeros = [_bf16(0.0)] * QKV_DIM
    state: list[int] = []
    for begin, end in zip(case_offsets, case_offsets[1:]):
        rows: list[list[int]] = []
        for src_token in range(max(begin, end - (CONV_WIDTH - 1)), end):
            row_begin = src_token * QKV_DIM
            rows.append(mixed_qkv_words[row_begin : row_begin + QKV_DIM])
        while len(rows) < CONV_WIDTH - 1:
            rows.insert(0, zeros)

        for channel in range(QKV_DIM):
            for history in range(CONV_WIDTH - 1):
                state.append(rows[history][channel])
    return state


def _split_qkv(
    conv_output_bf16: list[int],
    token_count: int = TOTAL_TOKENS,
) -> tuple[list[int], list[int], list[int]]:
    q: list[int] = []
    k: list[int] = []
    v: list[int] = []
    for token in range(token_count):
        base = token * QKV_DIM
        q.extend(conv_output_bf16[base : base + Q_DIM])
        k.extend(conv_output_bf16[base + Q_DIM : base + Q_DIM + K_DIM])
        v.extend(conv_output_bf16[base + Q_DIM + K_DIM : base + QKV_DIM])
    return q, k, v


def _l2_normalize_heads(
    words: list[int],
    heads: int,
    dim: int,
) -> tuple[list[float], list[int]]:
    out_f32: list[float] = []
    out_bf16: list[int] = []
    rows = TOTAL_TOKENS * heads
    for row in range(rows):
        base = row * dim
        x = [bf16_bits_to_float32(word) for word in words[base : base + dim]]
        norm = math.sqrt(sum(value * value for value in x))
        denom = max(norm, L2_EPS)
        for value in x:
            y = _f32(value / denom)
            out_f32.append(y)
            out_bf16.append(_bf16(y))
    return out_f32, out_bf16


def _build_a_log() -> list[float]:
    return [
        _f32(-2.25 + 0.0625 * ((head * 7) % 17) - 0.015625 * (head % 3))
        for head in range(VALUE_HEADS)
    ]


def _build_dt_bias() -> list[float]:
    return [
        _f32(-0.375 + 0.03125 * ((head * 5) % 19) + 0.0078125 * ((head % 4) - 1.5))
        for head in range(VALUE_HEADS)
    ]


def _gating(
    a_words: list[int],
    b_words: list[int],
    a_log: list[float],
    dt_bias: list[float],
) -> tuple[list[float], list[float], list[float]]:
    tokens = len(a_words) // VALUE_HEADS
    if len(a_words) != tokens * VALUE_HEADS or len(b_words) != len(a_words):
        raise AssertionError("GDN gate tensors must be [tokens, value_heads]")
    g: list[float] = []
    decay_exp: list[float] = []
    beta: list[float] = []
    for token in range(tokens):
        for head in range(VALUE_HEADS):
            idx = token * VALUE_HEADS + head
            a = bf16_bits_to_float32(a_words[idx])
            b = bf16_bits_to_float32(b_words[idx])
            x = _f32(a + dt_bias[head])
            g_value = _f32(-math.exp(a_log[head]) * _softplus(x))
            g.append(g_value)
            decay_exp.append(_f32(math.exp(g_value)))
            beta.append(_f32(_sigmoid(b)))
    return g, decay_exp, beta


def _build_gated_rmsnorm_weight() -> tuple[list[int], list[float]]:
    words: list[int] = []
    values: list[float] = []
    for lane in range(VALUE_DIM):
        value = 1.0 + 0.00390625 * ((lane * 5) % 23 - 11)
        value += 0.001953125 if lane % 2 == 0 else -0.00146484375
        word = _bf16(value)
        words.append(word)
        values.append(bf16_bits_to_float32(word))
    return words, values


def _gated_rmsnorm_silu(
    x_words: list[int],
    z_words: list[int],
    weight: list[float],
) -> tuple[list[float], list[int]]:
    out_f32: list[float] = []
    out_bf16: list[int] = []
    rows = TOTAL_TOKENS * VALUE_HEADS
    for row in range(rows):
        base = row * VALUE_DIM
        x = [bf16_bits_to_float32(word) for word in x_words[base : base + VALUE_DIM]]
        z = [bf16_bits_to_float32(word) for word in z_words[base : base + VALUE_DIM]]
        variance = sum(value * value for value in x) / VALUE_DIM
        inv_rms = 1.0 / math.sqrt(variance + RMS_EPS)
        for lane, value in enumerate(x):
            y = _f32(value * inv_rms * weight[lane] * _silu(z[lane]))
            out_f32.append(y)
            out_bf16.append(_bf16(y))
    return out_f32, out_bf16


def _recurrent_qk_values(
    q_words: list[int],
    k_words: list[int],
    *,
    normalize_inside: bool,
    token_count: int = TOTAL_TOKENS,
) -> tuple[list[float], list[float]]:
    q_values: list[float] = []
    k_values: list[float] = []
    rows = token_count * KEY_HEADS
    for row in range(rows):
        base = row * KEY_DIM
        q_row = [bf16_bits_to_float32(word) for word in q_words[base : base + KEY_DIM]]
        k_row = [bf16_bits_to_float32(word) for word in k_words[base : base + KEY_DIM]]
        if normalize_inside:
            q_norm = math.sqrt(sum(value * value for value in q_row))
            k_norm = math.sqrt(sum(value * value for value in k_row))
            q_factor = RECURRENT_SCALE / max(q_norm, L2_EPS)
            k_factor = 1.0 / max(k_norm, L2_EPS)
        else:
            q_factor = RECURRENT_SCALE
            k_factor = 1.0
        for value in q_row:
            q_values.append(_f32(value * q_factor))
        for value in k_row:
            k_values.append(_f32(value * k_factor))
    return q_values, k_values


def _gdn_prefill_recurrence(
    q_values: list[float],
    k_values: list[float],
    v_words: list[int],
    decay_exp: list[float],
    beta: list[float],
    case_offsets: list[int],
) -> tuple[list[float], list[int], list[float], list[int]]:
    output_f32: list[float] = []
    output_bf16: list[int] = []
    final_state_f32: list[float] = []
    final_state_bf16: list[int] = []

    for begin, end in zip(case_offsets, case_offsets[1:]):
        state, case_output = _run_gdn_recurrence_case(
            q_values,
            k_values,
            v_words,
            decay_exp,
            beta,
            begin,
            end,
        )
        output_f32.extend(case_output)
        output_bf16.extend(_bf16(value) for value in case_output)
        final_state_f32.extend(state)
        final_state_bf16.extend(_bf16(value) for value in state)

    return output_f32, output_bf16, final_state_f32, final_state_bf16


def _gdn_decode_continuation(
    mixed_qkv_words: list[int],
    conv_weight: list[float],
    a_log: list[float],
    dt_bias: list[float],
    recurrent_final_state_bf16: list[int],
    case_offsets: list[int],
) -> dict[str, dict[str, list[float] | list[int]]]:
    step_payloads: dict[str, dict[str, list[float] | list[int]]] = {
        f"step{step}": _empty_decode_step_payload()
        for step in range(1, DECODE_STEPS + 1)
    }
    state_size = VALUE_HEADS * VALUE_DIM * KEY_DIM

    for case_id, (begin, end) in enumerate(zip(case_offsets, case_offsets[1:])):
        prompt_rows = [
            mixed_qkv_words[token * QKV_DIM : (token + 1) * QKV_DIM]
            for token in range(begin, end)
        ]
        decode_rows: list[tuple[list[int], list[int], list[int]]] = []
        for step_idx in range(DECODE_STEPS):
            position = SEQ_LENS[case_id] + step_idx
            decode_rows.append(_decode_projected_row(case_id, position))

        state_base = case_id * state_size
        state = [
            bf16_bits_to_float32(word)
            for word in recurrent_final_state_bf16[state_base : state_base + state_size]
        ]

        for step_idx, (mixed_qkv, b_words, a_words) in enumerate(decode_rows, start=1):
            history = prompt_rows + [row[0] for row in decode_rows[: step_idx - 1]]
            history.append(mixed_qkv)
            local_pos = len(prompt_rows) + step_idx - 1
            conv_row_f32, conv_row_bf16 = _continued_conv_row(
                history,
                conv_weight,
                local_pos,
            )
            conv_state = _continued_conv_final_state(history)
            q_words, k_words, v_words = _split_qkv(conv_row_bf16, token_count=1)
            g, decay_exp, beta = _gating(a_words, b_words, a_log, dt_bias)
            q_values, k_values = _recurrent_qk_values(
                q_words,
                k_words,
                normalize_inside=True,
                token_count=1,
            )
            state_f32, output_f32 = _run_gdn_recurrence_case(
                q_values,
                k_values,
                v_words,
                decay_exp,
                beta,
                0,
                1,
                initial_state=state,
            )
            output_bf16 = [_bf16(value) for value in output_f32]
            state_bf16 = [_bf16(value) for value in state_f32]

            payload = step_payloads[f"step{step_idx}"]
            payload["mixed_qkv"].extend(mixed_qkv)
            payload["conv_output_f32"].extend(conv_row_f32)
            payload["conv_output_bf16"].extend(conv_row_bf16)
            payload["conv_final_state"].extend(conv_state)
            payload["q_raw"].extend(q_words)
            payload["k_raw"].extend(k_words)
            payload["v_raw"].extend(v_words)
            payload["a"].extend(a_words)
            payload["b"].extend(b_words)
            payload["g"].extend(g)
            payload["decay_exp"].extend(decay_exp)
            payload["beta"].extend(beta)
            payload["output_f32"].extend(output_f32)
            payload["output_bf16"].extend(output_bf16)
            payload["final_state_f32"].extend(state_f32)
            payload["final_state_bf16"].extend(state_bf16)

            state = [bf16_bits_to_float32(word) for word in state_bf16]

    return step_payloads


def _empty_decode_step_payload() -> dict[str, list[float] | list[int]]:
    return {
        "mixed_qkv": [],
        "conv_output_f32": [],
        "conv_output_bf16": [],
        "conv_final_state": [],
        "q_raw": [],
        "k_raw": [],
        "v_raw": [],
        "a": [],
        "b": [],
        "g": [],
        "decay_exp": [],
        "beta": [],
        "output_f32": [],
        "output_bf16": [],
        "final_state_f32": [],
        "final_state_bf16": [],
    }


def _decode_projected_row(case_id: int, pos: int) -> tuple[list[int], list[int], list[int]]:
    mixed_qkv: list[int] = []
    for head in range(KEY_HEADS):
        for lane in range(KEY_DIM):
            mixed_qkv.append(_bf16(_q_value(case_id, pos, head, lane)))
    for head in range(KEY_HEADS):
        for lane in range(KEY_DIM):
            mixed_qkv.append(_bf16(_k_value(case_id, pos, head, lane)))
    for head in range(VALUE_HEADS):
        for lane in range(VALUE_DIM):
            mixed_qkv.append(_bf16(_v_value(case_id, pos, head, lane)))

    b_words = [_bf16(_b_value(case_id, pos, head)) for head in range(VALUE_HEADS)]
    a_words = [_bf16(_a_value(case_id, pos, head)) for head in range(VALUE_HEADS)]
    return mixed_qkv, b_words, a_words


def _continued_conv_row(
    history_rows: list[list[int]],
    conv_weight: list[float],
    local_pos: int,
) -> tuple[list[float], list[int]]:
    out_f32: list[float] = []
    out_bf16: list[int] = []
    for channel in range(QKV_DIM):
        acc = 0.0
        weight_base = channel * CONV_WIDTH
        for tap in range(CONV_WIDTH):
            src_local = local_pos - (CONV_WIDTH - 1) + tap
            if src_local < 0:
                continue
            acc += (
                bf16_bits_to_float32(history_rows[src_local][channel])
                * conv_weight[weight_base + tap]
            )
        value = _f32(_silu(acc))
        out_f32.append(value)
        out_bf16.append(_bf16(value))
    return out_f32, out_bf16


def _continued_conv_final_state(history_rows: list[list[int]]) -> list[int]:
    zeros = [_bf16(0.0)] * QKV_DIM
    rows = history_rows[-(CONV_WIDTH - 1) :]
    while len(rows) < CONV_WIDTH - 1:
        rows.insert(0, zeros)

    state: list[int] = []
    for channel in range(QKV_DIM):
        for history in range(CONV_WIDTH - 1):
            state.append(rows[history][channel])
    return state


def _run_gdn_recurrence_case(
    q_values: list[float],
    k_values: list[float],
    v_words: list[int],
    decay_exp: list[float],
    beta: list[float],
    begin: int,
    end: int,
    initial_state: list[float] | None = None,
) -> tuple[list[float], list[float]]:
    state_size = VALUE_HEADS * VALUE_DIM * KEY_DIM
    value_head_stride = VALUE_DIM * KEY_DIM
    qk_token_stride = KEY_HEADS * KEY_DIM
    output: list[float] = []
    state = [0.0] * state_size if initial_state is None else list(initial_state)
    if len(state) != state_size:
        raise AssertionError("initial GDN state has the wrong element count")
    head_repeat = VALUE_HEADS // KEY_HEADS
    key_lanes = range(KEY_DIM)

    for token in range(begin, end):
        qk_token_base = token * qk_token_stride
        gate_base = token * VALUE_HEADS
        v_token_base = token * VALUE_HEADS * VALUE_DIM
        for v_head in range(VALUE_HEADS):
            qk_head = v_head // head_repeat
            q_base = qk_token_base + qk_head * KEY_DIM
            k_base = q_base
            decay = decay_exp[gate_base + v_head]
            beta_gate = beta[gate_base + v_head]
            state_head_base = v_head * value_head_stride
            v_head_base = v_token_base + v_head * VALUE_DIM
            for value_lane in range(VALUE_DIM):
                row_base = state_head_base + value_lane * KEY_DIM
                kv_mem = 0.0
                for key_lane in key_lanes:
                    state_idx = row_base + key_lane
                    h = state[state_idx] * decay
                    state[state_idx] = h
                    kv_mem += h * k_values[k_base + key_lane]

                v_value = bf16_bits_to_float32(v_words[v_head_base + value_lane])
                delta = (v_value - kv_mem) * beta_gate
                out = 0.0
                for key_lane in key_lanes:
                    state_idx = row_base + key_lane
                    h = state[state_idx] + delta * k_values[k_base + key_lane]
                    state[state_idx] = h
                    out += h * q_values[q_base + key_lane]
                output.append(_f32(out))

    return [_f32(value) for value in state], output


def _vllm_prefill_recurrence_debug(
    q_l2norm_bf16: list[int],
    k_l2norm_bf16: list[int],
    v_words: list[int],
    decay_exp: list[float],
    beta: list[float],
    case_offsets: list[int],
    recurrent_output_f32: list[float],
    recurrent_final_state_f32: list[float],
) -> dict[str, object]:
    q_values, k_values = _recurrent_qk_values(
        q_l2norm_bf16,
        k_l2norm_bf16,
        normalize_inside=False,
    )
    long_case_id = len(SEQ_LENS) - 1
    begin = case_offsets[long_case_id]
    end = case_offsets[long_case_id + 1]
    state, output = _run_gdn_recurrence_case(
        q_values,
        k_values,
        v_words,
        decay_exp,
        beta,
        begin,
        end,
    )

    output_base = begin * VALUE_HEADS * VALUE_DIM
    expected_output = recurrent_output_f32[output_base : output_base + len(output)]
    state_size = VALUE_HEADS * VALUE_DIM * KEY_DIM
    state_base = long_case_id * state_size
    expected_state = recurrent_final_state_f32[state_base : state_base + state_size]

    output_delta, output_idx = _max_abs_delta(output, expected_output)
    state_delta, state_idx = _max_abs_delta(state, expected_state)
    return {
        "family": "vllm_prefill_debug_normalized_in_prep",
        "do_not_compare": True,
        "production_qwen36_expected": False,
        "normalization_boundary": (
            "q/k L2-normalized and BF16-rounded during prep, then consumed by "
            "recurrence without inside normalization"
        ),
        "compared_against": "qs3_raw_qk_normalized_inside_recurrence",
        "long_case_id": long_case_id,
        "long_case_length": end - begin,
        "long_case_output_max_abs_delta": _f32(output_delta),
        "long_case_output_max_abs_delta_flat_index": output_idx,
        "long_case_final_state_max_abs_delta": _f32(state_delta),
        "long_case_final_state_max_abs_delta_flat_index": state_idx,
        "differs_on_long_case": output_delta > 0.0 or state_delta > 0.0,
    }


def _max_abs_delta(values: list[float], expected: list[float]) -> tuple[float, int]:
    if len(values) != len(expected):
        raise AssertionError("delta inputs must have equal length")
    max_delta = 0.0
    max_idx = 0
    for idx, (value, expected_value) in enumerate(zip(values, expected)):
        delta = abs(value - expected_value)
        if delta > max_delta:
            max_delta = delta
            max_idx = idx
    return max_delta, max_idx


def _q_value(case_id: int, pos: int, head: int, lane: int) -> float:
    centered = ((case_id * 17 + pos * 29 + head * 7 + lane * 3) % 43) - 21
    return (
        centered * 0.015625
        + (head - 7.5) * 0.00390625
        + ((lane % 11) - 5) * 0.001953125
        + (pos - 2) * 0.0029296875
    )


def _k_value(case_id: int, pos: int, head: int, lane: int) -> float:
    centered = ((case_id * 13 + pos * 31 + head * 11 + lane * 5) % 47) - 23
    return (
        centered * 0.013671875
        + (7.5 - head) * 0.00341796875
        - ((lane % 13) - 6) * 0.001708984375
        + (case_id - 2) * 0.0048828125
    )


def _v_value(case_id: int, pos: int, head: int, lane: int) -> float:
    centered = ((case_id * 23 + pos * 19 + head * 5 + lane * 7) % 53) - 26
    sign = 1.0 if (case_id + pos + head + lane) % 2 == 0 else -1.0
    return (
        centered * 0.0107421875
        + sign * 0.0068359375
        + ((head % 8) - 3.5) * 0.00439453125
        + ((lane % 17) - 8) * 0.001220703125
    )


def _z_value(case_id: int, pos: int, head: int, lane: int) -> float:
    centered = ((case_id * 5 + pos * 7 + head * 13 + lane * 11) % 31) - 15
    return (
        centered * 0.02734375
        + ((head % 4) - 1.5) * 0.03125
        - ((lane % 9) - 4) * 0.00390625
    )


def _b_value(case_id: int, pos: int, head: int) -> float:
    centered = ((case_id * 3 + pos * 11 + head * 7) % 29) - 14
    return centered * 0.078125 + (head % 5 - 2) * 0.03125


def _a_value(case_id: int, pos: int, head: int) -> float:
    centered = ((case_id * 7 + pos * 5 + head * 3) % 23) - 11
    return centered * 0.0625 - (head % 7 - 3) * 0.0234375


def _conv_weight_value(channel: int, tap: int) -> float:
    centered = ((channel * 3 + tap * 11) % 37) - 18
    current_tap_boost = 0.01953125 if tap == CONV_WIDTH - 1 else 0.0
    return centered * 0.0029296875 + current_tap_boost


def _softplus(value: float) -> float:
    if value > SOFTPLUS_THRESHOLD:
        return value
    return math.log1p(math.exp(-abs(value))) + max(value, 0.0)


def _silu(value: float) -> float:
    return value * _sigmoid(value)


def _sigmoid(value: float) -> float:
    if value >= 0.0:
        z = math.exp(-value)
        return 1.0 / (1.0 + z)
    z = math.exp(value)
    return z / (1.0 + z)


def _bf16(value: float) -> int:
    return float32_to_bf16_bits(value)


def _f32(value: float) -> float:
    return struct.unpack("<f", struct.pack("<f", float(value)))[0]
