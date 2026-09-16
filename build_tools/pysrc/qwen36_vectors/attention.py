"""Deterministic Qwen3.6 full-attention primitive vectors.

The vectors in this module intentionally model only the primitive preparation
path around full attention: packed Q/output-gate extraction, q/k Gemma RMSNorm,
partial RoPE, and output gating. They do not try to encode paged attention.
"""

from __future__ import annotations

import hashlib
import math
import struct
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

from .config import VectorConfig
from .io import write_artifact
from .paths import DEFAULT_VECTOR_ROOT
from .schema import MANIFEST_FILE, TensorSpec, VectorManifest

POSITIONS = (0, 1, 7, 64, 65, 511)
NUM_TOKENS = len(POSITIONS)


ROPE_SCALE = 1.0
DEFAULT_OUTPUT = DEFAULT_VECTOR_ROOT / "full_attention_primitives"


@dataclass(frozen=True)
class TensorData:
    name: str
    dtype: str
    shape: tuple[int, ...]
    data: tuple[int, ...] | tuple[float, ...]
    role: str
    op: str
    description: str
    strides: tuple[int, ...] | None = None

    @property
    def file_name(self) -> str:
        return f"attention_{self.name}.{self.dtype}"

    @property
    def element_count(self) -> int:
        total = 1
        for extent in self.shape:
            total *= extent
        return total


def f32_to_bf16_bits(value: float) -> int:
    """Round a Python float to BF16 using round-to-nearest-even."""

    bits = struct.unpack("<I", struct.pack("<f", float(value)))[0]
    lsb = (bits >> 16) & 1
    return ((bits + 0x7FFF + lsb) >> 16) & 0xFFFF


def bf16_bits_to_f32(bits: int) -> float:
    return struct.unpack("<f", struct.pack("<I", (bits & 0xFFFF) << 16))[0]


def to_bf16_values(values: Iterable[float]) -> tuple[int, ...]:
    return tuple(f32_to_bf16_bits(value) for value in values)


def bf16_values_to_f32(values: Iterable[int]) -> tuple[float, ...]:
    return tuple(bf16_bits_to_f32(value) for value in values)


def tensor_bytes(tensor: TensorData) -> bytes:
    if tensor.dtype == "bf16":
        return struct.pack(f"<{len(tensor.data)}H", *tensor.data)
    if tensor.dtype == "f32":
        return struct.pack(f"<{len(tensor.data)}f", *tensor.data)
    if tensor.dtype == "i32":
        return struct.pack(f"<{len(tensor.data)}i", *tensor.data)
    raise ValueError(f"unsupported tensor dtype {tensor.dtype!r}")


def contiguous_strides(shape: tuple[int, ...]) -> tuple[int, ...]:
    stride = 1
    out = []
    for extent in reversed(shape):
        out.append(stride)
        stride *= extent
    return tuple(reversed(out))


def _q_lane_value(token: int, head: int, lane: int) -> float:
    centered = ((token * 37 + head * 17 + lane * 5) % 43) - 21
    lane_bias = ((lane % 13) - 6) * 0.00390625
    head_bias = (head - 7.5) * 0.005859375
    token_bias = (token - 2.5) * 0.013671875
    tail_marker = 0.03125 if lane in (64, 65, 127, 128, 255) else 0.0
    return centered * 0.03125 + lane_bias + head_bias + token_bias + tail_marker


def _gate_lane_value(token: int, head: int, lane: int) -> float:
    pattern = ((token * 11 + head * 19 + lane * 7) % 17) - 8
    lane_bias = ((lane % 5) - 2) * 0.0625
    head_bias = (head % 4 - 1.5) * 0.125
    token_bias = (token - 2.5) * 0.078125
    return pattern * 0.3125 + lane_bias + head_bias + token_bias


def _k_lane_value(token: int, head: int, lane: int) -> float:
    centered = ((token * 29 + head * 23 + lane * 3) % 37) - 18
    lane_bias = ((lane % 11) - 5) * 0.0048828125
    head_bias = (head - 0.5) * 0.0390625
    token_bias = (2.5 - token) * 0.01171875
    tail_marker = -0.02734375 if lane in (64, 65, 127, 128, 255) else 0.0
    return centered * 0.03515625 + lane_bias + head_bias + token_bias + tail_marker


def _q_raw_weight_value(lane: int) -> float:
    # Nonzero signed raw Gemma weights; the effective multiplier is raw + 1.
    base = ((lane * 7) % 31 - 15) * 0.0078125
    return base + (0.015625 if lane % 2 == 0 else -0.01171875)


def _k_raw_weight_value(lane: int) -> float:
    base = ((lane * 5 + 3) % 29 - 14) * 0.0087890625
    return base + (-0.013671875 if lane % 3 == 0 else 0.017578125)


def _attention_out_value(token: int, head: int, lane: int) -> float:
    pattern = ((token * 31 + head * 13 + lane * 9) % 23) - 11
    sign = -1.0 if (token + head + lane) % 2 else 1.0
    return sign * (0.0625 + abs(pattern) * 0.046875)


def _sigmoid(value: float) -> float:
    if value >= 0.0:
        z = math.exp(-value)
        return 1.0 / (1.0 + z)
    z = math.exp(value)
    return z / (1.0 + z)


def _build_q_gate(
    config: VectorConfig,
) -> tuple[tuple[int, ...], tuple[int, ...], tuple[int, ...]]:
    packed: list[int] = []
    q: list[int] = []
    gate: list[int] = []
    for token in range(NUM_TOKENS):
        for head in range(config.q_heads):
            q_head = [
                f32_to_bf16_bits(_q_lane_value(token, head, lane))
                for lane in range(config.head_dim)
            ]
            gate_head = [
                f32_to_bf16_bits(_gate_lane_value(token, head, lane))
                for lane in range(config.head_dim)
            ]
            packed.extend(q_head)
            packed.extend(gate_head)
            q.extend(q_head)
            gate.extend(gate_head)
    return tuple(packed), tuple(q), tuple(gate)


def _build_k(config: VectorConfig) -> tuple[int, ...]:
    return tuple(
        f32_to_bf16_bits(_k_lane_value(token, head, lane))
        for token in range(NUM_TOKENS)
        for head in range(config.kv_heads)
        for lane in range(config.head_dim)
    )


def _gemma_rmsnorm(
    config: VectorConfig,
    x_bf16: tuple[int, ...],
    raw_weight_bf16: tuple[int, ...],
    rows: int,
) -> tuple[tuple[float, ...], tuple[int, ...]]:
    weights = [bf16_bits_to_f32(value) + 1.0 for value in raw_weight_bf16]
    out_f32: list[float] = []
    out_bf16: list[int] = []
    for row in range(rows):
        base = row * config.head_dim
        x = [bf16_bits_to_f32(bits) for bits in x_bf16[base : base + config.head_dim]]
        variance = sum(value * value for value in x) / config.head_dim
        inv_rms = 1.0 / math.sqrt(variance + config.rms_eps)
        for lane, value in enumerate(x):
            y = value * inv_rms * weights[lane]
            out_f32.append(y)
            out_bf16.append(f32_to_bf16_bits(y))
    return tuple(out_f32), tuple(out_bf16)


def _cos_sin_cache(config: VectorConfig) -> tuple[tuple[float, ...], tuple[int, ...]]:
    max_pos = max(POSITIONS) + 1
    half = config.rotary_dim // 2
    cache_f32: list[float] = []
    cache_bf16: list[int] = []
    inv_freq = [
        1.0 / (config.rope_theta ** (float(i * 2) / config.rotary_dim))
        for i in range(half)
    ]
    for pos in range(max_pos):
        cos_values = [math.cos((pos / ROPE_SCALE) * freq) for freq in inv_freq]
        sin_values = [math.sin((pos / ROPE_SCALE) * freq) for freq in inv_freq]
        packed = cos_values + sin_values
        cache_f32.extend(packed)
        cache_bf16.extend(f32_to_bf16_bits(value) for value in packed)
    return tuple(cache_f32), tuple(cache_bf16)


def _apply_partial_rope(
    config: VectorConfig,
    x_norm_bf16: tuple[int, ...],
    heads: int,
    cos_sin_cache_bf16: tuple[int, ...],
) -> tuple[tuple[float, ...], tuple[int, ...]]:
    half = config.rotary_dim // 2
    out_f32: list[float] = []
    out_bf16: list[int] = []
    for token, pos in enumerate(POSITIONS):
        cache_base = pos * config.rotary_dim
        cos_values = [
            bf16_bits_to_f32(cos_sin_cache_bf16[cache_base + lane])
            for lane in range(half)
        ]
        sin_values = [
            bf16_bits_to_f32(cos_sin_cache_bf16[cache_base + half + lane])
            for lane in range(half)
        ]
        for head in range(heads):
            base = (token * heads + head) * config.head_dim
            x = [
                bf16_bits_to_f32(bits)
                for bits in x_norm_bf16[base : base + config.head_dim]
            ]
            y = list(x)
            for lane in range(half):
                x1 = x[lane]
                x2 = x[half + lane]
                cos = cos_values[lane]
                sin = sin_values[lane]
                y[lane] = x1 * cos - x2 * sin
                y[half + lane] = x2 * cos + x1 * sin
            out_f32.extend(y)
            out_bf16.extend(f32_to_bf16_bits(value) for value in y)
    return tuple(out_f32), tuple(out_bf16)


def _apply_output_gate(
    config: VectorConfig,
    gate_bf16: tuple[int, ...],
) -> tuple[tuple[int, ...], tuple[float, ...], tuple[int, ...]]:
    out_initial: list[int] = []
    out_f32: list[float] = []
    out_bf16: list[int] = []
    for token in range(NUM_TOKENS):
        for head in range(config.q_heads):
            for lane in range(config.head_dim):
                idx = (token * config.q_heads + head) * config.head_dim + lane
                current_bits = f32_to_bf16_bits(_attention_out_value(token, head, lane))
                current = bf16_bits_to_f32(current_bits)
                gate = bf16_bits_to_f32(gate_bf16[idx])
                value = current * _sigmoid(gate)
                out_initial.append(current_bits)
                out_f32.append(value)
                out_bf16.append(f32_to_bf16_bits(value))
    return tuple(out_initial), tuple(out_f32), tuple(out_bf16)


def generate_tensors(config: VectorConfig) -> tuple[TensorData, ...]:
    packed_q_gate, q_extracted, gate_extracted = _build_q_gate(config)
    k_input = _build_k(config)
    q_raw_weight = to_bf16_values(
        _q_raw_weight_value(lane) for lane in range(config.head_dim)
    )
    k_raw_weight = to_bf16_values(
        _k_raw_weight_value(lane) for lane in range(config.head_dim)
    )

    q_norm_f32, q_norm_bf16 = _gemma_rmsnorm(
        config, q_extracted, q_raw_weight, NUM_TOKENS * config.q_heads
    )
    k_norm_f32, k_norm_bf16 = _gemma_rmsnorm(
        config, k_input, k_raw_weight, NUM_TOKENS * config.kv_heads
    )
    cos_sin_f32, cos_sin_bf16 = _cos_sin_cache(config)
    q_rope_f32, q_rope_bf16 = _apply_partial_rope(
        config, q_norm_bf16, config.q_heads, cos_sin_bf16
    )
    k_rope_f32, k_rope_bf16 = _apply_partial_rope(
        config, k_norm_bf16, config.kv_heads, cos_sin_bf16
    )
    gate_out_initial, gated_out_f32, gated_out_bf16 = _apply_output_gate(
        config, gate_extracted
    )

    return (
        TensorData(
            "packed_q_gate",
            "bf16",
            (NUM_TOKENS, config.q_heads, 2, config.head_dim),
            packed_q_gate,
            "input",
            "packed_q_gate_extract",
            "q projection output laid out per head as [q, output_gate]",
        ),
        TensorData(
            "q_extracted",
            "bf16",
            (NUM_TOKENS, config.q_heads, config.head_dim),
            q_extracted,
            "expected",
            "packed_q_gate_extract",
            "contiguous q heads extracted from packed_q_gate",
        ),
        TensorData(
            "gate_extracted",
            "bf16",
            (NUM_TOKENS, config.q_heads, config.head_dim),
            gate_extracted,
            "expected",
            "packed_q_gate_extract",
            "contiguous pre-sigmoid output gate heads extracted from packed_q_gate",
        ),
        TensorData(
            "k_input",
            "bf16",
            (NUM_TOKENS, config.kv_heads, config.head_dim),
            k_input,
            "input",
            "qk_gemma_rmsnorm",
            "contiguous k heads before per-head q/k norm",
        ),
        TensorData(
            "q_norm_raw_weight",
            "bf16",
            (config.head_dim,),
            q_raw_weight,
            "input",
            "qk_gemma_rmsnorm",
            "raw q_norm weight; effective Gemma multiplier is bf16(raw) + 1",
        ),
        TensorData(
            "k_norm_raw_weight",
            "bf16",
            (config.head_dim,),
            k_raw_weight,
            "input",
            "qk_gemma_rmsnorm",
            "raw k_norm weight; effective Gemma multiplier is bf16(raw) + 1",
        ),
        TensorData(
            "q_norm_f32",
            "f32",
            (NUM_TOKENS, config.q_heads, config.head_dim),
            q_norm_f32,
            "reference",
            "qk_gemma_rmsnorm",
            "f32 reference before BF16 store rounding",
        ),
        TensorData(
            "q_norm_bf16",
            "bf16",
            (NUM_TOKENS, config.q_heads, config.head_dim),
            q_norm_bf16,
            "expected",
            "qk_gemma_rmsnorm",
            "BF16-stored q Gemma RMSNorm output",
        ),
        TensorData(
            "k_norm_f32",
            "f32",
            (NUM_TOKENS, config.kv_heads, config.head_dim),
            k_norm_f32,
            "reference",
            "qk_gemma_rmsnorm",
            "f32 reference before BF16 store rounding",
        ),
        TensorData(
            "k_norm_bf16",
            "bf16",
            (NUM_TOKENS, config.kv_heads, config.head_dim),
            k_norm_bf16,
            "expected",
            "qk_gemma_rmsnorm",
            "BF16-stored k Gemma RMSNorm output",
        ),
        TensorData(
            "positions",
            "i32",
            (NUM_TOKENS,),
            POSITIONS,
            "input",
            "partial_rope",
            "token positions chosen to exercise zero, small, boundary, and large angles",
        ),
        TensorData(
            "cos_sin_cache_f32",
            "f32",
            (max(POSITIONS) + 1, config.rotary_dim),
            cos_sin_f32,
            "reference",
            "partial_rope",
            "vLLM NeoX-style RoPE cache before dtype conversion, packed [cos, sin]",
        ),
        TensorData(
            "cos_sin_cache_bf16",
            "bf16",
            (max(POSITIONS) + 1, config.rotary_dim),
            cos_sin_bf16,
            "input",
            "partial_rope",
            "vLLM BF16 RoPE cache used by BF16 query/key tensors, packed [cos, sin]",
        ),
        TensorData(
            "q_rope_f32",
            "f32",
            (NUM_TOKENS, config.q_heads, config.head_dim),
            q_rope_f32,
            "reference",
            "partial_rope",
            "f32 partial RoPE reference from BF16 norm output and BF16 cos/sin cache",
        ),
        TensorData(
            "q_rope_bf16",
            "bf16",
            (NUM_TOKENS, config.q_heads, config.head_dim),
            q_rope_bf16,
            "expected",
            "partial_rope",
            "BF16-stored q after partial RoPE over lanes [0, 64)",
        ),
        TensorData(
            "k_rope_f32",
            "f32",
            (NUM_TOKENS, config.kv_heads, config.head_dim),
            k_rope_f32,
            "reference",
            "partial_rope",
            "f32 partial RoPE reference from BF16 norm output and BF16 cos/sin cache",
        ),
        TensorData(
            "k_rope_bf16",
            "bf16",
            (NUM_TOKENS, config.kv_heads, config.head_dim),
            k_rope_bf16,
            "expected",
            "partial_rope",
            "BF16-stored k after partial RoPE over lanes [0, 64)",
        ),
        TensorData(
            "gate_out_initial",
            "bf16",
            (NUM_TOKENS, config.q_heads, config.head_dim),
            gate_out_initial,
            "input",
            "output_gate",
            "attention output before in-place sigmoid gate application",
        ),
        TensorData(
            "gated_output_f32",
            "f32",
            (NUM_TOKENS, config.q_heads, config.head_dim),
            gated_out_f32,
            "reference",
            "output_gate",
            "f32 reference for out * sigmoid(gate)",
        ),
        TensorData(
            "gated_output_bf16",
            "bf16",
            (NUM_TOKENS, config.q_heads, config.head_dim),
            gated_out_bf16,
            "expected",
            "output_gate",
            "BF16-stored output after sigmoid gate application",
        ),
    )


def _tail_checks() -> list[dict[str, object]]:
    lanes = (64, 65, 127, 128, 255)
    return [
        {
            "tensor": "q_rope_bf16",
            "equals": "q_norm_bf16",
            "lanes": list(lanes),
            "all_tokens": True,
            "all_heads": True,
            "reason": "partial RoPE rotates only lanes [0, 64); tail lanes are pass-through",
        },
        {
            "tensor": "k_rope_bf16",
            "equals": "k_norm_bf16",
            "lanes": list(lanes),
            "all_tokens": True,
            "all_heads": True,
            "reason": "partial RoPE rotates only lanes [0, 64); tail lanes are pass-through",
        },
    ]


def build_manifest(
    config: VectorConfig, tensors: tuple[TensorData, ...]
) -> VectorManifest:
    tensor_specs = []
    for tensor in tensors:
        blob = tensor_bytes(tensor)
        tensor_specs.append(
            TensorSpec(
                tensor.name,
                tensor.dtype,
                tensor.shape,
                file=tensor.file_name,
                role=tensor.role,
                description=tensor.description,
                metadata={
                    "op": tensor.op,
                    "strides": list(tensor.strides or contiguous_strides(tensor.shape)),
                    "elements": tensor.element_count,
                    "bytes": len(blob),
                    "sha256": hashlib.sha256(blob).hexdigest(),
                },
            )
        )

    return VectorManifest(
        name="qwen36_full_attention_primitives",
        groups=("qwen36_semantics", "attention", "full_attention_primitives"),
        description=(
            "Deterministic primitive bundle for Qwen3.6 full-attention preparation: "
            "packed [q, gate] extraction, per-head q/k Gemma RMSNorm, "
            "NeoX-style partial RoPE, and sigmoid output gate."
        ),
        metadata={
            "case": "full_attention_primitives_v1",
            "dimensions": {
                "hidden_size": config.hidden_size,
                "num_tokens": NUM_TOKENS,
                "positions": list(POSITIONS),
                "q_heads": config.q_heads,
                "kv_heads": config.kv_heads,
                "head_dim": config.head_dim,
                "rotary_dim": config.rotary_dim,
                "q_hidden": config.q_hidden,
                "kv_hidden": config.kv_hidden,
                "q_proj_out": config.q_proj_out,
            },
            "params": {
                "rms_eps": config.rms_eps,
                "rope_theta": config.rope_theta,
                "rope_scale": ROPE_SCALE,
                "rope_interleave": False,
                "rope_style": "neox_non_interleaved",
                "output_gate_activation": "sigmoid",
            },
            "rounding": [
                "All BF16 tensors store raw IEEE BF16 bits as little-endian u16 values.",
                (
                    "Synthetic q, gate, k, output, and raw norm weights are "
                    "rounded to BF16 before reference math."
                ),
                (
                    "Gemma q/k RMSNorm accumulates variance in f32 from BF16 "
                    "inputs, uses effective_weight = bf16(raw_weight) + 1, "
                    "then rounds the stored output to BF16."
                ),
                (
                    "RoPE reads the BF16-stored norm output and the BF16 vLLM "
                    "cos/sin cache as f32, rotates lanes [0, 64), copies lanes "
                    f"[{config.rotary_dim}, {config.head_dim}), then rounds the stored output to BF16."
                ),
                (
                    "Output gating reads BF16 gate/out values as f32, computes "
                    "out * sigmoid(gate), then rounds the stored output to BF16."
                ),
            ],
            "source_semantics": [
                "3pty/vllm/vllm/model_executor/models/qwen3_next.py::_project_qkv_gate",
                (
                    "3pty/vllm/vllm/model_executor/layers/"
                    "fused_qk_norm_rope.py::fused_qk_rmsnorm_rope_gate"
                ),
                (
                    "3pty/vllm/vllm/model_executor/layers/rotary_embedding/"
                    "base.py::RotaryEmbeddingBase"
                ),
            ],
            "tail_lane_checks": _tail_checks(),
        },
        tensors=tuple(tensor_specs),
    )


def build_attention_artifact(
    config: VectorConfig,
) -> tuple[
    VectorManifest,
    dict[str, tuple[int, ...] | tuple[float, ...]],
]:
    tensors = generate_tensors(config)
    manifest = build_manifest(config, tensors)
    return manifest, {tensor.name: tensor.data for tensor in tensors}


def write_bundle(
    config: VectorConfig, output_dir: Path = DEFAULT_OUTPUT, *, force: bool = False
) -> VectorManifest:
    manifest, tensors = build_attention_artifact(config)
    output_dir.mkdir(parents=True, exist_ok=True)

    output_files = [output_dir / MANIFEST_FILE]
    output_files.extend(output_dir / tensor.file for tensor in manifest.tensors)
    if not force:
        existing = [path for path in output_files if path.exists()]
        if existing:
            names = ", ".join(str(path) for path in existing[:4])
            if len(existing) > 4:
                names += f", ... ({len(existing)} files)"
            raise FileExistsError(
                f"refusing to overwrite existing vector artifacts: {names}"
            )

    write_artifact(output_dir, manifest, tensors)
    return manifest
