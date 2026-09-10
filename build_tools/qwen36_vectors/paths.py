from __future__ import annotations

from pathlib import Path

TOOL_ROOT = Path(__file__).resolve().parent

DEFAULT_VECTOR_ROOT = Path("build/vectors/qwen36_semantics")
DEFAULT_ORACLE_ROOT = TOOL_ROOT / "oracle_hashes"

VECTOR_GROUPS: tuple[str, ...] = (
    "full_attention_block",
    "full_attention_primitives",
    "gdn_decoder_layer",
    "gdn_post_conv_prep",
    "model_logits",
    "moe_shared_expert",
    "norms",
)
