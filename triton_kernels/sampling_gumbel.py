# Triton JIT call/annotation types are not Python typing types.
# ty: ignore[invalid-argument-type, invalid-type-form]
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# Adapted from vLLM 51a99565c398c8320de8131e07731c75c52eb87c,
# vllm/v1/worker/gpu/sample/gumbel.py. See LICENSE.vllm.
# Changes: one row, AOT arguments, preprocessed logits, invalid-row sentinel.

import triton
import triton.language as tl
from triton.language.extra.cuda import libdevice


@triton.jit
def kernel(
    logits: tl.tensor,
    original: tl.tensor,
    local_max: tl.tensor,
    local_ids: tl.tensor,
    position: tl.tensor,
    seed: tl.uint64,
    vocab: tl.int32,
    temperature: tl.float32,
    BLOCK: tl.constexpr,
):
    block_idx = tl.program_id(0)
    block = block_idx * BLOCK + tl.arange(0, BLOCK)
    mask = block < vocab
    values = tl.load(logits + block, mask, other=-float("inf"))
    raw = tl.load(original + block, mask, other=-float("inf"))
    overflow = (tl.abs(raw) < float("inf")) & (
        tl.abs(raw / temperature) == float("inf")
    )
    invalid = (
        tl.sum((libdevice.isnan(raw) | (raw == float("inf")) | overflow).to(tl.int32))
        > 0
    )
    pos = tl.load(position)
    gumbel_seed = tl.randint(seed, pos)
    u = tl.maximum(tl.rand(gumbel_seed, block), 4.6566127342e-10)
    noise = -tl.log(-libdevice.log1p(-u))
    values = tl.where(mask, values + noise, -float("inf"))
    value, idx = tl.max(values, axis=0, return_indices=True)
    tl.store(local_max + block_idx, value)
    tl.store(local_ids + block_idx, tl.where(invalid, -1, block_idx * BLOCK + idx))
