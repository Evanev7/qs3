"""Offline vLLM comparison for qs3's fixed 102-token core workload."""

import argparse
import importlib.metadata
import json
import statistics
import time
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument("--model", required=True)
parser.add_argument("--eager", action="store_true")
parser.add_argument("--samples", type=int, default=32)
parser.add_argument("--context", type=int, default=102)
parser.add_argument("--output", required=True)
args = parser.parse_args()
assert args.samples > 0 and args.context >= 102

from tokenizers import Tokenizer
from vllm import LLM, SamplingParams

prompt = (
    "<|im_start|>system\n"
    "You are a concise technical assistant. Explain systems accurately, distinguish measured facts from estimates, and avoid unnecessary jargon.\n"
    "<|im_end|>\n"
    "<|im_start|>user\n"
    "Explain how a transformer language model turns a prompt into the next token. Cover tokenization, embeddings, attention, feed-forward layers, normalization, logits, and sampling. Distinguish prompt processing from autoregressive decoding, and mention why memory bandwidth matters. Use plain language and keep the answer under 250 words.\n"
    "<|im_end|>\n"
    "<|im_start|>assistant\n"
)
ids = (
    Tokenizer.from_file(str(Path(args.model) / "tokenizer.json"))
    .encode(prompt, add_special_tokens=False)
    .ids
)
fingerprint = 0xCBF29CE484222325
for token in ids:
    for byte in token.to_bytes(4, "little", signed=True):
        fingerprint = ((fingerprint ^ byte) * 0x100000001B3) & ((1 << 64) - 1)
assert len(ids) == 102 and fingerprint == 0x6CA602A4CC15238D, (
    len(ids),
    hex(fingerprint),
)
ids = (ids * ((args.context + 101) // 102))[: args.context]
prompt_fingerprint = 0xCBF29CE484222325
for token in ids:
    for byte in token.to_bytes(4, "little", signed=True):
        prompt_fingerprint = ((prompt_fingerprint ^ byte) * 0x100000001B3) & (
            (1 << 64) - 1
        )

started = time.perf_counter()
llm = LLM(
    model=args.model,
    dtype="bfloat16",
    quantization=None,
    skip_tokenizer_init=True,
    tensor_parallel_size=1,
    max_num_seqs=1,
    max_model_len=args.context + args.samples + 5,
    enable_prefix_caching=False,
    enable_chunked_prefill=False,
    enforce_eager=args.eager,
    gpu_memory_utilization=0.85,
    kv_cache_memory_bytes=512 << 20,
    mamba_cache_dtype="auto",
    mamba_ssm_cache_dtype="auto",
    stream_interval=1,
    limit_mm_per_prompt={"image": 0, "video": 0},
    seed=0,
)
setup_ms = (time.perf_counter() - started) * 1000
engine = llm.llm_engine
from vllm.model_executor.layers.mamba.mamba_utils import MambaStateDtypeCalculator

cache = engine.vllm_config.cache_config
state_dtypes = MambaStateDtypeCalculator.gated_delta_net_state_dtype(
    engine.vllm_config.model_config.dtype,
    cache.mamba_cache_dtype,
    cache.mamba_ssm_cache_dtype,
)
print(
    json.dumps({"resolved_gdn_state_dtypes": [str(dtype) for dtype in state_dtypes]}),
    flush=True,
)


def request(index, decode_steps):
    # vLLM emits token 1 at prefill. An extra output is needed to time exactly
    # decode_steps forwards, matching qs3's processing of each generated token.
    params = SamplingParams(
        temperature=0, max_tokens=decode_steps + 1, ignore_eos=True, detokenize=False
    )
    engine.add_request(str(index), {"prompt_token_ids": ids}, params)
    durations, token_ids, empty_steps = [], [], 0
    before = time.perf_counter()
    while engine.has_unfinished_requests():
        outputs = engine.step()
        if not outputs:
            empty_steps += 1
            continue
        delivered = time.perf_counter()
        elapsed_ms = (delivered - before) * 1000
        assert len(outputs) == 1 and len(outputs[0].outputs) == 1
        next_ids = list(outputs[0].outputs[0].token_ids)
        assert len(next_ids) == len(token_ids) + 1, (len(token_ids), len(next_ids))
        token_ids = next_ids
        durations.append(elapsed_ms)
        before = delivered
    assert len(durations) == decode_steps + 1
    print(
        json.dumps(
            {"request": index, "empty_steps": empty_steps, "outputs": len(durations)}
        ),
        flush=True,
    )
    return durations, token_ids


prefill = [request(i, 0)[0][0] for i in range(7)]
steps, generated = request(100, 4 + args.samples)
decode = steps[5:]
assert len(decode) == args.samples
result = {
    "metadata": {
        p: importlib.metadata.version(p) for p in ["vllm", "torch", "transformers"]
    },
    "model": args.model,
    "cache_config": {
        key: str(getattr(engine.vllm_config.cache_config, key, None))
        for key in [
            "cache_dtype",
            "mamba_cache_dtype",
            "mamba_ssm_cache_dtype",
            "enable_prefix_caching",
        ]
    },
    "execution": {
        "precision": "bf16",
        "mamba_cache_dtype": "auto",
        "mamba_ssm_cache_dtype": "auto",
        "resolved_gdn_state_dtypes": [str(dtype) for dtype in state_dtypes],
        "enforce_eager": args.eager,
        "sampling": "greedy",
        "prefix_caching": False,
        "chunked_prefill": False,
        "detokenize": False,
        "stream_interval": 1,
        "timing": "wall time between token deliveries including empty engine steps",
        "kv_cache_memory_bytes": 512 << 20,
    },
    "prompt": {
        "tokens": len(ids),
        "base_token_id_fnv1a": f"{fingerprint:016x}",
        "token_id_fnv1a": f"{prompt_fingerprint:016x}",
        "construction": "repeat_base_token_ids",
    },
    "setup_ms": setup_ms,
    "prefill": {
        "warmup_ms": prefill[:2],
        "sample_ms": prefill[2:],
        "p50_ms": statistics.median(prefill[2:]),
    },
    "decode": {
        "warmup_ms": steps[1:5],
        "sample_ms": decode,
        "samples": len(decode),
        "context_start": len(ids) + 4,
        "context_end": len(ids) + 4 + len(decode),
        "p50_ms": statistics.median(decode),
        "tokens_per_second": len(decode) * 1000 / sum(decode),
    },
    "generated_tokens": generated,
}
Path(args.output).write_text(json.dumps(result, indent=2) + "\n")
print(json.dumps(result))
