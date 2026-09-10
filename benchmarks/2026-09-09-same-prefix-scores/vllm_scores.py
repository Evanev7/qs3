"""Diagnostic only: retain normal model execution, force a supplied decode prefix."""

import argparse
import importlib.metadata
import json
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument("--model", required=True)
parser.add_argument("--input", type=Path, required=True)
parser.add_argument("--output", type=Path, required=True)
args = parser.parse_args()
spec = json.loads(args.input.read_text())
prompt = spec["prompt_ids"]
forced = spec["forced_ids"]
capture_steps = set(spec["capture_steps"])
assert prompt and forced and all(0 <= x < 248070 for x in prompt + forced)
assert all(0 <= x <= len(forced) for x in capture_steps)
args.output.mkdir(parents=True, exist_ok=True)

import torch
from vllm import LLM, SamplingParams

llm = LLM(
    model=args.model,
    dtype="bfloat16",
    quantization=None,
    skip_tokenizer_init=True,
    tensor_parallel_size=1,
    max_num_seqs=1,
    max_model_len=len(prompt) + len(forced) + 1,
    enable_prefix_caching=False,
    enable_chunked_prefill=False,
    enforce_eager=False,
    gpu_memory_utilization=0.85,
    kv_cache_memory_bytes=512 << 20,
    mamba_cache_dtype="auto",
    mamba_ssm_cache_dtype="auto",
    stream_interval=1,
    limit_mm_per_prompt={"image": 0, "video": 0},
    seed=0,
)


def install(model):
    # compute_logits runs after the model forward, outside the captured forward.
    # Preserve the unmodified output before forcing only the sampler's choice.
    original = model.compute_logits
    records = []
    model._qs3_score_records = records

    def compute(hidden_states, *a, **kw):
        logits = original(hidden_states, *a, **kw)
        assert logits is not None and logits.ndim == 2 and logits.shape[0] == 1
        step = len(records)
        assert step <= len(forced)
        raw = logits[0].detach().float().cpu().contiguous()
        assert raw.numel() == 248320 and torch.isfinite(raw).all()
        values, ids = torch.topk(raw, 20)
        row = {
            "step": step,
            "argmax": int(raw.argmax()),
            "top_ids": ids.tolist(),
            "top_values": values.tolist(),
            "logsumexp": float(torch.logsumexp(raw.double(), dim=0)),
            "logits_dtype": str(logits.dtype),
        }
        if step < len(forced):
            row["forced_id"] = forced[step]
            row["forced_logit"] = float(raw[forced[step]])
            row["forced_nll"] = row["logsumexp"] - row["forced_logit"]
        if step in capture_steps:
            filename = f"logits-{step:04d}.f32"
            raw.numpy().astype("<f4", copy=False).tofile(args.output / filename)
            row["file"] = filename
        records.append(row)
        if step < len(forced):
            logits.fill_(float("-inf"))
            logits[0, forced[step]] = 0
        return logits

    model.compute_logits = compute
    gates = [
        (name, module)
        for name, module in model.named_modules()
        if name.endswith(".mlp.gate") and hasattr(module, "weight")
    ]
    assert len(gates) == 40
    name, gate = gates[0]
    probe = torch.zeros(
        (1, gate.weight.shape[1]), device=gate.weight.device, dtype=gate.weight.dtype
    )
    output = gate(probe)
    if isinstance(output, tuple):
        output = output[0]
    return {
        "model_class": type(model).__name__,
        "router_weight_dtypes": [(n, str(m.weight.dtype)) for n, m in gates],
        "router_probe": {
            "module": name,
            "class": type(gate).__name__,
            "input_dtype": str(probe.dtype),
            "output_dtype": str(output.dtype),
        },
    }


model_info = llm.apply_model(install)
outputs = llm.generate(
    {"prompt_token_ids": prompt},
    SamplingParams(
        temperature=0, max_tokens=len(forced) + 1, ignore_eos=True, detokenize=False
    ),
    use_tqdm=False,
)
generated = list(outputs[0].outputs[0].token_ids)
assert generated[: len(forced)] == forced and len(generated) == len(forced) + 1
(records,) = llm.apply_model(lambda model: model._qs3_score_records)
assert len(records) == len(forced) + 1
result = {
    "model": args.model,
    "model_info": model_info,
    "versions": {
        p: importlib.metadata.version(p) for p in ["vllm", "torch", "transformers"]
    },
    "protocol": "one prefill then forced token-by-token decode; logits captured before forcing",
    "input": spec,
    "generated_ids": generated,
    "records": records,
}
(args.output / "scores.json").write_text(json.dumps(result, indent=2) + "\n")
print(json.dumps({"steps": len(records), "output": str(args.output)}), flush=True)
