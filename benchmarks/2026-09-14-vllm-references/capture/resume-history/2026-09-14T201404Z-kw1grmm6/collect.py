"""Capture pinned Qwen BF16 raw logits with one vLLM load; not a timing benchmark."""

import argparse
import hashlib
import importlib.metadata
import json
import os
import shutil
import time
from pathlib import Path

VOCAB = 248320
TOKENIZER_VOCABS = {"qwen3.6-35b-a3b": 248070, "qwen3.6-27b": 248070,
                    "qwen3.8-27b": 248077}
WORKLOADS = ((102, 32), (500, 200), (4000, 800))
WARMUP_STEPS = 4


def write_json(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def fingerprint(ids):
    value = 0xCBF29CE484222325
    for token in ids:
        for byte in token.to_bytes(4, "little", signed=True):
            value = ((value ^ byte) * 0x100000001B3) & ((1 << 64) - 1)
    return f"{value:016x}"


def make_workloads(base_ids, reference_dir, model_spec):
    tokenizer_vocab = TOKENIZER_VOCABS[model_spec["name"]]
    assert len(base_ids) == 102 and fingerprint(base_ids) == "6ca602a4cc15238d"
    if reference_dir is not None:
        assert model_spec["name"] == "qwen3.6-35b-a3b", "saved benchmark continuations are 35B-only"
    # The real-loader smoke test uses this tiny prompt. Keep all four predicted
    # rows so each model gets a compact, independent prefill/decode reference.
    workloads = [dict(name="4-4", prompt_ids=[1, 2, 3, 4], context=4, samples=4,
                      decode_steps=3, warmup_steps=0, forced_ids=None)]
    for context, samples in WORKLOADS:
        prompt = (base_ids * ((context + 101) // 102))[:context]
        item = dict(name=f"{context}-{samples}", prompt_ids=prompt,
                    context=context, samples=samples,
                    decode_steps=WARMUP_STEPS + samples,
                    warmup_steps=WARMUP_STEPS, forced_ids=None)
        if reference_dir is not None and context in (102, 500):
            source = reference_dir / f"{item['name']}.json"
            reference = json.loads(source.read_text())
            assert Path(reference["model"]).name == model_spec["rev"]
            assert reference["prompt"]["tokens"] == context
            assert reference["prompt"]["token_id_fnv1a"] == fingerprint(prompt)
            assert reference["decode"]["samples"] == samples
            assert reference["execution"]["precision"] == "bf16"
            assert reference["execution"]["sampling"] == "greedy"
            assert reference["execution"]["resolved_gdn_state_dtypes"] == [
                "torch.bfloat16", "torch.float32"]
            ids = reference["generated_tokens"]
            assert len(ids) == item["decode_steps"] + 1
            assert all(type(t) is int and 0 <= t < tokenizer_vocab for t in ids)
            item.update(forced_ids=ids[:-1], reference_generated_ids=ids,
                        source=str(source), source_sha256=hashlib.sha256(source.read_bytes()).hexdigest())
        workloads.append(item)
    return workloads


def install_capture(model, output_dir, workload, tokenizer_vocab):
    import torch

    original = model.compute_logits
    records = []
    output_dir = Path(output_dir)
    model._qs3_capture = (original, records)

    def compute(hidden_states, *args, **kwargs):
        logits = original(hidden_states, *args, **kwargs)
        assert logits is not None and tuple(logits.shape) == (1, VOCAB)
        step = len(records)
        assert step <= workload["decode_steps"], "unexpected extra forward"
        # clone is essential: CPU tensors in local tests must not alias the
        # sampler tensor, and the persisted data must precede every mutation.
        raw = logits[0].detach().float().cpu().contiguous().clone()
        assert bool(torch.isfinite(raw).all()), f"non-finite logits at step {step}"
        argmax = int(raw.argmax())
        assert argmax < tokenizer_vocab, f"padded token selected at step {step}"
        forced = workload["forced_ids"]
        chosen = forced[step] if forced is not None and step < len(forced) else argmax
        values, ids = torch.topk(raw, 20)
        lse = float(torch.logsumexp(raw.double(), dim=0))
        valid_lse = float(torch.logsumexp(raw[:tokenizer_vocab].double(), dim=0))
        filename = f"logits-{step:04d}.f32"
        payload = raw.numpy().astype("<f4", copy=False).tobytes()
        assert len(payload) == VOCAB * 4
        with (output_dir / filename).open("xb") as output:
            output.write(payload)
        row = dict(step=step, context_tokens=workload["context"] + step,
                   phase="prefill" if step == 0 else "decode",
                   file=filename, sha256=hashlib.sha256(payload).hexdigest(),
                   argmax=argmax, top_ids=ids.tolist(), top_values=values.tolist(),
                   top2_margin=float(values[0] - values[1]), logsumexp=lse,
                   addressable_logsumexp=valid_lse, logits_dtype=str(logits.dtype),
                   selected_id=chosen, selected_logit=float(raw[chosen]))
        if step < workload["decode_steps"]:
            row.update(forced_id=chosen, forced_logit=float(raw[chosen]),
                       forced_nll=lse - float(raw[chosen]))
        # Append a small journal so an interrupted capture remains inspectable.
        with (output_dir / "records.jsonl").open("a") as journal:
            journal.write(json.dumps(row, allow_nan=False) + "\n")
        records.append(row)
        # Force the chosen continuation only AFTER retaining the raw logits.
        # This also removes any ambiguity from sampler defaults or argmax ties.
        logits.fill_(float("-inf"))
        logits[0, chosen] = 0
        if step % 100 == 0:
            print(f"{workload['name']}: captured step {step}/{workload['decode_steps']}", flush=True)
        return logits

    model.compute_logits = compute
    return {"class": type(model).__name__}


def finish_capture(model):
    original, records = model._qs3_capture
    model.compute_logits = original
    del model._qs3_capture
    return records


def finish_workload(workload, generated, records, model_spec):
    assert len(generated) == len(records) == workload["decode_steps"] + 1
    assert [r["step"] for r in records] == list(range(len(records)))
    assert generated == [r["selected_id"] for r in records]
    if workload["forced_ids"] is not None:
        assert generated[:-1] == workload["forced_ids"]
    # Same input schema as same_prefix_scores/qs3_test.rs.
    spec = dict(model_revision=model_spec["rev"], model_name=model_spec["name"],
                model_repo=model_spec["repo"], prompt_ids=workload["prompt_ids"],
                forced_ids=generated[:-1], capture_steps=list(range(len(records))),
                source="vllm_correctness/collect.py")
    return spec


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--model-spec", type=Path, required=True,
                        help="pinned source and text config exported by run.py")
    parser.add_argument("--output", type=Path, required=True,
                        help="new directory; existing captures are never overwritten")
    parser.add_argument("--reference-dir", type=Path,
                        help="reuse 102-32.json and 500-200.json benchmark continuations")
    args = parser.parse_args()
    assert os.environ.get("VLLM_USE_V2_MODEL_RUNNER") == "0", (
        "raw compute_logits capture requires VLLM_USE_V2_MODEL_RUNNER=0")
    model_spec = json.loads(args.model_spec.read_text())
    tokenizer_vocab = TOKENIZER_VOCABS[model_spec["name"]]
    assert args.model.name == model_spec["rev"], "snapshot does not match selected revision"
    text_config = json.loads((args.model / "config.json").read_text())["text_config"]
    assert text_config == model_spec["text_config"], "snapshot text config differs from models/"
    assert text_config["vocab_size"] == VOCAB
    tokenizer = json.loads((args.model / "tokenizer.json").read_text())
    token_ids = set(tokenizer["model"]["vocab"].values())
    token_ids.update(token["id"] for token in tokenizer["added_tokens"])
    assert token_ids == set(range(tokenizer_vocab)), "unexpected addressable tokenizer IDs"
    base = Path(__file__).resolve().parent.parent / "same_prefix_scores/base_ids.json"
    workloads = make_workloads(json.loads(base.read_text()), args.reference_dir, model_spec)
    args.output.mkdir(parents=True, exist_ok=False)
    shutil.copy2(__file__, args.output / "collector.py")
    write_json(args.output / "model-spec.json", model_spec)
    shutil.copy2(base, args.output / "base_ids.json")
    assets = {}
    for name in ("config.json", "generation_config.json", "tokenizer.json", "tokenizer_config.json"):
        source = args.model / name
        shutil.copy2(source, args.output / name)
        assets[name] = hashlib.sha256(source.read_bytes()).hexdigest()
    for workload in workloads:
        if "source" in workload:
            shutil.copy2(workload["source"], args.output / f"source-{workload['name']}.json")
    write_json(args.output / "plan.json", workloads)

    import torch
    from vllm import LLM, SamplingParams
    from vllm import envs as vllm_envs
    from vllm.model_executor.layers.mamba.mamba_utils import MambaStateDtypeCalculator
    assert vllm_envs.VLLM_USE_V2_MODEL_RUNNER is False, "vLLM did not select model runner v1"

    options = dict(model=str(args.model), dtype="bfloat16", quantization=None,
                   skip_tokenizer_init=True, tensor_parallel_size=1, max_num_seqs=1,
                   max_model_len=max(w["context"] + w["decode_steps"] + 1 for w in workloads),
                   enable_prefix_caching=False, enable_chunked_prefill=False,
                   enforce_eager=False, gpu_memory_utilization=0.85,
                   kv_cache_memory_bytes=(512 if model_spec["name"] == "qwen3.6-35b-a3b" else 1024) << 20,
                   mamba_cache_dtype="auto",
                   mamba_ssm_cache_dtype="float32", stream_interval=1,
                   limit_mm_per_prompt={"image": 0, "video": 0}, seed=0)
    started = time.perf_counter()
    llm = LLM(**options)
    config = llm.llm_engine.vllm_config
    cache = config.cache_config
    state_dtypes = MambaStateDtypeCalculator.gated_delta_net_state_dtype(
        config.model_config.dtype, cache.mamba_cache_dtype, cache.mamba_ssm_cache_dtype)
    assert [str(x) for x in state_dtypes] == ["torch.bfloat16", "torch.float32"]
    metadata = dict(model=str(args.model), model_revision=model_spec["rev"],
                    model_name=model_spec["name"], model_repo=model_spec["repo"], asset_sha256=assets,
                    container_image=os.environ.get("QS3_REFERENCE_IMAGE"),
                    versions={p: importlib.metadata.version(p) for p in
                              ("vllm", "torch", "transformers", "flashinfer-python")},
                    engine_options=options, resolved_vllm_config=str(config),
                    model_runner="v1", capture_hook="model.compute_logits",
                    resolved_gdn_state_dtypes=[str(x) for x in state_dtypes],
                    cuda=torch.version.cuda, gpu=torch.cuda.get_device_name(),
                    setup_seconds=time.perf_counter() - started,
                    logits_format={"dtype": "little-endian float32", "shape": [VOCAB],
                                   "addressable_token_count": tokenizer_vocab},
                    timing="instrumented correctness capture; not a performance baseline")
    write_json(args.output / "metadata.json", metadata)

    completed = []
    for workload in workloads:
        run_dir = args.output / workload["name"]
        vllm_dir = run_dir / "vllm"
        vllm_dir.mkdir(parents=True)
        model_info = llm.apply_model(lambda model: install_capture(model, vllm_dir, workload, tokenizer_vocab))
        try:
            outputs = llm.generate({"prompt_token_ids": workload["prompt_ids"]},
                SamplingParams(temperature=0, top_p=1, top_k=-1, min_p=0,
                               presence_penalty=0, frequency_penalty=0, repetition_penalty=1,
                               max_tokens=workload["decode_steps"] + 1,
                               ignore_eos=True, detokenize=False), use_tqdm=False)
        finally:
            records, = llm.apply_model(finish_capture)
        assert len(outputs) == 1 and len(outputs[0].outputs) == 1
        generated = list(outputs[0].outputs[0].token_ids)
        spec = finish_workload(workload, generated, records, model_spec)
        write_json(run_dir / "input.json", spec)
        scores = dict(input=spec, records=records, generated_ids=generated,
                      protocol="one prefill then token-by-token decode; raw logits before forcing/sampling",
                      model_info=model_info, workload=workload, metadata="../../metadata.json",
                      original_greedy_difference_steps=[r["step"] for r, token in
                          zip(records, workload.get("reference_generated_ids", generated))
                          if r["argmax"] != token])
        write_json(vllm_dir / "scores.json", scores)
        if workload["name"] == "4-4":
            write_json(args.output / "small-reference.json", dict(
                model_name=model_spec["name"], model_repo=model_spec["repo"],
                model_revision=model_spec["rev"], container_image=metadata["container_image"],
                versions=metadata["versions"], metadata="metadata.json",
                prompt_ids=workload["prompt_ids"], generated_ids=generated,
                forced_ids=generated[:-1], rows=[dict(
                    step=row["step"], phase=row["phase"], argmax=row["argmax"],
                    top_ids=row["top_ids"][:8], top_values=row["top_values"][:8],
                    top2_margin=row["top2_margin"], logits_dtype=row["logits_dtype"],
                    file=f"4-4/vllm/{row['file']}", sha256=row["sha256"],
                ) for row in records]))
        completed.append(workload["name"])
        write_json(args.output / "completed.json", completed)
        print(json.dumps({"completed": workload["name"], "rows": len(records),
                          "original_greedy_difference_steps": scores["original_greedy_difference_steps"]}), flush=True)
    write_json(args.output / "SUCCESS.json", {"workloads": completed})


if __name__ == "__main__":
    main()
