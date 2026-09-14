# Pinned vLLM references — 2026-09-14

Official vLLM 0.29.0 is pinned in [pins.json](pins.json), including its ARM64
Docker digest and the three BF16 model revisions. The matching vendored source
pin is `98dff2a81d747d1dba01a47f939f48c3526d4206`. Future core vLLM benchmark
runs use the same image pin and model runner v1; the historical benchmark
results retain their original pins. No refreshed core timing result is claimed.

The capture records full, unmodified float32 logit rows before sampling or
forcing tokens. Each model has four workloads:

| Prompt / generated workload | Captured rows | Continuation |
| --- | ---: | --- |
| IDs `[1,2,3,4]`, four predictions | 4 | Own greedy |
| 102 / 32 (+ four warmup decode steps) | 37 | Saved 35B baseline; own greedy for 27B |
| 500 / 200 (+ four warmup decode steps) | 205 | Saved 35B baseline; own greedy for 27B |
| 4000 / 800 (+ four warmup decode steps) | 805 | Own greedy |

Each row has 248320 little-endian float32 values. The pinned 3.6 tokenizers
have 248070 addressable IDs; 3.8 has 248077, including seven additional audio
special tokens at IDs 248070–248076. The collector validates each exact ID set.
Completed and verified: 1051 rows/model, 3153 rows across all three.
These are real-model references, separate from `qwen36_vectors` synthetic
operator-composition fixtures. They do not establish qs3 runtime correctness
until qs3 is replayed on the recorded prefixes.

The collector explicitly selects model runner v1 because it hooks
`model.compute_logits`. Model forward execution may use vLLM CUDA graphs;
this is instrumented reference collection, not a performance benchmark or a
change to qs3 execution. BF16 activations/weights and FP32 recurrent GDN state
remain explicit. The 35B reference has 512 MiB KV workspace; 27B uses 1 GiB.
The first 27B attempt showed that 512 MiB only supports 4704 tokens after vLLM's
hybrid-cache layout overhead, short of the 4805-token configured maximum.
That attempt is retained in the run's `failed/` directory.

Capture location: `.prototypes/out/vllm-correctness-2026-09-14T194223Z-jn1cisug`.
A resumable runner preserves completed models, checks the image/model pins,
and archives incomplete attempts before retrying. Resume source versions and
configuration changes are retained in `resume-history/`.

Verification completed on sp10 at 20:28:14 UTC. It checked all file sizes and
SHA256 hashes, completion markers, model/image identities, raw-logit journal
consistency, replay token bookkeeping and resume provenance: 3264 files,
including 3,131,811,840 bytes of raw logits. The manifest digest is
`dae278dbeb39f91f799d13da9170e18731b2e43aa66de47d93ad5c618112065b`.
The compact export retains per-model metadata, small references, scores,
replay inputs, exact capture sources and hashes of all omitted binary rows.

Full binary rows remain in the ignored capture directory on sp10. The initial local raw-file transfer was
stopped after capture because it was slow; the local raw directory is partial.
Verification runs against the complete sp10 copy before compact export.


[Verification summary](summary.json) and [compact capture](capture/) contain
all three models. Small-prompt references are:

| Model | Greedy IDs for `[1,2,3,4]` | Reference |
| --- | --- | --- |
| 3.6-35B-A3B | `[5,6,24218,10]` | [JSON](capture/data/qwen3.6-35b-a3b/small-reference.json) |
| 3.6-27B | `[5,9,0,31]` | [JSON](capture/data/qwen3.6-27b/small-reference.json) |
| 3.8-27B | `[5,0,31,46474]` | [JSON](capture/data/qwen3.8-27b/small-reference.json) |

All 108 retained files were checked against the original manifest after transfer.
Restore [capture-sources.tar.gz](capture-sources.tar.gz) into the repository root
to recover the prototype, its inputs and recipes. The model-local `collector.py`
copies in the capture record the exact version used for each completed model.
