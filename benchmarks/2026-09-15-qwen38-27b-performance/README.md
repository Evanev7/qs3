# Qwen3.8-27B BF16 performance baseline — 2026-09-15

Steady decode is effectively tied with pinned vLLM at about **4.5 tok/s** on
both routine workloads. vLLM prefill is faster. This records the BF16 starting
point for NVFP4 work; no BF16 kernel tuning was performed.

## Measurements

| Prompt / decode samples | Runtime | Prefill p50 ms | Decode p50 ms | Decode p95 ms | Decode tok/s |
| --- | --- | ---: | ---: | ---: | ---: |
| 102 / 32 | qs3 eager | 295.535 | 221.605 | 222.471 | 4.509 |
| 102 / 32 | vLLM graphs | 267.016 | 221.786 | 222.894 | 4.508 |
| 1024 / 256 | qs3 eager | 1026.327 | 222.602 | 223.903 | 4.491 |
| 1024 / 256 | vLLM graphs | 796.080 | 222.820 | 223.981 | 4.487 |

The measured decode throughput differences are below 0.1%; this single baseline
does not establish a decode performance advantage for either runtime. Throughput
is the reciprocal of mean latency, not p50. [summary.json](summary.json) is
recomputed from all raw latency samples by [summarize.py](summarize.py).

Both workloads have matching prompt fingerprints and decode context ranges.
They use independent greedy continuations. The first generated-token differences
are at index 18 for 102/32 and index 3 for 1024/256. The short case shares its
first 18 tokens; the sustained case shares its first three. Later performance
samples therefore follow different token prefixes. The separate
[forced-prefix correctness capture](../2026-09-15-qwen38-27b-correctness/README.md)
records numerical differences; these timings do not resolve them.

## Configuration and timing boundaries

- Host `sp10`, NVIDIA GB10, driver 580.142. qs3 uses CUDA runtime 13.0 and
  `rustc 1.98.1 (48a229cea 2026-09-01)`, release build.
- Both use `Qwen/Qwen3.8-27B` BF16 snapshot
  `1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0`, BF16 activations/KV/conv state,
  FP32 GDN recurrent state, batch one and greedy sampling.
- qs3 uses its current eager schedule, pinned-upload final device weights,
  prepared cuBLASLt dense projections, local GDN, FlashInfer attention, and
  Triton LM-head/GDN-QKV decode projections. Its LM-head output is FP32.
- vLLM 0.29.0 uses model runner v1 and default CUDA graphs, with prefix caching
  and chunked prefill disabled. Torch is 2.13.0+cu130; Transformers is 5.16.1.
  The image is pinned to
  `sha256:c2914767605584b6d8f45686b82de173ecc99e781897aa3d0a66dacd72c51ae1`.
  vLLM uses BF16 LM-head output. Full resolved configurations and image
  inspection are retained in the capture.
- vLLM reserves 1 GiB KV cache. qs3 sizes its cache to the workload and records
  its separate 64 MiB linear/attention workspace capacities in each result.
- Each runtime/workload loads the model separately. Execution order is qs3
  102/32, vLLM 102/32, qs3 1024/256, then vLLM 1024/256. Loading, compilation
  and graph preparation are outside the steady-state samples. File-cache
  residency is uncontrolled; recorded setup times are not a loader comparison.
- Each runtime has two fresh-prefill warmups and five prefill samples, followed
  by four decode warmups and 32 or 256 measured decode forwards. Longer prompts
  repeat the exact base token IDs. No profiler or raw-logit collector runs
  alongside the timing samples.
- qs3 uses the existing benchmark's unprofiled measurement pass and wall-clock
  runner calls. vLLM measures wall time between token deliveries, including
  empty engine steps. Its prefill includes first-token delivery; qs3 prefill
  computes/samples the next token internally but returns no generated token for
  `max_new_tokens=0`. vLLM records one extra predicted token to time the same
  number of decode forwards; token comparison omits that final prediction.

## Reproduction and validation

Run the gitignored prototype workflow with `models/config.nix` selecting 3.8-27B:

```sh
QS3_PROTOTYPE_OUTPUT_PREFIX=performance-38-27b ./remote.sh prototype performance-baseline-38-27b
```

`remote.sh` was not modified. The added prototype recipe runs
`.prototypes/performance_baseline/run.py`; its exact source is saved as
[capture/run.py](capture/run.py), alongside the vLLM benchmark script. The only
production-code change for this performance run corrects benchmark metadata:
the compiled model name, GQA ratio six, and dense MLP replace misleading 35B/MoE
labels. No inference arithmetic, scheduling, or allocation policy was changed.

Native/Triton and release Rust builds succeeded. All four benchmark contract
tests passed through the prototype workflow. The driver checks snapshot identity,
prompt fingerprints, measured decode counts/context ranges, and effective GDN
state precision. Both workload comparisons and the final completion marker
were produced. The local report recomputes latency summaries and throughput
from the archived samples.

The complete artifact was retrieved from
`sp10:qs3/.prototypes/out/performance-38-27b-2026-09-15T155238Z-ny115g7j`.
`capture/` retains exact sources, the working-tree diff against commit
`78cfbe52f2bc0b7f4aa9ecac099bae9de457f9cf`, generated build configuration,
container commands/image inspection, build/test/run logs, numeric timing samples,
generated tokens, and per-workload comparisons. [manifest.json](manifest.json)
records fresh remote file sizes/SHA256 hashes used to verify the local copy.
