# Experiment index

This file is the current handoff, not a chronological notebook. Detailed results
live beside their sources in `benchmarks/`. Update a row when starting, rejecting,
or adopting a candidate. TODO.md is the implementation checklist.

## Current baseline

- Target: Qwen3.8-27B NVIDIA NVFP4, GB10 / SM121, eager, no MTP implementation.
- Source: `1b298a0`; CuTe FP8 GDN QKV and NVFP4 small-M linears, QuTLASS prefill.
- Full `./remote.sh test` and both standard benchmarks pass.
- 102/32: 128.506 ms prefill, 84.938 ms/token, 11.766 tok/s.
- 1024/256: 414.435 ms prefill, 85.606 ms/token, 11.676 tok/s.
- Current measurements: `benchmarks/2026-09-20T111616.693817464Z-1b298a0.json`
  and `benchmarks/2026-09-20T111653.436956098Z-1b298a0.json`.
- Earlier QuTLASS adoption reduced large prefill from 518 to 413 ms with
  unchanged logits. CuTe FP8 subsequently reduced decode by about 8 ms/token;
  its accepted numerical differences are recorded under KR03.
- Snapshot: `dbb8f445b3145f8a4c18ddc769f032d57d32867c`.
- [Prior baseline and 28-case survey](benchmarks/2026-09-19-nvfp4-performance/README.md).
- Full-model vLLM parity after chunked GDN integration remains open. Matching
  independent tokens and matching logits on identical prefixes are different tests.

## Active work and ownership

GPU checkout: `sp10@sp10:qs3`; one GPU workflow at a time.
GPU owner: none; KR06 integration, full tests, and paired measurements complete.
The matched stock vLLM no-MTP baseline is complete (BL01).
KR03's required test suite passes; its numerical drift is accepted explicitly.
Working sources: `.prototypes/kernel_replacements/`; curated snapshots/results
are linked below. Update ownership here before starting another probe.
Do not run `remote.sh test`, `benchmark`, or another GPU probe concurrently.
Preserve Rust scheduling/state transactions and AOT inference without Python/JIT.

| ID | Candidate | Status / evidence | Next decisive check |
| --- | --- | --- | --- |
| KR01 | vLLM Triton fused Q/K norm + partial RoPE + gate extraction | 8.13→1.53 µs at M1, 404→253 µs at M1024; gate exact, Q/K differ | Match current RoPE/reduction; native AOT and full model before adoption |
| KR02 | vLLM packed GDN decode / b12x batched recurrence | Source inspection; current prep + recurrence ~0.99 ms/token | Compare state layout/rounding and small batched sequence semantics; retain separate read/write state slots |
| KR03 | b12x tensor FP8 / dense NVFP4 CuTeDSL | M1/N10240/K5120, FP32 split2 + Triton reduction; full tests pass. Snapshot `c9138b3`: 88.594/89.306 ms per token at 102/1024 context. 91/96 same-prefix decode winners match; numerical drift accepted. [Evidence](benchmarks/2026-09-20-qscute-fp8/README.md) | Small-batch shapes remain unqualified |
| KR04 | QuTLASS SM120 NVFP4 GEMM / fused quantization | Adopted `b1db445`/`f284ac6`: tests pass, 69/69 logit rows exact; M1024 prefill 518→413 ms | Preserve this baseline; compare future fusion against it |
| KR05 | cuTile Rust | AOT SM121 native launches pass: SAXPY 64/64, NVFP4 768/768 exact with isolated tileiras 13.4.92 | Real-shape performance and useful fusion; 13.3 compiler fails NVFP4 on SM121 |
| KR06 | Fused SiLU×up + NVFP4 quantization | Integrated using FlashInfer helpers; 81/81 packed-byte cases exact, full tests and real reset/replay pass. K17408 M1 4.67→3.67 µs, M1024 654→371 µs, including zero padding. Paired 1024-token prefill 417.179→399.731 ms; all generated IDs exact. [Evidence](benchmarks/2026-09-20-silu-nvfp4/README.md) | Committed benchmark/Nsight after review; decode gain is small relative to drift |
| KR07 | Row-dependent NVFP4 selection | N32 below 128 rows, N64 from 128, QuTLASS from 512; KR09 supersedes the 27B M<=16 path with CuTe | Revisit when measuring speculative batches and rollback |
| DR01 | Decode slowdown following `b143912` | Short workload +3.1 ms; captured FP8 QKV group explains ~2.96 ms; same binary can run fast; workspace-only test negative | [Evidence](benchmarks/2026-09-19-decode-regression/README.md); isolate algorithm and activation/output placement |
| KR09 | CuTe NVFP4 small-M MLP/LM head | Integrated 32x64x512, no split-K. 360 synthetic cases pass; selected recipe has 1,544/1,544 real intermediates exact. Full tests and real reset/replay pass. Final 102/32 and 1024/256: 85.126/85.558 ms/token. [Evidence](benchmarks/2026-09-20-cute-nvfp4/README.md) | Committed `1b298a0` benchmark/Nsight confirms 84.938/85.606 ms/token |
| BL01 | Stock vLLM NVFP4 without MTP | v0.29.0, same checkpoint; 82.707/83.582 ms per token, 12.093/11.951 tok/s at 102/32 and 1024/256. [Raw samples/settings](benchmarks/2026-09-20-vllm-nvfp4-nomtp/README.md) | CuTe NVFP4 working tree is ~2 ms/token behind vLLM with full decode graphs; independent continuations |
| KR08 | 16 MiB cuBLASLt workspace | Hold: ~4.9% faster decode, changes outputs at 55/29 tokens | Same-prefix vLLM reference and selected-algorithm investigation; keep 64 MiB default |

## Rules for comparing candidates

Current artifacts and pinned sources: [kernel replacement experiments](benchmarks/2026-09-19-kernel-replacements/README.md).

1. Record source commits, model/config/precision, shape, compiled flags, and toolchain.
2. Record same-input numerical checks before timings; compare recurrent state too.
3. Measure warmed and weight-evicted kernels separately. Include quantization,
   packing and reduction costs. Never extrapolate hot-cache GEMM timing to tok/s.
4. For a promising candidate, run the real model, full required tests, and the
   standard committed benchmark. Retain negative results and exact blockers.
5. Update this index with artifact paths and the next command. Work by another
   context/agent must state its owned files and whether it owns the GPU.

## Next experiment

After KR06 review and a committed benchmark/Nsight pass, map the remaining
cuBLAS FP8 projections to CuTe candidates. The `1b298a0` long trace spends about
23 ms/token in those GEMMs, beyond the CuTe GDN QKV path.

## Ideas to retain

- Small speculative batches (roughly <=16) need dedicated measurement; the
  existing N=32 GEMM winner at M=1 does not choose batched GDN or rollback design.
- Fuse Q/K preparation while leaving paged KV append separate initially.
- GDN prefill KKT + solve (+ W/U if useful) may reduce launches; preserve BF16
  intermediates. SM100 CuTeDSL code is not automatically suitable for SM121.
- Preserve checkpoint W4A4 / FP8 semantics. W4A16 substitution is a separate
  numerical experiment, not a transparent faster implementation.

## Historical evidence

- [Archived findings through 2026-09-19](benchmarks/2026-09-19-kernel-replacements/prior-findings.md)
  retain BF16 baselines, loader work, parity investigations, and architectural history.
- [Earlier NVFP4 kernel survey](benchmarks/2026-09-14-nvfp4-survey/README.md)
  already qualifies native b12x exports and FlashInfer; extend its evidence rather
  than repeating the export feasibility work.
