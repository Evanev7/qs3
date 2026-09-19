# Experiment index

This file is the current handoff, not a chronological notebook. Detailed results
live beside their sources in `benchmarks/`. Update a row when starting, rejecting,
or adopting a candidate. TODO.md is the implementation checklist.

## Current baseline

- Target: Qwen3.8-27B NVIDIA NVFP4, GB10 / SM121, eager, no MTP implementation.
- Source: `6511c0a`; latest production code `1516ae4` (trace reporting only).
- Full `./remote.sh test` and both standard benchmarks pass.
- 102/32: 128.745 ms prefill, 10.366 tok/s. 1024/256: 518.081 ms, 10.607 tok/s.
- Snapshot: `dbb8f445b3145f8a4c18ddc769f032d57d32867c`.
- [Baseline and 28-case survey](benchmarks/2026-09-19-nvfp4-performance/README.md).
- Full-model vLLM parity after chunked GDN integration remains open. Matching
  independent tokens and matching logits on identical prefixes are different tests.

## Active work and ownership

Owner: main agent. GPU checkout: `sp10@sp10:qs3`; one GPU workflow at a time.
Working sources: `.prototypes/kernel_replacements/`. Main agent owns ongoing
kernel probes; check this handoff before taking the GPU.
Do not run `remote.sh test`, `benchmark`, or another GPU probe concurrently.
Preserve Rust scheduling/state transactions and AOT inference without Python/JIT.

| ID | Candidate | Status / evidence | Next decisive check |
| --- | --- | --- | --- |
| KR01 | vLLM Triton fused Q/K norm + partial RoPE + gate extraction | 8.13→1.53 µs at M1, 404→253 µs at M1024; gate exact, Q/K differ | Match current RoPE/reduction; native AOT and full model before adoption |
| KR02 | vLLM packed GDN decode / b12x batched recurrence | Source inspection; current prep + recurrence ~0.99 ms/token | Compare state layout/rounding and small batched sequence semantics; retain separate read/write state slots |
| KR03 | b12x tensor FP8 / dense NVFP4 CuTeDSL | FP32 two-slice reduction: M1 evicted 350→284 µs on largest projection, exact output on fixture | Native export, production-library reference, and real-model comparison; smaller shapes only ~2% gain |
| KR04 | QuTLASS SM120 NVFP4 GEMM / fused quantization | `b1db445`: full tests pass; 69/69 full-model logit rows exact; raw M1024 ~17% faster than N64DP | Committed standard benchmark next |
| KR05 | cuTile Rust | SAXPY AOT SM121 + native launch pass; NVFP4 example fails SM121 compilation, SM120 cubin cannot load on GB10 | Try a newer isolated compiler; do not infer NVFP4 support from SAXPY success |
| KR06 | vLLM fused SiLU×up + NVFP4 quantization | Source available; scope includes BF16 rounding boundary | Same-input packed values/scales against current two-stage path |
| KR07 | Row-dependent NVFP4 selection | `b1db445`: N32 below 128 rows, N64 from 128, QuTLASS from 512; configured AOT | Committed benchmark; small decode/MTP shapes retain N32 |
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
