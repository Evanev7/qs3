# Experiment index

This file is the current handoff, not a chronological notebook. Detailed results
live beside their sources in `benchmarks/`. Update a row when starting, rejecting,
or adopting a candidate. TODO.md is the implementation checklist.

## Current baseline

- Target: Qwen3.8-27B NVIDIA NVFP4, GB10 / SM121, eager, no MTP implementation.
- Source: `f284ac6`; QuTLASS prefill and AOT row selection in `b1db445`.
- Full `./remote.sh test` and both standard benchmarks pass.
- 102/32: 129.014 ms prefill, 10.356 tok/s. 1024/256: 413.405 ms, 10.568 tok/s.
- Large prefill is 20.2% faster than N32 baseline (518.081 ms). Decode is flat
  within 0.4%; all 36/260 generated IDs and 69/69 diagnostic logit rows match.
- Snapshot: `dbb8f445b3145f8a4c18ddc769f032d57d32867c`.
- [Prior baseline and 28-case survey](benchmarks/2026-09-19-nvfp4-performance/README.md).
- Full-model vLLM parity after chunked GDN integration remains open. Matching
  independent tokens and matching logits on identical prefixes are different tests.

## Active work and ownership

GPU checkout: `sp10@sp10:qs3`; one GPU workflow at a time.
This experiment pass is finished; no GPU workflow is running or reserved.
Working sources: `.prototypes/kernel_replacements/`; curated snapshots/results
are linked below. Update ownership here before starting another probe.
Do not run `remote.sh test`, `benchmark`, or another GPU probe concurrently.
Preserve Rust scheduling/state transactions and AOT inference without Python/JIT.

| ID | Candidate | Status / evidence | Next decisive check |
| --- | --- | --- | --- |
| KR01 | vLLM Triton fused Q/K norm + partial RoPE + gate extraction | 8.13→1.53 µs at M1, 404→253 µs at M1024; gate exact, Q/K differ | Match current RoPE/reduction; native AOT and full model before adoption |
| KR02 | vLLM packed GDN decode / b12x batched recurrence | Source inspection; current prep + recurrence ~0.99 ms/token | Compare state layout/rounding and small batched sequence semantics; retain separate read/write state slots |
| KR03 | b12x tensor FP8 / dense NVFP4 CuTeDSL | Native CuTe + Triton reducer passes; M1 evicted ~350→284 µs on largest projection; rare BF16 differences across fixtures | Production-library reference and real-model comparison; smaller shapes only ~2% gain |
| KR04 | QuTLASS SM120 NVFP4 GEMM / fused quantization | Adopted `b1db445`/`f284ac6`: tests pass, 69/69 logit rows exact; M1024 prefill 518→413 ms | Preserve this baseline; compare future fusion against it |
| KR05 | cuTile Rust | AOT SM121 native launches pass: SAXPY 64/64, NVFP4 768/768 exact with isolated tileiras 13.4.92 | Real-shape performance and useful fusion; 13.3 compiler fails NVFP4 on SM121 |
| KR06 | vLLM fused SiLU×up + NVFP4 quantization | 42/42 packed-value/scale cases exact; K17408 M1 5.38→1.87 µs, M1024 655→359 µs | Small production implementation, then full-model/tests/benchmark |
| KR07 | Row-dependent NVFP4 selection | Adopted: N32 below 128 rows, N64 from 128, QuTLASS from 512; configured AOT | Small decode/MTP shapes retain N32; revisit when measuring sequential batches |
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

KR06 has the strongest exact intermediate evidence. Start from
`.prototypes/kernel_replacements/silu_quant_native.cu`; replace the probe's
vLLM helper copy with existing FlashInfer packed-vector/conversion helpers,
then rerun `silu_quant_probe.py`. Only wire Rust dispatch after packed-byte and
scale equality holds. Full `./remote.sh test`, the saved-logit diagnostic, and
`./remote.sh benchmark` are the adoption checks, in that order. KR03 is the
larger possible decode gain, but its reduction differences require model evidence.

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
