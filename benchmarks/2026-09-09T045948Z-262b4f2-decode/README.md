# 35B BF16 decode after warp GDN recurrence

Source `262b4f2`, sp10 GB10, pinned 35B snapshot, FP32 recurrence, eager Rust
execution, 96-block grouped MoE. Nsight captures 32 forwards after four warmups,
102-token prompt, context 106–138. See `runner.json` for exact settings and IDs.

The capture contains 38,948 kernels, 1278.544 ms summed kernel duration and
1379.419 ms first-to-last-kernel span. GPU work covers 1283.637 ms of that span;
95.782 ms has no GPU work. Pinned allocation/free time is zero.

GEMV kernels total about 775 ms and grouped MoE 392.597 ms. Decode convolution
still takes 37.418 ms over 960 calls; warp GDN takes 13.351 ms over 960 calls.
The output projection identified by grid 62,080 has 32 calls and 197.355 ms;
its kernel signature uses BF16 weights and FP32 internal input/output (recorded
separately). Rust supplies BF16 input through BF16 cuBLASLt matrix layouts; the
FP32 kernel input reflects internal library handling, while FP32 logits are the
requested output dtype.

Compare with the aligned vLLM trace in `../2026-09-09T050426Z-vllm-bf16-decode/`.
Profiler timings diagnose work and gaps; use the unprofiled core artifacts for
end-to-end throughput. Full trace files remain in `.prototypes/profiles/` and
`sp10@sp10:~/qs3-profiles/` under this capture name.
