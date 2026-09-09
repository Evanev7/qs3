# BF16 grouped-MoE tile probe

Source `7f0e658`, sp10 GB10, CUDA 13. This standalone AOT probe compares
CTA/warp tiles 128x128x32 / 64x64x32, 32x128x64 / 32x64x64, and
16x128x64 / 16x64x64. All use 96 persistent threadblocks, SM80 BF16 tensor-core
arithmetic and two pipeline stages. The provider API and routing work are held
constant. No runtime JIT or vendored source changes are needed.

All 72 complete-output comparisons are bitwise identical to the 128-row control:
two repetitions, 1/16/102/1024 tokens, and routes spread over 8/32/256 of the 256
experts. Hidden width is 2048, intermediate width 512, top-k eight. Inputs and
weights use deterministic nonzero BF16 values. Every output is checked and each
case has nonzero outputs. CUDA events measure the full MoE stage after five
warmups, with 100 repetitions for 1/16 tokens and 30 for 102/1024 tokens.

Selected means across the two repetitions, microseconds:

| Routed experts | Tokens | Tile 128 | Tile 32 | Tile 16 |
| ---: | ---: | ---: | ---: | ---: |
| 256 | 1 | 324.91 | 283.66 | 274.94 |
| 256 | 102 | 7970.11 | 7246.84 | 7230.39 |
| 256 | 1024 | 9362.59 | 8609.82 | 8587.46 |
| 32 | 1024 | 2523.96 | 2566.17 | 3156.74 |
| 8 | 1024 | 1865.07 | 2119.45 | 2843.54 |

Smaller M tiles reduce padding for sparse expert rows, but can increase work and
weight traffic when many rows hit the same expert. The 32-row tile is a candidate
for real-model decode; retain the 128-row control and measure real prefill before
choosing policy. The 16-row tile offers little consistent benefit and much worse
skewed-prefill behavior. These probes do not establish core throughput gains.

Run `python3 build.py` from the pinned checkout after a normal native build, then
execute `.prototypes/moe_tiles/probe`. Candidate MoE symbols are renamed only in
the disposable probe so it can link with the normal context/provider object.
