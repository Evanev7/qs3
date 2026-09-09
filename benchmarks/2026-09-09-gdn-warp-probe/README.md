# Ordered warp GDN recurrence probe

sp10 GB10, CUDA 13, serial reference from source commit `b8a0828`.
The AOT candidate replaces a 128-thread shared-memory reduction with one warp
per value row, four rows per 128-thread block. Each lane keeps four state values.
Its explicit rounded additions reproduce the original tree: pairs at stride 64,
then stride 32, followed by warp shuffles at strides 16/8/4/2/1. This preserves
FP32 addition order and prevents contracting multiplication into reduction adds.

The initial 24-case probe and expanded 192-case probe compare every BF16 output
bit and every state byte. Coverage includes 1/2/4/31/128/1024 tokens, Q/K L2 norm
on/off, BF16/FP32 state, separate/same-slot state writes, disabled state updates,
one/two sequences, empty segments and negative read slots. The expanded probe
also instantiates the decode kernel at one token. All comparisons pass.

CUDA-event means over 20 calls after three warmups in the expanded run:

| Tokens | Q/K norm | State | Block row | Warp row |
| ---: | --- | --- | ---: | ---: |
| 1 | on | FP32 | 0.035101 ms | 0.010226 ms |
| 1024 | on | FP32 | 32.076317 ms | 4.390481 ms |
| 1024 | on | BF16 | 32.071556 ms | 4.448955 ms |
| 1024 | off | FP32 | 17.487783 ms | 3.994461 ms |

These are standalone kernel probes alongside the asset download, not isolated
core benchmarks. Reproduce with the repository's CUDA flags and native objects,
excluding `qscu_gdn.o` because the probe includes that source. The initial and
expanded source/log pairs are retained. Production integration must separately
pass the prescribed suite and real-model/core comparisons.
