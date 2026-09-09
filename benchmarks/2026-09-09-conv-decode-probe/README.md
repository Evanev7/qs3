# Tiled decode convolution probe

sp10 GB10, CUDA 13, `qscu.cu` from `1efaee8`. The candidate changes channel
assignment only: one 256-thread CTA per channel tile instead of one CTA looping
through every channel. Sequence indexing, arithmetic and state updates are
unchanged. This is an AOT standalone probe run alongside the asset download.

| Channels | State | Original, microseconds | Tiled, microseconds |
| --- | --- | ---: | ---: |
| 8192 | BF16 | 18.444 | 4.096 |
| 8192 | FP32 | 18.440 | 4.102 |
| 10240 | BF16 | 22.522 | 4.100 |
| 10240 | FP32 | 22.526 | 4.083 |

CUDA events measure 500 calls after ten warmups, including final-state writeback.
All output and state bytes match in 768 cases: both widths and state dtypes,
one/three sequences, same/separate state slots, updates enabled/disabled,
negative/valid reads, exact input/output alias or disjoint buffers, no activation
or SiLU, and no bias/BF16 bias/FP32 bias. The 10240-channel case probes 27B geometry;
it does not enable that shape in the production descriptors or Rust runner.

Reproduce from the pinned checkout after the normal native build:

```sh
nvcc -std=c++17 -arch=sm_121 --expt-relaxed-constexpr -diag-suppress 20012 \
  -I. -Ibuild -DQSFI_ENABLE_CHECKED_VALIDATION=0 \
  benchmarks/2026-09-09-conv-decode-probe/probe.cu \
  build/qsfi.o build/qscu_gdn.o build/qscb.o -o /tmp/conv-decode-probe \
  -lcuda -lcublas -lcublasLt
/tmp/conv-decode-probe
```
