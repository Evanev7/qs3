# Qwen3.8-27B NVFP4 performance experiments

These are sequential sp10/GB10 experiments on `b143912` with the pinned NVIDIA
snapshot `dbb8f445b3145f8a4c18ddc769f032d57d32867c`, eager execution, FP32 GDN
state, and the new 64-token GDN prefill. No MTP implementation is involved.

## Committed GDN baseline

The standard `./remote.sh benchmark` completed both workloads:

| Context / measured decode tokens | Prefill p50 | Decode tok/s | Decode p50 |
| --- | ---: | ---: | ---: |
| 102 / 32 | 129.391 ms | 10.374 | 96.313 ms |
| 1024 / 256 | 515.756 ms | 10.616 | 94.147 ms |

Full measurement samples and the separate Nsight capture are in
[102/32](../2026-09-19T005405.800847842Z-b143912.json) and
[1024/256](../2026-09-19T005439.162702917Z-b143912.json).
The previous `df6f07e` long prefill measured 736.564 ms: this run is about 30%
faster. Decode did not improve. This is a historical comparison, not an
interleaved experiment isolating the GDN change.

The long trace attributes about 49.75 ms/token to NVFP4 matrix kernels and
38.56 ms/token to FP8 matrix kernels, out of 93.01 ms summed kernel time.
The local GDN decode recurrence takes about 0.94 ms/token; gaps total 2.17 ms/token.
Matrix projections are the main performance target. Nsight reported unsupported
UMA fault tracing, unavailable CPU kernel stacks, and possible missing events;
the ordinary CUDA and CPU user-stack trace data passed the benchmark checks.

`weight-storage.json` inventories checkpoint headers. Non-embedding text tensors
occupy about 17.61 GB, including scales. Dividing that by kernel time gives a
rough 189 GB/s checkpoint-byte rate, **not measured DRAM bandwidth**: cache
reuse, repeated reads, activations, and recurrent state are not accounted for.

## Tactic and workspace survey

[The survey](survey/summary.json) contains 28 fresh, unprofiled processes. Each
uses two prefill warmups and five samples, then four decode warmups and 64
measured steps. Both contexts run each choice twice, reversing the order.
The tables average the two runs' reported statistics.

| NVFP4 tactic, 64 MiB cuBLASLt workspace | 102 prefill ms | 102 decode tok/s | 1024 prefill ms | 1024 decode tok/s |
| --- | ---: | ---: | ---: | ---: |
| Tile N=32, DP (default) | 129.870 | 10.657 | 519.637 | 10.558 |
| Tile N=32, Stream-K | 130.213 | 10.643 | 1746.673 | 10.569 |
| Tile N=64, DP | 128.980 | 10.598 | 437.923 | 10.518 |
| Tile N=64, Stream-K | 172.883 | 10.603 | 442.965 | 10.532 |

N=64 DP reduces long prefill by 15.7%, while its decode is slightly slower.
All tactic runs produce the same 68-token independent greedy continuations as
the default at each context. This alone does not establish logit parity or a
threshold for switching tactics at smaller prefill / future MTP shapes.

| cuBLASLt workspace, N=32 DP | 102 prefill ms | 102 decode tok/s | 1024 prefill ms | 1024 decode tok/s |
| --- | ---: | ---: | ---: | ---: |
| 16 MiB | 128.615 | 11.087 | 519.127 | 11.063 |
| 64 MiB (default) | 130.037 | 10.581 | 522.043 | 10.547 |
| 256 MiB | 129.233 | 10.577 | 522.551 | 10.543 |

The 16 MiB choice is 4.8% / 4.9% faster in decode but changes both continuations:
they share only 55 / 29 generated tokens with the default. Each choice repeats
its own tokens exactly. The workspace is an input to cuBLASLt algorithm selection
in `qscb.cu`; reducing it is not a numerically neutral memory saving.
Production retains 64 MiB and the N=32 DP tactic.

Each survey case retains its full result, command/environment overrides and run
log. `survey/metadata.json` identifies the source/config/native archive;
`survey/overlay.diff` records the isolated benchmark-only edit.
For reproduction, copy `survey/run.py` to `.prototypes/nvfp4_perf/run.py` and run
it on the matching built checkout with the normal remote CUDA/Rust environment.
Never overlap these runs with tests or the standard benchmark.

## Numerical follow-up

[Comparisons](logits/comparisons.json) distinguish identical-prefix rows from
rows after independent greedy decoding has diverged. Six additional processes
capture prefill plus 68 decode logit rows each. These diagnostic runs perform
per-step downloads and **must not be used as timing evidence**. Hashes/exact
comparisons cover all 248320 logits; distribution metrics use the tokenizer's
248077 addressable IDs.

At 1024 tokens, N=64 DP and N=32 DP produce **69/69 bit-identical rows**, tested
separately at 64 MiB and 16 MiB workspace. This is stronger evidence for the
prefill tactic than token agreement alone. Before adopting a row-dependent
choice, measure the crossover at smaller row counts, including the shapes
needed for MTP. A global N=64 default would sacrifice decode performance.

The 16 MiB workspace differs from 64 MiB on 56/56 identical-prefix rows at
102 tokens and 29/30 at 1024 tokens. The long prefill row is exact. On the short
prefill row, max absolute error is 4.5 and RMSE is 0.751, but KL(control ||
candidate) is only 1.81e-7: large low-probability logit changes alone do not
describe the distribution. At the first differing greedy choice, KL is 0.234
at short context and 0.265 at long context. This does not establish either
choice's quality or agreement with vLLM; the workspace speedup remains
experimental pending an identical-prefix reference comparison and FP8 algorithm
investigation.

`logits/validate.py` uses the survey helper `run.py`; `logits/overlay.diff` retains
its diagnostic changes. `logits/COMPLETE.json` describes the initial five cases;
the sixth, `1024-64dp-64m`, uses the same binary and has its command recorded in
that case directory. `logits/analyze.py` reduces all six and separately compares
the two tactics at 16 MiB. Raw rows remain on sp10 in
`.prototypes/out/nvfp4-perf-logits-skr3f9xb/` (about 411 MB), outside Git.

## Trace report reduction

The baseline's CPU report reduction spent minutes scanning the entire GPU trace
for every sampled CPU event. The change in `src/bin/qs3_bench/profile.rs` merges
GPU work intervals into a temporary indexed table and uses a predecessor lookup.
Nested intervals, copies, memsets, half-open boundaries and repeated reductions
are covered by the existing reducer test extended for these cases.

An isolated host harness passes all four reducer tests and exactly reproduces
**every CPU report field** from the saved 1024/256 trace. Reduction takes 0.930 s
on the local development host; this is not a same-host before/after speed ratio.
`trace-reducer/validation.json` records the trace/source hashes and reference JSON;
the adjacent logs retain the test and comparison output. Raw SQLite remains in
`.prototypes/out/trace-reducer-validation/` locally and the original profile
directory on sp10, rather than in Git.

The archived harness files use `.txt` extensions so they do not become inputs
to the main Nix/Cargo build. To reproduce the host-only check, copy them to a
temporary crate as `Cargo.toml`, `Cargo.lock`, and `src/main.rs`. Copy the current
`src/bin/qs3_bench/profile.rs` to its `src/profile.rs`, and append this wrapper
to the copied file:

```rust
pub fn offline_cpu_summary(db: &rusqlite::Connection) -> Result<tinyjson::JsonValue> {
    cpu_summary(db)
}
```

Run that harness with `QS3_BUILD_PROFILE=release cargo test`, then
`QS3_BUILD_PROFILE=release cargo run --release -- <decode.sqlite> <baseline.json>`.
It opens SQLite read-only and asserts exact equality against `nsight.cpu`.

The complete required `./remote.sh test` workflow passed on the GDN integration
source before these experiments (`gdn-integration-tests.log`), and again with
the reducer change now committed as `1516ae4` (`full-tests.log`). The latter
includes 17 Python tests, three standalone AOT launcher tests, 164 library tests
(with six existing ignored cases), the benchmark/integration suites, and all
four native CUDA suites. The standard benchmark of committed source `a6cb1ba`
(the reducer plus this experiment record) then passed both workloads:

| Context / measured decode tokens | Prefill p50 | Decode tok/s | Decode p50 |
| --- | ---: | ---: | ---: |
| 102 / 32 | 128.745 ms | 10.366 | 96.395 ms |
| 1024 / 256 | 518.081 ms | 10.607 | 94.229 ms |

Results: [102/32](../2026-09-19T012145.016839315Z-a6cb1ba.json) and
[1024/256](../2026-09-19T012210.420121283Z-a6cb1ba.json).
The complete 36 / 260 generated IDs, including warmups, exactly match the earlier
`b143912` benchmark. `committed-benchmark-validation.json` records those checks.
There is no inference-performance improvement from changing the report reducer.
The complete command, including the Nix build and both measurement/profile
workloads, took 3m27s (`committed-benchmark.log`). Both CPU reports completed,
with 3746 / 30078 sampled events. The final evidence-only commit renames the
archived harness files to `.txt`; production code is unchanged from this run.
