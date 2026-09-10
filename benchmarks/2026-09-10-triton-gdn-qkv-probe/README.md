# Real-weight GDN QKV Triton candidates

The b12x-derived row kernel with eight warps is the best tested device-backed
candidate. After eviction it reduces the 8192x2048 BF16 projection from
198.400 to 185.056 µs. The median paired saving is 13.312 µs (6.7%); all nine
layer/repetition pairs save 12.288–14.336 µs. Managed weights show little gain.
No production provider changes or end-to-end speedup claims accompany this probe.

## Results

Medians over three layers and three repetitions, after cache eviction. Each
entry in the raw CSV is itself the median of 50 CUDA event measurements.

| Allocation | Candidate rows/warps | cuBLASLt µs | Triton µs | Median paired saving µs |
| --- | --- | ---: | ---: | ---: |
| device | row 1/4 | 198.368 | 190.176 | 8.192 |
| device | row 1/8 | 198.400 | 185.056 | 13.312 |
| device | tile 4/4 | 198.816 | 195.296 | 4.048 |
| device | tile 4/8 | 198.400 | 196.320 | 3.072 |
| device | tile 8/4 | 199.280 | 207.584 | -8.336 |
| device | tile 8/8 | 198.384 | 197.344 | 1.056 |
| managed CPU copy | row 1/8 | 205.440 | 203.488 | 1.952 |
| managed upload | row 1/8 | 204.512 | 202.464 | 1.968 |

The managed CPU-copy row/eight-warp saving ranges from -2.016 to 3.072 µs;
managed upload ranges from 1.024 to 3.056 µs. Most multirow managed cases regress.
See [summary.json](summary.json) for every variant, including repeated-weight
measurements and paired ranges. Differences between marginal medians need not
equal the median paired saving.

All **54 validation cases** pass full-output cuBLASLt comparison. Maximum absolute
difference is 0.000488281 across all variants and 0.000244141 for row/eight-warps;
layer zero is numerically identical. Both outputs also pass CPU double
references at 65 spread rows, including boundaries. This covers six variants,
three real tensors and three allocation modes with one reproducible synthetic
BF16 activation vector. These are real weights, not captured model activations.
BF16 validation tolerance is 2e-5 absolute plus 0.008 relative; full-model
logits and continuation remain separate gates.

Thirty calls per decode times the device saving suggest **0.399 ms/token** of
kernel time. This is an extrapolation from isolated kernels, not measured
decode improvement. QKV is a modest candidate; routed-expert GEMV against the
complete existing MoE path remains the larger-budget next experiment.

## Allocation control

The initial sweep allocated new tensors and prepared a new cuBLASLt plan for
each variant. Its device cuBLASLt medians shifted from about 197 µs for the first
variant to 156–157 µs for later variants. Managed rankings also differed.
Those records and the original harness remain under [initial/](initial/).

The final sweep shares input, weight, output and workspace allocations and one
prepared cuBLASLt plan across all six variants within each process. Its device
control stays near 198–199 µs. This removes allocation and plan replacement
between candidates; it does not establish the underlying cause of the initial
timing shift. Compare each candidate with its paired control and do not mix
absolute times across these runs. Multiple full-model allocation contexts remain
an integration gate.

Weights use cudaMalloc or cudaMallocManaged. `managed_upload` copies host data
with cudaMemcpyAsync; `managed_cpu` copies through the CPU. Setup, upload,
validation, warmup and eviction are outside timed intervals. The probe does not
measure loader throughput, first GPU use, or full-model memory pressure.

## Reproduction and provenance

Run from repository commit `1f39b0bddec48132a3dc2ed420ddbb323ab665a3` with the
archived prototype sources. Native qscb is unchanged. The host is sp10, NVIDIA
GB10 SM121, driver 580.142, CUDA 13.0.88; compiler and assembler details and hashes
are in [manifest.json](manifest.json), and linked libraries in [host.txt](host.txt).
The pinned snapshot and SHA256s of layer 0/18/38 QKV tensors are in
[weights.json](weights.json).

Compilation uses pinned Triton 3.8.0, FP32 accumulation, one stage, and disabled
FP fusion. The row kernel is imported from the existing
`build_tools/qstriton/kernels/lm_head.py` with its b12x attribution/license; the
multirow kernel is copied from the previous qs3 GEMV probe. Scratch and cluster
requirements are rejected at export. Native execution loads cubins without
runtime Python or JIT. cuBLASLt uses the existing prepared path and 64 MiB workspace.

```sh
mkdir -p .prototypes/gdn_qkv_aot
cp benchmarks/2026-09-10-triton-gdn-qkv-probe/sources/* .prototypes/gdn_qkv_aot/
tar -cf - .prototypes/gdn_qkv_aot/{compile.py,extract.py,kernel.py,bench.cu,run.sh} \
  | ssh -F /dev/null sp10@sp10 'cd qs3 && tar -xf -'
./run_cuda_test.sh bash .prototypes/gdn_qkv_aot/run.sh
```

Run sequentially with other test/benchmark jobs on the disposable checkout.
The test script does not transfer ignored `.prototypes` sources itself.
Outputs are in remote `.prototypes/gdn_qkv_aot/out/`. Retrieve CSV/log/JSON files
and `host.txt`, then run `sources/summarize.py OUTPUT_DIRECTORY` locally.

Each control/candidate uses ten warmups, then 50 event samples per repetition;
three repetitions alternate which implementation runs first. The evicted case
touches at least 128 MiB or four times L2, whichever is larger, before each sample.
The repeated-weight case omits eviction. Both completed sweeps passed through
the prescribed test script; production tests were not rerun for this probe-only
change.
