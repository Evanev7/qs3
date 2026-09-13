# Identical-prefix numerical check after decode kernel changes

Candidate runtime: `efc944447ed60a6586e44504d4ee59dcbaf66247`, SM121, native
CUDA 13 release build, pinned-upload device weights, BF16 router, FP32 GDN state,
Triton GDN QKV and LM head, `decode_gemv64` routed MoE. The captured candidate
source diff is documentation only. This instrumented score replay is not a
performance measurement.

The reference is the previously saved vLLM BF16 capture for snapshot
`995ad96eacd98c81ed38be0c5b274b04031597b0`. Each system receives exactly the same
prompt and forced continuation. Every position saves all 248320 FP32 logits
(37, 205, and 805 frames including prefill and decode warmups). No new vLLM run
was needed. All arrays are finite; raw output SHA256 hashes are in the reports.

The earlier qs3 control is `4cdc270` plus its archived `baseline-source.diff`:
that patch selects pinned-upload weights and adds diagnostic assertions/metadata.
It predates optional stochastic sampling; both score replays use greedy mode.
The three prefill arrays are bit-for-bit identical between that control and the
candidate. Decode arrays are not bit-for-bit identical.

| Workload | Earlier qs3 / vLLM argmax agreement | Candidate / vLLM | Earlier forced mean NLL | Candidate NLL | Earlier centered RMSE vs vLLM | Candidate RMSE |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 102/32 | 37/37 | 37/37 | 0.000395 | 0.000446 | 0.118996 | 0.129258 |
| 500/200 | 204/205 | 205/205 | 0.095672 | 0.093924 | 0.142159 | 0.147809 |
| 4000/800 | 798/805 | 799/805 | 0.087547 | 0.088508 | 0.179475 | 0.179357 |

Relative to earlier qs3, argmax changes at no short positions, at intermediate
position 193, and at long positions 26, 148, and 539. The first two long changes
agree with vLLM; the last introduces a new disagreement. Remaining long vLLM
mismatches are 47, 49, 210, 271, 278, and 539. Mean likelihood and score distances
move in both directions. These are vLLM-selected continuations, not an independent
quality evaluation or evidence that one summation order is generally superior.

Candidate-versus-earlier-qs3 full-vocabulary differences are explicit:

| Workload | Mean centered RMSE | Maximum absolute logit change |
| --- | ---: | ---: |
| 102/32 | 0.124974 | 3.576336 |
| 500/200 | 0.112670 | 3.905744 |
| 4000/800 | 0.148895 | 4.893288 |

These maxima span the full vocabulary; they are not only winning-token errors.
Projection reductions change floating-point summation order while retaining all
BF16 rounding stages. Do not describe the optimized runner as bitwise equivalent.
The existing loaded 35B score/token/reset/rollback regression passed separately
with BF16 and FP32 recurrent state.

## Evidence and reproduction

- `candidate/kernel-comparison.json`: per-frame candidate/control hashes and
  full-vocabulary drift, plus ranking and likelihood summaries.
- `candidate/{102-32,500-200,4000-800}/comparison.json`: complete comparisons with
  the saved vLLM reference, including ranking margins and selected token scores.
- `candidate/*/input.json` and `candidate/*/qs3/scores.json`: exact input token IDs,
  capture positions and per-position rankings/likelihoods.
- `candidate/metadata.json`, `candidate/source.diff`, `candidate/SUCCESS.json`:
  source and execution provenance plus completion of all three workloads.
- `baseline-summary.json` and `baseline-source.diff`: earlier control provenance
  and scalar comparisons.

Raw candidate logits remain on sp10 under
`/home/sp10/qs3/.prototypes/out/qs3-correctness-2026-09-13T161939Z-v1v4zcx7/`.
The earlier qs3 directory is
`/home/sp10/qs3/.prototypes/out/qs3-correctness-2026-09-13T130939Z-jyc_aku1/`;
vLLM is under `qs3/.prototypes/out/vllm-correctness-2026-09-13T121347Z-8y4EGb/data/`.
Large raw arrays are not duplicated into Git.

`run.py` is the existing `.prototypes/qs3_correctness/run.py` used through
`run_cuda_test.sh`; restore it to that location to reproduce with the cached
reference. It invokes the committed `loader::tests::scores::real_qwen36_same_prefix_scores`
release test. `compare_vllm.py` is its original
`.prototypes/same_prefix_scores/compare.py`. `compare.py BASELINE CANDIDATE OUTPUT`
computes the additional qs3-to-qs3 drift beside the complete raw captures.
