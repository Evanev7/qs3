# Qwen3.8-27B BF16 comparison — 2026-09-15

All four full-model qs3 replays completed on sp10. Short-prompt logits are close
to the saved vLLM reference, but longer prompts expose substantial differences
at prefill / first decode. This is an initial numerical comparison, not a
completed correctness gate or a performance benchmark.

## Results

| Workload | Matching argmax / rows | Differences within a vLLM maximum tie | Mean centered logit RMSE | Maximum KL(vLLM ‖ qs3) |
| --- | ---: | ---: | ---: | ---: |
| 4 / 4 | 4 / 4 | 0 | 0.02232 | 0.000414 |
| 102 / 32 | 36 / 37 | 1 | 0.03922 | 0.002551 |
| 500 / 200 | 204 / 205 | 0 | 0.06372 | 0.50450 |
| 4000 / 800 | 794 / 805 | 5 | 0.07797 | 19.69938 |

There are 1,038 exact argmax matches out of 1,051 rows. Six of the thirteen
different choices also attain the maximum score in vLLM; seven do not.
These counts describe identical forced prefixes, not free-running generation.
The four small-case predictions agree on `[5, 0, 31, 46474]`.

- **102/32, step 18:** vLLM ties tokens 11 and 13 at 28.125 and selects 11.
  qs3 selects 13 with a 0.019493 FP32 margin; both qs3 candidate scores round
  to the same BF16 value, 28.125.
- **500/200, step 70:** vLLM selects 7308 over 41755 by 0.125; qs3 selects
  41755 over 7308 by 0.047049. The largest distribution difference is instead
  at prefill (step 0): KL 0.50450 and centered RMSE 1.83470, despite agreeing
  on token 248046. Its qs3/vLLM scores are 20.48551 / 27.25.
- **4000/800, step 1:** after consuming the same first token, qs3 selects
  248046 while vLLM selects 492. Token 492 scores 7.62726 / 23.5 in qs3/vLLM;
  token 248046 scores 27.83768 / 19.625. KL is 19.69938 and centered RMSE is
  2.14798. This difference cannot be explained by final-logit BF16 rounding.
  Prefill already differs (centered RMSE 1.70195), although its winner agrees.
  The other non-tied differences occur at steps 7, 170, 345, 395 and 445;
  steps 201, 317, 410, 475 and 542 involve tied vLLM maxima.

The next numerical investigation should isolate long-prompt prefill and its
first decode. Later agreement under forced tokens does not clear those early
differences. No cause is established by this capture, and these repeated-prompt
workloads do not measure independent language quality.

## Protocol and evidence

- BF16 snapshot: `Qwen/Qwen3.8-27B`, revision
  `1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0`.
- Reference: pinned vLLM 0.29.0, using the
  [September 14 capture](../2026-09-14-vllm-references/README.md).
  All 1,051 original 3.8 rows passed fresh size/SHA256 checks before this run.
- qs3 source: `78cfbe52f2bc0b7f4aa9ecac099bae9de457f9cf` plus
  [the recorded working-tree diff](capture/source.diff). The build selects
  3.8-27B; the score diagnostic verifies the model name and snapshot revision,
  records the loaded tokenizer count, and reports dense MLP metadata.
- AOT native and Triton builds succeeded. Each workload ran the ignored release
  score test through `./remote.sh prototype qs3-correctness`, loading weights
  separately via pinned staging into device allocations. All four test
  invocations passed. The general test suite and prefix rebuild/reset cases
  were not run in this comparison.
- Both runtimes use BF16 weights/activations and FP32 GDN recurrent state.
  qs3 uses cuBLASLt dense MLP, local GDN, FlashInfer full attention, and Triton
  LM-head/GDN-QKV decode projections. qs3 retains FP32 LM-head output; vLLM's
  BF16 output was saved as FP32 rows. No inference arithmetic was changed here.
- Each replay performs one prefill, then forces the saved continuation one token
  at a time. Step 0 is prefill; step N observes `prompt_ids + forced_ids[:N]`.
  Benchmark-shaped cases include four warmup decode steps and the final
  prediction. Each of the 1,051 rows contains 248320 logits.
- Distribution metrics include padded logit slots to match the existing
  comparator. The [summary](summary.json) also reports forced-token NLL over
  the 248077 addressable tokenizer IDs. This is diagnostic NLL on vLLM's own
  continuation, not an independent quality score.

Full qs3 logits and the completed run remain on sp10:
`/home/sp10/qs3/.prototypes/out/qs3-correctness-38-27b-2026-09-15T153412Z-m17x9euc`.
The reference rows remain in the adjacent
`vllm-correctness-2026-09-14T194223Z-jn1cisug/data/qwen3.8-27b` directory.
The automatic raw-row download was stopped after the compact export completed;
the local ignored raw copy is partial. The remote inference/comparison run has
its own [completion marker](capture/SUCCESS.json).

`capture/` retains all qs3 score summaries, replay inputs, full per-row numerical
comparisons and raw-row hashes, exact runner/diagnostic/comparator sources,
build/replay logs and provenance. It omits binary rows and remote vLLM symlinks;
use the linked original reference artifact for vLLM score summaries and pins.
`manifest.json` records hashes/sizes of the compact capture files retrieved here.

To reproduce, restore `capture/run.py` as `.prototypes/qs3_correctness/run.py`
and `capture/compare.py` as `.prototypes/same_prefix_scores/compare.py`, use the
recorded source/configuration, and run the `qs3-correctness` prototype recipe
with the original reference directory available. Run `python3 summarize.py`
from this artifact to regenerate `summary.json` without loading a model.
