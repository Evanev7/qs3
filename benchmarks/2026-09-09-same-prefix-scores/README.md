# Identical-prefix scores, 35B BF16

Both runtimes process the exact 1024-token sustained core prompt followed by
260 forced IDs from the pinned vLLM baseline. Record logits at prefill and after
every decode: 261 positions. The vLLM hook preserves raw logits before forcing
the sampler's choice. qs3 uses its actual `decode_one` path, not one-token append
or repeated whole-prefix prefill. This is a numerical diagnostic, not a timed run.

qs3 diagnostic source: `c2fef72`; runtime kernels: `0dc68a2`. The corresponding
command is now `./run_cuda_test.sh just scores-test 2026-09-09-long-forced`.
Its Rust test loads `~/qs3-scores/<run>/input.json` and writes `<run>/qs3/`.
The full diagnostic passed. The pinned vLLM image is recorded in `image.txt`;
its versions and model revision are also in `vllm/scores.json`. Both use BF16
weights and convolution/KV caches with FP32 GDN recurrence. vLLM retains its
normal graph execution; qs3 uses eager AOT.

Results:

- Argmax agrees at 256/261 positions. Differences are at 162, 163, 173, 218, 223.
- vLLM reproduces all 260 original greedy IDs despite the diagnostic hook.
- Mean negative log likelihood of the 260 forced IDs is 0.108287 for qs3 and
  0.105211 for vLLM, in natural-log units. These IDs are vLLM's own greedy output
  on one repeated benchmark prompt; this is not independent language-quality
  evaluation or a quality-parity threshold.
- Twelve full-vocabulary captures have RMSE 0.0789–0.5464. At position 162,
  KL(vLLM || qs3) is 0.074231 nats. See `comparison.json` for per-position
  centered errors, KL, candidate scores and binary SHA256 hashes.

At the first disagreement:

| ID | qs3 FP32 logit | qs3 rounded to BF16 | vLLM logit |
| ---: | ---: | ---: | ---: |
| 8340 | 21.865154 | 21.875 | 21.125 |
| 79091 | 21.409761 | 21.375 | 21.375 |

Rounding the final qs3 logit vector alone therefore does not resolve this choice.
It does not isolate differences in the preceding computation or projection
algorithms. A standalone probe of the loaded vLLM `ReplicatedLinear` router
confirms BF16 input/output and BF16 weights; qs3 computes FP32 router logits.
The earlier vLLM decode trace also passes BF16 router logits to top-k gating.
Router/shared-gate precision and projection packing merit controlled comparison;
no cause of the remaining score differences is established here.

`qs3/scores.json` and `vllm/scores.json` retain top-20 and forced-token scores at
every step. Full binary captures remain at
`sp10@sp10:~/qs3-scores/2026-09-09-long-forced/{qs3,vllm}/`. Local binary copying
was stopped after the remote run finished because transfer was slow; the complete
comparison ran on sp10 using the retained binaries. Run `compare.py <run-dir>`
where both binary directories are present to reproduce it. The launcher copies
JSON/logs by default; binary download is optional.
