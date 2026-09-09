# Pinned Qwen3.6-27B BF16 manifest

Repository [Qwen/Qwen3.6-27B](https://huggingface.co/Qwen/Qwen3.6-27B), revision
`6a9e13bd6fc8f0983b9b99948120bc37f49c13e9`, selected from the model API on
2026-09-09. [Pinned config](https://huggingface.co/Qwen/Qwen3.6-27B/blob/6a9e13bd6fc8f0983b9b99948120bc37f49c13e9/config.json).
These metadata files and bounded HTTP-range shard headers were fetched before
weight download or CUDA allocation. `probe.py` reproduces collection;
`python3 validate.py .` verifies the headers and exact text tensor shapes.

The manifest has 1199 BF16 tensors in 15 shards: 851 text tensors totaling
53,791,996,928 bytes, plus 333 visual tensors and 15 MTP tensors. All tensor data
is 55,562,855,904 bytes. Validation checks duplicate JSON keys/tensor names,
index/header agreement, dtype, tensor byte sizes, contiguous nonoverlapping
ranges within each reported file, and the exact dense text model shape set.
It does not validate weight payload contents or prove runtime 27B support.

The text model has hidden size 5120 and 64 layers in a three-GDN/one-full-attention
pattern. Full attention uses 24 Q and four KV heads, dimension 256, with a
12288-wide packed Q/gate projection and 6144-wide attention output. GDN has
16 key and 48 value heads, dimensions 128, conv width four; QKV projection width
is 10240 and Z/output width is 6144. Dense gate/up projections are separate
`[17408,5120]` tensors, with down `[5120,17408]`. The model config requests FP32
recurrent state. All safetensor storage, including A_log/dt_bias, is BF16.

The tokenizer JSON is retained on sp10 at
`~/qs3-27b-assets/6a9e13bd6fc8f0983b9b99948120bc37f49c13e9/tokenizer.json`.
Its SHA-256 is `5f9e4d4901a92b997e463c1f46055088b6cca5ca61a6522d1b9f64c4bb81cb42`,
identical to the already validated pinned 35B tokenizer. The 12.8 MB file is not
copied into this manifest directory. `validation.json` records counts and hashes;
`probe.json` records per-shard physical sizes and header hashes. Production Rust
manifest validation, dense materialization, AOT shape support and inference
reference checks remain TODO items.
