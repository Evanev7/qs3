# Integrated runtime validation

`required-tests.log`: `./remote.sh test` exited 0 with the final `Nvfp4Linears`
wrapper. Python, Triton/CuTe launcher, Rust, and all four native CUDA suites pass.

`real-checkpoint.log`: the following command exited 0 on the same GPU checkout
with the pinned Qwen3.8-27B NVIDIA NVFP4 snapshot selected by `models/config.nix`:

```sh
LIBRARY_PATH=/usr/local/cuda/lib64 LD_LIBRARY_PATH=/usr/local/cuda/lib64 \
  cargo test --release --lib real_nvfp4_dense_prefill_decode_and_reset \
  -- --ignored --nocapture --test-threads=1
```

The real-model check covers finite, nonconstant logits at four forced prefixes
and exact reset/replay. It does not assert full-model parity with vLLM.
