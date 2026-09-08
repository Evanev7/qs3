# qwen36-vectors

This subproject generates synthetic correctness vectors for qs3's narrow
Qwen3.6-35B-A3B runtime path.

The vector suite is intentionally separate from the production loader. It should
exercise model semantics by directly emitting small deterministic tensors and
expected outputs, then qs3 tests can upload those tensors into kernels or model
test fixtures without needing safetensors loading to exist first.

The vendored vLLM Qwen3.5/Qwen3-Next code is the semantic reference. The first
generator implementation should use small local formulas derived from that code,
with optional executable-vLLM comparison only if the environment can support it.

Generated artifacts live under `build/vectors/qwen36_semantics/` by default and
are not committed. Byte-level oracle hashes live next to this generator under
`build_tools/qwen36-vectors/oracle_hashes/`.

## Artifact format

A vector artifact is a directory containing a `manifest.json` file and flat raw
tensor files. The primary format is intentionally simple for Rust tests to
consume without Python-only container formats.

```text
artifact-name/
  manifest.json
  input_hidden.f32
  expected_logits.bf16
  token_ids.i32
```

All tensor files are little-endian, contiguous, row-major scalar dumps. There is
no per-file header; shape and dtype live in the manifest.

Supported dtype names are `bf16`, `f32`, `f64`, `i8`, `u8`, `i16`, `u16`, `i32`,
`u32`, `i64`, and `u64`. File suffixes match dtype names, for example `.bf16`
or `.i32`. BF16 tensors are stored as raw little-endian `u16` bit patterns.

```json
{
  "schema_version": 1,
  "byte_order": "little",
  "name": "full_attention_smoke",
  "groups": ["full_attention"],
  "tensors": [
    {
      "name": "input_hidden",
      "dtype": "f32",
      "shape": [2, 2048],
      "file": "input_hidden.f32",
      "role": "input"
    },
    {
      "name": "expected_logits",
      "dtype": "bf16",
      "shape": [2, 151936],
      "file": "expected_logits.bf16",
      "role": "expected"
    }
  ]
}
```

The manifest validator rejects unsupported schema versions, non-little-endian
artifacts, duplicate tensor names, duplicate tensor files, negative dimensions,
unsupported dtypes, and nested tensor paths.

## Python helpers

The package has no runtime dependency on torch or NumPy. `qwen36_vectors.schema`
defines `VectorManifest` and `TensorSpec`; `qwen36_vectors.io` reads and writes
manifests and raw tensors using only the Python standard library.

BF16 helpers are provided for deterministic conversion at vector-generation
time:

- `float32_to_bf16_bits(value)` converts one Python float to a raw BF16 `u16`
  word with round-to-nearest-even behavior.
- `bf16_bits_to_float32(bits)` expands one raw BF16 `u16` word to float32.
- `float32_to_bf16_words(values)` and `bf16_words_to_float32(values)` handle
  iterables.

Generate the norm semantics vectors from the repository root with:

```sh
python3 build_tools/qwen36-vectors/src/qwen36_vectors/__main__.py generate-norms
```

The `norms` group emits Gemma RMSNorm and Gemma fused add RMSNorm artifacts.
Those vectors encode vLLM/FlashInfer semantics where the effective weight is
`raw_weight + 1.0`, and they include standard RMSNorm comparison tensors so
tests can fail directly when a path multiplies by `raw_weight`.

Generate and verify the full suite with:

```sh
just generate-vectors
```

That recipe writes all groups under `build/vectors/qwen36_semantics/` and checks
the generated file set, byte lengths, and SHA256 hashes against
`build_tools/qwen36-vectors/oracle_hashes/*.oracle.json`.

The Ninja stamp depends on the committed oracle hash metadata, not on Python
generator sources. After editing any generator Python, explicitly regenerate and
check the vectors before testing or committing; remove
`build/vectors/qwen36_semantics/.oracle-ok` first if you need to force a clean
`just generate-vectors` run without changing oracle metadata.

The `check-oracles` command can emit a Make-style depfile after verification:

```sh
python3 build_tools/qwen36-vectors/src/qwen36_vectors/__main__.py check-oracles \
  --input-root build/vectors/qwen36_semantics \
  --depfile build/vectors/qwen36_semantics/.oracle-ok.d \
  --depfile-target build/vectors/qwen36_semantics/.oracle-ok
```

Both depfile options must be supplied together. The target must match the build
output's path as seen by Ninja, relative to its working directory. Dependencies
use absolute paths and include generator Python sources, oracle JSON files, and
their directories to detect additions/removals. Generated vectors and
`__pycache__` contents are excluded. Ninja consumes this depfile for the
`.oracle-ok` target.

When a generator change intentionally updates bytes, refresh the oracle metadata
from a clean generated tree with:

```sh
just refresh-vector-oracles
```

## Rust consumption

Rust tests should consume these artifacts through `tests/vector_harness.rs`.
That harness is intentionally independent from `src/weight_loader.rs` and the
production safetensors path. It parses only `manifest.json`, validates flat
tensor paths and byte lengths, and reads tensor files as little-endian scalar
dumps.

Typed host views map directly to the manifest dtype:

- `bf16` tensors are raw `u16` words. Upload them with
  `DeviceTensor::<u16>::from_bf16(...)` and pass the resulting BF16 descriptor
  to qsfi/qscu primitive tests.
- `u16` tensors are raw words for bit-level checks or explicit reinterpretation
  in a test. They do not imply a qsfi dtype by themselves.
- `f32` tensors upload with `DeviceTensor::<f32>::from_f32(...)`, for example
  router logits, route weights, or f32 oracle comparison buffers.
- `i32` tensors upload with `DeviceTensor::<i32>::from_i32(...)`, for example
  top-k ids, positions, sequence indptrs, or state indices.

Primitive tests should build `ffi::Tensor1`, `ffi::Tensor2`, or `ffi::Tensor3`
from the uploaded device tensor and pass those descriptors into the existing
kernel boundary. Model-fixture tests should fill fixture device buffers from the
same typed host views and still avoid the safetensors loader. Use the test CUDA
stream/context already owned by the target test, and synchronize only at test
assertion boundaries.

The harness and model tests read `build/vectors/qwen36_semantics/`. Required
vector data is not optional: missing vector roots or manifests fail loudly. The
harness validates every manifest and raw tensor length before future
kernel-specific tests run.

Current Rust coverage includes:

- Gemma RMSNorm: upload `x.bf16` and `raw_weight.bf16`, run `qsfi_rmsnorm`,
  and compare both BF16 bits and f32 tolerances against `expected_output_bf16`
  / `expected_output_f32`.
- Fused residual RMSNorm: upload `x.bf16`, `residual.bf16`, and
  `raw_weight.bf16`, run `qsfi_fused_add_rmsnorm`, and compare normalized
  output plus updated residual output.
- Attention primitive bundle: upload prepared Q/K/V, q/k norm weights,
  positions, RoPE outputs, append page metadata, and expected attention output;
  assert explicit q/k norm plus partial RoPE before `POS_ENCODING_NONE`
  attention.
- MoE routing: upload router logits, run qwen3.6 top-k routing, and compare
  `topk_ids.i32`, unrenormalized weights, and renormalized weights including
  tie-break cases.
- GDN prep: upload packed post-conv tensors, `a_log`, `dt_bias`, q/k/v/gate
  weights, and state indices; compare causal conv output, post-conv Q/K/V,
  decay/beta materialization, and gated RMSNorm outputs before recurrent GDN
  execution.
- GDN decoder layer: load the private model fixture weights, run
  `execute_gdn_layer`, assert the output projection in `scratch.attn_proj`, then
  run `execute_post_attention_mlp` and compare router decisions, shared expert,
  residual, and next-layer norm.

Generate the full-attention primitive vectors with:

```sh
python3 build_tools/qwen36-vectors/src/qwen36_vectors/__main__.py generate-attention
```

The `full_attention_primitives` group emits packed per-head `[q, output_gate]`
extraction, q/k Gemma RMSNorm over head dim 256, vLLM NeoX-style partial RoPE
over rotary dim 64, and sigmoid output-gate artifacts.

Generate the GDN decoder-layer vector with:

```sh
python3 build_tools/qwen36-vectors/src/qwen36_vectors/__main__.py generate-gdn-decoder-layer
```

The `gdn_decoder_layer` group emits row-coded GDN projection weights for six
rows, real Qwen3.6 hidden/GDN dimensions, output projection weights, and the
shared MoE/shared-expert post-attention oracle with 256 experts, top-8 routing,
and synthetic intermediate width 8.
