# Triton AOT builder

Run from the repository root after `just build_tools/uv-sync`:

```sh
nix eval --json --file models/config.nix > build/triton-config.json
build_tools/.venv/bin/qstriton \
  --config build/triton-config.json --out build/triton \
  build_tools/qstriton/kernels/lm_head.py
```

`qstriton` is a uv workspace member, installed as `qstriton` to avoid
shadowing the upstream `triton` compiler package. The shared workspace lock pins
its compiler dependency. Kernel sources and their license ship as package data.

Each source file contains one locally defined `@triton.jit` kernel. Its filename
stem selects `config.kernels.<stem>`; the builder discovers the function itself.
There is no build hook, argument list, or symbol name to maintain separately.

Parameters annotated `tl.tensor` (or left unannotated) are device pointers.
Their names select element formats
from the entry's `precision` attrset, which references the model's precision
policy. Scalar annotations use Triton's type metadata. `tl.constexpr` parameters
select values from `constants`; `{ dtype = "f32"; }` denotes a dtype-valued
constant. `grid` and `options` provide launch dimensions and compiler options.
All of these inputs are resolved in Nix, then passed through `ASTSource`.

The b12x row kernel names its input `activation` to match the precision field and
takes accumulation precision as a constexpr. Its source needs no configuration
imports or factory. The source and its license are under `kernels/`.

Outputs per file are a cubin, PTX, a JSON compilation manifest, and a standalone
Rust module. Rust embeds the cubin and packs runtime arguments in the inspected
order, excluding constexprs. Pointer types describe storage (`u16` for BF16);
this is an unsafe launch wrapper, not a tensor shape/aliasing checker.
Integer constexprs are exposed in `constants` so the typed Rust adapter can
check the compiled dimensions without repeating configuration values.

Load `Kernel` in the caller's existing CUDA context before capture or timing;
launch on that context's stream. Keep the module alive until queued work and
graphs using it are finished, and keep its context current through destruction.
The generated module links `libcuda`; inference needs neither Python nor JIT.

The current launcher covers this ordinary single-CTA row GEMV with no scratch
allocations. Further argument and launch forms are listed in `TODO.md`.

`ninja -C build` builds the LM head alongside the native archive. `nix build
.#triton` produces the same artifacts through `default.nix`; the Rust Nix packages
include that output. `backend::qstriton::LmHead` checks typed tensor dimensions
and layout, and the runner loads it once for the compiled model shape. Other
shapes (including the small randomized fixtures) use the existing cuBLASLt path.

`just build_tools/triton-test` builds and exercises generated Rust, including a BF16-output
specialization with a masked reduction tail. Run it on sp10 with
`./run_cuda_test.sh just build_tools/triton-test`; the host needs Nix, the installed Python
workspace, Rust, and CUDA.
