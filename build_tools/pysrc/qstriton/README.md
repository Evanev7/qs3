# Triton AOT compiler

Compile a Python source exposing one `@triton.jit` function named `kernel`:

```sh
qstriton --source gemv.py --spec "$spec_json" --target "$target_json" --prefix lm_head
```

Pointer parameter names select `spec.precision` entries; scalar annotations use
Triton types. Constexprs select `spec.constants`, with `{ dtype = "f32"; }` for
dtype constants. `spec.grid` and `spec.options` set launch dimensions and compiler
options. Outputs are `.cubin`, `.ptx`, `.json`, `.rs`, and a `.d` depfile.

The Rust launcher embeds the cubin. Load it before capture or timing; retain its
CUDA context and module until queued work and graphs finish. Callers validate
pointer shapes and aliasing. The current launcher supports one CTA without scratch.
