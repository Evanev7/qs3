# Triton AOT compiler

Compile a Python source exposing one `@triton.jit` function named `kernel`:

```sh
qstriton --source gemv.py --spec "$spec_json" --target "$target_json" --prefix lm_head
```

Pointer parameter names select `spec.precision` entries; scalar annotations use
Triton types. Constexprs select `spec.constants`, with `{ dtype = "f32"; }` for
dtype constants. `spec.grid` sets fixed launch dimensions, or is `null` for a
runtime grid. `spec.options` sets compiler options. Outputs are `.cubin`, `.ptx`,
`.json`, `.rs`, and a `.d` depfile.

The Rust launcher embeds the cubin and takes `DevicePtr<DType>` tensor arguments
using qs3's dtype markers. Scalars retain their primitive Rust types. Device
pointers are erased only inside the launcher for CUDA argument packing.
Load it before capture or timing; retain its
CUDA context and module until queued work and graphs finish. Callers validate
pointer shapes and aliasing. Fixed-grid launchers embed the generated `GRID`
constant; only runtime-grid launchers take a `[u32; 3]` parameter. The launcher
supports one CTA per cluster and no compiler-managed global scratch.
