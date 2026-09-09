# Qwen3.6-27B attention shape probe

On sp10 GB10, a disposable copy of source commit
`f23e5572122a8e668c9d7e3e86c17e821e4c9328` compiled and passed native tests at
24 Q heads, four KV heads, head dimension 256, GQA ratio six. The fixture reuses
the existing CPU attention reference with these exact head counts: paged decode,
causal prefill, decode reprepare, and paged append cases all pass in release.

The probe changes qs3's plan dispatch from group 8 to groups 6/8 and accepts
only the existing 16Q/2KV shape plus 24Q/4KV. FlashInfer's standard decode dispatch
supports groups 1/2/3/4/8 and therefore also needs changing: after including
`utils.cuh`, the probe maps that macro to qs3's 6/8 dispatch before including
the attention kernels. This remains an AOT FlashInfer-owned translation unit;
no vendored file changes or runtime JIT are needed.

`probe.py` builds the disposable source and fixture under
`.prototypes/qwen27_attention/` from an existing Ninja build. Run the resulting
`probe` executable separately. The log records compilation and successful
checks. Warnings are unused test functions excluded from this focused main.
This proves the native shape is viable; production Rust validation/dispatch,
checked-build coverage, GDN/MLP shape work and full 27B inference remain open.
