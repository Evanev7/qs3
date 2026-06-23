this is quasar3, a minimal qwen3.6 runtime built from flashinfer, inspired by dwarfstar4
rarely record durable facts that take significant information gathering here
- do not worry about backward compatibility, abi stability or versioning
- no cmake, we'll figure out a build system later

todo:
- implement `flashinfer.h` first: context/error handling, scratch allocation,
  kernel loading, paged kv append, batch decode planning/execution, and batch
  prefill planning/execution.
- write focused tests before building higher-level runtime code. cover tensor
  descriptor validation, paged kv table shape/layout checks, append position
  math, decode/prefill plan lifecycle, error reporting, and basic flashinfer
  output parity against small reference cases.
- build a narrow `engine/session` layer above flashinfer after the wrapper is
  tested. the session should own tokens, per-layer paged kv caches, page tables,
  flashinfer plans, logits, and prefix sync/rebuild policy.
- treat paged kv state as first-class runtime state: page allocator, indptr,
  indices, last-page lengths, and append positions should be explicit session
  data rather than temporary sampler-loop arrays.
- reuse flashinfer plans across layers and tokens where the shape permits.
  separate prefill and decode paths, with chunked prefill for long suffixes and
  decode for short/live generation.
- keep the runtime qwen3.6-specific. validate early, fail loudly, avoid generic 
  runtime compatibility work.
