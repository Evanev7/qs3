#![allow(dead_code)]

use crate::engine::{DType, Status};
use crate::ffi;

use std::ffi::c_void;
use std::fs::File;

/// Backend interface for the future real-model weight loader.
///
/// The qwen3.6-specific loader should stay below this trait: parse
/// `config.json`, safetensors indexes, and all shard headers first; reject
/// duplicate, missing, unexpected, wrong-dtype, wrong-shape, overlapping, or
/// out-of-range tensors before allocating CUDA-visible memory. Once the exact
/// BF16 tensor plan is validated, it should call this backend to allocate the
/// final qs3-owned storage and fill it from safetensors byte ranges.
///
/// Intended implementations:
/// - `ManagedUmaBackend` for GB10/UMA: `cudaMallocManaged` final weights,
///   `preadv` directly into the host-visible pointer, then `cudaMemAdvise`
///   read-mostly/preferred-device and `cudaMemPrefetchAsync` in `seal`.
/// - `PinnedUploadBackend` for dGPU fallback: `cudaMalloc` final weights,
///   a small `cudaHostAlloc` ring, `preadv`, `cudaMemcpyAsync`, and events for
///   staging-buffer reuse.
/// - `CufileBackend` only later and only behind a hard capability probe; the
///   current GB10 box reports GDS compatibility fallback, and safetensors tensor
///   offsets are not guaranteed to be 4 KiB aligned.
///
/// The backend must not expose or retain mmap/InstantTensor staging pointers as
/// committed model weights. Returned spans are the durable pointers that
/// `QwenWeights` will wrap after load completion. `seal` is the only load-time
/// synchronization point required before the runner uses the weights.
pub(crate) trait WeightLoadBackend {
    fn alloc_tensor(&mut self, desc: WeightTensorDesc<'_>) -> Result<WeightLoadSpan, Status>;
    fn read_exact(&mut self, src: WeightFileRange<'_>, dst: WeightLoadSpan) -> Result<(), Status>;
    fn zero_fill(&mut self, dst: WeightLoadSpan) -> Result<(), Status>;
    fn seal(&mut self, stream: *mut c_void) -> Result<(), Status>;
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) struct WeightTensorDesc<'a> {
    name: &'a str,
    dtype: DType,
    shape: &'a [u32],
    bytes: usize,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum WeightLoadMemory {
    ManagedUma,
    Device,
}

#[derive(Clone, Copy, Debug)]
pub(crate) struct WeightLoadSpan {
    ptr: ffi::DevicePtr,
    bytes: usize,
    memory: WeightLoadMemory,
}

#[derive(Clone, Copy, Debug)]
pub(crate) struct WeightFileRange<'a> {
    file: &'a File,
    offset: u64,
    bytes: usize,
}

