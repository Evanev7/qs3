#![allow(dead_code)]

use crate::engine::{DynDType, Status};
use crate::ffi;
use crate::{
    QWEN36_FULL_ATTN_HEAD_DIM, QWEN36_FULL_ATTN_KV_HEADS, QWEN36_FULL_ATTN_KV_HIDDEN,
    QWEN36_FULL_ATTN_Q_HEADS, QWEN36_FULL_ATTN_Q_HIDDEN, QWEN36_FULL_ATTN_Q_PROJ_OUT,
    QWEN36_FULL_ATTN_ROTARY_DIM, QWEN36_GDN_CONV_WIDTH, QWEN36_GDN_KEY_DIM, QWEN36_GDN_NUM_K_HEADS,
    QWEN36_GDN_NUM_V_HEADS, QWEN36_GDN_OUTPUT_DIM, QWEN36_GDN_PACKED_DIM, QWEN36_GDN_VALUE_DIM,
    QWEN36_HIDDEN_SIZE, QWEN36_MOE_INTERMEDIATE_SIZE, QWEN36_MOE_NUM_EXPERTS,
    QWEN36_MOE_SHARED_EXPERT_INTERMEDIATE_SIZE, QWEN36_MOE_TOP_K,
};

use std::collections::{BTreeMap, BTreeSet, HashMap, HashSet};
use std::ffi::c_void;
use std::fmt;
use std::fs::File;
use std::io::{self, Read};
use std::os::fd::AsRawFd;
use std::path::{Component, Path, PathBuf};
use std::ptr;
use std::time::Instant;
use tinyjson::JsonValue;

const DEFAULT_MAX_JSON_BYTES: usize = 64 << 20;
const DEFAULT_MAX_HEADER_BYTES: usize = 256 << 20;
const CONFIG_FILE: &str = "config.json";
const SAFETENSORS_INDEX_FILE: &str = "model.safetensors.index.json";
const TEXT_PREFIX: &str = "model.language_model.";
const PINNED_UPLOAD_BUFFER_COUNT: usize = 4;
const PINNED_UPLOAD_BUFFER_BYTES: usize = 1 << 30;

fn read_exact_at_raw(file: &File, dst: *mut u8, bytes: usize, offset: u64) -> io::Result<()> {
    if dst.is_null() && bytes != 0 {
        return Err(io::Error::new(
            io::ErrorKind::InvalidInput,
            "null destination pointer",
        ));
    }
    let fd = file.as_raw_fd();
    let mut done = 0usize;
    while done < bytes {
        let absolute_offset = offset
            .checked_add(done as u64)
            .ok_or_else(|| io::Error::new(io::ErrorKind::InvalidInput, "pread offset overflow"))?;
        if absolute_offset > libc::off_t::MAX as u64 {
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                "pread offset exceeds off_t range",
            ));
        }
        let count = (bytes - done).min(isize::MAX as usize);
        let read = unsafe {
            libc::pread(
                fd,
                dst.add(done).cast::<c_void>(),
                count,
                absolute_offset as libc::off_t,
            )
        };
        if read < 0 {
            let err = io::Error::last_os_error();
            if err.kind() == io::ErrorKind::Interrupted {
                continue;
            }
            return Err(err);
        }
        if read == 0 {
            return Err(io::Error::new(
                io::ErrorKind::UnexpectedEof,
                "short pread while loading tensor",
            ));
        }
        done = done
            .checked_add(read as usize)
            .ok_or_else(|| io::Error::new(io::ErrorKind::InvalidData, "pread count overflow"))?;
    }
    Ok(())
}

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
///   direct offset reads into the host-visible pointer, and one CUDA stream sync
///   in `seal`.
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
    fn device_ordinal(&self) -> i32;
    fn allocations(&self) -> &[WeightLoadSpan];
    fn take_allocations(&mut self) -> Vec<WeightLoadSpan>;
    fn alloc_tensor(&mut self, desc: WeightTensorDesc<'_>) -> Result<WeightLoadSpan, Status>;
    fn read_exact(
        &mut self,
        src: WeightFileRange<'_>,
        dst: &WeightLoadSpan,
        stream: *mut c_void,
    ) -> Result<(), Status>;
    fn zero_fill(&mut self, dst: &WeightLoadSpan, stream: *mut c_void) -> Result<(), Status>;
    fn seal(&mut self, stream: *mut c_void) -> Result<(), Status>;
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) struct WeightTensorDesc<'a> {
    name: &'a str,
    dtype: DynDType,
    shape: &'a [u32],
    bytes: usize,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum WeightLoadMemory {
    ManagedUma,
    Device,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
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

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub(crate) struct ManagedUmaLoadStats {
    tensors: usize,
    allocated_bytes: usize,
    read_bytes: usize,
    zero_fill_bytes: usize,
    alloc_us: u128,
    read_us: u128,
    zero_fill_us: u128,
    seal_us: u128,
}

/// GB10/UMA loader backend: final weights live in CUDA managed memory and file
/// payloads are read directly into those committed allocations.
pub(crate) struct ManagedUmaBackend {
    device_ordinal: i32,
    allocations: Vec<WeightLoadSpan>,
    stats: ManagedUmaLoadStats,
}

impl ManagedUmaBackend {
    pub(crate) fn new(device_ordinal: i32) -> Result<Self, Status> {
        activate_device(device_ordinal)?;
        Ok(Self {
            device_ordinal,
            allocations: Vec::new(),
            stats: ManagedUmaLoadStats::default(),
        })
    }

    pub(crate) fn stats(&self) -> ManagedUmaLoadStats {
        self.stats
    }

    fn add_stat(value: &mut usize, add: usize) -> Result<(), Status> {
        *value = value.checked_add(add).ok_or(Status::InvalidArgument)?;
        Ok(())
    }

    fn add_elapsed_us(value: &mut u128, started: Instant) -> Result<(), Status> {
        *value = value
            .checked_add(started.elapsed().as_micros())
            .ok_or(Status::InvalidArgument)?;
        Ok(())
    }

    fn owns_span(&self, span: &WeightLoadSpan) -> bool {
        self.allocations
            .iter()
            .any(|allocation| allocation.ptr == span.ptr && allocation.bytes == span.bytes)
    }

    fn validate_span(&self, span: &WeightLoadSpan, bytes: usize) -> Result<(), Status> {
        if span.memory != WeightLoadMemory::ManagedUma
            || span.ptr.is_null()
            || span.bytes != bytes
            || !self.owns_span(span)
        {
            return Err(Status::InvalidArgument);
        }
        Ok(())
    }
}

impl WeightLoadBackend for ManagedUmaBackend {
    fn device_ordinal(&self) -> i32 {
        self.device_ordinal
    }

    fn allocations(&self) -> &[WeightLoadSpan] {
        &self.allocations
    }

    fn take_allocations(&mut self) -> Vec<WeightLoadSpan> {
        std::mem::take(&mut self.allocations)
    }

    fn alloc_tensor(&mut self, desc: WeightTensorDesc<'_>) -> Result<WeightLoadSpan, Status> {
        if desc.bytes == 0 {
            return Err(Status::InvalidArgument);
        }
        activate_device(self.device_ordinal)?;
        let started = Instant::now();
        let mut ptr = ptr::null_mut();
        result_from_cuda(unsafe {
            ffi::cuda::cudaMallocManaged(&mut ptr, desc.bytes, ffi::cuda::CUDA_MEM_ATTACH_GLOBAL)
        })?;
        if ptr.is_null() {
            return Err(Status::InternalError);
        }
        let span = WeightLoadSpan {
            ptr,
            bytes: desc.bytes,
            memory: WeightLoadMemory::ManagedUma,
        };
        self.allocations.push(span);
        self.stats.tensors = self
            .stats
            .tensors
            .checked_add(1)
            .ok_or(Status::InvalidArgument)?;
        Self::add_stat(&mut self.stats.allocated_bytes, desc.bytes)?;
        Self::add_elapsed_us(&mut self.stats.alloc_us, started)?;
        Ok(span)
    }

    fn read_exact(
        &mut self,
        src: WeightFileRange<'_>,
        dst: &WeightLoadSpan,
        _stream: *mut c_void,
    ) -> Result<(), Status> {
        self.validate_span(dst, src.bytes)?;
        let started = Instant::now();
        read_exact_at_raw(src.file, dst.ptr.cast::<u8>(), src.bytes, src.offset)
            .map_err(|_| Status::BackendError)?;
        Self::add_elapsed_us(&mut self.stats.read_us, started)?;
        Self::add_stat(&mut self.stats.read_bytes, src.bytes)
    }

    fn zero_fill(&mut self, dst: &WeightLoadSpan, _stream: *mut c_void) -> Result<(), Status> {
        self.validate_span(dst, dst.bytes)?;
        let started = Instant::now();
        unsafe {
            ptr::write_bytes(dst.ptr.cast::<u8>(), 0, dst.bytes);
        }
        Self::add_elapsed_us(&mut self.stats.zero_fill_us, started)?;
        Self::add_stat(&mut self.stats.zero_fill_bytes, dst.bytes)
    }

    fn seal(&mut self, stream: *mut c_void) -> Result<(), Status> {
        activate_device(self.device_ordinal)?;
        let started = Instant::now();
        result_from_cuda(unsafe { ffi::cuda::cudaStreamSynchronize(stream) })?;
        Self::add_elapsed_us(&mut self.stats.seal_us, started)
    }
}

impl Drop for ManagedUmaBackend {
    fn drop(&mut self) {
        let _ = activate_device(self.device_ordinal);
        for allocation in self.allocations.drain(..) {
            unsafe {
                ffi::cuda::cudaFree(allocation.ptr);
            }
        }
    }
}

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub(crate) struct PinnedUploadLoadStats {
    tensors: usize,
    allocated_bytes: usize,
    read_bytes: usize,
    zero_fill_bytes: usize,
    chunks: usize,
    alloc_us: u128,
    wait_us: u128,
    read_us: u128,
    copy_enqueue_us: u128,
    zero_fill_us: u128,
    seal_us: u128,
}

#[derive(Debug)]
struct PinnedUploadSlot {
    ptr: *mut c_void,
    event: *mut c_void,
    in_use: bool,
    event_recorded: bool,
    stream: *mut c_void,
}

/// dGPU-style comparison backend: final weights live in device memory and file
/// payloads pass through a fixed pinned host ring.
pub(crate) struct PinnedUploadBackend {
    device_ordinal: i32,
    allocations: Vec<WeightLoadSpan>,
    slots: Vec<PinnedUploadSlot>,
    next_slot: usize,
    stats: PinnedUploadLoadStats,
}

impl PinnedUploadBackend {
    pub(crate) fn new(device_ordinal: i32) -> Result<Self, Status> {
        activate_device(device_ordinal)?;
        let mut backend = Self {
            device_ordinal,
            allocations: Vec::new(),
            slots: Vec::with_capacity(PINNED_UPLOAD_BUFFER_COUNT),
            next_slot: 0,
            stats: PinnedUploadLoadStats::default(),
        };
        for _ in 0..PINNED_UPLOAD_BUFFER_COUNT {
            let mut ptr = ptr::null_mut();
            result_from_cuda(unsafe {
                ffi::cuda::cudaHostAlloc(
                    &mut ptr,
                    PINNED_UPLOAD_BUFFER_BYTES,
                    ffi::cuda::CUDA_HOST_ALLOC_DEFAULT,
                )
            })?;
            if ptr.is_null() {
                return Err(Status::InternalError);
            }
            let mut event = ptr::null_mut();
            let event_result = result_from_cuda(unsafe {
                ffi::cuda::cudaEventCreateWithFlags(
                    &mut event,
                    ffi::cuda::CUDA_EVENT_DISABLE_TIMING,
                )
            });
            if let Err(err) = event_result {
                unsafe {
                    ffi::cuda::cudaFreeHost(ptr);
                }
                return Err(err);
            }
            if event.is_null() {
                unsafe {
                    ffi::cuda::cudaFreeHost(ptr);
                }
                return Err(Status::InternalError);
            }
            backend.slots.push(PinnedUploadSlot {
                ptr,
                event,
                in_use: false,
                event_recorded: false,
                stream: ptr::null_mut(),
            });
        }
        Ok(backend)
    }

    pub(crate) fn stats(&self) -> PinnedUploadLoadStats {
        self.stats
    }

    fn add_stat(value: &mut usize, add: usize) -> Result<(), Status> {
        *value = value.checked_add(add).ok_or(Status::InvalidArgument)?;
        Ok(())
    }

    fn add_elapsed_us(value: &mut u128, started: Instant) -> Result<(), Status> {
        *value = value
            .checked_add(started.elapsed().as_micros())
            .ok_or(Status::InvalidArgument)?;
        Ok(())
    }

    fn owns_span(&self, span: &WeightLoadSpan) -> bool {
        self.allocations
            .iter()
            .any(|allocation| allocation.ptr == span.ptr && allocation.bytes == span.bytes)
    }

    fn validate_span(&self, span: &WeightLoadSpan, bytes: usize) -> Result<(), Status> {
        if span.memory != WeightLoadMemory::Device
            || span.ptr.is_null()
            || span.bytes != bytes
            || !self.owns_span(span)
        {
            return Err(Status::InvalidArgument);
        }
        Ok(())
    }

    fn synchronize_slot(&mut self, idx: usize) -> Result<(), Status> {
        if self.slots[idx].in_use {
            if self.slots[idx].event_recorded {
                result_from_cuda(unsafe {
                    ffi::cuda::cudaEventSynchronize(self.slots[idx].event)
                })?;
            } else {
                result_from_cuda(unsafe {
                    ffi::cuda::cudaStreamSynchronize(self.slots[idx].stream)
                })?;
            }
            self.slots[idx].in_use = false;
            self.slots[idx].event_recorded = false;
            self.slots[idx].stream = ptr::null_mut();
        }
        Ok(())
    }

    fn wait_slot(&mut self, idx: usize) -> Result<(), Status> {
        if self.slots[idx].in_use {
            let started = Instant::now();
            self.synchronize_slot(idx)?;
            Self::add_elapsed_us(&mut self.stats.wait_us, started)?;
        }
        Ok(())
    }
}

impl WeightLoadBackend for PinnedUploadBackend {
    fn device_ordinal(&self) -> i32 {
        self.device_ordinal
    }

    fn allocations(&self) -> &[WeightLoadSpan] {
        &self.allocations
    }

    fn take_allocations(&mut self) -> Vec<WeightLoadSpan> {
        std::mem::take(&mut self.allocations)
    }

    fn alloc_tensor(&mut self, desc: WeightTensorDesc<'_>) -> Result<WeightLoadSpan, Status> {
        if desc.bytes == 0 {
            return Err(Status::InvalidArgument);
        }
        activate_device(self.device_ordinal)?;
        let started = Instant::now();
        let mut ptr = ptr::null_mut();
        result_from_cuda(unsafe { ffi::cuda::cudaMalloc(&mut ptr, desc.bytes) })?;
        if ptr.is_null() {
            return Err(Status::InternalError);
        }
        let span = WeightLoadSpan {
            ptr,
            bytes: desc.bytes,
            memory: WeightLoadMemory::Device,
        };
        self.allocations.push(span);
        self.stats.tensors = self
            .stats
            .tensors
            .checked_add(1)
            .ok_or(Status::InvalidArgument)?;
        Self::add_stat(&mut self.stats.allocated_bytes, desc.bytes)?;
        Self::add_elapsed_us(&mut self.stats.alloc_us, started)?;
        Ok(span)
    }

    fn read_exact(
        &mut self,
        src: WeightFileRange<'_>,
        dst: &WeightLoadSpan,
        stream: *mut c_void,
    ) -> Result<(), Status> {
        self.validate_span(dst, src.bytes)?;
        let mut done = 0usize;
        while done < src.bytes {
            let slot_idx = self.next_slot % self.slots.len();
            self.next_slot = self
                .next_slot
                .checked_add(1)
                .ok_or(Status::InvalidArgument)?;
            self.wait_slot(slot_idx)?;
            let chunk = (src.bytes - done).min(PINNED_UPLOAD_BUFFER_BYTES);
            let slot_ptr = self.slots[slot_idx].ptr;
            let started = Instant::now();
            read_exact_at_raw(
                src.file,
                slot_ptr.cast::<u8>(),
                chunk,
                src.offset
                    .checked_add(done as u64)
                    .ok_or(Status::InvalidArgument)?,
            )
            .map_err(|_| Status::BackendError)?;
            Self::add_elapsed_us(&mut self.stats.read_us, started)?;

            let started = Instant::now();
            let dst_ptr = unsafe { dst.ptr.cast::<u8>().add(done).cast::<c_void>() };
            result_from_cuda(unsafe {
                ffi::cuda::cudaMemcpyAsync(
                    dst_ptr,
                    slot_ptr.cast_const(),
                    chunk,
                    ffi::cuda::CUDA_MEMCPY_HOST_TO_DEVICE,
                    stream,
                )
            })?;
            self.slots[slot_idx].in_use = true;
            self.slots[slot_idx].event_recorded = false;
            self.slots[slot_idx].stream = stream;
            result_from_cuda(unsafe {
                ffi::cuda::cudaEventRecord(self.slots[slot_idx].event, stream)
            })?;
            self.slots[slot_idx].event_recorded = true;
            Self::add_elapsed_us(&mut self.stats.copy_enqueue_us, started)?;
            Self::add_stat(&mut self.stats.read_bytes, chunk)?;
            self.stats.chunks = self
                .stats
                .chunks
                .checked_add(1)
                .ok_or(Status::InvalidArgument)?;
            done = done.checked_add(chunk).ok_or(Status::InvalidArgument)?;
        }
        Ok(())
    }

    fn zero_fill(&mut self, dst: &WeightLoadSpan, stream: *mut c_void) -> Result<(), Status> {
        self.validate_span(dst, dst.bytes)?;
        let started = Instant::now();
        result_from_cuda(unsafe { ffi::cuda::cudaMemsetAsync(dst.ptr, 0, dst.bytes, stream) })?;
        Self::add_elapsed_us(&mut self.stats.zero_fill_us, started)?;
        Self::add_stat(&mut self.stats.zero_fill_bytes, dst.bytes)
    }

    fn seal(&mut self, stream: *mut c_void) -> Result<(), Status> {
        activate_device(self.device_ordinal)?;
        let started = Instant::now();
        result_from_cuda(unsafe { ffi::cuda::cudaStreamSynchronize(stream) })?;
        for idx in 0..self.slots.len() {
            self.synchronize_slot(idx)?;
        }
        Self::add_elapsed_us(&mut self.stats.seal_us, started)
    }
}

impl Drop for PinnedUploadBackend {
    fn drop(&mut self) {
        let _ = activate_device(self.device_ordinal);
        for slot in self.slots.drain(..) {
            if slot.in_use {
                if slot.event_recorded && !slot.event.is_null() {
                    unsafe {
                        ffi::cuda::cudaEventSynchronize(slot.event);
                    }
                } else {
                    unsafe {
                        ffi::cuda::cudaStreamSynchronize(slot.stream);
                    }
                }
            }
            if !slot.event.is_null() {
                unsafe {
                    ffi::cuda::cudaEventDestroy(slot.event);
                }
            }
            if !slot.ptr.is_null() {
                unsafe {
                    ffi::cuda::cudaFreeHost(slot.ptr);
                }
            }
        }
        for allocation in self.allocations.drain(..) {
            unsafe {
                ffi::cuda::cudaFree(allocation.ptr);
            }
        }
    }
}

#[derive(Debug)]
pub(crate) enum WeightLoadError {
    Io(String),
    Json(String),
    InvalidConfig(String),
    InvalidIndex(String),
    InvalidSafetensors(String),
    TensorTable(String),
    Backend(Status),
}

impl WeightLoadError {
    fn json(message: impl Into<String>) -> Self {
        Self::Json(message.into())
    }

    fn invalid_config(message: impl Into<String>) -> Self {
        Self::InvalidConfig(message.into())
    }

    fn invalid_index(message: impl Into<String>) -> Self {
        Self::InvalidIndex(message.into())
    }

    fn invalid_safetensors(message: impl Into<String>) -> Self {
        Self::InvalidSafetensors(message.into())
    }

    fn tensor_table(message: impl Into<String>) -> Self {
        Self::TensorTable(message.into())
    }
}

impl From<io::Error> for WeightLoadError {
    fn from(err: io::Error) -> Self {
        Self::Io(err.to_string())
    }
}

impl fmt::Display for WeightLoadError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Io(message) => write!(f, "io error: {message}"),
            Self::Json(message) => write!(f, "json error: {message}"),
            Self::InvalidConfig(message) => write!(f, "invalid qwen3.6 config: {message}"),
            Self::InvalidIndex(message) => write!(f, "invalid safetensors index: {message}"),
            Self::InvalidSafetensors(message) => write!(f, "invalid safetensors: {message}"),
            Self::TensorTable(message) => write!(f, "invalid tensor table: {message}"),
            Self::Backend(status) => write!(f, "weight load backend failed: {status:?}"),
        }
    }
}

impl std::error::Error for WeightLoadError {}

type LoadResult<T> = Result<T, WeightLoadError>;

fn result_from_cuda(err: i32) -> Result<(), Status> {
    if err == ffi::cuda::CUDA_SUCCESS {
        Ok(())
    } else if err == ffi::cuda::CUDA_ERROR_MEMORY_ALLOCATION {
        Err(Status::OutOfMemory)
    } else {
        Err(Status::CudaError)
    }
}

fn activate_device(device_ordinal: i32) -> Result<(), Status> {
    if device_ordinal < 0 {
        return Ok(());
    }
    result_from_cuda(unsafe { ffi::cuda::cudaSetDevice(device_ordinal) })
}

fn parse_json_object(src: &str) -> LoadResult<HashMap<String, JsonValue>> {
    src.parse::<JsonValue>()
        .map_err(|err| WeightLoadError::json(err.to_string()))?
        .try_into()
        .map_err(|_| WeightLoadError::json("expected top-level object"))
}

fn read_json_object_with_limit(
    path: impl AsRef<Path>,
    max_bytes: usize,
) -> LoadResult<HashMap<String, JsonValue>> {
    let path = path.as_ref();
    let mut file = File::open(path)?;
    let len = file.metadata()?.len();
    if len > max_bytes as u64 {
        return Err(WeightLoadError::json(format!(
            "{} is {len} bytes, limit is {max_bytes}",
            path.display()
        )));
    }
    let mut src = String::new();
    file.read_to_string(&mut src)?;
    parse_json_object(&src)
}

fn string_field(object: &HashMap<String, JsonValue>, name: &str) -> LoadResult<String> {
    object
        .get(name)
        .and_then(JsonValue::get::<String>)
        .cloned()
        .ok_or_else(|| WeightLoadError::json(format!("field {name:?} must be a string")))
}

fn opt_string_field(object: &HashMap<String, JsonValue>, name: &str) -> LoadResult<Option<String>> {
    match object.get(name) {
        Some(value) if value.is_null() => Ok(None),
        Some(value) => value
            .get::<String>()
            .cloned()
            .map(Some)
            .ok_or_else(|| WeightLoadError::json(format!("field {name:?} must be a string"))),
        None => Ok(None),
    }
}

fn bool_field_with_default(
    object: &HashMap<String, JsonValue>,
    name: &str,
    default: bool,
) -> LoadResult<bool> {
    match object.get(name) {
        Some(value) if value.is_null() => Ok(default),
        Some(value) => value
            .get::<bool>()
            .copied()
            .ok_or_else(|| WeightLoadError::json(format!("field {name:?} must be a bool"))),
        None => Ok(default),
    }
}

fn u64_from_value(value: &JsonValue, context: impl fmt::Display) -> LoadResult<u64> {
    const MAX_SAFE_JSON_INTEGER: f64 = 9_007_199_254_740_992.0;
    let raw = value
        .get::<f64>()
        .ok_or_else(|| WeightLoadError::json(format!("{context} must be an integer")))?;
    if !raw.is_finite() || *raw < 0.0 || raw.fract() != 0.0 || *raw > MAX_SAFE_JSON_INTEGER {
        return Err(WeightLoadError::json(format!(
            "{context} must be a non-negative JSON integer <= 2^53"
        )));
    }
    Ok(*raw as u64)
}

fn u32_field(object: &HashMap<String, JsonValue>, name: &str) -> LoadResult<u32> {
    let value = object
        .get(name)
        .ok_or_else(|| WeightLoadError::json(format!("missing field {name:?}")))
        .and_then(|value| u64_from_value(value, format!("field {name:?}")))?;
    u32::try_from(value).map_err(|_| WeightLoadError::json(format!("field {name:?} overflows u32")))
}

fn opt_u32_field(object: &HashMap<String, JsonValue>, name: &str) -> LoadResult<Option<u32>> {
    object
        .get(name)
        .map(|value| {
            let value = u64_from_value(value, format!("field {name:?}"))?;
            u32::try_from(value)
                .map_err(|_| WeightLoadError::json(format!("field {name:?} overflows u32")))
        })
        .transpose()
}

fn f32_from_value(value: &JsonValue, context: impl fmt::Display) -> LoadResult<f32> {
    let raw = value
        .get::<f64>()
        .ok_or_else(|| WeightLoadError::json(format!("{context} must be a number")))?;
    if !raw.is_finite() || *raw < f32::MIN as f64 || *raw > f32::MAX as f64 {
        return Err(WeightLoadError::json(format!(
            "{context} must be a finite f32"
        )));
    }
    Ok(*raw as f32)
}

fn f32_field_with_default(
    object: &HashMap<String, JsonValue>,
    name: &str,
    default: f32,
) -> LoadResult<f32> {
    match object.get(name) {
        Some(value) if value.is_null() => Ok(default),
        Some(value) => f32_from_value(value, format!("field {name:?}")),
        None => Ok(default),
    }
}

fn string_array_field(object: &HashMap<String, JsonValue>, name: &str) -> LoadResult<Vec<String>> {
    let values = object
        .get(name)
        .and_then(JsonValue::get::<Vec<JsonValue>>)
        .ok_or_else(|| WeightLoadError::json(format!("field {name:?} must be an array")))?;
    let mut out = Vec::with_capacity(values.len());
    for value in values {
        let Some(value) = value.get::<String>() else {
            return Err(WeightLoadError::json(format!(
                "field {name:?} must contain only strings"
            )));
        };
        out.push(value.clone());
    }
    Ok(out)
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum QwenLayerKind {
    LinearAttention,
    FullAttention,
}

#[derive(Clone, Debug, PartialEq)]
pub(crate) struct Qwen36TextConfig {
    num_hidden_layers: u32,
    hidden_size: u32,
    intermediate_size: u32,
    vocab_size: u32,
    num_attention_heads: u32,
    num_key_value_heads: u32,
    head_dim: u32,
    rms_norm_eps: f32,
    rope_theta: f32,
    logits_soft_cap: f32,
    num_experts: u32,
    num_experts_per_tok: u32,
    moe_intermediate_size: u32,
    shared_expert_intermediate_size: u32,
    linear_num_key_heads: u32,
    linear_num_value_heads: u32,
    linear_key_head_dim: u32,
    linear_value_head_dim: u32,
    linear_conv_kernel_dim: u32,
    layer_types: Vec<QwenLayerKind>,
    tie_word_embeddings: bool,
    dtype: Option<String>,
    attn_output_gate: bool,
    full_attention_interval: Option<u32>,
    partial_rotary_factor: Option<f32>,
}

impl Qwen36TextConfig {
    fn read(model_dir: impl AsRef<Path>) -> LoadResult<Self> {
        Self::read_with_limit(model_dir, DEFAULT_MAX_JSON_BYTES)
    }

    fn read_with_limit(model_dir: impl AsRef<Path>, max_bytes: usize) -> LoadResult<Self> {
        let root = read_json_object_with_limit(model_dir.as_ref().join(CONFIG_FILE), max_bytes)?;
        Self::from_config_object(&root)
    }

    fn from_config_object(root: &HashMap<String, JsonValue>) -> LoadResult<Self> {
        let text = match root.get("text_config") {
            Some(value) => value
                .get::<HashMap<String, JsonValue>>()
                .ok_or_else(|| WeightLoadError::json("field \"text_config\" must be an object"))?,
            None => root,
        };
        let tie_word_embeddings = bool_field_with_default(
            text,
            "tie_word_embeddings",
            bool_field_with_default(root, "tie_word_embeddings", false)?,
        )?;
        let num_attention_heads = u32_field(text, "num_attention_heads")?;
        let hidden_size = u32_field(text, "hidden_size")?;
        let head_dim =
            opt_u32_field(text, "head_dim")?.unwrap_or_else(|| hidden_size / num_attention_heads);
        let intermediate_size = opt_u32_field(text, "intermediate_size")?
            .unwrap_or(u32_field(text, "moe_intermediate_size")?);
        let rope_parameters = match text.get("rope_parameters") {
            Some(value) if value.is_null() => None,
            Some(value) => Some(value.get::<HashMap<String, JsonValue>>().ok_or_else(|| {
                WeightLoadError::json("field \"rope_parameters\" must be an object")
            })?),
            None => None,
        };
        let rope_theta = match rope_parameters {
            Some(rope) => f32_field_with_default(rope, "rope_theta", 10_000.0)?,
            None => f32_field_with_default(text, "rope_theta", 10_000.0)?,
        };
        let partial_rotary_factor = match rope_parameters {
            Some(rope) => match rope.get("partial_rotary_factor") {
                Some(value) => Some(f32_from_value(
                    value,
                    "field \"rope_parameters.partial_rotary_factor\"",
                )?),
                None => text
                    .get("partial_rotary_factor")
                    .map(|value| f32_from_value(value, "field \"partial_rotary_factor\""))
                    .transpose()?,
            },
            None => text
                .get("partial_rotary_factor")
                .map(|value| f32_from_value(value, "field \"partial_rotary_factor\""))
                .transpose()?,
        };
        let layer_types = string_array_field(text, "layer_types")?
            .into_iter()
            .map(|name| match name.as_str() {
                "linear_attention" => Ok(QwenLayerKind::LinearAttention),
                "full_attention" => Ok(QwenLayerKind::FullAttention),
                other => Err(WeightLoadError::invalid_config(format!(
                    "unknown layer_types entry {other:?}"
                ))),
            })
            .collect::<LoadResult<Vec<_>>>()?;
        let config = Self {
            num_hidden_layers: u32_field(text, "num_hidden_layers")?,
            hidden_size,
            intermediate_size,
            vocab_size: u32_field(text, "vocab_size")?,
            num_attention_heads,
            num_key_value_heads: u32_field(text, "num_key_value_heads")?,
            head_dim,
            rms_norm_eps: f32_field_with_default(text, "rms_norm_eps", 1.0e-6)?,
            rope_theta,
            logits_soft_cap: f32_field_with_default(text, "logits_soft_cap", 0.0)?,
            num_experts: u32_field(text, "num_experts")?,
            num_experts_per_tok: u32_field(text, "num_experts_per_tok")?,
            moe_intermediate_size: u32_field(text, "moe_intermediate_size")?,
            shared_expert_intermediate_size: u32_field(text, "shared_expert_intermediate_size")?,
            linear_num_key_heads: u32_field(text, "linear_num_key_heads")?,
            linear_num_value_heads: u32_field(text, "linear_num_value_heads")?,
            linear_key_head_dim: u32_field(text, "linear_key_head_dim")?,
            linear_value_head_dim: u32_field(text, "linear_value_head_dim")?,
            linear_conv_kernel_dim: u32_field(text, "linear_conv_kernel_dim")?,
            layer_types,
            tie_word_embeddings,
            dtype: opt_string_field(text, "dtype")?,
            attn_output_gate: bool_field_with_default(text, "attn_output_gate", true)?,
            full_attention_interval: opt_u32_field(text, "full_attention_interval")?,
            partial_rotary_factor,
        };
        config.validate_supported()?;
        Ok(config)
    }

    fn validate_supported(&self) -> LoadResult<()> {
        if self
            .dtype
            .as_deref()
            .is_some_and(|dtype| dtype != "bfloat16")
        {
            return Err(WeightLoadError::invalid_config(format!(
                "dtype must be bfloat16, got {:?}",
                self.dtype
            )));
        }
        if self.hidden_size != QWEN36_HIDDEN_SIZE {
            return Err(WeightLoadError::invalid_config(format!(
                "hidden_size must be {QWEN36_HIDDEN_SIZE}, got {}",
                self.hidden_size
            )));
        }
        if self.num_hidden_layers == 0 || !self.num_hidden_layers.is_multiple_of(4) {
            return Err(WeightLoadError::invalid_config(
                "num_hidden_layers must be a positive multiple of 4",
            ));
        }
        if self.layer_types.len() != self.num_hidden_layers as usize {
            return Err(WeightLoadError::invalid_config(format!(
                "layer_types has {} entries, expected {}",
                self.layer_types.len(),
                self.num_hidden_layers
            )));
        }
        for (idx, layer_type) in self.layer_types.iter().enumerate() {
            let expected = if idx % 4 == 3 {
                QwenLayerKind::FullAttention
            } else {
                QwenLayerKind::LinearAttention
            };
            if *layer_type != expected {
                return Err(WeightLoadError::invalid_config(format!(
                    "layer {idx} must be {expected:?}, got {layer_type:?}"
                )));
            }
        }
        if self
            .full_attention_interval
            .is_some_and(|interval| interval != 4)
        {
            return Err(WeightLoadError::invalid_config(format!(
                "full_attention_interval must be 4, got {:?}",
                self.full_attention_interval
            )));
        }
        if self.num_attention_heads != QWEN36_FULL_ATTN_Q_HEADS
            || self.num_key_value_heads != QWEN36_FULL_ATTN_KV_HEADS
            || self.head_dim != QWEN36_FULL_ATTN_HEAD_DIM
        {
            return Err(WeightLoadError::invalid_config(format!(
                "full attention shape must be q_heads={QWEN36_FULL_ATTN_Q_HEADS} \
                 kv_heads={QWEN36_FULL_ATTN_KV_HEADS} head_dim={QWEN36_FULL_ATTN_HEAD_DIM}, \
                 got q_heads={} kv_heads={} head_dim={}",
                self.num_attention_heads, self.num_key_value_heads, self.head_dim
            )));
        }
        if self.num_experts != QWEN36_MOE_NUM_EXPERTS
            || self.num_experts_per_tok != QWEN36_MOE_TOP_K
            || self.moe_intermediate_size != QWEN36_MOE_INTERMEDIATE_SIZE
            || self.intermediate_size != QWEN36_MOE_INTERMEDIATE_SIZE
            || self.shared_expert_intermediate_size != QWEN36_MOE_SHARED_EXPERT_INTERMEDIATE_SIZE
        {
            return Err(WeightLoadError::invalid_config(
                "MoE fields do not match qs3 Qwen3.6-35B-A3B constants",
            ));
        }
        if self.linear_num_key_heads != QWEN36_GDN_NUM_K_HEADS
            || self.linear_num_value_heads != QWEN36_GDN_NUM_V_HEADS
            || self.linear_key_head_dim != QWEN36_GDN_KEY_DIM
            || self.linear_value_head_dim != QWEN36_GDN_VALUE_DIM
            || self.linear_conv_kernel_dim != QWEN36_GDN_CONV_WIDTH
        {
            return Err(WeightLoadError::invalid_config(
                "GDN fields do not match qs3 Qwen3.6 constants",
            ));
        }
        if !self.attn_output_gate {
            return Err(WeightLoadError::invalid_config(
                "attn_output_gate must be true for packed q_proj gate extraction",
            ));
        }
        if self.tie_word_embeddings {
            return Err(WeightLoadError::invalid_config(
                "tie_word_embeddings is unsupported; lm_head.weight must be present",
            ));
        }
        if let Some(partial_rotary_factor) = self.partial_rotary_factor {
            if !partial_rotary_factor.is_finite() || partial_rotary_factor <= 0.0 {
                return Err(WeightLoadError::invalid_config(
                    "partial_rotary_factor must be finite and positive",
                ));
            }
            let rotary_dim = self.head_dim as f32 * partial_rotary_factor;
            if (rotary_dim - QWEN36_FULL_ATTN_ROTARY_DIM as f32).abs() > f32::EPSILON {
                return Err(WeightLoadError::invalid_config(format!(
                    "partial rotary dim must be {QWEN36_FULL_ATTN_ROTARY_DIM}, got {rotary_dim}"
                )));
            }
        }
        if !self.rms_norm_eps.is_finite()
            || self.rms_norm_eps <= 0.0
            || !self.rope_theta.is_finite()
            || self.rope_theta <= 0.0
            || !self.logits_soft_cap.is_finite()
            || self.logits_soft_cap < 0.0
        {
            return Err(WeightLoadError::invalid_config(
                "non-finite or invalid numeric config field",
            ));
        }
        Ok(())
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum WeightTensorDType {
    Bf16,
    F32,
}

impl WeightTensorDType {
    fn from_safetensors_name(name: &str, tensor_name: &str) -> LoadResult<Self> {
        match name {
            "BF16" => Ok(Self::Bf16),
            "F32" => Ok(Self::F32),
            other => Err(WeightLoadError::invalid_safetensors(format!(
                "tensor {tensor_name:?} has unsupported dtype {other:?}; BF16 loader rejects quantized/NVFP4 tensors"
            ))),
        }
    }

    fn to_runtime_dtype(self) -> DynDType {
        match self {
            Self::Bf16 => DynDType::BF16,
            Self::F32 => DynDType::F32,
        }
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub(crate) struct TensorMeta {
    dtype: WeightTensorDType,
    shape: Vec<u32>,
    data_offsets: (u64, u64),
}

impl TensorMeta {
    fn byte_len(&self) -> LoadResult<usize> {
        storage_bytes(self.dtype, &self.shape)
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
struct TensorFileMeta {
    shard: String,
    absolute_offset: u64,
    meta: TensorMeta,
}

#[derive(Clone, Debug, Eq, PartialEq)]
struct SafetensorsHeader {
    tensors: BTreeMap<String, TensorMeta>,
    data_start: u64,
}

impl SafetensorsHeader {
    fn read(path: impl AsRef<Path>) -> LoadResult<Self> {
        Self::read_with_limit(path, DEFAULT_MAX_HEADER_BYTES)
    }

    fn read_with_limit(path: impl AsRef<Path>, max_header_bytes: usize) -> LoadResult<Self> {
        let path = path.as_ref();
        let mut file = File::open(path)?;
        let file_len = file.metadata()?.len();
        let mut prefix = [0u8; 8];
        file.read_exact(&mut prefix)?;
        let header_len = u64::from_le_bytes(prefix);
        if header_len == 0 {
            return Err(WeightLoadError::invalid_safetensors(format!(
                "{} has an empty header",
                path.display()
            )));
        }
        if header_len > max_header_bytes as u64 {
            return Err(WeightLoadError::invalid_safetensors(format!(
                "{} header is {header_len} bytes, limit is {max_header_bytes}",
                path.display()
            )));
        }
        let data_start = 8u64
            .checked_add(header_len)
            .ok_or_else(|| WeightLoadError::invalid_safetensors("header end overflow"))?;
        if data_start > file_len {
            return Err(WeightLoadError::invalid_safetensors(format!(
                "{} is shorter than its declared header",
                path.display()
            )));
        }
        let mut header = vec![0u8; header_len as usize];
        file.read_exact(&mut header)?;
        Self::parse_header_bytes(&header, data_start, file_len - data_start)
    }

    fn parse_safetensors_bytes(bytes: &[u8]) -> LoadResult<Self> {
        if bytes.len() < 8 {
            return Err(WeightLoadError::invalid_safetensors(
                "file is shorter than safetensors prefix",
            ));
        }
        let header_len = u64::from_le_bytes(bytes[..8].try_into().unwrap());
        let data_start = 8u64
            .checked_add(header_len)
            .ok_or_else(|| WeightLoadError::invalid_safetensors("header end overflow"))?;
        if data_start > bytes.len() as u64 {
            return Err(WeightLoadError::invalid_safetensors(
                "file is shorter than declared header",
            ));
        }
        Self::parse_header_bytes(
            &bytes[8..data_start as usize],
            data_start,
            bytes.len() as u64 - data_start,
        )
    }

    fn parse_header_bytes(header: &[u8], data_start: u64, data_len: u64) -> LoadResult<Self> {
        let src = std::str::from_utf8(header)
            .map_err(|err| WeightLoadError::invalid_safetensors(err.to_string()))?;
        let root = parse_json_object(src)?;
        let mut tensors = BTreeMap::new();
        for (name, value) in root {
            if name == "__metadata__" {
                if value.get::<HashMap<String, JsonValue>>().is_none() {
                    return Err(WeightLoadError::json(
                        "field \"__metadata__\" must be an object",
                    ));
                }
                continue;
            }
            let object = value.get::<HashMap<String, JsonValue>>().ok_or_else(|| {
                WeightLoadError::json(format!("tensor {name:?} must be an object"))
            })?;
            let dtype = WeightTensorDType::from_safetensors_name(
                string_field(object, "dtype")?.as_str(),
                &name,
            )?;
            let shape = parse_shape(object, &name)?;
            let data_offsets = parse_data_offsets(object, &name)?;
            if data_offsets.0 > data_offsets.1 {
                return Err(WeightLoadError::invalid_safetensors(format!(
                    "tensor {name:?} has decreasing data_offsets"
                )));
            }
            let expected_bytes = storage_bytes(dtype, &shape)? as u64;
            let actual_bytes = data_offsets.1 - data_offsets.0;
            if actual_bytes != expected_bytes {
                return Err(WeightLoadError::invalid_safetensors(format!(
                    "tensor {name:?} has {actual_bytes} data bytes, expected {expected_bytes}"
                )));
            }
            if tensors
                .insert(
                    name,
                    TensorMeta {
                        dtype,
                        shape,
                        data_offsets,
                    },
                )
                .is_some()
            {
                return Err(WeightLoadError::invalid_safetensors(
                    "duplicate tensor name in safetensors header",
                ));
            }
        }
        validate_safetensors_spans(&tensors, data_len)?;
        Ok(Self {
            tensors,
            data_start,
        })
    }
}

fn parse_shape(object: &HashMap<String, JsonValue>, tensor_name: &str) -> LoadResult<Vec<u32>> {
    let values = object
        .get("shape")
        .and_then(JsonValue::get::<Vec<JsonValue>>)
        .ok_or_else(|| {
            WeightLoadError::invalid_safetensors(format!(
                "tensor {tensor_name:?} shape must be an array"
            ))
        })?;
    let mut shape = Vec::with_capacity(values.len());
    for (idx, value) in values.iter().enumerate() {
        let dim = u64_from_value(value, format!("tensor {tensor_name:?} shape[{idx}]"))?;
        shape.push(u32::try_from(dim).map_err(|_| {
            WeightLoadError::invalid_safetensors(format!(
                "tensor {tensor_name:?} shape[{idx}] overflows u32"
            ))
        })?);
    }
    Ok(shape)
}

fn parse_data_offsets(
    object: &HashMap<String, JsonValue>,
    tensor_name: &str,
) -> LoadResult<(u64, u64)> {
    let values = object
        .get("data_offsets")
        .and_then(JsonValue::get::<Vec<JsonValue>>)
        .ok_or_else(|| {
            WeightLoadError::invalid_safetensors(format!(
                "tensor {tensor_name:?} data_offsets must be an array"
            ))
        })?;
    if values.len() != 2 {
        return Err(WeightLoadError::invalid_safetensors(format!(
            "tensor {tensor_name:?} data_offsets must have two entries"
        )));
    }
    Ok((
        u64_from_value(
            &values[0],
            format!("tensor {tensor_name:?} data_offsets[0]"),
        )?,
        u64_from_value(
            &values[1],
            format!("tensor {tensor_name:?} data_offsets[1]"),
        )?,
    ))
}

fn validate_safetensors_spans(
    tensors: &BTreeMap<String, TensorMeta>,
    data_len: u64,
) -> LoadResult<()> {
    let mut spans = tensors
        .iter()
        .map(|(name, meta)| (meta.data_offsets.0, meta.data_offsets.1, name.as_str()))
        .collect::<Vec<_>>();
    spans.sort_by_key(|(start, _, _)| *start);
    let mut cursor = 0u64;
    for (start, end, name) in spans {
        if start != cursor {
            return Err(WeightLoadError::invalid_safetensors(format!(
                "tensor {name:?} starts at {start}, expected contiguous offset {cursor}"
            )));
        }
        if end > data_len {
            return Err(WeightLoadError::invalid_safetensors(format!(
                "tensor {name:?} ends at {end}, past data length {data_len}"
            )));
        }
        cursor = end;
    }
    if cursor != data_len {
        return Err(WeightLoadError::invalid_safetensors(format!(
            "safetensors data has trailing bytes: consumed {cursor}, data length {data_len}"
        )));
    }
    Ok(())
}

fn storage_bytes(dtype: WeightTensorDType, shape: &[u32]) -> LoadResult<usize> {
    let elements = shape.iter().try_fold(1usize, |acc, dim| {
        acc.checked_mul(*dim as usize)
            .ok_or_else(|| WeightLoadError::tensor_table("tensor element count overflow"))
    })?;
    let bytes_per_element = match dtype {
        WeightTensorDType::Bf16 => 2,
        WeightTensorDType::F32 => 4,
    };
    elements
        .checked_mul(bytes_per_element)
        .ok_or_else(|| WeightLoadError::tensor_table("tensor byte count overflow"))
}

#[derive(Clone, Debug, Eq, PartialEq)]
struct SafetensorsIndex {
    weight_map: BTreeMap<String, String>,
}

impl SafetensorsIndex {
    fn read(model_dir: impl AsRef<Path>) -> LoadResult<Self> {
        Self::read_with_limit(model_dir, DEFAULT_MAX_JSON_BYTES)
    }

    fn read_with_limit(model_dir: impl AsRef<Path>, max_bytes: usize) -> LoadResult<Self> {
        let root = read_json_object_with_limit(
            model_dir.as_ref().join(SAFETENSORS_INDEX_FILE),
            max_bytes,
        )?;
        let weight_map = root
            .get("weight_map")
            .and_then(JsonValue::get::<HashMap<String, JsonValue>>)
            .ok_or_else(|| {
                WeightLoadError::invalid_index("field \"weight_map\" must be an object")
            })?;
        let mut out = BTreeMap::new();
        for (name, shard) in weight_map {
            let shard = shard.get::<String>().ok_or_else(|| {
                WeightLoadError::invalid_index(format!(
                    "weight_map entry {name:?} must be a shard string"
                ))
            })?;
            validate_shard_name(shard)?;
            out.insert(name.clone(), shard.clone());
        }
        if out.is_empty() {
            return Err(WeightLoadError::invalid_index("weight_map is empty"));
        }
        Ok(Self { weight_map: out })
    }

    fn shard_names(&self) -> BTreeSet<String> {
        self.weight_map.values().cloned().collect()
    }
}

fn validate_shard_name(shard: &str) -> LoadResult<()> {
    let path = Path::new(shard);
    if path.is_absolute() || !shard.ends_with(".safetensors") {
        return Err(WeightLoadError::invalid_index(format!(
            "invalid shard path {shard:?}"
        )));
    }
    let mut components = path.components();
    match (components.next(), components.next()) {
        (Some(Component::Normal(_)), None) => Ok(()),
        _ => Err(WeightLoadError::invalid_index(format!(
            "invalid shard path {shard:?}"
        ))),
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum WeightTensorSource {
    Safetensors,
    ZeroFill,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub(crate) struct WeightTensorTarget {
    layer: Option<u32>,
    slot: &'static str,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub(crate) struct WeightTensorSpec {
    name: String,
    dtype: WeightTensorDType,
    shape: Vec<u32>,
    source: WeightTensorSource,
    target: WeightTensorTarget,
}

impl WeightTensorSpec {
    fn byte_len(&self) -> LoadResult<usize> {
        storage_bytes(self.dtype, &self.shape)
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
struct ValidatedTensor {
    spec: WeightTensorSpec,
    file_meta: Option<TensorFileMeta>,
}

pub(crate) fn expected_qwen36_bf16_specs(
    config: &Qwen36TextConfig,
) -> LoadResult<Vec<WeightTensorSpec>> {
    config.validate_supported()?;
    let hidden = config.hidden_size;
    let vocab = config.vocab_size;
    let experts = config.num_experts;
    let moe_i = config.moe_intermediate_size;
    let shared_i = config.shared_expert_intermediate_size;
    let gdn_v_heads = config.linear_num_value_heads;
    let gdn_v_dim = config.linear_value_head_dim;

    let mut specs = Vec::new();
    push_spec(
        &mut specs,
        format!("{TEXT_PREFIX}embed_tokens.weight"),
        WeightTensorDType::Bf16,
        &[vocab, hidden],
        WeightTensorSource::Safetensors,
        None,
        "token_embedding",
    );
    push_spec(
        &mut specs,
        format!("{TEXT_PREFIX}norm.weight"),
        WeightTensorDType::Bf16,
        &[hidden],
        WeightTensorSource::Safetensors,
        None,
        "final_norm",
    );
    push_spec(
        &mut specs,
        "lm_head.weight",
        WeightTensorDType::Bf16,
        &[vocab, hidden],
        WeightTensorSource::Safetensors,
        None,
        "lm_head",
    );

    for (layer_idx, layer_type) in config.layer_types.iter().copied().enumerate() {
        let layer = layer_idx as u32;
        let prefix = format!("{TEXT_PREFIX}layers.{layer_idx}");
        push_spec(
            &mut specs,
            format!("{prefix}.input_layernorm.weight"),
            WeightTensorDType::Bf16,
            &[hidden],
            WeightTensorSource::Safetensors,
            Some(layer),
            "input_layernorm",
        );
        push_spec(
            &mut specs,
            format!("{prefix}.post_attention_layernorm.weight"),
            WeightTensorDType::Bf16,
            &[hidden],
            WeightTensorSource::Safetensors,
            Some(layer),
            "post_attention_layernorm",
        );

        match layer_type {
            QwenLayerKind::FullAttention => {
                let attn = format!("{prefix}.self_attn");
                push_spec(
                    &mut specs,
                    format!("{attn}.q_norm.weight"),
                    WeightTensorDType::Bf16,
                    &[QWEN36_FULL_ATTN_HEAD_DIM],
                    WeightTensorSource::Safetensors,
                    Some(layer),
                    "attn.q_norm",
                );
                push_spec(
                    &mut specs,
                    format!("{attn}.k_norm.weight"),
                    WeightTensorDType::Bf16,
                    &[QWEN36_FULL_ATTN_HEAD_DIM],
                    WeightTensorSource::Safetensors,
                    Some(layer),
                    "attn.k_norm",
                );
                push_spec(
                    &mut specs,
                    format!("{attn}.q_proj.weight"),
                    WeightTensorDType::Bf16,
                    &[QWEN36_FULL_ATTN_Q_PROJ_OUT, hidden],
                    WeightTensorSource::Safetensors,
                    Some(layer),
                    "attn.q_proj",
                );
                push_spec(
                    &mut specs,
                    format!("{attn}.k_proj.weight"),
                    WeightTensorDType::Bf16,
                    &[QWEN36_FULL_ATTN_KV_HIDDEN, hidden],
                    WeightTensorSource::Safetensors,
                    Some(layer),
                    "attn.k_proj",
                );
                push_spec(
                    &mut specs,
                    format!("{attn}.v_proj.weight"),
                    WeightTensorDType::Bf16,
                    &[QWEN36_FULL_ATTN_KV_HIDDEN, hidden],
                    WeightTensorSource::Safetensors,
                    Some(layer),
                    "attn.v_proj",
                );
                push_spec(
                    &mut specs,
                    format!("{attn}.o_proj.weight"),
                    WeightTensorDType::Bf16,
                    &[hidden, QWEN36_FULL_ATTN_Q_HIDDEN],
                    WeightTensorSource::Safetensors,
                    Some(layer),
                    "attn.o_proj",
                );
            }
            QwenLayerKind::LinearAttention => {
                let gdn = format!("{prefix}.linear_attn");
                push_spec(
                    &mut specs,
                    format!("{gdn}.in_proj_qkv.weight"),
                    WeightTensorDType::Bf16,
                    &[QWEN36_GDN_PACKED_DIM, hidden],
                    WeightTensorSource::Safetensors,
                    Some(layer),
                    "gdn.in_proj_qkv",
                );
                push_spec(
                    &mut specs,
                    format!("{gdn}.in_proj_z.weight"),
                    WeightTensorDType::Bf16,
                    &[QWEN36_GDN_OUTPUT_DIM, hidden],
                    WeightTensorSource::Safetensors,
                    Some(layer),
                    "gdn.gate_proj_z",
                );
                push_spec(
                    &mut specs,
                    format!("{gdn}.in_proj_a.weight"),
                    WeightTensorDType::Bf16,
                    &[gdn_v_heads, hidden],
                    WeightTensorSource::Safetensors,
                    Some(layer),
                    "gdn.a_proj",
                );
                push_spec(
                    &mut specs,
                    format!("{gdn}.in_proj_b.weight"),
                    WeightTensorDType::Bf16,
                    &[gdn_v_heads, hidden],
                    WeightTensorSource::Safetensors,
                    Some(layer),
                    "gdn.b_proj",
                );
                push_spec(
                    &mut specs,
                    format!("{gdn}.conv1d.weight"),
                    WeightTensorDType::Bf16,
                    &[QWEN36_GDN_PACKED_DIM, 1, QWEN36_GDN_CONV_WIDTH],
                    WeightTensorSource::Safetensors,
                    Some(layer),
                    "gdn.conv_weight",
                );
                push_spec(
                    &mut specs,
                    format!("{gdn}.conv1d.bias"),
                    WeightTensorDType::Bf16,
                    &[QWEN36_GDN_PACKED_DIM],
                    WeightTensorSource::ZeroFill,
                    Some(layer),
                    "gdn.conv_bias.zero",
                );
                push_spec(
                    &mut specs,
                    format!("{gdn}.A_log"),
                    WeightTensorDType::Bf16,
                    &[gdn_v_heads],
                    WeightTensorSource::Safetensors,
                    Some(layer),
                    "gdn.a_log",
                );
                push_spec(
                    &mut specs,
                    format!("{gdn}.dt_bias"),
                    WeightTensorDType::Bf16,
                    &[gdn_v_heads],
                    WeightTensorSource::Safetensors,
                    Some(layer),
                    "gdn.dt_bias",
                );
                push_spec(
                    &mut specs,
                    format!("{gdn}.norm.weight"),
                    WeightTensorDType::Bf16,
                    &[gdn_v_dim],
                    WeightTensorSource::Safetensors,
                    Some(layer),
                    "gdn.rms_weight",
                );
                push_spec(
                    &mut specs,
                    format!("{gdn}.out_proj.weight"),
                    WeightTensorDType::Bf16,
                    &[hidden, QWEN36_GDN_OUTPUT_DIM],
                    WeightTensorSource::Safetensors,
                    Some(layer),
                    "gdn.out_proj",
                );
            }
        }

        let mlp = format!("{prefix}.mlp");
        push_spec(
            &mut specs,
            format!("{mlp}.gate.weight"),
            WeightTensorDType::Bf16,
            &[experts, hidden],
            WeightTensorSource::Safetensors,
            Some(layer),
            "mlp.router",
        );
        push_spec(
            &mut specs,
            format!("{mlp}.experts.gate_up_proj"),
            WeightTensorDType::Bf16,
            &[experts, 2 * moe_i, hidden],
            WeightTensorSource::Safetensors,
            Some(layer),
            "mlp.experts.gate_up",
        );
        push_spec(
            &mut specs,
            format!("{mlp}.experts.down_proj"),
            WeightTensorDType::Bf16,
            &[experts, hidden, moe_i],
            WeightTensorSource::Safetensors,
            Some(layer),
            "mlp.experts.down",
        );
        push_spec(
            &mut specs,
            format!("{mlp}.shared_expert.gate_proj.weight"),
            WeightTensorDType::Bf16,
            &[shared_i, hidden],
            WeightTensorSource::Safetensors,
            Some(layer),
            "mlp.shared.gate",
        );
        push_spec(
            &mut specs,
            format!("{mlp}.shared_expert.up_proj.weight"),
            WeightTensorDType::Bf16,
            &[shared_i, hidden],
            WeightTensorSource::Safetensors,
            Some(layer),
            "mlp.shared.up",
        );
        push_spec(
            &mut specs,
            format!("{mlp}.shared_expert.down_proj.weight"),
            WeightTensorDType::Bf16,
            &[hidden, shared_i],
            WeightTensorSource::Safetensors,
            Some(layer),
            "mlp.shared.down",
        );
        push_spec(
            &mut specs,
            format!("{mlp}.shared_expert_gate.weight"),
            WeightTensorDType::Bf16,
            &[1, hidden],
            WeightTensorSource::Safetensors,
            Some(layer),
            "mlp.shared.gate_score",
        );
    }

    Ok(specs)
}

fn push_spec(
    specs: &mut Vec<WeightTensorSpec>,
    name: impl Into<String>,
    dtype: WeightTensorDType,
    shape: &[u32],
    source: WeightTensorSource,
    layer: Option<u32>,
    slot: &'static str,
) {
    specs.push(WeightTensorSpec {
        name: name.into(),
        dtype,
        shape: shape.to_vec(),
        source,
        target: WeightTensorTarget { layer, slot },
    });
}

fn ignored_qwen36_tensor(name: &str) -> bool {
    name.starts_with("model.visual.")
        || name.starts_with("mtp.")
        || name.ends_with("rotary_emb.inv_freq")
}

fn validate_qwen36_bf16_tensor_table(
    tensors: &BTreeMap<String, TensorFileMeta>,
    config: &Qwen36TextConfig,
) -> LoadResult<Vec<ValidatedTensor>> {
    let specs = expected_qwen36_bf16_specs(config)?;
    let mut expected_names = HashSet::new();
    let mut validated = Vec::with_capacity(specs.len());
    for spec in specs {
        if spec.source == WeightTensorSource::Safetensors {
            expected_names.insert(spec.name.clone());
            let Some(file_meta) = tensors.get(&spec.name) else {
                return Err(WeightLoadError::tensor_table(format!(
                    "missing required tensor {:?}",
                    spec.name
                )));
            };
            if file_meta.meta.dtype != spec.dtype {
                return Err(WeightLoadError::tensor_table(format!(
                    "tensor {:?} dtype {:?}, expected {:?}",
                    spec.name, file_meta.meta.dtype, spec.dtype
                )));
            }
            if file_meta.meta.shape != spec.shape {
                return Err(WeightLoadError::tensor_table(format!(
                    "tensor {:?} shape {:?}, expected {:?}",
                    spec.name, file_meta.meta.shape, spec.shape
                )));
            }
            validated.push(ValidatedTensor {
                spec,
                file_meta: Some(file_meta.clone()),
            });
        } else {
            validated.push(ValidatedTensor {
                spec,
                file_meta: None,
            });
        }
    }
    for name in tensors.keys() {
        if !expected_names.contains(name) && !ignored_qwen36_tensor(name) {
            return Err(WeightLoadError::tensor_table(format!(
                "unexpected tensor {name:?}"
            )));
        }
    }
    Ok(validated)
}

#[derive(Clone, Debug)]
pub(crate) enum QwenLoadSource {
    FileRange {
        shard_path: PathBuf,
        absolute_offset: u64,
        bytes: usize,
    },
    ZeroFill {
        bytes: usize,
    },
}

#[derive(Clone, Debug)]
pub(crate) struct QwenLoadPlanEntry {
    spec: WeightTensorSpec,
    source: QwenLoadSource,
}

#[derive(Clone, Debug)]
pub(crate) struct QwenBf16LoadPlan {
    config: Qwen36TextConfig,
    entries: Vec<QwenLoadPlanEntry>,
}

impl QwenBf16LoadPlan {
    pub(crate) fn read(model_dir: impl AsRef<Path>) -> LoadResult<Self> {
        Self::read_with_limits(model_dir, DEFAULT_MAX_JSON_BYTES, DEFAULT_MAX_HEADER_BYTES)
    }

    pub(crate) fn tensor_count(&self) -> usize {
        self.entries.len()
    }

    pub(crate) fn file_bytes(&self) -> LoadResult<usize> {
        self.entries.iter().try_fold(0usize, |acc, entry| {
            let bytes = match &entry.source {
                QwenLoadSource::FileRange { bytes, .. } => *bytes,
                QwenLoadSource::ZeroFill { .. } => 0,
            };
            acc.checked_add(bytes)
                .ok_or_else(|| WeightLoadError::tensor_table("file byte count overflow"))
        })
    }

    pub(crate) fn zero_fill_bytes(&self) -> LoadResult<usize> {
        self.entries.iter().try_fold(0usize, |acc, entry| {
            let bytes = match &entry.source {
                QwenLoadSource::FileRange { .. } => 0,
                QwenLoadSource::ZeroFill { bytes } => *bytes,
            };
            acc.checked_add(bytes)
                .ok_or_else(|| WeightLoadError::tensor_table("zero-fill byte count overflow"))
        })
    }

    pub(crate) fn total_bytes(&self) -> LoadResult<usize> {
        self.file_bytes()?
            .checked_add(self.zero_fill_bytes()?)
            .ok_or_else(|| WeightLoadError::tensor_table("total byte count overflow"))
    }

    fn read_with_limits(
        model_dir: impl AsRef<Path>,
        max_json_bytes: usize,
        max_header_bytes: usize,
    ) -> LoadResult<Self> {
        let model_dir = model_dir.as_ref();
        let config = Qwen36TextConfig::read_with_limit(model_dir, max_json_bytes)?;
        let index = SafetensorsIndex::read_with_limit(model_dir, max_json_bytes)?;
        let table = read_indexed_safetensors_table(model_dir, &index, max_header_bytes)?;
        let validated = validate_qwen36_bf16_tensor_table(&table, &config)?;
        let mut entries = Vec::with_capacity(validated.len());
        for validated in validated {
            let bytes = validated.spec.byte_len()?;
            let source = match validated.file_meta {
                Some(file_meta) => QwenLoadSource::FileRange {
                    shard_path: model_dir.join(file_meta.shard),
                    absolute_offset: file_meta.absolute_offset,
                    bytes,
                },
                None => QwenLoadSource::ZeroFill { bytes },
            };
            entries.push(QwenLoadPlanEntry {
                spec: validated.spec,
                source,
            });
        }
        Ok(Self { config, entries })
    }
}

fn read_indexed_safetensors_table(
    model_dir: &Path,
    index: &SafetensorsIndex,
    max_header_bytes: usize,
) -> LoadResult<BTreeMap<String, TensorFileMeta>> {
    let mut table = BTreeMap::new();
    for shard in index.shard_names() {
        let header = SafetensorsHeader::read_with_limit(model_dir.join(&shard), max_header_bytes)?;
        for (name, meta) in header.tensors {
            if ignored_qwen36_tensor(&name) {
                continue;
            }
            match index.weight_map.get(&name) {
                Some(mapped_shard) if mapped_shard == &shard => {}
                Some(mapped_shard) => {
                    return Err(WeightLoadError::invalid_index(format!(
                        "tensor {name:?} is in shard {shard:?}, but index maps it to {mapped_shard:?}"
                    )));
                }
                None => {
                    return Err(WeightLoadError::invalid_index(format!(
                        "header tensor {name:?} is missing from weight_map"
                    )));
                }
            }
            let absolute_offset = header
                .data_start
                .checked_add(meta.data_offsets.0)
                .ok_or_else(|| {
                    WeightLoadError::invalid_safetensors(format!(
                        "absolute offset overflow for tensor {name:?}"
                    ))
                })?;
            if table
                .insert(
                    name,
                    TensorFileMeta {
                        shard: shard.clone(),
                        absolute_offset,
                        meta,
                    },
                )
                .is_some()
            {
                return Err(WeightLoadError::tensor_table(format!(
                    "duplicate tensor across shards in {shard:?}"
                )));
            }
        }
    }
    for (name, shard) in &index.weight_map {
        if ignored_qwen36_tensor(name) {
            continue;
        }
        if !table.contains_key(name) {
            return Err(WeightLoadError::invalid_index(format!(
                "index maps tensor {name:?} to shard {shard:?}, but the shard header does not contain it"
            )));
        }
    }
    Ok(table)
}

#[derive(Debug)]
pub(crate) struct LoadedWeightTensor {
    spec: WeightTensorSpec,
    span: WeightLoadSpan,
}

#[derive(Debug)]
pub(crate) struct LoadedWeightPlan<B: WeightLoadBackend> {
    config: Qwen36TextConfig,
    tensors: Vec<LoadedWeightTensor>,
    backend: B,
}

pub(crate) fn execute_qwen36_bf16_load_plan<B: WeightLoadBackend>(
    plan: &QwenBf16LoadPlan,
    mut backend: B,
    stream: *mut c_void,
) -> LoadResult<LoadedWeightPlan<B>> {
    let files = open_plan_files(plan)?;
    let mut tensors = Vec::with_capacity(plan.entries.len());
    for entry in &plan.entries {
        let bytes = match &entry.source {
            QwenLoadSource::FileRange { bytes, .. } | QwenLoadSource::ZeroFill { bytes } => *bytes,
        };
        let span = backend
            .alloc_tensor(WeightTensorDesc {
                name: &entry.spec.name,
                dtype: entry.spec.dtype.to_runtime_dtype(),
                shape: &entry.spec.shape,
                bytes,
            })
            .map_err(WeightLoadError::Backend)?;
        tensors.push(LoadedWeightTensor {
            spec: entry.spec.clone(),
            span,
        });
    }

    let mut read_order = Vec::new();
    let mut zero_order = Vec::new();
    for (idx, entry) in plan.entries.iter().enumerate() {
        match &entry.source {
            QwenLoadSource::FileRange { .. } => read_order.push(idx),
            QwenLoadSource::ZeroFill { .. } => zero_order.push(idx),
        }
    }
    read_order.sort_by(
        |&lhs, &rhs| match (&plan.entries[lhs].source, &plan.entries[rhs].source) {
            (
                QwenLoadSource::FileRange {
                    shard_path: lhs_shard,
                    absolute_offset: lhs_offset,
                    ..
                },
                QwenLoadSource::FileRange {
                    shard_path: rhs_shard,
                    absolute_offset: rhs_offset,
                    ..
                },
            ) => lhs_shard
                .cmp(rhs_shard)
                .then_with(|| lhs_offset.cmp(rhs_offset)),
            _ => std::cmp::Ordering::Equal,
        },
    );

    for idx in read_order {
        let QwenLoadSource::FileRange {
            shard_path,
            absolute_offset,
            bytes,
        } = &plan.entries[idx].source
        else {
            unreachable!();
        };
        let file = files.get(shard_path).ok_or_else(|| {
            WeightLoadError::Io(format!(
                "load plan file {} was not opened",
                shard_path.display()
            ))
        })?;
        backend
            .read_exact(
                WeightFileRange {
                    file,
                    offset: *absolute_offset,
                    bytes: *bytes,
                },
                &tensors[idx].span,
                stream,
            )
            .map_err(WeightLoadError::Backend)?;
    }

    for idx in zero_order {
        backend
            .zero_fill(&tensors[idx].span, stream)
            .map_err(WeightLoadError::Backend)?;
    }

    backend.seal(stream).map_err(WeightLoadError::Backend)?;
    Ok(LoadedWeightPlan {
        config: plan.config.clone(),
        tensors,
        backend,
    })
}

fn open_plan_files(plan: &QwenBf16LoadPlan) -> LoadResult<BTreeMap<PathBuf, File>> {
    let mut paths = BTreeSet::new();
    for entry in &plan.entries {
        if let QwenLoadSource::FileRange { shard_path, .. } = &entry.source {
            paths.insert(shard_path.clone());
        }
    }
    let mut files = BTreeMap::new();
    for path in paths {
        files.insert(path.clone(), File::open(path)?);
    }
    Ok(files)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::{env, path::PathBuf, ptr, time::Instant};

    const DEFAULT_REAL_QWEN36_BF16_DIR: &str = "/home/exo/.cache/huggingface/hub/models--Qwen--Qwen3.6-35B-A3B/snapshots/995ad96eacd98c81ed38be0c5b274b04031597b0";

    fn synthetic_safetensors(header: &str, data_len: usize) -> Vec<u8> {
        let mut out = Vec::new();
        out.extend_from_slice(&(header.len() as u64).to_le_bytes());
        out.extend_from_slice(header.as_bytes());
        out.resize(out.len() + data_len, 0);
        out
    }

    fn layer_types_json(layers: usize) -> String {
        (0..layers)
            .map(|idx| {
                if idx % 4 == 3 {
                    "\"full_attention\""
                } else {
                    "\"linear_attention\""
                }
            })
            .collect::<Vec<_>>()
            .join(",")
    }

    fn nested_text_config_json(layers: usize, vocab: u32) -> String {
        format!(
            r#"{{
                "model_type": "qwen3_5_moe",
                "tie_word_embeddings": false,
                "text_config": {{
                    "model_type": "qwen3_5_moe_text",
                    "dtype": "bfloat16",
                    "num_hidden_layers": {layers},
                    "hidden_size": 2048,
                    "vocab_size": {vocab},
                    "num_attention_heads": 16,
                    "num_key_value_heads": 2,
                    "head_dim": 256,
                    "num_experts": 256,
                    "num_experts_per_tok": 8,
                    "moe_intermediate_size": 512,
                    "shared_expert_intermediate_size": 512,
                    "linear_num_key_heads": 16,
                    "linear_num_value_heads": 32,
                    "linear_key_head_dim": 128,
                    "linear_value_head_dim": 128,
                    "linear_conv_kernel_dim": 4,
                    "full_attention_interval": 4,
                    "attn_output_gate": true,
                    "rms_norm_eps": 1e-6,
                    "rope_parameters": {{
                        "partial_rotary_factor": 0.25,
                        "rope_theta": 10000000
                    }},
                    "layer_types": [{}]
                }}
            }}"#,
            layer_types_json(layers)
        )
    }

    fn qwen_text_config(layers: usize, vocab: u32) -> Qwen36TextConfig {
        let root = parse_json_object(&nested_text_config_json(layers, vocab)).unwrap();
        Qwen36TextConfig::from_config_object(&root).unwrap()
    }

    fn real_qwen36_bf16_model_dir() -> Option<PathBuf> {
        let path = PathBuf::from(DEFAULT_REAL_QWEN36_BF16_DIR);
        path.exists().then_some(path)
    }

    fn cuda_device_from_env() -> i32 {
        env::var("QS3_CUDA_DEVICE")
            .ok()
            .and_then(|value| value.parse::<i32>().ok())
            .unwrap_or(0)
    }

    fn tensor_table_from_specs(specs: &[WeightTensorSpec]) -> BTreeMap<String, TensorFileMeta> {
        let mut table = BTreeMap::new();
        let mut offset = 0u64;
        for spec in specs {
            if spec.source == WeightTensorSource::ZeroFill {
                continue;
            }
            let bytes = spec.byte_len().unwrap() as u64;
            table.insert(
                spec.name.clone(),
                TensorFileMeta {
                    shard: "model-00001-of-00001.safetensors".to_owned(),
                    absolute_offset: offset,
                    meta: TensorMeta {
                        dtype: spec.dtype,
                        shape: spec.shape.clone(),
                        data_offsets: (offset, offset + bytes),
                    },
                },
            );
            offset += bytes;
        }
        table
    }

    #[test]
    fn parses_nested_qwen36_text_config_and_manifest() {
        let config = qwen_text_config(40, 248_320);
        assert_eq!(config.hidden_size, QWEN36_HIDDEN_SIZE);
        assert_eq!(config.rope_theta, 10_000_000.0);

        let specs = expected_qwen36_bf16_specs(&config).unwrap();
        assert_eq!(specs.len(), 723);
        assert!(specs.iter().any(|spec| {
            spec.name == "model.language_model.layers.3.self_attn.q_norm.weight"
                && spec.shape == vec![QWEN36_FULL_ATTN_HEAD_DIM]
        }));
        assert!(specs.iter().any(|spec| {
            spec.name == "model.language_model.layers.0.mlp.experts.gate_up_proj"
                && spec.shape == vec![256, 1024, 2048]
        }));
        assert!(specs.iter().any(|spec| {
            spec.name == "model.language_model.layers.0.linear_attn.conv1d.bias"
                && spec.source == WeightTensorSource::ZeroFill
        }));
        assert!(
            !specs
                .iter()
                .any(|spec| spec.name.ends_with("experts.gate_up_proj.weight"))
        );
    }

    #[test]
    fn rejects_bad_layer_schedule() {
        let mut json = nested_text_config_json(4, 16);
        json = json.replacen("\"full_attention\"", "\"linear_attention\"", 1);
        let root = parse_json_object(&json).unwrap();
        let err = Qwen36TextConfig::from_config_object(&root).unwrap_err();
        assert!(matches!(err, WeightLoadError::InvalidConfig(_)));
        assert!(err.to_string().contains("layer 3"));
    }

    #[test]
    fn parses_safetensors_header_and_rejects_trailing_data() {
        let header = r#"{"a":{"dtype":"BF16","shape":[2,3],"data_offsets":[0,12]},"b":{"dtype":"F32","shape":[1],"data_offsets":[12,16]}}"#;
        let parsed =
            SafetensorsHeader::parse_safetensors_bytes(&synthetic_safetensors(header, 16)).unwrap();
        assert_eq!(parsed.data_start, 8 + header.len() as u64);
        assert_eq!(parsed.tensors["a"].shape, vec![2, 3]);

        let err = SafetensorsHeader::parse_safetensors_bytes(&synthetic_safetensors(header, 17))
            .unwrap_err();
        assert!(matches!(err, WeightLoadError::InvalidSafetensors(_)));
        assert!(err.to_string().contains("trailing bytes"));
    }

    #[test]
    fn rejects_safetensors_gaps_and_wrong_byte_count() {
        let gap = r#"{"a":{"dtype":"BF16","shape":[1],"data_offsets":[1,3]}}"#;
        let err =
            SafetensorsHeader::parse_safetensors_bytes(&synthetic_safetensors(gap, 3)).unwrap_err();
        assert!(err.to_string().contains("expected contiguous offset 0"));

        let wrong_size = r#"{"a":{"dtype":"BF16","shape":[2],"data_offsets":[0,2]}}"#;
        let err = SafetensorsHeader::parse_safetensors_bytes(&synthetic_safetensors(wrong_size, 2))
            .unwrap_err();
        assert!(err.to_string().contains("expected 4"));
    }

    #[test]
    fn validates_complete_tensor_table_and_rejects_unexpected_text_tensor() {
        let config = qwen_text_config(4, 16);
        let specs = expected_qwen36_bf16_specs(&config).unwrap();
        let mut table = tensor_table_from_specs(&specs);
        let validated = validate_qwen36_bf16_tensor_table(&table, &config).unwrap();
        assert_eq!(validated.len(), specs.len());

        table.insert(
            "lm_head.input_scale".to_owned(),
            TensorFileMeta {
                shard: "model-00001-of-00001.safetensors".to_owned(),
                absolute_offset: 0,
                meta: TensorMeta {
                    dtype: WeightTensorDType::F32,
                    shape: vec![1],
                    data_offsets: (0, 4),
                },
            },
        );
        let err = validate_qwen36_bf16_tensor_table(&table, &config).unwrap_err();
        assert!(err.to_string().contains("unexpected tensor"));
    }

    #[test]
    fn rejects_missing_and_wrong_shape_required_tensor() {
        let config = qwen_text_config(4, 16);
        let specs = expected_qwen36_bf16_specs(&config).unwrap();
        let mut table = tensor_table_from_specs(&specs);
        table.remove("model.language_model.layers.3.self_attn.o_proj.weight");
        let err = validate_qwen36_bf16_tensor_table(&table, &config).unwrap_err();
        assert!(err.to_string().contains("missing required tensor"));

        let mut table = tensor_table_from_specs(&specs);
        table
            .get_mut("model.language_model.layers.0.linear_attn.A_log")
            .unwrap()
            .meta
            .shape = vec![31];
        let err = validate_qwen36_bf16_tensor_table(&table, &config).unwrap_err();
        assert!(err.to_string().contains("shape"));
    }

    #[derive(Default)]
    struct RecordingBackend {
        next_addr: usize,
        allocations: Vec<WeightLoadSpan>,
        allocs: Vec<(String, DynDType, Vec<u32>, usize)>,
        reads: Vec<(u64, usize)>,
        zeros: Vec<usize>,
        sealed: bool,
        dropped: Option<std::rc::Rc<std::cell::Cell<bool>>>,
    }

    impl Drop for RecordingBackend {
        fn drop(&mut self) {
            if let Some(dropped) = &self.dropped {
                dropped.set(true);
            }
        }
    }

    impl WeightLoadBackend for RecordingBackend {
        fn device_ordinal(&self) -> i32 {
            0
        }

        fn allocations(&self) -> &[WeightLoadSpan] {
            &self.allocations
        }

        fn take_allocations(&mut self) -> Vec<WeightLoadSpan> {
            std::mem::take(&mut self.allocations)
        }

        fn alloc_tensor(&mut self, desc: WeightTensorDesc<'_>) -> Result<WeightLoadSpan, Status> {
            self.next_addr += 0x1000;
            self.allocs.push((
                desc.name.to_owned(),
                desc.dtype,
                desc.shape.to_vec(),
                desc.bytes,
            ));
            let span = WeightLoadSpan {
                ptr: self.next_addr as ffi::DevicePtr,
                bytes: desc.bytes,
                memory: WeightLoadMemory::ManagedUma,
            };
            self.allocations.push(span);
            Ok(span)
        }

        fn read_exact(
            &mut self,
            src: WeightFileRange<'_>,
            _dst: &WeightLoadSpan,
            _stream: *mut c_void,
        ) -> Result<(), Status> {
            self.reads.push((src.offset, src.bytes));
            Ok(())
        }

        fn zero_fill(&mut self, dst: &WeightLoadSpan, _stream: *mut c_void) -> Result<(), Status> {
            self.zeros.push(dst.bytes);
            Ok(())
        }

        fn seal(&mut self, _stream: *mut c_void) -> Result<(), Status> {
            self.sealed = true;
            Ok(())
        }
    }

    #[test]
    fn executes_validated_plan_against_backend() {
        let tmp =
            std::env::temp_dir().join(format!("qs3-weight-loader-test-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&tmp);
        std::fs::create_dir(&tmp).unwrap();
        let shard = tmp.join("model-00001-of-00001.safetensors");
        std::fs::write(&shard, [0u8; 8]).unwrap();

        let plan = QwenBf16LoadPlan {
            config: qwen_text_config(4, 16),
            entries: vec![
                QwenLoadPlanEntry {
                    spec: WeightTensorSpec {
                        name: "a".to_owned(),
                        dtype: WeightTensorDType::Bf16,
                        shape: vec![2],
                        source: WeightTensorSource::Safetensors,
                        target: WeightTensorTarget {
                            layer: None,
                            slot: "a",
                        },
                    },
                    source: QwenLoadSource::FileRange {
                        shard_path: shard,
                        absolute_offset: 8,
                        bytes: 4,
                    },
                },
                QwenLoadPlanEntry {
                    spec: WeightTensorSpec {
                        name: "b".to_owned(),
                        dtype: WeightTensorDType::Bf16,
                        shape: vec![4],
                        source: WeightTensorSource::ZeroFill,
                        target: WeightTensorTarget {
                            layer: None,
                            slot: "b",
                        },
                    },
                    source: QwenLoadSource::ZeroFill { bytes: 8 },
                },
            ],
        };
        let dropped = std::rc::Rc::new(std::cell::Cell::new(false));
        let mut backend = RecordingBackend::default();
        backend.dropped = Some(dropped.clone());
        let loaded = execute_qwen36_bf16_load_plan(&plan, backend, ptr::null_mut()).unwrap();

        assert_eq!(loaded.tensors.len(), 2);
        assert_eq!(loaded.backend.allocs.len(), 2);
        assert_eq!(loaded.backend.reads, vec![(8, 4)]);
        assert_eq!(loaded.backend.zeros, vec![8]);
        assert!(loaded.backend.sealed);
        assert!(!dropped.get());
        drop(loaded);
        assert!(dropped.get());

        let _ = std::fs::remove_dir_all(&tmp);
    }

    #[test]
    fn validates_real_qwen36_bf16_manifest_when_available() {
        let Some(model_dir) = real_qwen36_bf16_model_dir() else {
            eprintln!(
                "skipping real Qwen3.6 BF16 manifest smoke; {} does not exist",
                DEFAULT_REAL_QWEN36_BF16_DIR
            );
            return;
        };

        let started = Instant::now();
        let plan = QwenBf16LoadPlan::read(&model_dir).unwrap();
        let file_bytes = plan.file_bytes().unwrap();
        let zero_fill_bytes = plan.zero_fill_bytes().unwrap();
        println!(
            "validated {} tensors from {} in {:.3}s: {:.3} GiB file bytes, {:.3} MiB zero-fill",
            plan.tensor_count(),
            model_dir.display(),
            started.elapsed().as_secs_f64(),
            file_bytes as f64 / (1u64 << 30) as f64,
            zero_fill_bytes as f64 / (1u64 << 20) as f64
        );

        assert_eq!(plan.config.num_hidden_layers, 40);
        assert_eq!(plan.tensor_count(), 723);
        assert_eq!(zero_fill_bytes, 30 * QWEN36_GDN_PACKED_DIM as usize * 2);
        assert!(file_bytes > 60usize << 30);
    }

    #[test]
    #[ignore = "reads the full real BF16 model into CUDA managed memory"]
    fn bench_real_qwen36_bf16_managed_uma_load() {
        let model_dir = real_qwen36_bf16_model_dir()
            .expect("run on spark-1565 with the hardcoded BF16 snapshot path");
        let plan_started = Instant::now();
        let plan = QwenBf16LoadPlan::read(&model_dir).unwrap();
        println!(
            "planned {} tensors in {:.3}s from {}",
            plan.tensor_count(),
            plan_started.elapsed().as_secs_f64(),
            model_dir.display()
        );

        let backend = ManagedUmaBackend::new(cuda_device_from_env()).unwrap();

        let load_started = Instant::now();
        let loaded = execute_qwen36_bf16_load_plan(&plan, backend, ptr::null_mut()).unwrap();
        let elapsed = load_started.elapsed();
        let loaded_tensors = loaded.tensors.len();
        let stats = loaded.backend.stats();
        let read_gib = stats.read_bytes as f64 / (1u64 << 30) as f64;
        println!(
            "loaded {} tensors: {:.3} GiB read, {:.3} MiB zero-fill, {:.3}s, {:.3} GiB/s",
            loaded_tensors,
            read_gib,
            stats.zero_fill_bytes as f64 / (1u64 << 20) as f64,
            elapsed.as_secs_f64(),
            read_gib / elapsed.as_secs_f64()
        );
        println!(
            "phases: alloc {:.3}s, read {:.3}s ({:.3} GiB/s), zero {:.6}s, seal {:.6}s",
            stats.alloc_us as f64 / 1_000_000.0,
            stats.read_us as f64 / 1_000_000.0,
            read_gib / (stats.read_us as f64 / 1_000_000.0),
            stats.zero_fill_us as f64 / 1_000_000.0,
            stats.seal_us as f64 / 1_000_000.0
        );

        assert_eq!(loaded_tensors, plan.tensor_count());
        assert_eq!(stats.tensors, plan.tensor_count());
        assert_eq!(stats.read_bytes, plan.file_bytes().unwrap());
        assert_eq!(stats.zero_fill_bytes, plan.zero_fill_bytes().unwrap());
        assert_eq!(stats.allocated_bytes, plan.total_bytes().unwrap());
        drop(loaded);
    }

    #[test]
    #[ignore = "reads the full real BF16 model through pinned staging into CUDA device memory"]
    fn bench_real_qwen36_bf16_pinned_upload_load() {
        let model_dir = real_qwen36_bf16_model_dir()
            .expect("run on spark-1565 with the hardcoded BF16 snapshot path");
        let plan_started = Instant::now();
        let plan = QwenBf16LoadPlan::read(&model_dir).unwrap();
        println!(
            "planned {} tensors in {:.3}s from {}",
            plan.tensor_count(),
            plan_started.elapsed().as_secs_f64(),
            model_dir.display()
        );

        let backend = PinnedUploadBackend::new(cuda_device_from_env()).unwrap();

        let load_started = Instant::now();
        let loaded = execute_qwen36_bf16_load_plan(&plan, backend, ptr::null_mut()).unwrap();
        let elapsed = load_started.elapsed();
        let loaded_tensors = loaded.tensors.len();
        let stats = loaded.backend.stats();
        let read_gib = stats.read_bytes as f64 / (1u64 << 30) as f64;
        println!(
            "loaded {} tensors: {:.3} GiB read, {:.3} MiB zero-fill, {:.3}s, {:.3} GiB/s",
            loaded_tensors,
            read_gib,
            stats.zero_fill_bytes as f64 / (1u64 << 20) as f64,
            elapsed.as_secs_f64(),
            read_gib / elapsed.as_secs_f64()
        );
        println!(
            "phases: alloc {:.3}s, wait {:.3}s, read {:.3}s ({:.3} GiB/s), copy-enqueue {:.3}s, zero {:.6}s, seal {:.6}s, chunks {}, buffers {} x {:.0} MiB",
            stats.alloc_us as f64 / 1_000_000.0,
            stats.wait_us as f64 / 1_000_000.0,
            stats.read_us as f64 / 1_000_000.0,
            read_gib / (stats.read_us as f64 / 1_000_000.0),
            stats.copy_enqueue_us as f64 / 1_000_000.0,
            stats.zero_fill_us as f64 / 1_000_000.0,
            stats.seal_us as f64 / 1_000_000.0,
            stats.chunks,
            PINNED_UPLOAD_BUFFER_COUNT,
            PINNED_UPLOAD_BUFFER_BYTES as f64 / (1u64 << 20) as f64
        );

        assert_eq!(loaded_tensors, plan.tensor_count());
        assert_eq!(stats.tensors, plan.tensor_count());
        assert_eq!(stats.read_bytes, plan.file_bytes().unwrap());
        assert_eq!(stats.zero_fill_bytes, plan.zero_fill_bytes().unwrap());
        assert_eq!(stats.allocated_bytes, plan.total_bytes().unwrap());
        drop(loaded);
    }
}
