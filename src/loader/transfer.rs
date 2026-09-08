use super::{PINNED_UPLOAD_BUFFER_BYTES, PINNED_UPLOAD_BUFFER_COUNT};
use crate::{
    engine::{DynDType, Status},
    ffi,
};

use std::{ffi::c_void, fs::File, io, os::fd::AsRawFd, ptr, time::Instant};

pub(super) fn result_from_cuda(err: i32) -> Result<(), Status> {
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
    let status = unsafe { ffi::cuda::cudaSetDevice(device_ordinal) };
    if status != ffi::cuda::CUDA_SUCCESS {
        let message = unsafe { std::ffi::CStr::from_ptr(ffi::cuda::cudaGetErrorString(status)) };
        eprintln!(
            "cudaSetDevice({device_ordinal}) failed: {} (CUDA error {status})",
            message.to_string_lossy(),
        );
    }
    result_from_cuda(status)
}

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
    /// Returns exactly the spans currently exposed by `allocations`, in the
    /// same order, and leaves `allocations()` empty.
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
    pub(super) name: &'a str,
    pub(super) dtype: DynDType,
    pub(super) shape: &'a [u32],
    pub(super) bytes: usize,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum WeightLoadMemory {
    ManagedUma,
    Device,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) struct WeightLoadSpan {
    pub(super) ptr: ffi::DevicePtr,
    pub(super) bytes: usize,
    pub(super) memory: WeightLoadMemory,
}

#[derive(Clone, Copy, Debug)]
pub(crate) struct WeightFileRange<'a> {
    pub(super) file: &'a File,
    pub(super) offset: u64,
    pub(super) bytes: usize,
}

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub(crate) struct ManagedUmaLoadStats {
    pub(super) tensors: usize,
    pub(super) allocated_bytes: usize,
    pub(super) read_bytes: usize,
    pub(super) zero_fill_bytes: usize,
    pub(super) alloc_us: u128,
    pub(super) read_us: u128,
    pub(super) zero_fill_us: u128,
    pub(super) seal_us: u128,
}

/// GB10/UMA loader backend: final weights live in CUDA managed memory and file
/// payloads are read directly into those committed allocations.
pub(crate) struct ManagedUmaBackend {
    pub(super) device_ordinal: i32,
    pub(super) allocations: Vec<WeightLoadSpan>,
    pub(super) stats: ManagedUmaLoadStats,
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
    pub(super) tensors: usize,
    pub(super) allocated_bytes: usize,
    pub(super) read_bytes: usize,
    pub(super) zero_fill_bytes: usize,
    pub(super) chunks: usize,
    pub(super) alloc_us: u128,
    pub(super) wait_us: u128,
    pub(super) read_us: u128,
    pub(super) copy_enqueue_us: u128,
    pub(super) zero_fill_us: u128,
    pub(super) seal_us: u128,
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
    pub(super) device_ordinal: i32,
    pub(super) allocations: Vec<WeightLoadSpan>,
    slots: Vec<PinnedUploadSlot>,
    pub(super) next_slot: usize,
    pub(super) stats: PinnedUploadLoadStats,
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
