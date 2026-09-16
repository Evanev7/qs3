//! Device storage and operations ordered on an owned CUDA stream.
//!
//! Except for `synchronize` and `HostBuffer::upload`, operations enqueue
//! work without establishing host-visible completion. A successful stream wait,
//! or a successful wait/query on an event recorded after the operation, establishes
//! completion. An error does not establish completion. Rust borrows alone do not
//! keep transfer storage alive after an asynchronous method returns.

use crate::{
    backend::{Bf16Heads, DMat, DTensor3, DVec, Workspace},
    dtype::{BF16, DType, U8},
    engine::Status,
    ffi,
    ffi::cuda,
    model::result_from_cuda,
};
use std::{
    alloc::{Layout, alloc_zeroed, dealloc, handle_alloc_error},
    marker::PhantomData,
    mem,
    ops::Deref,
    ptr,
    rc::Rc,
};

/// Owns a CUDA stream on one device.
///
/// Raw-handle users must not replace or destroy this stream. Cross-stream uses of
/// allocations require dependencies ordering allocation, accesses, and release.
/// Dropping the context destroys the stream handle but does not wait for pending
/// host transfers; their host storage must still survive until completion.
#[derive(Debug)]
pub struct CudaCtx {
    device_ordinal: i32,
    // TODO: make private. or more private, anyway
    pub(crate) stream: ffi::CudaStream,
}

impl CudaCtx {
    pub fn default() -> Result<Self, Status> {
        let mut dev = 0;
        unsafe {
            result_from_cuda(cuda::cudaGetDevice(&raw mut dev))?;
        }
        CudaCtx::new(dev)
    }

    pub fn new(device_ordinal: i32) -> Result<Self, Status> {
        let mut stream = ptr::null_mut();
        unsafe {
            result_from_cuda(cuda::cudaSetDevice(device_ordinal))?;
            result_from_cuda(cuda::cudaStreamCreateWithFlags(
                &mut stream,
                cuda::CUDA_STREAM_NON_BLOCKING,
            ))?;
        }
        Ok(Self {
            device_ordinal,
            stream,
        })
    }
    pub(crate) fn device_ordinal(&self) -> i32 {
        self.device_ordinal
    }

    pub(crate) fn activate(&self) -> Result<(), Status> {
        let mut current_dev = 0;
        result_from_cuda(unsafe { cuda::cudaGetDevice(&raw mut current_dev) })?;
        if self.device_ordinal < 0 || self.device_ordinal == current_dev {
            return Ok(());
        }
        result_from_cuda(unsafe { cuda::cudaSetDevice(self.device_ordinal) })
    }
    /// Waits for work previously submitted to this stream.
    ///
    /// Only `Ok(())` establishes successful completion. Errors may originate from
    /// earlier asynchronous operations.
    pub fn synchronize(&self) -> Result<(), Status> {
        self.activate()?;
        result_from_cuda(unsafe { cuda::cudaStreamSynchronize(self.stream) })
    }
    /// Allocates uninitialized storage for `bytes` bytes on this stream.
    ///
    /// The returned pointer carries ownership responsibility, not a Rust value or
    /// lifetime. Its allocation must precede every use, and its eventual release
    /// must follow every use. This method does not initialize the allocation.
    fn alloc(&self, bytes: usize) -> Result<*mut u8, Status> {
        if bytes == 0 {
            return Err(Status::InvalidArgument);
        };
        let mut ptr = ptr::null_mut();
        self.activate()?;
        result_from_cuda(unsafe { cuda::cudaMallocAsync(&mut ptr, bytes, self.stream) })?;
        Ok(ptr.cast())
    }

    /// Enqueues release of an allocation on this stream.
    ///
    /// # Safety
    /// `ptr` must be an allocation base accepted by `cudaFreeAsync`, and the caller
    /// must have exclusive authority to release it. Its allocation and all uses,
    /// including uses on other streams, must be ordered before this release.
    /// After a successful call, do not submit further uses or free it again.
    unsafe fn free(&self, ptr: *mut u8) -> Result<(), Status> {
        result_from_cuda(unsafe { cuda::cudaFreeAsync(ptr.cast(), self.stream) })
    }

    /// Grows an allocation if necessary, preserving `old_bytes` bytes.
    ///
    /// On success, ownership responsibility moves to the returned pointer. If no
    /// growth was needed it equals `ptr`; otherwise release of `ptr` was enqueued
    /// after the copy, and the added capacity is uninitialized. On error, no
    /// replacement is handed off and the caller remains responsible for `ptr`;
    /// an error does not establish completion or permit assuming pending work stopped.
    ///
    /// # Safety
    /// `ptr` must be a live allocation base accepted by `cudaFreeAsync`, covering
    /// at least `old_bytes` bytes, with exclusive release authority
    /// held by the caller. Prior writes must be ordered before the copy, and all
    /// uses of the old allocation must be ordered before its release. No other
    /// owner may later free the retired pointer or submit further accesses to it.
    unsafe fn realloc(
        &self,
        ptr: *mut u8,
        old_bytes: usize,
        new_bytes: usize,
    ) -> Result<*mut u8, Status> {
        if new_bytes <= old_bytes {
            return Ok(ptr);
        }
        self.activate()?;
        let dst = self.alloc(new_bytes)?;
        unsafe {
            if let Err(e) = self.memcpy(ptr, dst, old_bytes) {
                _ = self.free(dst);
                return Err(e);
            }
            if let Err(e) = self.free(ptr) {
                _ = self.free(dst);
                return Err(e);
            }
        };
        Ok(dst)
    }

    /// Enqueues a device-to-device copy of `bytes` bytes.
    ///
    /// # Safety
    /// `src` and `dst` must describe non-overlapping CUDA-accessible device regions
    /// covering that byte count. Both allocations must be available before the
    /// copy and remain valid through completion. Source writes and conflicting
    /// destination accesses must be ordered around the copy, including accesses
    /// on other streams. Allocation release may be enqueued after the copy.
    unsafe fn memcpy(&self, src: *const u8, dst: *mut u8, bytes: usize) -> Result<(), Status> {
        self.activate()?;
        result_from_cuda(unsafe {
            cuda::cudaMemcpyAsync(
                dst.cast(),
                src.cast(),
                bytes,
                cuda::CUDA_MEMCPY_DEVICE_TO_DEVICE,
                self.stream,
            )
        })
    }

    /// Enqueues a host-to-device copy of `src.len()` bytes.
    ///
    /// # Safety
    /// `dst` must cover that many writable device bytes and must not overlap the
    /// source. The host source allocation must remain at the same address and
    /// unmodified until completion, beyond the lifetime of this call's borrow.
    /// The destination allocation must be available before the copy and remain
    /// valid through it; conflicting accesses and release must be stream-ordered.
    unsafe fn upload(&self, src: &[u8], dst: *mut u8) -> Result<(), Status> {
        self.activate()?;
        result_from_cuda(unsafe {
            cuda::cudaMemcpyAsync(
                dst.cast(),
                src.as_ptr().cast(),
                src.len(),
                cuda::CUDA_MEMCPY_HOST_TO_DEVICE,
                self.stream,
            )
        })
    }

    /// Enqueues a device-to-host copy of `dst.len()` bytes.
    ///
    /// # Safety
    /// `src` must cover that many readable device bytes and must not overlap
    /// the destination. Its allocation must remain
    /// valid through the copy, with writes and release ordered around it.
    /// The host destination allocation must remain at the same address and must
    /// not be read, modified, reallocated, or dropped until completion, beyond
    /// the lifetime of this call's mutable borrow.
    pub(crate) unsafe fn download(&self, src: *const u8, dst: &mut [u8]) -> Result<(), Status> {
        self.activate()?;
        result_from_cuda(unsafe {
            cuda::cudaMemcpyAsync(
                dst.as_mut_ptr().cast(),
                src.cast(),
                dst.len(),
                cuda::CUDA_MEMCPY_DEVICE_TO_HOST,
                self.stream,
            )
        })
    }

    /// Enqueues a zero-byte fill of `bytes` bytes.
    ///
    /// # Safety
    /// `ptr` must cover that many writable device bytes. Its allocation must be
    /// available before the fill and remain valid through completion. Conflicting
    /// accesses and release, including on other streams, must be ordered around
    /// the fill.
    unsafe fn zero(&self, ptr: *mut u8, bytes: usize) -> Result<(), Status> {
        self.activate()?;
        result_from_cuda(unsafe { cuda::cudaMemsetAsync(ptr.cast(), 0, bytes, self.stream) })
    }
}

impl Drop for CudaCtx {
    fn drop(&mut self) {
        unsafe { cuda::cudaStreamDestroy(self.stream) };
    }
}

#[derive(Debug)]
pub(crate) struct DeviceSpan<D: DType> {
    ptr: ffi::DevicePtr<D>,
    pub(crate) len: usize,
}

impl<D: DType> DeviceSpan<D> {
    /// Describes device storage without taking ownership or proving its lifetime.
    /// `len` counts elements. Unsafe users must establish that the allocation is
    /// live, accessible, and large enough before submitting device work.
    pub(crate) fn new(ptr: *mut u8, len: usize) -> Result<Self, Status> {
        D::size_of(len)?;
        if !(ptr as usize).is_multiple_of(D::ALIGN) {
            return Err(Status::InvalidArgument);
        }
        Ok(Self {
            ptr: ffi::DevicePtr::new(ptr).ok_or(Status::InvalidArgument)?,
            len,
        })
    }

    pub(crate) fn check_view_len(&self, len: usize) -> Result<(), Status> {
        if len == 0 || len > self.len {
            return Err(Status::InvalidArgument);
        }
        Ok(())
    }

    pub fn as_ptr(&self) -> ffi::DevicePtr<D> {
        self.ptr
    }
}

impl<D: DType> Deref for DeviceSpan<D> {
    type Target = ffi::DevicePtr<D>;

    fn deref(&self) -> &Self::Target {
        &self.ptr
    }
}

// These descriptors check shapes against the declared extent, but do not own or
// borrow the allocation. Unsafe launches must establish allocation validity and
// retain its owner until all uses complete.
impl<D: DType> DeviceSpan<D> {
    pub(crate) fn vector(&self, len: u32) -> Result<DVec<D>, Status> {
        self.check_view_len(len as usize)?;
        DVec::contiguous(self.ptr, len)
    }

    pub(crate) fn matrix(&self, rows: u32, cols: u32) -> Result<DMat<D>, Status> {
        let len = (rows as usize)
            .checked_mul(cols as usize)
            .ok_or(Status::InvalidArgument)?;
        self.check_view_len(len)?;
        DMat::contiguous(self.ptr, rows, cols)
    }

    pub(crate) fn tensor3(&self, a: u32, b: u32, c: u32) -> Result<DTensor3<D>, Status> {
        let len = [a, b, c].into_iter().try_fold(1usize, |len, dim| {
            len.checked_mul(dim as usize).ok_or(Status::InvalidArgument)
        })?;
        self.check_view_len(len)?;
        DTensor3::contiguous(self.ptr, a, b, c)
    }
}

impl DeviceSpan<BF16> {
    pub(crate) fn heads(&self, rows: u32, heads: u32, dim: u32) -> Result<Bf16Heads, Status> {
        let len = [rows, heads, dim]
            .into_iter()
            .try_fold(1usize, |len, dim| {
                len.checked_mul(dim as usize).ok_or(Status::InvalidArgument)
            })?;
        self.check_view_len(len)?;
        Bf16Heads::contiguous(self.ptr, rows, heads, dim)
    }
}

impl DeviceSpan<U8> {
    pub(crate) fn workspace(&self, bytes: usize) -> Result<Workspace, Status> {
        if bytes == 0 {
            return Ok(Workspace::none());
        }
        self.check_view_len(bytes)?;
        Workspace::new(self.erase(), bytes)
    }
}

/// Owns dtype-sized device storage; Drop releases it on the context's stream.
/// `len` counts logical elements, including for packed dtypes.
#[derive(Debug)]
pub(crate) struct DeviceBuffer<D: DType> {
    pub(crate) span: DeviceSpan<D>,
    ctx: Rc<CudaCtx>,
}

impl<D: DType> DeviceBuffer<D> {
    pub(crate) fn with_capacity(ctx: Rc<CudaCtx>, len: usize) -> Result<Self, Status> {
        let bytes = D::size_of(len)?;
        Ok(Self {
            span: DeviceSpan::new(ctx.alloc(bytes)?, len)?,
            ctx,
        })
    }

    pub(crate) fn realloc(&mut self, len: usize) -> Result<&mut Self, Status> {
        let old_bytes = D::size_of(self.len)?;
        let new_bytes = D::size_of(len)?;
        // SAFETY: self owns storage; all accesses must be ordered on this stream.
        let ptr = unsafe { self.ctx.realloc(self.as_raw(), old_bytes, new_bytes)? };
        self.span = DeviceSpan::new(ptr, self.len.max(len))?;
        Ok(self)
    }

    /// Grows storage as needed and uploads an encoded element prefix.
    /// Host buffers contain whole bytes and whole DTypes, no slot is partially filled.
    ///
    /// # Safety
    /// Source bytes must remain live and unmodified until the stream completes.
    /// External device accesses must be ordered around the upload and any growth.
    pub(crate) unsafe fn upload(&mut self, buf: &HostBuffer<D>) -> Result<(), Status> {
        if buf.len() == 0 {
            return Ok(());
        }
        self.realloc(buf.len())?;
        unsafe { self.ctx.upload(buf.as_ref(), self.as_raw()) }
    }

    /// Downloads an encoded prefix sized by the host destination.
    ///
    /// # Safety
    /// Destination must remain live and untouched until the stream completes.
    /// External device writes must be ordered around this copy.
    pub(crate) unsafe fn download(&self, out: &mut HostBuffer<D>) -> Result<(), Status> {
        unsafe { self.download_range(0, out) }
    }

    /// Downloads encoded elements from a byte-aligned element offset.
    ///
    /// # Safety
    /// Same lifetime and ordering requirements as `download`.
    pub(crate) unsafe fn download_range(
        &self,
        offset: usize,
        out: &mut HostBuffer<D>,
    ) -> Result<(), Status> {
        if offset
            .checked_add(out.len())
            .ok_or(Status::InvalidArgument)?
            > self.len
        {
            return Err(Status::InvalidArgument);
        }
        if out.is_empty() {
            return Ok(());
        }
        let bytes = D::size_of(offset)?;
        unsafe { self.ctx.download(self.as_raw().add(bytes), out.as_mut()) }
    }

    pub(crate) fn zero(&mut self) -> Result<(), Status> {
        unsafe { self.ctx.zero(self.as_raw(), D::size_of(self.len())?) }
    }

    pub fn len(&self) -> usize {
        self.len
    }
}

impl<D: DType> Drop for DeviceBuffer<D> {
    fn drop(&mut self) {
        unsafe { _ = self.ctx.free(self.as_raw()) };
    }
}
impl<D: DType> Deref for DeviceBuffer<D> {
    type Target = DeviceSpan<D>;
    fn deref(&self) -> &Self::Target {
        &self.span
    }
}

// no HostSlice if offered until proven useful
#[derive(Debug)]
pub(crate) struct HostBuffer<D: DType> {
    // contract: always a valid number of bytes based on dtype.
    // always a valid number of dtype elements based on bytes.
    // i.e. storage.len() is a multiple of 4 for F32s
    // and always stores an even number of F4s
    ptr: ptr::NonNull<u8>,
    size: usize,
    dtype: PhantomData<D>,
}

impl<D: DType> HostBuffer<D> {
    pub(crate) fn new(len: usize) -> Result<Self, Status> {
        let size = D::size_of(len)?;
        let layout =
            Layout::from_size_align(size, D::ALIGN).map_err(|_| Status::InvalidArgument)?;
        // SAFETY: 0 is a u8
        let ptr = if size == 0 {
            layout.dangling_ptr()
        } else {
            ptr::NonNull::new(unsafe { alloc_zeroed(layout) })
                .unwrap_or_else(|| handle_alloc_error(layout))
        };

        Ok(Self {
            ptr,
            size,
            dtype: PhantomData,
        })
    }
    pub(crate) fn len(&self) -> usize {
        D::len_of(self.size).expect("assertion of container")
    }
    pub(crate) fn is_empty(&self) -> bool {
        self.len() == 0
    }
    pub(crate) fn upload(self, ctx: Rc<CudaCtx>) -> Result<DeviceBuffer<D>, Status> {
        let mut buffer = DeviceBuffer::with_capacity(ctx, self.len())?;
        // SAFETY: source remains borrowed until the stream completes.
        unsafe {
            buffer.upload(&self)?;
        }
        buffer.ctx.synchronize().inspect_err(|_| {
            eprintln!(
                "CUDA upload failed to synchronize as a previous operation failed
                We cannot be certain that the upload has been cancelled, so {}
                bytes have been leaked.",
                self.as_ref().len()
            );
            mem::forget(self)
        })?;
        Ok(buffer)
    }

}
impl<D: DType> AsRef<[u8]> for HostBuffer<D> {
    fn as_ref(&self) -> &[u8] {
        unsafe { &*ptr::slice_from_raw_parts(self.ptr.as_ptr(), self.size) }
    }
}
impl<D: DType> AsMut<[u8]> for HostBuffer<D> {
    fn as_mut(&mut self) -> &mut [u8] {
        unsafe { &mut *ptr::slice_from_raw_parts_mut(self.ptr.as_ptr(), self.size) }
    }
}

impl<D: DType> Drop for HostBuffer<D> {
    fn drop(&mut self) {
        if self.size != 0 {
            let layout =
                Layout::from_size_align(self.size, D::ALIGN).expect("validated on construction");
            unsafe { dealloc(self.ptr.as_ptr(), layout) }
        }
    }
}
