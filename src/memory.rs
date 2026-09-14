//! Device storage and operations ordered on an owned CUDA stream.
//!
//! Except for `synchronize` and `DeviceBuffer::from_slice`, operations enqueue
//! work without establishing host-visible completion. A successful stream wait,
//! or a successful wait/query on an event recorded after the operation, establishes
//! completion. An error does not establish completion. Rust borrows alone do not
//! keep transfer storage alive after an asynchronous method returns.

use crate::{backend::DeviceElement, engine::Status, ffi, ffi::cuda, model::result_from_cuda};
use std::{mem, ops::Deref, ptr, rc::Rc};

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
    /// Allocates uninitialized storage for `len` elements on this stream.
    ///
    /// The returned pointer carries ownership responsibility, not a Rust value or
    /// lifetime. Its allocation must precede every use, and its eventual release
    /// must follow every use. This method does not initialize valid `T` values.
    fn alloc<T>(&self, len: usize) -> Result<*mut T, Status> {
        if len == 0 {
            return Err(Status::InvalidArgument);
        };
        let bytes = len
            .checked_mul(mem::size_of::<T>())
            .ok_or(Status::InvalidArgument)?;
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
    unsafe fn free<T>(&self, ptr: *mut T) -> Result<(), Status> {
        result_from_cuda(unsafe { cuda::cudaFreeAsync(ptr.cast(), self.stream) })
    }

    /// Grows an allocation if necessary, preserving `old_cap` elements' bytes.
    ///
    /// On success, ownership responsibility moves to the returned pointer. If no
    /// growth was needed it equals `ptr`; otherwise release of `ptr` was enqueued
    /// after the copy, and the added capacity is uninitialized. On error, no
    /// replacement is handed off and the caller remains responsible for `ptr`;
    /// an error does not establish completion or permit assuming pending work stopped.
    ///
    /// # Safety
    /// `ptr` must be a live allocation base accepted by `cudaFreeAsync`, covering
    /// at least `old_cap * size_of::<T>()` bytes, with exclusive release authority
    /// held by the caller. Prior writes must be ordered before the copy, and all
    /// uses of the old allocation must be ordered before its release. No other
    /// owner may later free the retired pointer or submit further accesses to it.
    unsafe fn realloc<T>(
        &self,
        ptr: *mut T,
        old_cap: usize,
        new_cap: usize,
    ) -> Result<*mut T, Status> {
        if new_cap <= old_cap {
            return Ok(ptr);
        }
        self.activate()?;
        let dst = self.alloc::<T>(new_cap)?;
        unsafe {
            if let Err(e) = self.memcpy(ptr, dst, old_cap) {
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

    /// Enqueues a device-to-device copy of `cap * size_of::<T>()` bytes.
    ///
    /// # Safety
    /// `src` and `dst` must describe non-overlapping CUDA-accessible device regions
    /// covering that byte count. Both allocations must be available before the
    /// copy and remain valid through completion. Source writes and conflicting
    /// destination accesses must be ordered around the copy, including accesses
    /// on other streams. Allocation release may be enqueued after the copy.
    unsafe fn memcpy<T>(&self, src: *const T, dst: *mut T, cap: usize) -> Result<(), Status> {
        let bytes = cap
            .checked_mul(mem::size_of::<T>())
            .ok_or(Status::InvalidArgument)?;
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

    /// Enqueues a host-to-device copy of `size_of_val(src)` bytes.
    ///
    /// # Safety
    /// `dst` must cover that many writable device bytes and must not overlap the
    /// source. The host source allocation must remain at the same address and
    /// unmodified until completion, beyond the lifetime of this call's borrow.
    /// The destination allocation must be available before the copy and remain
    /// valid through it; conflicting accesses and release must be stream-ordered.
    unsafe fn upload<T>(&self, src: &[T], dst: *mut T) -> Result<(), Status> {
        self.activate()?;
        result_from_cuda(unsafe {
            cuda::cudaMemcpyAsync(
                dst.cast(),
                src.as_ptr().cast(),
                mem::size_of_val(src),
                cuda::CUDA_MEMCPY_HOST_TO_DEVICE,
                self.stream,
            )
        })
    }

    /// Enqueues a device-to-host copy of `size_of_val(dst)` bytes.
    ///
    /// # Safety
    /// `src` must cover that many readable device bytes representing valid `T`
    /// values and must not overlap the destination. Its allocation must remain
    /// valid through the copy, with writes and release ordered around it.
    /// The host destination allocation must remain at the same address and must
    /// not be read, modified, reallocated, or dropped until completion, beyond
    /// the lifetime of this call's mutable borrow.
    pub(crate) unsafe fn download<T>(&self, src: *const T, dst: &mut [T]) -> Result<(), Status> {
        self.activate()?;
        result_from_cuda(unsafe {
            cuda::cudaMemcpyAsync(
                dst.as_mut_ptr().cast(),
                src.cast(),
                mem::size_of_val(dst),
                cuda::CUDA_MEMCPY_DEVICE_TO_HOST,
                self.stream,
            )
        })
    }

    /// Enqueues a zero-byte fill of `len * size_of::<T>()` bytes.
    ///
    /// # Safety
    /// `ptr` must cover that many writable device bytes. Its allocation must be
    /// available before the fill and remain valid through completion. Conflicting
    /// accesses and release, including on other streams, must be ordered around
    /// the fill.
    unsafe fn zero<T: DeviceElement>(&self, ptr: *mut T, len: usize) -> Result<(), Status> {
        let bytes = len
            .checked_mul(mem::size_of::<T>())
            .ok_or(Status::InvalidArgument)?;
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
pub(crate) struct DeviceSpan<T> {
    ptr: ffi::DevicePtr<T>,
    pub(crate) cap: usize,
}

impl<T> DeviceSpan<T> {
    /// Describes device storage without taking ownership or proving its lifetime.
    /// `cap` counts elements. Unsafe users must establish that the allocation is
    /// live, accessible, and large enough before submitting device work.
    pub(crate) fn new(ptr: *mut T, cap: usize) -> Result<Self, Status> {
        Ok(Self {
            ptr: ffi::DevicePtr::new(ptr).ok_or(Status::InvalidArgument)?,
            cap,
        })
    }

    pub(crate) fn check_view_len(&self, len: usize) -> Result<(), Status> {
        if len == 0 || len > self.cap {
            return Err(Status::InvalidArgument);
        }
        Ok(())
    }
}

impl<T> Deref for DeviceSpan<T> {
    type Target = ffi::DevicePtr<T>;

    fn deref(&self) -> &Self::Target {
        &self.ptr
    }
}

// These descriptors check shapes against the declared extent, but do not own or
// borrow the allocation. Unsafe launches must establish allocation validity and
// retain its owner until all uses complete.
impl<T: DeviceElement> DeviceSpan<T> {
    pub(crate) fn matrix(
        &self,
        rows: u32,
        cols: u32,
    ) -> Result<crate::backend::DMat<T::DType>, Status> {
        self.matrix_at(0, rows, cols)
    }

    pub(crate) fn matrix_at(
        &self,
        offset: usize,
        rows: u32,
        cols: u32,
    ) -> Result<crate::backend::DMat<T::DType>, Status> {
        let len = (rows as usize)
            .checked_mul(cols as usize)
            .ok_or(Status::InvalidArgument)?;
        self.check_view_len(offset.checked_add(len).ok_or(Status::InvalidArgument)?)?;
        // A span does not prove that the backing allocation is still live.
        // Wrapping arithmetic constructs metadata without requiring that proof;
        // dereferencing the resulting address remains an unsafe launch obligation.
        crate::backend::DMat::contiguous(self.ptr.as_raw().wrapping_add(offset).cast(), rows, cols)
    }

    pub(crate) fn vector(&self, len: u32) -> Result<crate::backend::DVec<T::DType>, Status> {
        self.check_view_len(len as usize)?;
        crate::backend::DVec::contiguous(self.erase(), len)
    }

    pub(crate) fn tensor3(
        &self,
        a: u32,
        b: u32,
        c: u32,
    ) -> Result<crate::backend::DTensor3<T::DType>, Status> {
        let len = [a, b, c].into_iter().try_fold(1usize, |len, dim| {
            len.checked_mul(dim as usize).ok_or(Status::InvalidArgument)
        })?;
        self.check_view_len(len)?;
        crate::backend::DTensor3::contiguous(self.erase(), a, b, c)
    }
}

impl DeviceSpan<u16> {
    pub(crate) fn heads(
        &self,
        rows: u32,
        heads: u32,
        dim: u32,
    ) -> Result<crate::backend::Bf16Heads, Status> {
        let len = [rows, heads, dim]
            .into_iter()
            .try_fold(1usize, |len, dim| {
                len.checked_mul(dim as usize).ok_or(Status::InvalidArgument)
            })?;
        self.check_view_len(len)?;
        crate::backend::Bf16Heads::contiguous(self.erase(), rows, heads, dim)
    }
}

impl DeviceSpan<u8> {
    pub(crate) fn workspace(&self, bytes: usize) -> Result<crate::backend::Workspace, Status> {
        if bytes == 0 {
            return Ok(crate::backend::Workspace::none());
        }
        self.check_view_len(bytes)?;
        crate::backend::Workspace::new(self.erase(), bytes)
    }
}

/// Owns device storage; dropping it enqueues release on its context's stream.
///
/// Raw-pointer users must preserve the pointer/capacity ownership invariants and
/// order all external uses before growth or destruction can release the storage.
/// Allocation alone does not establish initialized contents.
#[derive(Debug)]
pub(crate) struct DeviceBuffer<T> {
    pub(crate) span: DeviceSpan<T>,
    ctx: Rc<CudaCtx>,
}

impl<T> DeviceBuffer<T> {
    pub(crate) fn with_capacity(ctx: Rc<CudaCtx>, cap: usize) -> Result<Self, Status> {
        Ok(Self {
            span: DeviceSpan::new(ctx.alloc::<T>(cap)?, cap)?,
            ctx,
        })
    }
    pub(crate) fn realloc(&mut self, cap: usize) -> Result<&mut Self, Status> {
        // SAFETY: self owns the allocation described by ptr/cap. Buffer operations
        // share the context stream; raw-pointer users must order external uses.
        // On success, replace the retired pointer without separately freeing it.
        self.span = DeviceSpan::new(
            unsafe { self.ctx.realloc(self.ptr.as_raw(), self.cap, cap) }?,
            self.cap.max(cap),
        )?;
        Ok(self)
    }

    pub(crate) fn from_slice(ctx: Rc<CudaCtx>, values: &[T]) -> Result<Self, Status>
    where
        T: Copy,
    {
        let mut buffer = Self::with_capacity(ctx, values.len())?;
        // SAFETY: values stays borrowed through the following synchronization;
        // on success the upload has completed before returning to the caller.
        unsafe {
            buffer.upload(values)?;
        }
        buffer.ctx.synchronize()?;
        Ok(buffer)
    }

    /// Grows storage if needed and enqueues an upload of `values`.
    ///
    /// # Safety
    /// The host allocation backing `values` must remain at the same address and
    /// unmodified until completion. The Rust borrow need not last that long, so
    /// the caller must enforce this separately. External device accesses must be
    /// ordered around the upload and any release caused by growth.
    pub(crate) unsafe fn upload(&mut self, values: &[T]) -> Result<(), Status>
    where
        T: Copy,
    {
        if values.is_empty() {
            return Ok(());
        }
        self.realloc(values.len())?;
        unsafe {
            self.ctx.upload(values, self.ptr.as_raw())?;
        }
        Ok(())
    }

    /// Enqueues a download of the first `out.len()` elements.
    ///
    /// # Safety
    /// The source bytes must represent valid `T` values. The host allocation
    /// backing `out` must remain at the same address and must not be read,
    /// modified, reallocated, or dropped until completion. External device writes
    /// must be ordered around the copy. This call does not wait for completion.
    pub(crate) unsafe fn download(&self, out: &mut [T]) -> Result<(), Status>
    where
        T: Copy,
    {
        unsafe { self.download_range(0, out) }
    }

    /// Enqueues a download of `out.len()` elements starting at element `offset`.
    /// Checks the requested extent against capacity, not initialization.
    ///
    /// # Safety
    /// The selected source bytes must represent valid `T` values. The host
    /// allocation backing `out` must remain at the same address and must not be
    /// read, modified, reallocated, or dropped until completion. External device
    /// writes must be ordered around the copy. This call does not wait.
    pub(crate) unsafe fn download_range(&self, offset: usize, out: &mut [T]) -> Result<(), Status>
    where
        T: Copy,
    {
        if out.is_empty() {
            return Ok(());
        }
        let end = offset
            .checked_add(out.len())
            .ok_or(Status::InvalidArgument)?;
        if end > self.cap {
            return Err(Status::InvalidArgument);
        }
        unsafe {
            self.ctx.download(self.ptr.as_raw().add(offset), out)?;
        }
        Ok(())
    }
}

impl<T: DeviceElement> DeviceBuffer<T> {
    pub(crate) fn zero(&mut self, len: usize) -> Result<(), Status> {
        if len > self.cap {
            return Err(Status::InvalidArgument);
        };
        unsafe { self.ctx.zero(self.ptr.as_raw(), len) }
    }
}

impl<T> Drop for DeviceBuffer<T> {
    fn drop(&mut self) {
        unsafe { _ = self.ctx.free(self.ptr.as_raw()) };
    }
}
impl<T> Deref for DeviceBuffer<T> {
    type Target = DeviceSpan<T>;
    fn deref(&self) -> &Self::Target {
        &self.span
    }
}
