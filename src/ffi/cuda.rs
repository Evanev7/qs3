use std::ffi::c_void;

pub(crate) const CUDA_SUCCESS: i32 = 0;
pub(crate) const CUDA_ERROR_MEMORY_ALLOCATION: i32 = 2;
pub(crate) const CUDA_MEMCPY_HOST_TO_DEVICE: i32 = 1;
pub(crate) const CUDA_MEMCPY_DEVICE_TO_HOST: i32 = 2;
pub(crate) const CUDA_MEMCPY_DEVICE_TO_DEVICE: i32 = 3;

unsafe extern "C" {
    pub(crate) fn cudaGetDevice(device: *mut i32) -> i32;
    pub(crate) fn cudaSetDevice(device: i32) -> i32;
    pub(crate) fn cudaMalloc(dev_ptr: *mut *mut c_void, size: usize) -> i32;
    pub(crate) fn cudaFree(dev_ptr: *mut c_void) -> i32;
    pub(crate) fn cudaMemcpyAsync(
        dst: *mut c_void,
        src: *const c_void,
        count: usize,
        kind: i32,
        stream: *mut c_void,
    ) -> i32;
    pub(crate) fn cudaMemcpy2DAsync(
        dst: *mut c_void,
        dpitch: usize,
        src: *const c_void,
        spitch: usize,
        width: usize,
        height: usize,
        kind: i32,
        stream: *mut c_void,
    ) -> i32;
    pub(crate) fn cudaMemsetAsync(
        dev_ptr: *mut c_void,
        value: i32,
        count: usize,
        stream: *mut c_void,
    ) -> i32;
    pub(crate) fn cudaStreamSynchronize(stream: *mut c_void) -> i32;
}
