//! CuTe FP8 decode GEMM producing two FP32 split-K partials.
use crate::{
    Status,
    backend::{DMat, DVec},
    dtype::{F32, Fp8E4M3},
    ffi,
};

mod fp8_decode {
    include!(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/build/cute/fp8_decode.rs"
    ));
}
pub(crate) struct Fp8Decode(fp8_decode::Kernel);

impl Fp8Decode {
    pub(crate) const N: u32 = fp8_decode::constants::N as u32;
    pub(crate) const K: u32 = fp8_decode::constants::K as u32;

    pub(crate) fn supports(k: u32, n: u32) -> bool {
        k == Self::K && n == Self::N
    }

    /// The current CUDA context must survive this owner and its queued work.
    /// Loading a second live instance of the CuTe specialization panics.
    pub(crate) unsafe fn load() -> Result<Self, Status> {
        unsafe { fp8_decode::Kernel::load() }
            .map(Self)
            .map_err(|_| Status::CudaError)
    }

    /// Bindings must be live, nonaliasing and ordered on the supplied stream.
    pub(crate) unsafe fn launch(
        &self,
        stream: ffi::CudaStream,
        input: DMat<Fp8E4M3>,
        weight: DMat<Fp8E4M3>,
        scales: [DVec<F32>; 2],
        partials: DMat<F32>,
    ) -> Result<(), Status> {
        if input.shape() != [1, Self::K]
            || weight.shape() != [Self::N, Self::K]
            || partials.shape() != [2, Self::N]
            || scales.iter().any(|s| s.len != 1 || !s.is_contiguous())
        {
            return Err(Status::InvalidArgument);
        }
        input.require_contiguous()?;
        weight.require_contiguous()?;
        partials.require_contiguous()?;
        for address in [
            input.data.erase(),
            weight.data.erase(),
            partials.data.erase(),
        ] {
            if !(address as usize).is_multiple_of(16) {
                return Err(Status::InvalidArgument);
            }
        }
        if scales
            .iter()
            .any(|s| !(s.data.erase() as usize).is_multiple_of(4))
        {
            return Err(Status::InvalidArgument);
        }
        unsafe {
            self.0
                .launch(
                    input.data,
                    weight.data,
                    partials.data,
                    scales[0].data,
                    scales[1].data,
                    stream,
                )
                .map_err(|_| Status::CudaError)
        }
    }
}

#[cfg(test)]
mod tests;

// The backend and full-runner tests exercise the same single-owner AOT export.
#[cfg(test)]
pub(crate) static TEST_LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());
