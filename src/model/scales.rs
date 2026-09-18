use crate::{
    Status,
    backend::DVec,
    dtype::{DType, F32},
    ffi::DevicePtr,
    memory::DeviceSpan,
};
use std::mem::{offset_of, size_of};

#[repr(C)]
#[derive(Clone, Copy)]
pub struct Fp8Scales {
    pub input_scale: f32,
    pub weight_scale: f32,
}

#[repr(C)]
#[derive(Clone, Copy)]
pub struct Nvfp4Scales {
    pub quant_multiplier: f32,
    pub alpha: f32,
}

// Structured storage has no native tensor element dtype. Only its named F32
// fields are passed to kernels; treating the entire record as a tensor fails.
unsafe impl DType for Fp8Scales {}
unsafe impl DType for Nvfp4Scales {}

const _: () = assert!(size_of::<Fp8Scales>() == 8 && size_of::<Nvfp4Scales>() == 8);

fn field<T: DType>(record: &DeviceSpan<T>, offset: usize) -> Result<DVec<F32>, Status> {
    record.check_view_len(1)?;
    // Called only with offsets of the F32 fields in the two records above.
    DVec::contiguous(
        DevicePtr::new(unsafe { record.as_raw().add(offset) }).ok_or(Status::InvalidArgument)?,
        1,
    )
}
impl DeviceSpan<Fp8Scales> {
    pub(crate) fn input_scale(&self) -> Result<DVec<F32>, Status> {
        field(self, offset_of!(Fp8Scales, input_scale))
    }
    pub(crate) fn weight_scale(&self) -> Result<DVec<F32>, Status> {
        field(self, offset_of!(Fp8Scales, weight_scale))
    }
}
impl DeviceSpan<Nvfp4Scales> {
    pub(crate) fn quant_multiplier(&self) -> Result<DVec<F32>, Status> {
        field(self, offset_of!(Nvfp4Scales, quant_multiplier))
    }
    pub(crate) fn alpha(&self) -> Result<DVec<F32>, Status> {
        field(self, offset_of!(Nvfp4Scales, alpha))
    }
}
