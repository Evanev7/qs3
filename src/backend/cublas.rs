use crate::{Status, ffi::qscb};

pub(crate) use super::Bf16Gemm;

/// Thin typed access to qscb's cuBLAS implementation.
pub(crate) struct Cublas<'a> {
    context: &'a mut qscb::Context,
}

impl<'a> Cublas<'a> {
    pub(super) fn new(context: &'a mut qscb::Context) -> Self {
        Self { context }
    }

    pub(crate) unsafe fn gemm_bf16(&mut self, desc: &Bf16Gemm) -> Result<(), Status> {
        unsafe { self.context.gemm_bf16(&desc.raw) }
    }
}
