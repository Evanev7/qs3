use super::{BF16, DMat, F32, Workspace, result_from_raw};
use crate::{
    Status,
    ffi::{self, sys},
};
use std::{
    mem::MaybeUninit,
    ptr::{self, NonNull},
};

/// Typed access to qscb's cuBLAS implementation.
pub(crate) struct Qscb {
    raw: NonNull<sys::qscb_context>,
}

impl Qscb {
    pub(crate) fn new(device_ordinal: i32, stream: ffi::CudaStream) -> Result<Self, Status> {
        let desc = sys::qscb_context_desc {
            device_ordinal,
            stream,
        };
        let mut raw = ptr::null_mut();
        result_from_raw(unsafe { sys::qscb_context_create(&desc, &mut raw) })?;
        Ok(Self {
            raw: NonNull::new(raw).ok_or(Status::InternalError)?,
        })
    }

    pub(crate) unsafe fn linear_bf16(
        &mut self,
        input: DMat<BF16>,
        weight: DMat<BF16>,
        output: DMat<BF16>,
        workspace: Workspace,
    ) -> Result<(), Status> {
        let desc = linear_bf16_desc(input, weight, output, workspace)?;
        unsafe { self.execute_linear(&desc) }
    }

    pub(crate) unsafe fn linear_f32(
        &mut self,
        input: DMat<BF16>,
        weight: DMat<BF16>,
        output: DMat<F32>,
        workspace: Workspace,
    ) -> Result<(), Status> {
        let desc = linear_f32_desc(input, weight, output, workspace)?;
        unsafe { self.execute_linear(&desc) }
    }

    unsafe fn execute_linear(&mut self, desc: &sys::qscb_linear_desc) -> Result<(), Status> {
        result_from_raw(unsafe { sys::qscb_linear(self.raw.as_ptr(), desc) })
            .inspect_err(|_| _ = self.last_error())
    }

    fn last_error(&self) -> Result<ffi::ErrorInfo, Status> {
        let mut out = MaybeUninit::uninit();
        result_from_raw(unsafe {
            sys::qscb_context_get_last_error(self.raw.as_ptr(), out.as_mut_ptr())
        })?;
        Ok(unsafe { out.assume_init() })
    }
}

impl Drop for Qscb {
    fn drop(&mut self) {
        unsafe { sys::qscb_context_destroy(self.raw.as_ptr()) }
    }
}

pub(super) fn linear_bf16_desc(
    input: DMat<BF16>,
    weight: DMat<BF16>,
    output: DMat<BF16>,
    workspace: Workspace,
) -> Result<sys::qscb_linear_desc, Status> {
    linear_desc(
        input,
        weight,
        output.tensor(),
        output.rows,
        output.cols,
        workspace,
    )
}

pub(super) fn linear_f32_desc(
    input: DMat<BF16>,
    weight: DMat<BF16>,
    output: DMat<F32>,
    workspace: Workspace,
) -> Result<sys::qscb_linear_desc, Status> {
    linear_desc(
        input,
        weight,
        output.tensor(),
        output.rows,
        output.cols,
        workspace,
    )
}

fn linear_desc(
    input: DMat<BF16>,
    weight: DMat<BF16>,
    output: ffi::Tensor2,
    output_rows: u32,
    output_cols: u32,
    workspace: Workspace,
) -> Result<sys::qscb_linear_desc, Status> {
    workspace.validate()?;
    if weight.cols != input.cols || output_rows != input.rows || output_cols != weight.rows {
        return Err(Status::InvalidArgument);
    }
    Ok(sys::qscb_linear_desc {
        x: input.tensor(),
        weight: weight.tensor(),
        out: output,
        rows: input.rows,
        in_features: input.cols,
        out_features: weight.rows,
        alpha: 0.0,
        beta: 0.0,
        workspace: workspace.data,
        workspace_bytes: workspace.bytes,
    })
}
