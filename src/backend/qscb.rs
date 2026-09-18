use super::{DMat, DVec, Workspace, result_from_raw};
use crate::{
    Status,
    dtype::{BF16, DType, F32, Fp8E4M3},
    ffi::{self, sys},
    memory::CudaCtx,
};
use std::{
    collections::{HashMap, hash_map::Entry},
    mem::MaybeUninit,
    ptr::{self, NonNull},
};

/// Typed access to qscb's cuBLAS implementation.
pub(crate) struct Qscb {
    raw: NonNull<sys::qscb_context>,
    plans: HashMap<LinearKey, LinearPlan>,
}

impl Qscb {
    pub(crate) unsafe fn requantize_fp8(
        &mut self,
        weight: DMat<Fp8E4M3>,
        source_scale: f32,
        shared_scale: f32,
    ) -> Result<(), Status> {
        if [source_scale, shared_scale]
            .iter()
            .any(|s| !s.is_finite() || *s <= 0.0)
        {
            return Err(Status::InvalidArgument);
        }
        result_from_raw(unsafe {
            sys::qscb_fp8_requantize(
                self.raw.as_ptr(),
                &weight.tensor(),
                source_scale,
                shared_scale,
            )
        })
    }

    pub(crate) unsafe fn quantize_fp8(
        &mut self,
        input: DMat<BF16>,
        output: DMat<Fp8E4M3>,
        scale: DVec<F32>,
    ) -> Result<(), Status> {
        if !input.same_shape(output) || scale.len != 1 {
            return Err(Status::InvalidArgument);
        }
        scale.require_contiguous()?;
        let desc = sys::qscb_fp8_quantize_desc {
            x: input.tensor(),
            out: output.tensor(),
            scale: scale.tensor(),
        };
        result_from_raw(unsafe { sys::qscb_fp8_quantize(self.raw.as_ptr(), &desc) })
    }

    pub(crate) unsafe fn linear_fp8<Output: LinearOutput>(
        &mut self,
        input: DMat<Fp8E4M3>,
        weight: DMat<Fp8E4M3>,
        output: DMat<Output>,
        scales: [DVec<F32>; 2],
        workspace: Workspace,
    ) -> Result<(), Status> {
        for scale in scales {
            scale.require_contiguous()?;
            if scale.len != 1 {
                return Err(Status::InvalidArgument);
            }
        }
        let desc = sys::qscb_fp8_linear_desc {
            linear: linear_desc(input, weight, output, workspace)?,
            x_scale: scales[0].tensor(),
            weight_scale: scales[1].tensor(),
        };
        let key = LinearKey::new(&desc.linear);
        let plan = match self.plans.entry(key) {
            Entry::Occupied(entry) => entry.into_mut(),
            Entry::Vacant(entry) => {
                let mut raw = ptr::null_mut();
                result_from_raw(unsafe {
                    sys::qscb_fp8_linear_plan_create(self.raw.as_ptr(), &desc, &mut raw)
                })?;
                entry.insert(LinearPlan(NonNull::new(raw).ok_or(Status::InternalError)?))
            }
        };
        result_from_raw(unsafe {
            sys::qscb_fp8_linear_execute(self.raw.as_ptr(), plan.0.as_ptr(), &desc)
        })
    }

    pub(crate) fn new(ctx: &CudaCtx) -> Result<Self, Status> {
        let desc = sys::qscb_context_desc {
            device_ordinal: ctx.device_ordinal(),
            stream: ctx.stream,
        };
        let mut raw = ptr::null_mut();
        result_from_raw(unsafe { sys::qscb_context_create(&desc, &mut raw) })?;
        Ok(Self {
            raw: NonNull::new(raw).ok_or(Status::InternalError)?,
            plans: HashMap::new(),
        })
    }

    pub(crate) unsafe fn linear<Output: LinearOutput>(
        &mut self,
        input: DMat<BF16>,
        weight: DMat<BF16>,
        output: DMat<Output>,
        workspace: Workspace,
    ) -> Result<(), Status> {
        let desc = linear_desc(input, weight, output, workspace)?;
        let key = LinearKey::new(&desc);
        let plan = match self.plans.entry(key) {
            Entry::Occupied(entry) => entry.into_mut(),
            Entry::Vacant(entry) => {
                let mut raw = ptr::null_mut();
                result_from_raw(unsafe {
                    sys::qscb_linear_plan_create(self.raw.as_ptr(), &desc, &mut raw)
                })?;
                entry.insert(LinearPlan(NonNull::new(raw).ok_or(Status::InternalError)?))
            }
        };
        result_from_raw(unsafe {
            sys::qscb_linear_execute(self.raw.as_ptr(), plan.0.as_ptr(), &desc)
        })
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
        self.plans.clear();
        unsafe { sys::qscb_context_destroy(self.raw.as_ptr()) }
    }
}

pub(crate) trait LinearOutput: DType {}
impl LinearOutput for BF16 {}
impl LinearOutput for F32 {}

struct LinearPlan(NonNull<sys::qscb_linear_plan>);

impl Drop for LinearPlan {
    fn drop(&mut self) {
        unsafe { sys::qscb_linear_plan_destroy(self.0.as_ptr()) }
    }
}

// Context/device ownership is implicit in Qscb. Scalars and tensor addresses
// may change; algorithms depend on their alignment, not their identity.
#[derive(Debug, PartialEq, Eq, Hash)]
struct LinearKey {
    dimensions: [u32; 3],
    strides: [i64; 3],
    output_dtype: ffi::DTypeRaw,
    input_dtype: ffi::DTypeRaw,
    weight_dtype: ffi::DTypeRaw,
    alignments: [u32; 3],
    workspace_bytes: usize,
}

impl LinearKey {
    fn new(desc: &sys::qscb_linear_desc) -> Self {
        Self {
            dimensions: [desc.rows, desc.in_features, desc.out_features],
            strides: [desc.x.stride[0], desc.weight.stride[0], desc.out.stride[0]],
            output_dtype: desc.out.dtype,
            input_dtype: desc.x.dtype,
            weight_dtype: desc.weight.dtype,
            alignments: [desc.x.data, desc.weight.data, desc.out.data]
                .map(|ptr| 1 << (ptr as usize).trailing_zeros().min(8)),
            workspace_bytes: desc.workspace_bytes,
        }
    }
}

pub(super) fn linear_desc<Input: DType, Output: LinearOutput>(
    input: DMat<Input>,
    weight: DMat<Input>,
    output: DMat<Output>,
    workspace: Workspace,
) -> Result<sys::qscb_linear_desc, Status> {
    workspace.validate()?;
    if !(workspace.data as usize).is_multiple_of(256)
        || weight.cols != input.cols
        || output.rows != input.rows
        || output.cols != weight.rows
    {
        return Err(Status::InvalidArgument);
    }
    Ok(sys::qscb_linear_desc {
        x: input.tensor(),
        weight: weight.tensor(),
        out: output.tensor(),
        rows: input.rows,
        in_features: input.cols,
        out_features: weight.rows,
        alpha: 0.0,
        beta: 0.0,
        workspace: workspace.data,
        workspace_bytes: workspace.bytes,
    })
}
