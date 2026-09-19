use super::Qsfi;
use crate::{
    Status,
    backend::{DMat, DVec, Workspace, result_from_raw},
    constants::nvfp4,
    dtype::{BF16, F32, Fp8E4M3, Nvfp4E2M1},
    ffi::sys,
};
use std::ptr::{self, NonNull};

/// Explicit AOT tactic. The build's configured subset must include this choice.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum Nvfp4Tactic {
    Tile128x32Dp,
    Tile128x32StreamK,
    Tile128x64Dp,
    Tile128x64StreamK,
    Qutlass256x128,
}
impl Nvfp4Tactic {
    const fn from_name(name: &str) -> Self {
        match name.as_bytes() {
            b"tile128x32_dp" => Self::Tile128x32Dp,
            b"tile128x32_stream_k" => Self::Tile128x32StreamK,
            b"tile128x64_dp" => Self::Tile128x64Dp,
            b"tile128x64_stream_k" => Self::Tile128x64StreamK,
            b"qutlass256x128" => Self::Qutlass256x128,
            _ => panic!("unknown NVFP4 AOT tactic"),
        }
    }

    pub(crate) const fn for_rows(rows: u32) -> Self {
        const SMALL: Nvfp4Tactic = Nvfp4Tactic::from_name(nvfp4::SMALL_TACTIC);
        const MEDIUM: Nvfp4Tactic = Nvfp4Tactic::from_name(nvfp4::MEDIUM_TACTIC);
        const PREFILL: Nvfp4Tactic = Nvfp4Tactic::from_name(nvfp4::PREFILL_TACTIC);
        if rows >= nvfp4::PREFILL_ROWS {
            PREFILL
        } else if rows >= nvfp4::MEDIUM_ROWS {
            MEDIUM
        } else {
            SMALL
        }
    }

    pub(crate) const fn name(self) -> &'static str {
        match self {
            Self::Tile128x32Dp => "tile128x32_dp",
            Self::Tile128x32StreamK => "tile128x32_stream_k",
            Self::Tile128x64Dp => "tile128x64_dp",
            Self::Tile128x64StreamK => "tile128x64_stream_k",
            Self::Qutlass256x128 => "qutlass256x128",
        }
    }

    fn raw(self) -> sys::qsfi_nvfp4_tactic {
        match self {
            Self::Tile128x32Dp => sys::QSFI_NVFP4_TILE128X32_DP,
            Self::Tile128x32StreamK => sys::QSFI_NVFP4_TILE128X32_STREAM_K,
            Self::Tile128x64Dp => sys::QSFI_NVFP4_TILE128X64_DP,
            Self::Tile128x64StreamK => sys::QSFI_NVFP4_TILE128X64_STREAM_K,
            Self::Qutlass256x128 => sys::QSFI_NVFP4_QUTLASS256X128,
        }
    }
}

pub(crate) fn scale_count(rows: u32, k: u32) -> Result<u32, Status> {
    if rows == 0 || k == 0 || !k.is_multiple_of(128) {
        return Err(Status::InvalidArgument);
    }
    rows.div_ceil(128)
        .checked_mul(128)
        .and_then(|v| v.checked_mul(k / 16))
        .ok_or(Status::InvalidArgument)
}

pub(crate) struct Nvfp4Plan {
    raw: NonNull<sys::qsfi_nvfp4_plan>,
    shape: [u32; 3], // M, N, K
    pub(crate) workspace_bytes: usize,
}
impl Drop for Nvfp4Plan {
    fn drop(&mut self) {
        unsafe { sys::qsfi_nvfp4_plan_destroy(self.raw.as_ptr()) }
    }
}

impl Qsfi {
    pub(crate) fn create_nvfp4_plan(
        &mut self,
        shape: [u32; 3],
        tactic: Nvfp4Tactic,
    ) -> Result<Nvfp4Plan, Status> {
        let [m, n, k] = shape;
        scale_count(m, k)?;
        scale_count(n, k)?;
        if !n.is_multiple_of(8) {
            return Err(Status::InvalidArgument);
        }
        let desc = sys::qsfi_nvfp4_plan_desc {
            rows: m,
            out_features: n,
            in_features: k,
            tactic: tactic.raw(),
        };
        let mut raw = ptr::null_mut();
        let mut workspace_bytes = 0;
        result_from_raw(unsafe {
            sys::qsfi_nvfp4_plan_create(self.raw.as_ptr(), &desc, &mut raw, &mut workspace_bytes)
        })?;
        Ok(Nvfp4Plan {
            raw: NonNull::new(raw).ok_or(Status::InternalError)?,
            shape,
            workspace_bytes,
        })
    }

    pub(crate) unsafe fn nvfp4_swizzle_scales(
        &mut self,
        input: DMat<Fp8E4M3>,
        output: DVec<Fp8E4M3>,
    ) -> Result<(), Status> {
        input.require_contiguous()?;
        output.require_contiguous()?;
        let k = input.cols.checked_mul(16).ok_or(Status::InvalidArgument)?;
        if output.len != scale_count(input.rows, k)? {
            return Err(Status::InvalidArgument);
        }
        result_from_raw(unsafe {
            sys::qsfi_nvfp4_swizzle_scales(self.raw.as_ptr(), &input.tensor(), &output.tensor())
        })
    }

    pub(crate) unsafe fn nvfp4_quantize(
        &mut self,
        input: DMat<BF16>,
        output: DMat<Nvfp4E2M1>,
        scales: DVec<Fp8E4M3>,
        multiplier: DVec<F32>,
    ) -> Result<(), Status> {
        input.require_contiguous()?;
        output.require_contiguous()?;
        scales.require_contiguous()?;
        multiplier.require_contiguous()?;
        if !input.same_shape(output)
            || scales.len != scale_count(input.rows, input.cols)?
            || multiplier.len != 1
        {
            return Err(Status::InvalidArgument);
        }
        let desc = sys::qsfi_nvfp4_quantize_desc {
            x: input.tensor(),
            out: output.tensor(),
            scales: scales.tensor(),
            quant_multiplier: multiplier.tensor(),
        };
        result_from_raw(unsafe { sys::qsfi_nvfp4_quantize(self.raw.as_ptr(), &desc) })
    }

    pub(crate) unsafe fn nvfp4_execute(
        &mut self,
        plan: &Nvfp4Plan,
        input: DMat<Nvfp4E2M1>,
        weight: DMat<Nvfp4E2M1>,
        scales: [DVec<Fp8E4M3>; 2],
        alpha: DVec<F32>,
        output: DMat<BF16>,
        workspace: Workspace,
    ) -> Result<(), Status> {
        let [m, n, k] = plan.shape;
        input.require_contiguous()?;
        weight.require_contiguous()?;
        output.require_contiguous()?;
        for scale in scales {
            scale.require_contiguous()?;
        }
        alpha.require_contiguous()?;
        workspace.validate()?;
        if [
            input.rows,
            input.cols,
            weight.rows,
            weight.cols,
            output.rows,
            output.cols,
        ] != [m, k, n, k, m, n]
            || scales[0].len != scale_count(m, k)?
            || scales[1].len != scale_count(n, k)?
            || alpha.len != 1
            || workspace.bytes < plan.workspace_bytes
            || !(workspace.data as usize).is_multiple_of(256)
        {
            return Err(Status::InvalidArgument);
        }
        let desc = sys::qsfi_nvfp4_execute_desc {
            x: input.tensor(),
            weight: weight.tensor(),
            x_scales: scales[0].tensor(),
            weight_scales: scales[1].tensor(),
            alpha: alpha.tensor(),
            out: output.tensor(),
            workspace: workspace.data,
            workspace_bytes: workspace.bytes,
        };
        result_from_raw(unsafe {
            sys::qsfi_nvfp4_execute(self.raw.as_ptr(), plan.raw.as_ptr(), &desc)
        })
    }
}
