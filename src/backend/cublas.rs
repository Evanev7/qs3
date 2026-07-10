use super::*;

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

#[derive(Clone, Copy, Debug)]
pub(crate) struct Bf16Gemm {
    pub(super) raw: qscb::Bf16GemmDesc,
}

impl Bf16Gemm {
    pub(crate) fn new(
        x: DMat<BF16>,
        weight: DMat<BF16>,
        out: Bf16OrF32Mat,
        workspace: Workspace,
    ) -> Result<Self, Status> {
        Self::with_alpha_beta(x, weight, out, workspace, 0.0, 0.0)
    }

    pub(crate) fn with_alpha_beta(
        x: DMat<BF16>,
        weight: DMat<BF16>,
        out: Bf16OrF32Mat,
        workspace: Workspace,
        alpha: f32,
        beta: f32,
    ) -> Result<Self, Status> {
        workspace.validate()?;
        if !alpha.is_finite() || !beta.is_finite() {
            return Err(Status::InvalidArgument);
        }
        if weight.cols != x.cols || out.rows() != x.rows || out.cols() != weight.rows {
            return Err(Status::InvalidArgument);
        }
        Ok(Self {
            raw: qscb::Bf16GemmDesc {
                x: x.tensor(),
                weight: weight.tensor(),
                out: out.tensor(),
                rows: x.rows,
                in_features: x.cols,
                out_features: weight.rows,
                alpha,
                beta,
                workspace: workspace.data,
                workspace_bytes: workspace.bytes,
            },
        })
    }
}
