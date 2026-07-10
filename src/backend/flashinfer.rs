use super::*;

pub(crate) use super::{MoePlan, Workspace};

/// Thin typed access to qsfi's FlashInfer-owned operations.
pub(crate) struct FlashInfer<'a> {
    context: &'a mut qsfi::Context,
}

impl<'a> FlashInfer<'a> {
    pub(super) fn new(context: &'a mut qsfi::Context) -> Self {
        Self { context }
    }

    pub(crate) unsafe fn rmsnorm_bf16(&mut self, desc: &RmsNormBf16) -> Result<(), Status> {
        unsafe { self.context.rmsnorm(&desc.raw) }
    }

    pub(crate) unsafe fn fused_add_rmsnorm_bf16(
        &mut self,
        desc: &FusedAddRmsNormBf16,
    ) -> Result<(), Status> {
        unsafe { self.context.fused_add_rmsnorm(&desc.raw) }
    }

    pub(crate) unsafe fn rope_apply_bf16(&mut self, desc: &RopeApplyBf16) -> Result<(), Status> {
        unsafe { self.context.rope_apply(&desc.raw) }
    }

    pub(crate) unsafe fn create_moe_bf16_plan(
        &mut self,
        config: MoeBf16PlanConfig,
    ) -> Result<MoePlan, Status> {
        let desc = config.desc()?;
        unsafe { self.context.create_moe_plan(&desc) }
    }

    pub(crate) unsafe fn moe_workspace_size(
        &mut self,
        plan: &MoePlan,
        num_tokens: u32,
    ) -> Result<usize, Status> {
        unsafe { self.context.moe_workspace_size(plan, num_tokens) }
    }

    pub(crate) unsafe fn moe_execute_bf16(
        &mut self,
        plan: &MoePlan,
        desc: &MoeBf16Execute,
    ) -> Result<(), Status> {
        unsafe { self.context.moe_execute_bf16(plan, &desc.raw) }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) struct MoeBf16PlanConfig {
    pub(crate) max_num_tokens: u32,
    pub(crate) hidden_size: u32,
    pub(crate) intermediate_size: u32,
    pub(crate) num_experts: u32,
    pub(crate) top_k: u32,
}

impl MoeBf16PlanConfig {
    fn desc(self) -> Result<ffi::MoePlanDesc, Status> {
        validate_nonzero(&[
            self.max_num_tokens,
            self.hidden_size,
            self.intermediate_size,
            self.num_experts,
            self.top_k,
        ])?;
        if self.top_k > self.num_experts {
            return Err(Status::InvalidArgument);
        }
        Ok(ffi::MoePlanDesc {
            backend: ffi::MOE_BACKEND_FLASHINFER_STAGED_BF16,
            route_mode: ffi::MOE_ROUTE_PRECOMPUTED_TOPK,
            max_num_tokens: self.max_num_tokens,
            hidden_size: self.hidden_size,
            intermediate_size: self.intermediate_size,
            num_experts: self.num_experts,
            top_k: self.top_k,
            local_expert_offset: 0,
            local_num_experts: self.num_experts,
            activation_dtype: ffi::DTYPE_BF16,
            weight_dtype: ffi::DTYPE_BF16,
            output_dtype: ffi::DTYPE_BF16,
            reserved0: 0,
        })
    }
}

#[derive(Clone, Copy, Debug)]
pub(crate) struct MoeBf16ExecuteArgs {
    pub(crate) hidden: DMat<BF16>,
    pub(crate) topk_ids: DMat<I32>,
    pub(crate) topk_weights: DMat<F32>,
    pub(crate) gate_up_weight: DTensor3<BF16>,
    pub(crate) down_weight: DTensor3<BF16>,
    pub(crate) out: DMat<BF16>,
    pub(crate) workspace: Workspace,
}

#[derive(Clone, Copy, Debug)]
pub(crate) struct MoeBf16Execute {
    pub(super) raw: ffi::MoeBf16ExecuteDesc,
}

impl MoeBf16Execute {
    pub(crate) fn new(args: MoeBf16ExecuteArgs) -> Result<Self, Status> {
        args.hidden.require_contiguous()?;
        args.topk_ids.require_contiguous()?;
        args.topk_weights.require_contiguous()?;
        args.gate_up_weight.require_contiguous()?;
        args.down_weight.require_contiguous()?;
        args.out.require_contiguous()?;
        if args.hidden.rows == 0
            || args.topk_ids.rows != args.hidden.rows
            || args.topk_weights.rows != args.hidden.rows
            || args.topk_ids.cols != args.topk_weights.cols
            || args.topk_ids.cols == 0
            || !args.hidden.same_shape(args.out)
        {
            return Err(Status::InvalidArgument);
        }
        Ok(Self {
            raw: ffi::MoeBf16ExecuteDesc {
                hidden: args.hidden.tensor(),
                topk_ids: args.topk_ids.tensor(),
                topk_weights: args.topk_weights.tensor(),
                gate_up_weight: args.gate_up_weight.tensor(),
                down_weight: args.down_weight.tensor(),
                out: args.out.tensor(),
                workspace: args.workspace.tensor(ffi::DTYPE_U8)?,
                num_tokens: args.hidden.rows,
            },
        })
    }
}

#[derive(Clone, Copy, Debug)]
pub(crate) struct RmsNormBf16 {
    pub(super) raw: ffi::RmsnormDesc,
}

impl RmsNormBf16 {
    pub(crate) fn new(
        x: DMat<BF16>,
        weight: DVec<BF16>,
        out: DMat<BF16>,
        eps: f32,
    ) -> Result<Self, Status> {
        Self::qwen_decoder_norm(x, weight, out, eps)
    }

    pub(crate) fn qwen_qk_norm(
        x: DMat<BF16>,
        weight: DVec<BF16>,
        out: DMat<BF16>,
        eps: f32,
    ) -> Result<Self, Status> {
        Self::qwen_decoder_norm(x, weight, out, eps)
    }

    pub(crate) fn qwen_decoder_norm(
        x: DMat<BF16>,
        weight: DVec<BF16>,
        out: DMat<BF16>,
        eps: f32,
    ) -> Result<Self, Status> {
        validate_eps(eps)?;
        weight.require_contiguous()?;
        if !x.same_shape(out) || weight.len != x.cols {
            return Err(Status::InvalidArgument);
        }
        Ok(Self {
            raw: ffi::RmsnormDesc {
                x: x.tensor(),
                weight: weight.tensor(),
                out: out.tensor(),
                hidden_size: x.cols,
                eps,
            },
        })
    }
}

#[derive(Clone, Copy, Debug)]
pub(crate) struct FusedAddRmsNormBf16 {
    pub(super) raw: ffi::FusedAddRmsnormDesc,
}

impl FusedAddRmsNormBf16 {
    pub(crate) fn new(
        x: DMat<BF16>,
        residual_inout: DMat<BF16>,
        weight: DVec<BF16>,
        eps: f32,
    ) -> Result<Self, Status> {
        Self::qwen_decoder_norm(x, residual_inout, weight, eps)
    }

    pub(crate) fn qwen_decoder_norm(
        x: DMat<BF16>,
        residual_inout: DMat<BF16>,
        weight: DVec<BF16>,
        eps: f32,
    ) -> Result<Self, Status> {
        validate_eps(eps)?;
        weight.require_contiguous()?;
        if !x.same_shape(residual_inout) || weight.len != x.cols {
            return Err(Status::InvalidArgument);
        }
        Ok(Self {
            raw: ffi::FusedAddRmsnormDesc {
                x: x.tensor(),
                residual_inout: residual_inout.tensor(),
                weight: weight.tensor(),
                out: x.tensor(),
                hidden_size: x.cols,
                eps,
            },
        })
    }
}

#[derive(Clone, Copy, Debug)]
pub(crate) struct RopeApplyBf16 {
    pub(super) raw: ffi::RopeApplyDesc,
}

impl RopeApplyBf16 {
    pub(crate) fn new(
        q: Bf16Heads,
        k: Bf16Heads,
        q_out: Bf16Heads,
        k_out: Bf16Heads,
        positions: DVec<I32>,
        rotary_dim: u32,
    ) -> Result<Self, Status> {
        Self::with_params(q, k, q_out, k_out, positions, rotary_dim, 0.0, 0.0)
    }

    pub(crate) fn with_params(
        q: Bf16Heads,
        k: Bf16Heads,
        q_out: Bf16Heads,
        k_out: Bf16Heads,
        positions: DVec<I32>,
        rotary_dim: u32,
        rope_scale: f32,
        rope_theta: f32,
    ) -> Result<Self, Status> {
        require_supported_rope_dims(q.head_dim, rotary_dim)?;
        if q.head_dim != k.head_dim
            || !q.same_shape(q_out)
            || !k.same_shape(k_out)
            || k.tokens != q.tokens
            || positions.len != q.tokens
        {
            return Err(Status::InvalidArgument);
        }
        if q_out.data == q.data && !q.same_strides(q_out) {
            return Err(Status::InvalidArgument);
        }
        if k_out.data == k.data && !k.same_strides(k_out) {
            return Err(Status::InvalidArgument);
        }
        positions.require_contiguous()?;
        if !rope_scale.is_finite()
            || rope_scale < 0.0
            || !rope_theta.is_finite()
            || rope_theta < 0.0
        {
            return Err(Status::InvalidArgument);
        }
        Ok(Self {
            raw: ffi::RopeApplyDesc {
                q: q.tensor(),
                k: k.tensor(),
                q_out: q_out.tensor(),
                k_out: k_out.tensor(),
                positions: positions.tensor(),
                num_qo_heads: q.heads,
                num_kv_heads: k.heads,
                head_dim: q.head_dim,
                rotary_dim,
                rope_scale,
                rope_theta,
                interleave: 0,
            },
        })
    }
}
