use super::ModelRunner;
use crate::{
    QWEN36_MOE_ROUTER_RENORMALIZE, QWEN36_MOE_ROUTER_SCALING_FACTOR, QWEN36_MOE_ROUTER_SCORE,
    backend::{
        BF16, DMat, DTensor3, F32, FloatDType,
        qsfi::{MoeBf16Execute, MoeBf16ExecuteArgs, Workspace},
    },
    engine::Status,
    ffi,
    model::{
        MoeRouterPrecision, QwenMoeConfig,
        weights::{QwenMlpPtrs, QwenSharedExpertPtrs},
    },
};

impl ModelRunner {
    pub(super) fn execute_post_attention_mlp(
        &mut self,
        rows: u32,
        hidden: u32,
        intermediate: u32,
        mlp_norm: ffi::DevicePtr,
        mlp: QwenMlpPtrs,
        next_weight: ffi::DevicePtr,
    ) -> Result<(), Status> {
        self.fused_add_rmsnorm(
            self.scratch.attn_proj.as_device_ptr(),
            self.scratch.residual.as_device_ptr(),
            mlp_norm,
            rows,
        )?;
        match mlp {
            QwenMlpPtrs::Dense {
                gate_proj,
                up_proj,
                down_proj,
            } => {
                self.execute_dense_mlp(rows, hidden, intermediate, gate_proj, up_proj, down_proj)?
            }
            QwenMlpPtrs::Moe {
                router_proj,
                gate_up_proj,
                down_proj,
                shared,
            } => {
                self.execute_moe_mlp(rows, hidden, router_proj, gate_up_proj, down_proj, shared)?
            }
        }
        self.fused_add_rmsnorm(
            self.scratch.mlp_out.as_device_ptr(),
            self.scratch.residual.as_device_ptr(),
            next_weight,
            rows,
        )
    }

    fn execute_dense_mlp(
        &mut self,
        rows: u32,
        hidden: u32,
        intermediate: u32,
        gate_proj: ffi::DevicePtr,
        up_proj: ffi::DevicePtr,
        down_proj: ffi::DevicePtr,
    ) -> Result<(), Status> {
        self.linear_bf16(
            self.scratch.attn_proj.as_device_ptr(),
            rows,
            hidden,
            gate_proj,
            self.scratch.gate.as_device_ptr(),
            intermediate,
        )?;
        self.linear_bf16(
            self.scratch.attn_proj.as_device_ptr(),
            rows,
            hidden,
            up_proj,
            self.scratch.up.as_device_ptr(),
            intermediate,
        )?;
        self.silu_and_mul(
            rows,
            intermediate,
            self.scratch.gate.as_device_ptr(),
            self.scratch.up.as_device_ptr(),
            self.scratch.mlp.as_device_ptr(),
        )?;
        self.linear_bf16(
            self.scratch.mlp.as_device_ptr(),
            rows,
            intermediate,
            down_proj,
            self.scratch.mlp_out.as_device_ptr(),
            hidden,
        )
    }

    fn route_moe_logits<T: FloatDType>(
        &mut self,
        logits: DMat<T>,
        moe: QwenMoeConfig,
    ) -> Result<(), Status> {
        let ids = DMat::contiguous(
            self.scratch.topk_ids.as_device_ptr(),
            logits.rows(),
            moe.num_experts_per_tok,
        )?;
        let weights = DMat::contiguous(
            self.scratch.topk_weights.as_device_ptr(),
            logits.rows(),
            moe.num_experts_per_tok,
        )?;
        let mut ops = self.engine.operators();
        unsafe {
            ops.qscu().router_topk(
                logits,
                ids,
                weights,
                QWEN36_MOE_ROUTER_SCORE,
                QWEN36_MOE_ROUTER_RENORMALIZE,
                QWEN36_MOE_ROUTER_SCALING_FACTOR,
            )
        }
    }

    pub(super) fn execute_moe_mlp(
        &mut self,
        rows: u32,
        hidden: u32,
        router_proj: ffi::DevicePtr,
        gate_up_proj: ffi::DevicePtr,
        down_proj: ffi::DevicePtr,
        shared: Option<QwenSharedExpertPtrs>,
    ) -> Result<(), Status> {
        let moe = self.config.moe_config().ok_or(Status::InternalError)?;
        match self.config.moe_router_precision {
            MoeRouterPrecision::F32 => {
                self.linear_f32(
                    self.scratch.attn_proj.as_device_ptr(),
                    rows,
                    hidden,
                    router_proj,
                    self.scratch.router_logits.as_device_ptr(),
                    moe.num_experts,
                )?;
                self.route_moe_logits(
                    DMat::<F32>::contiguous(
                        self.scratch.router_logits.as_device_ptr(),
                        rows,
                        moe.num_experts,
                    )?,
                    moe,
                )?;
            }
            MoeRouterPrecision::Bf16 => {
                self.linear_bf16(
                    self.scratch.attn_proj.as_device_ptr(),
                    rows,
                    hidden,
                    router_proj,
                    self.scratch.router_logits_bf16.as_device_ptr(),
                    moe.num_experts,
                )?;
                self.route_moe_logits(
                    DMat::<BF16>::contiguous(
                        self.scratch.router_logits_bf16.as_device_ptr(),
                        rows,
                        moe.num_experts,
                    )?,
                    moe,
                )?;
            }
        }

        let plan = self.moe_plan.as_ref().ok_or(Status::InternalError)?;
        let execute = MoeBf16Execute::new(MoeBf16ExecuteArgs {
            hidden: DMat::contiguous(self.scratch.attn_proj.as_device_ptr(), rows, hidden)?,
            topk_ids: DMat::contiguous(
                self.scratch.topk_ids.as_device_ptr(),
                rows,
                moe.num_experts_per_tok,
            )?,
            topk_weights: DMat::contiguous(
                self.scratch.topk_weights.as_device_ptr(),
                rows,
                moe.num_experts_per_tok,
            )?,
            gate_up_weight: DTensor3::contiguous(
                gate_up_proj,
                moe.num_experts,
                moe.moe_intermediate_size
                    .checked_mul(2)
                    .ok_or(Status::InvalidArgument)?,
                hidden,
            )?,
            down_weight: DTensor3::contiguous(
                down_proj,
                moe.num_experts,
                hidden,
                moe.moe_intermediate_size,
            )?,
            out: DMat::contiguous(self.scratch.mlp_out.as_device_ptr(), rows, hidden)?,
            workspace: Workspace::new(
                self.scratch.moe_workspace.as_device_ptr(),
                self.scratch.moe_workspace.cap,
            )?,
        })?;
        {
            let mut ops = self.engine.operators();
            unsafe { ops.qsfi().moe_execute_bf16(plan, &execute)? };
        }

        if let Some(shared) = shared {
            self.execute_shared_expert_mlp(
                rows,
                hidden,
                moe.shared_expert_intermediate_size,
                shared,
            )?;
        }
        Ok(())
    }

    fn execute_shared_expert_mlp(
        &mut self,
        rows: u32,
        hidden: u32,
        intermediate: u32,
        shared: QwenSharedExpertPtrs,
    ) -> Result<(), Status> {
        if intermediate == 0 {
            return Err(Status::InternalError);
        }
        self.linear_bf16(
            self.scratch.attn_proj.as_device_ptr(),
            rows,
            hidden,
            shared.gate_proj,
            self.scratch.shared_gate.as_device_ptr(),
            intermediate,
        )?;
        self.linear_bf16(
            self.scratch.attn_proj.as_device_ptr(),
            rows,
            hidden,
            shared.up_proj,
            self.scratch.shared_up.as_device_ptr(),
            intermediate,
        )?;
        self.silu_and_mul(
            rows,
            intermediate,
            self.scratch.shared_gate.as_device_ptr(),
            self.scratch.shared_up.as_device_ptr(),
            self.scratch.shared_mlp.as_device_ptr(),
        )?;
        self.linear_bf16(
            self.scratch.shared_mlp.as_device_ptr(),
            rows,
            intermediate,
            shared.down_proj,
            self.scratch.shared_out.as_device_ptr(),
            hidden,
        )?;
        self.linear_f32(
            self.scratch.attn_proj.as_device_ptr(),
            rows,
            hidden,
            shared.shared_expert_gate,
            self.scratch.shared_gate_logits.as_device_ptr(),
            1,
        )?;
        self.shared_expert_gate_add(rows, hidden)
    }
}
