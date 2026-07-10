use crate::model::*;

impl ModelRunner {
    pub(in crate::model) fn execute_post_attention_mlp(
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

    pub(in crate::model) fn execute_dense_mlp(
        &mut self,
        rows: u32,
        hidden: u32,
        intermediate: u32,
        gate_proj: ffi::DevicePtr,
        up_proj: ffi::DevicePtr,
        down_proj: ffi::DevicePtr,
    ) -> Result<(), Status> {
        self.gemm_bf16(
            self.scratch.attn_proj.as_device_ptr(),
            rows,
            hidden,
            gate_proj,
            self.scratch.gate.as_device_ptr(),
            intermediate,
            GemmOut::Bf16,
        )?;
        self.gemm_bf16(
            self.scratch.attn_proj.as_device_ptr(),
            rows,
            hidden,
            up_proj,
            self.scratch.up.as_device_ptr(),
            intermediate,
            GemmOut::Bf16,
        )?;
        self.silu_and_mul(
            rows,
            intermediate,
            self.scratch.gate.as_device_ptr(),
            self.scratch.up.as_device_ptr(),
            self.scratch.mlp.as_device_ptr(),
        )?;
        self.gemm_bf16(
            self.scratch.mlp.as_device_ptr(),
            rows,
            intermediate,
            down_proj,
            self.scratch.mlp_out.as_device_ptr(),
            hidden,
            GemmOut::Bf16,
        )
    }

    pub(in crate::model) fn execute_moe_mlp(
        &mut self,
        rows: u32,
        hidden: u32,
        router_proj: ffi::DevicePtr,
        gate_up_proj: ffi::DevicePtr,
        down_proj: ffi::DevicePtr,
        shared: Option<QwenSharedExpertPtrs>,
    ) -> Result<(), Status> {
        let moe = self.config.moe_config().ok_or(Status::InternalError)?;
        self.gemm_bf16(
            self.scratch.attn_proj.as_device_ptr(),
            rows,
            hidden,
            router_proj,
            self.scratch.router_logits.as_device_ptr(),
            moe.num_experts,
            GemmOut::F32,
        )?;

        let router = RouterTopK::new(
            Bf16OrF32Mat::F32(DMat::contiguous(
                self.scratch.router_logits.as_device_ptr(),
                rows,
                moe.num_experts,
            )?),
            DMat::contiguous(
                self.scratch.topk_ids.as_device_ptr(),
                rows,
                moe.num_experts_per_tok,
            )?,
            DMat::contiguous(
                self.scratch.topk_weights.as_device_ptr(),
                rows,
                moe.num_experts_per_tok,
            )?,
            QWEN36_MOE_ROUTER_SCORE,
            QWEN36_MOE_ROUTER_RENORMALIZE,
            QWEN36_MOE_ROUTER_SCALING_FACTOR,
        )?;
        {
            let mut ops = self.engine.operators();
            unsafe { ops.cuda.router_topk(&router)? };
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
            unsafe { ops.flashinfer.moe_execute_bf16(plan, &execute)? };
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

    pub(in crate::model) fn execute_shared_expert_mlp(
        &mut self,
        rows: u32,
        hidden: u32,
        intermediate: u32,
        shared: QwenSharedExpertPtrs,
    ) -> Result<(), Status> {
        if intermediate == 0 {
            return Err(Status::InternalError);
        }
        self.gemm_bf16(
            self.scratch.attn_proj.as_device_ptr(),
            rows,
            hidden,
            shared.gate_proj,
            self.scratch.shared_gate.as_device_ptr(),
            intermediate,
            GemmOut::Bf16,
        )?;
        self.gemm_bf16(
            self.scratch.attn_proj.as_device_ptr(),
            rows,
            hidden,
            shared.up_proj,
            self.scratch.shared_up.as_device_ptr(),
            intermediate,
            GemmOut::Bf16,
        )?;
        self.silu_and_mul(
            rows,
            intermediate,
            self.scratch.shared_gate.as_device_ptr(),
            self.scratch.shared_up.as_device_ptr(),
            self.scratch.shared_mlp.as_device_ptr(),
        )?;
        self.gemm_bf16(
            self.scratch.shared_mlp.as_device_ptr(),
            rows,
            intermediate,
            shared.down_proj,
            self.scratch.shared_out.as_device_ptr(),
            hidden,
            GemmOut::Bf16,
        )?;
        self.gemm_bf16(
            self.scratch.attn_proj.as_device_ptr(),
            rows,
            hidden,
            shared.shared_expert_gate,
            self.scratch.shared_gate_logits.as_device_ptr(),
            1,
            GemmOut::F32,
        )?;
        self.shared_expert_gate_add(rows, hidden)
    }
}
