use super::BatchExecution;
use crate::{
    QWEN36_MOE_ROUTER_RENORMALIZE, QWEN36_MOE_ROUTER_SCALING_FACTOR, QWEN36_MOE_ROUTER_SCORE,
    backend::qsfi::{FusedAddRmsNormBf16, MoeBf16Execute, MoeBf16ExecuteArgs},
    engine::Status,
    model::{
        scratch::DeviceBuffer,
        weights::{QwenMlpWeights, QwenSharedExpertWeights},
    },
};

impl BatchExecution<'_> {
    pub(super) unsafe fn execute_post_attention_mlp(
        &mut self,
        rows: u32,
        norm: &DeviceBuffer<u16>,
        mlp: &QwenMlpWeights,
        next_norm: &DeviceBuffer<u16>,
    ) -> Result<(), Status> {
        let hidden = self.config.hidden_size;
        let residual = self.scratch.residual.matrix(rows, hidden)?;
        let before = FusedAddRmsNormBf16::qwen_decoder_norm(
            self.scratch.attn_proj.matrix(rows, hidden)?,
            residual,
            norm.vector(hidden)?,
            self.config.rms_norm_eps,
        )?;
        unsafe {
            self.engine
                .operators()
                .qsfi()
                .fused_add_rmsnorm_bf16(&before)?;
        }
        match mlp {
            QwenMlpWeights::Dense {
                gate_proj,
                up_proj,
                down_proj,
            } => {
                let intermediate = self.config.intermediate_size;
                let input = self.scratch.attn_proj.matrix(rows, hidden)?;
                let gate = self.scratch.gate.matrix(rows, intermediate)?;
                let up = self.scratch.up.matrix(rows, intermediate)?;
                let activated = self.scratch.mlp.matrix(rows, intermediate)?;
                let mut ops = self.engine.operators();
                unsafe {
                    ops.qscb().linear(
                        input,
                        gate_proj.matrix(intermediate, hidden)?,
                        gate,
                        self.linear_workspace,
                    )?;
                    ops.qscb().linear(
                        input,
                        up_proj.matrix(intermediate, hidden)?,
                        up,
                        self.linear_workspace,
                    )?;
                    ops.qscu().silu_and_mul_bf16(gate, up, activated)?;
                    ops.qscb().linear(
                        activated,
                        down_proj.matrix(hidden, intermediate)?,
                        self.scratch.mlp_out.matrix(rows, hidden)?,
                        self.linear_workspace,
                    )?;
                }
            }
            QwenMlpWeights::Moe {
                router_proj,
                gate_up_proj,
                down_proj,
                shared,
            } => unsafe {
                self.execute_moe_mlp(rows, router_proj, gate_up_proj, down_proj, shared.as_ref())?;
            },
        }
        let after = FusedAddRmsNormBf16::qwen_decoder_norm(
            self.scratch.mlp_out.matrix(rows, hidden)?,
            residual,
            next_norm.vector(hidden)?,
            self.config.rms_norm_eps,
        )?;
        unsafe {
            self.engine
                .operators()
                .qsfi()
                .fused_add_rmsnorm_bf16(&after)
        }
    }

    pub(super) unsafe fn execute_moe_mlp(
        &mut self,
        rows: u32,
        router_proj: &DeviceBuffer<u16>,
        gate_up_proj: &DeviceBuffer<u16>,
        down_proj: &DeviceBuffer<u16>,
        shared: Option<&QwenSharedExpertWeights>,
    ) -> Result<(), Status> {
        let hidden = self.config.hidden_size;
        let moe = self.config.moe_config().ok_or(Status::InternalError)?;
        let input = self.scratch.attn_proj.matrix(rows, hidden)?;
        let logits = self.scratch.router_logits.matrix(rows, moe.num_experts)?;
        let ids = self
            .scratch
            .topk_ids
            .matrix(rows, moe.num_experts_per_tok)?;
        let weights = self
            .scratch
            .topk_weights
            .matrix(rows, moe.num_experts_per_tok)?;
        let plan = self.moe_plan.ok_or(Status::InternalError)?;
        let execute = MoeBf16Execute::new(MoeBf16ExecuteArgs {
            hidden: input,
            topk_ids: ids,
            topk_weights: weights,
            gate_up_weight: gate_up_proj.tensor3(
                moe.num_experts,
                moe.moe_intermediate_size
                    .checked_mul(2)
                    .ok_or(Status::InvalidArgument)?,
                hidden,
            )?,
            down_weight: down_proj.tensor3(moe.num_experts, hidden, moe.moe_intermediate_size)?,
            out: self.scratch.mlp_out.matrix(rows, hidden)?,
            workspace: self.moe_workspace,
        })?;
        {
            let mut ops = self.engine.operators();
            unsafe {
                ops.qscb().linear(
                    input,
                    router_proj.matrix(moe.num_experts, hidden)?,
                    logits,
                    self.linear_workspace,
                )?;
                ops.qscu().router_topk(
                    logits,
                    ids,
                    weights,
                    QWEN36_MOE_ROUTER_SCORE,
                    QWEN36_MOE_ROUTER_RENORMALIZE,
                    QWEN36_MOE_ROUTER_SCALING_FACTOR,
                )?;
                ops.qsfi().moe_execute_bf16(plan, &execute)?;
            }
        }
        if let Some(shared) = shared {
            unsafe {
                self.execute_shared_expert_mlp(rows, moe.shared_expert_intermediate_size, shared)?;
            }
        }
        Ok(())
    }

    unsafe fn execute_shared_expert_mlp(
        &mut self,
        rows: u32,
        intermediate: u32,
        shared: &QwenSharedExpertWeights,
    ) -> Result<(), Status> {
        let hidden = self.config.hidden_size;
        let input = self.scratch.attn_proj.matrix(rows, hidden)?;
        let gate = self.scratch.shared_gate.matrix(rows, intermediate)?;
        let up = self.scratch.shared_up.matrix(rows, intermediate)?;
        let activated = self.scratch.shared_mlp.matrix(rows, intermediate)?;
        let output = self.scratch.shared_out.matrix(rows, hidden)?;
        let logits = self.scratch.shared_gate_logits.matrix(rows, 1)?;
        let mut ops = self.engine.operators();
        unsafe {
            ops.qscb().linear(
                input,
                shared.gate_proj.matrix(intermediate, hidden)?,
                gate,
                self.linear_workspace,
            )?;
            ops.qscb().linear(
                input,
                shared.up_proj.matrix(intermediate, hidden)?,
                up,
                self.linear_workspace,
            )?;
            ops.qscu().silu_and_mul_bf16(gate, up, activated)?;
            ops.qscb().linear(
                activated,
                shared.down_proj.matrix(hidden, intermediate)?,
                output,
                self.linear_workspace,
            )?;
            ops.qscb().linear(
                input,
                shared.shared_expert_gate.matrix(1, hidden)?,
                logits,
                self.linear_workspace,
            )?;
            ops.qscu().qwen36_shared_expert_gate_add_bf16(
                logits,
                output,
                self.scratch.mlp_out.matrix(rows, hidden)?,
            )
        }
    }
}
