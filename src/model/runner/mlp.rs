use super::BatchExecution;
use crate::dtype::BF16;
use crate::{
    QWEN36_MOE_ROUTER_RENORMALIZE, QWEN36_MOE_ROUTER_SCALING_FACTOR, QWEN36_MOE_ROUTER_SCORE,
    backend::qsfi::{FusedAddRmsNormBf16, MoeBf16Execute, MoeBf16ExecuteArgs},
    engine::Status,
    memory::DeviceSpan,
    model::{
        scratch::MlpScratch,
        weights::{QwenMlpWeights, QwenSharedExpertWeights},
    },
};

impl BatchExecution<'_> {
    pub(super) unsafe fn execute_post_attention_mlp(
        &mut self,
        rows: u32,
        norm: &DeviceSpan<BF16>,
        mlp: &QwenMlpWeights,
        next_norm: &DeviceSpan<BF16>,
    ) -> Result<(), Status> {
        let hidden = self.config.hidden_size();
        let residual = self.scratch.residual.matrix(rows, hidden)?;
        let before = FusedAddRmsNormBf16::qwen_decoder_norm(
            self.scratch.attn_proj.matrix(rows, hidden)?,
            residual,
            norm.vector(hidden)?,
            self.config.rms_norm_eps(),
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
                let MlpScratch::Dense(scratch) = &self.scratch.mlp else {
                    return Err(Status::InternalError);
                };
                let intermediate = self.config.intermediate_size();
                let input = self.scratch.attn_proj.matrix(rows, hidden)?;
                let gate = scratch.gate.matrix(rows, intermediate)?;
                let up = scratch.up.matrix(rows, intermediate)?;
                let activated = scratch.activated.matrix(rows, intermediate)?;
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
            self.config.rms_norm_eps(),
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
        router_proj: &DeviceSpan<BF16>,
        gate_up_proj: &DeviceSpan<BF16>,
        down_proj: &DeviceSpan<BF16>,
        shared: Option<&QwenSharedExpertWeights>,
    ) -> Result<(), Status> {
        let hidden = self.config.hidden_size();
        let moe = self.config.moe_config().ok_or(Status::InternalError)?;
        let MlpScratch::Moe(scratch) = &self.scratch.mlp else {
            return Err(Status::InternalError);
        };
        let input = self.scratch.attn_proj.matrix(rows, hidden)?;
        let logits = scratch.router_logits.matrix(rows, moe.num_experts)?;
        let ids = scratch.topk_ids.matrix(rows, moe.num_experts_per_tok)?;
        let weights = scratch.topk_weights.matrix(rows, moe.num_experts_per_tok)?;
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
            workspace: scratch.workspace.workspace(scratch.workspace.len)?,
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
        let MlpScratch::Moe(moe) = &self.scratch.mlp else {
            return Err(Status::InternalError);
        };
        let scratch = moe.shared.as_ref().ok_or(Status::InternalError)?;
        let hidden = self.config.hidden_size();
        let input = self.scratch.attn_proj.matrix(rows, hidden)?;
        let gate = scratch.gate.matrix(rows, intermediate)?;
        let up = scratch.up.matrix(rows, intermediate)?;
        let activated = scratch.activated.matrix(rows, intermediate)?;
        let output = scratch.out.matrix(rows, hidden)?;
        let logits = scratch.gate_logits.matrix(rows, 1)?;
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
