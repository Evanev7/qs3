use super::{BatchExecution, Projection, linear::LinearWeight};
use crate::dtype::BF16;
use crate::{
    QWEN36_MOE_ROUTER_RENORMALIZE, QWEN36_MOE_ROUTER_SCALING_FACTOR, QWEN36_MOE_ROUTER_SCORE,
    backend::{
        DMat, DTensor3,
        qsfi::{FusedAddRmsNormBf16, MoeBf16Execute, MoeBf16ExecuteArgs},
    },
    engine::Status,
    memory::DeviceSpan,
    model::{
        QwenConfig,
        scratch::MlpScratch,
        weights::{DenseMlp, FusedExperts, MoeMlp, SharedExpert, W},
    },
};

pub(super) type SharedExpertView = SharedExpert<DMat<BF16>, DMat<BF16>>;
pub(super) type MoeView = MoeMlp<FusedExperts<DTensor3<BF16>>, DMat<BF16>, DMat<BF16>>;

// Both variants contain non-owning device descriptors. The caller retains the
// original allocations until execution completes, as for LinearWeight views.
pub(super) enum MlpView {
    Dense(DenseMlp<LinearWeight>),
    Moe(MoeView),
}

pub(super) trait Mlp {
    fn view(&self, config: &QwenConfig) -> Result<MlpView, Status>;
}
impl<P: Projection> Mlp for DenseMlp<P> {
    fn view(&self, config: &QwenConfig) -> Result<MlpView, Status> {
        let hidden = config.hidden_size();
        let intermediate = config.intermediate_size();
        Ok(MlpView::Dense(DenseMlp {
            gate_proj: self.gate_proj.view(intermediate, hidden)?,
            up_proj: self.up_proj.view(intermediate, hidden)?,
            down_proj: self.down_proj.view(hidden, intermediate)?,
        }))
    }
}
impl Mlp for MoeMlp<FusedExperts<W<BF16>>, W<BF16>> {
    fn view(&self, config: &QwenConfig) -> Result<MlpView, Status> {
        let hidden = config.hidden_size();
        let moe = config.moe_config().ok_or(Status::InvalidArgument)?;
        let intermediate = moe.moe_intermediate_size;
        let shared = self
            .shared
            .as_ref()
            .map(|shared| -> Result<SharedExpertView, Status> {
                let intermediate = moe.shared_expert_intermediate_size;
                Ok(SharedExpert {
                    projections: DenseMlp {
                        gate_proj: shared.projections.gate_proj.matrix(intermediate, hidden)?,
                        up_proj: shared.projections.up_proj.matrix(intermediate, hidden)?,
                        down_proj: shared.projections.down_proj.matrix(hidden, intermediate)?,
                    },
                    gate: shared.gate.matrix(1, hidden)?,
                })
            })
            .transpose()?;
        Ok(MlpView::Moe(MoeMlp {
            router_proj: self.router_proj.matrix(moe.num_experts, hidden)?,
            experts: FusedExperts {
                gate_up_proj: self.experts.gate_up_proj.tensor3(
                    moe.num_experts,
                    intermediate.checked_mul(2).ok_or(Status::InvalidArgument)?,
                    hidden,
                )?,
                down_proj: self
                    .experts
                    .down_proj
                    .tensor3(moe.num_experts, hidden, intermediate)?,
            },
            shared,
        }))
    }
}

impl BatchExecution<'_> {
    pub(super) unsafe fn execute_post_attention_mlp<M: Mlp>(
        &mut self,
        rows: u32,
        norm: &DeviceSpan<BF16>,
        mlp: &M,
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
        unsafe {
            match mlp.view(self.config)? {
                MlpView::Dense(weights) => self.execute_dense_mlp(rows, weights)?,
                MlpView::Moe(weights) => self.execute_moe_mlp(rows, weights)?,
            }
        };
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

    unsafe fn execute_dense_mlp(
        &mut self,
        rows: u32,
        weights: DenseMlp<LinearWeight>,
    ) -> Result<(), Status> {
        let MlpScratch::Dense(scratch) = &self.scratch.mlp else {
            return Err(Status::InternalError);
        };
        let hidden = self.config.hidden_size();
        let intermediate = self.config.intermediate_size();
        let input = self.scratch.attn_proj.matrix(rows, hidden)?;
        let gate = scratch.gate.matrix(rows, intermediate)?;
        let up = scratch.up.matrix(rows, intermediate)?;
        let activated = scratch.activated.matrix(rows, intermediate)?;
        let mut ops = self.engine.operators();
        unsafe {
            ops.linear(
                input,
                weights.gate_proj,
                gate,
                self.quantized_scratch,
                self.linear_workspace,
            )?;
            ops.linear(
                input,
                weights.up_proj,
                up,
                self.quantized_scratch,
                self.linear_workspace,
            )?;
            ops.silu_mul_linear(
                [gate, up],
                weights.down_proj,
                activated,
                self.scratch.mlp_out.matrix(rows, hidden)?,
                self.quantized_scratch,
                self.linear_workspace,
            )
        }
    }

    pub(super) unsafe fn execute_moe_mlp(&mut self, rows: u32, mlp: MoeView) -> Result<(), Status> {
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
            gate_up_weight: mlp.experts.gate_up_proj,
            down_weight: mlp.experts.down_proj,
            out: self.scratch.mlp_out.matrix(rows, hidden)?,
            workspace: scratch.workspace.workspace(scratch.workspace.len)?,
        })?;
        {
            let mut ops = self.engine.operators();
            unsafe {
                ops.qscb()
                    .linear(input, mlp.router_proj, logits, self.linear_workspace)?;
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
        if let Some(shared) = mlp.shared {
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
        shared: SharedExpertView,
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
                shared.projections.gate_proj,
                gate,
                self.linear_workspace,
            )?;
            ops.qscb()
                .linear(input, shared.projections.up_proj, up, self.linear_workspace)?;
            ops.qscu().silu_and_mul_bf16(gate, up, activated)?;
            ops.qscb().linear(
                activated,
                shared.projections.down_proj,
                output,
                self.linear_workspace,
            )?;
            ops.qscb()
                .linear(input, shared.gate, logits, self.linear_workspace)?;
            ops.qscu().qwen36_shared_expert_gate_add_bf16(
                logits,
                output,
                self.scratch.mlp_out.matrix(rows, hidden)?,
            )
        }
    }
}
