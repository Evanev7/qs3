use super::BatchExecution;
use crate::{
    QWEN36_FULL_ATTN_Q_PROJ_OUT, QWEN36_FULL_ATTN_ROTARY_DIM,
    backend::{
        BF16, DMat,
        qsfi::{RmsNormBf16, RopeApplyBf16},
    },
    engine::{AttentionLayer, Status},
    model::{ActiveRunKind, weights::QwenAttentionMlpWeights},
};

impl BatchExecution<'_> {
    pub(super) fn execute_attention_layer(
        &mut self,
        attention_layer_idx: u32,
        rows: u32,
        input: DMat<BF16>,
        layer: &QwenAttentionMlpWeights,
        kind: ActiveRunKind,
    ) -> Result<(), Status> {
        let hidden = self.config.hidden_size;
        let q_hidden = self.config.q_hidden_size()?;
        let kv_hidden = self.config.kv_hidden_size()?;
        {
            let mut ops = self.engine.operators();
            unsafe {
                ops.qscb().linear(
                    input,
                    layer.q_proj.matrix(QWEN36_FULL_ATTN_Q_PROJ_OUT, hidden)?,
                    self.scratch
                        .q_proj_out
                        .matrix(rows, QWEN36_FULL_ATTN_Q_PROJ_OUT)?,
                    self.linear_workspace,
                )?;
                ops.qscu().qwen36_extract_q_and_gate_bf16(
                    self.scratch
                        .q_proj_out
                        .matrix(rows, QWEN36_FULL_ATTN_Q_PROJ_OUT)?,
                    self.scratch.q.matrix(rows, q_hidden)?,
                    self.scratch.attn_gate.matrix(rows, q_hidden)?,
                )?;
                ops.qscb().linear(
                    input,
                    layer.k_proj.matrix(kv_hidden, hidden)?,
                    self.scratch.k.matrix(rows, kv_hidden)?,
                    self.linear_workspace,
                )?;
                ops.qscb().linear(
                    input,
                    layer.v_proj.matrix(kv_hidden, hidden)?,
                    self.scratch.v.matrix(rows, kv_hidden)?,
                    self.linear_workspace,
                )?;
                for (buffer, weight, heads) in [
                    (&self.scratch.q, &layer.q_norm, self.config.num_q_heads),
                    (&self.scratch.k, &layer.k_norm, self.config.num_kv_heads),
                ] {
                    let flattened = buffer.matrix(
                        rows.checked_mul(heads).ok_or(Status::InvalidArgument)?,
                        self.config.head_dim,
                    )?;
                    let norm = RmsNormBf16::qwen_qk_norm(
                        flattened,
                        weight.vector(self.config.head_dim)?,
                        flattened,
                        self.config.rms_norm_eps,
                    )?;
                    ops.qsfi().rmsnorm_bf16(&norm)?;
                }
            }
        }
        self.apply_attention_rope(rows)?;
        let q = self
            .scratch
            .q
            .heads(rows, self.config.num_q_heads, self.config.head_dim)?;
        let k = self
            .scratch
            .k
            .heads(rows, self.config.num_kv_heads, self.config.head_dim)?;
        let v = self
            .scratch
            .v
            .heads(rows, self.config.num_kv_heads, self.config.head_dim)?;
        let output =
            self.scratch
                .attn_out
                .heads(rows, self.config.num_q_heads, self.config.head_dim)?;
        let engine_layer = AttentionLayer::bf16_attention(
            attention_layer_idx,
            q,
            k,
            v,
            output,
            self.scratch.positions.vector(rows)?,
        );
        unsafe {
            match kind {
                ActiveRunKind::Append => self.engine.append_attention(&engine_layer)?,
                ActiveRunKind::Decode => self.engine.decode_attention(&engine_layer)?,
            }
        }
        let out = self.scratch.attn_out.matrix(rows, q_hidden)?;
        let mut ops = self.engine.operators();
        unsafe {
            ops.qscu().qwen36_full_attention_output_gate_bf16(
                self.scratch.attn_gate.matrix(rows, q_hidden)?,
                out,
            )?;
            ops.qscb().linear(
                out,
                layer.o_proj.matrix(hidden, q_hidden)?,
                self.scratch.attn_proj.matrix(rows, hidden)?,
                self.linear_workspace,
            )
        }
    }

    pub(super) fn apply_attention_rope(&mut self, rows: u32) -> Result<(), Status> {
        let q = self
            .scratch
            .q
            .heads(rows, self.config.num_q_heads, self.config.head_dim)?;
        let k = self
            .scratch
            .k
            .heads(rows, self.config.num_kv_heads, self.config.head_dim)?;
        let desc = RopeApplyBf16::with_params(
            q,
            k,
            q,
            k,
            self.scratch.positions.vector(rows)?,
            QWEN36_FULL_ATTN_ROTARY_DIM,
            self.config.rope_scale,
            self.config.rope_theta,
        )?;
        unsafe { self.engine.operators().qsfi().rope_apply_bf16(&desc) }
    }
}
