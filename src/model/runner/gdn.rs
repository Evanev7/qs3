use super::BatchExecution;
use crate::{
    backend::DMat,
    constants::gdn::{
        CONV_WIDTH, KEY_HEAD_DIM, NUM_KEY_HEADS, NUM_VALUE_HEADS, OUTPUT_WIDTH,
        PACKED_QKV_CHANNELS, VALUE_HEAD_DIM,
    },
    dtype::BF16,
    engine::Status,
    memory::CudaCtx,
    model::{ActiveRunKind, weights::QwenGdnWeights},
};

impl BatchExecution<'_> {
    // Append metadata must be prepared for `rows` on this stream before execution.
    pub(super) unsafe fn execute_gdn_layer<M>(
        &mut self,
        ctx: &CudaCtx,
        gdn_layer_idx: u32,
        rows: u32,
        input: DMat<BF16>,
        layer: &QwenGdnWeights<M>,
        kind: ActiveRunKind,
    ) -> Result<(), Status> {
        if matches!(kind, ActiveRunKind::Decode) && rows != 1 {
            return Err(Status::InvalidArgument);
        }
        let hidden = self.config.hidden_size();
        let state = self.gdn_state.ok_or(Status::InternalError)?;
        let slots = state.layer_slots(gdn_layer_idx)?;
        let conv_state = state.conv_view()?;
        let recurrent_state = state.recurrent_view()?;
        let scratch = self.scratch.gdn.as_ref().ok_or(Status::InternalError)?;
        let seq_indptr = match kind {
            ActiveRunKind::Append => Some(scratch.seq_indptr.vector(2)?),
            ActiveRunKind::Decode => None,
        };
        let read_indices = scratch.state_index(slots.live_slot)?;
        let write_indices = Some(scratch.state_index(slots.staged_slot)?);
        let packed = scratch.packed.matrix(rows, PACKED_QKV_CHANNELS)?;
        let conv_out = scratch.conv_out.matrix(rows, PACKED_QKV_CHANNELS)?;
        let a = scratch.a.matrix(rows, NUM_VALUE_HEADS)?;
        let b = scratch.b.matrix(rows, NUM_VALUE_HEADS)?;
        let a_log = layer.a_log.vector(NUM_VALUE_HEADS)?;
        let dt_bias = layer.dt_bias.vector(NUM_VALUE_HEADS)?;
        let q = scratch.q.heads(rows, NUM_KEY_HEADS, KEY_HEAD_DIM)?;
        let k = scratch.k.heads(rows, NUM_KEY_HEADS, KEY_HEAD_DIM)?;
        let v = scratch.v.heads(rows, NUM_VALUE_HEADS, VALUE_HEAD_DIM)?;
        let out = scratch
            .recurrent_out
            .heads(rows, NUM_VALUE_HEADS, VALUE_HEAD_DIM)?;
        let gate = scratch.gate.heads(rows, NUM_VALUE_HEADS, VALUE_HEAD_DIM)?;
        let norm_out = scratch
            .norm_out
            .heads(rows, NUM_VALUE_HEADS, VALUE_HEAD_DIM)?;
        let mut ops = self.engine.operators();
        unsafe {
            let weight = layer.in_proj.matrix(PACKED_QKV_CHANNELS, hidden)?;
            match (kind, self.gdn_qkv) {
                (ActiveRunKind::Decode, Some(kernel)) => {
                    kernel.launch(ctx.stream, input, weight, packed)?;
                }
                _ => {
                    ops.qscb()
                        .linear(input, weight, packed, self.linear_workspace)?;
                }
            }
            ops.qscb().linear(
                input,
                layer.a_proj.matrix(NUM_VALUE_HEADS, hidden)?,
                a,
                self.linear_workspace,
            )?;
            ops.qscb().linear(
                input,
                layer.b_proj.matrix(NUM_VALUE_HEADS, hidden)?,
                b,
                self.linear_workspace,
            )?;
            ops.qscb().linear(
                input,
                layer.gate_proj.matrix(OUTPUT_WIDTH, hidden)?,
                scratch.gate.matrix(rows, OUTPUT_WIDTH)?,
                self.linear_workspace,
            )?;
            ops.qscu().qwen36_gdn_causal_conv1d_bf16(
                packed,
                layer.conv_weight.matrix(PACKED_QKV_CHANNELS, CONV_WIDTH)?,
                layer.conv_bias.vector(PACKED_QKV_CHANNELS)?,
                conv_state,
                Some(read_indices),
                write_indices,
                seq_indptr,
                conv_out,
                1,
            )?;
            ops.qscu()
                .qwen36_gdn_post_conv_prepare_bf16(conv_out, a, b, a_log, dt_bias, q, k, v)?;
            match kind {
                ActiveRunKind::Append => ops.qscu().qwen36_gdn_prefill_bf16(
                    q,
                    k,
                    v,
                    a,
                    b,
                    a_log,
                    dt_bias,
                    recurrent_state,
                    seq_indptr.ok_or(Status::InternalError)?,
                    read_indices,
                    write_indices,
                    out,
                    1,
                )?,
                ActiveRunKind::Decode => ops.qscu().qwen36_gdn_decode_bf16(
                    q,
                    k,
                    v,
                    a,
                    b,
                    a_log,
                    dt_bias,
                    recurrent_state,
                    read_indices,
                    write_indices,
                    out,
                )?,
            }
            ops.qscu().qwen36_gdn_gated_rmsnorm_bf16(
                out,
                gate,
                layer.rms_weight.vector(VALUE_HEAD_DIM)?,
                norm_out,
                self.config.rms_norm_eps(),
            )?;
            ops.qscb().linear(
                scratch.norm_out.matrix(rows, OUTPUT_WIDTH)?,
                layer.out_proj.matrix(hidden, OUTPUT_WIDTH)?,
                self.scratch.attn_proj.matrix(rows, hidden)?,
                self.linear_workspace,
            )
        }
    }
}
