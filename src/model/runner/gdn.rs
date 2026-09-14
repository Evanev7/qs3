use super::BatchExecution;
use crate::{
    backend::{BF16, DMat},
    constants::gdn::{
        CONV_WIDTH, KEY_HEAD_DIM, NUM_KEY_HEADS, NUM_VALUE_HEADS, OUTPUT_WIDTH,
        PACKED_QKV_CHANNELS, VALUE_HEAD_DIM,
    },
    engine::Status,
    memory::CudaCtx,
    model::{ActiveRunKind, weights::QwenGdnWeights},
};

impl BatchExecution<'_> {
    pub(super) unsafe fn execute_gdn_layer(
        &mut self,
        ctx: &CudaCtx,
        gdn_layer_idx: u32,
        rows: u32,
        input: DMat<BF16>,
        layer: &QwenGdnWeights,
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
        let live_slot = [i32::try_from(slots.live_slot).map_err(|_| Status::InvalidArgument)?];
        let staged_slot = [i32::try_from(slots.staged_slot).map_err(|_| Status::InvalidArgument)?];
        let sequence = [0, i32::try_from(rows).map_err(|_| Status::InvalidArgument)?];
        let scratch = self.scratch.gdn.as_mut().ok_or(Status::InternalError)?;
        // Keep host metadata alive through the explicit layer completion boundary.
        // The closure also routes enqueue/validation failures through that wait.
        let result = (|| -> Result<(), Status> {
            unsafe {
                scratch.state_indices.upload(&live_slot)?;
                scratch.state_out_indices.upload(&staged_slot)?;
            }
            let seq_indptr = if matches!(kind, ActiveRunKind::Append) {
                unsafe {
                    scratch.seq_indptr.upload(&sequence)?;
                }
                Some(scratch.seq_indptr.vector(2)?)
            } else {
                None
            };
            let read_indices = scratch.state_indices.vector(1)?;
            let write_indices = Some(scratch.state_out_indices.vector(1)?);
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
        })();
        // TODO(async-upload-lifetimes): retain GDN host metadata until batch
        // completion, then remove this temporary per-layer stream wait.
        let completion = ctx.synchronize();
        result.and(completion)
    }
}
