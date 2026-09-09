use super::BatchExecution;
use crate::{
    QWEN36_GDN_CONV_WIDTH, QWEN36_GDN_KEY_DIM, QWEN36_GDN_NUM_K_HEADS, QWEN36_GDN_NUM_Q_HEADS,
    QWEN36_GDN_NUM_V_HEADS, QWEN36_GDN_OUTPUT_DIM, QWEN36_GDN_PACKED_DIM, QWEN36_GDN_VALUE_DIM,
    backend::{BF16, DMat},
    engine::Status,
    model::{ActiveRunKind, weights::QwenGdnWeights},
};

impl BatchExecution<'_> {
    pub(super) fn execute_gdn_layer(
        &mut self,
        gdn_layer_idx: u32,
        rows: u32,
        input: DMat<BF16>,
        layer: &QwenGdnWeights,
        kind: ActiveRunKind,
    ) -> Result<(), Status> {
        if matches!(kind, ActiveRunKind::Decode) && rows != 1 {
            return Err(Status::InvalidArgument);
        }
        let hidden = self.config.hidden_size;
        let state = self.gdn_state.ok_or(Status::InternalError)?;
        let slots = state.layer_slots(gdn_layer_idx)?;
        let conv_state = state.conv_view()?;
        let recurrent_state = state.recurrent_view()?;
        self.scratch.gdn_state_indices.upload(
            self.config.stream,
            &[i32::try_from(slots.live_slot).map_err(|_| Status::InvalidArgument)?],
        )?;
        self.scratch.gdn_state_out_indices.upload(
            self.config.stream,
            &[i32::try_from(slots.staged_slot).map_err(|_| Status::InvalidArgument)?],
        )?;
        let seq_indptr = if matches!(kind, ActiveRunKind::Append) {
            self.scratch.gdn_seq_indptr.upload(
                self.config.stream,
                &[0, i32::try_from(rows).map_err(|_| Status::InvalidArgument)?],
            )?;
            Some(self.scratch.gdn_seq_indptr.vector(2)?)
        } else {
            None
        };
        let read_indices = self.scratch.gdn_state_indices.vector(1)?;
        let write_indices = Some(self.scratch.gdn_state_out_indices.vector(1)?);
        let packed = self
            .scratch
            .gdn_packed
            .matrix(rows, QWEN36_GDN_PACKED_DIM)?;
        let conv_out = self
            .scratch
            .gdn_conv_out
            .matrix(rows, QWEN36_GDN_PACKED_DIM)?;
        let a = self.scratch.gdn_a.matrix(rows, QWEN36_GDN_NUM_V_HEADS)?;
        let b = self.scratch.gdn_b.matrix(rows, QWEN36_GDN_NUM_V_HEADS)?;
        let a_log = layer.a_log.vector(QWEN36_GDN_NUM_V_HEADS)?;
        let dt_bias = layer.dt_bias.vector(QWEN36_GDN_NUM_V_HEADS)?;
        let q = self
            .scratch
            .gdn_q
            .heads(rows, QWEN36_GDN_NUM_Q_HEADS, QWEN36_GDN_KEY_DIM)?;
        let k = self
            .scratch
            .gdn_k
            .heads(rows, QWEN36_GDN_NUM_K_HEADS, QWEN36_GDN_KEY_DIM)?;
        let v = self
            .scratch
            .gdn_v
            .heads(rows, QWEN36_GDN_NUM_V_HEADS, QWEN36_GDN_VALUE_DIM)?;
        let out = self.scratch.gdn_recurrent_out.heads(
            rows,
            QWEN36_GDN_NUM_V_HEADS,
            QWEN36_GDN_VALUE_DIM,
        )?;
        let gate =
            self.scratch
                .gdn_gate
                .heads(rows, QWEN36_GDN_NUM_V_HEADS, QWEN36_GDN_VALUE_DIM)?;
        let norm_out =
            self.scratch
                .gdn_norm_out
                .heads(rows, QWEN36_GDN_NUM_V_HEADS, QWEN36_GDN_VALUE_DIM)?;
        let mut ops = self.engine.operators();
        unsafe {
            ops.qscb().linear(
                input,
                layer.in_proj.matrix(QWEN36_GDN_PACKED_DIM, hidden)?,
                packed,
                self.linear_workspace,
            )?;
            ops.qscb().linear(
                input,
                layer.a_proj.matrix(QWEN36_GDN_NUM_V_HEADS, hidden)?,
                a,
                self.linear_workspace,
            )?;
            ops.qscb().linear(
                input,
                layer.b_proj.matrix(QWEN36_GDN_NUM_V_HEADS, hidden)?,
                b,
                self.linear_workspace,
            )?;
            ops.qscb().linear(
                input,
                layer.gate_proj.matrix(QWEN36_GDN_OUTPUT_DIM, hidden)?,
                self.scratch.gdn_gate.matrix(rows, QWEN36_GDN_OUTPUT_DIM)?,
                self.linear_workspace,
            )?;
            ops.qscu().qwen36_gdn_causal_conv1d_bf16(
                packed,
                layer
                    .conv_weight
                    .matrix(QWEN36_GDN_PACKED_DIM, QWEN36_GDN_CONV_WIDTH)?,
                layer.conv_bias.vector(QWEN36_GDN_PACKED_DIM)?,
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
                layer.rms_weight.vector(QWEN36_GDN_VALUE_DIM)?,
                norm_out,
                self.config.rms_norm_eps,
            )?;
            ops.qscb().linear(
                self.scratch
                    .gdn_norm_out
                    .matrix(rows, QWEN36_GDN_OUTPUT_DIM)?,
                layer.out_proj.matrix(hidden, QWEN36_GDN_OUTPUT_DIM)?,
                self.scratch.attn_proj.matrix(rows, hidden)?,
                self.linear_workspace,
            )
        }
    }
}
