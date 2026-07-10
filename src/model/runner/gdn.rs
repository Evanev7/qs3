use super::ModelRunner;
use crate::{
    QWEN36_GDN_CONV_WIDTH, QWEN36_GDN_NUM_V_HEADS, QWEN36_GDN_OUTPUT_DIM, QWEN36_GDN_PACKED_DIM,
    QWEN36_GDN_VALUE_DIM,
    backend::{DMat, DVec},
    engine::Status,
    ffi,
    model::{ActiveRunKind, weights::QwenGdnPtrs},
};

impl ModelRunner {
    pub(super) fn execute_gdn_layer(
        &mut self,
        gdn_layer_idx: u32,
        rows: u32,
        hidden: u32,
        layer_input: ffi::DevicePtr,
        layer: QwenGdnPtrs,
        kind: ActiveRunKind,
    ) -> Result<(), Status> {
        if matches!(kind, ActiveRunKind::Decode) && rows != 1 {
            return Err(Status::InvalidArgument);
        }
        let (_state_pool, live_slot, staged_slot, conv_state, recurrent_state) =
            self.gdn_state_views(gdn_layer_idx)?;
        let live_slot_i32 = i32::try_from(live_slot).map_err(|_| Status::InvalidArgument)?;
        let staged_slot_i32 = i32::try_from(staged_slot).map_err(|_| Status::InvalidArgument)?;
        self.scratch
            .gdn_state_indices
            .upload(self.config.stream, &[live_slot_i32])?;
        self.scratch
            .gdn_state_out_indices
            .upload(self.config.stream, &[staged_slot_i32])?;

        self.linear_bf16(
            layer_input,
            rows,
            hidden,
            layer.in_proj,
            self.scratch.gdn_packed.as_device_ptr(),
            QWEN36_GDN_PACKED_DIM,
        )?;
        self.linear_bf16(
            layer_input,
            rows,
            hidden,
            layer.a_proj,
            self.scratch.gdn_a.as_device_ptr(),
            QWEN36_GDN_NUM_V_HEADS,
        )?;
        self.linear_bf16(
            layer_input,
            rows,
            hidden,
            layer.b_proj,
            self.scratch.gdn_b.as_device_ptr(),
            QWEN36_GDN_NUM_V_HEADS,
        )?;
        self.linear_bf16(
            layer_input,
            rows,
            hidden,
            layer.gate_proj,
            self.scratch.gdn_gate.as_device_ptr(),
            QWEN36_GDN_OUTPUT_DIM,
        )?;

        let seq_indptr = if matches!(kind, ActiveRunKind::Append) {
            let rows_i32 = i32::try_from(rows).map_err(|_| Status::InvalidArgument)?;
            self.scratch
                .gdn_seq_indptr
                .upload(self.config.stream, &[0, rows_i32])?;
            Some(DVec::contiguous(
                self.scratch.gdn_seq_indptr.as_device_ptr(),
                2,
            )?)
        } else {
            None
        };

        let packed = DMat::contiguous(
            self.scratch.gdn_packed.as_device_ptr(),
            rows,
            QWEN36_GDN_PACKED_DIM,
        )?;
        let conv_weight = DMat::contiguous(
            layer.conv_weight,
            QWEN36_GDN_PACKED_DIM,
            QWEN36_GDN_CONV_WIDTH,
        )?;
        let conv_bias = DVec::contiguous(layer.conv_bias, QWEN36_GDN_PACKED_DIM)?;
        let state_read_indices = Some(DVec::contiguous(
            self.scratch.gdn_state_indices.as_device_ptr(),
            1,
        )?);
        let state_write_indices = Some(DVec::contiguous(
            self.scratch.gdn_state_out_indices.as_device_ptr(),
            1,
        )?);
        let conv_out = DMat::contiguous(
            self.scratch.gdn_conv_out.as_device_ptr(),
            rows,
            QWEN36_GDN_PACKED_DIM,
        )?;
        {
            let mut ops = self.engine.operators();
            unsafe {
                ops.qscu().qwen36_gdn_causal_conv1d_bf16(
                    packed,
                    conv_weight,
                    conv_bias,
                    conv_state,
                    state_read_indices,
                    state_write_indices,
                    seq_indptr,
                    conv_out,
                    1,
                )?
            };
        }

        let a = DMat::contiguous(
            self.scratch.gdn_a.as_device_ptr(),
            rows,
            QWEN36_GDN_NUM_V_HEADS,
        )?;
        let b = DMat::contiguous(
            self.scratch.gdn_b.as_device_ptr(),
            rows,
            QWEN36_GDN_NUM_V_HEADS,
        )?;
        let a_log = DVec::contiguous(layer.a_log, QWEN36_GDN_NUM_V_HEADS)?;
        let dt_bias = DVec::contiguous(layer.dt_bias, QWEN36_GDN_NUM_V_HEADS)?;
        let q = self.gdn_q_heads(self.scratch.gdn_q.as_device_ptr(), rows)?;
        let k = self.gdn_k_heads(self.scratch.gdn_k.as_device_ptr(), rows)?;
        let v = self.gdn_v_heads(self.scratch.gdn_v.as_device_ptr(), rows)?;
        {
            let mut ops = self.engine.operators();
            unsafe {
                ops.qscu()
                    .qwen36_gdn_post_conv_prepare_bf16(conv_out, a, b, a_log, dt_bias, q, k, v)?
            };
        }

        match kind {
            ActiveRunKind::Append => {
                let rows_i32 = i32::try_from(rows).map_err(|_| Status::InvalidArgument)?;
                self.scratch
                    .gdn_seq_indptr
                    .upload(self.config.stream, &[0, rows_i32])?;
                let seq_indptr = DVec::contiguous(self.scratch.gdn_seq_indptr.as_device_ptr(), 2)?;
                let state_indices =
                    DVec::contiguous(self.scratch.gdn_state_indices.as_device_ptr(), 1)?;
                let state_out_indices = Some(DVec::contiguous(
                    self.scratch.gdn_state_out_indices.as_device_ptr(),
                    1,
                )?);
                let out = self.gdn_v_heads(self.scratch.gdn_recurrent_out.as_device_ptr(), rows)?;
                let mut ops = self.engine.operators();
                unsafe {
                    ops.qscu().qwen36_gdn_prefill_bf16(
                        q,
                        k,
                        v,
                        a,
                        b,
                        a_log,
                        dt_bias,
                        recurrent_state,
                        seq_indptr,
                        state_indices,
                        state_out_indices,
                        out,
                        1,
                    )?
                };
            }
            ActiveRunKind::Decode => {
                let state_indices =
                    DVec::contiguous(self.scratch.gdn_state_indices.as_device_ptr(), rows)?;
                let state_out_indices = Some(DVec::contiguous(
                    self.scratch.gdn_state_out_indices.as_device_ptr(),
                    rows,
                )?);
                let out = self.gdn_v_heads(self.scratch.gdn_recurrent_out.as_device_ptr(), rows)?;
                let mut ops = self.engine.operators();
                unsafe {
                    ops.qscu().qwen36_gdn_decode_bf16(
                        q,
                        k,
                        v,
                        a,
                        b,
                        a_log,
                        dt_bias,
                        recurrent_state,
                        state_indices,
                        state_out_indices,
                        out,
                    )?
                };
            }
        }

        let recurrent_out =
            self.gdn_v_heads(self.scratch.gdn_recurrent_out.as_device_ptr(), rows)?;
        let gate = self.gdn_v_heads(self.scratch.gdn_gate.as_device_ptr(), rows)?;
        let rms_weight = DVec::contiguous(layer.rms_weight, QWEN36_GDN_VALUE_DIM)?;
        let norm_out = self.gdn_v_heads(self.scratch.gdn_norm_out.as_device_ptr(), rows)?;
        {
            let mut ops = self.engine.operators();
            unsafe {
                ops.qscu().qwen36_gdn_gated_rmsnorm_bf16(
                    recurrent_out,
                    gate,
                    rms_weight,
                    norm_out,
                    self.config.rms_norm_eps,
                )?
            };
        }

        self.linear_bf16(
            self.scratch.gdn_norm_out.as_device_ptr(),
            rows,
            QWEN36_GDN_OUTPUT_DIM,
            layer.out_proj,
            self.scratch.attn_proj.as_device_ptr(),
            hidden,
        )?;
        Ok(())
    }
}
