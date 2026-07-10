use crate::model::*;

impl ModelRunner {
    pub(in crate::model) fn execute_gdn_layer(
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

        self.gemm_bf16(
            layer_input,
            rows,
            hidden,
            layer.in_proj,
            self.scratch.gdn_packed.as_device_ptr(),
            QWEN36_GDN_PACKED_DIM,
            GemmOut::Bf16,
        )?;
        self.gemm_bf16(
            layer_input,
            rows,
            hidden,
            layer.a_proj,
            self.scratch.gdn_a.as_device_ptr(),
            QWEN36_GDN_NUM_V_HEADS,
            GemmOut::Bf16,
        )?;
        self.gemm_bf16(
            layer_input,
            rows,
            hidden,
            layer.b_proj,
            self.scratch.gdn_b.as_device_ptr(),
            QWEN36_GDN_NUM_V_HEADS,
            GemmOut::Bf16,
        )?;
        self.gemm_bf16(
            layer_input,
            rows,
            hidden,
            layer.gate_proj,
            self.scratch.gdn_gate.as_device_ptr(),
            QWEN36_GDN_OUTPUT_DIM,
            GemmOut::Bf16,
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

        let conv = GdnCausalConv1dBf16::new(GdnCausalConv1dBf16Args {
            x: DMat::contiguous(
                self.scratch.gdn_packed.as_device_ptr(),
                rows,
                QWEN36_GDN_PACKED_DIM,
            )?,
            weight: DMat::contiguous(
                layer.conv_weight,
                QWEN36_GDN_PACKED_DIM,
                QWEN36_GDN_CONV_WIDTH,
            )?,
            bias: Some(Bf16OrF32Vec::Bf16(DVec::contiguous(
                layer.conv_bias,
                QWEN36_GDN_PACKED_DIM,
            )?)),
            state: conv_state,
            state_read_indices: Some(DVec::contiguous(
                self.scratch.gdn_state_indices.as_device_ptr(),
                1,
            )?),
            state_write_indices: Some(DVec::contiguous(
                self.scratch.gdn_state_out_indices.as_device_ptr(),
                1,
            )?),
            seq_indptr,
            out: DMat::contiguous(
                self.scratch.gdn_conv_out.as_device_ptr(),
                rows,
                QWEN36_GDN_PACKED_DIM,
            )?,
            batch_size: 1,
            activation: Activation::Silu,
            update_state: true,
        })?;
        {
            let mut ops = self.engine.operators();
            unsafe { ops.cuda().qwen36_gdn_causal_conv1d_bf16(&conv)? };
        }

        let post = GdnPostConvPrepareBf16::new(GdnPostConvPrepareBf16Args {
            conv_out: DMat::contiguous(
                self.scratch.gdn_conv_out.as_device_ptr(),
                rows,
                QWEN36_GDN_PACKED_DIM,
            )?,
            a: DMat::contiguous(
                self.scratch.gdn_a.as_device_ptr(),
                rows,
                QWEN36_GDN_NUM_V_HEADS,
            )?,
            b: DMat::contiguous(
                self.scratch.gdn_b.as_device_ptr(),
                rows,
                QWEN36_GDN_NUM_V_HEADS,
            )?,
            a_log: DVec::contiguous(layer.a_log, QWEN36_GDN_NUM_V_HEADS)?,
            dt_bias: DVec::contiguous(layer.dt_bias, QWEN36_GDN_NUM_V_HEADS)?,
            q: self.gdn_q_heads(self.scratch.gdn_q.as_device_ptr(), rows)?,
            k: self.gdn_k_heads(self.scratch.gdn_k.as_device_ptr(), rows)?,
            v: self.gdn_v_heads(self.scratch.gdn_v.as_device_ptr(), rows)?,
            g_out: None,
            beta_out: None,
            apply_qk_l2norm: false,
            l2norm_eps: self.config.rms_norm_eps,
            forget_gate_output: GdnForgetGateOutput::LogDecay,
        })?;
        {
            let mut ops = self.engine.operators();
            unsafe { ops.cuda().qwen36_gdn_post_conv_prepare_bf16(&post)? };
        }

        match kind {
            ActiveRunKind::Append => {
                let rows_i32 = i32::try_from(rows).map_err(|_| Status::InvalidArgument)?;
                self.scratch
                    .gdn_seq_indptr
                    .upload(self.config.stream, &[0, rows_i32])?;
                let prefill = GdnPrefillBf16::new(GdnPrefillBf16Args {
                    q: self.gdn_q_heads(self.scratch.gdn_q.as_device_ptr(), rows)?,
                    k: self.gdn_k_heads(self.scratch.gdn_k.as_device_ptr(), rows)?,
                    v: self.gdn_v_heads(self.scratch.gdn_v.as_device_ptr(), rows)?,
                    a: DMat::contiguous(
                        self.scratch.gdn_a.as_device_ptr(),
                        rows,
                        QWEN36_GDN_NUM_V_HEADS,
                    )?,
                    b: DMat::contiguous(
                        self.scratch.gdn_b.as_device_ptr(),
                        rows,
                        QWEN36_GDN_NUM_V_HEADS,
                    )?,
                    a_log: DVec::contiguous(layer.a_log, QWEN36_GDN_NUM_V_HEADS)?,
                    dt_bias: DVec::contiguous(layer.dt_bias, QWEN36_GDN_NUM_V_HEADS)?,
                    state: recurrent_state,
                    seq_indptr: DVec::contiguous(self.scratch.gdn_seq_indptr.as_device_ptr(), 2)?,
                    state_indices: DVec::contiguous(
                        self.scratch.gdn_state_indices.as_device_ptr(),
                        1,
                    )?,
                    state_out_indices: Some(DVec::contiguous(
                        self.scratch.gdn_state_out_indices.as_device_ptr(),
                        1,
                    )?),
                    out: self.gdn_v_heads(self.scratch.gdn_recurrent_out.as_device_ptr(), rows)?,
                    batch_size: 1,
                    scale: qwen36_gdn_scale(),
                    use_qk_l2norm: true,
                    disable_state_update: false,
                })?;
                let mut ops = self.engine.operators();
                unsafe { ops.cuda().gdn_prefill_bf16(&prefill)? };
            }
            ActiveRunKind::Decode => {
                let decode = GdnDecodeBf16::new(GdnDecodeBf16Args {
                    q: self.gdn_q_heads(self.scratch.gdn_q.as_device_ptr(), rows)?,
                    k: self.gdn_k_heads(self.scratch.gdn_k.as_device_ptr(), rows)?,
                    v: self.gdn_v_heads(self.scratch.gdn_v.as_device_ptr(), rows)?,
                    a: DMat::contiguous(
                        self.scratch.gdn_a.as_device_ptr(),
                        rows,
                        QWEN36_GDN_NUM_V_HEADS,
                    )?,
                    b: DMat::contiguous(
                        self.scratch.gdn_b.as_device_ptr(),
                        rows,
                        QWEN36_GDN_NUM_V_HEADS,
                    )?,
                    a_log: DVec::contiguous(layer.a_log, QWEN36_GDN_NUM_V_HEADS)?,
                    dt_bias: DVec::contiguous(layer.dt_bias, QWEN36_GDN_NUM_V_HEADS)?,
                    state: recurrent_state,
                    state_indices: DVec::contiguous(
                        self.scratch.gdn_state_indices.as_device_ptr(),
                        rows,
                    )?,
                    state_out_indices: Some(DVec::contiguous(
                        self.scratch.gdn_state_out_indices.as_device_ptr(),
                        rows,
                    )?),
                    out: self.gdn_v_heads(self.scratch.gdn_recurrent_out.as_device_ptr(), rows)?,
                    scale: qwen36_gdn_scale(),
                    use_qk_l2norm: true,
                    disable_state_update: false,
                })?;
                let mut ops = self.engine.operators();
                unsafe { ops.cuda().gdn_decode_bf16(&decode)? };
            }
        }

        let gated = GdnRmsNormGatedBf16::new(GdnRmsNormGatedBf16Args {
            x: self.gdn_v_heads(self.scratch.gdn_recurrent_out.as_device_ptr(), rows)?,
            gate: self.gdn_v_heads(self.scratch.gdn_gate.as_device_ptr(), rows)?,
            weight: Bf16OrF32Vec::Bf16(DVec::contiguous(layer.rms_weight, QWEN36_GDN_VALUE_DIM)?),
            out: self.gdn_v_heads(self.scratch.gdn_norm_out.as_device_ptr(), rows)?,
            eps: self.config.rms_norm_eps,
            gate_activation: Activation::Silu,
        })?;
        {
            let mut ops = self.engine.operators();
            unsafe { ops.cuda().qwen36_gdn_rmsnorm_gated_bf16(&gated)? };
        }

        self.gemm_bf16(
            self.scratch.gdn_norm_out.as_device_ptr(),
            rows,
            QWEN36_GDN_OUTPUT_DIM,
            layer.out_proj,
            self.scratch.attn_proj.as_device_ptr(),
            hidden,
            GemmOut::Bf16,
        )?;
        Ok(())
    }
}
