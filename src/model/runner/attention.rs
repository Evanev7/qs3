use crate::model::*;

impl ModelRunner {
    pub(in crate::model) fn execute_attention_layer(
        &mut self,
        attention_layer_idx: u32,
        rows: u32,
        hidden: u32,
        q_hidden: u32,
        kv_hidden: u32,
        layer_input: ffi::DevicePtr,
        layer: QwenAttentionMlpPtrs,
        kind: ActiveRunKind,
    ) -> Result<(), Status> {
        self.gemm_bf16(
            layer_input,
            rows,
            hidden,
            layer.q_proj,
            self.scratch.q_proj_out.as_device_ptr(),
            QWEN36_FULL_ATTN_Q_PROJ_OUT,
            GemmOut::Bf16,
        )?;
        self.extract_attention_q_and_gate(rows)?;
        self.gemm_bf16(
            layer_input,
            rows,
            hidden,
            layer.k_proj,
            self.scratch.k.as_device_ptr(),
            kv_hidden,
            GemmOut::Bf16,
        )?;
        self.gemm_bf16(
            layer_input,
            rows,
            hidden,
            layer.v_proj,
            self.scratch.v.as_device_ptr(),
            kv_hidden,
            GemmOut::Bf16,
        )?;
        self.qwen_qk_norm_heads(
            self.scratch.q.as_device_ptr(),
            layer.q_norm,
            self.scratch.q.as_device_ptr(),
            rows,
            self.config.num_q_heads,
        )?;
        self.qwen_qk_norm_heads(
            self.scratch.k.as_device_ptr(),
            layer.k_norm,
            self.scratch.k.as_device_ptr(),
            rows,
            self.config.num_kv_heads,
        )?;
        self.apply_attention_rope(rows)?;

        let engine_layer = EngineLayer::bf16_attention(
            attention_layer_idx,
            self.attention_heads(
                self.scratch.q.as_device_ptr(),
                rows,
                self.config.num_q_heads,
            )?,
            self.attention_heads(
                self.scratch.k.as_device_ptr(),
                rows,
                self.config.num_kv_heads,
            )?,
            self.attention_heads(
                self.scratch.v.as_device_ptr(),
                rows,
                self.config.num_kv_heads,
            )?,
            self.attention_heads(
                self.scratch.attn_out.as_device_ptr(),
                rows,
                self.config.num_q_heads,
            )?,
            self.scratch.positions.as_device_ptr(),
        );
        unsafe {
            match kind {
                ActiveRunKind::Append => self.engine.append_layer(&engine_layer)?,
                ActiveRunKind::Decode => self.engine.decode_layer(&engine_layer)?,
            }
        }
        self.apply_attention_output_gate(rows, q_hidden)?;

        self.gemm_bf16(
            self.scratch.attn_out.as_device_ptr(),
            rows,
            q_hidden,
            layer.o_proj,
            self.scratch.attn_proj.as_device_ptr(),
            hidden,
            GemmOut::Bf16,
        )
    }

    pub(in crate::model) fn extract_attention_q_and_gate(
        &mut self,
        rows: u32,
    ) -> Result<(), Status> {
        if self.config.num_q_heads != QWEN36_FULL_ATTN_Q_HEADS
            || self.config.head_dim != QWEN36_FULL_ATTN_HEAD_DIM
            || self.config.q_hidden_size()? != QWEN36_FULL_ATTN_Q_HIDDEN
        {
            return Err(Status::Unsupported);
        }
        unsafe {
            extract_qwen36_packed_attention_q_and_gate_bf16(
                self.scratch.q_proj_out.as_device_ptr(),
                self.scratch.q.as_device_ptr(),
                self.scratch.attn_gate.as_device_ptr(),
                rows,
                self.config.stream,
            )
        }
    }

    pub(in crate::model) fn qwen_qk_norm_heads(
        &mut self,
        x: ffi::DevicePtr,
        weight: ffi::DevicePtr,
        out: ffi::DevicePtr,
        rows: u32,
        heads: u32,
    ) -> Result<(), Status> {
        if heads == 0 || self.config.head_dim != QWEN36_FULL_ATTN_HEAD_DIM {
            return Err(Status::InvalidArgument);
        }
        let norm_rows = rows.checked_mul(heads).ok_or(Status::InvalidArgument)?;
        let desc = RmsNormBf16::qwen_qk_norm(
            DMat::contiguous(x, norm_rows, self.config.head_dim)?,
            DVec::contiguous(weight, self.config.head_dim)?,
            DMat::contiguous(out, norm_rows, self.config.head_dim)?,
            self.config.rms_norm_eps,
        )?;
        let mut ops = self.engine.operators();
        unsafe { ops.flashinfer.rmsnorm_bf16(&desc) }
    }

    pub(in crate::model) fn apply_attention_rope(&mut self, rows: u32) -> Result<(), Status> {
        let q = self.attention_heads(
            self.scratch.q.as_device_ptr(),
            rows,
            self.config.num_q_heads,
        )?;
        let k = self.attention_heads(
            self.scratch.k.as_device_ptr(),
            rows,
            self.config.num_kv_heads,
        )?;
        let desc = RopeApplyBf16::with_params(
            q,
            k,
            q,
            k,
            DVec::contiguous(self.scratch.positions.as_device_ptr(), rows)?,
            QWEN36_FULL_ATTN_ROTARY_DIM,
            self.config.rope_scale,
            self.config.rope_theta,
        )?;
        let mut ops = self.engine.operators();
        unsafe { ops.flashinfer.rope_apply_bf16(&desc) }
    }

    pub(in crate::model) fn apply_attention_output_gate(
        &mut self,
        rows: u32,
        q_hidden: u32,
    ) -> Result<(), Status> {
        if q_hidden != QWEN36_FULL_ATTN_Q_HIDDEN {
            return Err(Status::Unsupported);
        }
        let desc = Qwen36FullAttentionOutputGateBf16::new(
            DMat::contiguous(self.scratch.attn_gate.as_device_ptr(), rows, q_hidden)?,
            DMat::contiguous(self.scratch.attn_out.as_device_ptr(), rows, q_hidden)?,
        )?;
        let mut ops = self.engine.operators();
        unsafe { ops.cuda.qwen36_full_attention_output_gate_bf16(&desc) }
    }
}
