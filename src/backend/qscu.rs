use super::{
    BF16, Bf16Heads, DMat, DVec, F32, FloatDType, GdnStateIndexPolicy, I32, Qsfi,
    require_gdn_state_index_vec,
    require_i32_vec, require_qwen36_gdn_heads, result_from_raw, validate_eps,
    validate_gdn_recurrent_tensors, validate_nonzero, validate_soft_cap, zero_tensor1,
    zero_tensor2,
};
use crate::{
    QWEN36_FULL_ATTN_Q_HIDDEN, QWEN36_GDN_CONV_WIDTH, QWEN36_GDN_KEY_DIM, QWEN36_GDN_NUM_K_HEADS,
    QWEN36_GDN_NUM_Q_HEADS, QWEN36_GDN_NUM_V_HEADS, QWEN36_GDN_PACKED_DIM, QWEN36_GDN_VALUE_DIM,
    QWEN36_MOE_MAX_EXPERTS, QWEN36_MOE_MAX_TOP_K, Status,
    ffi::{self, sys},
};

use std::ptr;

pub(crate) use super::{GdnConvState, GdnRecurrentState, RouterScore};

/// Typed access to qscu's stream-ordered CUDA kernels.
pub(crate) struct Qscu<'a> {
    stream: &'a ffi::CudaStream,
    qsfi: &'a mut Qsfi,
}

impl<'a> Qscu<'a> {
    pub(super) fn new(stream: &'a ffi::CudaStream, qsfi: &'a mut Qsfi) -> Self {
        Self { stream, qsfi }
    }

    pub(crate) unsafe fn embedding_gather_bf16(
        &mut self,
        token_ids: DVec<I32>,
        embedding: DMat<BF16>,
        out: DMat<BF16>,
        padding_token_id: Option<i32>,
        validate_token_ids: bool,
    ) -> Result<(), Status> {
        let desc = embedding_gather_desc(
            token_ids,
            embedding,
            out,
            padding_token_id,
            validate_token_ids,
        )?;
        result_from_raw(unsafe { sys::qscu_embedding_gather_bf16(&desc, *self.stream) })
    }

    pub(crate) unsafe fn silu_and_mul_bf16(
        &mut self,
        gate: DMat<BF16>,
        up: DMat<BF16>,
        out: DMat<BF16>,
    ) -> Result<(), Status> {
        let desc = silu_and_mul_desc(gate, up, out)?;
        result_from_raw(unsafe { sys::qscu_silu_and_mul_bf16(&desc, *self.stream) })
    }

    pub(crate) unsafe fn qwen36_shared_expert_gate_add_bf16(
        &mut self,
        gate_logits: DMat<F32>,
        shared: DMat<BF16>,
        out: DMat<BF16>,
    ) -> Result<(), Status> {
        let desc = qwen36_shared_expert_gate_add_desc(gate_logits, shared, out)?;
        result_from_raw(unsafe {
            sys::qscu_qwen36_shared_expert_gate_add_bf16(&desc, *self.stream)
        })
    }

    pub(crate) unsafe fn qwen36_full_attention_output_gate_bf16(
        &mut self,
        gate: DMat<BF16>,
        out: DMat<BF16>,
    ) -> Result<(), Status> {
        let desc = qwen36_full_attention_output_gate_desc(gate, out)?;
        result_from_raw(unsafe {
            sys::qscu_qwen36_full_attention_output_gate_bf16(&desc, *self.stream)
        })
    }

    pub(crate) unsafe fn logits_soft_cap_f32(
        &mut self,
        logits: DMat<F32>,
        soft_cap: f32,
    ) -> Result<(), Status> {
        validate_logits_soft_cap(logits, soft_cap)?;
        result_from_raw(unsafe {
            sys::qscu_logits_soft_cap_f32(
                &logits.tensor(),
                logits.rows,
                logits.cols,
                soft_cap,
                *self.stream,
            )
        })
    }

    pub(crate) unsafe fn greedy_argmax_f32(
        &mut self,
        logits: DMat<F32>,
        next_token_ids: DVec<I32>,
    ) -> Result<(), Status> {
        let desc = greedy_argmax_desc(logits, next_token_ids)?;
        result_from_raw(unsafe { sys::qscu_greedy_argmax_f32(&desc, *self.stream) })
    }

    pub(crate) unsafe fn router_topk<T: FloatDType>(
        &mut self,
        logits: DMat<T>,
        topk_ids: DMat<I32>,
        topk_weights: DMat<F32>,
        score: RouterScore,
        renormalize: bool,
        routed_scaling_factor: f32,
    ) -> Result<(), Status> {
        let desc = router_topk_desc(
            logits,
            topk_ids,
            topk_weights,
            score,
            renormalize,
            routed_scaling_factor,
        )?;
        result_from_raw(unsafe { sys::qscu_router_topk(&desc, *self.stream) })
    }

    pub(crate) unsafe fn qwen36_gdn_causal_conv1d_bf16(
        &mut self,
        x: DMat<BF16>,
        weight: DMat<BF16>,
        bias: DVec<BF16>,
        state: GdnConvState,
        state_read_indices: Option<DVec<I32>>,
        state_write_indices: Option<DVec<I32>>,
        seq_indptr: Option<DVec<I32>>,
        out: DMat<BF16>,
        batch_size: u32,
    ) -> Result<(), Status> {
        let desc = qwen36_gdn_causal_conv1d_desc(
            x,
            weight,
            bias,
            state,
            state_read_indices,
            state_write_indices,
            seq_indptr,
            out,
            batch_size,
        )?;
        result_from_raw(unsafe { sys::qscu_qwen36_gdn_causal_conv1d_bf16(&desc, *self.stream) })
    }

    pub(crate) unsafe fn qwen36_gdn_post_conv_prepare_bf16(
        &mut self,
        conv_out: DMat<BF16>,
        a: DMat<BF16>,
        b: DMat<BF16>,
        a_log: DVec<BF16>,
        dt_bias: DVec<BF16>,
        q: Bf16Heads,
        k: Bf16Heads,
        v: Bf16Heads,
    ) -> Result<(), Status> {
        let desc = qwen36_gdn_post_conv_prepare_desc(conv_out, a, b, a_log, dt_bias, q, k, v)?;
        result_from_raw(unsafe { sys::qscu_qwen36_gdn_post_conv_prepare_bf16(&desc, *self.stream) })
    }

    pub(crate) unsafe fn qwen36_gdn_gated_rmsnorm_bf16(
        &mut self,
        x: Bf16Heads,
        gate: Bf16Heads,
        weight: DVec<BF16>,
        out: Bf16Heads,
        eps: f32,
    ) -> Result<(), Status> {
        let desc = qwen36_gdn_gated_rmsnorm_desc(x, gate, weight, out, eps)?;
        result_from_raw(unsafe { sys::qscu_qwen36_gdn_rmsnorm_gated_bf16(&desc, *self.stream) })
    }

    pub(crate) unsafe fn qwen36_gdn_decode_bf16(
        &mut self,
        q: Bf16Heads,
        k: Bf16Heads,
        v: Bf16Heads,
        a: DMat<BF16>,
        b: DMat<BF16>,
        a_log: DVec<BF16>,
        dt_bias: DVec<BF16>,
        state: GdnRecurrentState,
        state_indices: DVec<I32>,
        state_out_indices: Option<DVec<I32>>,
        out: Bf16Heads,
    ) -> Result<(), Status> {
        let desc = qwen36_gdn_decode_desc(
            q,
            k,
            v,
            a,
            b,
            a_log,
            dt_bias,
            state,
            state_indices,
            state_out_indices,
            out,
        )?;
        result_from_raw(unsafe { sys::qscu_gdn_decode(self.qsfi.as_raw(), &desc) })
            .inspect_err(|_| _ = self.qsfi.last_error())
    }

    pub(crate) unsafe fn qwen36_gdn_prefill_bf16(
        &mut self,
        q: Bf16Heads,
        k: Bf16Heads,
        v: Bf16Heads,
        a: DMat<BF16>,
        b: DMat<BF16>,
        a_log: DVec<BF16>,
        dt_bias: DVec<BF16>,
        state: GdnRecurrentState,
        seq_indptr: DVec<I32>,
        state_indices: DVec<I32>,
        state_out_indices: Option<DVec<I32>>,
        out: Bf16Heads,
        batch_size: u32,
    ) -> Result<(), Status> {
        let desc = qwen36_gdn_prefill_desc(
            q,
            k,
            v,
            a,
            b,
            a_log,
            dt_bias,
            state,
            seq_indptr,
            state_indices,
            state_out_indices,
            out,
            batch_size,
        )?;
        result_from_raw(unsafe { sys::qscu_gdn_prefill(self.qsfi.as_raw(), &desc) })
            .inspect_err(|_| _ = self.qsfi.last_error())
    }
}

pub(super) fn qwen36_gdn_gated_rmsnorm_desc(
    x: Bf16Heads,
    gate: Bf16Heads,
    weight: DVec<BF16>,
    out: Bf16Heads,
    eps: f32,
) -> Result<sys::qscu_qwen36_gdn_rmsnorm_gated_desc, Status> {
    validate_eps(eps)?;
    x.require_contiguous()?;
    gate.require_contiguous()?;
    weight.require_contiguous()?;
    out.require_contiguous()?;
    if weight.len != QWEN36_GDN_VALUE_DIM {
        return Err(Status::InvalidArgument);
    }
    let tokens = x.tokens;
    require_qwen36_gdn_heads(x, QWEN36_GDN_NUM_V_HEADS, tokens)?;
    require_qwen36_gdn_heads(gate, QWEN36_GDN_NUM_V_HEADS, tokens)?;
    require_qwen36_gdn_heads(out, QWEN36_GDN_NUM_V_HEADS, tokens)?;
    Ok(sys::qscu_qwen36_gdn_rmsnorm_gated_desc {
        x: x.tensor(),
        gate: gate.tensor(),
        weight: weight.tensor(),
        out: out.tensor(),
        num_tokens: tokens,
        eps,
        gate_activation: sys::QSCU_ACTIVATION_SILU,
    })
}

pub(super) fn embedding_gather_desc(
    token_ids: DVec<I32>,
    embedding: DMat<BF16>,
    out: DMat<BF16>,
    padding_token_id: Option<i32>,
    validate_token_ids: bool,
) -> Result<sys::qscu_embedding_gather_desc, Status> {
    token_ids.require_contiguous()?;
    embedding.require_contiguous()?;
    out.require_contiguous()?;
    if out.rows != token_ids.len || out.cols != embedding.cols {
        return Err(Status::InvalidArgument);
    }
    Ok(sys::qscu_embedding_gather_desc {
        token_ids: token_ids.tensor(),
        embedding: embedding.tensor(),
        out: out.tensor(),
        padding_token_id: padding_token_id.unwrap_or(-1),
        validate_token_ids: u32::from(validate_token_ids),
    })
}

pub(super) fn silu_and_mul_desc(
    gate: DMat<BF16>,
    up: DMat<BF16>,
    out: DMat<BF16>,
) -> Result<sys::qscu_silu_and_mul_desc, Status> {
    gate.require_contiguous()?;
    up.require_contiguous()?;
    out.require_contiguous()?;
    if !gate.same_shape(up) || !gate.same_shape(out) {
        return Err(Status::InvalidArgument);
    }
    Ok(sys::qscu_silu_and_mul_desc {
        gate: gate.tensor(),
        up: up.tensor(),
        out: out.tensor(),
        num_tokens: gate.rows,
        intermediate_size: gate.cols,
    })
}

pub(super) fn qwen36_shared_expert_gate_add_desc(
    gate_logits: DMat<F32>,
    shared: DMat<BF16>,
    out: DMat<BF16>,
) -> Result<sys::qscu_qwen36_shared_expert_gate_add_desc, Status> {
    if !gate_logits.is_contiguous() || !shared.is_contiguous() || !out.is_contiguous() {
        return Err(Status::InvalidArgument);
    }
    if gate_logits.rows != shared.rows || gate_logits.cols != 1 || !shared.same_shape(out) {
        return Err(Status::InvalidArgument);
    }
    Ok(sys::qscu_qwen36_shared_expert_gate_add_desc {
        gate_logits: gate_logits.tensor(),
        shared: shared.tensor(),
        out: out.tensor(),
        num_tokens: shared.rows,
        hidden_size: shared.cols,
    })
}

pub(super) fn qwen36_full_attention_output_gate_desc(
    gate: DMat<BF16>,
    out: DMat<BF16>,
) -> Result<sys::qscu_qwen36_full_attention_output_gate_desc, Status> {
    gate.require_contiguous()?;
    out.require_contiguous()?;
    if !gate.same_shape(out) || gate.cols != QWEN36_FULL_ATTN_Q_HIDDEN {
        return Err(Status::InvalidArgument);
    }
    Ok(sys::qscu_qwen36_full_attention_output_gate_desc {
        gate: gate.tensor(),
        out: out.tensor(),
        num_tokens: gate.rows,
        q_hidden: gate.cols,
    })
}

pub(super) fn validate_logits_soft_cap(logits: DMat<F32>, soft_cap: f32) -> Result<(), Status> {
    logits.require_contiguous()?;
    validate_soft_cap(soft_cap)
}

pub(super) fn greedy_argmax_desc(
    logits: DMat<F32>,
    next_token_ids: DVec<I32>,
) -> Result<sys::qscu_sampling_desc, Status> {
    logits.require_contiguous()?;
    next_token_ids.require_contiguous()?;
    if next_token_ids.len != logits.rows {
        return Err(Status::InvalidArgument);
    }
    if logits.cols > i32::MAX as u32 {
        return Err(Status::Unsupported);
    }
    Ok(sys::qscu_sampling_desc {
        logits: logits.tensor(),
        uniform_samples: zero_tensor1(ffi::DTYPE_F32),
        next_token_ids: next_token_ids.tensor(),
        selected_logprobs: zero_tensor1(ffi::DTYPE_F32),
        selected_probs: zero_tensor1(ffi::DTYPE_F32),
        batch_size: logits.rows,
        vocab_size: logits.cols,
        top_k: 0,
        top_p: 0.0,
        min_p: 0.0,
        temperature: 0.0,
    })
}

pub(super) fn router_topk_desc<T: FloatDType>(
    logits: DMat<T>,
    topk_ids: DMat<I32>,
    topk_weights: DMat<F32>,
    score: RouterScore,
    renormalize: bool,
    routed_scaling_factor: f32,
) -> Result<sys::qscu_router_topk_desc, Status> {
    if !logits.is_contiguous() || !topk_ids.is_contiguous() || !topk_weights.is_contiguous() {
        return Err(Status::InvalidArgument);
    }
    if logits.rows == 0
        || logits.cols == 0
        || topk_ids.rows != logits.rows
        || topk_weights.rows != logits.rows
        || topk_ids.cols != topk_weights.cols
        || topk_ids.cols == 0
    {
        return Err(Status::InvalidArgument);
    }
    if topk_ids.cols > QWEN36_MOE_MAX_TOP_K
        || logits.cols > QWEN36_MOE_MAX_EXPERTS
        || topk_ids.cols > logits.cols
    {
        return Err(Status::Unsupported);
    }
    if !routed_scaling_factor.is_finite() || routed_scaling_factor <= 0.0 {
        return Err(Status::InvalidArgument);
    }
    Ok(sys::qscu_router_topk_desc {
        logits: logits.tensor(),
        topk_ids: topk_ids.tensor(),
        topk_weights: topk_weights.tensor(),
        num_tokens: logits.rows,
        num_experts: logits.cols,
        top_k: topk_ids.cols,
        score: score.raw(),
        renormalize: u32::from(renormalize),
        routed_scaling_factor,
    })
}

pub(super) fn qwen36_gdn_causal_conv1d_desc(
    x: DMat<BF16>,
    weight: DMat<BF16>,
    bias: DVec<BF16>,
    state: GdnConvState,
    state_read_indices: Option<DVec<I32>>,
    state_write_indices: Option<DVec<I32>>,
    seq_indptr: Option<DVec<I32>>,
    out: DMat<BF16>,
    batch_size: u32,
) -> Result<sys::qscu_qwen36_gdn_causal_conv1d_desc, Status> {
    validate_nonzero(&[batch_size])?;
    x.require_contiguous()?;
    weight.require_contiguous()?;
    out.require_contiguous()?;
    if x.rows == 0
        || x.cols != QWEN36_GDN_PACKED_DIM
        || weight.rows != QWEN36_GDN_PACKED_DIM
        || weight.cols != QWEN36_GDN_CONV_WIDTH
        || out.rows != x.rows
        || out.cols != QWEN36_GDN_PACKED_DIM
    {
        return Err(Status::InvalidArgument);
    }
    bias.require_contiguous()?;
    if bias.len != QWEN36_GDN_PACKED_DIM {
        return Err(Status::InvalidArgument);
    }
    if let Some(indices) = state_read_indices {
        require_i32_vec(indices, batch_size)?;
    }
    if let Some(indices) = state_write_indices {
        require_i32_vec(indices, batch_size)?;
    }
    if state_read_indices.is_none() && state_write_indices.is_none() {
        return Err(Status::InvalidArgument);
    }
    let seq_indptr = if let Some(seq_indptr) = seq_indptr {
        require_i32_vec(
            seq_indptr,
            batch_size.checked_add(1).ok_or(Status::InvalidArgument)?,
        )?;
        seq_indptr.data
    } else {
        if batch_size != x.rows {
            return Err(Status::InvalidArgument);
        }
        ptr::null_mut()
    };
    Ok(sys::qscu_qwen36_gdn_causal_conv1d_desc {
        x: x.tensor(),
        weight: weight.tensor(),
        bias: bias.tensor(),
        state: state.tensor(),
        state_read_indices: state_read_indices.map_or(zero_tensor1(ffi::DTYPE_I32), DVec::tensor),
        state_write_indices: state_write_indices.map_or(zero_tensor1(ffi::DTYPE_I32), DVec::tensor),
        seq_indptr,
        out: out.tensor(),
        num_tokens: x.rows,
        batch_size,
        activation: sys::QSCU_ACTIVATION_SILU,
        update_state: 1,
    })
}

pub(super) fn qwen36_gdn_post_conv_prepare_desc(
    conv_out: DMat<BF16>,
    a: DMat<BF16>,
    b: DMat<BF16>,
    a_log: DVec<BF16>,
    dt_bias: DVec<BF16>,
    q: Bf16Heads,
    k: Bf16Heads,
    v: Bf16Heads,
) -> Result<sys::qscu_qwen36_gdn_post_conv_prepare_desc, Status> {
    conv_out.require_contiguous()?;
    a.require_contiguous()?;
    b.require_contiguous()?;
    a_log.require_contiguous()?;
    dt_bias.require_contiguous()?;
    q.require_contiguous()?;
    k.require_contiguous()?;
    v.require_contiguous()?;
    let tokens = conv_out.rows;
    if conv_out.cols != QWEN36_GDN_PACKED_DIM
        || a.rows != tokens
        || a.cols != QWEN36_GDN_NUM_V_HEADS
        || b.rows != tokens
        || b.cols != QWEN36_GDN_NUM_V_HEADS
        || a_log.len != QWEN36_GDN_NUM_V_HEADS
        || dt_bias.len != QWEN36_GDN_NUM_V_HEADS
    {
        return Err(Status::InvalidArgument);
    }
    require_qwen36_gdn_heads(q, QWEN36_GDN_NUM_Q_HEADS, tokens)?;
    require_qwen36_gdn_heads(k, QWEN36_GDN_NUM_K_HEADS, tokens)?;
    require_qwen36_gdn_heads(v, QWEN36_GDN_NUM_V_HEADS, tokens)?;
    Ok(sys::qscu_qwen36_gdn_post_conv_prepare_desc {
        conv_out: conv_out.tensor(),
        a: a.tensor(),
        b: b.tensor(),
        a_log: a_log.tensor(),
        dt_bias: dt_bias.tensor(),
        q: q.tensor(),
        k: k.tensor(),
        v: v.tensor(),
        g_out: zero_tensor2(ffi::DTYPE_F32),
        beta_out: zero_tensor2(ffi::DTYPE_F32),
        num_tokens: tokens,
        apply_qk_l2norm: 0,
        l2norm_eps: 0.0,
        forget_gate_output: sys::QSCU_GDN_FORGET_LOG_DECAY,
    })
}

pub(super) fn qwen36_gdn_decode_desc(
    q: Bf16Heads,
    k: Bf16Heads,
    v: Bf16Heads,
    a: DMat<BF16>,
    b: DMat<BF16>,
    a_log: DVec<BF16>,
    dt_bias: DVec<BF16>,
    state: GdnRecurrentState,
    state_indices: DVec<I32>,
    state_out_indices: Option<DVec<I32>>,
    out: Bf16Heads,
) -> Result<sys::qscu_gdn_decode_desc, Status> {
    let tokens = q.tokens;
    validate_gdn_recurrent_tensors(q, k, v, a, b, a_log, dt_bias, out, tokens)?;
    require_gdn_state_index_vec(state_indices, tokens, GdnStateIndexPolicy::NegativeSkips)?;
    if let Some(indices) = state_out_indices {
        require_gdn_state_index_vec(indices, tokens, GdnStateIndexPolicy::NegativeSkips)?;
    }
    Ok(sys::qscu_gdn_decode_desc {
        q: q.tensor(),
        k: k.tensor(),
        v: v.tensor(),
        a: a.tensor(),
        b: b.tensor(),
        a_log: a_log.tensor(),
        dt_bias: dt_bias.tensor(),
        state: state.tensor(),
        state_indices: state_indices.tensor(),
        state_out_indices: state_out_indices.map_or(zero_tensor1(ffi::DTYPE_I32), DVec::tensor),
        out: out.tensor(),
        num_tokens: tokens,
        num_q_heads: QWEN36_GDN_NUM_Q_HEADS,
        num_k_heads: QWEN36_GDN_NUM_K_HEADS,
        num_v_heads: QWEN36_GDN_NUM_V_HEADS,
        key_dim: QWEN36_GDN_KEY_DIM,
        value_dim: QWEN36_GDN_VALUE_DIM,
        state_layout: sys::QSCU_GDN_STATE_LAYOUT_VK,
        scale: qwen36_gdn_scale(),
        use_qk_l2norm: 1,
        disable_state_update: 0,
    })
}

pub(super) fn qwen36_gdn_prefill_desc(
    q: Bf16Heads,
    k: Bf16Heads,
    v: Bf16Heads,
    a: DMat<BF16>,
    b: DMat<BF16>,
    a_log: DVec<BF16>,
    dt_bias: DVec<BF16>,
    state: GdnRecurrentState,
    seq_indptr: DVec<I32>,
    state_indices: DVec<I32>,
    state_out_indices: Option<DVec<I32>>,
    out: Bf16Heads,
    batch_size: u32,
) -> Result<sys::qscu_gdn_prefill_desc, Status> {
    validate_nonzero(&[batch_size])?;
    let total_tokens = q.tokens;
    validate_gdn_recurrent_tensors(q, k, v, a, b, a_log, dt_bias, out, total_tokens)?;
    require_i32_vec(
        seq_indptr,
        batch_size.checked_add(1).ok_or(Status::InvalidArgument)?,
    )?;
    require_gdn_state_index_vec(
        state_indices,
        batch_size,
        GdnStateIndexPolicy::NegativeSkips,
    )?;
    if let Some(indices) = state_out_indices {
        require_gdn_state_index_vec(indices, batch_size, GdnStateIndexPolicy::NegativeSkips)?;
    }
    Ok(sys::qscu_gdn_prefill_desc {
        q: q.tensor(),
        k: k.tensor(),
        v: v.tensor(),
        a: a.tensor(),
        b: b.tensor(),
        a_log: a_log.tensor(),
        dt_bias: dt_bias.tensor(),
        state: state.tensor(),
        seq_indptr: seq_indptr.data,
        state_indices: state_indices.tensor(),
        state_out_indices: state_out_indices.map_or(zero_tensor1(ffi::DTYPE_I32), DVec::tensor),
        out: out.tensor(),
        batch_size,
        total_tokens,
        num_q_heads: QWEN36_GDN_NUM_Q_HEADS,
        num_k_heads: QWEN36_GDN_NUM_K_HEADS,
        num_v_heads: QWEN36_GDN_NUM_V_HEADS,
        key_dim: QWEN36_GDN_KEY_DIM,
        value_dim: QWEN36_GDN_VALUE_DIM,
        state_layout: sys::QSCU_GDN_STATE_LAYOUT_VK,
        scale: qwen36_gdn_scale(),
        use_qk_l2norm: 1,
        disable_state_update: 0,
    })
}

fn qwen36_gdn_scale() -> f32 {
    1.0 / (QWEN36_GDN_KEY_DIM as f32).sqrt()
}
