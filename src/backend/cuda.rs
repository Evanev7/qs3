use super::*;

pub(crate) use super::{
    Activation, GdnConvState, GdnForgetGateOutput, GdnRecurrentState, RouterScore,
};

/// Thin typed access to qscu's stream-ordered CUDA kernels.
pub(crate) struct Cuda<'a> {
    stream: &'a ffi::CudaStream,
    context: &'a mut qsfi::Context,
}

impl<'a> Cuda<'a> {
    pub(super) fn new(stream: &'a ffi::CudaStream, context: &'a mut qsfi::Context) -> Self {
        Self { stream, context }
    }

    pub(crate) unsafe fn embedding_gather_bf16(
        &mut self,
        desc: &EmbeddingGatherBf16,
    ) -> Result<(), Status> {
        unsafe { qscu::embedding_gather_bf16(&desc.raw, *self.stream) }
    }

    pub(crate) unsafe fn silu_and_mul_bf16(&mut self, desc: &SiluAndMulBf16) -> Result<(), Status> {
        unsafe { qscu::silu_and_mul_bf16(&desc.raw, *self.stream) }
    }

    pub(crate) unsafe fn qwen36_shared_expert_gate_add_bf16(
        &mut self,
        desc: &Qwen36SharedExpertGateAddBf16,
    ) -> Result<(), Status> {
        unsafe { qscu::qwen36_shared_expert_gate_add_bf16(&desc.raw, *self.stream) }
    }

    pub(crate) unsafe fn qwen36_full_attention_output_gate_bf16(
        &mut self,
        desc: &Qwen36FullAttentionOutputGateBf16,
    ) -> Result<(), Status> {
        unsafe { qscu::qwen36_full_attention_output_gate_bf16(&desc.raw, *self.stream) }
    }

    pub(crate) unsafe fn logits_soft_cap_f32(
        &mut self,
        desc: &LogitsSoftCapF32,
    ) -> Result<(), Status> {
        unsafe {
            qscu::logits_soft_cap_f32(
                &desc.logits,
                desc.rows,
                desc.vocab_size,
                desc.soft_cap,
                *self.stream,
            )
        }
    }

    pub(crate) unsafe fn greedy_argmax_f32(
        &mut self,
        desc: &GreedyArgmaxF32,
    ) -> Result<(), Status> {
        unsafe { qscu::greedy_argmax_f32(&desc.raw, *self.stream) }
    }

    pub(crate) unsafe fn router_topk(&mut self, desc: &RouterTopK) -> Result<(), Status> {
        unsafe { qscu::router_topk(&desc.raw, *self.stream) }
    }

    pub(crate) unsafe fn qwen36_gdn_causal_conv1d_bf16(
        &mut self,
        desc: &GdnCausalConv1dBf16,
    ) -> Result<(), Status> {
        unsafe { qscu::qwen36_gdn_causal_conv1d_bf16(&desc.raw, *self.stream) }
    }

    pub(crate) unsafe fn qwen36_gdn_post_conv_prepare_bf16(
        &mut self,
        desc: &GdnPostConvPrepareBf16,
    ) -> Result<(), Status> {
        unsafe { qscu::qwen36_gdn_post_conv_prepare_bf16(&desc.raw, *self.stream) }
    }

    pub(crate) unsafe fn qwen36_gdn_rmsnorm_gated_bf16(
        &mut self,
        desc: &GdnRmsNormGatedBf16,
    ) -> Result<(), Status> {
        unsafe { qscu::qwen36_gdn_rmsnorm_gated_bf16(&desc.raw, *self.stream) }
    }

    pub(crate) unsafe fn gdn_decode_bf16(&mut self, desc: &GdnDecodeBf16) -> Result<(), Status> {
        unsafe { qscu::gdn_decode(self.context, &desc.raw) }
    }

    pub(crate) unsafe fn gdn_prefill_bf16(&mut self, desc: &GdnPrefillBf16) -> Result<(), Status> {
        unsafe { qscu::gdn_prefill(self.context, &desc.raw) }
    }
}

#[derive(Clone, Copy, Debug)]
pub(crate) struct EmbeddingGatherBf16 {
    pub(super) raw: qscu::EmbeddingGatherDesc,
}

impl EmbeddingGatherBf16 {
    pub(crate) fn new(
        token_ids: DVec<I32>,
        embedding: DMat<BF16>,
        out: DMat<BF16>,
    ) -> Result<Self, Status> {
        Self::with_options(token_ids, embedding, out, None, false)
    }

    pub(crate) fn with_options(
        token_ids: DVec<I32>,
        embedding: DMat<BF16>,
        out: DMat<BF16>,
        padding_token_id: Option<i32>,
        validate_token_ids: bool,
    ) -> Result<Self, Status> {
        token_ids.require_contiguous()?;
        embedding.require_contiguous()?;
        out.require_contiguous()?;
        if out.rows != token_ids.len || out.cols != embedding.cols {
            return Err(Status::InvalidArgument);
        }
        Ok(Self {
            raw: qscu::EmbeddingGatherDesc {
                token_ids: token_ids.tensor(),
                embedding: embedding.tensor(),
                out: out.tensor(),
                padding_token_id: padding_token_id.unwrap_or(-1),
                validate_token_ids: u32::from(validate_token_ids),
            },
        })
    }
}

#[derive(Clone, Copy, Debug)]
pub(crate) struct SiluAndMulBf16 {
    pub(super) raw: qscu::SiluAndMulDesc,
}

impl SiluAndMulBf16 {
    pub(crate) fn new(gate: DMat<BF16>, up: DMat<BF16>, out: DMat<BF16>) -> Result<Self, Status> {
        gate.require_contiguous()?;
        up.require_contiguous()?;
        out.require_contiguous()?;
        if !gate.same_shape(up) || !gate.same_shape(out) {
            return Err(Status::InvalidArgument);
        }
        Ok(Self {
            raw: qscu::SiluAndMulDesc {
                gate: gate.tensor(),
                up: up.tensor(),
                out: out.tensor(),
                num_tokens: gate.rows,
                intermediate_size: gate.cols,
            },
        })
    }
}

#[derive(Clone, Copy, Debug)]
pub(crate) struct Qwen36SharedExpertGateAddBf16 {
    pub(super) raw: qscu::Qwen36SharedExpertGateAddDesc,
}

impl Qwen36SharedExpertGateAddBf16 {
    pub(crate) fn new(
        gate_logits: Bf16OrF32Mat,
        shared: DMat<BF16>,
        out: DMat<BF16>,
    ) -> Result<Self, Status> {
        if !gate_logits.is_contiguous() || !shared.is_contiguous() || !out.is_contiguous() {
            return Err(Status::InvalidArgument);
        }
        if gate_logits.rows() != shared.rows || gate_logits.cols() != 1 || !shared.same_shape(out) {
            return Err(Status::InvalidArgument);
        }
        Ok(Self {
            raw: qscu::Qwen36SharedExpertGateAddDesc {
                gate_logits: gate_logits.tensor(),
                shared: shared.tensor(),
                out: out.tensor(),
                num_tokens: shared.rows,
                hidden_size: shared.cols,
            },
        })
    }
}

#[derive(Clone, Copy, Debug)]
pub(crate) struct Qwen36FullAttentionOutputGateBf16 {
    pub(super) raw: qscu::Qwen36FullAttentionOutputGateDesc,
}

impl Qwen36FullAttentionOutputGateBf16 {
    // `gate` is already extracted from q_proj's interleaved per-head [q, gate]
    // output. q_norm/RoPE are upstream of this in-place attention-output gate.
    pub(crate) fn new(gate: DMat<BF16>, out: DMat<BF16>) -> Result<Self, Status> {
        gate.require_contiguous()?;
        out.require_contiguous()?;
        if !gate.same_shape(out) || gate.cols != QWEN36_FULL_ATTN_Q_HIDDEN {
            return Err(Status::InvalidArgument);
        }
        Ok(Self {
            raw: qscu::Qwen36FullAttentionOutputGateDesc {
                gate: gate.tensor(),
                out: out.tensor(),
                num_tokens: gate.rows,
                q_hidden: gate.cols,
            },
        })
    }
}

#[derive(Clone, Copy, Debug)]
pub(crate) struct LogitsSoftCapF32 {
    logits: ffi::Tensor2,
    rows: u32,
    vocab_size: u32,
    soft_cap: f32,
}

impl LogitsSoftCapF32 {
    pub(crate) fn new(logits: DMat<F32>, soft_cap: f32) -> Result<Self, Status> {
        logits.require_contiguous()?;
        validate_soft_cap(soft_cap)?;
        Ok(Self {
            logits: logits.tensor(),
            rows: logits.rows,
            vocab_size: logits.cols,
            soft_cap,
        })
    }
}

#[derive(Clone, Copy, Debug)]
pub(crate) struct GreedyArgmaxF32 {
    pub(super) raw: qscu::SamplingDesc,
}

impl GreedyArgmaxF32 {
    pub(crate) fn new(logits: DMat<F32>, next_token_ids: DVec<I32>) -> Result<Self, Status> {
        logits.require_contiguous()?;
        next_token_ids.require_contiguous()?;
        if next_token_ids.len != logits.rows {
            return Err(Status::InvalidArgument);
        }
        if logits.cols > i32::MAX as u32 {
            return Err(Status::Unsupported);
        }
        Ok(Self {
            raw: qscu::SamplingDesc {
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
            },
        })
    }
}

#[derive(Clone, Copy, Debug)]
pub(crate) struct RouterTopK {
    pub(super) raw: qscu::RouterTopkDesc,
}

impl RouterTopK {
    pub(crate) fn new(
        logits: Bf16OrF32Mat,
        topk_ids: DMat<I32>,
        topk_weights: DMat<F32>,
        score: RouterScore,
        renormalize: bool,
        routed_scaling_factor: f32,
    ) -> Result<Self, Status> {
        if !logits.is_contiguous() || !topk_ids.is_contiguous() || !topk_weights.is_contiguous() {
            return Err(Status::InvalidArgument);
        }
        if logits.rows() == 0
            || logits.cols() == 0
            || topk_ids.rows != logits.rows()
            || topk_weights.rows != logits.rows()
            || topk_ids.cols != topk_weights.cols
            || topk_ids.cols == 0
        {
            return Err(Status::InvalidArgument);
        }
        if topk_ids.cols > QWEN36_MOE_MAX_TOP_K
            || logits.cols() > QWEN36_MOE_MAX_EXPERTS
            || topk_ids.cols > logits.cols()
        {
            return Err(Status::Unsupported);
        }
        if !routed_scaling_factor.is_finite() || routed_scaling_factor <= 0.0 {
            return Err(Status::InvalidArgument);
        }
        Ok(Self {
            raw: qscu::RouterTopkDesc {
                logits: logits.tensor(),
                topk_ids: topk_ids.tensor(),
                topk_weights: topk_weights.tensor(),
                num_tokens: logits.rows(),
                num_experts: logits.cols(),
                top_k: topk_ids.cols,
                score: score.raw(),
                renormalize: u32::from(renormalize),
                routed_scaling_factor,
            },
        })
    }
}

#[derive(Clone, Copy, Debug)]
pub(crate) struct GdnCausalConv1dBf16Args {
    pub(crate) x: DMat<BF16>,
    pub(crate) weight: DMat<BF16>,
    pub(crate) bias: Option<Bf16OrF32Vec>,
    pub(crate) state: GdnConvState,
    pub(crate) state_read_indices: Option<DVec<I32>>,
    pub(crate) state_write_indices: Option<DVec<I32>>,
    pub(crate) seq_indptr: Option<DVec<I32>>,
    pub(crate) out: DMat<BF16>,
    pub(crate) batch_size: u32,
    pub(crate) activation: Activation,
    pub(crate) update_state: bool,
}

#[derive(Clone, Copy, Debug)]
pub(crate) struct GdnCausalConv1dBf16 {
    pub(super) raw: qscu::GdnCausalConv1dDesc,
}

impl GdnCausalConv1dBf16 {
    pub(crate) fn new(args: GdnCausalConv1dBf16Args) -> Result<Self, Status> {
        validate_nonzero(&[args.batch_size])?;
        args.x.require_contiguous()?;
        args.weight.require_contiguous()?;
        args.out.require_contiguous()?;
        if args.x.rows == 0
            || args.x.cols != QWEN36_GDN_PACKED_DIM
            || args.weight.rows != QWEN36_GDN_PACKED_DIM
            || args.weight.cols != QWEN36_GDN_CONV_WIDTH
            || args.out.rows != args.x.rows
            || args.out.cols != QWEN36_GDN_PACKED_DIM
        {
            return Err(Status::InvalidArgument);
        }
        if let Some(bias) = args.bias {
            require_float_vec(bias, QWEN36_GDN_PACKED_DIM)?;
        }
        if let Some(indices) = args.state_read_indices {
            require_i32_vec(indices, args.batch_size)?;
        }
        if let Some(indices) = args.state_write_indices {
            require_i32_vec(indices, args.batch_size)?;
        }
        if args.state_read_indices.is_none() && args.state_write_indices.is_none() {
            return Err(Status::InvalidArgument);
        }
        let seq_indptr = if let Some(seq_indptr) = args.seq_indptr {
            let expected = args
                .batch_size
                .checked_add(1)
                .ok_or(Status::InvalidArgument)?;
            require_i32_vec(seq_indptr, expected)?;
            seq_indptr.data
        } else {
            if args.batch_size != args.x.rows {
                return Err(Status::InvalidArgument);
            }
            ptr::null_mut()
        };
        if !matches!(args.activation, Activation::None | Activation::Silu) {
            return Err(Status::Unsupported);
        }

        Ok(Self {
            raw: qscu::GdnCausalConv1dDesc {
                x: args.x.tensor(),
                weight: args.weight.tensor(),
                bias: args
                    .bias
                    .map_or(zero_tensor1(ffi::DTYPE_BF16), |bias| bias.tensor()),
                state: args.state.tensor(),
                state_read_indices: args
                    .state_read_indices
                    .map_or(zero_tensor1(ffi::DTYPE_I32), DVec::tensor),
                state_write_indices: args
                    .state_write_indices
                    .map_or(zero_tensor1(ffi::DTYPE_I32), DVec::tensor),
                seq_indptr,
                out: args.out.tensor(),
                num_tokens: args.x.rows,
                batch_size: args.batch_size,
                activation: args.activation.raw(),
                update_state: u32::from(args.update_state),
            },
        })
    }
}

#[derive(Clone, Copy, Debug)]
pub(crate) struct GdnPostConvPrepareBf16Args {
    pub(crate) conv_out: DMat<BF16>,
    pub(crate) a: DMat<BF16>,
    pub(crate) b: DMat<BF16>,
    pub(crate) a_log: DVec<BF16>,
    pub(crate) dt_bias: DVec<BF16>,
    pub(crate) q: Bf16Heads,
    pub(crate) k: Bf16Heads,
    pub(crate) v: Bf16Heads,
    pub(crate) g_out: Option<DMat<F32>>,
    pub(crate) beta_out: Option<DMat<F32>>,
    pub(crate) apply_qk_l2norm: bool,
    pub(crate) l2norm_eps: f32,
    pub(crate) forget_gate_output: GdnForgetGateOutput,
}

#[derive(Clone, Copy, Debug)]
pub(crate) struct GdnPostConvPrepareBf16 {
    pub(super) raw: qscu::GdnPostConvPrepareDesc,
}

impl GdnPostConvPrepareBf16 {
    pub(crate) fn new(args: GdnPostConvPrepareBf16Args) -> Result<Self, Status> {
        validate_eps(args.l2norm_eps)?;
        args.conv_out.require_contiguous()?;
        args.a.require_contiguous()?;
        args.b.require_contiguous()?;
        args.a_log.require_contiguous()?;
        args.dt_bias.require_contiguous()?;
        args.q.require_contiguous()?;
        args.k.require_contiguous()?;
        args.v.require_contiguous()?;
        let tokens = args.conv_out.rows;
        if args.conv_out.cols != QWEN36_GDN_PACKED_DIM
            || args.a.rows != tokens
            || args.a.cols != QWEN36_GDN_NUM_V_HEADS
            || args.b.rows != tokens
            || args.b.cols != QWEN36_GDN_NUM_V_HEADS
            || args.a_log.len != QWEN36_GDN_NUM_V_HEADS
            || args.dt_bias.len != QWEN36_GDN_NUM_V_HEADS
        {
            return Err(Status::InvalidArgument);
        }
        require_qwen36_gdn_heads(args.q, QWEN36_GDN_NUM_Q_HEADS, tokens)?;
        require_qwen36_gdn_heads(args.k, QWEN36_GDN_NUM_K_HEADS, tokens)?;
        require_qwen36_gdn_heads(args.v, QWEN36_GDN_NUM_V_HEADS, tokens)?;
        if let Some(g_out) = args.g_out {
            g_out.require_contiguous()?;
            if g_out.rows != tokens || g_out.cols != QWEN36_GDN_NUM_V_HEADS {
                return Err(Status::InvalidArgument);
            }
        }
        if let Some(beta_out) = args.beta_out {
            beta_out.require_contiguous()?;
            if beta_out.rows != tokens || beta_out.cols != QWEN36_GDN_NUM_V_HEADS {
                return Err(Status::InvalidArgument);
            }
        }

        Ok(Self {
            raw: qscu::GdnPostConvPrepareDesc {
                conv_out: args.conv_out.tensor(),
                a: args.a.tensor(),
                b: args.b.tensor(),
                a_log: args.a_log.tensor(),
                dt_bias: args.dt_bias.tensor(),
                q: args.q.tensor(),
                k: args.k.tensor(),
                v: args.v.tensor(),
                g_out: args
                    .g_out
                    .map_or(zero_tensor2(ffi::DTYPE_F32), DMat::tensor),
                beta_out: args
                    .beta_out
                    .map_or(zero_tensor2(ffi::DTYPE_F32), DMat::tensor),
                num_tokens: tokens,
                apply_qk_l2norm: u32::from(args.apply_qk_l2norm),
                l2norm_eps: args.l2norm_eps,
                forget_gate_output: args.forget_gate_output.raw(),
            },
        })
    }
}

#[derive(Clone, Copy, Debug)]
pub(crate) struct GdnRmsNormGatedBf16Args {
    pub(crate) x: Bf16Heads,
    pub(crate) gate: Bf16Heads,
    pub(crate) weight: Bf16OrF32Vec,
    pub(crate) out: Bf16Heads,
    pub(crate) eps: f32,
    pub(crate) gate_activation: Activation,
}

#[derive(Clone, Copy, Debug)]
pub(crate) struct GdnRmsNormGatedBf16 {
    pub(super) raw: qscu::GdnRmsnormGatedDesc,
}

impl GdnRmsNormGatedBf16 {
    pub(crate) fn new(args: GdnRmsNormGatedBf16Args) -> Result<Self, Status> {
        validate_eps(args.eps)?;
        if !matches!(args.gate_activation, Activation::Silu | Activation::Sigmoid) {
            return Err(Status::Unsupported);
        }
        args.x.require_contiguous()?;
        args.gate.require_contiguous()?;
        args.out.require_contiguous()?;
        require_float_vec(args.weight, QWEN36_GDN_VALUE_DIM)?;
        let tokens = args.x.tokens;
        require_qwen36_gdn_heads(args.x, QWEN36_GDN_NUM_V_HEADS, tokens)?;
        require_qwen36_gdn_heads(args.gate, QWEN36_GDN_NUM_V_HEADS, tokens)?;
        require_qwen36_gdn_heads(args.out, QWEN36_GDN_NUM_V_HEADS, tokens)?;
        Ok(Self {
            raw: qscu::GdnRmsnormGatedDesc {
                x: args.x.tensor(),
                gate: args.gate.tensor(),
                weight: args.weight.tensor(),
                out: args.out.tensor(),
                num_tokens: tokens,
                eps: args.eps,
                gate_activation: args.gate_activation.raw(),
            },
        })
    }
}

#[derive(Clone, Copy, Debug)]
pub(crate) struct GdnDecodeBf16Args {
    pub(crate) q: Bf16Heads,
    pub(crate) k: Bf16Heads,
    pub(crate) v: Bf16Heads,
    pub(crate) a: DMat<BF16>,
    pub(crate) b: DMat<BF16>,
    pub(crate) a_log: DVec<BF16>,
    pub(crate) dt_bias: DVec<BF16>,
    pub(crate) state: GdnRecurrentState,
    pub(crate) state_indices: DVec<I32>,
    pub(crate) state_out_indices: Option<DVec<I32>>,
    pub(crate) out: Bf16Heads,
    pub(crate) scale: f32,
    pub(crate) use_qk_l2norm: bool,
    pub(crate) disable_state_update: bool,
}

#[derive(Clone, Copy, Debug)]
pub(crate) struct GdnDecodeBf16 {
    pub(super) raw: qscu::GdnDecodeDesc,
}

impl GdnDecodeBf16 {
    pub(crate) fn new(args: GdnDecodeBf16Args) -> Result<Self, Status> {
        validate_gdn_scale(args.scale)?;
        let tokens = args.q.tokens;
        validate_gdn_recurrent_tensors(
            args.q,
            args.k,
            args.v,
            args.a,
            args.b,
            args.a_log,
            args.dt_bias,
            args.out,
            tokens,
        )?;
        require_gdn_state_index_vec(
            args.state_indices,
            tokens,
            GdnStateIndexPolicy::NegativeSkips,
        )?;
        if let Some(indices) = args.state_out_indices {
            require_gdn_state_index_vec(indices, tokens, GdnStateIndexPolicy::NegativeSkips)?;
        }
        Ok(Self {
            raw: qscu::GdnDecodeDesc {
                q: args.q.tensor(),
                k: args.k.tensor(),
                v: args.v.tensor(),
                a: args.a.tensor(),
                b: args.b.tensor(),
                a_log: args.a_log.tensor(),
                dt_bias: args.dt_bias.tensor(),
                state: args.state.tensor(),
                state_indices: args.state_indices.tensor(),
                state_out_indices: args
                    .state_out_indices
                    .map_or(zero_tensor1(ffi::DTYPE_I32), DVec::tensor),
                out: args.out.tensor(),
                num_tokens: tokens,
                num_q_heads: QWEN36_GDN_NUM_Q_HEADS,
                num_k_heads: QWEN36_GDN_NUM_K_HEADS,
                num_v_heads: QWEN36_GDN_NUM_V_HEADS,
                key_dim: QWEN36_GDN_KEY_DIM,
                value_dim: QWEN36_GDN_VALUE_DIM,
                state_layout: qscu::GDN_STATE_LAYOUT_VK,
                scale: args.scale,
                use_qk_l2norm: u32::from(args.use_qk_l2norm),
                disable_state_update: u32::from(args.disable_state_update),
            },
        })
    }
}

#[derive(Clone, Copy, Debug)]
pub(crate) struct GdnPrefillBf16Args {
    pub(crate) q: Bf16Heads,
    pub(crate) k: Bf16Heads,
    pub(crate) v: Bf16Heads,
    pub(crate) a: DMat<BF16>,
    pub(crate) b: DMat<BF16>,
    pub(crate) a_log: DVec<BF16>,
    pub(crate) dt_bias: DVec<BF16>,
    pub(crate) state: GdnRecurrentState,
    pub(crate) seq_indptr: DVec<I32>,
    pub(crate) state_indices: DVec<I32>,
    pub(crate) state_out_indices: Option<DVec<I32>>,
    pub(crate) out: Bf16Heads,
    pub(crate) batch_size: u32,
    pub(crate) scale: f32,
    pub(crate) use_qk_l2norm: bool,
    pub(crate) disable_state_update: bool,
}

#[derive(Clone, Copy, Debug)]
pub(crate) struct GdnPrefillBf16 {
    pub(super) raw: qscu::GdnPrefillDesc,
}

impl GdnPrefillBf16 {
    pub(crate) fn new(args: GdnPrefillBf16Args) -> Result<Self, Status> {
        validate_gdn_scale(args.scale)?;
        validate_nonzero(&[args.batch_size])?;
        let total_tokens = args.q.tokens;
        validate_gdn_recurrent_tensors(
            args.q,
            args.k,
            args.v,
            args.a,
            args.b,
            args.a_log,
            args.dt_bias,
            args.out,
            total_tokens,
        )?;
        let seq_indptr_len = args
            .batch_size
            .checked_add(1)
            .ok_or(Status::InvalidArgument)?;
        require_i32_vec(args.seq_indptr, seq_indptr_len)?;
        require_gdn_state_index_vec(
            args.state_indices,
            args.batch_size,
            GdnStateIndexPolicy::NegativeSkips,
        )?;
        if let Some(indices) = args.state_out_indices {
            require_gdn_state_index_vec(
                indices,
                args.batch_size,
                GdnStateIndexPolicy::NegativeSkips,
            )?;
        }
        Ok(Self {
            raw: qscu::GdnPrefillDesc {
                q: args.q.tensor(),
                k: args.k.tensor(),
                v: args.v.tensor(),
                a: args.a.tensor(),
                b: args.b.tensor(),
                a_log: args.a_log.tensor(),
                dt_bias: args.dt_bias.tensor(),
                state: args.state.tensor(),
                seq_indptr: args.seq_indptr.data,
                state_indices: args.state_indices.tensor(),
                state_out_indices: args
                    .state_out_indices
                    .map_or(zero_tensor1(ffi::DTYPE_I32), DVec::tensor),
                out: args.out.tensor(),
                batch_size: args.batch_size,
                total_tokens,
                num_q_heads: QWEN36_GDN_NUM_Q_HEADS,
                num_k_heads: QWEN36_GDN_NUM_K_HEADS,
                num_v_heads: QWEN36_GDN_NUM_V_HEADS,
                key_dim: QWEN36_GDN_KEY_DIM,
                value_dim: QWEN36_GDN_VALUE_DIM,
                state_layout: qscu::GDN_STATE_LAYOUT_VK,
                scale: args.scale,
                use_qk_l2norm: u32::from(args.use_qk_l2norm),
                disable_state_update: u32::from(args.disable_state_update),
            },
        })
    }
}
