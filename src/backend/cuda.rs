use crate::{
    Status,
    ffi::{self, qscu},
};

pub(crate) use super::{
    Activation, EmbeddingGatherBf16, GdnCausalConv1dBf16, GdnCausalConv1dBf16Args, GdnConvState,
    GdnForgetGateOutput, GdnPostConvPrepareBf16, GdnPostConvPrepareBf16Args, GdnRecurrentState,
    GdnRmsNormGatedBf16, GdnRmsNormGatedBf16Args, GreedyArgmaxF32, LogitsSoftCapF32,
    Qwen36FullAttentionOutputGateBf16, Qwen36SharedExpertGateAddBf16, RouterScore, RouterTopK,
    SiluAndMulBf16,
};

/// Thin typed access to qscu's stream-ordered CUDA kernels.
pub(crate) struct Cuda<'a> {
    stream: &'a ffi::CudaStream,
}

impl<'a> Cuda<'a> {
    pub(super) fn new(stream: &'a ffi::CudaStream) -> Self {
        Self { stream }
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
}
