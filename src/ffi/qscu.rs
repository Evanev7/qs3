#![allow(dead_code)]

use crate::Status;

use super::{self as ffi, sys};

pub(crate) type SiluAndMulDesc = sys::qscu_silu_and_mul_desc;
pub(crate) type Qwen36SharedExpertGateAddDesc = sys::qscu_qwen36_shared_expert_gate_add_desc;
pub(crate) type EmbeddingGatherDesc = sys::qscu_embedding_gather_desc;
pub(crate) type SamplingDesc = sys::qscu_sampling_desc;
pub(crate) type GdnDecodeDesc = sys::qscu_gdn_decode_desc;
pub(crate) type GdnPrefillDesc = sys::qscu_gdn_prefill_desc;
pub(crate) type GdnCausalConv1dDesc = sys::qscu_qwen36_gdn_causal_conv1d_desc;
pub(crate) type GdnPostConvPrepareDesc = sys::qscu_qwen36_gdn_post_conv_prepare_desc;
pub(crate) type GdnRmsnormGatedDesc = sys::qscu_qwen36_gdn_rmsnorm_gated_desc;
pub(crate) type RouterTopkDesc = sys::qscu_router_topk_desc;

pub(crate) type ActivationRaw = sys::qscu_activation;
pub(crate) type GdnForgetGateOutputRaw = sys::qscu_gdn_forget_gate_output;
pub(crate) type GdnStateLayoutRaw = sys::qscu_gdn_state_layout;
pub(crate) type RouterScoreRaw = sys::qscu_router_score;

pub(crate) const ACTIVATION_NONE: ActivationRaw = sys::QSCU_ACTIVATION_NONE;
pub(crate) const ACTIVATION_SILU: ActivationRaw = sys::QSCU_ACTIVATION_SILU;
pub(crate) const ACTIVATION_SIGMOID: ActivationRaw = sys::QSCU_ACTIVATION_SIGMOID;

pub(crate) const GDN_FORGET_LOG_DECAY: GdnForgetGateOutputRaw = sys::QSCU_GDN_FORGET_LOG_DECAY;
pub(crate) const GDN_FORGET_LINEAR_ALPHA: GdnForgetGateOutputRaw =
    sys::QSCU_GDN_FORGET_LINEAR_ALPHA;
pub(crate) const GDN_STATE_LAYOUT_VK: GdnStateLayoutRaw = sys::QSCU_GDN_STATE_LAYOUT_VK;

pub(crate) const ROUTER_SCORE_SOFTMAX: RouterScoreRaw = sys::QSCU_ROUTER_SCORE_SOFTMAX;
pub(crate) const ROUTER_SCORE_SIGMOID: RouterScoreRaw = sys::QSCU_ROUTER_SCORE_SIGMOID;

pub(crate) unsafe fn silu_and_mul_bf16(
    desc: &SiluAndMulDesc,
    stream: ffi::CudaStream,
) -> Result<(), Status> {
    ffi::result_from_raw(unsafe { sys::qscu_silu_and_mul_bf16(desc, stream) })
}

pub(crate) unsafe fn qwen36_shared_expert_gate_add_bf16(
    desc: &Qwen36SharedExpertGateAddDesc,
    stream: ffi::CudaStream,
) -> Result<(), Status> {
    ffi::result_from_raw(unsafe { sys::qscu_qwen36_shared_expert_gate_add_bf16(desc, stream) })
}

pub(crate) unsafe fn embedding_gather_bf16(
    desc: &EmbeddingGatherDesc,
    stream: ffi::CudaStream,
) -> Result<(), Status> {
    ffi::result_from_raw(unsafe { sys::qscu_embedding_gather_bf16(desc, stream) })
}

pub(crate) unsafe fn logits_soft_cap_f32(
    logits: &ffi::Tensor2,
    rows: u32,
    vocab_size: u32,
    soft_cap: f32,
    stream: ffi::CudaStream,
) -> Result<(), Status> {
    ffi::result_from_raw(unsafe {
        sys::qscu_logits_soft_cap_f32(logits, rows, vocab_size, soft_cap, stream)
    })
}

pub(crate) unsafe fn greedy_argmax_f32(
    desc: &SamplingDesc,
    stream: ffi::CudaStream,
) -> Result<(), Status> {
    ffi::result_from_raw(unsafe { sys::qscu_greedy_argmax_f32(desc, stream) })
}

pub(crate) unsafe fn qwen36_gdn_causal_conv1d_bf16(
    desc: &GdnCausalConv1dDesc,
    stream: ffi::CudaStream,
) -> Result<(), Status> {
    ffi::result_from_raw(unsafe { sys::qscu_qwen36_gdn_causal_conv1d_bf16(desc, stream) })
}

pub(crate) unsafe fn qwen36_gdn_post_conv_prepare_bf16(
    desc: &GdnPostConvPrepareDesc,
    stream: ffi::CudaStream,
) -> Result<(), Status> {
    ffi::result_from_raw(unsafe { sys::qscu_qwen36_gdn_post_conv_prepare_bf16(desc, stream) })
}

pub(crate) unsafe fn qwen36_gdn_rmsnorm_gated_bf16(
    desc: &GdnRmsnormGatedDesc,
    stream: ffi::CudaStream,
) -> Result<(), Status> {
    ffi::result_from_raw(unsafe { sys::qscu_qwen36_gdn_rmsnorm_gated_bf16(desc, stream) })
}

pub(crate) unsafe fn gdn_decode(
    ctx: &mut ffi::qsfi::Context,
    desc: &GdnDecodeDesc,
) -> Result<(), Status> {
    ffi::result_from_raw(unsafe { sys::qscu_gdn_decode(ctx.as_raw(), desc) })
        .inspect_err(|_| _ = ctx.last_error())
}

pub(crate) unsafe fn gdn_prefill(
    ctx: &mut ffi::qsfi::Context,
    desc: &GdnPrefillDesc,
) -> Result<(), Status> {
    ffi::result_from_raw(unsafe { sys::qscu_gdn_prefill(ctx.as_raw(), desc) })
        .inspect_err(|_| _ = ctx.last_error())
}

pub(crate) unsafe fn router_topk(
    desc: &RouterTopkDesc,
    stream: ffi::CudaStream,
) -> Result<(), Status> {
    ffi::result_from_raw(unsafe { sys::qscu_router_topk(desc, stream) })
}
