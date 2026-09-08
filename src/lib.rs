#![allow(clippy::missing_safety_doc)]

pub(crate) const QWEN36_FULL_ATTN_Q_HEADS: u32 = 16;
pub(crate) const QWEN36_FULL_ATTN_KV_HEADS: u32 = 2;
pub(crate) const QWEN36_FULL_ATTN_GROUP_SIZE: u32 =
    QWEN36_FULL_ATTN_Q_HEADS / QWEN36_FULL_ATTN_KV_HEADS;
pub(crate) const QWEN36_FULL_ATTN_HEAD_DIM: u32 = 256;
pub(crate) const QWEN36_FULL_ATTN_Q_HIDDEN: u32 =
    QWEN36_FULL_ATTN_Q_HEADS * QWEN36_FULL_ATTN_HEAD_DIM;
pub(crate) const QWEN36_FULL_ATTN_KV_HIDDEN: u32 =
    QWEN36_FULL_ATTN_KV_HEADS * QWEN36_FULL_ATTN_HEAD_DIM;
pub(crate) const QWEN36_FULL_ATTN_Q_PROJ_OUT: u32 = 2 * QWEN36_FULL_ATTN_Q_HIDDEN;
pub(crate) const QWEN36_FULL_ATTN_ROTARY_DIM: u32 = 64;
pub(crate) const QWEN36_HIDDEN_SIZE: u32 = 2048;
pub(crate) const QWEN36_MOE_NUM_EXPERTS: u32 = 256;
pub(crate) const QWEN36_MOE_TOP_K: u32 = 8;
pub(crate) const QWEN36_MOE_INTERMEDIATE_SIZE: u32 = 512;
pub(crate) const QWEN36_MOE_SHARED_EXPERT_INTERMEDIATE_SIZE: u32 = 512;
pub(crate) const QWEN36_MOE_MAX_TOP_K: u32 = 16;
pub(crate) const QWEN36_MOE_MAX_EXPERTS: u32 = 4096;
pub(crate) const QWEN36_MOE_ROUTER_SCORE: backend::qscu::RouterScore =
    backend::qscu::RouterScore::Softmax;
pub(crate) const QWEN36_MOE_ROUTER_RENORMALIZE: bool = true;
pub(crate) const QWEN36_MOE_ROUTER_SCALING_FACTOR: f32 = 1.0;
pub(crate) const QWEN36_GDN_NUM_Q_HEADS: u32 = 16;
pub(crate) const QWEN36_GDN_NUM_K_HEADS: u32 = 16;
pub(crate) const QWEN36_GDN_NUM_V_HEADS: u32 = 32;
pub(crate) const QWEN36_GDN_KEY_DIM: u32 = 128;
pub(crate) const QWEN36_GDN_VALUE_DIM: u32 = 128;
pub(crate) const QWEN36_GDN_CONV_WIDTH: u32 = 4;
pub(crate) const QWEN36_GDN_CONV_STATE: u32 = QWEN36_GDN_CONV_WIDTH - 1;
pub(crate) const QWEN36_GDN_PACKED_DIM: u32 =
    2 * QWEN36_GDN_NUM_K_HEADS * QWEN36_GDN_KEY_DIM + QWEN36_GDN_NUM_V_HEADS * QWEN36_GDN_VALUE_DIM;
pub(crate) const QWEN36_GDN_OUTPUT_DIM: u32 = QWEN36_GDN_NUM_V_HEADS * QWEN36_GDN_VALUE_DIM;
pub(crate) const QWEN36_GDN_STATE_SLOTS_PER_LAYER: u32 = 2;

const _: () = assert!(QWEN36_GDN_PACKED_DIM == 8192);
const _: () = assert!(QWEN36_GDN_OUTPUT_DIM == 4096);
const _: () = assert!(QWEN36_GDN_NUM_Q_HEADS == 16);
const _: () = assert!(QWEN36_GDN_NUM_K_HEADS == 16);
const _: () = assert!(QWEN36_GDN_NUM_V_HEADS == 32);
const _: () = assert!(QWEN36_GDN_KEY_DIM == 128);
const _: () = assert!(QWEN36_GDN_VALUE_DIM == 128);
const _: () = assert!(QWEN36_GDN_CONV_STATE == 3);
const _: () = assert!(QWEN36_FULL_ATTN_GROUP_SIZE == 8);
const _: () = assert!(QWEN36_FULL_ATTN_Q_HIDDEN == 4096);
const _: () = assert!(QWEN36_FULL_ATTN_KV_HIDDEN == 512);
const _: () = assert!(QWEN36_FULL_ATTN_Q_PROJ_OUT == 8192);

mod backend;
pub mod engine;
pub(crate) mod ext;
pub mod ffi;
mod loader;
pub mod model;
pub mod tokenizer;

mod test_assets;

pub use engine::{
    AppendBatch, AttentionLayer, BatchKind, Commit, CoreState, DecodeBatch, DynDType, Engine,
    EngineConfig, KvLayout, RequestId, Status,
};
pub use loader::benchmark::run_core_benchmark;
pub use model::{ModelRunner, QwenConfig, QwenMoeConfig, QwenRequest, QwenResult, QwenWeights};
pub use tokenizer::{QwenTokenizer, TokenizerError};

#[cfg(test)]
mod backend_contract_tests {
    use crate::Status;
    use crate::backend::{
        BF16, DMat, DVec, F32, I32, Workspace,
        qscb::Qscb,
        qscu::Qscu,
        qsfi::{Qsfi, RmsNormBf16},
    };

    unsafe fn qscu_owns_embedding(
        qscu: &mut Qscu<'_>,
        token_ids: DVec<I32>,
        embedding: DMat<BF16>,
        output: DMat<BF16>,
    ) -> Result<(), Status> {
        unsafe { qscu.embedding_gather_bf16(token_ids, embedding, output, None, true) }
    }
    unsafe fn qscb_owns_linear(
        qscb: &mut Qscb,
        input: DMat<BF16>,
        weight: DMat<BF16>,
        bf16_output: DMat<BF16>,
        f32_output: DMat<F32>,
        workspace: Workspace,
    ) -> Result<(), Status> {
        unsafe {
            qscb.linear_bf16(input, weight, bf16_output, workspace)?;
            qscb.linear_f32(input, weight, f32_output, workspace)
        }
    }
    fn qsfi_owns_norm(_: &mut Qsfi, _: &RmsNormBf16) {}

    #[test]
    fn native_operations_have_explicit_backend_types() {
        let _ = qscu_owns_embedding;
        let _ = qscb_owns_linear;
        let _ = qsfi_owns_norm;
    }
}
