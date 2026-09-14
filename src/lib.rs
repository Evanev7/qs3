#![allow(clippy::missing_safety_doc)]

/// Build configuration generated from `models/config.nix`.
pub mod constants {
    include!(concat!(env!("CARGO_MANIFEST_DIR"), "/build/constants.rs"));
}

pub(crate) const QWEN36_MOE_MAX_TOP_K: u32 = 8;
pub(crate) const QWEN36_MOE_MAX_EXPERTS: u32 = 256;
pub(crate) const QWEN36_MOE_ROUTER_SCORE: backend::qscu::RouterScore =
    backend::qscu::RouterScore::Softmax;
pub(crate) const QWEN36_MOE_ROUTER_RENORMALIZE: bool = true;
pub(crate) const QWEN36_MOE_ROUTER_SCALING_FACTOR: f32 = 1.0;
pub(crate) const QWEN36_GDN_STATE_SLOTS_PER_LAYER: u32 = 2;

mod backend;
pub mod engine;
pub(crate) mod ext;
pub mod ffi;
mod loader;
pub mod memory;
pub mod model;
pub mod tokenizer;

mod test_assets;

pub use engine::{
    AppendBatch, AttentionLayer, BatchKind, Commit, CoreState, DecodeBatch, DynDType, Engine,
    EngineConfig, KvLayout, RequestId, Status,
};
pub use loader::benchmark::run_core_benchmark;
pub use model::{ModelRunner, QwenConfig, QwenRequest, QwenResult, QwenWeights};
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
            qscb.linear(input, weight, bf16_output, workspace)?;
            qscb.linear(input, weight, f32_output, workspace)
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
