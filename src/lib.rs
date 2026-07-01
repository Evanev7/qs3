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
pub(crate) const QWEN36_MOE_ROUTER_SCORE: runtime::kernels::RouterScore =
    runtime::kernels::RouterScore::Softmax;
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

pub mod engine;
pub mod ffi;
pub mod model;
mod runtime;
mod weight_loader;

pub use engine::{
    AppendBatch, BatchKind, Commit, CoreState, DType, DecodeBatch, Engine, EngineConfig,
    EngineLayer, EngineTrait, KvLayout, RequestId, Status,
};
pub use model::{ModelRunner, QwenConfig, QwenMoeConfig, QwenRequest, QwenResult, QwenWeights};
