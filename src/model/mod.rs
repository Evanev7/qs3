use crate::backend::cublas::Bf16Gemm;
use crate::backend::cuda::{
    Activation, EmbeddingGatherBf16, GdnCausalConv1dBf16, GdnCausalConv1dBf16Args, GdnConvState,
    GdnDecodeBf16, GdnDecodeBf16Args, GdnForgetGateOutput, GdnPostConvPrepareBf16,
    GdnPostConvPrepareBf16Args, GdnPrefillBf16, GdnPrefillBf16Args, GdnRecurrentState,
    GdnRmsNormGatedBf16, GdnRmsNormGatedBf16Args, GreedyArgmaxF32, LogitsSoftCapF32,
    Qwen36FullAttentionOutputGateBf16, Qwen36SharedExpertGateAddBf16, RouterTopK, SiluAndMulBf16,
};
use crate::backend::flashinfer::{
    FusedAddRmsNormBf16, MoeBf16Execute, MoeBf16ExecuteArgs, MoeBf16PlanConfig, MoePlan,
    RmsNormBf16, RopeApplyBf16, Workspace,
};
use crate::backend::{Bf16Heads, Bf16OrF32Mat, Bf16OrF32Vec, DMat, DTensor3, DVec, FloatStorage};
use crate::engine::{
    AppendBatch, AttentionLayer, Commit, DecodeBatch, DynDType, Engine, EngineConfig, KvLayout,
    RequestId, Status, validate_supported_attention_grouping,
    validate_supported_attention_head_dim,
};
use crate::ext::{SafeVec, try_clone_slice};
use crate::ffi::{self, cuda};
use crate::{
    QWEN36_FULL_ATTN_GROUP_SIZE, QWEN36_FULL_ATTN_HEAD_DIM, QWEN36_FULL_ATTN_KV_HEADS,
    QWEN36_FULL_ATTN_KV_HIDDEN, QWEN36_FULL_ATTN_Q_HEADS, QWEN36_FULL_ATTN_Q_HIDDEN,
    QWEN36_FULL_ATTN_Q_PROJ_OUT, QWEN36_FULL_ATTN_ROTARY_DIM, QWEN36_GDN_CONV_STATE,
    QWEN36_GDN_CONV_WIDTH, QWEN36_GDN_KEY_DIM, QWEN36_GDN_NUM_K_HEADS, QWEN36_GDN_NUM_Q_HEADS,
    QWEN36_GDN_NUM_V_HEADS, QWEN36_GDN_OUTPUT_DIM, QWEN36_GDN_PACKED_DIM,
    QWEN36_GDN_STATE_SLOTS_PER_LAYER, QWEN36_GDN_VALUE_DIM, QWEN36_HIDDEN_SIZE,
    QWEN36_MOE_INTERMEDIATE_SIZE, QWEN36_MOE_MAX_EXPERTS, QWEN36_MOE_MAX_TOP_K,
    QWEN36_MOE_NUM_EXPERTS, QWEN36_MOE_ROUTER_RENORMALIZE, QWEN36_MOE_ROUTER_SCALING_FACTOR,
    QWEN36_MOE_ROUTER_SCORE, QWEN36_MOE_SHARED_EXPERT_INTERMEDIATE_SIZE, QWEN36_MOE_TOP_K,
};

use std::ffi::c_void;
use std::{mem, ptr};

mod config;
mod runner;

use config::QwenBlockKind;
pub use config::{QwenConfig, QwenMoeConfig};
#[cfg(test)]
use config::{QwenGdnShape, QwenLayerPattern, QwenModelShape};
pub use runner::{ModelRunner, QwenRequest, QwenResult};
mod weights;

pub use weights::QwenWeights;
use weights::{
    QwenAttentionMlpPtrs, QwenGdnPtrs, QwenLayerPtrs, QwenMlpPtrs, QwenSharedExpertPtrs,
};
#[cfg(test)]
use weights::{
    QwenAttentionMlpWeights, QwenGdnWeights, QwenLayerWeights, QwenMlpWeights,
    QwenSharedExpertWeights,
};
#[derive(Clone, Copy)]
struct BatchRun<'a> {
    tokens: &'a [i32],
    start_pos: u32,
    kind: ActiveRunKind,
}

#[derive(Clone, Copy)]
enum ActiveRunKind {
    Append,
    Decode,
}

#[derive(Clone, Copy)]
enum GemmOut {
    Bf16,
    F32,
}

mod state;

#[cfg(test)]
use state::GdnSlotMap;
use state::GdnState;
mod scratch;

pub(crate) use scratch::DeviceBuffer;
use scratch::RunnerScratch;
struct DeterministicRng {
    state: u64,
}

impl DeterministicRng {
    fn new(seed: u64) -> Self {
        Self {
            state: seed ^ 0x9e37_79b9_7f4a_7c15,
        }
    }

    fn next_u32(&mut self) -> u32 {
        let mut x = self.state;
        x ^= x >> 12;
        x ^= x << 25;
        x ^= x >> 27;
        self.state = x;
        ((x.wrapping_mul(0x2545_f491_4f6c_dd1d)) >> 32) as u32
    }

    fn next_unit_f32(&mut self) -> f32 {
        let bits = 0x3f80_0000 | (self.next_u32() >> 9);
        f32::from_bits(bits) - 1.0
    }
}

fn random_bf16_values(
    rng: &mut DeterministicRng,
    count: usize,
    scale: f32,
) -> Result<Vec<u16>, Status> {
    let mut out = Vec::safe_new(count)?;
    for _ in 0..count {
        let value = (rng.next_unit_f32() * 2.0 - 1.0) * scale;
        out.push(f32_to_bf16_bits(value));
    }
    Ok(out)
}

fn constant_bf16_values(count: usize, value: f32) -> Result<Vec<u16>, Status> {
    let mut out = Vec::safe_new(count)?;
    out.resize(count, f32_to_bf16_bits(value));
    Ok(out)
}

fn qwen36_gdn_scale() -> f32 {
    1.0 / (QWEN36_GDN_KEY_DIM as f32).sqrt()
}

/// Extract Qwen3.6 full-attention Q and output-gate heads from packed BF16
/// q_proj output laid out as `[q0, gate0, q1, gate1, ...]`.
///
/// # Safety
///
/// `packed`, `q`, and `gate` must be valid device pointers on `stream`'s device.
/// `packed` must have at least `rows * FULL_ATTN_Q_HEADS * 2 * FULL_ATTN_HEAD_DIM`
/// BF16 elements, and `q` and `gate` must each have at least
/// `rows * FULL_ATTN_Q_HEADS * FULL_ATTN_HEAD_DIM` BF16 elements.
unsafe fn extract_qwen36_packed_attention_q_and_gate_bf16(
    packed: ffi::DevicePtr,
    q: ffi::DevicePtr,
    gate: ffi::DevicePtr,
    rows: u32,
    stream: *mut c_void,
) -> Result<(), Status> {
    let head_bytes = checked_usize_product(&[
        QWEN36_FULL_ATTN_HEAD_DIM,
        u32::try_from(mem::size_of::<u16>()).map_err(|_| Status::InvalidArgument)?,
    ])?;
    let packed_head_bytes = head_bytes.checked_mul(2).ok_or(Status::InvalidArgument)?;
    let height = checked_usize_product(&[rows, QWEN36_FULL_ATTN_Q_HEADS])?;
    device_copy_2d_on_stream(
        q,
        head_bytes,
        packed,
        packed_head_bytes,
        head_bytes,
        height,
        stream,
    )?;
    device_copy_2d_on_stream(
        gate,
        head_bytes,
        device_ptr_byte_offset(packed, head_bytes)?,
        packed_head_bytes,
        head_bytes,
        height,
        stream,
    )
}

fn f32_to_bf16_bits(value: f32) -> u16 {
    let bits = value.to_bits();
    let lsb = (bits >> 16) & 1;
    ((bits.wrapping_add(0x7fff + lsb)) >> 16) as u16
}

fn validate_token_ids(tokens: &[i32], vocab_size: u32) -> Result<(), Status> {
    let vocab_size = i32::try_from(vocab_size).map_err(|_| Status::Unsupported)?;
    for token in tokens {
        if *token < 0 || *token >= vocab_size {
            return Err(Status::InvalidArgument);
        }
    }
    Ok(())
}

fn checked_usize_product(values: &[u32]) -> Result<usize, Status> {
    let mut product = 1usize;
    for value in values {
        product = product
            .checked_mul(*value as usize)
            .ok_or(Status::InvalidArgument)?;
    }
    Ok(product)
}

fn device_ptr_byte_offset(ptr: ffi::DevicePtr, bytes: usize) -> Result<ffi::DevicePtr, Status> {
    if ptr.is_null() {
        return Err(Status::InvalidArgument);
    }
    Ok(unsafe { ptr.cast::<u8>().add(bytes).cast() })
}

fn device_copy_2d_on_stream(
    dst: ffi::DevicePtr,
    dst_pitch_bytes: usize,
    src: ffi::DevicePtr,
    src_pitch_bytes: usize,
    width_bytes: usize,
    height: usize,
    stream: *mut c_void,
) -> Result<(), Status> {
    if dst.is_null()
        || src.is_null()
        || dst_pitch_bytes == 0
        || src_pitch_bytes == 0
        || width_bytes == 0
        || height == 0
        || width_bytes > dst_pitch_bytes
        || width_bytes > src_pitch_bytes
    {
        return Err(Status::InvalidArgument);
    }
    result_from_cuda(unsafe {
        cuda::cudaMemcpy2DAsync(
            dst,
            dst_pitch_bytes,
            src.cast_const(),
            src_pitch_bytes,
            width_bytes,
            height,
            cuda::CUDA_MEMCPY_DEVICE_TO_DEVICE,
            stream,
        )
    })
}

fn activate_device(device_ordinal: i32) -> Result<(), Status> {
    if device_ordinal < 0 {
        return Ok(());
    }
    result_from_cuda(unsafe { cuda::cudaSetDevice(device_ordinal) })
}

fn resolve_device_ordinal(device_ordinal: i32) -> Result<i32, Status> {
    if device_ordinal >= 0 {
        return Ok(device_ordinal);
    }
    let mut current = 0;
    result_from_cuda(unsafe { cuda::cudaGetDevice(&mut current) })?;
    Ok(current)
}

fn synchronize_stream(stream: *mut c_void) -> Result<(), Status> {
    result_from_cuda(unsafe { cuda::cudaStreamSynchronize(stream) })
}

fn result_from_cuda(err: i32) -> Result<(), Status> {
    if err == cuda::CUDA_SUCCESS {
        Ok(())
    } else if err == cuda::CUDA_ERROR_MEMORY_ALLOCATION {
        Err(Status::OutOfMemory)
    } else {
        Err(Status::CudaError)
    }
}

#[cfg(test)]
mod tests;
