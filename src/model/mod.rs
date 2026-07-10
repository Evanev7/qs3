use crate::engine::Status;
use crate::ext::SafeVec;
use crate::ffi::{self, cuda};
use crate::{QWEN36_FULL_ATTN_HEAD_DIM, QWEN36_FULL_ATTN_Q_HEADS};

use std::ffi::c_void;
use std::mem;

mod config;
mod runner;

use config::QwenBlockKind;
pub use config::{QwenConfig, QwenMoeConfig};
pub use runner::{ModelRunner, QwenRequest, QwenResult};
mod weights;

pub use weights::QwenWeights;
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

mod state;

mod scratch;

pub(crate) use scratch::DeviceBuffer;
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
