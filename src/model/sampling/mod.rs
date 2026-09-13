use super::scratch::DeviceBuffer;
use crate::{Status, backend::qstriton::sampling as kernels, ffi};

mod tables;

/// Sampling for a single Qwen request. Temperature zero selects greedy decoding
/// and ignores filtering. Otherwise apply temperature, top-k, then top-p.
/// Top-k zero disables it; top-p must be in (0, 1], with one disabling it.
/// Boundary ties retain lower token IDs first.
/// NaN, positive infinity, temperature scaling overflow, or a row with no finite
/// candidate fails sampling. Negative infinity masks an individual token.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SamplingParams {
    pub temperature: f32,
    pub top_k: u32,
    pub top_p: f32,
    /// Philox seed. Noise is indexed by the absolute position of the input token
    /// producing the logit row and by token ID. Reset/rebuild repeats those draws;
    /// failed attempts do not consume RNG state. Different requests with the same
    /// seed and position use the same noise.
    pub seed: u64,
}

impl Default for SamplingParams {
    fn default() -> Self {
        Self {
            temperature: 0.0,
            top_k: 0,
            top_p: 1.0,
            seed: 0,
        }
    }
}

impl SamplingParams {
    pub(super) fn validate(self, vocab: u32) -> Result<(), Status> {
        if !self.temperature.is_finite()
            || self.temperature < 0.0
            || (self.temperature > 0.0 && !self.temperature.recip().is_finite())
            || self.top_k > vocab
            || !self.top_p.is_finite()
            || self.top_p <= 0.0
            || self.top_p > 1.0
        {
            return Err(Status::InvalidArgument);
        }
        Ok(())
    }
}

/// Modules and allocations are prepared once, before any decode/capture.
pub(super) struct Sampler {
    prepare: kernels::prepare::Kernel,
    filter: kernels::filter::Kernel,
    gumbel: kernels::gumbel::Kernel,
    reduce: kernels::reduce::Kernel,
    params: SamplingParams,
    vocab: u32,
    processed: DeviceBuffer<f32>,
    buffer: DeviceBuffer<f32>,
    percentile: DeviceBuffer<f32>,
    normal: DeviceBuffer<f32>,
    local_max: DeviceBuffer<f32>,
    local_ids: DeviceBuffer<i32>,
}

impl Sampler {
    pub(super) fn new(
        device: i32,
        stream: ffi::CudaStream,
        vocab: u32,
        params: SamplingParams,
    ) -> Result<Self, Status> {
        params.validate(vocab)?;
        if vocab == 0 || i64::from(vocab) > kernels::prepare::constants::VOCAB {
            return Err(Status::Unsupported);
        }
        let mut processed = DeviceBuffer::empty(device);
        let mut buffer = DeviceBuffer::empty(device);
        let mut local_max = DeviceBuffer::empty(device);
        let mut local_ids = DeviceBuffer::empty(device);
        processed.ensure(kernels::prepare::constants::VOCAB as usize)?;
        buffer.ensure(kernels::prepare::constants::VOCAB as usize)?;
        local_max.ensure(kernels::gumbel::GRID[0] as usize)?;
        local_ids.ensure(kernels::gumbel::GRID[0] as usize)?;
        let percentile =
            DeviceBuffer::from_slice(device, stream, &tables::PERCENTILE_TO_STD_TABLE)?;
        let normal = DeviceBuffer::from_slice(device, stream, &tables::NORMAL_CDF_TO_SIGMA_TABLE)?;
        // Allocations establish the device's primary context before module load.
        unsafe {
            Ok(Self {
                prepare: kernels::prepare::Kernel::load().map_err(|_| Status::CudaError)?,
                filter: kernels::filter::Kernel::load().map_err(|_| Status::CudaError)?,
                gumbel: kernels::gumbel::Kernel::load().map_err(|_| Status::CudaError)?,
                reduce: kernels::reduce::Kernel::load().map_err(|_| Status::CudaError)?,
                params,
                vocab,
                processed,
                buffer,
                percentile,
                normal,
                local_max,
                local_ids,
            })
        }
    }

    // Checked views of runner-owned inputs; scratch is private and never aliases
    // inputs/output. The runner keeps everything alive through stream completion.
    pub(super) fn launch(
        &mut self,
        stream: ffi::CudaStream,
        logits: &DeviceBuffer<f32>,
        positions: &DeviceBuffer<i32>,
        position_index: u32,
        output: &DeviceBuffer<i32>,
    ) -> Result<(), Status> {
        logits.vector(self.vocab)?;
        positions.vector(
            position_index
                .checked_add(1)
                .ok_or(Status::InvalidArgument)?,
        )?;
        output.vector(1)?;
        let compiled_vocab = kernels::prepare::constants::VOCAB as i32;
        let k = if self.params.top_k == 0 {
            compiled_vocab
        } else {
            self.params.top_k as i32
        };
        unsafe {
            self.prepare
                .launch(
                    stream,
                    logits.ptr,
                    self.processed.ptr,
                    self.vocab as i32,
                    self.params.temperature,
                )
                .map_err(|_| Status::CudaError)?;
            if self.params.top_k != 0 || self.params.top_p < 1.0 {
                self.filter
                    .launch(
                        stream,
                        self.processed.ptr,
                        compiled_vocab,
                        self.buffer.ptr,
                        self.percentile.ptr,
                        self.normal.ptr,
                        k,
                        self.params.top_p,
                    )
                    .map_err(|_| Status::CudaError)?;
            }
            self.gumbel
                .launch(
                    stream,
                    self.processed.ptr,
                    logits.ptr,
                    self.local_max.ptr,
                    self.local_ids.ptr,
                    positions.ptr.add(position_index as usize),
                    self.params.seed,
                    self.vocab as i32,
                    self.params.temperature,
                )
                .map_err(|_| Status::CudaError)?;
            self.reduce
                .launch(stream, self.local_max.ptr, self.local_ids.ptr, output.ptr)
                .map_err(|_| Status::CudaError)
        }
    }
}

#[cfg(test)]
mod tests;
