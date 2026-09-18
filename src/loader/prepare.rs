//! Compiled CUDA weight layouts, enqueued before the loader's ownership handoff.
use super::{
    format::{LoadResult, WeightLoadError},
    plan::{LoadedWeightTensor, QwenLoadPlan},
    quantization::ProjectionQuantization,
    transfer::{
        WeightBuffer, WeightLoadBackend, WeightLoadSpan, WeightTensorDesc, result_from_cuda,
    },
};
use crate::{
    backend::{Qscb, Qsfi, qsfi::scale_count},
    dtype::{DynDType, F32, Fp8E4M3},
    ffi::cuda,
    memory::{CudaCtx, DeviceSpan, HostBuffer},
};
use std::collections::{BTreeMap, BTreeSet};

enum Transform {
    Fp8 {
        source: usize,
        n: u32,
        k: u32,
        source_scale: f32,
        shared_scale: f32,
    },
    Scales {
        source: usize,
        n: u32,
        k: u32,
        count: u32,
    },
    Parameters {
        name: String,
        host: HostBuffer<F32>,
    },
}
enum Destination {
    Tensor(usize),
    Parameters(String),
}
struct Allocation {
    destination: Destination,
    span: WeightLoadSpan,
    dtype: DynDType,
}

pub(super) struct Preparation {
    transforms: Vec<Transform>,
    allocations: Vec<Allocation>,
    provider: Option<Qsfi>,
    fp8_provider: Option<Qscb>,
}
impl Preparation {
    /// Validate the complete preparation recipe before any device allocation.
    pub(super) fn plan(plan: &QwenLoadPlan) -> LoadResult<Self> {
        let mut transforms = Vec::new();
        if let Some(q) = &plan.quantization {
            let mut used = BTreeSet::new();
            let mut scalar = |name: String| -> LoadResult<f32> {
                let value = *q.globals.get(&name).ok_or_else(|| {
                    WeightLoadError::tensor_table(format!("missing scale {name}"))
                })?;
                if !value.is_finite() || value <= 0.0 {
                    return Err(WeightLoadError::tensor_table(format!(
                        "invalid scale {name}"
                    )));
                }
                used.insert(name);
                Ok(value)
            };
            for (source, entry) in plan.entries.iter().enumerate() {
                let Some(name) = entry.spec.name.strip_suffix(".weight") else {
                    continue;
                };
                if !matches!(entry.spec.dtype, DynDType::FP8E4M3 | DynDType::U8) {
                    continue;
                }
                let input = scalar(format!("{name}.input_scale"))?;
                let values = match q.recipe(name)? {
                    ProjectionQuantization::Fp8 => {
                        let weight = scalar(format!("{name}.weight_scale"))?;
                        let [n, k] = entry.spec.shape.as_slice() else {
                            return Err(WeightLoadError::tensor_table("invalid FP8 weight shape"));
                        };
                        let mut shared_input = input;
                        let mut shared_weight = weight;
                        let mut requantize = false;
                        for sibling in fp8_group(name)? {
                            let tensor = plan
                                .entries
                                .iter()
                                .find(|e| e.spec.name == format!("{sibling}.weight"))
                                .ok_or_else(|| {
                                    WeightLoadError::tensor_table(format!(
                                        "missing fused FP8 projection {sibling}"
                                    ))
                                })?;
                            if q.recipe(&sibling)? != ProjectionQuantization::Fp8
                                || tensor.spec.dtype != DynDType::FP8E4M3
                                || tensor.spec.shape.len() != 2
                                || tensor.spec.shape[1] != *k
                            {
                                return Err(WeightLoadError::tensor_table(
                                    "incompatible fused FP8 projection",
                                ));
                            }
                            shared_input =
                                shared_input.max(scalar(format!("{sibling}.input_scale"))?);
                            let sibling_weight = scalar(format!("{sibling}.weight_scale"))?;
                            shared_weight = shared_weight.max(sibling_weight);
                            requantize |= sibling_weight != weight;
                        }
                        // vLLM requantizes every segment when any group scales differ,
                        // including the segment already using the maximum scale.
                        if requantize {
                            transforms.push(Transform::Fp8 {
                                source,
                                n: *n,
                                k: *k,
                                source_scale: weight,
                                shared_scale: shared_weight,
                            });
                        }
                        [shared_input, shared_weight]
                    }
                    ProjectionQuantization::Nvfp4 | ProjectionQuantization::W4A16Nvfp4 => {
                        let scale_name = format!("{name}.weight_scale");
                        let source = plan
                            .entries
                            .iter()
                            .position(|e| e.spec.name == scale_name)
                            .ok_or_else(|| {
                                WeightLoadError::tensor_table(format!("missing {scale_name}"))
                            })?;
                        let scale = &plan.entries[source].spec;
                        let [n, cols] = scale.shape.as_slice() else {
                            return Err(WeightLoadError::tensor_table("invalid NVFP4 scale shape"));
                        };
                        let k = cols
                            .checked_mul(16)
                            .ok_or_else(|| WeightLoadError::tensor_table("NVFP4 K overflow"))?;
                        if scale.dtype != DynDType::FP8E4M3
                            || entry.spec.dtype != DynDType::U8
                            || entry.spec.shape != [*n, k / 2]
                        {
                            return Err(WeightLoadError::tensor_table(
                                "NVFP4 weight/scale shape mismatch",
                            ));
                        }
                        let count = scale_count(*n, k).map_err(WeightLoadError::Backend)?;
                        transforms.push(Transform::Scales {
                            source,
                            n: *n,
                            k,
                            count,
                        });
                        [
                            1.0 / input,
                            input * scalar(format!("{name}.weight_scale_2"))?,
                        ]
                    }
                };
                if values.iter().any(|v| !v.is_finite() || *v <= 0.0) {
                    return Err(WeightLoadError::tensor_table("invalid prepared scale"));
                }
                let mut host = HostBuffer::<F32>::new(2).map_err(WeightLoadError::Backend)?;
                for (bytes, value) in host.as_mut().chunks_exact_mut(4).zip(values) {
                    bytes.copy_from_slice(&value.to_ne_bytes());
                }
                transforms.push(Transform::Parameters {
                    name: name.into(),
                    host,
                });
            }
            if used != q.globals.keys().cloned().collect() {
                return Err(WeightLoadError::tensor_table("unused quantization scales"));
            }
        }
        Ok(Self {
            transforms,
            allocations: Vec::new(),
            provider: None,
            fp8_provider: None,
        })
    }

    pub(super) fn enqueue<B: WeightLoadBackend>(
        &mut self,
        backend: &mut B,
        sources: &[WeightLoadSpan],
        ctx: &CudaCtx,
    ) -> LoadResult<()> {
        for transform in &self.transforms {
            if let Transform::Fp8 {
                source,
                n,
                k,
                source_scale,
                shared_scale,
            } = transform
            {
                if self.fp8_provider.is_none() {
                    self.fp8_provider = Some(Qscb::new(ctx).map_err(WeightLoadError::Backend)?);
                }
                let weight =
                    DeviceSpan::<Fp8E4M3>::new(sources[*source].ptr.cast(), sources[*source].bytes)
                        .map_err(WeightLoadError::Backend)?;
                unsafe {
                    self.fp8_provider.as_mut().unwrap().requantize_fp8(
                        weight.matrix(*n, *k).map_err(WeightLoadError::Backend)?,
                        *source_scale,
                        *shared_scale,
                    )
                }
                .map_err(WeightLoadError::Backend)?;
                continue;
            }
            let (name, dtype, count) = match transform {
                Transform::Fp8 { .. } => unreachable!(),
                Transform::Scales { source, count, .. } => (
                    format!("prepared.scales.{source}"),
                    DynDType::FP8E4M3,
                    *count,
                ),
                Transform::Parameters { name, .. } => {
                    (format!("{name}.parameters"), DynDType::F32, 2)
                }
            };
            let bytes = dtype
                .storage_bytes_for_shape(&[count])
                .map_err(WeightLoadError::Backend)?;
            let span = backend
                .alloc_tensor(WeightTensorDesc {
                    name: &name,
                    dtype,
                    shape: &[count],
                    bytes,
                })
                .map_err(WeightLoadError::Backend)?;
            match transform {
                Transform::Fp8 { .. } => unreachable!(),
                Transform::Scales {
                    source,
                    n,
                    k,
                    count,
                } => {
                    if self.provider.is_none() {
                        self.provider = Some(Qsfi::new(ctx).map_err(WeightLoadError::Backend)?);
                    }
                    let input = DeviceSpan::<Fp8E4M3>::new(
                        sources[*source].ptr.cast(),
                        sources[*source].bytes,
                    )
                    .map_err(WeightLoadError::Backend)?;
                    let output = DeviceSpan::<Fp8E4M3>::new(span.ptr.cast(), span.bytes)
                        .map_err(WeightLoadError::Backend)?;
                    unsafe {
                        self.provider.as_mut().unwrap().nvfp4_swizzle_scales(
                            input.matrix(*n, k / 16).map_err(WeightLoadError::Backend)?,
                            output.vector(*count).map_err(WeightLoadError::Backend)?,
                        )
                    }
                    .map_err(WeightLoadError::Backend)?;
                    self.allocations.push(Allocation {
                        destination: Destination::Tensor(*source),
                        span,
                        dtype,
                    });
                }
                Transform::Parameters { name, host } => {
                    // HostBuffer storage is retained until backend.finish establishes completion.
                    result_from_cuda(unsafe {
                        cuda::cudaMemcpyAsync(
                            span.ptr,
                            host.as_ref().as_ptr().cast(),
                            host.as_ref().len(),
                            cuda::CUDA_MEMCPY_HOST_TO_DEVICE,
                            ctx.stream,
                        )
                    })
                    .map_err(WeightLoadError::Backend)?;
                    self.allocations.push(Allocation {
                        destination: Destination::Parameters(name.clone()),
                        span,
                        dtype,
                    });
                }
            }
        }
        Ok(())
    }

    pub(super) fn commit<B: WeightLoadBackend>(
        self,
        tensors: &mut [LoadedWeightTensor<B>],
        buffers: Vec<WeightBuffer<B>>,
    ) -> LoadResult<BTreeMap<String, WeightBuffer<B>>> {
        if buffers.len() != self.allocations.len() {
            return Err(WeightLoadError::tensor_table(
                "prepared allocation count mismatch",
            ));
        }
        let mut scale_parameters = BTreeMap::new();
        for (allocation, buffer) in self.allocations.into_iter().zip(buffers) {
            if buffer.dtype() != allocation.dtype || !buffer.matches_span(allocation.span) {
                return Err(WeightLoadError::tensor_table(
                    "prepared allocation mismatch",
                ));
            }
            match allocation.destination {
                // The canonical allocation is retired only after the load-end wait.
                Destination::Tensor(index) => tensors[index].buffer = buffer,
                Destination::Parameters(name) => {
                    scale_parameters.insert(name, buffer);
                }
            }
        }
        Ok(scale_parameters)
    }
}

/// Qwen's FP8 groups in vLLM's Qwen3_5Model. Keep the checkpoint's split storage,
/// but use the same represented weights and activation scales as fused serving.
fn fp8_group(name: &str) -> LoadResult<Vec<String>> {
    for suffixes in [
        &[".linear_attn.in_proj_qkv", ".linear_attn.in_proj_z"][..],
        &[
            ".self_attn.q_proj",
            ".self_attn.k_proj",
            ".self_attn.v_proj",
        ][..],
    ] {
        for suffix in suffixes {
            if let Some(prefix) = name.strip_suffix(suffix) {
                return Ok(suffixes.iter().map(|s| format!("{prefix}{s}")).collect());
            }
        }
    }
    if name.ends_with(".linear_attn.out_proj") || name.ends_with(".self_attn.o_proj") {
        return Ok(vec![name.into()]);
    }
    Err(WeightLoadError::tensor_table(format!(
        "unexpected Qwen FP8 projection {name}"
    )))
}
