use super::{
    format::{LoadResult, WeightLoadError, WeightTensorSource, WeightTensorSpec},
    plan::LoadedWeightPlan,
    quantization::{ProjectionQuantization, Quantization},
    schema::{self, Tensors},
    transfer::{WeightBuffer, WeightLoadBackend},
};
use crate::{
    dtype::DType,
    memory::DeviceSpan,
    model::{QwenConfig, QwenWeights, weights::W},
};
use std::{collections::BTreeMap, ops::Deref};

/// Retains the backend's release policy, including U8 storage viewed as E2M1.
struct TensorOwner<B: WeightLoadBackend, D: DType> {
    _owner: WeightBuffer<B>,
    span: DeviceSpan<D>,
}
impl<B: WeightLoadBackend, D: DType> Deref for TensorOwner<B, D> {
    type Target = DeviceSpan<D>;
    fn deref(&self) -> &Self::Target {
        &self.span
    }
}
struct LoadedTensors<B: WeightLoadBackend> {
    tensors: BTreeMap<String, (WeightTensorSpec, WeightBuffer<B>)>,
    quantization: Option<Quantization>,
    // Generated Fp8Scales/Nvfp4Scales storage, keyed by projection name.
    scale_parameters: BTreeMap<String, WeightBuffer<B>>,
}
impl<B: WeightLoadBackend + 'static> Tensors for LoadedTensors<B> {
    type Tensor<D: DType> = W<D>;
    fn scale_parameters<D: DType>(&mut self, name: &str) -> LoadResult<W<D>> {
        let owner = self.scale_parameters.remove(name).ok_or_else(|| {
            WeightLoadError::tensor_table(format!("missing prepared scale parameters {name}"))
        })?;
        let WeightBuffer::F32(buffer) = &owner else {
            return Err(WeightLoadError::tensor_table(
                "prepared scale parameters dtype mismatch",
            ));
        };
        if crate::dtype::F32::size_of(buffer.len).map_err(WeightLoadError::Backend)?
            != D::size_of(1).map_err(WeightLoadError::Backend)?
        {
            return Err(WeightLoadError::tensor_table(
                "prepared scale parameters size mismatch",
            ));
        }
        let span = DeviceSpan::new(buffer.as_raw(), 1).map_err(WeightLoadError::Backend)?;
        Ok(Box::new(TensorOwner {
            _owner: owner,
            span,
        }))
    }
    fn tensor<D: DType>(
        &mut self,
        name: &str,
        shape: &[u32],
        source: WeightTensorSource,
    ) -> LoadResult<W<D>> {
        let (spec, owner) = self.tensors.remove(name).ok_or_else(|| {
            WeightLoadError::tensor_table(format!("missing loaded tensor {name}"))
        })?;
        if spec != schema::tensor_spec::<D>(name, shape, source) {
            return Err(WeightLoadError::tensor_table(format!(
                "loaded tensor descriptor mismatch: {name}"
            )));
        }
        let ptr = match &owner {
            WeightBuffer::Bf16(b) => b.erase(),
            WeightBuffer::Fp8E4m3(b) => b.erase(),
            WeightBuffer::U8(b) => b.erase(),
            WeightBuffer::F32(b) => b.erase(),
        };
        let len = if self.quantization.is_some() && name.ends_with(".weight_scale") {
            let [n, cols] = shape else {
                return Err(WeightLoadError::tensor_table("invalid block scale shape"));
            };
            crate::backend::qsfi::scale_count(
                *n,
                cols.checked_mul(16)
                    .ok_or_else(|| WeightLoadError::tensor_table("scale extent overflow"))?,
            )
            .map_err(WeightLoadError::Backend)? as usize
        } else {
            D::len_of(spec.byte_len()?).map_err(WeightLoadError::Backend)?
        };
        let span = DeviceSpan::new(ptr.cast(), len).map_err(WeightLoadError::Backend)?;
        Ok(Box::new(TensorOwner {
            _owner: owner,
            span,
        }))
    }
    fn scalar(&mut self, name: &str) -> LoadResult<f32> {
        self.quantization
            .as_mut()
            .and_then(|q| q.globals.remove(name))
            .ok_or_else(|| WeightLoadError::tensor_table(format!("missing loaded scalar {name}")))
    }
    fn recipe(&mut self, name: &str) -> LoadResult<ProjectionQuantization> {
        self.quantization
            .as_ref()
            .ok_or_else(|| WeightLoadError::invalid_config("missing mixed recipe"))?
            .recipe(name)
    }
}

impl<B: WeightLoadBackend + 'static> LoadedWeightPlan<B> {
    pub(crate) fn into_qwen_model(self, max_seq_len: u32) -> LoadResult<(QwenConfig, QwenWeights)> {
        self.config.validate_compiled_model()?;
        let config = QwenConfig::new(max_seq_len).map_err(|s| {
            WeightLoadError::invalid_config(format!("invalid runtime resources: {s:?}"))
        })?;
        self.into_model(config)
    }
    #[cfg(test)]
    pub(super) fn into_fixture_model(
        self,
        max_seq_len: u32,
    ) -> LoadResult<(QwenConfig, QwenWeights)> {
        let config = QwenConfig::loaded_fixture(
            self.config.num_hidden_layers,
            self.config.vocab_size,
            max_seq_len,
        );
        self.into_model(config)
    }
    fn into_model(self, config: QwenConfig) -> LoadResult<(QwenConfig, QwenWeights)> {
        let mut tensors = BTreeMap::new();
        for tensor in self.tensors {
            if tensors
                .insert(tensor.spec.name.clone(), (tensor.spec, tensor.buffer))
                .is_some()
            {
                return Err(WeightLoadError::tensor_table("duplicate loaded tensor"));
            }
        }
        let mut s = LoadedTensors {
            tensors,
            quantization: self.quantization,
            scale_parameters: self.scale_parameters,
        };
        let weights = if s.quantization.is_none() {
            if self.config.num_experts > 0 {
                QwenWeights::MoeBf16(schema::moe_bf16_model(&mut s, &self.config)?)
            } else {
                QwenWeights::DenseBf16(schema::dense_bf16_model(&mut s, &self.config)?)
            }
        } else if self.config.num_experts > 0 {
            QwenWeights::MoeNvfp4(schema::moe_nvfp4_model(&mut s, &self.config)?)
        } else {
            QwenWeights::DenseNvfp4(schema::dense_nvfp4_model(&mut s, &self.config)?)
        };
        if !s.scale_parameters.is_empty()
            || !s.tensors.is_empty()
            || s.quantization
                .as_ref()
                .is_some_and(|q| !q.globals.is_empty())
        {
            return Err(WeightLoadError::tensor_table(
                "unused loaded tensors or scales",
            ));
        }
        Ok((config, weights))
    }
}
