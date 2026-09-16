use super::{
    format::{LoadResult, WeightLoadError},
    plan::LoadedWeightPlan,
    transfer::{WeightBuffer, WeightLoadBackend},
};
use crate::{
    engine::Status,
    model::{QwenConfig, QwenWeights},
};
use std::collections::BTreeMap;

impl<B: WeightLoadBackend> LoadedWeightPlan<B> {
    pub(crate) fn into_qwen_model(self, max_seq_len: u32) -> LoadResult<(QwenConfig, QwenWeights)> {
        self.config.validate_compiled_model()?;
        let config = QwenConfig::new(max_seq_len).map_err(|status| {
            WeightLoadError::invalid_config(format!("invalid runtime resources: {status:?}"))
        })?;
        self.into_model_with_config(config)
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
        self.into_model_with_config(config)
    }

    fn into_model_with_config(self, config: QwenConfig) -> LoadResult<(QwenConfig, QwenWeights)> {
        let mut tensors_by_target = BTreeMap::new();
        for tensor in self.tensors {
            let key = (tensor.spec.target.layer, tensor.spec.target.slot);
            let WeightBuffer::Bf16(buffer) = tensor.buffer else {
                return Err(WeightLoadError::tensor_table(
                    "BF16 model assembly requires BF16 tensor storage",
                ));
            };
            tensors_by_target.insert(key, buffer);
        }

        let mut missing = None;
        let weights = QwenWeights::from_bf16_allocations(config, |layer, slot| {
            tensors_by_target.remove(&(layer, slot)).ok_or_else(|| {
                missing = Some((layer, slot));
                Status::InternalError
            })
        })
        .map_err(|status| {
            let detail = missing
                .map(|(layer, slot)| format!("missing target {layer:?}/{slot}"))
                .unwrap_or_else(|| format!("model factory failed with {status:?}"));
            WeightLoadError::tensor_table(detail)
        })?;

        if let Some(((layer, slot), _)) = tensors_by_target.into_iter().next() {
            return Err(WeightLoadError::tensor_table(format!(
                "unused loaded target {layer:?}/{slot}"
            )));
        }
        Ok((config, weights))
    }
}
