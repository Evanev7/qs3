use super::{
    format::{LoadResult, WeightLoadError, WeightTensorDType},
    plan::LoadedWeightPlan,
    transfer::WeightLoadBackend,
};
use crate::{
    engine::Status,
    model::{QwenConfig, QwenWeights},
};
use std::collections::{BTreeMap, BTreeSet};
use std::mem;

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
        let LoadedWeightPlan {
            tensors,
            mut backend,
            ..
        } = self;

        if tensors.len() != backend.allocations().len() {
            return Err(WeightLoadError::tensor_table(format!(
                "loaded tensor count {} does not match backend allocation count {}",
                tensors.len(),
                backend.allocations().len()
            )));
        }
        let mut targets = BTreeSet::new();
        let mut pointers = BTreeSet::new();
        for (tensor, allocation) in tensors.iter().zip(backend.allocations()) {
            let expected_bytes = tensor.spec.byte_len()?;
            if tensor.spec.dtype != WeightTensorDType::Bf16
                || tensor.span != *allocation
                || allocation.ptr.is_null()
                || allocation.bytes != expected_bytes
                || allocation.bytes % mem::size_of::<u16>() != 0
            {
                return Err(WeightLoadError::tensor_table(format!(
                    "loaded allocation does not match tensor {:?}",
                    tensor.spec.name
                )));
            }
            let key = (tensor.spec.target.layer, tensor.spec.target.slot);
            if !targets.insert(key) {
                return Err(WeightLoadError::tensor_table(format!(
                    "duplicate loaded target {:?}/{}",
                    key.0, key.1
                )));
            }
            if !pointers.insert(allocation.ptr as usize) {
                return Err(WeightLoadError::tensor_table(format!(
                    "duplicate loaded allocation pointer for tensor {:?}",
                    tensor.spec.name
                )));
            }
        }

        let expected_allocations = backend.allocations().to_vec();
        let allocations = backend.take_allocations();
        if allocations.len() != expected_allocations.len()
            || allocations
                .iter()
                .zip(&expected_allocations)
                .any(|(owner, expected)| {
                    owner.erase() != expected.ptr
                        || owner.cap.checked_mul(mem::size_of::<u16>()) != Some(expected.bytes)
                })
        {
            return Err(WeightLoadError::tensor_table(
                "backend returned allocation list does not match validated allocations",
            ));
        }
        if !backend.allocations().is_empty() {
            return Err(WeightLoadError::tensor_table(
                "backend retained allocations after transfer",
            ));
        }
        let mut allocations_by_target: BTreeMap<(Option<u32>, &'static str), B::Allocation> =
            BTreeMap::new();

        for (tensor, allocation) in tensors.into_iter().zip(allocations) {
            let key = (tensor.spec.target.layer, tensor.spec.target.slot);
            let replaced = allocations_by_target.insert(key, allocation);
            debug_assert!(replaced.is_none());
        }
        drop(backend);

        let mut missing = None;
        let weights = QwenWeights::from_bf16_allocations(config, |layer, slot| {
            allocations_by_target.remove(&(layer, slot)).ok_or_else(|| {
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

        if let Some(((layer, slot), _)) = allocations_by_target.into_iter().next() {
            return Err(WeightLoadError::tensor_table(format!(
                "unused loaded target {layer:?}/{slot}"
            )));
        }
        Ok((config, weights))
    }
}
