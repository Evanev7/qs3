//! Qwen ModelOpt mixed NVFP4/FP8 checkpoint contract, before CUDA allocation.
use super::format::*;
use crate::dtype::DynDType;
use std::{
    collections::{BTreeMap, HashMap},
    fs::File,
    os::unix::fs::FileExt,
    path::Path,
};
use tinyjson::JsonValue;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(super) enum ProjectionQuantization {
    Fp8,
    Nvfp4,
    W4A16Nvfp4,
}

#[derive(Clone, Debug)]
pub(super) struct Quantization {
    pub projections: BTreeMap<String, ProjectionQuantization>,
    pub globals: BTreeMap<String, f32>,
}

fn object<'a>(value: &'a JsonValue, label: &str) -> LoadResult<&'a HashMap<String, JsonValue>> {
    value
        .get()
        .ok_or_else(|| WeightLoadError::invalid_config(format!("{label} must be an object")))
}

impl Quantization {
    pub fn parse(root: &HashMap<String, JsonValue>) -> LoadResult<Option<Self>> {
        let Some(value) = root.get("quantization_config") else {
            return Ok(None);
        };
        let q = object(value, "quantization_config")?;
        if string_field(q, "quant_method")? != "modelopt"
            || string_field(q, "quant_algo")? != "MIXED_PRECISION"
        {
            return Err(WeightLoadError::invalid_config(
                "expected ModelOpt MIXED_PRECISION",
            ));
        }
        // This map is authoritative. Older ModelOpt config_groups incorrectly
        // declares A4 for the pinned 35B W4A16 checkpoint.
        let map = object(
            q.get("quantized_layers")
                .ok_or_else(|| WeightLoadError::invalid_config("missing quantized_layers"))?,
            "quantized_layers",
        )?;
        let mut projections = BTreeMap::new();
        for (name, value) in map {
            let entry = object(value, name)?;
            let recipe = match string_field(entry, "quant_algo")?.as_str() {
                "FP8" if entry.len() == 1 => ProjectionQuantization::Fp8,
                algo @ ("NVFP4" | "W4A16_NVFP4")
                    if entry.len() == 2 && u32_field(entry, "group_size")? == 16 =>
                {
                    if algo == "NVFP4" {
                        ProjectionQuantization::Nvfp4
                    } else {
                        ProjectionQuantization::W4A16Nvfp4
                    }
                }
                _ => {
                    return Err(WeightLoadError::invalid_config(format!(
                        "unsupported quantization recipe for {name}"
                    )));
                }
            };
            projections.insert(name.clone(), recipe);
        }
        Ok(Some(Self {
            projections,
            globals: BTreeMap::new(),
        }))
    }

    pub fn recipe_name(projection: &str) -> String {
        match projection.split_once(".experts.") {
            Some((prefix, _)) => format!("{prefix}.experts"),
            None => projection.to_owned(),
        }
    }

    pub fn recipe(&self, projection: &str) -> LoadResult<ProjectionQuantization> {
        let key = Self::recipe_name(projection);
        self.projections.get(&key).copied().ok_or_else(|| {
            WeightLoadError::invalid_config(format!("missing quantization recipe for {key}"))
        })
    }

    /// All headers have passed before reading values. Validate scale payloads
    /// before allocating any CUDA-visible storage, including pinned staging.
    pub fn read_scales(&mut self, directory: &Path, tensors: &[ValidatedTensor]) -> LoadResult<()> {
        let mut files = BTreeMap::new();
        let mut scratch = vec![0u8; 1 << 20];
        for tensor in tensors {
            let scalar = tensor.spec.dtype == DynDType::F32;
            let block = tensor.spec.name.ends_with(".weight_scale")
                && tensor.spec.dtype == DynDType::FP8E4M3;
            if !scalar && !block {
                continue;
            }
            let meta = tensor
                .file_meta
                .as_ref()
                .ok_or_else(|| WeightLoadError::tensor_table("scale has no source"))?;
            if !files.contains_key(&meta.shard) {
                files.insert(meta.shard.clone(), File::open(directory.join(&meta.shard))?);
            }
            let file = &files[&meta.shard];
            if scalar {
                let mut bytes = [0u8; 4];
                file.read_exact_at(&mut bytes, meta.absolute_offset)?;
                let value = f32::from_le_bytes(bytes);
                if !value.is_finite() || value <= 0.0 {
                    return Err(WeightLoadError::tensor_table(format!(
                        "{} must be finite and positive",
                        tensor.spec.name
                    )));
                }
                self.globals.insert(tensor.spec.name.clone(), value);
            } else {
                let mut remaining = tensor.spec.byte_len()?;
                let mut offset = meta.absolute_offset;
                while remaining > 0 {
                    let count = remaining.min(scratch.len());
                    file.read_exact_at(&mut scratch[..count], offset)?;
                    // E4M3FN: sign bit and 0x7f NaN are invalid scales. +0 is
                    // valid for zero blocks; there are no infinity encodings.
                    if scratch[..count].iter().any(|v| v & 0x80 != 0 || *v == 0x7f) {
                        return Err(WeightLoadError::tensor_table(format!(
                            "invalid E4M3 block scale in {}",
                            tensor.spec.name
                        )));
                    }
                    remaining -= count;
                    offset += count as u64;
                }
            }
        }
        Ok(())
    }
}
