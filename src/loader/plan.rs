use super::format::{LoadResult, Qwen36TextConfig, WeightLoadError, WeightTensorSpec};
use super::format::{
    SafetensorsIndex, read_indexed_safetensors_table, read_json_object_with_limit,
    validate_tensor_specs,
};
use super::quantization::Quantization;
use super::transfer::{
    WeightBuffer, WeightFileRange, WeightLoadBackend, WeightLoadSpan, WeightTensorDesc,
};
use super::{DEFAULT_MAX_HEADER_BYTES, DEFAULT_MAX_JSON_BYTES};

use crate::memory::CudaCtx;

use std::{
    collections::{BTreeMap, BTreeSet},
    fs::File,
    path::{Path, PathBuf},
};

#[derive(Clone, Debug)]
pub(crate) enum QwenLoadSource {
    FileRange {
        shard_path: PathBuf,
        absolute_offset: u64,
        bytes: usize,
    },
    ZeroFill {
        bytes: usize,
    },
}

#[derive(Clone, Debug)]
pub(crate) struct QwenLoadPlanEntry {
    pub(super) spec: WeightTensorSpec,
    pub(super) source: QwenLoadSource,
}

#[derive(Clone, Debug)]
pub(crate) struct QwenLoadPlan {
    pub(super) config: Qwen36TextConfig,
    pub(super) entries: Vec<QwenLoadPlanEntry>,
    pub(super) quantization: Option<Quantization>,
}

impl QwenLoadPlan {
    pub(crate) fn read(model_dir: impl AsRef<Path>) -> LoadResult<Self> {
        Self::read_with_limits(model_dir, DEFAULT_MAX_JSON_BYTES, DEFAULT_MAX_HEADER_BYTES)
    }

    pub(crate) fn tensor_count(&self) -> usize {
        self.entries.len()
    }

    pub(crate) fn file_bytes(&self) -> LoadResult<usize> {
        self.entries.iter().try_fold(0usize, |acc, entry| {
            let bytes = match &entry.source {
                QwenLoadSource::FileRange { bytes, .. } => *bytes,
                QwenLoadSource::ZeroFill { .. } => 0,
            };
            acc.checked_add(bytes)
                .ok_or_else(|| WeightLoadError::tensor_table("file byte count overflow"))
        })
    }

    pub(crate) fn zero_fill_bytes(&self) -> LoadResult<usize> {
        self.entries.iter().try_fold(0usize, |acc, entry| {
            let bytes = match &entry.source {
                QwenLoadSource::FileRange { .. } => 0,
                QwenLoadSource::ZeroFill { bytes } => *bytes,
            };
            acc.checked_add(bytes)
                .ok_or_else(|| WeightLoadError::tensor_table("zero-fill byte count overflow"))
        })
    }

    pub(crate) fn total_bytes(&self) -> LoadResult<usize> {
        self.file_bytes()?
            .checked_add(self.zero_fill_bytes()?)
            .ok_or_else(|| WeightLoadError::tensor_table("total byte count overflow"))
    }

    pub(super) fn read_with_limits(
        model_dir: impl AsRef<Path>,
        max_json_bytes: usize,
        max_header_bytes: usize,
    ) -> LoadResult<Self> {
        let model_dir = model_dir.as_ref();
        let root = read_json_object_with_limit(model_dir.join(super::CONFIG_FILE), max_json_bytes)?;
        let config = Qwen36TextConfig::from_config_object(&root)?;
        config.validate_compiled_model()?;
        let mut quantization = Quantization::parse(&root)?;
        let specs = super::schema::expected_specs(&config, quantization.as_ref())?;
        let index = SafetensorsIndex::read_with_limit(model_dir, max_json_bytes)?;
        let table = read_indexed_safetensors_table(model_dir, &index, max_header_bytes)?;
        let tensors = validate_tensor_specs(&table, specs)?;
        if let Some(q) = &mut quantization {
            q.read_scales(model_dir, &tensors)?;
        }
        let mut entries = Vec::with_capacity(tensors.len());
        for tensor in tensors {
            if quantization.is_some() && tensor.spec.dtype == crate::dtype::DynDType::F32 {
                continue;
            }
            let bytes = tensor.spec.byte_len()?;
            let source = match tensor.file_meta {
                Some(file_meta) => QwenLoadSource::FileRange {
                    shard_path: model_dir.join(file_meta.shard),
                    absolute_offset: file_meta.absolute_offset,
                    bytes,
                },
                None => QwenLoadSource::ZeroFill { bytes },
            };
            entries.push(QwenLoadPlanEntry {
                spec: tensor.spec,
                source,
            });
        }
        Ok(Self {
            config,
            entries,
            quantization,
        })
    }
}

pub(crate) struct LoadedWeightTensor<B: WeightLoadBackend> {
    pub(super) spec: WeightTensorSpec,
    pub(super) buffer: WeightBuffer<B>,
}

pub(crate) struct LoadedWeightPlan<B: WeightLoadBackend> {
    pub(super) config: Qwen36TextConfig,
    pub(super) tensors: Vec<LoadedWeightTensor<B>>,
    pub(super) stats: B::Stats,
    pub(super) scale_parameters: BTreeMap<String, WeightBuffer<B>>,
    pub(super) quantization: Option<Quantization>,
}

pub(crate) fn execute_qwen_load_plan<B: WeightLoadBackend>(
    plan: &QwenLoadPlan,
    mut backend: B,
    ctx: &CudaCtx,
) -> LoadResult<LoadedWeightPlan<B>> {
    let mut names = BTreeSet::new();
    for entry in &plan.entries {
        if !names.insert(&entry.spec.name) {
            return Err(WeightLoadError::tensor_table("duplicate loaded tensor"));
        }
    }
    let mut preparation = super::prepare::Preparation::plan(plan)?;
    let files = open_plan_files(plan)?;
    let mut spans = Vec::with_capacity(plan.entries.len());
    for entry in &plan.entries {
        let bytes = match &entry.source {
            QwenLoadSource::FileRange { bytes, .. } | QwenLoadSource::ZeroFill { bytes } => *bytes,
        };
        let span = backend
            .alloc_tensor(WeightTensorDesc {
                name: &entry.spec.name,
                dtype: entry.spec.dtype,
                shape: &entry.spec.shape,
                bytes,
            })
            .map_err(WeightLoadError::Backend)?;
        spans.push(span);
    }

    let mut read_order = Vec::new();
    let mut zero_order = Vec::new();
    for (idx, entry) in plan.entries.iter().enumerate() {
        match &entry.source {
            QwenLoadSource::FileRange { .. } => read_order.push(idx),
            QwenLoadSource::ZeroFill { .. } => zero_order.push(idx),
        }
    }
    read_order.sort_by(
        |&lhs, &rhs| match (&plan.entries[lhs].source, &plan.entries[rhs].source) {
            (
                QwenLoadSource::FileRange {
                    shard_path: lhs_shard,
                    absolute_offset: lhs_offset,
                    ..
                },
                QwenLoadSource::FileRange {
                    shard_path: rhs_shard,
                    absolute_offset: rhs_offset,
                    ..
                },
            ) => lhs_shard
                .cmp(rhs_shard)
                .then_with(|| lhs_offset.cmp(rhs_offset)),
            _ => std::cmp::Ordering::Equal,
        },
    );

    for idx in read_order {
        let QwenLoadSource::FileRange {
            shard_path,
            absolute_offset,
            bytes,
        } = &plan.entries[idx].source
        else {
            unreachable!();
        };
        let file = files.get(shard_path).ok_or_else(|| {
            WeightLoadError::Io(format!(
                "load plan file {} was not opened",
                shard_path.display()
            ))
        })?;
        backend
            .read_exact(
                WeightFileRange {
                    file,
                    offset: *absolute_offset,
                    bytes: *bytes,
                },
                &spans[idx],
                ctx.stream,
            )
            .map_err(WeightLoadError::Backend)?;
    }

    for idx in zero_order {
        backend
            .zero_fill(&spans[idx], ctx.stream)
            .map_err(WeightLoadError::Backend)?;
    }

    let prepared = preparation.enqueue(&mut backend, &spans, ctx);
    // Finish also settles any transfers submitted before a preparation error.
    // An unsuccessful finish does not establish that host uploads are complete.
    let finished = backend.finish(ctx.stream);
    let (mut buffers, stats) = match finished {
        Ok(result) => result,
        Err(status) => {
            std::mem::forget(preparation);
            return Err(WeightLoadError::Backend(status));
        }
    };
    prepared?;
    if buffers.len() < plan.entries.len() {
        return Err(WeightLoadError::tensor_table("missing loaded buffers"));
    }
    let extra = buffers.split_off(plan.entries.len());
    let mut tensors = pair_loaded_tensors(plan, spans, buffers)?;
    let scale_parameters = preparation.commit(&mut tensors, extra)?;
    Ok(LoadedWeightPlan {
        config: plan.config.clone(),
        tensors,
        stats,
        scale_parameters,
        quantization: plan.quantization.clone(),
    })
}

fn pair_loaded_tensors<B: WeightLoadBackend>(
    plan: &QwenLoadPlan,
    spans: Vec<WeightLoadSpan>,
    buffers: Vec<WeightBuffer<B>>,
) -> LoadResult<Vec<LoadedWeightTensor<B>>> {
    if buffers.len() != plan.entries.len() {
        return Err(WeightLoadError::tensor_table(
            "loaded tensor count does not match plan",
        ));
    }
    plan.entries
        .iter()
        .zip(spans)
        .zip(buffers)
        .map(|((entry, span), buffer)| {
            if buffer.dtype() != entry.spec.dtype
                || entry.spec.byte_len()? != span.bytes
                || !buffer.matches_span(span)
            {
                return Err(WeightLoadError::tensor_table(format!(
                    "returned buffer does not match tensor {:?}",
                    entry.spec.name,
                )));
            }
            Ok(LoadedWeightTensor {
                spec: entry.spec.clone(),
                buffer,
            })
        })
        .collect()
}

fn open_plan_files(plan: &QwenLoadPlan) -> LoadResult<BTreeMap<PathBuf, File>> {
    let mut paths = BTreeSet::new();
    for entry in &plan.entries {
        if let QwenLoadSource::FileRange { shard_path, .. } = &entry.source {
            paths.insert(shard_path.clone());
        }
    }
    let mut files = BTreeMap::new();
    for path in paths {
        files.insert(path.clone(), File::open(path)?);
    }
    Ok(files)
}
