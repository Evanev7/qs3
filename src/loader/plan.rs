use super::format::{
    LoadResult, Qwen36TextConfig, WeightLoadError, WeightTensorSpec, validate_qwen36_bf16_dir,
};
use super::transfer::{WeightFileRange, WeightLoadBackend, WeightLoadSpan, WeightTensorDesc};
use super::{DEFAULT_MAX_HEADER_BYTES, DEFAULT_MAX_JSON_BYTES};

use std::{
    collections::{BTreeMap, BTreeSet},
    ffi::c_void,
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
pub(crate) struct QwenBf16LoadPlan {
    pub(super) config: Qwen36TextConfig,
    pub(super) entries: Vec<QwenLoadPlanEntry>,
}

impl QwenBf16LoadPlan {
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
        let validated = validate_qwen36_bf16_dir(model_dir, max_json_bytes, max_header_bytes)?;
        let mut entries = Vec::with_capacity(validated.tensors.len());
        for tensor in validated.tensors {
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
            config: validated.config,
            entries,
        })
    }
}

#[derive(Debug)]
pub(crate) struct LoadedWeightTensor {
    pub(super) spec: WeightTensorSpec,
    pub(super) span: WeightLoadSpan,
}

#[derive(Debug)]
pub(crate) struct LoadedWeightPlan<B: WeightLoadBackend> {
    pub(super) config: Qwen36TextConfig,
    pub(super) tensors: Vec<LoadedWeightTensor>,
    pub(super) backend: B,
}

pub(crate) fn execute_qwen36_bf16_load_plan<B: WeightLoadBackend>(
    plan: &QwenBf16LoadPlan,
    mut backend: B,
    stream: *mut c_void,
) -> LoadResult<LoadedWeightPlan<B>> {
    let files = open_plan_files(plan)?;
    let mut tensors = Vec::with_capacity(plan.entries.len());
    for entry in &plan.entries {
        let bytes = match &entry.source {
            QwenLoadSource::FileRange { bytes, .. } | QwenLoadSource::ZeroFill { bytes } => *bytes,
        };
        let span = backend
            .alloc_tensor(WeightTensorDesc {
                name: &entry.spec.name,
                dtype: entry.spec.dtype.to_runtime_dtype(),
                shape: &entry.spec.shape,
                bytes,
            })
            .map_err(WeightLoadError::Backend)?;
        tensors.push(LoadedWeightTensor {
            spec: entry.spec.clone(),
            span,
        });
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
                &tensors[idx].span,
                stream,
            )
            .map_err(WeightLoadError::Backend)?;
    }

    for idx in zero_order {
        backend
            .zero_fill(&tensors[idx].span, stream)
            .map_err(WeightLoadError::Backend)?;
    }

    backend.seal(stream).map_err(WeightLoadError::Backend)?;
    Ok(LoadedWeightPlan {
        config: plan.config.clone(),
        tensors,
        backend,
    })
}

fn open_plan_files(plan: &QwenBf16LoadPlan) -> LoadResult<BTreeMap<PathBuf, File>> {
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
