use super::{
    CONFIG_FILE, DEFAULT_MAX_HEADER_BYTES, DEFAULT_MAX_JSON_BYTES, SAFETENSORS_INDEX_FILE,
};
use crate::{
    constants::{
        attention::{HEAD_DIM, NUM_KV_HEADS, NUM_Q_HEADS, ROTARY_DIM},
        gdn::{CONV_WIDTH, KEY_HEAD_DIM, NUM_KEY_HEADS, NUM_VALUE_HEADS, VALUE_HEAD_DIM},
        mlp::{
            HAS_EXPERTS, INTERMEDIATE_SIZE, NUM_EXPERTS, NUM_EXPERTS_PER_TOKEN,
            SHARED_EXPERT_INTERMEDIATE_SIZE,
        },
        model::HIDDEN_SIZE,
    },
    dtype::DynDType,
    engine::Status,
};

use std::{
    collections::{BTreeMap, BTreeSet, HashMap, HashSet},
    fmt,
    fs::File,
    io::{self, Read},
    path::{Component, Path},
};
use tinyjson::JsonValue;

pub(super) fn read_indexed_safetensors_table(
    model_dir: &Path,
    index: &SafetensorsIndex,
    max_header_bytes: usize,
) -> LoadResult<BTreeMap<String, TensorFileMeta>> {
    let mut table = BTreeMap::new();
    for shard in index.shard_names() {
        let header = SafetensorsHeader::read_with_limit(model_dir.join(&shard), max_header_bytes)?;
        for (name, meta) in header.tensors {
            if ignored_qwen36_tensor(&name) {
                continue;
            }
            match index.weight_map.get(&name) {
                Some(mapped_shard) if mapped_shard == &shard => {}
                Some(mapped_shard) => {
                    return Err(WeightLoadError::invalid_index(format!(
                        "tensor {name:?} is in shard {shard:?}, but index maps it to {mapped_shard:?}"
                    )));
                }
                None => {
                    return Err(WeightLoadError::invalid_index(format!(
                        "header tensor {name:?} is missing from weight_map"
                    )));
                }
            }
            let absolute_offset = header
                .data_start
                .checked_add(meta.data_offsets.0)
                .ok_or_else(|| {
                    WeightLoadError::invalid_safetensors(format!(
                        "absolute offset overflow for tensor {name:?}"
                    ))
                })?;
            if table
                .insert(
                    name,
                    TensorFileMeta {
                        shard: shard.clone(),
                        absolute_offset,
                        meta,
                    },
                )
                .is_some()
            {
                return Err(WeightLoadError::tensor_table(format!(
                    "duplicate tensor across shards in {shard:?}"
                )));
            }
        }
    }
    for (name, shard) in &index.weight_map {
        if ignored_qwen36_tensor(name) {
            continue;
        }
        if !table.contains_key(name) {
            return Err(WeightLoadError::invalid_index(format!(
                "index maps tensor {name:?} to shard {shard:?}, but the shard header does not contain it"
            )));
        }
    }
    Ok(table)
}

#[derive(Debug)]
pub(crate) enum WeightLoadError {
    Io(String),
    Json(String),
    InvalidConfig(String),
    InvalidIndex(String),
    InvalidSafetensors(String),
    TensorTable(String),
    Backend(Status),
}

impl WeightLoadError {
    pub(super) fn json(message: impl Into<String>) -> Self {
        Self::Json(message.into())
    }

    pub(super) fn invalid_config(message: impl Into<String>) -> Self {
        Self::InvalidConfig(message.into())
    }

    pub(super) fn invalid_index(message: impl Into<String>) -> Self {
        Self::InvalidIndex(message.into())
    }

    pub(super) fn invalid_safetensors(message: impl Into<String>) -> Self {
        Self::InvalidSafetensors(message.into())
    }

    pub(super) fn tensor_table(message: impl Into<String>) -> Self {
        Self::TensorTable(message.into())
    }
}

impl From<io::Error> for WeightLoadError {
    fn from(err: io::Error) -> Self {
        Self::Io(err.to_string())
    }
}

impl fmt::Display for WeightLoadError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Io(message) => write!(f, "io error: {message}"),
            Self::Json(message) => write!(f, "json error: {message}"),
            Self::InvalidConfig(message) => write!(f, "invalid qwen3.6 config: {message}"),
            Self::InvalidIndex(message) => write!(f, "invalid safetensors index: {message}"),
            Self::InvalidSafetensors(message) => write!(f, "invalid safetensors: {message}"),
            Self::TensorTable(message) => write!(f, "invalid tensor table: {message}"),
            Self::Backend(status) => write!(f, "weight load backend failed: {status:?}"),
        }
    }
}

impl std::error::Error for WeightLoadError {}

pub(super) type LoadResult<T> = Result<T, WeightLoadError>;

pub(super) fn parse_json_object(src: &str) -> LoadResult<HashMap<String, JsonValue>> {
    src.parse::<JsonValue>()
        .map_err(|err| WeightLoadError::json(err.to_string()))?
        .try_into()
        .map_err(|_| WeightLoadError::json("expected top-level object"))
}

pub(super) fn read_json_object_with_limit(
    path: impl AsRef<Path>,
    max_bytes: usize,
) -> LoadResult<HashMap<String, JsonValue>> {
    let path = path.as_ref();
    let mut file = File::open(path)?;
    let len = file.metadata()?.len();
    if len > max_bytes as u64 {
        return Err(WeightLoadError::json(format!(
            "{} is {len} bytes, limit is {max_bytes}",
            path.display()
        )));
    }
    let mut src = String::new();
    file.read_to_string(&mut src)?;
    parse_json_object(&src)
}

pub(super) fn string_field(object: &HashMap<String, JsonValue>, name: &str) -> LoadResult<String> {
    object
        .get(name)
        .and_then(JsonValue::get::<String>)
        .cloned()
        .ok_or_else(|| WeightLoadError::json(format!("field {name:?} must be a string")))
}

pub(super) fn opt_string_field(
    object: &HashMap<String, JsonValue>,
    name: &str,
) -> LoadResult<Option<String>> {
    match object.get(name) {
        Some(value) if value.is_null() => Ok(None),
        Some(value) => value
            .get::<String>()
            .cloned()
            .map(Some)
            .ok_or_else(|| WeightLoadError::json(format!("field {name:?} must be a string"))),
        None => Ok(None),
    }
}

pub(super) fn bool_field_with_default(
    object: &HashMap<String, JsonValue>,
    name: &str,
    default: bool,
) -> LoadResult<bool> {
    match object.get(name) {
        Some(value) if value.is_null() => Ok(default),
        Some(value) => value
            .get::<bool>()
            .copied()
            .ok_or_else(|| WeightLoadError::json(format!("field {name:?} must be a bool"))),
        None => Ok(default),
    }
}

pub(super) fn u64_from_value(value: &JsonValue, context: impl fmt::Display) -> LoadResult<u64> {
    const MAX_SAFE_JSON_INTEGER: f64 = 9_007_199_254_740_992.0;
    let raw = value
        .get::<f64>()
        .ok_or_else(|| WeightLoadError::json(format!("{context} must be an integer")))?;
    if !raw.is_finite() || *raw < 0.0 || raw.fract() != 0.0 || *raw > MAX_SAFE_JSON_INTEGER {
        return Err(WeightLoadError::json(format!(
            "{context} must be a non-negative JSON integer <= 2^53"
        )));
    }
    Ok(*raw as u64)
}

pub(super) fn u32_field(object: &HashMap<String, JsonValue>, name: &str) -> LoadResult<u32> {
    let value = object
        .get(name)
        .ok_or_else(|| WeightLoadError::json(format!("missing field {name:?}")))
        .and_then(|value| u64_from_value(value, format!("field {name:?}")))?;
    u32::try_from(value).map_err(|_| WeightLoadError::json(format!("field {name:?} overflows u32")))
}

pub(super) fn opt_u32_field(
    object: &HashMap<String, JsonValue>,
    name: &str,
) -> LoadResult<Option<u32>> {
    object
        .get(name)
        .map(|value| {
            let value = u64_from_value(value, format!("field {name:?}"))?;
            u32::try_from(value)
                .map_err(|_| WeightLoadError::json(format!("field {name:?} overflows u32")))
        })
        .transpose()
}

pub(super) fn f32_from_value(value: &JsonValue, context: impl fmt::Display) -> LoadResult<f32> {
    let raw = value
        .get::<f64>()
        .ok_or_else(|| WeightLoadError::json(format!("{context} must be a number")))?;
    if !raw.is_finite() || *raw < f32::MIN as f64 || *raw > f32::MAX as f64 {
        return Err(WeightLoadError::json(format!(
            "{context} must be a finite f32"
        )));
    }
    Ok(*raw as f32)
}

pub(super) fn f32_field_with_default(
    object: &HashMap<String, JsonValue>,
    name: &str,
    default: f32,
) -> LoadResult<f32> {
    match object.get(name) {
        Some(value) if value.is_null() => Ok(default),
        Some(value) => f32_from_value(value, format!("field {name:?}")),
        None => Ok(default),
    }
}

pub(super) fn string_array_field(
    object: &HashMap<String, JsonValue>,
    name: &str,
) -> LoadResult<Vec<String>> {
    let values = object
        .get(name)
        .and_then(JsonValue::get::<Vec<JsonValue>>)
        .ok_or_else(|| WeightLoadError::json(format!("field {name:?} must be an array")))?;
    let mut out = Vec::with_capacity(values.len());
    for value in values {
        let Some(value) = value.get::<String>() else {
            return Err(WeightLoadError::json(format!(
                "field {name:?} must contain only strings"
            )));
        };
        out.push(value.clone());
    }
    Ok(out)
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(super) enum QwenLayerKind {
    LinearAttention,
    FullAttention,
}

#[derive(Clone, Debug, PartialEq)]
pub(crate) struct Qwen36TextConfig {
    pub(super) num_hidden_layers: u32,
    pub(super) hidden_size: u32,
    pub(super) intermediate_size: u32,
    pub(super) vocab_size: u32,
    pub(super) num_attention_heads: u32,
    pub(super) num_key_value_heads: u32,
    pub(super) head_dim: u32,
    pub(super) rms_norm_eps: f32,
    pub(super) rope_theta: f32,
    pub(super) logits_soft_cap: f32,
    pub(super) num_experts: u32,
    pub(super) num_experts_per_tok: u32,
    pub(super) moe_intermediate_size: u32,
    pub(super) shared_expert_intermediate_size: u32,
    pub(super) linear_num_key_heads: u32,
    pub(super) linear_num_value_heads: u32,
    pub(super) linear_key_head_dim: u32,
    pub(super) linear_value_head_dim: u32,
    pub(super) linear_conv_kernel_dim: u32,
    pub(super) layer_types: Vec<QwenLayerKind>,
    pub(super) tie_word_embeddings: bool,
    pub(super) dtype: Option<String>,
    pub(super) attn_output_gate: bool,
    pub(super) full_attention_interval: Option<u32>,
    pub(super) partial_rotary_factor: Option<f32>,
}

impl Qwen36TextConfig {
    pub(super) fn read(model_dir: impl AsRef<Path>) -> LoadResult<Self> {
        Self::read_with_limit(model_dir, DEFAULT_MAX_JSON_BYTES)
    }

    pub(super) fn read_with_limit(
        model_dir: impl AsRef<Path>,
        max_bytes: usize,
    ) -> LoadResult<Self> {
        let root = read_json_object_with_limit(model_dir.as_ref().join(CONFIG_FILE), max_bytes)?;
        Self::from_config_object(&root)
    }

    pub(super) fn from_config_object(root: &HashMap<String, JsonValue>) -> LoadResult<Self> {
        let text = match root.get("text_config") {
            Some(value) => value
                .get::<HashMap<String, JsonValue>>()
                .ok_or_else(|| WeightLoadError::json("field \"text_config\" must be an object"))?,
            None => root,
        };
        let tie_word_embeddings = bool_field_with_default(
            text,
            "tie_word_embeddings",
            bool_field_with_default(root, "tie_word_embeddings", false)?,
        )?;
        let num_attention_heads = u32_field(text, "num_attention_heads")?;
        let hidden_size = u32_field(text, "hidden_size")?;
        let head_dim = u32_field(text, "head_dim")?;
        let (
            intermediate_size,
            num_experts,
            num_experts_per_tok,
            moe_intermediate_size,
            shared_expert_intermediate_size,
        ) = if HAS_EXPERTS {
            let moe_intermediate_size = u32_field(text, "moe_intermediate_size")?;
            (
                opt_u32_field(text, "intermediate_size")?.unwrap_or(moe_intermediate_size),
                u32_field(text, "num_experts")?,
                u32_field(text, "num_experts_per_tok")?,
                moe_intermediate_size,
                u32_field(text, "shared_expert_intermediate_size")?,
            )
        } else {
            for field in [
                "num_experts",
                "num_experts_per_tok",
                "moe_intermediate_size",
                "shared_expert_intermediate_size",
            ] {
                if text.contains_key(field) {
                    return Err(WeightLoadError::invalid_config(format!(
                        "dense checkpoint must not contain MoE field {field:?}"
                    )));
                }
            }
            (u32_field(text, "intermediate_size")?, 0, 0, 0, 0)
        };
        let rope_parameters = match text.get("rope_parameters") {
            Some(value) if value.is_null() => None,
            Some(value) => Some(value.get::<HashMap<String, JsonValue>>().ok_or_else(|| {
                WeightLoadError::json("field \"rope_parameters\" must be an object")
            })?),
            None => None,
        };
        let rope_theta = match rope_parameters {
            Some(rope) => f32_field_with_default(rope, "rope_theta", 10_000.0)?,
            None => f32_field_with_default(text, "rope_theta", 10_000.0)?,
        };
        let partial_rotary_factor = match rope_parameters {
            Some(rope) => match rope.get("partial_rotary_factor") {
                Some(value) => Some(f32_from_value(
                    value,
                    "field \"rope_parameters.partial_rotary_factor\"",
                )?),
                None => text
                    .get("partial_rotary_factor")
                    .map(|value| f32_from_value(value, "field \"partial_rotary_factor\""))
                    .transpose()?,
            },
            None => text
                .get("partial_rotary_factor")
                .map(|value| f32_from_value(value, "field \"partial_rotary_factor\""))
                .transpose()?,
        };
        let layer_types = string_array_field(text, "layer_types")?
            .into_iter()
            .map(|name| match name.as_str() {
                "linear_attention" => Ok(QwenLayerKind::LinearAttention),
                "full_attention" => Ok(QwenLayerKind::FullAttention),
                other => Err(WeightLoadError::invalid_config(format!(
                    "unknown layer_types entry {other:?}"
                ))),
            })
            .collect::<LoadResult<Vec<_>>>()?;
        let config = Self {
            num_hidden_layers: u32_field(text, "num_hidden_layers")?,
            hidden_size,
            intermediate_size,
            vocab_size: u32_field(text, "vocab_size")?,
            num_attention_heads,
            num_key_value_heads: u32_field(text, "num_key_value_heads")?,
            head_dim,
            rms_norm_eps: f32_field_with_default(text, "rms_norm_eps", 1.0e-6)?,
            rope_theta,
            logits_soft_cap: f32_field_with_default(text, "logits_soft_cap", 0.0)?,
            num_experts,
            num_experts_per_tok,
            moe_intermediate_size,
            shared_expert_intermediate_size,
            linear_num_key_heads: u32_field(text, "linear_num_key_heads")?,
            linear_num_value_heads: u32_field(text, "linear_num_value_heads")?,
            linear_key_head_dim: u32_field(text, "linear_key_head_dim")?,
            linear_value_head_dim: u32_field(text, "linear_value_head_dim")?,
            linear_conv_kernel_dim: u32_field(text, "linear_conv_kernel_dim")?,
            layer_types,
            tie_word_embeddings,
            dtype: opt_string_field(text, "dtype")?,
            attn_output_gate: bool_field_with_default(text, "attn_output_gate", true)?,
            full_attention_interval: opt_u32_field(text, "full_attention_interval")?,
            partial_rotary_factor,
        };
        config.validate_supported()?;
        Ok(config)
    }

    /// Check the supplied checkpoint against this engine before any weight allocation.
    /// Source revisions and weight values may differ; architecture and math must match.
    pub(super) fn validate_compiled_model(&self) -> LoadResult<()> {
        use crate::constants::{attention, model};
        self.validate_supported()?;
        if self.num_hidden_layers != model::NUM_HIDDEN_LAYERS
            || self.vocab_size != model::VOCAB_SIZE
            || self.rms_norm_eps != model::RMS_NORM_EPS
            || self.rope_theta != attention::ROPE_THETA
            || self.logits_soft_cap != 0.0
        {
            return Err(WeightLoadError::invalid_config(
                "checkpoint layers, vocabulary, normalization, or RoPE do not match the compiled model",
            ));
        }
        for (actual, expected) in self.layer_types.iter().zip(model::LAYER_TYPES) {
            let expected = match expected {
                "full_attention" => QwenLayerKind::FullAttention,
                "linear_attention" => QwenLayerKind::LinearAttention,
                _ => {
                    return Err(WeightLoadError::invalid_config(
                        "unsupported compiled layer type",
                    ));
                }
            };
            if *actual != expected {
                return Err(WeightLoadError::invalid_config(
                    "checkpoint layer order differs from the compiled model",
                ));
            }
        }
        Ok(())
    }

    pub(super) fn validate_supported(&self) -> LoadResult<()> {
        if self
            .dtype
            .as_deref()
            .is_some_and(|dtype| dtype != "bfloat16")
        {
            return Err(WeightLoadError::invalid_config(format!(
                "dtype must be bfloat16, got {:?}",
                self.dtype
            )));
        }
        if self.hidden_size != HIDDEN_SIZE {
            return Err(WeightLoadError::invalid_config(format!(
                "hidden_size must be {HIDDEN_SIZE}, got {}",
                self.hidden_size
            )));
        }
        if self.num_hidden_layers == 0 || !self.num_hidden_layers.is_multiple_of(4) {
            return Err(WeightLoadError::invalid_config(
                "num_hidden_layers must be a positive multiple of 4",
            ));
        }
        if self.layer_types.len() != self.num_hidden_layers as usize {
            return Err(WeightLoadError::invalid_config(format!(
                "layer_types has {} entries, expected {}",
                self.layer_types.len(),
                self.num_hidden_layers
            )));
        }
        for (idx, layer_type) in self.layer_types.iter().enumerate() {
            let expected = if idx % 4 == 3 {
                QwenLayerKind::FullAttention
            } else {
                QwenLayerKind::LinearAttention
            };
            if *layer_type != expected {
                return Err(WeightLoadError::invalid_config(format!(
                    "layer {idx} must be {expected:?}, got {layer_type:?}"
                )));
            }
        }
        if self
            .full_attention_interval
            .is_some_and(|interval| interval != 4)
        {
            return Err(WeightLoadError::invalid_config(format!(
                "full_attention_interval must be 4, got {:?}",
                self.full_attention_interval
            )));
        }
        if self.num_attention_heads != NUM_Q_HEADS
            || self.num_key_value_heads != NUM_KV_HEADS
            || self.head_dim != HEAD_DIM
        {
            return Err(WeightLoadError::invalid_config(format!(
                "full attention shape must be q_heads={NUM_Q_HEADS} \
                 kv_heads={NUM_KV_HEADS} head_dim={HEAD_DIM}, \
                 got q_heads={} kv_heads={} head_dim={}",
                self.num_attention_heads, self.num_key_value_heads, self.head_dim
            )));
        }
        if self.intermediate_size != INTERMEDIATE_SIZE {
            return Err(WeightLoadError::invalid_config(format!(
                "MLP intermediate_size must be {INTERMEDIATE_SIZE}, got {}",
                self.intermediate_size
            )));
        }
        if self.num_experts != NUM_EXPERTS
            || self.num_experts_per_tok != NUM_EXPERTS_PER_TOKEN
            || self.moe_intermediate_size != if HAS_EXPERTS { INTERMEDIATE_SIZE } else { 0 }
            || self.shared_expert_intermediate_size != SHARED_EXPERT_INTERMEDIATE_SIZE
        {
            return Err(WeightLoadError::invalid_config(
                "expert fields do not match the compiled model",
            ));
        }
        if self.linear_num_key_heads != NUM_KEY_HEADS
            || self.linear_num_value_heads != NUM_VALUE_HEADS
            || self.linear_key_head_dim != KEY_HEAD_DIM
            || self.linear_value_head_dim != VALUE_HEAD_DIM
            || self.linear_conv_kernel_dim != CONV_WIDTH
        {
            return Err(WeightLoadError::invalid_config(
                "GDN fields do not match qs3 Qwen3.6 constants",
            ));
        }
        if !self.attn_output_gate {
            return Err(WeightLoadError::invalid_config(
                "attn_output_gate must be true for packed q_proj gate extraction",
            ));
        }
        if self.tie_word_embeddings {
            return Err(WeightLoadError::invalid_config(
                "tie_word_embeddings is unsupported; lm_head.weight must be present",
            ));
        }
        if let Some(partial_rotary_factor) = self.partial_rotary_factor {
            if !partial_rotary_factor.is_finite() || partial_rotary_factor <= 0.0 {
                return Err(WeightLoadError::invalid_config(
                    "partial_rotary_factor must be finite and positive",
                ));
            }
            let rotary_dim = self.head_dim as f32 * partial_rotary_factor;
            if (rotary_dim - ROTARY_DIM as f32).abs() > f32::EPSILON {
                return Err(WeightLoadError::invalid_config(format!(
                    "partial rotary dim must be {ROTARY_DIM}, got {rotary_dim}"
                )));
            }
        }
        if !self.rms_norm_eps.is_finite()
            || self.rms_norm_eps <= 0.0
            || !self.rope_theta.is_finite()
            || self.rope_theta <= 0.0
            || !self.logits_soft_cap.is_finite()
            || self.logits_soft_cap < 0.0
        {
            return Err(WeightLoadError::invalid_config(
                "non-finite or invalid numeric config field",
            ));
        }
        Ok(())
    }
}
impl DynDType {
    pub(super) fn from_safetensors_name(name: &str, tensor_name: &str) -> LoadResult<Self> {
        match name {
            "BF16" => Ok(Self::BF16),
            "F32" => Ok(Self::F32),
            "U8" => Ok(Self::U8),
            "F8_E4M3" => Ok(Self::FP8E4M3),
            other => Err(WeightLoadError::invalid_safetensors(format!(
                "tensor {tensor_name:?} has unsupported dtype {other:?}"
            ))),
        }
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub(crate) struct TensorMeta {
    pub(super) dtype: DynDType,
    pub(super) shape: Vec<u32>,
    pub(super) data_offsets: (u64, u64),
}

impl TensorMeta {
    pub(super) fn byte_len(&self) -> LoadResult<usize> {
        self.dtype
            .storage_bytes_for_shape(&self.shape)
            .map_err(|_| WeightLoadError::tensor_table("tensor byte count overflow"))
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub(super) struct TensorFileMeta {
    pub(super) shard: String,
    pub(super) absolute_offset: u64,
    pub(super) meta: TensorMeta,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub(super) struct SafetensorsHeader {
    pub(super) tensors: BTreeMap<String, TensorMeta>,
    pub(super) data_start: u64,
}

impl SafetensorsHeader {
    pub(super) fn read(path: impl AsRef<Path>) -> LoadResult<Self> {
        Self::read_with_limit(path, DEFAULT_MAX_HEADER_BYTES)
    }

    pub(super) fn read_with_limit(
        path: impl AsRef<Path>,
        max_header_bytes: usize,
    ) -> LoadResult<Self> {
        let path = path.as_ref();
        let mut file = File::open(path)?;
        let file_len = file.metadata()?.len();
        let mut prefix = [0u8; 8];
        file.read_exact(&mut prefix)?;
        let header_len = u64::from_le_bytes(prefix);
        if header_len == 0 {
            return Err(WeightLoadError::invalid_safetensors(format!(
                "{} has an empty header",
                path.display()
            )));
        }
        if header_len > max_header_bytes as u64 {
            return Err(WeightLoadError::invalid_safetensors(format!(
                "{} header is {header_len} bytes, limit is {max_header_bytes}",
                path.display()
            )));
        }
        let data_start = 8u64
            .checked_add(header_len)
            .ok_or_else(|| WeightLoadError::invalid_safetensors("header end overflow"))?;
        if data_start > file_len {
            return Err(WeightLoadError::invalid_safetensors(format!(
                "{} is shorter than its declared header",
                path.display()
            )));
        }
        let mut header = vec![0u8; header_len as usize];
        file.read_exact(&mut header)?;
        Self::parse_header_bytes(&header, data_start, file_len - data_start)
    }

    pub(super) fn parse_safetensors_bytes(bytes: &[u8]) -> LoadResult<Self> {
        if bytes.len() < 8 {
            return Err(WeightLoadError::invalid_safetensors(
                "file is shorter than safetensors prefix",
            ));
        }
        let header_len = u64::from_le_bytes(bytes[..8].try_into().unwrap());
        let data_start = 8u64
            .checked_add(header_len)
            .ok_or_else(|| WeightLoadError::invalid_safetensors("header end overflow"))?;
        if data_start > bytes.len() as u64 {
            return Err(WeightLoadError::invalid_safetensors(
                "file is shorter than declared header",
            ));
        }
        Self::parse_header_bytes(
            &bytes[8..data_start as usize],
            data_start,
            bytes.len() as u64 - data_start,
        )
    }

    pub(super) fn parse_header_bytes(
        header: &[u8],
        data_start: u64,
        data_len: u64,
    ) -> LoadResult<Self> {
        let src = std::str::from_utf8(header)
            .map_err(|err| WeightLoadError::invalid_safetensors(err.to_string()))?;
        let root = parse_json_object(src)?;
        let mut tensors = BTreeMap::new();
        for (name, value) in root {
            if name == "__metadata__" {
                if value.get::<HashMap<String, JsonValue>>().is_none() {
                    return Err(WeightLoadError::json(
                        "field \"__metadata__\" must be an object",
                    ));
                }
                continue;
            }
            let object = value.get::<HashMap<String, JsonValue>>().ok_or_else(|| {
                WeightLoadError::json(format!("tensor {name:?} must be an object"))
            })?;
            let dtype =
                DynDType::from_safetensors_name(string_field(object, "dtype")?.as_str(), &name)?;
            let shape = parse_shape(object, &name)?;
            let data_offsets = parse_data_offsets(object, &name)?;
            if data_offsets.0 > data_offsets.1 {
                return Err(WeightLoadError::invalid_safetensors(format!(
                    "tensor {name:?} has decreasing data_offsets"
                )));
            }
            let expected_bytes = dtype
                .storage_bytes_for_shape(&shape)
                .map_err(|_| WeightLoadError::tensor_table("tensor byte count overflow"))?
                as u64;
            let actual_bytes = data_offsets.1 - data_offsets.0;
            if actual_bytes != expected_bytes {
                return Err(WeightLoadError::invalid_safetensors(format!(
                    "tensor {name:?} has {actual_bytes} data bytes, expected {expected_bytes}"
                )));
            }
            if tensors
                .insert(
                    name,
                    TensorMeta {
                        dtype,
                        shape,
                        data_offsets,
                    },
                )
                .is_some()
            {
                return Err(WeightLoadError::invalid_safetensors(
                    "duplicate tensor name in safetensors header",
                ));
            }
        }
        validate_safetensors_spans(&tensors, data_len)?;
        Ok(Self {
            tensors,
            data_start,
        })
    }
}

pub(super) fn parse_shape(
    object: &HashMap<String, JsonValue>,
    tensor_name: &str,
) -> LoadResult<Vec<u32>> {
    let values = object
        .get("shape")
        .and_then(JsonValue::get::<Vec<JsonValue>>)
        .ok_or_else(|| {
            WeightLoadError::invalid_safetensors(format!(
                "tensor {tensor_name:?} shape must be an array"
            ))
        })?;
    let mut shape = Vec::with_capacity(values.len());
    for (idx, value) in values.iter().enumerate() {
        let dim = u64_from_value(value, format!("tensor {tensor_name:?} shape[{idx}]"))?;
        shape.push(u32::try_from(dim).map_err(|_| {
            WeightLoadError::invalid_safetensors(format!(
                "tensor {tensor_name:?} shape[{idx}] overflows u32"
            ))
        })?);
    }
    Ok(shape)
}

pub(super) fn parse_data_offsets(
    object: &HashMap<String, JsonValue>,
    tensor_name: &str,
) -> LoadResult<(u64, u64)> {
    let values = object
        .get("data_offsets")
        .and_then(JsonValue::get::<Vec<JsonValue>>)
        .ok_or_else(|| {
            WeightLoadError::invalid_safetensors(format!(
                "tensor {tensor_name:?} data_offsets must be an array"
            ))
        })?;
    if values.len() != 2 {
        return Err(WeightLoadError::invalid_safetensors(format!(
            "tensor {tensor_name:?} data_offsets must have two entries"
        )));
    }
    Ok((
        u64_from_value(
            &values[0],
            format!("tensor {tensor_name:?} data_offsets[0]"),
        )?,
        u64_from_value(
            &values[1],
            format!("tensor {tensor_name:?} data_offsets[1]"),
        )?,
    ))
}

pub(super) fn validate_safetensors_spans(
    tensors: &BTreeMap<String, TensorMeta>,
    data_len: u64,
) -> LoadResult<()> {
    let mut spans = tensors
        .iter()
        .map(|(name, meta)| (meta.data_offsets.0, meta.data_offsets.1, name.as_str()))
        .collect::<Vec<_>>();
    spans.sort_by_key(|(start, _, _)| *start);
    let mut cursor = 0u64;
    for (start, end, name) in spans {
        if start != cursor {
            return Err(WeightLoadError::invalid_safetensors(format!(
                "tensor {name:?} starts at {start}, expected contiguous offset {cursor}"
            )));
        }
        if end > data_len {
            return Err(WeightLoadError::invalid_safetensors(format!(
                "tensor {name:?} ends at {end}, past data length {data_len}"
            )));
        }
        cursor = end;
    }
    if cursor != data_len {
        return Err(WeightLoadError::invalid_safetensors(format!(
            "safetensors data has trailing bytes: consumed {cursor}, data length {data_len}"
        )));
    }
    Ok(())
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub(super) struct SafetensorsIndex {
    pub(super) weight_map: BTreeMap<String, String>,
}

impl SafetensorsIndex {
    pub(super) fn read(model_dir: impl AsRef<Path>) -> LoadResult<Self> {
        Self::read_with_limit(model_dir, DEFAULT_MAX_JSON_BYTES)
    }

    pub(super) fn read_with_limit(
        model_dir: impl AsRef<Path>,
        max_bytes: usize,
    ) -> LoadResult<Self> {
        let root = read_json_object_with_limit(
            model_dir.as_ref().join(SAFETENSORS_INDEX_FILE),
            max_bytes,
        )?;
        let weight_map = root
            .get("weight_map")
            .and_then(JsonValue::get::<HashMap<String, JsonValue>>)
            .ok_or_else(|| {
                WeightLoadError::invalid_index("field \"weight_map\" must be an object")
            })?;
        let mut out = BTreeMap::new();
        for (name, shard) in weight_map {
            let shard = shard.get::<String>().ok_or_else(|| {
                WeightLoadError::invalid_index(format!(
                    "weight_map entry {name:?} must be a shard string"
                ))
            })?;
            validate_shard_name(shard)?;
            out.insert(name.clone(), shard.clone());
        }
        if out.is_empty() {
            return Err(WeightLoadError::invalid_index("weight_map is empty"));
        }
        Ok(Self { weight_map: out })
    }

    pub(super) fn shard_names(&self) -> BTreeSet<String> {
        self.weight_map.values().cloned().collect()
    }
}

pub(super) fn validate_shard_name(shard: &str) -> LoadResult<()> {
    let path = Path::new(shard);
    if path.is_absolute() || !shard.ends_with(".safetensors") {
        return Err(WeightLoadError::invalid_index(format!(
            "invalid shard path {shard:?}"
        )));
    }
    let mut components = path.components();
    match (components.next(), components.next()) {
        (Some(Component::Normal(_)), None) => Ok(()),
        _ => Err(WeightLoadError::invalid_index(format!(
            "invalid shard path {shard:?}"
        ))),
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum WeightTensorSource {
    Safetensors,
    ZeroFill,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub(crate) struct WeightTensorSpec {
    pub(super) name: String,
    pub(super) dtype: DynDType,
    pub(super) shape: Vec<u32>,
    pub(super) source: WeightTensorSource,
}

impl WeightTensorSpec {
    pub(super) fn byte_len(&self) -> LoadResult<usize> {
        self.dtype
            .storage_bytes_for_shape(&self.shape)
            .map_err(|_| WeightLoadError::tensor_table("tensor byte count overflow"))
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub(super) struct ValidatedTensor {
    pub(super) spec: WeightTensorSpec,
    pub(super) file_meta: Option<TensorFileMeta>,
}

pub(super) fn ignored_qwen36_tensor(name: &str) -> bool {
    name.starts_with("model.visual.")
        || name.starts_with("mtp.")
        || name.ends_with("rotary_emb.inv_freq")
}

pub(super) fn validate_tensor_specs(
    tensors: &BTreeMap<String, TensorFileMeta>,
    specs: Vec<WeightTensorSpec>,
) -> LoadResult<Vec<ValidatedTensor>> {
    let mut expected_names = HashSet::new();
    let mut validated = Vec::with_capacity(specs.len());
    for spec in specs {
        if spec.source == WeightTensorSource::Safetensors {
            expected_names.insert(spec.name.clone());
            let Some(file_meta) = tensors.get(&spec.name) else {
                return Err(WeightLoadError::tensor_table(format!(
                    "missing required tensor {:?}",
                    spec.name
                )));
            };
            if file_meta.meta.dtype != spec.dtype {
                return Err(WeightLoadError::tensor_table(format!(
                    "tensor {:?} dtype {:?}, expected {:?}",
                    spec.name, file_meta.meta.dtype, spec.dtype
                )));
            }
            if file_meta.meta.shape != spec.shape {
                return Err(WeightLoadError::tensor_table(format!(
                    "tensor {:?} shape {:?}, expected {:?}",
                    spec.name, file_meta.meta.shape, spec.shape
                )));
            }
            validated.push(ValidatedTensor {
                spec,
                file_meta: Some(file_meta.clone()),
            });
        } else {
            validated.push(ValidatedTensor {
                spec,
                file_meta: None,
            });
        }
    }
    for name in tensors.keys() {
        if !expected_names.contains(name) && !ignored_qwen36_tensor(name) {
            return Err(WeightLoadError::tensor_table(format!(
                "unexpected tensor {name:?}"
            )));
        }
    }
    Ok(validated)
}
