use crate::{
    constants::{attention, model},
    dtype::DynDType,
    engine::{EngineConfig, KvLayout, Status},
};

use super::weights::MoeShape;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(super) enum QwenBlockKind {
    FullAttention,
    LinearAttention,
}

/// Runtime resources for the model selected at build time.
/// Model geometry and layer ordering come from `crate::constants`.
#[derive(Clone, Copy, Debug)]
pub struct QwenConfig {
    pub max_live_requests: u32,
    pub max_batch_rows: u32,
    pub max_batch_tokens: u32,
    pub max_seq_len: u32,
    pub max_pages: u32,
    pub page_size: u32,
    pub qsfi_float_workspace_bytes: usize,
    pub qsfi_int_workspace_bytes: usize,
    pub qsfi_host_int_workspace_bytes: usize,
    pub qscb_workspace_bytes: usize,
    #[cfg(test)]
    pub(super) fixture: Option<super::fixtures::FixtureShape>,
}

// These compile to the generated constants in production. Only numerical test
// fixtures can substitute reduced layer/vocabulary/MLP shapes.
macro_rules! model_values {
    ($($name:ident: $ty:ty = $value:expr;)*) => {$(
        pub(crate) fn $name(&self) -> $ty {
            #[cfg(test)]
            if let Some(fixture) = self.fixture {
                return fixture.$name;
            }
            $value
        }
    )*};
}

impl QwenConfig {
    pub fn new(max_seq_len: u32) -> Result<Self, Status> {
        let page_size = 4;
        let config = Self {
            max_live_requests: 1,
            max_batch_rows: 1,
            max_batch_tokens: max_seq_len,
            max_seq_len,
            max_pages: max_seq_len.div_ceil(page_size),
            page_size,
            qsfi_float_workspace_bytes: 64 << 20,
            qsfi_int_workspace_bytes: 64 << 20,
            qsfi_host_int_workspace_bytes: 64 << 20,
            qscb_workspace_bytes: 64 << 20,
            #[cfg(test)]
            fixture: None,
        };
        config.validate()?;
        Ok(config)
    }

    model_values! {
        num_layers: u32 = model::NUM_HIDDEN_LAYERS;
        hidden_size: u32 = model::HIDDEN_SIZE;
        intermediate_size: u32 = crate::constants::mlp::INTERMEDIATE_SIZE;
        vocab_size: u32 = model::VOCAB_SIZE;
        rms_norm_eps: f32 = model::RMS_NORM_EPS;
        rope_theta: f32 = attention::ROPE_THETA;
    }

    pub(crate) const fn num_q_heads(&self) -> u32 {
        attention::NUM_Q_HEADS
    }

    pub(crate) const fn num_kv_heads(&self) -> u32 {
        attention::NUM_KV_HEADS
    }

    pub(crate) const fn head_dim(&self) -> u32 {
        attention::HEAD_DIM
    }

    pub fn validate(&self) -> Result<(), Status> {
        if self.max_live_requests == 0
            || self.max_batch_rows == 0
            || self.max_batch_tokens == 0
            || self.max_seq_len == 0
            || self.max_pages == 0
            || self.page_size == 0
        {
            return Err(Status::InvalidArgument);
        }
        if self.max_live_requests != 1 || self.max_batch_rows != 1 {
            return Err(Status::Unsupported);
        }
        if self.max_batch_tokens < self.max_batch_rows {
            return Err(Status::InvalidArgument);
        }
        let capacity = self
            .max_pages
            .checked_mul(self.page_size)
            .ok_or(Status::InvalidArgument)?;
        if self.max_seq_len > capacity || self.max_seq_len > model::MAX_POSITION_EMBEDDINGS {
            return Err(Status::InvalidArgument);
        }
        if self.qsfi_float_workspace_bytes == 0
            || self.qsfi_int_workspace_bytes == 0
            || self.qsfi_host_int_workspace_bytes == 0
            || self.qscb_workspace_bytes == 0
        {
            return Err(Status::InvalidArgument);
        }
        #[cfg(test)]
        if let Some(fixture) = self.fixture {
            fixture.validate()?;
        }
        Ok(())
    }

    pub(super) fn kv_hidden_size(&self) -> Result<u32, Status> {
        self.num_kv_heads()
            .checked_mul(self.head_dim())
            .ok_or(Status::InvalidArgument)
    }

    pub(super) fn q_hidden_size(&self) -> Result<u32, Status> {
        self.num_q_heads()
            .checked_mul(self.head_dim())
            .ok_or(Status::InvalidArgument)
    }

    pub(super) fn layer_kind(&self, layer_idx: u32) -> QwenBlockKind {
        #[cfg(test)]
        if self
            .fixture
            .is_some_and(|fixture| fixture.full_attention_only)
        {
            return QwenBlockKind::FullAttention;
        }
        match model::LAYER_TYPES[layer_idx as usize] {
            "full_attention" => QwenBlockKind::FullAttention,
            "linear_attention" => QwenBlockKind::LinearAttention,
            _ => unreachable!("unsupported compiled layer type"),
        }
    }

    pub(super) fn has_gdn_layers(&self) -> bool {
        self.gdn_layer_count() != 0
    }

    pub(super) fn attention_layer_count(&self) -> u32 {
        (0..self.num_layers())
            .filter(|&i| self.layer_kind(i) == QwenBlockKind::FullAttention)
            .count() as u32
    }

    pub(super) fn gdn_layer_count(&self) -> u32 {
        self.num_layers() - self.attention_layer_count()
    }

    fn layer_index(&self, model_layer_idx: u32, kind: QwenBlockKind) -> Result<u32, Status> {
        if model_layer_idx >= self.num_layers() || self.layer_kind(model_layer_idx) != kind {
            return Err(Status::InternalError);
        }
        Ok((0..model_layer_idx)
            .filter(|&i| self.layer_kind(i) == kind)
            .count() as u32)
    }

    pub(super) fn attention_layer_index(&self, model_layer_idx: u32) -> Result<u32, Status> {
        self.layer_index(model_layer_idx, QwenBlockKind::FullAttention)
    }

    pub(super) fn gdn_layer_index(&self, model_layer_idx: u32) -> Result<u32, Status> {
        self.layer_index(model_layer_idx, QwenBlockKind::LinearAttention)
    }

    pub(super) fn moe_config(&self) -> Option<MoeShape> {
        #[cfg(test)]
        if let Some(fixture) = self.fixture {
            return fixture.moe;
        }
        crate::constants::mlp::HAS_EXPERTS.then(MoeShape::compiled)
    }

    pub(super) fn engine_config(&self) -> EngineConfig {
        EngineConfig {
            num_layers: self.attention_layer_count(),
            max_live_requests: self.max_live_requests,
            max_batch_rows: self.max_batch_rows,
            max_batch_tokens: self.max_batch_tokens,
            max_seq_len: self.max_seq_len,
            max_pages: self.max_pages,
            page_size: self.page_size,
            hidden_size: self.hidden_size(),
            intermediate_size: self.intermediate_size(),
            vocab_size: self.vocab_size(),
            num_q_heads: self.num_q_heads(),
            num_kv_heads: self.num_kv_heads(),
            head_dim: self.head_dim(),
            activation_dtype: DynDType::BF16,
            kv_dtype: DynDType::BF16,
            kv_layout: KvLayout::NHD,
            rope_theta: self.rope_theta(),
            rope_scale: 1.0,
            logits_soft_cap: 0.0,
            qsfi_float_workspace_bytes: self.qsfi_float_workspace_bytes,
            qsfi_int_workspace_bytes: self.qsfi_int_workspace_bytes,
            qsfi_host_int_workspace_bytes: self.qsfi_host_int_workspace_bytes,
        }
    }
}
