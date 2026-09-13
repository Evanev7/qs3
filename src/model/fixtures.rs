//! Small, explicit shapes used by numerical and runner-lifecycle tests only.
use super::{MoeShape, QwenConfig};
use crate::{
    constants::{attention, model},
    engine::Status,
};
use std::ptr;

#[derive(Clone, Copy, Debug, PartialEq)]
pub(super) struct FixtureShape {
    pub num_layers: u32,
    pub hidden_size: u32,
    pub intermediate_size: u32,
    pub vocab_size: u32,
    pub rms_norm_eps: f32,
    pub rope_theta: f32,
    pub moe: Option<MoeShape>,
    pub full_attention_only: bool,
}

impl FixtureShape {
    fn dense() -> Self {
        Self {
            num_layers: 2,
            hidden_size: model::HIDDEN_SIZE,
            intermediate_size: 256,
            vocab_size: 64,
            rms_norm_eps: 1.0e-6,
            rope_theta: 10_000.0,
            moe: None,
            full_attention_only: true,
        }
    }

    pub(super) fn validate(self) -> Result<(), Status> {
        if self.num_layers == 0
            || self.num_layers > model::NUM_HIDDEN_LAYERS
            || self.hidden_size != model::HIDDEN_SIZE
            || self.vocab_size == 0
            || self.intermediate_size == 0
            || !self.intermediate_size.is_multiple_of(8)
        {
            return Err(Status::InvalidArgument);
        }
        if !self.full_attention_only && !self.num_layers.is_multiple_of(4) {
            return Err(Status::InvalidArgument);
        }
        if let Some(moe) = self.moe {
            if moe.num_experts == 0
                || moe.num_experts_per_tok == 0
                || moe.num_experts_per_tok > moe.num_experts
                || moe.num_experts > crate::QWEN36_MOE_MAX_EXPERTS
                || moe.num_experts_per_tok > crate::QWEN36_MOE_MAX_TOP_K
                || moe.moe_intermediate_size != self.intermediate_size
                || !moe.shared_expert_intermediate_size.is_multiple_of(8)
            {
                return Err(Status::InvalidArgument);
            }
        }
        Ok(())
    }
}

impl QwenConfig {
    pub(super) fn fixture_mut(&mut self) -> &mut FixtureShape {
        self.fixture
            .as_mut()
            .expect("explicit test fixture required")
    }

    pub(super) fn randomized_dense_tiny_fixture(device_ordinal: i32) -> Self {
        let mut config = Self::new(device_ordinal, ptr::null_mut(), 16).unwrap();
        config.max_pages = 8;
        config.qscb_workspace_bytes = 16 << 20;
        config.fixture = Some(FixtureShape::dense());
        config
    }

    pub(super) fn randomized_moe_tiny_fixture(device_ordinal: i32) -> Self {
        let mut config = Self::randomized_dense_tiny_fixture(device_ordinal);
        let fixture = config.fixture_mut();
        fixture.intermediate_size = 64;
        fixture.moe = Some(MoeShape {
            num_experts: 4,
            num_experts_per_tok: 2,
            moe_intermediate_size: 64,
            shared_expert_intermediate_size: 0,
        });
        config
    }

    pub(super) fn randomized_shared_moe_tiny_fixture(device_ordinal: i32) -> Self {
        let mut config = Self::randomized_moe_tiny_fixture(device_ordinal);
        config
            .fixture_mut()
            .moe
            .as_mut()
            .unwrap()
            .shared_expert_intermediate_size = 32;
        config
    }

    pub(super) fn randomized_qwen36_moe_gdn_one_block_fixture(device_ordinal: i32) -> Self {
        let mut config = Self::randomized_dense_tiny_fixture(device_ordinal);
        config.max_seq_len = 8;
        config.max_batch_tokens = 8;
        config.max_pages = 2;
        config.qscb_workspace_bytes = 64 << 20;
        let fixture = config.fixture_mut();
        fixture.full_attention_only = false;
        fixture.num_layers = 4;
        fixture.vocab_size = 32;
        fixture.intermediate_size = crate::constants::mlp::INTERMEDIATE_SIZE;
        fixture.moe = Some(MoeShape::compiled());
        config
    }

    pub(crate) fn loaded_fixture(
        device_ordinal: i32,
        stream: *mut std::ffi::c_void,
        num_layers: u32,
        vocab_size: u32,
        max_seq_len: u32,
    ) -> Self {
        let mut config = Self::new(device_ordinal, stream, max_seq_len).unwrap();
        config.fixture = Some(FixtureShape {
            num_layers,
            vocab_size,
            intermediate_size: crate::constants::mlp::INTERMEDIATE_SIZE,
            rms_norm_eps: model::RMS_NORM_EPS,
            rope_theta: attention::ROPE_THETA,
            moe: Some(MoeShape::compiled()),
            full_attention_only: false,
            ..FixtureShape::dense()
        });
        config
    }
}
