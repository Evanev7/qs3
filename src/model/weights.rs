#[cfg(test)]
use super::{DeterministicRng, checked_usize_product, constant_bf16_values, random_bf16_values};
use super::{QwenBlockKind, QwenConfig, scratch::DeviceBuffer};
#[cfg(test)]
use crate::constants::{
    attention::PACKED_Q_GATE_WIDTH,
    gdn::{CONV_WIDTH, NUM_VALUE_HEADS, OUTPUT_WIDTH, PACKED_QKV_CHANNELS, VALUE_HEAD_DIM},
};
use crate::{engine::Status, ext::SafeVec};

/// Dimensions needed to bind routed and shared expert tensors.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(super) struct MoeShape {
    pub num_experts: u32,
    pub num_experts_per_tok: u32,
    pub moe_intermediate_size: u32,
    pub shared_expert_intermediate_size: u32,
}

impl MoeShape {
    pub(super) const fn compiled() -> Self {
        use crate::constants::mlp;
        Self {
            num_experts: mlp::NUM_EXPERTS,
            num_experts_per_tok: mlp::NUM_EXPERTS_PER_TOKEN,
            moe_intermediate_size: mlp::INTERMEDIATE_SIZE,
            shared_expert_intermediate_size: mlp::SHARED_EXPERT_INTERMEDIATE_SIZE,
        }
    }
}

pub struct QwenWeights {
    pub(super) device_ordinal: i32,
    pub(super) stream: crate::ffi::CudaStream,
    #[cfg(test)]
    pub(super) fixture: Option<super::fixtures::FixtureShape>,
    pub(super) token_embedding: DeviceBuffer<u16>,
    pub(super) final_norm: DeviceBuffer<u16>,
    pub(super) lm_head: DeviceBuffer<u16>,
    pub(super) layers: Vec<QwenLayerWeights>,
}

fn loaded_mlp_weights<F>(
    layer: u32,
    moe: Option<MoeShape>,
    take: &mut F,
) -> Result<QwenMlpWeights, Status>
where
    F: FnMut(Option<u32>, &'static str) -> Result<DeviceBuffer<u16>, Status>,
{
    let Some(moe) = moe else {
        return Ok(QwenMlpWeights::Dense {
            gate_proj: take(Some(layer), "mlp.gate")?,
            up_proj: take(Some(layer), "mlp.up")?,
            down_proj: take(Some(layer), "mlp.down")?,
        });
    };
    Ok(QwenMlpWeights::Moe {
        router_proj: take(Some(layer), "mlp.router")?,
        gate_up_proj: take(Some(layer), "mlp.experts.gate_up")?,
        down_proj: take(Some(layer), "mlp.experts.down")?,
        shared: if moe.shared_expert_intermediate_size == 0 {
            None
        } else {
            Some(QwenSharedExpertWeights {
                gate_proj: take(Some(layer), "mlp.shared.gate")?,
                up_proj: take(Some(layer), "mlp.shared.up")?,
                down_proj: take(Some(layer), "mlp.shared.down")?,
                shared_expert_gate: take(Some(layer), "mlp.shared.gate_score")?,
            })
        },
    })
}

impl QwenWeights {
    pub(crate) fn from_bf16_buffers<F>(config: QwenConfig, mut take: F) -> Result<Self, Status>
    where
        F: FnMut(Option<u32>, &'static str) -> Result<DeviceBuffer<u16>, Status>,
    {
        config.validate()?;
        let token_embedding = take(None, "token_embedding")?;
        let final_norm = take(None, "final_norm")?;
        let lm_head = take(None, "lm_head")?;
        let mut layers = Vec::safe_new(config.num_layers() as usize)?;

        for layer_idx in 0..config.num_layers() {
            let input_norm = take(Some(layer_idx), "input_layernorm")?;
            let mlp_norm = take(Some(layer_idx), "post_attention_layernorm")?;
            let mlp = loaded_mlp_weights(layer_idx, config.moe_config(), &mut take)?;
            let layer = match config.layer_kind(layer_idx) {
                QwenBlockKind::FullAttention => {
                    QwenLayerWeights::AttentionMlp(QwenAttentionMlpWeights {
                        attn_norm: input_norm,
                        q_norm: take(Some(layer_idx), "attn.q_norm")?,
                        k_norm: take(Some(layer_idx), "attn.k_norm")?,
                        q_proj: take(Some(layer_idx), "attn.q_proj")?,
                        k_proj: take(Some(layer_idx), "attn.k_proj")?,
                        v_proj: take(Some(layer_idx), "attn.v_proj")?,
                        o_proj: take(Some(layer_idx), "attn.o_proj")?,
                        mlp_norm,
                        mlp,
                    })
                }
                QwenBlockKind::LinearAttention => QwenLayerWeights::Gdn(QwenGdnWeights {
                    norm: input_norm,
                    in_proj: take(Some(layer_idx), "gdn.in_proj_qkv")?,
                    gate_proj: take(Some(layer_idx), "gdn.gate_proj_z")?,
                    a_proj: take(Some(layer_idx), "gdn.a_proj")?,
                    b_proj: take(Some(layer_idx), "gdn.b_proj")?,
                    conv_weight: take(Some(layer_idx), "gdn.conv_weight")?,
                    conv_bias: take(Some(layer_idx), "gdn.conv_bias.zero")?,
                    a_log: take(Some(layer_idx), "gdn.a_log")?,
                    dt_bias: take(Some(layer_idx), "gdn.dt_bias")?,
                    rms_weight: take(Some(layer_idx), "gdn.rms_weight")?,
                    out_proj: take(Some(layer_idx), "gdn.out_proj")?,
                    mlp_norm,
                    mlp,
                }),
            };
            layers.push(layer);
        }

        Ok(Self {
            device_ordinal: config.device_ordinal,
            stream: config.stream,
            #[cfg(test)]
            fixture: config.fixture,
            token_embedding,
            final_norm,
            lm_head,
            layers,
        })
    }

    #[cfg(test)]
    pub(crate) fn random_bf16(config: &QwenConfig, seed: u64) -> Result<Self, Status> {
        config.validate()?;
        let config = config.resolved_device_config()?;
        let mut rng = DeterministicRng::new(seed);
        let device = config.device_ordinal;
        let stream = config.stream;
        let hidden = config.hidden_size();
        let q_hidden = config.q_hidden_size()?;
        let vocab = config.vocab_size();

        let token_embedding = DeviceBuffer::from_slice(
            device,
            stream,
            &random_bf16_values(&mut rng, checked_usize_product(&[vocab, hidden])?, 0.08)?,
        )?;
        let final_norm =
            DeviceBuffer::from_slice(device, stream, &constant_bf16_values(hidden as usize, 0.0)?)?;
        let lm_head = DeviceBuffer::from_slice(
            device,
            stream,
            &random_bf16_values(&mut rng, checked_usize_product(&[vocab, hidden])?, 0.04)?,
        )?;

        let mut layers = Vec::safe_new(config.num_layers() as usize)?;
        for layer_idx in 0..config.num_layers() {
            let mlp_norm = DeviceBuffer::from_slice(
                device,
                stream,
                &constant_bf16_values(hidden as usize, 0.0)?,
            )?;
            let mlp = Self::random_mlp_weights(&config, &mut rng)?;

            if config.layer_kind(layer_idx) == QwenBlockKind::LinearAttention {
                layers.push(QwenLayerWeights::Gdn(QwenGdnWeights {
                    norm: DeviceBuffer::from_slice(
                        device,
                        stream,
                        &constant_bf16_values(hidden as usize, 0.0)?,
                    )?,
                    in_proj: DeviceBuffer::from_slice(
                        device,
                        stream,
                        &random_bf16_values(
                            &mut rng,
                            checked_usize_product(&[PACKED_QKV_CHANNELS, hidden])?,
                            0.01,
                        )?,
                    )?,
                    gate_proj: DeviceBuffer::from_slice(
                        device,
                        stream,
                        &random_bf16_values(
                            &mut rng,
                            checked_usize_product(&[OUTPUT_WIDTH, hidden])?,
                            0.01,
                        )?,
                    )?,
                    a_proj: DeviceBuffer::from_slice(
                        device,
                        stream,
                        &random_bf16_values(
                            &mut rng,
                            checked_usize_product(&[NUM_VALUE_HEADS, hidden])?,
                            0.005,
                        )?,
                    )?,
                    b_proj: DeviceBuffer::from_slice(
                        device,
                        stream,
                        &random_bf16_values(
                            &mut rng,
                            checked_usize_product(&[NUM_VALUE_HEADS, hidden])?,
                            0.005,
                        )?,
                    )?,
                    conv_weight: DeviceBuffer::from_slice(
                        device,
                        stream,
                        &random_bf16_values(
                            &mut rng,
                            checked_usize_product(&[PACKED_QKV_CHANNELS, CONV_WIDTH])?,
                            0.25,
                        )?,
                    )?,
                    conv_bias: DeviceBuffer::from_slice(
                        device,
                        stream,
                        &constant_bf16_values(PACKED_QKV_CHANNELS as usize, 0.0)?,
                    )?,
                    a_log: DeviceBuffer::from_slice(
                        device,
                        stream,
                        &constant_bf16_values(NUM_VALUE_HEADS as usize, -2.0)?,
                    )?,
                    dt_bias: DeviceBuffer::from_slice(
                        device,
                        stream,
                        &constant_bf16_values(NUM_VALUE_HEADS as usize, -1.0)?,
                    )?,
                    rms_weight: DeviceBuffer::from_slice(
                        device,
                        stream,
                        &constant_bf16_values(VALUE_HEAD_DIM as usize, 1.0)?,
                    )?,
                    out_proj: DeviceBuffer::from_slice(
                        device,
                        stream,
                        &random_bf16_values(
                            &mut rng,
                            checked_usize_product(&[hidden, OUTPUT_WIDTH])?,
                            0.01,
                        )?,
                    )?,
                    mlp_norm,
                    mlp,
                }));
                continue;
            }

            let kv_hidden = config.kv_hidden_size()?;
            layers.push(QwenLayerWeights::AttentionMlp(QwenAttentionMlpWeights {
                attn_norm: DeviceBuffer::from_slice(
                    device,
                    stream,
                    &constant_bf16_values(hidden as usize, 0.0)?,
                )?,
                // Raw Qwen q/k norm weights use Gemma-style RMSNorm semantics:
                // effective weight is raw BF16 + 1.0 in f32.
                q_norm: DeviceBuffer::from_slice(
                    device,
                    stream,
                    &constant_bf16_values(config.head_dim() as usize, 0.0)?,
                )?,
                k_norm: DeviceBuffer::from_slice(
                    device,
                    stream,
                    &constant_bf16_values(config.head_dim() as usize, 0.0)?,
                )?,
                q_proj: DeviceBuffer::from_slice(
                    device,
                    stream,
                    &random_bf16_values(
                        &mut rng,
                        checked_usize_product(&[PACKED_Q_GATE_WIDTH, hidden])?,
                        0.04,
                    )?,
                )?,
                k_proj: DeviceBuffer::from_slice(
                    device,
                    stream,
                    &random_bf16_values(
                        &mut rng,
                        checked_usize_product(&[kv_hidden, hidden])?,
                        0.04,
                    )?,
                )?,
                v_proj: DeviceBuffer::from_slice(
                    device,
                    stream,
                    &random_bf16_values(
                        &mut rng,
                        checked_usize_product(&[kv_hidden, hidden])?,
                        0.04,
                    )?,
                )?,
                o_proj: DeviceBuffer::from_slice(
                    device,
                    stream,
                    &random_bf16_values(
                        &mut rng,
                        checked_usize_product(&[hidden, q_hidden])?,
                        0.04,
                    )?,
                )?,
                mlp_norm,
                mlp,
            }));
        }

        Ok(Self {
            device_ordinal: config.device_ordinal,
            stream: config.stream,
            #[cfg(test)]
            fixture: config.fixture,
            token_embedding,
            final_norm,
            lm_head,
            layers,
        })
    }

    #[cfg(test)]
    pub(super) fn random_mlp_weights(
        config: &QwenConfig,
        rng: &mut DeterministicRng,
    ) -> Result<QwenMlpWeights, Status> {
        let device = config.device_ordinal;
        let stream = config.stream;
        let hidden = config.hidden_size();
        let intermediate = config.intermediate_size();
        if let Some(moe) = config.moe_config() {
            let shared = if moe.shared_expert_intermediate_size == 0 {
                None
            } else {
                Some(QwenSharedExpertWeights {
                    gate_proj: DeviceBuffer::from_slice(
                        device,
                        stream,
                        &random_bf16_values(
                            rng,
                            checked_usize_product(&[moe.shared_expert_intermediate_size, hidden])?,
                            0.03,
                        )?,
                    )?,
                    up_proj: DeviceBuffer::from_slice(
                        device,
                        stream,
                        &random_bf16_values(
                            rng,
                            checked_usize_product(&[moe.shared_expert_intermediate_size, hidden])?,
                            0.03,
                        )?,
                    )?,
                    down_proj: DeviceBuffer::from_slice(
                        device,
                        stream,
                        &random_bf16_values(
                            rng,
                            checked_usize_product(&[hidden, moe.shared_expert_intermediate_size])?,
                            0.03,
                        )?,
                    )?,
                    shared_expert_gate: DeviceBuffer::from_slice(
                        device,
                        stream,
                        &random_bf16_values(rng, checked_usize_product(&[1, hidden])?, 0.03)?,
                    )?,
                })
            };
            Ok(QwenMlpWeights::Moe {
                router_proj: DeviceBuffer::from_slice(
                    device,
                    stream,
                    &random_bf16_values(
                        rng,
                        checked_usize_product(&[moe.num_experts, hidden])?,
                        0.03,
                    )?,
                )?,
                gate_up_proj: DeviceBuffer::from_slice(
                    device,
                    stream,
                    &random_bf16_values(
                        rng,
                        checked_usize_product(&[
                            moe.num_experts,
                            2,
                            moe.moe_intermediate_size,
                            hidden,
                        ])?,
                        0.03,
                    )?,
                )?,
                down_proj: DeviceBuffer::from_slice(
                    device,
                    stream,
                    &random_bf16_values(
                        rng,
                        checked_usize_product(&[
                            moe.num_experts,
                            hidden,
                            moe.moe_intermediate_size,
                        ])?,
                        0.03,
                    )?,
                )?,
                shared,
            })
        } else {
            Ok(QwenMlpWeights::Dense {
                gate_proj: DeviceBuffer::from_slice(
                    device,
                    stream,
                    &random_bf16_values(
                        rng,
                        checked_usize_product(&[intermediate, hidden])?,
                        0.035,
                    )?,
                )?,
                up_proj: DeviceBuffer::from_slice(
                    device,
                    stream,
                    &random_bf16_values(
                        rng,
                        checked_usize_product(&[intermediate, hidden])?,
                        0.035,
                    )?,
                )?,
                down_proj: DeviceBuffer::from_slice(
                    device,
                    stream,
                    &random_bf16_values(
                        rng,
                        checked_usize_product(&[hidden, intermediate])?,
                        0.035,
                    )?,
                )?,
            })
        }
    }

    pub(crate) fn validate_for(&self, config: &QwenConfig) -> Result<(), Status> {
        #[cfg(test)]
        if self.fixture != config.fixture {
            return Err(Status::InvalidArgument);
        }
        if self.device_ordinal != config.device_ordinal || self.stream != config.stream {
            return Err(Status::InvalidArgument);
        }
        let expected_layers = config.num_layers() as usize;
        if self.layers.len() != expected_layers {
            return Err(Status::InvalidArgument);
        }
        for (idx, layer) in self.layers.iter().enumerate() {
            match (config.layer_kind(idx as u32), layer) {
                (QwenBlockKind::FullAttention, QwenLayerWeights::AttentionMlp(layer)) => {
                    layer.mlp.validate_for(config.moe_config())?;
                }
                (QwenBlockKind::LinearAttention, QwenLayerWeights::Gdn(layer)) => {
                    layer.mlp.validate_for(config.moe_config())?;
                }
                _ => return Err(Status::InvalidArgument),
            }
        }
        Ok(())
    }
}

pub(super) enum QwenLayerWeights {
    AttentionMlp(QwenAttentionMlpWeights),
    Gdn(QwenGdnWeights),
}

pub(super) struct QwenAttentionMlpWeights {
    pub(super) attn_norm: DeviceBuffer<u16>,
    pub(super) q_norm: DeviceBuffer<u16>,
    pub(super) k_norm: DeviceBuffer<u16>,
    pub(super) q_proj: DeviceBuffer<u16>,
    pub(super) k_proj: DeviceBuffer<u16>,
    pub(super) v_proj: DeviceBuffer<u16>,
    pub(super) o_proj: DeviceBuffer<u16>,
    pub(super) mlp_norm: DeviceBuffer<u16>,
    pub(super) mlp: QwenMlpWeights,
}

pub(super) struct QwenGdnWeights {
    pub(super) norm: DeviceBuffer<u16>,
    pub(super) in_proj: DeviceBuffer<u16>,
    pub(super) gate_proj: DeviceBuffer<u16>,
    pub(super) a_proj: DeviceBuffer<u16>,
    pub(super) b_proj: DeviceBuffer<u16>,
    pub(super) conv_weight: DeviceBuffer<u16>,
    pub(super) conv_bias: DeviceBuffer<u16>,
    pub(super) a_log: DeviceBuffer<u16>,
    pub(super) dt_bias: DeviceBuffer<u16>,
    pub(super) rms_weight: DeviceBuffer<u16>,
    pub(super) out_proj: DeviceBuffer<u16>,
    pub(super) mlp_norm: DeviceBuffer<u16>,
    pub(super) mlp: QwenMlpWeights,
}

pub(super) enum QwenMlpWeights {
    Dense {
        gate_proj: DeviceBuffer<u16>,
        up_proj: DeviceBuffer<u16>,
        down_proj: DeviceBuffer<u16>,
    },
    Moe {
        router_proj: DeviceBuffer<u16>,
        gate_up_proj: DeviceBuffer<u16>,
        down_proj: DeviceBuffer<u16>,
        shared: Option<QwenSharedExpertWeights>,
    },
}

pub(super) struct QwenSharedExpertWeights {
    pub(super) gate_proj: DeviceBuffer<u16>,
    pub(super) up_proj: DeviceBuffer<u16>,
    pub(super) down_proj: DeviceBuffer<u16>,
    pub(super) shared_expert_gate: DeviceBuffer<u16>,
}

impl QwenMlpWeights {
    pub(super) fn validate_for(&self, moe: Option<MoeShape>) -> Result<(), Status> {
        match (self, moe) {
            (Self::Dense { .. }, None) => Ok(()),
            (Self::Moe { shared, .. }, Some(moe)) => {
                if shared.is_some() == (moe.shared_expert_intermediate_size != 0) {
                    Ok(())
                } else {
                    Err(Status::InvalidArgument)
                }
            }
            _ => Err(Status::InvalidArgument),
        }
    }
}

impl QwenLayerWeights {
    pub(super) fn input_norm(&self) -> &DeviceBuffer<u16> {
        match self {
            Self::AttentionMlp(layer) => &layer.attn_norm,
            Self::Gdn(layer) => &layer.norm,
        }
    }

    pub(super) fn post_attention_mlp(&self) -> (&DeviceBuffer<u16>, &QwenMlpWeights) {
        match self {
            Self::AttentionMlp(layer) => (&layer.mlp_norm, &layer.mlp),
            Self::Gdn(layer) => (&layer.mlp_norm, &layer.mlp),
        }
    }
}
