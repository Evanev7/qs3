use super::{
    DeterministicRng, QwenBlockKind, QwenConfig, QwenMoeConfig, checked_usize_product,
    constant_bf16_values, random_bf16_values, scratch::DeviceBuffer,
};
use crate::{
    QWEN36_FULL_ATTN_Q_PROJ_OUT, QWEN36_GDN_CONV_WIDTH, QWEN36_GDN_NUM_V_HEADS,
    QWEN36_GDN_OUTPUT_DIM, QWEN36_GDN_PACKED_DIM, QWEN36_GDN_VALUE_DIM, engine::Status,
    ext::SafeVec, ffi,
};

pub struct QwenWeights {
    pub(super) config: QwenConfig,
    pub(super) token_embedding: DeviceBuffer<u16>,
    pub(super) final_norm: DeviceBuffer<u16>,
    pub(super) lm_head: DeviceBuffer<u16>,
    pub(super) layers: Vec<QwenLayerWeights>,
}

fn loaded_qwen36_moe_weights<F>(layer: u32, take: &mut F) -> Result<QwenMlpWeights, Status>
where
    F: FnMut(Option<u32>, &'static str) -> Result<DeviceBuffer<u16>, Status>,
{
    Ok(QwenMlpWeights::Moe {
        router_proj: take(Some(layer), "mlp.router")?,
        gate_up_proj: take(Some(layer), "mlp.experts.gate_up")?,
        down_proj: take(Some(layer), "mlp.experts.down")?,
        shared: Some(QwenSharedExpertWeights {
            gate_proj: take(Some(layer), "mlp.shared.gate")?,
            up_proj: take(Some(layer), "mlp.shared.up")?,
            down_proj: take(Some(layer), "mlp.shared.down")?,
            shared_expert_gate: take(Some(layer), "mlp.shared.gate_score")?,
        }),
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
        let mut layers = Vec::safe_new(config.num_layers as usize)?;

        for layer_idx in 0..config.num_layers {
            let input_norm = take(Some(layer_idx), "input_layernorm")?;
            let mlp_norm = take(Some(layer_idx), "post_attention_layernorm")?;
            let mlp = loaded_qwen36_moe_weights(layer_idx, &mut take)?;
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
            config,
            token_embedding,
            final_norm,
            lm_head,
            layers,
        })
    }

    pub fn random_bf16(config: &QwenConfig, seed: u64) -> Result<Self, Status> {
        config.validate()?;
        let config = config.resolved_device_config()?;
        let mut rng = DeterministicRng::new(seed);
        let device = config.device_ordinal;
        let stream = config.stream;
        let hidden = config.hidden_size;
        let q_hidden = config.q_hidden_size()?;
        let vocab = config.vocab_size;

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

        let mut layers = Vec::safe_new(config.num_layers as usize)?;
        for layer_idx in 0..config.num_layers {
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
                            checked_usize_product(&[QWEN36_GDN_PACKED_DIM, hidden])?,
                            0.01,
                        )?,
                    )?,
                    gate_proj: DeviceBuffer::from_slice(
                        device,
                        stream,
                        &random_bf16_values(
                            &mut rng,
                            checked_usize_product(&[QWEN36_GDN_OUTPUT_DIM, hidden])?,
                            0.01,
                        )?,
                    )?,
                    a_proj: DeviceBuffer::from_slice(
                        device,
                        stream,
                        &random_bf16_values(
                            &mut rng,
                            checked_usize_product(&[QWEN36_GDN_NUM_V_HEADS, hidden])?,
                            0.005,
                        )?,
                    )?,
                    b_proj: DeviceBuffer::from_slice(
                        device,
                        stream,
                        &random_bf16_values(
                            &mut rng,
                            checked_usize_product(&[QWEN36_GDN_NUM_V_HEADS, hidden])?,
                            0.005,
                        )?,
                    )?,
                    conv_weight: DeviceBuffer::from_slice(
                        device,
                        stream,
                        &random_bf16_values(
                            &mut rng,
                            checked_usize_product(&[QWEN36_GDN_PACKED_DIM, QWEN36_GDN_CONV_WIDTH])?,
                            0.25,
                        )?,
                    )?,
                    conv_bias: DeviceBuffer::from_slice(
                        device,
                        stream,
                        &constant_bf16_values(QWEN36_GDN_PACKED_DIM as usize, 0.0)?,
                    )?,
                    a_log: DeviceBuffer::from_slice(
                        device,
                        stream,
                        &constant_bf16_values(QWEN36_GDN_NUM_V_HEADS as usize, -2.0)?,
                    )?,
                    dt_bias: DeviceBuffer::from_slice(
                        device,
                        stream,
                        &constant_bf16_values(QWEN36_GDN_NUM_V_HEADS as usize, -1.0)?,
                    )?,
                    rms_weight: DeviceBuffer::from_slice(
                        device,
                        stream,
                        &constant_bf16_values(QWEN36_GDN_VALUE_DIM as usize, 1.0)?,
                    )?,
                    out_proj: DeviceBuffer::from_slice(
                        device,
                        stream,
                        &random_bf16_values(
                            &mut rng,
                            checked_usize_product(&[hidden, QWEN36_GDN_OUTPUT_DIM])?,
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
                    &constant_bf16_values(config.head_dim as usize, 0.0)?,
                )?,
                k_norm: DeviceBuffer::from_slice(
                    device,
                    stream,
                    &constant_bf16_values(config.head_dim as usize, 0.0)?,
                )?,
                q_proj: DeviceBuffer::from_slice(
                    device,
                    stream,
                    &random_bf16_values(
                        &mut rng,
                        checked_usize_product(&[QWEN36_FULL_ATTN_Q_PROJ_OUT, hidden])?,
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
            config,
            token_embedding,
            final_norm,
            lm_head,
            layers,
        })
    }

    pub(super) fn random_mlp_weights(
        config: &QwenConfig,
        rng: &mut DeterministicRng,
    ) -> Result<QwenMlpWeights, Status> {
        let device = config.device_ordinal;
        let stream = config.stream;
        let hidden = config.hidden_size;
        let intermediate = config.intermediate_size;
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
        if !self.config.same_model_shape(config) {
            return Err(Status::InvalidArgument);
        }
        if self.config.device_ordinal != config.device_ordinal
            || self.config.stream != config.stream
        {
            return Err(Status::InvalidArgument);
        }
        let expected_layers = config.num_layers as usize;
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

#[derive(Clone, Copy)]
pub(super) enum QwenMlpPtrs {
    Dense {
        gate_proj: ffi::DevicePtr,
        up_proj: ffi::DevicePtr,
        down_proj: ffi::DevicePtr,
    },
    Moe {
        router_proj: ffi::DevicePtr,
        gate_up_proj: ffi::DevicePtr,
        down_proj: ffi::DevicePtr,
        shared: Option<QwenSharedExpertPtrs>,
    },
}

#[derive(Clone, Copy)]
pub(super) struct QwenSharedExpertPtrs {
    pub(super) gate_proj: ffi::DevicePtr,
    pub(super) up_proj: ffi::DevicePtr,
    pub(super) down_proj: ffi::DevicePtr,
    pub(super) shared_expert_gate: ffi::DevicePtr,
}

impl QwenMlpWeights {
    pub(super) fn validate_for(&self, moe: Option<QwenMoeConfig>) -> Result<(), Status> {
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

    pub(super) fn ptrs(&self) -> QwenMlpPtrs {
        match self {
            Self::Dense {
                gate_proj,
                up_proj,
                down_proj,
            } => QwenMlpPtrs::Dense {
                gate_proj: gate_proj.as_device_ptr(),
                up_proj: up_proj.as_device_ptr(),
                down_proj: down_proj.as_device_ptr(),
            },
            Self::Moe {
                router_proj,
                gate_up_proj,
                down_proj,
                shared,
            } => QwenMlpPtrs::Moe {
                router_proj: router_proj.as_device_ptr(),
                gate_up_proj: gate_up_proj.as_device_ptr(),
                down_proj: down_proj.as_device_ptr(),
                shared: shared.as_ref().map(|shared| QwenSharedExpertPtrs {
                    gate_proj: shared.gate_proj.as_device_ptr(),
                    up_proj: shared.up_proj.as_device_ptr(),
                    down_proj: shared.down_proj.as_device_ptr(),
                    shared_expert_gate: shared.shared_expert_gate.as_device_ptr(),
                }),
            },
        }
    }
}

#[derive(Clone, Copy)]
pub(super) enum QwenLayerPtrs {
    AttentionMlp(QwenAttentionMlpPtrs),
    Gdn(QwenGdnPtrs),
}

#[derive(Clone, Copy)]
pub(super) struct QwenAttentionMlpPtrs {
    pub(super) attn_norm: ffi::DevicePtr,
    pub(super) q_norm: ffi::DevicePtr,
    pub(super) k_norm: ffi::DevicePtr,
    pub(super) q_proj: ffi::DevicePtr,
    pub(super) k_proj: ffi::DevicePtr,
    pub(super) v_proj: ffi::DevicePtr,
    pub(super) o_proj: ffi::DevicePtr,
    pub(super) mlp_norm: ffi::DevicePtr,
    pub(super) mlp: QwenMlpPtrs,
}

impl QwenLayerWeights {
    pub(super) fn ptrs(&self) -> QwenLayerPtrs {
        match self {
            Self::AttentionMlp(layer) => QwenLayerPtrs::AttentionMlp(QwenAttentionMlpPtrs {
                attn_norm: layer.attn_norm.as_device_ptr(),
                q_norm: layer.q_norm.as_device_ptr(),
                k_norm: layer.k_norm.as_device_ptr(),
                q_proj: layer.q_proj.as_device_ptr(),
                k_proj: layer.k_proj.as_device_ptr(),
                v_proj: layer.v_proj.as_device_ptr(),
                o_proj: layer.o_proj.as_device_ptr(),
                mlp_norm: layer.mlp_norm.as_device_ptr(),
                mlp: layer.mlp.ptrs(),
            }),
            Self::Gdn(layer) => QwenLayerPtrs::Gdn(QwenGdnPtrs {
                norm: layer.norm.as_device_ptr(),
                in_proj: layer.in_proj.as_device_ptr(),
                gate_proj: layer.gate_proj.as_device_ptr(),
                a_proj: layer.a_proj.as_device_ptr(),
                b_proj: layer.b_proj.as_device_ptr(),
                conv_weight: layer.conv_weight.as_device_ptr(),
                conv_bias: layer.conv_bias.as_device_ptr(),
                a_log: layer.a_log.as_device_ptr(),
                dt_bias: layer.dt_bias.as_device_ptr(),
                rms_weight: layer.rms_weight.as_device_ptr(),
                out_proj: layer.out_proj.as_device_ptr(),
                mlp_norm: layer.mlp_norm.as_device_ptr(),
                mlp: layer.mlp.ptrs(),
            }),
        }
    }
}

impl QwenLayerPtrs {
    pub(super) fn input_norm(self) -> ffi::DevicePtr {
        match self {
            Self::AttentionMlp(layer) => layer.attn_norm,
            Self::Gdn(layer) => layer.norm,
        }
    }

    pub(super) fn post_attention_mlp(self) -> QwenPostAttentionMlpPtrs {
        match self {
            Self::AttentionMlp(layer) => QwenPostAttentionMlpPtrs {
                norm: layer.mlp_norm,
                mlp: layer.mlp,
            },
            Self::Gdn(layer) => QwenPostAttentionMlpPtrs {
                norm: layer.mlp_norm,
                mlp: layer.mlp,
            },
        }
    }
}

#[derive(Clone, Copy)]
pub(super) struct QwenGdnPtrs {
    pub(super) norm: ffi::DevicePtr,
    pub(super) in_proj: ffi::DevicePtr,
    pub(super) gate_proj: ffi::DevicePtr,
    pub(super) a_proj: ffi::DevicePtr,
    pub(super) b_proj: ffi::DevicePtr,
    pub(super) conv_weight: ffi::DevicePtr,
    pub(super) conv_bias: ffi::DevicePtr,
    pub(super) a_log: ffi::DevicePtr,
    pub(super) dt_bias: ffi::DevicePtr,
    pub(super) rms_weight: ffi::DevicePtr,
    pub(super) out_proj: ffi::DevicePtr,
    pub(super) mlp_norm: ffi::DevicePtr,
    pub(super) mlp: QwenMlpPtrs,
}

#[derive(Clone, Copy)]
pub(super) struct QwenPostAttentionMlpPtrs {
    pub(super) norm: ffi::DevicePtr,
    pub(super) mlp: QwenMlpPtrs,
}
