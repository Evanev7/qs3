#[cfg(test)]
use super::{DeterministicRng, checked_usize_product, constant_bf16_values, random_bf16_values};
use super::{QwenBlockKind, QwenConfig};
#[cfg(test)]
use crate::constants::{
    attention::PACKED_Q_GATE_WIDTH,
    gdn::{CONV_WIDTH, NUM_VALUE_HEADS, OUTPUT_WIDTH, PACKED_QKV_CHANNELS, VALUE_HEAD_DIM},
};
#[cfg(test)]
use crate::memory::{CudaCtx, DeviceBuffer};
use crate::{engine::Status, ext::SafeVec, memory::DeviceSpan};
use std::ops::Deref;
#[cfg(test)]
use std::rc::Rc;

/// Keeps the concrete allocation owner while exposing only the common view.
/// Backends retain control over allocation and destruction; no growth or transfer
/// behavior is required of weights. Boxing occurs only during materialization.
pub(super) type QwenWeight = Box<dyn Deref<Target = DeviceSpan<u16>>>;

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
    pub(super) token_embedding: QwenWeight,
    pub(super) final_norm: QwenWeight,
    pub(super) lm_head: QwenWeight,
    pub(super) layers: Vec<QwenLayerWeights>,
}

#[cfg(test)]
fn bf16_weight(ctx: &Rc<CudaCtx>, values: &[u16]) -> Result<QwenWeight, Status> {
    Ok(Box::new(DeviceBuffer::from_slice(ctx.clone(), values)?))
}

fn loaded_mlp_weights<F>(
    layer: u32,
    moe: Option<MoeShape>,
    take: &mut F,
) -> Result<QwenMlpWeights, Status>
where
    F: FnMut(Option<u32>, &'static str) -> Result<QwenWeight, Status>,
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
    pub(crate) fn from_bf16_allocations<F, A>(
        config: QwenConfig,
        mut take: F,
    ) -> Result<Self, Status>
    where
        F: FnMut(Option<u32>, &'static str) -> Result<A, Status>,
        A: Deref<Target = DeviceSpan<u16>> + 'static,
    {
        config.validate()?;
        let mut take = |layer, slot| take(layer, slot).map(|owner| Box::new(owner) as QwenWeight);
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
            token_embedding,
            final_norm,
            lm_head,
            layers,
        })
    }

    #[cfg(test)]
    pub(crate) fn random_bf16(
        ctx: Rc<CudaCtx>,
        config: &QwenConfig,
        seed: u64,
    ) -> Result<Self, Status> {
        config.validate()?;
        let mut rng = DeterministicRng::new(seed);
        let hidden = config.hidden_size();
        let q_hidden = config.q_hidden_size()?;
        let vocab = config.vocab_size();

        let token_embedding = bf16_weight(
            &ctx,
            &random_bf16_values(&mut rng, checked_usize_product(&[vocab, hidden])?, 0.08)?,
        )?;
        let final_norm = bf16_weight(&ctx, &constant_bf16_values(hidden as usize, 0.0)?)?;
        let lm_head = bf16_weight(
            &ctx,
            &random_bf16_values(&mut rng, checked_usize_product(&[vocab, hidden])?, 0.04)?,
        )?;

        let mut layers = Vec::safe_new(config.num_layers() as usize)?;
        for layer_idx in 0..config.num_layers() {
            let mlp_norm = bf16_weight(&ctx, &constant_bf16_values(hidden as usize, 0.0)?)?;
            let mlp = Self::random_mlp_weights(ctx.clone(), config, &mut rng)?;

            if config.layer_kind(layer_idx) == QwenBlockKind::LinearAttention {
                layers.push(QwenLayerWeights::Gdn(QwenGdnWeights {
                    norm: bf16_weight(&ctx, &constant_bf16_values(hidden as usize, 0.0)?)?,
                    in_proj: bf16_weight(
                        &ctx,
                        &random_bf16_values(
                            &mut rng,
                            checked_usize_product(&[PACKED_QKV_CHANNELS, hidden])?,
                            0.01,
                        )?,
                    )?,
                    gate_proj: bf16_weight(
                        &ctx,
                        &random_bf16_values(
                            &mut rng,
                            checked_usize_product(&[OUTPUT_WIDTH, hidden])?,
                            0.01,
                        )?,
                    )?,
                    a_proj: bf16_weight(
                        &ctx,
                        &random_bf16_values(
                            &mut rng,
                            checked_usize_product(&[NUM_VALUE_HEADS, hidden])?,
                            0.005,
                        )?,
                    )?,
                    b_proj: bf16_weight(
                        &ctx,
                        &random_bf16_values(
                            &mut rng,
                            checked_usize_product(&[NUM_VALUE_HEADS, hidden])?,
                            0.005,
                        )?,
                    )?,
                    conv_weight: bf16_weight(
                        &ctx,
                        &random_bf16_values(
                            &mut rng,
                            checked_usize_product(&[PACKED_QKV_CHANNELS, CONV_WIDTH])?,
                            0.25,
                        )?,
                    )?,
                    conv_bias: bf16_weight(
                        &ctx,
                        &constant_bf16_values(PACKED_QKV_CHANNELS as usize, 0.0)?,
                    )?,
                    a_log: bf16_weight(
                        &ctx,
                        &constant_bf16_values(NUM_VALUE_HEADS as usize, -2.0)?,
                    )?,
                    dt_bias: bf16_weight(
                        &ctx,
                        &constant_bf16_values(NUM_VALUE_HEADS as usize, -1.0)?,
                    )?,
                    rms_weight: bf16_weight(
                        &ctx,
                        &constant_bf16_values(VALUE_HEAD_DIM as usize, 1.0)?,
                    )?,
                    out_proj: bf16_weight(
                        &ctx,
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
                attn_norm: bf16_weight(&ctx, &constant_bf16_values(hidden as usize, 0.0)?)?,
                // Raw Qwen q/k norm weights use Gemma-style RMSNorm semantics:
                // effective weight is raw BF16 + 1.0 in f32.
                q_norm: bf16_weight(
                    &ctx,
                    &constant_bf16_values(config.head_dim() as usize, 0.0)?,
                )?,
                k_norm: bf16_weight(
                    &ctx,
                    &constant_bf16_values(config.head_dim() as usize, 0.0)?,
                )?,
                q_proj: bf16_weight(
                    &ctx,
                    &random_bf16_values(
                        &mut rng,
                        checked_usize_product(&[PACKED_Q_GATE_WIDTH, hidden])?,
                        0.04,
                    )?,
                )?,
                k_proj: bf16_weight(
                    &ctx,
                    &random_bf16_values(
                        &mut rng,
                        checked_usize_product(&[kv_hidden, hidden])?,
                        0.04,
                    )?,
                )?,
                v_proj: bf16_weight(
                    &ctx,
                    &random_bf16_values(
                        &mut rng,
                        checked_usize_product(&[kv_hidden, hidden])?,
                        0.04,
                    )?,
                )?,
                o_proj: bf16_weight(
                    &ctx,
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
            token_embedding,
            final_norm,
            lm_head,
            layers,
        })
    }

    #[cfg(test)]
    pub(super) fn random_mlp_weights(
        ctx: Rc<CudaCtx>,
        config: &QwenConfig,
        rng: &mut DeterministicRng,
    ) -> Result<QwenMlpWeights, Status> {
        let hidden = config.hidden_size();
        let intermediate = config.intermediate_size();
        if let Some(moe) = config.moe_config() {
            let shared = if moe.shared_expert_intermediate_size == 0 {
                None
            } else {
                Some(QwenSharedExpertWeights {
                    gate_proj: bf16_weight(
                        &ctx,
                        &random_bf16_values(
                            rng,
                            checked_usize_product(&[moe.shared_expert_intermediate_size, hidden])?,
                            0.03,
                        )?,
                    )?,
                    up_proj: bf16_weight(
                        &ctx,
                        &random_bf16_values(
                            rng,
                            checked_usize_product(&[moe.shared_expert_intermediate_size, hidden])?,
                            0.03,
                        )?,
                    )?,
                    down_proj: bf16_weight(
                        &ctx,
                        &random_bf16_values(
                            rng,
                            checked_usize_product(&[hidden, moe.shared_expert_intermediate_size])?,
                            0.03,
                        )?,
                    )?,
                    shared_expert_gate: bf16_weight(
                        &ctx,
                        &random_bf16_values(rng, checked_usize_product(&[1, hidden])?, 0.03)?,
                    )?,
                })
            };
            Ok(QwenMlpWeights::Moe {
                router_proj: bf16_weight(
                    &ctx,
                    &random_bf16_values(
                        rng,
                        checked_usize_product(&[moe.num_experts, hidden])?,
                        0.03,
                    )?,
                )?,
                gate_up_proj: bf16_weight(
                    &ctx,
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
                down_proj: bf16_weight(
                    &ctx,
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
                gate_proj: bf16_weight(
                    &ctx,
                    &random_bf16_values(
                        rng,
                        checked_usize_product(&[intermediate, hidden])?,
                        0.035,
                    )?,
                )?,
                up_proj: bf16_weight(
                    &ctx,
                    &random_bf16_values(
                        rng,
                        checked_usize_product(&[intermediate, hidden])?,
                        0.035,
                    )?,
                )?,
                down_proj: bf16_weight(
                    &ctx,
                    &random_bf16_values(
                        rng,
                        checked_usize_product(&[hidden, intermediate])?,
                        0.035,
                    )?,
                )?,
            })
        }
    }
}

pub(super) enum QwenLayerWeights {
    AttentionMlp(QwenAttentionMlpWeights),
    Gdn(QwenGdnWeights),
}

pub(super) struct QwenAttentionMlpWeights {
    pub(super) attn_norm: QwenWeight,
    pub(super) q_norm: QwenWeight,
    pub(super) k_norm: QwenWeight,
    pub(super) q_proj: QwenWeight,
    pub(super) k_proj: QwenWeight,
    pub(super) v_proj: QwenWeight,
    pub(super) o_proj: QwenWeight,
    pub(super) mlp_norm: QwenWeight,
    pub(super) mlp: QwenMlpWeights,
}

pub(super) struct QwenGdnWeights {
    pub(super) norm: QwenWeight,
    pub(super) in_proj: QwenWeight,
    pub(super) gate_proj: QwenWeight,
    pub(super) a_proj: QwenWeight,
    pub(super) b_proj: QwenWeight,
    pub(super) conv_weight: QwenWeight,
    pub(super) conv_bias: QwenWeight,
    pub(super) a_log: QwenWeight,
    pub(super) dt_bias: QwenWeight,
    pub(super) rms_weight: QwenWeight,
    pub(super) out_proj: QwenWeight,
    pub(super) mlp_norm: QwenWeight,
    pub(super) mlp: QwenMlpWeights,
}

pub(super) enum QwenMlpWeights {
    Dense {
        gate_proj: QwenWeight,
        up_proj: QwenWeight,
        down_proj: QwenWeight,
    },
    Moe {
        router_proj: QwenWeight,
        gate_up_proj: QwenWeight,
        down_proj: QwenWeight,
        shared: Option<QwenSharedExpertWeights>,
    },
}

pub(super) struct QwenSharedExpertWeights {
    pub(super) gate_proj: QwenWeight,
    pub(super) up_proj: QwenWeight,
    pub(super) down_proj: QwenWeight,
    pub(super) shared_expert_gate: QwenWeight,
}

impl QwenLayerWeights {
    pub(super) fn input_norm(&self) -> &DeviceSpan<u16> {
        match self {
            Self::AttentionMlp(layer) => &layer.attn_norm,
            Self::Gdn(layer) => &layer.norm,
        }
    }

    pub(super) fn post_attention_mlp(&self) -> (&DeviceSpan<u16>, &QwenMlpWeights) {
        match self {
            Self::AttentionMlp(layer) => (&layer.mlp_norm, &layer.mlp),
            Self::Gdn(layer) => (&layer.mlp_norm, &layer.mlp),
        }
    }
}
