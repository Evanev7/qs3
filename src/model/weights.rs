#[cfg(test)]
use super::{DeterministicRng, checked_usize_product, constant_bf16_values, random_bf16_values};
#[cfg(test)]
use super::{QwenBlockKind, QwenConfig};
#[cfg(test)]
use crate::constants::{
    attention::PACKED_Q_GATE_WIDTH,
    gdn::{CONV_WIDTH, NUM_VALUE_HEADS, OUTPUT_WIDTH, PACKED_QKV_CHANNELS, VALUE_HEAD_DIM},
};
#[cfg(test)]
use crate::engine::Status;
#[cfg(test)]
use crate::ext::SafeVec;
#[cfg(test)]
use crate::memory::{CudaCtx, HostBuffer};
use crate::{dtype::BF16, memory::DeviceSpan};
use std::ops::Deref;
#[cfg(test)]
use std::rc::Rc;

/// Keeps the concrete allocation owner while exposing only the common view.
/// Backends retain control over allocation and destruction; no growth or transfer
/// behavior is required of weights. Boxing occurs only during materialization.
pub type W<T> = Box<dyn Deref<Target = DeviceSpan<T>>>;

/// Dimensions needed to bind routed and shared expert tensors.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) struct MoeShape {
    pub num_experts: u32,
    pub num_experts_per_tok: u32,
    pub moe_intermediate_size: u32,
    pub shared_expert_intermediate_size: u32,
}

impl MoeShape {
    pub(crate) const fn compiled() -> Self {
        use crate::constants::mlp;
        Self {
            num_experts: mlp::NUM_EXPERTS,
            num_experts_per_tok: mlp::NUM_EXPERTS_PER_TOKEN,
            moe_intermediate_size: mlp::INTERMEDIATE_SIZE,
            shared_expert_intermediate_size: mlp::SHARED_EXPERT_INTERMEDIATE_SIZE,
        }
    }
}

pub struct QwenModel<M, A, H, T = W<BF16>> {
    pub(crate) token_embedding: T,
    pub(crate) final_norm: T,
    pub(crate) lm_head: H,
    pub(crate) layers: Vec<QwenLayerWeights<M, A, T>>,
}

#[cfg(test)]
fn bf16_weight(ctx: &Rc<CudaCtx>, values: &[u16]) -> Result<W<BF16>, Status> {
    let mut host = HostBuffer::<BF16>::new(values.len())?;
    for (bytes, value) in host.as_mut().chunks_exact_mut(2).zip(values) {
        bytes.copy_from_slice(&value.to_ne_bytes());
    }
    Ok(Box::new(host.upload(ctx.clone())?))
}

#[cfg(test)]
impl<M> QwenModel<M, W<BF16>, W<BF16>> {
    fn random_bf16(
        ctx: Rc<CudaCtx>,
        config: &QwenConfig,
        seed: u64,
        mlp: fn(Rc<CudaCtx>, &QwenConfig, &mut DeterministicRng) -> Result<M, Status>,
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
            let mlp = mlp(ctx.clone(), config, &mut rng)?;

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
}

#[cfg(test)]
impl MoeMlp<FusedExperts<W<BF16>>, W<BF16>> {
    fn random_bf16(
        ctx: Rc<CudaCtx>,
        config: &QwenConfig,
        rng: &mut DeterministicRng,
    ) -> Result<Self, Status> {
        let hidden = config.hidden_size();
        let moe = config.moe_config().ok_or(Status::InvalidArgument)?;
        let shared = if moe.shared_expert_intermediate_size == 0 {
            None
        } else {
            Some(SharedExpert {
                projections: DenseMlp {
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
                },
                gate: bf16_weight(
                    &ctx,
                    &random_bf16_values(rng, checked_usize_product(&[1, hidden])?, 0.03)?,
                )?,
            })
        };
        Ok(Self {
            router_proj: bf16_weight(
                &ctx,
                &random_bf16_values(
                    rng,
                    checked_usize_product(&[moe.num_experts, hidden])?,
                    0.03,
                )?,
            )?,
            experts: FusedExperts {
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
            },
            shared,
        })
    }
}

#[cfg(test)]
impl DenseMlp<W<BF16>> {
    fn random_bf16(
        ctx: Rc<CudaCtx>,
        config: &QwenConfig,
        rng: &mut DeterministicRng,
    ) -> Result<Self, Status> {
        let hidden = config.hidden_size();
        let intermediate = config.intermediate_size();
        Ok(Self {
            gate_proj: bf16_weight(
                &ctx,
                &random_bf16_values(rng, checked_usize_product(&[intermediate, hidden])?, 0.035)?,
            )?,
            up_proj: bf16_weight(
                &ctx,
                &random_bf16_values(rng, checked_usize_product(&[intermediate, hidden])?, 0.035)?,
            )?,
            down_proj: bf16_weight(
                &ctx,
                &random_bf16_values(rng, checked_usize_product(&[hidden, intermediate])?, 0.035)?,
            )?,
        })
    }
}

pub(crate) enum QwenLayerWeights<M, A = W<BF16>, T = W<BF16>> {
    AttentionMlp(QwenAttentionMlpWeights<M, A, T>),
    Gdn(QwenGdnWeights<M, A, T>),
}

pub(crate) struct QwenAttentionMlpWeights<M, A = W<BF16>, T = W<BF16>> {
    pub(crate) attn_norm: T,
    pub(crate) q_norm: T,
    pub(crate) k_norm: T,
    pub(crate) q_proj: A,
    pub(crate) k_proj: A,
    pub(crate) v_proj: A,
    pub(crate) o_proj: A,
    pub(crate) mlp_norm: T,
    pub(crate) mlp: M,
}

pub(crate) struct QwenGdnWeights<M, A = W<BF16>, T = W<BF16>> {
    pub(crate) norm: T,
    pub(crate) in_proj: A,
    pub(crate) gate_proj: A,
    pub(crate) a_proj: T,
    pub(crate) b_proj: T,
    pub(crate) conv_weight: T,
    pub(crate) conv_bias: T,
    pub(crate) a_log: T,
    pub(crate) dt_bias: T,
    pub(crate) rms_weight: T,
    pub(crate) out_proj: A,
    pub(crate) mlp_norm: T,
    pub(crate) mlp: M,
}

impl<M, A> QwenLayerWeights<M, A> {
    pub(crate) fn input_norm(&self) -> &DeviceSpan<BF16> {
        match self {
            Self::AttentionMlp(layer) => &layer.attn_norm,
            Self::Gdn(layer) => &layer.norm,
        }
    }

    pub(crate) fn post_attention_mlp(&self) -> (&DeviceSpan<BF16>, &M) {
        match self {
            Self::AttentionMlp(layer) => (&layer.mlp_norm, &layer.mlp),
            Self::Gdn(layer) => (&layer.mlp_norm, &layer.mlp),
        }
    }
}

/// Model storage ready for the compiled providers; the loader prepares scale layouts.
pub enum QwenWeights {
    DenseBf16(QwenModel<DenseMlp<W<BF16>>, W<BF16>, W<BF16>>),
    MoeBf16(QwenModel<MoeMlp<FusedExperts<W<BF16>>, W<BF16>>, W<BF16>, W<BF16>>),
    DenseNvfp4(QwenModel<DenseMlp<Nvfp4Block>, Fp8Block, Nvfp4Block>),
    MoeNvfp4(QwenModel<MoeMlp<SplitExperts<Nvfp4Block>, Nvfp4Block>, Fp8Block, Nvfp4Block>),
}

impl QwenWeights {
    pub(crate) fn is_quantized(&self) -> bool {
        matches!(self, Self::DenseNvfp4(_) | Self::MoeNvfp4(_))
    }
}

#[cfg(test)]
impl QwenWeights {
    pub(crate) fn random_bf16(
        ctx: Rc<CudaCtx>,
        config: &QwenConfig,
        seed: u64,
    ) -> Result<Self, Status> {
        if config.moe_config().is_some() {
            Ok(Self::MoeBf16(QwenModel::random_bf16(
                ctx,
                config,
                seed,
                MoeMlp::random_bf16,
            )?))
        } else {
            Ok(Self::DenseBf16(QwenModel::random_bf16(
                ctx,
                config,
                seed,
                DenseMlp::random_bf16,
            )?))
        }
    }
}

#[cfg_attr(not(test), allow(dead_code))]
pub struct Nvfp4Block<
    P = W<crate::dtype::Nvfp4E2M1>,
    S = W<crate::dtype::Fp8E4M3>,
    G = W<super::scales::Nvfp4Scales>,
> {
    pub(crate) parameters: G,
    pub(crate) activation: super::Nvfp4Activation,
    pub(crate) weight: P,
    pub(crate) weight_scale: S,
    pub(crate) shape: [u32; 2],
}

#[cfg_attr(not(test), allow(dead_code))]
pub struct Fp8Block<T = W<crate::dtype::Fp8E4M3>, G = W<super::scales::Fp8Scales>> {
    pub(crate) scales: G,
    pub(crate) weight: T,
    pub(crate) shape: [u32; 2],
}

pub struct DenseMlp<P> {
    pub(crate) gate_proj: P,
    pub(crate) up_proj: P,
    pub(crate) down_proj: P,
}

/// Experts stacked along the first dimension, with gate/up fused in each expert.
pub struct FusedExperts<T> {
    pub(crate) gate_up_proj: T,
    pub(crate) down_proj: T,
}

/// Individually stored experts, each with separate gate, up and down projections.
#[cfg_attr(not(test), allow(dead_code))]
pub struct SplitExperts<P> {
    pub(crate) projections: Vec<DenseMlp<P>>,
}

pub struct SharedExpert<P, T = W<BF16>> {
    pub(crate) projections: DenseMlp<P>,
    pub(crate) gate: T,
}

pub struct MoeMlp<E, P, T = W<BF16>> {
    pub(crate) router_proj: T,
    pub(crate) experts: E,
    pub(crate) shared: Option<SharedExpert<P, T>>,
}
