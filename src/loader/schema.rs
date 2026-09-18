//! The Qwen checkpoint structure, walked once to plan and again to construct.
//! Planning records typed tensor requests; materialization supplies their owners.
use super::{
    format::*,
    quantization::{ProjectionQuantization, Quantization},
};
use crate::{
    dtype::{BF16, DType, DynDType, F32, Fp8E4M3, Nvfp4E2M1},
    model::weights::{
        DenseMlp, Fp8Block, FusedExperts, MoeMlp, Nvfp4Block, QwenAttentionMlpWeights,
        QwenGdnWeights, QwenLayerWeights, QwenModel, SharedExpert, SplitExperts,
    },
};
use std::collections::BTreeSet;

pub(super) trait Tensors {
    type Tensor<D: DType>;
    fn scale_parameters<D: DType>(&mut self, name: &str) -> LoadResult<Self::Tensor<D>>;
    fn tensor<D: DType>(
        &mut self,
        name: &str,
        shape: &[u32],
        source: WeightTensorSource,
    ) -> LoadResult<Self::Tensor<D>>;
    fn scalar(&mut self, name: &str) -> LoadResult<f32>;
    fn recipe(&mut self, name: &str) -> LoadResult<ProjectionQuantization>;
}

type Bf16<S> = <S as Tensors>::Tensor<BF16>;
type Fp8<S> = Fp8Block<
    <S as Tensors>::Tensor<Fp8E4M3>,
    <S as Tensors>::Tensor<crate::model::scales::Fp8Scales>,
>;
type Fp4<S> = Nvfp4Block<
    <S as Tensors>::Tensor<Nvfp4E2M1>,
    <S as Tensors>::Tensor<Fp8E4M3>,
    <S as Tensors>::Tensor<crate::model::scales::Nvfp4Scales>,
>;
type DenseBf16Model<S> = QwenModel<DenseMlp<Bf16<S>>, Bf16<S>, Bf16<S>, Bf16<S>>;
type MoeBf16Model<S> =
    QwenModel<MoeMlp<FusedExperts<Bf16<S>>, Bf16<S>, Bf16<S>>, Bf16<S>, Bf16<S>, Bf16<S>>;
type DenseNvfp4Model<S> = QwenModel<DenseMlp<Fp4<S>>, Fp8<S>, Fp4<S>, Bf16<S>>;
type MoeNvfp4Model<S> =
    QwenModel<MoeMlp<SplitExperts<Fp4<S>>, Fp4<S>, Bf16<S>>, Fp8<S>, Fp4<S>, Bf16<S>>;

fn bf16<S: Tensors>(s: &mut S, name: &str, shape: [u32; 2]) -> LoadResult<Bf16<S>> {
    s.tensor::<BF16>(
        &format!("{name}.weight"),
        &shape,
        WeightTensorSource::Safetensors,
    )
}
fn fp8<S: Tensors>(s: &mut S, name: &str, shape: [u32; 2]) -> LoadResult<Fp8<S>> {
    if s.recipe(name)? != ProjectionQuantization::Fp8 {
        return Err(WeightLoadError::invalid_config(format!(
            "expected FP8 recipe for {name}"
        )));
    }
    s.scalar(&format!("{name}.weight_scale"))?;
    s.scalar(&format!("{name}.input_scale"))?;
    Ok(Fp8Block {
        scales: s.scale_parameters(name)?,
        weight: s.tensor::<Fp8E4M3>(
            &format!("{name}.weight"),
            &shape,
            WeightTensorSource::Safetensors,
        )?,
        shape,
    })
}
fn nvfp4<S: Tensors>(s: &mut S, name: &str, shape: [u32; 2]) -> LoadResult<Fp4<S>> {
    let recipe = s.recipe(name)?;
    let (ProjectionQuantization::Nvfp4 | ProjectionQuantization::W4A16Nvfp4) = recipe else {
        return Err(WeightLoadError::invalid_config(format!(
            "expected NVFP4 recipe for {name}"
        )));
    };
    let [n, k] = shape;
    if n == 0 || k == 0 || !k.is_multiple_of(16) {
        return Err(WeightLoadError::tensor_table(
            "NVFP4 requires positive N and K divisible by 16",
        ));
    }
    s.scalar(&format!("{name}.weight_scale_2"))?;
    s.scalar(&format!("{name}.input_scale"))?;
    Ok(Nvfp4Block {
        parameters: s.scale_parameters(name)?,
        activation: match recipe {
            ProjectionQuantization::Nvfp4 => crate::model::Nvfp4Activation::A4,
            ProjectionQuantization::W4A16Nvfp4 => crate::model::Nvfp4Activation::A16,
            ProjectionQuantization::Fp8 => unreachable!(),
        },
        weight: s.tensor::<Nvfp4E2M1>(
            &format!("{name}.weight"),
            &shape,
            WeightTensorSource::Safetensors,
        )?,
        weight_scale: s.tensor::<Fp8E4M3>(
            &format!("{name}.weight_scale"),
            &[n, k / 16],
            WeightTensorSource::Safetensors,
        )?,
        shape,
    })
}

fn dense<S: Tensors, P>(
    s: &mut S,
    prefix: &str,
    hidden: u32,
    intermediate: u32,
    projection: fn(&mut S, &str, [u32; 2]) -> LoadResult<P>,
) -> LoadResult<DenseMlp<P>> {
    Ok(DenseMlp {
        gate_proj: projection(s, &format!("{prefix}.gate_proj"), [intermediate, hidden])?,
        up_proj: projection(s, &format!("{prefix}.up_proj"), [intermediate, hidden])?,
        down_proj: projection(s, &format!("{prefix}.down_proj"), [hidden, intermediate])?,
    })
}

pub(super) fn dense_bf16_model<S: Tensors>(
    s: &mut S,
    c: &Qwen36TextConfig,
) -> LoadResult<DenseBf16Model<S>> {
    model(s, c, bf16::<S>, bf16::<S>, |s, p| {
        dense(s, p, c.hidden_size, c.intermediate_size, bf16::<S>)
    })
}

pub(super) fn moe_bf16_model<S: Tensors>(
    s: &mut S,
    c: &Qwen36TextConfig,
) -> LoadResult<MoeBf16Model<S>> {
    model(s, c, bf16::<S>, bf16::<S>, |s, p| {
        Ok(MoeMlp {
            router_proj: bf16(s, &format!("{p}.gate"), [c.num_experts, c.hidden_size])?,
            experts: FusedExperts {
                gate_up_proj: s.tensor::<BF16>(
                    &format!("{p}.experts.gate_up_proj"),
                    &[c.num_experts, 2 * c.moe_intermediate_size, c.hidden_size],
                    WeightTensorSource::Safetensors,
                )?,
                down_proj: s.tensor::<BF16>(
                    &format!("{p}.experts.down_proj"),
                    &[c.num_experts, c.hidden_size, c.moe_intermediate_size],
                    WeightTensorSource::Safetensors,
                )?,
            },
            shared: Some(shared_expert(s, p, c, bf16::<S>)?),
        })
    })
}

fn shared_expert<S: Tensors, P>(
    s: &mut S,
    p: &str,
    c: &Qwen36TextConfig,
    projection: fn(&mut S, &str, [u32; 2]) -> LoadResult<P>,
) -> LoadResult<SharedExpert<P, Bf16<S>>> {
    Ok(SharedExpert {
        projections: dense(
            s,
            &format!("{p}.shared_expert"),
            c.hidden_size,
            c.shared_expert_intermediate_size,
            projection,
        )?,
        gate: bf16(s, &format!("{p}.shared_expert_gate"), [1, c.hidden_size])?,
    })
}
pub(super) fn dense_nvfp4_model<S: Tensors>(
    s: &mut S,
    c: &Qwen36TextConfig,
) -> LoadResult<DenseNvfp4Model<S>> {
    model(s, c, nvfp4::<S>, fp8::<S>, |s, p| {
        dense(s, p, c.hidden_size, c.intermediate_size, nvfp4::<S>)
    })
}
pub(super) fn moe_nvfp4_model<S: Tensors>(
    s: &mut S,
    c: &Qwen36TextConfig,
) -> LoadResult<MoeNvfp4Model<S>> {
    model(s, c, nvfp4::<S>, fp8::<S>, |s, p| {
        let experts = (0..c.num_experts)
            .map(|e| {
                dense(
                    s,
                    &format!("{p}.experts.{e}"),
                    c.hidden_size,
                    c.moe_intermediate_size,
                    nvfp4::<S>,
                )
            })
            .collect::<LoadResult<_>>()?;
        Ok(MoeMlp {
            router_proj: bf16(s, &format!("{p}.gate"), [c.num_experts, c.hidden_size])?,
            experts: SplitExperts {
                projections: experts,
            },
            shared: Some(shared_expert(s, p, c, nvfp4::<S>)?),
        })
    })
}

fn model<S: Tensors, H, A, M>(
    s: &mut S,
    c: &Qwen36TextConfig,
    head: fn(&mut S, &str, [u32; 2]) -> LoadResult<H>,
    attn: fn(&mut S, &str, [u32; 2]) -> LoadResult<A>,
    mlp: impl Fn(&mut S, &str) -> LoadResult<M>,
) -> LoadResult<QwenModel<M, A, H, Bf16<S>>> {
    use crate::constants::{
        attention::{HEAD_DIM, KV_WIDTH, PACKED_Q_GATE_WIDTH, Q_WIDTH},
        gdn::{CONV_WIDTH, OUTPUT_WIDTH, PACKED_QKV_CHANNELS},
    };
    let h = c.hidden_size;
    let token_embedding = bf16(s, "model.language_model.embed_tokens", [c.vocab_size, h])?;
    let final_norm = s.tensor::<BF16>(
        "model.language_model.norm.weight",
        &[h],
        WeightTensorSource::Safetensors,
    )?;
    let lm_head = head(s, "lm_head", [c.vocab_size, h])?;
    let mut layers = Vec::with_capacity(c.layer_types.len());
    for (idx, kind) in c.layer_types.iter().enumerate() {
        let p = format!("model.language_model.layers.{idx}");
        let norm = s.tensor::<BF16>(
            &format!("{p}.input_layernorm.weight"),
            &[h],
            WeightTensorSource::Safetensors,
        )?;
        let mlp_norm = s.tensor::<BF16>(
            &format!("{p}.post_attention_layernorm.weight"),
            &[h],
            WeightTensorSource::Safetensors,
        )?;
        let mlp = mlp(s, &format!("{p}.mlp"))?;
        layers.push(match kind {
            QwenLayerKind::FullAttention => {
                let p = format!("{p}.self_attn");
                QwenLayerWeights::AttentionMlp(QwenAttentionMlpWeights {
                    attn_norm: norm,
                    mlp_norm,
                    mlp,
                    q_norm: s.tensor::<BF16>(
                        &format!("{p}.q_norm.weight"),
                        &[HEAD_DIM],
                        WeightTensorSource::Safetensors,
                    )?,
                    k_norm: s.tensor::<BF16>(
                        &format!("{p}.k_norm.weight"),
                        &[HEAD_DIM],
                        WeightTensorSource::Safetensors,
                    )?,
                    q_proj: attn(s, &format!("{p}.q_proj"), [PACKED_Q_GATE_WIDTH, h])?,
                    k_proj: attn(s, &format!("{p}.k_proj"), [KV_WIDTH, h])?,
                    v_proj: attn(s, &format!("{p}.v_proj"), [KV_WIDTH, h])?,
                    o_proj: attn(s, &format!("{p}.o_proj"), [h, Q_WIDTH])?,
                })
            }
            QwenLayerKind::LinearAttention => {
                let p = format!("{p}.linear_attn");
                QwenLayerWeights::Gdn(QwenGdnWeights {
                    norm,
                    mlp_norm,
                    mlp,
                    in_proj: attn(s, &format!("{p}.in_proj_qkv"), [PACKED_QKV_CHANNELS, h])?,
                    gate_proj: attn(s, &format!("{p}.in_proj_z"), [OUTPUT_WIDTH, h])?,
                    out_proj: attn(s, &format!("{p}.out_proj"), [h, OUTPUT_WIDTH])?,
                    a_proj: bf16(s, &format!("{p}.in_proj_a"), [c.linear_num_value_heads, h])?,
                    b_proj: bf16(s, &format!("{p}.in_proj_b"), [c.linear_num_value_heads, h])?,
                    conv_weight: s.tensor::<BF16>(
                        &format!("{p}.conv1d.weight"),
                        &[PACKED_QKV_CHANNELS, 1, CONV_WIDTH],
                        WeightTensorSource::Safetensors,
                    )?,
                    conv_bias: s.tensor::<BF16>(
                        &format!("{p}.conv1d.bias"),
                        &[PACKED_QKV_CHANNELS],
                        WeightTensorSource::ZeroFill,
                    )?,
                    a_log: s.tensor::<BF16>(
                        &format!("{p}.A_log"),
                        &[c.linear_num_value_heads],
                        WeightTensorSource::Safetensors,
                    )?,
                    dt_bias: s.tensor::<BF16>(
                        &format!("{p}.dt_bias"),
                        &[c.linear_num_value_heads],
                        WeightTensorSource::Safetensors,
                    )?,
                    rms_weight: s.tensor::<BF16>(
                        &format!("{p}.norm.weight"),
                        &[c.linear_value_head_dim],
                        WeightTensorSource::Safetensors,
                    )?,
                })
            }
        });
    }
    Ok(QwenModel {
        token_embedding,
        final_norm,
        lm_head,
        layers,
    })
}

/// Safetensors uses U8 pairs for logical E2M1 tensors.
pub(super) fn tensor_spec<D: DType>(
    name: &str,
    shape: &[u32],
    source: WeightTensorSource,
) -> WeightTensorSpec {
    let mut shape = shape.to_vec();
    let dtype = match D::RAW {
        crate::ffi::DTYPE_BF16 => DynDType::BF16,
        crate::ffi::DTYPE_F32 => DynDType::F32,
        crate::ffi::DTYPE_FP8_E4M3 => DynDType::FP8E4M3,
        crate::ffi::DTYPE_NVFP4_E2M1 => {
            *shape.last_mut().unwrap() /= 2;
            DynDType::U8
        }
        _ => unreachable!("unsupported Qwen storage type"),
    };
    WeightTensorSpec {
        name: name.into(),
        dtype,
        shape,
        source,
    }
}

pub(super) fn expected_specs(
    c: &Qwen36TextConfig,
    quantization: Option<&Quantization>,
) -> LoadResult<Vec<WeightTensorSpec>> {
    struct Specs<'a> {
        tensors: Vec<WeightTensorSpec>,
        quantization: Option<&'a Quantization>,
        recipes: BTreeSet<String>,
    }
    impl Tensors for Specs<'_> {
        type Tensor<D: DType> = ();
        fn scale_parameters<D: DType>(&mut self, _: &str) -> LoadResult<()> {
            Ok(())
        }
        fn tensor<D: DType>(
            &mut self,
            name: &str,
            shape: &[u32],
            source: WeightTensorSource,
        ) -> LoadResult<()> {
            self.tensors.push(tensor_spec::<D>(name, shape, source));
            Ok(())
        }
        fn scalar(&mut self, name: &str) -> LoadResult<f32> {
            self.tensor::<F32>(name, &[], WeightTensorSource::Safetensors)?;
            Ok(1.0)
        }
        fn recipe(&mut self, name: &str) -> LoadResult<ProjectionQuantization> {
            self.recipes.insert(Quantization::recipe_name(name));
            self.quantization.unwrap().recipe(name)
        }
    }
    c.validate_supported()?;
    let mut s = Specs {
        tensors: Vec::new(),
        quantization,
        recipes: BTreeSet::new(),
    };
    if quantization.is_some() {
        if c.num_experts > 0 {
            moe_nvfp4_model(&mut s, c)?;
        } else {
            dense_nvfp4_model(&mut s, c)?;
        }
    } else if c.num_experts > 0 {
        moe_bf16_model(&mut s, c)?;
    } else {
        dense_bf16_model(&mut s, c)?;
    }
    if let Some(q) = quantization {
        if s.recipes != q.projections.keys().cloned().collect() {
            return Err(WeightLoadError::invalid_config(
                "quantized_layers does not match the compiled Qwen mixed recipe",
            ));
        }
    }
    Ok(s.tensors)
}
