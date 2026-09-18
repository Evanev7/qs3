use super::*;
use crate::loader::{
    format::validate_tensor_specs,
    quantization::{ProjectionQuantization, Quantization},
};
use crate::model::{Nvfp4Activation, QwenWeights, weights::QwenLayerWeights};
use std::collections::HashMap;
use tinyjson::JsonValue;

fn selected_quantized_config() -> HashMap<String, JsonValue> {
    let model = crate::constants::engine::MODEL.trim_end_matches("-nvfp4");
    let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("models")
        .join(format!("{model}-nvfp4"))
        .join("model.json");
    parse_json_object(&std::fs::read_to_string(path).unwrap()).unwrap()
}

fn small_quantization(mode: ProjectionQuantization) -> (Qwen36TextConfig, Quantization) {
    let config = qwen_text_config(4, 16);
    let mut q = Quantization::parse(&selected_quantized_config())
        .unwrap()
        .unwrap();
    q.projections.retain(|name, _| {
        name == "lm_head"
            || name
                .strip_prefix("model.language_model.layers.")
                .unwrap()
                .split('.')
                .next()
                .unwrap()
                .parse::<usize>()
                .unwrap()
                < 4
    });
    for recipe in q.projections.values_mut() {
        if matches!(
            recipe,
            ProjectionQuantization::Nvfp4 | ProjectionQuantization::W4A16Nvfp4
        ) {
            *recipe = mode;
        }
    }
    (config, q)
}

fn zero_plan(mode: ProjectionQuantization) -> QwenLoadPlan {
    let (config, mut q) = small_quantization(mode);
    let mut entries = Vec::new();
    for spec in expected_specs(&config, Some(&q)).unwrap() {
        if spec.dtype == DynDType::F32 {
            // Distinct non-unit scalars detect conflation of scale roles.
            let value = if spec.name.ends_with("input_scale") {
                0.125
            } else {
                0.25
            };
            q.globals.insert(spec.name, value);
        } else {
            let bytes = spec.byte_len().unwrap();
            entries.push(QwenLoadPlanEntry {
                spec,
                source: QwenLoadSource::ZeroFill { bytes },
            });
        }
    }
    QwenLoadPlan {
        config,
        entries,
        quantization: Some(q),
    }
}

#[test]
fn pinned_mixed_recipe_matches_selected_geometry() {
    let root = selected_quantized_config();
    let config = Qwen36TextConfig::from_config_object(&root).unwrap();
    config.validate_compiled_model().unwrap();
    let q = Quantization::parse(&root).unwrap().unwrap();
    let specs = expected_specs(&config, Some(&q)).unwrap();
    let table = tensor_table_from_specs(&specs);
    validate_tensor_specs(&table, specs).unwrap();
}

#[test]
fn mixed_manifest_rejects_missing_extra_and_wrong_recipes() {
    let (config, q) = small_quantization(ProjectionQuantization::Nvfp4);
    for action in 0..4 {
        let mut bad = q.clone();
        match action {
            0 => {
                bad.projections.remove("lm_head");
            }
            1 => {
                bad.projections.insert(
                    "model.language_model.norm".into(),
                    ProjectionQuantization::Fp8,
                );
            }
            2 => {
                bad.projections
                    .insert("lm_head".into(), ProjectionQuantization::Fp8);
            }
            _ => {
                bad.projections.insert(
                    "model.language_model.layers.0.linear_attn.in_proj_qkv".into(),
                    ProjectionQuantization::Nvfp4,
                );
            }
        }
        assert!(expected_specs(&config, Some(&bad)).is_err());
    }
}

#[test]
fn mixed_manifest_parses_quantization_metadata() {
    for (replacement, expected) in [
        (r#"{"quant_algo":"FP8"}"#, Some(ProjectionQuantization::Fp8)),
        (
            r#"{"quant_algo":"NVFP4","group_size":16}"#,
            Some(ProjectionQuantization::Nvfp4),
        ),
        (
            r#"{"quant_algo":"W4A16_NVFP4","group_size":16}"#,
            Some(ProjectionQuantization::W4A16Nvfp4),
        ),
        (r#"{"quant_algo":"NVFP4","group_size":32}"#, None),
        (
            r#"{"quant_algo":"NVFP4","group_size":16,"zero_point":true}"#,
            None,
        ),
        (r#"{"quant_algo":"MXFP4","group_size":16}"#, None),
        (r#"{"quant_algo":"FP8","group_size":128}"#, None),
    ] {
        let mut root = selected_quantized_config();
        let q = root
            .get_mut("quantization_config")
            .unwrap()
            .get_mut::<HashMap<String, JsonValue>>()
            .unwrap();
        let layers = q
            .get_mut("quantized_layers")
            .unwrap()
            .get_mut::<HashMap<String, JsonValue>>()
            .unwrap();
        layers.insert(
            "lm_head".into(),
            JsonValue::Object(parse_json_object(replacement).unwrap()),
        );
        let parsed = Quantization::parse(&root);
        match expected {
            Some(expected) => assert_eq!(
                parsed.unwrap().unwrap().recipe("lm_head").unwrap(),
                expected
            ),
            None => assert!(parsed.is_err()),
        }
    }
}

#[test]
fn mixed_manifest_checks_packed_weight_and_scale_headers() {
    let (config, q) = small_quantization(ProjectionQuantization::Nvfp4);
    let specs = expected_specs(&config, Some(&q)).unwrap();
    for name in [
        "lm_head.weight",
        "lm_head.weight_scale",
        "lm_head.weight_scale_2",
        "lm_head.input_scale",
        "model.language_model.layers.0.linear_attn.in_proj_qkv.weight_scale",
    ] {
        for mutation in 0..3 {
            let mut table = tensor_table_from_specs(&specs);
            match mutation {
                0 => {
                    table.remove(name);
                }
                1 => {
                    table.get_mut(name).unwrap().meta.dtype = DynDType::BF16;
                }
                _ => {
                    table.get_mut(name).unwrap().meta.shape.push(1);
                }
            }
            assert!(
                validate_tensor_specs(&table, specs.clone()).is_err(),
                "{name}, mutation {mutation}"
            );
        }
    }
    let mut table = tensor_table_from_specs(&specs);
    table.insert(
        "lm_head.unexpected_scale".into(),
        table["lm_head.input_scale"].clone(),
    );
    assert!(validate_tensor_specs(&table, specs).is_err());
}

#[test]
fn both_activation_modes_materialize_and_preserve_scales() {
    let ctx = std::rc::Rc::new(crate::memory::CudaCtx::new(cuda_device_from_env()).unwrap());
    for checkpoint in [
        ProjectionQuantization::Nvfp4,
        ProjectionQuantization::W4A16Nvfp4,
    ] {
        for execution in [None, Some(Nvfp4Activation::A4), Some(Nvfp4Activation::A16)] {
            let plan = zero_plan(checkpoint);
            let loaded =
                execute_qwen_load_plan(&plan, RecordingBackend::default(), ptr::null_mut())
                    .unwrap();
            let (mut config, weights) = loaded.into_fixture_model(8).unwrap();
            assert_eq!(config.nvfp4_activation_override, None);
            config.nvfp4_activation_override = execution;
            for recipe in plan.quantization.as_ref().unwrap().projections.values() {
                if matches!(
                    recipe,
                    ProjectionQuantization::Nvfp4 | ProjectionQuantization::W4A16Nvfp4
                ) {
                    assert_eq!(*recipe, checkpoint);
                }
            }
            let head = match &weights {
                QwenWeights::DenseNvfp4(model) => {
                    let QwenLayerWeights::Gdn(layer) = &model.layers[0] else {
                        panic!("expected GDN")
                    };
                    assert_eq!(layer.in_proj.weight_scale, 0.25);
                    assert_eq!(layer.in_proj.input_scale, 0.125);
                    &model.lm_head
                }
                QwenWeights::MoeNvfp4(model) => {
                    let QwenLayerWeights::Gdn(layer) = &model.layers[0] else {
                        panic!("expected GDN")
                    };
                    assert_eq!(
                        layer.mlp.experts.projections.len(),
                        plan.config.num_experts as usize
                    );
                    &model.lm_head
                }
                _ => panic!("expected quantized model"),
            };
            assert_eq!(head.weight_scale_2, 0.25);
            assert_eq!(head.input_scale, Some(0.125));
            assert_eq!(head.shape, [16, plan.config.hidden_size]);
            assert_eq!(head.weight.len, 16 * plan.config.hidden_size as usize);
            assert_eq!(head.weight_scale.len, head.weight.len / 16);
            assert!(matches!(
                crate::model::ModelRunner::new(ctx.clone(), config, weights, 16),
                Err(Status::Unsupported)
            ));
        }
    }
    let (mut config, weights) = execute_qwen_load_plan(
        &bf16_zero_plan(tensor::selected_text_config()),
        RecordingBackend::default(),
        ptr::null_mut(),
    )
    .unwrap()
    .into_fixture_model(8)
    .unwrap();
    config.validate().unwrap();
    config.nvfp4_activation_override = Some(Nvfp4Activation::A4);
    assert!(matches!(
        crate::model::ModelRunner::new(ctx, config, weights, 16),
        Err(Status::InvalidArgument)
    ));
}

#[test]
fn materialization_rejects_missing_scales_and_unconsumed_storage() {
    for extra in [false, true] {
        let mut plan = zero_plan(ProjectionQuantization::Nvfp4);
        let globals = &mut plan.quantization.as_mut().unwrap().globals;
        if extra {
            globals.insert("unused.scale".into(), 1.0);
        } else {
            globals.remove("lm_head.weight_scale_2");
        }
        let loaded =
            execute_qwen_load_plan(&plan, RecordingBackend::default(), ptr::null_mut()).unwrap();
        assert!(loaded.into_fixture_model(8).is_err());
    }
}

#[test]
fn scale_payload_validation_rejects_nonfinite_negative_and_zero_globals() {
    let dir = env::temp_dir().join(format!("qs3-scale-values-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let (_, q) = small_quantization(ProjectionQuantization::Nvfp4);
    for (value, block, valid) in [
        (0.25f32, 0u8, true),
        (0.25, 0x7e, true),
        (0.0, 0x38, false),
        (-1.0, 0x38, false),
        (f32::NAN, 0x38, false),
        (f32::INFINITY, 0x38, false),
        (0.25, 0x7f, false),
        (0.25, 0x80, false),
        (0.25, 0xb8, false),
    ] {
        let mut payload = value.to_le_bytes().to_vec();
        payload.push(block);
        std::fs::write(dir.join("scales.safetensors"), &payload).unwrap();
        let tensors = [
            ("lm_head.weight_scale_2", DynDType::F32, vec![], 0, 4),
            ("lm_head.weight_scale", DynDType::FP8E4M3, vec![1], 4, 1),
        ]
        .into_iter()
        .map(
            |(name, dtype, shape, offset, len)| crate::loader::format::ValidatedTensor {
                spec: WeightTensorSpec {
                    name: name.into(),
                    dtype,
                    shape: shape.clone(),
                    source: WeightTensorSource::Safetensors,
                },
                file_meta: Some(TensorFileMeta {
                    shard: "scales.safetensors".into(),
                    absolute_offset: offset,
                    meta: TensorMeta {
                        dtype,
                        shape,
                        data_offsets: (offset, offset + len),
                    },
                }),
            },
        )
        .collect::<Vec<_>>();
        assert_eq!(
            q.clone().read_scales(&dir, &tensors).is_ok(),
            valid,
            "{value}, {block}"
        );
    }
    std::fs::remove_dir_all(dir).unwrap();
}

fn real_nvfp4_dir() -> std::path::PathBuf {
    if let Some(path) = env::var_os("QS3_NVFP4_MODEL_DIR") {
        return path.into();
    }
    let model = crate::constants::engine::MODEL.trim_end_matches("-nvfp4");
    let source = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("models")
        .join(format!("{model}-nvfp4"))
        .join("source.nix");
    let output = std::process::Command::new("nix").args(["eval", "--offline", "--raw", "--file"]).arg(source).args(["--apply", r#"source: "models--${builtins.replaceStrings ["/"] ["--"] source.repo}/snapshots/${source.rev}""#]).output().unwrap();
    assert!(output.status.success());
    std::path::PathBuf::from(env::var_os("HOME").unwrap())
        .join(".cache/huggingface/hub")
        .join(std::str::from_utf8(&output.stdout).unwrap())
}

#[test]
fn real_nvfp4_manifest_and_scales_validate() {
    let dir = real_nvfp4_dir();
    assert!(dir.is_dir(), "missing NVFP4 snapshot: {}", dir.display());
    let plan = QwenLoadPlan::read(dir).unwrap();
    assert!(plan.quantization.is_some());
    assert!(plan.total_bytes().unwrap() > 1 << 30);
}

#[test]
#[ignore = "full checkpoint allocations; run explicitly after the required suite"]
fn real_nvfp4_loads_both_backends() {
    let plan = QwenLoadPlan::read(real_nvfp4_dir()).unwrap();
    let ctx = crate::memory::CudaCtx::new(cuda_device_from_env()).unwrap();
    for pinned in [false, true] {
        let (_, weights) = if pinned {
            execute_qwen_load_plan(
                &plan,
                PinnedUploadBackend::new(cuda_device_from_env()).unwrap(),
                ctx.stream,
            )
            .unwrap()
            .into_qwen_model(8)
            .unwrap()
        } else {
            execute_qwen_load_plan(
                &plan,
                ManagedUmaBackend::new(cuda_device_from_env()).unwrap(),
                ctx.stream,
            )
            .unwrap()
            .into_qwen_model(8)
            .unwrap()
        };
        match &weights {
            QwenWeights::DenseNvfp4(m) => assert_model_samples(&plan, m, &ctx, |prefix, mlp| {
                assert_dense_samples(&plan, prefix, mlp, &ctx)
            }),
            QwenWeights::MoeNvfp4(m) => assert_model_samples(&plan, m, &ctx, |prefix, mlp| {
                assert_file_samples(
                    &plan,
                    &format!("{prefix}.gate.weight"),
                    &mlp.router_proj,
                    &ctx,
                );
                assert_file_samples(
                    &plan,
                    &format!("{prefix}.shared_expert_gate.weight"),
                    &mlp.shared.as_ref().unwrap().gate,
                    &ctx,
                );
                assert_dense_samples(
                    &plan,
                    &format!("{prefix}.shared_expert"),
                    &mlp.shared.as_ref().unwrap().projections,
                    &ctx,
                );
                for expert in [
                    0,
                    mlp.experts.projections.len() / 2,
                    mlp.experts.projections.len() - 1,
                ] {
                    assert_dense_samples(
                        &plan,
                        &format!("{prefix}.experts.{expert}"),
                        &mlp.experts.projections[expert],
                        &ctx,
                    );
                }
            }),
            _ => panic!("expected quantized weights"),
        }
        drop(weights);
    }
}

fn assert_file_samples<D: DType>(
    plan: &QwenLoadPlan,
    name: &str,
    span: &DeviceSpan<D>,
    ctx: &crate::memory::CudaCtx,
) {
    use std::os::unix::fs::FileExt;
    let entry = plan
        .entries
        .iter()
        .find(|entry| entry.spec.name == name)
        .unwrap();
    let QwenLoadSource::FileRange {
        shard_path,
        absolute_offset,
        bytes,
    } = &entry.source
    else {
        panic!("expected file payload")
    };
    assert_eq!(D::size_of(span.len).unwrap(), *bytes);
    let file = std::fs::File::open(shard_path).unwrap();
    let count = (*bytes).min(32);
    for offset in [0, (bytes - count) / 2, bytes - count] {
        let mut expected = vec![0u8; count];
        let mut got = vec![0u8; count];
        file.read_exact_at(&mut expected, absolute_offset + offset as u64)
            .unwrap();
        unsafe {
            result_from_cuda(ffi::cuda::cudaMemcpyAsync(
                got.as_mut_ptr().cast(),
                span.erase().cast::<u8>().add(offset).cast(),
                count,
                ffi::cuda::CUDA_MEMCPY_DEVICE_TO_HOST,
                ctx.stream,
            ))
            .unwrap();
        }
        ctx.synchronize().unwrap();
        assert_eq!(got, expected, "{name} at byte {offset}");
    }
}

fn assert_block_samples(
    plan: &QwenLoadPlan,
    name: &str,
    block: &crate::model::weights::Nvfp4Block,
    ctx: &crate::memory::CudaCtx,
) {
    assert_file_samples(plan, &format!("{name}.weight"), &block.weight, ctx);
    assert_file_samples(
        plan,
        &format!("{name}.weight_scale"),
        &block.weight_scale,
        ctx,
    );
    let globals = &plan.quantization.as_ref().unwrap().globals;
    assert_eq!(
        block.weight_scale_2,
        globals[&format!("{name}.weight_scale_2")]
    );
    assert_eq!(
        block.input_scale,
        Some(globals[&format!("{name}.input_scale")])
    );
}

fn assert_model_samples<M>(
    plan: &QwenLoadPlan,
    model: &crate::model::weights::QwenModel<
        M,
        crate::model::weights::Fp8Block,
        crate::model::weights::Nvfp4Block,
    >,
    ctx: &crate::memory::CudaCtx,
    check_mlp: impl Fn(&str, &M),
) {
    assert_eq!(model.layers.len(), plan.config.num_hidden_layers as usize);
    assert_file_samples(
        plan,
        "model.language_model.embed_tokens.weight",
        &model.token_embedding,
        ctx,
    );
    assert_file_samples(
        plan,
        "model.language_model.norm.weight",
        &model.final_norm,
        ctx,
    );
    assert_block_samples(plan, "lm_head", &model.lm_head, ctx);
    for idx in [0, model.layers.len() / 2, model.layers.len() - 1] {
        let prefix = format!("model.language_model.layers.{idx}");
        let (name, fp8, mlp) = match &model.layers[idx] {
            QwenLayerWeights::Gdn(layer) => (
                format!("{prefix}.linear_attn.in_proj_qkv"),
                &layer.in_proj,
                &layer.mlp,
            ),
            QwenLayerWeights::AttentionMlp(layer) => (
                format!("{prefix}.self_attn.q_proj"),
                &layer.q_proj,
                &layer.mlp,
            ),
        };
        assert_file_samples(plan, &format!("{name}.weight"), &fp8.weight, ctx);
        assert_eq!(
            fp8.weight.len,
            fp8.shape[0] as usize * fp8.shape[1] as usize
        );
        let globals = &plan.quantization.as_ref().unwrap().globals;
        assert_eq!(fp8.weight_scale, globals[&format!("{name}.weight_scale")]);
        assert_eq!(fp8.input_scale, globals[&format!("{name}.input_scale")]);
        check_mlp(&format!("{prefix}.mlp"), mlp);
    }
}

fn assert_dense_samples(
    plan: &QwenLoadPlan,
    prefix: &str,
    mlp: &crate::model::weights::DenseMlp<crate::model::weights::Nvfp4Block>,
    ctx: &crate::memory::CudaCtx,
) {
    assert_block_samples(plan, &format!("{prefix}.gate_proj"), &mlp.gate_proj, ctx);
    assert_block_samples(plan, &format!("{prefix}.up_proj"), &mlp.up_proj, ctx);
    assert_block_samples(plan, &format!("{prefix}.down_proj"), &mlp.down_proj, ctx);
}

#[test]
fn quantized_owners_survive_handoff_and_release_on_assembly_failure() {
    use std::{cell::Cell, rc::Rc};
    for fail_assembly in [false, true] {
        let mut plan = zero_plan(ProjectionQuantization::Nvfp4);
        let count = plan.tensor_count();
        if fail_assembly {
            plan.quantization
                .as_mut()
                .unwrap()
                .globals
                .remove("lm_head.input_scale");
        }
        let dropped = Rc::new(Cell::new(false));
        let owner_drops = Rc::new(Cell::new(0));
        let mut backend = RecordingBackend::default();
        backend.dropped = Some(dropped.clone());
        backend.owner_drops = Some(owner_drops.clone());
        let loaded = execute_qwen_load_plan(&plan, backend, ptr::null_mut()).unwrap();
        assert!(dropped.get());
        assert_eq!(owner_drops.get(), 0);
        let result = loaded.into_fixture_model(8);
        if fail_assembly {
            assert!(result.is_err());
        } else {
            let (_, weights) = result.unwrap();
            assert_eq!(owner_drops.get(), 0);
            drop(weights);
        }
        assert_eq!(owner_drops.get(), count);
    }
}
