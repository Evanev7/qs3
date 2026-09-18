use super::*;
use crate::loader::{
    format::validate_tensor_specs,
    quantization::{ProjectionQuantization, Quantization},
};
use crate::model::{Nvfp4Activation, QwenWeights, weights::QwenLayerWeights};
use std::collections::HashMap;
use tinyjson::JsonValue;

fn scalar_record<D: DType>(span: &DeviceSpan<D>, ctx: &crate::memory::CudaCtx) -> [f32; 2] {
    let mut bytes = [0u8; 8];
    assert_eq!(D::size_of(span.len).unwrap(), bytes.len());
    unsafe {
        ctx.download(span.as_raw(), &mut bytes).unwrap();
    }
    ctx.synchronize().unwrap();
    [
        f32::from_ne_bytes(bytes[..4].try_into().unwrap()),
        f32::from_ne_bytes(bytes[4..].try_into().unwrap()),
    ]
}

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
            let loaded = execute_qwen_load_plan(
                &plan,
                ManagedUmaBackend::new(cuda_device_from_env()).unwrap(),
                &ctx,
            )
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
                    assert_eq!(scalar_record(&layer.in_proj.scales, &ctx), [0.125, 0.25]);
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
            let activation = if checkpoint == ProjectionQuantization::Nvfp4 {
                Nvfp4Activation::A4
            } else {
                Nvfp4Activation::A16
            };
            assert_eq!(head.activation, activation);
            assert_eq!(scalar_record(&head.parameters, &ctx), [8.0, 0.03125]);
            assert_eq!(head.shape, [16, plan.config.hidden_size]);
            assert_eq!(head.weight.len, 16 * plan.config.hidden_size as usize);
            assert_eq!(
                head.weight_scale.len,
                crate::backend::qsfi::scale_count(head.shape[0], head.shape[1]).unwrap() as usize
            );
            // A16 and quantized MoE remain explicit execution rejections.
            if execution.unwrap_or(activation) == Nvfp4Activation::A16
                || matches!(weights, QwenWeights::MoeNvfp4(_))
            {
                assert!(matches!(
                    crate::model::ModelRunner::new(ctx.clone(), config, weights, 16),
                    Err(Status::Unsupported)
                ));
            }
        }
    }
    let (mut config, weights) = execute_qwen_load_plan(
        &bf16_zero_plan(tensor::selected_text_config()),
        RecordingBackend::default(),
        &ctx,
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
    let ctx = crate::memory::CudaCtx::new(cuda_device_from_env()).unwrap();
    for extra in [false, true] {
        let plan = zero_plan(ProjectionQuantization::Nvfp4);
        let mut loaded = execute_qwen_load_plan(
            &plan,
            ManagedUmaBackend::new(cuda_device_from_env()).unwrap(),
            &ctx,
        )
        .unwrap();
        let globals = &mut loaded.quantization.as_mut().unwrap().globals;
        if extra {
            globals.insert("unused.scale".into(), 1.0);
        } else {
            globals.remove("lm_head.weight_scale_2");
        }
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
                &ctx,
            )
            .unwrap()
            .into_qwen_model(8)
            .unwrap()
        } else {
            execute_qwen_load_plan(
                &plan,
                ManagedUmaBackend::new(cuda_device_from_env()).unwrap(),
                &ctx,
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

#[test]
#[ignore = "full dense checkpoint execution; run explicitly after the required suite"]
fn real_nvfp4_dense_prefill_decode_and_reset() {
    use crate::{ModelRunner, QwenRequest, QwenTokenizer};
    let directory = real_nvfp4_dir();
    let plan = QwenLoadPlan::read(&directory).unwrap();
    assert_eq!(
        plan.config.num_experts, 0,
        "this probe exercises dense W4A4"
    );
    let tokenizer = QwenTokenizer::from_model_dir(&directory).unwrap();
    let ctx = std::rc::Rc::new(crate::memory::CudaCtx::new(cuda_device_from_env()).unwrap());
    let (config, weights) = execute_qwen_load_plan(
        &plan,
        PinnedUploadBackend::new(cuda_device_from_env()).unwrap(),
        &ctx,
    )
    .unwrap()
    .into_qwen_model(16)
    .unwrap();
    let mut runner = ModelRunner::new(ctx, config, weights, tokenizer.token_count()).unwrap();
    // Saved vLLM probe: nvfp4-vllm_checkpoint_probe-20260914T203250Z-qpls36fp.
    // Force the same prefixes; report numerical differences without claiming
    // this smoke/replay check is an external model-quality acceptance test.
    let prompt = QwenRequest {
        request_id: 93,
        tokens: &[1, 2, 3, 4],
        max_new_tokens: 0,
    };
    let mut first = Vec::new();
    for pass in 0..2 {
        runner.reset().unwrap();
        runner.run(prompt).unwrap();
        for step in 0..4 {
            if step > 0 {
                runner
                    .decode_forced_token_for_test([5, 0, 31][step - 1])
                    .unwrap();
            }
            let logits = runner.last_logits_row_for_test().unwrap();
            assert!(logits.iter().all(|v| v.is_finite()));
            assert!(logits.windows(2).any(|v| v[0] != v[1]));
            if pass == 0 {
                let mut ids: Vec<_> = (0..tokenizer.token_count()).collect();
                ids.sort_by(|&a, &b| logits[b].total_cmp(&logits[a]).then(a.cmp(&b)));
                let max = f64::from(logits[ids[0]]);
                let log_z = max
                    + logits
                        .iter()
                        .map(|v| (f64::from(*v) - max).exp())
                        .sum::<f64>()
                        .ln();
                let top: Vec<_> = ids[..5]
                    .iter()
                    .map(|&i| (i, logits[i], f64::from(logits[i]) - log_z))
                    .collect();
                eprintln!("NVFP4 forced-prefix step {step} (id, logit, logprob): {top:?}");
                first.push(logits);
            } else {
                assert_eq!(logits, first[step], "reset/replay step {step}");
            }
        }
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
        if entry.spec.dtype == DynDType::FP8E4M3
            && let Some(projection) = name.strip_suffix(".weight")
            && let Some(q) = &plan.quantization
        {
            let (scales, requantize) = reference_fp8_scales(q, projection);
            if requantize {
                let old = q.globals[&format!("{projection}.weight_scale")];
                for code in &mut expected {
                    *code = reference_requantize(*code, old, scales[1]);
                }
            }
        }
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
    assert_swizzled_scale_samples(
        plan,
        &format!("{name}.weight_scale"),
        &block.weight_scale,
        ctx,
    );
    let globals = &plan.quantization.as_ref().unwrap().globals;
    let input = globals[&format!("{name}.input_scale")];
    let weight = globals[&format!("{name}.weight_scale_2")];
    assert_eq!(
        scalar_record(&block.parameters, ctx),
        [1.0 / input, input * weight]
    );
}

fn assert_swizzled_scale_samples(
    plan: &QwenLoadPlan,
    name: &str,
    span: &DeviceSpan<Fp8E4M3>,
    ctx: &crate::memory::CudaCtx,
) {
    use std::os::unix::fs::FileExt;
    let entry = plan.entries.iter().find(|e| e.spec.name == name).unwrap();
    let [rows, cols] = entry.spec.shape.as_slice() else {
        panic!("scale matrix");
    };
    let rows = *rows as usize;
    let cols = *cols as usize;
    assert_eq!(span.len, rows.div_ceil(128) * 128 * cols);
    let QwenLoadSource::FileRange {
        shard_path,
        absolute_offset,
        ..
    } = &entry.source
    else {
        panic!("file scales");
    };
    let file = std::fs::File::open(shard_path).unwrap();
    // Compare complete sampled rows, covering both swizzle axes and padded rows.
    for r in [0, rows / 2, rows - 1, rows.div_ceil(128) * 128 - 1] {
        let mut expected = vec![0u8; cols];
        if r < rows {
            file.read_exact_at(&mut expected, absolute_offset + (r * cols) as u64)
                .unwrap();
        }
        let mut tile = vec![0u8; 128 * cols];
        unsafe {
            ctx.download(span.as_raw().add((r / 128) * 128 * cols), &mut tile)
                .unwrap();
        }
        ctx.synchronize().unwrap();
        let got: Vec<_> = (0..cols)
            .map(|c| tile[(c / 4) * 512 + (r % 32) * 16 + ((r % 128) / 32) * 4 + c % 4])
            .collect();
        assert_eq!(got, expected, "{name} row {r}");
    }
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
        assert_eq!(
            scalar_record(&fp8.scales, ctx),
            reference_fp8_scales(plan.quantization.as_ref().unwrap(), &name).0
        );
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

struct TrackingBuffer<D: DType> {
    inner: crate::loader::transfer::CudaWeightBuffer<D>,
    drops: std::rc::Rc<std::cell::Cell<usize>>,
}
impl<D: DType> std::ops::Deref for TrackingBuffer<D> {
    type Target = DeviceSpan<D>;
    fn deref(&self) -> &Self::Target {
        &self.inner
    }
}
impl<D: DType> Drop for TrackingBuffer<D> {
    fn drop(&mut self) {
        self.drops.set(self.drops.get() + 1);
    }
}
struct TrackingBackend {
    inner: Option<ManagedUmaBackend>,
    allocations: std::rc::Rc<std::cell::Cell<usize>>,
    drops: std::rc::Rc<std::cell::Cell<usize>>,
    dropped: std::rc::Rc<std::cell::Cell<bool>>,
    fail_at: Option<usize>,
}
impl Drop for TrackingBackend {
    fn drop(&mut self) {
        self.dropped.set(true);
    }
}
impl WeightLoadBackend for TrackingBackend {
    type Buffer<D: DType> = TrackingBuffer<D>;
    type Stats = ();
    fn alloc_tensor(&mut self, desc: WeightTensorDesc<'_>) -> Result<WeightLoadSpan, Status> {
        if self.fail_at == Some(self.allocations.get()) {
            return Err(Status::OutOfMemory);
        }
        let span = self.inner.as_mut().unwrap().alloc_tensor(desc)?;
        self.allocations.set(self.allocations.get() + 1);
        Ok(span)
    }
    fn read_exact(
        &mut self,
        src: WeightFileRange<'_>,
        dst: &WeightLoadSpan,
        stream: *mut c_void,
    ) -> Result<(), Status> {
        self.inner.as_mut().unwrap().read_exact(src, dst, stream)
    }
    fn zero_fill(&mut self, dst: &WeightLoadSpan, stream: *mut c_void) -> Result<(), Status> {
        self.inner.as_mut().unwrap().zero_fill(dst, stream)
    }
    fn finish(mut self, stream: *mut c_void) -> Result<(Vec<WeightBuffer<Self>>, ()), Status> {
        let (buffers, _) = self.inner.take().unwrap().finish(stream)?;
        let buffers = buffers
            .into_iter()
            .map(|b| match b {
                WeightBuffer::Bf16(inner) => WeightBuffer::Bf16(TrackingBuffer {
                    inner,
                    drops: self.drops.clone(),
                }),
                WeightBuffer::F32(inner) => WeightBuffer::F32(TrackingBuffer {
                    inner,
                    drops: self.drops.clone(),
                }),
                WeightBuffer::Fp8E4m3(inner) => WeightBuffer::Fp8E4m3(TrackingBuffer {
                    inner,
                    drops: self.drops.clone(),
                }),
                WeightBuffer::U8(inner) => WeightBuffer::U8(TrackingBuffer {
                    inner,
                    drops: self.drops.clone(),
                }),
            })
            .collect();
        Ok((buffers, ()))
    }
}

#[test]
fn quantized_owners_survive_handoff_and_release_on_assembly_failure() {
    use std::{cell::Cell, rc::Rc};
    let ctx = crate::memory::CudaCtx::new(cuda_device_from_env()).unwrap();
    for failure in 0..3 {
        let plan = zero_plan(ProjectionQuantization::Nvfp4);
        let source_count = plan
            .entries
            .iter()
            .filter(|e| e.spec.name.ends_with(".weight_scale"))
            .count();
        let allocations = Rc::new(Cell::new(0));
        let dropped = Rc::new(Cell::new(false));
        let drops = Rc::new(Cell::new(0));
        let backend = TrackingBackend {
            inner: Some(ManagedUmaBackend::new(cuda_device_from_env()).unwrap()),
            allocations: allocations.clone(),
            drops: drops.clone(),
            dropped: dropped.clone(),
            // Fail after the first scale swizzle was submitted, before its scalar upload.
            fail_at: (failure == 2).then_some(plan.tensor_count() + 1),
        };
        let result = execute_qwen_load_plan(&plan, backend, &ctx);
        assert!(dropped.get());
        if failure == 2 {
            assert!(result.is_err());
        } else {
            let mut loaded = result.unwrap();
            assert_eq!(
                drops.get(),
                source_count,
                "only retired source scales are released"
            );
            if failure == 1 {
                loaded
                    .quantization
                    .as_mut()
                    .unwrap()
                    .globals
                    .remove("lm_head.input_scale");
            }
            let result = loaded.into_fixture_model(8);
            if failure == 1 {
                assert!(result.is_err());
            } else {
                let (_, weights) = result.unwrap();
                assert_eq!(drops.get(), source_count);
                drop(weights);
            }
        }
        assert_eq!(drops.get(), allocations.get());
    }
}

#[test]
fn invalid_preparation_scalars_fail_before_device_allocation() {
    let ctx = crate::memory::CudaCtx::new(cuda_device_from_env()).unwrap();
    use std::{cell::Cell, rc::Rc};
    for fault in 0..4 {
        let mut plan = zero_plan(ProjectionQuantization::Nvfp4);
        let globals = &mut plan.quantization.as_mut().unwrap().globals;
        match fault {
            0 => {
                globals.remove("lm_head.input_scale");
            }
            1 => {
                globals.insert("unused.scale".into(), 1.0);
            }
            2 => {
                globals.insert("lm_head.input_scale".into(), f32::from_bits(1));
            }
            _ => {
                globals.insert("lm_head.weight_scale_2".into(), f32::NAN);
            }
        }
        let allocations = Rc::new(Cell::new(0));
        let backend = TrackingBackend {
            inner: Some(ManagedUmaBackend::new(cuda_device_from_env()).unwrap()),
            allocations: allocations.clone(),
            drops: Rc::new(Cell::new(0)),
            dropped: Rc::new(Cell::new(false)),
            fail_at: None,
        };
        assert!(execute_qwen_load_plan(&plan, backend, &ctx).is_err());
        assert_eq!(allocations.get(), 0);
    }
}

#[test]
fn both_load_backends_swizzle_scales_and_zero_padding_before_handoff() {
    use crate::loader::plan::LoadedWeightPlan;
    let (n, k) = (136u32, 256u32);
    let cols = k / 16;
    let path = env::temp_dir().join(format!("qs3-swizzle-{}", std::process::id()));
    let packed = vec![0x12u8; (n * k / 2) as usize];
    let scales: Vec<_> = (0..n * cols)
        .map(|i| [0x30, 0x38, 0x40][i as usize % 3])
        .collect();
    let mut bytes = packed.clone();
    bytes.extend_from_slice(&scales);
    std::fs::write(&path, bytes).unwrap();
    let plan = QwenLoadPlan {
        config: qwen_text_config(4, 16),
        entries: vec![
            QwenLoadPlanEntry {
                spec: WeightTensorSpec {
                    name: "lm_head.weight".into(),
                    dtype: DynDType::U8,
                    shape: vec![n, k / 2],
                    source: WeightTensorSource::Safetensors,
                },
                source: QwenLoadSource::FileRange {
                    shard_path: path.clone(),
                    absolute_offset: 0,
                    bytes: packed.len(),
                },
            },
            QwenLoadPlanEntry {
                spec: WeightTensorSpec {
                    name: "lm_head.weight_scale".into(),
                    dtype: DynDType::FP8E4M3,
                    shape: vec![n, cols],
                    source: WeightTensorSource::Safetensors,
                },
                source: QwenLoadSource::FileRange {
                    shard_path: path.clone(),
                    absolute_offset: packed.len() as u64,
                    bytes: scales.len(),
                },
            },
        ],
        quantization: Some(Quantization {
            projections: [("lm_head".into(), ProjectionQuantization::Nvfp4)]
                .into_iter()
                .collect(),
            globals: [
                ("lm_head.input_scale".into(), 0.125),
                ("lm_head.weight_scale_2".into(), 0.25),
            ]
            .into_iter()
            .collect(),
        }),
    };
    fn check<B: WeightLoadBackend>(
        loaded: LoadedWeightPlan<B>,
        expected: &[u8],
        n: u32,
        cols: u32,
        ctx: &crate::memory::CudaCtx,
    ) {
        let WeightBuffer::Fp8E4m3(buffer) = &loaded.tensors[1].buffer else {
            panic!("scale storage");
        };
        assert_eq!(buffer.len, n.div_ceil(128) as usize * 128 * cols as usize);
        let mut bytes = vec![0u8; buffer.len];
        unsafe {
            ctx.download(buffer.as_raw(), &mut bytes).unwrap();
        }
        ctx.synchronize().unwrap();
        // Walk destination coordinates in layout order, independent of the CUDA
        // kernel's source-index-to-destination-offset calculation.
        let mut offset = 0;
        for tile in 0..n.div_ceil(128) {
            for column_group in 0..cols / 4 {
                for row in 0..32 {
                    for group in 0..4 {
                        for column in 0..4 {
                            let r = tile * 128 + group * 32 + row;
                            let c = column_group * 4 + column;
                            let value = if r < n {
                                expected[(r * cols + c) as usize]
                            } else {
                                0
                            };
                            assert_eq!(bytes[offset], value, "scale ({r}, {c})");
                            offset += 1;
                        }
                    }
                }
            }
        }
        let WeightBuffer::F32(record) = &loaded.scale_parameters["lm_head"] else {
            panic!("scalar record");
        };
        assert_eq!(scalar_record(&**record, ctx), [8.0, 0.03125]);
    }
    let ctx = crate::memory::CudaCtx::new(cuda_device_from_env()).unwrap();
    check(
        execute_qwen_load_plan(
            &plan,
            ManagedUmaBackend::new(cuda_device_from_env()).unwrap(),
            &ctx,
        )
        .unwrap(),
        &scales,
        n,
        cols,
        &ctx,
    );
    check(
        execute_qwen_load_plan(
            &plan,
            PinnedUploadBackend::new(cuda_device_from_env()).unwrap(),
            &ctx,
        )
        .unwrap(),
        &scales,
        n,
        cols,
        &ctx,
    );
    std::fs::remove_file(path).unwrap();
}

// Independent test reference: decode E4M3 arithmetically, round onto the FP16
// number grid, then search the finite E4M3 codebook with ties to even.
fn reference_requantize(code: u8, old: f32, new: f32) -> u8 {
    fn decode(code: u8) -> f32 {
        let e = (code >> 3) & 15;
        let m = code & 7;
        if e == 0 {
            f32::from(m) * 2.0f32.powi(-9)
        } else {
            (1.0 + f32::from(m) / 8.0) * 2.0f32.powi(i32::from(e) - 7)
        }
    }
    assert_ne!(code & 127, 127, "nonfinite checkpoint weight");
    let value = decode(code & 127) * old;
    let half = if value >= 65520.0 {
        f32::INFINITY
    } else if value == 0.0 {
        0.0
    } else {
        let quantum = 2.0f32.powi((value.log2().floor() as i32 - 10).max(-24));
        (value / quantum).round_ties_even() * quantum
    };
    let target = (half / new).min(448.0);
    let result = (0..=126u8)
        .min_by(|&a, &b| {
            (decode(a) - target)
                .abs()
                .total_cmp(&(decode(b) - target).abs())
                .then((a & 1).cmp(&(b & 1)))
        })
        .unwrap();
    result | (code & 128)
}

fn reference_fp8_scales(q: &Quantization, name: &str) -> ([f32; 2], bool) {
    let (prefix, leaf) = name.rsplit_once('.').unwrap();
    let group = match leaf {
        "in_proj_qkv" | "in_proj_z" => vec!["in_proj_qkv", "in_proj_z"],
        "q_proj" | "k_proj" | "v_proj" => vec!["q_proj", "k_proj", "v_proj"],
        _ => vec![leaf],
    };
    let mut values = [0.0f32; 2];
    let old = q.globals[&format!("{name}.weight_scale")];
    let mut requantize = false;
    for member in group {
        for (i, scale) in ["input_scale", "weight_scale"].iter().enumerate() {
            let v = q.globals[&format!("{prefix}.{member}.{scale}")];
            values[i] = values[i].max(v);
            requantize |= i == 1 && v != old;
        }
    }
    (values, requantize)
}

#[test]
fn fp8_fused_groups_prepare_weights_and_scales_on_both_backends() {
    let ctx = crate::memory::CudaCtx::new(cuda_device_from_env()).unwrap();
    let path = env::temp_dir().join(format!("qs3-fp8-fusion-{}", std::process::id()));
    let payload = [0x38u8, 0x40, 0x48, 0xb8, 0xc0, 0xc8, 0, 0x80];
    std::fs::write(&path, payload).unwrap();
    let names = [
        "model.language_model.layers.0.linear_attn.in_proj_qkv",
        "model.language_model.layers.0.linear_attn.in_proj_z",
        "model.language_model.layers.3.self_attn.q_proj",
        "model.language_model.layers.3.self_attn.k_proj",
        "model.language_model.layers.3.self_attn.v_proj",
        "model.language_model.layers.3.self_attn.o_proj",
    ];
    let weights = [0.25, 0.5, 0.25, 0.5, 1.0, 0.25];
    let inputs = [0.125, 0.25, 0.5, 0.25, 0.125, 0.125];
    let mut q = Quantization {
        projections: BTreeMap::new(),
        globals: BTreeMap::new(),
    };
    let mut entries = Vec::new();
    for (i, name) in names.iter().enumerate() {
        q.projections
            .insert((*name).into(), ProjectionQuantization::Fp8);
        q.globals.insert(format!("{name}.input_scale"), inputs[i]);
        q.globals.insert(format!("{name}.weight_scale"), weights[i]);
        entries.push(QwenLoadPlanEntry {
            spec: WeightTensorSpec {
                name: format!("{name}.weight"),
                dtype: DynDType::FP8E4M3,
                shape: vec![2, 4],
                source: WeightTensorSource::Safetensors,
            },
            source: QwenLoadSource::FileRange {
                shard_path: path.clone(),
                absolute_offset: 0,
                bytes: payload.len(),
            },
        });
    }
    let plan = QwenLoadPlan {
        config: qwen_text_config(4, 16),
        entries,
        quantization: Some(q),
    };
    fn check<B: WeightLoadBackend>(
        loaded: crate::loader::plan::LoadedWeightPlan<B>,
        ctx: &crate::memory::CudaCtx,
    ) {
        let expected = [
            [0x30u8, 0x38, 0x40, 0xb0, 0xb8, 0xc0, 0, 0x80],
            [0x38, 0x40, 0x48, 0xb8, 0xc0, 0xc8, 0, 0x80],
            [0x28, 0x30, 0x38, 0xa8, 0xb0, 0xb8, 0, 0x80],
            [0x30, 0x38, 0x40, 0xb0, 0xb8, 0xc0, 0, 0x80],
            [0x38, 0x40, 0x48, 0xb8, 0xc0, 0xc8, 0, 0x80],
            [0x38, 0x40, 0x48, 0xb8, 0xc0, 0xc8, 0, 0x80],
        ];
        let scales = [
            [0.25, 0.5],
            [0.25, 0.5],
            [0.5, 1.0],
            [0.5, 1.0],
            [0.5, 1.0],
            [0.125, 0.25],
        ];
        for (i, tensor) in loaded.tensors.iter().enumerate() {
            let WeightBuffer::Fp8E4m3(buffer) = &tensor.buffer else {
                panic!("FP8 buffer")
            };
            let mut got = [0u8; 8];
            unsafe {
                ctx.download(buffer.as_raw(), &mut got).unwrap();
            }
            ctx.synchronize().unwrap();
            assert_eq!(got, expected[i]);
            let name = tensor.spec.name.strip_suffix(".weight").unwrap();
            let WeightBuffer::F32(params) = &loaded.scale_parameters[name] else {
                panic!("scale buffer")
            };
            assert_eq!(scalar_record(params, ctx), scales[i]);
        }
    }
    check(
        execute_qwen_load_plan(
            &plan,
            ManagedUmaBackend::new(cuda_device_from_env()).unwrap(),
            &ctx,
        )
        .unwrap(),
        &ctx,
    );
    check(
        execute_qwen_load_plan(
            &plan,
            PinnedUploadBackend::new(cuda_device_from_env()).unwrap(),
            &ctx,
        )
        .unwrap(),
        &ctx,
    );
    std::fs::remove_file(path).unwrap();
}
