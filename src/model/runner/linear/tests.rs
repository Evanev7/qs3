use super::*;
use crate::{
    dtype::DType,
    memory::HostBuffer,
    model::{ModelRunner, QwenRequest},
    model::{
        Nvfp4Activation,
        scales::{Fp8Scales, Nvfp4Scales},
    },
};

fn device<D: DType>(ctx: &Rc<CudaCtx>, bytes: &[u8]) -> DeviceBuffer<D> {
    let mut host = HostBuffer::<D>::new(D::len_of(bytes.len()).unwrap()).unwrap();
    host.as_mut().copy_from_slice(bytes);
    host.upload(ctx.clone()).unwrap()
}
fn record<D: DType>(ctx: &Rc<CudaCtx>, values: [f32; 2]) -> W<D> {
    Box::new(device::<D>(
        ctx,
        &values
            .into_iter()
            .flat_map(f32::to_ne_bytes)
            .collect::<Vec<_>>(),
    ))
}
fn bf16(ctx: &Rc<CudaCtx>, values: &[f32]) -> DeviceBuffer<BF16> {
    device(
        ctx,
        &values
            .iter()
            .flat_map(|v| ((v.to_bits() >> 16) as u16).to_ne_bytes())
            .collect::<Vec<_>>(),
    )
}
fn download(buffer: &DeviceBuffer<BF16>, ctx: &CudaCtx) -> Vec<f32> {
    let mut host = HostBuffer::new(buffer.len()).unwrap();
    unsafe {
        buffer.download(&mut host).unwrap();
    }
    ctx.synchronize().unwrap();
    host.as_ref()
        .chunks_exact(2)
        .map(|b| f32::from_bits(u32::from(u16::from_ne_bytes(b.try_into().unwrap())) << 16))
        .collect()
}
fn fp4_value(code: u8) -> f32 {
    let value = [0., 0.5, 1., 1.5, 2., 3., 4., 6.][(code & 7) as usize];
    if code & 8 == 0 { value } else { -value }
}
fn fp8_value(code: u8) -> f32 {
    let e = (code >> 3) & 15;
    let m = code & 7;
    let v = if e == 0 {
        f32::from(m) / 512.0
    } else {
        (1.0 + f32::from(m) / 8.0) * 2f32.powi(i32::from(e) - 7)
    };
    if code & 128 == 0 { v } else { -v }
}
fn round_fp8(value: f32) -> f32 {
    // Independent exhaustive nearest-even reference over finite positive E4M3.
    let best = (0u8..127)
        .min_by(|a, b| {
            (fp8_value(*a) - value.abs())
                .abs()
                .total_cmp(&(fp8_value(*b) - value.abs()).abs())
                .then_with(|| (a & 1).cmp(&(b & 1)))
        })
        .unwrap();
    fp8_value(best).copysign(value)
}
fn fp4_block(ctx: &Rc<CudaCtx>, n: u32, k: u32) -> Nvfp4Block {
    let bytes: Vec<u8> = (0..n as usize * k as usize / 2)
        .map(|i| {
            let code = |j: usize| ((j + j / k as usize * 3) % 16) as u8;
            code(2 * i) | (code(2 * i + 1) << 4)
        })
        .collect();
    let scales: Vec<u8> = (0..n * k / 16)
        .map(|i| [0x30, 0x38, 0x40][i as usize % 3])
        .collect();
    // Independent CPU construction of the provider layout for numerical tests.
    let cols = (k / 16) as usize;
    let mut swizzled = vec![0; scale_count(n, k).unwrap() as usize];
    for r in 0..n as usize {
        for c in 0..cols {
            let offset = (r / 128) * (128 * cols)
                + (c / 4) * 512
                + (r % 32) * 16
                + ((r % 128) / 32) * 4
                + c % 4;
            swizzled[offset] = scales[r * cols + c];
        }
    }
    Nvfp4Block {
        activation: Nvfp4Activation::A4,
        weight: Box::new(device::<Nvfp4E2M1>(ctx, &bytes)),
        weight_scale: Box::new(device::<Fp8E4M3>(ctx, &swizzled)),
        parameters: record::<Nvfp4Scales>(ctx, [8.0, 0.03125]),
        shape: [n, k],
    }
}

#[test]
fn mixed_linear_preparation_scales_tails_and_reused_plans_match_reference() {
    let ctx = Rc::new(CudaCtx::new(0).unwrap());
    let config = QwenConfig::randomized_dense_tiny_fixture();
    let mut engine = Engine::new(ctx.clone(), config.engine_config()).unwrap();
    let (n, k) = (136, 256); // Scale rows require padding; N is not a CTA multiple.
    let mut scratch = QuantizedScratch::new(ctx.clone(), n).unwrap();
    let workspace = DeviceBuffer::<U8>::with_capacity(ctx.clone(), 8 << 20).unwrap();
    let fp4 = fp4_block(&ctx, n, k);
    let weight_codes: Vec<_> = (0..n * k).map(|i| ((i + i / k * 3) % 16) as u8).collect();
    let fp8_bytes: Vec<u8> = (0..n * k)
        .map(|i| 0x20 + (i % 25) as u8 | if i % 3 == 0 { 128 } else { 0 })
        .collect();
    let fp8: Vec<Fp8Block> = [0.125, 0.25]
        .into_iter()
        .map(|input_scale| Fp8Block {
            weight: Box::new(device::<Fp8E4M3>(&ctx, &fp8_bytes)) as W<Fp8E4M3>,
            scales: record::<Fp8Scales>(&ctx, [input_scale, 0.5]),
            shape: [n, k],
        })
        .collect();
    ctx.synchronize().unwrap();
    for rows in [1, 2, 5, 16, 128, 512, 513, 1] {
        scratch
            .reserve_shapes(&mut engine, rows, k, vec![[rows, n, k]])
            .unwrap();
        let values: Vec<f32> = (0..rows * k)
            .map(|i| fp4_value(((i + i / k) % 16) as u8) * 0.125)
            .collect();
        let input = bf16(&ctx, &values);
        let output = DeviceBuffer::<BF16>::with_capacity(ctx.clone(), (rows * n) as usize).unwrap();
        unsafe {
            engine
                .operators()
                .linear(
                    input.matrix(rows, k).unwrap(),
                    fp4.view(n, k).unwrap(),
                    output.matrix(rows, n).unwrap(),
                    Some(&scratch),
                    Workspace::none(),
                )
                .unwrap();
        }
        let actual = download(&output, &ctx);
        if rows == 1 {
            let widened = DeviceBuffer::<F32>::with_capacity(ctx.clone(), n as usize).unwrap();
            unsafe {
                engine
                    .operators()
                    .qscu()
                    .logits_bf16_to_f32(output.matrix(1, n).unwrap(), widened.matrix(1, n).unwrap())
                    .unwrap();
            }
            let mut host = HostBuffer::<F32>::new(n as usize).unwrap();
            unsafe {
                widened.download(&mut host).unwrap();
            }
            ctx.synchronize().unwrap();
            let values: Vec<_> = host
                .as_ref()
                .chunks_exact(4)
                .map(|v| f32::from_ne_bytes(v.try_into().unwrap()))
                .collect();
            assert_eq!(values, actual);
        }
        for r in 0..rows {
            for c in 0..n {
                let sum: f32 = (0..k)
                    .map(|j| {
                        values[(r * k + j) as usize]
                            * fp4_value(weight_codes[(c * k + j) as usize])
                            * [0.5, 1., 2.][((c * k / 16 + j / 16) % 3) as usize]
                            * 0.25
                    })
                    .sum();
                assert!(
                    (actual[(r * n + c) as usize] - sum).abs() <= 0.01 + sum.abs() * 0.004,
                    "FP4 M={rows} r={r} c={c}: {} vs {sum}",
                    actual[(r * n + c) as usize]
                );
            }
        }
        for (index, weight) in fp8.iter().enumerate() {
            unsafe {
                engine
                    .operators()
                    .linear(
                        input.matrix(rows, k).unwrap(),
                        weight.view(n, k).unwrap(),
                        output.matrix(rows, n).unwrap(),
                        Some(&scratch),
                        workspace.workspace(workspace.len()).unwrap(),
                    )
                    .unwrap();
            }
            let actual = download(&output, &ctx);
            let scale = [0.125, 0.25][index];
            let quantized: Vec<_> = values
                .iter()
                .map(|x| round_fp8(*x / scale) * scale)
                .collect();
            for r in 0..rows {
                for c in 0..n {
                    let sum: f32 = (0..k)
                        .map(|j| {
                            quantized[(r * k + j) as usize]
                                * fp8_value(fp8_bytes[(c * k + j) as usize])
                                * 0.5
                        })
                        .sum();
                    assert!(
                        (actual[(r * n + c) as usize] - sum).abs() <= 0.01 + sum.abs() * 0.004,
                        "FP8 M={rows} scale={scale} r={r} c={c}: {} vs {sum}",
                        actual[(r * n + c) as usize]
                    );
                }
            }
        }
    }
    assert_eq!(scratch.plans.len(), 7);
}

fn quantized_fixture(ctx: &Rc<CudaCtx>, config: &QwenConfig) -> QwenWeights {
    let QwenWeights::DenseBf16(model) = QwenWeights::random_bf16(ctx.clone(), config, 91).unwrap()
    else {
        unreachable!()
    };
    let fp8 = |weight: W<BF16>, n, k| -> Fp8Block {
        let scale: f32 = 0.0009765625; // 2^-10: enough range for the randomized weights.
        let globals = device::<F32>(ctx, &scale.to_ne_bytes());
        let output = DeviceBuffer::<Fp8E4M3>::with_capacity(ctx.clone(), (n * k) as usize).unwrap();
        let mut qscb = crate::backend::Qscb::new(ctx).unwrap();
        unsafe {
            qscb.quantize_fp8(
                weight.matrix(n, k).unwrap(),
                output.matrix(n, k).unwrap(),
                globals.vector(1).unwrap(),
            )
            .unwrap();
        }
        ctx.synchronize().unwrap();
        Fp8Block {
            weight: Box::new(output),
            scales: record::<Fp8Scales>(ctx, [0.015625, scale]),
            shape: [n, k],
        }
    };
    let hidden = config.hidden_size();
    let intermediate = config.intermediate_size();
    let mlp = || DenseMlp {
        gate_proj: fp4_block(ctx, intermediate, hidden),
        up_proj: fp4_block(ctx, intermediate, hidden),
        down_proj: fp4_block(ctx, hidden, intermediate),
    };
    let layers = model
        .layers
        .into_iter()
        .map(|l| match l {
            QwenLayerWeights::AttentionMlp(l) => {
                QwenLayerWeights::AttentionMlp(QwenAttentionMlpWeights {
                    attn_norm: l.attn_norm,
                    q_norm: l.q_norm,
                    k_norm: l.k_norm,
                    q_proj: fp8(
                        l.q_proj,
                        crate::constants::attention::PACKED_Q_GATE_WIDTH,
                        hidden,
                    ),
                    k_proj: fp8(l.k_proj, config.kv_hidden_size().unwrap(), hidden),
                    v_proj: fp8(l.v_proj, config.kv_hidden_size().unwrap(), hidden),
                    o_proj: fp8(l.o_proj, hidden, config.q_hidden_size().unwrap()),
                    mlp_norm: l.mlp_norm,
                    mlp: mlp(),
                })
            }
            QwenLayerWeights::Gdn(l) => QwenLayerWeights::Gdn(QwenGdnWeights {
                norm: l.norm,
                in_proj: fp8(
                    l.in_proj,
                    crate::constants::gdn::PACKED_QKV_CHANNELS,
                    hidden,
                ),
                gate_proj: fp8(l.gate_proj, crate::constants::gdn::OUTPUT_WIDTH, hidden),
                out_proj: fp8(l.out_proj, hidden, crate::constants::gdn::OUTPUT_WIDTH),
                a_proj: l.a_proj,
                b_proj: l.b_proj,
                conv_weight: l.conv_weight,
                conv_bias: l.conv_bias,
                a_log: l.a_log,
                dt_bias: l.dt_bias,
                rms_weight: l.rms_weight,
                mlp_norm: l.mlp_norm,
                mlp: mlp(),
            }),
        })
        .collect();
    QwenWeights::DenseNvfp4(QwenModel {
        token_embedding: model.token_embedding,
        final_norm: model.final_norm,
        lm_head: fp4_block(ctx, config.vocab_size(), hidden),
        layers,
    })
}

#[test]
fn mixed_dense_runner_prefill_decode_rebuild_and_reset_reuse_preparation() {
    let _kernel_owner = crate::backend::qscute::TEST_LOCK.lock().unwrap();
    let ctx = Rc::new(CudaCtx::new(0).unwrap());
    let mut config = QwenConfig::randomized_dense_tiny_fixture();
    config.fixture_mut().num_layers = 4;
    config.fixture_mut().full_attention_only = false;
    let mut runner = ModelRunner::new(
        ctx.clone(),
        config,
        quantized_fixture(&ctx, &config),
        config.vocab_size() as usize,
    )
    .unwrap();
    assert_eq!(runner.lm_head_provider(), "flashinfer-cutlass-nvfp4");
    assert_eq!(
        runner.gdn_qkv_provider(),
        if crate::backend::qscute::Fp8Decode::supports(
            config.hidden_size(),
            crate::constants::gdn::PACKED_QKV_CHANNELS,
        ) {
            "cute-fp8-split2"
        } else {
            "cublaslt-fp8"
        }
    );
    let request = QwenRequest {
        request_id: 17,
        tokens: &[1, 2, 3, 4],
        max_new_tokens: 2,
    };
    let first = runner.run(request).unwrap();
    let logits = runner.last_logits_row_for_test().unwrap();
    assert!(logits.iter().all(|v| v.is_finite()));
    assert!(logits.windows(2).any(|v| v[0] != v[1]));
    let plan_count = runner.quantized_scratch.as_ref().unwrap().plans.len();
    let scales = match &runner.weights {
        QwenWeights::DenseNvfp4(m) => m.lm_head.weight_scale.as_raw(),
        _ => unreachable!(),
    };
    runner.reset().unwrap();
    let second = runner.run(request).unwrap();
    assert_eq!(first.generated_tokens, second.generated_tokens);
    assert_eq!(logits, runner.last_logits_row_for_test().unwrap());
    assert_eq!(
        plan_count,
        runner.quantized_scratch.as_ref().unwrap().plans.len()
    );
    runner.assert_late_rebuild_failure_preserves_prefix(17, &[1, 4, 3, 2]);
    let rewritten = QwenRequest {
        tokens: &[1, 3, 2, 4],
        ..request
    };
    let rebuilt = runner.run(rewritten).unwrap();
    runner.reset().unwrap();
    let fresh = runner.run(rewritten).unwrap();
    assert_eq!(rebuilt.generated_tokens, fresh.generated_tokens);
    assert!(
        matches!(&runner.weights, QwenWeights::DenseNvfp4(m) if m.lm_head.weight_scale.as_raw() == scales)
    );
}
