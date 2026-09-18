use super::{
    BatchRun, ModelRunner, QwenRequest,
    mlp::{Mlp, MlpView},
};
use crate::dtype::{BF16, F32, I32};
use crate::memory::{CudaCtx, DeviceBuffer, DeviceSpan, HostBuffer};
use crate::{
    QWEN36_MOE_ROUTER_SCALING_FACTOR, QWEN36_MOE_ROUTER_SCORE,
    backend::{
        DMat, DTensor3,
        qsfi::{MoeBf16Execute, MoeBf16ExecuteArgs, MoeBf16PlanConfig, Workspace},
    },
    constants::{
        attention::{
            HEAD_DIM, KV_WIDTH, NUM_KV_HEADS, NUM_Q_HEADS, PACKED_Q_GATE_WIDTH, Q_WIDTH, ROPE_THETA,
        },
        gdn::{CONV_WIDTH, NUM_VALUE_HEADS, OUTPUT_WIDTH, PACKED_QKV_CHANNELS, VALUE_HEAD_DIM},
        mlp::HAS_EXPERTS,
        model::{HIDDEN_SIZE, RMS_NORM_EPS},
    },
    engine::{AppendBatch, AttentionLayer, Commit, Engine, Status},
    ffi::cuda,
    model::{
        ActiveRunKind, MoeShape, QwenBlockKind, QwenConfig, QwenWeights, checked_usize_product,
        constant_bf16_values,
        scratch::{MlpScratch, MoeScratch, RunnerScratch, SharedExpertScratch},
        state::{GdnSlotMap, GdnState},
        weights::{
            DenseMlp, FusedExperts, MoeMlp, QwenAttentionMlpWeights, QwenGdnWeights,
            QwenLayerWeights, QwenModel, SharedExpert, W,
        },
    },
};
use std::rc::Rc;
use std::{mem, path::PathBuf};

const DEFAULT_VECTOR_ROOT: &str = "build/vectors/qwen36_semantics";
const FULL_ATTN_VECTOR_ROWS: u32 = 6;
const FULL_ATTN_BLOCK_VECTOR_ROWS: u32 = 6;
const FULL_ATTN_BLOCK_MOE_INTERMEDIATE: u32 = 8;
const GDN_DECODER_LAYER_VECTOR_ROWS: u32 = 6;
const GDN_DECODER_LAYER_MOE_INTERMEDIATE: u32 = 8;
const MODEL_LOGITS_PROMPT_LEN: usize = 4;
const MODEL_LOGITS_TOTAL_ROWS: usize = 5;
const MODEL_LOGITS_VOCAB: u32 = 16;
const MODEL_LOGITS_INTERMEDIATE: u32 = 8;
// Fixed MoE oracle dimensions, shared by the primitive and composed MoE vectors.
// A dense selected model has no experts; it must not zero these test dimensions.
const VECTOR_EXPERTS: u32 = 256;
const VECTOR_TOP_K: u32 = 8;
const MOE_VECTOR_ROWS: u32 = 5;
const MOE_VECTOR_HIDDEN: u32 = 8;
const MOE_VECTOR_INTERMEDIATE: u32 = 8;
const MOE_VECTOR_ELEMENTS: usize = (MOE_VECTOR_ROWS * MOE_VECTOR_HIDDEN) as usize;
const MOE_ROUTER_WEIGHT_ABS_TOL: f32 = 1.0e-5;
const MOE_ROUTER_WEIGHT_REL_TOL: f32 = 1.0e-5;
const BF16_MOE_ABS_TOL: f32 = 0.02;
// FlashInfer computes RoPE factors internally; the vectors store BF16-cache
// vLLM references, so RoPE uses the same f32-oracle tolerance style as the
// existing semantic vector harness.
const BF16_ROPE_ABS_TOL: f32 = 0.018;
const BF16_BLOCK_NORM_ABS_TOL: f32 = 0.03;
const BF16_BLOCK_PROJ_ABS_TOL: f32 = 0.04;
const BF16_BLOCK_ATTN_ABS_TOL: f32 = 0.04;
const BF16_BLOCK_ROPE_ABS_TOL: f32 = 0.033;
const BF16_BLOCK_MOE_ABS_TOL: f32 = 0.06;
const BF16_BLOCK_FINAL_NORM_ABS_TOL: f32 = 0.05;
const BF16_GDN_DECODER_PROJ_ABS_TOL: f32 = 0.08;
const BF16_GDN_DECODER_NORM_ABS_TOL: f32 = 0.06;
const BF16_GDN_DECODER_MOE_ABS_TOL: f32 = 0.08;
const BF16_GDN_DECODER_FINAL_NORM_ABS_TOL: f32 = 0.08;
const MODEL_LOGITS_ABS_TOL: f32 = 0.35;
const MODEL_LOGITS_REL_TOL: f32 = 0.03;

fn qwen36_hybrid_fixture_with_supported_attention(num_layers: u32) -> QwenConfig {
    let mut config = QwenConfig::randomized_qwen36_moe_gdn_one_block_fixture();
    config.fixture_mut().num_layers = num_layers;
    config
}

fn placeholder_weight() -> crate::model::weights::W<BF16> {
    Box::new(Rc::new(
        DeviceSpan::new(std::ptr::NonNull::<u16>::dangling().as_ptr().cast(), 0).unwrap(),
    ))
}

fn moe_scratch(scratch: &RunnerScratch) -> &MoeScratch {
    let MlpScratch::Moe(moe) = &scratch.mlp else {
        panic!("expected MoE scratch")
    };
    moe
}

fn moe_scratch_mut(scratch: &mut RunnerScratch) -> &mut MoeScratch {
    let MlpScratch::Moe(moe) = &mut scratch.mlp else {
        panic!("expected MoE scratch")
    };
    moe
}

fn shared_scratch(scratch: &RunnerScratch) -> &SharedExpertScratch {
    moe_scratch(scratch).shared.as_ref().unwrap()
}

// Fixture sources are initialized primitive arrays; transfers explicitly use bytes.
macro_rules! upload {
    ($ctx:expr, $buffer:expr, $values:expr $(,)?) => {{
        let values = $values;
        let mut host = HostBuffer::new(values.len()).unwrap();
        for (dst, src) in host
            .as_mut()
            .iter_mut()
            .zip(values.iter().flat_map(|value| value.to_ne_bytes()))
        {
            *dst = src;
        }
        unsafe {
            ($buffer).upload(&host).unwrap();
        }
        let result = ($ctx).synchronize();
        if result.is_err() {
            std::mem::forget(host);
        }
        result
    }};
}

fn bf16_buffer(ctx: Rc<CudaCtx>, values: &[u16]) -> Result<DeviceBuffer<BF16>, Status> {
    let mut host = HostBuffer::<BF16>::new(values.len())?;
    for (bytes, value) in host.as_mut().chunks_exact_mut(2).zip(values) {
        bytes.copy_from_slice(&value.to_ne_bytes());
    }
    host.upload(ctx)
}

type FusedBf16Mlp = MoeMlp<FusedExperts<W<BF16>>, W<BF16>>;

fn empty_shared_expert_weights() -> SharedExpert<W<BF16>> {
    SharedExpert {
        projections: DenseMlp {
            gate_proj: placeholder_weight(),
            up_proj: placeholder_weight(),
            down_proj: placeholder_weight(),
        },
        gate: placeholder_weight(),
    }
}

fn empty_qwen36_moe_mlp_weights() -> FusedBf16Mlp {
    MoeMlp {
        router_proj: placeholder_weight(),
        experts: FusedExperts {
            gate_up_proj: placeholder_weight(),
            down_proj: placeholder_weight(),
        },
        shared: Some(empty_shared_expert_weights()),
    }
}

unsafe extern "C" {
    fn cudaGetDeviceCount(count: *mut i32) -> i32;
}

fn cuda_device_available() -> bool {
    let mut device_count = 0;
    let err = unsafe { cudaGetDeviceCount(&mut device_count) };
    if err != cuda::CUDA_SUCCESS || device_count == 0 {
        eprintln!("SKIP: no CUDA device available");
        return false;
    }
    assert_eq!(unsafe { cuda::cudaSetDevice(0) }, cuda::CUDA_SUCCESS);
    true
}

fn filled_bf16_buffer(ctx: &Rc<CudaCtx>, len: usize, value: f32) -> DeviceBuffer<BF16> {
    bf16_buffer(ctx.clone(), &constant_bf16_values(len, value).unwrap()).unwrap()
}

fn download_bf16(buffer: &DeviceSpan<BF16>, ctx: &CudaCtx, len: usize) -> Vec<u16> {
    buffer.check_view_len(len).unwrap();
    let mut values = HostBuffer::<BF16>::new(len).unwrap();
    unsafe {
        ctx.download(buffer.as_raw(), values.as_mut()).unwrap();
    }
    if let Err(status) = ctx.synchronize() {
        std::mem::forget(values);
        panic!("download did not complete: {status:?}");
    }
    values
        .as_ref()
        .chunks_exact(2)
        .map(|bytes| u16::from_ne_bytes(bytes.try_into().unwrap()))
        .collect()
}

fn download_i32(buffer: &DeviceSpan<I32>, ctx: &CudaCtx, len: usize) -> Vec<i32> {
    buffer.check_view_len(len).unwrap();
    let mut values = HostBuffer::<I32>::new(len).unwrap();
    unsafe {
        ctx.download(buffer.as_raw(), values.as_mut()).unwrap();
    }
    if let Err(status) = ctx.synchronize() {
        std::mem::forget(values);
        panic!("download did not complete: {status:?}");
    }
    values
        .as_ref()
        .chunks_exact(4)
        .map(|bytes| i32::from_ne_bytes(bytes.try_into().unwrap()))
        .collect()
}

fn download_f32(buffer: &DeviceSpan<F32>, ctx: &CudaCtx, len: usize) -> Vec<f32> {
    buffer.check_view_len(len).unwrap();
    let mut values = HostBuffer::<F32>::new(len).unwrap();
    unsafe {
        ctx.download(buffer.as_raw(), values.as_mut()).unwrap();
    }
    if let Err(status) = ctx.synchronize() {
        std::mem::forget(values);
        panic!("download did not complete: {status:?}");
    }
    values
        .as_ref()
        .chunks_exact(4)
        .map(|bytes| f32::from_ne_bytes(bytes.try_into().unwrap()))
        .collect()
}

fn vector_root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join(DEFAULT_VECTOR_ROOT)
}

fn read_vector_bytes_at(group: &str, file: &str, elem_size: usize, elements: usize) -> Vec<u8> {
    let path = vector_root().join(group).join(file);
    let bytes = std::fs::read(&path).unwrap_or_else(|err| {
        panic!(
            "failed to read required Qwen3.6 vector {}: {err}. \
             Generate vectors with `just build_tools/generate-vectors`.",
            path.display()
        );
    });
    let expected_bytes = elements.checked_mul(elem_size).unwrap();
    assert_eq!(bytes.len(), expected_bytes, "{file} byte length changed");
    bytes
}

fn read_vector_bytes(file: &str, elem_size: usize, elements: usize) -> Vec<u8> {
    read_vector_bytes_at("full_attention_primitives", file, elem_size, elements)
}

fn read_moe_vector_bytes(file: &str, elem_size: usize, elements: usize) -> Vec<u8> {
    read_vector_bytes_at("moe_shared_expert", file, elem_size, elements)
}

fn read_block_vector_bytes(file: &str, elem_size: usize, elements: usize) -> Vec<u8> {
    read_vector_bytes_at("full_attention_block", file, elem_size, elements)
}

fn read_gdn_decoder_vector_bytes(file: &str, elem_size: usize, elements: usize) -> Vec<u8> {
    read_vector_bytes_at("gdn_decoder_layer", file, elem_size, elements)
}

fn read_model_logits_vector_bytes(file: &str, elem_size: usize, elements: usize) -> Vec<u8> {
    read_vector_bytes_at("model_logits", file, elem_size, elements)
}

fn read_bf16_vector(file: &str, elements: usize) -> Vec<u16> {
    read_vector_bytes(file, mem::size_of::<u16>(), elements)
        .chunks_exact(mem::size_of::<u16>())
        .map(|chunk| u16::from_le_bytes([chunk[0], chunk[1]]))
        .collect()
}

fn read_i32_vector(file: &str, elements: usize) -> Vec<i32> {
    read_vector_bytes(file, mem::size_of::<i32>(), elements)
        .chunks_exact(mem::size_of::<i32>())
        .map(|chunk| i32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
        .collect()
}

fn read_f32_vector(file: &str, elements: usize) -> Vec<f32> {
    read_vector_bytes(file, mem::size_of::<f32>(), elements)
        .chunks_exact(mem::size_of::<f32>())
        .map(|chunk| f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
        .collect()
}

fn read_block_bf16_vector(file: &str, elements: usize) -> Vec<u16> {
    read_block_vector_bytes(file, mem::size_of::<u16>(), elements)
        .chunks_exact(mem::size_of::<u16>())
        .map(|chunk| u16::from_le_bytes([chunk[0], chunk[1]]))
        .collect()
}

fn read_block_i32_vector(file: &str, elements: usize) -> Vec<i32> {
    read_block_vector_bytes(file, mem::size_of::<i32>(), elements)
        .chunks_exact(mem::size_of::<i32>())
        .map(|chunk| i32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
        .collect()
}

fn read_block_f32_vector(file: &str, elements: usize) -> Vec<f32> {
    read_block_vector_bytes(file, mem::size_of::<f32>(), elements)
        .chunks_exact(mem::size_of::<f32>())
        .map(|chunk| f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
        .collect()
}

fn read_gdn_decoder_bf16_vector(file: &str, elements: usize) -> Vec<u16> {
    read_gdn_decoder_vector_bytes(file, mem::size_of::<u16>(), elements)
        .chunks_exact(mem::size_of::<u16>())
        .map(|chunk| u16::from_le_bytes([chunk[0], chunk[1]]))
        .collect()
}

fn read_gdn_decoder_i32_vector(file: &str, elements: usize) -> Vec<i32> {
    read_gdn_decoder_vector_bytes(file, mem::size_of::<i32>(), elements)
        .chunks_exact(mem::size_of::<i32>())
        .map(|chunk| i32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
        .collect()
}

fn read_gdn_decoder_f32_vector(file: &str, elements: usize) -> Vec<f32> {
    read_gdn_decoder_vector_bytes(file, mem::size_of::<f32>(), elements)
        .chunks_exact(mem::size_of::<f32>())
        .map(|chunk| f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
        .collect()
}

fn read_model_logits_bf16_vector(file: &str, elements: usize) -> Vec<u16> {
    read_model_logits_vector_bytes(file, mem::size_of::<u16>(), elements)
        .chunks_exact(mem::size_of::<u16>())
        .map(|chunk| u16::from_le_bytes([chunk[0], chunk[1]]))
        .collect()
}

fn read_model_logits_i32_vector(file: &str, elements: usize) -> Vec<i32> {
    read_model_logits_vector_bytes(file, mem::size_of::<i32>(), elements)
        .chunks_exact(mem::size_of::<i32>())
        .map(|chunk| i32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
        .collect()
}

fn read_model_logits_f32_vector(file: &str, elements: usize) -> Vec<f32> {
    read_model_logits_vector_bytes(file, mem::size_of::<f32>(), elements)
        .chunks_exact(mem::size_of::<f32>())
        .map(|chunk| f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
        .collect()
}

fn read_moe_bf16_vector(file: &str, elements: usize) -> Vec<u16> {
    read_moe_vector_bytes(file, mem::size_of::<u16>(), elements)
        .chunks_exact(mem::size_of::<u16>())
        .map(|chunk| u16::from_le_bytes([chunk[0], chunk[1]]))
        .collect()
}

fn read_moe_i32_vector(file: &str, elements: usize) -> Vec<i32> {
    read_moe_vector_bytes(file, mem::size_of::<i32>(), elements)
        .chunks_exact(mem::size_of::<i32>())
        .map(|chunk| i32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
        .collect()
}

fn read_moe_f32_vector(file: &str, elements: usize) -> Vec<f32> {
    read_moe_vector_bytes(file, mem::size_of::<f32>(), elements)
        .chunks_exact(mem::size_of::<f32>())
        .map(|chunk| f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
        .collect()
}

fn bf16_bits_to_f32(bits: u16) -> f32 {
    f32::from_bits(u32::from(bits) << 16)
}

fn assert_bf16_exact(what: &str, got: &[u16], expected: &[u16]) {
    assert_eq!(
        got.len(),
        expected.len(),
        "{what}: BF16 element count changed"
    );
    if got == expected {
        return;
    }

    let first_mismatch = got
        .iter()
        .zip(expected)
        .position(|(got, expected)| got != expected)
        .unwrap();
    let mismatch_count = got
        .iter()
        .zip(expected)
        .filter(|(got, expected)| got != expected)
        .count();
    panic!(
        "{what}: BF16 mismatch_count={mismatch_count}; first mismatch at {first_mismatch}: got=0x{:04x}, expected=0x{:04x}",
        got[first_mismatch], expected[first_mismatch]
    );
}

fn assert_f32_close(what: &str, got: &[f32], expected: &[f32], abs_tol: f32, rel_tol: f32) {
    assert_eq!(got.len(), expected.len(), "{what}: element count changed");

    let mut max_abs_delta = 0.0f32;
    let mut max_abs_delta_idx = 0usize;
    for (idx, (&got, &expected)) in got.iter().zip(expected).enumerate() {
        let delta = (got - expected).abs();
        if delta > max_abs_delta {
            max_abs_delta = delta;
            max_abs_delta_idx = idx;
        }
    }

    for (idx, (&got, &expected)) in got.iter().zip(expected).enumerate() {
        let delta = (got - expected).abs();
        let tolerance = abs_tol.max(rel_tol * expected.abs());
        assert!(
            delta <= tolerance,
            "{what}[{idx}]: got {got:.8}, expected {expected:.8}, \
             delta={delta:.8}, tolerance={tolerance:.8}; \
             max_abs_delta={max_abs_delta:.8} at {max_abs_delta_idx}"
        );
    }
}

// Projection checks allow normal BF16 rounding differences. Check routing
// arithmetic separately against the actual validated projection output, so
// its tighter tolerance does not accidentally constrain the preceding GEMM.
fn assert_moe_router_matches_cpu(runner: &ModelRunner, rows: u32) {
    let moe = runner.config.moe_config().unwrap();
    let experts = moe.num_experts as usize;
    let topk = moe.num_experts_per_tok as usize;
    let logits = download_bf16(
        &moe_scratch(&runner.scratch).router_logits,
        &runner.ctx,
        rows as usize * experts,
    );
    let ids = download_i32(
        &moe_scratch(&runner.scratch).topk_ids,
        &runner.ctx,
        rows as usize * topk,
    );
    let weights = download_f32(
        &moe_scratch(&runner.scratch).topk_weights,
        &runner.ctx,
        ids.len(),
    );
    let mut expected_weights = Vec::with_capacity(weights.len());
    for (row_idx, row) in logits.chunks_exact(experts).enumerate() {
        let values: Vec<f64> = row
            .iter()
            .map(|&x| f64::from(bf16_bits_to_f32(x)))
            .collect();
        let mut order: Vec<usize> = (0..experts).collect();
        order.sort_by(|&a, &b| values[b].total_cmp(&values[a]).then(a.cmp(&b)));
        let max = values[order[0]];
        let denom: f64 = order[..topk].iter().map(|&i| (values[i] - max).exp()).sum();
        for rank in 0..topk {
            assert_eq!(ids[row_idx * topk + rank], order[rank] as i32);
            expected_weights.push(((values[order[rank]] - max).exp() / denom) as f32);
        }
    }
    assert_f32_close(
        "router weights from validated BF16 logits",
        &weights,
        &expected_weights,
        MOE_ROUTER_WEIGHT_ABS_TOL,
        MOE_ROUTER_WEIGHT_REL_TOL,
    );
}

fn assert_bf16_close_to_f32_oracle(
    what: &str,
    got_bf16: &[u16],
    expected_bf16: &[u16],
    expected_f32: &[f32],
    abs_tol: f32,
) {
    assert_eq!(
        got_bf16.len(),
        expected_bf16.len(),
        "{what}: BF16 element count changed"
    );
    assert_eq!(
        got_bf16.len(),
        expected_f32.len(),
        "{what}: f32 element count changed"
    );

    let bf16_mismatch_count = got_bf16
        .iter()
        .zip(expected_bf16)
        .filter(|(got, expected)| got != expected)
        .count();
    let mut max_abs_delta = 0.0f32;
    let mut max_abs_delta_idx = 0usize;
    for (idx, (&got, &expected)) in got_bf16.iter().zip(expected_f32).enumerate() {
        let delta = (bf16_bits_to_f32(got) - expected).abs();
        if delta > max_abs_delta {
            max_abs_delta = delta;
            max_abs_delta_idx = idx;
        }
    }

    for (idx, ((&got, &expected_bits), &expected)) in got_bf16
        .iter()
        .zip(expected_bf16)
        .zip(expected_f32)
        .enumerate()
    {
        let got_f32 = bf16_bits_to_f32(got);
        let delta = (got_f32 - expected).abs();
        assert!(
            got == expected_bits || delta <= abs_tol,
            "{what}[{idx}]: got {got_f32:.8} (0x{got:04x}), expected {expected:.8} \
             +/- {abs_tol:.8} or expected_bf16=0x{expected_bits:04x}; \
             bf16_mismatch_count={bf16_mismatch_count}, \
             max_abs_delta={max_abs_delta:.8} at {max_abs_delta_idx}"
        );
    }
}

fn full_attention_vector_config() -> QwenConfig {
    let mut config = QwenConfig::randomized_dense_tiny_fixture();
    config.fixture_mut().rms_norm_eps = RMS_NORM_EPS;
    config.fixture_mut().rope_theta = ROPE_THETA;
    config.fixture_mut().num_layers = 1;
    config
}

fn placeholder_weights() -> super::QwenWeights {
    super::QwenWeights::DenseBf16(QwenModel {
        token_embedding: placeholder_weight(),
        final_norm: placeholder_weight(),
        lm_head: placeholder_weight(),
        layers: Vec::new(),
    })
}

fn moe_vector_config() -> QwenConfig {
    let mut config = QwenConfig::randomized_dense_tiny_fixture();
    config.fixture_mut().num_layers = 1;
    config.max_seq_len = MOE_VECTOR_ROWS;
    config.max_pages = 2;
    config.page_size = 4;
    config.fixture_mut().hidden_size = MOE_VECTOR_HIDDEN;
    config.fixture_mut().intermediate_size = MOE_VECTOR_INTERMEDIATE;
    config.fixture_mut().moe = Some(MoeShape {
        num_experts: VECTOR_EXPERTS,
        num_experts_per_tok: VECTOR_TOP_K,
        moe_intermediate_size: MOE_VECTOR_INTERMEDIATE,
        shared_expert_intermediate_size: MOE_VECTOR_INTERMEDIATE,
    });
    config.fixture_mut().vocab_size = 16;
    config
}

fn moe_vector_runner() -> ModelRunner {
    let ctx = Rc::new(CudaCtx::default().unwrap());
    let config = moe_vector_config();
    let mut engine = Engine::new(ctx.clone(), config.engine_config()).unwrap();
    let moe = config.moe_config().unwrap();
    let (moe_plan, workspace_bytes) = {
        let mut ops = engine.operators();
        let plan = unsafe {
            ops.qsfi()
                .create_moe_bf16_plan(MoeBf16PlanConfig {
                    kernel: crate::backend::qsfi::MoeBf16Kernel::COMPILED,
                    max_num_tokens: config.max_seq_len,
                    hidden_size: config.hidden_size(),
                    intermediate_size: moe.moe_intermediate_size,
                    num_experts: moe.num_experts,
                    top_k: moe.num_experts_per_tok,
                })
                .unwrap()
        };
        let workspace_bytes =
            unsafe { ops.qsfi().moe_workspace_size(&plan, config.max_seq_len) }.unwrap();
        (plan, workspace_bytes)
    };
    let scratch = RunnerScratch::new(ctx.clone(), &config, workspace_bytes).unwrap();
    let qscb_workspace =
        DeviceBuffer::with_capacity(ctx.clone(), (config.qscb_workspace_bytes).max(1)).unwrap();

    ModelRunner {
        ctx: ctx.clone(),
        sampler: None,
        lm_head: None,
        gdn_qkv: None,
        config,
        tokenizer_token_count: config.vocab_size(),
        weights: placeholder_weights(),
        quantized_scratch: None,
        engine,
        moe_plan: Some(moe_plan),
        gdn_state: None,
        scratch,
        qscb_workspace,
        live_request_id: None,
        live_tokens: Vec::new(),
        last_next_tokens: Vec::new(),
        last_logits_rows: 0,
        last_logits_vocab_size: 0,
    }
}

fn upload_moe_vector_inputs(runner: &mut ModelRunner) {
    runner.scratch.reserve(MOE_VECTOR_ROWS).unwrap();
    upload!(
        &runner.ctx,
        &mut runner.scratch.attn_proj,
        &read_moe_bf16_vector("hidden.bf16", MOE_VECTOR_ELEMENTS),
    )
    .unwrap();
    upload!(
        &runner.ctx,
        &mut moe_scratch_mut(&mut runner.scratch).router_logits,
        &read_moe_bf16_vector(
            "router_logits.bf16",
            (MOE_VECTOR_ROWS * VECTOR_EXPERTS) as usize,
        ),
    )
    .unwrap();
}

fn run_moe_vector_router(runner: &mut ModelRunner, renormalize: bool) {
    let moe = runner.config.moe_config().unwrap();
    let router_logits = DMat::contiguous(
        **moe_scratch(&runner.scratch).router_logits,
        MOE_VECTOR_ROWS,
        moe.num_experts,
    )
    .unwrap();
    let topk_ids = DMat::contiguous(
        **moe_scratch(&runner.scratch).topk_ids,
        MOE_VECTOR_ROWS,
        moe.num_experts_per_tok,
    )
    .unwrap();
    let topk_weights = DMat::contiguous(
        **moe_scratch(&runner.scratch).topk_weights,
        MOE_VECTOR_ROWS,
        moe.num_experts_per_tok,
    )
    .unwrap();
    let mut ops = runner.engine.operators();
    unsafe {
        ops.qscu().router_topk(
            router_logits,
            topk_ids,
            topk_weights,
            QWEN36_MOE_ROUTER_SCORE,
            renormalize,
            QWEN36_MOE_ROUTER_SCALING_FACTOR,
        )
    }
    .unwrap();
}

fn execute_moe_vector_routed_output(runner: &mut ModelRunner) {
    let ctx = runner.ctx.clone();
    let moe = runner.config.moe_config().unwrap();
    let gate_up_weight = bf16_buffer(
        ctx.clone(),
        &read_moe_bf16_vector(
            "gate_up_weight.bf16",
            (VECTOR_EXPERTS * 2 * MOE_VECTOR_INTERMEDIATE * MOE_VECTOR_HIDDEN) as usize,
        ),
    )
    .unwrap();
    let down_weight = bf16_buffer(
        ctx.clone(),
        &read_moe_bf16_vector(
            "down_weight.bf16",
            (VECTOR_EXPERTS * MOE_VECTOR_HIDDEN * MOE_VECTOR_INTERMEDIATE) as usize,
        ),
    )
    .unwrap();
    let execute = MoeBf16Execute::new(MoeBf16ExecuteArgs {
        hidden: DMat::contiguous(
            **runner.scratch.attn_proj,
            MOE_VECTOR_ROWS,
            MOE_VECTOR_HIDDEN,
        )
        .unwrap(),
        topk_ids: DMat::contiguous(
            **moe_scratch(&runner.scratch).topk_ids,
            MOE_VECTOR_ROWS,
            moe.num_experts_per_tok,
        )
        .unwrap(),
        topk_weights: DMat::contiguous(
            **moe_scratch(&runner.scratch).topk_weights,
            MOE_VECTOR_ROWS,
            moe.num_experts_per_tok,
        )
        .unwrap(),
        gate_up_weight: DTensor3::contiguous(
            **gate_up_weight,
            moe.num_experts,
            2 * MOE_VECTOR_INTERMEDIATE,
            MOE_VECTOR_HIDDEN,
        )
        .unwrap(),
        down_weight: DTensor3::contiguous(
            **down_weight,
            moe.num_experts,
            MOE_VECTOR_HIDDEN,
            MOE_VECTOR_INTERMEDIATE,
        )
        .unwrap(),
        out: DMat::contiguous(**runner.scratch.mlp_out, MOE_VECTOR_ROWS, MOE_VECTOR_HIDDEN)
            .unwrap(),
        workspace: Workspace::new(
            moe_scratch(&runner.scratch).workspace.erase(),
            moe_scratch(&runner.scratch).workspace.len,
        )
        .unwrap(),
    })
    .unwrap();
    let mut ops = runner.engine.operators();
    unsafe {
        ops.qsfi()
            .moe_execute_bf16(runner.moe_plan.as_ref().unwrap(), &execute)
    }
    .unwrap();
    runner.ctx.synchronize().unwrap();
}

fn execute_moe_vector_shared_gate_add(runner: &mut ModelRunner) {
    let ctx = runner.ctx.clone();
    let shared_gate_up_weight = bf16_buffer(
        ctx.clone(),
        &read_moe_bf16_vector(
            "shared_gate_up_weight.bf16",
            (2 * MOE_VECTOR_INTERMEDIATE * MOE_VECTOR_HIDDEN) as usize,
        ),
    )
    .unwrap();
    let shared_down_weight = bf16_buffer(
        ctx.clone(),
        &read_moe_bf16_vector(
            "shared_down_weight.bf16",
            (MOE_VECTOR_HIDDEN * MOE_VECTOR_INTERMEDIATE) as usize,
        ),
    )
    .unwrap();
    let shared_up_weight = bf16_buffer(
        ctx.clone(),
        &read_moe_bf16_vector(
            "shared_gate_up_weight.bf16",
            (2 * MOE_VECTOR_INTERMEDIATE * MOE_VECTOR_HIDDEN) as usize,
        )[(MOE_VECTOR_INTERMEDIATE * MOE_VECTOR_HIDDEN) as usize..],
    )
    .unwrap();
    let shared_up_proj = shared_up_weight
        .matrix(MOE_VECTOR_INTERMEDIATE, MOE_VECTOR_HIDDEN)
        .unwrap();

    unsafe {
        runner.engine.operators().qscb().linear(
            runner
                .scratch
                .attn_proj
                .matrix(MOE_VECTOR_ROWS, MOE_VECTOR_HIDDEN)
                .unwrap(),
            shared_gate_up_weight
                .matrix(MOE_VECTOR_INTERMEDIATE, MOE_VECTOR_HIDDEN)
                .unwrap(),
            shared_scratch(&runner.scratch)
                .gate
                .matrix(MOE_VECTOR_ROWS, MOE_VECTOR_INTERMEDIATE)
                .unwrap(),
            runner
                .qscb_workspace
                .workspace(runner.config.qscb_workspace_bytes)
                .unwrap(),
        )
    }
    .unwrap();
    unsafe {
        runner.engine.operators().qscb().linear(
            runner
                .scratch
                .attn_proj
                .matrix(MOE_VECTOR_ROWS, MOE_VECTOR_HIDDEN)
                .unwrap(),
            shared_up_proj,
            shared_scratch(&runner.scratch)
                .up
                .matrix(MOE_VECTOR_ROWS, MOE_VECTOR_INTERMEDIATE)
                .unwrap(),
            runner
                .qscb_workspace
                .workspace(runner.config.qscb_workspace_bytes)
                .unwrap(),
        )
    }
    .unwrap();
    unsafe {
        runner.engine.operators().qscu().silu_and_mul_bf16(
            shared_scratch(&runner.scratch)
                .gate
                .matrix(MOE_VECTOR_ROWS, MOE_VECTOR_INTERMEDIATE)
                .unwrap(),
            shared_scratch(&runner.scratch)
                .up
                .matrix(MOE_VECTOR_ROWS, MOE_VECTOR_INTERMEDIATE)
                .unwrap(),
            shared_scratch(&runner.scratch)
                .activated
                .matrix(MOE_VECTOR_ROWS, MOE_VECTOR_INTERMEDIATE)
                .unwrap(),
        )
    }
    .unwrap();
    unsafe {
        runner.engine.operators().qscb().linear(
            shared_scratch(&runner.scratch)
                .activated
                .matrix(MOE_VECTOR_ROWS, MOE_VECTOR_INTERMEDIATE)
                .unwrap(),
            shared_down_weight
                .matrix(MOE_VECTOR_HIDDEN, MOE_VECTOR_INTERMEDIATE)
                .unwrap(),
            shared_scratch(&runner.scratch)
                .out
                .matrix(MOE_VECTOR_ROWS, MOE_VECTOR_HIDDEN)
                .unwrap(),
            runner
                .qscb_workspace
                .workspace(runner.config.qscb_workspace_bytes)
                .unwrap(),
        )
    }
    .unwrap();
    upload!(
        &runner.ctx,
        &mut moe_scratch_mut(&mut runner.scratch)
            .shared
            .as_mut()
            .unwrap()
            .gate_logits,
        &read_moe_f32_vector("shared_gate_logits.f32", MOE_VECTOR_ROWS as usize),
    )
    .unwrap();

    unsafe {
        runner
            .engine
            .operators()
            .qscu()
            .qwen36_shared_expert_gate_add_bf16(
                shared_scratch(&runner.scratch)
                    .gate_logits
                    .matrix(MOE_VECTOR_ROWS, 1)
                    .unwrap(),
                shared_scratch(&runner.scratch)
                    .out
                    .matrix(MOE_VECTOR_ROWS, MOE_VECTOR_HIDDEN)
                    .unwrap(),
                runner
                    .scratch
                    .mlp_out
                    .matrix(MOE_VECTOR_ROWS, MOE_VECTOR_HIDDEN)
                    .unwrap(),
            )
    }
    .unwrap();
    runner.ctx.synchronize().unwrap();
}

fn full_attention_vector_runner() -> ModelRunner {
    let ctx = Rc::new(CudaCtx::default().unwrap());
    let mut config = full_attention_vector_config();
    config.qscb_workspace_bytes = 0; // This fixture runs only attention prep kernels.
    let engine = Engine::new(ctx.clone(), config.engine_config()).unwrap();
    ModelRunner {
        ctx: ctx.clone(),
        sampler: None,
        gdn_qkv: None,
        lm_head: None,
        config,
        tokenizer_token_count: config.vocab_size(),
        weights: placeholder_weights(),
        quantized_scratch: None,
        engine,
        moe_plan: None,
        gdn_state: None,
        scratch: RunnerScratch::new(ctx.clone(), &config, 0).unwrap(),
        qscb_workspace: DeviceBuffer::with_capacity(ctx.clone(), 1).unwrap(),
        live_request_id: None,
        live_tokens: Vec::new(),
        last_next_tokens: Vec::new(),
        last_logits_rows: 0,
        last_logits_vocab_size: 0,
    }
}

fn full_attention_block_vector_runner() -> ModelRunner {
    let ctx = Rc::new(CudaCtx::default().unwrap());
    let mut config = full_attention_vector_config();
    config.qscb_workspace_bytes = 0; // This fixture runs only attention prep kernels.
    let engine = Engine::new(ctx.clone(), config.engine_config()).unwrap();
    let qscb_workspace =
        DeviceBuffer::with_capacity(ctx.clone(), (config.qscb_workspace_bytes).max(1)).unwrap();
    ModelRunner {
        ctx: ctx.clone(),
        sampler: None,
        gdn_qkv: None,
        lm_head: None,
        config,
        tokenizer_token_count: config.vocab_size(),
        weights: placeholder_weights(),
        quantized_scratch: None,
        engine,
        moe_plan: None,
        gdn_state: None,
        scratch: RunnerScratch::new(ctx.clone(), &config, 0).unwrap(),
        qscb_workspace,
        live_request_id: None,
        live_tokens: Vec::new(),
        last_next_tokens: Vec::new(),
        last_logits_rows: 0,
        last_logits_vocab_size: 0,
    }
}

fn full_attention_block_moe_vector_config() -> QwenConfig {
    let mut config = full_attention_vector_config();
    config.max_seq_len = FULL_ATTN_BLOCK_VECTOR_ROWS;
    config.max_pages = 2;
    config.page_size = 4;
    config.fixture_mut().intermediate_size = FULL_ATTN_BLOCK_MOE_INTERMEDIATE;
    config.fixture_mut().moe = Some(MoeShape {
        num_experts: VECTOR_EXPERTS,
        num_experts_per_tok: VECTOR_TOP_K,
        moe_intermediate_size: FULL_ATTN_BLOCK_MOE_INTERMEDIATE,
        shared_expert_intermediate_size: FULL_ATTN_BLOCK_MOE_INTERMEDIATE,
    });
    config
}

fn full_attention_block_moe_vector_runner() -> ModelRunner {
    let ctx = Rc::new(CudaCtx::default().unwrap());
    let config = full_attention_block_moe_vector_config();
    let mut engine = Engine::new(ctx.clone(), config.engine_config()).unwrap();
    let moe = config.moe_config().unwrap();
    let (moe_plan, workspace_bytes) = {
        let mut ops = engine.operators();
        let plan = unsafe {
            ops.qsfi()
                .create_moe_bf16_plan(MoeBf16PlanConfig {
                    kernel: crate::backend::qsfi::MoeBf16Kernel::COMPILED,
                    max_num_tokens: config.max_seq_len,
                    hidden_size: config.hidden_size(),
                    intermediate_size: moe.moe_intermediate_size,
                    num_experts: moe.num_experts,
                    top_k: moe.num_experts_per_tok,
                })
                .unwrap()
        };
        let workspace_bytes =
            unsafe { ops.qsfi().moe_workspace_size(&plan, config.max_seq_len) }.unwrap();
        (plan, workspace_bytes)
    };
    let scratch = RunnerScratch::new(ctx.clone(), &config, workspace_bytes).unwrap();
    let qscb_workspace =
        DeviceBuffer::with_capacity(ctx.clone(), (config.qscb_workspace_bytes).max(1)).unwrap();
    ModelRunner {
        ctx: ctx.clone(),
        sampler: None,
        gdn_qkv: None,
        lm_head: None,
        config,
        tokenizer_token_count: config.vocab_size(),
        weights: placeholder_weights(),
        quantized_scratch: None,
        engine,
        moe_plan: Some(moe_plan),
        gdn_state: None,
        scratch,
        qscb_workspace,
        live_request_id: None,
        live_tokens: Vec::new(),
        last_next_tokens: Vec::new(),
        last_logits_rows: 0,
        last_logits_vocab_size: 0,
    }
}

struct FullAttentionBlockMoeLayer {
    layer: QwenLayerWeights<FusedBf16Mlp>,
    next_norm: DeviceBuffer<BF16>,
}

impl FullAttentionBlockMoeLayer {
    fn weights(&self) -> &QwenAttentionMlpWeights<FusedBf16Mlp> {
        match &self.layer {
            QwenLayerWeights::AttentionMlp(layer) => layer,
            QwenLayerWeights::Gdn(_) => unreachable!("full-attention block fixture is not GDN"),
        }
    }
}

fn full_attention_block_moe_layer(ctx: &Rc<CudaCtx>) -> FullAttentionBlockMoeLayer {
    let hidden = HIDDEN_SIZE;
    let q_hidden = Q_WIDTH;
    let kv_hidden = KV_WIDTH;
    let intermediate = FULL_ATTN_BLOCK_MOE_INTERMEDIATE;
    let layer: QwenLayerWeights<FusedBf16Mlp> =
        QwenLayerWeights::AttentionMlp(QwenAttentionMlpWeights {
            attn_norm: Box::new(
                bf16_buffer(
                    ctx.clone(),
                    &read_block_bf16_vector("block_attn_norm_raw_weight.bf16", hidden as usize),
                )
                .unwrap(),
            ),
            q_norm: Box::new(
                bf16_buffer(
                    ctx.clone(),
                    &read_block_bf16_vector("block_q_norm_raw_weight.bf16", HEAD_DIM as usize),
                )
                .unwrap(),
            ),
            k_norm: Box::new(
                bf16_buffer(
                    ctx.clone(),
                    &read_block_bf16_vector("block_k_norm_raw_weight.bf16", HEAD_DIM as usize),
                )
                .unwrap(),
            ),
            q_proj: Box::new(
                bf16_buffer(
                    ctx.clone(),
                    &read_block_bf16_vector(
                        "block_q_proj_weight.bf16",
                        checked_usize_product(&[PACKED_Q_GATE_WIDTH, hidden]).unwrap(),
                    ),
                )
                .unwrap(),
            ),
            k_proj: Box::new(
                bf16_buffer(
                    ctx.clone(),
                    &read_block_bf16_vector(
                        "block_k_proj_weight.bf16",
                        checked_usize_product(&[kv_hidden, hidden]).unwrap(),
                    ),
                )
                .unwrap(),
            ),
            v_proj: Box::new(
                bf16_buffer(
                    ctx.clone(),
                    &read_block_bf16_vector(
                        "block_v_proj_weight.bf16",
                        checked_usize_product(&[kv_hidden, hidden]).unwrap(),
                    ),
                )
                .unwrap(),
            ),
            o_proj: Box::new(
                bf16_buffer(
                    ctx.clone(),
                    &read_block_bf16_vector(
                        "block_o_proj_weight.bf16",
                        checked_usize_product(&[hidden, q_hidden]).unwrap(),
                    ),
                )
                .unwrap(),
            ),
            mlp_norm: Box::new(
                bf16_buffer(
                    ctx.clone(),
                    &read_block_bf16_vector(
                        "block_post_attn_norm_raw_weight.bf16",
                        hidden as usize,
                    ),
                )
                .unwrap(),
            ),
            mlp: MoeMlp {
                router_proj: Box::new(
                    bf16_buffer(
                        ctx.clone(),
                        &read_block_bf16_vector(
                            "block_moe_router_proj_weight.bf16",
                            checked_usize_product(&[VECTOR_EXPERTS, hidden]).unwrap(),
                        ),
                    )
                    .unwrap(),
                ),
                experts: FusedExperts {
                    gate_up_proj: Box::new(
                        bf16_buffer(
                            ctx.clone(),
                            &read_block_bf16_vector(
                                "block_moe_gate_up_proj_weight.bf16",
                                checked_usize_product(&[VECTOR_EXPERTS, 2, intermediate, hidden])
                                    .unwrap(),
                            ),
                        )
                        .unwrap(),
                    ),
                    down_proj: Box::new(
                        bf16_buffer(
                            ctx.clone(),
                            &read_block_bf16_vector(
                                "block_moe_down_proj_weight.bf16",
                                checked_usize_product(&[VECTOR_EXPERTS, hidden, intermediate])
                                    .unwrap(),
                            ),
                        )
                        .unwrap(),
                    ),
                },
                shared: Some(SharedExpert {
                    projections: DenseMlp {
                        gate_proj: Box::new(
                            bf16_buffer(
                                ctx.clone(),
                                &read_block_bf16_vector(
                                    "block_moe_shared_gate_proj_weight.bf16",
                                    checked_usize_product(&[intermediate, hidden]).unwrap(),
                                ),
                            )
                            .unwrap(),
                        ),
                        up_proj: Box::new(
                            bf16_buffer(
                                ctx.clone(),
                                &read_block_bf16_vector(
                                    "block_moe_shared_up_proj_weight.bf16",
                                    checked_usize_product(&[intermediate, hidden]).unwrap(),
                                ),
                            )
                            .unwrap(),
                        ),
                        down_proj: Box::new(
                            bf16_buffer(
                                ctx.clone(),
                                &read_block_bf16_vector(
                                    "block_moe_shared_down_proj_weight.bf16",
                                    checked_usize_product(&[hidden, intermediate]).unwrap(),
                                ),
                            )
                            .unwrap(),
                        ),
                    },
                    gate: Box::new(
                        bf16_buffer(
                            ctx.clone(),
                            &read_block_bf16_vector(
                                "block_moe_shared_expert_gate_weight.bf16",
                                hidden as usize,
                            ),
                        )
                        .unwrap(),
                    ),
                }),
            },
        });
    let next_norm = bf16_buffer(
        ctx.clone(),
        &read_block_bf16_vector("block_next_layer_norm_raw_weight.bf16", hidden as usize),
    )
    .unwrap();
    FullAttentionBlockMoeLayer { layer, next_norm }
}

fn gdn_decoder_layer_vector_config() -> QwenConfig {
    let mut config = QwenConfig::randomized_qwen36_moe_gdn_one_block_fixture();
    config.fixture_mut().rms_norm_eps = RMS_NORM_EPS;
    config.fixture_mut().rope_theta = ROPE_THETA;
    config.max_seq_len = GDN_DECODER_LAYER_VECTOR_ROWS;
    config.max_pages = 2;
    config.page_size = 4;
    config.fixture_mut().intermediate_size = GDN_DECODER_LAYER_MOE_INTERMEDIATE;
    config.fixture_mut().moe = Some(MoeShape {
        num_experts: VECTOR_EXPERTS,
        num_experts_per_tok: VECTOR_TOP_K,
        moe_intermediate_size: GDN_DECODER_LAYER_MOE_INTERMEDIATE,
        shared_expert_intermediate_size: GDN_DECODER_LAYER_MOE_INTERMEDIATE,
    });
    config.fixture_mut().vocab_size = 16;
    config
}

fn gdn_decoder_layer_vector_runner() -> ModelRunner {
    let ctx = Rc::new(CudaCtx::default().unwrap());
    let config = gdn_decoder_layer_vector_config();
    let mut engine = Engine::new(ctx.clone(), config.engine_config()).unwrap();
    let moe = config.moe_config().unwrap();
    let (moe_plan, workspace_bytes) = {
        let mut ops = engine.operators();
        let plan = unsafe {
            ops.qsfi()
                .create_moe_bf16_plan(MoeBf16PlanConfig {
                    kernel: crate::backend::qsfi::MoeBf16Kernel::COMPILED,
                    max_num_tokens: config.max_seq_len,
                    hidden_size: config.hidden_size(),
                    intermediate_size: moe.moe_intermediate_size,
                    num_experts: moe.num_experts,
                    top_k: moe.num_experts_per_tok,
                })
                .unwrap()
        };
        let workspace_bytes =
            unsafe { ops.qsfi().moe_workspace_size(&plan, config.max_seq_len) }.unwrap();
        (plan, workspace_bytes)
    };
    let scratch = RunnerScratch::new(ctx.clone(), &config, workspace_bytes).unwrap();
    let qscb_workspace =
        DeviceBuffer::with_capacity(ctx.clone(), (config.qscb_workspace_bytes).max(1)).unwrap();
    let gdn_state = Some(GdnState::new(ctx.clone(), &config).unwrap());

    ModelRunner {
        ctx: ctx.clone(),
        sampler: None,
        gdn_qkv: None,
        lm_head: None,
        config,
        tokenizer_token_count: config.vocab_size(),
        weights: placeholder_weights(),
        quantized_scratch: None,
        engine,
        moe_plan: Some(moe_plan),
        gdn_state,
        scratch,
        qscb_workspace,
        live_request_id: None,
        live_tokens: Vec::new(),
        last_next_tokens: Vec::new(),
        last_logits_rows: 0,
        last_logits_vocab_size: 0,
    }
}

struct GdnDecoderLayerFixture {
    layer: QwenLayerWeights<FusedBf16Mlp>,
    next_norm: DeviceBuffer<BF16>,
}

impl GdnDecoderLayerFixture {
    fn weights(&self) -> &QwenGdnWeights<FusedBf16Mlp> {
        match &self.layer {
            QwenLayerWeights::Gdn(layer) => layer,
            QwenLayerWeights::AttentionMlp(_) => {
                unreachable!("GDN decoder-layer fixture is not full attention")
            }
        }
    }
}

fn gdn_decoder_layer_fixture(ctx: &Rc<CudaCtx>) -> GdnDecoderLayerFixture {
    let hidden = HIDDEN_SIZE;
    let intermediate = GDN_DECODER_LAYER_MOE_INTERMEDIATE;
    let layer: QwenLayerWeights<FusedBf16Mlp> = QwenLayerWeights::Gdn(QwenGdnWeights {
        norm: Box::new(filled_bf16_buffer(ctx, hidden as usize, 0.0)),
        in_proj: Box::new(
            bf16_buffer(
                ctx.clone(),
                &read_gdn_decoder_bf16_vector(
                    "gdn_decoder_in_proj_weight.bf16",
                    checked_usize_product(&[PACKED_QKV_CHANNELS, hidden]).unwrap(),
                ),
            )
            .unwrap(),
        ),
        gate_proj: Box::new(
            bf16_buffer(
                ctx.clone(),
                &read_gdn_decoder_bf16_vector(
                    "gdn_decoder_gate_proj_weight.bf16",
                    checked_usize_product(&[OUTPUT_WIDTH, hidden]).unwrap(),
                ),
            )
            .unwrap(),
        ),
        a_proj: Box::new(
            bf16_buffer(
                ctx.clone(),
                &read_gdn_decoder_bf16_vector(
                    "gdn_decoder_a_proj_weight.bf16",
                    checked_usize_product(&[NUM_VALUE_HEADS, hidden]).unwrap(),
                ),
            )
            .unwrap(),
        ),
        b_proj: Box::new(
            bf16_buffer(
                ctx.clone(),
                &read_gdn_decoder_bf16_vector(
                    "gdn_decoder_b_proj_weight.bf16",
                    checked_usize_product(&[NUM_VALUE_HEADS, hidden]).unwrap(),
                ),
            )
            .unwrap(),
        ),
        conv_weight: Box::new(
            bf16_buffer(
                ctx.clone(),
                &read_gdn_decoder_bf16_vector(
                    "gdn_decoder_conv_weight.bf16",
                    checked_usize_product(&[PACKED_QKV_CHANNELS, CONV_WIDTH]).unwrap(),
                ),
            )
            .unwrap(),
        ),
        conv_bias: Box::new(
            bf16_buffer(
                ctx.clone(),
                &read_gdn_decoder_bf16_vector(
                    "gdn_decoder_conv_bias.bf16",
                    PACKED_QKV_CHANNELS as usize,
                ),
            )
            .unwrap(),
        ),
        a_log: Box::new(
            bf16_buffer(
                ctx.clone(),
                &read_gdn_decoder_bf16_vector("gdn_decoder_A_log.bf16", NUM_VALUE_HEADS as usize),
            )
            .unwrap(),
        ),
        dt_bias: Box::new(
            bf16_buffer(
                ctx.clone(),
                &read_gdn_decoder_bf16_vector("gdn_decoder_dt_bias.bf16", NUM_VALUE_HEADS as usize),
            )
            .unwrap(),
        ),
        rms_weight: Box::new(
            bf16_buffer(
                ctx.clone(),
                &read_gdn_decoder_bf16_vector(
                    "gdn_decoder_rms_weight.bf16",
                    VALUE_HEAD_DIM as usize,
                ),
            )
            .unwrap(),
        ),
        out_proj: Box::new(
            bf16_buffer(
                ctx.clone(),
                &read_gdn_decoder_bf16_vector(
                    "gdn_decoder_out_proj_weight.bf16",
                    checked_usize_product(&[hidden, OUTPUT_WIDTH]).unwrap(),
                ),
            )
            .unwrap(),
        ),
        mlp_norm: Box::new(
            bf16_buffer(
                ctx.clone(),
                &read_gdn_decoder_bf16_vector(
                    "gdn_decoder_mlp_norm_raw_weight.bf16",
                    hidden as usize,
                ),
            )
            .unwrap(),
        ),
        mlp: MoeMlp {
            router_proj: Box::new(
                bf16_buffer(
                    ctx.clone(),
                    &read_gdn_decoder_bf16_vector(
                        "gdn_decoder_moe_router_proj_weight.bf16",
                        checked_usize_product(&[VECTOR_EXPERTS, hidden]).unwrap(),
                    ),
                )
                .unwrap(),
            ),
            experts: FusedExperts {
                gate_up_proj: Box::new(
                    bf16_buffer(
                        ctx.clone(),
                        &read_gdn_decoder_bf16_vector(
                            "gdn_decoder_moe_gate_up_proj_weight.bf16",
                            checked_usize_product(&[VECTOR_EXPERTS, 2, intermediate, hidden])
                                .unwrap(),
                        ),
                    )
                    .unwrap(),
                ),
                down_proj: Box::new(
                    bf16_buffer(
                        ctx.clone(),
                        &read_gdn_decoder_bf16_vector(
                            "gdn_decoder_moe_down_proj_weight.bf16",
                            checked_usize_product(&[VECTOR_EXPERTS, hidden, intermediate]).unwrap(),
                        ),
                    )
                    .unwrap(),
                ),
            },
            shared: Some(SharedExpert {
                projections: DenseMlp {
                    gate_proj: Box::new(
                        bf16_buffer(
                            ctx.clone(),
                            &read_gdn_decoder_bf16_vector(
                                "gdn_decoder_moe_shared_gate_proj_weight.bf16",
                                checked_usize_product(&[intermediate, hidden]).unwrap(),
                            ),
                        )
                        .unwrap(),
                    ),
                    up_proj: Box::new(
                        bf16_buffer(
                            ctx.clone(),
                            &read_gdn_decoder_bf16_vector(
                                "gdn_decoder_moe_shared_up_proj_weight.bf16",
                                checked_usize_product(&[intermediate, hidden]).unwrap(),
                            ),
                        )
                        .unwrap(),
                    ),
                    down_proj: Box::new(
                        bf16_buffer(
                            ctx.clone(),
                            &read_gdn_decoder_bf16_vector(
                                "gdn_decoder_moe_shared_down_proj_weight.bf16",
                                checked_usize_product(&[hidden, intermediate]).unwrap(),
                            ),
                        )
                        .unwrap(),
                    ),
                },
                gate: Box::new(
                    bf16_buffer(
                        ctx.clone(),
                        &read_gdn_decoder_bf16_vector(
                            "gdn_decoder_moe_shared_expert_gate_weight.bf16",
                            hidden as usize,
                        ),
                    )
                    .unwrap(),
                ),
            }),
        },
    });
    let next_norm = bf16_buffer(
        ctx.clone(),
        &read_gdn_decoder_bf16_vector(
            "gdn_decoder_next_layer_norm_raw_weight.bf16",
            hidden as usize,
        ),
    )
    .unwrap();
    GdnDecoderLayerFixture { layer, next_norm }
}

fn model_logits_vector_config() -> QwenConfig {
    let mut config = QwenConfig::randomized_dense_tiny_fixture();
    config.fixture_mut().rms_norm_eps = RMS_NORM_EPS;
    config.fixture_mut().rope_theta = ROPE_THETA;
    config.fixture_mut().num_layers = 1;
    config.max_seq_len = MODEL_LOGITS_TOTAL_ROWS as u32;
    config.max_pages = 2;
    config.page_size = MODEL_LOGITS_PROMPT_LEN as u32;
    config.fixture_mut().intermediate_size = MODEL_LOGITS_INTERMEDIATE;
    config.fixture_mut().moe = HAS_EXPERTS.then_some(MoeShape {
        num_experts: VECTOR_EXPERTS,
        num_experts_per_tok: VECTOR_TOP_K,
        moe_intermediate_size: MODEL_LOGITS_INTERMEDIATE,
        shared_expert_intermediate_size: MODEL_LOGITS_INTERMEDIATE,
    });
    config.fixture_mut().vocab_size = MODEL_LOGITS_VOCAB;
    config
}

fn model_logits_vector_weights(ctx: &Rc<CudaCtx>, config: QwenConfig) -> QwenWeights {
    fn assemble<M>(
        ctx: &Rc<CudaCtx>,
        config: QwenConfig,
        mlp: M,
    ) -> QwenModel<M, W<BF16>, W<BF16>> {
        let hidden = config.hidden_size();
        let q_hidden = config.q_hidden_size().unwrap();
        let kv_hidden = config.kv_hidden_size().unwrap();
        let vocab = config.vocab_size();

        let token_embedding = bf16_buffer(
            ctx.clone(),
            &read_model_logits_bf16_vector(
                "model_token_embedding_weight.bf16",
                checked_usize_product(&[vocab, hidden]).unwrap(),
            ),
        )
        .unwrap();
        let final_norm = bf16_buffer(
            ctx.clone(),
            &read_model_logits_bf16_vector("model_final_norm_raw_weight.bf16", hidden as usize),
        )
        .unwrap();
        let lm_head = bf16_buffer(
            ctx.clone(),
            &read_model_logits_bf16_vector(
                "model_lm_head_weight.bf16",
                checked_usize_product(&[vocab, hidden]).unwrap(),
            ),
        )
        .unwrap();
        let layer: QwenLayerWeights<M> = QwenLayerWeights::AttentionMlp(QwenAttentionMlpWeights {
            attn_norm: Box::new(
                bf16_buffer(
                    ctx.clone(),
                    &read_model_logits_bf16_vector(
                        "model_attn_norm_raw_weight.bf16",
                        hidden as usize,
                    ),
                )
                .unwrap(),
            ),
            q_norm: Box::new(
                bf16_buffer(
                    ctx.clone(),
                    &read_model_logits_bf16_vector(
                        "model_q_norm_raw_weight.bf16",
                        config.head_dim() as usize,
                    ),
                )
                .unwrap(),
            ),
            k_norm: Box::new(
                bf16_buffer(
                    ctx.clone(),
                    &read_model_logits_bf16_vector(
                        "model_k_norm_raw_weight.bf16",
                        config.head_dim() as usize,
                    ),
                )
                .unwrap(),
            ),
            q_proj: Box::new(
                bf16_buffer(
                    ctx.clone(),
                    &read_model_logits_bf16_vector(
                        "model_q_proj_weight.bf16",
                        checked_usize_product(&[PACKED_Q_GATE_WIDTH, hidden]).unwrap(),
                    ),
                )
                .unwrap(),
            ),
            k_proj: Box::new(
                bf16_buffer(
                    ctx.clone(),
                    &read_model_logits_bf16_vector(
                        "model_k_proj_weight.bf16",
                        checked_usize_product(&[kv_hidden, hidden]).unwrap(),
                    ),
                )
                .unwrap(),
            ),
            v_proj: Box::new(
                bf16_buffer(
                    ctx.clone(),
                    &read_model_logits_bf16_vector(
                        "model_v_proj_weight.bf16",
                        checked_usize_product(&[kv_hidden, hidden]).unwrap(),
                    ),
                )
                .unwrap(),
            ),
            o_proj: Box::new(
                bf16_buffer(
                    ctx.clone(),
                    &read_model_logits_bf16_vector(
                        "model_o_proj_weight.bf16",
                        checked_usize_product(&[hidden, q_hidden]).unwrap(),
                    ),
                )
                .unwrap(),
            ),
            mlp_norm: Box::new(
                bf16_buffer(
                    ctx.clone(),
                    &read_model_logits_bf16_vector(
                        "model_mlp_norm_raw_weight.bf16",
                        hidden as usize,
                    ),
                )
                .unwrap(),
            ),
            mlp,
        });

        QwenModel {
            token_embedding: Box::new(token_embedding),
            final_norm: Box::new(final_norm),
            lm_head: Box::new(lm_head),
            layers: vec![layer],
        }
    }

    let hidden = config.hidden_size();
    if let Some(moe) = config.moe_config() {
        let mlp: FusedBf16Mlp = MoeMlp {
            router_proj: Box::new(
                bf16_buffer(
                    ctx.clone(),
                    &read_model_logits_bf16_vector(
                        "model_moe_router_proj_weight.bf16",
                        checked_usize_product(&[moe.num_experts, hidden]).unwrap(),
                    ),
                )
                .unwrap(),
            ),
            experts: FusedExperts {
                gate_up_proj: Box::new(
                    bf16_buffer(
                        ctx.clone(),
                        &read_model_logits_bf16_vector(
                            "model_moe_gate_up_proj_weight.bf16",
                            checked_usize_product(&[
                                moe.num_experts,
                                2,
                                moe.moe_intermediate_size,
                                hidden,
                            ])
                            .unwrap(),
                        ),
                    )
                    .unwrap(),
                ),
                down_proj: Box::new(
                    bf16_buffer(
                        ctx.clone(),
                        &read_model_logits_bf16_vector(
                            "model_moe_down_proj_weight.bf16",
                            checked_usize_product(&[
                                moe.num_experts,
                                hidden,
                                moe.moe_intermediate_size,
                            ])
                            .unwrap(),
                        ),
                    )
                    .unwrap(),
                ),
            },
            shared: Some(SharedExpert {
                projections: DenseMlp {
                    gate_proj: Box::new(
                        bf16_buffer(
                            ctx.clone(),
                            &read_model_logits_bf16_vector(
                                "model_moe_shared_gate_proj_weight.bf16",
                                checked_usize_product(&[
                                    moe.shared_expert_intermediate_size,
                                    hidden,
                                ])
                                .unwrap(),
                            ),
                        )
                        .unwrap(),
                    ),
                    up_proj: Box::new(
                        bf16_buffer(
                            ctx.clone(),
                            &read_model_logits_bf16_vector(
                                "model_moe_shared_up_proj_weight.bf16",
                                checked_usize_product(&[
                                    moe.shared_expert_intermediate_size,
                                    hidden,
                                ])
                                .unwrap(),
                            ),
                        )
                        .unwrap(),
                    ),
                    down_proj: Box::new(
                        bf16_buffer(
                            ctx.clone(),
                            &read_model_logits_bf16_vector(
                                "model_moe_shared_down_proj_weight.bf16",
                                checked_usize_product(&[
                                    hidden,
                                    moe.shared_expert_intermediate_size,
                                ])
                                .unwrap(),
                            ),
                        )
                        .unwrap(),
                    ),
                },
                gate: Box::new(
                    bf16_buffer(
                        ctx.clone(),
                        &read_model_logits_bf16_vector(
                            "model_moe_shared_expert_gate_weight.bf16",
                            hidden as usize,
                        ),
                    )
                    .unwrap(),
                ),
            }),
        };
        QwenWeights::MoeBf16(assemble(ctx, config, mlp))
    } else {
        let weight = |file: &str| {
            Box::new(
                bf16_buffer(
                    ctx.clone(),
                    &read_model_logits_bf16_vector(
                        file,
                        checked_usize_product(&[hidden, MODEL_LOGITS_INTERMEDIATE]).unwrap(),
                    ),
                )
                .unwrap(),
            )
        };
        let mlp: DenseMlp<W<BF16>> = DenseMlp {
            gate_proj: weight("model_dense_gate_proj_weight.bf16"),
            up_proj: weight("model_dense_up_proj_weight.bf16"),
            down_proj: weight("model_dense_down_proj_weight.bf16"),
        };
        QwenWeights::DenseBf16(assemble(ctx, config, mlp))
    }
}

fn full_attention_block_hidden_len() -> usize {
    checked_usize_product(&[FULL_ATTN_BLOCK_VECTOR_ROWS, HIDDEN_SIZE]).unwrap()
}

fn full_attention_block_q_len() -> usize {
    checked_usize_product(&[FULL_ATTN_BLOCK_VECTOR_ROWS, NUM_Q_HEADS, HEAD_DIM]).unwrap()
}

fn full_attention_block_kv_len() -> usize {
    checked_usize_product(&[FULL_ATTN_BLOCK_VECTOR_ROWS, NUM_KV_HEADS, HEAD_DIM]).unwrap()
}

fn full_attention_block_q_proj_len() -> usize {
    checked_usize_product(&[FULL_ATTN_BLOCK_VECTOR_ROWS, PACKED_Q_GATE_WIDTH]).unwrap()
}

fn gdn_decoder_layer_hidden_len() -> usize {
    checked_usize_product(&[GDN_DECODER_LAYER_VECTOR_ROWS, HIDDEN_SIZE]).unwrap()
}

fn gdn_decoder_layer_topk_len() -> usize {
    checked_usize_product(&[GDN_DECODER_LAYER_VECTOR_ROWS, VECTOR_TOP_K]).unwrap()
}

fn full_attention_q_len() -> usize {
    checked_usize_product(&[FULL_ATTN_VECTOR_ROWS, NUM_Q_HEADS, HEAD_DIM]).unwrap()
}

fn full_attention_kv_len() -> usize {
    checked_usize_product(&[FULL_ATTN_VECTOR_ROWS, NUM_KV_HEADS, HEAD_DIM]).unwrap()
}

fn full_attention_packed_q_gate_len() -> usize {
    checked_usize_product(&[FULL_ATTN_VECTOR_ROWS, PACKED_Q_GATE_WIDTH]).unwrap()
}

fn has_nonzero_bf16(values: &[u16]) -> bool {
    values.iter().any(|value| *value != 0)
}

fn qwen36_packed_q_gate_pattern(row: usize, head: usize, lane: usize, gate: bool) -> u16 {
    let value = (((row + 1) as u16) << 12) | ((head as u16) << 8) | lane as u16;
    if gate { value ^ 0x8000 } else { value }
}

#[test]
fn qwen36_packed_attention_q_gate_extraction_preserves_rows_heads_and_lanes() {
    if !cuda_device_available() {
        return;
    }

    const ROWS: usize = 3;
    const HEADS: usize = NUM_Q_HEADS as usize;
    const HEAD_LANES: usize = HEAD_DIM as usize;
    const PACKED_HEAD_DIM: usize = 2 * HEAD_LANES;

    let mut packed = vec![0_u16; ROWS * HEADS * PACKED_HEAD_DIM];
    let mut expected_q = vec![0_u16; ROWS * HEADS * HEAD_LANES];
    let mut expected_gate = vec![0_u16; ROWS * HEADS * HEAD_LANES];
    for row in 0..ROWS {
        for head in 0..HEADS {
            for lane in 0..HEAD_LANES {
                let packed_base = (row * HEADS + head) * PACKED_HEAD_DIM;
                let out_idx = (row * HEADS + head) * HEAD_LANES + lane;
                let q = qwen36_packed_q_gate_pattern(row, head, lane, false);
                let gate = qwen36_packed_q_gate_pattern(row, head, lane, true);
                packed[packed_base + lane] = q;
                packed[packed_base + HEAD_LANES + lane] = gate;
                expected_q[out_idx] = q;
                expected_gate[out_idx] = gate;
            }
        }
    }

    let ctx = Rc::new(CudaCtx::default().unwrap());
    let packed_device = bf16_buffer(ctx.clone(), &packed).unwrap();
    let q_device = DeviceBuffer::with_capacity(ctx.clone(), expected_q.len()).unwrap();
    let gate_device = DeviceBuffer::with_capacity(ctx.clone(), expected_gate.len()).unwrap();

    unsafe {
        crate::engine::Engine::new(ctx.clone(), full_attention_vector_config().engine_config())
            .unwrap()
            .operators()
            .qscu()
            .qwen36_extract_q_and_gate_bf16(
                packed_device
                    .matrix(ROWS as u32, PACKED_Q_GATE_WIDTH)
                    .unwrap(),
                q_device.matrix(ROWS as u32, Q_WIDTH).unwrap(),
                gate_device.matrix(ROWS as u32, Q_WIDTH).unwrap(),
            )
            .unwrap();
    }
    ctx.synchronize().unwrap();

    assert_eq!(download_bf16(&q_device, &ctx, expected_q.len()), expected_q);
    assert_eq!(
        download_bf16(&gate_device, &ctx, expected_gate.len()),
        expected_gate
    );
}

#[test]
fn qwen36_full_attention_vectors_validate_packed_q_gate_extraction() {
    if !cuda_device_available() {
        return;
    }

    let mut runner = full_attention_vector_runner();
    let ctx = runner.ctx.clone();
    let rows = FULL_ATTN_VECTOR_ROWS;
    let q_len = full_attention_q_len();
    runner.scratch.reserve(rows).unwrap();

    let packed = read_bf16_vector(
        "attention_packed_q_gate.bf16",
        full_attention_packed_q_gate_len(),
    );
    upload!(&runner.ctx, &mut runner.scratch.q_proj_out, &packed).unwrap();

    unsafe {
        runner
            .engine
            .operators()
            .qscu()
            .qwen36_extract_q_and_gate_bf16(
                runner
                    .scratch
                    .q_proj_out
                    .matrix(rows, PACKED_Q_GATE_WIDTH)
                    .unwrap(),
                runner.scratch.q.matrix(rows, Q_WIDTH).unwrap(),
                runner.scratch.attn_gate.matrix(rows, Q_WIDTH).unwrap(),
            )
    }
    .unwrap();
    ctx.synchronize().unwrap();

    let got_q = download_bf16(&runner.scratch.q, &ctx, q_len);
    let got_gate = download_bf16(&runner.scratch.attn_gate, &ctx, q_len);
    let expected_q = read_bf16_vector("attention_q_extracted.bf16", q_len);
    let expected_gate = read_bf16_vector("attention_gate_extracted.bf16", q_len);
    assert_bf16_exact("attention q extraction vector", &got_q, &expected_q);
    assert_bf16_exact(
        "attention gate extraction vector",
        &got_gate,
        &expected_gate,
    );
}

#[test]
fn qwen36_full_attention_vectors_validate_qk_norm_and_rope_pipeline() {
    if !cuda_device_available() {
        return;
    }

    let mut runner = full_attention_vector_runner();
    let ctx = runner.ctx.clone();
    let rows = FULL_ATTN_VECTOR_ROWS;
    let q_len = full_attention_q_len();
    let kv_len = full_attention_kv_len();
    let head_dim = HEAD_DIM as usize;
    runner.scratch.reserve(rows).unwrap();

    let packed = read_bf16_vector(
        "attention_packed_q_gate.bf16",
        full_attention_packed_q_gate_len(),
    );
    let k_input = read_bf16_vector("attention_k_input.bf16", kv_len);
    let positions = read_i32_vector("attention_positions.i32", rows as usize);
    upload!(&runner.ctx, &mut runner.scratch.q_proj_out, &packed).unwrap();
    upload!(&runner.ctx, &mut runner.scratch.k, &k_input).unwrap();
    upload!(&runner.ctx, &mut runner.scratch.positions, &positions).unwrap();

    unsafe {
        runner
            .engine
            .operators()
            .qscu()
            .qwen36_extract_q_and_gate_bf16(
                runner
                    .scratch
                    .q_proj_out
                    .matrix(rows, PACKED_Q_GATE_WIDTH)
                    .unwrap(),
                runner.scratch.q.matrix(rows, Q_WIDTH).unwrap(),
                runner.scratch.attn_gate.matrix(rows, Q_WIDTH).unwrap(),
            )
    }
    .unwrap();

    let q_norm_weight = bf16_buffer(
        ctx.clone(),
        &read_bf16_vector("attention_q_norm_raw_weight.bf16", head_dim),
    )
    .unwrap();
    let k_norm_weight = bf16_buffer(
        ctx.clone(),
        &read_bf16_vector("attention_k_norm_raw_weight.bf16", head_dim),
    )
    .unwrap();

    let q_ptr = runner
        .scratch
        .q
        .matrix(rows * runner.config.num_q_heads(), runner.config.head_dim())
        .unwrap();
    unsafe {
        runner.engine.operators().qsfi().rmsnorm_bf16(
            &crate::backend::qsfi::RmsNormBf16::qwen_qk_norm(
                q_ptr,
                q_norm_weight.vector(runner.config.head_dim()).unwrap(),
                q_ptr,
                runner.config.rms_norm_eps(),
            )
            .unwrap(),
        )
    }
    .unwrap();
    let k_ptr = runner
        .scratch
        .k
        .matrix(
            rows * runner.config.num_kv_heads(),
            runner.config.head_dim(),
        )
        .unwrap();
    unsafe {
        runner.engine.operators().qsfi().rmsnorm_bf16(
            &crate::backend::qsfi::RmsNormBf16::qwen_qk_norm(
                k_ptr,
                k_norm_weight.vector(runner.config.head_dim()).unwrap(),
                k_ptr,
                runner.config.rms_norm_eps(),
            )
            .unwrap(),
        )
    }
    .unwrap();
    ctx.synchronize().unwrap();

    let got_q_norm = download_bf16(&runner.scratch.q, &ctx, q_len);
    let got_k_norm = download_bf16(&runner.scratch.k, &ctx, kv_len);
    let expected_q_norm = read_bf16_vector("attention_q_norm_bf16.bf16", q_len);
    let expected_k_norm = read_bf16_vector("attention_k_norm_bf16.bf16", kv_len);
    assert_bf16_exact(
        "attention q Gemma RMSNorm vector",
        &got_q_norm,
        &expected_q_norm,
    );
    assert_bf16_exact(
        "attention k Gemma RMSNorm vector",
        &got_k_norm,
        &expected_k_norm,
    );

    unsafe {
        runner
            .execution()
            .unwrap()
            .1
            .apply_attention_rope(rows)
            .unwrap();
    }
    ctx.synchronize().unwrap();

    let got_q_rope = download_bf16(&runner.scratch.q, &ctx, q_len);
    let got_k_rope = download_bf16(&runner.scratch.k, &ctx, kv_len);
    let expected_q_rope = read_bf16_vector("attention_q_rope_bf16.bf16", q_len);
    let expected_k_rope = read_bf16_vector("attention_k_rope_bf16.bf16", kv_len);
    let expected_q_rope_f32 = read_f32_vector("attention_q_rope_f32.f32", q_len);
    let expected_k_rope_f32 = read_f32_vector("attention_k_rope_f32.f32", kv_len);
    assert_bf16_close_to_f32_oracle(
        "attention q partial RoPE vector",
        &got_q_rope,
        &expected_q_rope,
        &expected_q_rope_f32,
        BF16_ROPE_ABS_TOL,
    );
    assert_bf16_close_to_f32_oracle(
        "attention k partial RoPE vector",
        &got_k_rope,
        &expected_k_rope,
        &expected_k_rope_f32,
        BF16_ROPE_ABS_TOL,
    );
}

#[test]
fn qwen36_full_attention_vectors_validate_output_gate() {
    if !cuda_device_available() {
        return;
    }

    let mut runner = full_attention_vector_runner();
    let ctx = runner.ctx.clone();
    let rows = FULL_ATTN_VECTOR_ROWS;
    let q_len = full_attention_q_len();
    runner.scratch.reserve(rows).unwrap();

    let gate = read_bf16_vector("attention_gate_extracted.bf16", q_len);
    let out_initial = read_bf16_vector("attention_gate_out_initial.bf16", q_len);
    upload!(&runner.ctx, &mut runner.scratch.attn_gate, &gate).unwrap();
    upload!(&runner.ctx, &mut runner.scratch.attn_out, &out_initial).unwrap();

    unsafe {
        runner
            .engine
            .operators()
            .qscu()
            .qwen36_full_attention_output_gate_bf16(
                runner.scratch.attn_gate.matrix(rows, Q_WIDTH).unwrap(),
                runner.scratch.attn_out.matrix(rows, Q_WIDTH).unwrap(),
            )
    }
    .unwrap();
    ctx.synchronize().unwrap();

    let got = download_bf16(&runner.scratch.attn_out, &ctx, q_len);
    let expected = read_bf16_vector("attention_gated_output_bf16.bf16", q_len);
    assert_bf16_exact("attention output gate vector", &got, &expected);
}

#[test]
fn qwen36_full_attention_block_vector_validates_attention_residual_norm_composition() {
    if !cuda_device_available() {
        return;
    }

    let mut runner = full_attention_block_vector_runner();
    let ctx = runner.ctx.clone();
    let rows = FULL_ATTN_BLOCK_VECTOR_ROWS;
    let hidden = HIDDEN_SIZE;
    let q_hidden = Q_WIDTH;
    let kv_hidden = KV_WIDTH;
    let hidden_len = full_attention_block_hidden_len();
    let q_len = full_attention_block_q_len();
    let kv_len = full_attention_block_kv_len();
    let q_proj_len = full_attention_block_q_proj_len();
    runner.scratch.reserve(rows).unwrap();

    let input_residual = read_block_bf16_vector("block_input_residual.bf16", hidden_len);
    let positions = read_block_i32_vector("block_positions.i32", rows as usize);
    upload!(&runner.ctx, &mut runner.scratch.residual, &input_residual).unwrap();
    upload!(&runner.ctx, &mut runner.scratch.positions, &positions).unwrap();

    let attn_norm_weight = bf16_buffer(
        ctx.clone(),
        &read_block_bf16_vector("block_attn_norm_raw_weight.bf16", hidden as usize),
    )
    .unwrap();
    let q_norm_weight = bf16_buffer(
        ctx.clone(),
        &read_block_bf16_vector("block_q_norm_raw_weight.bf16", HEAD_DIM as usize),
    )
    .unwrap();
    let k_norm_weight = bf16_buffer(
        ctx.clone(),
        &read_block_bf16_vector("block_k_norm_raw_weight.bf16", HEAD_DIM as usize),
    )
    .unwrap();
    let post_norm_weight = bf16_buffer(
        ctx.clone(),
        &read_block_bf16_vector("block_post_attn_norm_raw_weight.bf16", hidden as usize),
    )
    .unwrap();
    let q_proj_weight = bf16_buffer(
        ctx.clone(),
        &read_block_bf16_vector(
            "block_q_proj_weight.bf16",
            checked_usize_product(&[PACKED_Q_GATE_WIDTH, hidden]).unwrap(),
        ),
    )
    .unwrap();
    let k_proj_weight = bf16_buffer(
        ctx.clone(),
        &read_block_bf16_vector(
            "block_k_proj_weight.bf16",
            checked_usize_product(&[kv_hidden, hidden]).unwrap(),
        ),
    )
    .unwrap();
    let v_proj_weight = bf16_buffer(
        ctx.clone(),
        &read_block_bf16_vector(
            "block_v_proj_weight.bf16",
            checked_usize_product(&[kv_hidden, hidden]).unwrap(),
        ),
    )
    .unwrap();
    let o_proj_weight = bf16_buffer(
        ctx.clone(),
        &read_block_bf16_vector(
            "block_o_proj_weight.bf16",
            checked_usize_product(&[hidden, q_hidden]).unwrap(),
        ),
    )
    .unwrap();

    let residual_ptr = runner.scratch.residual.matrix(rows, hidden).unwrap();
    let norm_ptr = runner.scratch.norm.matrix(rows, hidden).unwrap();
    unsafe {
        runner.engine.operators().qsfi().rmsnorm_bf16(
            &crate::backend::qsfi::RmsNormBf16::qwen_decoder_norm(
                residual_ptr,
                attn_norm_weight
                    .vector(runner.config.hidden_size())
                    .unwrap(),
                norm_ptr,
                runner.config.rms_norm_eps(),
            )
            .unwrap(),
        )
    }
    .unwrap();
    ctx.synchronize().unwrap();
    let got_attn_norm = download_bf16(&runner.scratch.norm, &ctx, hidden_len);
    assert_bf16_close_to_f32_oracle(
        "full-attention block input Gemma RMSNorm",
        &got_attn_norm,
        &read_block_bf16_vector("block_expected_attn_norm_output_bf16.bf16", hidden_len),
        &read_block_f32_vector("block_expected_attn_norm_output_f32.f32", hidden_len),
        BF16_BLOCK_NORM_ABS_TOL,
    );

    unsafe {
        runner.engine.operators().qscb().linear(
            norm_ptr,
            q_proj_weight.matrix(PACKED_Q_GATE_WIDTH, hidden).unwrap(),
            runner
                .scratch
                .q_proj_out
                .matrix(rows, PACKED_Q_GATE_WIDTH)
                .unwrap(),
            runner
                .qscb_workspace
                .workspace(runner.config.qscb_workspace_bytes)
                .unwrap(),
        )
    }
    .unwrap();
    unsafe {
        runner
            .engine
            .operators()
            .qscu()
            .qwen36_extract_q_and_gate_bf16(
                runner
                    .scratch
                    .q_proj_out
                    .matrix(rows, PACKED_Q_GATE_WIDTH)
                    .unwrap(),
                runner.scratch.q.matrix(rows, Q_WIDTH).unwrap(),
                runner.scratch.attn_gate.matrix(rows, Q_WIDTH).unwrap(),
            )
    }
    .unwrap();
    unsafe {
        runner.engine.operators().qscb().linear(
            norm_ptr,
            k_proj_weight.matrix(kv_hidden, hidden).unwrap(),
            runner.scratch.k.matrix(rows, kv_hidden).unwrap(),
            runner
                .qscb_workspace
                .workspace(runner.config.qscb_workspace_bytes)
                .unwrap(),
        )
    }
    .unwrap();
    unsafe {
        runner.engine.operators().qscb().linear(
            norm_ptr,
            v_proj_weight.matrix(kv_hidden, hidden).unwrap(),
            runner.scratch.v.matrix(rows, kv_hidden).unwrap(),
            runner
                .qscb_workspace
                .workspace(runner.config.qscb_workspace_bytes)
                .unwrap(),
        )
    }
    .unwrap();
    ctx.synchronize().unwrap();

    let got_q_proj = download_bf16(&runner.scratch.q_proj_out, &ctx, q_proj_len);
    assert_bf16_close_to_f32_oracle(
        "full-attention block q projection",
        &got_q_proj,
        &read_block_bf16_vector("block_expected_q_proj_output_bf16.bf16", q_proj_len),
        &read_block_f32_vector("block_expected_q_proj_output_f32.f32", q_proj_len),
        BF16_BLOCK_PROJ_ABS_TOL,
    );
    assert_bf16_exact(
        "full-attention block q extraction",
        &download_bf16(&runner.scratch.q, &ctx, q_len),
        &read_block_bf16_vector("block_expected_q_extracted_bf16.bf16", q_len),
    );
    assert_bf16_exact(
        "full-attention block output-gate extraction",
        &download_bf16(&runner.scratch.attn_gate, &ctx, q_len),
        &read_block_bf16_vector("block_expected_gate_extracted_bf16.bf16", q_len),
    );
    assert_bf16_close_to_f32_oracle(
        "full-attention block k projection",
        &download_bf16(&runner.scratch.k, &ctx, kv_len),
        &read_block_bf16_vector("block_expected_k_proj_output_bf16.bf16", kv_len),
        &read_block_f32_vector("block_expected_k_proj_output_f32.f32", kv_len),
        BF16_BLOCK_PROJ_ABS_TOL,
    );
    assert_bf16_close_to_f32_oracle(
        "full-attention block v projection",
        &download_bf16(&runner.scratch.v, &ctx, kv_len),
        &read_block_bf16_vector("block_expected_v_proj_output_bf16.bf16", kv_len),
        &read_block_f32_vector("block_expected_v_proj_output_f32.f32", kv_len),
        BF16_BLOCK_PROJ_ABS_TOL,
    );

    let q_ptr = runner
        .scratch
        .q
        .matrix(rows * runner.config.num_q_heads(), runner.config.head_dim())
        .unwrap();
    unsafe {
        runner.engine.operators().qsfi().rmsnorm_bf16(
            &crate::backend::qsfi::RmsNormBf16::qwen_qk_norm(
                q_ptr,
                q_norm_weight.vector(runner.config.head_dim()).unwrap(),
                q_ptr,
                runner.config.rms_norm_eps(),
            )
            .unwrap(),
        )
    }
    .unwrap();
    let k_ptr = runner
        .scratch
        .k
        .matrix(
            rows * runner.config.num_kv_heads(),
            runner.config.head_dim(),
        )
        .unwrap();
    unsafe {
        runner.engine.operators().qsfi().rmsnorm_bf16(
            &crate::backend::qsfi::RmsNormBf16::qwen_qk_norm(
                k_ptr,
                k_norm_weight.vector(runner.config.head_dim()).unwrap(),
                k_ptr,
                runner.config.rms_norm_eps(),
            )
            .unwrap(),
        )
    }
    .unwrap();
    ctx.synchronize().unwrap();
    assert_bf16_close_to_f32_oracle(
        "full-attention block q Gemma RMSNorm",
        &download_bf16(&runner.scratch.q, &ctx, q_len),
        &read_block_bf16_vector("block_expected_q_norm_output_bf16.bf16", q_len),
        &read_block_f32_vector("block_expected_q_norm_output_f32.f32", q_len),
        BF16_BLOCK_NORM_ABS_TOL,
    );
    assert_bf16_close_to_f32_oracle(
        "full-attention block k Gemma RMSNorm",
        &download_bf16(&runner.scratch.k, &ctx, kv_len),
        &read_block_bf16_vector("block_expected_k_norm_output_bf16.bf16", kv_len),
        &read_block_f32_vector("block_expected_k_norm_output_f32.f32", kv_len),
        BF16_BLOCK_NORM_ABS_TOL,
    );

    unsafe {
        runner
            .execution()
            .unwrap()
            .1
            .apply_attention_rope(rows)
            .unwrap();
    }
    ctx.synchronize().unwrap();
    assert_bf16_close_to_f32_oracle(
        "full-attention block q partial RoPE",
        &download_bf16(&runner.scratch.q, &ctx, q_len),
        &read_block_bf16_vector("block_expected_q_rope_output_bf16.bf16", q_len),
        &read_block_f32_vector("block_expected_q_rope_output_f32.f32", q_len),
        BF16_BLOCK_ROPE_ABS_TOL,
    );
    assert_bf16_close_to_f32_oracle(
        "full-attention block k partial RoPE",
        &download_bf16(&runner.scratch.k, &ctx, kv_len),
        &read_block_bf16_vector("block_expected_k_rope_output_bf16.bf16", kv_len),
        &read_block_f32_vector("block_expected_k_rope_output_f32.f32", kv_len),
        BF16_BLOCK_ROPE_ABS_TOL,
    );

    let tokens = [101, 102, 103, 104, 105, 106];
    let token_indptr = [0, rows as i32];
    runner
        .engine
        .begin_append(AppendBatch {
            request_ids: &[0xFA77_0001],
            token_indptr: &token_indptr,
            tokens: &tokens,
        })
        .unwrap();
    let engine_layer = AttentionLayer::bf16_attention(
        0,
        runner
            .scratch
            .q
            .heads(rows, runner.config.num_q_heads(), runner.config.head_dim())
            .unwrap(),
        runner
            .scratch
            .k
            .heads(rows, runner.config.num_kv_heads(), runner.config.head_dim())
            .unwrap(),
        runner
            .scratch
            .v
            .heads(rows, runner.config.num_kv_heads(), runner.config.head_dim())
            .unwrap(),
        runner
            .scratch
            .attn_out
            .heads(rows, runner.config.num_q_heads(), runner.config.head_dim())
            .unwrap(),
        runner.scratch.positions.vector(rows).unwrap(),
    );
    unsafe { runner.engine.append_attention(&engine_layer) }.unwrap();
    ctx.synchronize().unwrap();
    assert_bf16_close_to_f32_oracle(
        "full-attention block raw causal GQA attention",
        &download_bf16(&runner.scratch.attn_out, &ctx, q_len),
        &read_block_bf16_vector("block_expected_raw_attention_output_bf16.bf16", q_len),
        &read_block_f32_vector("block_expected_raw_attention_output_f32.f32", q_len),
        BF16_BLOCK_ATTN_ABS_TOL,
    );

    unsafe {
        runner
            .engine
            .operators()
            .qscu()
            .qwen36_full_attention_output_gate_bf16(
                runner.scratch.attn_gate.matrix(rows, Q_WIDTH).unwrap(),
                runner.scratch.attn_out.matrix(rows, Q_WIDTH).unwrap(),
            )
    }
    .unwrap();
    ctx.synchronize().unwrap();
    assert_bf16_close_to_f32_oracle(
        "full-attention block sigmoid output gate",
        &download_bf16(&runner.scratch.attn_out, &ctx, q_len),
        &read_block_bf16_vector("block_expected_gated_attention_output_bf16.bf16", q_len),
        &read_block_f32_vector("block_expected_gated_attention_output_f32.f32", q_len),
        BF16_BLOCK_ATTN_ABS_TOL,
    );

    unsafe {
        runner.engine.operators().qscb().linear(
            runner.scratch.attn_out.matrix(rows, q_hidden).unwrap(),
            o_proj_weight.matrix(hidden, q_hidden).unwrap(),
            runner.scratch.attn_proj.matrix(rows, hidden).unwrap(),
            runner
                .qscb_workspace
                .workspace(runner.config.qscb_workspace_bytes)
                .unwrap(),
        )
    }
    .unwrap();
    ctx.synchronize().unwrap();
    assert_bf16_close_to_f32_oracle(
        "full-attention block output projection",
        &download_bf16(&runner.scratch.attn_proj, &ctx, hidden_len),
        &read_block_bf16_vector("block_expected_o_proj_output_bf16.bf16", hidden_len),
        &read_block_f32_vector("block_expected_o_proj_output_f32.f32", hidden_len),
        BF16_BLOCK_PROJ_ABS_TOL,
    );

    unsafe {
        runner.engine.operators().qsfi().fused_add_rmsnorm_bf16(
            &crate::backend::qsfi::FusedAddRmsNormBf16::qwen_decoder_norm(
                runner
                    .scratch
                    .attn_proj
                    .matrix(rows, runner.config.hidden_size())
                    .unwrap(),
                runner
                    .scratch
                    .residual
                    .matrix(rows, runner.config.hidden_size())
                    .unwrap(),
                post_norm_weight
                    .vector(runner.config.hidden_size())
                    .unwrap(),
                runner.config.rms_norm_eps(),
            )
            .unwrap(),
        )
    }
    .unwrap();
    ctx.synchronize().unwrap();
    assert_bf16_close_to_f32_oracle(
        "full-attention block residual after attention add",
        &download_bf16(&runner.scratch.residual, &ctx, hidden_len),
        &read_block_bf16_vector(
            "block_expected_residual_after_attention_bf16.bf16",
            hidden_len,
        ),
        &read_block_f32_vector(
            "block_expected_residual_after_attention_f32.f32",
            hidden_len,
        ),
        BF16_BLOCK_NORM_ABS_TOL,
    );
    assert_bf16_close_to_f32_oracle(
        "full-attention block post-attention Gemma RMSNorm",
        &download_bf16(&runner.scratch.attn_proj, &ctx, hidden_len),
        &read_block_bf16_vector("block_expected_post_attn_norm_output_bf16.bf16", hidden_len),
        &read_block_f32_vector("block_expected_post_attn_norm_output_f32.f32", hidden_len),
        BF16_BLOCK_NORM_ABS_TOL,
    );

    runner.engine.commit_batch(Commit::default()).unwrap();
}

#[test]
fn qwen36_full_attention_decoder_slice_chains_attention_into_moe_and_next_norm() {
    if !cuda_device_available() {
        return;
    }

    let mut runner = full_attention_block_moe_vector_runner();
    let ctx = runner.ctx.clone();
    let rows = FULL_ATTN_BLOCK_VECTOR_ROWS;
    let hidden = HIDDEN_SIZE;
    let hidden_len = full_attention_block_hidden_len();
    runner.scratch.reserve(rows).unwrap();

    let layer = full_attention_block_moe_layer(&ctx);
    let layer_weights = layer.weights();
    upload!(
        &runner.ctx,
        &mut runner.scratch.residual,
        &read_block_bf16_vector("block_input_residual.bf16", hidden_len),
    )
    .unwrap();
    upload!(
        &runner.ctx,
        &mut runner.scratch.positions,
        &read_block_i32_vector("block_positions.i32", rows as usize),
    )
    .unwrap();

    let residual_ptr = runner.scratch.residual.matrix(rows, hidden).unwrap();
    let norm_ptr = runner.scratch.norm.matrix(rows, hidden).unwrap();
    unsafe {
        runner.engine.operators().qsfi().rmsnorm_bf16(
            &crate::backend::qsfi::RmsNormBf16::qwen_decoder_norm(
                residual_ptr,
                layer_weights
                    .attn_norm
                    .vector(runner.config.hidden_size())
                    .unwrap(),
                norm_ptr,
                runner.config.rms_norm_eps(),
            )
            .unwrap(),
        )
    }
    .unwrap();

    let tokens = [101, 102, 103, 104, 105, 106];
    let token_indptr = [0, rows as i32];
    runner
        .engine
        .begin_append(AppendBatch {
            request_ids: &[0xFA77_0002],
            token_indptr: &token_indptr,
            tokens: &tokens,
        })
        .unwrap();
    let attention_layer_idx = runner.config.attention_layer_index(0).unwrap();
    unsafe {
        runner
            .execution()
            .unwrap()
            .1
            .execute_attention_layer(
                attention_layer_idx,
                rows,
                norm_ptr,
                layer_weights,
                ActiveRunKind::Append,
            )
            .unwrap();
    }
    ctx.synchronize().unwrap();
    assert_bf16_close_to_f32_oracle(
        "decoder slice produced attention output projection",
        &download_bf16(&runner.scratch.attn_proj, &ctx, hidden_len),
        &read_block_bf16_vector("block_expected_o_proj_output_bf16.bf16", hidden_len),
        &read_block_f32_vector("block_expected_o_proj_output_f32.f32", hidden_len),
        BF16_BLOCK_PROJ_ABS_TOL,
    );

    unsafe {
        runner
            .execution()
            .unwrap()
            .1
            .execute_post_attention_mlp(
                rows,
                &layer_weights.mlp_norm,
                &layer_weights.mlp,
                &layer.next_norm,
            )
            .unwrap();
    }
    ctx.synchronize().unwrap();

    assert_bf16_close_to_f32_oracle(
        "decoder slice post-attention Gemma RMSNorm handoff",
        &download_bf16(&runner.scratch.attn_proj, &ctx, hidden_len),
        &read_block_bf16_vector("block_expected_post_attn_norm_output_bf16.bf16", hidden_len),
        &read_block_f32_vector("block_expected_post_attn_norm_output_f32.f32", hidden_len),
        BF16_BLOCK_NORM_ABS_TOL,
    );
    assert_bf16_close_to_f32_oracle(
        "decoder slice residual after attention plus MoE/shared expert",
        &download_bf16(&runner.scratch.residual, &ctx, hidden_len),
        &read_block_bf16_vector("block_expected_residual_after_mlp_bf16.bf16", hidden_len),
        &read_block_f32_vector("block_expected_residual_after_mlp_f32.f32", hidden_len),
        BF16_BLOCK_FINAL_NORM_ABS_TOL,
    );
    assert_bf16_close_to_f32_oracle(
        "decoder slice next-layer Gemma RMSNorm",
        &download_bf16(&runner.scratch.mlp_out, &ctx, hidden_len),
        &read_block_bf16_vector(
            "block_expected_next_layer_norm_output_bf16.bf16",
            hidden_len,
        ),
        &read_block_f32_vector("block_expected_next_layer_norm_output_f32.f32", hidden_len),
        BF16_BLOCK_FINAL_NORM_ABS_TOL,
    );

    runner.engine.commit_batch(Commit::default()).unwrap();
}

#[test]
fn qwen36_gdn_qkv_triton_decode_matches_cublaslt() {
    if !cuda_device_available() {
        return;
    }

    let mut runner = gdn_decoder_layer_vector_runner();
    let ctx = runner.ctx.clone();
    let hidden = runner.config.hidden_size();
    let packed = PACKED_QKV_CHANNELS;
    runner.scratch.reserve(1).unwrap();
    runner.gdn_qkv = Some(unsafe { crate::backend::qstriton::GdnQkv::load().unwrap() });
    assert_eq!(runner.gdn_qkv_provider(), "triton");

    // Exactly representable BF16 values with varying signs and magnitudes.
    let values: Vec<u16> = (0..hidden)
        .map(|col| {
            let value = ((col % 31) as i32 - 15) as f32 / 32.0;
            (value.to_bits() >> 16) as u16
        })
        .collect();
    let input = bf16_buffer(ctx.clone(), &values).unwrap();
    let layer = gdn_decoder_layer_fixture(&ctx);
    let reference =
        DeviceBuffer::<BF16>::with_capacity(ctx.clone(), (packed as usize).max(1)).unwrap();
    unsafe {
        runner
            .engine
            .operators()
            .qscb()
            .linear(
                input.matrix(1, hidden).unwrap(),
                layer.weights().in_proj.matrix(packed, hidden).unwrap(),
                reference.matrix(1, packed).unwrap(),
                runner
                    .qscb_workspace
                    .workspace(runner.config.qscb_workspace_bytes)
                    .unwrap(),
            )
            .unwrap();
        runner
            .execution()
            .unwrap()
            .1
            .execute_gdn_layer(
                &ctx,
                0,
                1,
                input.matrix(1, hidden).unwrap(),
                layer.weights(),
                ActiveRunKind::Decode,
            )
            .unwrap();
    }

    let actual = download_bf16(
        &runner.scratch.gdn.as_ref().unwrap().packed,
        &ctx,
        packed as usize,
    );
    let expected = download_bf16(&reference, &ctx, packed as usize);
    let mut max_error = 0.0_f32;
    for (index, (&actual, &expected)) in actual.iter().zip(&expected).enumerate() {
        let actual = bf16_bits_to_f32(actual);
        let expected = bf16_bits_to_f32(expected);
        let error = (actual - expected).abs();
        // Same tolerance as the real-weight QKV prototype comparison.
        assert!(
            actual.is_finite() && expected.is_finite() && error <= 2e-5 + 0.008 * expected.abs(),
            "GDN QKV output {index}: Triton {actual}, cuBLASLt {expected}"
        );
        max_error = max_error.max(error);
    }
    eprintln!("Triton GDN QKV vs cuBLASLt: {packed} outputs, max abs error {max_error}");
}

#[test]
fn qwen36_gdn_decoder_layer_vector_chains_gdn_into_moe_and_next_norm() {
    if !cuda_device_available() {
        return;
    }

    let mut runner = gdn_decoder_layer_vector_runner();
    let ctx = runner.ctx.clone();
    let rows = GDN_DECODER_LAYER_VECTOR_ROWS;
    let hidden = HIDDEN_SIZE;
    let hidden_len = gdn_decoder_layer_hidden_len();
    let gdn_out_len = checked_usize_product(&[rows, OUTPUT_WIDTH]).unwrap();
    let topk_len = gdn_decoder_layer_topk_len();
    runner.scratch.reserve(rows).unwrap();
    runner
        .upload_batch_inputs(BatchRun {
            tokens: &vec![0; rows as usize],
            start_pos: 0,
            kind: ActiveRunKind::Append,
        })
        .unwrap();

    upload!(
        &runner.ctx,
        &mut runner.scratch.norm,
        &read_gdn_decoder_bf16_vector("gdn_decoder_layer_input.bf16", hidden_len),
    )
    .unwrap();
    upload!(
        &runner.ctx,
        &mut runner.scratch.residual,
        &read_gdn_decoder_bf16_vector("gdn_decoder_input_residual.bf16", hidden_len),
    )
    .unwrap();

    let layer = gdn_decoder_layer_fixture(&ctx);
    let layer_weights = layer.weights();
    let norm_ptr = runner.scratch.norm.matrix(rows, hidden).unwrap();
    unsafe {
        runner
            .execution()
            .unwrap()
            .1
            .execute_gdn_layer(
                &ctx,
                0,
                rows,
                norm_ptr,
                layer_weights,
                ActiveRunKind::Append,
            )
            .unwrap();
    }
    ctx.synchronize().unwrap();

    assert_bf16_close_to_f32_oracle(
        "GDN decoder slice gated recurrent RMSNorm",
        &download_bf16(
            &runner.scratch.gdn.as_ref().unwrap().norm_out,
            &ctx,
            gdn_out_len,
        ),
        &read_gdn_decoder_bf16_vector(
            "gdn_decoder_expected_gated_norm_output_bf16.bf16",
            gdn_out_len,
        ),
        &read_gdn_decoder_f32_vector(
            "gdn_decoder_expected_gated_norm_output_f32.f32",
            gdn_out_len,
        ),
        BF16_GDN_DECODER_NORM_ABS_TOL,
    );
    assert_bf16_close_to_f32_oracle(
        "GDN decoder slice output projection",
        &download_bf16(&runner.scratch.attn_proj, &ctx, hidden_len),
        &read_gdn_decoder_bf16_vector("gdn_decoder_expected_output_proj_bf16.bf16", hidden_len),
        &read_gdn_decoder_f32_vector("gdn_decoder_expected_output_proj_f32.f32", hidden_len),
        BF16_GDN_DECODER_PROJ_ABS_TOL,
    );

    unsafe {
        runner
            .execution()
            .unwrap()
            .1
            .execute_post_attention_mlp(
                rows,
                &layer_weights.mlp_norm,
                &layer_weights.mlp,
                &layer.next_norm,
            )
            .unwrap();
    }
    ctx.synchronize().unwrap();

    assert_bf16_close_to_f32_oracle(
        "GDN decoder slice post-attention Gemma RMSNorm handoff",
        &download_bf16(&runner.scratch.attn_proj, &ctx, hidden_len),
        &read_gdn_decoder_bf16_vector(
            "gdn_decoder_expected_post_attn_norm_output_bf16.bf16",
            hidden_len,
        ),
        &read_gdn_decoder_f32_vector(
            "gdn_decoder_expected_post_attn_norm_output_f32.f32",
            hidden_len,
        ),
        BF16_GDN_DECODER_NORM_ABS_TOL,
    );
    assert_bf16_close_to_f32_oracle(
        "GDN decoder slice MoE router logits",
        &download_bf16(
            &moe_scratch(&runner.scratch).router_logits,
            &ctx,
            (rows * VECTOR_EXPERTS) as usize,
        ),
        &read_gdn_decoder_bf16_vector(
            "gdn_decoder_expected_moe_router_logits_bf16.bf16",
            (rows * VECTOR_EXPERTS) as usize,
        ),
        &read_gdn_decoder_f32_vector(
            "gdn_decoder_expected_moe_router_logits_f32.f32",
            (rows * VECTOR_EXPERTS) as usize,
        ),
        BF16_GDN_DECODER_PROJ_ABS_TOL,
    );
    assert_eq!(
        download_i32(&moe_scratch(&runner.scratch).topk_ids, &ctx, topk_len),
        read_gdn_decoder_i32_vector("gdn_decoder_expected_moe_topk_ids.i32", topk_len),
        "GDN decoder slice MoE router top-k ids changed"
    );
    assert_moe_router_matches_cpu(&runner, rows);
    assert_f32_close(
        "GDN decoder slice shared expert gate logits",
        &download_f32(
            &shared_scratch(&runner.scratch).gate_logits,
            &ctx,
            rows as usize,
        ),
        &read_gdn_decoder_f32_vector(
            "gdn_decoder_expected_shared_gate_logits_f32.f32",
            rows as usize,
        ),
        BF16_GDN_DECODER_PROJ_ABS_TOL,
        1.0e-4,
    );
    assert_bf16_close_to_f32_oracle(
        "GDN decoder slice shared expert output",
        &download_bf16(&shared_scratch(&runner.scratch).out, &ctx, hidden_len),
        &read_gdn_decoder_bf16_vector(
            "gdn_decoder_expected_shared_expert_output_bf16.bf16",
            hidden_len,
        ),
        &read_gdn_decoder_f32_vector(
            "gdn_decoder_expected_shared_expert_output_f32.f32",
            hidden_len,
        ),
        BF16_GDN_DECODER_MOE_ABS_TOL,
    );
    assert_bf16_close_to_f32_oracle(
        "GDN decoder slice residual after MoE/shared expert",
        &download_bf16(&runner.scratch.residual, &ctx, hidden_len),
        &read_gdn_decoder_bf16_vector(
            "gdn_decoder_expected_residual_after_mlp_bf16.bf16",
            hidden_len,
        ),
        &read_gdn_decoder_f32_vector(
            "gdn_decoder_expected_residual_after_mlp_f32.f32",
            hidden_len,
        ),
        BF16_GDN_DECODER_FINAL_NORM_ABS_TOL,
    );
    assert_bf16_close_to_f32_oracle(
        "GDN decoder slice next-layer Gemma RMSNorm",
        &download_bf16(&runner.scratch.mlp_out, &ctx, hidden_len),
        &read_gdn_decoder_bf16_vector(
            "gdn_decoder_expected_next_layer_norm_output_bf16.bf16",
            hidden_len,
        ),
        &read_gdn_decoder_f32_vector(
            "gdn_decoder_expected_next_layer_norm_output_f32.f32",
            hidden_len,
        ),
        BF16_GDN_DECODER_FINAL_NORM_ABS_TOL,
    );

    runner.commit_gdn_state();
}

#[test]
fn qwen36_full_attention_block_vector_validates_oracle_seeded_moe_shared_and_next_norm() {
    if !cuda_device_available() {
        return;
    }

    let mut runner = full_attention_block_moe_vector_runner();
    let ctx = runner.ctx.clone();
    let rows = FULL_ATTN_BLOCK_VECTOR_ROWS;
    let hidden_len = full_attention_block_hidden_len();
    runner.scratch.reserve(rows).unwrap();

    upload!(
        &runner.ctx,
        &mut runner.scratch.attn_proj,
        &read_block_bf16_vector("block_expected_post_attn_norm_output_bf16.bf16", hidden_len),
    )
    .unwrap();
    upload!(
        &runner.ctx,
        &mut runner.scratch.residual,
        &read_block_bf16_vector(
            "block_expected_residual_after_attention_bf16.bf16",
            hidden_len,
        ),
    )
    .unwrap();

    let layer = full_attention_block_moe_layer(&ctx);
    let layer_weights = layer.weights();
    let mlp = &layer_weights.mlp;
    let MlpView::Moe(view) = mlp.view(&runner.config).unwrap() else {
        panic!("expected MoE view");
    };

    unsafe {
        runner
            .execution()
            .unwrap()
            .1
            .execute_moe_mlp(rows, view)
            .unwrap();
    }
    ctx.synchronize().unwrap();

    assert_bf16_close_to_f32_oracle(
        "full-attention block MoE router logits",
        &download_bf16(
            &moe_scratch(&runner.scratch).router_logits,
            &ctx,
            (rows * VECTOR_EXPERTS) as usize,
        ),
        &read_block_bf16_vector(
            "block_expected_moe_router_logits_bf16.bf16",
            (rows * VECTOR_EXPERTS) as usize,
        ),
        &read_block_f32_vector(
            "block_expected_moe_router_logits_f32.f32",
            (rows * VECTOR_EXPERTS) as usize,
        ),
        BF16_BLOCK_PROJ_ABS_TOL,
    );
    // Projection rounding can change nearly tied ranks. Check exact routing
    // against the validated GPU logits; primitive router tests use fixed logits
    // and retain exact comparisons with the generated top-k oracle.
    assert_moe_router_matches_cpu(&runner, rows);
    assert_f32_close(
        "full-attention block shared expert gate logits",
        &download_f32(
            &shared_scratch(&runner.scratch).gate_logits,
            &ctx,
            rows as usize,
        ),
        &read_block_f32_vector("block_expected_shared_gate_logits_f32.f32", rows as usize),
        BF16_BLOCK_PROJ_ABS_TOL,
        1.0e-4,
    );
    assert_bf16_close_to_f32_oracle(
        "full-attention block shared expert output",
        &download_bf16(&shared_scratch(&runner.scratch).out, &ctx, hidden_len),
        &read_block_bf16_vector("block_expected_shared_expert_output_bf16.bf16", hidden_len),
        &read_block_f32_vector("block_expected_shared_expert_output_f32.f32", hidden_len),
        BF16_BLOCK_MOE_ABS_TOL,
    );
    assert_bf16_close_to_f32_oracle(
        "full-attention block MoE/shared expert output",
        &download_bf16(&runner.scratch.mlp_out, &ctx, hidden_len),
        &read_block_bf16_vector("block_expected_moe_shared_output_bf16.bf16", hidden_len),
        &read_block_f32_vector("block_expected_moe_shared_output_f32.f32", hidden_len),
        BF16_BLOCK_MOE_ABS_TOL,
    );

    unsafe {
        runner.engine.operators().qsfi().fused_add_rmsnorm_bf16(
            &crate::backend::qsfi::FusedAddRmsNormBf16::qwen_decoder_norm(
                runner
                    .scratch
                    .mlp_out
                    .matrix(rows, runner.config.hidden_size())
                    .unwrap(),
                runner
                    .scratch
                    .residual
                    .matrix(rows, runner.config.hidden_size())
                    .unwrap(),
                layer.next_norm.vector(runner.config.hidden_size()).unwrap(),
                runner.config.rms_norm_eps(),
            )
            .unwrap(),
        )
    }
    .unwrap();
    ctx.synchronize().unwrap();
    assert_bf16_close_to_f32_oracle(
        "full-attention block residual after MoE/shared expert",
        &download_bf16(&runner.scratch.residual, &ctx, hidden_len),
        &read_block_bf16_vector("block_expected_residual_after_mlp_bf16.bf16", hidden_len),
        &read_block_f32_vector("block_expected_residual_after_mlp_f32.f32", hidden_len),
        BF16_BLOCK_FINAL_NORM_ABS_TOL,
    );
    assert_bf16_close_to_f32_oracle(
        "full-attention block next-layer Gemma RMSNorm",
        &download_bf16(&runner.scratch.mlp_out, &ctx, hidden_len),
        &read_block_bf16_vector(
            "block_expected_next_layer_norm_output_bf16.bf16",
            hidden_len,
        ),
        &read_block_f32_vector("block_expected_next_layer_norm_output_f32.f32", hidden_len),
        BF16_BLOCK_FINAL_NORM_ABS_TOL,
    );
}

#[test]
fn qwen36_model_logits_vector_validates_public_run_logits_handoff() {
    if !cuda_device_available() {
        return;
    }

    let ctx = Rc::new(CudaCtx::default().unwrap());
    let config = model_logits_vector_config();
    let request_id = 0x4D4C_0001;
    let weights = model_logits_vector_weights(&ctx, config);
    let mut runner =
        ModelRunner::new(ctx.clone(), config, weights, config.vocab_size() as usize).unwrap();
    let prompt = read_model_logits_i32_vector("model_prompt_tokens.i32", MODEL_LOGITS_PROMPT_LEN);
    assert_eq!(prompt.len() as u32, runner.config.page_size);

    let prefill_result = runner
        .run(QwenRequest {
            request_id,
            tokens: &prompt,
            max_new_tokens: 0,
        })
        .unwrap();
    ctx.synchronize().unwrap();
    assert_eq!(prefill_result.request_id, request_id);
    assert_eq!(prefill_result.prompt_tokens, MODEL_LOGITS_PROMPT_LEN as u32);
    assert!(prefill_result.generated_tokens.is_empty());
    assert_eq!(prefill_result.live_tokens, prompt);
    assert_eq!(prefill_result.logits_rows, 1);
    assert_eq!(prefill_result.logits_vocab_size, MODEL_LOGITS_VOCAB);
    assert_eq!(runner.live_tokens(), prompt.as_slice());
    assert_eq!(runner.last_logits_rows, 1);
    assert_eq!(runner.last_logits_vocab_size, MODEL_LOGITS_VOCAB);

    let vocab = MODEL_LOGITS_VOCAB as usize;
    let prefill_logits_len = MODEL_LOGITS_PROMPT_LEN * vocab;
    let expected_prefill =
        read_model_logits_f32_vector("model_expected_prefill_logits_f32.f32", prefill_logits_len);
    assert_f32_close(
        "model logits final prefill row",
        &download_f32(&runner.scratch.logits, &ctx, vocab),
        &expected_prefill[prefill_logits_len - vocab..],
        MODEL_LOGITS_ABS_TOL,
        MODEL_LOGITS_REL_TOL,
    );
    let prefill_top = download_i32(&runner.scratch.next_token_ids, &ctx, 1);
    let expected_prefill_top = read_model_logits_i32_vector(
        "model_expected_prefill_top_ids.i32",
        MODEL_LOGITS_PROMPT_LEN,
    );
    assert_eq!(
        prefill_top,
        expected_prefill_top[MODEL_LOGITS_PROMPT_LEN - 1..],
        "final prefill greedy top id changed"
    );
    assert_eq!(
        runner.last_next_tokens, prefill_top,
        "prefill runner state top ids changed"
    );
    for margin in read_model_logits_f32_vector(
        "model_expected_prefill_top_margin_f32.f32",
        MODEL_LOGITS_PROMPT_LEN,
    ) {
        assert!(
            margin > 0.1,
            "prefill top-token margin is too small: {margin}"
        );
    }

    let decode_input = *prefill_top.last().unwrap();
    assert_eq!(
        vec![decode_input],
        read_model_logits_i32_vector("model_decode_input_token.i32", 1),
        "decode input must be the final prefill greedy token"
    );
    let decode_result = runner
        .run(QwenRequest {
            request_id,
            tokens: &prompt,
            max_new_tokens: 1,
        })
        .unwrap();
    ctx.synchronize().unwrap();
    assert_eq!(decode_result.request_id, request_id);
    assert_eq!(decode_result.prompt_tokens, MODEL_LOGITS_PROMPT_LEN as u32);
    assert_eq!(decode_result.generated_tokens, vec![decode_input]);
    assert_eq!(
        decode_result.live_tokens,
        vec![prompt[0], prompt[1], prompt[2], prompt[3], decode_input]
    );
    assert_eq!(decode_result.logits_rows, 1);
    assert_eq!(decode_result.logits_vocab_size, MODEL_LOGITS_VOCAB);
    assert_eq!(
        runner.live_tokens(),
        &[prompt[0], prompt[1], prompt[2], prompt[3], decode_input]
    );
    assert_eq!(runner.last_logits_rows, 1);
    assert_eq!(runner.last_logits_vocab_size, MODEL_LOGITS_VOCAB);

    assert_f32_close(
        "model logits decode row",
        &download_f32(&runner.scratch.logits, &ctx, MODEL_LOGITS_VOCAB as usize),
        &read_model_logits_f32_vector(
            "model_expected_decode_logits_f32.f32",
            MODEL_LOGITS_VOCAB as usize,
        ),
        MODEL_LOGITS_ABS_TOL,
        MODEL_LOGITS_REL_TOL,
    );
    let expected_decode_top = read_model_logits_i32_vector("model_expected_decode_top_ids.i32", 1);
    assert_eq!(
        download_i32(&runner.scratch.next_token_ids, &ctx, 1),
        expected_decode_top,
        "decode greedy top id changed"
    );
    assert_eq!(
        runner.last_next_tokens, expected_decode_top,
        "decode runner state top id changed"
    );
    let decode_margin =
        read_model_logits_f32_vector("model_expected_decode_top_margin_f32.f32", 1)[0];
    assert!(
        decode_margin > 0.1,
        "decode top-token margin is too small: {decode_margin}"
    );
}

#[test]
fn qwen36_moe_vectors_validate_router_topk_and_renormalized_weights() {
    if !cuda_device_available() {
        return;
    }

    let mut runner = moe_vector_runner();
    upload_moe_vector_inputs(&mut runner);
    run_moe_vector_router(&mut runner, true);

    let topk_len = (MOE_VECTOR_ROWS * VECTOR_TOP_K) as usize;
    let got_ids = download_i32(
        &moe_scratch(&runner.scratch).topk_ids,
        &runner.ctx,
        topk_len,
    );
    let got_weights = download_f32(
        &moe_scratch(&runner.scratch).topk_weights,
        &runner.ctx,
        topk_len,
    );
    assert_eq!(
        got_ids,
        read_moe_i32_vector("topk_ids.i32", topk_len),
        "router top-k ids changed"
    );
    assert_f32_close(
        "renormalized router top-k weights",
        &got_weights,
        &read_moe_f32_vector("topk_weights.f32", topk_len),
        MOE_ROUTER_WEIGHT_ABS_TOL,
        MOE_ROUTER_WEIGHT_REL_TOL,
    );
}

#[test]
fn qwen36_moe_vectors_validate_router_unrenormalized_weights() {
    if !cuda_device_available() {
        return;
    }

    let mut runner = moe_vector_runner();
    upload_moe_vector_inputs(&mut runner);
    run_moe_vector_router(&mut runner, false);

    let topk_len = (MOE_VECTOR_ROWS * VECTOR_TOP_K) as usize;
    let got_ids = download_i32(
        &moe_scratch(&runner.scratch).topk_ids,
        &runner.ctx,
        topk_len,
    );
    let got_weights = download_f32(
        &moe_scratch(&runner.scratch).topk_weights,
        &runner.ctx,
        topk_len,
    );
    assert_eq!(
        got_ids,
        read_moe_i32_vector("topk_ids.i32", topk_len),
        "router top-k ids changed with renormalization disabled"
    );
    assert_f32_close(
        "unrenormalized router top-k weights",
        &got_weights,
        &read_moe_f32_vector("topk_unrenormalized_weights.f32", topk_len),
        MOE_ROUTER_WEIGHT_ABS_TOL,
        MOE_ROUTER_WEIGHT_REL_TOL,
    );
}

#[test]
fn qwen36_moe_vectors_validate_staged_bf16_routed_output() {
    if !cuda_device_available() {
        return;
    }

    let mut runner = moe_vector_runner();
    upload_moe_vector_inputs(&mut runner);
    run_moe_vector_router(&mut runner, true);
    execute_moe_vector_routed_output(&mut runner);

    let got = download_bf16(&runner.scratch.mlp_out, &runner.ctx, MOE_VECTOR_ELEMENTS);
    assert_bf16_close_to_f32_oracle(
        "routed MoE staged BF16 vector",
        &got,
        &read_moe_bf16_vector("routed_moe_output_bf16.bf16", MOE_VECTOR_ELEMENTS),
        &read_moe_f32_vector("routed_moe_output.f32", MOE_VECTOR_ELEMENTS),
        BF16_MOE_ABS_TOL,
    );
}

#[test]
fn qwen36_moe_vectors_validate_shared_expert_gate_add_output() {
    if !cuda_device_available() {
        return;
    }

    let mut runner = moe_vector_runner();
    upload_moe_vector_inputs(&mut runner);
    run_moe_vector_router(&mut runner, true);
    execute_moe_vector_routed_output(&mut runner);
    execute_moe_vector_shared_gate_add(&mut runner);

    let got = download_bf16(&runner.scratch.mlp_out, &runner.ctx, MOE_VECTOR_ELEMENTS);
    assert_bf16_close_to_f32_oracle(
        "shared expert gate-add BF16 vector",
        &got,
        &read_moe_bf16_vector("combined_output_bf16.bf16", MOE_VECTOR_ELEMENTS),
        &read_moe_f32_vector("combined_output.f32", MOE_VECTOR_ELEMENTS),
        BF16_MOE_ABS_TOL,
    );
}

#[test]
fn randomized_full_attention_weights_seed_qwen_norm_raw_weights_as_zero() {
    if !cuda_device_available() {
        return;
    }

    let config = QwenConfig::randomized_dense_tiny_fixture();
    let ctx = Rc::new(CudaCtx::default().unwrap());
    let weights = QwenWeights::random_bf16(ctx.clone(), &config, 0x5153_3300_7177_6e6b).unwrap();
    let QwenWeights::DenseBf16(weights) = weights else {
        panic!("expected dense BF16");
    };
    let hidden = config.hidden_size() as usize;
    let head_dim = config.head_dim() as usize;

    assert_eq!(weights.final_norm.len, hidden);
    assert_eq!(
        download_bf16(&weights.final_norm, &ctx, hidden),
        vec![0_u16; hidden]
    );

    match &weights.layers[0] {
        QwenLayerWeights::AttentionMlp(layer) => {
            assert_eq!(layer.attn_norm.len, hidden);
            assert_eq!(layer.mlp_norm.len, hidden);
            assert_eq!(layer.q_norm.len, head_dim);
            assert_eq!(layer.k_norm.len, head_dim);
            assert_eq!(
                layer.q_proj.len,
                checked_usize_product(&[PACKED_Q_GATE_WIDTH, config.hidden_size()]).unwrap()
            );
            assert_eq!(
                download_bf16(&layer.attn_norm, &ctx, hidden),
                vec![0_u16; hidden]
            );
            assert_eq!(
                download_bf16(&layer.mlp_norm, &ctx, hidden),
                vec![0_u16; hidden]
            );
            assert_eq!(
                download_bf16(&layer.q_norm, &ctx, head_dim),
                vec![0_u16; head_dim]
            );
            assert_eq!(
                download_bf16(&layer.k_norm, &ctx, head_dim),
                vec![0_u16; head_dim]
            );
        }
        QwenLayerWeights::Gdn(_) => unreachable!(),
    }
}

#[test]
fn qwen36_gdn_weights_carry_post_attention_shared_moe() {
    let layer: QwenLayerWeights<FusedBf16Mlp> = QwenLayerWeights::Gdn(QwenGdnWeights {
        norm: placeholder_weight(),
        in_proj: placeholder_weight(),
        gate_proj: placeholder_weight(),
        a_proj: placeholder_weight(),
        b_proj: placeholder_weight(),
        conv_weight: placeholder_weight(),
        conv_bias: placeholder_weight(),
        a_log: placeholder_weight(),
        dt_bias: placeholder_weight(),
        rms_weight: placeholder_weight(),
        out_proj: placeholder_weight(),
        mlp_norm: placeholder_weight(),
        mlp: empty_qwen36_moe_mlp_weights(),
    });

    match &layer {
        QwenLayerWeights::Gdn(gdn) => assert!(
            gdn.mlp.shared.is_some(),
            "GDN layers must carry shared MoE post-attention weights"
        ),
        QwenLayerWeights::AttentionMlp(_) => unreachable!(),
    }
}

#[test]
fn shared_moe_execution_produces_routed_and_shared_outputs() {
    if !cuda_device_available() {
        return;
    }

    let config = QwenConfig::randomized_shared_moe_tiny_fixture();
    let ctx = Rc::new(CudaCtx::default().unwrap());
    let mut runner = ModelRunner::random_bf16(ctx.clone(), config, 0x5153_3300_5a5a_5a5a).unwrap();
    let rows = 1;
    let hidden = config.hidden_size();
    let hidden_len = hidden as usize;
    let moe = config.moe_config().unwrap();
    runner.scratch.reserve(rows).unwrap();

    let input = constant_bf16_values(hidden_len, 1.0).unwrap();
    upload!(&runner.ctx, &mut runner.scratch.attn_proj, &input).unwrap();

    let router_proj = filled_bf16_buffer(
        &ctx,
        checked_usize_product(&[moe.num_experts, hidden]).unwrap(),
        0.0,
    );
    let gate_up_proj = filled_bf16_buffer(
        &ctx,
        checked_usize_product(&[moe.num_experts, 2, moe.moe_intermediate_size, hidden]).unwrap(),
        0.003,
    );
    let down_proj = filled_bf16_buffer(
        &ctx,
        checked_usize_product(&[moe.num_experts, hidden, moe.moe_intermediate_size]).unwrap(),
        0.01,
    );

    let mut mlp: FusedBf16Mlp = MoeMlp {
        router_proj: Box::new(router_proj),
        experts: FusedExperts {
            gate_up_proj: Box::new(gate_up_proj),
            down_proj: Box::new(down_proj),
        },
        shared: None,
    };
    let MlpView::Moe(view) = mlp.view(&config).unwrap() else {
        panic!("expected MoE view");
    };
    unsafe {
        runner
            .execution()
            .unwrap()
            .1
            .execute_moe_mlp(rows, view)
            .unwrap();
    }
    let routed = download_bf16(&runner.scratch.mlp_out, &ctx, hidden_len);
    assert!(has_nonzero_bf16(&routed));

    upload!(&runner.ctx, &mut runner.scratch.attn_proj, &input).unwrap();
    let shared_gate_proj = filled_bf16_buffer(
        &ctx,
        checked_usize_product(&[moe.shared_expert_intermediate_size, hidden]).unwrap(),
        0.004,
    );
    let shared_up_proj = filled_bf16_buffer(
        &ctx,
        checked_usize_product(&[moe.shared_expert_intermediate_size, hidden]).unwrap(),
        0.004,
    );
    let shared_down_proj = filled_bf16_buffer(
        &ctx,
        checked_usize_product(&[hidden, moe.shared_expert_intermediate_size]).unwrap(),
        0.02,
    );
    let shared_expert_gate = filled_bf16_buffer(&ctx, hidden_len, 0.02);
    let shared: SharedExpert<W<BF16>> = SharedExpert {
        projections: DenseMlp {
            gate_proj: Box::new(shared_gate_proj),
            up_proj: Box::new(shared_up_proj),
            down_proj: Box::new(shared_down_proj),
        },
        gate: Box::new(shared_expert_gate),
    };

    mlp.shared = Some(shared);
    let MlpView::Moe(view) = mlp.view(&config).unwrap() else {
        panic!("expected MoE view");
    };
    unsafe {
        runner
            .execution()
            .unwrap()
            .1
            .execute_moe_mlp(rows, view)
            .unwrap();
    }

    let shared_out = download_bf16(&shared_scratch(&runner.scratch).out, &ctx, hidden_len);
    let combined = download_bf16(&runner.scratch.mlp_out, &ctx, hidden_len);
    let gate_logits = download_f32(
        &shared_scratch(&runner.scratch).gate_logits,
        &ctx,
        rows as usize,
    );

    assert!(has_nonzero_bf16(&shared_out));
    assert!(gate_logits.iter().any(|value| *value != 0.0));
    assert_ne!(combined, routed);
}

#[test]
fn qwen36_hybrid_schedule_maps_model_layers_to_attention_and_gdn_indices() {
    let config = qwen36_hybrid_fixture_with_supported_attention(8);
    assert_eq!(config.attention_layer_count(), 2);
    assert_eq!(config.gdn_layer_count(), 6);
    assert_eq!(config.layer_kind(0), QwenBlockKind::LinearAttention);
    assert_eq!(config.layer_kind(1), QwenBlockKind::LinearAttention);
    assert_eq!(config.layer_kind(2), QwenBlockKind::LinearAttention);
    assert_eq!(config.layer_kind(3), QwenBlockKind::FullAttention);
    assert_eq!(config.layer_kind(4), QwenBlockKind::LinearAttention);
    assert_eq!(config.layer_kind(7), QwenBlockKind::FullAttention);

    assert_eq!(config.gdn_layer_index(0), Ok(0));
    assert_eq!(config.gdn_layer_index(1), Ok(1));
    assert_eq!(config.gdn_layer_index(2), Ok(2));
    assert_eq!(config.gdn_layer_index(4), Ok(3));
    assert_eq!(config.gdn_layer_index(6), Ok(5));
    assert_eq!(config.attention_layer_index(3), Ok(0));
    assert_eq!(config.attention_layer_index(7), Ok(1));
    assert_eq!(config.attention_layer_index(0), Err(Status::InternalError));
    assert_eq!(config.gdn_layer_index(3), Err(Status::InternalError));

    let engine = config.engine_config();
    assert_eq!(engine.num_layers, 2);
    assert_eq!(engine.num_q_heads, config.num_q_heads());
    assert_eq!(engine.num_kv_heads, config.num_kv_heads());
    assert_eq!(engine.head_dim, config.head_dim());
    assert_eq!(config.num_q_heads(), NUM_Q_HEADS);
    assert_eq!(config.num_kv_heads(), NUM_KV_HEADS);
}

#[test]
fn gdn_slot_map_commit_is_explicit_and_uses_gdn_layer_count() {
    let mut slots = GdnSlotMap::new(3).unwrap();
    assert_eq!(slots.state_pool, 6);
    assert_eq!(
        slots.layer_slots(0).map(|s| (s.live_slot, s.staged_slot)),
        Ok((0, 1))
    );
    assert_eq!(
        slots.layer_slots(2).map(|s| (s.live_slot, s.staged_slot)),
        Ok((4, 5))
    );
    assert_eq!(slots.layer_slots(3).err(), Some(Status::InvalidArgument));

    let before = slots.layer_slots(1).unwrap();
    let without_commit = slots.layer_slots(1).unwrap();
    assert_eq!(without_commit.live_slot, before.live_slot);
    assert_eq!(without_commit.staged_slot, before.staged_slot);

    slots.commit();
    let committed = slots.layer_slots(1).unwrap();
    assert_eq!(committed.live_slot, before.staged_slot);
    assert_eq!(committed.staged_slot, before.live_slot);

    slots.reset(3).unwrap();
    let reset = slots.layer_slots(1).unwrap();
    assert_eq!(reset.live_slot, before.live_slot);
    assert_eq!(reset.staged_slot, before.staged_slot);
}

#[test]
fn sampling_respects_the_loaded_tokenizer_boundary() {
    if !cuda_device_available() {
        return;
    }
    let config = QwenConfig::randomized_shared_moe_tiny_fixture();
    let ctx = Rc::new(CudaCtx::default().unwrap());
    let mut runner = ModelRunner::random_bf16(ctx.clone(), config, 78).unwrap();
    // Exercise sampling directly with synthetic full-width logits; no model
    // execution uses these tiny weights with the larger vocabulary.
    runner.config.fixture_mut().vocab_size = 248320;
    runner.scratch.reserve(1).unwrap();
    upload!(&runner.ctx, &mut runner.scratch.positions, &[0_i32]).unwrap();
    let mut logits = vec![f32::NEG_INFINITY; 248320];
    // Include a non-pinned boundary so this tests the supplied count itself.
    for token_count in [257, 248070, 248077] {
        runner.tokenizer_token_count = token_count;
        for temperature in [0.0, 1.0] {
            runner
                .set_sampling(crate::model::SamplingParams {
                    temperature,
                    top_k: 1,
                    ..Default::default()
                })
                .unwrap();
            for token in [token_count - 1, token_count] {
                logits.fill(f32::NEG_INFINITY);
                logits[token as usize] = 1.0;
                upload!(&runner.ctx, &mut runner.scratch.logits, &logits).unwrap();
                let expected = if token < token_count {
                    Ok(vec![token as i32])
                } else {
                    Err(Status::InternalError)
                };
                assert_eq!(runner.sample_logits(0), expected);
                assert!(runner.live_tokens.is_empty());
            }
        }
    }
}

#[test]
fn runner_rejects_invalid_tokenizer_counts() {
    if !cuda_device_available() {
        return;
    }
    let ctx = Rc::new(CudaCtx::default().unwrap());
    let config = QwenConfig::randomized_shared_moe_tiny_fixture();
    for token_count in [0, config.vocab_size() as usize + 1, usize::MAX] {
        let weights = QwenWeights::random_bf16(ctx.clone(), &config, 78).unwrap();
        assert_eq!(
            ModelRunner::new(ctx.clone(), config, weights, token_count).err(),
            Some(Status::InvalidArgument)
        );
    }
}

#[test]
fn stochastic_sampling_survives_reset_release_and_failed_rebuild() {
    if !cuda_device_available() {
        return;
    }
    let config = QwenConfig::randomized_shared_moe_tiny_fixture();
    let ctx = Rc::new(CudaCtx::default().unwrap());
    let mut runner = ModelRunner::random_bf16(ctx.clone(), config, 78).unwrap();
    let params = crate::model::SamplingParams {
        temperature: 0.8,
        top_k: 8,
        top_p: 0.9,
        seed: 0xabcdef0123456789,
    };
    runner.set_sampling(params).unwrap();
    let prompt = [1, 2, 3];
    let request = QwenRequest {
        request_id: 41,
        tokens: &prompt,
        max_new_tokens: 3,
    };
    let expected = runner.run(request).unwrap();
    assert_eq!(runner.set_sampling(params), Err(Status::InvalidArgument));
    runner.reset().unwrap();
    let initial = runner
        .run(QwenRequest {
            max_new_tokens: 1,
            ..request
        })
        .unwrap();
    assert_eq!(initial.generated_tokens, expected.generated_tokens[..1]);
    runner.assert_late_rebuild_failure_preserves_prefix(41, &[7, 6, 5, 4, 3]);
    let continued = runner
        .run(QwenRequest {
            tokens: &initial.live_tokens,
            max_new_tokens: 2,
            ..request
        })
        .unwrap();
    assert_eq!(continued.generated_tokens, expected.generated_tokens[1..]);
    runner.release_requests(&[41]).unwrap();
    assert_eq!(
        runner.run(request).unwrap().generated_tokens,
        expected.generated_tokens
    );
    // Rebuild a different live prefix, then restore the original prompt.
    runner
        .run(QwenRequest {
            tokens: &[4, 3, 2],
            ..request
        })
        .unwrap();
    assert_eq!(
        runner.run(request).unwrap().generated_tokens,
        expected.generated_tokens
    );
    runner.reset().unwrap();
    runner
        .set_sampling(crate::model::SamplingParams::default())
        .unwrap();
    assert!(runner.sampler.is_none());
}

#[test]
fn failed_rebuild_after_final_layer_preserves_prefix_and_continuation() {
    if !cuda_device_available() {
        return;
    }
    let config = QwenConfig::randomized_shared_moe_tiny_fixture();
    let seed = 0x5245_4255_494c_4401;
    let prompt = [1, 2, 3];
    let request_id = 41;
    let ctx = Rc::new(CudaCtx::default().unwrap());
    let mut runner = ModelRunner::random_bf16(ctx.clone(), config, seed).unwrap();
    let initial = runner
        .run(QwenRequest {
            request_id,
            tokens: &prompt,
            max_new_tokens: 1,
        })
        .unwrap();
    runner.assert_late_rebuild_failure_preserves_prefix(request_id, &[7, 6, 5, 4, 3]);

    let continued = runner
        .run(QwenRequest {
            request_id,
            tokens: &initial.live_tokens,
            max_new_tokens: 2,
        })
        .unwrap();
    let continued_logits = runner.last_logits_row_for_test().unwrap();
    drop(runner);

    let mut control = ModelRunner::random_bf16(ctx.clone(), config, seed).unwrap();
    let expected = control
        .run(QwenRequest {
            request_id,
            tokens: &prompt,
            max_new_tokens: 3,
        })
        .unwrap();
    assert_eq!(continued.generated_tokens, expected.generated_tokens[1..]);
    assert_eq!(continued.live_tokens, expected.live_tokens);
    assert_f32_close(
        "continuation after failed prefix rebuild",
        &continued_logits,
        &control.last_logits_row_for_test().unwrap(),
        1.0e-4,
        1.0e-5,
    );
}

impl ModelRunner {
    pub(crate) fn assert_lm_head_matches_cublaslt(&mut self, rows: u32) {
        let ctx = self.ctx.clone();
        assert_eq!(self.lm_head_provider(), "triton");
        let vocab = self.config.vocab_size();
        let hidden = self.config.hidden_size();
        let reference =
            DeviceBuffer::<F32>::with_capacity(ctx.clone(), (vocab as usize).max(1)).unwrap();
        unsafe {
            self.engine
                .operators()
                .qscb()
                .linear(
                    self.scratch
                        .mlp_out
                        .matrix(rows, hidden)
                        .unwrap()
                        .row(rows - 1)
                        .unwrap(),
                    match &self.weights {
                        super::QwenWeights::DenseBf16(m) => {
                            m.lm_head.matrix(vocab, hidden).unwrap()
                        }
                        super::QwenWeights::MoeBf16(m) => m.lm_head.matrix(vocab, hidden).unwrap(),
                        _ => unreachable!(),
                    },
                    reference.matrix(1, vocab).unwrap(),
                    self.qscb_workspace
                        .workspace(self.config.qscb_workspace_bytes)
                        .unwrap(),
                )
                .unwrap();
        }
        let mut expected = HostBuffer::<F32>::new(vocab as usize).unwrap();
        unsafe {
            reference.download(&mut expected).unwrap();
        }
        self.ctx.synchronize().unwrap();
        let expected: Vec<f32> = expected
            .as_ref()
            .chunks_exact(4)
            .map(|bytes| f32::from_ne_bytes(bytes.try_into().unwrap()))
            .collect();
        let actual = self.last_logits_row_for_test().unwrap();
        let mut max_error = 0.0_f32;
        for (index, (&actual, &expected)) in actual.iter().zip(&expected).enumerate() {
            let error = (actual - expected).abs();
            assert!(
                error <= 1e-3 + 1e-5 * expected.abs(),
                "LM head logit {index}: Triton {actual}, cuBLASLt {expected}"
            );
            max_error = max_error.max(error);
        }
        eprintln!("Triton LM head vs cuBLASLt: {vocab} logits, max abs error {max_error}");
    }

    pub(crate) fn decode_forced_token_for_test(&mut self, token: i32) -> Result<(), Status> {
        let request_id = self.live_request_id.ok_or(Status::InvalidArgument)?;
        super::validate_token_ids(&[token], self.config.vocab_size())?;
        self.decode_one(request_id, token)
    }

    // Shared with the full loaded-model regression to cover GDN state rollback.
    pub(crate) fn assert_late_rebuild_failure_preserves_prefix(
        &mut self,
        request_id: crate::RequestId,
        rewritten_prompt: &[i32],
    ) {
        let live_before = self.live_tokens().to_vec();
        let logits_before = self.last_logits_row_for_test().unwrap();
        // Fail after all candidate decoder layers, before replacing logits.
        // Restore ownership before assertions or dropping the runner.
        let final_norm = match &mut self.weights {
            super::QwenWeights::DenseBf16(m) => &mut m.final_norm,
            super::QwenWeights::MoeBf16(m) => &mut m.final_norm,
            super::QwenWeights::DenseNvfp4(m) => &mut m.final_norm,
            super::QwenWeights::MoeNvfp4(m) => &mut m.final_norm,
        };
        let final_norm = std::mem::replace(final_norm, placeholder_weight());
        let failed = self.run(QwenRequest {
            request_id,
            tokens: rewritten_prompt,
            max_new_tokens: 0,
        });
        match &mut self.weights {
            super::QwenWeights::DenseBf16(m) => m.final_norm = final_norm,
            super::QwenWeights::MoeBf16(m) => m.final_norm = final_norm,
            super::QwenWeights::DenseNvfp4(m) => m.final_norm = final_norm,
            super::QwenWeights::MoeNvfp4(m) => m.final_norm = final_norm,
        };
        assert_eq!(failed.unwrap_err(), Status::InvalidArgument);
        assert_eq!(self.live_tokens(), live_before);
        assert_eq!(self.last_logits_row_for_test().unwrap(), logits_before);
    }
}

#[test]
fn host_buffer_requires_whole_dtype_storage() {
    use crate::dtype::{DType, DynDType, Nvfp4E2M1};

    assert!(F32::len_of(3).is_err());
    assert!(HostBuffer::<Nvfp4E2M1>::new(3).is_err());
    assert_eq!(HostBuffer::<Nvfp4E2M1>::new(6).unwrap().as_ref().len(), 3);
    assert_eq!(HostBuffer::<F32>::new(3).unwrap().len(), 3);
    for len in [0, 1, 3] {
        let mut host = HostBuffer::<F32>::new(len).unwrap();
        assert_eq!(host.len(), len);
        assert_eq!(host.is_empty(), len == 0);
        assert!((host.as_ref().as_ptr() as usize).is_multiple_of(F32::ALIGN));
        assert_eq!(host.as_ref(), vec![0; len * 4]);
        host.as_mut().fill(0xab);
        assert_eq!(host.as_ref(), vec![0xab; len * 4]);
    }
    assert_eq!(
        Nvfp4E2M1::size_of(3),
        DynDType::NVFP4E2M1.storage_bytes_for(3)
    );
    assert!(F32::size_of(usize::MAX).is_err());
}

#[test]
fn host_buffer_download_ranges_use_element_offsets_and_zero_full_storage() {
    if !cuda_device_available() {
        return;
    }
    let ctx = Rc::new(CudaCtx::default().unwrap());
    let mut host = HostBuffer::<F32>::new(4).unwrap();
    for (bytes, value) in host
        .as_mut()
        .chunks_exact_mut(4)
        .zip([1.5_f32, -2.0, 3.25, 4.5])
    {
        bytes.copy_from_slice(&value.to_ne_bytes());
    }
    let mut device = host.upload(ctx.clone()).unwrap();
    let mut row = HostBuffer::<F32>::new(2).unwrap();
    unsafe {
        device.download_range(1, &mut row).unwrap();
    }
    ctx.synchronize().unwrap();
    assert_eq!(
        row.as_ref(),
        [-2.0_f32, 3.25]
            .into_iter()
            .flat_map(f32::to_ne_bytes)
            .collect::<Vec<_>>()
    );
    assert_eq!(
        unsafe { device.download_range(3, &mut row) },
        Err(Status::InvalidArgument)
    );
    assert_eq!(
        unsafe { device.download_range(usize::MAX, &mut row) },
        Err(Status::InvalidArgument)
    );
    device.zero().unwrap();
    let mut all = HostBuffer::<F32>::new(4).unwrap();
    unsafe {
        device.download(&mut all).unwrap();
    }
    ctx.synchronize().unwrap();
    assert_eq!(all.as_ref(), &[0; 16]);
}

#[test]
fn packed_host_buffer_growth_preserves_bytes_and_rejects_half_byte_offsets() {
    use crate::dtype::Nvfp4E2M1;

    if !cuda_device_available() {
        return;
    }
    let ctx = Rc::new(CudaCtx::default().unwrap());
    let mut host = HostBuffer::<Nvfp4E2M1>::new(4).unwrap();
    host.as_mut().copy_from_slice(&[0x12, 0x34]);
    let mut device = host.upload(ctx.clone()).unwrap();
    device.realloc(8).unwrap();
    assert_eq!(device.len(), 8);
    let mut pair = HostBuffer::<Nvfp4E2M1>::new(2).unwrap();
    assert_eq!(
        unsafe { device.download_range(1, &mut pair) },
        Err(Status::InvalidArgument)
    );
    unsafe {
        device.download_range(2, &mut pair).unwrap();
    }
    ctx.synchronize().unwrap();
    assert_eq!(pair.as_ref(), &[0x34]);
    let mut prefix = HostBuffer::<Nvfp4E2M1>::new(2).unwrap();
    prefix.as_mut().copy_from_slice(&[0xab]);
    unsafe {
        device.upload(&prefix).unwrap();
    }
    let mut first = HostBuffer::<Nvfp4E2M1>::new(4).unwrap();
    unsafe {
        device.download(&mut first).unwrap();
    }
    ctx.synchronize().unwrap();
    assert_eq!(first.as_ref(), &[0xab, 0x34]);
    device.zero().unwrap();
    let mut all = HostBuffer::<Nvfp4E2M1>::new(8).unwrap();
    unsafe {
        device.download(&mut all).unwrap();
    }
    ctx.synchronize().unwrap();
    assert_eq!(all.as_ref(), &[0; 4]);
}
