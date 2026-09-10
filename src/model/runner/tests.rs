use super::{ModelRunner, QwenRequest};
use crate::{
    QWEN36_FULL_ATTN_HEAD_DIM, QWEN36_FULL_ATTN_KV_HEADS, QWEN36_FULL_ATTN_KV_HIDDEN,
    QWEN36_FULL_ATTN_Q_HEADS, QWEN36_FULL_ATTN_Q_HIDDEN, QWEN36_FULL_ATTN_Q_PROJ_OUT,
    QWEN36_GDN_CONV_WIDTH, QWEN36_GDN_NUM_V_HEADS, QWEN36_GDN_OUTPUT_DIM, QWEN36_GDN_PACKED_DIM,
    QWEN36_GDN_VALUE_DIM, QWEN36_HIDDEN_SIZE, QWEN36_MOE_INTERMEDIATE_SIZE, QWEN36_MOE_NUM_EXPERTS,
    QWEN36_MOE_ROUTER_SCALING_FACTOR, QWEN36_MOE_ROUTER_SCORE,
    QWEN36_MOE_SHARED_EXPERT_INTERMEDIATE_SIZE, QWEN36_MOE_TOP_K,
    backend::{
        DMat, DTensor3,
        qsfi::{MoeBf16Execute, MoeBf16ExecuteArgs, MoeBf16PlanConfig, Workspace},
    },
    engine::{AppendBatch, AttentionLayer, Commit, Engine, Status},
    ffi::cuda,
    model::{
        ActiveRunKind, QwenBlockKind, QwenConfig, QwenMoeConfig, QwenWeights,
        checked_usize_product,
        config::{QwenGdnShape, QwenLayerPattern, QwenModelShape},
        constant_bf16_values,
        scratch::{DeviceBuffer, RunnerScratch},
        state::{GdnSlotMap, GdnState},
        synchronize_stream,
        weights::{
            QwenAttentionMlpWeights, QwenGdnWeights, QwenLayerWeights, QwenMlpWeights,
            QwenSharedExpertWeights,
        },
    },
};
use std::{ffi::c_void, mem, path::PathBuf, ptr};

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
    let mut config = QwenConfig::randomized_qwen36_moe_gdn_one_block_fixture(-1);
    config.num_layers = num_layers;
    config
}

fn empty_bf16_buffer() -> DeviceBuffer<u16> {
    DeviceBuffer::empty(-1)
}

fn empty_shared_expert_weights() -> QwenSharedExpertWeights {
    QwenSharedExpertWeights {
        gate_proj: empty_bf16_buffer(),
        up_proj: empty_bf16_buffer(),
        down_proj: empty_bf16_buffer(),
        shared_expert_gate: empty_bf16_buffer(),
    }
}

fn empty_qwen36_moe_mlp_weights() -> QwenMlpWeights {
    QwenMlpWeights::Moe {
        router_proj: empty_bf16_buffer(),
        gate_up_proj: empty_bf16_buffer(),
        down_proj: empty_bf16_buffer(),
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

fn filled_bf16_buffer(
    device_ordinal: i32,
    stream: *mut c_void,
    len: usize,
    value: f32,
) -> DeviceBuffer<u16> {
    DeviceBuffer::from_slice(
        device_ordinal,
        stream,
        &constant_bf16_values(len, value).unwrap(),
    )
    .unwrap()
}

fn download_bf16(buffer: &DeviceBuffer<u16>, stream: *mut c_void, len: usize) -> Vec<u16> {
    let mut values = vec![0_u16; len];
    buffer.download(stream, &mut values).unwrap();
    values
}

fn download_i32(buffer: &DeviceBuffer<i32>, stream: *mut c_void, len: usize) -> Vec<i32> {
    let mut values = vec![0_i32; len];
    buffer.download(stream, &mut values).unwrap();
    values
}

fn download_f32(buffer: &DeviceBuffer<f32>, stream: *mut c_void, len: usize) -> Vec<f32> {
    let mut values = vec![0.0_f32; len];
    buffer.download(stream, &mut values).unwrap();
    values
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
        &runner.scratch.router_logits,
        runner.config.stream,
        rows as usize * experts,
    );
    let ids = download_i32(
        &runner.scratch.topk_ids,
        runner.config.stream,
        rows as usize * topk,
    );
    let weights = download_f32(
        &runner.scratch.topk_weights,
        runner.config.stream,
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
    let mut config = QwenConfig::randomized_dense_tiny_fixture(0);
    config.num_layers = 1;
    config
}

fn empty_weights_for_config(config: QwenConfig) -> QwenWeights {
    QwenWeights {
        config,
        token_embedding: empty_bf16_buffer(),
        final_norm: empty_bf16_buffer(),
        lm_head: empty_bf16_buffer(),
        layers: Vec::new(),
    }
}

fn moe_vector_config() -> QwenConfig {
    let mut config = QwenConfig::randomized_dense_tiny_fixture(0);
    config.num_layers = 1;
    config.max_seq_len = MOE_VECTOR_ROWS;
    config.max_pages = 2;
    config.page_size = 4;
    config.hidden_size = MOE_VECTOR_HIDDEN;
    config.intermediate_size = MOE_VECTOR_INTERMEDIATE;
    config.moe = Some(QwenMoeConfig {
        num_experts: QWEN36_MOE_NUM_EXPERTS,
        num_experts_per_tok: QWEN36_MOE_TOP_K,
        moe_intermediate_size: MOE_VECTOR_INTERMEDIATE,
        shared_expert_intermediate_size: MOE_VECTOR_INTERMEDIATE,
    });
    config.vocab_size = 16;
    config
}

fn moe_vector_runner() -> ModelRunner {
    let config = moe_vector_config();
    let mut engine = Engine::new(config.engine_config()).unwrap();
    let moe = config.moe_config().unwrap();
    let (moe_plan, workspace_bytes) = {
        let mut ops = engine.operators();
        let plan = unsafe {
            ops.qsfi()
                .create_moe_bf16_plan(MoeBf16PlanConfig {
                    kernel: config.moe_bf16_kernel,
                    max_num_tokens: config.max_seq_len,
                    hidden_size: config.hidden_size,
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
    let mut scratch = RunnerScratch::new(config.device_ordinal);
    scratch.ensure_moe_workspace(workspace_bytes).unwrap();
    let mut qscb_workspace = DeviceBuffer::empty(config.device_ordinal);
    qscb_workspace.ensure(config.qscb_workspace_bytes).unwrap();

    ModelRunner {
        lm_head: None,
        config,
        weights: empty_weights_for_config(config),
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
    runner
        .scratch
        .ensure(&runner.config, MOE_VECTOR_ROWS)
        .unwrap();
    runner
        .scratch
        .attn_proj
        .upload(
            runner.config.stream,
            &read_moe_bf16_vector("hidden.bf16", MOE_VECTOR_ELEMENTS),
        )
        .unwrap();
    runner
        .scratch
        .router_logits
        .upload(
            runner.config.stream,
            &read_moe_bf16_vector(
                "router_logits.bf16",
                (MOE_VECTOR_ROWS * QWEN36_MOE_NUM_EXPERTS) as usize,
            ),
        )
        .unwrap();
}

fn run_moe_vector_router(runner: &mut ModelRunner, renormalize: bool) {
    let moe = runner.config.moe_config().unwrap();
    let router_logits = DMat::contiguous(
        runner.scratch.router_logits.as_device_ptr(),
        MOE_VECTOR_ROWS,
        moe.num_experts,
    )
    .unwrap();
    let topk_ids = DMat::contiguous(
        runner.scratch.topk_ids.as_device_ptr(),
        MOE_VECTOR_ROWS,
        moe.num_experts_per_tok,
    )
    .unwrap();
    let topk_weights = DMat::contiguous(
        runner.scratch.topk_weights.as_device_ptr(),
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
    let moe = runner.config.moe_config().unwrap();
    let gate_up_weight = DeviceBuffer::from_slice(
        runner.config.device_ordinal,
        runner.config.stream,
        &read_moe_bf16_vector(
            "gate_up_weight.bf16",
            (QWEN36_MOE_NUM_EXPERTS * 2 * MOE_VECTOR_INTERMEDIATE * MOE_VECTOR_HIDDEN) as usize,
        ),
    )
    .unwrap();
    let down_weight = DeviceBuffer::from_slice(
        runner.config.device_ordinal,
        runner.config.stream,
        &read_moe_bf16_vector(
            "down_weight.bf16",
            (QWEN36_MOE_NUM_EXPERTS * MOE_VECTOR_HIDDEN * MOE_VECTOR_INTERMEDIATE) as usize,
        ),
    )
    .unwrap();
    let execute = MoeBf16Execute::new(MoeBf16ExecuteArgs {
        hidden: DMat::contiguous(
            runner.scratch.attn_proj.as_device_ptr(),
            MOE_VECTOR_ROWS,
            MOE_VECTOR_HIDDEN,
        )
        .unwrap(),
        topk_ids: DMat::contiguous(
            runner.scratch.topk_ids.as_device_ptr(),
            MOE_VECTOR_ROWS,
            moe.num_experts_per_tok,
        )
        .unwrap(),
        topk_weights: DMat::contiguous(
            runner.scratch.topk_weights.as_device_ptr(),
            MOE_VECTOR_ROWS,
            moe.num_experts_per_tok,
        )
        .unwrap(),
        gate_up_weight: DTensor3::contiguous(
            gate_up_weight.as_device_ptr(),
            moe.num_experts,
            2 * MOE_VECTOR_INTERMEDIATE,
            MOE_VECTOR_HIDDEN,
        )
        .unwrap(),
        down_weight: DTensor3::contiguous(
            down_weight.as_device_ptr(),
            moe.num_experts,
            MOE_VECTOR_HIDDEN,
            MOE_VECTOR_INTERMEDIATE,
        )
        .unwrap(),
        out: DMat::contiguous(
            runner.scratch.mlp_out.as_device_ptr(),
            MOE_VECTOR_ROWS,
            MOE_VECTOR_HIDDEN,
        )
        .unwrap(),
        workspace: Workspace::new(
            runner.scratch.moe_workspace.as_device_ptr(),
            runner.scratch.moe_workspace.cap,
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
    synchronize_stream(runner.config.stream).unwrap();
}

fn execute_moe_vector_shared_gate_add(runner: &mut ModelRunner) {
    let shared_gate_up_weight = DeviceBuffer::from_slice(
        runner.config.device_ordinal,
        runner.config.stream,
        &read_moe_bf16_vector(
            "shared_gate_up_weight.bf16",
            (2 * MOE_VECTOR_INTERMEDIATE * MOE_VECTOR_HIDDEN) as usize,
        ),
    )
    .unwrap();
    let shared_down_weight = DeviceBuffer::from_slice(
        runner.config.device_ordinal,
        runner.config.stream,
        &read_moe_bf16_vector(
            "shared_down_weight.bf16",
            (MOE_VECTOR_HIDDEN * MOE_VECTOR_INTERMEDIATE) as usize,
        ),
    )
    .unwrap();
    let shared_up_proj = shared_gate_up_weight
        .matrix_at(
            (MOE_VECTOR_INTERMEDIATE * MOE_VECTOR_HIDDEN) as usize,
            MOE_VECTOR_INTERMEDIATE,
            MOE_VECTOR_HIDDEN,
        )
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
            runner
                .scratch
                .shared_gate
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
            runner
                .scratch
                .shared_up
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
            runner
                .scratch
                .shared_gate
                .matrix(MOE_VECTOR_ROWS, MOE_VECTOR_INTERMEDIATE)
                .unwrap(),
            runner
                .scratch
                .shared_up
                .matrix(MOE_VECTOR_ROWS, MOE_VECTOR_INTERMEDIATE)
                .unwrap(),
            runner
                .scratch
                .shared_mlp
                .matrix(MOE_VECTOR_ROWS, MOE_VECTOR_INTERMEDIATE)
                .unwrap(),
        )
    }
    .unwrap();
    unsafe {
        runner.engine.operators().qscb().linear(
            runner
                .scratch
                .shared_mlp
                .matrix(MOE_VECTOR_ROWS, MOE_VECTOR_INTERMEDIATE)
                .unwrap(),
            shared_down_weight
                .matrix(MOE_VECTOR_HIDDEN, MOE_VECTOR_INTERMEDIATE)
                .unwrap(),
            runner
                .scratch
                .shared_out
                .matrix(MOE_VECTOR_ROWS, MOE_VECTOR_HIDDEN)
                .unwrap(),
            runner
                .qscb_workspace
                .workspace(runner.config.qscb_workspace_bytes)
                .unwrap(),
        )
    }
    .unwrap();
    runner
        .scratch
        .shared_gate_logits
        .upload(
            runner.config.stream,
            &read_moe_f32_vector("shared_gate_logits.f32", MOE_VECTOR_ROWS as usize),
        )
        .unwrap();
    unsafe {
        runner
            .engine
            .operators()
            .qscu()
            .qwen36_shared_expert_gate_add_bf16(
                runner
                    .scratch
                    .shared_gate_logits
                    .matrix(MOE_VECTOR_ROWS, 1)
                    .unwrap(),
                runner
                    .scratch
                    .shared_out
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
    synchronize_stream(runner.config.stream).unwrap();
}

fn full_attention_vector_runner() -> ModelRunner {
    let mut config = full_attention_vector_config();
    config.qscb_workspace_bytes = 0; // This fixture runs only attention prep kernels.
    let engine = Engine::new(config.engine_config()).unwrap();
    ModelRunner {
        lm_head: None,
        config,
        weights: empty_weights_for_config(config),
        engine,
        moe_plan: None,
        gdn_state: None,
        scratch: RunnerScratch::new(config.device_ordinal),
        qscb_workspace: DeviceBuffer::empty(config.device_ordinal),
        live_request_id: None,
        live_tokens: Vec::new(),
        last_next_tokens: Vec::new(),
        last_logits_rows: 0,
        last_logits_vocab_size: 0,
    }
}

fn full_attention_block_vector_runner() -> ModelRunner {
    let mut config = full_attention_vector_config();
    config.qscb_workspace_bytes = 0; // This fixture runs only attention prep kernels.
    let engine = Engine::new(config.engine_config()).unwrap();
    let mut qscb_workspace = DeviceBuffer::empty(config.device_ordinal);
    qscb_workspace.ensure(config.qscb_workspace_bytes).unwrap();
    ModelRunner {
        lm_head: None,
        config,
        weights: empty_weights_for_config(config),
        engine,
        moe_plan: None,
        gdn_state: None,
        scratch: RunnerScratch::new(config.device_ordinal),
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
    config.intermediate_size = FULL_ATTN_BLOCK_MOE_INTERMEDIATE;
    config.moe = Some(QwenMoeConfig {
        num_experts: QWEN36_MOE_NUM_EXPERTS,
        num_experts_per_tok: QWEN36_MOE_TOP_K,
        moe_intermediate_size: FULL_ATTN_BLOCK_MOE_INTERMEDIATE,
        shared_expert_intermediate_size: FULL_ATTN_BLOCK_MOE_INTERMEDIATE,
    });
    config
}

fn full_attention_block_moe_vector_runner() -> ModelRunner {
    let config = full_attention_block_moe_vector_config();
    let mut engine = Engine::new(config.engine_config()).unwrap();
    let moe = config.moe_config().unwrap();
    let (moe_plan, workspace_bytes) = {
        let mut ops = engine.operators();
        let plan = unsafe {
            ops.qsfi()
                .create_moe_bf16_plan(MoeBf16PlanConfig {
                    kernel: config.moe_bf16_kernel,
                    max_num_tokens: config.max_seq_len,
                    hidden_size: config.hidden_size,
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
    let mut scratch = RunnerScratch::new(config.device_ordinal);
    scratch.ensure_moe_workspace(workspace_bytes).unwrap();
    let mut qscb_workspace = DeviceBuffer::empty(config.device_ordinal);
    qscb_workspace.ensure(config.qscb_workspace_bytes).unwrap();
    ModelRunner {
        lm_head: None,
        config,
        weights: empty_weights_for_config(config),
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
    layer: QwenLayerWeights,
    next_norm: DeviceBuffer<u16>,
}

impl FullAttentionBlockMoeLayer {
    fn weights(&self) -> &QwenAttentionMlpWeights {
        match &self.layer {
            QwenLayerWeights::AttentionMlp(layer) => layer,
            QwenLayerWeights::Gdn(_) => unreachable!("full-attention block fixture is not GDN"),
        }
    }
}

fn full_attention_block_moe_layer(device: i32, stream: *mut c_void) -> FullAttentionBlockMoeLayer {
    let hidden = QWEN36_HIDDEN_SIZE;
    let q_hidden = QWEN36_FULL_ATTN_Q_HIDDEN;
    let kv_hidden = QWEN36_FULL_ATTN_KV_HIDDEN;
    let intermediate = FULL_ATTN_BLOCK_MOE_INTERMEDIATE;
    let layer = QwenLayerWeights::AttentionMlp(QwenAttentionMlpWeights {
        attn_norm: DeviceBuffer::from_slice(
            device,
            stream,
            &read_block_bf16_vector("block_attn_norm_raw_weight.bf16", hidden as usize),
        )
        .unwrap(),
        q_norm: DeviceBuffer::from_slice(
            device,
            stream,
            &read_block_bf16_vector(
                "block_q_norm_raw_weight.bf16",
                QWEN36_FULL_ATTN_HEAD_DIM as usize,
            ),
        )
        .unwrap(),
        k_norm: DeviceBuffer::from_slice(
            device,
            stream,
            &read_block_bf16_vector(
                "block_k_norm_raw_weight.bf16",
                QWEN36_FULL_ATTN_HEAD_DIM as usize,
            ),
        )
        .unwrap(),
        q_proj: DeviceBuffer::from_slice(
            device,
            stream,
            &read_block_bf16_vector(
                "block_q_proj_weight.bf16",
                checked_usize_product(&[QWEN36_FULL_ATTN_Q_PROJ_OUT, hidden]).unwrap(),
            ),
        )
        .unwrap(),
        k_proj: DeviceBuffer::from_slice(
            device,
            stream,
            &read_block_bf16_vector(
                "block_k_proj_weight.bf16",
                checked_usize_product(&[kv_hidden, hidden]).unwrap(),
            ),
        )
        .unwrap(),
        v_proj: DeviceBuffer::from_slice(
            device,
            stream,
            &read_block_bf16_vector(
                "block_v_proj_weight.bf16",
                checked_usize_product(&[kv_hidden, hidden]).unwrap(),
            ),
        )
        .unwrap(),
        o_proj: DeviceBuffer::from_slice(
            device,
            stream,
            &read_block_bf16_vector(
                "block_o_proj_weight.bf16",
                checked_usize_product(&[hidden, q_hidden]).unwrap(),
            ),
        )
        .unwrap(),
        mlp_norm: DeviceBuffer::from_slice(
            device,
            stream,
            &read_block_bf16_vector("block_post_attn_norm_raw_weight.bf16", hidden as usize),
        )
        .unwrap(),
        mlp: QwenMlpWeights::Moe {
            router_proj: DeviceBuffer::from_slice(
                device,
                stream,
                &read_block_bf16_vector(
                    "block_moe_router_proj_weight.bf16",
                    checked_usize_product(&[QWEN36_MOE_NUM_EXPERTS, hidden]).unwrap(),
                ),
            )
            .unwrap(),
            gate_up_proj: DeviceBuffer::from_slice(
                device,
                stream,
                &read_block_bf16_vector(
                    "block_moe_gate_up_proj_weight.bf16",
                    checked_usize_product(&[QWEN36_MOE_NUM_EXPERTS, 2, intermediate, hidden])
                        .unwrap(),
                ),
            )
            .unwrap(),
            down_proj: DeviceBuffer::from_slice(
                device,
                stream,
                &read_block_bf16_vector(
                    "block_moe_down_proj_weight.bf16",
                    checked_usize_product(&[QWEN36_MOE_NUM_EXPERTS, hidden, intermediate]).unwrap(),
                ),
            )
            .unwrap(),
            shared: Some(QwenSharedExpertWeights {
                gate_proj: DeviceBuffer::from_slice(
                    device,
                    stream,
                    &read_block_bf16_vector(
                        "block_moe_shared_gate_proj_weight.bf16",
                        checked_usize_product(&[intermediate, hidden]).unwrap(),
                    ),
                )
                .unwrap(),
                up_proj: DeviceBuffer::from_slice(
                    device,
                    stream,
                    &read_block_bf16_vector(
                        "block_moe_shared_up_proj_weight.bf16",
                        checked_usize_product(&[intermediate, hidden]).unwrap(),
                    ),
                )
                .unwrap(),
                down_proj: DeviceBuffer::from_slice(
                    device,
                    stream,
                    &read_block_bf16_vector(
                        "block_moe_shared_down_proj_weight.bf16",
                        checked_usize_product(&[hidden, intermediate]).unwrap(),
                    ),
                )
                .unwrap(),
                shared_expert_gate: DeviceBuffer::from_slice(
                    device,
                    stream,
                    &read_block_bf16_vector(
                        "block_moe_shared_expert_gate_weight.bf16",
                        hidden as usize,
                    ),
                )
                .unwrap(),
            }),
        },
    });
    let next_norm = DeviceBuffer::from_slice(
        device,
        stream,
        &read_block_bf16_vector("block_next_layer_norm_raw_weight.bf16", hidden as usize),
    )
    .unwrap();
    FullAttentionBlockMoeLayer { layer, next_norm }
}

fn gdn_decoder_layer_vector_config() -> QwenConfig {
    let mut config = QwenConfig::randomized_qwen36_moe_gdn_one_block_fixture(0);
    config.max_seq_len = GDN_DECODER_LAYER_VECTOR_ROWS;
    config.max_pages = 2;
    config.page_size = 4;
    config.intermediate_size = GDN_DECODER_LAYER_MOE_INTERMEDIATE;
    config.moe = Some(QwenMoeConfig {
        num_experts: QWEN36_MOE_NUM_EXPERTS,
        num_experts_per_tok: QWEN36_MOE_TOP_K,
        moe_intermediate_size: GDN_DECODER_LAYER_MOE_INTERMEDIATE,
        shared_expert_intermediate_size: GDN_DECODER_LAYER_MOE_INTERMEDIATE,
    });
    config.vocab_size = 16;
    config
}

fn gdn_decoder_layer_vector_runner() -> ModelRunner {
    let config = gdn_decoder_layer_vector_config();
    let mut engine = Engine::new(config.engine_config()).unwrap();
    let moe = config.moe_config().unwrap();
    let (moe_plan, workspace_bytes) = {
        let mut ops = engine.operators();
        let plan = unsafe {
            ops.qsfi()
                .create_moe_bf16_plan(MoeBf16PlanConfig {
                    kernel: config.moe_bf16_kernel,
                    max_num_tokens: config.max_seq_len,
                    hidden_size: config.hidden_size,
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
    let mut scratch = RunnerScratch::new(config.device_ordinal);
    scratch.ensure_moe_workspace(workspace_bytes).unwrap();
    let mut qscb_workspace = DeviceBuffer::empty(config.device_ordinal);
    qscb_workspace.ensure(config.qscb_workspace_bytes).unwrap();
    let gdn_state = Some(GdnState::new(&config).unwrap());

    ModelRunner {
        lm_head: None,
        config,
        weights: empty_weights_for_config(config),
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
    layer: QwenLayerWeights,
    next_norm: DeviceBuffer<u16>,
}

impl GdnDecoderLayerFixture {
    fn weights(&self) -> &QwenGdnWeights {
        match &self.layer {
            QwenLayerWeights::Gdn(layer) => layer,
            QwenLayerWeights::AttentionMlp(_) => {
                unreachable!("GDN decoder-layer fixture is not full attention")
            }
        }
    }
}

fn gdn_decoder_layer_fixture(device: i32, stream: *mut c_void) -> GdnDecoderLayerFixture {
    let hidden = QWEN36_HIDDEN_SIZE;
    let intermediate = GDN_DECODER_LAYER_MOE_INTERMEDIATE;
    let layer = QwenLayerWeights::Gdn(QwenGdnWeights {
        norm: filled_bf16_buffer(device, stream, hidden as usize, 0.0),
        in_proj: DeviceBuffer::from_slice(
            device,
            stream,
            &read_gdn_decoder_bf16_vector(
                "gdn_decoder_in_proj_weight.bf16",
                checked_usize_product(&[QWEN36_GDN_PACKED_DIM, hidden]).unwrap(),
            ),
        )
        .unwrap(),
        gate_proj: DeviceBuffer::from_slice(
            device,
            stream,
            &read_gdn_decoder_bf16_vector(
                "gdn_decoder_gate_proj_weight.bf16",
                checked_usize_product(&[QWEN36_GDN_OUTPUT_DIM, hidden]).unwrap(),
            ),
        )
        .unwrap(),
        a_proj: DeviceBuffer::from_slice(
            device,
            stream,
            &read_gdn_decoder_bf16_vector(
                "gdn_decoder_a_proj_weight.bf16",
                checked_usize_product(&[QWEN36_GDN_NUM_V_HEADS, hidden]).unwrap(),
            ),
        )
        .unwrap(),
        b_proj: DeviceBuffer::from_slice(
            device,
            stream,
            &read_gdn_decoder_bf16_vector(
                "gdn_decoder_b_proj_weight.bf16",
                checked_usize_product(&[QWEN36_GDN_NUM_V_HEADS, hidden]).unwrap(),
            ),
        )
        .unwrap(),
        conv_weight: DeviceBuffer::from_slice(
            device,
            stream,
            &read_gdn_decoder_bf16_vector(
                "gdn_decoder_conv_weight.bf16",
                checked_usize_product(&[QWEN36_GDN_PACKED_DIM, QWEN36_GDN_CONV_WIDTH]).unwrap(),
            ),
        )
        .unwrap(),
        conv_bias: DeviceBuffer::from_slice(
            device,
            stream,
            &read_gdn_decoder_bf16_vector(
                "gdn_decoder_conv_bias.bf16",
                QWEN36_GDN_PACKED_DIM as usize,
            ),
        )
        .unwrap(),
        a_log: DeviceBuffer::from_slice(
            device,
            stream,
            &read_gdn_decoder_bf16_vector(
                "gdn_decoder_A_log.bf16",
                QWEN36_GDN_NUM_V_HEADS as usize,
            ),
        )
        .unwrap(),
        dt_bias: DeviceBuffer::from_slice(
            device,
            stream,
            &read_gdn_decoder_bf16_vector(
                "gdn_decoder_dt_bias.bf16",
                QWEN36_GDN_NUM_V_HEADS as usize,
            ),
        )
        .unwrap(),
        rms_weight: DeviceBuffer::from_slice(
            device,
            stream,
            &read_gdn_decoder_bf16_vector(
                "gdn_decoder_rms_weight.bf16",
                QWEN36_GDN_VALUE_DIM as usize,
            ),
        )
        .unwrap(),
        out_proj: DeviceBuffer::from_slice(
            device,
            stream,
            &read_gdn_decoder_bf16_vector(
                "gdn_decoder_out_proj_weight.bf16",
                checked_usize_product(&[hidden, QWEN36_GDN_OUTPUT_DIM]).unwrap(),
            ),
        )
        .unwrap(),
        mlp_norm: DeviceBuffer::from_slice(
            device,
            stream,
            &read_gdn_decoder_bf16_vector("gdn_decoder_mlp_norm_raw_weight.bf16", hidden as usize),
        )
        .unwrap(),
        mlp: QwenMlpWeights::Moe {
            router_proj: DeviceBuffer::from_slice(
                device,
                stream,
                &read_gdn_decoder_bf16_vector(
                    "gdn_decoder_moe_router_proj_weight.bf16",
                    checked_usize_product(&[QWEN36_MOE_NUM_EXPERTS, hidden]).unwrap(),
                ),
            )
            .unwrap(),
            gate_up_proj: DeviceBuffer::from_slice(
                device,
                stream,
                &read_gdn_decoder_bf16_vector(
                    "gdn_decoder_moe_gate_up_proj_weight.bf16",
                    checked_usize_product(&[QWEN36_MOE_NUM_EXPERTS, 2, intermediate, hidden])
                        .unwrap(),
                ),
            )
            .unwrap(),
            down_proj: DeviceBuffer::from_slice(
                device,
                stream,
                &read_gdn_decoder_bf16_vector(
                    "gdn_decoder_moe_down_proj_weight.bf16",
                    checked_usize_product(&[QWEN36_MOE_NUM_EXPERTS, hidden, intermediate]).unwrap(),
                ),
            )
            .unwrap(),
            shared: Some(QwenSharedExpertWeights {
                gate_proj: DeviceBuffer::from_slice(
                    device,
                    stream,
                    &read_gdn_decoder_bf16_vector(
                        "gdn_decoder_moe_shared_gate_proj_weight.bf16",
                        checked_usize_product(&[intermediate, hidden]).unwrap(),
                    ),
                )
                .unwrap(),
                up_proj: DeviceBuffer::from_slice(
                    device,
                    stream,
                    &read_gdn_decoder_bf16_vector(
                        "gdn_decoder_moe_shared_up_proj_weight.bf16",
                        checked_usize_product(&[intermediate, hidden]).unwrap(),
                    ),
                )
                .unwrap(),
                down_proj: DeviceBuffer::from_slice(
                    device,
                    stream,
                    &read_gdn_decoder_bf16_vector(
                        "gdn_decoder_moe_shared_down_proj_weight.bf16",
                        checked_usize_product(&[hidden, intermediate]).unwrap(),
                    ),
                )
                .unwrap(),
                shared_expert_gate: DeviceBuffer::from_slice(
                    device,
                    stream,
                    &read_gdn_decoder_bf16_vector(
                        "gdn_decoder_moe_shared_expert_gate_weight.bf16",
                        hidden as usize,
                    ),
                )
                .unwrap(),
            }),
        },
    });
    let next_norm = DeviceBuffer::from_slice(
        device,
        stream,
        &read_gdn_decoder_bf16_vector(
            "gdn_decoder_next_layer_norm_raw_weight.bf16",
            hidden as usize,
        ),
    )
    .unwrap();
    GdnDecoderLayerFixture { layer, next_norm }
}

fn model_logits_vector_config() -> QwenConfig {
    let mut config = QwenConfig::randomized_dense_tiny_fixture(0);
    config.num_layers = 1;
    config.max_seq_len = MODEL_LOGITS_TOTAL_ROWS as u32;
    config.max_pages = 2;
    config.page_size = MODEL_LOGITS_PROMPT_LEN as u32;
    config.intermediate_size = MODEL_LOGITS_INTERMEDIATE;
    config.moe = Some(QwenMoeConfig {
        num_experts: QWEN36_MOE_NUM_EXPERTS,
        num_experts_per_tok: QWEN36_MOE_TOP_K,
        moe_intermediate_size: MODEL_LOGITS_INTERMEDIATE,
        shared_expert_intermediate_size: MODEL_LOGITS_INTERMEDIATE,
    });
    config.vocab_size = MODEL_LOGITS_VOCAB;
    config
}

fn model_logits_vector_weights(config: QwenConfig) -> QwenWeights {
    let device = config.device_ordinal;
    let stream = config.stream;
    let hidden = config.hidden_size;
    let q_hidden = config.q_hidden_size().unwrap();
    let kv_hidden = config.kv_hidden_size().unwrap();
    let vocab = config.vocab_size;
    let moe = config.moe_config().unwrap();

    let token_embedding = DeviceBuffer::from_slice(
        device,
        stream,
        &read_model_logits_bf16_vector(
            "model_token_embedding_weight.bf16",
            checked_usize_product(&[vocab, hidden]).unwrap(),
        ),
    )
    .unwrap();
    let final_norm = DeviceBuffer::from_slice(
        device,
        stream,
        &read_model_logits_bf16_vector("model_final_norm_raw_weight.bf16", hidden as usize),
    )
    .unwrap();
    let lm_head = DeviceBuffer::from_slice(
        device,
        stream,
        &read_model_logits_bf16_vector(
            "model_lm_head_weight.bf16",
            checked_usize_product(&[vocab, hidden]).unwrap(),
        ),
    )
    .unwrap();
    let layer = QwenLayerWeights::AttentionMlp(QwenAttentionMlpWeights {
        attn_norm: DeviceBuffer::from_slice(
            device,
            stream,
            &read_model_logits_bf16_vector("model_attn_norm_raw_weight.bf16", hidden as usize),
        )
        .unwrap(),
        q_norm: DeviceBuffer::from_slice(
            device,
            stream,
            &read_model_logits_bf16_vector(
                "model_q_norm_raw_weight.bf16",
                config.head_dim as usize,
            ),
        )
        .unwrap(),
        k_norm: DeviceBuffer::from_slice(
            device,
            stream,
            &read_model_logits_bf16_vector(
                "model_k_norm_raw_weight.bf16",
                config.head_dim as usize,
            ),
        )
        .unwrap(),
        q_proj: DeviceBuffer::from_slice(
            device,
            stream,
            &read_model_logits_bf16_vector(
                "model_q_proj_weight.bf16",
                checked_usize_product(&[QWEN36_FULL_ATTN_Q_PROJ_OUT, hidden]).unwrap(),
            ),
        )
        .unwrap(),
        k_proj: DeviceBuffer::from_slice(
            device,
            stream,
            &read_model_logits_bf16_vector(
                "model_k_proj_weight.bf16",
                checked_usize_product(&[kv_hidden, hidden]).unwrap(),
            ),
        )
        .unwrap(),
        v_proj: DeviceBuffer::from_slice(
            device,
            stream,
            &read_model_logits_bf16_vector(
                "model_v_proj_weight.bf16",
                checked_usize_product(&[kv_hidden, hidden]).unwrap(),
            ),
        )
        .unwrap(),
        o_proj: DeviceBuffer::from_slice(
            device,
            stream,
            &read_model_logits_bf16_vector(
                "model_o_proj_weight.bf16",
                checked_usize_product(&[hidden, q_hidden]).unwrap(),
            ),
        )
        .unwrap(),
        mlp_norm: DeviceBuffer::from_slice(
            device,
            stream,
            &read_model_logits_bf16_vector("model_mlp_norm_raw_weight.bf16", hidden as usize),
        )
        .unwrap(),
        mlp: QwenMlpWeights::Moe {
            router_proj: DeviceBuffer::from_slice(
                device,
                stream,
                &read_model_logits_bf16_vector(
                    "model_moe_router_proj_weight.bf16",
                    checked_usize_product(&[moe.num_experts, hidden]).unwrap(),
                ),
            )
            .unwrap(),
            gate_up_proj: DeviceBuffer::from_slice(
                device,
                stream,
                &read_model_logits_bf16_vector(
                    "model_moe_gate_up_proj_weight.bf16",
                    checked_usize_product(&[moe.num_experts, 2, moe.moe_intermediate_size, hidden])
                        .unwrap(),
                ),
            )
            .unwrap(),
            down_proj: DeviceBuffer::from_slice(
                device,
                stream,
                &read_model_logits_bf16_vector(
                    "model_moe_down_proj_weight.bf16",
                    checked_usize_product(&[moe.num_experts, hidden, moe.moe_intermediate_size])
                        .unwrap(),
                ),
            )
            .unwrap(),
            shared: Some(QwenSharedExpertWeights {
                gate_proj: DeviceBuffer::from_slice(
                    device,
                    stream,
                    &read_model_logits_bf16_vector(
                        "model_moe_shared_gate_proj_weight.bf16",
                        checked_usize_product(&[moe.shared_expert_intermediate_size, hidden])
                            .unwrap(),
                    ),
                )
                .unwrap(),
                up_proj: DeviceBuffer::from_slice(
                    device,
                    stream,
                    &read_model_logits_bf16_vector(
                        "model_moe_shared_up_proj_weight.bf16",
                        checked_usize_product(&[moe.shared_expert_intermediate_size, hidden])
                            .unwrap(),
                    ),
                )
                .unwrap(),
                down_proj: DeviceBuffer::from_slice(
                    device,
                    stream,
                    &read_model_logits_bf16_vector(
                        "model_moe_shared_down_proj_weight.bf16",
                        checked_usize_product(&[hidden, moe.shared_expert_intermediate_size])
                            .unwrap(),
                    ),
                )
                .unwrap(),
                shared_expert_gate: DeviceBuffer::from_slice(
                    device,
                    stream,
                    &read_model_logits_bf16_vector(
                        "model_moe_shared_expert_gate_weight.bf16",
                        hidden as usize,
                    ),
                )
                .unwrap(),
            }),
        },
    });

    QwenWeights {
        config,
        token_embedding,
        final_norm,
        lm_head,
        layers: vec![layer],
    }
}

fn full_attention_block_hidden_len() -> usize {
    checked_usize_product(&[FULL_ATTN_BLOCK_VECTOR_ROWS, QWEN36_HIDDEN_SIZE]).unwrap()
}

fn full_attention_block_q_len() -> usize {
    checked_usize_product(&[
        FULL_ATTN_BLOCK_VECTOR_ROWS,
        QWEN36_FULL_ATTN_Q_HEADS,
        QWEN36_FULL_ATTN_HEAD_DIM,
    ])
    .unwrap()
}

fn full_attention_block_kv_len() -> usize {
    checked_usize_product(&[
        FULL_ATTN_BLOCK_VECTOR_ROWS,
        QWEN36_FULL_ATTN_KV_HEADS,
        QWEN36_FULL_ATTN_HEAD_DIM,
    ])
    .unwrap()
}

fn full_attention_block_q_proj_len() -> usize {
    checked_usize_product(&[FULL_ATTN_BLOCK_VECTOR_ROWS, QWEN36_FULL_ATTN_Q_PROJ_OUT]).unwrap()
}

fn full_attention_block_moe_topk_len() -> usize {
    checked_usize_product(&[FULL_ATTN_BLOCK_VECTOR_ROWS, QWEN36_MOE_TOP_K]).unwrap()
}

fn gdn_decoder_layer_hidden_len() -> usize {
    checked_usize_product(&[GDN_DECODER_LAYER_VECTOR_ROWS, QWEN36_HIDDEN_SIZE]).unwrap()
}

fn gdn_decoder_layer_topk_len() -> usize {
    checked_usize_product(&[GDN_DECODER_LAYER_VECTOR_ROWS, QWEN36_MOE_TOP_K]).unwrap()
}

fn full_attention_q_len() -> usize {
    checked_usize_product(&[
        FULL_ATTN_VECTOR_ROWS,
        QWEN36_FULL_ATTN_Q_HEADS,
        QWEN36_FULL_ATTN_HEAD_DIM,
    ])
    .unwrap()
}

fn full_attention_kv_len() -> usize {
    checked_usize_product(&[
        FULL_ATTN_VECTOR_ROWS,
        QWEN36_FULL_ATTN_KV_HEADS,
        QWEN36_FULL_ATTN_HEAD_DIM,
    ])
    .unwrap()
}

fn full_attention_packed_q_gate_len() -> usize {
    checked_usize_product(&[FULL_ATTN_VECTOR_ROWS, QWEN36_FULL_ATTN_Q_PROJ_OUT]).unwrap()
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
    const HEADS: usize = QWEN36_FULL_ATTN_Q_HEADS as usize;
    const HEAD_DIM: usize = QWEN36_FULL_ATTN_HEAD_DIM as usize;
    const PACKED_HEAD_DIM: usize = 2 * HEAD_DIM;

    let mut packed = vec![0_u16; ROWS * HEADS * PACKED_HEAD_DIM];
    let mut expected_q = vec![0_u16; ROWS * HEADS * HEAD_DIM];
    let mut expected_gate = vec![0_u16; ROWS * HEADS * HEAD_DIM];
    for row in 0..ROWS {
        for head in 0..HEADS {
            for lane in 0..HEAD_DIM {
                let packed_base = (row * HEADS + head) * PACKED_HEAD_DIM;
                let out_idx = (row * HEADS + head) * HEAD_DIM + lane;
                let q = qwen36_packed_q_gate_pattern(row, head, lane, false);
                let gate = qwen36_packed_q_gate_pattern(row, head, lane, true);
                packed[packed_base + lane] = q;
                packed[packed_base + HEAD_DIM + lane] = gate;
                expected_q[out_idx] = q;
                expected_gate[out_idx] = gate;
            }
        }
    }

    let stream = ptr::null_mut();
    let packed_device = DeviceBuffer::from_slice(-1, stream, &packed).unwrap();
    let mut q_device = DeviceBuffer::empty(-1);
    let mut gate_device = DeviceBuffer::empty(-1);
    q_device.ensure(expected_q.len()).unwrap();
    gate_device.ensure(expected_gate.len()).unwrap();

    unsafe {
        crate::engine::Engine::new(full_attention_vector_config().engine_config())
            .unwrap()
            .operators()
            .qscu()
            .qwen36_extract_q_and_gate_bf16(
                packed_device
                    .matrix(ROWS as u32, QWEN36_FULL_ATTN_Q_PROJ_OUT)
                    .unwrap(),
                q_device
                    .matrix(ROWS as u32, QWEN36_FULL_ATTN_Q_HIDDEN)
                    .unwrap(),
                gate_device
                    .matrix(ROWS as u32, QWEN36_FULL_ATTN_Q_HIDDEN)
                    .unwrap(),
            )
            .unwrap();
    }
    synchronize_stream(stream).unwrap();

    assert_eq!(
        download_bf16(&q_device, stream, expected_q.len()),
        expected_q
    );
    assert_eq!(
        download_bf16(&gate_device, stream, expected_gate.len()),
        expected_gate
    );
}

#[test]
fn qwen36_full_attention_vectors_validate_packed_q_gate_extraction() {
    if !cuda_device_available() {
        return;
    }

    let mut runner = full_attention_vector_runner();
    let stream = runner.config.stream;
    let rows = FULL_ATTN_VECTOR_ROWS;
    let q_len = full_attention_q_len();
    runner.scratch.ensure(&runner.config, rows).unwrap();

    let packed = read_bf16_vector(
        "attention_packed_q_gate.bf16",
        full_attention_packed_q_gate_len(),
    );
    runner.scratch.q_proj_out.upload(stream, &packed).unwrap();

    unsafe {
        runner
            .engine
            .operators()
            .qscu()
            .qwen36_extract_q_and_gate_bf16(
                runner
                    .scratch
                    .q_proj_out
                    .matrix(rows, QWEN36_FULL_ATTN_Q_PROJ_OUT)
                    .unwrap(),
                runner
                    .scratch
                    .q
                    .matrix(rows, QWEN36_FULL_ATTN_Q_HIDDEN)
                    .unwrap(),
                runner
                    .scratch
                    .attn_gate
                    .matrix(rows, QWEN36_FULL_ATTN_Q_HIDDEN)
                    .unwrap(),
            )
    }
    .unwrap();
    synchronize_stream(stream).unwrap();

    let got_q = download_bf16(&runner.scratch.q, stream, q_len);
    let got_gate = download_bf16(&runner.scratch.attn_gate, stream, q_len);
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
    let stream = runner.config.stream;
    let device = runner.config.device_ordinal;
    let rows = FULL_ATTN_VECTOR_ROWS;
    let q_len = full_attention_q_len();
    let kv_len = full_attention_kv_len();
    let head_dim = QWEN36_FULL_ATTN_HEAD_DIM as usize;
    runner.scratch.ensure(&runner.config, rows).unwrap();

    let packed = read_bf16_vector(
        "attention_packed_q_gate.bf16",
        full_attention_packed_q_gate_len(),
    );
    let k_input = read_bf16_vector("attention_k_input.bf16", kv_len);
    let positions = read_i32_vector("attention_positions.i32", rows as usize);
    runner.scratch.q_proj_out.upload(stream, &packed).unwrap();
    runner.scratch.k.upload(stream, &k_input).unwrap();
    runner.scratch.positions.upload(stream, &positions).unwrap();

    unsafe {
        runner
            .engine
            .operators()
            .qscu()
            .qwen36_extract_q_and_gate_bf16(
                runner
                    .scratch
                    .q_proj_out
                    .matrix(rows, QWEN36_FULL_ATTN_Q_PROJ_OUT)
                    .unwrap(),
                runner
                    .scratch
                    .q
                    .matrix(rows, QWEN36_FULL_ATTN_Q_HIDDEN)
                    .unwrap(),
                runner
                    .scratch
                    .attn_gate
                    .matrix(rows, QWEN36_FULL_ATTN_Q_HIDDEN)
                    .unwrap(),
            )
    }
    .unwrap();

    let q_norm_weight = DeviceBuffer::from_slice(
        device,
        stream,
        &read_bf16_vector("attention_q_norm_raw_weight.bf16", head_dim),
    )
    .unwrap();
    let k_norm_weight = DeviceBuffer::from_slice(
        device,
        stream,
        &read_bf16_vector("attention_k_norm_raw_weight.bf16", head_dim),
    )
    .unwrap();

    let q_ptr = runner
        .scratch
        .q
        .matrix(rows * runner.config.num_q_heads, runner.config.head_dim)
        .unwrap();
    unsafe {
        runner.engine.operators().qsfi().rmsnorm_bf16(
            &crate::backend::qsfi::RmsNormBf16::qwen_qk_norm(
                q_ptr,
                q_norm_weight.vector(runner.config.head_dim).unwrap(),
                q_ptr,
                runner.config.rms_norm_eps,
            )
            .unwrap(),
        )
    }
    .unwrap();
    let k_ptr = runner
        .scratch
        .k
        .matrix(rows * runner.config.num_kv_heads, runner.config.head_dim)
        .unwrap();
    unsafe {
        runner.engine.operators().qsfi().rmsnorm_bf16(
            &crate::backend::qsfi::RmsNormBf16::qwen_qk_norm(
                k_ptr,
                k_norm_weight.vector(runner.config.head_dim).unwrap(),
                k_ptr,
                runner.config.rms_norm_eps,
            )
            .unwrap(),
        )
    }
    .unwrap();
    synchronize_stream(stream).unwrap();

    let got_q_norm = download_bf16(&runner.scratch.q, stream, q_len);
    let got_k_norm = download_bf16(&runner.scratch.k, stream, kv_len);
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
    synchronize_stream(stream).unwrap();

    let got_q_rope = download_bf16(&runner.scratch.q, stream, q_len);
    let got_k_rope = download_bf16(&runner.scratch.k, stream, kv_len);
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
    let stream = runner.config.stream;
    let rows = FULL_ATTN_VECTOR_ROWS;
    let q_len = full_attention_q_len();
    runner.scratch.ensure(&runner.config, rows).unwrap();

    let gate = read_bf16_vector("attention_gate_extracted.bf16", q_len);
    let out_initial = read_bf16_vector("attention_gate_out_initial.bf16", q_len);
    runner.scratch.attn_gate.upload(stream, &gate).unwrap();
    runner
        .scratch
        .attn_out
        .upload(stream, &out_initial)
        .unwrap();

    unsafe {
        runner
            .engine
            .operators()
            .qscu()
            .qwen36_full_attention_output_gate_bf16(
                runner
                    .scratch
                    .attn_gate
                    .matrix(rows, QWEN36_FULL_ATTN_Q_HIDDEN)
                    .unwrap(),
                runner
                    .scratch
                    .attn_out
                    .matrix(rows, QWEN36_FULL_ATTN_Q_HIDDEN)
                    .unwrap(),
            )
    }
    .unwrap();
    synchronize_stream(stream).unwrap();

    let got = download_bf16(&runner.scratch.attn_out, stream, q_len);
    let expected = read_bf16_vector("attention_gated_output_bf16.bf16", q_len);
    assert_bf16_exact("attention output gate vector", &got, &expected);
}

#[test]
fn qwen36_full_attention_block_vector_validates_attention_residual_norm_composition() {
    if !cuda_device_available() {
        return;
    }

    let mut runner = full_attention_block_vector_runner();
    let stream = runner.config.stream;
    let device = runner.config.device_ordinal;
    let rows = FULL_ATTN_BLOCK_VECTOR_ROWS;
    let hidden = QWEN36_HIDDEN_SIZE;
    let q_hidden = QWEN36_FULL_ATTN_Q_HIDDEN;
    let kv_hidden = QWEN36_FULL_ATTN_KV_HIDDEN;
    let hidden_len = full_attention_block_hidden_len();
    let q_len = full_attention_block_q_len();
    let kv_len = full_attention_block_kv_len();
    let q_proj_len = full_attention_block_q_proj_len();
    runner.scratch.ensure(&runner.config, rows).unwrap();

    let input_residual = read_block_bf16_vector("block_input_residual.bf16", hidden_len);
    let positions = read_block_i32_vector("block_positions.i32", rows as usize);
    runner
        .scratch
        .residual
        .upload(stream, &input_residual)
        .unwrap();
    runner.scratch.positions.upload(stream, &positions).unwrap();

    let attn_norm_weight = DeviceBuffer::from_slice(
        device,
        stream,
        &read_block_bf16_vector("block_attn_norm_raw_weight.bf16", hidden as usize),
    )
    .unwrap();
    let q_norm_weight = DeviceBuffer::from_slice(
        device,
        stream,
        &read_block_bf16_vector(
            "block_q_norm_raw_weight.bf16",
            QWEN36_FULL_ATTN_HEAD_DIM as usize,
        ),
    )
    .unwrap();
    let k_norm_weight = DeviceBuffer::from_slice(
        device,
        stream,
        &read_block_bf16_vector(
            "block_k_norm_raw_weight.bf16",
            QWEN36_FULL_ATTN_HEAD_DIM as usize,
        ),
    )
    .unwrap();
    let post_norm_weight = DeviceBuffer::from_slice(
        device,
        stream,
        &read_block_bf16_vector("block_post_attn_norm_raw_weight.bf16", hidden as usize),
    )
    .unwrap();
    let q_proj_weight = DeviceBuffer::from_slice(
        device,
        stream,
        &read_block_bf16_vector(
            "block_q_proj_weight.bf16",
            checked_usize_product(&[QWEN36_FULL_ATTN_Q_PROJ_OUT, hidden]).unwrap(),
        ),
    )
    .unwrap();
    let k_proj_weight = DeviceBuffer::from_slice(
        device,
        stream,
        &read_block_bf16_vector(
            "block_k_proj_weight.bf16",
            checked_usize_product(&[kv_hidden, hidden]).unwrap(),
        ),
    )
    .unwrap();
    let v_proj_weight = DeviceBuffer::from_slice(
        device,
        stream,
        &read_block_bf16_vector(
            "block_v_proj_weight.bf16",
            checked_usize_product(&[kv_hidden, hidden]).unwrap(),
        ),
    )
    .unwrap();
    let o_proj_weight = DeviceBuffer::from_slice(
        device,
        stream,
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
                attn_norm_weight.vector(runner.config.hidden_size).unwrap(),
                norm_ptr,
                runner.config.rms_norm_eps,
            )
            .unwrap(),
        )
    }
    .unwrap();
    synchronize_stream(stream).unwrap();
    let got_attn_norm = download_bf16(&runner.scratch.norm, stream, hidden_len);
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
            q_proj_weight
                .matrix(QWEN36_FULL_ATTN_Q_PROJ_OUT, hidden)
                .unwrap(),
            runner
                .scratch
                .q_proj_out
                .matrix(rows, QWEN36_FULL_ATTN_Q_PROJ_OUT)
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
                    .matrix(rows, QWEN36_FULL_ATTN_Q_PROJ_OUT)
                    .unwrap(),
                runner
                    .scratch
                    .q
                    .matrix(rows, QWEN36_FULL_ATTN_Q_HIDDEN)
                    .unwrap(),
                runner
                    .scratch
                    .attn_gate
                    .matrix(rows, QWEN36_FULL_ATTN_Q_HIDDEN)
                    .unwrap(),
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
    synchronize_stream(stream).unwrap();

    let got_q_proj = download_bf16(&runner.scratch.q_proj_out, stream, q_proj_len);
    assert_bf16_close_to_f32_oracle(
        "full-attention block q projection",
        &got_q_proj,
        &read_block_bf16_vector("block_expected_q_proj_output_bf16.bf16", q_proj_len),
        &read_block_f32_vector("block_expected_q_proj_output_f32.f32", q_proj_len),
        BF16_BLOCK_PROJ_ABS_TOL,
    );
    assert_bf16_exact(
        "full-attention block q extraction",
        &download_bf16(&runner.scratch.q, stream, q_len),
        &read_block_bf16_vector("block_expected_q_extracted_bf16.bf16", q_len),
    );
    assert_bf16_exact(
        "full-attention block output-gate extraction",
        &download_bf16(&runner.scratch.attn_gate, stream, q_len),
        &read_block_bf16_vector("block_expected_gate_extracted_bf16.bf16", q_len),
    );
    assert_bf16_close_to_f32_oracle(
        "full-attention block k projection",
        &download_bf16(&runner.scratch.k, stream, kv_len),
        &read_block_bf16_vector("block_expected_k_proj_output_bf16.bf16", kv_len),
        &read_block_f32_vector("block_expected_k_proj_output_f32.f32", kv_len),
        BF16_BLOCK_PROJ_ABS_TOL,
    );
    assert_bf16_close_to_f32_oracle(
        "full-attention block v projection",
        &download_bf16(&runner.scratch.v, stream, kv_len),
        &read_block_bf16_vector("block_expected_v_proj_output_bf16.bf16", kv_len),
        &read_block_f32_vector("block_expected_v_proj_output_f32.f32", kv_len),
        BF16_BLOCK_PROJ_ABS_TOL,
    );

    let q_ptr = runner
        .scratch
        .q
        .matrix(rows * runner.config.num_q_heads, runner.config.head_dim)
        .unwrap();
    unsafe {
        runner.engine.operators().qsfi().rmsnorm_bf16(
            &crate::backend::qsfi::RmsNormBf16::qwen_qk_norm(
                q_ptr,
                q_norm_weight.vector(runner.config.head_dim).unwrap(),
                q_ptr,
                runner.config.rms_norm_eps,
            )
            .unwrap(),
        )
    }
    .unwrap();
    let k_ptr = runner
        .scratch
        .k
        .matrix(rows * runner.config.num_kv_heads, runner.config.head_dim)
        .unwrap();
    unsafe {
        runner.engine.operators().qsfi().rmsnorm_bf16(
            &crate::backend::qsfi::RmsNormBf16::qwen_qk_norm(
                k_ptr,
                k_norm_weight.vector(runner.config.head_dim).unwrap(),
                k_ptr,
                runner.config.rms_norm_eps,
            )
            .unwrap(),
        )
    }
    .unwrap();
    synchronize_stream(stream).unwrap();
    assert_bf16_close_to_f32_oracle(
        "full-attention block q Gemma RMSNorm",
        &download_bf16(&runner.scratch.q, stream, q_len),
        &read_block_bf16_vector("block_expected_q_norm_output_bf16.bf16", q_len),
        &read_block_f32_vector("block_expected_q_norm_output_f32.f32", q_len),
        BF16_BLOCK_NORM_ABS_TOL,
    );
    assert_bf16_close_to_f32_oracle(
        "full-attention block k Gemma RMSNorm",
        &download_bf16(&runner.scratch.k, stream, kv_len),
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
    synchronize_stream(stream).unwrap();
    assert_bf16_close_to_f32_oracle(
        "full-attention block q partial RoPE",
        &download_bf16(&runner.scratch.q, stream, q_len),
        &read_block_bf16_vector("block_expected_q_rope_output_bf16.bf16", q_len),
        &read_block_f32_vector("block_expected_q_rope_output_f32.f32", q_len),
        BF16_BLOCK_ROPE_ABS_TOL,
    );
    assert_bf16_close_to_f32_oracle(
        "full-attention block k partial RoPE",
        &download_bf16(&runner.scratch.k, stream, kv_len),
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
            .heads(rows, runner.config.num_q_heads, runner.config.head_dim)
            .unwrap(),
        runner
            .scratch
            .k
            .heads(rows, runner.config.num_kv_heads, runner.config.head_dim)
            .unwrap(),
        runner
            .scratch
            .v
            .heads(rows, runner.config.num_kv_heads, runner.config.head_dim)
            .unwrap(),
        runner
            .scratch
            .attn_out
            .heads(rows, runner.config.num_q_heads, runner.config.head_dim)
            .unwrap(),
        runner.scratch.positions.vector(rows).unwrap(),
    );
    unsafe { runner.engine.append_attention(&engine_layer) }.unwrap();
    synchronize_stream(stream).unwrap();
    assert_bf16_close_to_f32_oracle(
        "full-attention block raw causal GQA attention",
        &download_bf16(&runner.scratch.attn_out, stream, q_len),
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
                runner
                    .scratch
                    .attn_gate
                    .matrix(rows, QWEN36_FULL_ATTN_Q_HIDDEN)
                    .unwrap(),
                runner
                    .scratch
                    .attn_out
                    .matrix(rows, QWEN36_FULL_ATTN_Q_HIDDEN)
                    .unwrap(),
            )
    }
    .unwrap();
    synchronize_stream(stream).unwrap();
    assert_bf16_close_to_f32_oracle(
        "full-attention block sigmoid output gate",
        &download_bf16(&runner.scratch.attn_out, stream, q_len),
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
    synchronize_stream(stream).unwrap();
    assert_bf16_close_to_f32_oracle(
        "full-attention block output projection",
        &download_bf16(&runner.scratch.attn_proj, stream, hidden_len),
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
                    .matrix(rows, runner.config.hidden_size)
                    .unwrap(),
                runner
                    .scratch
                    .residual
                    .matrix(rows, runner.config.hidden_size)
                    .unwrap(),
                post_norm_weight.vector(runner.config.hidden_size).unwrap(),
                runner.config.rms_norm_eps,
            )
            .unwrap(),
        )
    }
    .unwrap();
    synchronize_stream(stream).unwrap();
    assert_bf16_close_to_f32_oracle(
        "full-attention block residual after attention add",
        &download_bf16(&runner.scratch.residual, stream, hidden_len),
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
        &download_bf16(&runner.scratch.attn_proj, stream, hidden_len),
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
    let stream = runner.config.stream;
    let device = runner.config.device_ordinal;
    let rows = FULL_ATTN_BLOCK_VECTOR_ROWS;
    let hidden = QWEN36_HIDDEN_SIZE;
    let hidden_len = full_attention_block_hidden_len();
    runner.scratch.ensure(&runner.config, rows).unwrap();

    let layer = full_attention_block_moe_layer(device, stream);
    let layer_weights = layer.weights();
    runner
        .scratch
        .residual
        .upload(
            stream,
            &read_block_bf16_vector("block_input_residual.bf16", hidden_len),
        )
        .unwrap();
    runner
        .scratch
        .positions
        .upload(
            stream,
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
                    .vector(runner.config.hidden_size)
                    .unwrap(),
                norm_ptr,
                runner.config.rms_norm_eps,
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
    synchronize_stream(stream).unwrap();
    assert_bf16_close_to_f32_oracle(
        "decoder slice produced attention output projection",
        &download_bf16(&runner.scratch.attn_proj, stream, hidden_len),
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
    synchronize_stream(stream).unwrap();

    assert_bf16_close_to_f32_oracle(
        "decoder slice post-attention Gemma RMSNorm handoff",
        &download_bf16(&runner.scratch.attn_proj, stream, hidden_len),
        &read_block_bf16_vector("block_expected_post_attn_norm_output_bf16.bf16", hidden_len),
        &read_block_f32_vector("block_expected_post_attn_norm_output_f32.f32", hidden_len),
        BF16_BLOCK_NORM_ABS_TOL,
    );
    assert_bf16_close_to_f32_oracle(
        "decoder slice residual after attention plus MoE/shared expert",
        &download_bf16(&runner.scratch.residual, stream, hidden_len),
        &read_block_bf16_vector("block_expected_residual_after_mlp_bf16.bf16", hidden_len),
        &read_block_f32_vector("block_expected_residual_after_mlp_f32.f32", hidden_len),
        BF16_BLOCK_FINAL_NORM_ABS_TOL,
    );
    assert_bf16_close_to_f32_oracle(
        "decoder slice next-layer Gemma RMSNorm",
        &download_bf16(&runner.scratch.mlp_out, stream, hidden_len),
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
fn qwen36_gdn_decoder_layer_vector_chains_gdn_into_moe_and_next_norm() {
    if !cuda_device_available() {
        return;
    }

    let mut runner = gdn_decoder_layer_vector_runner();
    let stream = runner.config.stream;
    let device = runner.config.device_ordinal;
    let rows = GDN_DECODER_LAYER_VECTOR_ROWS;
    let hidden = QWEN36_HIDDEN_SIZE;
    let hidden_len = gdn_decoder_layer_hidden_len();
    let gdn_out_len = checked_usize_product(&[rows, QWEN36_GDN_OUTPUT_DIM]).unwrap();
    let topk_len = gdn_decoder_layer_topk_len();
    runner.scratch.ensure(&runner.config, rows).unwrap();

    runner
        .scratch
        .norm
        .upload(
            stream,
            &read_gdn_decoder_bf16_vector("gdn_decoder_layer_input.bf16", hidden_len),
        )
        .unwrap();
    runner
        .scratch
        .residual
        .upload(
            stream,
            &read_gdn_decoder_bf16_vector("gdn_decoder_input_residual.bf16", hidden_len),
        )
        .unwrap();

    let layer = gdn_decoder_layer_fixture(device, stream);
    let layer_weights = layer.weights();
    let norm_ptr = runner.scratch.norm.matrix(rows, hidden).unwrap();
    unsafe {
        runner
            .execution()
            .unwrap()
            .1
            .execute_gdn_layer(0, rows, norm_ptr, layer_weights, ActiveRunKind::Append)
            .unwrap();
    }
    synchronize_stream(stream).unwrap();

    assert_bf16_close_to_f32_oracle(
        "GDN decoder slice gated recurrent RMSNorm",
        &download_bf16(&runner.scratch.gdn_norm_out, stream, gdn_out_len),
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
        &download_bf16(&runner.scratch.attn_proj, stream, hidden_len),
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
    synchronize_stream(stream).unwrap();

    assert_bf16_close_to_f32_oracle(
        "GDN decoder slice post-attention Gemma RMSNorm handoff",
        &download_bf16(&runner.scratch.attn_proj, stream, hidden_len),
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
            &runner.scratch.router_logits,
            stream,
            (rows * QWEN36_MOE_NUM_EXPERTS) as usize,
        ),
        &read_gdn_decoder_bf16_vector(
            "gdn_decoder_expected_moe_router_logits_bf16.bf16",
            (rows * QWEN36_MOE_NUM_EXPERTS) as usize,
        ),
        &read_gdn_decoder_f32_vector(
            "gdn_decoder_expected_moe_router_logits_f32.f32",
            (rows * QWEN36_MOE_NUM_EXPERTS) as usize,
        ),
        BF16_GDN_DECODER_PROJ_ABS_TOL,
    );
    assert_eq!(
        download_i32(&runner.scratch.topk_ids, stream, topk_len),
        read_gdn_decoder_i32_vector("gdn_decoder_expected_moe_topk_ids.i32", topk_len),
        "GDN decoder slice MoE router top-k ids changed"
    );
    assert_moe_router_matches_cpu(&runner, rows);
    assert_f32_close(
        "GDN decoder slice shared expert gate logits",
        &download_f32(&runner.scratch.shared_gate_logits, stream, rows as usize),
        &read_gdn_decoder_f32_vector(
            "gdn_decoder_expected_shared_gate_logits_f32.f32",
            rows as usize,
        ),
        BF16_GDN_DECODER_PROJ_ABS_TOL,
        1.0e-4,
    );
    assert_bf16_close_to_f32_oracle(
        "GDN decoder slice shared expert output",
        &download_bf16(&runner.scratch.shared_out, stream, hidden_len),
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
        &download_bf16(&runner.scratch.residual, stream, hidden_len),
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
        &download_bf16(&runner.scratch.mlp_out, stream, hidden_len),
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
    let stream = runner.config.stream;
    let device = runner.config.device_ordinal;
    let rows = FULL_ATTN_BLOCK_VECTOR_ROWS;
    let hidden_len = full_attention_block_hidden_len();
    let topk_len = full_attention_block_moe_topk_len();
    runner.scratch.ensure(&runner.config, rows).unwrap();

    runner
        .scratch
        .attn_proj
        .upload(
            stream,
            &read_block_bf16_vector("block_expected_post_attn_norm_output_bf16.bf16", hidden_len),
        )
        .unwrap();
    runner
        .scratch
        .residual
        .upload(
            stream,
            &read_block_bf16_vector(
                "block_expected_residual_after_attention_bf16.bf16",
                hidden_len,
            ),
        )
        .unwrap();

    let layer = full_attention_block_moe_layer(device, stream);
    let layer_weights = layer.weights();
    let (router_proj, gate_up_proj, down_proj, shared) = match &layer_weights.mlp {
        QwenMlpWeights::Moe {
            router_proj,
            gate_up_proj,
            down_proj,
            shared,
        } => (router_proj, gate_up_proj, down_proj, shared.as_ref()),
        QwenMlpWeights::Dense { .. } => unreachable!("block fixture must use MoE"),
    };

    unsafe {
        runner
            .execution()
            .unwrap()
            .1
            .execute_moe_mlp(rows, router_proj, gate_up_proj, down_proj, shared)
            .unwrap();
    }
    synchronize_stream(stream).unwrap();

    assert_bf16_close_to_f32_oracle(
        "full-attention block MoE router logits",
        &download_bf16(
            &runner.scratch.router_logits,
            stream,
            (rows * QWEN36_MOE_NUM_EXPERTS) as usize,
        ),
        &read_block_bf16_vector(
            "block_expected_moe_router_logits_bf16.bf16",
            (rows * QWEN36_MOE_NUM_EXPERTS) as usize,
        ),
        &read_block_f32_vector(
            "block_expected_moe_router_logits_f32.f32",
            (rows * QWEN36_MOE_NUM_EXPERTS) as usize,
        ),
        BF16_BLOCK_PROJ_ABS_TOL,
    );
    assert_eq!(
        download_i32(&runner.scratch.topk_ids, stream, topk_len),
        read_block_i32_vector("block_expected_moe_topk_ids.i32", topk_len),
        "full-attention block MoE router top-k ids changed"
    );
    assert_moe_router_matches_cpu(&runner, rows);
    assert_f32_close(
        "full-attention block shared expert gate logits",
        &download_f32(&runner.scratch.shared_gate_logits, stream, rows as usize),
        &read_block_f32_vector("block_expected_shared_gate_logits_f32.f32", rows as usize),
        BF16_BLOCK_PROJ_ABS_TOL,
        1.0e-4,
    );
    assert_bf16_close_to_f32_oracle(
        "full-attention block shared expert output",
        &download_bf16(&runner.scratch.shared_out, stream, hidden_len),
        &read_block_bf16_vector("block_expected_shared_expert_output_bf16.bf16", hidden_len),
        &read_block_f32_vector("block_expected_shared_expert_output_f32.f32", hidden_len),
        BF16_BLOCK_MOE_ABS_TOL,
    );
    assert_bf16_close_to_f32_oracle(
        "full-attention block MoE/shared expert output",
        &download_bf16(&runner.scratch.mlp_out, stream, hidden_len),
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
                    .matrix(rows, runner.config.hidden_size)
                    .unwrap(),
                runner
                    .scratch
                    .residual
                    .matrix(rows, runner.config.hidden_size)
                    .unwrap(),
                layer.next_norm.vector(runner.config.hidden_size).unwrap(),
                runner.config.rms_norm_eps,
            )
            .unwrap(),
        )
    }
    .unwrap();
    synchronize_stream(stream).unwrap();
    assert_bf16_close_to_f32_oracle(
        "full-attention block residual after MoE/shared expert",
        &download_bf16(&runner.scratch.residual, stream, hidden_len),
        &read_block_bf16_vector("block_expected_residual_after_mlp_bf16.bf16", hidden_len),
        &read_block_f32_vector("block_expected_residual_after_mlp_f32.f32", hidden_len),
        BF16_BLOCK_FINAL_NORM_ABS_TOL,
    );
    assert_bf16_close_to_f32_oracle(
        "full-attention block next-layer Gemma RMSNorm",
        &download_bf16(&runner.scratch.mlp_out, stream, hidden_len),
        &read_block_bf16_vector(
            "block_expected_next_layer_norm_output_bf16.bf16",
            hidden_len,
        ),
        &read_block_f32_vector("block_expected_next_layer_norm_output_f32.f32", hidden_len),
        BF16_BLOCK_FINAL_NORM_ABS_TOL,
    );
}

#[test]
fn qwen36_model_logits_vector_validates_public_run_moe_logits_handoff() {
    if !cuda_device_available() {
        return;
    }

    let config = model_logits_vector_config();
    let stream = config.stream;
    let request_id = 0x4D4C_0001;
    let weights = model_logits_vector_weights(config);
    let mut runner = ModelRunner::new(config, weights).unwrap();
    let prompt = read_model_logits_i32_vector("model_prompt_tokens.i32", MODEL_LOGITS_PROMPT_LEN);
    assert_eq!(prompt.len() as u32, runner.config.page_size);

    let prefill_result = runner
        .run(QwenRequest {
            request_id,
            tokens: &prompt,
            max_new_tokens: 0,
        })
        .unwrap();
    synchronize_stream(stream).unwrap();
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
        &download_f32(&runner.scratch.logits, stream, vocab),
        &expected_prefill[prefill_logits_len - vocab..],
        MODEL_LOGITS_ABS_TOL,
        MODEL_LOGITS_REL_TOL,
    );
    let prefill_top = download_i32(&runner.scratch.next_token_ids, stream, 1);
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
    synchronize_stream(stream).unwrap();
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
        &download_f32(&runner.scratch.logits, stream, MODEL_LOGITS_VOCAB as usize),
        &read_model_logits_f32_vector(
            "model_expected_decode_logits_f32.f32",
            MODEL_LOGITS_VOCAB as usize,
        ),
        MODEL_LOGITS_ABS_TOL,
        MODEL_LOGITS_REL_TOL,
    );
    let expected_decode_top = read_model_logits_i32_vector("model_expected_decode_top_ids.i32", 1);
    assert_eq!(
        download_i32(&runner.scratch.next_token_ids, stream, 1),
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

    let topk_len = (MOE_VECTOR_ROWS * QWEN36_MOE_TOP_K) as usize;
    let got_ids = download_i32(&runner.scratch.topk_ids, runner.config.stream, topk_len);
    let got_weights = download_f32(&runner.scratch.topk_weights, runner.config.stream, topk_len);
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

    let topk_len = (MOE_VECTOR_ROWS * QWEN36_MOE_TOP_K) as usize;
    let got_ids = download_i32(&runner.scratch.topk_ids, runner.config.stream, topk_len);
    let got_weights = download_f32(&runner.scratch.topk_weights, runner.config.stream, topk_len);
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

    let got = download_bf16(
        &runner.scratch.mlp_out,
        runner.config.stream,
        MOE_VECTOR_ELEMENTS,
    );
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

    let got = download_bf16(
        &runner.scratch.mlp_out,
        runner.config.stream,
        MOE_VECTOR_ELEMENTS,
    );
    assert_bf16_close_to_f32_oracle(
        "shared expert gate-add BF16 vector",
        &got,
        &read_moe_bf16_vector("combined_output_bf16.bf16", MOE_VECTOR_ELEMENTS),
        &read_moe_f32_vector("combined_output.f32", MOE_VECTOR_ELEMENTS),
        BF16_MOE_ABS_TOL,
    );
}

#[test]
fn public_moe_config_validation_rejects_invalid_config_json_shapes() {
    assert_eq!(
        QwenMoeConfig::qwen36_35b_a3b(),
        QwenMoeConfig {
            num_experts: QWEN36_MOE_NUM_EXPERTS,
            num_experts_per_tok: QWEN36_MOE_TOP_K,
            moe_intermediate_size: QWEN36_MOE_INTERMEDIATE_SIZE,
            shared_expert_intermediate_size: QWEN36_MOE_SHARED_EXPERT_INTERMEDIATE_SIZE,
        }
    );
    assert_eq!(
        QwenMoeConfig {
            num_experts: 4,
            num_experts_per_tok: 0,
            moe_intermediate_size: 64,
            shared_expert_intermediate_size: 0,
        }
        .validate(128),
        Err(Status::InvalidArgument)
    );
    assert_eq!(
        QwenMoeConfig {
            num_experts: 4,
            num_experts_per_tok: 5,
            moe_intermediate_size: 64,
            shared_expert_intermediate_size: 0,
        }
        .validate(128),
        Err(Status::InvalidArgument)
    );
    assert_eq!(
        QwenMoeConfig {
            num_experts: 32,
            num_experts_per_tok: 17,
            moe_intermediate_size: 64,
            shared_expert_intermediate_size: 0,
        }
        .validate(128),
        Err(Status::Unsupported)
    );
    assert_eq!(
        QwenMoeConfig {
            num_experts: 4097,
            num_experts_per_tok: 2,
            moe_intermediate_size: 64,
            shared_expert_intermediate_size: 0,
        }
        .validate(128),
        Err(Status::Unsupported)
    );
    assert_eq!(
        QwenMoeConfig {
            num_experts: 4,
            num_experts_per_tok: 2,
            moe_intermediate_size: 66,
            shared_expert_intermediate_size: 0,
        }
        .validate(128),
        Err(Status::InvalidArgument)
    );
    assert_eq!(
        QwenMoeConfig {
            num_experts: 4,
            num_experts_per_tok: 2,
            moe_intermediate_size: 64,
            shared_expert_intermediate_size: 66,
        }
        .validate(128),
        Err(Status::InvalidArgument)
    );
}

#[test]
fn private_qwen36_gdn_validation_is_fixed_to_supported_moe_shape() {
    let mut dense_gdn = QwenConfig::randomized_qwen36_moe_gdn_one_block_fixture(-1);
    dense_gdn.moe = None;
    dense_gdn.hidden_size = 10240;
    dense_gdn.intermediate_size = 17408;
    dense_gdn.model_shape = QwenModelShape {
        layer_pattern: QwenLayerPattern::Qwen36HybridGdn {
            gdn: QwenGdnShape::qwen36_dense_27b(),
        },
    };
    assert_eq!(dense_gdn.validate(), Err(Status::Unsupported));

    let mut wrong_hidden = QwenConfig::randomized_qwen36_moe_gdn_one_block_fixture(-1);
    wrong_hidden.hidden_size = 4096;
    assert_eq!(wrong_hidden.validate(), Err(Status::InvalidArgument));

    let mut missing_full_attention = QwenConfig::randomized_qwen36_moe_gdn_one_block_fixture(-1);
    missing_full_attention.num_layers = 1;
    assert_eq!(
        missing_full_attention.validate(),
        Err(Status::InvalidArgument)
    );

    let introduces_full_attention = QwenConfig::randomized_qwen36_moe_gdn_one_block_fixture(-1);
    assert_eq!(introduces_full_attention.validate(), Ok(()));

    let mut missing_shared = QwenConfig::randomized_qwen36_moe_gdn_one_block_fixture(-1);
    missing_shared.moe = Some(QwenMoeConfig {
        shared_expert_intermediate_size: 0,
        ..QwenMoeConfig::qwen36_35b_a3b()
    });
    assert_eq!(missing_shared.validate(), Err(Status::Unsupported));
}

#[test]
fn qwen36_no_full_attention_config_is_invalid() {
    let mut config = QwenConfig::randomized_qwen36_moe_gdn_one_block_fixture(-1);
    config.num_layers = 1;
    assert_eq!(config.attention_layer_count(), 0);
    assert_eq!(config.gdn_layer_count(), 1);
    assert_eq!(config.validate(), Err(Status::InvalidArgument));
}

#[test]
fn qwen36_incomplete_hybrid_schedule_is_invalid() {
    let mut config = QwenConfig::randomized_qwen36_moe_gdn_one_block_fixture(-1);
    config.num_layers = 5;
    assert_eq!(config.attention_layer_count(), 1);
    assert_eq!(config.gdn_layer_count(), 4);
    assert_eq!(config.validate(), Err(Status::InvalidArgument));
}

#[test]
fn qwen36_one_schedule_block_engine_config_uses_real_attention_dimensions() {
    let config = QwenConfig::randomized_qwen36_moe_gdn_one_block_fixture(-1);
    assert_eq!(config.validate(), Ok(()));
    assert_eq!(config.attention_layer_count(), 1);
    assert_eq!(config.gdn_layer_count(), 3);
    assert_eq!(config.hidden_size, QWEN36_HIDDEN_SIZE);
    assert_eq!(config.num_q_heads, QWEN36_FULL_ATTN_Q_HEADS);
    assert_eq!(config.num_kv_heads, QWEN36_FULL_ATTN_KV_HEADS);
    assert_eq!(config.head_dim, QWEN36_FULL_ATTN_HEAD_DIM);
    assert_eq!(config.q_hidden_size(), Ok(QWEN36_FULL_ATTN_Q_HIDDEN));
    assert_eq!(config.kv_hidden_size(), Ok(QWEN36_FULL_ATTN_KV_HIDDEN));

    let engine = config.engine_config();
    assert_eq!(engine.num_layers, 1);
    assert_eq!(engine.num_q_heads, config.num_q_heads);
    assert_eq!(engine.num_kv_heads, config.num_kv_heads);
    assert_eq!(engine.head_dim, config.head_dim);
}

#[test]
fn loaded_qwen36_factories_request_every_manifest_target_once() {
    use std::collections::BTreeSet;

    let config = QwenConfig::qwen36_bf16_runtime(
        0,
        ptr::null_mut(),
        40,
        248_320,
        1.0e-6,
        10_000_000.0,
        0.0,
        8,
    )
    .unwrap();
    let mut seen = BTreeSet::new();
    let weights = QwenWeights::from_bf16_buffers(config, |layer, slot| {
        assert!(
            seen.insert((layer, slot)),
            "duplicate target {layer:?}/{slot}"
        );
        Ok(DeviceBuffer::empty(config.device_ordinal))
    })
    .unwrap();

    assert_eq!(seen.len(), 723);
    assert!(seen.contains(&(None, "token_embedding")));
    assert!(seen.contains(&(None, "final_norm")));
    assert!(seen.contains(&(None, "lm_head")));
    assert!(seen.contains(&(Some(0), "gdn.in_proj_qkv")));
    assert!(seen.contains(&(Some(3), "attn.q_proj")));
    assert!(seen.contains(&(Some(39), "mlp.shared.gate_score")));
    weights.validate_for(&config).unwrap();
}

#[test]
fn randomized_full_attention_weights_seed_qwen_norm_raw_weights_as_zero() {
    if !cuda_device_available() {
        return;
    }

    let config = QwenConfig::randomized_dense_tiny_fixture(0);
    let weights = QwenWeights::random_bf16(&config, 0x5153_3300_7177_6e6b).unwrap();
    let hidden = config.hidden_size as usize;
    let head_dim = config.head_dim as usize;

    assert_eq!(weights.final_norm.cap, hidden);
    assert_eq!(
        download_bf16(&weights.final_norm, config.stream, hidden),
        vec![0_u16; hidden]
    );

    match &weights.layers[0] {
        QwenLayerWeights::AttentionMlp(layer) => {
            assert_eq!(layer.attn_norm.cap, hidden);
            assert_eq!(layer.mlp_norm.cap, hidden);
            assert_eq!(layer.q_norm.cap, head_dim);
            assert_eq!(layer.k_norm.cap, head_dim);
            assert_eq!(
                layer.q_proj.cap,
                checked_usize_product(&[QWEN36_FULL_ATTN_Q_PROJ_OUT, config.hidden_size]).unwrap()
            );
            assert_eq!(
                download_bf16(&layer.attn_norm, config.stream, hidden),
                vec![0_u16; hidden]
            );
            assert_eq!(
                download_bf16(&layer.mlp_norm, config.stream, hidden),
                vec![0_u16; hidden]
            );
            assert_eq!(
                download_bf16(&layer.q_norm, config.stream, head_dim),
                vec![0_u16; head_dim]
            );
            assert_eq!(
                download_bf16(&layer.k_norm, config.stream, head_dim),
                vec![0_u16; head_dim]
            );
        }
        QwenLayerWeights::Gdn(_) => unreachable!(),
    }
}

#[test]
fn qwen36_gdn_weights_carry_post_attention_shared_moe() {
    let layer = QwenLayerWeights::Gdn(QwenGdnWeights {
        norm: empty_bf16_buffer(),
        in_proj: empty_bf16_buffer(),
        gate_proj: empty_bf16_buffer(),
        a_proj: empty_bf16_buffer(),
        b_proj: empty_bf16_buffer(),
        conv_weight: empty_bf16_buffer(),
        conv_bias: empty_bf16_buffer(),
        a_log: empty_bf16_buffer(),
        dt_bias: empty_bf16_buffer(),
        rms_weight: empty_bf16_buffer(),
        out_proj: empty_bf16_buffer(),
        mlp_norm: empty_bf16_buffer(),
        mlp: empty_qwen36_moe_mlp_weights(),
    });

    match &layer {
        QwenLayerWeights::Gdn(gdn) => {
            assert_eq!(
                gdn.mlp.validate_for(Some(QwenMoeConfig::qwen36_35b_a3b())),
                Ok(())
            );
        }
        QwenLayerWeights::AttentionMlp(_) => unreachable!(),
    }

    match &layer {
        QwenLayerWeights::Gdn(gdn) => match gdn.mlp {
            QwenMlpWeights::Moe {
                shared: Some(_), ..
            } => {}
            _ => panic!("GDN layers must carry shared MoE post-attention weights"),
        },
        QwenLayerWeights::AttentionMlp(_) => unreachable!(),
    }
}

#[test]
fn shared_moe_execution_produces_routed_and_shared_outputs() {
    if !cuda_device_available() {
        return;
    }

    let config = QwenConfig::randomized_shared_moe_tiny_fixture(0);
    let mut runner = ModelRunner::random_bf16(config, 0x5153_3300_5a5a_5a5a).unwrap();
    let rows = 1;
    let hidden = config.hidden_size;
    let hidden_len = hidden as usize;
    let moe = config.moe_config().unwrap();
    runner.scratch.ensure(&config, rows).unwrap();

    let input = constant_bf16_values(hidden_len, 1.0).unwrap();
    runner
        .scratch
        .attn_proj
        .upload(config.stream, &input)
        .unwrap();

    let router_proj = filled_bf16_buffer(
        config.device_ordinal,
        config.stream,
        checked_usize_product(&[moe.num_experts, hidden]).unwrap(),
        0.0,
    );
    let gate_up_proj = filled_bf16_buffer(
        config.device_ordinal,
        config.stream,
        checked_usize_product(&[moe.num_experts, 2, moe.moe_intermediate_size, hidden]).unwrap(),
        0.003,
    );
    let down_proj = filled_bf16_buffer(
        config.device_ordinal,
        config.stream,
        checked_usize_product(&[moe.num_experts, hidden, moe.moe_intermediate_size]).unwrap(),
        0.01,
    );

    unsafe {
        runner
            .execution()
            .unwrap()
            .1
            .execute_moe_mlp(rows, &router_proj, &gate_up_proj, &down_proj, None)
            .unwrap();
    }
    let routed = download_bf16(&runner.scratch.mlp_out, config.stream, hidden_len);
    assert!(has_nonzero_bf16(&routed));

    runner
        .scratch
        .attn_proj
        .upload(config.stream, &input)
        .unwrap();
    let shared_gate_proj = filled_bf16_buffer(
        config.device_ordinal,
        config.stream,
        checked_usize_product(&[moe.shared_expert_intermediate_size, hidden]).unwrap(),
        0.004,
    );
    let shared_up_proj = filled_bf16_buffer(
        config.device_ordinal,
        config.stream,
        checked_usize_product(&[moe.shared_expert_intermediate_size, hidden]).unwrap(),
        0.004,
    );
    let shared_down_proj = filled_bf16_buffer(
        config.device_ordinal,
        config.stream,
        checked_usize_product(&[hidden, moe.shared_expert_intermediate_size]).unwrap(),
        0.02,
    );
    let shared_expert_gate =
        filled_bf16_buffer(config.device_ordinal, config.stream, hidden_len, 0.02);
    let shared = QwenSharedExpertWeights {
        gate_proj: shared_gate_proj,
        up_proj: shared_up_proj,
        down_proj: shared_down_proj,
        shared_expert_gate: shared_expert_gate,
    };

    unsafe {
        runner
            .execution()
            .unwrap()
            .1
            .execute_moe_mlp(rows, &router_proj, &gate_up_proj, &down_proj, Some(&shared))
            .unwrap();
    }

    let shared_out = download_bf16(&runner.scratch.shared_out, config.stream, hidden_len);
    let combined = download_bf16(&runner.scratch.mlp_out, config.stream, hidden_len);
    let mut gate_logits = vec![0.0_f32; rows as usize];
    runner
        .scratch
        .shared_gate_logits
        .download(config.stream, &mut gate_logits)
        .unwrap();

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
    assert_eq!(engine.num_q_heads, config.num_q_heads);
    assert_eq!(engine.num_kv_heads, config.num_kv_heads);
    assert_eq!(engine.head_dim, config.head_dim);
    assert_eq!(config.num_q_heads, QWEN36_FULL_ATTN_Q_HEADS);
    assert_eq!(config.num_kv_heads, QWEN36_FULL_ATTN_KV_HEADS);
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
fn failed_rebuild_after_final_layer_preserves_prefix_and_continuation() {
    if !cuda_device_available() {
        return;
    }
    let config = QwenConfig::randomized_shared_moe_tiny_fixture(0);
    let seed = 0x5245_4255_494c_4401;
    let prompt = [1, 2, 3];
    let request_id = 41;
    let mut runner = ModelRunner::random_bf16(config, seed).unwrap();
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

    let mut control = ModelRunner::random_bf16(config, seed).unwrap();
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
        assert_eq!(self.lm_head_provider(), "triton");
        let vocab = self.config.vocab_size;
        let hidden = self.config.hidden_size;
        let mut reference = DeviceBuffer::<f32>::empty(self.config.device_ordinal);
        reference.ensure(vocab as usize).unwrap();
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
                    self.weights.lm_head.matrix(vocab, hidden).unwrap(),
                    reference.matrix(1, vocab).unwrap(),
                    self.qscb_workspace
                        .workspace(self.config.qscb_workspace_bytes)
                        .unwrap(),
                )
                .unwrap();
        }
        let mut expected = vec![0.0; vocab as usize];
        reference
            .download(self.config.stream, &mut expected)
            .unwrap();
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
        super::validate_token_ids(&[token], self.config.vocab_size)?;
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
        let final_norm = self.weights.final_norm.ptr;
        self.weights.final_norm.ptr = ptr::null_mut();
        let failed = self.run(QwenRequest {
            request_id,
            tokens: rewritten_prompt,
            max_new_tokens: 0,
        });
        self.weights.final_norm.ptr = final_norm;
        assert_eq!(failed.unwrap_err(), Status::InvalidArgument);
        assert_eq!(self.live_tokens(), live_before);
        assert_eq!(self.last_logits_row_for_test().unwrap(), logits_before);
    }
}
