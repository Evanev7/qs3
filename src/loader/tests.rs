use super::{
    DEFAULT_MAX_HEADER_BYTES, DEFAULT_MAX_JSON_BYTES, PINNED_UPLOAD_BUFFER_BYTES,
    PINNED_UPLOAD_BUFFER_COUNT,
    format::{
        Qwen36TextConfig, SafetensorsHeader, TensorFileMeta, TensorMeta, WeightLoadError,
        WeightTensorDType, WeightTensorSource, WeightTensorSpec, WeightTensorTarget,
        expected_qwen36_bf16_specs, parse_json_object, validate_qwen36_bf16_dir,
        validate_qwen36_bf16_tensor_table,
    },
    plan::{
        LoadedWeightPlan, LoadedWeightTensor, QwenBf16LoadPlan, QwenLoadPlanEntry, QwenLoadSource,
        execute_qwen36_bf16_load_plan,
    },
    transfer::{
        ManagedUmaBackend, PinnedUploadBackend, WeightFileRange, WeightLoadBackend,
        WeightLoadMemory, WeightLoadSpan, WeightTensorDesc, result_from_cuda,
    },
};
use crate::test_assets::{real_qwen36_model_dir, require_real_qwen36_model_dir};
use crate::{
    QWEN36_FULL_ATTN_HEAD_DIM, QWEN36_GDN_PACKED_DIM, QWEN36_HIDDEN_SIZE,
    engine::{DynDType, Status},
    ffi,
};
use std::collections::BTreeMap;
use std::ffi::c_void;
use std::{env, ptr, time::Instant};

mod scores;

const REAL_PROMPT: [i32; 4] = [1, 2, 3, 4];
const REAL_GENERATED: [i32; 4] = [5, 6, 24_218, 10];
const REAL_PREFILL_TOP_IDS: [i32; 8] = [5, 3, 2, 220, 61, 198, 26_972, 271];
const REAL_PREFILL_TOP_VALUES: [f32; 8] = [
    13.9375, 12.0625, 10.875, 10.75, 10.4375, 10.3125, 9.9375, 9.875,
];
const REAL_DECODE_TOP_IDS: [i32; 8] = [6, 9, 9_867, 61, 41_813, 1, 3, 14_482];
const REAL_DECODE_TOP_VALUES: [f32; 8] = [
    14.875, 11.25, 11.1875, 11.125, 10.8125, 10.5625, 10.5625, 10.5625,
];
const REAL_PREFILL_STABLE_PREFIX_LEN: usize = REAL_PREFILL_TOP_IDS.len();
// Reference decode ranks 5-7 are BF16-tied at 10.5625 with no cutoff
// margin, so exact cross-runtime ID membership is stable only through 5.
const REAL_DECODE_STABLE_PREFIX_LEN: usize = 5;
const REAL_LOGIT_ABS_TOL: f32 = 0.35;
const REAL_LOGIT_REL_TOL: f32 = 0.03;

fn assert_real_top_logits(
    label: &str,
    logits: &[f32],
    expected_ids: &[i32; 8],
    expected_values: &[f32; 8],
    expected_margin: f32,
    stable_prefix_len: usize,
) {
    assert!((1..=expected_ids.len()).contains(&stable_prefix_len));
    assert!(logits.iter().all(|value| value.is_finite()));
    let mut ids = (0..logits.len()).collect::<Vec<_>>();
    ids.sort_unstable_by(|lhs, rhs| {
        logits[*rhs]
            .total_cmp(&logits[*lhs])
            .then_with(|| lhs.cmp(rhs))
    });
    let got_ids = ids[..8]
        .iter()
        .map(|id| i32::try_from(*id).unwrap())
        .collect::<Vec<_>>();
    assert_eq!(
        &got_ids[..stable_prefix_len],
        &expected_ids[..stable_prefix_len],
        "{label} stable top-id prefix changed"
    );

    for (&id, &expected) in expected_ids.iter().zip(expected_values) {
        let got = logits[id as usize];
        let tolerance = REAL_LOGIT_ABS_TOL.max(REAL_LOGIT_REL_TOL * expected.abs());
        assert!(
            (got - expected).abs() <= tolerance,
            "{label} token {id}: got {got}, expected {expected}, tolerance {tolerance}"
        );
    }
    let got_margin = logits[expected_ids[0] as usize] - logits[expected_ids[1] as usize];
    assert!(expected_margin > 0.1);
    assert!(
        got_margin > 0.1,
        "{label} top margin collapsed to {got_margin}"
    );
    assert!(
        (got_margin - expected_margin).abs() <= 2.0 * REAL_LOGIT_ABS_TOL,
        "{label} margin: got {got_margin}, expected {expected_margin}"
    );
}

#[test]
fn real_logit_helper_allows_unstable_ids_after_stable_prefix() {
    let expected_ids = [0, 1, 2, 3, 4, 5, 6, 7];
    let expected_values = [2.0, 1.5, 1.2, 1.1, 1.0, 0.9, 0.8, 0.7];
    let logits = [2.0, 1.5, 1.2, 1.1, 1.0, 0.9, 0.8, 0.7, 0.95, 0.85];

    assert_real_top_logits(
        "stable prefix fixture",
        &logits,
        &expected_ids,
        &expected_values,
        0.5,
        5,
    );
}

#[test]
#[should_panic(expected = "token 7")]
fn real_logit_helper_checks_values_after_stable_prefix() {
    let expected_ids = [0, 1, 2, 3, 4, 5, 6, 7];
    let expected_values = [2.0, 1.5, 1.2, 1.1, 1.0, 0.9, 0.8, 0.7];
    let logits = [2.0, 1.5, 1.2, 1.1, 1.0, 0.9, 0.8, -1.0, 0.95, 0.85];

    assert_real_top_logits(
        "all values fixture",
        &logits,
        &expected_ids,
        &expected_values,
        0.5,
        5,
    );
}

fn synthetic_safetensors(header: &str, data_len: usize) -> Vec<u8> {
    let mut out = Vec::new();
    out.extend_from_slice(&(header.len() as u64).to_le_bytes());
    out.extend_from_slice(header.as_bytes());
    out.resize(out.len() + data_len, 0);
    out
}

fn layer_types_json(layers: usize) -> String {
    (0..layers)
        .map(|idx| {
            if idx % 4 == 3 {
                "\"full_attention\""
            } else {
                "\"linear_attention\""
            }
        })
        .collect::<Vec<_>>()
        .join(",")
}

fn nested_text_config_json(layers: usize, vocab: u32) -> String {
    format!(
        r#"{{
            "model_type": "qwen3_5_moe",
            "tie_word_embeddings": false,
            "text_config": {{
                "model_type": "qwen3_5_moe_text",
                "dtype": "bfloat16",
                "num_hidden_layers": {layers},
                "hidden_size": 2048,
                "vocab_size": {vocab},
                "num_attention_heads": 16,
                "num_key_value_heads": 2,
                "head_dim": 256,
                "num_experts": 256,
                "num_experts_per_tok": 8,
                "moe_intermediate_size": 512,
                "shared_expert_intermediate_size": 512,
                "linear_num_key_heads": 16,
                "linear_num_value_heads": 32,
                "linear_key_head_dim": 128,
                "linear_value_head_dim": 128,
                "linear_conv_kernel_dim": 4,
                "full_attention_interval": 4,
                "attn_output_gate": true,
                "rms_norm_eps": 1e-6,
                "rope_parameters": {{
                    "partial_rotary_factor": 0.25,
                    "rope_theta": 10000000
                }},
                "layer_types": [{}]
            }}
        }}"#,
        layer_types_json(layers)
    )
}

fn qwen_text_config(layers: usize, vocab: u32) -> Qwen36TextConfig {
    let root = parse_json_object(&nested_text_config_json(layers, vocab)).unwrap();
    Qwen36TextConfig::from_config_object(&root).unwrap()
}

#[derive(Default)]
struct TinyBackendState {
    take_calls: std::cell::Cell<usize>,
    drop_calls: std::cell::Cell<usize>,
    remaining_at_drop: std::cell::Cell<usize>,
}

struct TinyCudaBackend {
    device_ordinal: i32,
    allocations: Vec<WeightLoadSpan>,
    state: std::rc::Rc<TinyBackendState>,
}

impl TinyCudaBackend {
    fn new(device_ordinal: i32, state: std::rc::Rc<TinyBackendState>) -> Self {
        Self {
            device_ordinal,
            allocations: Vec::new(),
            state,
        }
    }
}

impl WeightLoadBackend for TinyCudaBackend {
    fn device_ordinal(&self) -> i32 {
        self.device_ordinal
    }

    fn allocations(&self) -> &[WeightLoadSpan] {
        &self.allocations
    }

    fn take_allocations(&mut self) -> Vec<WeightLoadSpan> {
        self.state.take_calls.set(self.state.take_calls.get() + 1);
        std::mem::take(&mut self.allocations)
    }

    fn alloc_tensor(&mut self, desc: WeightTensorDesc<'_>) -> Result<WeightLoadSpan, Status> {
        let mut ptr = ptr::null_mut();
        // This drop-only fixture never dereferences model weights, so it
        // allocates one BF16 element while retaining the logical span size.
        result_from_cuda(unsafe { ffi::cuda::cudaMalloc(&mut ptr, 2) })?;
        let span = WeightLoadSpan {
            ptr,
            bytes: desc.bytes,
            memory: WeightLoadMemory::Device,
        };
        self.allocations.push(span);
        Ok(span)
    }

    fn read_exact(
        &mut self,
        _src: WeightFileRange<'_>,
        _dst: &WeightLoadSpan,
        _stream: *mut c_void,
    ) -> Result<(), Status> {
        Ok(())
    }

    fn zero_fill(&mut self, _dst: &WeightLoadSpan, _stream: *mut c_void) -> Result<(), Status> {
        Ok(())
    }

    fn seal(&mut self, _stream: *mut c_void) -> Result<(), Status> {
        Ok(())
    }
}

impl Drop for TinyCudaBackend {
    fn drop(&mut self) {
        self.state.drop_calls.set(self.state.drop_calls.get() + 1);
        self.state.remaining_at_drop.set(self.allocations.len());
        for allocation in self.allocations.drain(..) {
            unsafe {
                ffi::cuda::cudaFree(allocation.ptr);
            }
        }
    }
}

#[derive(Default)]
struct AdversarialBackendState {
    take_calls: std::cell::Cell<usize>,
    drop_calls: std::cell::Cell<usize>,
    remaining_at_drop: std::cell::Cell<usize>,
}

#[derive(Clone, Copy)]
enum AdversarialTake {
    ReturnEmptyAndKeep,
    ReturnEmptyAndClear,
    ReturnEmptyThenRetain(WeightLoadSpan),
}

struct AdversarialBackend {
    allocations: Vec<WeightLoadSpan>,
    take: AdversarialTake,
    state: std::rc::Rc<AdversarialBackendState>,
}

impl WeightLoadBackend for AdversarialBackend {
    fn device_ordinal(&self) -> i32 {
        0
    }

    fn allocations(&self) -> &[WeightLoadSpan] {
        &self.allocations
    }

    fn take_allocations(&mut self) -> Vec<WeightLoadSpan> {
        self.state.take_calls.set(self.state.take_calls.get() + 1);
        match self.take {
            AdversarialTake::ReturnEmptyAndKeep => Vec::new(),
            AdversarialTake::ReturnEmptyAndClear => {
                self.allocations.clear();
                Vec::new()
            }
            AdversarialTake::ReturnEmptyThenRetain(allocation) => {
                self.allocations.push(allocation);
                Vec::new()
            }
        }
    }

    fn alloc_tensor(&mut self, _desc: WeightTensorDesc<'_>) -> Result<WeightLoadSpan, Status> {
        Err(Status::InternalError)
    }

    fn read_exact(
        &mut self,
        _src: WeightFileRange<'_>,
        _dst: &WeightLoadSpan,
        _stream: *mut c_void,
    ) -> Result<(), Status> {
        Err(Status::InternalError)
    }

    fn zero_fill(&mut self, _dst: &WeightLoadSpan, _stream: *mut c_void) -> Result<(), Status> {
        Err(Status::InternalError)
    }

    fn seal(&mut self, _stream: *mut c_void) -> Result<(), Status> {
        Err(Status::InternalError)
    }
}

impl Drop for AdversarialBackend {
    fn drop(&mut self) {
        self.state.drop_calls.set(self.state.drop_calls.get() + 1);
        self.state.remaining_at_drop.set(self.allocations.len());
    }
}

fn fake_span(address: usize, bytes: usize) -> WeightLoadSpan {
    WeightLoadSpan {
        ptr: address as ffi::DevicePtr,
        bytes,
        memory: WeightLoadMemory::Device,
    }
}

fn adversarial_loaded_plan(
    config: Qwen36TextConfig,
    specs: Vec<WeightTensorSpec>,
    spans: Vec<WeightLoadSpan>,
    take: AdversarialTake,
    state: std::rc::Rc<AdversarialBackendState>,
) -> LoadedWeightPlan<AdversarialBackend> {
    assert_eq!(specs.len(), spans.len());
    let tensors = specs
        .into_iter()
        .zip(spans.iter().copied())
        .map(|(spec, span)| LoadedWeightTensor { spec, span })
        .collect();
    LoadedWeightPlan {
        config,
        tensors,
        backend: AdversarialBackend {
            allocations: spans,
            take,
            state,
        },
    }
}

unsafe extern "C" {
    fn cudaGetDeviceCount(count: *mut i32) -> i32;
}

fn cuda_device_available_for_loader_test() -> bool {
    let mut count = 0;
    unsafe { cudaGetDeviceCount(&mut count) == ffi::cuda::CUDA_SUCCESS && count > 0 }
}

fn cuda_device_from_env() -> i32 {
    env::var("QS3_CUDA_DEVICE")
        .ok()
        .and_then(|value| value.parse::<i32>().ok())
        .unwrap_or(0)
}

fn tensor_table_from_specs(specs: &[WeightTensorSpec]) -> BTreeMap<String, TensorFileMeta> {
    let mut table = BTreeMap::new();
    let mut offset = 0u64;
    for spec in specs {
        if spec.source == WeightTensorSource::ZeroFill {
            continue;
        }
        let bytes = spec.byte_len().unwrap() as u64;
        table.insert(
            spec.name.clone(),
            TensorFileMeta {
                shard: "model-00001-of-00001.safetensors".to_owned(),
                absolute_offset: offset,
                meta: TensorMeta {
                    dtype: spec.dtype,
                    shape: spec.shape.clone(),
                    data_offsets: (offset, offset + bytes),
                },
            },
        );
        offset += bytes;
    }
    table
}

#[test]
fn parses_nested_qwen36_text_config_and_manifest() {
    let config = qwen_text_config(40, 248_320);
    assert_eq!(config.hidden_size, QWEN36_HIDDEN_SIZE);
    assert_eq!(config.rope_theta, 10_000_000.0);

    let specs = expected_qwen36_bf16_specs(&config).unwrap();
    assert_eq!(specs.len(), 723);
    assert!(specs.iter().any(|spec| {
        spec.name == "model.language_model.layers.3.self_attn.q_norm.weight"
            && spec.shape == vec![QWEN36_FULL_ATTN_HEAD_DIM]
    }));
    assert!(specs.iter().any(|spec| {
        spec.name == "model.language_model.layers.0.mlp.experts.gate_up_proj"
            && spec.shape == vec![256, 1024, 2048]
    }));
    assert!(specs.iter().any(|spec| {
        spec.name == "model.language_model.layers.0.linear_attn.conv1d.bias"
            && spec.source == WeightTensorSource::ZeroFill
    }));
    assert!(
        !specs
            .iter()
            .any(|spec| spec.name.ends_with("experts.gate_up_proj.weight"))
    );
}

#[test]
fn rejects_bad_layer_schedule() {
    let mut json = nested_text_config_json(4, 16);
    json = json.replacen("\"full_attention\"", "\"linear_attention\"", 1);
    let root = parse_json_object(&json).unwrap();
    let err = Qwen36TextConfig::from_config_object(&root).unwrap_err();
    assert!(matches!(err, WeightLoadError::InvalidConfig(_)));
    assert!(err.to_string().contains("layer 3"));
}

#[test]
fn parses_safetensors_header_and_rejects_trailing_data() {
    let header = r#"{"a":{"dtype":"BF16","shape":[2,3],"data_offsets":[0,12]},"b":{"dtype":"F32","shape":[1],"data_offsets":[12,16]}}"#;
    let parsed =
        SafetensorsHeader::parse_safetensors_bytes(&synthetic_safetensors(header, 16)).unwrap();
    assert_eq!(parsed.data_start, 8 + header.len() as u64);
    assert_eq!(parsed.tensors["a"].shape, vec![2, 3]);

    let err =
        SafetensorsHeader::parse_safetensors_bytes(&synthetic_safetensors(header, 17)).unwrap_err();
    assert!(matches!(err, WeightLoadError::InvalidSafetensors(_)));
    assert!(err.to_string().contains("trailing bytes"));
}

#[test]
fn rejects_safetensors_gaps_and_wrong_byte_count() {
    let gap = r#"{"a":{"dtype":"BF16","shape":[1],"data_offsets":[1,3]}}"#;
    let err =
        SafetensorsHeader::parse_safetensors_bytes(&synthetic_safetensors(gap, 3)).unwrap_err();
    assert!(err.to_string().contains("expected contiguous offset 0"));

    let wrong_size = r#"{"a":{"dtype":"BF16","shape":[2],"data_offsets":[0,2]}}"#;
    let err = SafetensorsHeader::parse_safetensors_bytes(&synthetic_safetensors(wrong_size, 2))
        .unwrap_err();
    assert!(err.to_string().contains("expected 4"));
}

#[test]
fn validates_complete_tensor_table_and_rejects_unexpected_text_tensor() {
    let config = qwen_text_config(4, 16);
    let specs = expected_qwen36_bf16_specs(&config).unwrap();
    let mut table = tensor_table_from_specs(&specs);
    let validated = validate_qwen36_bf16_tensor_table(&table, &config).unwrap();
    assert_eq!(validated.len(), specs.len());

    table.insert(
        "lm_head.input_scale".to_owned(),
        TensorFileMeta {
            shard: "model-00001-of-00001.safetensors".to_owned(),
            absolute_offset: 0,
            meta: TensorMeta {
                dtype: WeightTensorDType::F32,
                shape: vec![1],
                data_offsets: (0, 4),
            },
        },
    );
    let err = validate_qwen36_bf16_tensor_table(&table, &config).unwrap_err();
    assert!(err.to_string().contains("unexpected tensor"));
}

#[test]
fn rejects_missing_and_wrong_shape_required_tensor() {
    let config = qwen_text_config(4, 16);
    let specs = expected_qwen36_bf16_specs(&config).unwrap();
    let mut table = tensor_table_from_specs(&specs);
    table.remove("model.language_model.layers.3.self_attn.o_proj.weight");
    let err = validate_qwen36_bf16_tensor_table(&table, &config).unwrap_err();
    assert!(err.to_string().contains("missing required tensor"));

    let mut table = tensor_table_from_specs(&specs);
    table
        .get_mut("model.language_model.layers.0.linear_attn.A_log")
        .unwrap()
        .meta
        .shape = vec![31];
    let err = validate_qwen36_bf16_tensor_table(&table, &config).unwrap_err();
    assert!(err.to_string().contains("shape"));
}

#[derive(Default)]
struct RecordingBackend {
    next_addr: usize,
    allocations: Vec<WeightLoadSpan>,
    allocs: Vec<(String, DynDType, Vec<u32>, usize)>,
    reads: Vec<(u64, usize)>,
    zeros: Vec<usize>,
    sealed: bool,
    dropped: Option<std::rc::Rc<std::cell::Cell<bool>>>,
}

impl Drop for RecordingBackend {
    fn drop(&mut self) {
        if let Some(dropped) = &self.dropped {
            dropped.set(true);
        }
    }
}

impl WeightLoadBackend for RecordingBackend {
    fn device_ordinal(&self) -> i32 {
        0
    }

    fn allocations(&self) -> &[WeightLoadSpan] {
        &self.allocations
    }

    fn take_allocations(&mut self) -> Vec<WeightLoadSpan> {
        std::mem::take(&mut self.allocations)
    }

    fn alloc_tensor(&mut self, desc: WeightTensorDesc<'_>) -> Result<WeightLoadSpan, Status> {
        self.next_addr += 0x1000;
        self.allocs.push((
            desc.name.to_owned(),
            desc.dtype,
            desc.shape.to_vec(),
            desc.bytes,
        ));
        let span = WeightLoadSpan {
            ptr: self.next_addr as ffi::DevicePtr,
            bytes: desc.bytes,
            memory: WeightLoadMemory::ManagedUma,
        };
        self.allocations.push(span);
        Ok(span)
    }

    fn read_exact(
        &mut self,
        src: WeightFileRange<'_>,
        _dst: &WeightLoadSpan,
        _stream: *mut c_void,
    ) -> Result<(), Status> {
        self.reads.push((src.offset, src.bytes));
        Ok(())
    }

    fn zero_fill(&mut self, dst: &WeightLoadSpan, _stream: *mut c_void) -> Result<(), Status> {
        self.zeros.push(dst.bytes);
        Ok(())
    }

    fn seal(&mut self, _stream: *mut c_void) -> Result<(), Status> {
        self.sealed = true;
        Ok(())
    }
}

#[test]
fn executes_validated_plan_against_backend() {
    let tmp = std::env::temp_dir().join(format!("qs3-weight-loader-test-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&tmp);
    std::fs::create_dir(&tmp).unwrap();
    let shard = tmp.join("model-00001-of-00001.safetensors");
    std::fs::write(&shard, [0u8; 8]).unwrap();

    let plan = QwenBf16LoadPlan {
        config: qwen_text_config(4, 16),
        entries: vec![
            QwenLoadPlanEntry {
                spec: WeightTensorSpec {
                    name: "a".to_owned(),
                    dtype: WeightTensorDType::Bf16,
                    shape: vec![2],
                    source: WeightTensorSource::Safetensors,
                    target: WeightTensorTarget {
                        layer: None,
                        slot: "a",
                    },
                },
                source: QwenLoadSource::FileRange {
                    shard_path: shard,
                    absolute_offset: 8,
                    bytes: 4,
                },
            },
            QwenLoadPlanEntry {
                spec: WeightTensorSpec {
                    name: "b".to_owned(),
                    dtype: WeightTensorDType::Bf16,
                    shape: vec![4],
                    source: WeightTensorSource::ZeroFill,
                    target: WeightTensorTarget {
                        layer: None,
                        slot: "b",
                    },
                },
                source: QwenLoadSource::ZeroFill { bytes: 8 },
            },
        ],
    };
    let dropped = std::rc::Rc::new(std::cell::Cell::new(false));
    let mut backend = RecordingBackend::default();
    backend.dropped = Some(dropped.clone());
    let loaded = execute_qwen36_bf16_load_plan(&plan, backend, ptr::null_mut()).unwrap();

    assert_eq!(loaded.tensors.len(), 2);
    assert_eq!(loaded.backend.allocs.len(), 2);
    assert_eq!(loaded.backend.reads, vec![(8, 4)]);
    assert_eq!(loaded.backend.zeros, vec![8]);
    assert!(loaded.backend.sealed);
    assert!(!dropped.get());
    drop(loaded);
    assert!(dropped.get());

    let _ = std::fs::remove_dir_all(&tmp);
}

#[test]
fn loaded_plan_transfers_allocations_into_qwen_weights_once() {
    if !cuda_device_available_for_loader_test() {
        return;
    }
    let text = qwen_text_config(4, 32);
    let entries = expected_qwen36_bf16_specs(&text)
        .unwrap()
        .into_iter()
        .map(|spec| {
            let bytes = spec.byte_len().unwrap();
            QwenLoadPlanEntry {
                spec,
                source: QwenLoadSource::ZeroFill { bytes },
            }
        })
        .collect::<Vec<_>>();
    let plan = QwenBf16LoadPlan {
        config: text,
        entries,
    };
    let count = plan.tensor_count();
    let state = std::rc::Rc::new(TinyBackendState::default());
    let backend = TinyCudaBackend::new(0, state.clone());

    let loaded = execute_qwen36_bf16_load_plan(&plan, backend, ptr::null_mut()).unwrap();
    let (config, weights) = loaded.into_qwen_model(ptr::null_mut(), 8).unwrap();

    assert_eq!(count, 75);
    assert_eq!(state.take_calls.get(), 1);
    assert_eq!(state.drop_calls.get(), 1);
    assert_eq!(state.remaining_at_drop.get(), 0);
    weights.validate_for(&config).unwrap();
    drop(weights);
}

#[test]
fn materialization_rejects_duplicate_targets_before_transfer() {
    let text = qwen_text_config(4, 32);
    let mut specs = expected_qwen36_bf16_specs(&text)
        .unwrap()
        .into_iter()
        .take(2)
        .collect::<Vec<_>>();
    specs[1].target = specs[0].target.clone();
    let spans = specs
        .iter()
        .enumerate()
        .map(|(idx, spec)| fake_span(0x1000 + idx * 0x1000, spec.byte_len().unwrap()))
        .collect();
    let state = std::rc::Rc::new(AdversarialBackendState::default());
    let loaded = adversarial_loaded_plan(
        text,
        specs,
        spans,
        AdversarialTake::ReturnEmptyAndKeep,
        state.clone(),
    );

    let err = loaded
        .into_qwen_model(ptr::null_mut(), 8)
        .err()
        .expect("duplicate target must fail materialization");

    assert!(err.to_string().contains("duplicate loaded target"));
    assert_eq!(state.take_calls.get(), 0);
    assert_eq!(state.drop_calls.get(), 1);
    assert_eq!(state.remaining_at_drop.get(), 2);
}

#[test]
fn materialization_rejects_duplicate_pointers_before_transfer() {
    let text = qwen_text_config(4, 32);
    let specs = expected_qwen36_bf16_specs(&text)
        .unwrap()
        .into_iter()
        .take(2)
        .collect::<Vec<_>>();
    let spans = specs
        .iter()
        .map(|spec| fake_span(0x1000, spec.byte_len().unwrap()))
        .collect();
    let state = std::rc::Rc::new(AdversarialBackendState::default());
    let loaded = adversarial_loaded_plan(
        text,
        specs,
        spans,
        AdversarialTake::ReturnEmptyAndKeep,
        state.clone(),
    );

    let err = loaded
        .into_qwen_model(ptr::null_mut(), 8)
        .err()
        .expect("duplicate pointer must fail materialization");

    assert!(
        err.to_string()
            .contains("duplicate loaded allocation pointer")
    );
    assert_eq!(state.take_calls.get(), 0);
    assert_eq!(state.drop_calls.get(), 1);
    assert_eq!(state.remaining_at_drop.get(), 2);
}

#[test]
fn materialization_rejects_changed_taken_allocations_before_adoption() {
    let text = qwen_text_config(4, 32);
    let specs = expected_qwen36_bf16_specs(&text)
        .unwrap()
        .into_iter()
        .take(1)
        .collect::<Vec<_>>();
    let spans = vec![fake_span(0x1000, specs[0].byte_len().unwrap())];
    let state = std::rc::Rc::new(AdversarialBackendState::default());
    let loaded = adversarial_loaded_plan(
        text,
        specs,
        spans,
        AdversarialTake::ReturnEmptyAndClear,
        state.clone(),
    );

    let err = loaded
        .into_qwen_model(ptr::null_mut(), 8)
        .err()
        .expect("changed allocation list must fail materialization");

    assert!(
        err.to_string()
            .contains("returned allocation list does not match validated allocations")
    );
    assert_eq!(state.take_calls.get(), 1);
    assert_eq!(state.drop_calls.get(), 1);
    assert_eq!(state.remaining_at_drop.get(), 0);
}

#[test]
fn materialization_rejects_backend_that_retains_allocations_after_take() {
    let text = qwen_text_config(4, 32);
    let state = std::rc::Rc::new(AdversarialBackendState::default());
    let loaded = adversarial_loaded_plan(
        text,
        Vec::new(),
        Vec::new(),
        AdversarialTake::ReturnEmptyThenRetain(fake_span(0x1000, 2)),
        state.clone(),
    );

    let err = loaded
        .into_qwen_model(ptr::null_mut(), 8)
        .err()
        .expect("retained allocation list must fail materialization");

    assert!(
        err.to_string()
            .contains("backend retained allocations after transfer")
    );
    assert_eq!(state.take_calls.get(), 1);
    assert_eq!(state.drop_calls.get(), 1);
    assert_eq!(state.remaining_at_drop.get(), 1);
}

#[test]
fn validates_real_qwen36_bf16_manifest_when_available() {
    let model_dir = real_qwen36_model_dir();
    if !model_dir.is_dir() {
        eprintln!(
            "skipping real Qwen3.6 BF16 manifest smoke; {} does not exist",
            model_dir.display()
        );
        return;
    }

    let started = Instant::now();
    let validated =
        validate_qwen36_bf16_dir(&model_dir, DEFAULT_MAX_JSON_BYTES, DEFAULT_MAX_HEADER_BYTES)
            .unwrap();
    let plan = QwenBf16LoadPlan::read(&model_dir).unwrap();
    assert_eq!(validated.tensor_count(), plan.tensor_count());
    let file_bytes = plan.file_bytes().unwrap();
    let zero_fill_bytes = plan.zero_fill_bytes().unwrap();
    println!(
        "validated {} tensors from {} in {:.3}s: {:.3} GiB file bytes, {:.3} MiB zero-fill",
        plan.tensor_count(),
        model_dir.display(),
        started.elapsed().as_secs_f64(),
        file_bytes as f64 / (1u64 << 30) as f64,
        zero_fill_bytes as f64 / (1u64 << 20) as f64
    );

    assert_eq!(plan.config.num_hidden_layers, 40);
    assert_eq!(plan.tensor_count(), 723);
    assert_eq!(zero_fill_bytes, 30 * QWEN36_GDN_PACKED_DIM as usize * 2);
    assert!(file_bytes > 60usize << 30);
}

#[test]
#[ignore = "loads and executes the full real Qwen3.6 BF16 model"]
fn real_qwen36_bf16_generates_reference_tokens() {
    for precision in [
        crate::model::GdnRecurrentPrecision::Bf16,
        crate::model::GdnRecurrentPrecision::F32,
    ] {
        eprintln!("real BF16 model with {} GDN recurrence", precision.as_str());
        check_real_bf16_reference(precision);
    }
}

fn check_real_bf16_reference(precision: crate::model::GdnRecurrentPrecision) {
    let model_dir = require_real_qwen36_model_dir();
    let plan = QwenBf16LoadPlan::read(model_dir).unwrap();
    let backend = ManagedUmaBackend::new(cuda_device_from_env()).unwrap();
    let loaded = execute_qwen36_bf16_load_plan(&plan, backend, ptr::null_mut()).unwrap();
    let (mut config, weights) = loaded.into_qwen_model(ptr::null_mut(), 8).unwrap();
    config.gdn_recurrent_precision = precision;
    let mut runner = crate::model::ModelRunner::new(config, weights).unwrap();
    let request_id = 0xBF16_0001;

    let prefill = runner
        .run(crate::model::QwenRequest {
            request_id,
            tokens: &REAL_PROMPT,
            max_new_tokens: 0,
        })
        .unwrap();
    assert!(prefill.generated_tokens.is_empty());
    assert_real_top_logits(
        "real prefill",
        &runner.last_logits_row_for_test().unwrap(),
        &REAL_PREFILL_TOP_IDS,
        &REAL_PREFILL_TOP_VALUES,
        1.875,
        REAL_PREFILL_STABLE_PREFIX_LEN,
    );

    let first = runner
        .run(crate::model::QwenRequest {
            request_id,
            tokens: &REAL_PROMPT,
            max_new_tokens: 1,
        })
        .unwrap();
    assert_eq!(first.generated_tokens, vec![REAL_GENERATED[0]]);
    assert_real_top_logits(
        "real first decode",
        &runner.last_logits_row_for_test().unwrap(),
        &REAL_DECODE_TOP_IDS,
        &REAL_DECODE_TOP_VALUES,
        3.625,
        REAL_DECODE_STABLE_PREFIX_LEN,
    );

    runner.assert_late_rebuild_failure_preserves_prefix(request_id, &[7, 6, 5, 4]);

    let tail = runner
        .run(crate::model::QwenRequest {
            request_id,
            tokens: &first.live_tokens,
            max_new_tokens: 3,
        })
        .unwrap();
    assert_eq!(tail.generated_tokens, REAL_GENERATED[1..]);

    runner.reset().unwrap();
    let replay = runner
        .run(crate::model::QwenRequest {
            request_id,
            tokens: &REAL_PROMPT,
            max_new_tokens: 4,
        })
        .unwrap();
    assert_eq!(replay.generated_tokens, REAL_GENERATED);
}

#[test]
#[ignore = "reads the full real BF16 model into CUDA managed memory"]
fn bench_real_qwen36_bf16_managed_uma_load() {
    let model_dir = require_real_qwen36_model_dir();
    let plan_started = Instant::now();
    let plan = QwenBf16LoadPlan::read(&model_dir).unwrap();
    println!(
        "planned {} tensors in {:.3}s from {}",
        plan.tensor_count(),
        plan_started.elapsed().as_secs_f64(),
        model_dir.display()
    );

    let backend = ManagedUmaBackend::new(cuda_device_from_env()).unwrap();

    let load_started = Instant::now();
    let loaded = execute_qwen36_bf16_load_plan(&plan, backend, ptr::null_mut()).unwrap();
    let elapsed = load_started.elapsed();
    let loaded_tensors = loaded.tensors.len();
    let stats = loaded.backend.stats();
    let read_gib = stats.read_bytes as f64 / (1u64 << 30) as f64;
    println!(
        "loaded {} tensors: {:.3} GiB read, {:.3} MiB zero-fill, {:.3}s, {:.3} GiB/s",
        loaded_tensors,
        read_gib,
        stats.zero_fill_bytes as f64 / (1u64 << 20) as f64,
        elapsed.as_secs_f64(),
        read_gib / elapsed.as_secs_f64()
    );
    println!(
        "phases: alloc {:.3}s, read {:.3}s ({:.3} GiB/s), zero {:.6}s, seal {:.6}s",
        stats.alloc_us as f64 / 1_000_000.0,
        stats.read_us as f64 / 1_000_000.0,
        read_gib / (stats.read_us as f64 / 1_000_000.0),
        stats.zero_fill_us as f64 / 1_000_000.0,
        stats.seal_us as f64 / 1_000_000.0
    );

    assert_eq!(loaded_tensors, plan.tensor_count());
    assert_eq!(stats.tensors, plan.tensor_count());
    assert_eq!(stats.read_bytes, plan.file_bytes().unwrap());
    assert_eq!(stats.zero_fill_bytes, plan.zero_fill_bytes().unwrap());
    assert_eq!(stats.allocated_bytes, plan.total_bytes().unwrap());
    drop(loaded);
}

#[test]
#[ignore = "reads the full real BF16 model through pinned staging into CUDA device memory"]
fn bench_real_qwen36_bf16_pinned_upload_load() {
    let model_dir = require_real_qwen36_model_dir();
    let plan_started = Instant::now();
    let plan = QwenBf16LoadPlan::read(&model_dir).unwrap();
    println!(
        "planned {} tensors in {:.3}s from {}",
        plan.tensor_count(),
        plan_started.elapsed().as_secs_f64(),
        model_dir.display()
    );

    let backend = PinnedUploadBackend::new(cuda_device_from_env()).unwrap();

    let load_started = Instant::now();
    let loaded = execute_qwen36_bf16_load_plan(&plan, backend, ptr::null_mut()).unwrap();
    let elapsed = load_started.elapsed();
    let loaded_tensors = loaded.tensors.len();
    let stats = loaded.backend.stats();
    let read_gib = stats.read_bytes as f64 / (1u64 << 30) as f64;
    println!(
        "loaded {} tensors: {:.3} GiB read, {:.3} MiB zero-fill, {:.3}s, {:.3} GiB/s",
        loaded_tensors,
        read_gib,
        stats.zero_fill_bytes as f64 / (1u64 << 20) as f64,
        elapsed.as_secs_f64(),
        read_gib / elapsed.as_secs_f64()
    );
    println!(
        "phases: alloc {:.3}s, wait {:.3}s, read {:.3}s ({:.3} GiB/s), copy-enqueue {:.3}s, zero {:.6}s, seal {:.6}s, chunks {}, buffers {} x {:.0} MiB",
        stats.alloc_us as f64 / 1_000_000.0,
        stats.wait_us as f64 / 1_000_000.0,
        stats.read_us as f64 / 1_000_000.0,
        read_gib / (stats.read_us as f64 / 1_000_000.0),
        stats.copy_enqueue_us as f64 / 1_000_000.0,
        stats.zero_fill_us as f64 / 1_000_000.0,
        stats.seal_us as f64 / 1_000_000.0,
        stats.chunks,
        PINNED_UPLOAD_BUFFER_COUNT,
        PINNED_UPLOAD_BUFFER_BYTES as f64 / (1u64 << 20) as f64
    );

    assert_eq!(loaded_tensors, plan.tensor_count());
    assert_eq!(stats.tensors, plan.tensor_count());
    assert_eq!(stats.read_bytes, plan.file_bytes().unwrap());
    assert_eq!(stats.zero_fill_bytes, plan.zero_fill_bytes().unwrap());
    assert_eq!(stats.allocated_bytes, plan.total_bytes().unwrap());
    drop(loaded);
}
