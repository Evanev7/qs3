use super::schema::expected_specs;
use super::{
    PINNED_UPLOAD_BUFFER_BYTES, PINNED_UPLOAD_BUFFER_COUNT,
    format::{
        Qwen36TextConfig, SafetensorsHeader, TensorFileMeta, TensorMeta, WeightLoadError,
        WeightTensorSource, WeightTensorSpec, parse_json_object, validate_tensor_specs,
    },
    plan::{QwenLoadPlan, QwenLoadPlanEntry, QwenLoadSource, execute_qwen_load_plan},
    transfer::{
        ManagedUmaBackend, PinnedUploadBackend, WeightBuffer, WeightFileRange, WeightLoadBackend,
        WeightLoadMemory, WeightLoadSpan, WeightTensorDesc, result_from_cuda,
    },
};
use crate::dtype::{BF16, DType, F32, Fp8E4M3, U8};
use crate::test_assets::{real_selected_model_dir, require_real_qwen36_model_dir};
use crate::{dtype::DynDType, engine::Status, ffi, memory::DeviceSpan};
use std::collections::BTreeMap;
use std::ffi::c_void;
use std::{env, ptr, time::Instant};

mod checkpoint;
mod quantization;
mod scores;
mod tensor;

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
const REAL_PREFILL_STABLE_TOP_COUNT: usize = REAL_PREFILL_TOP_IDS.len();
// Reference decode ranks 5-7 are BF16-tied at 10.5625 with no cutoff
// margin, so exact cross-runtime ID membership is stable only through 5.
// Secondary ordering is not fixed: BF16 router-output rounding can swap scores
// within the existing numerical tolerances. Keep the winner and top-set checks.
const REAL_DECODE_STABLE_TOP_COUNT: usize = 5;
const REAL_LOGIT_ABS_TOL: f32 = 0.35;
const REAL_LOGIT_REL_TOL: f32 = 0.03;

fn assert_real_top_logits(
    label: &str,
    logits: &[f32],
    expected_ids: &[i32; 8],
    expected_values: &[f32; 8],
    expected_margin: f32,
    stable_top_count: usize,
) {
    assert!((1..=expected_ids.len()).contains(&stable_top_count));
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
    assert_eq!(got_ids[0], expected_ids[0], "{label} greedy winner changed");
    let mut got_top = got_ids[..stable_top_count].to_vec();
    let mut expected_top = expected_ids[..stable_top_count].to_vec();
    got_top.sort_unstable();
    expected_top.sort_unstable();
    assert_eq!(
        got_top, expected_top,
        "{label} stable top-id membership changed"
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

#[test]
fn real_logit_helper_allows_close_secondary_rank_swaps() {
    let expected_ids = [0, 1, 2, 3, 4, 5, 6, 7];
    let expected_values = [3.0, 1.5, 1.4, 1.1, 1.0, 0.9, 0.8, 0.7];
    let logits = [3.0, 1.4, 1.5, 1.1, 1.0, 0.9, 0.8, 0.7];
    assert_real_top_logits(
        "close ranks",
        &logits,
        &expected_ids,
        &expected_values,
        1.5,
        5,
    );
}

#[test]
#[should_panic(expected = "greedy winner changed")]
fn real_logit_helper_rejects_changed_winner() {
    let expected_ids = [0, 1, 2, 3, 4, 5, 6, 7];
    let expected_values = [2.0, 1.8, 1.4, 1.1, 1.0, 0.9, 0.8, 0.7];
    let logits = [1.8, 2.0, 1.4, 1.1, 1.0, 0.9, 0.8, 0.7];
    assert_real_top_logits(
        "changed winner",
        &logits,
        &expected_ids,
        &expected_values,
        0.2,
        5,
    );
}

#[test]
#[should_panic(expected = "stable top-id membership changed")]
fn real_logit_helper_rejects_changed_top_membership() {
    let expected_ids = [0, 1, 2, 3, 4, 5, 6, 7];
    let expected_values = [3.0, 1.5, 1.4, 1.1, 1.0, 0.9, 0.8, 0.7];
    let logits = [3.0, 1.5, 1.4, 1.1, 1.0, 1.05, 0.8, 0.7];
    assert_real_top_logits(
        "changed membership",
        &logits,
        &expected_ids,
        &expected_values,
        1.5,
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

fn nested_text_config_json(layers: usize, vocab: u32) -> String {
    let mut root = checkpoint::selected_config();
    let text = checkpoint::text(&mut root);
    text.insert("num_hidden_layers".into(), (layers as f64).into());
    text.insert("vocab_size".into(), (vocab as f64).into());
    let schedule = text
        .get_mut("layer_types")
        .unwrap()
        .get_mut::<Vec<tinyjson::JsonValue>>()
        .unwrap();
    assert!(layers <= schedule.len());
    schedule.truncate(layers);
    tinyjson::JsonValue::Object(root).stringify().unwrap()
}

fn qwen_text_config(layers: usize, vocab: u32) -> Qwen36TextConfig {
    let root = parse_json_object(&nested_text_config_json(layers, vocab)).unwrap();
    Qwen36TextConfig::from_config_object(&root).unwrap()
}

#[test]
fn compiled_model_rejects_checkpoint_geometry_and_math_mismatches() {
    use crate::constants::{attention, model};
    let valid = qwen_text_config(model::NUM_HIDDEN_LAYERS as usize, model::VOCAB_SIZE);
    valid.validate_compiled_model().unwrap();
    let mut wrong_layers = valid.clone();
    wrong_layers.num_hidden_layers -= 4;
    wrong_layers
        .layer_types
        .truncate(wrong_layers.num_hidden_layers as usize);
    let mut wrong_vocab = valid.clone();
    wrong_vocab.vocab_size -= 1;
    let mut wrong_norm = valid.clone();
    wrong_norm.rms_norm_eps *= 2.0;
    let mut wrong_rope = valid.clone();
    wrong_rope.rope_theta = attention::ROPE_THETA * 2.0;
    let mut wrong_soft_cap = valid.clone();
    wrong_soft_cap.logits_soft_cap = 2.0;
    for wrong in [
        wrong_layers,
        wrong_vocab,
        wrong_norm,
        wrong_rope,
        wrong_soft_cap,
    ] {
        assert!(matches!(
            wrong.validate_compiled_model(),
            Err(WeightLoadError::InvalidConfig(_))
        ));
    }
}

#[test]
fn incompatible_checkpoint_is_rejected_before_reading_weight_index() {
    let directory =
        std::env::temp_dir().join(format!("qs3-compiled-model-reject-{}", std::process::id()));
    std::fs::create_dir_all(&directory).unwrap();
    std::fs::write(
        directory.join("config.json"),
        nested_text_config_json(4, 32),
    )
    .unwrap();
    // No index or weights exist. Reject the config before attempting to open them.
    assert!(matches!(
        QwenLoadPlan::read(&directory),
        Err(WeightLoadError::InvalidConfig(_))
    ));
    std::fs::remove_dir_all(directory).unwrap();
}

#[derive(Default)]
struct TinyBackendState {
    transferred: std::cell::Cell<usize>,
    drop_calls: std::cell::Cell<usize>,
    remaining_at_drop: std::cell::Cell<usize>,
    allocation_drops: std::cell::Cell<usize>,
}

// A test backend's allocation policy is retained through QwenWeights' boxes.
// Synthetic backends expose metadata only; TinyCudaBackend owns real CUDA storage.
struct TestWeightBuffer<D: DType> {
    span: DeviceSpan<D>,
    state: Option<std::rc::Rc<TinyBackendState>>,
    owner_drops: Option<std::rc::Rc<std::cell::Cell<usize>>>,
}
impl<D: DType> std::ops::Deref for TestWeightBuffer<D> {
    type Target = DeviceSpan<D>;
    fn deref(&self) -> &Self::Target {
        &self.span
    }
}
impl<D: DType> Drop for TestWeightBuffer<D> {
    fn drop(&mut self) {
        if let Some(count) = &self.owner_drops {
            count.set(count.get() + 1);
        }
        if let Some(state) = &self.state {
            state.allocation_drops.set(state.allocation_drops.get() + 1);
            unsafe {
                ffi::cuda::cudaFree(self.span.erase());
            }
        }
    }
}

struct TinyCudaBackend {
    device_ordinal: i32,
    fail_after: Option<usize>,
    allocations: Vec<WeightLoadSpan>,
    state: std::rc::Rc<TinyBackendState>,
}

impl TinyCudaBackend {
    fn new(device_ordinal: i32, state: std::rc::Rc<TinyBackendState>) -> Self {
        Self {
            device_ordinal,
            fail_after: None,
            allocations: Vec::new(),
            state,
        }
    }
}

impl WeightLoadBackend for TinyCudaBackend {
    type Buffer<D: DType> = TestWeightBuffer<D>;
    type Stats = ();

    fn alloc_tensor(&mut self, desc: WeightTensorDesc<'_>) -> Result<WeightLoadSpan, Status> {
        assert_eq!(desc.dtype, DynDType::BF16);
        result_from_cuda(unsafe { ffi::cuda::cudaSetDevice(self.device_ordinal) })?;
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

    fn finish(mut self, _stream: *mut c_void) -> Result<(Vec<WeightBuffer<Self>>, ()), Status> {
        let mut buffers = Vec::new();
        while let Some(&allocation) = self.allocations.last() {
            if self.fail_after == Some(buffers.len()) {
                return Err(Status::BackendError);
            }
            let span = DeviceSpan::new(allocation.ptr.cast(), BF16::len_of(allocation.bytes)?)?;
            self.allocations.pop();
            buffers.push(WeightBuffer::Bf16(TestWeightBuffer {
                span,
                state: Some(self.state.clone()),
                owner_drops: None,
            }));
            self.state.transferred.set(self.state.transferred.get() + 1);
        }
        buffers.reverse();
        Ok((buffers, ()))
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
fn manifest_preserves_checkpoint_names_shapes_and_synthetic_bias() {
    let root = checkpoint::selected_config();
    let config = Qwen36TextConfig::from_config_object(&root).unwrap();
    let specs = expected_specs(&config, None).unwrap();
    let by_name: BTreeMap<_, _> = specs
        .iter()
        .map(|spec| (spec.name.as_str(), spec))
        .collect();
    assert_eq!(by_name.len(), specs.len(), "duplicate checkpoint names");
    let prefix = "model.language_model.layers.0";
    let (name, shape) = if config.num_experts != 0 {
        (
            format!("{prefix}.mlp.experts.gate_up_proj"),
            vec![
                config.num_experts,
                2 * config.moe_intermediate_size,
                config.hidden_size,
            ],
        )
    } else {
        (
            format!("{prefix}.mlp.gate_proj.weight"),
            vec![config.intermediate_size, config.hidden_size],
        )
    };
    assert_eq!(by_name[name.as_str()].shape, shape);
    assert_eq!(
        by_name["model.language_model.layers.3.self_attn.q_norm.weight"].shape,
        vec![config.head_dim]
    );
    let bias = by_name[format!("{prefix}.linear_attn.conv1d.bias").as_str()];
    assert_eq!(bias.source, WeightTensorSource::ZeroFill);
    assert!(!by_name.contains_key(format!("{prefix}.mlp.experts.gate_up_proj.weight").as_str()));
}

#[test]
fn materialization_consumes_every_planned_tensor_once() {
    let plan = bf16_zero_plan(tensor::selected_text_config());
    let count = plan.tensor_count();
    let drops = std::rc::Rc::new(std::cell::Cell::new(0));
    let mut backend = RecordingBackend::default();
    backend.owner_drops = Some(drops.clone());
    let loaded = execute_qwen_load_plan(&plan, backend, ptr::null_mut()).unwrap();
    assert_eq!(loaded.stats.allocs.len(), count);
    let (_, weights) = loaded.into_qwen_model(8).unwrap();
    assert_eq!(drops.get(), 0);
    drop(weights);
    assert_eq!(drops.get(), count);
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
    let specs = expected_specs(&config, None).unwrap();
    let mut table = tensor_table_from_specs(&specs);
    let validated = validate_tensor_specs(&table, expected_specs(&config, None).unwrap()).unwrap();
    assert_eq!(validated.len(), specs.len());

    table.insert(
        "lm_head.input_scale".to_owned(),
        TensorFileMeta {
            shard: "model-00001-of-00001.safetensors".to_owned(),
            absolute_offset: 0,
            meta: TensorMeta {
                dtype: DynDType::F32,
                shape: vec![1],
                data_offsets: (0, 4),
            },
        },
    );
    let err = validate_tensor_specs(&table, expected_specs(&config, None).unwrap()).unwrap_err();
    assert!(err.to_string().contains("unexpected tensor"));
}

#[test]
fn rejects_missing_and_wrong_shape_required_tensor() {
    let config = qwen_text_config(4, 16);
    let specs = expected_specs(&config, None).unwrap();
    let mut table = tensor_table_from_specs(&specs);
    table.remove("model.language_model.layers.3.self_attn.o_proj.weight");
    let err = validate_tensor_specs(&table, expected_specs(&config, None).unwrap()).unwrap_err();
    assert!(err.to_string().contains("missing required tensor"));

    let mut table = tensor_table_from_specs(&specs);
    table
        .get_mut("model.language_model.layers.0.linear_attn.A_log")
        .unwrap()
        .meta
        .shape[0] += 1;
    let err = validate_tensor_specs(&table, expected_specs(&config, None).unwrap()).unwrap_err();
    assert!(err.to_string().contains("shape"));
}

#[derive(Default)]
struct RecordingStats {
    allocs: Vec<(String, DynDType, Vec<u32>, usize)>,
    reads: Vec<(u64, usize)>,
    zeros: Vec<usize>,
}

#[derive(Clone, Copy, Default)]
enum FinishFault {
    #[default]
    None,
    Reorder,
    Missing,
    WrongDtype,
    WrongSize,
}

#[derive(Default)]
struct RecordingBackend {
    next_addr: usize,
    allocations: Vec<(DynDType, WeightLoadSpan)>,
    stats: RecordingStats,
    fault: FinishFault,
    dropped: Option<std::rc::Rc<std::cell::Cell<bool>>>,
    owner_drops: Option<std::rc::Rc<std::cell::Cell<usize>>>,
}

impl Drop for RecordingBackend {
    fn drop(&mut self) {
        if let Some(dropped) = &self.dropped {
            dropped.set(true);
        }
    }
}

impl WeightLoadBackend for RecordingBackend {
    type Buffer<D: DType> = TestWeightBuffer<D>;
    type Stats = RecordingStats;

    fn alloc_tensor(&mut self, desc: WeightTensorDesc<'_>) -> Result<WeightLoadSpan, Status> {
        self.next_addr += 0x1000;
        self.stats.allocs.push((
            desc.name.to_owned(),
            desc.dtype,
            desc.shape.to_vec(),
            desc.bytes,
        ));
        let span = WeightLoadSpan {
            ptr: self.next_addr as ffi::ErasedDevicePtr,
            bytes: desc.bytes,
            memory: WeightLoadMemory::ManagedUma,
        };
        self.allocations.push((desc.dtype, span));
        Ok(span)
    }

    fn read_exact(
        &mut self,
        src: WeightFileRange<'_>,
        _dst: &WeightLoadSpan,
        _stream: *mut c_void,
    ) -> Result<(), Status> {
        self.stats.reads.push((src.offset, src.bytes));
        Ok(())
    }

    fn zero_fill(&mut self, dst: &WeightLoadSpan, _stream: *mut c_void) -> Result<(), Status> {
        self.stats.zeros.push(dst.bytes);
        Ok(())
    }

    fn finish(
        mut self,
        _stream: *mut c_void,
    ) -> Result<(Vec<WeightBuffer<Self>>, Self::Stats), Status> {
        fn buffer<D: DType>(
            span: WeightLoadSpan,
            owner_drops: Option<std::rc::Rc<std::cell::Cell<usize>>>,
        ) -> Result<TestWeightBuffer<D>, Status> {
            Ok(TestWeightBuffer {
                span: DeviceSpan::new(span.ptr.cast(), D::len_of(span.bytes)?)?,
                state: None,
                owner_drops,
            })
        }
        let mut buffers = Vec::new();
        for (mut dtype, mut span) in self.allocations.drain(..) {
            if matches!(self.fault, FinishFault::WrongDtype) {
                dtype = DynDType::U8;
            }
            if matches!(self.fault, FinishFault::WrongSize) {
                span.bytes += 4;
            }
            buffers.push(match dtype {
                DynDType::BF16 => {
                    WeightBuffer::Bf16(buffer::<BF16>(span, self.owner_drops.clone())?)
                }
                DynDType::F32 => WeightBuffer::F32(buffer::<F32>(span, self.owner_drops.clone())?),
                DynDType::FP8E4M3 => {
                    WeightBuffer::Fp8E4m3(buffer::<Fp8E4M3>(span, self.owner_drops.clone())?)
                }
                DynDType::U8 => WeightBuffer::U8(buffer::<U8>(span, self.owner_drops.clone())?),
                _ => return Err(Status::InvalidArgument),
            });
        }
        match self.fault {
            FinishFault::Reorder => buffers.reverse(),
            FinishFault::Missing => {
                buffers.pop();
            }
            _ => (),
        }
        Ok((buffers, std::mem::take(&mut self.stats)))
    }
}

#[test]
fn executes_validated_plan_against_backend() {
    let tmp = std::env::temp_dir().join(format!("qs3-weight-loader-test-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&tmp);
    std::fs::create_dir(&tmp).unwrap();
    let shard = tmp.join("model-00001-of-00001.safetensors");
    std::fs::write(&shard, [0u8; 8]).unwrap();

    let plan = QwenLoadPlan {
        quantization: None,
        config: tensor::selected_text_config(),
        entries: vec![
            QwenLoadPlanEntry {
                spec: WeightTensorSpec {
                    name: "a".to_owned(),
                    dtype: DynDType::BF16,
                    shape: vec![2],
                    source: WeightTensorSource::Safetensors,
                },
                source: QwenLoadSource::FileRange {
                    shard_path: shard.clone(),
                    absolute_offset: 8,
                    bytes: 4,
                },
            },
            QwenLoadPlanEntry {
                spec: WeightTensorSpec {
                    name: "b".to_owned(),
                    dtype: DynDType::BF16,
                    shape: vec![4],
                    source: WeightTensorSource::ZeroFill,
                },
                source: QwenLoadSource::ZeroFill { bytes: 8 },
            },
            QwenLoadPlanEntry {
                spec: WeightTensorSpec {
                    name: "c".to_owned(),
                    dtype: DynDType::F32,
                    shape: vec![1],
                    source: WeightTensorSource::Safetensors,
                },
                source: QwenLoadSource::FileRange {
                    shard_path: shard,
                    absolute_offset: 0,
                    bytes: 4,
                },
            },
        ],
    };
    let dropped = std::rc::Rc::new(std::cell::Cell::new(false));
    let mut backend = RecordingBackend::default();
    backend.dropped = Some(dropped.clone());
    let loaded = execute_qwen_load_plan(&plan, backend, ptr::null_mut()).unwrap();

    assert_eq!(loaded.tensors.len(), 3);
    assert_eq!(loaded.stats.allocs.len(), 3);
    assert_eq!(loaded.stats.reads, vec![(0, 4), (8, 4)]);
    assert_eq!(loaded.stats.zeros, vec![8]);
    assert!(dropped.get());
    assert_eq!(
        loaded
            .tensors
            .iter()
            .map(|tensor| tensor.spec.name.as_str())
            .collect::<Vec<_>>(),
        ["a", "b", "c"]
    );
    let WeightBuffer::Bf16(a) = &loaded.tensors[0].buffer else {
        panic!("expected BF16")
    };
    let WeightBuffer::Bf16(b) = &loaded.tensors[1].buffer else {
        panic!("expected BF16")
    };
    let WeightBuffer::F32(c) = &loaded.tensors[2].buffer else {
        panic!("expected F32")
    };
    assert_eq!(a.erase() as usize, 0x1000);
    assert_eq!(b.erase() as usize, 0x2000);
    assert_eq!(c.erase() as usize, 0x3000);
    drop(loaded);
    assert!(dropped.get());

    let _ = std::fs::remove_dir_all(&tmp);
}

#[test]
fn loaded_plan_transfers_allocations_into_qwen_weights_once() {
    if !cuda_device_available_for_loader_test() {
        return;
    }
    let plan = bf16_zero_plan(qwen_text_config(4, 32));
    let count = plan.tensor_count();
    let state = std::rc::Rc::new(TinyBackendState::default());
    let backend = TinyCudaBackend::new(0, state.clone());

    let loaded = execute_qwen_load_plan(&plan, backend, ptr::null_mut()).unwrap();
    let (_, weights) = loaded.into_fixture_model(8).unwrap();

    assert_eq!(state.transferred.get(), count);
    assert_eq!(state.drop_calls.get(), 1);
    assert_eq!(state.remaining_at_drop.get(), 0);
    assert_eq!(state.allocation_drops.get(), 0);
    drop(weights);
    assert_eq!(state.allocation_drops.get(), count);
}

fn bf16_zero_plan(config: Qwen36TextConfig) -> QwenLoadPlan {
    let entries = expected_specs(&config, None)
        .unwrap()
        .into_iter()
        .map(|spec| {
            let bytes = spec.byte_len().unwrap();
            QwenLoadPlanEntry {
                spec,
                source: QwenLoadSource::ZeroFill { bytes },
            }
        })
        .collect();
    QwenLoadPlan {
        config,
        entries,
        quantization: None,
    }
}

fn small_zero_plan() -> QwenLoadPlan {
    QwenLoadPlan {
        quantization: None,
        config: tensor::selected_text_config(),
        entries: ["a", "b"]
            .into_iter()
            .map(|slot| QwenLoadPlanEntry {
                spec: WeightTensorSpec {
                    name: slot.to_owned(),
                    dtype: DynDType::BF16,
                    shape: vec![2],
                    source: WeightTensorSource::ZeroFill,
                },
                source: QwenLoadSource::ZeroFill { bytes: 4 },
            })
            .collect(),
    }
}

#[test]
fn load_rejects_duplicate_names_before_allocation() {
    let mut plan = small_zero_plan();
    plan.entries[1].spec.name = plan.entries[0].spec.name.clone();
    // An invalid device makes accidental allocation fail with a different error.
    let backend = TinyCudaBackend::new(-1, Default::default());
    let err = execute_qwen_load_plan(&plan, backend, ptr::null_mut())
        .err()
        .unwrap();
    assert!(err.to_string().contains("duplicate loaded tensor"));
}

#[test]
fn load_rejects_mismatched_finished_buffers() {
    for fault in [
        FinishFault::Reorder,
        FinishFault::Missing,
        FinishFault::WrongDtype,
        FinishFault::WrongSize,
    ] {
        let dropped = std::rc::Rc::new(std::cell::Cell::new(false));
        let mut backend = RecordingBackend::default();
        backend.fault = fault;
        backend.dropped = Some(dropped.clone());
        let err = execute_qwen_load_plan(&small_zero_plan(), backend, ptr::null_mut())
            .err()
            .unwrap();
        assert!(matches!(err, WeightLoadError::TensorTable(_)));
        assert!(dropped.get());
    }
}

#[test]
fn failed_finish_drops_transferred_and_remaining_allocations() {
    if !cuda_device_available_for_loader_test() {
        return;
    }
    let state = std::rc::Rc::new(TinyBackendState::default());
    let mut backend = TinyCudaBackend::new(cuda_device_from_env(), state.clone());
    backend.fail_after = Some(1);
    assert!(execute_qwen_load_plan(&small_zero_plan(), backend, ptr::null_mut()).is_err());
    assert_eq!(state.transferred.get(), 1);
    assert_eq!(state.allocation_drops.get(), 1);
    assert_eq!(state.drop_calls.get(), 1);
    assert_eq!(state.remaining_at_drop.get(), 1);
}

#[test]
fn validates_selected_checkpoint_manifest_when_available() {
    let model_dir = real_selected_model_dir();
    if !model_dir.is_dir() {
        eprintln!(
            "skipping selected BF16 manifest smoke; {} does not exist",
            model_dir.display()
        );
        return;
    }

    let started = Instant::now();
    // This validates the actual checkpoint's config, complete index, tensor
    // shapes/dtypes and shard ranges against the selected build.
    let plan = QwenLoadPlan::read(&model_dir).unwrap();
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
}

#[test]
#[ignore = "loads and executes the full real Qwen3.6 BF16 model"]
fn real_qwen36_bf16_generates_reference_tokens() {
    eprintln!(
        "real BF16 model with {} GDN recurrence and {} MoE",
        crate::constants::precision::GDN_RECURRENT_STATE,
        crate::backend::qsfi::MoeBf16Kernel::COMPILED.as_str(),
    );
    let model_dir = require_real_qwen36_model_dir();
    let tokenizer = crate::tokenizer::QwenTokenizer::from_model_dir(&model_dir).unwrap();
    let plan = QwenLoadPlan::read(model_dir).unwrap();
    let backend = ManagedUmaBackend::new(cuda_device_from_env()).unwrap();
    let loaded = execute_qwen_load_plan(&plan, backend, ptr::null_mut()).unwrap();
    let (config, weights) = loaded.into_qwen_model(8).unwrap();
    let mut runner = crate::model::ModelRunner::new(
        std::rc::Rc::new(crate::memory::CudaCtx::default().unwrap()),
        config,
        weights,
        tokenizer.token_count(),
    )
    .unwrap();
    assert_eq!(runner.gdn_qkv_provider(), "triton");
    let request_id = 0xBF16_0001;

    let prefill = runner
        .run(crate::model::QwenRequest {
            request_id,
            tokens: &REAL_PROMPT,
            max_new_tokens: 0,
        })
        .unwrap();
    assert!(prefill.generated_tokens.is_empty());
    runner.assert_lm_head_matches_cublaslt(REAL_PROMPT.len() as u32);
    assert_real_top_logits(
        "real prefill",
        &runner.last_logits_row_for_test().unwrap(),
        &REAL_PREFILL_TOP_IDS,
        &REAL_PREFILL_TOP_VALUES,
        1.875,
        REAL_PREFILL_STABLE_TOP_COUNT,
    );

    let first = runner
        .run(crate::model::QwenRequest {
            request_id,
            tokens: &REAL_PROMPT,
            max_new_tokens: 1,
        })
        .unwrap();
    assert_eq!(first.generated_tokens, vec![REAL_GENERATED[0]]);
    runner.assert_lm_head_matches_cublaslt(1);
    assert_real_top_logits(
        "real first decode",
        &runner.last_logits_row_for_test().unwrap(),
        &REAL_DECODE_TOP_IDS,
        &REAL_DECODE_TOP_VALUES,
        3.625,
        REAL_DECODE_STABLE_TOP_COUNT,
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
    let plan = QwenLoadPlan::read(&model_dir).unwrap();
    println!(
        "planned {} tensors in {:.3}s from {}",
        plan.tensor_count(),
        plan_started.elapsed().as_secs_f64(),
        model_dir.display()
    );

    let setup_started = Instant::now();
    let backend = ManagedUmaBackend::new(cuda_device_from_env()).unwrap();
    let backend_setup = setup_started.elapsed();
    println!("backend setup {:.3}s", backend_setup.as_secs_f64());

    let load_started = Instant::now();
    let loaded = execute_qwen_load_plan(&plan, backend, ptr::null_mut()).unwrap();
    let elapsed = load_started.elapsed();
    println!(
        "backend setup + load {:.3}s",
        (backend_setup + elapsed).as_secs_f64()
    );
    let loaded_tensors = loaded.tensors.len();
    let stats = loaded.stats;
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
    let plan = QwenLoadPlan::read(&model_dir).unwrap();
    println!(
        "planned {} tensors in {:.3}s from {}",
        plan.tensor_count(),
        plan_started.elapsed().as_secs_f64(),
        model_dir.display()
    );

    let setup_started = Instant::now();
    let backend = PinnedUploadBackend::new(cuda_device_from_env()).unwrap();
    let backend_setup = setup_started.elapsed();
    println!("backend setup {:.3}s", backend_setup.as_secs_f64());

    let load_started = Instant::now();
    let loaded = execute_qwen_load_plan(&plan, backend, ptr::null_mut()).unwrap();
    let elapsed = load_started.elapsed();
    println!(
        "backend setup + load {:.3}s",
        (backend_setup + elapsed).as_secs_f64()
    );
    let loaded_tensors = loaded.tensors.len();
    let stats = loaded.stats;
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
