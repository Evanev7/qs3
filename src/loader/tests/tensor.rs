use super::*;
use std::rc::Rc;

pub(super) fn selected_text_config() -> Qwen36TextConfig {
    Qwen36TextConfig::from_config_object(&checkpoint::selected_config()).unwrap()
}

#[test]
fn parses_mixed_storage_without_claiming_a_quantization_recipe() {
    let header = r#"{"packed":{"dtype":"U8","shape":[3],"data_offsets":[0,3]},"blocks":{"dtype":"F8_E4M3","shape":[3],"data_offsets":[3,6]},"global":{"dtype":"F32","shape":[],"data_offsets":[6,10]}}"#;
    let parsed =
        SafetensorsHeader::parse_safetensors_bytes(&synthetic_safetensors(header, 10)).unwrap();
    assert_eq!(parsed.tensors["packed"].dtype, DynDType::U8);
    assert_eq!(parsed.tensors["blocks"].dtype, DynDType::FP8E4M3);
    assert_eq!(parsed.tensors["global"].byte_len().unwrap(), 4);

    let malformed = header.replace("\"shape\":[3]", "\"shape\":[4]");
    assert!(
        SafetensorsHeader::parse_safetensors_bytes(&synthetic_safetensors(&malformed, 10)).is_err()
    );
    let unsupported = header.replace("F8_E4M3", "F8_E4M3FNUZ");
    assert!(
        SafetensorsHeader::parse_safetensors_bytes(&synthetic_safetensors(&unsupported, 10))
            .is_err()
    );
}

#[test]
fn bf16_manifest_still_rejects_quantized_tensor_storage() {
    let config = selected_text_config();
    let specs = expected_specs(&config, None).unwrap();
    for dtype in [DynDType::U8, DynDType::FP8E4M3] {
        let mut table = tensor_table_from_specs(&specs);
        table
            .get_mut("model.language_model.layers.3.self_attn.o_proj.weight")
            .unwrap()
            .meta
            .dtype = dtype;
        let err =
            validate_tensor_specs(&table, expected_specs(&config, None).unwrap()).unwrap_err();
        assert!(err.to_string().contains("dtype"));
    }
}

#[test]
fn bf16_model_rejects_other_storage() {
    let mut plan = bf16_zero_plan(qwen_text_config(4, 16));
    plan.entries[0].spec.dtype = DynDType::FP8E4M3;
    let bytes = plan.entries[0].spec.byte_len().unwrap();
    plan.entries[0].source = QwenLoadSource::ZeroFill { bytes };
    let loaded =
        execute_qwen_load_plan(&plan, RecordingBackend::default(), ptr::null_mut()).unwrap();
    let err = loaded.into_fixture_model(8).err().unwrap();
    assert!(
        err.to_string()
            .contains("loaded tensor descriptor mismatch")
    );
}

#[test]
fn selected_bf16_model_retains_all_allocation_owners() {
    if !cuda_device_available_for_loader_test() {
        return;
    }
    let plan = bf16_zero_plan(selected_text_config());
    let count = plan.tensor_count();
    let state = Rc::new(TinyBackendState::default());
    let backend = TinyCudaBackend::new(cuda_device_from_env(), state.clone());
    let loaded = execute_qwen_load_plan(&plan, backend, ptr::null_mut()).unwrap();
    let (_, weights) = loaded.into_qwen_model(8).unwrap();
    assert_eq!(state.transferred.get(), count);
    assert_eq!(state.drop_calls.get(), 1);
    assert_eq!(state.remaining_at_drop.get(), 0);
    assert_eq!(state.allocation_drops.get(), 0);
    drop(weights);
    assert_eq!(state.allocation_drops.get(), count);
}

#[test]
fn managed_mixed_tensor_handoff_retains_exact_file_bytes() {
    if !cuda_device_available_for_loader_test() {
        return;
    }
    let path = env::temp_dir().join(format!("qs3-mixed-tensor-{}", std::process::id()));
    // Odd-sized packed payload and block scales followed by an unaligned source
    // F32 scalar. Each final allocation must have its own correct alignment.
    let payload = [0x12u8, 0x34, 0x56, 0x38, 0x40, 0x00, 0x00, 0x00, 0xc0, 0x3f];
    std::fs::write(&path, payload).unwrap();
    let file = std::fs::File::open(&path).unwrap();
    let mut backend = ManagedUmaBackend::new(cuda_device_from_env()).unwrap();
    let mut offset = 0;
    for (slot, dtype, shape) in [
        ("packed", DynDType::U8, vec![3]),
        ("blocks", DynDType::FP8E4M3, vec![3]),
        ("global", DynDType::F32, vec![]),
    ] {
        let bytes = dtype.storage_bytes_for_shape(&shape).unwrap();
        let span = backend
            .alloc_tensor(WeightTensorDesc {
                name: slot,
                dtype,
                shape: &shape,
                bytes,
            })
            .unwrap();
        backend
            .read_exact(
                WeightFileRange {
                    file: &file,
                    offset,
                    bytes,
                },
                &span,
                ptr::null_mut(),
            )
            .unwrap();
        offset += bytes as u64;
    }
    let (weights, stats) = backend.finish(ptr::null_mut()).unwrap();
    assert_eq!(stats.read_bytes, payload.len());
    let mut weights = weights.into_iter();
    let WeightBuffer::U8(packed) = weights.next().unwrap() else {
        panic!("expected packed bytes")
    };
    let WeightBuffer::Fp8E4m3(blocks) = weights.next().unwrap() else {
        panic!("expected FP8 scales")
    };
    let WeightBuffer::F32(global) = weights.next().unwrap() else {
        panic!("expected F32 scale")
    };
    // Managed memory is host-visible after finish; backend has already been dropped.
    unsafe {
        assert_eq!(
            std::slice::from_raw_parts(packed.erase().cast::<u8>(), packed.len),
            &payload[..3]
        );
        assert_eq!(
            std::slice::from_raw_parts(blocks.erase().cast::<u8>(), blocks.len),
            &payload[3..6]
        );
        assert_eq!(*global.erase().cast::<f32>(), 1.5);
    }
    drop(weights);
    std::fs::remove_file(path).unwrap();
}

#[test]
fn pinned_finish_returns_completed_owned_buffers() {
    if !cuda_device_available_for_loader_test() {
        return;
    }
    let path = env::temp_dir().join(format!("qs3-pinned-finish-{}", std::process::id()));
    let payload = [0x12u8, 0x34, 0x56, 0x38, 0x40, 0x00];
    std::fs::write(&path, payload).unwrap();
    let file = std::fs::File::open(&path).unwrap();
    let ctx = crate::memory::CudaCtx::new(cuda_device_from_env()).unwrap();
    let mut backend = PinnedUploadBackend::new(cuda_device_from_env()).unwrap();
    let mut spans = Vec::new();
    for (name, dtype, shape, bytes) in [
        ("packed", DynDType::U8, [3], 3),
        ("blocks", DynDType::FP8E4M3, [3], 3),
        ("zeros", DynDType::BF16, [2], 4),
    ] {
        spans.push(
            backend
                .alloc_tensor(WeightTensorDesc {
                    name,
                    dtype,
                    shape: &shape,
                    bytes,
                })
                .unwrap(),
        );
    }
    // Submission order must not become output order.
    for idx in [1, 0] {
        backend
            .read_exact(
                WeightFileRange {
                    file: &file,
                    offset: (idx * 3) as u64,
                    bytes: 3,
                },
                &spans[idx],
                ctx.stream,
            )
            .unwrap();
    }
    backend.zero_fill(&spans[2], ctx.stream).unwrap();
    let (buffers, stats) = backend.finish(ctx.stream).unwrap();
    assert_eq!(stats.read_bytes, 6);
    assert_eq!(stats.zero_fill_bytes, 4);
    assert_eq!(
        buffers.iter().map(WeightBuffer::dtype).collect::<Vec<_>>(),
        [DynDType::U8, DynDType::FP8E4M3, DynDType::BF16]
    );
    // Staging and backend have been released. Only these owners keep storage live.
    let mut observed = [0u8; 10];
    let mut offset = 0;
    for (buffer, span) in buffers.iter().zip(spans) {
        assert!(buffer.matches_span(span));
        unsafe {
            result_from_cuda(ffi::cuda::cudaMemcpyAsync(
                observed.as_mut_ptr().add(offset).cast(),
                span.ptr,
                span.bytes,
                ffi::cuda::CUDA_MEMCPY_DEVICE_TO_HOST,
                ctx.stream,
            ))
            .unwrap();
        }
        offset += span.bytes;
    }
    ctx.synchronize().unwrap();
    assert_eq!(&observed[..6], &payload);
    assert_eq!(&observed[6..], &[0u8; 4]);
    drop(buffers);
    std::fs::remove_file(path).unwrap();
}
