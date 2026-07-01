use qs3::{ModelRunner, QwenConfig, QwenMoeConfig, QwenRequest, QwenResult, QwenWeights, Status};

use std::ffi::{CStr, c_char};

const CUDA_SUCCESS: i32 = 0;
const RANDOM_MODEL_SEED: u64 = 0x5153_3300_d15e_a5e5;

unsafe extern "C" {
    fn cudaGetDeviceCount(count: *mut i32) -> i32;
    fn cudaGetErrorString(error: i32) -> *const c_char;
    fn cudaSetDevice(device: i32) -> i32;
    fn cudaDeviceSynchronize() -> i32;
}

fn cuda_error_string(err: i32) -> String {
    if err == CUDA_SUCCESS {
        return "cudaSuccess".to_owned();
    }
    let ptr = unsafe { cudaGetErrorString(err) };
    if ptr.is_null() {
        return format!("CUDA error {err}");
    }
    unsafe { CStr::from_ptr(ptr) }
        .to_string_lossy()
        .into_owned()
}

fn assert_cuda(err: i32, what: &str) {
    assert_eq!(
        err,
        CUDA_SUCCESS,
        "{what}: {} ({err})",
        cuda_error_string(err)
    );
}

fn cuda_device_available() -> bool {
    let Some(device_count) = cuda_device_count() else {
        return false;
    };
    if device_count == 0 {
        eprintln!("SKIP: no CUDA device available");
        return false;
    }
    assert_cuda(unsafe { cudaSetDevice(0) }, "set CUDA test device");
    true
}

fn cuda_device_count() -> Option<i32> {
    let mut device_count = 0;
    let err = unsafe { cudaGetDeviceCount(&mut device_count) };
    if err != CUDA_SUCCESS {
        eprintln!(
            "SKIP: CUDA device count unavailable: {} ({err})",
            cuda_error_string(err)
        );
        return None;
    }
    Some(device_count)
}

fn run_random_model(
    config: QwenConfig,
    request_id: u64,
    tokens: &[i32],
    max_new_tokens: u32,
) -> QwenResult {
    let mut runner = ModelRunner::random_bf16(config, RANDOM_MODEL_SEED).unwrap();
    runner
        .run(QwenRequest {
            request_id,
            tokens,
            max_new_tokens,
        })
        .unwrap()
}

#[test]
fn randomized_dense_model_runs_prefill_and_two_decodes() {
    if !cuda_device_available() {
        return;
    }

    let config = QwenConfig::randomized_dense_tiny_fixture(0);
    let mut runner = ModelRunner::random_bf16(config, RANDOM_MODEL_SEED).unwrap();

    assert_eq!(
        runner
            .run(QwenRequest {
                request_id: 11,
                tokens: &[-1],
                max_new_tokens: 0,
            })
            .unwrap_err(),
        Status::InvalidArgument
    );
    assert_eq!(
        runner
            .run(QwenRequest {
                request_id: 11,
                tokens: &[config.vocab_size as i32],
                max_new_tokens: 0,
            })
            .unwrap_err(),
        Status::InvalidArgument
    );

    let result = runner
        .run(QwenRequest {
            request_id: 11,
            tokens: &[1, 7, 13],
            max_new_tokens: 2,
        })
        .unwrap();
    assert_eq!(result.generated_tokens.len(), 2);
    assert_eq!(result.live_tokens.len(), 5);
    assert_eq!(&result.live_tokens[..3], &[1, 7, 13]);
    assert_eq!(result.logits_rows, 1);
    assert_eq!(result.logits_vocab_size, config.vocab_size);
    for token in &result.generated_tokens {
        assert!(*token >= 0 && *token < config.vocab_size as i32);
    }

    let live = result.live_tokens.clone();
    let continued = runner
        .run(QwenRequest {
            request_id: 11,
            tokens: &live,
            max_new_tokens: 1,
        })
        .unwrap();
    assert_eq!(continued.generated_tokens.len(), 1);
    assert_eq!(continued.live_tokens.len(), 6);
    assert_eq!(&continued.live_tokens[..live.len()], live.as_slice());

    assert_cuda(
        unsafe { cudaDeviceSynchronize() },
        "sync randomized model test",
    );
}

#[test]
fn exact_prefix_extension_accepts_caller_suffix_tokens() {
    if !cuda_device_available() {
        return;
    }

    let config = QwenConfig::randomized_dense_tiny_fixture(0);
    let mut runner = ModelRunner::random_bf16(config, RANDOM_MODEL_SEED).unwrap();

    let base = runner
        .run(QwenRequest {
            request_id: 17,
            tokens: &[1, 7, 13],
            max_new_tokens: 0,
        })
        .unwrap();
    assert!(base.generated_tokens.is_empty());
    assert_eq!(base.live_tokens, [1, 7, 13]);

    let extended_prompt = [1, 7, 13, 21, 34];
    let extended = runner
        .run(QwenRequest {
            request_id: 17,
            tokens: &extended_prompt,
            max_new_tokens: 2,
        })
        .unwrap();
    let fresh = run_random_model(config, 17, &extended_prompt, 2);

    assert_eq!(extended.generated_tokens, fresh.generated_tokens);
    assert_eq!(extended.live_tokens, fresh.live_tokens);
    assert_eq!(extended.logits_rows, fresh.logits_rows);
    assert_eq!(extended.logits_vocab_size, fresh.logits_vocab_size);
    assert_eq!(
        &extended.live_tokens[..extended_prompt.len()],
        &extended_prompt
    );

    assert_cuda(
        unsafe { cudaDeviceSynchronize() },
        "sync caller-suffix exact-prefix extension test",
    );
}

#[test]
fn qwen36_gdn_configs_require_full_attention_and_shared_expert() {
    let mut no_attention = QwenConfig::randomized_qwen36_moe_gdn_one_block_fixture(-1);
    no_attention.num_layers = 1;
    assert_eq!(no_attention.validate(), Err(Status::InvalidArgument));
    assert_eq!(
        ModelRunner::random_bf16(no_attention, RANDOM_MODEL_SEED).err(),
        Some(Status::InvalidArgument)
    );

    let config = QwenConfig::randomized_qwen36_moe_gdn_one_block_fixture(-1);
    assert_eq!(config.validate(), Ok(()));

    let mut incomplete_schedule = config;
    incomplete_schedule.num_layers = 5;
    assert_eq!(incomplete_schedule.validate(), Err(Status::InvalidArgument));
    assert_eq!(
        ModelRunner::random_bf16(incomplete_schedule, RANDOM_MODEL_SEED).err(),
        Some(Status::InvalidArgument)
    );

    let mut missing_shared = config;
    missing_shared.moe = Some(QwenMoeConfig {
        shared_expert_intermediate_size: 0,
        ..QwenMoeConfig::qwen36_35b_a3b()
    });
    assert_eq!(missing_shared.validate(), Err(Status::Unsupported));
}

#[test]
fn randomized_moe_model_runs_prefill_decode_rebuild_and_failed_rewrite() {
    if !cuda_device_available() {
        return;
    }

    let config = QwenConfig::randomized_moe_tiny_fixture(0);
    let mut runner = ModelRunner::random_bf16(config, RANDOM_MODEL_SEED).unwrap();

    let prefill = runner
        .run(QwenRequest {
            request_id: 71,
            tokens: &[1, 7, 13],
            max_new_tokens: 0,
        })
        .unwrap();
    assert!(prefill.generated_tokens.is_empty());
    assert_eq!(prefill.live_tokens, vec![1, 7, 13]);
    assert_eq!(prefill.logits_rows, 3);
    assert_eq!(prefill.logits_vocab_size, config.vocab_size);

    let decoded = runner
        .run(QwenRequest {
            request_id: 71,
            tokens: &prefill.live_tokens,
            max_new_tokens: 2,
        })
        .unwrap();
    assert_eq!(decoded.generated_tokens.len(), 2);
    assert_eq!(decoded.live_tokens.len(), 5);
    assert_eq!(
        &decoded.live_tokens[..prefill.live_tokens.len()],
        [1, 7, 13]
    );
    for token in &decoded.generated_tokens {
        assert!(*token >= 0 && *token < config.vocab_size as i32);
    }

    let first = run_random_model(config, 73, &[3, 5, 8, 13], 2);
    let second = run_random_model(config, 73, &[3, 5, 8, 13], 2);
    assert_eq!(first, second);
    assert_eq!(first.generated_tokens.len(), 2);

    let original = runner
        .run(QwenRequest {
            request_id: 79,
            tokens: &[2, 4, 6, 8],
            max_new_tokens: 2,
        })
        .unwrap();
    assert_eq!(original.live_tokens.len(), 6);

    let rewritten_prompt = [2, 4, 9, 8];
    let rebuilt = runner
        .run(QwenRequest {
            request_id: 79,
            tokens: &rewritten_prompt,
            max_new_tokens: 2,
        })
        .unwrap();
    let fresh = run_random_model(config, 79, &rewritten_prompt, 2);
    assert_eq!(rebuilt.generated_tokens, fresh.generated_tokens);
    assert_eq!(rebuilt.live_tokens, fresh.live_tokens);
    assert_eq!(rebuilt.logits_rows, fresh.logits_rows);
    assert_eq!(rebuilt.logits_vocab_size, fresh.logits_vocab_size);

    let live_before_failed_rewrite = runner.live_tokens().to_vec();
    assert_eq!(
        runner
            .run(QwenRequest {
                request_id: 79,
                tokens: &[2, config.vocab_size as i32, 9, 8],
                max_new_tokens: 1,
            })
            .unwrap_err(),
        Status::InvalidArgument
    );
    assert_eq!(runner.live_tokens(), live_before_failed_rewrite.as_slice());

    let continued = runner
        .run(QwenRequest {
            request_id: 79,
            tokens: &live_before_failed_rewrite,
            max_new_tokens: 1,
        })
        .unwrap();
    assert_eq!(continued.generated_tokens.len(), 1);
    assert_eq!(
        &continued.live_tokens[..live_before_failed_rewrite.len()],
        live_before_failed_rewrite.as_slice()
    );

    assert_cuda(
        unsafe { cudaDeviceSynchronize() },
        "sync randomized MoE model test",
    );
}

#[test]
fn randomized_shared_moe_model_runs_prefill_and_decode() {
    if !cuda_device_available() {
        return;
    }

    let config = QwenConfig::randomized_shared_moe_tiny_fixture(0);
    let mut runner = ModelRunner::random_bf16(config, RANDOM_MODEL_SEED).unwrap();

    let result = runner
        .run(QwenRequest {
            request_id: 81,
            tokens: &[1, 7, 13],
            max_new_tokens: 2,
        })
        .unwrap();
    assert_eq!(result.generated_tokens.len(), 2);
    assert_eq!(result.live_tokens.len(), 5);
    assert_eq!(&result.live_tokens[..3], &[1, 7, 13]);
    assert_eq!(result.logits_rows, 1);
    assert_eq!(result.logits_vocab_size, config.vocab_size);
    for token in &result.generated_tokens {
        assert!(*token >= 0 && *token < config.vocab_size as i32);
    }

    assert_cuda(
        unsafe { cudaDeviceSynchronize() },
        "sync randomized shared MoE model test",
    );
}

#[test]
fn randomized_dense_model_runs_with_logits_soft_cap() {
    if !cuda_device_available() {
        return;
    }

    let mut config = QwenConfig::randomized_dense_tiny_fixture(0);
    config.logits_soft_cap = 2.0;

    let first = run_random_model(config, 25, &[3, 5, 8, 13], 2);
    let second = run_random_model(config, 25, &[3, 5, 8, 13], 2);

    assert_eq!(first, second);
    assert_eq!(first.generated_tokens.len(), 2);
    assert_eq!(first.live_tokens.len(), 6);
    assert_eq!(first.logits_rows, 1);
    assert_eq!(first.logits_vocab_size, config.vocab_size);

    assert_cuda(
        unsafe { cudaDeviceSynchronize() },
        "sync soft-capped randomized model test",
    );
}

#[test]
fn randomized_dense_model_is_repeatable_for_same_seed_and_request() {
    if !cuda_device_available() {
        return;
    }

    let config = QwenConfig::randomized_dense_tiny_fixture(0);
    let first = run_random_model(config, 21, &[3, 5, 8, 13], 3);
    let second = run_random_model(config, 21, &[3, 5, 8, 13], 3);

    assert_eq!(first, second);
    assert_eq!(first.generated_tokens.len(), 3);
    assert_eq!(first.live_tokens.len(), 7);
    assert_eq!(&first.live_tokens[..4], &[3, 5, 8, 13]);
    assert_eq!(first.logits_rows, 1);
    assert_eq!(first.logits_vocab_size, config.vocab_size);

    assert_cuda(
        unsafe { cudaDeviceSynchronize() },
        "sync repeatable randomized model test",
    );
}

#[test]
fn prompt_rewrite_behind_live_tail_rebuilds_like_fresh_runner() {
    if !cuda_device_available() {
        return;
    }

    let config = QwenConfig::randomized_dense_tiny_fixture(0);
    let mut runner = ModelRunner::random_bf16(config, RANDOM_MODEL_SEED).unwrap();
    let original = runner
        .run(QwenRequest {
            request_id: 31,
            tokens: &[2, 4, 6, 8],
            max_new_tokens: 2,
        })
        .unwrap();
    assert_eq!(original.live_tokens.len(), 6);

    let rewritten_prompt = [2, 4, 9, 8];
    let rebuilt = runner
        .run(QwenRequest {
            request_id: 31,
            tokens: &rewritten_prompt,
            max_new_tokens: 2,
        })
        .unwrap();
    let fresh = run_random_model(config, 31, &rewritten_prompt, 2);

    assert_eq!(rebuilt.generated_tokens, fresh.generated_tokens);
    assert_eq!(rebuilt.live_tokens, fresh.live_tokens);
    assert_eq!(rebuilt.logits_rows, fresh.logits_rows);
    assert_eq!(rebuilt.logits_vocab_size, fresh.logits_vocab_size);
    assert_eq!(
        &rebuilt.live_tokens[..rewritten_prompt.len()],
        &rewritten_prompt
    );

    assert_cuda(
        unsafe { cudaDeviceSynchronize() },
        "sync prompt rewrite rebuild randomized model test",
    );
}

#[test]
fn failed_prompt_rewrite_keeps_previous_live_state() {
    if !cuda_device_available() {
        return;
    }

    let config = QwenConfig::randomized_dense_tiny_fixture(0);
    let mut runner = ModelRunner::random_bf16(config, RANDOM_MODEL_SEED).unwrap();
    let original = runner
        .run(QwenRequest {
            request_id: 37,
            tokens: &[1, 2, 3],
            max_new_tokens: 1,
        })
        .unwrap();
    let original_live = original.live_tokens;

    assert_eq!(
        runner
            .run(QwenRequest {
                request_id: 37,
                tokens: &[1, config.vocab_size as i32, 3],
                max_new_tokens: 1,
            })
            .unwrap_err(),
        Status::InvalidArgument
    );
    assert_eq!(runner.live_tokens(), original_live.as_slice());

    let continued = runner
        .run(QwenRequest {
            request_id: 37,
            tokens: &original_live,
            max_new_tokens: 1,
        })
        .unwrap();
    assert_eq!(
        &continued.live_tokens[..original_live.len()],
        original_live.as_slice()
    );
    assert_eq!(continued.generated_tokens.len(), 1);

    assert_cuda(
        unsafe { cudaDeviceSynchronize() },
        "sync failed prompt rewrite transactionality test",
    );
}

#[test]
fn oversized_run_request_is_rejected_without_mutating_live_state() {
    if !cuda_device_available() {
        return;
    }

    let config = QwenConfig::randomized_dense_tiny_fixture(0);
    let mut runner = ModelRunner::random_bf16(config, RANDOM_MODEL_SEED).unwrap();
    let original = runner
        .run(QwenRequest {
            request_id: 41,
            tokens: &[1, 2, 3],
            max_new_tokens: 1,
        })
        .unwrap();
    let original_live = original.live_tokens;

    assert_eq!(
        runner
            .run(QwenRequest {
                request_id: 42,
                tokens: &[4],
                max_new_tokens: u32::MAX,
            })
            .unwrap_err(),
        Status::InvalidArgument
    );
    assert_eq!(runner.live_tokens(), original_live.as_slice());

    let too_long_prompt = [5; 15];
    assert_eq!(
        runner
            .run(QwenRequest {
                request_id: 43,
                tokens: &too_long_prompt,
                max_new_tokens: 2,
            })
            .unwrap_err(),
        Status::InvalidArgument
    );
    assert_eq!(runner.live_tokens(), original_live.as_slice());

    assert_cuda(
        unsafe { cudaDeviceSynchronize() },
        "sync oversized run rejection randomized model test",
    );
}

#[test]
fn qwen_config_rejects_unsupported_dense_runner_shapes() {
    let config = QwenConfig::randomized_dense_tiny_fixture(-1);
    assert_eq!(config.validate(), Ok(()));
    assert_ne!(config.hidden_size, config.num_q_heads * config.head_dim);

    let mut config = QwenConfig::randomized_dense_tiny_fixture(-1);
    config.hidden_size = 97;
    assert_eq!(config.validate(), Err(Status::InvalidArgument));

    let mut config = QwenConfig::randomized_dense_tiny_fixture(-1);
    config.max_batch_size = 2;
    assert_eq!(config.validate(), Err(Status::Unsupported));

    let mut config = QwenConfig::randomized_dense_tiny_fixture(-1);
    config.head_dim = 80;
    assert_eq!(config.validate(), Err(Status::Unsupported));

    for head_dim in [64, 128, 512] {
        let mut config = QwenConfig::randomized_dense_tiny_fixture(-1);
        config.head_dim = head_dim;
        assert_eq!(config.validate(), Err(Status::Unsupported));
    }

    let mut config = QwenConfig::randomized_dense_tiny_fixture(-1);
    config.num_q_heads = 4;
    config.num_kv_heads = 2;
    assert_eq!(config.validate(), Err(Status::Unsupported));

    let mut config = QwenConfig::randomized_dense_tiny_fixture(-1);
    config.num_q_heads = 8;
    config.num_kv_heads = 1;
    assert_eq!(config.validate(), Err(Status::Unsupported));

    let mut config = QwenConfig::randomized_dense_tiny_fixture(-1);
    config.num_q_heads = 32;
    config.num_kv_heads = 4;
    assert_eq!(config.validate(), Err(Status::Unsupported));

    let mut config = QwenConfig::randomized_dense_tiny_fixture(-1);
    config.rope_theta = 0.0;
    assert_eq!(config.validate(), Err(Status::InvalidArgument));

    let mut config = QwenConfig::randomized_dense_tiny_fixture(-1);
    config.rope_scale = 0.0;
    assert_eq!(config.validate(), Err(Status::InvalidArgument));

    let mut config = QwenConfig::randomized_dense_tiny_fixture(-1);
    config.logits_soft_cap = -1.0;
    assert_eq!(config.validate(), Err(Status::InvalidArgument));

    let mut config = QwenConfig::randomized_dense_tiny_fixture(-1);
    config.logits_soft_cap = f32::INFINITY;
    assert_eq!(config.validate(), Err(Status::InvalidArgument));
}

#[test]
fn qwen_config_validates_public_moe_config_json_fields() {
    let tiny = QwenConfig::randomized_moe_tiny_fixture(-1);
    assert_eq!(
        tiny.moe,
        Some(QwenMoeConfig {
            num_experts: 4,
            num_experts_per_tok: 2,
            moe_intermediate_size: 64,
            shared_expert_intermediate_size: 0,
        })
    );

    let qwen36 = QwenConfig::randomized_qwen36_moe_gdn_one_block_fixture(-1);
    assert_eq!(qwen36.moe, Some(QwenMoeConfig::qwen36_35b_a3b()));
    assert_eq!(qwen36.hidden_size, 2048);
    assert_eq!(qwen36.num_q_heads, 16);
    assert_eq!(qwen36.num_kv_heads, 2);
    assert_eq!(qwen36.head_dim, 256);
    assert_eq!(qwen36.num_q_heads * qwen36.head_dim, 4096);

    let shared_tiny = QwenConfig::randomized_shared_moe_tiny_fixture(-1);
    assert_eq!(
        shared_tiny.moe,
        Some(QwenMoeConfig {
            num_experts: 4,
            num_experts_per_tok: 2,
            moe_intermediate_size: 64,
            shared_expert_intermediate_size: 32,
        })
    );

    let mut config = QwenConfig::randomized_moe_tiny_fixture(-1);
    config.moe = Some(QwenMoeConfig {
        num_experts: 4,
        num_experts_per_tok: 0,
        moe_intermediate_size: 64,
        shared_expert_intermediate_size: 0,
    });
    assert_eq!(config.validate(), Err(Status::InvalidArgument));

    let mut config = QwenConfig::randomized_moe_tiny_fixture(-1);
    config.moe = Some(QwenMoeConfig {
        num_experts: 4,
        num_experts_per_tok: 5,
        moe_intermediate_size: 64,
        shared_expert_intermediate_size: 0,
    });
    assert_eq!(config.validate(), Err(Status::InvalidArgument));

    let mut config = QwenConfig::randomized_moe_tiny_fixture(-1);
    config.moe = Some(QwenMoeConfig {
        num_experts: 32,
        num_experts_per_tok: 17,
        moe_intermediate_size: 64,
        shared_expert_intermediate_size: 0,
    });
    assert_eq!(config.validate(), Err(Status::Unsupported));

    let mut config = QwenConfig::randomized_moe_tiny_fixture(-1);
    config.moe = Some(QwenMoeConfig {
        num_experts: 4097,
        num_experts_per_tok: 2,
        moe_intermediate_size: 64,
        shared_expert_intermediate_size: 0,
    });
    assert_eq!(config.validate(), Err(Status::Unsupported));

    let mut config = QwenConfig::randomized_moe_tiny_fixture(-1);
    config.moe = Some(QwenMoeConfig {
        num_experts: 4,
        num_experts_per_tok: 2,
        moe_intermediate_size: 66,
        shared_expert_intermediate_size: 0,
    });
    assert_eq!(config.validate(), Err(Status::InvalidArgument));

    let mut config = QwenConfig::randomized_moe_tiny_fixture(-1);
    config.moe = Some(QwenMoeConfig {
        num_experts: 4,
        num_experts_per_tok: 2,
        moe_intermediate_size: 64,
        shared_expert_intermediate_size: 66,
    });
    assert_eq!(config.validate(), Err(Status::InvalidArgument));

    let mut config = QwenConfig::randomized_moe_tiny_fixture(-1);
    config.intermediate_size = 256;
    assert_eq!(config.validate(), Err(Status::InvalidArgument));

    let mut config = QwenConfig::randomized_moe_tiny_fixture(-1);
    config.moe = Some(QwenMoeConfig {
        num_experts: 4,
        num_experts_per_tok: 2,
        moe_intermediate_size: 64,
        shared_expert_intermediate_size: 64,
    });
    assert_eq!(config.validate(), Ok(()));

    let mut qwen36_without_shared = qwen36;
    qwen36_without_shared.moe = Some(QwenMoeConfig {
        shared_expert_intermediate_size: 0,
        ..QwenMoeConfig::qwen36_35b_a3b()
    });
    assert_eq!(qwen36_without_shared.validate(), Err(Status::Unsupported));
}

#[test]
fn qwen_weights_are_bound_to_device_and_stream_config() {
    if !cuda_device_available() {
        return;
    }

    let config = QwenConfig::randomized_dense_tiny_fixture(0);
    let weights = QwenWeights::random_bf16(&config, RANDOM_MODEL_SEED).unwrap();

    let mut mismatched_device = config;
    mismatched_device.device_ordinal = 1;
    assert_eq!(
        ModelRunner::new(mismatched_device, weights).err(),
        Some(Status::InvalidArgument)
    );

    let weights = QwenWeights::random_bf16(&config, RANDOM_MODEL_SEED).unwrap();
    let mut mismatched_stream = config;
    mismatched_stream.stream = 1_usize as *mut _;
    assert_eq!(
        ModelRunner::new(mismatched_stream, weights).err(),
        Some(Status::InvalidArgument)
    );
}

#[test]
fn current_device_sentinel_is_resolved_for_weights_and_runner() {
    if !cuda_device_available() {
        return;
    }

    let explicit = QwenConfig::randomized_dense_tiny_fixture(0);
    let sentinel = QwenConfig::randomized_dense_tiny_fixture(-1);

    assert_cuda(
        unsafe { cudaSetDevice(0) },
        "set CUDA current device before sentinel weights",
    );
    let weights = QwenWeights::random_bf16(&sentinel, RANDOM_MODEL_SEED).unwrap();
    let runner = ModelRunner::new(explicit, weights).unwrap();
    drop(runner);

    assert_cuda(
        unsafe { cudaSetDevice(0) },
        "set CUDA current device before sentinel runner",
    );
    let weights = QwenWeights::random_bf16(&explicit, RANDOM_MODEL_SEED).unwrap();
    let runner = ModelRunner::new(sentinel, weights).unwrap();
    drop(runner);
}

#[test]
fn sentinel_weights_reject_runner_after_current_device_switch() {
    let Some(device_count) = cuda_device_count() else {
        return;
    };
    if device_count < 2 {
        eprintln!("SKIP: requires at least two CUDA devices");
        return;
    }

    let sentinel = QwenConfig::randomized_dense_tiny_fixture(-1);
    assert_cuda(
        unsafe { cudaSetDevice(0) },
        "set CUDA current device before sentinel weights",
    );
    let weights = QwenWeights::random_bf16(&sentinel, RANDOM_MODEL_SEED).unwrap();

    assert_cuda(
        unsafe { cudaSetDevice(1) },
        "switch CUDA current device before runner",
    );
    assert_eq!(
        ModelRunner::new(sentinel, weights).err(),
        Some(Status::InvalidArgument)
    );
    assert_cuda(unsafe { cudaSetDevice(0) }, "restore CUDA test device");
}

#[test]
fn runner_reactivates_bound_device_when_reusing_buffers() {
    let Some(device_count) = cuda_device_count() else {
        return;
    };
    if device_count < 2 {
        eprintln!("SKIP: requires at least two CUDA devices");
        return;
    }

    assert_cuda(
        unsafe { cudaSetDevice(0) },
        "set CUDA current device before runner",
    );
    let config = QwenConfig::randomized_dense_tiny_fixture(0);
    let mut runner = ModelRunner::random_bf16(config, RANDOM_MODEL_SEED).unwrap();
    let first = runner
        .run(QwenRequest {
            request_id: 61,
            tokens: &[1, 2, 3],
            max_new_tokens: 0,
        })
        .unwrap();

    assert_cuda(
        unsafe { cudaSetDevice(1) },
        "switch CUDA current device before buffer reuse",
    );
    let continued = runner
        .run(QwenRequest {
            request_id: 61,
            tokens: &first.live_tokens,
            max_new_tokens: 1,
        })
        .unwrap();

    assert_eq!(continued.generated_tokens.len(), 1);
    assert_eq!(
        &continued.live_tokens[..first.live_tokens.len()],
        first.live_tokens
    );
    assert_cuda(unsafe { cudaSetDevice(0) }, "restore CUDA test device");
}
