use std::{collections::HashMap, time::Duration};

use tinyjson::JsonValue;

use super::{
    plan::{QwenBf16LoadPlan, execute_qwen36_bf16_load_plan},
    transfer::{ManagedUmaBackend, result_from_cuda},
};
use crate::test_assets::require_real_qwen36_model_dir;
use crate::{
    QwenTokenizer, ffi,
    model::{GdnRecurrentPrecision, ModelRunner, MoeBf16Kernel, QwenRequest},
};
use std::{ptr, time::Instant};

unsafe extern "C" {
    fn cudaProfilerStart() -> i32;
    fn cudaProfilerStop() -> i32;
}

const REQUEST_ID: u64 = 0x5450_5300_0000_0001;
const TOKENIZE_SAMPLES: usize = 100;
const PREFILL_WARMUPS: usize = 2;
const PREFILL_SAMPLES: usize = 5;
const DECODE_WARMUPS: usize = 4;
const DEFAULT_DECODE_SAMPLES: usize = 32;

const FIXED_PROMPT: &str = concat!(
    "<|im_start|>system\n",
    "You are a concise technical assistant. Explain systems accurately, distinguish measured facts from estimates, and avoid unnecessary jargon.\n",
    "<|im_end|>\n",
    "<|im_start|>user\n",
    "Explain how a transformer language model turns a prompt into the next token. Cover tokenization, embeddings, attention, feed-forward layers, normalization, logits, and sampling. Distinguish prompt processing from autoregressive decoding, and mention why memory bandwidth matters. Use plain language and keep the answer under 250 words.\n",
    "<|im_end|>\n",
    "<|im_start|>assistant\n",
);

const FNV_OFFSET: u64 = 0xcbf2_9ce4_8422_2325;
const FNV_PRIME: u64 = 0x0000_0100_0000_01b3;

fn object(fields: impl IntoIterator<Item = (&'static str, JsonValue)>) -> JsonValue {
    fields
        .into_iter()
        .map(|(key, value)| (key.to_owned(), value))
        .collect::<HashMap<_, _>>()
        .into()
}

fn milliseconds(elapsed: Duration) -> JsonValue {
    (elapsed.as_secs_f64() * 1_000.0).into()
}

fn sample_milliseconds(samples: &[Duration]) -> JsonValue {
    samples
        .iter()
        .copied()
        .map(milliseconds)
        .collect::<Vec<_>>()
        .into()
}

fn fnv1a_update(mut hash: u64, bytes: &[u8]) -> u64 {
    for &byte in bytes {
        hash ^= u64::from(byte);
        hash = hash.wrapping_mul(FNV_PRIME);
    }
    hash
}

#[cfg(test)]
fn byte_fingerprint(bytes: &[u8]) -> u64 {
    fnv1a_update(FNV_OFFSET, bytes)
}

fn token_id_fingerprint(tokens: &[i32]) -> u64 {
    tokens.iter().fold(FNV_OFFSET, |hash, token| {
        fnv1a_update(hash, &token.to_le_bytes())
    })
}

fn sorted_nanos(samples: &[Duration]) -> Option<Vec<u128>> {
    if samples.is_empty() {
        return None;
    }
    let mut values = samples.iter().map(Duration::as_nanos).collect::<Vec<_>>();
    values.sort_unstable();
    Some(values)
}

fn median_seconds(samples: &[Duration]) -> Option<f64> {
    let values = sorted_nanos(samples)?;
    let middle = values.len() / 2;
    let nanos = if values.len() % 2 == 0 {
        (values[middle - 1] + values[middle]) as f64 / 2.0
    } else {
        values[middle] as f64
    };
    Some(nanos / 1_000_000_000.0)
}

fn nearest_rank_seconds(samples: &[Duration], numerator: usize, denominator: usize) -> Option<f64> {
    if numerator == 0 || denominator == 0 || numerator > denominator {
        return None;
    }
    let values = sorted_nanos(samples)?;
    let rank = values
        .len()
        .checked_mul(numerator)?
        .div_ceil(denominator)
        .clamp(1, values.len());
    Some(values[rank - 1] as f64 / 1_000_000_000.0)
}

fn tokens_per_second(tokens: usize, elapsed: Duration) -> Option<f64> {
    if tokens == 0 || elapsed.is_zero() {
        return None;
    }
    Some(tokens as f64 / elapsed.as_secs_f64())
}

fn synchronize_stream() {
    result_from_cuda(unsafe { ffi::cuda::cudaStreamSynchronize(ptr::null_mut()) })
        .expect("benchmark stream synchronization failed");
}

fn positive_env(name: &str, default: usize) -> usize {
    match std::env::var(name) {
        Err(std::env::VarError::NotPresent) => default,
        Ok(value) => value
            .parse::<usize>()
            .ok()
            .filter(|&value| value > 0)
            .unwrap_or_else(|| panic!("{name} must be a positive integer")),
        Err(error) => panic!("invalid {name}: {error}"),
    }
}

fn measure_fresh_prefill(runner: &mut ModelRunner, prompt: &[i32], profile: bool) -> Duration {
    runner.reset().expect("benchmark runner reset failed");
    synchronize_stream();
    if profile {
        result_from_cuda(unsafe { cudaProfilerStart() }).expect("start prefill profiling range");
    }
    let started = Instant::now();
    let result = runner
        .run(QwenRequest {
            request_id: REQUEST_ID,
            tokens: prompt,
            max_new_tokens: 0,
        })
        .expect("fresh benchmark prefill failed");
    let elapsed = started.elapsed();
    if profile {
        result_from_cuda(unsafe { cudaProfilerStop() }).expect("stop prefill profiling range");
    }
    assert!(result.generated_tokens.is_empty());
    assert_eq!(result.live_tokens, prompt);
    elapsed
}

fn measure_decode_step(
    runner: &mut ModelRunner,
    tokenizer: &QwenTokenizer,
    live_tokens: &[i32],
) -> (Duration, Vec<i32>) {
    let started = Instant::now();
    let result = runner
        .run(QwenRequest {
            request_id: REQUEST_ID,
            tokens: live_tokens,
            max_new_tokens: 1,
        })
        .expect("sequential benchmark decode failed");
    let elapsed = started.elapsed();
    assert_eq!(result.generated_tokens.len(), 1);
    assert_eq!(result.live_tokens.len(), live_tokens.len() + 1);
    tokenizer
        .decode(&result.generated_tokens)
        .expect("model generated a non-tokenizer or padded vocabulary ID");
    (elapsed, result.live_tokens)
}

pub fn run_core_benchmark() -> JsonValue {
    let decode_samples = positive_env("QS3_BENCH_DECODE_SAMPLES", DEFAULT_DECODE_SAMPLES);
    let profile_phase = match std::env::var("QS3_PROFILE") {
        Err(std::env::VarError::NotPresent) => "none",
        Ok(value) if value == "decode" => "decode",
        Ok(value) if value == "prefill" => "prefill",
        _ => panic!("QS3_PROFILE must be unset, decode or prefill"),
    };
    let model_dir = require_real_qwen36_model_dir();
    let started = Instant::now();
    let tokenizer =
        QwenTokenizer::from_model_dir(&model_dir).expect("failed to load pinned Qwen tokenizer");
    let tokenizer_load = started.elapsed();

    let started = Instant::now();
    let prompt = tokenizer.encode(FIXED_PROMPT);
    let cold_tokenize = started.elapsed();
    assert!(!prompt.is_empty());
    let mut tokenize_times = Vec::with_capacity(TOKENIZE_SAMPLES);
    for _ in 0..TOKENIZE_SAMPLES {
        let started = Instant::now();
        let repeated = tokenizer.encode(FIXED_PROMPT);
        tokenize_times.push(started.elapsed());
        assert_eq!(
            repeated, prompt,
            "fixed prompt tokenization changed within one run"
        );
    }
    let tokenize_median = median_seconds(&tokenize_times).expect("tokenizer samples are non-empty");
    let context_tokens = positive_env("QS3_BENCH_CONTEXT_TOKENS", prompt.len());
    // Longer-context comparisons repeat the exact base token IDs, avoiding
    // tokenizer boundary differences between runtimes.
    let prompt = prompt
        .iter()
        .copied()
        .cycle()
        .take(context_tokens)
        .collect::<Vec<_>>();

    let started = Instant::now();
    let plan = QwenBf16LoadPlan::read(&model_dir).expect("failed to build BF16 load plan");
    let weight_plan = started.elapsed();
    let max_seq_len = u32::try_from(
        prompt
            .len()
            .checked_add(DECODE_WARMUPS)
            .and_then(|len| len.checked_add(decode_samples))
            .and_then(|len| len.checked_add(1))
            .expect("benchmark sequence length overflow"),
    )
    .expect("fixed benchmark sequence length exceeds u32");
    let backend = ManagedUmaBackend::new(0).expect("failed to create managed-UMA backend");
    let started = Instant::now();
    let loaded = execute_qwen36_bf16_load_plan(&plan, backend, ptr::null_mut())
        .expect("failed to load BF16 tensors");
    let weight_load = started.elapsed();

    let started = Instant::now();
    let (mut config, weights) = loaded
        .into_qwen_model(ptr::null_mut(), max_seq_len)
        .expect("failed to materialize Qwen model weights");
    let weight_materialize = started.elapsed();
    config.gdn_recurrent_precision = match std::env::var("QS3_BENCH_GDN_STATE") {
        Err(std::env::VarError::NotPresent) => config.gdn_recurrent_precision,
        Ok(value) if value == "bf16" => GdnRecurrentPrecision::Bf16,
        Ok(value) if value == "f32" => GdnRecurrentPrecision::F32,
        _ => panic!("QS3_BENCH_GDN_STATE must be unset, bf16 or f32"),
    };
    config.moe_bf16_kernel = match std::env::var("QS3_BENCH_MOE_KERNEL") {
        Err(std::env::VarError::NotPresent) => config.moe_bf16_kernel,
        Ok(value) if value == "tile128_blocks4" => MoeBf16Kernel::CutlassTile128Blocks4,
        Ok(value) if value == "tile128_blocks96" => MoeBf16Kernel::CutlassTile128Blocks96,
        Ok(value) if value == "tile32_blocks96" => MoeBf16Kernel::CutlassTile32Blocks96,
        _ => panic!("QS3_BENCH_MOE_KERNEL must name tile128_blocks4, tile128_blocks96 or tile32_blocks96"),
    };
    let started = Instant::now();
    let mut runner = ModelRunner::new(config, weights).expect("failed to construct ModelRunner");
    let runner_init = started.elapsed();

    let mut prefill_warmup_times = Vec::with_capacity(PREFILL_WARMUPS);
    for _ in 0..PREFILL_WARMUPS {
        prefill_warmup_times.push(measure_fresh_prefill(&mut runner, &prompt, false));
    }
    let mut prefill_times = Vec::with_capacity(PREFILL_SAMPLES);
    for sample in 0..PREFILL_SAMPLES {
        prefill_times.push(measure_fresh_prefill(
            &mut runner,
            &prompt,
            profile_phase == "prefill" && sample == 0,
        ));
    }
    let prefill_median = median_seconds(&prefill_times).expect("prefill samples are non-empty");

    runner.reset().expect("decode setup reset failed");
    let prefill = runner
        .run(QwenRequest {
            request_id: REQUEST_ID,
            tokens: &prompt,
            max_new_tokens: 0,
        })
        .expect("decode setup prefill failed");
    let mut live_tokens = prefill.live_tokens;
    let mut decode_warmup_times = Vec::with_capacity(DECODE_WARMUPS);
    for _ in 0..DECODE_WARMUPS {
        let (elapsed, next_live_tokens) =
            measure_decode_step(&mut runner, &tokenizer, &live_tokens);
        decode_warmup_times.push(elapsed);
        live_tokens = next_live_tokens;
    }
    let decode_context_start = live_tokens.len();
    let mut decode_times = Vec::with_capacity(decode_samples);
    // Nsight Systems can capture only this prepared, steady-decode interval.
    // Profiler API calls stay outside the per-step latency samples.
    if profile_phase == "decode" {
        result_from_cuda(unsafe { cudaProfilerStart() }).expect("start decode profiling range");
    }
    for _ in 0..decode_samples {
        let (elapsed, next_live_tokens) =
            measure_decode_step(&mut runner, &tokenizer, &live_tokens);
        decode_times.push(elapsed);
        live_tokens = next_live_tokens;
    }
    if profile_phase == "decode" {
        result_from_cuda(unsafe { cudaProfilerStop() }).expect("stop decode profiling range");
    }
    let decode_total = decode_times.iter().copied().sum::<Duration>();
    let decode_p50 = median_seconds(&decode_times).expect("decode samples are non-empty");
    let decode_p95 =
        nearest_rank_seconds(&decode_times, 95, 100).expect("decode p95 samples are non-empty");

    // Emit numeric samples and summaries together, without a second text format.
    object([
        ("model", model_dir.to_string_lossy().into_owned().into()),
        (
            "execution",
            object([
                ("mode", "eager".to_owned().into()),
                ("precision", "bf16".to_owned().into()),
                ("cuda_profiler_range", (profile_phase != "none").into()),
                ("cuda_profiler_phase", profile_phase.to_owned().into()),
                ("linear", "cublaslt_prepared_f32_accum".to_owned().into()),
                ("attention", "flashinfer_paged_hd256_gqa8".to_owned().into()),
                ("norm", "flashinfer_gemma_aot".to_owned().into()),
                ("gdn", "qscu_warp4_row128_bf16".to_owned().into()),
                ("gdn_prefill_conv", "qscu_parallel_tokens".to_owned().into()),
                ("gdn_decode_conv", "qscu_tiled_channels".to_owned().into()),
                ("router", "qscu_warp32_top8_e256".to_owned().into()),
                ("router_logits_dtype", "bf16".to_owned().into()),
                (
                    "moe_threadblocks",
                    f64::from(config.moe_bf16_kernel.threadblocks()).into(),
                ),
                ("moe_kernel", config.moe_bf16_kernel.as_str().to_owned().into()),
                ("moe_cta_tile", config.moe_bf16_kernel.cta_tile().into_iter()
                    .map(|dim| JsonValue::Number(f64::from(dim))).collect::<Vec<_>>().into()),
                ("gdn_conv_state_dtype", "bf16".to_owned().into()),
                (
                    "gdn_recurrent_state_dtype",
                    config.gdn_recurrent_precision.as_str().to_owned().into(),
                ),
                (
                    "moe",
                    "cutlass_sm80_stages2_bf16".to_owned().into(),
                ),
                (
                    "linear_workspace_bytes",
                    (config.qscb_workspace_bytes as f64).into(),
                ),
                (
                    "attention_float_workspace_bytes",
                    (config.qsfi_float_workspace_bytes as f64).into(),
                ),
                (
                    "attention_int_workspace_bytes",
                    (config.qsfi_int_workspace_bytes as f64).into(),
                ),
                (
                    "attention_host_workspace_bytes",
                    (config.qsfi_host_int_workspace_bytes as f64).into(),
                ),
                ("weight_backend", "managed_uma".to_owned().into()),
                ("sampling", "greedy".to_owned().into()),
                ("logits_rows", "final_token".to_owned().into()),
            ]),
        ),
        (
            "prompt",
            object([
                ("base_bytes", (FIXED_PROMPT.len() as f64).into()),
                ("construction", "repeat_base_token_ids".to_owned().into()),
                ("tokens", (prompt.len() as f64).into()),
                (
                    "token_id_fnv1a",
                    format!("{:016x}", token_id_fingerprint(&prompt)).into(),
                ),
            ]),
        ),
        (
            "setup_ms",
            object([
                ("tokenizer_load", milliseconds(tokenizer_load)),
                ("weight_plan", milliseconds(weight_plan)),
                ("weight_load", milliseconds(weight_load)),
                ("weight_materialize", milliseconds(weight_materialize)),
                ("runner_init", milliseconds(runner_init)),
            ]),
        ),
        (
            "tokenize",
            object([
                ("samples", (TOKENIZE_SAMPLES as f64).into()),
                ("cold_ms", milliseconds(cold_tokenize)),
                ("p50_ms", (tokenize_median * 1_000.0).into()),
            ]),
        ),
        (
            "prefill",
            object([
                ("warmups", (PREFILL_WARMUPS as f64).into()),
                ("samples", (PREFILL_SAMPLES as f64).into()),
                ("warmup_ms", sample_milliseconds(&prefill_warmup_times)),
                ("sample_ms", sample_milliseconds(&prefill_times)),
                ("context_start", 0.0.into()),
                ("context_end", (prompt.len() as f64).into()),
                ("p50_ms", (prefill_median * 1_000.0).into()),
                (
                    "tokens_per_second",
                    (prompt.len() as f64 / prefill_median).into(),
                ),
            ]),
        ),
        (
            "decode",
            object([
                ("warmups", (DECODE_WARMUPS as f64).into()),
                ("samples", (decode_samples as f64).into()),
                ("warmup_ms", sample_milliseconds(&decode_warmup_times)),
                ("sample_ms", sample_milliseconds(&decode_times)),
                ("context_start", (decode_context_start as f64).into()),
                ("context_end", (live_tokens.len() as f64).into()),
                (
                    "generated_token_ids",
                    live_tokens[prompt.len()..]
                        .iter()
                        .map(|&token| JsonValue::Number(f64::from(token)))
                        .collect::<Vec<_>>()
                        .into(),
                ),
                ("total_ms", milliseconds(decode_total)),
                (
                    "tokens_per_second",
                    tokens_per_second(decode_samples, decode_total)
                        .expect("nonzero decode time")
                        .into(),
                ),
                ("p50_ms", (decode_p50 * 1_000.0).into()),
                ("p95_ms", (decode_p95 * 1_000.0).into()),
            ]),
        ),
    ])
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fixed_workload_contract_does_not_drift() {
        assert!(FIXED_PROMPT.starts_with("<|im_start|>system\n"));
        assert!(FIXED_PROMPT.ends_with("<|im_start|>assistant\n"));
        assert_eq!(FIXED_PROMPT.len(), 556);
        assert_eq!(
            byte_fingerprint(FIXED_PROMPT.as_bytes()),
            0x329f_8f7b_8aef_d628
        );
        assert_eq!(TOKENIZE_SAMPLES, 100);
        assert_eq!(PREFILL_WARMUPS, 2);
        assert_eq!(PREFILL_SAMPLES, 5);
        assert_eq!(DECODE_WARMUPS, 4);
        assert_eq!(DEFAULT_DECODE_SAMPLES, 32);
    }

    #[test]
    fn token_fingerprint_is_stable_over_little_endian_i32_bytes() {
        assert_eq!(token_id_fingerprint(&[1, -2]), 0x222a_d8e9_836c_c591);
        assert_ne!(
            token_id_fingerprint(&[1, -2]),
            token_id_fingerprint(&[-2, 1])
        );
    }

    #[test]
    fn latency_summaries_handle_odd_even_and_nearest_rank_samples() {
        let odd = [
            Duration::from_millis(3),
            Duration::from_millis(1),
            Duration::from_millis(2),
        ];
        let even = [
            Duration::from_millis(4),
            Duration::from_millis(1),
            Duration::from_millis(3),
            Duration::from_millis(2),
        ];
        let ranks = (1..=20).map(Duration::from_millis).collect::<Vec<_>>();

        assert_eq!(median_seconds(&odd), Some(0.002));
        assert_eq!(median_seconds(&even), Some(0.0025));
        assert_eq!(nearest_rank_seconds(&ranks, 95, 100), Some(0.019));
        assert_eq!(median_seconds(&[]), None);
        assert_eq!(nearest_rank_seconds(&odd, 0, 100), None);
    }

    #[test]
    fn throughput_rejects_empty_work_and_zero_time() {
        assert_eq!(tokens_per_second(8, Duration::from_secs(2)), Some(4.0));
        assert_eq!(tokens_per_second(0, Duration::from_secs(2)), None);
        assert_eq!(tokens_per_second(8, Duration::ZERO), None);
    }
}
