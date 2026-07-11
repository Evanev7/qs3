use std::{fmt, time::Duration};

use super::{
    plan::{QwenBf16LoadPlan, execute_qwen36_bf16_load_plan},
    transfer::{ManagedUmaBackend, result_from_cuda},
};
use crate::{
    QwenTokenizer, ffi,
    model::{ModelRunner, QwenRequest},
};
use std::{path::Path, ptr, time::Instant};

const MODEL_DIR: &str = "/home/exo/.cache/huggingface/hub/models--Qwen--Qwen3.6-35B-A3B/snapshots/995ad96eacd98c81ed38be0c5b274b04031597b0";
const REQUEST_ID: u64 = 0x5450_5300_0000_0001;
const TOKENIZE_SAMPLES: usize = 100;
const PREFILL_WARMUPS: usize = 2;
const PREFILL_SAMPLES: usize = 5;
const DECODE_WARMUPS: usize = 4;
const DECODE_SAMPLES: usize = 32;

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

const TSV_HEADER: &str =
    "metric\tclass\tsample\ttokens\tcontext_start\tcontext_end\telapsed_ms\ttokens_per_second";

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum SampleClass {
    Setup,
    Warmup,
    Measure,
    Summary,
}

impl fmt::Display for SampleClass {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(match self {
            Self::Setup => "setup",
            Self::Warmup => "warmup",
            Self::Measure => "measure",
            Self::Summary => "summary",
        })
    }
}

#[derive(Clone, Copy, Debug)]
struct BenchRow<'a> {
    metric: &'a str,
    class: SampleClass,
    sample: Option<usize>,
    tokens: Option<usize>,
    context_start: Option<usize>,
    context_end: Option<usize>,
    elapsed_ms: Option<f64>,
    tokens_per_second: Option<f64>,
}

impl<'a> BenchRow<'a> {
    fn setup(metric: &'a str, elapsed: Duration) -> Self {
        Self {
            metric,
            class: SampleClass::Setup,
            sample: None,
            tokens: None,
            context_start: None,
            context_end: None,
            elapsed_ms: Some(elapsed.as_secs_f64() * 1_000.0),
            tokens_per_second: None,
        }
    }

    fn sample(
        metric: &'a str,
        class: SampleClass,
        sample: usize,
        tokens: usize,
        context_start: usize,
        context_end: usize,
        elapsed: Duration,
    ) -> Self {
        Self {
            metric,
            class,
            sample: Some(sample),
            tokens: Some(tokens),
            context_start: Some(context_start),
            context_end: Some(context_end),
            elapsed_ms: Some(elapsed.as_secs_f64() * 1_000.0),
            tokens_per_second: tokens_per_second(tokens, elapsed),
        }
    }

    fn summary(
        metric: &'a str,
        tokens: usize,
        context_start: usize,
        context_end: usize,
        elapsed_seconds: f64,
        throughput: Option<f64>,
    ) -> Self {
        Self {
            metric,
            class: SampleClass::Summary,
            sample: None,
            tokens: Some(tokens),
            context_start: Some(context_start),
            context_end: Some(context_end),
            elapsed_ms: Some(elapsed_seconds * 1_000.0),
            tokens_per_second: throughput,
        }
    }
}

impl fmt::Display for BenchRow<'_> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        fn optional<T: fmt::Display>(value: Option<T>) -> String {
            value.map_or_else(|| "-".to_owned(), |value| value.to_string())
        }
        fn decimal(value: Option<f64>) -> String {
            value.map_or_else(|| "-".to_owned(), |value| format!("{value:.3}"))
        }

        write!(
            f,
            "{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}",
            self.metric,
            self.class,
            optional(self.sample),
            optional(self.tokens),
            optional(self.context_start),
            optional(self.context_end),
            decimal(self.elapsed_ms),
            decimal(self.tokens_per_second),
        )
    }
}

fn fnv1a_update(mut hash: u64, bytes: &[u8]) -> u64 {
    for &byte in bytes {
        hash ^= u64::from(byte);
        hash = hash.wrapping_mul(FNV_PRIME);
    }
    hash
}

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

fn measure_fresh_prefill(runner: &mut ModelRunner, prompt: &[i32]) -> Duration {
    runner.reset().expect("benchmark runner reset failed");
    synchronize_stream();
    let started = Instant::now();
    let result = runner
        .run(QwenRequest {
            request_id: REQUEST_ID,
            tokens: prompt,
            max_new_tokens: 0,
        })
        .expect("fresh benchmark prefill failed");
    let elapsed = started.elapsed();
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

fn run_real_qwen36_bf16_tps() {
    let model_dir = Path::new(MODEL_DIR);
    assert!(
        model_dir.is_dir(),
        "pinned model snapshot is missing: {MODEL_DIR}"
    );
    println!("\n{TSV_HEADER}");

    let started = Instant::now();
    let tokenizer =
        QwenTokenizer::from_model_dir(model_dir).expect("failed to load pinned Qwen tokenizer");
    println!("{}", BenchRow::setup("tokenizer_load", started.elapsed()));

    let started = Instant::now();
    let prompt = tokenizer.encode(FIXED_PROMPT);
    let cold_tokenize = started.elapsed();
    assert!(!prompt.is_empty());
    println!(
        "{}",
        BenchRow::sample(
            "tokenize_cold",
            SampleClass::Measure,
            0,
            prompt.len(),
            0,
            prompt.len(),
            cold_tokenize,
        )
    );
    println!(
        "# prompt_bytes={} prompt_tokens={} prompt_fnv1a={:016x}",
        FIXED_PROMPT.len(),
        prompt.len(),
        token_id_fingerprint(&prompt),
    );

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
    println!(
        "{}",
        BenchRow::summary(
            "tokenize_steady_p50",
            prompt.len(),
            0,
            prompt.len(),
            tokenize_median,
            Some(prompt.len() as f64 / tokenize_median),
        )
    );

    let started = Instant::now();
    let plan = QwenBf16LoadPlan::read(model_dir).expect("failed to build BF16 load plan");
    println!("{}", BenchRow::setup("weight_plan", started.elapsed()));

    let max_seq_len = u32::try_from(prompt.len() + DECODE_WARMUPS + DECODE_SAMPLES + 1)
        .expect("fixed benchmark sequence length exceeds u32");
    let backend = ManagedUmaBackend::new(0).expect("failed to create managed-UMA backend");
    let started = Instant::now();
    let loaded = execute_qwen36_bf16_load_plan(&plan, backend, ptr::null_mut())
        .expect("failed to load BF16 tensors");
    println!("{}", BenchRow::setup("weight_load", started.elapsed()));

    let started = Instant::now();
    let (config, weights) = loaded
        .into_qwen_model(ptr::null_mut(), max_seq_len)
        .expect("failed to materialize Qwen model weights");
    println!(
        "{}",
        BenchRow::setup("weight_materialize", started.elapsed())
    );

    let started = Instant::now();
    let mut runner = ModelRunner::new(config, weights).expect("failed to construct ModelRunner");
    println!("{}", BenchRow::setup("runner_init", started.elapsed()));

    for sample in 0..PREFILL_WARMUPS {
        let elapsed = measure_fresh_prefill(&mut runner, &prompt);
        println!(
            "{}",
            BenchRow::sample(
                "prefill_fresh_e2e",
                SampleClass::Warmup,
                sample,
                prompt.len(),
                0,
                prompt.len(),
                elapsed,
            )
        );
    }

    let mut prefill_times = Vec::with_capacity(PREFILL_SAMPLES);
    for sample in 0..PREFILL_SAMPLES {
        let elapsed = measure_fresh_prefill(&mut runner, &prompt);
        prefill_times.push(elapsed);
        println!(
            "{}",
            BenchRow::sample(
                "prefill_fresh_e2e",
                SampleClass::Measure,
                sample,
                prompt.len(),
                0,
                prompt.len(),
                elapsed,
            )
        );
    }
    let prefill_median = median_seconds(&prefill_times).expect("prefill samples are non-empty");
    println!(
        "{}",
        BenchRow::summary(
            "prefill_fresh_e2e_p50",
            prompt.len(),
            0,
            prompt.len(),
            prefill_median,
            Some(prompt.len() as f64 / prefill_median),
        )
    );

    runner.reset().expect("decode setup reset failed");
    let prefill = runner
        .run(QwenRequest {
            request_id: REQUEST_ID,
            tokens: &prompt,
            max_new_tokens: 0,
        })
        .expect("decode setup prefill failed");
    let mut live_tokens = prefill.live_tokens;

    for sample in 0..DECODE_WARMUPS {
        let context_start = live_tokens.len();
        let (elapsed, next_live_tokens) =
            measure_decode_step(&mut runner, &tokenizer, &live_tokens);
        live_tokens = next_live_tokens;
        println!(
            "{}",
            BenchRow::sample(
                "decode_sequential_e2e",
                SampleClass::Warmup,
                sample,
                1,
                context_start,
                live_tokens.len(),
                elapsed,
            )
        );
    }

    let decode_context_start = live_tokens.len();
    let mut decode_times = Vec::with_capacity(DECODE_SAMPLES);
    for sample in 0..DECODE_SAMPLES {
        let context_start = live_tokens.len();
        let (elapsed, next_live_tokens) =
            measure_decode_step(&mut runner, &tokenizer, &live_tokens);
        decode_times.push(elapsed);
        live_tokens = next_live_tokens;
        println!(
            "{}",
            BenchRow::sample(
                "decode_sequential_e2e",
                SampleClass::Measure,
                sample,
                1,
                context_start,
                live_tokens.len(),
                elapsed,
            )
        );
    }

    let decode_total = decode_times.iter().copied().sum::<Duration>();
    let decode_p50 = median_seconds(&decode_times).expect("decode samples are non-empty");
    let decode_p95 =
        nearest_rank_seconds(&decode_times, 95, 100).expect("decode p95 samples are non-empty");
    println!(
        "{}",
        BenchRow::summary(
            "decode_sequential_e2e_total",
            DECODE_SAMPLES,
            decode_context_start,
            live_tokens.len(),
            decode_total.as_secs_f64(),
            tokens_per_second(DECODE_SAMPLES, decode_total),
        )
    );
    println!(
        "{}",
        BenchRow::summary(
            "decode_sequential_e2e_p50",
            1,
            decode_context_start,
            live_tokens.len(),
            decode_p50,
            Some(1.0 / decode_p50),
        )
    );
    println!(
        "{}",
        BenchRow::summary(
            "decode_sequential_e2e_p95",
            1,
            decode_context_start,
            live_tokens.len(),
            decode_p95,
            Some(1.0 / decode_p95),
        )
    );
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
        assert_eq!(DECODE_SAMPLES, 32);
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

    #[test]
    fn tsv_rows_keep_missing_fields_explicit() {
        let row = BenchRow {
            metric: "decode_sequential_e2e",
            class: SampleClass::Measure,
            sample: Some(3),
            tokens: Some(1),
            context_start: Some(128),
            context_end: Some(129),
            elapsed_ms: Some(12.5),
            tokens_per_second: Some(80.0),
        };
        assert_eq!(
            row.to_string(),
            "decode_sequential_e2e\tmeasure\t3\t1\t128\t129\t12.500\t80.000"
        );

        let setup = BenchRow::setup("weight_plan", Duration::from_millis(7));
        assert_eq!(
            setup.to_string(),
            "weight_plan\tsetup\t-\t-\t-\t-\t7.000\t-"
        );

        let warmup = BenchRow::sample(
            "prefill_fresh_e2e",
            SampleClass::Warmup,
            0,
            96,
            0,
            96,
            Duration::from_millis(500),
        );
        assert_eq!(
            warmup.to_string(),
            "prefill_fresh_e2e\twarmup\t0\t96\t0\t96\t500.000\t192.000"
        );

        let summary = BenchRow::summary("prefill_fresh_e2e_p50", 96, 0, 96, 0.5, Some(192.0));
        assert_eq!(
            summary.to_string(),
            "prefill_fresh_e2e_p50\tsummary\t-\t96\t0\t96\t500.000\t192.000"
        );
    }
}

#[test]
#[ignore = "loads and benchmarks the full real Qwen3.6 BF16 model"]
fn real_qwen36_bf16_tps() {
    run_real_qwen36_bf16_tps();
}
