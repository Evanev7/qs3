use qs3::{ModelRunner, QwenConfig, QwenMoeConfig, QwenRequest};

use std::env;
use std::ffi::{CStr, c_char};
use std::time::{Duration, Instant};

const CUDA_SUCCESS: i32 = 0;
const RANDOM_MODEL_SEED: u64 = 0x5153_3300_b36c_0001;
const DEFAULT_WARMUPS: u32 = 5;
const DEFAULT_ITERS: u32 = 20;

type BenchResult<T> = Result<T, String>;

unsafe extern "C" {
    fn cudaGetDeviceCount(count: *mut i32) -> i32;
    fn cudaGetErrorString(error: i32) -> *const c_char;
    fn cudaSetDevice(device: i32) -> i32;
    fn cudaDeviceSynchronize() -> i32;
}

#[derive(Debug)]
struct Options {
    warmups: u32,
    iters: u32,
    tokens: Vec<u32>,
}

impl Default for Options {
    fn default() -> Self {
        Self {
            warmups: DEFAULT_WARMUPS,
            iters: DEFAULT_ITERS,
            tokens: vec![1, 4, 8],
        }
    }
}

fn main() {
    if let Err(err) = run() {
        eprintln!("{err}");
        std::process::exit(1);
    }
}

fn run() -> BenchResult<()> {
    let options = parse_options(env::args().collect())?;
    ensure_cuda_device()?;

    println!("case\ttokens\tphase\twarmups\titers\tavg_us");
    for tokens in &options.tokens {
        let avg_us = bench_prefill(&options, *tokens)?;
        print_row(*tokens, "prefill", &options, avg_us);

        let avg_us = bench_hot_decode(&options, *tokens)?;
        print_row(*tokens, "decode_hot", &options, avg_us);
    }
    Ok(())
}

fn parse_options(args: Vec<String>) -> BenchResult<Options> {
    let mut options = Options::default();
    let mut idx = 1;
    while idx < args.len() {
        match args[idx].as_str() {
            "--warmups" => {
                idx += 1;
                let value = args.get(idx).ok_or_else(usage)?;
                options.warmups = parse_u32(value, true)?;
            }
            "--iters" => {
                idx += 1;
                let value = args.get(idx).ok_or_else(usage)?;
                options.iters = parse_u32(value, false)?;
            }
            "--tokens" => {
                idx += 1;
                let value = args.get(idx).ok_or_else(usage)?;
                options.tokens = parse_token_list(value)?;
            }
            "--help" | "-h" => return Err(usage()),
            _ => return Err(usage()),
        }
        idx += 1;
    }
    if options.iters == 0 || options.tokens.is_empty() {
        return Err(usage());
    }
    Ok(options)
}

fn usage() -> String {
    "usage: qs3_model_bench [--warmups N] [--iters N] [--tokens A,B,C]".to_owned()
}

fn parse_u32(text: &str, allow_zero: bool) -> BenchResult<u32> {
    let value = text
        .parse::<u32>()
        .map_err(|_| format!("invalid integer: {text}"))?;
    if !allow_zero && value == 0 {
        return Err(format!("value must be non-zero: {text}"));
    }
    Ok(value)
}

fn parse_token_list(text: &str) -> BenchResult<Vec<u32>> {
    let mut tokens = Vec::new();
    for item in text.split(',') {
        tokens.push(parse_u32(item, false)?);
    }
    if tokens.is_empty() {
        return Err("token list must not be empty".to_owned());
    }
    Ok(tokens)
}

fn print_row(tokens: u32, phase: &str, options: &Options, avg_us: f64) {
    println!(
        "full_attention_full_moe\t{}\t{}\t{}\t{}\t{:.3}",
        tokens, phase, options.warmups, options.iters, avg_us
    );
}

fn ensure_cuda_device() -> BenchResult<()> {
    let mut count = 0;
    cuda_check(
        unsafe { cudaGetDeviceCount(&mut count) },
        "cudaGetDeviceCount",
    )?;
    if count <= 0 {
        return Err("no CUDA device available".to_owned());
    }
    cuda_check(unsafe { cudaSetDevice(0) }, "cudaSetDevice")?;
    Ok(())
}

fn cuda_check(err: i32, what: &str) -> BenchResult<()> {
    if err == CUDA_SUCCESS {
        return Ok(());
    }
    Err(format!("{what} failed: {} ({err})", cuda_error_string(err)))
}

fn cuda_synchronize() -> BenchResult<()> {
    cuda_check(unsafe { cudaDeviceSynchronize() }, "cudaDeviceSynchronize")
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

fn model_config(options: &Options, prompt_tokens: u32) -> BenchResult<QwenConfig> {
    let measured_decode_tokens = options
        .warmups
        .checked_add(options.iters)
        .and_then(|value| value.checked_add(1))
        .ok_or_else(|| "warmups + iters overflows u32".to_owned())?;
    let required_seq_len = prompt_tokens
        .checked_add(measured_decode_tokens)
        .ok_or_else(|| "prompt tokens + decode tokens overflows u32".to_owned())?;

    let moe = QwenMoeConfig::qwen36_35b_a3b();
    let mut config = QwenConfig::randomized_dense_tiny_fixture(0);
    config.intermediate_size = moe.moe_intermediate_size;
    config.moe = Some(moe);
    config.max_seq_len = config.max_seq_len.max(required_seq_len);
    config.max_pages = config
        .max_pages
        .max(div_ceil(config.max_seq_len, config.page_size).saturating_add(1));
    Ok(config)
}

fn div_ceil(value: u32, divisor: u32) -> u32 {
    value / divisor + u32::from(value % divisor != 0)
}

fn bench_prefill(options: &Options, tokens: u32) -> BenchResult<f64> {
    let config = model_config(options, tokens)?;
    let prompt = make_prompt(tokens, config.vocab_size)?;
    let mut runner = make_runner(config)?;

    for _ in 0..options.warmups {
        runner_reset(&mut runner)?;
        let result = runner_run(
            &mut runner,
            QwenRequest {
                request_id: 11,
                tokens: &prompt,
                max_new_tokens: 0,
            },
            "prefill warmup",
        )?;
        expect_live_len(&result.live_tokens, tokens, "prefill warmup")?;
        cuda_synchronize()?;
    }

    let mut elapsed = Duration::ZERO;
    for _ in 0..options.iters {
        runner_reset(&mut runner)?;
        let start = Instant::now();
        let result = runner_run(
            &mut runner,
            QwenRequest {
                request_id: 11,
                tokens: &prompt,
                max_new_tokens: 0,
            },
            "prefill iteration",
        )?;
        cuda_synchronize()?;
        elapsed += start.elapsed();
        expect_live_len(&result.live_tokens, tokens, "prefill iteration")?;
    }
    Ok(avg_us(elapsed, options.iters))
}

fn bench_hot_decode(options: &Options, tokens: u32) -> BenchResult<f64> {
    let config = model_config(options, tokens)?;
    let prompt = make_prompt(tokens, config.vocab_size)?;
    let mut runner = make_runner(config)?;

    let prefill = runner_run(
        &mut runner,
        QwenRequest {
            request_id: 29,
            tokens: &prompt,
            max_new_tokens: 0,
        },
        "decode prefill",
    )?;
    let mut live_tokens = prefill.live_tokens;
    expect_live_len(&live_tokens, tokens, "decode prefill")?;

    for _ in 0..options.warmups {
        let result = runner_run(
            &mut runner,
            QwenRequest {
                request_id: 29,
                tokens: &live_tokens,
                max_new_tokens: 1,
            },
            "decode warmup",
        )?;
        expect_generated_len(&result.generated_tokens, 1, "decode warmup")?;
        live_tokens = result.live_tokens;
        cuda_synchronize()?;
    }

    let mut elapsed = Duration::ZERO;
    for _ in 0..options.iters {
        let start = Instant::now();
        let result = runner_run(
            &mut runner,
            QwenRequest {
                request_id: 29,
                tokens: &live_tokens,
                max_new_tokens: 1,
            },
            "decode iteration",
        )?;
        cuda_synchronize()?;
        elapsed += start.elapsed();
        expect_generated_len(&result.generated_tokens, 1, "decode iteration")?;
        live_tokens = result.live_tokens;
    }
    Ok(avg_us(elapsed, options.iters))
}

fn make_runner(config: QwenConfig) -> BenchResult<ModelRunner> {
    ModelRunner::random_bf16(config, RANDOM_MODEL_SEED)
        .map_err(|status| format!("ModelRunner::random_bf16 failed: {status:?}"))
}

fn runner_reset(runner: &mut ModelRunner) -> BenchResult<()> {
    runner
        .reset()
        .map_err(|status| format!("ModelRunner::reset failed: {status:?}"))
}

fn runner_run<'a>(
    runner: &mut ModelRunner,
    request: QwenRequest<'a>,
    label: &str,
) -> BenchResult<qs3::QwenResult> {
    runner
        .run(request)
        .map_err(|status| format!("{label} ModelRunner::run failed: {status:?}"))
}

fn make_prompt(tokens: u32, vocab_size: u32) -> BenchResult<Vec<i32>> {
    let mut prompt = Vec::new();
    prompt
        .try_reserve(tokens as usize)
        .map_err(|_| "failed to reserve prompt tokens".to_owned())?;
    for idx in 0..tokens {
        let token = (idx.wrapping_mul(17).wrapping_add(3) % vocab_size) as i32;
        prompt.push(token);
    }
    Ok(prompt)
}

fn expect_live_len(tokens: &[i32], expected: u32, label: &str) -> BenchResult<()> {
    if tokens.len() != expected as usize {
        return Err(format!(
            "{label} produced {} live tokens, expected {expected}",
            tokens.len()
        ));
    }
    Ok(())
}

fn expect_generated_len(tokens: &[i32], expected: usize, label: &str) -> BenchResult<()> {
    if tokens.len() != expected {
        return Err(format!(
            "{label} produced {} generated tokens, expected {expected}",
            tokens.len()
        ));
    }
    Ok(())
}

fn avg_us(elapsed: Duration, iters: u32) -> f64 {
    elapsed.as_secs_f64() * 1_000_000.0 / f64::from(iters)
}
