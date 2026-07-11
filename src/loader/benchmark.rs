use std::{fmt, time::Duration};

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

fn token_id_fingerprint(tokens: &[i32]) -> u64 {
    tokens.iter().fold(FNV_OFFSET, |mut hash, token| {
        for byte in token.to_le_bytes() {
            hash ^= u64::from(byte);
            hash = hash.wrapping_mul(FNV_PRIME);
        }
        hash
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

#[cfg(test)]
mod tests {
    use super::*;

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
