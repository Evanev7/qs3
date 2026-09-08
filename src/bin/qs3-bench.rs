use std::{
    collections::HashMap,
    process::{Command, Stdio},
    time::{SystemTime, UNIX_EPOCH},
};

use tinyjson::JsonValue;

unsafe extern "C" {
    fn cudaRuntimeGetVersion(version: *mut i32) -> i32;
}

fn object(fields: impl IntoIterator<Item = (&'static str, JsonValue)>) -> JsonValue {
    fields
        .into_iter()
        .map(|(key, value)| (key.to_owned(), value))
        .collect::<HashMap<_, _>>()
        .into()
}

fn probe(program: &str, args: &[&str]) -> Option<String> {
    let output = Command::new(program)
        .args(args)
        .stdin(Stdio::null())
        .output()
        .ok()?;
    if !output.status.success() {
        return None;
    }
    let value = String::from_utf8(output.stdout).ok()?.trim().to_owned();
    (!value.is_empty()).then_some(value)
}

fn optional(value: Option<String>) -> JsonValue {
    value.map_or(JsonValue::Null, JsonValue::from)
}

fn cuda_runtime_version() -> JsonValue {
    let mut version = 0;
    // Query the linked runtime. A runtime PATH probe can report a different
    // toolkit from the one Nix linked, or find no toolkit at all.
    let status = unsafe { cudaRuntimeGetVersion(&mut version) };
    assert_eq!(status, 0, "query linked CUDA runtime version");
    format!("{}.{}", version / 1000, (version % 1000) / 10).into()
}

fn collect() -> JsonValue {
    object([
        ("host", optional(probe("hostname", &[]))),
        (
            "gpu",
            optional(probe(
                "nvidia-smi",
                &["--id=0", "--query-gpu=name", "--format=csv,noheader"],
            )),
        ),
        (
            "driver",
            optional(probe(
                "nvidia-smi",
                &[
                    "--id=0",
                    "--query-gpu=driver_version",
                    "--format=csv,noheader",
                ],
            )),
        ),
        ("cuda_runtime", cuda_runtime_version()),
        ("rust", env!("QS3_RUSTC_VERSION").to_owned().into()),
    ])
}

fn timestamp(time: SystemTime) -> String {
    let elapsed = time
        .duration_since(UNIX_EPOCH)
        .expect("clock before Unix epoch");
    let seconds: libc::time_t = elapsed.as_secs().try_into().expect("timestamp overflow");
    let mut calendar = std::mem::MaybeUninit::<libc::tm>::uninit();
    // gmtime_r initializes the caller-owned calendar and is independent of TZ.
    let calendar = unsafe {
        assert!(!libc::gmtime_r(&seconds, calendar.as_mut_ptr()).is_null());
        calendar.assume_init()
    };
    format!(
        "{:04}-{:02}-{:02}T{:02}:{:02}:{:02}.{:09}+00:00",
        calendar.tm_year + 1900,
        calendar.tm_mon + 1,
        calendar.tm_mday,
        calendar.tm_hour,
        calendar.tm_min,
        calendar.tm_sec,
        elapsed.subsec_nanos(),
    )
}

fn record(measure: impl FnOnce() -> JsonValue) -> JsonValue {
    // Probe before timing; the measurement function owns only workload metrics.
    let metadata = collect();
    let started_at = timestamp(SystemTime::now());
    let mut measurement = measure();
    let finished_at = timestamp(SystemTime::now());
    let fields = measurement
        .get_mut::<HashMap<String, JsonValue>>()
        .expect("benchmark measurement is an object");
    fields.insert(
        "profile".to_owned(),
        env!("QS3_BUILD_PROFILE").to_owned().into(),
    );
    fields.insert("started_at".to_owned(), started_at.into());
    fields.insert("finished_at".to_owned(), finished_at.into());
    object([("metadata", metadata), ("measurement", measurement)])
}

#[test]
fn utc_timestamps_preserve_subseconds_and_calendar_boundaries() {
    use std::time::Duration;

    assert_eq!(timestamp(UNIX_EPOCH), "1970-01-01T00:00:00.000000000+00:00");
    assert_eq!(
        timestamp(UNIX_EPOCH + Duration::new(951_782_400, 123_456_789)),
        "2000-02-29T00:00:00.123456789+00:00",
    );
    assert_eq!(
        timestamp(UNIX_EPOCH + Duration::new(1_735_689_599, 999_999_999)),
        "2024-12-31T23:59:59.999999999+00:00",
    );
}

fn main() {
    let result = record(qs3::run_core_benchmark);
    println!(
        "{}",
        result.stringify().expect("benchmark result is valid JSON")
    );
}
