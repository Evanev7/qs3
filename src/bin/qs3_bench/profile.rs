//! Host-side benchmark orchestration and reduction of Nsight's SQLite export.
//! No deployment policy lives here: this runs wherever the benchmark is installed.

use std::{
    collections::HashMap,
    error::Error,
    fs::{self, File},
    path::{Path, PathBuf},
    process::{Command, Stdio},
    time::SystemTime,
};

use rusqlite::{Connection, OpenFlags, Row};
use tinyjson::JsonValue;

use super::{object, timestamp};

type Result<T> = std::result::Result<T, Box<dyn Error>>;
type Fields = HashMap<String, JsonValue>;
type Interval = (i64, i64);

fn text(command: &mut Command) -> Result<String> {
    let output = command.stdin(Stdio::null()).output()?;
    if !output.status.success() {
        return Err(format!(
            "{command:?}: {}\n{}",
            output.status,
            String::from_utf8_lossy(&output.stderr)
        )
        .into());
    }
    Ok(String::from_utf8(output.stdout)?.trim().to_owned())
}

fn logged(command: &mut Command, directory: &Path, name: &str) -> Result<()> {
    let status = command
        .stdin(Stdio::null())
        .stdout(File::create(directory.join(format!("{name}.stdout.log")))?)
        .stderr(File::create(directory.join(format!("{name}.stderr.log")))?)
        .status()?;
    if !status.success() {
        return Err(format!("{command:?}: {status}; see {}", directory.display()).into());
    }
    Ok(())
}

fn field<'a>(value: &'a JsonValue, key: &str) -> Result<&'a JsonValue> {
    value
        .get::<Fields>()
        .and_then(|fields| fields.get(key))
        .ok_or_else(|| format!("missing JSON field {key}").into())
}

fn number(value: &JsonValue, key: &str) -> Result<f64> {
    field(value, key)?
        .get::<f64>()
        .copied()
        .filter(|v| v.is_finite())
        .ok_or_else(|| format!("invalid numeric field {key}").into())
}

fn rows(value: &JsonValue) -> Result<&Vec<JsonValue>> {
    value
        .get::<Vec<JsonValue>>()
        .ok_or_else(|| "expected a JSON array".into())
}

fn insert(value: &mut JsonValue, key: &str, entry: JsonValue) -> Result<()> {
    value
        .get_mut::<Fields>()
        .ok_or("expected a JSON object")?
        .insert(key.to_owned(), entry);
    Ok(())
}

fn query<T>(
    database: &Connection,
    sql: &str,
    map: impl FnMut(&Row<'_>) -> rusqlite::Result<T>,
) -> Result<Vec<T>> {
    Ok(database
        .prepare(sql)?
        .query_map([], map)?
        .collect::<rusqlite::Result<Vec<_>>>()?)
}

fn intervals(database: &Connection, sql: &str) -> Result<Vec<Interval>> {
    let spans = query(database, sql, |row| {
        Ok((row.get::<_, i64>(0)?, row.get::<_, i64>(1)?))
    })?;
    if spans.iter().any(|(start, end)| end < start) {
        return Err("trace interval ends before it starts".into());
    }
    Ok(spans)
}

#[derive(Debug)]
struct Timing {
    calls: i64,
    total_ms: f64,
    mean_us: f64,
    min_us: f64,
    max_us: f64,
}

impl Timing {
    fn read(row: &Row<'_>) -> rusqlite::Result<Self> {
        Ok(Self {
            calls: row.get("calls")?,
            total_ms: row.get("total_ms")?,
            mean_us: row.get("mean_us")?,
            min_us: row.get("min_us")?,
            max_us: row.get("max_us")?,
        })
    }

    fn json(&self, steps: f64) -> Fields {
        HashMap::from([
            ("calls".into(), (self.calls as f64).into()),
            ("total_ms".into(), self.total_ms.into()),
            ("ms_per_decode".into(), (self.total_ms / steps).into()),
            ("mean_us".into(), self.mean_us.into()),
            ("min_us".into(), self.min_us.into()),
            ("max_us".into(), self.max_us.into()),
        ])
    }
}

struct Kernel {
    name: String,
    grid: [u32; 3],
    block: [u32; 3],
    registers: u32,
    static_shared: i64,
    dynamic_shared: i64,
    timing: Timing,
}

impl Kernel {
    fn read(row: &Row<'_>) -> rusqlite::Result<Self> {
        Ok(Self {
            name: row.get("name")?,
            grid: [row.get("grid_x")?, row.get("grid_y")?, row.get("grid_z")?],
            block: [
                row.get("block_x")?,
                row.get("block_y")?,
                row.get("block_z")?,
            ],
            registers: row.get("registers_per_thread")?,
            static_shared: row.get("static_shared_bytes")?,
            dynamic_shared: row.get("dynamic_shared_bytes")?,
            timing: Timing::read(row)?,
        })
    }

    fn json(self, steps: f64) -> JsonValue {
        let mut result = self.timing.json(steps);
        result.insert("name".into(), self.name.into());
        for (name, value) in [
            ("grid_x", self.grid[0]),
            ("grid_y", self.grid[1]),
            ("grid_z", self.grid[2]),
            ("block_x", self.block[0]),
            ("block_y", self.block[1]),
            ("block_z", self.block[2]),
            ("registers_per_thread", self.registers),
        ] {
            result.insert(name.into(), (value as f64).into());
        }
        result.insert(
            "static_shared_bytes".into(),
            (self.static_shared as f64).into(),
        );
        result.insert(
            "dynamic_shared_bytes".into(),
            (self.dynamic_shared as f64).into(),
        );
        result.into()
    }
}

struct Memory {
    operation: &'static str,
    kind: i64,
    calls: i64,
    bytes: i64,
    total_ms: f64,
}

impl Memory {
    fn json(self, steps: f64) -> JsonValue {
        object([
            ("operation", self.operation.to_owned().into()),
            ("kind_id", (self.kind as f64).into()),
            ("calls", (self.calls as f64).into()),
            ("bytes", (self.bytes as f64).into()),
            ("total_ms", self.total_ms.into()),
            ("ms_per_decode", (self.total_ms / steps).into()),
        ])
    }
}

struct Api {
    domain: &'static str,
    name: String,
    timing: Timing,
}

impl Api {
    fn json(self, steps: f64) -> JsonValue {
        let mut result = self.timing.json(steps);
        result.insert("domain".into(), self.domain.to_owned().into());
        result.insert("name".into(), self.name.into());
        result.into()
    }
}

fn has_table(database: &Connection, name: &str) -> Result<bool> {
    Ok(database.query_row(
        "SELECT EXISTS(SELECT 1 FROM sqlite_master WHERE type='table' AND name=?1)",
        [name],
        |row| row.get(0),
    )?)
}

fn merge(mut spans: Vec<Interval>) -> Result<Vec<Interval>> {
    spans.sort_unstable();
    let mut merged: Vec<Interval> = Vec::new();
    for (start, end) in spans {
        if end < start {
            return Err("trace interval ends before it starts".into());
        }
        if let Some(previous) = merged.last_mut().filter(|previous| start <= previous.1) {
            previous.1 = previous.1.max(end);
        } else {
            merged.push((start, end));
        }
    }
    Ok(merged)
}

fn duration(spans: &[Interval]) -> i64 {
    spans.iter().map(|(start, end)| end - start).sum()
}

fn overlap(a: &[Interval], b: &[Interval]) -> i64 {
    let (mut i, mut j, mut total) = (0, 0, 0);
    while i < a.len() && j < b.len() {
        total += (a[i].1.min(b[j].1) - a[i].0.max(b[j].0)).max(0);
        if a[i].1 < b[j].1 {
            i += 1;
        } else {
            j += 1;
        }
    }
    total
}

fn summarize(database: &Connection, steps: f64) -> Result<JsonValue> {
    if steps <= 0.0 || !steps.is_finite() || steps.fract() != 0.0 {
        return Err("decode sample count must be a positive integer".into());
    }
    let kernel_spans = intervals(database, "SELECT start,end FROM CUPTI_ACTIVITY_KIND_KERNEL")?;
    let kernel_union = merge(kernel_spans.clone())?;
    let span = (
        kernel_union
            .first()
            .ok_or("capture contains no CUDA kernels")?
            .0,
        kernel_union.last().unwrap().1,
    );
    if span.1 <= span.0 {
        return Err("empty kernel span".into());
    }
    let contexts = query(
        database,
        "SELECT DISTINCT deviceId,contextId FROM CUPTI_ACTIVITY_KIND_KERNEL",
        |row| Ok((row.get::<_, i64>(0)?, row.get::<_, i64>(1)?)),
    )?;
    if contexts.len() != 1 {
        return Err("expected one CUDA device/context".into());
    }
    let kernels = query(database, "
        SELECT s.value AS name, k.gridX AS grid_x, k.gridY AS grid_y, k.gridZ AS grid_z,
            k.blockX AS block_x, k.blockY AS block_y, k.blockZ AS block_z,
            k.registersPerThread AS registers_per_thread,
            k.staticSharedMemory AS static_shared_bytes, k.dynamicSharedMemory AS dynamic_shared_bytes,
            COUNT(*) AS calls, SUM(k.end-k.start)/1e6 AS total_ms,
            AVG(k.end-k.start)/1e3 AS mean_us, MIN(k.end-k.start)/1e3 AS min_us, MAX(k.end-k.start)/1e3 AS max_us
        FROM CUPTI_ACTIVITY_KIND_KERNEL k JOIN StringIds s ON s.id=k.demangledName
        GROUP BY k.demangledName, k.gridX,k.gridY,k.gridZ, k.blockX,k.blockY,k.blockZ,
            k.registersPerThread,k.staticSharedMemory,k.dynamicSharedMemory
        ORDER BY total_ms DESC, name, grid_x,grid_y,grid_z,block_x,block_y,block_z
    ", Kernel::read)?;
    let named_count: i64 = kernels.iter().map(|k| k.timing.calls).sum();
    if named_count != kernel_spans.len() as i64 {
        return Err("export is missing kernel names".into());
    }

    let mut gpu_spans = kernel_spans.clone();
    let mut memory = Vec::new();
    for (table, operation, kind) in [
        ("CUPTI_ACTIVITY_KIND_MEMCPY", "copy", "copyKind"),
        ("CUPTI_ACTIVITY_KIND_MEMSET", "set", "memKind"),
    ] {
        if !has_table(database, table)? {
            continue;
        } // Lazy export omits empty tables.
        let memory_contexts = query(
            database,
            &format!("SELECT DISTINCT deviceId,contextId FROM {table}"),
            |row| Ok((row.get::<_, i64>(0)?, row.get::<_, i64>(1)?)),
        )?;
        if memory_contexts
            .iter()
            .any(|context| *context != contexts[0])
        {
            return Err("memory trace belongs to another CUDA device/context".into());
        }
        gpu_spans.extend(intervals(
            database,
            &format!("SELECT start,end FROM {table}"),
        )?);
        memory.extend(query(database, &format!("
            SELECT {kind} AS kind_id, COUNT(*) AS calls, SUM(bytes) AS bytes, SUM(end-start)/1e6 AS total_ms
            FROM {table} GROUP BY {kind} ORDER BY {kind}"), |row| Ok(Memory {
                operation, kind: row.get("kind_id")?, calls: row.get("calls")?,
                bytes: row.get("bytes")?, total_ms: row.get("total_ms")?,
            }))?);
    }
    let gpu_union = merge(gpu_spans)?;
    let active = overlap(&gpu_union, &[span]);
    let pinned = merge(intervals(
        database,
        "
        SELECT r.start,r.end FROM CUPTI_ACTIVITY_KIND_RUNTIME r JOIN StringIds s ON s.id=r.nameId
        WHERE s.value LIKE 'cudaHostAlloc%' OR s.value LIKE 'cudaFreeHost%'
    ",
    )?)?;
    let mut api = Vec::new();
    for (table, domain) in [
        ("CUPTI_ACTIVITY_KIND_RUNTIME", "runtime"),
        ("CUPTI_ACTIVITY_KIND_DRIVER", "driver"),
    ] {
        if !has_table(database, table)? {
            continue;
        }
        api.extend(query(database, &format!("
            SELECT s.value AS name, COUNT(*) AS calls, SUM(r.end-r.start)/1e6 AS total_ms,
                AVG(r.end-r.start)/1e3 AS mean_us, MIN(r.end-r.start)/1e3 AS min_us, MAX(r.end-r.start)/1e3 AS max_us
            FROM {table} r JOIN StringIds s ON s.id=r.nameId
            GROUP BY r.nameId ORDER BY total_ms DESC, name"), |row| Ok(Api {
                domain, name: row.get("name")?, timing: Timing::read(row)?,
            }))?);
    }
    let diagnostics: Vec<(String, String)> = if has_table(database, "DIAGNOSTIC_EVENT")? {
        query(
            database,
            "SELECT e.name,d.text FROM DIAGNOSTIC_EVENT d
            JOIN ENUM_DIAGNOSTIC_SEVERITY_LEVEL e ON e.id=d.severity
            WHERE e.name IN ('Warning','Error') ORDER BY d.timestamp",
            |row| Ok((row.get(0)?, row.get(1)?)),
        )?
    } else {
        Vec::new()
    };
    Ok(object([
        ("decode_steps", steps.into()),
        (
            "timeline",
            object([
                ("kernel_count", (kernel_spans.len() as f64).into()),
                (
                    "kernel_duration_sum_ms",
                    (duration(&kernel_spans) as f64 / 1e6).into(),
                ),
                (
                    "kernel_ms_per_decode",
                    (duration(&kernel_spans) as f64 / 1e6 / steps).into(),
                ),
                ("kernel_span_ms", ((span.1 - span.0) as f64 / 1e6).into()),
                (
                    "gpu_work_union_within_kernel_span_ms",
                    (active as f64 / 1e6).into(),
                ),
                (
                    "no_gpu_work_within_kernel_span_ms",
                    ((span.1 - span.0 - active) as f64 / 1e6).into(),
                ),
                (
                    "no_gpu_work_ms_per_decode",
                    ((span.1 - span.0 - active) as f64 / 1e6 / steps).into(),
                ),
                (
                    "pinned_host_api_union_ms",
                    (duration(&pinned) as f64 / 1e6).into(),
                ),
                (
                    "pinned_host_api_overlap_gpu_work_ms",
                    (overlap(&pinned, &gpu_union) as f64 / 1e6).into(),
                ),
            ]),
        ),
        (
            "kernels",
            kernels
                .into_iter()
                .map(|k| k.json(steps))
                .collect::<Vec<_>>()
                .into(),
        ),
        (
            "cuda_api",
            api.into_iter()
                .map(|a| a.json(steps))
                .collect::<Vec<_>>()
                .into(),
        ),
        (
            "memory",
            memory
                .into_iter()
                .map(|m| m.json(steps))
                .collect::<Vec<_>>()
                .into(),
        ),
        (
            "diagnostics",
            diagnostics
                .into_iter()
                .map(|(severity, text)| {
                    object([("severity", severity.into()), ("text", text.into())])
                })
                .collect::<Vec<_>>()
                .into(),
        ),
    ]))
}

fn read_pass(json: &str, profiled: bool) -> Result<JsonValue> {
    let value: JsonValue = json.trim().parse()?;
    let m = field(&value, "measurement")?;
    let e = field(m, "execution")?;
    if field(m, "profile")?.get::<String>().map(String::as_str) != Some("release")
        || field(e, "cuda_profiler_range")?.get::<bool>() != Some(&profiled)
        || field(e, "cuda_profiler_phase")?
            .get::<String>()
            .map(String::as_str)
            != Some(if profiled { "decode" } else { "none" })
    {
        return Err("benchmark pass used the wrong build or profiling mode".into());
    }
    let decode = field(m, "decode")?;
    let n = number(decode, "samples")?;
    if n <= 0.0
        || n != rows(field(decode, "sample_ms")?)?.len() as f64
        || n + number(decode, "warmups")?
            != rows(field(decode, "generated_token_ids")?)?.len() as f64
    {
        return Err("incomplete benchmark pass".into());
    }
    Ok(value)
}

fn cpu_summary(database: &Connection) -> Result<JsonValue> {
    for table in ["COMPOSITE_EVENTS", "SAMPLING_CALLCHAINS", "SCHED_EVENTS"] {
        if !has_table(database, table)? {
            return Err(format!(
                "CPU capture is missing {table}; check Nsight diagnostics and host perf permissions"
            )
            .into());
        }
    }
    // Match the GPU gap metric's endpoints. Sample counts are statistical
    // observations, not durations; inclusive stack counts overlap by design.
    let bounds =
        "bounds AS (SELECT MIN(start) AS first, MAX(end) AS last FROM CUPTI_ACTIVITY_KIND_KERNEL)";
    let sample_count: i64 = database.query_row(
        &format!(
            "WITH {bounds}
        SELECT COUNT(*) FROM COMPOSITE_EVENTS,bounds WHERE start >= first AND start < last"
        ),
        [],
        |row| row.get(0),
    )?;
    if sample_count == 0 {
        return Err("CPU sampling produced no samples during steady decode".into());
    }
    let mut gpu = "SELECT start,end FROM CUPTI_ACTIVITY_KIND_KERNEL".to_owned();
    for table in ["CUPTI_ACTIVITY_KIND_MEMCPY", "CUPTI_ACTIVITY_KIND_MEMSET"] {
        if has_table(database, table)? {
            gpu.push_str(&format!(" UNION ALL SELECT start,end FROM {table}"));
        }
    }
    let selected = format!("WITH {bounds}, gpu AS ({gpu}), samples AS MATERIALIZED (
        SELECT e.*, NOT EXISTS(SELECT 1 FROM gpu WHERE gpu.start <= e.start AND gpu.end > e.start) AS in_gpu_gap
        FROM COMPOSITE_EVENTS e,bounds WHERE e.start >= first AND e.start < last)");
    let functions = query(database, &format!("{selected}
        SELECT s.value AS function, m.value AS module, c.unresolved,
            COUNT(DISTINCT e.id) AS inclusive_samples,
            SUM(c.stackDepth=0) AS leaf_samples,
            COUNT(DISTINCT CASE WHEN e.in_gpu_gap THEN e.id END) AS inclusive_samples_without_gpu_work,
            SUM(CASE WHEN c.stackDepth=0 AND e.in_gpu_gap THEN 1 ELSE 0 END) AS leaf_samples_without_gpu_work
        FROM samples e JOIN SAMPLING_CALLCHAINS c ON c.id=e.id
        LEFT JOIN StringIds s ON s.id=c.symbol LEFT JOIN StringIds m ON m.id=c.module
        WHERE COALESCE(c.specialEntry,0)=0
        GROUP BY c.symbol,c.module,c.unresolved
        ORDER BY leaf_samples_without_gpu_work DESC,leaf_samples DESC,inclusive_samples DESC,function,module"), |row| Ok(CpuFunction {
            function: row.get("function")?, module: row.get("module")?, unresolved: row.get("unresolved")?,
            inclusive: row.get("inclusive_samples")?, leaf: row.get("leaf_samples")?,
            inclusive_gap: row.get("inclusive_samples_without_gpu_work")?, leaf_gap: row.get("leaf_samples_without_gpu_work")?,
        }))?;
    if functions.is_empty() {
        return Err("CPU samples contain no function stacks".into());
    }
    let threads: Vec<(String, Option<String>, i64, i64)> = query(
        database,
        &format!(
            "{selected}
        SELECT CAST(e.globalTid AS TEXT) AS thread_id, t.label AS state,
            COUNT(*) AS samples, SUM(e.in_gpu_gap) AS samples_without_gpu_work
        FROM samples e LEFT JOIN ENUM_SAMPLING_THREAD_STATE t ON t.id=e.threadState
        GROUP BY e.globalTid,e.threadState ORDER BY samples DESC,thread_id,state"
        ),
        |row| Ok((row.get(0)?, row.get(1)?, row.get(2)?, row.get(3)?)),
    )?;
    let scheduling: Vec<(String, i64, i64, i64)> = query(
        database,
        &format!(
            "WITH {bounds}
        SELECT CAST(e.globalTid AS TEXT) AS thread_id, COUNT(*) AS events,
            SUM(e.isSchedIn!=0) AS scheduled_in, SUM(e.isSchedIn=0) AS scheduled_out
        FROM SCHED_EVENTS e,bounds WHERE e.start >= first AND e.start < last
        GROUP BY e.globalTid ORDER BY events DESC,thread_id"
        ),
        |row| Ok((row.get(0)?, row.get(1)?, row.get(2)?, row.get(3)?)),
    )?;
    Ok(object([
        ("scope", "process-tree".to_owned().into()),
        ("backtrace", "dwarf".to_owned().into()),
        ("samples", (sample_count as f64).into()),
        (
            "threads",
            threads
                .into_iter()
                .map(|(id, state, samples, gap)| {
                    object([
                        ("thread_id", id.into()),
                        ("state", super::optional(state)),
                        ("samples", (samples as f64).into()),
                        ("samples_without_gpu_work", (gap as f64).into()),
                    ])
                })
                .collect::<Vec<_>>()
                .into(),
        ),
        (
            "functions",
            functions
                .into_iter()
                .map(CpuFunction::json)
                .collect::<Vec<_>>()
                .into(),
        ),
        (
            "scheduling",
            scheduling
                .into_iter()
                .map(|(id, events, ins, outs)| {
                    object([
                        ("thread_id", id.into()),
                        ("events", (events as f64).into()),
                        ("scheduled_in", (ins as f64).into()),
                        ("scheduled_out", (outs as f64).into()),
                    ])
                })
                .collect::<Vec<_>>()
                .into(),
        ),
    ]))
}

struct CpuFunction {
    function: Option<String>,
    module: Option<String>,
    unresolved: Option<bool>,
    inclusive: i64,
    leaf: i64,
    inclusive_gap: i64,
    leaf_gap: i64,
}

impl CpuFunction {
    fn json(self) -> JsonValue {
        object([
            ("function", super::optional(self.function)),
            ("module", super::optional(self.module)),
            (
                "unresolved",
                self.unresolved.map_or(JsonValue::Null, JsonValue::from),
            ),
            ("inclusive_samples", (self.inclusive as f64).into()),
            ("leaf_samples", (self.leaf as f64).into()),
            (
                "inclusive_samples_without_gpu_work",
                (self.inclusive_gap as f64).into(),
            ),
            (
                "leaf_samples_without_gpu_work",
                (self.leaf_gap as f64).into(),
            ),
        ])
    }
}

fn validate_pair(a: &JsonValue, b: &JsonValue) -> Result<()> {
    if field(a, "metadata")? != field(b, "metadata")? {
        return Err("paired metadata differs".into());
    }
    let (a, b) = (field(a, "measurement")?, field(b, "measurement")?);
    for key in ["model", "prompt"] {
        if field(a, key)? != field(b, key)? {
            return Err(format!("paired {key} differs").into());
        }
    }
    let execution = |m| -> Result<Fields> {
        let mut e = field(m, "execution")?
            .get::<Fields>()
            .ok_or("expected execution object")?
            .clone();
        e.remove("cuda_profiler_range");
        e.remove("cuda_profiler_phase");
        Ok(e)
    };
    if execution(a)? != execution(b)? {
        return Err("paired execution settings differ".into());
    }
    for key in [
        "samples",
        "warmups",
        "context_start",
        "context_end",
        "generated_token_ids",
    ] {
        if field(field(a, "decode")?, key)? != field(field(b, "decode")?, key)? {
            return Err(format!("paired decode {key} differs").into());
        }
    }
    Ok(())
}

pub(super) fn run() -> Result<JsonValue> {
    if env!("QS3_BUILD_PROFILE") != "release" {
        return Err("run the core benchmark in release mode".into());
    }
    // Check CPU sampling permissions before loading the model. Never silently
    // fall back to a GPU-only capture.
    let version = text(Command::new("nsys").arg("--version"))?;
    let environment = text(Command::new("nsys").args(["status", "--environment"]))?;
    if !environment.contains("CPU Profiling Environment (process-tree): OK") {
        return Err(format!("CPU sampling is unavailable; configure host perf permissions before benchmarking.\n{environment}").into());
    }
    let root = std::env::var_os("QS3_BENCH_ARTIFACT_DIR")
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from(".prototypes/profiles"));
    fs::create_dir_all(&root)?;
    let directory = root.join(timestamp(SystemTime::now()).replace(':', ""));
    fs::create_dir(&directory)?;
    let directory = directory.canonicalize()?;
    eprintln!("Benchmark artifacts: {}", directory.display());
    let executable = std::env::current_exe()?;
    let mut benchmark = Command::new(&executable);
    benchmark.arg("--measure-pass").env_remove("QS3_PROFILE");
    eprintln!("Running unprofiled core benchmark");
    logged(&mut benchmark, &directory, "benchmark")?;
    let mut result = read_pass(
        &fs::read_to_string(directory.join("benchmark.stdout.log"))?,
        false,
    )?;

    let mut capture = Command::new("nsys");
    capture
        .args([
            "profile",
            "--trace=cuda",
            "--sample=process-tree",
            "--cpuctxsw=process-tree",
            "--backtrace=dwarf",
            "--samples-per-backtrace=1",
            "--capture-range=cudaProfilerApi",
            "--capture-range-end=stop",
            "--cuda-memory-usage=true",
        ])
        .arg(format!("--output={}", directory.join("decode").display()))
        .arg(&executable)
        .arg("--measure-pass")
        .env("QS3_PROFILE", "decode");
    eprintln!("Capturing warmed decode with Nsight Systems");
    logged(&mut capture, &directory, "nsight")?;
    // Nsight writes its own progress to stdout; process output is parsed from
    // the single JSON line, while the full log remains in the artifact directory.
    let log = fs::read_to_string(directory.join("nsight.stdout.log"))?;
    let lines: Vec<_> = log.lines().filter(|line| line.starts_with('{')).collect();
    if lines.len() != 1 {
        return Err("Nsight pass did not emit exactly one benchmark result".into());
    }
    let profiled = read_pass(lines[0], true)?;
    validate_pair(&result, &profiled)?;
    let database = directory.join("decode.sqlite");
    eprintln!("Exporting and reducing the CUDA trace");
    logged(
        Command::new("nsys")
            .args(["export", "--type=sqlite"])
            .arg(format!("--output={}", database.display()))
            .arg(directory.join("decode.nsys-rep")),
        &directory,
        "export",
    )?;
    let steps = number(
        field(field(&profiled, "measurement")?, "decode")?,
        "samples",
    )?;
    let database = Connection::open_with_flags(database, OpenFlags::SQLITE_OPEN_READ_ONLY)?;
    let mut summary = summarize(&database, steps)?;
    for diagnostic in rows(field(&summary, "diagnostics")?)? {
        eprintln!("Nsight diagnostic: {}", diagnostic.stringify()?);
    }
    insert(&mut summary, "cpu", cpu_summary(&database)?)?;
    insert(&mut summary, "version", version.into())?;
    insert(&mut summary, "cpu_environment", environment.into())?;
    insert(
        &mut summary,
        "sqlite_version",
        rusqlite::version().to_owned().into(),
    )?;
    insert(
        &mut summary,
        "artifact_directory",
        directory.to_string_lossy().into_owned().into(),
    )?;
    insert(
        &mut summary,
        "command",
        std::iter::once(capture.get_program())
            .chain(capture.get_args())
            .map(|s| JsonValue::from(s.to_string_lossy().into_owned()))
            .collect::<Vec<_>>()
            .into(),
    )?;
    insert(
        &mut summary,
        "profiled_measurement",
        field(&profiled, "measurement")?.clone(),
    )?;
    insert(&mut result, "nsight", summary)?;
    Ok(result)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn interval_union_and_clipping_do_not_double_count_overlapping_work() {
        let work = merge(vec![(30, 40), (5, 15), (0, 10), (10, 12)]).unwrap();
        assert_eq!(work, vec![(0, 15), (30, 40)]);
        assert_eq!(duration(&work), 25);
        assert_eq!(overlap(&work, &[(8, 35)]), 12);
        let pins = merge(vec![(-2, 3), (38, 45)]).unwrap();
        assert_eq!(duration(&pins), 12);
        assert_eq!(overlap(&pins, &work), 5);
        assert!(merge(vec![(9, 2)]).is_err());
    }

    #[test]
    fn sqlite_reduction_handles_overlap_lazy_tables_and_launch_shapes() -> Result<()> {
        let database = Connection::open_in_memory()?;
        database.execute_batch("
            CREATE TABLE StringIds(id INTEGER PRIMARY KEY,value TEXT);
            INSERT INTO StringIds VALUES(1,'GemmGrouped<bf16>'),(2,'cudaHostAlloc_v3020');
            CREATE TABLE CUPTI_ACTIVITY_KIND_KERNEL(start INTEGER,end INTEGER,deviceId INTEGER,contextId INTEGER,
                demangledName INTEGER,gridX INTEGER,gridY INTEGER,gridZ INTEGER,blockX INTEGER,blockY INTEGER,
                blockZ INTEGER,registersPerThread INTEGER,staticSharedMemory INTEGER,dynamicSharedMemory INTEGER);
            INSERT INTO CUPTI_ACTIVITY_KIND_KERNEL VALUES
                (0,10000000,0,1,1,4,1,1,128,1,1,32,0,0),
                (5000000,15000000,0,1,1,96,1,1,128,1,1,32,0,0),
                (30000000,40000000,0,1,1,96,1,1,128,1,1,32,0,0);
            CREATE TABLE CUPTI_ACTIVITY_KIND_RUNTIME(start INTEGER,end INTEGER,nameId INTEGER);
            INSERT INTO CUPTI_ACTIVITY_KIND_RUNTIME VALUES(-2000000,3000000,2);
        ")?;
        let reduced = summarize(&database, 2.0)?;
        assert_eq!(
            number(&reduced["timeline"], "kernel_duration_sum_ms")?,
            30.0
        );
        assert_eq!(
            number(&reduced["timeline"], "no_gpu_work_within_kernel_span_ms")?,
            15.0
        );
        assert_eq!(
            number(&reduced["timeline"], "pinned_host_api_overlap_gpu_work_ms")?,
            3.0
        );
        assert_eq!(rows(&reduced["kernels"])?.len(), 2);
        assert_eq!(number(&reduced["kernels"][0], "grid_x")?, 96.0);
        assert_eq!(number(&reduced["kernels"][0], "ms_per_decode")?, 10.0);
        assert!(rows(&reduced["memory"])?.is_empty());
        database.execute_batch("
            CREATE TABLE CUPTI_ACTIVITY_KIND_MEMCPY(start INTEGER,end INTEGER,deviceId INTEGER,contextId INTEGER,copyKind INTEGER,bytes INTEGER);
            INSERT INTO CUPTI_ACTIVITY_KIND_MEMCPY VALUES(12000000,32000000,0,1,1,4096);
        ")?;
        let reduced = summarize(&database, 2.0)?;
        assert_eq!(
            number(&reduced["timeline"], "no_gpu_work_within_kernel_span_ms")?,
            0.0
        );
        assert_eq!(number(&reduced["memory"][0], "bytes")?, 4096.0);
        assert!(summarize(&database, 0.0).is_err());
        database.execute_batch("DELETE FROM CUPTI_ACTIVITY_KIND_KERNEL")?;
        assert!(summarize(&database, 2.0).is_err());
        Ok(())
    }

    #[test]
    fn cpu_samples_are_clipped_and_distinguish_leaf_from_inclusive_stacks() -> Result<()> {
        let db = Connection::open_in_memory()?;
        assert!(cpu_summary(&db).is_err());
        db.execute_batch("
            CREATE TABLE CUPTI_ACTIVITY_KIND_KERNEL(start INTEGER,end INTEGER);
            INSERT INTO CUPTI_ACTIVITY_KIND_KERNEL VALUES(10,20),(40,50);
            CREATE TABLE COMPOSITE_EVENTS(id INTEGER,start INTEGER,globalTid INTEGER,threadState INTEGER);
            INSERT INTO COMPOSITE_EVENTS VALUES(1,5,99,1),(2,15,99,1),(3,30,99,1),(4,50,99,1);
            CREATE TABLE SAMPLING_CALLCHAINS(id INTEGER,symbol INTEGER,module INTEGER,unresolved INTEGER,specialEntry INTEGER,stackDepth INTEGER);
            INSERT INTO SAMPLING_CALLCHAINS VALUES(2,1,3,0,0,0),(2,2,3,0,0,1),
                (3,1,3,0,0,0),(3,2,3,0,0,1),(3,2,3,0,0,2);
            CREATE TABLE StringIds(id INTEGER,value TEXT);
            INSERT INTO StringIds VALUES(1,'leaf'),(2,'caller'),(3,'qs3-bench');
            CREATE TABLE ENUM_SAMPLING_THREAD_STATE(id INTEGER,label TEXT);
            INSERT INTO ENUM_SAMPLING_THREAD_STATE VALUES(1,'Running');
            CREATE TABLE SCHED_EVENTS(start INTEGER,globalTid INTEGER,isSchedIn INTEGER);
            INSERT INTO SCHED_EVENTS VALUES(5,99,1),(25,99,0),(35,99,1);
        ")?;
        let result = cpu_summary(&db)?;
        assert_eq!(number(&result, "samples")?, 2.0);
        assert_eq!(
            number(&result["threads"][0], "samples_without_gpu_work")?,
            1.0
        );
        assert_eq!(
            number(&result["functions"][0], "leaf_samples_without_gpu_work")?,
            1.0
        );
        assert_eq!(number(&result["functions"][1], "inclusive_samples")?, 2.0);
        assert_eq!(number(&result["functions"][1], "leaf_samples")?, 0.0);
        assert_eq!(number(&result["scheduling"][0], "events")?, 2.0);
        db.execute_batch("DELETE FROM COMPOSITE_EVENTS")?;
        assert!(cpu_summary(&db).is_err());
        Ok(())
    }

    #[test]
    fn paired_runs_must_match_workload_precision_and_generated_tokens() -> Result<()> {
        let plain: JsonValue = r#"{
            "metadata":{"gpu":"GB10"},
            "measurement":{"profile":"release","model":"pinned-model","prompt":{"tokens":102},
                "execution":{"precision":"bf16","cuda_profiler_range":false,"cuda_profiler_phase":"none"},
                "decode":{"samples":2,"warmups":1,"context_start":103,"context_end":105,
                    "sample_ms":[1,1],"generated_token_ids":[1,2,3]}}
        }"#.parse()?;
        let mut profiled = plain.clone();
        profiled["measurement"]["execution"]["cuda_profiler_range"] = true.into();
        profiled["measurement"]["execution"]["cuda_profiler_phase"] = "decode".to_owned().into();
        read_pass(&plain.stringify()?, false)?;
        read_pass(&profiled.stringify()?, true)?;
        validate_pair(&plain, &profiled)?;
        assert!(read_pass(&plain.stringify()?, true).is_err());
        profiled["measurement"]["decode"]["generated_token_ids"][2] = 99.0.into();
        assert!(validate_pair(&plain, &profiled).is_err());
        profiled = plain.clone();
        profiled["measurement"]["execution"]["precision"] = "fp8".to_owned().into();
        assert!(validate_pair(&plain, &profiled).is_err());
        profiled["measurement"]["decode"]["samples"] = 32.0.into();
        assert!(read_pass(&profiled.stringify()?, false).is_err());
        Ok(())
    }
}
