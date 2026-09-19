"""Render history and archived GPU time breakdowns: python3 benchmarks/render_history.py.

Import existing local or remote Nsight exports (no GPU execution):
  python3 benchmarks/render_history.py --import-gpu benchmarks/RUN.json --remote sp10@sp10

Keep RUN.gpu.json.gz beside RUN.json in git. HTML viewers are generated offline.
Future benchmark runs capture a warmed prefill and warmed decode separately;
older captures expose only available phases. The local remote.sh archives both.
"""

import argparse
from bisect import bisect_right
from itertools import groupby
import re
import gzip
import json
from pathlib import Path
import shlex
import sqlite3
import subprocess
import sys

ROOT = Path(__file__).resolve().parent


# Exact Nsight events are stored separately from the aggregate benchmark JSON.

def extract_phase(path: Path) -> dict:
    with sqlite3.connect(path.resolve().as_uri() + "?mode=ro", uri=True) as db:
        tables = {r[0] for r in db.execute("SELECT name FROM sqlite_master WHERE type='table'")}
        names = dict(db.execute("SELECT id,value FROM StringIds"))
        definitions, lookup, events = [], {}, []

        def add(row, definition):
            start, end, device, context, stream = row[:5]
            if end < start:
                raise ValueError("GPU event ends before it starts")
            key = json.dumps(definition, sort_keys=True)
            if key not in lookup:
                lookup[key] = len(definitions)
                definitions.append(definition)
            events.append([start, end - start, lookup[key], device, context, stream])

        for row in db.execute("""
            SELECT start,end,deviceId,contextId,streamId,demangledName,
                   gridX,gridY,gridZ,blockX,blockY,blockZ,
                   registersPerThread,staticSharedMemory,dynamicSharedMemory
            FROM CUPTI_ACTIVITY_KIND_KERNEL ORDER BY start,end
        """):
            add(row, dict(kind="kernel", name=names[row[5]], grid=list(row[6:9]),
                          block=list(row[9:12]), registers=row[12], shared_bytes=row[13]+row[14]))
        if not events:
            raise ValueError(f"{path}: no GPU kernels captured")
        if "CUPTI_ACTIVITY_KIND_MEMCPY" in tables:
            kinds = dict(db.execute("SELECT id,label FROM ENUM_CUDA_MEMCPY_OPER"))
            for row in db.execute("""
                SELECT start,end,deviceId,contextId,streamId,copyKind,bytes
                FROM CUPTI_ACTIVITY_KIND_MEMCPY ORDER BY start,end
            """):
                add(row, dict(kind="copy", name=kinds[row[5]], bytes=row[6]))
        if "CUPTI_ACTIVITY_KIND_MEMSET" in tables:
            for row in db.execute("""
                SELECT start,end,deviceId,contextId,streamId,bytes
                FROM CUPTI_ACTIVITY_KIND_MEMSET ORDER BY start,end
            """):
                add(row, dict(kind="memset", name="Memset", bytes=row[5]))
        events.sort()
        origin = events[0][0]
        finish = max(e[0] + e[1] for e in events)
        api_tables = [t for t in ("CUPTI_ACTIVITY_KIND_RUNTIME", "CUPTI_ACTIVITY_KIND_DRIVER") if t in tables]
        def api_intervals():
            kinds = {key: api_kind(value) for key, value in names.items()}
            for table in api_tables:
                for start, end, name in db.execute(
                    f"SELECT start,end,nameId FROM {table} WHERE end>? AND start<?",
                    (origin, finish),
                ):
                    yield start, end, kinds[name]
        gaps, gap_count = classify_gaps(events, api_intervals())
    for event in events:
        event[0] -= origin
    for gap in gaps:
        gap[0] -= origin
    return dict(source=str(path), origin_ns=origin, definitions=definitions, events=events,
                gaps=gaps, gap_count=gap_count, gap_api_available=bool(api_tables),
                event_columns=["start_ns", "duration_ns", "definition", "device", "context", "stream"])


def extract_directory(directory: Path) -> dict:
    phases = {}
    for phase in ("prefill", "decode"):
        path = directory / f"{phase}.sqlite"
        if path.exists():
            phases[phase] = extract_phase(path)
    if not phases:
        raise ValueError(f"{directory}: no prefill/decode SQLite captures")
    return phases


def archive(benchmark: Path, remote: str | None = None) -> Path:
    run = json.loads(benchmark.read_text())
    directory = Path(run["nsight"]["artifact_directory"])
    if remote:
        # Pass the checked-in extractor over stdin; do not install or write on the host.
        command = shlex.join(["python3", "-", "--extract-directory", str(directory)])
        result = subprocess.run(["ssh", "-F", "/dev/null", remote, command],
                                input=Path(__file__).read_bytes(), capture_output=True, check=True)
        phases = json.loads(gzip.decompress(result.stdout))
    else:
        phases = extract_directory(directory)
    for phase in ("decode", "prefill"):
        summary = run["nsight"].get("prefill") if phase == "prefill" else run["nsight"]["timeline"]
        if summary is None:
            continue
        if phase not in phases:
            raise ValueError(f"{benchmark}: recorded {phase} capture is missing")
        trace = phases[phase]
        count = sum(trace["definitions"][e[2]]["kind"] == "kernel" for e in trace["events"])
        if count != summary["kernel_count"]:
            raise ValueError(f"{benchmark}: {phase} kernel count differs from the recorded capture")
    result = dict(benchmark=benchmark.name, commit=run["metadata"]["commit_hash"],
                  model=run["measurement"]["model_name"],
                  prompt_tokens=run["measurement"]["prompt"]["tokens"],
                  decode_samples=run["measurement"]["decode"]["samples"], phases=phases)
    output = benchmark.with_suffix(".gpu.json.gz")
    output.write_bytes(gzip.compress(json.dumps(result, separators=(",", ":")).encode(), mtime=0))
    return output


GAP_KINDS = (
    "Synchronization API active",
    "Memory-management API active",
    "Transfer API active",
    "Launch / launch-setup API active",
    "Other CUDA API active",
    "No CUDA API call recorded",
)


def api_kind(name: str) -> int:
    name = name.lower()
    if "synchronize" in name or "waitevent" in name:
        return 0
    if any(x in name for x in ("malloc", "free", "hostalloc", "memalloc", "memmap", "memunmap")):
        return 1
    if "memcpy" in name or "memset" in name:
        return 2
    if any(x in name for x in ("launch", "tensormap", "kernelget", "driverentrypoint")):
        return 3
    return 4


def classify_gaps(events: list, api_intervals) -> tuple[list, int]:
    """Partition the complement of the GPU-work union by observed CUDA API overlap.

    Nested runtime/driver calls use GAP_KINDS priority. No double counting;
    overlapping API calls are evidence of host activity, not proof of a stall cause.
    Events use absolute start + duration; APIs use absolute start + end + kind.
    """
    end = events[0][0]
    gaps = []
    for event in sorted(events):
        start, duration = event[:2]
        if duration < 0:
            raise ValueError("GPU event ends before it starts")
        if start > end:
            gaps.append((end, start))
        end = max(end, start + duration)
    ends = [b for _, b in gaps]
    boundaries = [[(a, 0, 0), (b, 0, 0)] for a, b in gaps]
    for start, end, kind in api_intervals:
        if end < start:
            raise ValueError("CUDA API interval ends before it starts")
        index = bisect_right(ends, start)
        while index < len(gaps) and gaps[index][0] < end:
            a, b = gaps[index]
            a, b = max(a, start), min(b, end)
            if a < b:
                boundaries[index].extend(((a, 1, kind), (b, -1, kind)))
            index += 1
    result = []
    for boundary in boundaries:
        counts = [0] * 5
        previous = min(x[0] for x in boundary)
        for time, changes in groupby(sorted(boundary), key=lambda x: x[0]):
            kind = next((i for i, count in enumerate(counts) if count), 5)
            if time > previous:
                if result and result[-1][0] + result[-1][1] == previous and result[-1][2] == kind:
                    result[-1][1] += time - previous
                else:
                    result.append([previous, time - previous, kind])
            for _, delta, category in changes:
                counts[category] += delta
            previous = time
    return result, len(gaps)


def kernel_kind(definition: dict) -> tuple[str, str]:
    """Name-based operation families only: no inferred model layer or projection role."""
    name = definition["name"].lower()
    if definition["kind"] != "kernel":
        return "Memory operations", definition["name"]
    if "quantize" in name or "quant_to" in name:
        if "fp8_quantize" in name:
            kind = "FP8 quantization"
        elif "blockscalequantizationtype)0" in name:
            kind = "NVFP4 quantization"
        else:
            kind = "Block quantization"
        return "Quantization", kind
    if "gdn" in name or "conv1d" in name or "chunk_" in name:
        if "conv1d" in name:
            kind = "Causal convolution"
        elif "post_conv" in name or "prepare" in name:
            kind = "Q/K/V preparation"
        elif "rmsnorm" in name:
            kind = "Gated normalization"
        elif "warp_kernel" in name or "recurrent" in name:
            kind = "Recurrent state update"
        else:
            kind = "Chunked prefill"
        return "GDN", kind
    if "splitkreduce" in name:
        return "Matrix multiplication", "Split-K reduction"
    if "nvjet" in name:
        return "Matrix multiplication", "FP8 GEMM"
    if "gemmuniversal" in name or "gemm" in name:
        return "Matrix multiplication", "NVFP4 GEMM" if "float_e2m1" in name else "GEMM"
    if "gemvx" in name or "gemv" in name:
        return "Matrix multiplication", "GEMV"
    if "norm" in name:
        return "Normalization", "Residual + RMSNorm" if "fusedadd" in name else "RMSNorm"
    if "rotary" in name or "rope" in name:
        return "Full attention", "Rotary position encoding"
    if "appendpaged" in name:
        return "Full attention", "KV cache append"
    if "mergestates" in name:
        return "Full attention", "Attention-state merge"
    if "attention_output_gate" in name:
        return "Full attention", "Output gating"
    if "pagedkv" in name or "attention" in name:
        return "Full attention", "Attention kernel"
    if "silu" in name:
        return "MLP activation", "SiLU × up"
    if "embedding" in name:
        return "Embedding / sampling", "Embedding lookup"
    if any(x in name for x in ("argmax", "sample", "sampling", "logits")):
        return "Embedding / sampling", "Logits / token selection"
    return "Other kernels", definition["name"].split("(")[0].removeprefix("void ")[:100]


def variant_label(definition: dict, kind: str) -> str:
    if definition["kind"] != "kernel":
        return f'{definition["name"]} · {definition["bytes"]:,} bytes'
    name = definition["name"]
    tile = re.search(r"_mma_(\d+x\d+x\d+)_", name)
    prefix = f'tile {tile[1].replace("x", " × ")} · ' if tile else ""
    grid = " × ".join(map(str, definition["grid"]))
    return f"{kind} · {prefix}grid {grid}"


def summarize_phase(trace: dict) -> dict:
    definitions = trace["definitions"]
    stats = [[0, 0] for _ in definitions]
    end = trace["events"][0][0]
    busy = 0
    for event in trace["events"]:
        start, duration, index = event[:3]
        stats[index][0] += duration
        stats[index][1] += 1
        busy += max(0, start + duration - max(start, end))
        end = max(end, start + duration)
    span = end - trace["events"][0][0]
    if span <= 0:
        raise ValueError("capture has no positive GPU duration")
    groups = {}
    for definition, (duration, calls) in zip(definitions, stats):
        group, kind = kernel_kind(definition)
        groups.setdefault(group, {}).setdefault(kind, []).append(dict(
            label=variant_label(definition, kind), ns=duration, count=calls, definition=definition))
    gaps = {}
    for _, duration, kind in trace["gaps"]:
        label = GAP_KINDS[kind] if trace["gap_api_available"] else "CUDA API trace unavailable"
        row = gaps.setdefault(label, dict(ns=0, count=0))
        row["ns"] += duration
        row["count"] += 1
    gap_ns = sum(row["ns"] for row in gaps.values())
    if busy + gap_ns != span:
        raise ValueError("GPU work and classified gaps do not cover the capture span")
    groups["No GPU work"] = {name: [dict(label=name, **row)] for name, row in gaps.items()}
    work_ns = sum(row[0] for row in stats)
    return dict(source=trace["source"], span_ns=span, busy_ns=busy, work_ns=work_ns,
                gap_ns=gap_ns, overlap_ns=work_ns-busy, gap_count=trace["gap_count"], groups=groups)


def report_data(sidecar: Path) -> dict:
    raw = json.loads(gzip.decompress(sidecar.read_bytes()))
    raw["phases"] = {phase: summarize_phase(trace) for phase, trace in raw["phases"].items()}
    measurement = json.loads(sidecar.with_name(raw["benchmark"]).read_text())["measurement"]
    raw["unprofiled_ms"] = {phase: measurement[phase]["p50_ms"] for phase in ("prefill", "decode")}
    return raw


def render_gpu_report(sidecar: Path, reports: list[dict]) -> Path:
    selected = next(report for report in reports if report["benchmark"] == sidecar.name.removesuffix(".gpu.json.gz")+".json")
    # Embed only summaries for comparable workloads. No network or raw-event parsing in the viewer.
    matches = [r for r in reports if all(r[key] == selected[key] for key in ("model", "prompt_tokens", "decode_samples"))]
    data = json.dumps(dict(selected=selected["benchmark"], reports=matches), separators=(",", ":"))
    data = data.replace("&", "\\u0026").replace("<", "\\u003c").replace(">", "\\u003e")
    output = sidecar.with_name(sidecar.name.removesuffix(".json.gz") + ".html")
    output.write_text(GPU_HTML.replace("__GPU_DATA__", data))
    return output



def collect_runs() -> list[dict[str, object]]:
    runs: list[dict[str, object]] = []
    models = {p.parent.name for p in (ROOT.parent / "models").glob("*/model.json")}
    for path in sorted(ROOT.glob("*.json")):
        source = json.loads(path.read_text())
        measurement = source.get("measurement", {})
        if "decode" not in measurement or "metadata" not in source:
            continue
        metadata = source["metadata"]
        execution = measurement["execution"]
        model_name = measurement["model_name"]
        if model_name not in models:
            raise ValueError(f"{path.name}: unknown model_name {model_name!r}")
        runs.append({
            "model_name": model_name,
            "model": measurement["model"],
            "commit": metadata["commit_hash"],
            "file": path.name,
            "date": measurement["started_at"],
            "prompt": measurement["prompt"]["tokens"],
            "samples": measurement["decode"]["samples"],
            "warmups": measurement["decode"]["warmups"],
            "context_start": measurement["decode"]["context_start"],
            "context_end": measurement["decode"]["context_end"],
            "tps": measurement["decode"]["tokens_per_second"],
            "decode": measurement["decode"]["p50_ms"],
            "p95": measurement["decode"]["p95_ms"],
            "prefill": measurement["prefill"]["p50_ms"],
            "execution": execution,
            "metadata": metadata,
            "gpu_report": path.with_suffix(".gpu.html").name
                if path.with_suffix(".gpu.json.gz").exists() else None,
        })
    commits = list(dict.fromkeys(str(run["commit"]) for run in runs))
    output = subprocess.run(
        ["git", "show", "--no-patch", "--format=%H%x09%s", *commits],
        cwd=ROOT, check=True, capture_output=True, text=True,
    ).stdout
    subjects = dict(line.split("\t", 1) for line in output.splitlines() if "\t" in line)
    for run in runs:
        run["subject"] = subjects[str(run["commit"])]
    return sorted(runs, key=lambda run: str(run["date"]))


HTML = r'''<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Quasar3 · Benchmark history</title>
<style>
:root{color-scheme:dark;--bg:#10151b;--panel:#171e27;--line:#2b3542;--ink:#edf2f6;--muted:#a0aebc;--accent:#8fbded;--green:#83ddbb}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--ink);font:15px/1.5 system-ui,-apple-system,sans-serif}main{max-width:1540px;margin:auto;padding:40px 42px 54px}h1{font-size:35px;letter-spacing:-1.3px;font-weight:650;margin:4px 0 10px}p{margin:0;color:var(--muted)}.eyebrow{font:11px ui-monospace,monospace;letter-spacing:2px;color:var(--green)}.intro{max-width:900px}.top{display:flex;justify-content:space-between;align-items:flex-start;gap:20px}.stamp{font:12px ui-monospace,monospace;color:var(--muted);padding-top:12px;white-space:nowrap}.cards{display:grid;grid-template-columns:repeat(3,1fr);gap:14px;margin:25px 0}.card{border:1px solid var(--line);border-radius:10px;padding:17px 20px;background:var(--panel)}.card span{display:block;color:var(--muted);font-size:12px}.card strong{font-size:29px;font-weight:600;letter-spacing:-.6px}.card small{font-size:13px;font-weight:400;color:var(--muted);margin-left:8px}.card:last-child strong{color:var(--green)}.controls{display:flex;gap:14px;align-items:end;flex-wrap:wrap;margin-bottom:20px}label{font-size:12px;color:var(--muted);display:flex;flex-direction:column;gap:6px}select,input,button{font:inherit}select,input{color:var(--ink);background:var(--panel);border:1px solid #3b4858;border-radius:6px;padding:9px 11px;font-size:13px}select{min-width:205px}input{min-width:270px}.count{margin-left:auto;font-size:12px;padding-bottom:10px;color:var(--muted)}.chart-panel{border:1px solid var(--line);border-radius:10px;overflow:hidden;background:var(--panel)}.chart-heading{padding:18px 20px;border-bottom:1px solid var(--line);display:flex;gap:15px;align-items:center;justify-content:space-between}.chart-heading strong{font-size:15px}.chart-heading span{color:var(--muted);font-size:12px}.axis-row,.run{display:grid;grid-template-columns:minmax(290px,43%) 1fr;gap:22px;padding:0 20px}.axis-row{height:40px;align-items:center;background:#141b23}.axis-label{font-size:11px;text-transform:uppercase;letter-spacing:1px;color:var(--muted)}.axis{height:100%;position:relative;margin-right:78px}.tick{position:absolute;top:12px;transform:translateX(-50%);font:11px ui-monospace,monospace;color:var(--muted)}.tick:first-child{transform:none}.run{width:100%;border:0;border-bottom:1px solid #26313d;background:transparent;color:inherit;text-align:left;min-height:79px;cursor:pointer;align-items:center}.run{white-space:normal}.run>span:first-child{min-width:0}.subject{overflow-wrap:anywhere}.run:last-child{border-bottom:0}.run:hover{background:#202b37}.run[aria-pressed=true]{background:#23323b;box-shadow:inset 3px 0 var(--green)}.run:focus-visible,select:focus-visible,input:focus-visible{outline:2px solid var(--green);outline-offset:-2px}.subject{font-size:13px;font-weight:500;display:block;line-height:1.4;padding-top:9px}.revision{display:block;font:11px/1.6 ui-monospace,monospace;color:var(--muted);padding:4px 0 9px}.revision b{color:#c0cfdf;font-weight:400}.plot{position:relative;height:100%;min-height:78px;margin-right:78px;background:repeating-linear-gradient(to right,#33404c77 0,#33404c77 1px,transparent 1px,transparent 25%);display:flex;align-items:center}.bar{height:26px;background:var(--accent);border-radius:0 4px 4px 0;min-width:2px;position:relative}.latest .bar{background:var(--green)}.value{position:absolute;left:100%;top:3px;margin-left:10px;white-space:nowrap;font:13px ui-monospace,monospace;color:var(--ink)}.empty{padding:30px;text-align:center;color:var(--muted)}.detail{margin-top:18px;border:1px solid var(--line);border-radius:10px;padding:20px;background:var(--panel)}.detail h2{font-size:17px;margin:0 0 14px;font-weight:550}.detail-grid{display:grid;grid-template-columns:repeat(4,minmax(0,1fr));gap:15px 25px}.datum span{display:block;color:var(--muted);font-size:11px;margin-bottom:3px}.datum strong{font-size:12px;font-weight:450;overflow-wrap:anywhere;display:block}a{color:var(--green);text-decoration:none}a:hover{text-decoration:underline}.source{margin-top:16px;font-size:12px}.foot{font-size:12px;line-height:1.7;margin-top:18px;max-width:1050px}.legend{display:inline-flex;gap:7px;align-items:center}.dot{width:9px;height:9px;background:var(--green);border-radius:2px;display:inline-block}
@media(max-width:760px){main{padding:23px 16px}h1{font-size:28px}.top{display:block}.stamp{padding-top:12px}.cards{gap:8px}.card{padding:12px}.card strong{font-size:23px}.card small{display:block;margin:0}.controls{display:grid;grid-template-columns:1fr;gap:12px}.controls label{width:100%}select,input{min-width:0;width:100%}.count{margin-left:0}.axis-row,.run{grid-template-columns:minmax(170px,49%) 1fr;gap:12px;padding:0 12px}.plot,.axis{margin-right:58px}.run{min-height:100px}.subject{font-size:11px}.revision{font-size:10px}.value{font-size:11px;margin-left:5px}.detail-grid{grid-template-columns:repeat(2,minmax(0,1fr))}.chart-heading{align-items:flex-start;flex-direction:column;gap:5px}}
@media print{body{background:white;color:black}.controls{display:none}main{padding:10px}.run{break-inside:avoid}.chart-panel,.card,.detail{background:white}.subject,.value{color:black}.revision,.foot,p{color:#555}.run[aria-pressed=true]{background:#eef5f2}.bar{print-color-adjust:exact}}
</style>
</head>
<body><main>
<div class="top"><div class="intro"><div class="eyebrow">QUASAR3 / SPARK GB10</div><h1>Performance across commits</h1><p>Recorded core benchmark runs, labeled with their measured commit subjects. Select a run to inspect its configuration.</p></div><div class="stamp" id="stamp"></div></div>
<div class="cards"><div class="card"><span>Runs in this view</span><strong id="run-count"></strong></div><div class="card"><span>First recorded run</span><strong id="first"></strong><span id="first-id"></span></div><div class="card"><span>Latest recorded run</span><strong id="latest"></strong><span id="latest-id"></span></div></div>
<div class="controls"><label>Model<select id="model"><option value="all">All models</option></select></label><label>Workload<select id="workload"></select></label><label>Metric<select id="metric"><option value="tps">Decode throughput · tok/s</option><option value="decode">Decode latency · p50 ms</option><option value="p95">Decode latency · p95 ms</option><option value="prefill">Prefill latency · p50 ms</option></select></label><label>Filter commits<input id="search" type="search" placeholder="Commit subject or hash" autocomplete="off"></label><span class="count" id="count"></span></div>
<p id="mixed" class="foot" hidden>Mixed models: differences include model architecture and quantization, not just changes across commits.</p>
<section class="chart-panel" aria-label="Benchmark comparison"><div class="chart-heading"><strong id="chart-title"></strong><span><span class="legend"><i class="dot"></i> Latest run</span> · Oldest → newest</span></div><div class="axis-row"><span class="axis-label">Commit / configuration</span><div class="axis" id="axis"></div></div><div id="chart"></div></section>
<section class="detail" id="detail" aria-live="polite"></section>
<p class="foot">Each bar is one recorded run; repeated commits retain their separate measurements. The default view selects Qwen3.8-27B NVFP4. Model IDs match the models directory, including separate -nvfp4 variants. Workloads are separated by prompt length and measured decode steps. Precision, kernel settings and loading strategy changed over this history, so adjacent bars are not necessarily controlled A/B comparisons. Missing historical metadata is shown as “not recorded”. The chart contains the core JSON files in this directory; source links open the original measurements.</p>
<noscript><p>This interactive chart needs JavaScript enabled. All data is embedded in this file; no network access is required.</p></noscript>
</main>
<script id="benchmark-data" type="application/json">__DATA__</script>
<script>
'use strict';
const runs = JSON.parse(document.getElementById('benchmark-data').textContent);
const $ = id => document.getElementById(id);
const metrics = {tps:{label:'Decode throughput',unit:'tok/s',direction:'Higher is better'},decode:{label:'Decode latency · p50',unit:'ms',direction:'Lower is better'},p95:{label:'Decode latency · p95',unit:'ms',direction:'Lower is better'},prefill:{label:'Prefill latency · p50',unit:'ms',direction:'Lower is better'}};
const escape = s => String(s).replace(/[&<>"']/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
const format = n => Number(n).toLocaleString('en-US',{minimumFractionDigits:3,maximumFractionDigits:3});
const stamp = s => s.replace('T',' ').slice(0,19)+' UTC';
const defaultModel = 'qwen3.8-27b-nvfp4';
const models = [...new Set([defaultModel,...runs.map(r=>r.model_name)])].sort();
for(const model of models){const opt=document.createElement('option');opt.value=model;opt.textContent=model;$('model').append(opt);}
$('model').value=defaultModel;
const workloads = [...new Set(runs.map(r=>`${r.prompt}/${r.samples}`))].sort((a,b)=>Number(a.split('/')[0])-Number(b.split('/')[0]));
for(const key of workloads){const [p,n]=key.split('/');const opt=document.createElement('option');opt.value=key;opt.textContent=`${Number(p).toLocaleString()} prompt / ${n} decode`; $('workload').append(opt);}
const params = new URLSearchParams(location.search);
if(params.get('model')==='all'||models.includes(params.get('model'))) $('model').value=params.get('model');
if(workloads.includes(params.get('workload'))) $('workload').value=params.get('workload');
if(metrics[params.get('metric')]) $('metric').value=params.get('metric');
$('search').value=params.get('q')||'';
$('stamp').textContent=`${runs.length} runs · through ${runs.at(-1)?.date.slice(0,10)||'—'}`;
let selected = null;
function setting(r){const e=r.execution;const bits=[r.model_name, e.precision?e.precision.toUpperCase():'precision not recorded'];if(e.gdn_recurrent_state_dtype)bits.push(`GDN ${e.gdn_recurrent_state_dtype.toUpperCase()}`);if(e.moe_kernel)bits.push(e.moe_kernel);else if(e.moe_threadblocks)bits.push(`MoE ${e.moe_threadblocks} blocks`);if(e.weight_backend==='pinned_upload')bits.push('device weights');if(e.lm_head==='triton')bits.push('Triton LM');return bits.join(' · ');}
function datum(label,value){return `<div class="datum"><span>${escape(label)}</span><strong>${escape(value??'not recorded')}</strong></div>`;}
function inspect(r){
 selected=r.file;
 document.querySelectorAll('.run').forEach(el=>el.setAttribute('aria-pressed',String(el.dataset.file===selected)));
 const e=r.execution,m=r.metadata;
 const fields=[['Model',r.model_name],['Precision',e.precision],['Model path',r.model],['Commit',r.commit],['Recorded',stamp(r.date)],['Decode throughput',format(r.tps)+' tok/s'],['Decode p50 / p95',format(r.decode)+' / '+format(r.p95)+' ms'],['Prefill p50',format(r.prefill)+' ms'],['Workload',`${r.prompt} prompt · ${r.warmups} warmups · ${r.samples} measured decode`],['Decode context',`${r.context_start} → ${r.context_end}`],['Weight backend',e.weight_backend],['GDN recurrent state',e.gdn_recurrent_state_dtype],...(e.mlp?[['MLP kernel',e.mlp]]:[['Router logits',e.router_logits_dtype],['MoE kernel',e.moe_kernel??e.moe],['MoE threadblocks',e.moe_threadblocks]]),['LM head',e.lm_head],['CUDA / driver',`${m.cuda_runtime} / ${m.driver}`],['Execution',e.mode],['Rust toolchain',m.rust]];
 $('detail').innerHTML=`<h2>${escape(r.subject)}</h2><div class="detail-grid">${fields.map(x=>datum(...x)).join('')}</div><div class="source"><a href="${encodeURI(r.file)}">Open original benchmark JSON ↗</a>${r.gpu_report?` · <a href="${encodeURI(r.gpu_report)}">GPU time breakdown ↗</a>`:' · GPU breakdown not archived'}</div>`;
}
function render(){
 const metric=$('metric').value,meta=metrics[metric],query=$('search').value.toLowerCase().trim();
 const rows=runs.filter(r=>($('model').value==='all'||r.model_name===$('model').value) && `${r.prompt}/${r.samples}`===$('workload').value && `${r.subject} ${r.commit}`.toLowerCase().includes(query));
 $('mixed').hidden=new Set(rows.map(r=>r.model_name)).size<2;
 $('run-count').textContent=rows.length;
 $('count').textContent=`${rows.length} of ${runs.length} recorded runs`;
 $('chart-title').textContent=`${$('model').value==='all'?'All models':$('model').value} · ${meta.label} (${meta.unit}) · ${meta.direction}`;
 for(const [id,r] of [['first',rows[0]],['latest',rows.at(-1)]]){
  $(id).innerHTML=r?`${format(r[metric])}<small>${meta.unit}</small>`:'—';
  $(id+'-id').textContent=r?`${r.model_name} · ${r.commit.slice(0,7)} · ${r.date.slice(0,10)}`:'No matching runs';
 }
 if(!rows.length){$('chart').innerHTML='<div class="empty">No runs match this filter.</div>';$('axis').replaceChildren();$('detail').hidden=true;return;}
 $('detail').hidden=false;
 const max=Math.max(...rows.map(r=>r[metric]));
 const rough=max/4, magnitude=10**Math.floor(Math.log10(rough));
 const step=[1,2,2.5,5,10].map(n=>n*magnitude).find(n=>n>=rough);
 const limit=step*4;
 $('axis').innerHTML=Array.from({length:5},(_,i)=>`<span class="tick" style="left:${i*25}%">${Number((step*i).toFixed(2)).toLocaleString()}</span>`).join('');
 $('chart').replaceChildren();
 rows.forEach((r,i)=>{
  const button=document.createElement('button');button.className='run'+(i===rows.length-1?' latest':'');button.type='button';button.dataset.file=r.file;
  button.setAttribute('aria-label',`${r.model_name}, ${r.subject}, ${r.commit.slice(0,7)}, ${meta.label}: ${format(r[metric])} ${meta.unit}. Inspect run.`);
  button.innerHTML=`<span><span class="subject">${escape(r.subject)}</span><span class="revision"><b>${r.commit.slice(0,7)}</b> · ${escape(setting(r)||r.date.slice(0,10))}</span></span><span class="plot"><span class="bar" style="width:${r[metric]/limit*100}%"><span class="value">${format(r[metric])}</span></span></span>`;
  button.addEventListener('click',()=>{inspect(r);$('detail').scrollIntoView({behavior:'smooth',block:'nearest'});});$('chart').append(button);
 });
 inspect(rows.find(r=>r.file===selected)||rows.at(-1));
}
for(const id of ['model','workload','metric','search']) $(id).addEventListener('input',render);
render();
</script></body></html>
'''


# Accumulated GPU time, grouped across repeated model execution.
GPU_HTML = r'''<!doctype html>
<html lang="en">
<meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Quasar3 · GPU time breakdown</title>
<style>
:root{color-scheme:dark;--bg:#10151b;--panel:#171e27;--ink:#edf2f6;--muted:#a0aebc;--line:#334151;--green:#83ddbb;--red:#ef9d9d}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--ink);font:14px/1.5 system-ui,sans-serif}main{max-width:1250px;margin:auto;padding:32px}a{color:var(--green)}h1{font-size:30px;letter-spacing:-.7px;margin:12px 0 5px}h2{font-size:17px;margin:0 0 12px}p{color:var(--muted);margin:7px 0}.controls{display:flex;gap:18px;flex-wrap:wrap;margin:22px 0}label{display:flex;flex-direction:column;gap:5px;color:var(--muted);font-size:12px}select,button{font:inherit;color:inherit}select{background:var(--panel);color:var(--ink);border:1px solid var(--line);border-radius:6px;padding:9px;min-width:190px}button{cursor:pointer}button:focus-visible,summary:focus-visible,select:focus-visible{outline:2px solid var(--green)}.cards{display:grid;grid-template-columns:repeat(3,1fr);gap:14px}.card,.panel{background:var(--panel);border:1px solid var(--line);border-radius:9px;padding:19px}.card span{color:var(--muted);font-size:12px;display:block}.card strong{font-size:27px;font-weight:600}.card small{font-size:12px;color:var(--muted)}.panel{margin-top:18px}.stack{display:flex;height:48px;border-radius:5px;overflow:hidden;margin:18px 0 12px}.stack button{border:0;min-width:0;overflow:hidden;white-space:nowrap;text-overflow:ellipsis;font-size:12px;font-weight:600;color:#10151b;padding:0}.stack button:hover{filter:brightness(1.15)}.hint{font-size:12px;color:var(--muted)}#insight{color:var(--ink);font-size:16px;padding:4px 0 13px}.tree{margin-top:15px}.group{border-top:1px solid var(--line);padding:6px 0}.group>summary{cursor:pointer;list-style:none}.row{display:grid;grid-template-columns:minmax(220px,32%) 1fr 102px 100px;align-items:center;gap:13px;min-height:40px;width:100%;text-align:left;background:none;border:0;padding:5px 0;color:var(--ink)}.label{overflow-wrap:anywhere;font-size:13px}.group>summary .label:before{content:'▸ ';color:var(--muted)}.group[open]>summary .label:before{content:'▾ '}.row .barbox{height:15px;background:#253140;border-radius:3px;overflow:hidden}.row .bar{height:100%;border-radius:3px;min-width:1px}.time{font:13px ui-monospace,monospace;text-align:right;white-space:nowrap}.delta{font:12px ui-monospace,monospace;text-align:right;white-space:nowrap}.slower{color:var(--red)}.faster{color:var(--green)}.neutral{color:var(--muted)}.kind{padding-left:20px;cursor:pointer}.kind:hover{background:#202b37}.kind[aria-pressed=true]{background:#23333c}.head{color:var(--muted);font-size:11px}.head .time,.head .delta{font:11px system-ui,sans-serif}.head .label{font-size:11px}#detail{scroll-margin:15px}.detail-title{display:flex;align-items:center;justify-content:space-between;gap:15px}table{width:100%;border-collapse:collapse;font-size:12px}th{text-align:left;font-weight:400;color:var(--muted)}td,th{padding:10px 8px;border-bottom:1px solid var(--line);vertical-align:top}td:first-child{width:53%;overflow-wrap:anywhere}td:not(:first-child){white-space:nowrap;font-family:ui-monospace,monospace}pre{max-height:260px;overflow:auto;white-space:pre-wrap;overflow-wrap:anywhere;font:11px/1.5 ui-monospace,monospace}.raw{margin-top:7px}.raw summary{cursor:pointer;color:var(--muted)}.foot{font-size:12px;margin:18px 0}#provenance{overflow-wrap:anywhere}.unit{color:var(--muted);font-size:12px}
@media(max-width:760px){main{padding:18px}.cards{gap:7px}.card{padding:10px}.card strong{font-size:21px}.panel{padding:12px}.row{grid-template-columns:minmax(115px,36%) 1fr 75px 69px;gap:6px}.label{font-size:11px}.time,.delta{font-size:11px}.kind{padding-left:10px}.head .label{font-size:10px}td,th{padding:8px 4px;font-size:10px}select{max-width:100%;min-width:170px}.controls label{max-width:100%}.stack button{font-size:10px}}
body:not(.comparing) .delta{display:none}body:not(.comparing) .row{grid-template-columns:minmax(220px,32%) 1fr 102px}
@media(max-width:760px){body:not(.comparing) .row{grid-template-columns:minmax(115px,36%) 1fr 75px}}
</style>
<main>
<a href="performance.html">← Benchmark history</a><h1>Where GPU time goes</h1><p id="identity">Loading…</p>
<p>Repeated launches accumulated by kernel family, across every layer. Expand a family to inspect its kernels.</p><p class="hint">These are profiled GPU captures; benchmark history reports latency from separate unprofiled runs.</p>
<div class="controls"><label>Phase<select id="phase"></select></label><label>Optional capture comparison<select id="compare"></select></label></div>
<div class="cards"><div class="card"><span>Profiled GPU span</span><strong id="span"></strong><small class="unit"></small></div><div class="card"><span>Accumulated GPU work</span><strong id="work"></strong><small class="unit"></small></div><div class="card"><span>No GPU work</span><strong id="gap"></strong><small class="unit"></small></div></div>
<section class="panel"><h2>Accumulated time <span class="unit"></span></h2><div id="insight"></div><p id="comparison-source" class="hint"></p><p id="unprofiled" class="hint"></p><div id="stack" class="stack" aria-label="Accumulated work and gaps by family"></div><p id="accounting" class="hint"></p><div class="row head"><span class="label">Family / kernel kind</span><span>Share of accumulated work + gaps</span><span class="time" id="unit-head"></span><span class="delta">Δ capture timing</span></div><div id="tree" class="tree"></div></section>
<section class="panel" id="detail" hidden><div class="detail-title"><h2 id="detail-title"></h2><span id="detail-time"></span></div><p id="detail-note"></p><table><thead><tr><th>Kernel variant / evidence</th><th id="calls-head"></th><th id="detail-unit"></th><th>Mean µs</th></tr></thead><tbody id="variants"></tbody></table></section>
<details class="panel"><summary>How to read gaps and comparisons</summary><p>Gaps are intervals with no captured kernel, copy, or memset active on any stream. They cover only the time between the first and last GPU operation.</p><p>Gap labels describe CUDA API calls active on the host during those intervals. They do not prove why the GPU was idle. If calls overlap, synchronization takes priority, followed by memory management, transfer, launch/setup, and other CUDA calls. A gap can be split into several classified pieces. “No CUDA API call recorded” leaves host work, scheduling, and unobserved waits unresolved.</p><p>Kernel families come from kernel names. Layer numbers and projection roles are not inferred. Work totals sum individual GPU operations; concurrent work is counted in both operations. Empty gaps are counted once. Decode totals are divided by the captured decode sample count; prefill is one warmed prefill capture.</p><p>Comparisons use captured GPU timing from matching model, prompt length, and decode sample count. They can differ from the unprofiled latency in benchmark history. A difference between captures is not an optimisation result: showing a speedup requires a controlled A/B experiment. Time spent in a kernel is not necessarily removable overhead.</p></details>
<p id="provenance" class="foot"></p>
</main>
<script id="gpu-data" type="application/json">__GPU_DATA__</script>
<script>
'use strict';
const data=JSON.parse(document.getElementById('gpu-data').textContent),$=id=>document.getElementById(id);
const run=data.reports.find(r=>r.benchmark===data.selected);
const colors={'Matrix multiplication':'#a894e4','GDN':'#e8ac74','Full attention':'#e78eaa','Quantization':'#80bbed','Normalization':'#83ddbb','MLP activation':'#d9c885','Embedding / sampling':'#c5cd8d','Memory operations':'#92a1b2','Other kernels':'#adb5bd','No GPU work':'#dc9a77'};
let selected=null;
const fmt=(n,d=3)=>n.toLocaleString('en-US',{minimumFractionDigits:d,maximumFractionDigits:d});
const sum=rows=>rows.reduce((n,row)=>n+row.ns,0);
const groupTime=group=>Object.values(group||{}).reduce((n,rows)=>n+sum(rows),0);
const divisor=r=>$('phase').value==='decode'?r.decode_samples:1;
const ms=(ns,r=run)=>ns/1e6/divisor(r);
function element(tag,text,className){const e=document.createElement(tag);if(text!==undefined)e.textContent=text;if(className)e.className=className;return e;}
function delta(ns,old,reference){const d=reference?ms(ns)-ms(old,reference):null;return element('span',d===null?'—':`${d>0?'+':''}${fmt(d)}`,`delta ${d===null||Math.abs(d)<.0005?'neutral':d>0?'slower':'faster'}`);}
function row(label,ns,old,reference,total,color,tag='span'){
 const e=element(tag,undefined,'row');e.append(element('span',label,'label'));
 const box=element('span',undefined,'barbox'),bar=element('div',undefined,'bar');bar.style.width=(ns/total*100)+'%';bar.style.background=color;box.append(bar);e.append(box,element('span',fmt(ms(ns)),'time'),delta(ns,old,reference));return e;
}
function details(group,kind){
 selected=[group,kind];const phase=run.phases[$('phase').value],rows=phase.groups[group][kind];
 $('detail').hidden=false;$('detail-title').textContent=group+' / '+kind;$('detail-time').textContent=fmt(ms(sum(rows)))+' '+($('phase').value==='decode'?'ms/token':'ms/prefill');
 const gaps=group==='No GPU work';$('calls-head').textContent=(gaps?'Pieces':'Calls')+($('phase').value==='decode'?'/token':'');$('detail-unit').textContent=$('unit-head').textContent;
 $('detail-note').textContent=gaps?'Observed host API overlap during otherwise empty GPU intervals. Counts and means refer to classified pieces of gaps, not complete API calls. This is evidence of host activity, not a causal attribution.':'Variants group identical kernel names and launch shapes across the whole capture.';
 $('variants').replaceChildren();
 for(const r of [...rows].sort((a,b)=>b.ns-a.ns)){
  const tr=element('tr'),name=element('td',r.label);
  if(r.definition){const raw=element('details',undefined,'raw');raw.append(element('summary','Full name and launch parameters'),element('pre',JSON.stringify(r.definition,null,2)));name.append(raw);}
  tr.append(name,element('td',(r.count/divisor(run)).toLocaleString('en-US',{maximumFractionDigits:2})),element('td',fmt(ms(r.ns))),element('td',fmt(r.ns/r.count/1000)));$('variants').append(tr);
 }
 document.querySelectorAll('.kind').forEach(b=>b.setAttribute('aria-pressed',String(b.dataset.key===JSON.stringify(selected))));
}
function render(){
 const phase=run.phases[$('phase').value],reference=data.reports.find(r=>r.benchmark===$('compare').value),before=reference?.phases[$('phase').value];
 document.body.classList.toggle('comparing',Boolean(before));
 const unit=$('phase').value==='decode'?'ms/token':'ms/prefill';document.querySelectorAll('.unit').forEach(e=>e.textContent=' '+unit);$('unit-head').textContent=unit;
 $('span').textContent=fmt(ms(phase.span_ns));$('work').textContent=fmt(ms(phase.work_ns));$('gap').textContent=fmt(ms(phase.gap_ns));
 const total=phase.work_ns+phase.gap_ns;
 const families=[...new Set([...Object.keys(phase.groups),...Object.keys(before?.groups||{})])].sort((a,b)=>groupTime(phase.groups[b])-groupTime(phase.groups[a]));
 const open=new Set([...document.querySelectorAll('.group[open]')].map(e=>e.dataset.group));
 $('tree').replaceChildren();$('stack').replaceChildren();let largest=null;
 for(const family of families){
  const group=phase.groups[family]||{},old=before?.groups[family]||{},ns=groupTime(group),oldNs=groupTime(old),color=colors[family]||colors['Other kernels'];
  if(!ns&&!oldNs)continue;
  const section=element('details',undefined,'group');section.dataset.group=family;section.open=open.size?open.has(family):['Matrix multiplication','No GPU work'].includes(family);
  const summary=element('summary');summary.append(row(family,ns,oldNs,reference,total,color));section.append(summary);
  if(ns){const segment=element('button',`${family} · ${fmt(ms(ns),2)}`);segment.style.width=(ns/total*100)+'%';segment.style.background=color;segment.title=`${family}: ${fmt(ms(ns))} ${unit} (${fmt(ns/total*100,1)}%)`;segment.onclick=()=>{section.open=true;section.scrollIntoView({behavior:'smooth',block:'nearest'});};$('stack').append(segment);}
  const kinds=[...new Set([...Object.keys(group),...Object.keys(old)])].sort((a,b)=>sum(group[b]||[])-sum(group[a]||[]));
  for(const kind of kinds){
   const amount=sum(group[kind]||[]),previous=sum(old[kind]||[]),button=row(kind,amount,previous,reference,total,color,'button');button.classList.add('kind');button.dataset.key=JSON.stringify([family,kind]);
   button.setAttribute('aria-pressed',String(JSON.stringify(selected)===button.dataset.key));button.disabled=!group[kind];button.onclick=()=>{details(family,kind);$('detail').scrollIntoView({behavior:'smooth',block:'nearest'});};section.append(button);
   if(before){const change=ms(amount)-ms(previous,reference);if(!largest||Math.abs(change)>Math.abs(largest.change))largest={family,kind,change,amount,previous};}
  }
  $('tree').append(section);
 }
 if(largest)$('insight').textContent=`Largest timing difference: ${largest.kind} ${fmt(ms(largest.previous,reference))} → ${fmt(ms(largest.amount))} ${unit} (${largest.change>=0?'+':''}${fmt(largest.change)}).`;
 else{const [family,group]=Object.entries(phase.groups).sort((a,b)=>groupTime(b[1])-groupTime(a[1]))[0];$('insight').textContent=`${family} accounts for ${fmt(groupTime(group)/total*100,1)}% of accumulated work and gaps.`;}
 $('comparison-source').replaceChildren('Profiled capture sources: ');
 for(const [label,r] of [['selected',run],['comparison',reference]]){
  if(!r)continue;const link=element('a',`${label} ${r.commit.slice(0,7)}`);link.href=r.benchmark;$('comparison-source').append(link,' ');
 }
 const phaseName=$('phase').value;
 $('unprofiled').textContent=`Separately measured unprofiled ${phaseName} p50: ${fmt(run.unprofiled_ms[phaseName])} ms`+(reference?`; comparison ${fmt(reference.unprofiled_ms[phaseName])} ms.`:'.');
 $('accounting').textContent=`Bars accumulate repeated launches, not chronological positions. Concurrent GPU work adds ${fmt(ms(phase.overlap_ns))} ${unit} above the measured span; gaps are counted once.`;
 $('provenance').textContent=`${run.benchmark} · ${phase.source}`;
 if(selected&&phase.groups[selected[0]]?.[selected[1]])details(...selected);else{$('detail').hidden=true;selected=null;}
}
function choosePhase(){
 const previous=$('compare').value;$('compare').replaceChildren(new Option('No comparison',''));
 const comparable=data.reports.filter(r=>r.benchmark!==run.benchmark&&r.phases[$('phase').value]);
 for(const r of comparable)$('compare').add(new Option(`${r.commit.slice(0,7)} · ${r.benchmark.slice(0,10)} · ${r.benchmark.slice(11,17)}`,r.benchmark));
 $('compare').value=comparable.some(r=>r.benchmark===previous)?previous:'';render();
}
$('identity').textContent=`${run.model} · ${run.commit.slice(0,7)} · ${run.prompt_tokens} prompt / ${run.decode_samples} measured decode`;
for(const phase of ['prefill','decode']){const option=new Option(phase+(run.phases[phase]?'':' · not captured'),phase);option.disabled=!run.phases[phase];$('phase').add(option);}
$('phase').value=run.phases[location.hash.slice(1)]?location.hash.slice(1):run.phases.decode?'decode':'prefill';
$('phase').onchange=choosePhase;$('compare').onchange=render;choosePhase();
</script></html>
'''


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--import-gpu", nargs="+", type=Path,
                        help="Archive existing Nsight traces for these benchmark JSON files")
    parser.add_argument("--remote", help="Read traces through SSH, e.g. sp10@sp10")
    parser.add_argument("--extract-directory", type=Path, help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.extract_directory:
        data = json.dumps(extract_directory(args.extract_directory), separators=(",", ":")).encode()
        sys.stdout.buffer.write(gzip.compress(data, mtime=0))
        return
    if args.remote and not args.import_gpu:
        parser.error("--remote requires --import-gpu")
    for path in args.import_gpu or []:
        print(archive(path, args.remote))
    sidecars = sorted(ROOT.glob("*.gpu.json.gz"))
    reports = [report_data(path) for path in sidecars]
    for sidecar in sidecars:
        render_gpu_report(sidecar, reports)
    data = json.dumps(collect_runs(), separators=(",", ":"), ensure_ascii=False)
    # Keep data inside the non-executable JSON script even for unusual subjects.
    data = data.replace("&", "\\u0026").replace("<", "\\u003c").replace(">", "\\u003e")
    output = ROOT / "performance.html"
    output.write_text(HTML.replace("__DATA__", data))
    print(output)


if __name__ == "__main__":
    main()
