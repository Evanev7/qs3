"""Capture every logit row for workspace/tactic comparisons; timings are invalid."""

import array
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import tempfile

from run import REPO, REVISION, command, replace, write_json


def main():
    output = Path(tempfile.mkdtemp(prefix="nvfp4-perf-logits-", dir=REPO / ".prototypes/out"))
    print("OUTPUT", output, flush=True)
    work = output / "checkout"
    work.mkdir()
    for path in REPO.iterdir():
        if path.name not in {"src", "target", ".git", ".prototypes"}:
            (work / path.name).symlink_to(path, target_is_directory=path.is_dir())
    shutil.copytree(REPO / "src", work / "src")
    shutil.copy2(__file__, output / "validate.py")
    runner = work / "src/model/runner/mod.rs"
    replace(runner, "#[cfg(test)]\nuse crate::dtype::F32;", "use crate::dtype::F32;")
    replace(runner, "    #[cfg(test)]\n    pub(crate) fn last_logits_row_for_test", "    pub(crate) fn last_logits_row_for_test")
    benchmark = work / "src/loader/benchmark.rs"
    replace(benchmark, "let (config, weights) = loaded", "let (mut config, weights) = loaded")
    replace(benchmark, "    let quantized = weights.is_quantized();", '''    config.qscb_workspace_bytes = std::env::var("QS3_PERF_WORKSPACE_MIB").unwrap().parse::<usize>().unwrap() << 20;
    config.nvfp4_tactic = match std::env::var("QS3_PERF_TACTIC").unwrap().as_str() {
        "32dp" => Nvfp4Tactic::Tile128x32Dp,
        "64dp" => Nvfp4Tactic::Tile128x64Dp,
        _ => panic!("unknown tactic"),
    };
    let quantized = weights.is_quantized();''')
    replace(benchmark, "    let mut live_tokens = prefill.live_tokens;", "    dump_logits(&runner, prefill.live_tokens.len());\n    let mut live_tokens = prefill.live_tokens;")
    replace(benchmark, "    (elapsed, result.live_tokens)", "    dump_logits(runner, result.live_tokens.len());\n    (elapsed, result.live_tokens)")
    benchmark.write_text(benchmark.read_text() + '''
fn dump_logits(runner: &ModelRunner, tokens: usize) {
    let logits = runner.last_logits_row_for_test().unwrap();
    let bytes: Vec<u8> = logits.into_iter().flat_map(f32::to_le_bytes).collect();
    let directory = std::path::PathBuf::from(std::env::var_os("QS3_PERF_LOGITS").unwrap());
    std::fs::write(directory.join(format!("{tokens}.f32")), bytes).unwrap();
}
''')
    with (output / "overlay.diff").open("wb") as file:
        diff = subprocess.run(["git", "diff", "--no-index", str(REPO / "src"), str(work / "src")], stdout=file)
        assert diff.returncode == 1
    snapshot = Path.home() / ".cache/huggingface/hub/models--nvidia--Qwen3.8-27B-NVFP4/snapshots" / REVISION
    env = dict(os.environ, CARGO_TARGET_DIR=str(REPO / "target"), QS3_QWEN36_MODEL_DIR=str(snapshot))
    env.pop("QS3_PROFILE", None)
    command(["cargo", "build", "--release", "--bin", "qs3-bench"], output / "build.log", cwd=work, env=env)
    cases = [(102, "32dp", 64), (102, "32dp", 16), (1024, "32dp", 64), (1024, "32dp", 16), (1024, "64dp", 16)]
    summaries = []
    for context, tactic, workspace in cases:
        case = output / f"{context}-{tactic}-{workspace}m"
        case.mkdir()
        child_env = dict(env, QS3_PERF_WORKSPACE_MIB=str(workspace), QS3_PERF_TACTIC=tactic, QS3_PERF_LOGITS=str(case), QS3_BENCH_CONTEXT_TOKENS=str(context), QS3_BENCH_DECODE_SAMPLES="64")
        with (case / "result.json").open("w") as stdout, (case / "run.log").open("w") as stderr:
            subprocess.run([str(REPO / "target/release/qs3-bench"), "--measure-pass"], cwd=work, env=child_env, stdout=stdout, stderr=stderr, check=True)
        baseline = output / f"{context}-32dp-64m"
        tokens = json.loads((case / "result.json").read_text())["measurement"]["decode"]["generated_token_ids"]
        control_tokens = json.loads((baseline / "result.json").read_text())["measurement"]["decode"]["generated_token_ids"]
        assert len(tokens) == len(control_tokens) == 68
        rows = []
        for path in sorted(case.glob("*.f32"), key=lambda p: int(p.stem)):
            raw, control = path.read_bytes(), (baseline / path.name).read_bytes()
            assert len(raw) == len(control) == 248320 * 4
            a, b = array.array("f"), array.array("f")
            a.frombytes(raw)
            b.frombytes(control)
            assert all(math.isfinite(v) for v in a) and all(math.isfinite(v) for v in b)
            delta = [float(x) - float(y) for x, y in zip(a, b)]
            rows.append({"tokens": int(path.stem), "sha256": hashlib.sha256(raw).hexdigest(), "control_sha256": hashlib.sha256(control).hexdigest(), "exact": raw == control, "changed": sum(x != y for x, y in zip(a, b)), "max_abs": max(map(abs, delta)), "rmse": math.sqrt(sum(v*v for v in delta)/len(delta))})
        assert len(rows) == 69
        summary = {"case": case.name, "generated_tokens_exact": tokens == control_tokens, "rows": rows, "all_rows_exact": all(r["exact"] for r in rows)}
        summaries.append(summary)
        write_json(output / "summary.json", summaries)
        print(json.dumps({k:v for k,v in summary.items() if k != "rows"}), flush=True)
    write_json(output / "metadata.json", {"commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO, text=True).strip(), "model_revision": REVISION, "scope": "69 full logit rows per case: fresh prefill and 68 independent greedy steps. Timings invalid because of per-step downloads."})
    write_json(output / "COMPLETE.json", {"cases": len(cases)})


if __name__ == "__main__":
    main()
