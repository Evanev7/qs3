"""Plain 27B NVFP4 timing survey using an isolated Rust benchmark overlay."""

import datetime
import hashlib
import json
import os
from pathlib import Path
import shutil
import statistics
import subprocess
import tempfile

REPO = Path(__file__).resolve().parents[2]
PROTOTYPES = REPO / ".prototypes"
REVISION = "dbb8f445b3145f8a4c18ddc769f032d57d32867c"
TACTICS = ("32dp", "32streamk", "64dp", "64streamk")


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2) + "\n")


def command(args, output, *, cwd=REPO, env=None):
    print("Running:", " ".join(map(str, args)), flush=True)
    with output.open("w") as log:
        subprocess.run(
            args, cwd=cwd, env=env, stdout=log, stderr=subprocess.STDOUT, check=True
        )


def replace(path, before, after):
    source = path.read_text()
    assert source.count(before) == 1, (path, before)
    path.write_text(source.replace(before, after))


def main():
    config = json.loads(
        subprocess.check_output(
            ["nix", "eval", "--offline", "--json", "--file", "models/config.nix"],
            cwd=REPO,
        )
    )
    assert config["engine"]["model"] == "qwen3.8-27b-nvfp4"
    assert config["engine"]["mtp"] is False
    stamp = datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    output = Path(
        tempfile.mkdtemp(prefix=f"nvfp4-perf-{stamp}-", dir=PROTOTYPES / "out")
    )
    print("OUTPUT", output, flush=True)
    work = output / "checkout"
    work.mkdir()
    for path in REPO.iterdir():
        if path.name not in {"src", "target", ".git", ".prototypes"}:
            (work / path.name).symlink_to(path, target_is_directory=path.is_dir())
    shutil.copytree(REPO / "src", work / "src")
    shutil.copy2(__file__, output / "run.py")
    benchmark = work / "src/loader/benchmark.rs"
    replace(
        benchmark,
        "let (config, weights) = loaded",
        "let (mut config, weights) = loaded",
    )
    replace(
        benchmark,
        "    let quantized = weights.is_quantized();",
        """    config.nvfp4_tactic = match std::env::var("QS3_PERF_TACTIC").unwrap().as_str() {
        "32dp" => Nvfp4Tactic::Tile128x32Dp,
        "32streamk" => Nvfp4Tactic::Tile128x32StreamK,
        "64dp" => Nvfp4Tactic::Tile128x64Dp,
        "64streamk" => Nvfp4Tactic::Tile128x64StreamK,
        _ => panic!("unknown prototype tactic"),
    };
    config.qscb_workspace_bytes = std::env::var("QS3_PERF_WORKSPACE_MIB")
        .unwrap().parse::<usize>().unwrap().checked_mul(1 << 20).unwrap();
    let quantized = weights.is_quantized();""",
    )
    with (output / "overlay.diff").open("wb") as file:
        result = subprocess.run(
            ["git", "diff", "--no-index", str(REPO / "src"), str(work / "src")],
            stdout=file,
        )
        assert result.returncode == 1
    snapshot = (
        Path.home()
        / ".cache/huggingface/hub/models--nvidia--Qwen3.8-27B-NVFP4/snapshots"
        / REVISION
    )
    assert (snapshot / "model.safetensors.index.json").is_file()
    env = dict(
        os.environ,
        CARGO_TARGET_DIR=str(REPO / "target"),
        QS3_QWEN36_MODEL_DIR=str(snapshot),
    )
    env.pop("QS3_PROFILE", None)
    write_json(
        output / "metadata.json",
        {
            "commit": subprocess.check_output(
                ["git", "rev-parse", "HEAD"], cwd=REPO, text=True
            ).strip(),
            "model_revision": REVISION,
            "config": config,
            "native_archive_sha256": hashlib.sha256(
                (REPO / "build/libqs_native.a").read_bytes()
            ).hexdigest(),
            "protocol": "sequential unprofiled processes; 2 prefill warmups + 5 samples; 4 decode warmups + 64 samples; reverse-order repeats; independent greedy continuations compared",
            "scope": "existing NVFP4 tactics and FP8 cuBLASLt workspace; no MTP or new inference kernels",
        },
    )
    command(
        ["cargo", "build", "--release", "--bin", "qs3-bench"],
        output / "build.log",
        cwd=work,
        env=env,
    )
    binary = REPO / "target/release/qs3-bench"
    records = []

    def measure(tactic, workspace, context, phase, repeat):
        label = f"{phase}-{context}-{tactic}-{workspace}m-r{repeat}"
        case = output / label
        case.mkdir()
        child_env = dict(
            env,
            QS3_PERF_TACTIC=tactic,
            QS3_PERF_WORKSPACE_MIB=str(workspace),
            QS3_BENCH_CONTEXT_TOKENS=str(context),
            QS3_BENCH_DECODE_SAMPLES="64",
        )
        write_json(
            case / "command.json",
            {
                "argv": [str(binary), "--measure-pass"],
                "overrides": {
                    k: v for k, v in child_env.items() if k.startswith("QS3_")
                },
            },
        )
        with (
            (case / "result.json").open("w") as stdout,
            (case / "run.log").open("w") as stderr,
        ):
            subprocess.run(
                [str(binary), "--measure-pass"],
                cwd=work,
                env=child_env,
                stdout=stdout,
                stderr=stderr,
                check=True,
            )
        result = json.loads((case / "result.json").read_text())["measurement"]
        assert result["execution"]["mode"] == "eager"
        assert result["execution"]["precision"] == "nvfp4_fp8"
        assert result["decode"]["samples"] == 64
        row = {
            "case": label,
            "phase": phase,
            "repeat": repeat,
            "context": context,
            "tactic": tactic,
            "workspace_mib": workspace,
            "decode_tok_s": result["decode"]["tokens_per_second"],
            "decode_p50_ms": result["decode"]["p50_ms"],
            "prefill_p50_ms": result["prefill"]["p50_ms"],
            "tokens": result["decode"]["generated_token_ids"],
        }
        records.append(row)
        write_json(output / "records.json", records)
        print(json.dumps({k: v for k, v in row.items() if k != "tokens"}), flush=True)

    # Include both ends of the tactic order twice to expose drift.
    for context in (102, 1024):
        for repeat, tactics in enumerate((TACTICS, tuple(reversed(TACTICS)))):
            for tactic in tactics:
                measure(tactic, 64, context, "tactics", repeat)
    means = {
        t: statistics.mean(r["decode_tok_s"] for r in records if r["tactic"] == t)
        for t in TACTICS
    }
    best = max(means, key=means.get)
    # Workspace changes affect prepared cuBLASLt FP8 algorithms for both phases.
    for context in (102, 1024):
        for repeat, workspaces in enumerate(((16, 64, 256), (256, 64, 16))):
            for workspace in workspaces:
                measure(best, workspace, context, "workspace", repeat)
    summaries = []
    for phase in ("tactics", "workspace"):
        keys = sorted(
            {
                (r["context"], r["tactic"], r["workspace_mib"])
                for r in records
                if r["phase"] == phase
            }
        )
        for context, tactic, workspace in keys:
            rows = [
                r
                for r in records
                if r["phase"] == phase
                and (r["context"], r["tactic"], r["workspace_mib"])
                == (context, tactic, workspace)
            ]
            control = next(
                r
                for r in records
                if r["context"] == context
                and r["tactic"] == "32dp"
                and r["workspace_mib"] == 64
            )
            prefixes = [
                next(
                    (
                        i
                        for i, pair in enumerate(zip(r["tokens"], control["tokens"]))
                        if pair[0] != pair[1]
                    ),
                    len(r["tokens"]),
                )
                for r in rows
            ]
            summaries.append(
                {
                    "phase": phase,
                    "context": context,
                    "tactic": tactic,
                    "workspace_mib": workspace,
                    "decode_tok_s": statistics.mean(r["decode_tok_s"] for r in rows),
                    "prefill_p50_ms": statistics.mean(
                        r["prefill_p50_ms"] for r in rows
                    ),
                    "decode_tok_s_runs": [r["decode_tok_s"] for r in rows],
                    "common_prefix_with_control": prefixes,
                    "repeat_tokens_exact": all(
                        r["tokens"] == rows[0]["tokens"] for r in rows
                    ),
                }
            )
    write_json(output / "summary.json", summaries)
    write_json(
        output / "COMPLETE.json",
        {"cases": len(records), "best_tactic_for_workspace_sweep": best},
    )
    print("COMPLETE", output, flush=True)


if __name__ == "__main__":
    main()
