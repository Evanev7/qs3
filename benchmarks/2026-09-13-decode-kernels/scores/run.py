"""Replay the saved vLLM prefixes through the existing qs3 score diagnostic."""

import datetime
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile


def main():
    prototypes = Path(__file__).resolve().parents[1]
    repo = prototypes.parent
    reference = prototypes / "out/vllm-correctness-2026-09-13T121347Z-8y4EGb/data"
    workloads = ("102-32", "500-200", "4000-800")
    completed = json.loads((reference / "SUCCESS.json").read_text())["workloads"]
    assert all(name in completed for name in workloads)
    for name in workloads:
        spec = json.loads((reference / name / "input.json").read_text())
        scores = json.loads((reference / name / "vllm/scores.json").read_text())
        assert spec == scores["input"]
        assert len(scores["records"]) == len(spec["forced_ids"]) + 1
        assert all((reference / name / "vllm" / row["file"]).is_file()
                   for row in scores["records"])

    stamp = datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H%M%SZ")
    output = Path(tempfile.mkdtemp(prefix=f"qs3-correctness-{stamp}-", dir=prototypes / "out"))
    print(f"qs3 replay: {output}", flush=True)
    shutil.copy2(__file__, output / "run.py")
    shutil.copy2(repo / "src/loader/tests/scores.rs", output / "scores.rs")
    shutil.copy2(prototypes / "same_prefix_scores/compare.py", output / "compare.py")
    (output / "source.diff").write_bytes(subprocess.check_output(["git", "diff", "HEAD"], cwd=repo))
    metadata = {
        "commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=repo, text=True).strip(),
        "reference": os.path.relpath(reference, output),
        "profile": "release",
        "weight_backend": "pinned_upload",
        "timing": "instrumented correctness replay; not a performance benchmark",
    }
    (output / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")

    with (output / "build.log").open("w") as log:
        subprocess.run(["just", "ninja"], cwd=repo, stdout=log, stderr=subprocess.STDOUT, check=True)
        subprocess.run(["ninja", "-C", "build"], cwd=repo, stdout=log, stderr=subprocess.STDOUT, check=True)

    for name in workloads:
        case = output / name
        case.mkdir()
        shutil.copy2(reference / name / "input.json", case / "input.json")
        (case / "vllm").symlink_to(os.path.relpath(reference / name / "vllm", case), target_is_directory=True)
        env = dict(os.environ, QS3_SCORE_INPUT=str(case / "input.json"), QS3_SCORE_OUTPUT=str(case / "qs3"))
        print(f"Replaying {name}; log: {case / 'replay.log'}", flush=True)
        with (case / "replay.log").open("w") as log:
            subprocess.run([
                "cargo", "test", "--release", "--lib",
                "loader::tests::scores::real_qwen36_same_prefix_scores",
                "--", "--ignored", "--exact", "--nocapture", "--test-threads=1",
            ], cwd=repo, env=env, stdout=log, stderr=subprocess.STDOUT, check=True)
        # Missing output also catches an accidentally unmatched Cargo test filter.
        assert (case / "qs3/scores.json").is_file()
        subprocess.run([sys.executable, str(output / "compare.py"), str(case)], check=True)
    (output / "SUCCESS.json").write_text(json.dumps({"workloads": workloads}) + "\n")
    print(f"qs3 comparisons: {output}", flush=True)


if __name__ == "__main__":
    main()
