"""Capture each selected pinned Qwen model in a separate vLLM container."""

import argparse
import datetime
import json
import os
from pathlib import Path
import shlex
import shutil
import subprocess
import tempfile

from collect import write_json

MODELS = ("qwen3.6-35b-a3b", "qwen3.6-27b", "qwen3.8-27b")
IMAGE_PIN = Path(__file__).with_name("image.txt")


def model_spec(repo, name):
    directory = repo / "models" / name
    source = json.loads(subprocess.check_output([
        "nix", "eval", "--offline", "--json", "--file", str(directory / "source.nix"),
    ], text=True))
    return dict(name=name, **source,
                text_config=json.loads((directory / "model.json").read_text())["text_config"])


def capture_command(prototypes, run_dir, hub, cache, spec, image):
    name = spec["name"]
    model_repo = hub / ("models--" + spec["repo"].replace("/", "--"))
    container_run = Path("/bench") / run_dir.relative_to(prototypes)
    command = [
        "docker", "run", "--rm", "--gpus", "all", "--ipc=host", "--network=none",
        "-e", "VLLM_ENABLE_V1_MULTIPROCESSING=0", "-e", "VLLM_NO_USAGE_STATS=1",
        "-e", "VLLM_USE_V2_MODEL_RUNNER=0",
        "-e", f"QS3_REFERENCE_IMAGE={image}",
        "-e", "HF_HUB_OFFLINE=1", "-e", "TRANSFORMERS_OFFLINE=1",
        "-v", f"{model_repo}:/model-repo:ro", "-v", f"{prototypes}:/bench",
        "-v", f"{cache}:/root/.cache", "--entrypoint", "python3", image,
        "/bench/vllm_correctness/collect.py",
        "--model", f"/model-repo/snapshots/{spec['rev']}",
        "--model-spec", str(container_run / f"{name}.json"),
        "--output", str(container_run / "data" / name),
    ]
    if name == "qwen3.6-35b-a3b":
        command += ["--reference-dir", "/bench/vllm_correctness/references"]
    return command


def check_snapshot(hub, spec):
    snapshot = hub / ("models--" + spec["repo"].replace("/", "--")) / "snapshots" / spec["rev"]
    required = ["config.json", "generation_config.json", "tokenizer.json",
                "tokenizer_config.json", "model.safetensors.index.json"]
    index = snapshot / "model.safetensors.index.json"
    if index.is_file():
        required += sorted(set(json.loads(index.read_text())["weight_map"].values()))
    missing = [name for name in required if not (snapshot / name).is_file()]
    if missing:
        raise FileNotFoundError(
            f"{spec['name']}: missing snapshot files: {', '.join(missing)}\n"
            f"Download with: hf download {spec['repo']} --revision {spec['rev']}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models", nargs="+", choices=MODELS, default=list(MODELS))
    parser.add_argument("--image", default=IMAGE_PIN.read_text().strip(),
                        help="local Docker image; resolved once to its immutable image ID")
    parser.add_argument("--resume", type=Path,
                        help="existing run directory; retain completed models and archive failed attempts")
    parser.add_argument("--dry-run", action="store_true",
                        help="print pins and container commands without loading models or writing output")
    args = parser.parse_args()
    assert not (args.resume and args.dry_run), "--resume and --dry-run cannot be combined"
    assert len(args.models) == len(set(args.models)), "duplicate model selection"
    prototypes = Path(__file__).resolve().parents[1]
    specs = [model_spec(prototypes.parent, name) for name in args.models]
    hf_home = Path(os.environ.get("HF_HOME", Path.home() / ".cache/huggingface"))
    hub = Path(os.environ.get("HF_HUB_CACHE", hf_home / "hub")).expanduser().resolve()
    cache = Path.home() / "qs3-vllm-cache"
    if args.dry_run:
        for spec in specs:
            print(json.dumps(spec, indent=2))
            print(shlex.join(capture_command(prototypes, prototypes / "out/vllm-correctness-DRY-RUN",
                                             hub, cache, spec, args.image)))
        return

    # Fail before any GPU work if one of the requested snapshots is incomplete.
    for spec in specs:
        check_snapshot(hub, spec)
    inspected = json.loads(subprocess.check_output([
        "docker", "image", "inspect", args.image,
    ], text=True))
    assert len(inspected) == 1
    image_id = inspected[0]["Id"]
    assert image_id.startswith("sha256:"), "Docker did not return an immutable image ID"
    (prototypes / "out").mkdir(exist_ok=True)
    stamp = datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H%M%SZ")
    completed = []
    if args.resume:
        run_dir = args.resume.resolve(strict=True)
        run_dir.relative_to((prototypes / "out").resolve())
        plan = json.loads((run_dir / "plan.json").read_text())
        assert plan["models"] == args.models, "resume model selection/order changed"
        assert plan["image"] == image_id, "resume image ID changed"
        assert plan["model_runner"] == "v1", "resume model runner changed"
        for spec in specs:
            assert json.loads((run_dir / f"{spec['name']}.json").read_text()) == spec, (
                f"resume model pin/config changed: {spec['name']}")
        completed_path = run_dir / "completed.json"
        completed = json.loads(completed_path.read_text()) if completed_path.is_file() else []
        assert len(completed) == len(set(completed)) and set(completed) <= set(args.models)
        for name in completed:
            success = run_dir / "data" / name / "SUCCESS.json"
            assert success.is_file(), f"completed model has no success marker: {name}"
            assert json.loads(success.read_text())["workloads"] == ["4-4", "102-32", "500-200", "4000-800"]
        if (run_dir / "SUCCESS.json").is_file():
            assert completed == args.models, "root success disagrees with completed models"
            print(f"All selected models already completed: {run_dir}", flush=True)
            return
        history = run_dir / "resume-history"
        history.mkdir(exist_ok=True)
        attempt = Path(tempfile.mkdtemp(prefix=f"{stamp}-", dir=history))
        for source in (Path(__file__), Path(__file__).with_name("collect.py"), IMAGE_PIN):
            shutil.copy2(source, attempt / source.name)
        write_json(attempt / "resume.json", {"models": args.models, "image": image_id,
                                             "completed_before": completed, "run_dir": str(run_dir)})
        write_json(attempt / "image.json", inspected)
        # Move incomplete attempts intact before collector's exclusive mkdir.
        # Original run/plan and completed model source snapshots remain intact.
        for spec in specs:
            name = spec["name"]
            if name in completed:
                continue
            old = [run_dir / "data" / name, run_dir / f"{name}.log",
                   run_dir / f"{name}-command.json"]
            existing = [path for path in old if path.exists()]
            if existing:
                failed = run_dir / "failed" / attempt.name / name
                failed.mkdir(parents=True)
                # Containers own these directories. Renaming a directory across
                # parents also needs permission to update its '..' entry. Do
                # the atomic moves in the same image, without copy/delete
                # fallback or changing ownership of completed captures.
                moves = [[str(Path('/capture') / path.relative_to(run_dir)),
                          str(Path('/capture') / (failed / path.name).relative_to(run_dir))]
                         for path in existing]
                command = ["docker", "run", "--rm", "--network=none",
                           "-v", f"{run_dir}:/capture", "--entrypoint", "python3", image_id,
                           "-c", "import json,os,sys; moves=json.loads(sys.argv[1]); "
                           "assert all(not os.path.lexists(dst) for src,dst in moves); "
                           "[os.rename(src,dst) for src,dst in moves]", json.dumps(moves)]
                write_json(attempt / f"{name}-archive-command.json", command)
                subprocess.run(command, check=True)
    else:
        run_dir = Path(tempfile.mkdtemp(prefix=f"vllm-correctness-{stamp}-", dir=prototypes / "out"))
        attempt = run_dir
        shutil.copy2(__file__, run_dir / "run.py")
        for spec in specs:
            write_json(run_dir / f"{spec['name']}.json", spec)
        write_json(run_dir / "plan.json", {"models": args.models, "requested_image": args.image,
                                         "image": image_id, "model_runner": "v1"})
        write_json(run_dir / "image.json", inspected)
    (run_dir / "data").mkdir(exist_ok=True)
    print(f"vLLM correctness capture: {run_dir}", flush=True)
    with (attempt / "gpu.csv").open("w") as output:
        subprocess.run(["nvidia-smi", "--query-gpu=name,driver_version", "--format=csv"],
                       stdout=output, check=True)
    cache.mkdir(exist_ok=True)
    for spec in specs:
        name = spec["name"]
        if name in completed:
            print(f"Retaining completed {name}", flush=True)
            continue
        command = capture_command(prototypes, run_dir, hub, cache, spec, image_id)
        write_json(run_dir / f"{name}-command.json", command)
        log = run_dir / f"{name}.log"
        print(f"Capturing {name}; log: {log}", flush=True)
        with log.open("w") as output:
            subprocess.run(command, stdout=output, stderr=subprocess.STDOUT, check=True)
        assert (run_dir / "data" / name / "SUCCESS.json").is_file()
        completed.append(name)
        write_json(run_dir / "completed.json", completed)
    write_json(run_dir / "SUCCESS.json", {"models": completed})
    print(f"vLLM correctness data: {run_dir / 'data'}", flush=True)


if __name__ == "__main__":
    main()
