"""Verify all three completed captures and emit a compact pinned-reference summary.

Only standard-library JSON, bookkeeping, file sizes, and SHA256 are checked.
This does not rerun inference, recompute top-k, or establish model correctness.
"""

import argparse
import datetime
import hashlib
import json
from pathlib import Path
import shutil


MODELS = ("qwen3.6-35b-a3b", "qwen3.6-27b", "qwen3.8-27b")
WORKLOADS = {"4-4": (4, 4), "102-32": (102, 37),
             "500-200": (500, 205), "4000-800": (4000, 805)}
VOCAB = 248320
ROW_BYTES = VOCAB * 4
TOKENIZER_VOCABS = {"qwen3.6-35b-a3b": 248070, "qwen3.6-27b": 248070,
                    "qwen3.8-27b": 248077}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def canonical_sha(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     allow_nan=False).encode()).hexdigest()


def verify(run_dir, *, export_dir=None):
    manifest = {}

    def hashed(path):
        require(path.is_file(), f"missing file: {path}")
        with path.open("rb") as source:
            digest = hashlib.file_digest(source, "sha256").hexdigest()
        manifest[str(path.relative_to(run_dir))] = {"bytes": path.stat().st_size,
                                                  "sha256": digest}
        return digest

    def read(path):
        hashed(path)
        return json.loads(path.read_text())

    def exact(values, expected, label):
        require(len(values) == len(expected) and set(values) == set(expected),
                f"{label}: expected exactly {list(expected)}, got {values}")

    plan = read(run_dir / "plan.json")
    hashed(run_dir / "run.py")
    exact(plan["models"], MODELS, "capture plan models")
    exact(read(run_dir / "SUCCESS.json")["models"], MODELS, "capture SUCCESS models")
    exact(read(run_dir / "completed.json"), MODELS, "capture completed models")
    exact([p.name for p in (run_dir / "data").iterdir() if p.is_dir()], MODELS,
          "capture data directories")
    image = plan["image"]
    require(image.startswith("sha256:") and len(image) == 71, "plan image is not an immutable Docker image ID")
    inspected = read(run_dir / "image.json")
    require(len(inspected) == 1 and inspected[0]["Id"] == image, "image inspection does not match plan")
    # Completed models may have different collector revisions after a resume.
    # Preserve every recorded attempt, while model-local collector.py remains
    # authoritative for the rows that actually completed.
    history = run_dir / "resume-history"
    if history.is_dir():
        for attempt in sorted(history.iterdir()):
            require(attempt.is_dir(), f"unexpected resume history entry: {attempt}")
            resumed = read(attempt / "resume.json")
            require(resumed["image"] == image and resumed["models"] == plan["models"],
                    f"{attempt.name}: resume image/model selection mismatch")
            prior = resumed["completed_before"]
            require(len(prior) == len(set(prior)) and set(prior) <= set(MODELS),
                    f"{attempt.name}: invalid prior completed list")
            resume_image = read(attempt / "image.json")
            require(len(resume_image) == 1 and resume_image[0]["Id"] == image,
                    f"{attempt.name}: resume image inspection mismatch")
            for filename in ("run.py", "collect.py", "image.txt"):
                hashed(attempt / filename)
            for path in attempt.iterdir():
                if path.is_file():
                    hashed(path)
    if (run_dir / "gpu.csv").is_file():
        hashed(run_dir / "gpu.csv")
    models = {}
    total_rows = 0
    for name in MODELS:
        tokenizer_vocab = TOKENIZER_VOCABS[name]
        model_dir = run_dir / "data" / name
        pin_path = run_dir / f"{name}.json"
        pin = read(pin_path)
        command = read(run_dir / f"{name}-command.json")
        require(command[:2] == ["docker", "run"] and command.count(image) == 1,
                f"{name}: capture command image mismatch")
        for env in ("VLLM_USE_V2_MODEL_RUNNER=0", "VLLM_ENABLE_V1_MULTIPROCESSING=0",
                    f"QS3_REFERENCE_IMAGE={image}"):
            require(any(command[i:i + 2] == ["-e", env] for i in range(len(command))),
                    f"{name}: command is missing {env}")
        require(command[command.index("--model") + 1] == f"/model-repo/snapshots/{pin['rev']}",
                f"{name}: capture command snapshot mismatch")
        hashed(model_dir / "collector.py")
        hashed(model_dir / "base_ids.json")
        require(pin["name"] == name, f"{name}: pin name mismatch")
        require(len(pin["rev"]) == 40 and all(c in "0123456789abcdef" for c in pin["rev"]),
                f"{name}: model revision must be a full commit")
        require(read(model_dir / "model-spec.json") == pin, f"{name}: collector pin differs from run pin")
        exact(read(model_dir / "SUCCESS.json")["workloads"], WORKLOADS, f"{name} SUCCESS workloads")
        exact(read(model_dir / "completed.json"), WORKLOADS, f"{name} completed workloads")
        exact([p.name for p in model_dir.iterdir() if p.is_dir()], WORKLOADS, f"{name} workload directories")
        model_plan = read(model_dir / "plan.json")
        exact([w["name"] for w in model_plan], WORKLOADS, f"{name} planned workloads")
        planned = {w["name"]: w for w in model_plan}
        metadata_path = model_dir / "metadata.json"
        metadata = read(metadata_path)
        for field, expected in (("model_name", name), ("model_repo", pin["repo"]),
                                ("model_revision", pin["rev"]), ("container_image", image)):
            require(metadata[field] == expected, f"{name}: metadata {field} mismatch")
        require(Path(metadata["model"]).name == pin["rev"], f"{name}: snapshot path revision mismatch")
        require(metadata["logits_format"] == {"dtype": "little-endian float32", "shape": [VOCAB],
                                              "addressable_token_count": tokenizer_vocab}, f"{name}: unexpected logits format")
        require(metadata["model_runner"] == plan["model_runner"] == "v1", f"{name}: runner mismatch")
        require(metadata["capture_hook"] == "model.compute_logits", f"{name}: capture hook mismatch")
        require(metadata["resolved_gdn_state_dtypes"] == ["torch.bfloat16", "torch.float32"],
                f"{name}: GDN state precision mismatch")
        for asset, digest in metadata["asset_sha256"].items():
            require(Path(asset).name == asset, f"{name}: invalid asset filename")
            require(hashed(model_dir / asset) == digest, f"{name}: asset hash mismatch: {asset}")
        tokenizer = read(model_dir / "tokenizer.json")
        token_ids = set(tokenizer["model"]["vocab"].values())
        token_ids.update(token["id"] for token in tokenizer["added_tokens"])
        require(token_ids == set(range(tokenizer_vocab)), f"{name}: tokenizer addressability differs from pin")
        require(read(model_dir / "config.json")["text_config"] == pin["text_config"],
                f"{name}: config text geometry differs from pin")
        exact(metadata["asset_sha256"], ("config.json", "generation_config.json", "tokenizer.json",
                                         "tokenizer_config.json"), f"{name}: captured assets")
        small = read(model_dir / "small-reference.json")
        for field in ("model_name", "model_repo", "model_revision", "container_image", "versions"):
            require(small[field] == metadata[field], f"{name}: small reference {field} mismatch")
        require(small["prompt_ids"] == [1, 2, 3, 4], f"{name}: unexpected small prompt")
        workloads = {}
        for workload, (context, count) in WORKLOADS.items():
            directory = model_dir / workload
            scores_path = directory / "vllm" / "scores.json"
            scores = read(scores_path)
            input_path = directory / "input.json"
            inputs = read(input_path)
            require(scores["input"] == inputs, f"{name}/{workload}: input mismatch")
            require(scores["workload"] == planned[workload], f"{name}/{workload}: workload differs from plan")
            require(planned[workload]["context"] == context and planned[workload]["decode_steps"] + 1 == count,
                    f"{name}/{workload}: unexpected context or decode count")
            for field in ("model_name", "model_repo", "model_revision"):
                require(inputs[field] == metadata[field], f"{name}/{workload}: input {field} mismatch")
            require(inputs["prompt_ids"] == planned[workload]["prompt_ids"] and len(inputs["prompt_ids"]) == context,
                    f"{name}/{workload}: prompt mismatch")
            records, generated = scores["records"], scores["generated_ids"]
            require(len(records) == len(generated) == count, f"{name}/{workload}: expected {count} rows")
            require(all(type(token) is int and 0 <= token < tokenizer_vocab for token in generated),
                    f"{name}/{workload}: invalid generated token ID")
            require(inputs["capture_steps"] == list(range(count)), f"{name}/{workload}: capture steps mismatch")
            require(inputs["forced_ids"] == generated[:-1], f"{name}/{workload}: replay continuation mismatch")
            forced = planned[workload]["forced_ids"]
            if forced is not None:
                require(name == "qwen3.6-35b-a3b" and workload in ("102-32", "500-200"),
                        f"{name}/{workload}: unexpected forced source")
                require(forced == generated[:-1], f"{name}/{workload}: saved source continuation mismatch")
                source_path = model_dir / f"source-{workload}.json"
                require(hashed(source_path) == planned[workload]["source_sha256"],
                        f"{name}/{workload}: source reference hash mismatch")
                source = read(source_path)
                require(source["generated_tokens"] == planned[workload]["reference_generated_ids"],
                        f"{name}/{workload}: original reference IDs mismatch")
                require(source["generated_tokens"][:-1] == forced,
                        f"{name}/{workload}: forced IDs differ from original reference")
            require(generated == [r["selected_id"] for r in records], f"{name}/{workload}: generated IDs mismatch")
            journal = directory / "vllm" / "records.jsonl"
            hashed(journal)
            require([json.loads(line) for line in journal.read_text().splitlines()] == records,
                    f"{name}/{workload}: journal differs from scores")
            row_manifest = []
            for step, record in enumerate(records):
                label = f"{name}/{workload}/{step}"
                require(record["step"] == step and record["context_tokens"] == context + step, f"{label}: position mismatch")
                require(record["phase"] == ("prefill" if step == 0 else "decode"), f"{label}: phase mismatch")
                require(type(record["argmax"]) is int and 0 <= record["argmax"] < tokenizer_vocab,
                        f"{label}: invalid raw argmax")
                if forced is None or step == count - 1:
                    require(record["selected_id"] == record["argmax"], f"{label}: greedy selection mismatch")
                if step < count - 1:
                    require(record["forced_id"] == generated[step], f"{label}: forced record mismatch")
                require(record["file"] == f"logits-{step:04d}.f32", f"{label}: filename mismatch")
                path = directory / "vllm" / record["file"]
                require(path.stat().st_size == ROW_BYTES, f"{label}: expected {ROW_BYTES} bytes")
                require(hashed(path) == record["sha256"], f"{label}: logits hash mismatch")
                row_manifest.append({"file": str(path.relative_to(run_dir)), "sha256": record["sha256"], "bytes": ROW_BYTES})
            exact([p.name for p in (directory / "vllm").glob("*.f32")],
                  [r["file"] for r in records], f"{name}/{workload} raw row files")
            if workload == "4-4":
                require(small["generated_ids"] == generated and small["forced_ids"] == generated[:-1],
                        f"{name}: small reference token mismatch")
                require(len(small["rows"]) == count, f"{name}: small reference row count mismatch")
                for small_row, row in zip(small["rows"], records):
                    expected_small = {key: row[key] for key in
                                      ("step", "phase", "argmax", "top2_margin", "logits_dtype", "sha256")}
                    expected_small.update(top_ids=row["top_ids"][:8], top_values=row["top_values"][:8],
                                          file=f"4-4/vllm/{row['file']}")
                    require(small_row == expected_small, f"{name}: small reference row differs from scores")
            workloads[workload] = {"rows": count, "bytes": count * ROW_BYTES,
                                   "rows_manifest_sha256": canonical_sha(row_manifest),
                                   "scores": str(scores_path.relative_to(run_dir)),
                                   "scores_sha256": manifest[str(scores_path.relative_to(run_dir))]["sha256"],
                                   "input": str(input_path.relative_to(run_dir))}
            total_rows += count
        models[name] = {"repo": pin["repo"], "revision": pin["rev"], "image": image,
                        "versions": metadata["versions"], "small_generated_ids": small["generated_ids"],
                        "small_reference": str((model_dir / "small-reference.json").relative_to(run_dir)),
                        "metadata": str(metadata_path.relative_to(run_dir)),
                        "metadata_sha256": manifest[str(metadata_path.relative_to(run_dir))]["sha256"],
                        "pin_sha256": manifest[str(pin_path.relative_to(run_dir))]["sha256"],
                        "workloads": workloads}
    summary = {"verified": True, "run_dir": str(run_dir), "image": image,
            "verified_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
            "rows": total_rows, "logits_bytes": total_rows * ROW_BYTES,
            "manifest_sha256": canonical_sha(manifest), "manifest_files": len(manifest),
            "manifest_definition": "SHA256 of sorted-key compact JSON mapping verified run-relative filenames to bytes and SHA256",
            "plan_sha256": manifest["plan.json"]["sha256"], "models": models}
    if export_dir is not None:
        export_dir = export_dir.resolve()
        require(not export_dir.exists(), f"export directory already exists: {export_dir}")
        require(not export_dir.is_relative_to(run_dir), "compact export must be outside the raw run")
        export_dir.mkdir(parents=True)
        omitted = []
        for relative in sorted(manifest):
            path = Path(relative)
            if path.suffix == ".f32" or path.name == "tokenizer.json":
                omitted.append(relative)
                continue
            destination = export_dir / path
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(run_dir / path, destination)
        (export_dir / "verified-files.json").write_text(json.dumps(manifest, indent=2) + "\n")
        (export_dir / "verification.json").write_text(json.dumps(summary, indent=2) + "\n")
        (export_dir / "export.json").write_text(json.dumps({
            "source_run": str(run_dir), "omitted_files": omitted,
            "omission_reason": "Raw FP32 rows and tokenizer payload remain in the original artifact; their hashes and sizes are retained.",
            "verification": "Full original capture passed before export. This compact directory is not a complete capture and cannot be reverified without omitted files.",
        }, indent=2) + "\n")
        shutil.copy2(__file__, export_dir / "verifier.py")
        (export_dir / ".gitignore").write_text("*.f32\ntokenizer.json\n")
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--export-dir", type=Path,
                        help="new compact artifact directory outside the raw run; excludes raw rows/tokenizer payload")
    args = parser.parse_args()
    try:
        summary = verify(args.run_dir.resolve(), export_dir=args.export_dir)
    except (ValueError, KeyError, OSError, TypeError) as error:
        parser.exit(1, f"capture verification failed: {error}\n")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n")
    print(json.dumps({"verified": True, "models": len(summary["models"]), "rows": summary["rows"],
                      "logits_bytes": summary["logits_bytes"], "manifest_sha256": summary["manifest_sha256"],
                      "output": str(args.output.resolve())}))


if __name__ == "__main__":
    main()
