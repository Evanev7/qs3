"""Measure the 3.8-27B BF16 baseline; no inference tuning or logit capture."""
import datetime
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile

PROTOTYPES = Path(__file__).resolve().parents[1]
REPO = PROTOTYPES.parent
MODEL = 'qwen3.8-27b'
REVISION = '1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0'
WORKLOADS = ((102, 32), (1024, 256))


def save(path, value):
    path.write_text(json.dumps(value, indent=2) + '\n')


def run(command, *, stdout, stderr=None, env=None, cwd=REPO):
    print('Running:', ' '.join(map(str, command)), flush=True)
    with stdout.open('w') as out:
        if stderr is None:
            subprocess.run(command, cwd=cwd, env=env, stdout=out,
                           stderr=subprocess.STDOUT, check=True)
        else:
            with stderr.open('w') as err:
                subprocess.run(command, cwd=cwd, env=env, stdout=out, stderr=err, check=True)


def main():
    config = json.loads(subprocess.check_output([
        'nix', 'eval', '--offline', '--json', '--file', 'models/config.nix',
    ], cwd=REPO))
    assert config['engine']['model'] == MODEL
    model_repo = Path.home() / '.cache/huggingface/hub/models--Qwen--Qwen3.8-27B'
    snapshot = model_repo / 'snapshots' / REVISION
    assert (snapshot / 'model.safetensors.index.json').is_file()
    image_pin = (PROTOTYPES / 'vllm_correctness/image.txt').read_text().strip()
    inspection = json.loads(subprocess.check_output(['docker', 'image', 'inspect', image_pin]))
    image = inspection[0]['Id']
    stamp = datetime.datetime.now(datetime.timezone.utc).strftime('%Y-%m-%dT%H%M%SZ')
    output = Path(tempfile.mkdtemp(prefix=f'performance-38-27b-{stamp}-', dir=PROTOTYPES / 'out'))
    print(f'Performance baseline: {output}', flush=True)
    save(output / 'config.json', config)
    save(output / 'image.json', inspection)
    save(output / 'metadata.json', {
        'model': MODEL, 'revision': REVISION, 'image_pin': image_pin,
        'image_id': image, 'workloads': WORKLOADS,
        'commit': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=REPO, text=True).strip(),
        'protocol': 'unprofiled wall-clock baseline; independent greedy continuations; two prefill warmups, five samples; four decode warmups',
        'qs3_mode': 'eager', 'vllm_mode': 'default CUDA graphs; model runner v1',
    })
    (output / 'source.diff').write_bytes(subprocess.check_output(['git', 'diff', 'HEAD'], cwd=REPO))
    shutil.copy2(__file__, output / 'run.py')
    shutil.copy2(PROTOTYPES / 'vllm_core_benchmark.py', output / 'vllm_core_benchmark.py')
    shutil.copy2(REPO / 'src/loader/benchmark.rs', output / 'benchmark.rs')
    run(['nvidia-smi', '--query-gpu=name,driver_version', '--format=csv'], stdout=output / 'gpu.csv')
    run(['just', 'ninja'], stdout=output / 'configure.log')
    run(['ninja', '-C', 'build'], stdout=output / 'native-build.log')
    run(['cargo', 'build', '--release', '--bin', 'qs3-bench'], stdout=output / 'rust-build.log')
    run(['cargo', 'test', '--release', '--lib', 'loader::benchmark::tests', '--', '--nocapture'],
        stdout=output / 'contract-tests.log')
    cache = Path.home() / 'qs3-vllm-cache'
    cache.mkdir(exist_ok=True)
    completed = []
    for context, samples in WORKLOADS:
        case = output / f'{context}-{samples}'
        case.mkdir()
        env = dict(os.environ, QS3_QWEN36_MODEL_DIR=str(snapshot),
                   QS3_BENCH_CONTEXT_TOKENS=str(context), QS3_BENCH_DECODE_SAMPLES=str(samples))
        env.pop('QS3_PROFILE', None)
        started = datetime.datetime.now(datetime.timezone.utc).isoformat()
        run([str(REPO / 'target/release/qs3-bench'), '--measure-pass'],
            stdout=case / 'qs3.json', stderr=case / 'qs3.log', env=env)
        container_case = Path('/bench') / case.relative_to(PROTOTYPES)
        command = [
            'docker', 'run', '--rm', '--gpus', 'all', '--ipc=host', '--network=none',
            '-e', 'VLLM_ENABLE_V1_MULTIPROCESSING=0', '-e', 'VLLM_USE_V2_MODEL_RUNNER=0',
            '-e', 'VLLM_NO_USAGE_STATS=1', '-e', 'HF_HUB_OFFLINE=1', '-e', 'TRANSFORMERS_OFFLINE=1',
            '-v', f'{model_repo}:/model-repo:ro', '-v', f'{PROTOTYPES}:/bench',
            '-v', f'{cache}:/root/.cache', '--entrypoint', 'python3', image,
            '/bench/vllm_core_benchmark.py', '--model', f'/model-repo/snapshots/{REVISION}',
            '--context', str(context), '--samples', str(samples),
            '--output', str(container_case / 'vllm.json'),
        ]
        save(case / 'vllm-command.json', command)
        run(command, stdout=case / 'vllm.log')
        q = json.loads((case / 'qs3.json').read_text())['measurement']
        v = json.loads((case / 'vllm.json').read_text())
        assert q['model_name'] == MODEL
        assert Path(q['model']).name == Path(v['model']).name == REVISION
        assert q['prompt']['token_id_fnv1a'] == v['prompt']['token_id_fnv1a']
        assert q['decode']['samples'] == v['decode']['samples'] == samples
        assert q['decode']['context_start'] == v['decode']['context_start'] == context + 4
        assert q['decode']['context_end'] == v['decode']['context_end'] == context + 4 + samples
        assert q['execution']['gdn_recurrent_state_dtype'] == 'f32'
        assert v['execution']['resolved_gdn_state_dtypes'] == ['torch.bfloat16', 'torch.float32']
        qids = q['decode']['generated_token_ids']
        vids = v['generated_tokens'][:-1]
        assert len(qids) == len(vids) == samples + 4
        first_difference = next((i for i, (x,y) in enumerate(zip(qids, vids)) if x != y), None)
        summary = {
            'started_utc': started,
            'finished_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
            'context': context, 'decode_samples': samples,
            'prompt_fingerprint': q['prompt']['token_id_fnv1a'],
            'first_generated_token_difference': first_difference,
            'matching_generated_tokens': sum(x == y for x,y in zip(qids, vids)),
            'qs3': {'prefill_p50_ms': q['prefill']['p50_ms'], 'decode_p50_ms': q['decode']['p50_ms'],
                    'decode_tok_s': q['decode']['tokens_per_second']},
            'vllm': {'prefill_p50_ms': v['prefill']['p50_ms'], 'decode_p50_ms': v['decode']['p50_ms'],
                     'decode_tok_s': v['decode']['tokens_per_second']},
        }
        save(case / 'comparison.json', summary)
        print(json.dumps(summary, indent=2), flush=True)
        completed.append(case.name)
        save(output / 'completed.json', completed)
    save(output / 'SUCCESS.json', {'workloads': completed})
    print(f'Completed performance baseline: {output}', flush=True)


if __name__ == '__main__':
    main()
