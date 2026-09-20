"""Run sequential stock vLLM no-MTP baselines on the GPU host."""
import datetime
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import tempfile

ROOT = Path(__file__).resolve().parents[2]
REVISION = 'dbb8f445b3145f8a4c18ddc769f032d57d32867c'
MODEL_REPO = Path.home() / '.cache/huggingface/hub/models--nvidia--Qwen3.8-27B-NVFP4'
IMAGE = (ROOT / '.prototypes/vllm_correctness/image.txt').read_text().strip()

def save(path, data):
    path.write_text(json.dumps(data, indent=2) + '\n')

def main():
    snapshot = MODEL_REPO / 'snapshots' / REVISION
    assert (snapshot / 'model.safetensors.index.json').is_file()
    stamp = datetime.datetime.now(datetime.timezone.utc).strftime('%Y-%m-%dT%H%M%SZ')
    output = Path(tempfile.mkdtemp(prefix=f'vllm-nvfp4-nomtp-{stamp}-', dir=ROOT / '.prototypes/out'))
    print(output, flush=True)
    inspection = json.loads(subprocess.check_output(['docker', 'image', 'inspect', IMAGE]))
    save(output / 'image.json', inspection)
    for name in ('run.py', 'benchmark.py'):
        shutil.copy2(Path(__file__).with_name(name), output / name)
    raw = (snapshot / 'config.json').read_bytes()
    (output / 'checkpoint-config.json').write_bytes(raw)
    save(output / 'metadata.json', {
        'model': 'nvidia/Qwen3.8-27B-NVFP4', 'revision': REVISION,
        'config_sha256': hashlib.sha256(raw).hexdigest(), 'image_pin': IMAGE,
        'image_id': inspection[0]['Id'],
        'qs3_commit': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
        'workloads': [[102,32], [1024,256]],
        'protocol': 'batch 1 greedy; 2 prefill warmups + 5 samples; 4 decode warmups; independent continuations',
        'vllm_mode': 'stock image, default CUDA graphs, v1 runner, speculative_config=None',
    })
    (output / 'gpu.csv').write_bytes(subprocess.check_output(['nvidia-smi', '--query-gpu=name,driver_version,memory.total', '--format=csv']))
    cache = Path.home() / 'qs3-vllm-cache'
    cache.mkdir(exist_ok=True)
    completed = []
    for context, samples in ((102,32), (1024,256)):
        case = output / f'{context}-{samples}'
        case.mkdir()
        container_output = Path('/bench') / output.relative_to(ROOT / '.prototypes')
        command = ['docker', 'run', '--rm', '--gpus', 'all', '--ipc=host', '--network=none',
            '-e', 'VLLM_ENABLE_V1_MULTIPROCESSING=0', '-e', 'VLLM_USE_V2_MODEL_RUNNER=0',
            '-e', 'VLLM_NO_USAGE_STATS=1', '-e', 'HF_HUB_OFFLINE=1', '-e', 'TRANSFORMERS_OFFLINE=1',
            '-v', f'{MODEL_REPO}:/model-repo:ro', '-v', f'{ROOT / ".prototypes"}:/bench',
            '-v', f'{cache}:/root/.cache', '--entrypoint', 'python3', inspection[0]['Id'],
            str(container_output / 'benchmark.py'), '--model', f'/model-repo/snapshots/{REVISION}',
            '--context', str(context), '--samples', str(samples), '--output', str(container_output / case.name / 'vllm.json')]
        save(case / 'command.json', command)
        print(f'Starting {case.name}', flush=True)
        with (case / 'vllm.log').open('w') as log:
            subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, check=True)
        data = json.loads((case / 'vllm.json').read_text())
        assert data['prompt']['token_id_fnv1a'] == {102:'6ca602a4cc15238d',1024:'c8cd97958d34675c'}[context]
        completed.append(case.name)
        save(output / 'completed.json', completed)
        print(json.dumps({'case':case.name,'prefill_p50_ms':data['prefill']['p50_ms'], 'decode_p50_ms':data['decode']['p50_ms'], 'tok_s':data['decode']['tokens_per_second']}), flush=True)
    save(output / 'SUCCESS.json', {'workloads':completed})

if __name__ == '__main__':
    main()
