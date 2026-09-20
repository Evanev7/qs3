"""Measure the integrated working tree using the core benchmark's Nix libraries."""
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
from datetime import datetime, timezone

ROOT=Path(__file__).resolve().parents[2]
REV='dbb8f445b3145f8a4c18ddc769f032d57d32867c'

def main():
    output=Path(tempfile.mkdtemp(prefix='nvfp4-runtime-',dir=ROOT/'.prototypes/out'))
    print(output,flush=True)
    env=dict(os.environ,LIBRARY_PATH='/usr/local/cuda/lib64:'+os.environ.get('LIBRARY_PATH',''))
    binary=ROOT/'target/release/qs3-bench'
    with (output/'build.log').open('w') as log:
        subprocess.run(['cargo','build','--release','--bin','qs3-bench'],cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT,check=True)
    env.pop('LD_LIBRARY_PATH',None)
    reference=max(Path('/nix/store').glob('*-qs3-benchmark-0.1.0/bin/qs3-bench'),key=lambda p:p.stat().st_mtime)
    libs=subprocess.check_output(['ldd',str(reference)],env=env,text=True)
    interpreter=subprocess.check_output(['readelf','-l',str(reference)],text=True).split('Requesting program interpreter: ')[1].split(']')[0]
    dirs=list(dict.fromkeys(str(Path(line.split(' => ')[1].split()[0]).parent) for line in libs.splitlines() if ' => /nix/store/' in line))
    loader=[interpreter,'--library-path',':'.join(dirs+['/lib/aarch64-linux-gnu'])]
    (output/'libraries.txt').write_text(str(reference)+'\n'+subprocess.check_output(loader+['--list',str(binary)],env=env,text=True))
    sources=['cute_kernels/nvfp4_gemm.py','src/backend/qscute/nvfp4.rs','src/backend/qscute.rs','src/model/runner/linear/mod.rs','src/model/runner/mod.rs','models/config.nix','build/libqscute.a','build/libqs_native.a']
    metadata={'started_at':datetime.now(timezone.utc).isoformat(),
        'commit':subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
        'working_tree':True,'model_revision':REV,'binary_sha256':hashlib.sha256(binary.read_bytes()).hexdigest(),
        'sha256':{p:hashlib.sha256((ROOT/p).read_bytes()).hexdigest() for p in sources},
        'protocol':'core benchmark --measure-pass; Nix CUDA libraries; no profiler'}
    (output/'metadata.json').write_text(json.dumps(metadata,indent=2)+'\n')
    for source in sources:
        if source.startswith('build/'):
            continue
        target=output/'source'/source
        target.parent.mkdir(parents=True,exist_ok=True)
        shutil.copy2(ROOT/source,target)
    (output/'gpu.csv').write_bytes(subprocess.check_output(['nvidia-smi','--query-gpu=name,uuid,driver_version,pstate,temperature.gpu,clocks.sm,clocks.mem,power.draw','--format=csv']))
    (output/'source.diff').write_bytes(subprocess.check_output(['git','diff','HEAD','--','src','models','build.rs','build_tools','cute_kernels'],cwd=ROOT))
    shutil.copy2(__file__,output/'runtime_benchmark.py')
    for context,samples in [(102,32),(1024,256)]:
        env.update(QS3_QWEN36_MODEL_DIR=str(Path.home()/'.cache/huggingface/hub/models--nvidia--Qwen3.8-27B-NVFP4/snapshots'/REV),QS3_BENCH_CONTEXT_TOKENS=str(context),QS3_BENCH_DECODE_SAMPLES=str(samples))
        env.pop('QS3_PROFILE',None)
        with (output/f'{context}-{samples}.json').open('w') as out, (output/f'{context}-{samples}.log').open('w') as log:
            subprocess.run(loader+[str(binary),'--measure-pass'],cwd=ROOT,env=env,stdout=out,stderr=log,check=True)
        d=json.loads((output/f'{context}-{samples}.json').read_text())['measurement']
        assert d['execution']['nvfp4']['decode_tactic']=='cute_32x64x512'
        assert d['execution']['lm_head']=='cute-nvfp4'
        print(json.dumps({'context':context,'prefill_ms':d['prefill']['p50_ms'],'decode_ms':d['decode']['p50_ms'],'tok_s':d['decode']['tokens_per_second']}),flush=True)
    (output/'SUCCESS.json').write_text('{"workloads":["102-32","1024-256"]}\n')

if __name__=='__main__':main()
