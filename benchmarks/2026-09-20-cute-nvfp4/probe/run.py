"""Build a disposable Rust probe overlay; use --run on the GPU host."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import tempfile

ROOT = Path(__file__).resolve().parents[2]

def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--artifacts', type=Path, default=ROOT / '.prototypes/out/nvfp4-decode-aot')
    p.add_argument('--cases', nargs='+')
    p.add_argument('--run', action='store_true')
    p.add_argument('--model-benchmark', choices=['baseline','cute'])
    p.add_argument('--context', type=int, default=102)
    p.add_argument('--samples', type=int, default=32)
    p.add_argument('--real', action='store_true', help='Compare real checkpoint intermediates, retaining CUTLASS outputs')
    args = p.parse_args()
    if args.model_benchmark:
        args.real=True
    if args.real:
        assert args.run
        args.cases=args.cases or ['up_r0','down_r0','head_r0']
    artifacts = args.artifacts.resolve()
    data = json.loads((artifacts / 'cases.json').read_text())
    cases = [c for c in data['cases'] if args.cases is None or c['name'] in args.cases]
    assert cases and (args.cases is None or len(cases) == len(set(args.cases)))
    output = Path(tempfile.mkdtemp(prefix='nvfp4-decode-run-', dir=ROOT / '.prototypes/out'))
    checkout = output / 'checkout'
    checkout.mkdir()
    for source in ROOT.iterdir():
        if source.name in {'.git', '.prototypes', 'target', 'src', 'build.rs'}:
            continue
        (checkout / source.name).symlink_to(source, target_is_directory=source.is_dir())
    shutil.copytree(ROOT / 'src', checkout / 'src')
    build = (ROOT / 'build.rs').read_text().replace('fn main() {', 'fn main() {\n' +
        f'    println!("cargo:rustc-link-search=native={output}");\n' +
        '    println!("cargo:rustc-link-lib=static=nvfp4_candidates");', 1)
    (checkout / 'build.rs').write_text(build)
    subprocess.run(['ar', 'rcs', str(output / 'libnvfp4_candidates.a'),
                    *(str(artifacts / (c['name'] + '.o')) for c in cases)], check=True)
    source = Path(__file__).with_name('harness.rs').read_text()
    source += f'\nmod reduction {{ include!({json.dumps(str(artifacts / "reduction.rs"))}); }}\n'
    for c in cases:
        source += f'mod {c["name"]} {{ include!({json.dumps(str(artifacts / (c["name"] + ".rs")))}); }}\n'
    source += 'fn load_candidates() -> Vec<Candidate> {\nvec![\n'
    for c in cases:
        dtype = 'F32' if c['splits'] == 2 else 'BF16'
        pointer = 'partials' if c['splits'] == 2 else 'out'
        source += f'''{{ let kernel = unsafe {{ {c['name']}::Kernel::load().unwrap() }};
        Candidate {{ name: "{c['name']}", n: {c['n']}, k: {c['k']}, splits: {c['splits']},
          run: Box::new(move |a| unsafe {{ kernel.launch(
            DevicePtr::<U8>::new(a.a).unwrap(), DevicePtr::<U8>::new(a.b).unwrap(),
            DevicePtr::<Fp8E4M3>::new(a.sfa).unwrap(), DevicePtr::<Fp8E4M3>::new(a.sfb).unwrap(),
            DevicePtr::<{dtype}>::new(a.{pointer}).unwrap(), DevicePtr::<F32>::new(a.alpha).unwrap(), a.m, a.stream) }}) }} }},
'''
    source += ']\n}\n'
    if args.real:
        source += Path(__file__).with_name('real.rs').read_text()
        linear = checkout / 'src/model/runner/linear/mod.rs'
        text = linear.read_text().replace('pub(super) struct QuantizedScratch {', 'pub(super) struct QuantizedScratch {\n    #[cfg(test)] probe: crate::backend::nvfp4_decode_probe::Probe,')
        text = text.replace('            logits: DeviceBuffer::with_capacity(ctx, vocab as usize)?,', '            logits: DeviceBuffer::with_capacity(ctx.clone(), vocab as usize)?,\n            #[cfg(test)] probe: crate::backend::nvfp4_decode_probe::Probe::new(ctx),')
        needle = '                        scratch.workspace.workspace(scratch.workspace.len())?,\n                    )'
        assert text.count(needle)==1
        text=text.replace(needle,needle+'?;\n                    #[cfg(test)] if scratch.probe.compare_enabled { scratch.probe.compare(x, weight, [x_scales, block_scales], alpha, output); }\n                    Ok(())')
        needle='                    self.qsfi().nvfp4_execute('
        assert text.count(needle)==1
        text=text.replace(needle, '''                    #[cfg(test)] if scratch.probe.enabled && m<=16 {
                        scratch.probe.launch(x,weight,[x_scales,block_scales],alpha,output);
                        return Ok(());
                    }
'''+needle)
        linear.write_text(text)
    (checkout / 'src/backend/nvfp4_decode_probe.rs').write_text(source)
    with (checkout / 'src/backend/mod.rs').open('a') as f:
        f.write('\n#[cfg(test)] pub(crate) mod nvfp4_decode_probe;\n')
    metadata = {'donor':data['donor'], 'sm_count':data['sm_count'], 'cases':cases,
        'real_probe':args.real,'model_benchmark':args.model_benchmark,
        'commit':subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
        'hashes':{str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest()
            for p in [*Path(__file__).parent.glob('*.py'),*Path(__file__).parent.glob('*.rs'),*(artifacts / (c['name']+ext) for c in cases for ext in ['.o','.rs','.json'])]}}
    (output / 'metadata.json').write_text(json.dumps(metadata, indent=2)+'\n')
    if args.run:
        (output / 'gpu.csv').write_bytes(subprocess.check_output(['nvidia-smi','--query-gpu=name,driver_version,memory.total','--format=csv']))
    print(output, flush=True)
    env = dict(os.environ, CARGO_TARGET_DIR=str(ROOT / 'target'), NVFP4_OUTPUT=str(output))
    if args.run:
        for key in ['LIBRARY_PATH','LD_LIBRARY_PATH']:
            env[key] = '/usr/local/cuda/lib64' + (':' + env[key] if env.get(key) else '')
    if args.real:
        env['QS3_NVFP4_MODEL_DIR']=str(Path.home()/'.cache/huggingface/hub/models--nvidia--Qwen3.8-27B-NVFP4/snapshots/dbb8f445b3145f8a4c18ddc769f032d57d32867c')
    if args.model_benchmark:
        env.update(NVFP4_BENCH_PROVIDER=args.model_benchmark, QS3_BENCH_CONTEXT_TOKENS=str(args.context), QS3_BENCH_DECODE_SAMPLES=str(args.samples), QS3_QWEN36_MODEL_DIR=env['QS3_NVFP4_MODEL_DIR'])
    command = ['cargo','test','--release','--lib','nvfp4_model_benchmark' if args.model_benchmark else 'real_nvfp4_dense_prefill_decode_and_reset' if args.real else 'nvfp4_decode_candidates','--','--ignored','--nocapture','--test-threads=1'] if args.run else ['cargo','check','--tests']
    (output / 'command.json').write_text(json.dumps(command)+'\n')
    if args.model_benchmark:
        build=['cargo','test','--release','--lib','--no-run','--message-format=json']
        with (output/'build.log').open('w') as log:
            compiled=subprocess.run(build,cwd=checkout,env=env,stdout=subprocess.PIPE,stderr=log,text=True)
        if compiled.returncode:
            print((output/'build.log').read_text()[-10000:]);raise SystemExit(compiled.returncode)
        messages=[json.loads(line) for line in compiled.stdout.splitlines() if line.startswith('{')]
        binary=next(m['executable'] for m in messages if m.get('executable') and m.get('profile',{}).get('test'))
        reference=max(Path('/nix/store').glob('*-qs3-benchmark-0.1.0/bin/qs3-bench'),key=lambda p:p.stat().st_mtime)
        inspect_env=dict(env);inspect_env.pop('LD_LIBRARY_PATH',None)
        libs=subprocess.check_output(['ldd',str(reference)],env=inspect_env,text=True)
        interpreter=subprocess.check_output(['readelf','-l',str(reference)],text=True).split('Requesting program interpreter: ')[1].split(']')[0]
        dirs=list(dict.fromkeys(str(Path(line.split(' => ')[1].split()[0]).parent) for line in libs.splitlines() if ' => /nix/store/' in line))
        loader=[interpreter,'--library-path',':'.join(dirs+['/lib/aarch64-linux-gnu'])]
        (output/'libraries.txt').write_text(str(reference)+'\n'+subprocess.check_output(loader+['--list',binary],env=inspect_env,text=True))
        env=inspect_env
        command=loader+[binary,'nvfp4_model_benchmark','--ignored','--nocapture','--test-threads=1']
        (output/'command.json').write_text(json.dumps({'build':build,'run':command,'provider':args.model_benchmark,'context':args.context,'samples':args.samples},indent=2)+'\n')
    with (output / 'run.log').open('w') as log:
        result = subprocess.run(command,cwd=checkout,env=env,stdout=log,stderr=subprocess.STDOUT)
    if result.returncode:
        print((output / 'run.log').read_text()[-12000:])
        raise SystemExit(result.returncode)
    print('GPU probe passed' if args.run else 'Rust check passed; GPU UNVERIFIED', flush=True)

if __name__ == '__main__':
    main()
