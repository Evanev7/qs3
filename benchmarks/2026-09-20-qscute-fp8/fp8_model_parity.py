"""Same-prefix CuTe/cuBLASLt comparison in a diagnostic source overlay."""
import array
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import tempfile

ROOT = Path(__file__).resolve().parents[2]
REV = 'dbb8f445b3145f8a4c18ddc769f032d57d32867c'
TEST = r'''
#[test]
#[ignore]
fn kr03_fp8_model_parity() {
    use crate::{ModelRunner, QwenRequest, QwenTokenizer};
    let directory = require_real_qwen36_model_dir();
    let plan = QwenLoadPlan::read(&directory).unwrap();
    let tokenizer = QwenTokenizer::from_model_dir(&directory).unwrap();
    let ctx = std::rc::Rc::new(crate::memory::CudaCtx::new(0).unwrap());
    let (config, weights) = execute_qwen_load_plan(&plan, ManagedUmaBackend::new(0).unwrap(), &ctx)
        .unwrap().into_qwen_model(1200).unwrap();
    let mut runner = ModelRunner::new(ctx, config, weights, tokenizer.token_count()).unwrap();
    let mut cute = runner.gdn_qkv.take();
    let root = std::path::PathBuf::from(std::env::var_os("KR03_OUTPUT").unwrap());
    let source = tokenizer.encode("Explain how a computer represents numbers and why floating point arithmetic can round differently. Give a clear example and explain the practical consequences. ".repeat(100).as_str());
    for len in [4usize, 102, 1024] {
      for baseline in [true, false] {
        if !baseline { runner.gdn_qkv = cute.take(); }
        assert_eq!(runner.gdn_qkv_provider(), if baseline { "cublaslt-fp8" } else { "cute-fp8-split2" });
        let output = root.join(if baseline { "cublaslt" } else { "cute" });
        std::fs::create_dir_all(&output).unwrap();
        let prompt = if len == 4 { vec![1,2,3,4] } else { source[..len].to_vec() };
        let forced_file = root.join(format!("forced-{len}.bin"));
        let mut forced:Vec<i32> = if baseline { vec![] } else { std::fs::read(&forced_file).unwrap().chunks_exact(4).map(|b| i32::from_le_bytes(b.try_into().unwrap())).collect() };
        runner.reset().unwrap();
        runner.run(QwenRequest { request_id: 93, tokens: &prompt, max_new_tokens: 0 }).unwrap();
        for step in 0..=32 {
            let logits = runner.last_logits_row_for_test().unwrap();
            assert!(logits.iter().all(|v| v.is_finite()));
            std::fs::write(output.join(format!("{len}-{step}.f32")), logits.iter().flat_map(|v| v.to_le_bytes()).collect::<Vec<_>>()).unwrap();
            if step < 32 {
                if baseline {
                    let token = (0..tokenizer.token_count()).max_by(|&a,&b| logits[a].total_cmp(&logits[b]).then(b.cmp(&a))).unwrap();
                    forced.push(token as i32);
                }
                runner.decode_forced_token_for_test(forced[step]).unwrap();
            }
        }
        if baseline { std::fs::write(forced_file, forced.iter().flat_map(|v| v.to_le_bytes()).collect::<Vec<_>>()).unwrap(); }
        if !baseline { cute = runner.gdn_qkv.take(); }
      }
    }
}
'''

def main(expected_rows=99, protocol='Single runner, weights/scratch/other plans retained; three prefixes (4,102,1024), 32 forced baseline-greedy decode steps each; same tokens at every comparison. Diagnostic downloads invalidate timing.'):
    out = Path(tempfile.mkdtemp(prefix='kr03-runtime-parity-', dir=ROOT/'.prototypes/out'))
    print(out, flush=True)
    work = out/'checkout'; work.mkdir()
    for path in ROOT.iterdir():
        if path.name not in {'src', 'target', '.git', '.prototypes'}:
            (work/path.name).symlink_to(path, target_is_directory=path.is_dir())
    shutil.copytree(ROOT/'src', work/'src')
    runner = work/'src/model/runner/mod.rs'
    s = runner.read_text().replace('enum GdnQkv {', 'pub(crate) enum GdnQkv {')
    s = s.replace('    gdn_qkv: Option<GdnQkv>,', '    pub(crate) gdn_qkv: Option<GdnQkv>,')
    runner.write_text(s)
    tests = work/'src/loader/tests/mod.rs'; tests.write_text(tests.read_text()+TEST)
    with (out/'overlay.diff').open('w') as f:
        subprocess.run(['git','diff','--no-index',str(ROOT/'src'),str(work/'src')],stdout=f)
    env = dict(os.environ, CARGO_TARGET_DIR=str(ROOT/'target'), KR03_OUTPUT=str(out), QS3_QWEN36_MODEL_DIR=str(Path.home()/'.cache/huggingface/hub/models--nvidia--Qwen3.8-27B-NVFP4/snapshots'/REV))
    build = ['cargo','test','--release','--lib','--no-run','--message-format=json']
    with (out/'build.log').open('w') as log:
        result = subprocess.run(build, cwd=work, env=env, stdout=subprocess.PIPE, stderr=log, text=True, check=True)
    messages=[json.loads(line) for line in result.stdout.splitlines() if line.startswith('{')]
    executable=next(message['executable'] for message in messages if message.get('executable') and message.get('profile',{}).get('test'))
    # Execute with the release benchmark's exact Nix libraries and ELF loader.
    # This keeps its cuBLASLt recipe selection in the numerical comparison.
    reference=max(Path('/nix/store').glob('*-qs3-benchmark-0.1.0/bin/qs3-bench'),key=lambda p:p.stat().st_mtime)
    inspect_env = dict(os.environ); inspect_env.pop('LD_LIBRARY_PATH', None)
    libs=subprocess.check_output(['ldd',str(reference)],env=inspect_env,text=True)
    interpreter=subprocess.check_output(['readelf','-l',str(reference)],text=True).split('Requesting program interpreter: ')[1].split(']')[0]
    libdirs=list(dict.fromkeys(str(Path(line.split(' => ')[1].split()[0]).parent) for line in libs.splitlines() if ' => /nix/store/' in line))
    library_path=':'.join(libdirs+['/lib/aarch64-linux-gnu'])
    loader=[interpreter,'--library-path',library_path]
    (out/'libraries.txt').write_text(str(reference)+'\n'+subprocess.check_output(loader+['--list',executable],env=env,text=True))
    cmd=loader+[executable,'kr03_fp8_model_parity','--ignored','--nocapture','--test-threads=1']
    with (out/'run.log').open('w') as f:
        subprocess.run(cmd, cwd=work, env=env, stdout=f, stderr=subprocess.STDOUT, check=True)
    rows=[]
    for path in sorted((out/'cute').glob('*.f32')):
        base = (out/'cublaslt'/path.name).read_bytes(); data=path.read_bytes()
        a=array.array('f');a.frombytes(data);b=array.array('f');b.frombytes(base)
        assert len(a)==len(b)==248320
        i=max(range(248077),key=lambda i:b[i]);j=max(range(248077),key=lambda i:a[i])
        diff=[x-y for x,y in zip(a,b)]
        rows.append(dict(row=path.stem, exact=data==base, changed=sum(x!=y for x,y in zip(a,b)), max_abs=max(map(abs,diff)), relative_l2=math.sqrt(sum(x*x for x in diff)/sum(x*x for x in b)), baseline_argmax=i, cute_argmax=j, baseline_margin=b[i]-max(b[:i]+b[i+1:]), cute_margin=a[j]-max(a[:j]+a[j+1:]), sha256=hashlib.sha256(data).hexdigest(), baseline_sha256=hashlib.sha256(base).hexdigest()))
    result=dict(rows=rows, row_count=len(rows), exact_rows=sum(r['exact'] for r in rows), matching_argmax=sum(r['baseline_argmax']==r['cute_argmax'] for r in rows), protocol=protocol)
    (out/'comparison.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k!='rows'}),flush=True)
    assert len(rows)==expected_rows
    assert all(r['exact'] for r in rows if r['row'].endswith('-0')), 'unchanged prefill must match exactly'

if __name__ == '__main__': main()
