"""Compare the candidate with the saved qs3 scores at identical token prefixes."""
import argparse
import array
import hashlib
import json
import math
from pathlib import Path
import statistics
import sys

p = argparse.ArgumentParser()
p.add_argument('baseline', type=Path)
p.add_argument('candidate', type=Path)
p.add_argument('output', type=Path)
p.add_argument('--workloads', nargs='+', default=('102-32', '500-200', '4000-800'))
a = p.parse_args()
result = {'baseline': str(a.baseline), 'candidate': str(a.candidate), 'workloads': {}}
for name in a.workloads:
    baseline = a.baseline / name
    candidate = a.candidate / name
    before = json.loads((baseline / 'qs3/scores.json').read_text())
    after = json.loads((candidate / 'qs3/scores.json').read_text())
    assert before['input'] == after['input']
    br, ar = before['records'], after['records']
    assert len(br) == len(ar)
    frames = []
    for b, c in zip(br, ar):
        assert b['step'] == c['step'] and b.get('forced_id') == c.get('forced_id')
        assert ('file' in b) == ('file' in c)
        if 'file' not in b:
            continue
        arrays, hashes = [], []
        for root, record in ((baseline, b), (candidate, c)):
            raw = (root / 'qs3' / record['file']).read_bytes()
            values = array.array('f', raw)
            if sys.byteorder != 'little':
                values.byteswap()
            assert len(values) == 248320 and all(map(math.isfinite, values))
            arrays.append(values)
            hashes.append(hashlib.sha256(raw).hexdigest())
        delta = [float(y)-float(x) for x,y in zip(*arrays)]
        bias = statistics.fmean(delta)
        frames.append({'step': b['step'], 'baseline_sha256': hashes[0], 'candidate_sha256': hashes[1],
                       'identical': hashes[0] == hashes[1], 'max_abs_delta': max(map(abs, delta)),
                       'rmse': math.sqrt(statistics.fmean(x*x for x in delta)),
                       'centered_rmse': math.sqrt(statistics.fmean((x-bias)**2 for x in delta))})
    old_vllm = json.loads((baseline / 'comparison.json').read_text())
    new_vllm = json.loads((candidate / 'comparison.json').read_text())
    assert old_vllm['steps'] == new_vllm['steps'] == len(br)
    differing_steps = [b['step'] for b,c in zip(br,ar) if b['argmax'] != c['argmax']]
    result['workloads'][name] = {
        'positions': len(br), 'changed_qs3_argmax_steps': differing_steps,
        'baseline_vllm_agreement': old_vllm['argmax_agreement_count'],
        'candidate_vllm_agreement': new_vllm['argmax_agreement_count'],
        'baseline_forced_mean_nll': old_vllm['qs3_forced_mean_nll'],
        'candidate_forced_mean_nll': new_vllm['qs3_forced_mean_nll'],
        'baseline_vllm_mean_centered_rmse': old_vllm['mean_centered_rmse'],
        'candidate_vllm_mean_centered_rmse': new_vllm['mean_centered_rmse'],
        'captures': frames,
    }
a.output.write_text(json.dumps(result, indent=2)+'\n')
print(json.dumps({k: {n:v for n,v in x.items() if n != 'captures'} for k,x in result['workloads'].items()}, indent=2))
