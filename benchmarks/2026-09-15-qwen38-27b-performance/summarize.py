"""Recompute timing summaries from the archived unprofiled samples."""
import json
import math
from pathlib import Path
import statistics

ROOT = Path(__file__).resolve().parent


def metrics(data):
    prefill = data['prefill']['sample_ms']
    decode = data['decode']['sample_ms']
    assert len(prefill) == 5
    assert len(decode) == data['decode']['samples']
    assert all(math.isfinite(x) and x > 0 for x in prefill + decode)
    throughput = len(decode) * 1000 / sum(decode)
    assert math.isclose(throughput, data['decode']['tokens_per_second'], rel_tol=1e-10)
    return {
        'prefill_p50_ms': statistics.median(prefill),
        'decode_p50_ms': statistics.median(decode),
        'decode_p95_ms': sorted(decode)[math.ceil(0.95 * len(decode)) - 1],
        'decode_tok_s': throughput,
    }


def main():
    result = {}
    for name in ('102-32', '1024-256'):
        root = ROOT / 'capture' / name
        q = json.loads((root / 'qs3.json').read_text())['measurement']
        v = json.loads((root / 'vllm.json').read_text())
        assert q['prompt']['token_id_fnv1a'] == v['prompt']['token_id_fnv1a']
        assert q['decode']['samples'] == v['decode']['samples']
        qi, vi = q['decode']['generated_token_ids'], v['generated_tokens'][:-1]
        assert len(qi) == len(vi) == q['decode']['samples'] + 4
        qm, vm = metrics(q), metrics(v)
        result[name] = {
            'qs3': qm, 'vllm': vm,
            'qs3_decode_throughput_relative_to_vllm': qm['decode_tok_s'] / vm['decode_tok_s'],
            'first_generated_token_difference': next((i for i, (a,b) in enumerate(zip(qi,vi)) if a != b), None),
            'matching_generated_tokens': sum(a == b for a,b in zip(qi,vi)),
            'total_compared_generated_tokens': len(qi),
            'prompt_fingerprint': q['prompt']['token_id_fnv1a'],
        }
    (ROOT / 'summary.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
