"""Compare completed baseline files with the two e1befe1 core benchmarks."""
import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
p = argparse.ArgumentParser(description=__doc__)
p.add_argument('results',type=Path)
args = p.parse_args()
comparisons=[]
for qpath in sorted((ROOT/'benchmarks').glob('*-e1befe1.json')):
    q=json.loads(qpath.read_text())['measurement']
    context,samples=q['prompt']['tokens'],q['decode']['samples']
    v=json.loads((args.results/f'{context}-{samples}'/'vllm.json').read_text())
    assert Path(q['model']).name==Path(v['model']).name=='dbb8f445b3145f8a4c18ddc769f032d57d32867c'
    assert q['prompt']['token_id_fnv1a']==v['prompt']['token_id_fnv1a']
    for key in ['samples','context_start','context_end']:
        assert q['decode'][key]==v['decode'][key]
    assert q['execution']['gdn_recurrent_state_dtype']=='f32'
    assert v['execution']['resolved_gdn_state_dtypes']==['torch.bfloat16','torch.float32']
    assert v['execution']['speculative_config'] is None
    qids=q['decode']['generated_token_ids'];vids=v['generated_tokens'][:-1]
    assert len(qids)==len(vids)==samples+4
    comparisons.append({'context':context,'decode_samples':samples,
        'qs3_artifact':qpath.name,'prompt_fingerprint':q['prompt']['token_id_fnv1a'],
        'first_generated_token_difference':next((i for i,(a,b) in enumerate(zip(qids,vids)) if a!=b),None),
        'matching_generated_tokens':sum(a==b for a,b in zip(qids,vids)),
        'qs3':{'prefill_p50_ms':q['prefill']['p50_ms'],'decode_p50_ms':q['decode']['p50_ms'],'decode_tok_s':q['decode']['tokens_per_second']},
        'vllm':{'prefill_p50_ms':v['prefill']['p50_ms'],'decode_p50_ms':v['decode']['p50_ms'],'decode_tok_s':v['decode']['tokens_per_second']},
        'caveats':['qs3 eager vs vLLM default graphs','independent greedy continuations, not same-prefix parity',
                   'vLLM prefill includes first token sampling/delivery; qs3 prefill requests no new token']})
assert len(comparisons)==2
(args.results/'comparison.json').write_text(json.dumps(comparisons,indent=2)+'\n')
print(json.dumps(comparisons,indent=2))
