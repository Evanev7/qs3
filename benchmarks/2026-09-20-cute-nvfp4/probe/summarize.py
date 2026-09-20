"""Summarize complete GPU results; retain raw paired samples in results.jsonl."""
import argparse
import json
from pathlib import Path
import statistics

p=argparse.ArgumentParser(description=__doc__)
p.add_argument('output',type=Path)
a=p.parse_args()
data=[json.loads(line) for line in (a.output/'results.jsonl').read_text().splitlines()]
meta=json.loads((a.output/'metadata.json').read_text())
correct=[r for r in data if r['kind']=='correctness']
timings=[r for r in data if r['kind']=='timing']
expected={(c['name'],m,alpha) for c in meta['cases'] for m in [1,2,4,8,16] for alpha in [1,0.137]}
assert {(r['candidate'],r['m'],r['alpha']) for r in correct}==expected
assert len(correct)==len(expected)
assert {(r['candidate'],r['m'],r['evicted']) for r in timings} == {(c['name'],m,e) for c in meta['cases'] for m in [1,2,4,8,16] for e in [False,True]}
assert len(timings)==len(expected)
rows=[]
for r in timings:
    b=statistics.median(r['baseline_us']);c=statistics.median(r['candidate_us'])
    ratios=[x/y for x,y in zip(r['baseline_us'],r['candidate_us'])]
    rows.append({k:r[k] for k in ['candidate','m','n','k','evicted']} | {
        'baseline_median_us':b,'candidate_median_us':c,'speedup':b/c,
        'paired_median_speedup':statistics.median(ratios),
        'baseline_enqueue_median_us':statistics.median(r['baseline_enqueue_us']),
        'candidate_enqueue_median_us':statistics.median(r['candidate_enqueue_us'])})
(a.output/'summary.json').write_text(json.dumps({'correctness_cases':len(correct),'timings':rows},indent=2)+'\n')
for key in sorted({(r['n'],r['k'],r['m'],r['evicted']) for r in rows}):
    ranking=sorted((r for r in rows if (r['n'],r['k'],r['m'],r['evicted'])==key),key=lambda r:r['speedup'],reverse=True)
    winner=ranking[0]
    print(f"N{key[0]} K{key[1]} M{key[2]} {'evicted' if key[3] else 'warm'}: {winner['candidate']} {winner['baseline_median_us']:.2f} -> {winner['candidate_median_us']:.2f} us ({winner['speedup']:.3f}x)")
