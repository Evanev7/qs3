import json
import re
from pathlib import Path

root=Path(__file__).resolve().parent
runs=[]
for condition, filename in [('uncontrolled','uncontrolled.log'),('cold','cold.log')]:
    source=(root/filename).read_text()
    starts=list(re.finditer(r'(?:Cold )?Weight-load backend: (managed_uma|pinned_upload)',source,re.I))
    for i,start in enumerate(starts):
        section=source[start.start():starts[i+1].start() if i+1<len(starts) else len(source)]
        assert 'test result: ok. 1 passed; 0 failed' in section
        loaded=re.search(r'loaded (\d+) tensors: ([\d.]+) GiB read, ([\d.]+) MiB zero-fill, ([\d.]+)s, ([\d.]+) GiB/s',section)
        assert loaded
        row={'condition':condition,'backend':start.group(1),'sequence':i,
             'tensor_count':int(loaded[1]),'read_gib':float(loaded[2]),'zero_fill_mib':float(loaded[3]),
             'load_seconds':float(loaded[4]),'load_gib_per_second':float(loaded[5]),
             'backend_setup_seconds':float(re.search(r'backend setup ([\d.]+)s',section)[1]),
             'setup_plus_load_seconds':float(re.search(r'backend setup \+ load ([\d.]+)s',section)[1]),
             'phases':re.search(r'phases: (.*)',section)[1],
             'mem_available_kib':int(re.search(r'MemAvailable:\s+(\d+)',section)[1]),
             'cached_kib':int(re.search(r'Cached:\s+(\d+)',section)[1])}
        if condition=='cold':
            cache=json.loads(next(line for line in section.splitlines() if line.startswith('{"files"')))
            assert cache['resident_pages_after']==0
            row['cache_preparation']=cache
        runs.append(row)
assert len(runs)==5
result={'source_commit':'89a264e','snapshot':'995ad96eacd98c81ed38be0c5b274b04031597b0',
        'host':'sp10','cuda_runtime':'13.0','pinned_ring':{'buffers':4,'bytes_per_buffer':1<<30},
        'timing':'backend setup plus execute_qwen36_bf16_load_plan; manifest planning and teardown separate',
        'runs':runs}
(root/'results.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(result,indent=2))
