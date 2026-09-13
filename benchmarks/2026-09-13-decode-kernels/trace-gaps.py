"""Read-only attribution of the saved single-context Nsight decode trace."""
import bisect
import collections
import json
from pathlib import Path
import sqlite3
import sys

path=Path(sys.argv[1]).resolve()
c=sqlite3.connect(f'file:{path}?mode=ro',uri=True)
c.row_factory=sqlite3.Row
names=dict(c.execute('SELECT id,value FROM StringIds'))
tables={r[0] for r in c.execute("SELECT name FROM sqlite_master WHERE type='table'")}

def merge(intervals):
    out=[]
    for a,b in sorted(intervals):
        if out and a<=out[-1][1]:out[-1]=(out[-1][0],max(b,out[-1][1]))
        else:out.append((a,b))
    return out

class Coverage:
    def __init__(self,intervals):
        self.spans=merge(intervals)
        self.starts=[a for a,b in self.spans]
        self.prefix=[0]
        for a,b in self.spans:self.prefix.append(self.prefix[-1]+b-a)
    def upto(self,t):
        i=bisect.bisect_right(self.starts,t)-1
        if i<0:return 0
        a,b=self.spans[i]
        return self.prefix[i]+max(0,min(t,b)-a)
    def overlap(self,a,b):return self.upto(b)-self.upto(a)

kernels=[dict(r) for r in c.execute('SELECT start,end,correlationId,demangledName,streamId,contextId,deviceId FROM CUPTI_ACTIVITY_KIND_KERNEL ORDER BY start')]
assert len({(r['deviceId'],r['contextId']) for r in kernels})==1
span=(kernels[0]['start'],max(r['end'] for r in kernels))
emb=[r for r in kernels if 'embedding_gather' in names[r['demangledName']]]
arg=[r for r in kernels if 'greedy_argmax' in names[r['demangledName']]]
steps=len(arg)
assert len(emb)==steps and steps>1
apis={}
api_rows=[]
for table in ['CUPTI_ACTIVITY_KIND_RUNTIME','CUPTI_ACTIVITY_KIND_DRIVER']:
    if table not in tables:continue
    for row in c.execute(f'SELECT start,end,correlationId,nameId,globalTid FROM {table}'):
        r=dict(row);r['name']=names[r['nameId']];r['table']=table
        api_rows.append(r)
        apis.setdefault(r['correlationId'],r)

events=[(r['start'],r['end'],names[r['demangledName']],r['correlationId'],'kernel') for r in kernels]
copies=[]
for table,kind in [('CUPTI_ACTIVITY_KIND_MEMCPY','copyKind'),('CUPTI_ACTIVITY_KIND_MEMSET','memKind')]:
    if table not in tables:continue
    for row in c.execute(f'SELECT * FROM {table}'):
        r=dict(row)
        events.append((r['start'],r['end'],f'{table}:{r[kind]}',r['correlationId'],'memory'))
        if kind=='copyKind':copies.append(r)
events.sort()
active=Coverage([(max(a,span[0]),min(b,span[1])) for a,b,*_ in events if b>span[0] and a<span[1]])
gaps=[]
last_end=span[0]
previous='capture start'
for a,b,name,corr,kind in events:
    if b<=span[0] or a>=span[1]:continue
    a=max(a,span[0]);b=min(b,span[1])
    if a>last_end:gaps.append((last_end,a,previous,name,corr))
    if b>last_end:last_end=b;previous=name
assert last_end==span[1]
idle=Coverage([(a,b) for a,b,*_ in gaps])
assert active.prefix[-1]+idle.prefix[-1]==span[1]-span[0]

boundaries=[]
for i in range(steps-1):
    assert emb[i]['start']<arg[i]['end']<emb[i+1]['start']
    boundaries.append((arg[i]['end'],emb[i+1]['start']))
boundary=Coverage(boundaries)

category=collections.Counter();counts=collections.Counter();robust=collections.Counter();pairs=collections.defaultdict(lambda:[0,0]);hist=collections.defaultdict(lambda:[0,0])
for a,b,prev,nxt,corr in gaps:
    d=b-a;pairs[(prev,nxt)][0]+=1;pairs[(prev,nxt)][1]+=d
    us=d/1000
    bucket='<1us' if us<1 else '1-5us' if us<5 else '5-20us' if us<20 else '20-100us' if us<100 else '>=100us'
    hist[bucket][0]+=1;hist[bucket][1]+=d
    api=apis.get(corr)
    if api is not None:
        for threshold in [10000,100000,1000000]:
            if api['end']<=a-threshold:robust[str(threshold//1000)+'us']+=d
    if api is None:category['unmatched_next_api']+=d;counts['unmatched_next_api']+=1;continue
    category['before_next_api_start']+=max(0,min(b,api['start'])-a)
    category['inside_next_api_call']+=max(0,min(b,api['end'])-max(a,api['start']))
    category['after_next_api_return']+=max(0,b-max(a,api['end']))
    if api['end']<=a:counts['next_api_returned_before_gap']+=1
    elif api['start']>=b:counts['next_api_started_after_gpu_start']+=1
    else:counts['next_api_overlaps_or_starts_in_gap']+=1
assert sum(category.values())==idle.prefix[-1]

def ms(n):return n/1e6

def api_summary(rows):
    coverage=Coverage([(r['start'],r['end']) for r in rows])
    busy=sum(active.overlap(a,b) for a,b in coverage.spans)
    quiet=sum(idle.overlap(a,b) for a,b in coverage.spans)
    return {'calls':len(rows),'api_sum_ms_per_token':ms(sum(r['end']-r['start'] for r in rows))/steps,'api_union_ms_per_token':ms(coverage.prefix[-1])/steps,'overlap_gpu_active_ms_per_token':ms(busy)/steps,'overlap_gpu_idle_ms_per_token':ms(quiet)/steps}
copy_groups=collections.defaultdict(list)
copy_device=collections.defaultdict(list)
for r in copies:
    copy_device[r['copyKind']].append(r)
    if r['correlationId'] in apis:copy_groups[r['copyKind']].append(apis[r['correlationId']])

api_by_name=collections.defaultdict(list)
for r in api_rows:api_by_name[r['name']].append(r)
result={
 'source_database':str(path),'steps':steps,'gpu_stream_ids':sorted({r['streamId'] for r in kernels}),
 'kernel_count':len(kernels),'kernel_span_ms_per_token':ms(span[1]-span[0])/steps,
 'gpu_active_ms_per_token':ms(active.prefix[-1])/steps,'gpu_idle_ms_per_token':ms(idle.prefix[-1])/steps,
 'idle_fraction_of_kernel_span':idle.prefix[-1]/(span[1]-span[0]),
 'between_token_boundary_count':len(boundaries),
 'between_token_span_ms_per_boundary':ms(boundary.prefix[-1])/len(boundaries),
 'between_token_idle_ms_per_boundary':ms(sum(idle.overlap(a,b) for a,b in boundaries))/len(boundaries),
 'between_token_idle_ms_amortized_per_token':ms(sum(idle.overlap(a,b) for a,b in boundaries))/steps,
 'within_token_idle_ms_amortized_per_token':ms(idle.prefix[-1]-sum(idle.overlap(a,b) for a,b in boundaries))/steps,
 'gap_count':len(gaps),
 'gap_time_by_next_api_ms_per_token':{k:ms(v)/steps for k,v in category.items()},
 'gap_count_by_next_api':dict(counts),
 'idle_next_api_returned_at_least_before_gap_ms_per_token':{k:ms(v)/steps for k,v in robust.items()},
 'gap_histogram':{k:{'count':v[0],'ms_per_token':ms(v[1])/steps} for k,v in hist.items()},
 'copy_api_by_kind':{str(k):{**api_summary(v),'gpu_copy_ms_per_token':ms(sum(r['end']-r['start'] for r in copy_device[k]))/steps,'bytes_per_token':sum(r['bytes'] for r in copy_device[k])/steps} for k,v in copy_groups.items()},
 'top_api_overlap':{k:api_summary(v) for k,v in sorted(api_by_name.items(),key=lambda kv:sum(r['end']-r['start'] for r in kv[1]),reverse=True)[:12]},
 'largest_gap_pairs':[{'previous':p,'next':n,'count':v[0],'ms_per_token':ms(v[1])/steps} for (p,n),v in sorted(pairs.items(),key=lambda kv:kv[1][1],reverse=True)[:20]],
 'largest_gaps':[{'us':(b-a)/1000,'previous':p,'next':n,'next_api':apis.get(corr)} for a,b,p,n,corr in sorted(gaps,key=lambda x:x[1]-x[0],reverse=True)[:10]]
}
print(json.dumps(result,indent=2))
