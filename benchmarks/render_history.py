"""Render the recorded core runs: python3 benchmarks/render_history.py."""

import json
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def collect_runs() -> list[dict[str, object]]:
    runs: list[dict[str, object]] = []
    for path in sorted(ROOT.glob("*.json")):
        source = json.loads(path.read_text())
        measurement = source.get("measurement", {})
        if "decode" not in measurement or "metadata" not in source:
            continue
        metadata = source["metadata"]
        execution = measurement["execution"]
        runs.append({
            "commit": metadata["commit_hash"],
            "file": path.name,
            "date": measurement["started_at"],
            "prompt": measurement["prompt"]["tokens"],
            "samples": measurement["decode"]["samples"],
            "warmups": measurement["decode"]["warmups"],
            "context_start": measurement["decode"]["context_start"],
            "context_end": measurement["decode"]["context_end"],
            "tps": measurement["decode"]["tokens_per_second"],
            "decode": measurement["decode"]["p50_ms"],
            "p95": measurement["decode"]["p95_ms"],
            "prefill": measurement["prefill"]["p50_ms"],
            "execution": execution,
            "metadata": metadata,
        })
    commits = list(dict.fromkeys(str(run["commit"]) for run in runs))
    output = subprocess.run(
        ["git", "show", "--no-patch", "--format=%H%x09%s", *commits],
        cwd=ROOT, check=True, capture_output=True, text=True,
    ).stdout
    subjects = dict(line.split("\t", 1) for line in output.splitlines() if "\t" in line)
    for run in runs:
        run["subject"] = subjects[str(run["commit"])]
    return sorted(runs, key=lambda run: str(run["date"]))


HTML = r'''<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Quasar3 · Benchmark history</title>
<style>
:root{color-scheme:dark;--bg:#10151b;--panel:#171e27;--line:#2b3542;--ink:#edf2f6;--muted:#a0aebc;--accent:#8fbded;--green:#83ddbb}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--ink);font:15px/1.5 system-ui,-apple-system,sans-serif}main{max-width:1540px;margin:auto;padding:40px 42px 54px}h1{font-size:35px;letter-spacing:-1.3px;font-weight:650;margin:4px 0 10px}p{margin:0;color:var(--muted)}.eyebrow{font:11px ui-monospace,monospace;letter-spacing:2px;color:var(--green)}.intro{max-width:900px}.top{display:flex;justify-content:space-between;align-items:flex-start;gap:20px}.stamp{font:12px ui-monospace,monospace;color:var(--muted);padding-top:12px;white-space:nowrap}.cards{display:grid;grid-template-columns:repeat(3,1fr);gap:14px;margin:25px 0}.card{border:1px solid var(--line);border-radius:10px;padding:17px 20px;background:var(--panel)}.card span{display:block;color:var(--muted);font-size:12px}.card strong{font-size:29px;font-weight:600;letter-spacing:-.6px}.card small{font-size:13px;font-weight:400;color:var(--muted);margin-left:8px}.card:last-child strong{color:var(--green)}.controls{display:flex;gap:14px;align-items:end;flex-wrap:wrap;margin-bottom:20px}label{font-size:12px;color:var(--muted);display:flex;flex-direction:column;gap:6px}select,input,button{font:inherit}select,input{color:var(--ink);background:var(--panel);border:1px solid #3b4858;border-radius:6px;padding:9px 11px;font-size:13px}select{min-width:205px}input{min-width:270px}.count{margin-left:auto;font-size:12px;padding-bottom:10px;color:var(--muted)}.chart-panel{border:1px solid var(--line);border-radius:10px;overflow:hidden;background:var(--panel)}.chart-heading{padding:18px 20px;border-bottom:1px solid var(--line);display:flex;gap:15px;align-items:center;justify-content:space-between}.chart-heading strong{font-size:15px}.chart-heading span{color:var(--muted);font-size:12px}.axis-row,.run{display:grid;grid-template-columns:minmax(290px,43%) 1fr;gap:22px;padding:0 20px}.axis-row{height:40px;align-items:center;background:#141b23}.axis-label{font-size:11px;text-transform:uppercase;letter-spacing:1px;color:var(--muted)}.axis{height:100%;position:relative;margin-right:78px}.tick{position:absolute;top:12px;transform:translateX(-50%);font:11px ui-monospace,monospace;color:var(--muted)}.tick:first-child{transform:none}.run{width:100%;border:0;border-bottom:1px solid #26313d;background:transparent;color:inherit;text-align:left;min-height:79px;cursor:pointer;align-items:center}.run{white-space:normal}.run>span:first-child{min-width:0}.subject{overflow-wrap:anywhere}.run:last-child{border-bottom:0}.run:hover{background:#202b37}.run[aria-pressed=true]{background:#23323b;box-shadow:inset 3px 0 var(--green)}.run:focus-visible,select:focus-visible,input:focus-visible{outline:2px solid var(--green);outline-offset:-2px}.subject{font-size:13px;font-weight:500;display:block;line-height:1.4;padding-top:9px}.revision{display:block;font:11px/1.6 ui-monospace,monospace;color:var(--muted);padding:4px 0 9px}.revision b{color:#c0cfdf;font-weight:400}.plot{position:relative;height:100%;min-height:78px;margin-right:78px;background:repeating-linear-gradient(to right,#33404c77 0,#33404c77 1px,transparent 1px,transparent 25%);display:flex;align-items:center}.bar{height:26px;background:var(--accent);border-radius:0 4px 4px 0;min-width:2px;position:relative}.latest .bar{background:var(--green)}.value{position:absolute;left:100%;top:3px;margin-left:10px;white-space:nowrap;font:13px ui-monospace,monospace;color:var(--ink)}.empty{padding:30px;text-align:center;color:var(--muted)}.detail{margin-top:18px;border:1px solid var(--line);border-radius:10px;padding:20px;background:var(--panel)}.detail h2{font-size:17px;margin:0 0 14px;font-weight:550}.detail-grid{display:grid;grid-template-columns:repeat(4,minmax(0,1fr));gap:15px 25px}.datum span{display:block;color:var(--muted);font-size:11px;margin-bottom:3px}.datum strong{font-size:12px;font-weight:450;overflow-wrap:anywhere;display:block}a{color:var(--green);text-decoration:none}a:hover{text-decoration:underline}.source{margin-top:16px;font-size:12px}.foot{font-size:12px;line-height:1.7;margin-top:18px;max-width:1050px}.legend{display:inline-flex;gap:7px;align-items:center}.dot{width:9px;height:9px;background:var(--green);border-radius:2px;display:inline-block}
@media(max-width:760px){main{padding:23px 16px}h1{font-size:28px}.top{display:block}.stamp{padding-top:12px}.cards{gap:8px}.card{padding:12px}.card strong{font-size:23px}.card small{display:block;margin:0}.controls{display:grid;grid-template-columns:1fr;gap:12px}.controls label{width:100%}select,input{min-width:0;width:100%}.count{margin-left:0}.axis-row,.run{grid-template-columns:minmax(170px,49%) 1fr;gap:12px;padding:0 12px}.plot,.axis{margin-right:58px}.run{min-height:100px}.subject{font-size:11px}.revision{font-size:10px}.value{font-size:11px;margin-left:5px}.detail-grid{grid-template-columns:repeat(2,minmax(0,1fr))}.chart-heading{align-items:flex-start;flex-direction:column;gap:5px}}
@media print{body{background:white;color:black}.controls{display:none}main{padding:10px}.run{break-inside:avoid}.chart-panel,.card,.detail{background:white}.subject,.value{color:black}.revision,.foot,p{color:#555}.run[aria-pressed=true]{background:#eef5f2}.bar{print-color-adjust:exact}}
</style>
</head>
<body><main>
<div class="top"><div class="intro"><div class="eyebrow">QUASAR3 / SPARK GB10</div><h1>Performance across commits</h1><p>Recorded core benchmark runs, labeled with their measured commit subjects. Select a run to inspect its configuration.</p></div><div class="stamp" id="stamp"></div></div>
<div class="cards"><div class="card"><span>Runs in this view</span><strong id="run-count"></strong></div><div class="card"><span>First recorded run</span><strong id="first"></strong><span id="first-id"></span></div><div class="card"><span>Latest recorded run</span><strong id="latest"></strong><span id="latest-id"></span></div></div>
<div class="controls"><label>Workload<select id="workload"></select></label><label>Metric<select id="metric"><option value="tps">Decode throughput · tok/s</option><option value="decode">Decode latency · p50 ms</option><option value="p95">Decode latency · p95 ms</option><option value="prefill">Prefill latency · p50 ms</option></select></label><label>Filter commits<input id="search" type="search" placeholder="Commit subject or hash" autocomplete="off"></label><span class="count" id="count"></span></div>
<section class="chart-panel" aria-label="Benchmark comparison"><div class="chart-heading"><strong id="chart-title"></strong><span><span class="legend"><i class="dot"></i> Latest run</span> · Oldest → newest</span></div><div class="axis-row"><span class="axis-label">Commit / configuration</span><div class="axis" id="axis"></div></div><div id="chart"></div></section>
<section class="detail" id="detail" aria-live="polite"></section>
<p class="foot">Each bar is one recorded run; repeated commits retain their separate measurements. Workloads are separated by prompt length and measured decode steps. Precision, kernel settings and loading strategy changed over this history, so adjacent bars are not necessarily controlled A/B comparisons. Missing historical metadata is shown as “not recorded”. The chart contains the core JSON files in this directory; source links open the original measurements.</p>
<noscript><p>This interactive chart needs JavaScript enabled. All data is embedded in this file; no network access is required.</p></noscript>
</main>
<script id="benchmark-data" type="application/json">__DATA__</script>
<script>
'use strict';
const runs = JSON.parse(document.getElementById('benchmark-data').textContent);
const $ = id => document.getElementById(id);
const metrics = {tps:{label:'Decode throughput',unit:'tok/s',direction:'Higher is better'},decode:{label:'Decode latency · p50',unit:'ms',direction:'Lower is better'},p95:{label:'Decode latency · p95',unit:'ms',direction:'Lower is better'},prefill:{label:'Prefill latency · p50',unit:'ms',direction:'Lower is better'}};
const escape = s => String(s).replace(/[&<>"']/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
const format = n => Number(n).toLocaleString('en-US',{minimumFractionDigits:3,maximumFractionDigits:3});
const stamp = s => s.replace('T',' ').slice(0,19)+' UTC';
const workloads = [...new Set(runs.map(r=>`${r.prompt}/${r.samples}`))].sort((a,b)=>Number(a.split('/')[0])-Number(b.split('/')[0]));
for(const key of workloads){const [p,n]=key.split('/');const opt=document.createElement('option');opt.value=key;opt.textContent=`${Number(p).toLocaleString()} prompt / ${n} decode`; $('workload').append(opt);}
const params = new URLSearchParams(location.search);
if(workloads.includes(params.get('workload'))) $('workload').value=params.get('workload');
if(metrics[params.get('metric')]) $('metric').value=params.get('metric');
$('search').value=params.get('q')||'';
$('stamp').textContent=`${runs.length} runs · through ${runs.at(-1).date.slice(0,10)}`;
let selected = null;
function setting(r){const e=r.execution;const bits=[];if(e.gdn_recurrent_state_dtype)bits.push(`GDN ${e.gdn_recurrent_state_dtype.toUpperCase()}`);if(e.moe_kernel)bits.push(e.moe_kernel);else if(e.moe_threadblocks)bits.push(`MoE ${e.moe_threadblocks} blocks`);if(e.weight_backend==='pinned_upload')bits.push('device weights');if(e.lm_head==='triton')bits.push('Triton LM');return bits.join(' · ');}
function datum(label,value){return `<div class="datum"><span>${escape(label)}</span><strong>${escape(value??'not recorded')}</strong></div>`;}
function inspect(r){
 selected=r.file;
 document.querySelectorAll('.run').forEach(el=>el.setAttribute('aria-pressed',String(el.dataset.file===selected)));
 const e=r.execution,m=r.metadata;
 const fields=[['Commit',r.commit],['Recorded',stamp(r.date)],['Decode throughput',format(r.tps)+' tok/s'],['Decode p50 / p95',format(r.decode)+' / '+format(r.p95)+' ms'],['Prefill p50',format(r.prefill)+' ms'],['Workload',`${r.prompt} prompt · ${r.warmups} warmups · ${r.samples} measured decode`],['Decode context',`${r.context_start} → ${r.context_end}`],['Weight backend',e.weight_backend],['GDN recurrent state',e.gdn_recurrent_state_dtype],['Router logits',e.router_logits_dtype],['MoE kernel',e.moe_kernel??e.moe],['MoE threadblocks',e.moe_threadblocks],['LM head',e.lm_head],['CUDA / driver',`${m.cuda_runtime} / ${m.driver}`],['Execution',e.mode],['Rust toolchain',m.rust]];
 $('detail').innerHTML=`<h2>${escape(r.subject)}</h2><div class="detail-grid">${fields.map(x=>datum(...x)).join('')}</div><div class="source"><a href="${encodeURI(r.file)}">Open original benchmark JSON ↗</a></div>`;
}
function render(){
 const metric=$('metric').value,meta=metrics[metric],query=$('search').value.toLowerCase().trim();
 const rows=runs.filter(r=>`${r.prompt}/${r.samples}`===$('workload').value && `${r.subject} ${r.commit}`.toLowerCase().includes(query));
 $('run-count').textContent=rows.length;
 $('count').textContent=`${rows.length} of ${runs.length} recorded runs`;
 $('chart-title').textContent=`${meta.label} (${meta.unit}) · ${meta.direction}`;
 for(const [id,r] of [['first',rows[0]],['latest',rows.at(-1)]]){
  $(id).innerHTML=r?`${format(r[metric])}<small>${meta.unit}</small>`:'—';
  $(id+'-id').textContent=r?`${r.commit.slice(0,7)} · ${r.date.slice(0,10)}`:'No matching runs';
 }
 if(!rows.length){$('chart').innerHTML='<div class="empty">No runs match this filter.</div>';$('axis').replaceChildren();$('detail').hidden=true;return;}
 $('detail').hidden=false;
 const max=Math.max(...rows.map(r=>r[metric]));
 const rough=max/4, magnitude=10**Math.floor(Math.log10(rough));
 const step=[1,2,2.5,5,10].map(n=>n*magnitude).find(n=>n>=rough);
 const limit=step*4;
 $('axis').innerHTML=Array.from({length:5},(_,i)=>`<span class="tick" style="left:${i*25}%">${Number((step*i).toFixed(2)).toLocaleString()}</span>`).join('');
 $('chart').replaceChildren();
 rows.forEach((r,i)=>{
  const button=document.createElement('button');button.className='run'+(i===rows.length-1?' latest':'');button.type='button';button.dataset.file=r.file;
  button.setAttribute('aria-label',`${r.subject}, ${r.commit.slice(0,7)}, ${meta.label}: ${format(r[metric])} ${meta.unit}. Inspect run.`);
  button.innerHTML=`<span><span class="subject">${escape(r.subject)}</span><span class="revision"><b>${r.commit.slice(0,7)}</b> · ${escape(setting(r)||r.date.slice(0,10))}</span></span><span class="plot"><span class="bar" style="width:${r[metric]/limit*100}%"><span class="value">${format(r[metric])}</span></span></span>`;
  button.addEventListener('click',()=>{inspect(r);$('detail').scrollIntoView({behavior:'smooth',block:'nearest'});});$('chart').append(button);
 });
 inspect(rows.find(r=>r.file===selected)||rows.at(-1));
}
for(const id of ['workload','metric','search']) $(id).addEventListener('input',render);
render();
</script></body></html>
'''


def main() -> None:
    data = json.dumps(collect_runs(), separators=(",", ":"), ensure_ascii=False)
    # Keep data inside the non-executable JSON script even for unusual subjects.
    data = data.replace("&", "\\u0026").replace("<", "\\u003c").replace(">", "\\u003e")
    output = ROOT / "performance.html"
    output.write_text(HTML.replace("__DATA__", data))
    print(output)


if __name__ == "__main__":
    main()
