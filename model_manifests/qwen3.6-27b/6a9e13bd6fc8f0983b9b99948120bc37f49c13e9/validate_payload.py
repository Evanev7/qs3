"""Verify cached 27B shards against Hub SHA256 names and pinned header records."""
import hashlib,json,struct,time
from pathlib import Path
revision='6a9e13bd6fc8f0983b9b99948120bc37f49c13e9'
model=Path.home()/'.cache/huggingface/hub/models--Qwen--Qwen3.6-27B/snapshots'/revision
manifest=Path.home()/'qs3/model_manifests/qwen3.6-27b'/revision
index=json.loads((model/'model.safetensors.index.json').read_text())
assert index==json.loads((manifest/'model.safetensors.index.json').read_text())
assert json.loads((model/'config.json').read_text())==json.loads((manifest/'config.json').read_text())
shards=sorted(set(index['weight_map'].values()));assert len(shards)==15
results=[];started=time.monotonic()
for name in shards:
    path=model/name
    expected=path.resolve().name
    assert len(expected)==64 and all(c in '0123456789abcdef' for c in expected),expected
    with path.open('rb') as f:
        digest=hashlib.file_digest(f,'sha256').hexdigest()
    assert digest==expected,(name,digest,expected)
    with path.open('rb') as f:
        header_bytes=struct.unpack('<Q',f.read(8))[0]
        header=json.loads(f.read(header_bytes))
    assert header==json.loads((manifest/(name+'.header.json')).read_text()),name
    payload=max(v['data_offsets'][1] for k,v in header.items() if k!='__metadata__')
    assert path.stat().st_size==8+header_bytes+payload,name
    results.append({'shard':name,'bytes':path.stat().st_size,'sha256':digest,'matches_hub_blob_sha256':True,'matches_pinned_header':True})
tokenizer_sha256=hashlib.sha256((model/'tokenizer.json').read_bytes()).hexdigest()
assert tokenizer_sha256=='5f9e4d4901a92b997e463c1f46055088b6cca5ca61a6522d1b9f64c4bb81cb42'
print(json.dumps({'revision':revision,'host':'sp10','shard_count':len(shards),'total_file_bytes':sum(s['bytes'] for s in results),'tokenizer_sha256':tokenizer_sha256,'elapsed_seconds':time.monotonic()-started,'shards':results},indent=2))
