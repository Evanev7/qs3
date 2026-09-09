import hashlib,json,struct,urllib.request
from pathlib import Path
repo='Qwen/Qwen3.6-27B'
revision='6a9e13bd6fc8f0983b9b99948120bc37f49c13e9'
out=Path.home()/'qs3-27b-assets'/revision
out.mkdir(parents=True,exist_ok=True)
def fetch(name):
    with urllib.request.urlopen(f'https://huggingface.co/{repo}/resolve/{revision}/{name}',timeout=60) as r:
        data=r.read(32<<20)
        assert r.read(1)==b'', 'metadata file exceeds bound'
    (out/name).write_bytes(data)
    return data
for name in ['config.json','model.safetensors.index.json','tokenizer.json','tokenizer_config.json','generation_config.json']:
    data=fetch(name)
    print(name,len(data),hashlib.sha256(data).hexdigest(),flush=True)
index=json.loads((out/'model.safetensors.index.json').read_bytes())
headers={}
for name in sorted(set(index['weight_map'].values())):
    url=f'https://huggingface.co/{repo}/resolve/{revision}/{name}'
    request=urllib.request.Request(url,headers={'Range':'bytes=0-7'})
    with urllib.request.urlopen(request,timeout=60) as r:
        assert r.status==206,(name,r.status)
        assert r.headers['Content-Range'].startswith('bytes 0-7/'),r.headers['Content-Range']
        size=int(r.headers['Content-Range'].split('/')[1])
        first=r.read(8)
    header_len=struct.unpack('<Q',first)[0]
    assert 0<header_len<=16<<20,(name,header_len)
    request=urllib.request.Request(url,headers={'Range':f'bytes=8-{7+header_len}'})
    with urllib.request.urlopen(request,timeout=60) as r:
        assert r.status==206 and r.headers['Content-Range'].startswith(f'bytes 8-{7+header_len}/'),(r.status,r.headers.get('Content-Range'))
        raw=r.read(header_len)
        assert len(raw)==header_len
    header=json.loads(raw)
    (out/(name+'.header.json')).write_bytes(raw)
    headers[name]={'file_bytes':size,'header_bytes':header_len,'header_sha256':hashlib.sha256(raw).hexdigest(),'tensors':len(header)-('__metadata__' in header)}
    print(name,headers[name],flush=True)
(out/'probe.json').write_text(json.dumps({'repo':repo,'revision':revision,'shards':headers},indent=2)+'\n')
print('saved',out,flush=True)
