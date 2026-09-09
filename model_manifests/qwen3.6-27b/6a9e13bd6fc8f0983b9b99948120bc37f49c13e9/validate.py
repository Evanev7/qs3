import collections,hashlib,json,math,sys
from pathlib import Path
root=Path(sys.argv[1])
def unique(pairs):
    out={}
    for k,v in pairs:
        assert k not in out,('duplicate JSON key',k)
        out[k]=v
    return out
def read(name):return json.loads((root/name).read_text(),object_pairs_hook=unique)
probe=read('probe.json');cfg=read('config.json')['text_config'];index=read('model.safetensors.index.json')
assert cfg['num_hidden_layers']==64 and cfg['hidden_size']==5120
assert cfg['layer_types']==['linear_attention','linear_attention','linear_attention','full_attention']*16
assert (cfg['num_attention_heads'],cfg['num_key_value_heads'],cfg['head_dim'])==(24,4,256)
assert (cfg['linear_num_key_heads'],cfg['linear_num_value_heads'],cfg['linear_key_head_dim'],cfg['linear_value_head_dim'],cfg['linear_conv_kernel_dim'])==(16,48,128,128,4)
assert cfg['intermediate_size']==17408 and cfg['vocab_size']==248320
actual={};total_bytes=0
for file,meta in probe['shards'].items():
    data=read(file+'.header.json'); spans=[]
    for name,t in data.items():
        if name=='__metadata__':continue
        assert name not in actual,name
        assert index['weight_map'][name]==file,(name,file)
        assert t['dtype']=='BF16',(name,t['dtype'])
        assert all(type(d) is int and d>0 for d in t['shape'])
        begin,end=t['data_offsets']
        assert type(begin) is int and type(end) is int and 0<=begin<=end
        assert end-begin==2*math.prod(t['shape']),(name,t)
        assert end<=meta['file_bytes']-8-meta['header_bytes']
        spans.append((begin,end,name)); actual[name]=t
        total_bytes+=end-begin
    cursor=0
    for begin,end,name in sorted(spans):
        assert begin==cursor,('gap or overlap',name,begin,cursor)
        cursor=end
    assert cursor==meta['file_bytes']-8-meta['header_bytes']
assert set(actual)==set(index['weight_map'])
assert total_bytes==index['metadata']['total_size']
expected={'lm_head.weight':[248320,5120],'model.language_model.embed_tokens.weight':[248320,5120],'model.language_model.norm.weight':[5120]}
for layer in range(64):
    prefix=f'model.language_model.layers.{layer}.'
    shapes={'input_layernorm.weight':[5120],'post_attention_layernorm.weight':[5120],
            'mlp.gate_proj.weight':[17408,5120],'mlp.up_proj.weight':[17408,5120],'mlp.down_proj.weight':[5120,17408]}
    if layer%4==3:
        shapes.update({'self_attn.q_proj.weight':[12288,5120],'self_attn.k_proj.weight':[1024,5120],'self_attn.v_proj.weight':[1024,5120],'self_attn.o_proj.weight':[5120,6144],'self_attn.q_norm.weight':[256],'self_attn.k_norm.weight':[256]})
    else:
        shapes.update({'linear_attn.A_log':[48],'linear_attn.dt_bias':[48],'linear_attn.conv1d.weight':[10240,1,4],'linear_attn.in_proj_qkv.weight':[10240,5120],'linear_attn.in_proj_z.weight':[6144,5120],'linear_attn.in_proj_a.weight':[48,5120],'linear_attn.in_proj_b.weight':[48,5120],'linear_attn.norm.weight':[128],'linear_attn.out_proj.weight':[5120,6144]})
    expected.update({prefix+k:v for k,v in shapes.items()})
text_names={k for k in actual if k.startswith('model.language_model.') or k=='lm_head.weight'}
assert text_names==set(expected),(text_names-set(expected),set(expected)-text_names)
for name,shape in expected.items():assert actual[name]['shape']==shape,(name,actual[name]['shape'],shape)
assert all(k.startswith(('model.visual.','mtp.')) for k in set(actual)-text_names)
report={'repo':probe['repo'],'revision':probe['revision'],'shards':len(probe['shards']),'all_tensors':len(actual),'text_tensors':len(text_names),'all_tensor_bytes':total_bytes,'text_tensor_bytes':sum(2*math.prod(expected[k]) for k in expected),'ignored_tensor_prefix_counts':dict(collections.Counter('model.visual' if k.startswith('model.visual.') else 'mtp' for k in set(actual)-text_names)),'config_sha256':hashlib.sha256((root/'config.json').read_bytes()).hexdigest(),'index_sha256':hashlib.sha256((root/'model.safetensors.index.json').read_bytes()).hexdigest()}
(root/'validation.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2))
