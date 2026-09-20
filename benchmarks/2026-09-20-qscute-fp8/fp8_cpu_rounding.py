"""Compare captured real FP8 QKV with exact products and rounding orders."""
import sys, json
from pathlib import Path
import numpy as np
root=Path(sys.argv[1])
codes=np.arange(256,dtype=np.int32); e=(codes>>3)&15; m=codes&7
lut=np.where(e==0,m/512.,(1+m/8.)*np.exp2(e-7))*np.where(codes&128,-1,1)
def bf16(v):
    a=np.array(v,dtype=np.float32).view(np.uint32)
    return ((a+0x7fff+((a>>16)&1))>>16).astype(np.uint16)
for layer in range(3):
    pre=root/f'qkv-{layer}'
    x=lut[np.fromfile(str(pre)+'.x.fp8',dtype=np.uint8)]
    w=lut[np.fromfile(str(pre)+'.w.fp8',dtype=np.uint8).reshape(10240,5120)]
    xs,ws=np.fromfile(str(pre)+'.scales.f32',dtype=np.float32)
    exact=w@x
    parts=np.stack([w[:,:2560]@x[:2560],w[:,2560:]@x[2560:]])
    fp=parts.astype(np.float32); alpha=np.float32(xs*ws)
    variants={'exact':bf16(exact*float(xs)*float(ws)), 'sum_scale':bf16(exact.astype(np.float32)*alpha), 'split_scale':bf16(fp[0]*alpha+fp[1]*alpha), 'xs_ws':bf16((exact.astype(np.float32)*xs)*ws), 'ws_xs':bf16((exact.astype(np.float32)*ws)*xs)}
    cute=np.fromfile(str(pre)+'.cute.bf16',dtype=np.uint16);ref=np.fromfile(str(pre)+'.reference.bf16',dtype=np.uint16)
    print(json.dumps({'layer':layer,'scales':[float(xs),float(ws)],'mismatch':{k:{'cute':int(np.count_nonzero(v!=cute)),'reference':int(np.count_nonzero(v!=ref))} for k,v in variants.items()}},indent=2),flush=True)
    for i in np.flatnonzero(cute!=ref):
        print('column',i,'dot',exact[i], 'cute',hex(int(cute[i])),'reference',hex(int(ref[i])),'variants',{k:hex(int(v[i])) for k,v in variants.items()},flush=True)
