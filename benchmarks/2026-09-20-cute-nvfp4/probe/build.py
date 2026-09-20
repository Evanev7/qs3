"""GPU-free AOT export of small-M NVFP4 candidates for GB10."""
import argparse
import json
import importlib.metadata
import platform
import subprocess
from pathlib import Path
from qscute.builder import compile_source
from qsutil.config import CuteSpec, CudaTarget, parse
from qstriton.builder import build as build_triton
from qsutil.config import TritonSpec
from reduce import kernel as reduce_kernel
from extract import extract, PIN, ROOT, VENDOR

SHAPES={'up':(17408,5120),'down':(5120,17408),'head':(248320,5120)}
RECIPES=[(16,64,128,1),(16,64,128,2),(16,128,128,1),(16,128,128,2),
         (32,64,128,1),(32,64,128,2),(32,128,128,1),(32,128,128,2),
         (16,32,128,1),(32,32,128,1),(16,64,256,1),(32,64,256,1),
         (16,64,512,1),(32,64,512,1)]

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path,default=ROOT/'.prototypes/out/nvfp4-decode-aot')
    p.add_argument('--shapes',nargs='+',choices=SHAPES,default=list(SHAPES))
    p.add_argument('--recipe',type=int,nargs='*',default=list(range(8)))
    args=p.parse_args();args.output.mkdir(parents=True,exist_ok=True)
    assert subprocess.check_output(['git','-C',str(VENDOR),'rev-parse','HEAD'],text=True).strip()==PIN
    target=parse('{"backend":"cuda","computeCapability":{"major":12,"minor":1},"warpSize":32}',CudaTarget)
    build_triton(str(args.output/'reduction'),reduce_kernel,TritonSpec(precision={'partials':'f32','output':'bf16'},constants={'BLOCK':1024},grid=None,options={'num_warps':4}),target)
    (args.output/'toolchain.json').write_text(json.dumps({'python':platform.python_version(),'packages':{name:importlib.metadata.version(name) for name in ['nvidia-cutlass-dsl','triton','cuda-python']}},indent=2)+'\n')
    records=[]
    for recipe in args.recipe:
        tile=RECIPES[recipe];source=extract(args.output,*tile)
        for shape in args.shapes:
            n,k=SHAPES[shape];splits=tile[-1];name=f'{shape}_r{recipe}'
            precision={'a':'u8','b':'u8','sfa':'fp8_e4m3','sfb':'fp8_e4m3',
                       'output':'f32' if splits>1 else 'bf16','alpha':'f32'}
            spec=CuteSpec(precision=precision,
                alignments={name:16 if name!='alpha' else 4 for name in precision},
                constants={'N':n,'K':k},options={'gpu-arch':'sm_121a','host-target':'linux-aarch64'})
            compile_source(source,spec,target,str(args.output/name))
            obj=(args.output/name).with_suffix('.o').read_bytes()
            assert int.from_bytes(obj[18:20],'little')==183
            records.append(dict(name=name,shape=shape,n=n,k=k,tile=tile[:3],splits=splits,
                                source=str(source),prefix=str(args.output/name)))
            (args.output/'cases.json').write_text(json.dumps(dict(donor=PIN,sm_count=48,cases=records),indent=2)+'\n')
    print(f'{len(records)} AArch64/SM121 exports; GPU correctness and performance UNVERIFIED',flush=True)

if __name__=='__main__':main()
