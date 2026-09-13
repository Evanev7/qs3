import shlex, subprocess
from pathlib import Path
root=Path.cwd(); out=root/'.prototypes/moe_decode'; out.mkdir(exist_ok=True)
names=['qsfi_moe_plan_create','qsfi_moe_plan_destroy','qsfi_moe_workspace_size','qsfi_moe_execute_bf16','qsfi_moe_execute_nvfp4']
(out/'candidate_names.h').write_text('extern int candidate;\n'+''.join(f'#define {n} probe_{n}\n' for n in names))
s='#include "candidate_names.h"\n'+(root/'qsfi_moe.cu').read_text()
pos=s.index('cudaError_t launch_bf16_moe(')
s=s[:pos]+'#include "kernels.cuh"\n'+s[pos:]
needle='    const uint32_t num_tokens = desc->num_tokens;'
s=s.replace(needle,'''    if(desc->num_tokens==1 && candidate) {
        switch(candidate) {
        case 32:return launch_decode_moe<32>(desc,ws,ctx->stream);
        case 64:return launch_decode_moe<64>(desc,ws,ctx->stream);
        case 128:return launch_decode_moe<128>(desc,ws,ctx->stream);
        case 256:return launch_decode_moe<256>(desc,ws,ctx->stream);
        }
    }
'''+needle)
(out/'candidate.cu').write_text(s)
s='#include "candidate_names.h"\n#include <cuda_bf16.h>\n#include <cmath>\n#include <algorithm>\n'+(root/'bench_native.cu').read_text()
s=s[:s.index('int main(')]
a=s.index('bool bench_moe_execute_bf16(');b=s.index('\nbool ',a+1)
body=s[a:b]
helper='''
__global__ void fill_probe_values(uint16_t* data,size_t n,unsigned seed) {
    for(size_t i=size_t(blockIdx.x)*blockDim.x+threadIdx.x;i<n;i+=size_t(blockDim.x)*gridDim.x) {
        unsigned h=unsigned(i)^seed;h^=h>>16;h*=0x7feb352d;h^=h>>15;h*=0x846ca68b;h^=h>>16;
        reinterpret_cast<__nv_bfloat16*>(data)[i]=__float2bfloat16(float(int(h%1001)-500)*0.0003f);
    }
}
bool fill_probe(DeviceBuffer<uint16_t>& data,size_t n,unsigned seed,cudaStream_t stream) {
    fill_probe_values<<<4096,256,0,stream>>>(data.ptr,n,seed);return check_cuda(cudaGetLastError(),"fill probe");
}
'''
for old,new in [('hidden.zero(state.stream, "moe hidden")','fill_probe(hidden,size_t(tokens)*kHidden,3,state.stream)'),('gate_up_weight.zero(state.stream, "moe gate_up_weight")','fill_probe(gate_up_weight,size_t(kMoeExperts)*2*kMoeIntermediate*kHidden,7,state.stream)'),('down_weight.zero(state.stream, "moe down_weight")','fill_probe(down_weight,size_t(kMoeExperts)*kHidden*kMoeIntermediate,11,state.stream)')]:
    assert old in body;body=body.replace(old,new)
body=body.replace('route % kMoeExperts','(route*31+17) % kMoeExperts')
pos=body.index('    BenchRow row ')
body=body[:pos]+'''
    std::vector<uint16_t> reference(kHidden),actual(kHidden);
    candidate=0;
    if(qsfi_moe_execute_bf16(state.qsfi,plan.ptr,&desc)!=QSFI_STATUS_OK)return false;
    if(!check_cuda(cudaMemcpy(reference.data(),out.ptr,kHidden*2,cudaMemcpyDeviceToHost),"ref"))return false;
    for(int repeat=0;repeat<3;++repeat)for(int variant:{0,32,64,128,256}) {
    candidate=variant;
    if(qsfi_moe_execute_bf16(state.qsfi,plan.ptr,&desc)!=QSFI_STATUS_OK)return false;
    if(!check_cuda(cudaMemcpy(actual.data(),out.ptr,kHidden*2,cudaMemcpyDeviceToHost),"actual"))return false;
    double sq=0,ref_sq=0;float maxerr=0;size_t diff=0;
    for(size_t i=0;i<actual.size();++i) {
        float x=__bfloat162float(reinterpret_cast<const __nv_bfloat16*>(actual.data())[i]);
        float y=__bfloat162float(reinterpret_cast<const __nv_bfloat16*>(reference.data())[i]);
        if(!std::isfinite(x)||!std::isfinite(y))return false;
        maxerr=std::max(maxerr,std::fabs(x-y));sq+=(x-y)*(x-y);ref_sq+=y*y;diff+=actual[i]!=reference[i];
    }
    std::printf("repeat=%d variant=%d diff=%zu max_abs=%g relative_rms=%g ",repeat,candidate,diff,maxerr,sqrt(sq/ref_sq));
    if(sqrt(sq/ref_sq)>0.01)return false;
'''+body[pos:]
body=body.replace('    return run_timed(','    if(!run_timed(')
body=body.replace('    );\n}', '    ))return false;\n    }\n    return true;\n}')
s=s[:a]+helper+body+s[b:]
s+='\nint candidate=0;\nint main(){Options options{};options.warmups=5;options.iters=100;BenchState state{};if(!create_state(&state))return 1;bool ok=bench_moe_execute_bf16(state,options,1);destroy_state(&state);return ok?0:1;}\n'
(out/'probe.cu').write_text(s)
cmd=shlex.split(subprocess.check_output(['ninja','-t','commands','qsfi.o'],cwd=root/'build',text=True).splitlines()[-1]);cmd[0]='/usr/local/cuda/bin/nvcc';cmd.insert(1,'-I'+str(root));cmd[cmd.index('-c')+1]=str(out/'candidate.cu');cmd[cmd.index('-o')+1]=str(out/'candidate.o');subprocess.run(cmd,cwd=root/'build',check=True)
subprocess.run(['/usr/local/cuda/bin/nvcc','-std=c++17','-arch=sm_121','--expt-relaxed-constexpr','-diag-suppress','177','-I'+str(root),'-I'+str(root/'build'),'-DQSFI_ENABLE_CHECKED_VALIDATION=0',str(out/'probe.cu'),str(out/'candidate.o'),*[str(root/'build'/x) for x in ['qsfi.o','qscu_gdn.o','qscu.o','qscb.o']],'-lcuda','-lcublas','-lcublasLt','-o',str(out/'probe')],check=True)
subprocess.run([str(out/'probe')],check=True)
