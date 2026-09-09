from pathlib import Path
import shlex,subprocess
root=Path.cwd();out=root/'.prototypes/moe_tiles';out.mkdir(parents=True,exist_ok=True)
names=['qsfi_moe_plan_create','qsfi_moe_plan_destroy','qsfi_moe_workspace_size','qsfi_moe_execute_bf16','qsfi_moe_execute_nvfp4']
(out/'candidate_names.h').write_text('extern int qs3_moe_tile_m;\nextern unsigned qs3_moe_route_experts;\n'+''.join(f'#define {n} probe_{n}\n' for n in names))
s=(root/'qsfi_moe.cu').read_text();s='#include "candidate_names.h"\n'+s
s=s.replace('cudaError_t launch_grouped_bf16(', 'template <int TileM, int TileK>\ncudaError_t launch_grouped_bf16_impl(').replace('GemmShape<128, 128, 32>','GemmShape<TileM, 128, TileK>').replace('GemmShape<64, 64, 32>','GemmShape<(TileM < 64 ? TileM : 64), 64, TileK>')
pos=s.index('\nsize_t align_up(');s=s[:pos]+'''
cudaError_t launch_grouped_bf16(
    const moe_workspace& ws, uint32_t experts, uint32_t blocks, cudaStream_t stream
) {
    switch (qs3_moe_tile_m) {
    case 128: return launch_grouped_bf16_impl<128,32>(ws,experts,blocks,stream);
    case 32: return launch_grouped_bf16_impl<32,64>(ws,experts,blocks,stream);
    case 16: return launch_grouped_bf16_impl<16,64>(ws,experts,blocks,stream);
    default: throw std::runtime_error("invalid probe tile");
    }
}
'''+s[pos:];(out/'candidate.cu').write_text(s)
s=(root/'bench_native.cu').read_text();s='#include "candidate_names.h"\n#include <cuda_bf16.h>\n#include <cmath>\n#include <algorithm>\n'+s
s=s[:s.index('int main(')]
pos=s.index('bool bench_moe_execute_bf16(')
helper='''
__global__ void fill_probe_values(uint16_t* data, size_t n, unsigned seed) {
    for(size_t i=size_t(blockIdx.x)*blockDim.x+threadIdx.x;i<n;i+=size_t(blockDim.x)*gridDim.x)
        reinterpret_cast<__nv_bfloat16*>(data)[i]=__float2bfloat16(float(int((i*17+seed)%97)-48)*0.00371f);
}
bool fill_probe(DeviceBuffer<uint16_t>& data, size_t n, unsigned seed, cudaStream_t stream) {
    fill_probe_values<<<4096,256,0,stream>>>(data.ptr,n,seed);
    return check_cuda(cudaGetLastError(),"fill probe values");
}
'''
s=s[:pos]+helper+s[pos:];a=s.index('bool bench_moe_execute_bf16(');b=s.index('\nbool ',a+1);body=s[a:b]
for old,new in [
 ('hidden.zero(state.stream, "moe hidden")','fill_probe(hidden, size_t(tokens)*kHidden, 3, state.stream)'),
 ('gate_up_weight.zero(state.stream, "moe gate_up_weight")','fill_probe(gate_up_weight,size_t(kMoeExperts)*2*kMoeIntermediate*kHidden,7,state.stream)'),
 ('down_weight.zero(state.stream, "moe down_weight")','fill_probe(down_weight,size_t(kMoeExperts)*kHidden*kMoeIntermediate,11,state.stream)')]:
 assert old in body;body=body.replace(old,new)
body=body.replace('route % kMoeExperts','route % qs3_moe_route_experts')
pos=body.index('    BenchRow row ')
body=body[:pos]+'''
    const int candidate = qs3_moe_tile_m;
    std::vector<uint16_t> reference(size_t(tokens)*kHidden), actual(reference.size());
    qs3_moe_tile_m=128;
    if (qsfi_moe_execute_bf16(state.qsfi,plan.ptr,&desc)!=QSFI_STATUS_OK) return false;
    if (!check_cuda(cudaMemcpy(reference.data(),out.ptr,reference.size()*2,cudaMemcpyDeviceToHost),"reference output")) return false;
    qs3_moe_tile_m=candidate;
    if (qsfi_moe_execute_bf16(state.qsfi,plan.ptr,&desc)!=QSFI_STATUS_OK) return false;
    if (!check_cuda(cudaMemcpy(actual.data(),out.ptr,actual.size()*2,cudaMemcpyDeviceToHost),"candidate output")) return false;
    size_t differences=0,nonzero=0;float max_error=0;
    for(size_t i=0;i<actual.size();++i) {
        differences+=actual[i]!=reference[i];nonzero+=(reference[i]&0x7fff)!=0;
        float x=__bfloat162float(reinterpret_cast<const __nv_bfloat16*>(actual.data())[i]);
        float y=__bfloat162float(reinterpret_cast<const __nv_bfloat16*>(reference.data())[i]);
        if (!std::isfinite(x) || !std::isfinite(y)) return false;
        max_error=std::max(max_error,std::fabs(x-y));
    }
    std::fprintf(stderr,"tile=%d tokens=%u differing_bits=%zu nonzero=%zu max_abs=%g\\n",candidate,tokens,differences,nonzero,max_error);
    if (differences || !nonzero) return false;
'''+body[pos:];s=s[:a]+body+s[b:]
s+='''
int qs3_moe_tile_m=128;
unsigned qs3_moe_route_experts=256;
int main() {
    Options options{}; options.warmups=5;
    BenchState state{};if(!create_state(&state))return 1;
    for(int repeat=0;repeat<2;++repeat)for(unsigned active_experts:{8u,32u,256u})for(unsigned tokens:{1u,16u,102u,1024u})for(int tile:{128,32,16}) {
        qs3_moe_tile_m=tile;qs3_moe_route_experts=active_experts;options.iters=tokens>=102?30:100;
        std::printf("repeat=%d active_experts=%u tile=%d ",repeat,active_experts,tile);std::fflush(stdout);
        if(!bench_moe_execute_bf16(state,options,tokens)){destroy_state(&state);return 1;}
        std::fflush(stdout);
    }
    destroy_state(&state);return 0;
}
''';(out/'probe.cu').write_text(s)
cmd=shlex.split(subprocess.check_output(['ninja','-t','commands','qsfi.o'],cwd=root/'build',text=True).splitlines()[-1]);cmd[0]='/usr/local/cuda/bin/nvcc';cmd.insert(1,'-I'+str(root));cmd[cmd.index('-c')+1]=str(out/'candidate.cu');cmd[cmd.index('-o')+1]=str(out/'candidate.o');subprocess.run(cmd,cwd=root/'build',check=True)
flags=['/usr/local/cuda/bin/nvcc','-std=c++17','-arch=sm_121','--expt-relaxed-constexpr','-diag-suppress','177','-I'+str(root),'-I'+str(root/'build'),'-DQSFI_ENABLE_CHECKED_VALIDATION=0']
subprocess.run(flags+[str(out/'probe.cu'),str(out/'candidate.o'),*map(str,[root/'build'/x for x in ['qsfi.o','qscu_gdn.o','qscu.o','qscb.o']]),'-lcuda','-lcublas','-lcublasLt','-o',str(out/'probe')],check=True)
print('MoE tile probe built; execute separately.',flush=True)
