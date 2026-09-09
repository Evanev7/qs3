"""Build AOT GDN tests for 27B geometry without changing production dispatch."""
from pathlib import Path
import subprocess
root=Path.cwd();out=root/'.prototypes/qwen27_gdn';out.mkdir(parents=True,exist_ok=True)
for name,old,new in [
 ('qscu.cu','kQwen36GdnNumVHeads = QSFI_QWEN36_GDN_NUM_V_HEADS','kQwen36GdnNumVHeads = 48'),
 ('qscu_gdn.cu','kDefaultNumVHeads = QSFI_QWEN36_GDN_NUM_V_HEADS','kDefaultNumVHeads = 48')]:
 s=(root/name).read_text();assert s.count(old)==1;s=s.replace(old,new);(out/name).write_text(s)
s=(root/'test.cu').read_text().replace('constexpr int kGdnVHeads = 32;', 'constexpr int kGdnVHeads = 48;').replace('constexpr int kGdnActiveVHead = 3;', 'constexpr int kGdnActiveVHead = 47;').replace('kGdnActiveVHead / 2;', 'kGdnActiveVHead / 3;')
s=s[:s.index('int main()')]
# Reuse the analytic recurrence oracle with FP32 state as well as BF16.
extra='''
void check_float_state(const std::vector<float>& state, size_t active, float value,
                       float tolerance, const char* label) {
    for (size_t i=0;i<state.size();++i) {
        float expected=i==active?value:0.0f;
        if (std::fabs(state[i]-expected)>tolerance || !std::isfinite(state[i])) {
            std::fprintf(stderr,"FAIL: %s[%zu] got %g expected %g\\n",label,i,state[i],expected);
            ++failures; return;
        }
    }
}
'''
for name in ['test_gdn_decode_one_hot_recurrence','test_gdn_prefill_two_token_recurrence']:
 a=s.index('void '+name+'()');b=s.index('\nvoid ',a+1);body=s[a:b]
 body=body.replace(name,name+'_f32').replace('std::vector<uint16_t> h_state(state_elems, kBf16Zero);','std::vector<float> h_state(state_elems, 0.0f);').replace('uint16_t* d_state = nullptr;','float* d_state = nullptr;')
 assert body.count('desc.state = gdn_state_tensor_bf16(d_state);')==1
 body=body.replace('desc.state = gdn_state_tensor_bf16(d_state);','desc.state = gdn_state_tensor_bf16(d_state);\n    desc.state.dtype = QSFI_DTYPE_F32;')
 body=body.replace('h_state.size() * sizeof(uint16_t)', 'h_state.size() * sizeof(float)').replace('check_bf16_single_nonzero(\n        h_state,','check_float_state(\n        h_state,')
 extra+=body+'\n'
s+=extra+'''
int main() {
    if (!check_cuda(cudaSetDevice(0),"select device")) return 1;
    test_gdn_decode_one_hot_recurrence();
    test_gdn_prefill_two_token_recurrence();
    test_gdn_decode_one_hot_recurrence_f32();
    test_gdn_prefill_two_token_recurrence_f32();
    test_qscu_qwen36_gdn_causal_conv1d_bf16_cpu_reference();
    test_qscu_qwen36_gdn_post_conv_prepare_bf16_cpu_reference();
    test_qscu_qwen36_gdn_rmsnorm_gated_bf16_cpu_reference();
    test_qscu_gdn_router_validation_errors();
#if QSFI_ENABLE_CHECKED_VALIDATION
    test_checked_gdn_decode_rejects_invalid_state_index();
    test_checked_gdn_prefill_rejects_invalid_seq_indptr();
    test_qscu_qwen36_gdn_causal_conv1d_bf16_checked_validation();
#endif
    if (failures) return 1;
    std::puts("27B GDN: 16Q/16K/48V/128, conv 10240, BF16/FP32 recurrence and CPU-reference prep tests passed");
    return 0;
}
'''
(out/'test.cu').write_text(s)
inc=(root/'tests_cuda_qscu_gdn_router.inc').read_text().replace('constexpr int kVHead = 5;', 'constexpr int kVHead = 47;')
(out/'tests_cuda_qscu_gdn_router.inc').write_text(inc)
for checked in [0,1]:
 flags=['/usr/local/cuda/bin/nvcc','-std=c++17','-arch=sm_121','--expt-relaxed-constexpr','-diag-suppress','20012','-I'+str(root),'-I'+str(root/'build'),f'-DQSFI_ENABLE_CHECKED_VALIDATION={checked}']
 objects=[]
 for name in ['qscu','qscu_gdn']:
  obj=str(out/f'{name}_{checked}.o');subprocess.run(flags+['-c',str(out/(name+'.cu')),'-o',obj],check=True);objects.append(obj)
 suffix='_checked' if checked else ''
 binary=str(out/f'probe_{checked}')
 subprocess.run(flags+[str(out/'test.cu'),*objects,str(root/f'build/qsfi{suffix}.o'),str(root/'build/qscb.o'),'-lcuda','-lcublas','-lcublasLt','-o',binary],check=True)
 subprocess.run([binary],check=True)
