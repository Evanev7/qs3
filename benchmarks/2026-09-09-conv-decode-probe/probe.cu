#include "../../qscu.cu"
#include "kernels.inc"
#include <cstdio>
#include <vector>
#include <cstdlib>
void ck(cudaError_t s) { if(s!=cudaSuccess) { fprintf(stderr,"%s\n",cudaGetErrorString(s));exit(1); } }
template<class T> T* upload(const std::vector<T>& v) { T* p;ck(cudaMalloc(&p,v.size()*sizeof(T)));ck(cudaMemcpy(p,v.data(),v.size()*sizeof(T),cudaMemcpyHostToDevice));return p; }
template<class T> std::vector<T> download(T* p,size_t n) {std::vector<T> v(n);ck(cudaMemcpy(v.data(),p,n*sizeof(T),cudaMemcpyDeviceToHost));return v;}
template<class F> float bench(F f) { for(int i=0;i<10;++i)f();cudaEvent_t a,b;ck(cudaEventCreate(&a));ck(cudaEventCreate(&b));ck(cudaEventRecord(a));for(int i=0;i<500;++i)f();ck(cudaEventRecord(b));ck(cudaEventSynchronize(b));float ms;ck(cudaEventElapsedTime(&ms,a,b));ck(cudaEventDestroy(a));ck(cudaEventDestroy(b));return ms/500; }
template<class S> void run(unsigned dim,unsigned batch,bool same,bool update,bool negative,bool inplace,unsigned act,unsigned bias_kind,bool timing) {
 const unsigned slots=2*batch;
 std::vector<__nv_bfloat16> x(size_t(batch)*dim),w(size_t(dim)*4),bias(dim),z(x.size());std::vector<float> bias32(dim);
 std::vector<S> initial(size_t(slots)*dim*3);
 for(size_t i=0;i<x.size();++i)x[i]=__float2bfloat16(float(int(i%37)-18)*.03125f);
 for(size_t i=0;i<w.size();++i)w[i]=__float2bfloat16(float(int(i%13)-6)*.0625f);
 for(size_t i=0;i<dim;++i){bias32[i]=float(int(i%11)-5)*.041f;bias[i]=__float2bfloat16(bias32[i]);}
 for(size_t i=0;i<initial.size();++i)initial[i]=S(float(int(i%31)-15)*.021f);
 std::vector<int32_t> reads(batch),writes(batch);for(unsigned i=0;i<batch;++i){reads[i]=negative&&i==0?-1:int(i);writes[i]=same?i:batch+i;}
 auto* xa=upload(x);auto* xb=upload(x);auto* dw=upload(w);auto* db=upload(bias);auto* db32=upload(bias32);auto* a=upload(z);auto* b=upload(z);
 auto* sa=upload(initial);auto* sb=upload(initial);auto* dr=upload(reads);auto* ds=upload(writes);
 conv1d_params p{};p.x=xa;p.x_stride0=dim;p.x_stride1=1;p.weight=dw;p.weight_stride0=4;p.weight_stride1=1;
 p.bias_bf16=bias_kind==1?db:nullptr;p.bias_f32=bias_kind==2?db32:nullptr;p.bias_stride0=1;
 p.state_stride0=dim*3;p.state_stride1=3;p.state_stride2=1;p.read_indices=dr;p.write_indices=ds;
 p.out=inplace?xa:a;p.out_stride0=dim;p.out_stride1=1;p.conv_dim=dim;p.activation=static_cast<qscu_activation>(act);p.update_state=update;
 qwen36_gdn_causal_conv1d_kernel<<<batch,256>>>(p,sa);
 p.x=xb;p.out=inplace?xb:b;conv_decode_tiled<<<dim3(batch,(dim+255)/256),256>>>(p,sb);ck(cudaDeviceSynchronize());
 auto ha=download(inplace?xa:a,x.size()),hb=download(inplace?xb:b,x.size());auto hsa=download(sa,initial.size()),hsb=download(sb,initial.size());
 if(memcmp(ha.data(),hb.data(),ha.size()*sizeof(ha[0]))||memcmp(hsa.data(),hsb.data(),hsa.size()*sizeof(S))) {fprintf(stderr,"mismatch dim=%u batch=%u same=%d update=%d negative=%d inplace=%d act=%u bias=%u state=%zu\n",dim,batch,same,update,negative,inplace,act,bias_kind,sizeof(S));exit(1);}
 if(timing){p.x=xa;p.out=a;float old=bench([&]{qwen36_gdn_causal_conv1d_kernel<<<batch,256>>>(p,sa);});p.x=xb;p.out=b;float next=bench([&]{conv_decode_tiled<<<dim3(batch,(dim+255)/256),256>>>(p,sb);});printf("%u\t%zu\t%.6f\t%.6f\n",dim,sizeof(S),old,next);}
 for(void* ptr:{(void*)xa,(void*)xb,(void*)dw,(void*)db,(void*)db32,(void*)a,(void*)b,(void*)sa,(void*)sb,(void*)dr,(void*)ds})ck(cudaFree(ptr));
}
int main(){unsigned cases=0;puts("channels\tstate_bytes\tserial_ms\ttiled_ms");for(unsigned dim:{8192,10240})for(unsigned batch:{1,3})for(bool same:{false,true})for(bool update:{false,true})for(bool negative:{false,true})for(bool inplace:{false,true})for(unsigned act:{QSCU_ACTIVATION_NONE,QSCU_ACTIVATION_SILU})for(unsigned bias:{0,1,2}){bool timing=batch==1&&!same&&update&&!negative&&!inplace&&act==QSCU_ACTIVATION_SILU&&bias==0;run<__nv_bfloat16>(dim,batch,same,update,negative,inplace,act,bias,timing);run<float>(dim,batch,same,update,negative,inplace,act,bias,timing);cases+=2;}printf("%u bitwise output/state cases passed\n",cases);}
