// Single-token, Qwen3.6-35B BF16 routed experts. Prototype only.
template<int Threads>
__device__ float moe_sum(float x) {
    for(int d=16;d;d/=2) x+=__shfl_down_sync(0xffffffff,x,d);
    if constexpr(Threads>32) {
        __shared__ float partial[Threads/32];
        if(threadIdx.x%32==0) partial[threadIdx.x/32]=x;
        __syncthreads();
        if(threadIdx.x<32) {
            x=threadIdx.x<Threads/32?partial[threadIdx.x]:0.f;
            for(int d=16;d;d/=2) x+=__shfl_down_sync(0xffffffff,x,d);
        }
        __syncthreads();
    }
    return x;
}

template<int K,int N,int Threads,bool RoutedInput>
__global__ void moe_decode_gemv(const __nv_bfloat16* x,const __nv_bfloat16* w,
    const int32_t* ids,__nv_bfloat16* y) {
    const int row=blockIdx.x;
    const int route=blockIdx.y;
    const int expert=ids[route];
    if(expert<0 || expert>=256) {
        if(threadIdx.x==0) y[route*N+row]=__float2bfloat16(0.f);
        return;
    }
    x+=RoutedInput?route*K:0;
    w+=(size_t(expert)*N+row)*K;
    float sums[8]={};
    for(int k=threadIdx.x*8;k<K;k+=Threads*8) {
        uint4 xv=*reinterpret_cast<const uint4*>(x+k);
        uint4 wv=*reinterpret_cast<const uint4*>(w+k);
        const auto* xx=reinterpret_cast<const __nv_bfloat16*>(&xv);
        const auto* ww=reinterpret_cast<const __nv_bfloat16*>(&wv);
        #pragma unroll
        for(int j=0;j<8;++j) sums[j]+=__bfloat162float(xx[j])*__bfloat162float(ww[j]);
    }
    float acc=0.f;
    #pragma unroll
    for(int j=0;j<8;++j) acc+=sums[j];
    acc=moe_sum<Threads>(acc);
    if(threadIdx.x==0) y[route*N+row]=__float2bfloat16(acc);
}

__global__ void moe_decode_finalize(const float* scales,const __nv_bfloat16* expert_out,__nv_bfloat16* out) {
    int h=blockIdx.x*blockDim.x+threadIdx.x;
    float acc=0.f;
    #pragma unroll
    for(int r=0;r<8;++r) acc+=scales[r]*__bfloat162float(expert_out[r*2048+h]);
    out[h]=__float2bfloat16(acc);
}

template<int Threads>
cudaError_t launch_decode_moe(const qsfi_moe_bf16_execute_desc* desc,const moe_workspace& ws,cudaStream_t stream) {
    moe_decode_gemv<2048,1024,Threads,false><<<dim3(1024,8),Threads,0,stream>>>(
        static_cast<const __nv_bfloat16*>(desc->hidden.data),static_cast<const __nv_bfloat16*>(desc->gate_up_weight.data),
        static_cast<const int32_t*>(desc->topk_ids.data),ws.gemm1_out);
    auto err=cudaGetLastError(); if(err!=cudaSuccess)return err;
    swiglu_kernel<<<16,256,0,stream>>>(ws.gemm1_out,8,512,ws.act);
    err=cudaGetLastError(); if(err!=cudaSuccess)return err;
    moe_decode_gemv<512,2048,Threads,true><<<dim3(2048,8),Threads,0,stream>>>(
        ws.act,static_cast<const __nv_bfloat16*>(desc->down_weight.data),
        static_cast<const int32_t*>(desc->topk_ids.data),ws.gemm2_out);
    err=cudaGetLastError(); if(err!=cudaSuccess)return err;
    moe_decode_finalize<<<8,256,0,stream>>>(static_cast<const float*>(desc->topk_weights.data),ws.gemm2_out,static_cast<__nv_bfloat16*>(desc->out.data));
    return cudaGetLastError();
}
