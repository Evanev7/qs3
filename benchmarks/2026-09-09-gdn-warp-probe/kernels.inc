__device__ float ordered_warp_sum4(float a, float b, float c, float d) {
    // Match the original 128-thread shared-memory tree: strides 64, 32, then 16..1.
    float sum = __fadd_rn(__fadd_rn(a, c), __fadd_rn(b, d));
    for (unsigned step=16;step;step>>=1)
        sum = __fadd_rn(sum,__shfl_down_sync(0xffffffffu,sum,step));
    return __shfl_sync(0xffffffffu,sum,0);
}

template<class S, bool Prefill>
__global__ void gdn_warp_kernel(gdn_kernel_params p, S* state) {
    const unsigned lane=threadIdx.x%32;
    const uint64_t linear=uint64_t(blockIdx.x)*4+threadIdx.x/32;
    if(linear>=uint64_t(p.outer_count)*p.num_v_heads*p.value_dim)return;
    const unsigned dim=linear%p.value_dim;
    const unsigned head=(linear/p.value_dim)%p.num_v_heads;
    const unsigned seq=linear/(p.value_dim*p.num_v_heads);
    const int begin=Prefill?p.seq_indptr[seq]:int(seq);
    const int end=Prefill?p.seq_indptr[seq+1]:int(seq+1);
    const int read=p.state_indices[seq];
    const int write=p.state_out_indices==nullptr?read:p.state_out_indices[seq];
    if(read<0) {
        if(lane==0)for(int token=begin;token<end;++token)
            store_bf16(p.out.data+int64_t(token)*p.out.stride0+int64_t(head)*p.out.stride1+int64_t(dim)*p.out.stride2,0.f);
        return;
    }
    const unsigned qhead=mapped_head(head,p.num_v_heads,p.num_q_heads);
    const unsigned khead=mapped_head(head,p.num_v_heads,p.num_k_heads);
    const int64_t base=int64_t(read)*p.state.stride0+int64_t(head)*p.state.stride1+int64_t(dim)*p.state.stride2;
    float h[4];
    #pragma unroll
    for(unsigned j=0;j<4;++j)h[j]=load_state_value(state+base+int64_t(lane+32*j)*p.state.stride3);
    for(int token=begin;token<end;++token) {
        float qr[4],kr[4],q[4],k[4];
        #pragma unroll
        for(unsigned j=0;j<4;++j) {
            qr[j]=load_bf16(p.q.data+int64_t(token)*p.q.stride0+int64_t(qhead)*p.q.stride1+int64_t(lane+32*j)*p.q.stride2);
            kr[j]=load_bf16(p.k.data+int64_t(token)*p.k.stride0+int64_t(khead)*p.k.stride1+int64_t(lane+32*j)*p.k.stride2);
        }
        float qf=p.scale,kf=1.f;
        if(p.use_qk_l2norm) {
            float qn=ordered_warp_sum4(qr[0]*qr[0],qr[1]*qr[1],qr[2]*qr[2],qr[3]*qr[3]);
            float kn=ordered_warp_sum4(kr[0]*kr[0],kr[1]*kr[1],kr[2]*kr[2],kr[3]*kr[3]);
            qf*=rsqrtf(qn+1.e-8f);kf*=rsqrtf(kn+1.e-8f);
        }
        const float a=load_bf16(p.a.data+int64_t(token)*p.a.stride0+int64_t(head)*p.a.stride1);
        const float b=load_bf16(p.b.data+int64_t(token)*p.b.stride0+int64_t(head)*p.b.stride1);
        const float alog=load_bf16(p.a_log.data+int64_t(head)*p.a_log.stride0);
        const float dt=load_bf16(p.dt_bias.data+int64_t(head)*p.dt_bias.stride0);
        const float g=-expf(alog)*softplus(a+dt,QSFI_QWEN36_GDN_SOFTPLUS_BETA,QSFI_QWEN36_GDN_SOFTPLUS_THRESHOLD);
        const float beta=1.f/(1.f+expf(-b));
        #pragma unroll
        for(unsigned j=0;j<4;++j){q[j]=qr[j]*qf;k[j]=kr[j]*kf;h[j]*=expf(g);}
        const float dot=ordered_warp_sum4(k[0]*h[0],k[1]*h[1],k[2]*h[2],k[3]*h[3]);
        const float v=load_bf16(p.v.data+int64_t(token)*p.v.stride0+int64_t(head)*p.v.stride1+int64_t(dim)*p.v.stride2);
        const float delta=(v-dot)*beta;
        #pragma unroll
        for(unsigned j=0;j<4;++j)h[j]+=k[j]*delta;
        const float out=ordered_warp_sum4(q[0]*h[0],q[1]*h[1],q[2]*h[2],q[3]*h[3]);
        if(lane==0)store_bf16(p.out.data+int64_t(token)*p.out.stride0+int64_t(head)*p.out.stride1+int64_t(dim)*p.out.stride2,out);
    }
    if(!p.disable_state_update&&write>=0) {
        const int64_t dest=int64_t(write)*p.state.stride0+int64_t(head)*p.state.stride1+int64_t(dim)*p.state.stride2;
        #pragma unroll
        for(unsigned j=0;j<4;++j)state[dest+int64_t(lane+32*j)*p.state.stride3]=make_state_value<S>(h[j]);
    }
}
