template <typename StateT>
__device__ float conv_input_at(const conv1d_params& p, const StateT* state,
                              int token, int begin, int read_slot, unsigned dim) {
    if (token >= begin)
        return load_bf16(p.x + int64_t(token) * p.x_stride0 + int64_t(dim) * p.x_stride1);
    if (read_slot < 0) return 0.0f;
    return load_state(state + int64_t(read_slot) * p.state_stride0 + int64_t(dim) * p.state_stride1
                      + int64_t(token - begin + 3) * p.state_stride2);
}

template <typename StateT>
__global__ void conv_prefill_outputs(conv1d_params p, const StateT* state,
                                    unsigned tokens, unsigned batch) {
    const uint64_t count = uint64_t(tokens) * p.conv_dim;
    for (uint64_t i = uint64_t(blockIdx.x) * blockDim.x + threadIdx.x; i < count;
         i += uint64_t(gridDim.x) * blockDim.x) {
        const int token = i / p.conv_dim;
        const unsigned dim = i % p.conv_dim;
        unsigned lo = 0, hi = batch;
        while (lo + 1 < hi) {
            const unsigned mid = (lo + hi) / 2;
            if (p.seq_indptr[mid] <= token) lo = mid; else hi = mid;
        }
        const int begin = p.seq_indptr[lo];
        const int fallback = p.write_indices == nullptr ? -1 : p.write_indices[lo];
        const int read_slot = p.read_indices == nullptr ? fallback : p.read_indices[lo];
        const float h0 = conv_input_at(p, state, token-3, begin, read_slot, dim);
        const float h1 = conv_input_at(p, state, token-2, begin, read_slot, dim);
        const float h2 = conv_input_at(p, state, token-1, begin, read_slot, dim);
        const float x = conv_input_at(p, state, token, begin, read_slot, dim);
        const auto* w = p.weight + int64_t(dim) * p.weight_stride0;
        const float w0 = load_bf16(w), w1 = load_bf16(w+p.weight_stride1);
        const float w2 = load_bf16(w+2*p.weight_stride1), w3 = load_bf16(w+3*p.weight_stride1);
        const float bias = load_optional_bias(
            p.bias_bf16 == nullptr ? nullptr : p.bias_bf16 + int64_t(dim)*p.bias_stride0,
            p.bias_f32 == nullptr ? nullptr : p.bias_f32 + int64_t(dim)*p.bias_stride0, 0);
        const float out = apply_activation(h0*w0+h1*w1+h2*w2+x*w3+bias, p.activation);
        store_bf16(p.out + int64_t(token)*p.out_stride0 + int64_t(dim)*p.out_stride1, out);
    }
}

template <typename StateT>
__global__ void conv_prefill_final_state(conv1d_params p, StateT* state) {
    const unsigned seq = blockIdx.y;
    const unsigned dim = blockIdx.x * blockDim.x + threadIdx.x;
    if (dim >= p.conv_dim) return;
    const int fallback = p.write_indices == nullptr ? -1 : p.write_indices[seq];
    const int read_slot = p.read_indices == nullptr ? fallback : p.read_indices[seq];
    const int write_slot = p.write_indices == nullptr ? read_slot : p.write_indices[seq];
    if (write_slot < 0) return;
    const int begin=p.seq_indptr[seq], end=p.seq_indptr[seq+1];
    // Load all initial values before writing: read and write may be the same slot.
    const float h0=conv_input_at(p,state,end-3,begin,read_slot,dim);
    const float h1=conv_input_at(p,state,end-2,begin,read_slot,dim);
    const float h2=conv_input_at(p,state,end-1,begin,read_slot,dim);
    auto* out=state+int64_t(write_slot)*p.state_stride0+int64_t(dim)*p.state_stride1;
    out[0]=make_state<StateT>(h0);
    out[p.state_stride2]=make_state<StateT>(h1);
    out[2*p.state_stride2]=make_state<StateT>(h2);
}

template <typename StateT>
void launch_parallel(conv1d_params p, StateT* state, unsigned tokens, unsigned batch) {
    const unsigned blocks=std::min<uint64_t>((uint64_t(tokens)*p.conv_dim+255)/256,4096);
    conv_prefill_outputs<<<blocks,256>>>(p,state,tokens,batch);
    if (p.update_state)
        conv_prefill_final_state<<<dim3((p.conv_dim+255)/256,batch),256>>>(p,state);
}
