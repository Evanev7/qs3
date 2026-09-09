template <typename StateT>
__global__ void conv_decode_tiled(conv1d_params p, StateT* state)
{
    const uint32_t seq = blockIdx.x;
    const int32_t token_begin
        = p.seq_indptr == nullptr ? static_cast<int32_t>(seq) : p.seq_indptr[seq];
    const int32_t token_end
        = p.seq_indptr == nullptr ? static_cast<int32_t>(seq + 1) : p.seq_indptr[seq + 1];
    const int32_t fallback_slot = p.write_indices == nullptr ? -1 : p.write_indices[seq];
    const int32_t read_slot = p.read_indices == nullptr ? fallback_slot : p.read_indices[seq];
    const int32_t write_slot = p.update_state == 0
        ? -1
        : (p.write_indices == nullptr ? read_slot : p.write_indices[seq]);

    for (uint32_t dim = blockIdx.y * blockDim.x + threadIdx.x;
         dim < p.conv_dim; dim += blockDim.x * gridDim.y) {
        float h0 = 0.0f;
        float h1 = 0.0f;
        float h2 = 0.0f;
        if (read_slot >= 0) {
            const int64_t base = static_cast<int64_t>(read_slot) * p.state_stride0
                + static_cast<int64_t>(dim) * p.state_stride1;
            h0 = load_state(state + base);
            h1 = load_state(state + base + p.state_stride2);
            h2 = load_state(state + base + 2 * p.state_stride2);
        }

        const float w0 = load_bf16(p.weight + static_cast<int64_t>(dim) * p.weight_stride0);
        const float w1
            = load_bf16(p.weight + static_cast<int64_t>(dim) * p.weight_stride0 + p.weight_stride1);
        const float w2 = load_bf16(
            p.weight + static_cast<int64_t>(dim) * p.weight_stride0 + 2 * p.weight_stride1
        );
        const float w3 = load_bf16(
            p.weight + static_cast<int64_t>(dim) * p.weight_stride0 + 3 * p.weight_stride1
        );
        const float bias = load_optional_bias(
            p.bias_bf16 == nullptr ? nullptr
                                   : p.bias_bf16 + static_cast<int64_t>(dim) * p.bias_stride0,
            p.bias_f32 == nullptr ? nullptr
                                  : p.bias_f32 + static_cast<int64_t>(dim) * p.bias_stride0,
            0
        );

        for (int32_t token = token_begin; token < token_end; ++token) {
            const float x_value = load_bf16(
                p.x + static_cast<int64_t>(token) * p.x_stride0
                + static_cast<int64_t>(dim) * p.x_stride1
            );
            float out_value = h0 * w0 + h1 * w1 + h2 * w2 + x_value * w3 + bias;
            out_value = apply_activation(out_value, p.activation);
            store_bf16(
                p.out + static_cast<int64_t>(token) * p.out_stride0
                    + static_cast<int64_t>(dim) * p.out_stride1,
                out_value
            );
            h0 = h1;
            h1 = h2;
            h2 = x_value;
        }

        if (write_slot >= 0) {
            const int64_t base = static_cast<int64_t>(write_slot) * p.state_stride0
                + static_cast<int64_t>(dim) * p.state_stride1;
            state[base] = make_state<StateT>(h0);
            state[base + p.state_stride2] = make_state<StateT>(h1);
            state[base + 2 * p.state_stride2] = make_state<StateT>(h2);
        }
    }
}
