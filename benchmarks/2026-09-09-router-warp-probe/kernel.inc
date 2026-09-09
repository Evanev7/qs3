__global__ void router_topk_kernel(router_topk_params p)
{
    const uint32_t token = blockIdx.x;
    const uint32_t lane = threadIdx.x;
    constexpr unsigned mask = 0xffffffffu;
    __shared__ float scores[kRouterMaxExperts];
    __shared__ float probabilities[kRouterMaxExperts];
    float max_logit = kRouterNegInf;
    for (uint32_t expert = lane; expert < p.num_experts; expert += 32) {
        const float logit = load_router_logit(p, token, expert);
        scores[expert] = p.score == QSCU_ROUTER_SCORE_SOFTMAX ? logit : sigmoid(logit);
        if (logit > max_logit)
            max_logit = logit;
    }
    for (int delta = 16; delta; delta >>= 1)
        max_logit = fmaxf(max_logit, __shfl_xor_sync(mask, max_logit, delta));
    if (p.score == QSCU_ROUTER_SCORE_SOFTMAX) {
        for (uint32_t expert = lane; expert < p.num_experts; expert += 32)
            probabilities[expert] = expf(scores[expert] - max_logit);
    }
    __syncwarp(mask);
    // Preserve expert-order accumulation, including nonfinite propagation.
    float softmax_denom = 0.0f;
    if (lane == 0 && p.score == QSCU_ROUTER_SCORE_SOFTMAX) {
        for (uint32_t expert = 0; expert < p.num_experts; ++expert)
            softmax_denom += probabilities[expert];
    }
    uint32_t removed = 0;
    int32_t best_ids[kRouterMaxTopK];
    float weights[kRouterMaxTopK];
    float selected_sum = 0.0f;
    for (uint32_t pos = 0; pos < p.top_k; ++pos) {
        float best_score = kRouterNegInf;
        int32_t best_id = INT_MAX;
        for (uint32_t slot = 0; slot < (kRouterMaxExperts + 31) / 32; ++slot) {
            const uint32_t expert = lane + 32 * slot;
            if (expert < p.num_experts && (removed & (1u << slot)) == 0
                && better_router_candidate(scores[expert], expert, best_score, best_id)) {
                best_score = scores[expert];
                best_id = expert;
            }
        }
        for (int delta = 16; delta; delta >>= 1) {
            const float score = __shfl_xor_sync(mask, best_score, delta);
            const int32_t id = __shfl_xor_sync(mask, best_id, delta);
            if (better_router_candidate(score, id, best_score, best_id)) {
                best_score = score;
                best_id = id;
            }
        }
        if (best_id < p.num_experts && static_cast<uint32_t>(best_id) % 32 == lane)
            removed |= 1u << (static_cast<uint32_t>(best_id) / 32);
        if (lane == 0) {
            const bool present = best_id != INT_MAX;
            best_ids[pos] = present ? best_id : static_cast<int32_t>(pos);
            float weight = 0.0f;
            if (present) {
                weight = p.score == QSCU_ROUTER_SCORE_SOFTMAX
                    ? (softmax_denom > 0.0f ? probabilities[best_id] / softmax_denom : 0.0f)
                    : scores[best_id];
            }
            weights[pos] = weight;
            selected_sum += weight;
        }
    }
    if (lane == 0) {
        const float renorm = p.renormalize == 0 ? 1.0f : 1.0f / fmaxf(selected_sum, 1.0e-20f);
        for (uint32_t pos = 0; pos < p.top_k; ++pos) {
            p.topk_ids[static_cast<int64_t>(token) * p.ids_stride0 + static_cast<int64_t>(pos) * p.ids_stride1] = best_ids[pos];
            p.topk_weights[static_cast<int64_t>(token) * p.weights_stride0 + static_cast<int64_t>(pos) * p.weights_stride1] = weights[pos] * renorm * p.routed_scaling_factor;
        }
    }
}
