# Triton JIT call/annotation types are not Python typing types.
# ty: ignore[invalid-argument-type, invalid-type-form]
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# Adapted from vLLM 51a99565c398c8320de8131e07731c75c52eb87c,
# vllm/v1/sample/ops/topk_topp_triton.py. See LICENSE.vllm.
# Changes: AOT entrypoint/annotations; one row; runtime k/p scalars;
# always-enabled top-k/top-p compile flags folded away.
# Pivot search is unchanged. Uniform finite rows use exact count-based filtering
# instead of upstream's keep-everything fallback at a degenerate pivot.

import triton
import triton.language as tl


@triton.jit
def _update_min_larger_stats(data, above_mask, min_larger, num_min_larger, sentinel):
    """Update running (min, count) of values above a pivot across tiles.

    Tracks the smallest value strictly above a pivot and how many times
    it occurs.  Called once per tile per pivot; the running state is
    carried across tiles via `min_larger` / `num_min_larger`.

    Merge rule:
      - tile min < running min  → replace both
      - tile min == running min → accumulate count
      - tile min > running min  → keep running values
    """
    tile_min = tl.min(tl.where(above_mask, data, sentinel))
    tile_eq = above_mask & (tl.abs(data - tile_min) < 1e-9)
    tile_cnt = tl.sum(tile_eq)
    is_new = tile_min < min_larger
    is_same = tl.abs(tile_min - min_larger) < 1e-9
    num_min_larger = tl.where(is_new, tile_cnt, num_min_larger + tile_cnt * is_same)
    min_larger = tl.minimum(min_larger, tile_min)
    return min_larger, num_min_larger


@triton.jit
def kernel(
    LOGITS: tl.tensor,
    LOGITS_STRIDE_0: tl.int32,
    BUFFER: tl.tensor,
    PERCENTILE_TO_STD_TABLE: tl.tensor,
    NORMAL_CDF_TO_SIGMA_TABLE: tl.tensor,
    K: tl.int32,
    P: tl.float32,
    BATCH_SIZE: tl.constexpr,
    VOCAB_SIZE: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    BLOCK_SIZE_TRUNC: tl.constexpr,
):
    NUM_TILES: tl.constexpr = (VOCAB_SIZE + BLOCK_SIZE - 1) // BLOCK_SIZE
    pid = tl.program_id(0)
    num_programs = tl.num_programs(0)
    for row_id in tl.range(pid, BATCH_SIZE, num_programs):
        LOGITS_ROW = LOGITS + row_id.to(tl.int64) * LOGITS_STRIDE_0
        BUFFER_ROW = BUFFER + pid * VOCAB_SIZE

        final_pivot = -float("inf")
        duplicate_logit = float("inf")
        num_duplicate_logit = tl.zeros((), dtype=tl.uint32)
        num_keep = tl.zeros((), dtype=tl.uint32)
        num_kept = tl.zeros((), dtype=tl.uint32)

        max_logit = -float("inf")
        min_logit = float("inf")

        k = K
        if k < VOCAB_SIZE:
            # Zeroth pass: Compute avg and std from a sample block
            offs = tl.arange(0, BLOCK_SIZE)
            mask_n = offs < VOCAB_SIZE
            logits_blk0 = tl.load(LOGITS_ROW + offs, mask=mask_n, other=-float("inf"))
            # Exclude -inf values (e.g. from grammar bitmasks) from
            # statistics to avoid NaN in pivot computation.
            finite_mask = (logits_blk0 > -float("inf")) & mask_n
            num_finite = tl.sum(finite_mask)
            finite_logits = tl.where(finite_mask, logits_blk0, 0.0)
            avg_logit = tl.where(
                num_finite > 0, tl.sum(finite_logits) / num_finite, 0.0
            )
            sq_avg_logit = tl.where(
                num_finite > 0,
                tl.sum(finite_logits * finite_logits) / num_finite,
                0.0,
            )
            std_logit = tl.sqrt(tl.maximum(sq_avg_logit - avg_logit * avg_logit, 0.0))

            # Calculate outlier pivot t for Gaussian sigma-truncation
            percentile = tl.cast(k / VOCAB_SIZE * 200, tl.uint32)
            percentile = tl.minimum(percentile, 199)
            sigma = tl.load(PERCENTILE_TO_STD_TABLE + percentile)
            sigma = sigma + tl.abs(sigma) * -0.15
            outlier_pivot = avg_logit + std_logit * sigma
            num_outliers = tl.zeros((), dtype=tl.uint32)

            # First pass: compute max and min logits and gather outliers
            num_finite_total = tl.zeros((), dtype=tl.uint32)
            for i in range(NUM_TILES):
                offs_n = i * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
                mask_n = offs_n < VOCAB_SIZE
                logits_blk = tl.load(
                    LOGITS_ROW + offs_n, mask=mask_n, other=-float("inf")
                )

                max_logit = tl.maximum(max_logit, tl.max(logits_blk))
                # Exclude -inf from min to keep binary search bounds
                # finite (avoids NaN pivots).
                finite_blk_mask = logits_blk > -float("inf")
                finite_blk = tl.where(finite_blk_mask, logits_blk, float("inf"))
                min_logit = tl.minimum(min_logit, tl.min(finite_blk))
                num_finite_total += tl.sum(finite_blk_mask & mask_n)

                outlier_mask = (logits_blk > outlier_pivot) & mask_n
                cumulative_pos = tl.cast(
                    tl.cumsum(outlier_mask) - 1 + num_outliers, tl.int32
                )
                num_outliers += tl.sum(outlier_mask)
                write_pos = tl.where(outlier_mask, cumulative_pos, -1)
                tl.store(BUFFER_ROW + write_pos, logits_blk, mask=outlier_mask)

            # If no finite logits exist (all -inf), clamp min to
            # max so the search converges to -inf (no masking).
            min_logit = tl.minimum(min_logit, max_logit)

            # Second passes: Ternary search for pivot
            num_iters = 0
            k_pivot = float("inf")
            k_pivots_num = tl.zeros((), dtype=tl.uint32)
            min_larger = float("inf")
            num_min_larger = tl.zeros((), dtype=tl.uint32)
            if num_outliers > k:
                max_range = max_logit
                min_range = outlier_pivot
                search_range = tl.cast(num_outliers, tl.int32)
                search_iters = tl.cast(
                    (num_outliers + BLOCK_SIZE_TRUNC - 1) // BLOCK_SIZE_TRUNC,
                    tl.int32,
                )
                found_pivot = 0
                while found_pivot == 0:
                    k_pivot_0 = (max_range - min_range) * 1.0 / 3.0 + min_range
                    k_pivots_num_0 = tl.zeros((), dtype=tl.uint32)
                    min_larger_0 = float("inf")
                    num_min_larger_0 = tl.zeros((), dtype=tl.uint32)

                    k_pivot_1 = (max_range - min_range) * 2.0 / 3.0 + min_range
                    k_pivots_num_1 = tl.zeros((), dtype=tl.uint32)
                    min_larger_1 = float("inf")
                    num_min_larger_1 = tl.zeros((), dtype=tl.uint32)

                    # Single fused pass: compute k_pivots_num,
                    # min_larger, and num_min_larger together to avoid
                    # a second data scan. See _update_min_larger_stats
                    # for the tile-level merge logic.
                    for i in range(search_iters):
                        offs_n = i * BLOCK_SIZE_TRUNC + tl.arange(0, BLOCK_SIZE_TRUNC)
                        mask_n_2 = offs_n < search_range
                        logits_blk2 = tl.load(
                            BUFFER_ROW + offs_n, mask=mask_n_2, other=-float("inf")
                        )

                        above_0 = logits_blk2 > k_pivot_0
                        above_1 = logits_blk2 > k_pivot_1
                        k_pivots_num_0 += tl.sum(above_0)
                        k_pivots_num_1 += tl.sum(above_1)

                        min_larger_0, num_min_larger_0 = _update_min_larger_stats(
                            logits_blk2,
                            above_0,
                            min_larger_0,
                            num_min_larger_0,
                            float("inf"),
                        )
                        min_larger_1, num_min_larger_1 = _update_min_larger_stats(
                            logits_blk2,
                            above_1,
                            min_larger_1,
                            num_min_larger_1,
                            float("inf"),
                        )

                    # Check if any of the pivots satisfy termination condition
                    if k_pivots_num_0 >= k and k_pivots_num_0 - num_min_larger_0 < k:
                        k_pivot = k_pivot_0
                        k_pivots_num = k_pivots_num_0
                        min_larger = min_larger_0
                        num_min_larger = num_min_larger_0
                        found_pivot = 1
                    if k_pivots_num_1 >= k and k_pivots_num_1 - num_min_larger_1 < k:
                        k_pivot = k_pivot_1
                        k_pivots_num = k_pivots_num_1
                        min_larger = min_larger_1
                        num_min_larger = num_min_larger_1
                        found_pivot = 1

                    # Update range
                    if k_pivots_num_1 > k:
                        min_range = k_pivot_1
                    elif k_pivots_num_0 > k:
                        min_range = k_pivot_0

                    if k_pivots_num_0 < k:
                        max_range = k_pivot_0
                    elif k_pivots_num_1 < k:
                        max_range = k_pivot_1

                    num_iters += 1
                    if num_iters >= 18 or tl.abs(min_range - max_range) < 1e-9:
                        k_pivot = (max_range + min_range) / 2.0
                        min_larger = min_larger_0
                        num_min_larger = num_min_larger_0
                        found_pivot = 1
            else:
                # If top-k outlier gathering failed, search whole logit space
                max_range = max_logit
                min_range = min_logit
                found_pivot = 0
                while found_pivot == 0:
                    k_pivot_0 = (max_range - min_range) * 1.0 / 3.0 + min_range
                    k_pivots_num_0 = tl.zeros((), dtype=tl.uint32)
                    min_larger_0 = float("inf")
                    num_min_larger_0 = tl.zeros((), dtype=tl.uint32)

                    k_pivot_1 = (max_range - min_range) * 2.0 / 3.0 + min_range
                    k_pivots_num_1 = tl.zeros((), dtype=tl.uint32)
                    min_larger_1 = float("inf")
                    num_min_larger_1 = tl.zeros((), dtype=tl.uint32)

                    # Single fused pass over full vocab (same approach
                    # as the buffer path above).
                    for i in range(NUM_TILES):
                        offs_n = i * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
                        mask_n = offs_n < VOCAB_SIZE
                        logits_blk2 = tl.load(
                            LOGITS_ROW + offs_n, mask=mask_n, other=-float("inf")
                        )

                        above_0 = logits_blk2 > k_pivot_0
                        above_1 = logits_blk2 > k_pivot_1
                        k_pivots_num_0 += tl.sum(above_0)
                        k_pivots_num_1 += tl.sum(above_1)

                        min_larger_0, num_min_larger_0 = _update_min_larger_stats(
                            logits_blk2,
                            above_0,
                            min_larger_0,
                            num_min_larger_0,
                            float("inf"),
                        )
                        min_larger_1, num_min_larger_1 = _update_min_larger_stats(
                            logits_blk2,
                            above_1,
                            min_larger_1,
                            num_min_larger_1,
                            float("inf"),
                        )

                    # Check if any of the pivots satisfy termination condition
                    if k_pivots_num_0 >= k and k_pivots_num_0 - num_min_larger_0 < k:
                        k_pivot = k_pivot_0
                        k_pivots_num = k_pivots_num_0
                        min_larger = min_larger_0
                        num_min_larger = num_min_larger_0
                        found_pivot = 1
                    if k_pivots_num_1 >= k and k_pivots_num_1 - num_min_larger_1 < k:
                        k_pivot = k_pivot_1
                        k_pivots_num = k_pivots_num_1
                        min_larger = min_larger_1
                        num_min_larger = num_min_larger_1
                        found_pivot = 1

                    # Update range
                    if k_pivots_num_1 > k:
                        min_range = k_pivot_1
                    elif k_pivots_num_0 > k:
                        min_range = k_pivot_0

                    if k_pivots_num_0 < k:
                        max_range = k_pivot_0
                    elif k_pivots_num_1 < k:
                        max_range = k_pivot_1

                    num_iters += 1
                    if num_iters >= 18 or tl.abs(min_range - max_range) < 1e-9:
                        k_pivot = (max_range + min_range) / 2.0
                        min_larger = min_larger_0
                        num_min_larger = num_min_larger_0
                        found_pivot = 1

            duplicate_logit = min_larger
            num_duplicate_logit = num_min_larger
            num_keep = num_duplicate_logit - (k_pivots_num - k)
            num_kept = tl.zeros((), dtype=tl.uint32)

            # Top-k only path.  If there are fewer finite values
            # than k (e.g. grammar mask), keep everything.
            final_pivot = k_pivot if num_finite_total > k else -float("inf")

            if num_finite_total > k:
                #### TOP-P SAMPLING AFTER TOP-K ####
                p = P
                if p < 1.0:
                    min_logit = k_pivot
                    sum_exp_logits = 0.0
                    num_outliers_2 = tl.zeros((), dtype=tl.uint32)
                    search_range = tl.cast(num_outliers, tl.int32)
                    search_iters = tl.cast(
                        (num_outliers + BLOCK_SIZE_TRUNC - 1) // BLOCK_SIZE_TRUNC,
                        tl.int32,
                    )

                    # Third pass: Calculate exp logits and sum, gather outliers
                    if num_outliers > k:
                        for i in range(search_iters):
                            offs_n = i * BLOCK_SIZE_TRUNC + tl.arange(
                                0, BLOCK_SIZE_TRUNC
                            )
                            mask_n_2 = offs_n < search_range

                            probs_blk = tl.load(
                                BUFFER_ROW + offs_n,
                                mask=mask_n_2,
                                other=-float("inf"),
                            )

                            outlier_mask = (probs_blk > min_logit) & mask_n_2

                            # Duplicate logit handling for Top-k
                            if num_keep < num_duplicate_logit:
                                duplicate_mask = (
                                    tl.abs(probs_blk - duplicate_logit) < 1e-9
                                )
                                duplicate_count = tl.cumsum(duplicate_mask) + num_kept
                                duplicate_keep_mask = (
                                    duplicate_count <= num_keep
                                ) & duplicate_mask
                                duplicate_remove_mask = (
                                    duplicate_mask & ~duplicate_keep_mask
                                )
                                outlier_mask = outlier_mask & (~duplicate_remove_mask)
                                num_kept += tl.sum(duplicate_keep_mask)

                            probs_blk = tl.where(outlier_mask, probs_blk, -float("inf"))
                            probs_blk = probs_blk - max_logit
                            probs_blk = tl.exp(probs_blk)
                            sum_exp_logits += tl.sum(probs_blk)

                        # Fourth pass: Calculate BUFFER and get outliers
                        for i in range(search_iters):
                            offs_n = i * BLOCK_SIZE_TRUNC + tl.arange(
                                0, BLOCK_SIZE_TRUNC
                            )
                            mask_n_2 = offs_n < search_range

                            probs_blk = tl.load(
                                BUFFER_ROW + offs_n,
                                mask=mask_n_2,
                                other=-float("inf"),
                            )

                            probs_blk = probs_blk - max_logit
                            probs_blk = tl.exp(probs_blk)
                            probs_blk = probs_blk / sum_exp_logits
                            tl.store(BUFFER_ROW + offs_n, probs_blk, mask=mask_n_2)
                    else:
                        # If top-k outlier gathering failed,
                        # retry gathering using top-k pivot
                        for i in range(NUM_TILES):
                            offs_n = i * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
                            mask_n = offs_n < VOCAB_SIZE

                            probs_blk = tl.load(
                                LOGITS_ROW + offs_n,
                                mask=mask_n,
                                other=-float("inf"),
                            )

                            outlier_mask = (probs_blk > min_logit) & mask_n

                            # Duplicate logit handling for Top-k
                            duplicate_mask = tl.abs(probs_blk - duplicate_logit) < 1e-9
                            duplicate_count = tl.cumsum(duplicate_mask) + num_kept
                            duplicate_keep_mask = (
                                duplicate_count <= num_keep
                            ) & duplicate_mask
                            duplicate_remove_mask = (
                                duplicate_mask & ~duplicate_keep_mask
                            )
                            outlier_mask = outlier_mask & (~duplicate_remove_mask)
                            num_kept += tl.sum(duplicate_keep_mask)

                            probs_blk = tl.where(outlier_mask, probs_blk, -float("inf"))
                            probs_blk = probs_blk - max_logit
                            probs_blk = tl.exp(probs_blk)
                            sum_exp_logits += tl.sum(probs_blk)

                            cumulative_pos = tl.cast(
                                tl.cumsum(outlier_mask) - 1 + num_outliers_2,
                                tl.int32,
                            )
                            num_outliers_2 += tl.sum(outlier_mask)
                            write_pos = tl.where(outlier_mask, cumulative_pos, -1)
                            tl.store(
                                BUFFER_ROW + write_pos, probs_blk, mask=outlier_mask
                            )

                        search_range = tl.cast(num_outliers_2, tl.int32)
                        search_iters = tl.cast(
                            (num_outliers_2 + BLOCK_SIZE_TRUNC - 1) // BLOCK_SIZE_TRUNC,
                            tl.int32,
                        )

                        # Fourth pass: Calculate BUFFER and get outliers
                        for i in range(search_iters):
                            offs_n = i * BLOCK_SIZE_TRUNC + tl.arange(
                                0, BLOCK_SIZE_TRUNC
                            )
                            mask_n_2 = offs_n < search_range

                            probs_blk = tl.load(
                                BUFFER_ROW + offs_n, mask=mask_n_2, other=0.0
                            )
                            probs_blk = probs_blk / sum_exp_logits
                            tl.store(BUFFER_ROW + offs_n, probs_blk, mask=mask_n_2)

                    max_range = tl.exp(max_logit - max_logit) / sum_exp_logits
                    min_range = tl.exp(min_logit - max_logit) / sum_exp_logits

                    p_pivot = 1.0
                    num_iters = 0
                    min_larger_prob = 1.0
                    num_min_larger = tl.zeros((), dtype=tl.uint32)
                    p_pivots_sum = 0.0

                    # Fifth passes: Search for p_pivot
                    found_pivot = 0
                    while found_pivot == 0:
                        p_pivot_0 = (max_range - min_range) * 0.5 + min_range
                        p_pivots_sum_0 = 0.0
                        min_larger_0 = 1.0
                        num_min_larger_0 = tl.zeros((), dtype=tl.uint32)

                        # Single fused pass: compute p_pivots_sum,
                        # min_larger, and num_min_larger together.
                        # See _update_min_larger_stats for the
                        # tile-level merge logic.
                        for i in range(search_iters):
                            offs_n = i * BLOCK_SIZE_TRUNC + tl.arange(
                                0, BLOCK_SIZE_TRUNC
                            )
                            mask_n_2 = offs_n < search_range
                            probs_blk = tl.load(
                                BUFFER_ROW + offs_n, mask=mask_n_2, other=0.0
                            )

                            above_0 = probs_blk > p_pivot_0
                            p_pivots_sum_0 += tl.sum(probs_blk * above_0)

                            min_larger_0, num_min_larger_0 = _update_min_larger_stats(
                                probs_blk,
                                above_0,
                                min_larger_0,
                                num_min_larger_0,
                                1.0,
                            )

                        # Check if the pivot satisfies termination condition
                        if p_pivots_sum_0 >= p and (
                            p_pivots_sum_0 - (min_larger_0 * num_min_larger_0) < p
                        ):
                            p_pivot = p_pivot_0
                            min_larger_prob = min_larger_0
                            num_min_larger = num_min_larger_0
                            p_pivots_sum = p_pivots_sum_0
                            found_pivot = 1

                        # Update range
                        if p_pivots_sum_0 > p:
                            min_range = p_pivot_0
                        elif p_pivots_sum_0 < p:
                            max_range = p_pivot_0

                        num_iters += 1
                        if (max_range - min_range) < 1e-9 or num_iters >= 18:
                            p_pivot = (max_range + min_range) / 2.0
                            min_larger_prob = min_larger_0
                            num_min_larger = num_min_larger_0
                            p_pivots_sum = p_pivots_sum_0
                            found_pivot = 1

                    duplicate_logit = (
                        tl.log(min_larger_prob * sum_exp_logits) + max_logit
                    )
                    num_duplicate_logit = num_min_larger
                    num_keep = num_duplicate_logit - tl.cast(
                        (p_pivots_sum - p) / min_larger_prob, tl.uint32
                    )
                    num_kept = tl.zeros((), dtype=tl.uint32)

                    # Top-k + Top-p path
                    final_pivot = tl.log(p_pivot * sum_exp_logits) + max_logit

        if final_pivot == -float("inf"):
            #### STANDALONE TOP-P SAMPLING ####
            p = P
            if p < 1.0:
                # Zeroth pass: Compute avg and std from a sample block
                offs = tl.arange(0, BLOCK_SIZE)
                mask_n = offs < VOCAB_SIZE
                logits_blk0 = tl.load(
                    LOGITS_ROW + offs, mask=mask_n, other=-float("inf")
                )
                # Exclude -inf values (e.g. from grammar bitmasks) from
                # statistics to avoid NaN in pivot computation.
                finite_mask = (logits_blk0 > -float("inf")) & mask_n
                num_finite = tl.sum(finite_mask)
                finite_logits = tl.where(finite_mask, logits_blk0, 0.0)
                avg_logit = tl.where(
                    num_finite > 0, tl.sum(finite_logits) / num_finite, 0.0
                )
                sq_avg_logit = tl.where(
                    num_finite > 0,
                    tl.sum(finite_logits * finite_logits) / num_finite,
                    0.0,
                )
                std_logit = tl.sqrt(
                    tl.maximum(sq_avg_logit - avg_logit * avg_logit, 0.0)
                )
                max_sample = avg_logit + std_logit * 10.0
                sum_exp_logits = 0.0

                # First pass: compute max and min logits and sum_exp_logits
                for i in range(NUM_TILES):
                    offs_n = i * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
                    mask_n = offs_n < VOCAB_SIZE
                    logits_blk = tl.load(
                        LOGITS_ROW + offs_n, mask=mask_n, other=-float("inf")
                    )
                    max_logit = tl.maximum(max_logit, tl.max(logits_blk))
                    # Exclude -inf from min to keep binary search bounds
                    # finite (avoids NaN pivots).
                    finite_blk = tl.where(
                        logits_blk > -float("inf"), logits_blk, float("inf")
                    )
                    min_logit = tl.minimum(min_logit, tl.min(finite_blk))

                    probs_blk = tl.exp(logits_blk - max_sample)
                    probs_blk = tl.where(mask_n, probs_blk, 0.0)
                    sum_exp_logits += tl.sum(probs_blk)

                # If no finite logits exist (all -inf), clamp min to
                # max so the search converges to -inf (no masking).
                min_logit = tl.minimum(min_logit, max_logit)

                idx = tl.cast(p * 200, tl.int32)
                idx = tl.maximum(0, tl.minimum(idx, 199))
                sigma = tl.load(NORMAL_CDF_TO_SIGMA_TABLE + idx)
                sigma = sigma + tl.abs(sigma) * -0.25
                outlier_pivot = avg_logit + std_logit * sigma

                outlier_prob = tl.exp(outlier_pivot - max_sample) / sum_exp_logits
                sum_outlier_probs = 0.0
                num_outliers = tl.zeros((), dtype=tl.uint32)

                # Second pass: Calculate softmax and gather outliers
                for i in range(NUM_TILES):
                    offs_n = i * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
                    mask_n = offs_n < VOCAB_SIZE

                    probs_blk = tl.load(
                        LOGITS_ROW + offs_n, mask=mask_n, other=-float("inf")
                    )
                    probs_blk = tl.exp(probs_blk - max_sample)
                    probs_blk = probs_blk / sum_exp_logits

                    outlier_mask = (probs_blk > outlier_prob) & mask_n
                    sum_outlier_probs += tl.sum(outlier_mask * probs_blk)
                    cumulative_pos = tl.cast(
                        tl.cumsum(outlier_mask) - 1 + num_outliers, tl.int32
                    )
                    num_outliers += tl.sum(outlier_mask)
                    write_pos = tl.where(outlier_mask, cumulative_pos, -1)
                    tl.store(BUFFER_ROW + write_pos, probs_blk, mask=outlier_mask)

                max_range = tl.exp(max_logit - max_sample) / sum_exp_logits
                min_range = tl.exp(min_logit - max_sample) / sum_exp_logits

                p_pivot = 1.0
                num_iters = 0
                min_larger_prob = 1.0
                num_min_larger = tl.zeros((), dtype=tl.uint32)
                p_pivots_sum = 0.0

                # Third pass: Search for p_pivot
                if sum_outlier_probs > p:
                    min_range = outlier_prob
                    search_range = tl.cast(num_outliers, tl.int32)
                    search_iters = tl.cast(
                        (num_outliers + BLOCK_SIZE_TRUNC - 1) // BLOCK_SIZE_TRUNC,
                        tl.int32,
                    )

                    found_pivot = 0
                    while found_pivot == 0:
                        p_pivot_0 = (max_range - min_range) * 0.5 + min_range
                        p_pivots_sum_0 = 0.0
                        min_larger_0 = 1.0
                        num_min_larger_0 = tl.zeros((), dtype=tl.uint32)

                        # Single fused pass: compute p_pivots_sum,
                        # min_larger, and num_min_larger together.
                        # See _update_min_larger_stats for the
                        # tile-level merge logic.
                        for i in range(search_iters):
                            offs_n = i * BLOCK_SIZE_TRUNC + tl.arange(
                                0, BLOCK_SIZE_TRUNC
                            )
                            mask_n_2 = offs_n < search_range
                            probs_blk = tl.load(
                                BUFFER_ROW + offs_n, mask=mask_n_2, other=0.0
                            )

                            above_0 = probs_blk > p_pivot_0
                            p_pivots_sum_0 += tl.sum(probs_blk * above_0)

                            min_larger_0, num_min_larger_0 = _update_min_larger_stats(
                                probs_blk,
                                above_0,
                                min_larger_0,
                                num_min_larger_0,
                                1.0,
                            )

                        # Check if the pivot satisfies termination condition
                        if (
                            p_pivots_sum_0 >= p
                            and p_pivots_sum_0 - (min_larger_0 * num_min_larger_0) < p
                        ):
                            p_pivot = p_pivot_0
                            min_larger_prob = min_larger_0
                            num_min_larger = num_min_larger_0
                            p_pivots_sum = p_pivots_sum_0
                            found_pivot = 1

                        # Update range
                        if p_pivots_sum_0 > p:
                            min_range = p_pivot_0
                        elif p_pivots_sum_0 < p:
                            max_range = p_pivot_0

                        num_iters += 1
                        if (max_range - min_range) < 1e-9 or num_iters >= 18:
                            p_pivot = (max_range + min_range) / 2.0
                            min_larger_prob = min_larger_0
                            num_min_larger = num_min_larger_0
                            p_pivots_sum = p_pivots_sum_0
                            found_pivot = 1
                else:
                    # Re-populate the buffer with full softmax probabilities
                    for i in range(NUM_TILES):
                        offs_n = i * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
                        mask_n = offs_n < VOCAB_SIZE

                        probs_blk = tl.load(
                            LOGITS_ROW + offs_n, mask=mask_n, other=-float("inf")
                        )
                        probs_blk = tl.exp(probs_blk - max_sample)
                        probs_blk = probs_blk / sum_exp_logits
                        tl.store(BUFFER_ROW + offs_n, probs_blk, mask=mask_n)

                    found_pivot = 0
                    while found_pivot == 0:
                        p_pivot_0 = (max_range - min_range) * 0.5 + min_range
                        p_pivots_sum_0 = 0.0
                        min_larger_0 = 1.0
                        num_min_larger_0 = tl.zeros((), dtype=tl.uint32)

                        # Single fused pass: compute p_pivots_sum,
                        # min_larger, and num_min_larger together.
                        # See _update_min_larger_stats for the
                        # tile-level merge logic.
                        for i in range(NUM_TILES):
                            offs_n = i * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
                            mask_n = offs_n < VOCAB_SIZE
                            probs_blk = tl.load(
                                BUFFER_ROW + offs_n, mask=mask_n, other=0.0
                            )

                            above_0 = probs_blk > p_pivot_0
                            p_pivots_sum_0 += tl.sum(probs_blk * above_0)

                            min_larger_0, num_min_larger_0 = _update_min_larger_stats(
                                probs_blk,
                                above_0,
                                min_larger_0,
                                num_min_larger_0,
                                1.0,
                            )

                        # Check if the pivot satisfies termination condition
                        if (
                            p_pivots_sum_0 >= p
                            and p_pivots_sum_0 - (min_larger_0 * num_min_larger_0) < p
                        ):
                            p_pivot = p_pivot_0
                            min_larger_prob = min_larger_0
                            num_min_larger = num_min_larger_0
                            p_pivots_sum = p_pivots_sum_0
                            found_pivot = 1

                        # Update range
                        if p_pivots_sum_0 > p:
                            min_range = p_pivot_0
                        elif p_pivots_sum_0 < p:
                            max_range = p_pivot_0

                        num_iters += 1
                        if (max_range - min_range) < 1e-9 or num_iters >= 18:
                            p_pivot = (max_range + min_range) / 2.0
                            min_larger_prob = min_larger_0
                            num_min_larger = num_min_larger_0
                            p_pivots_sum = p_pivots_sum_0
                            found_pivot = 1

                duplicate_logit = tl.log(min_larger_prob * sum_exp_logits) + max_sample
                num_duplicate_logit = num_min_larger
                num_keep = num_duplicate_logit - tl.cast(
                    (p_pivots_sum - p) / min_larger_prob, tl.uint32
                )
                num_kept = tl.zeros((), dtype=tl.uint32)

                # Top-p only path
                final_pivot = tl.log(p_pivot * sum_exp_logits) + max_sample

        # Sixth pass: Apply mask and store final output.
        # If the pivot >= max logit (or is NaN), no token would
        # survive the strict `>` keep_mask.  Skip masking.
        # Using `not <` instead of `>=` so that NaN is also caught.
        if max_logit == min_logit and max_logit > -float("inf"):
            finite_count = tl.full((), 0, tl.int32)
            for i in range(NUM_TILES):
                offsets = i * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
                values = tl.load(
                    LOGITS_ROW + offsets, offsets < VOCAB_SIZE, other=-float("inf")
                )
                finite_count += tl.sum((values > -float("inf")).to(tl.int32))
            keep_count = tl.minimum(finite_count, K)
            keep_count = tl.maximum(1, tl.ceil(P * keep_count).to(tl.int32))
            seen = tl.full((), 0, tl.int32)
            for i in range(NUM_TILES):
                offsets = i * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
                values = tl.load(
                    LOGITS_ROW + offsets, offsets < VOCAB_SIZE, other=-float("inf")
                )
                finite = values > -float("inf")
                rank = seen + tl.cumsum(finite.to(tl.int32))
                seen += tl.sum(finite.to(tl.int32))
                tl.store(
                    LOGITS_ROW + offsets,
                    tl.where(finite & (rank <= keep_count), values, -float("inf")),
                    offsets < VOCAB_SIZE,
                )
        elif not (final_pivot < max_logit):
            final_pivot = -float("inf")
        elif final_pivot != -float("inf"):
            for i in range(NUM_TILES):
                offs_n = i * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
                mask_n = offs_n < VOCAB_SIZE
                logits_blk = tl.load(
                    LOGITS_ROW + offs_n, mask=mask_n, other=-float("inf")
                )
                keep_mask = (logits_blk > final_pivot) & mask_n

                # Duplicate logit handling
                if num_keep < num_duplicate_logit:
                    duplicate_mask = (
                        tl.abs(logits_blk - duplicate_logit) < 1e-9
                    ) & mask_n
                    duplicate_count = tl.cumsum(duplicate_mask) + num_kept
                    duplicate_keep_mask = (duplicate_count <= num_keep) & duplicate_mask
                    duplicate_remove_mask = duplicate_mask & ~duplicate_keep_mask
                    num_kept += tl.sum(duplicate_keep_mask)
                    keep_mask = keep_mask & (~duplicate_remove_mask)

                logits_blk = tl.where(keep_mask, logits_blk, -float("inf"))
                tl.store(LOGITS_ROW + offs_n, logits_blk, mask=mask_n)
