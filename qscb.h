#ifndef QSCB_H
#define QSCB_H

#include "qs_info.h"
#include "qs_tensor.h"

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/*
 * qscb is the narrow cuBLASLt lane for qwen3.6 dense projections.
 *
 * It intentionally models only the runtime shape used by qsfi linear/LM-head
 * callers:
 *   x      bf16 [rows, in_features]
 *   weight bf16 [out_features, in_features]
 *   out    bf16/f32 [rows, out_features]
 *
 * Tensors must be row-major with stride[1] == 1. Padded row strides are
 * accepted. Accumulation is always f32.
 */

typedef struct qscb_context qscb_context;

typedef struct {
    int32_t device_ordinal; /* < 0 means current CUDA device at create time. */
    qsfi_cuda_stream stream; /* NULL means the default stream. */
} qscb_context_desc;

typedef struct {
    qsfi_tensor2 x;
    qsfi_tensor2 weight;
    qsfi_tensor2 out;
    uint32_t rows;
    uint32_t in_features;
    uint32_t out_features;
    float alpha; /* 0 means 1, matching qsfi descriptor defaults. */
    float beta;
    qsfi_device_ptr workspace;
    size_t workspace_bytes;
} qscb_linear_desc;

qsfi_status qscb_context_create(const qscb_context_desc* desc, qscb_context** out);
void qscb_context_destroy(qscb_context* ctx);
qsfi_status qscb_context_get_last_error(const qscb_context* ctx, qsfi_error_info* out);
void qscb_context_clear_last_error(qscb_context* ctx);

typedef struct qscb_linear_plan qscb_linear_plan;

/* Preparation queries cuBLASLt once. Plans belong to ctx and must be destroyed
 * before it. Execution permits new addresses/scalars with identical dimensions,
 * row strides, tensor dtypes, capped (256-byte) pointer alignments and workspace
 * capacity. Workspace pointers must be 256-byte aligned. No device sync occurs.
 */
qsfi_status
qscb_linear_plan_create(qscb_context* ctx, const qscb_linear_desc* desc, qscb_linear_plan** out);
void qscb_linear_plan_destroy(qscb_linear_plan* plan);
qsfi_status
qscb_linear_execute(qscb_context* ctx, const qscb_linear_plan* plan, const qscb_linear_desc* desc);

/* Tensor-scaled E4M3 W8A8, with BF16/F32 output and F32 accumulation.
 * linear.x and linear.weight are E4M3; the remaining linear fields have the
 * same meaning as above. Scales are device F32[1] dequantization multipliers,
 * not reciprocals. Their values must be finite and positive. Callers own all
 * buffers and validate scale values before use. Context/plan use is serialized
 * by the caller. Scale addresses are rebound on every execution.
 */
typedef struct {
    qscb_linear_desc linear;
    qsfi_tensor1 x_scale;
    qsfi_tensor1 weight_scale;
} qscb_fp8_linear_desc;

qsfi_status qscb_fp8_linear_plan_create(
    qscb_context* ctx, const qscb_fp8_linear_desc* desc, qscb_linear_plan** out
);
qsfi_status qscb_fp8_linear_execute(
    qscb_context* ctx, const qscb_linear_plan* plan, const qscb_fp8_linear_desc* desc
);

/* BF16 -> E4M3 round-to-nearest, finite saturation. Row strides may be padded.
 * out = fp8(x / scale). scale is a caller-validated positive finite device
 * F32[1] dequantization multiplier, also supplied to the subsequent GEMM.
 * Input, output and scale storage must not overlap.
 */
typedef struct {
    qsfi_tensor2 x;
    qsfi_tensor2 out;
    qsfi_tensor1 scale;
} qscb_fp8_quantize_desc;
qsfi_status qscb_fp8_quantize(qscb_context* ctx, const qscb_fp8_quantize_desc* desc);

#ifdef __cplusplus
}
#endif

#endif
