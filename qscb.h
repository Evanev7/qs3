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
} qscb_bf16_gemm_desc;

qsfi_status qscb_context_create(const qscb_context_desc* desc, qscb_context** out);
void qscb_context_destroy(qscb_context* ctx);
qsfi_status qscb_context_get_last_error(const qscb_context* ctx, qsfi_error_info* out);
void qscb_context_clear_last_error(qscb_context* ctx);

qsfi_status qscb_gemm_bf16(qscb_context* ctx, const qscb_bf16_gemm_desc* desc);

#ifdef __cplusplus
}
#endif

#endif
