#ifndef QS_TENSOR_H
#define QS_TENSOR_H

#include <stdint.h>

/*
 * Shared C ABI tensor/device definitions. Strides are element strides, not byte
 * strides. Rank is part of the type name so public calls stay narrow and
 * operation validation can enforce qwen3.6 shape contracts locally.
 */

typedef void* qsfi_cuda_stream;
typedef void* qsfi_device_ptr;

typedef enum {
    QSFI_DTYPE_INVALID = 0,
    QSFI_DTYPE_F32 = 1,
    QSFI_DTYPE_F16 = 2,
    QSFI_DTYPE_BF16 = 3,
    QSFI_DTYPE_FP8_E4M3 = 4,
    QSFI_DTYPE_FP8_E5M2 = 5,
    QSFI_DTYPE_NVFP4_E2M1 = 6,
    QSFI_DTYPE_MXFP4_E2M1 = 7,
    QSFI_DTYPE_MXFP8_E4M3 = 8,
    QSFI_DTYPE_I32 = 9,
    QSFI_DTYPE_U32 = 10,
    QSFI_DTYPE_I8 = 11,
    QSFI_DTYPE_U8 = 12
} qsfi_dtype;

typedef struct {
    qsfi_device_ptr data;
    qsfi_dtype dtype;
    int64_t shape[1];
    int64_t stride[1];
} qsfi_tensor1;

typedef struct {
    qsfi_device_ptr data;
    qsfi_dtype dtype;
    int64_t shape[2];
    int64_t stride[2];
} qsfi_tensor2;

typedef struct {
    qsfi_device_ptr data;
    qsfi_dtype dtype;
    int64_t shape[3];
    int64_t stride[3];
} qsfi_tensor3;

typedef struct {
    qsfi_device_ptr data;
    qsfi_dtype dtype;
    int64_t shape[4];
    int64_t stride[4];
} qsfi_tensor4;

typedef struct {
    qsfi_device_ptr data;
    qsfi_dtype dtype;
    int64_t shape[5];
    int64_t stride[5];
} qsfi_tensor5;

typedef struct {
    qsfi_device_ptr data;
    qsfi_dtype dtype;
    int64_t shape[6];
    int64_t stride[6];
} qsfi_tensor6;

#endif
