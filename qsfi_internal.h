#ifndef QSFI_INTERNAL_H
#define QSFI_INTERNAL_H

#include "qsfi.h"

#include <cuda_runtime.h>

#include <exception>

struct qsfi_context {
    int32_t device_ordinal;
    cudaStream_t stream;
    void* float_workspace;
    size_t float_workspace_bytes;
    size_t int_workspace_bytes;
    size_t host_int_workspace_bytes;
    uint64_t scratch_generation;
    qsfi_error_info last_error;
};

void clear_error(qsfi_error_info* err);
qsfi_status set_error(
    qsfi_context* ctx,
    qsfi_status status,
    qsfi_error_source source,
    int32_t native_code,
    const char* fmt,
    ...
);
qsfi_status set_cuda_error(qsfi_context* ctx, cudaError_t err, const char* op);
qsfi_status set_flashinfer_error(qsfi_context* ctx, const char* op, const std::exception& ex);
qsfi_status activate_context(qsfi_context* ctx);
bool valid_dtype(qsfi_dtype dtype);
float default_one(float value);

template <typename Tensor>
qsfi_status validate_tensor(
    qsfi_context* ctx,
    const Tensor& tensor,
    const char* name,
    qsfi_dtype dtype,
    uint32_t expected_rank
)
{
    if (tensor.data == nullptr) {
        return set_error(
            ctx,
            QSFI_STATUS_INVALID_ARGUMENT,
            QSFI_ERROR_SOURCE_QSFI,
            0,
            "%s.data must not be null",
            name
        );
    }
    if (tensor.dtype != dtype) {
        return set_error(
            ctx,
            QSFI_STATUS_INVALID_ARGUMENT,
            QSFI_ERROR_SOURCE_QSFI,
            0,
            "%s dtype does not match expected dtype",
            name
        );
    }
    constexpr uint32_t rank = sizeof(tensor.shape) / sizeof(tensor.shape[0]);
    if (rank != expected_rank) {
        return set_error(
            ctx,
            QSFI_STATUS_INVALID_ARGUMENT,
            QSFI_ERROR_SOURCE_QSFI,
            0,
            "%s rank mismatch",
            name
        );
    }
    for (uint32_t i = 0; i < rank; ++i) {
        if (tensor.shape[i] <= 0 || tensor.stride[i] <= 0) {
            return set_error(
                ctx,
                QSFI_STATUS_INVALID_ARGUMENT,
                QSFI_ERROR_SOURCE_QSFI,
                0,
                "%s shape/stride entries must be positive",
                name
            );
        }
    }
    return QSFI_STATUS_OK;
}

#endif
