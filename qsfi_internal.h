#ifndef QSFI_INTERNAL_H
#define QSFI_INTERNAL_H

#include "qsfi.h"
#include "qsfi_native_common.h"

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

qsfi_status set_error(
    qsfi_context* ctx,
    qsfi_status status,
    qsfi_error_source source,
    int32_t native_code,
    const char* fmt,
    ...
);
qsfi_status set_invalid_arg(qsfi_context* ctx, const char* fmt, ...);
qsfi_status set_unsupported(qsfi_context* ctx, const char* fmt, ...);
qsfi_status set_out_of_memory(qsfi_context* ctx, const char* fmt, ...);
qsfi_status set_cuda_error(qsfi_context* ctx, cudaError_t err, const char* op);
qsfi_status set_flashinfer_error(qsfi_context* ctx, const char* op, const std::exception& ex);
qsfi_status activate_context(qsfi_context* ctx);
bool valid_dtype(qsfi_dtype dtype);

struct qsfi_context_error_reporter {
    qsfi_context* ctx;

    qsfi_status cuda_error(cudaError_t err, const char* op) const
    {
        return set_cuda_error(ctx, err, op);
    }

    template <typename... Args> qsfi_status invalid_arg(const char* fmt, Args... args) const
    {
        return set_invalid_arg(ctx, fmt, args...);
    }
};

template <typename Tensor>
qsfi_status validate_tensor(
    qsfi_context* ctx,
    const Tensor& tensor,
    const char* name,
    qsfi_dtype dtype,
    uint32_t expected_rank
)
{
    return qsfi_validate_native_tensor(
        qsfi_context_error_reporter { ctx },
        tensor,
        name,
        dtype,
        expected_rank
    );
}

#endif
