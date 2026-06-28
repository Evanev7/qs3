#ifndef QSFI_NATIVE_COMMON_H
#define QSFI_NATIVE_COMMON_H

#include "qsfi.h"

#include <cuda_runtime.h>

#include <cstdint>

#ifndef QSFI_ENABLE_CHECKED_VALIDATION
#define QSFI_ENABLE_CHECKED_VALIDATION 0
#endif

inline void qsfi_clear_error_info(qsfi_error_info* err)
{
    if (err == nullptr)
        return;
    err->status = QSFI_STATUS_OK;
    err->source = QSFI_ERROR_SOURCE_NONE;
    err->native_code = 0;
    err->message[0] = '\0';
}

inline float qsfi_default_one(float value)
{
    return value == 0.0f ? 1.0f : value;
}

template <typename Reporter, typename Tensor>
qsfi_status qsfi_validate_native_tensor(
    const Reporter& reporter,
    const Tensor& tensor,
    const char* name,
    qsfi_dtype dtype,
    uint32_t expected_rank
)
{
    if (tensor.data == nullptr) {
        return reporter.invalid_arg("%s.data must not be null", name);
    }
    if (tensor.dtype != dtype) {
        return reporter.invalid_arg("%s dtype does not match expected dtype", name);
    }
    constexpr uint32_t rank = sizeof(tensor.shape) / sizeof(tensor.shape[0]);
    if (rank != expected_rank) {
        return reporter.invalid_arg("%s rank mismatch", name);
    }
    for (uint32_t i = 0; i < rank; ++i) {
        if (tensor.shape[i] <= 0 || tensor.stride[i] <= 0) {
            return reporter.invalid_arg("%s shape/stride entries must be positive", name);
        }
    }
    return QSFI_STATUS_OK;
}

#if QSFI_ENABLE_CHECKED_VALIDATION
class qsfi_checked_validation_flag {
public:
    explicit qsfi_checked_validation_flag(cudaStream_t stream)
        : stream_(stream)
        , device_flag_(nullptr)
    {
    }

    qsfi_checked_validation_flag(const qsfi_checked_validation_flag&) = delete;
    qsfi_checked_validation_flag& operator=(const qsfi_checked_validation_flag&) = delete;

    ~qsfi_checked_validation_flag()
    {
        if (device_flag_ != nullptr)
            cudaFree(device_flag_);
    }

    int* device_ptr() const
    {
        return device_flag_;
    }

    template <typename Reporter>
    qsfi_status reset(const Reporter& reporter, const char* malloc_op, const char* memset_op)
    {
        cudaError_t err
            = cudaMalloc(reinterpret_cast<void**>(&device_flag_), sizeof(*device_flag_));
        if (err != cudaSuccess)
            return reporter.cuda_error(err, malloc_op);
        err = cudaMemsetAsync(device_flag_, 0, sizeof(*device_flag_), stream_);
        if (err != cudaSuccess)
            return reporter.cuda_error(err, memset_op);
        return QSFI_STATUS_OK;
    }

    template <typename Reporter> qsfi_status check_launch(const Reporter& reporter, const char* op)
    {
        const cudaError_t err = cudaGetLastError();
        if (err != cudaSuccess)
            return reporter.cuda_error(err, op);
        return QSFI_STATUS_OK;
    }

    template <typename Reporter>
    qsfi_status finish(
        const Reporter& reporter,
        const char* copy_op,
        const char* free_op,
        const char* invalid_message
    )
    {
        int host_error = 0;
        cudaError_t err = cudaMemcpyAsync(
            &host_error,
            device_flag_,
            sizeof(host_error),
            cudaMemcpyDeviceToHost,
            stream_
        );
        if (err == cudaSuccess)
            err = cudaStreamSynchronize(stream_);
        const cudaError_t free_err = release();
        if (err != cudaSuccess)
            return reporter.cuda_error(err, copy_op);
        if (free_err != cudaSuccess)
            return reporter.cuda_error(free_err, free_op);
        if (host_error != 0)
            return reporter.invalid_arg("%s", invalid_message);
        return QSFI_STATUS_OK;
    }

private:
    cudaError_t release()
    {
        int* flag = device_flag_;
        device_flag_ = nullptr;
        return cudaFree(flag);
    }

    cudaStream_t stream_;
    int* device_flag_;
};
#endif

#endif
