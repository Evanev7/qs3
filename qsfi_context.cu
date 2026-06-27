#include "qsfi_internal.h"
#include "qsfi_macros.h"

#include <cuda_runtime.h>

#include <cstdarg>
#include <cstdio>
#include <cstring>
#include <new>

void clear_error(qsfi_error_info* err)
{
    if (err == nullptr)
        return;
    err->status = QSFI_STATUS_OK;
    err->source = QSFI_ERROR_SOURCE_NONE;
    err->native_code = 0;
    err->message[0] = '\0';
}

static qsfi_status set_errorv(
    qsfi_context* ctx,
    qsfi_status status,
    qsfi_error_source source,
    int32_t native_code,
    const char* fmt,
    va_list args
)
{
    if (ctx == nullptr)
        return status;
    ctx->last_error.status = status;
    ctx->last_error.source = source;
    ctx->last_error.native_code = native_code;
    std::vsnprintf(ctx->last_error.message, QSFI_ERROR_MESSAGE_BYTES, fmt, args);
    ctx->last_error.message[QSFI_ERROR_MESSAGE_BYTES - 1] = '\0';
    return status;
}

qsfi_status set_error(
    qsfi_context* ctx,
    qsfi_status status,
    qsfi_error_source source,
    int32_t native_code,
    const char* fmt,
    ...
)
{
    va_list args;
    va_start(args, fmt);
    status = set_errorv(ctx, status, source, native_code, fmt, args);
    va_end(args);
    return status;
}

#define DEFINE_QSFI_ERROR_SETTER(name, code)                                                       \
    qsfi_status name(qsfi_context* ctx, const char* fmt, ...)                                      \
    {                                                                                              \
        va_list args;                                                                              \
        va_start(args, fmt);                                                                       \
        qsfi_status status = set_errorv(ctx, code, QSFI_ERROR_SOURCE_QSFI, 0, fmt, args);          \
        va_end(args);                                                                              \
        return status;                                                                             \
    }

DEFINE_QSFI_ERROR_SETTER(set_invalid_arg, QSFI_STATUS_INVALID_ARGUMENT)
DEFINE_QSFI_ERROR_SETTER(set_unsupported, QSFI_STATUS_UNSUPPORTED)
DEFINE_QSFI_ERROR_SETTER(set_out_of_memory, QSFI_STATUS_OUT_OF_MEMORY)

#undef DEFINE_QSFI_ERROR_SETTER

qsfi_status set_cuda_error(qsfi_context* ctx, cudaError_t err, const char* op)
{
    if (err == cudaSuccess)
        return QSFI_STATUS_OK;
    const qsfi_status status
        = (err == cudaErrorMemoryAllocation) ? QSFI_STATUS_OUT_OF_MEMORY : QSFI_STATUS_CUDA_ERROR;
    return set_error(
        ctx,
        status,
        QSFI_ERROR_SOURCE_CUDA,
        static_cast<int32_t>(err),
        "%s: %s",
        op,
        cudaGetErrorString(err)
    );
}

qsfi_status set_flashinfer_error(qsfi_context* ctx, const char* op, const std::exception& ex)
{
    return set_error(
        ctx,
        QSFI_STATUS_BACKEND_ERROR,
        QSFI_ERROR_SOURCE_FLASHINFER,
        0,
        "%s: %s",
        op,
        ex.what()
    );
}

qsfi_status activate_context(qsfi_context* ctx)
{
    if (ctx == nullptr)
        return QSFI_STATUS_INVALID_ARGUMENT;
    if (ctx->device_ordinal < 0)
        return QSFI_STATUS_OK;
    cudaError_t err = cudaSetDevice(ctx->device_ordinal);
    if (err != cudaSuccess)
        return set_cuda_error(ctx, err, "cudaSetDevice");
    return QSFI_STATUS_OK;
}

bool valid_dtype(qsfi_dtype dtype)
{
    return dtype == QSFI_DTYPE_F32 || dtype == QSFI_DTYPE_F16 || dtype == QSFI_DTYPE_BF16
        || dtype == QSFI_DTYPE_FP8_E4M3 || dtype == QSFI_DTYPE_FP8_E5M2
        || dtype == QSFI_DTYPE_NVFP4_E2M1 || dtype == QSFI_DTYPE_MXFP4_E2M1
        || dtype == QSFI_DTYPE_MXFP8_E4M3 || dtype == QSFI_DTYPE_I32 || dtype == QSFI_DTYPE_U32
        || dtype == QSFI_DTYPE_I8 || dtype == QSFI_DTYPE_U8;
}

float default_one(float value)
{
    return value == 0.0f ? 1.0f : value;
}

extern "C" {

const char* qsfi_status_string(qsfi_status status)
{
    switch (status) {
    case QSFI_STATUS_OK:
        return "ok";
    case QSFI_STATUS_INVALID_ARGUMENT:
        return "invalid argument";
    case QSFI_STATUS_UNSUPPORTED:
        return "unsupported";
    case QSFI_STATUS_OUT_OF_MEMORY:
        return "out of memory";
    case QSFI_STATUS_CUDA_ERROR:
        return "cuda error";
    case QSFI_STATUS_BACKEND_ERROR:
        return "backend error";
    case QSFI_STATUS_INTERNAL_ERROR:
        return "internal error";
    default:
        return "unknown status";
    }
}

qsfi_status qsfi_context_create(const qsfi_context_desc* desc, qsfi_context** out)
{
    if (out == nullptr)
        return QSFI_STATUS_INVALID_ARGUMENT;
    *out = nullptr;
    qsfi_context* ctx = new (std::nothrow) qsfi_context;
    if (ctx == nullptr)
        return QSFI_STATUS_OUT_OF_MEMORY;
    ctx->device_ordinal = desc != nullptr ? desc->device_ordinal : -1;
    ctx->stream = desc != nullptr ? static_cast<cudaStream_t>(desc->stream) : nullptr;
    ctx->float_workspace = nullptr;
    ctx->float_workspace_bytes = 0;
    ctx->int_workspace_bytes = 0;
    ctx->host_int_workspace_bytes = 0;
    ctx->scratch_generation = 0;
    clear_error(&ctx->last_error);
    if (ctx->device_ordinal >= 0) {
        cudaError_t err = cudaSetDevice(ctx->device_ordinal);
        if (err != cudaSuccess) {
            set_cuda_error(ctx, err, "cudaSetDevice");
            delete ctx;
            return QSFI_STATUS_CUDA_ERROR;
        }
    }
    *out = ctx;
    return QSFI_STATUS_OK;
}

void qsfi_context_destroy(qsfi_context* ctx)
{
    if (ctx == nullptr)
        return;
    if (ctx->device_ordinal >= 0)
        cudaSetDevice(ctx->device_ordinal);
    if (ctx->float_workspace != nullptr)
        cudaFree(ctx->float_workspace);
    delete ctx;
}

qsfi_status qsfi_context_set_stream(qsfi_context* ctx, qsfi_cuda_stream stream)
{
    if (ctx == nullptr)
        return QSFI_STATUS_INVALID_ARGUMENT;
    ctx->stream = static_cast<cudaStream_t>(stream);
    clear_error(&ctx->last_error);
    return QSFI_STATUS_OK;
}

qsfi_status qsfi_get_build_config(qsfi_build_config* out)
{
    if (out == nullptr)
        return QSFI_STATUS_INVALID_ARGUMENT;
    out->target_sm = QSFI_TARGET_SM;
    out->target_compute_capability_major = QSFI_TARGET_COMPUTE_CAPABILITY_MAJOR;
    out->target_compute_capability_minor = QSFI_TARGET_COMPUTE_CAPABILITY_MINOR;
    out->assume_fp8 = QSFI_ENABLE_FP8;
    out->assume_fp4 = QSFI_ENABLE_FP4;
    out->assume_pdl = QSFI_ENABLE_PDL;
    out->gemm_backend = QSFI_GEMM_BACKEND;
    return QSFI_STATUS_OK;
}

qsfi_status qsfi_context_get_info(qsfi_context* ctx, qsfi_context_info* out)
{
    if (ctx == nullptr || out == nullptr)
        return QSFI_STATUS_INVALID_ARGUMENT;
    clear_error(&ctx->last_error);
    qsfi_status status = activate_context(ctx);
    if (status != QSFI_STATUS_OK)
        return status;
    int device = ctx->device_ordinal;
    if (device < 0) {
        cudaError_t err = cudaGetDevice(&device);
        if (err != cudaSuccess)
            return set_cuda_error(ctx, err, "cudaGetDevice");
    }
    cudaDeviceProp prop {};
    cudaError_t err = cudaGetDeviceProperties(&prop, device);
    if (err != cudaSuccess)
        return set_cuda_error(ctx, err, "cudaGetDeviceProperties");
    out->runtime_compute_capability_major = static_cast<uint32_t>(prop.major);
    out->runtime_compute_capability_minor = static_cast<uint32_t>(prop.minor);
    return QSFI_STATUS_OK;
}

qsfi_status qsfi_context_validate_target(qsfi_context* ctx)
{
    qsfi_context_info info {};
    qsfi_status status = qsfi_context_get_info(ctx, &info);
    if (status != QSFI_STATUS_OK)
        return status;
    if (info.runtime_compute_capability_major != QSFI_TARGET_COMPUTE_CAPABILITY_MAJOR
        || info.runtime_compute_capability_minor != QSFI_TARGET_COMPUTE_CAPABILITY_MINOR) {
        return set_unsupported(
            ctx,
            "runtime compute capability %u.%u does not match qsfi build target %u.%u",
            info.runtime_compute_capability_major,
            info.runtime_compute_capability_minor,
            QSFI_TARGET_COMPUTE_CAPABILITY_MAJOR,
            QSFI_TARGET_COMPUTE_CAPABILITY_MINOR
        );
    }
    return QSFI_STATUS_OK;
}

qsfi_status qsfi_context_reserve_workspace(
    qsfi_context* ctx,
    size_t float_workspace_bytes,
    size_t int_workspace_bytes,
    size_t host_int_workspace_bytes
)
{
    if (ctx == nullptr)
        return QSFI_STATUS_INVALID_ARGUMENT;
    clear_error(&ctx->last_error);
    qsfi_status status = activate_context(ctx);
    if (status != QSFI_STATUS_OK)
        return status;
    void* new_float = nullptr;
    if (float_workspace_bytes != 0) {
        cudaError_t err = cudaMalloc(&new_float, float_workspace_bytes);
        if (err != cudaSuccess)
            return set_cuda_error(ctx, err, "cudaMalloc float workspace");
    }
    if (ctx->float_workspace != nullptr)
        cudaFree(ctx->float_workspace);
    ctx->float_workspace = new_float;
    ctx->float_workspace_bytes = float_workspace_bytes;
    ctx->int_workspace_bytes = int_workspace_bytes;
    ctx->host_int_workspace_bytes = host_int_workspace_bytes;
    ctx->scratch_generation += 1;
    return QSFI_STATUS_OK;
}

qsfi_status qsfi_context_get_last_error(const qsfi_context* ctx, qsfi_error_info* out)
{
    if (ctx == nullptr || out == nullptr)
        return QSFI_STATUS_INVALID_ARGUMENT;
    *out = ctx->last_error;
    return QSFI_STATUS_OK;
}

void qsfi_context_clear_last_error(qsfi_context* ctx)
{
    if (ctx == nullptr)
        return;
    clear_error(&ctx->last_error);
}

} // extern "C"
