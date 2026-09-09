#include "qscb.h"
#include "qsfi_native_common.h"

#include <cublasLt.h>
#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <cstdarg>
#include <cstdint>
#include <cstdio>
#include <memory>
#include <new>
#include <tuple>

struct qscb_context {
    cublasLtHandle_t lt;
    int32_t device_ordinal;
    cudaStream_t stream;
    qsfi_error_info last_error;
};

struct linear_descriptors {
    cublasLtMatmulDesc_t matmul;
    cublasLtMatrixLayout_t a;
    cublasLtMatrixLayout_t b;
    cublasLtMatrixLayout_t d;
    cublasLtMatmulPreference_t preference;

    linear_descriptors()
        : matmul(nullptr)
        , a(nullptr)
        , b(nullptr)
        , d(nullptr)
        , preference(nullptr)
    {
    }

    ~linear_descriptors()
    {
        if (preference != nullptr)
            cublasLtMatmulPreferenceDestroy(preference);
        if (d != nullptr)
            cublasLtMatrixLayoutDestroy(d);
        if (b != nullptr)
            cublasLtMatrixLayoutDestroy(b);
        if (a != nullptr)
            cublasLtMatrixLayoutDestroy(a);
        if (matmul != nullptr)
            cublasLtMatmulDescDestroy(matmul);
    }
    linear_descriptors(const linear_descriptors&) = delete;
    linear_descriptors& operator=(const linear_descriptors&) = delete;
};

// Only the low 8 address bits matter: cuBLASLt defaults to 256-byte alignment.
static uint32_t linear_alignment(const void* ptr)
{
    const uintptr_t address = reinterpret_cast<uintptr_t>(ptr);
    uint32_t alignment = 256;
    while (address % alignment != 0)
        alignment /= 2;
    return alignment;
}

static auto linear_key(const qscb_linear_desc& desc)
{
    return std::make_tuple(
        desc.rows,
        desc.in_features,
        desc.out_features,
        desc.x.stride[0],
        desc.weight.stride[0],
        desc.out.stride[0],
        desc.out.dtype,
        linear_alignment(desc.x.data),
        linear_alignment(desc.weight.data),
        linear_alignment(desc.out.data),
        desc.workspace_bytes
    );
}

struct qscb_linear_plan {
    linear_descriptors descriptors;
    cublasLtMatmulAlgo_t algorithm {};
    decltype(linear_key(qscb_linear_desc {})) key;
    qscb_context* owner = nullptr;
};

namespace {

const char* cublas_status_name(cublasStatus_t status)
{
    switch (status) {
    case CUBLAS_STATUS_SUCCESS:
        return "CUBLAS_STATUS_SUCCESS";
    case CUBLAS_STATUS_NOT_INITIALIZED:
        return "CUBLAS_STATUS_NOT_INITIALIZED";
    case CUBLAS_STATUS_ALLOC_FAILED:
        return "CUBLAS_STATUS_ALLOC_FAILED";
    case CUBLAS_STATUS_INVALID_VALUE:
        return "CUBLAS_STATUS_INVALID_VALUE";
    case CUBLAS_STATUS_ARCH_MISMATCH:
        return "CUBLAS_STATUS_ARCH_MISMATCH";
    case CUBLAS_STATUS_MAPPING_ERROR:
        return "CUBLAS_STATUS_MAPPING_ERROR";
    case CUBLAS_STATUS_EXECUTION_FAILED:
        return "CUBLAS_STATUS_EXECUTION_FAILED";
    case CUBLAS_STATUS_INTERNAL_ERROR:
        return "CUBLAS_STATUS_INTERNAL_ERROR";
    case CUBLAS_STATUS_NOT_SUPPORTED:
        return "CUBLAS_STATUS_NOT_SUPPORTED";
    case CUBLAS_STATUS_LICENSE_ERROR:
        return "CUBLAS_STATUS_LICENSE_ERROR";
    default:
        return "CUBLAS_STATUS_UNKNOWN";
    }
}

qsfi_status set_qscb_errorv(
    qscb_context* ctx,
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

qsfi_status set_qscb_error(
    qscb_context* ctx,
    qsfi_status status,
    qsfi_error_source source,
    int32_t native_code,
    const char* fmt,
    ...
)
{
    va_list args;
    va_start(args, fmt);
    status = set_qscb_errorv(ctx, status, source, native_code, fmt, args);
    va_end(args);
    return status;
}

qsfi_status set_qscb_invalid_arg(qscb_context* ctx, const char* fmt, ...)
{
    va_list args;
    va_start(args, fmt);
    qsfi_status status
        = set_qscb_errorv(ctx, QSFI_STATUS_INVALID_ARGUMENT, QSFI_ERROR_SOURCE_QSFI, 0, fmt, args);
    va_end(args);
    return status;
}

qsfi_status set_qscb_cuda_error(qscb_context* ctx, cudaError_t err, const char* op)
{
    if (err == cudaSuccess)
        return QSFI_STATUS_OK;
    return set_qscb_error(
        ctx,
        QSFI_STATUS_CUDA_ERROR,
        QSFI_ERROR_SOURCE_CUDA,
        static_cast<int32_t>(err),
        "%s: %s",
        op,
        cudaGetErrorString(err)
    );
}

qsfi_status set_qscb_cublaslt_error(qscb_context* ctx, cublasStatus_t status, const char* op)
{
    if (status == CUBLAS_STATUS_SUCCESS)
        return QSFI_STATUS_OK;
    return set_qscb_error(
        ctx,
        QSFI_STATUS_BACKEND_ERROR,
        QSFI_ERROR_SOURCE_CUBLASLT,
        static_cast<int32_t>(status),
        "%s: %s",
        op,
        cublas_status_name(status)
    );
}

cudaDataType_t output_data_type(qsfi_dtype dtype)
{
    return dtype == QSFI_DTYPE_F32 ? CUDA_R_32F : CUDA_R_16BF;
}

bool supported_output_dtype(qsfi_dtype dtype)
{
    return dtype == QSFI_DTYPE_BF16 || dtype == QSFI_DTYPE_F32;
}

bool tensor2_is_row_major(const qsfi_tensor2& tensor)
{
    return tensor.stride[1] == 1 && tensor.stride[0] >= tensor.shape[1];
}

struct qscb_error_reporter {
    qscb_context* ctx;

    template <typename... Args> qsfi_status invalid_arg(const char* fmt, Args... args) const
    {
        return set_qscb_invalid_arg(ctx, fmt, args...);
    }
};

template <typename Tensor>
qsfi_status qscb_validate_tensor(
    qscb_context* ctx,
    const Tensor& tensor,
    const char* name,
    qsfi_dtype dtype,
    uint32_t expected_rank
)
{
    return qsfi_validate_native_tensor(
        qscb_error_reporter { ctx },
        tensor,
        name,
        dtype,
        expected_rank
    );
}

qsfi_status validate_shape(
    qscb_context* ctx, const qsfi_tensor2& tensor, const char* name, int64_t rows, int64_t cols
)
{
    if (tensor.shape[0] != rows || tensor.shape[1] != cols) {
        return set_qscb_invalid_arg(
            ctx,
            "%s shape must be [%lld, %lld]",
            name,
            static_cast<long long>(rows),
            static_cast<long long>(cols)
        );
    }
    if (!tensor2_is_row_major(tensor)) {
        return set_qscb_invalid_arg(ctx, "%s must be row-major with stride[1] == 1", name);
    }
    return QSFI_STATUS_OK;
}

qsfi_status validate_linear_desc(qscb_context* ctx, const qscb_linear_desc* desc)
{
    if (desc == nullptr) {
        return set_qscb_invalid_arg(ctx, "qscb linear desc is null");
    }
    if (desc->rows == 0 || desc->in_features == 0 || desc->out_features == 0) {
        return set_qscb_invalid_arg(ctx, "qscb linear dimensions must be non-zero");
    }
    if (desc->workspace == nullptr && desc->workspace_bytes != 0) {
        return set_qscb_invalid_arg(
            ctx,
            "qscb linear workspace is null but workspace_bytes is set"
        );
    }

    if (desc->workspace != nullptr && linear_alignment(desc->workspace) < 256) {
        return set_qscb_invalid_arg(ctx, "qscb linear workspace must be 256-byte aligned");
    }

    qsfi_status status = qscb_validate_tensor(ctx, desc->x, "x", QSFI_DTYPE_BF16, 2);
    if (status != QSFI_STATUS_OK)
        return status;
    status = qscb_validate_tensor(ctx, desc->weight, "weight", QSFI_DTYPE_BF16, 2);
    if (status != QSFI_STATUS_OK)
        return status;
    if (!supported_output_dtype(desc->out.dtype)) {
        return set_qscb_invalid_arg(ctx, "out dtype must be bf16 or f32");
    }
    status = qscb_validate_tensor(ctx, desc->out, "out", desc->out.dtype, 2);
    if (status != QSFI_STATUS_OK)
        return status;

    const int64_t rows = static_cast<int64_t>(desc->rows);
    const int64_t in_features = static_cast<int64_t>(desc->in_features);
    const int64_t out_features = static_cast<int64_t>(desc->out_features);
    status = validate_shape(ctx, desc->x, "x", rows, in_features);
    if (status != QSFI_STATUS_OK)
        return status;
    status = validate_shape(ctx, desc->weight, "weight", out_features, in_features);
    if (status != QSFI_STATUS_OK)
        return status;
    return validate_shape(ctx, desc->out, "out", rows, out_features);
}

qsfi_status activate_qscb_context(qscb_context* ctx)
{
    if (ctx->device_ordinal < 0)
        return QSFI_STATUS_OK;
    cudaError_t err = cudaSetDevice(ctx->device_ordinal);
    return set_qscb_cuda_error(ctx, err, "cudaSetDevice");
}

cublasStatus_t create_descriptors(const qscb_linear_desc* desc, linear_descriptors* out)
{
    cublasStatus_t status = cublasLtMatmulDescCreate(&out->matmul, CUBLAS_COMPUTE_32F, CUDA_R_32F);
    if (status != CUBLAS_STATUS_SUCCESS)
        return status;

    cublasOperation_t transa = CUBLAS_OP_T;
    status = cublasLtMatmulDescSetAttribute(
        out->matmul,
        CUBLASLT_MATMUL_DESC_TRANSA,
        &transa,
        sizeof(transa)
    );
    if (status != CUBLAS_STATUS_SUCCESS)
        return status;
    cublasOperation_t transb = CUBLAS_OP_N;
    status = cublasLtMatmulDescSetAttribute(
        out->matmul,
        CUBLASLT_MATMUL_DESC_TRANSB,
        &transb,
        sizeof(transb)
    );
    if (status != CUBLAS_STATUS_SUCCESS)
        return status;

    status = cublasLtMatrixLayoutCreate(
        &out->a,
        CUDA_R_16BF,
        desc->in_features,
        desc->out_features,
        desc->weight.stride[0]
    );
    if (status != CUBLAS_STATUS_SUCCESS)
        return status;
    status = cublasLtMatrixLayoutCreate(
        &out->b,
        CUDA_R_16BF,
        desc->in_features,
        desc->rows,
        desc->x.stride[0]
    );
    if (status != CUBLAS_STATUS_SUCCESS)
        return status;
    status = cublasLtMatrixLayoutCreate(
        &out->d,
        output_data_type(desc->out.dtype),
        desc->out_features,
        desc->rows,
        desc->out.stride[0]
    );
    if (status != CUBLAS_STATUS_SUCCESS)
        return status;

    status = cublasLtMatmulPreferenceCreate(&out->preference);
    if (status != CUBLAS_STATUS_SUCCESS)
        return status;
    status = cublasLtMatmulPreferenceSetAttribute(
        out->preference,
        CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES,
        &desc->workspace_bytes,
        sizeof(desc->workspace_bytes)
    );
    if (status != CUBLAS_STATUS_SUCCESS)
        return status;
    const cublasLtMatmulPreferenceAttributes_t attributes[] = {
        CUBLASLT_MATMUL_PREF_MIN_ALIGNMENT_A_BYTES,
        CUBLASLT_MATMUL_PREF_MIN_ALIGNMENT_B_BYTES,
        CUBLASLT_MATMUL_PREF_MIN_ALIGNMENT_C_BYTES,
        CUBLASLT_MATMUL_PREF_MIN_ALIGNMENT_D_BYTES,
    };
    const uint32_t alignments[] = {
        linear_alignment(desc->weight.data),
        linear_alignment(desc->x.data),
        linear_alignment(desc->out.data),
        linear_alignment(desc->out.data),
    };
    for (int i = 0; i < 4; ++i) {
        status = cublasLtMatmulPreferenceSetAttribute(
            out->preference,
            attributes[i],
            &alignments[i],
            sizeof(alignments[i])
        );
        if (status != CUBLAS_STATUS_SUCCESS)
            return status;
    }
    return CUBLAS_STATUS_SUCCESS;
}

qsfi_status prepare_linear(qscb_context* ctx, const qscb_linear_desc* desc, qscb_linear_plan* plan)
{
    auto& descriptors = plan->descriptors;
    cublasStatus_t status = create_descriptors(desc, &descriptors);
    if (status != CUBLAS_STATUS_SUCCESS)
        return set_qscb_cublaslt_error(ctx, status, "cublasLt descriptor create");

    cublasLtMatmulHeuristicResult_t heuristic {};
    int returned_count = 0;
    status = cublasLtMatmulAlgoGetHeuristic(
        ctx->lt,
        descriptors.matmul,
        descriptors.a,
        descriptors.b,
        descriptors.d,
        descriptors.d,
        descriptors.preference,
        1,
        &heuristic,
        &returned_count
    );
    if (status != CUBLAS_STATUS_SUCCESS)
        return set_qscb_cublaslt_error(ctx, status, "cublasLtMatmulAlgoGetHeuristic");
    if (returned_count == 0) {
        return set_qscb_error(
            ctx,
            QSFI_STATUS_BACKEND_ERROR,
            QSFI_ERROR_SOURCE_CUBLASLT,
            0,
            "cublasLtMatmulAlgoGetHeuristic returned no qscb linear algorithms"
        );
    }

    if (heuristic.state != CUBLAS_STATUS_SUCCESS)
        return set_qscb_cublaslt_error(ctx, heuristic.state, "qscb linear heuristic state");
    plan->algorithm = heuristic.algo;
    plan->key = linear_key(*desc);
    plan->owner = ctx;
    return QSFI_STATUS_OK;
}

qsfi_status
run_linear(qscb_context* ctx, const qscb_linear_plan* plan, const qscb_linear_desc* desc)
{
    const auto& descriptors = plan->descriptors;
    const float alpha = qsfi_default_one(desc->alpha);
    const float beta = desc->beta;
    const void* c = beta == 0.0f ? nullptr : desc->out.data;
    cublasStatus_t status = cublasLtMatmul(
        ctx->lt,
        descriptors.matmul,
        &alpha,
        desc->weight.data,
        descriptors.a,
        desc->x.data,
        descriptors.b,
        &beta,
        c,
        descriptors.d,
        desc->out.data,
        descriptors.d,
        &plan->algorithm,
        desc->workspace,
        desc->workspace_bytes,
        ctx->stream
    );
    return set_qscb_cublaslt_error(ctx, status, "cublasLtMatmul qscb linear");
}

} // namespace

extern "C" {

qsfi_status qscb_context_create(const qscb_context_desc* desc, qscb_context** out)
{
    if (out == nullptr)
        return QSFI_STATUS_INVALID_ARGUMENT;
    *out = nullptr;
    if (desc == nullptr)
        return QSFI_STATUS_INVALID_ARGUMENT;

    qscb_context* ctx = new (std::nothrow) qscb_context;
    if (ctx == nullptr)
        return QSFI_STATUS_OUT_OF_MEMORY;
    ctx->lt = nullptr;
    ctx->device_ordinal = desc->device_ordinal;
    ctx->stream = static_cast<cudaStream_t>(desc->stream);
    qsfi_clear_error_info(&ctx->last_error);

    cudaError_t err = cudaSuccess;
    if (ctx->device_ordinal >= 0) {
        err = cudaSetDevice(ctx->device_ordinal);
        if (err != cudaSuccess) {
            delete ctx;
            return QSFI_STATUS_CUDA_ERROR;
        }
    } else {
        int device = -1;
        err = cudaGetDevice(&device);
        if (err != cudaSuccess) {
            delete ctx;
            return QSFI_STATUS_CUDA_ERROR;
        }
        ctx->device_ordinal = device;
    }

    cublasStatus_t status = cublasLtCreate(&ctx->lt);
    if (status != CUBLAS_STATUS_SUCCESS) {
        delete ctx;
        return QSFI_STATUS_BACKEND_ERROR;
    }
    *out = ctx;
    return QSFI_STATUS_OK;
}

void qscb_context_destroy(qscb_context* ctx)
{
    if (ctx == nullptr)
        return;
    if (ctx->device_ordinal >= 0)
        cudaSetDevice(ctx->device_ordinal);
    if (ctx->lt != nullptr)
        cublasLtDestroy(ctx->lt);
    delete ctx;
}

qsfi_status qscb_context_get_last_error(const qscb_context* ctx, qsfi_error_info* out)
{
    if (ctx == nullptr || out == nullptr)
        return QSFI_STATUS_INVALID_ARGUMENT;
    *out = ctx->last_error;
    return QSFI_STATUS_OK;
}

void qscb_context_clear_last_error(qscb_context* ctx)
{
    if (ctx != nullptr)
        qsfi_clear_error_info(&ctx->last_error);
}

qsfi_status
qscb_linear_plan_create(qscb_context* ctx, const qscb_linear_desc* desc, qscb_linear_plan** out)
{
    if (out == nullptr)
        return QSFI_STATUS_INVALID_ARGUMENT;
    *out = nullptr;
    if (ctx == nullptr)
        return QSFI_STATUS_INVALID_ARGUMENT;
    qsfi_clear_error_info(&ctx->last_error);

    qsfi_status status = activate_qscb_context(ctx);
    if (status != QSFI_STATUS_OK)
        return status;
    status = validate_linear_desc(ctx, desc);
    if (status != QSFI_STATUS_OK)
        return status;
    std::unique_ptr<qscb_linear_plan> plan(new (std::nothrow) qscb_linear_plan);
    if (!plan)
        return QSFI_STATUS_OUT_OF_MEMORY;
    status = prepare_linear(ctx, desc, plan.get());
    if (status != QSFI_STATUS_OK)
        return status;
    *out = plan.release();
    return QSFI_STATUS_OK;
}

void qscb_linear_plan_destroy(qscb_linear_plan* plan)
{
    delete plan;
}

qsfi_status
qscb_linear_execute(qscb_context* ctx, const qscb_linear_plan* plan, const qscb_linear_desc* desc)
{
    if (ctx == nullptr)
        return QSFI_STATUS_INVALID_ARGUMENT;
    qsfi_clear_error_info(&ctx->last_error);
    if (plan == nullptr || plan->owner != ctx)
        return set_qscb_invalid_arg(ctx, "qscb linear plan must belong to the execution context");
    qsfi_status status = validate_linear_desc(ctx, desc);
    if (status != QSFI_STATUS_OK)
        return status;
    if (plan->key != linear_key(*desc))
        return set_qscb_invalid_arg(ctx, "qscb linear plan layout/alignment/workspace mismatch");
    status = activate_qscb_context(ctx);
    if (status != QSFI_STATUS_OK)
        return status;
    return run_linear(ctx, plan, desc);
}

} // extern "C"
