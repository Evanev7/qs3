#ifndef QS_INFO_H
#define QS_INFO_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#define QSFI_ERROR_MESSAGE_BYTES 512u

#ifndef QSFI_ENABLE_CHECKED_VALIDATION
#ifdef NDEBUG
#define QSFI_ENABLE_CHECKED_VALIDATION 0
#else
#define QSFI_ENABLE_CHECKED_VALIDATION 1
#endif
#endif

typedef enum {
    QSFI_STATUS_OK = 0,
    QSFI_STATUS_INVALID_ARGUMENT = 1,
    QSFI_STATUS_UNSUPPORTED = 2,
    QSFI_STATUS_OUT_OF_MEMORY = 3,
    QSFI_STATUS_CUDA_ERROR = 4,
    QSFI_STATUS_BACKEND_ERROR = 5,
    QSFI_STATUS_INTERNAL_ERROR = 6
} qsfi_status;

typedef enum {
    QSFI_ERROR_SOURCE_NONE = 0,
    QSFI_ERROR_SOURCE_QSFI = 1,
    QSFI_ERROR_SOURCE_CUDA = 2,
    QSFI_ERROR_SOURCE_FLASHINFER = 3,
    QSFI_ERROR_SOURCE_CUBLASLT = 4
} qsfi_error_source;

typedef struct {
    qsfi_status status;
    qsfi_error_source source;
    int32_t native_code;
    char message[QSFI_ERROR_MESSAGE_BYTES];
} qsfi_error_info;

const char* qsfi_status_string(qsfi_status status);

#ifdef __cplusplus
}
#endif

#endif
