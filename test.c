#include "flashinfer.h"

#include <stdio.h>
#include <string.h>

static int failures = 0;

static void check(int condition, const char* what)
{
    if (!condition) {
        fprintf(stderr, "FAIL: %s\n", what);
        failures += 1;
    }
}

static void check_status(qsfi_status_t got, qsfi_status_t want, const char* what)
{
    if (got != want) {
        fprintf(
            stderr,
            "FAIL: %s: got %s (%d), want %s (%d)\n",
            what,
            qsfi_status_string(got),
            (int)got,
            qsfi_status_string(want),
            (int)want
        );
        failures += 1;
    }
}

static void check_status_strings(void)
{
    check(strcmp(qsfi_status_string(QSFI_STATUS_OK), "ok") == 0, "OK status string");
    check(qsfi_status_string((qsfi_status_t)999) != NULL, "unknown status string is non-null");
}

static void check_null_contracts(void)
{
    qsfi_plan_kind_t kind = QSFI_PLAN_BATCH_DECODE;
    qsfi_error_info_t err;

    check_status(
        qsfi_context_create(NULL, NULL),
        QSFI_STATUS_INVALID_ARGUMENT,
        "context_create rejects null out"
    );
    check_status(
        qsfi_context_get_last_error(NULL, &err),
        QSFI_STATUS_INVALID_ARGUMENT,
        "get_last_error rejects null ctx"
    );
    check_status(
        qsfi_context_get_last_error((const qsfi_context_t*)1, NULL),
        QSFI_STATUS_INVALID_ARGUMENT,
        "get_last_error rejects null out"
    );
    check_status(
        qsfi_plan_kind(NULL, &kind),
        QSFI_STATUS_INVALID_ARGUMENT,
        "plan_kind rejects null plan"
    );
    check_status(
        qsfi_plan_kind((const qsfi_plan_t*)1, NULL),
        QSFI_STATUS_INVALID_ARGUMENT,
        "plan_kind rejects null out"
    );
}

static void check_context_lifecycle(void)
{
    qsfi_context_desc_t desc;
    qsfi_context_t* ctx = NULL;
    qsfi_error_info_t err;
    qsfi_plan_t* plan = NULL;

    memset(&desc, 0, sizeof(desc));
    desc.device_ordinal = -1;
    desc.stream = NULL;

    check_status(
        qsfi_context_create(&desc, &ctx),
        QSFI_STATUS_OK,
        "context_create with no selected device"
    );
    check(ctx != NULL, "context_create returns context");
    if (ctx == NULL)
        return;

    check_status(
        qsfi_context_get_last_error(ctx, &err),
        QSFI_STATUS_OK,
        "get_last_error on fresh context"
    );
    check_status(err.status, QSFI_STATUS_OK, "fresh context last error is OK");

    check_status(
        qsfi_context_reserve_scratch(ctx, 0, 0, 0),
        QSFI_STATUS_OK,
        "reserve zero scratch"
    );
    check_status(
        qsfi_load_kernels(ctx, QSFI_KERNEL_MODULE_NONE),
        QSFI_STATUS_OK,
        "load no kernels"
    );

    check_status(
        qsfi_load_kernels(ctx, (qsfi_kernel_flags_t)0x80000000u),
        QSFI_STATUS_INVALID_ARGUMENT,
        "load rejects unknown kernel flag"
    );
    check_status(
        qsfi_context_get_last_error(ctx, &err),
        QSFI_STATUS_OK,
        "get_last_error after failure"
    );
    check_status(err.status, QSFI_STATUS_INVALID_ARGUMENT, "last error stores failure status");
    check(err.source == QSFI_ERROR_SOURCE_QSFI, "last error source is QSFI");
    check(err.message[0] != '\0', "last error stores message");

    qsfi_context_clear_last_error(ctx);
    check_status(
        qsfi_context_get_last_error(ctx, &err),
        QSFI_STATUS_OK,
        "get_last_error after clear"
    );
    check_status(err.status, QSFI_STATUS_OK, "clear resets status");

    check_status(
        qsfi_batch_decode_plan_create(ctx, NULL, NULL, &plan),
        QSFI_STATUS_INVALID_ARGUMENT,
        "decode plan rejects missing scratch/attention"
    );
    check(plan == NULL, "decode plan leaves out null on failure");

    qsfi_context_destroy(ctx);
}

int main(void)
{
    check_status_strings();
    check_null_contracts();
    check_context_lifecycle();

    if (failures != 0) {
        fprintf(stderr, "%d failure(s)\n", failures);
        return 1;
    }

    puts("flashinfer.h smoke test passed");
    return 0;
}
