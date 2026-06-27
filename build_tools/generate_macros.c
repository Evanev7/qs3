#if 0
exec tcc -run "$0" "$@"
#endif

#include <stdio.h>

typedef struct qsfi_dispatch_spec {
    const char* macro_name;
    const char* runtime_name;
    const char* template_name;
    const char* template_type;
    const char* unsupported_label;
    const int* values;
    int value_count;
} qsfi_dispatch_spec;

static void emit_build_constants(void)
{
    printf("#define QSFI_TARGET_SM 121u\n");
    printf("#define QSFI_TARGET_COMPUTE_CAPABILITY_MAJOR 12u\n");
    printf("#define QSFI_TARGET_COMPUTE_CAPABILITY_MINOR 1u\n\n");

    printf("#define QSFI_ENABLE_FP8 1u\n");
    printf("#define QSFI_ENABLE_FP4 1u\n");
    printf("#define QSFI_ENABLE_PDL 1u\n\n");

    printf("#define QSFI_GEMM_BACKEND_CUBLASLT 1u\n");
    printf("#define QSFI_GEMM_BACKEND_FLASHINFER 2u\n");
    printf("#define QSFI_GEMM_BACKEND_CUTLASS 3u\n\n");

    printf("#define QSFI_GEMM_BACKEND QSFI_GEMM_BACKEND_CUBLASLT\n\n");

    printf("#define QSFI_QWEN36_GDN_NUM_Q_HEADS 16u\n");
    printf("#define QSFI_QWEN36_GDN_NUM_K_HEADS 16u\n");
    printf("#define QSFI_QWEN36_GDN_NUM_V_HEADS 32u\n");
    printf("#define QSFI_QWEN36_GDN_KEY_DIM 128u\n");
    printf("#define QSFI_QWEN36_GDN_VALUE_DIM 128u\n");
    printf("#define QSFI_QWEN36_GDN_THREADS 128u\n");
    printf("#define QSFI_QWEN36_GDN_SOFTPLUS_BETA 1.0f\n");
    printf("#define QSFI_QWEN36_GDN_SOFTPLUS_THRESHOLD 20.0f\n\n");
}

static void emit_dispatch_case(const qsfi_dispatch_spec* spec, int value)
{
    printf("    case %d: { \\\n", value);
    printf("        constexpr %s %s = %d; \\\n", spec->template_type, spec->template_name, value);
    printf("        __VA_ARGS__ \\\n");
    printf("        break; \\\n");
    printf("    } \\\n");
}

static void emit_dispatch_macro(const qsfi_dispatch_spec* spec)
{
    printf(
        "#define %s(%s, %s, ...) \\\n",
        spec->macro_name,
        spec->runtime_name,
        spec->template_name
    );
    printf("    switch (%s) { \\\n", spec->runtime_name);

    for (int i = 0; i < spec->value_count; ++i) {
        emit_dispatch_case(spec, spec->values[i]);
    }

    printf("    default: { \\\n");
    printf("        std::ostringstream err_msg; \\\n");
    printf(
        "        err_msg << \"Unsupported %s: \" << %s; \\\n",
        spec->unsupported_label,
        spec->runtime_name
    );
    printf("        FLASHINFER_ERROR(err_msg.str()); \\\n");
    printf("    } \\\n");
    printf("}\n");
    putchar('\n');
}

int main(void)
{
    // static const int cta_tile_q_values[] = { 16, 32, 64, 128 };
    // static const int gqa_group_size_values[] = { 1, 2, 3, 4, 8 };
    // static const int head_dim_values[] = { 64, 128, 256, 512 };
    static const int cta_tile_q_values[] = { 16 };
    static const int gqa_group_size_values[] = { 1 };
    static const int head_dim_values[] = { 64 };

    static const qsfi_dispatch_spec specs[] = {
        {
            "QSFI_DISPATCH_CTA_TILE_Q",
            "cta_tile_q",
            "CTA_TILE_Q",
            "uint32_t",
            "cta_tile_q",
            cta_tile_q_values,
            (int)(sizeof(cta_tile_q_values) / sizeof(cta_tile_q_values[0])),
        },
        {
            "QSFI_DISPATCH_GQA_GROUP_SIZE",
            "group_size",
            "GROUP_SIZE",
            "size_t",
            "group_size",
            gqa_group_size_values,
            (int)(sizeof(gqa_group_size_values) / sizeof(gqa_group_size_values[0])),
        },
        {
            "QSFI_DISPATCH_HEAD_DIM",
            "head_dim",
            "HEAD_DIM",
            "size_t",
            "head_dim",
            head_dim_values,
            (int)(sizeof(head_dim_values) / sizeof(head_dim_values[0])),
        },
    };

    printf("#ifndef QSFI_MACROS_H\n");
    printf("#define QSFI_MACROS_H\n\n");

    {
        emit_build_constants();

        for (int i = 0; i < (int)(sizeof(specs) / sizeof(specs[0])); ++i) {
            emit_dispatch_macro(&specs[i]);
        }
    }

    printf("#endif\n");
    return 0;
}
