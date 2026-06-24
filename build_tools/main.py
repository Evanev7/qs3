def generate_dispatch_headers():
    CTA_TILE_Q_NS = [ 16, 32, 64, 128 ]
    QSFI_DISPATCH_CTA_TILE_Q = f"""\
#define QSFI_DISPATCH_CTA_TILE_Q(cta_tile_q, CTA_TILE_Q, ...)   \\
    switch (cta_tile_q) {{                                        \\
    {"".join(f"""\
    case {n}: {{                                               \\
            constexpr uint32_t CTA_TILE_Q = {n};                  \\
            __VA_ARGS__                                          \\
            break;                                               \\
    }}                                                            \\
    """ for n in CTA_TILE_Q_NS)}\
    default: {{                                               \\
            std::ostringstream err_msg;                          \\
            err_msg << "Unsupported cta_tile_q: " << cta_tile_q; \\
            FLASHINFER_ERROR(err_msg.str());                     \\
        }}                                                        \\
  }}
    """
    print(QSFI_DISPATCH_CTA_TILE_Q)

    GQA_GROUP_SIZE_NS = [ 1, 2, 3, 4, 8 ]
    QSFI_DISPATCH_GQA_GROUP_SIZE = f"""\
#define QSFI_DISPATCH_GQA_GROUP_SIZE(group_size, GROUP_SIZE, ...)  \\
    switch (group_size) {{                                          \\
    {"".join(f"""\
    case {n}: {{                                                  \\
            constexpr size_t GROUP_SIZE = {n};                       \\
            __VA_ARGS__                                            \\
            break;                                                 \\
    }}                                                              \\
    """ for n in GQA_GROUP_SIZE_NS)}\
    default: {{                                                 \\
            std::ostringstream err_msg;                            \\
            err_msg << "Unsupported group_size: " << group_size;   \\
            FLASHINFER_ERROR(err_msg.str());                       \\
        }}                                                          \\
}}
    """
    print(QSFI_DISPATCH_GQA_GROUP_SIZE)

    HEAD_DIM_NS = [64, 128, 256, 512]
    QSFI_DISPATCH_HEAD_DIM = f"""\
#define QSFI_DISPATCH_HEAD_DIM(head_dim, HEAD_DIM, ...)      \\
    switch (head_dim) {{                                      \\
    {"".join(f"""\
    case {n}: {{                                          \\
            constexpr size_t HEAD_DIM = {n};                 \\
            __VA_ARGS__                                      \\
            break;                                           \\
        }}                                                    \\
    """ for n in HEAD_DIM_NS)}\
    default: {{                                           \\
            std::ostringstream err_msg;                      \\
            err_msg << "Unsupported head_dim: " << head_dim; \\
            FLASHINFER_ERROR(err_msg.str());                 \\
        }}                                                    \\
}}
    """
    print(QSFI_DISPATCH_HEAD_DIM)

if __name__ == "__main__":
    generate_dispatch_headers()
