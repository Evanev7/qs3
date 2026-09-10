"""Build a disposable 24Q/4KV attention probe from the checked-out AOT source."""

import shlex
import subprocess
from pathlib import Path

root = Path.cwd()
probe = root / ".prototypes/qwen27_attention"
probe.mkdir(parents=True, exist_ok=True)
header = (root / "build/qsfi_macros.h").read_text()
start = header.index("#define QSFI_DISPATCH_GQA_GROUP_SIZE")
end = header.index("#define QSFI_DISPATCH_HEAD_DIM", start)
gqa = header[start:end]
case = gqa[gqa.index("    case 8:") : gqa.index("    default:")]
gqa = gqa.replace(
    case,
    case.replace("case 8:", "case 6:").replace("GROUP_SIZE = 8;", "GROUP_SIZE = 6;")
    + case,
)
(probe / "qsfi_macros.h").write_text(header[:start] + gqa + header[end:])
source = (root / "qsfi_attn.cu").read_text()
old = """    if (attention->head_dim_qk != 256 || attention->num_qo_heads != 16
        || attention->num_kv_heads != 2) {"""
new = """    if (attention->head_dim_qk != 256
        || !((attention->num_qo_heads == 16 && attention->num_kv_heads == 2)
             || (attention->num_qo_heads == 24 && attention->num_kv_heads == 4))) {"""
assert source.count(old) == 1
source = source.replace(old, new)
source = source.replace(
    "#include <flashinfer/attention/decode.cuh>",
    """#include <flashinfer/utils.cuh>
#undef DISPATCH_GQA_GROUP_SIZE
#define DISPATCH_GQA_GROUP_SIZE(...) QSFI_DISPATCH_GQA_GROUP_SIZE(__VA_ARGS__)
#include <flashinfer/attention/decode.cuh>""",
)
(probe / "qsfi_attn.cu").write_text(source)
(probe / "qsfi.cu").write_text(
    '#include "qsfi_attn.cu"\n'
    + "".join(
        f'#include "{root / name}"\n'
        for name in ["qsfi_context.cu", "qsfi_moe.cu", "qsfi_norm_rope.cu"]
    )
)
test = (
    (root / "test.cu")
    .read_text()
    .replace("constexpr int kQHeads = 16;", "constexpr int kQHeads = 24;")
    .replace("constexpr int kKvHeads = 2;", "constexpr int kKvHeads = 4;")
)
test = (
    test[: test.index("int main()")]
    + """int main() {
    if (!check_cuda(cudaSetDevice(0), "select device")) return 1;
    test_decode_append_uses_post_append_last_page_len();
    test_prefill_append_maps_positions_through_page_table();
    test_batch_decode_attention_matches_cpu_reference();
    test_batch_prefill_attention_matches_cpu_reference();
    if (failures) return 1;
    std::puts("24Q/4KV/256 AOT append, decode and prefill CPU-reference checks passed");
    return 0;
}
"""
)
(probe / "test.cu").write_text(test)
commands = subprocess.check_output(
    ["ninja", "-t", "commands", "qsfi.o"], cwd=root / "build", text=True
).splitlines()
compile_cmd = shlex.split(commands[-1])
assert compile_cmd[0] == "nvcc"
compile_cmd[0] = "/usr/local/cuda/bin/nvcc"
compile_cmd.insert(1, "-I" + str(root))
compile_cmd[compile_cmd.index("-c") + 1] = str(probe / "qsfi.cu")
compile_cmd[compile_cmd.index("-o") + 1] = str(probe / "qsfi.o")
print(shlex.join(compile_cmd), flush=True)
subprocess.run(compile_cmd, cwd=root / "build", check=True)
link = [
    "/usr/local/cuda/bin/nvcc",
    "-std=c++17",
    "-arch=sm_121",
    "--expt-relaxed-constexpr",
    "-DQSFI_ENABLE_CHECKED_VALIDATION=0",
    "-I" + str(root),
    "-I.",
    str(probe / "test.cu"),
    str(probe / "qsfi.o"),
    "qscu_gdn.o",
    "qscu.o",
    "qscb.o",
    "-lcuda",
    "-lcublas",
    "-lcublasLt",
    "-o",
    str(probe / "probe"),
]
subprocess.run(link, cwd=root / "build", check=True)
print(
    "Build complete. Execute .prototypes/qwen27_attention/probe separately.", flush=True
)
