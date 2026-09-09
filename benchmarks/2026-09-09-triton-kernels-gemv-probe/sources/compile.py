"""AOT the vendored triton_kernels matmul and copied b12x vocabulary GEMV."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys

import torch
import triton
from triton.backends.compiler import GPUTarget
from triton.backends.nvidia.compiler import get_ptxas
from triton.compiler import ASTSource

UPSTREAM = Path(__file__).parent / "upstream"
sys.path.insert(0, str(UPSTREAM))
from triton_kernels.matmul_details._matmul import _matmul
from b12x_kernel import _row_kernel


def matmul_constants(n, k, bn, bk):
    # Specialize unused host API features away. Runtime ABI is Y, X, W plus
    # Triton's two scratch pointers. Weight storage remains [N, K] row-major.
    values = dict.fromkeys(_matmul.arg_names)
    for name in ("Y", "X", "W"):
        del values[name]
    values.update(
        stride_y_k=0, stride_y_z=0, stride_y_m=n, stride_y_n=1,
        stride_x_z=0, stride_x_m=k, stride_x_k=1, X_TRANSPOSE=False,
        stride_w_e=0, stride_w_k=1, stride_w_n=k, W_TRANSPOSE=True,
        Y_ACC_IS_Y=False, M=1, N=n, K=k, K_W=k,
        batch_size=1, grid_m=1, grid_n=triton.cdiv(n, bn),
        activation_fn_args=(), ACTIVATION_REDUCTION_N=1, epilogue_fn_args=(),
        N_EXPTS_TOT=1, ALLOW_TF32=True, FLEXPOINT_SATURATE_INF=False,
        PER_BATCH_W_SCALE=False, PER_BATCH_OUT_SCALE=False, PER_BATCH_ACC_SCALE=False,
        BLOCK_M=16, BLOCK_N=bn, BLOCK_K=bk, GROUP_M=8, XCD_SWIZZLE=1,
        SWIZZLE_MX_VALUE="STRIDED", SWIZZLE_MX_SCALE="STRIDED",
        MX_BLOCK_SIZE=32, EPILOGUE_SUBTILE=1, EVEN_K=True, SPLIT_K=1,
        W_CACHE_MODIFIER="", NUM_SMS=0, OUT_N_TILE_ALIGNED=n % bn == 0,
        UPCAST_INDICES=False, SWAP_XW=False, IS_EPILOGUE_QUANT_MX=False,
        Y_VALUE_PACK_FACTOR=1, FLATTEN_LOOPS=True, reduce_rank=0, n_reduce_shards=1,
    )
    return values


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    n, k = 248320, 2048
    # FP32 is qs3's LM-head output. First matmul tile follows upstream's
    # nonpersistent heuristic; the rest are explicitly chosen tile probes.
    variants = [("b12x_row", _row_kernel, 1, 8, 1,
                 {"K": k, "BLOCK_K": k}, {"source": "*bf16", "weight": "*bf16", "output": "*fp32"})]
    for bn, bk, warps, stages in [(256, 64, 2, 2), (128, 64, 2, 4),
                                  (64, 64, 2, 4), (32, 128, 4, 4)]:
        variants.append((f"tk_n{bn}_k{bk}", _matmul, bn, warps, stages,
                         matmul_constants(n, k, bn, bk), {"Y": "*fp32", "X": "*bf16", "W": "*bf16"}))
    records = []
    for name, fn, rows, warps, stages, constants, signature in variants:
        options = dict(num_warps=warps, num_stages=stages, enable_fp_fusion=False)
        # cudaMalloc supplies the 16-byte alignment the normal JIT launcher
        # specializes on. Carry that information into the AOT compilation.
        attrs = {(fn.arg_names.index(arg),): [["tt.divisibility", 16]] for arg in signature}
        compiled = triton.compile(ASTSource(fn=fn, signature=signature, constexprs=constants, attrs=attrs),
                                  target=GPUTarget("cuda", 121, 32), options=options)
        meta = compiled.metadata
        if meta.global_scratch_size or meta.profile_scratch_size or meta.num_ctas != 1:
            raise RuntimeError("kernel needs scratch or a multi-CTA launch")
        cubin = compiled.asm["cubin"]
        (args.out / (name + ".cubin")).write_bytes(cubin)
        (args.out / (name + ".ptx")).write_text(compiled.asm["ptx"])
        records.append(dict(name=name, n=n, k=k, dtype="fp32", rows=rows, warps=warps,
                            matmul=fn is _matmul, constants=constants, options=options, pointer_alignment=16,
                            file=name + ".cubin", symbol=meta.name, shared=meta.shared,
                            sha256=hashlib.sha256(cubin).hexdigest()))
        print(f"Exported {name}: {meta.shared} bytes shared", flush=True)
    source_paths = [Path(__file__), Path(__file__).with_name("b12x_kernel.py")]
    source_paths += sorted((UPSTREAM / "triton_kernels").rglob("*.py"))
    source_hashes = {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in source_paths}
    assembler = get_ptxas(121)
    manifest = dict(triton_version=triton.__version__, torch_version=torch.__version__,
                    target="cuda:121:32", source_hashes=source_hashes,
                    upstream_revision="c01b6774b1865984607d89d89d3a10833de92037",
                    ptxas_path=assembler.path,
                    ptxas_version=subprocess.check_output([assembler.path, "--version"], text=True),
                    ptxas_sha256=hashlib.sha256(Path(assembler.path).read_bytes()).hexdigest(), kernels=records)
    (args.out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    entries = ['{"%s", "%s", "%s", %d, %d, %d, %d, %d, true, %s}' %
               (r["name"], r["file"], r["symbol"], n, k, r["rows"], r["warps"], r["shared"],
                str(r["matmul"]).lower()) for r in records]
    (args.out / "kernels.h").write_text(
        "struct KernelSpec { const char *name, *file, *symbol; "
        "int n, k, rows, warps, shared; bool fp32, matmul; };\n"
        "static const KernelSpec kernels[] = {\n" + ",\n".join(entries) + "\n};\n")
    (args.out / "kernels.d").write_text(str(args.out / "kernels.h") + ": " +
                                       " ".join(map(str, source_paths)) + "\n")


if __name__ == "__main__":
    main()
