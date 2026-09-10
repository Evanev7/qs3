"""AOT comparison of copied b12x vocabulary GEMV and the qs3 probe kernel."""

import argparse
import hashlib
import json
import subprocess
from pathlib import Path

import triton
from b12x_kernel import _row_kernel, _row_loop_kernel
from kernel import gemv
from triton.backends.compiler import GPUTarget
from triton.backends.nvidia.compiler import get_ptxas
from triton.compiler import ASTSource


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    n, k = 248320, 2048
    variants = [
        ("qs3_row", gemv, 4, {"N": n, "K": k, "ROWS": 1}, ("X", "W", "Y")),
        (
            "b12x_row",
            _row_kernel,
            4,
            {"K": k, "BLOCK_K": k},
            ("source", "weight", "output"),
        ),
        # The upstream vocabulary policy defaults to row / 8 warps.
        (
            "b12x_row",
            _row_kernel,
            8,
            {"K": k, "BLOCK_K": k},
            ("source", "weight", "output"),
        ),
        (
            "b12x_loop512",
            _row_loop_kernel,
            4,
            {"K": k, "BLOCK_K": 512},
            ("source", "weight", "output"),
        ),
    ]
    records = []
    for dtype in ("fp32", "bf16"):
        for name, fn, warps, constants, parameters in variants:
            signature = dict(zip(parameters, ("*bf16", "*bf16", "*" + dtype)))
            src = ASTSource(fn=fn, signature=signature, constexprs=constants)
            compiled = triton.compile(
                src,
                target=GPUTarget("cuda", 121, 32),
                options={
                    "num_warps": warps,
                    "num_stages": 1,
                    "enable_fp_fusion": False,
                },
            )
            meta = compiled.metadata
            if (
                meta.global_scratch_size
                or meta.profile_scratch_size
                or meta.num_ctas != 1
            ):
                raise RuntimeError("kernel needs scratch or a multi-CTA launch")
            stem = f"{name}_{dtype}_w{warps}"
            cubin = compiled.asm["cubin"]
            (args.out / (stem + ".cubin")).write_bytes(cubin)
            (args.out / (stem + ".ptx")).write_text(compiled.asm["ptx"])
            records.append(
                dict(
                    name=name,
                    n=n,
                    k=k,
                    dtype=dtype,
                    rows=1,
                    warps=warps,
                    constants=constants,
                    file=stem + ".cubin",
                    symbol=meta.name,
                    shared=meta.shared,
                    sha256=hashlib.sha256(cubin).hexdigest(),
                )
            )
    source_hashes = {
        name: hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest()
        for name in ("compile.py", "kernel.py", "b12x_kernel.py")
    }
    assembler = get_ptxas(121)
    manifest = dict(
        triton_version=triton.__version__,
        target="cuda:121:32",
        b12x_revision="75ffee6375b0577ce2c8d6931ffacefda3ecbdd6",
        ptxas_path=assembler.path,
        ptxas_version=subprocess.check_output([assembler.path, "--version"], text=True),
        ptxas_sha256=hashlib.sha256(Path(assembler.path).read_bytes()).hexdigest(),
        num_stages=1,
        enable_fp_fusion=False,
        source_hashes=source_hashes,
        kernels=records,
    )
    (args.out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    entries = [
        '{"%s", "%s", "%s", %d, %d, %d, %d, %d, %s}'
        % (
            r["name"],
            r["file"],
            r["symbol"],
            n,
            k,
            1,
            r["warps"],
            r["shared"],
            str(r["dtype"] == "fp32").lower(),
        )
        for r in records
    ]
    (args.out / "kernels.h").write_text(
        "struct KernelSpec { const char *name, *file, *symbol; "
        "int n, k, rows, warps, shared; bool fp32; };\n"
        "static const KernelSpec kernels[] = {\n" + ",\n".join(entries) + "\n};\n"
    )
    print(f"Exported {len(records)} SM121 cubins to {args.out}")


if __name__ == "__main__":
    main()
