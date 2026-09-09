"""Probe-only AOT exporter: cubins plus explicit native launch metadata."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess

import triton
from triton.backends.compiler import GPUTarget
from triton.compiler import ASTSource
from triton.backends.nvidia.compiler import get_ptxas

from kernel import gemv


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    # name, output rows, reduction width, output scalar type. These are 35B
    # projections, not a model-shape dispatch API. N=8192 includes packed Q/gate.
    shapes = [("qkv", 8192, 2048, "bf16"),
              ("z", 4096, 2048, "bf16"),
              ("out", 2048, 4096, "bf16"),
              ("lm_head", 248320, 2048, "fp32")]
    records = []
    for name, n, k, dtype in shapes:
        for rows, warps in [(1, 4), (4, 4), (8, 8)]:
            constants = {"N": n, "K": k, "ROWS": rows}
            src = ASTSource(fn=gemv,
                            signature={"X": "*bf16", "W": "*bf16", "Y": "*" + dtype},
                            constexprs=constants)
            compiled = triton.compile(src, target=GPUTarget("cuda", 121, 32),
                                      options={"num_warps": warps, "num_stages": 1,
                                               "enable_fp_fusion": False})
            meta = compiled.metadata
            for field in ["global_scratch_size", "profile_scratch_size"]:
                if getattr(meta, field, 0):
                    raise RuntimeError(f"{field} requires a different launch contract")
            if getattr(meta, "num_ctas", 1) != 1:
                raise RuntimeError("probe only supports single-CTA launches")
            stem = f"{name}_r{rows}_w{warps}"
            cubin = compiled.asm["cubin"]
            (args.out / (stem + ".cubin")).write_bytes(cubin)
            (args.out / (stem + ".ptx")).write_text(compiled.asm["ptx"])
            records.append(dict(name=name, n=n, k=k, dtype=dtype, rows=rows,
                                warps=warps, file=stem + ".cubin", symbol=meta.name,
                                shared=meta.shared, sha256=hashlib.sha256(cubin).hexdigest()))
    source_hashes = {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                     for p in [Path(__file__), Path(__file__).with_name("kernel.py")]}
    assembler = get_ptxas(121)
    manifest = dict(triton_version=triton.__version__, target="cuda:121:32",
                    ptxas_path=assembler.path,
                    ptxas_version=subprocess.check_output([assembler.path, "--version"], text=True),
                    ptxas_sha256=hashlib.sha256(Path(assembler.path).read_bytes()).hexdigest(),
                    num_stages=1, enable_fp_fusion=False,
                    source_hashes=source_hashes, kernels=records)
    (args.out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    entries = []
    for r in records:
        entries.append('{"%s", "%s", "%s", %d, %d, %d, %d, %d, %s}' %
                       (r["name"], r["file"], r["symbol"], r["n"], r["k"], r["rows"],
                        r["warps"], r["shared"], str(r["dtype"] == "fp32").lower()))
    (args.out / "kernels.h").write_text(
        "// Generated probe launch metadata; see manifest.json.\n"
        "struct KernelSpec { const char *name, *file, *symbol; "
        "int n, k, rows, warps, shared; bool fp32; };\n"
        "static const KernelSpec kernels[] = {\n" + ",\n".join(entries) + "\n};\n")
    print(f"Exported {len(records)} SM121 cubins to {args.out}")


if __name__ == "__main__":
    main()
