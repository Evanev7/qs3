# qs3 build tools

Just provides entrypoint commands that always run. Ninja handles rebuilds and
invokes compilers on source files. Nix generates Ninja manifests; Ninja tracks
the Nix inputs and reloads the generated manifests.

From the repository root:

```sh
just build_tools/uv-sync
just ninja
ninja -C build triton/kernels
```

`build_tools/` is one uv project; `pysrc/` contains `qstriton`, `qsutil`, and
`qwen36_vectors`. Raw kernels live in `triton_kernels/`. In `models/config.nix`,
each Triton recipe's attribute name determines output filenames; `source` is
repository-relative and `spec` supplies compiler inputs.

See [qstriton](pysrc/qstriton/README.md) for the compiler interface and
[qwen36_vectors](pysrc/qwen36_vectors/README.md) for vector generation.

The gitignored `remote.sh` prepares the remote CUDA/Python environment and runs
named workflows:

```sh
./remote.sh test
./remote.sh benchmark
./remote.sh prototype my-probe
```

`remote.sh` shares one CUDA/Python setup and then invokes Just. The root `test`
recipe runs the Python, Triton launcher, Rust and native CUDA suites against the
current working tree. The root `benchmark` recipe builds through Nix and emits
release/Nsight JSON. The remote benchmark action requires committed tracked changes,
transfers the exact commit, and saves that JSON under `benchmarks/`.
The benchmark runs two workloads sequentially: 102 prompt tokens / 32 measured
decode steps, then 1024 / 256. Each gets its own release/Nsight JSON, with the same
schema as earlier results. Both use FP32 GDN state and `tile32_blocks96` MoE.
See `./remote.sh --help`.

Every runnable prototype needs a named, no-argument recipe in
`.prototypes/justfile`. `./remote.sh prototype my-probe` invokes its `my-probe`
recipe with `.prototypes/` as the working directory. The runner uploads
`.prototypes/` and retrieves `.prototypes/out/` after a successful run. Uploads
exclude `out/`, `.venv/`, `__pycache__/` and `.git/`. Source deletions are mirrored,
and the recipe owns cleaning/rebuilding its outputs. For example:

```just
my-probe:
    mkdir -p out/my-probe
    nvcc -arch=sm_121 my-probe/bench.cu -o out/my-probe/bench
    out/my-probe/bench > out/my-probe/results.txt
```

The remote interface accepts no shell command or arbitrary recipe arguments.

`./remote.sh prototype vllm-benchmark` runs the same two workloads with the pinned
vLLM container and BF16 model snapshot. Results, generated token IDs, logs and
container/GPU metadata are retrieved under `.prototypes/out/vllm-*/`. vLLM stays
outside the build-tools venv. The harness uses greedy sampling, decode graphs,
four decode warmups and records the resolved GDN state precision. Compare prompt
fingerprints before comparing token sequences. vLLM records one extra output token
because it emits a token at prefill; compare qs3's 36/260 IDs against vLLM's first
36/260 IDs. Prefill timing includes first-token delivery in vLLM.

The original `run_cuda_test.sh` and `run_core_benchmark.sh` remain for comparison.
All runners share the disposable checkout; run them sequentially.
