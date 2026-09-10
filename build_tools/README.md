# qs3 Python build tools

This uv workspace uses Python 3.14. Run its tools from the repository
root with:

```sh
just build_tools/uv-sync
build_tools/.venv/bin/qwen36-vectors list-groups
```

`qwen36-vectors` is a standard-library-only workspace member in `qwen36_vectors/`.
Its Python modules live directly beside its `pyproject.toml`; setuptools maps that directory to
the installed `qwen36_vectors` package. Oracle hashes ship as package data.
Ninja's `qwen36_vectors` variable points to
`../build_tools/.venv/bin/qwen36-vectors` from its `build/` working directory.
The recipes in `build_tools/justfile` run from `build_tools/` and use
`.venv/bin/qwen36-vectors`. Vector output paths are
relative to the caller's working directory, so Ninja writes inside `build/`.

`just build_tools/uv-sync` explicitly runs `uv sync --locked` in this directory.
Prepare the environment before invoking tests or generators; those commands
use the installed tools without installing or updating packages.

The root `just check` and `just fmt` delegate Python work to this justfile.
ty uses `--project .` here to find the workspace's Python 3.14 environment and
installed compiler dependencies. The
default checks exclude vendored code and archived benchmark experiments.

`just build_tools/python-test` also regenerates the vectors and checks their pinned oracle
hashes. `just build_tools/triton-test` evaluates the Nix configuration, compiles the kernels,
and runs the generated Rust launchers on CUDA.

The gitignored remote runner syncs the checkout and executes its arguments:

```sh
./run_cuda_test.sh just build_tools/python-test triton-test
./run_cuda_test.sh just test
./run_cuda_test.sh just model-test 20 --test-threads=1
```

With no arguments it runs `just test`. It does not install tools or packages.
Remote test and benchmark invocations share a disposable checkout; run them
sequentially. Use `just --list` for runtime recipes and
`just --justfile build_tools/justfile --list` for build-tool recipes.

Triton 3.8.0 is pinned as a build-time compiler dependency. The official wheel
supplies the compiler; the vendored source checkout is not built by this project.
The Python upper bound follows this Triton release's supported interpreter range.

`uv.lock` is shared by workspace members. `qstriton/` is the `qstriton` member,
with a `qstriton` CLI and the `qstriton` Python package. It owns the upstream
Triton compiler dependency and ships the kernel sources; see
[its README](qstriton/README.md).
