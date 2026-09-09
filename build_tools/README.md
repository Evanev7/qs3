# qs3 Python build tools

This uv workspace uses Python 3.14. Run its tools from the repository
root with:

```sh
just uv-sync
build_tools/.venv/bin/qwen36-vectors list-groups
```

`qwen36-vectors` is a standard-library-only workspace member. Its Python modules
live directly beside its `pyproject.toml`; setuptools maps that directory to
the installed `qwen36_vectors` package. Oracle hashes ship as package data.
Ninja's `qwen36_vectors` variable points to
`../build_tools/.venv/bin/qwen36-vectors` from its `build/` working directory.
Just's matching variable uses `build_tools/.venv/bin/qwen36-vectors` from the
repository root. Vector output paths are
relative to the caller's working directory, so Ninja writes inside `build/`.

`just uv-sync` runs `uv sync --locked --project build_tools`.
`just generate-vectors` and `just refresh-vector-oracles` depend on that recipe.
Ninja only runs the installed generator; direct Ninja callers prepare the
environment first.

Triton 3.8.0 is pinned as a build-time compiler dependency. The official wheel
supplies the compiler; the vendored source checkout is not built by this project.
The Python upper bound follows this Triton release's supported interpreter range.

`uv.lock` is shared by workspace members. The reserved `triton/` directory is
not a workspace member; the compiler dependency is the upstream PyPI package.
