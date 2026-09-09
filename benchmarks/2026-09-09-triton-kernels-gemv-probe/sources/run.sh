#!/usr/bin/env bash
# Adapted from run_cuda_test.sh. Uses the same disposable checkout: run serially
# with the normal test and core benchmark scripts.
set -euo pipefail
cd "$(git rev-parse --show-toplevel)"

mkdir -p .prototypes/triton_kernels_gemv/upstream/triton_kernels
git -C 3pty/triton archive v3.8.0:python/triton_kernels/triton_kernels \
  | tar -x -C .prototypes/triton_kernels_gemv/upstream/triton_kernels
git -C 3pty/triton show v3.8.0:LICENSE > .prototypes/triton_kernels_gemv/LICENSE.triton

ssh -F /dev/null sp10@sp10 -T <<'SCRIPT'
set -euo pipefail
cd qs3
git reset --hard
git clean -fd
git pull --rebase
SCRIPT

rg --files --hidden -g '!3pty/**' -g '!.git/**' \
  | rsync -az -e 'ssh -F /dev/null' --files-from=- --relative ./ sp10@sp10:qs3/
# Prototypes are intentionally ignored, so transfer this probe's source files
# explicitly. Cubins, compiler cache entries and logs are built on the host.
rg --files --no-ignore .prototypes/triton_kernels_gemv \
    -g '*.py' -g '*.cu' -g 'build.ninja' -g 'README.md' -g 'run.sh' -g 'LICENSE.b12x' -g 'LICENSE.triton' \
  | rsync -az -e 'ssh -F /dev/null' --files-from=- --relative ./ sp10@sp10:qs3/

ssh -F /dev/null sp10@sp10 -T bash -ls <<'SCRIPT'
set -euo pipefail
export PATH="$HOME/.local/bin:$PATH"
cd qs3
ninja -f .prototypes/triton_kernels_gemv/build.ninja
{
    date -u
    nvcc --version
    nvidia-smi --query-gpu=name,driver_version,utilization.gpu --format=csv
    sha256sum qscb.cu .prototypes/triton_kernels_gemv/bench.cu build_tools/uv.lock
} > .prototypes/triton_kernels_gemv/out/host.txt
if ! .prototypes/triton_kernels_gemv/out/probe .prototypes/triton_kernels_gemv/out \
    > .prototypes/triton_kernels_gemv/out/results.csv \
    2> .prototypes/triton_kernels_gemv/out/validation.log; then
    cat .prototypes/triton_kernels_gemv/out/validation.log
    exit 1
fi
cat .prototypes/triton_kernels_gemv/out/validation.log
cat .prototypes/triton_kernels_gemv/out/results.csv
SCRIPT

rsync -az -e 'ssh -F /dev/null' \
  sp10@sp10:qs3/.prototypes/triton_kernels_gemv/out/ .prototypes/triton_kernels_gemv/sp10/
