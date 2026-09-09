#!/usr/bin/env bash
# Adapted from run_cuda_test.sh. Uses the same disposable checkout: run serially
# with the normal test and core benchmark scripts.
set -euo pipefail
cd "$(git rev-parse --show-toplevel)"

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
rg --files --no-ignore .prototypes/gemv_aot \
    -g '*.py' -g '*.cu' -g 'build.ninja' -g 'README.md' -g 'run.sh' \
  | rsync -az -e 'ssh -F /dev/null' --files-from=- --relative ./ sp10@sp10:qs3/

ssh -F /dev/null sp10@sp10 -T bash -ls <<'SCRIPT'
set -euo pipefail
export PATH="$HOME/.local/bin:$PATH"
cd qs3
uv sync --locked --project build_tools
ninja -f .prototypes/gemv_aot/build.ninja
{
    date -u
    nvcc --version
    nvidia-smi --query-gpu=name,driver_version,utilization.gpu --format=csv
    sha256sum qscb.cu .prototypes/gemv_aot/bench.cu build_tools/uv.lock
} > .prototypes/gemv_aot/out/host.txt
if ! .prototypes/gemv_aot/out/probe .prototypes/gemv_aot/out \
    > .prototypes/gemv_aot/out/results.csv \
    2> .prototypes/gemv_aot/out/validation.log; then
    cat .prototypes/gemv_aot/out/validation.log
    exit 1
fi
cat .prototypes/gemv_aot/out/validation.log
cat .prototypes/gemv_aot/out/results.csv
SCRIPT

rsync -az -e 'ssh -F /dev/null' \
  sp10@sp10:qs3/.prototypes/gemv_aot/out/ .prototypes/gemv_aot/sp10/
