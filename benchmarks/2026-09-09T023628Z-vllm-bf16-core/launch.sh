#!/usr/bin/env bash
set -euo pipefail
cd "$(git rev-parse --show-toplevel)"
vllm_run="$(date -u +%Y-%m-%dT%H%M%SZ)-vllm-bf16-core"
ssh -F /dev/null sp10@sp10 'mkdir -p ~/qs3-vllm-bench ~/qs3-vllm-cache'
rsync -az -e 'ssh -F /dev/null' .prototypes/vllm_core_benchmark.py sp10@sp10:qs3-vllm-bench/
ssh -F /dev/null sp10@sp10 -T bash -s -- "$vllm_run" <<'REMOTE'
set -euo pipefail
vllm_image=vllm-node@sha256:d966c1831d5da55c0cc52c6bd40f7d02cfc3d83404c3bd599139b055232d3970
mkdir -p "$HOME/qs3-vllm-bench/$1"
docker image inspect "$vllm_image" --format '{{.Id}} {{.Architecture}}' > "$HOME/qs3-vllm-bench/$1/image.txt"
nvidia-smi --query-gpu=name,driver_version --format=csv > "$HOME/qs3-vllm-bench/$1/gpu.csv"
docker run --rm --gpus all --ipc=host --network=none \
    -e VLLM_ENABLE_V1_MULTIPROCESSING=0 -e VLLM_NO_USAGE_STATS=1 \
    -e HF_HUB_OFFLINE=1 -e TRANSFORMERS_OFFLINE=1 \
    -v "$HOME/.cache/huggingface/hub/models--Qwen--Qwen3.6-35B-A3B:/model-repo:ro" \
    -v "$HOME/qs3-vllm-bench:/bench" \
    -v "$HOME/qs3-vllm-cache:/root/.cache" \
    --entrypoint python3 "$vllm_image" /bench/vllm_core_benchmark.py \
    --model /model-repo/snapshots/995ad96eacd98c81ed38be0c5b274b04031597b0 \
    --output "/bench/$1/result.json" > "$HOME/qs3-vllm-bench/$1/run.log" 2>&1
REMOTE
mkdir -p ".prototypes/vllm-runs/$vllm_run"
rsync -az -e 'ssh -F /dev/null' "sp10@sp10:qs3-vllm-bench/$vllm_run/" ".prototypes/vllm-runs/$vllm_run/"
printf 'vLLM run saved to .prototypes/vllm-runs/%s\n' "$vllm_run"
