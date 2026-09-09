#!/usr/bin/env bash
set -euo pipefail
cd "$(git rev-parse --show-toplevel)"
score_run=${QS3_SCORE_RUN:?set QS3_SCORE_RUN to the shared run name}
[[ $score_run =~ ^[A-Za-z0-9_.-]+$ ]] || exit 2
ssh -F /dev/null sp10@sp10 -T bash -s -- "$score_run" <<'REMOTE'
set -euo pipefail
mkdir -p "$HOME/qs3-scores/$1"
REMOTE
rsync -az -e 'ssh -F /dev/null' .prototypes/same_prefix_scores/input.json .prototypes/same_prefix_scores/vllm_scores.py "sp10@sp10:qs3-scores/$score_run/"
ssh -F /dev/null sp10@sp10 -T bash -s -- "$score_run" <<'REMOTE'
set -euo pipefail
score_dir="$HOME/qs3-scores/$1"
score_image=vllm-node@sha256:d966c1831d5da55c0cc52c6bd40f7d02cfc3d83404c3bd599139b055232d3970
docker image inspect "$score_image" --format '{{.Id}} {{.Architecture}}' > "$score_dir/image.txt"
docker run --rm --gpus all --ipc=host --network=none \
    -e VLLM_ENABLE_V1_MULTIPROCESSING=0 -e VLLM_NO_USAGE_STATS=1 \
    -e HF_HUB_OFFLINE=1 -e TRANSFORMERS_OFFLINE=1 \
    -v "$HOME/.cache/huggingface/hub/models--Qwen--Qwen3.6-35B-A3B:/model-repo:ro" \
    -v "$score_dir:/scores" -v "$HOME/qs3-vllm-cache:/root/.cache" \
    --entrypoint python3 "$score_image" /scores/vllm_scores.py \
    --model /model-repo/snapshots/995ad96eacd98c81ed38be0c5b274b04031597b0 \
    --input /scores/input.json --output /scores/vllm > "$score_dir/vllm.log" 2>&1
REMOTE
mkdir -p ".prototypes/same_prefix_scores/runs/$score_run"
rsync -az --exclude '*.f32' -e 'ssh -F /dev/null' "sp10@sp10:qs3-scores/$score_run/" ".prototypes/same_prefix_scores/runs/$score_run/"
