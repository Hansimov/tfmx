#!/usr/bin/env bash
set -euo pipefail

# Accept port/model/instance from EmbedServerByTEI while keeping sane defaults
PORT=${PORT:-28888}
MODEL_NAME=${MODEL_NAME:-"Alibaba-NLP/gte-multilingual-base"}
INSTANCE_ID=${INSTANCE_ID:-"Alibaba-NLP--gte-multilingual-base"}

while [[ $# -gt 0 ]]; do
    case "$1" in
        -p)
            PORT="$2"
            shift 2
            ;;
        -m)
            MODEL_NAME="$2"
            shift 2
            ;;
        -id)
            INSTANCE_ID="$2"
            shift 2
            ;;
        *)
            echo "Usage: $0 [-p PORT] [-m MODEL_NAME] [-id INSTANCE_ID]" >&2
            exit 1
            ;;
    esac
done

# Basic directories reused across runs
CONFIG_SENTFM_JSON=${CONFIG_SENTFM_JSON:-"config_sentence_transformers.json"}
TFMX_DIR=${TFMX_DIR:-"$HOME/repos/tfmx"}
CACHE_HF=${CACHE_HF:-".cache/huggingface"}
CACHE_HF_HUB=${CACHE_HF_HUB:-"$CACHE_HF/hub"}
TEI_IMAGE=${TEI_IMAGE:-"ghcr.io/huggingface/text-embeddings-inference:1.8"}

MODEL_NAME_DASH="$(printf '%s' "$MODEL_NAME" | sed 's,/,--,g')"

MODEL_SNAPSHOT_DIR=$(find "$HOME/$CACHE_HF_HUB" -type d -path "*/models--$MODEL_NAME_DASH/snapshots/*" -print -quit || true)
if [[ -n "${MODEL_SNAPSHOT_DIR:-}" ]]; then
    # Ensure TEI uses the local config to avoid repeated downloads
    cp -v "$TFMX_DIR/src/tfmx/$CONFIG_SENTFM_JSON" "$MODEL_SNAPSHOT_DIR/$CONFIG_SENTFM_JSON"
fi

mkdir -p "$TFMX_DIR/data/docker_data"

if docker ps -a --format '{{.Names}}' | grep -Fxq "$INSTANCE_ID"; then
    docker start "$INSTANCE_ID"
    echo "[tfmx] Container '$INSTANCE_ID' started on port $PORT"
    exit 0
fi

docker run --gpus all -d --name "$INSTANCE_ID" -p "$PORT:80" \
    -v "$HOME/$CACHE_HF":"/root/$CACHE_HF" \
    -v "$TFMX_DIR/data/docker_data":/data \
    -e HF_HOME="/root/$CACHE_HF" \
    -e HF_HUB_CACHE="/root/$CACHE_HF_HUB" \
    -e HUGGINGFACE_HUB_CACHE="/root/$CACHE_HF_HUB" \
    --pull always "$TEI_IMAGE" \
    --huggingface-hub-cache "/root/$CACHE_HF_HUB" \
    --model-id "$MODEL_NAME" --dtype float16

echo "[tfmx] Container '$INSTANCE_ID' is running on port $PORT"


# Kill all containers from TEI_IMAGE
# docker ps -q --filter "ancestor=$TEI_IMAGE" | xargs -r docker stop