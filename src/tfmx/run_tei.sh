#!/usr/bin/env bash
set -euo pipefail

# envs
PORT=${PORT:-28888}
MODEL_NAME=${MODEL_NAME:-"Alibaba-NLP/gte-multilingual-base"}
INSTANCE_ID=${INSTANCE_ID:-"Alibaba-NLP--gte-multilingual-base"}

# args
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

# paths
TFMX_DIR=${TFMX_DIR:-"$HOME/repos/tfmx"}
CACHE_HF=${CACHE_HF:-".cache/huggingface"}
CACHE_HF_HUB=${CACHE_HF_HUB:-"$CACHE_HF/hub"}
HF_ENDPOINT="https://hf-mirror.com"

# patch config file to avoid redundant download
MODEL_NAME_DASH="$(printf '%s' "$MODEL_NAME" | sed 's,/,--,g')"
MODEL_SNAPSHOT_DIR=$(find "$HOME/$CACHE_HF_HUB" -type d -path "*/models--$MODEL_NAME_DASH/snapshots/*" -print -quit || true)
if [[ -n "${MODEL_SNAPSHOT_DIR:-}" ]]; then
    CONFIG_SENTFM_JSON="config_sentence_transformers.json"
    TARGET_SENTFM_CONFIG="$MODEL_SNAPSHOT_DIR/$CONFIG_SENTFM_JSON"
    SOURCE_SENTFM_CONFIG="$TFMX_DIR/src/tfmx/$CONFIG_SENTFM_JSON"
    if [[ -f "$TARGET_SENTFM_CONFIG" ]]; then
        echo "[tfmx] Skip copy existed: '$TARGET_SENTFM_CONFIG'"
    else
        cp -v "$SOURCE_SENTFM_CONFIG" "$TARGET_SENTFM_CONFIG"
    fi
fi

# run docker
CONTAINER_EXISTS=$(docker ps -a --filter "name=^/${INSTANCE_ID}$" --format '{{.Names}}')
if [[ -n "$CONTAINER_EXISTS" ]]; then
    docker start "$INSTANCE_ID"
    echo "[tfmx] Container '$INSTANCE_ID' (:$PORT) is existed"
else
    ROOT_CACHE_HF_HUB="/root/$CACHE_HF_HUB"
    ROOT_CACHE_HF="/root/$CACHE_HF"
    TFMX_DOCKER_DATA_DIR="$TFMX_DIR/data/docker_data"
    TEI_IMAGE=${TEI_IMAGE:-"ghcr.io/huggingface/text-embeddings-inference:1.8"}
    mkdir -p "$TFMX_DOCKER_DATA_DIR"
    docker run --gpus all -d --name "$INSTANCE_ID" -p "$PORT:80" \
        -v "$HOME/$CACHE_HF":"$ROOT_CACHE_HF" \
        -v "$TFMX_DOCKER_DATA_DIR":/data \
        -e HF_ENDPOINT="$HF_ENDPOINT" \
        -e HF_HOME="$ROOT_CACHE_HF" \
        -e HF_HUB_CACHE="$ROOT_CACHE_HF_HUB" \
        -e HUGGINGFACE_HUB_CACHE="$ROOT_CACHE_HF_HUB" \
        --pull always "$TEI_IMAGE" \
        --huggingface-hub-cache "$ROOT_CACHE_HF_HUB" \
        --model-id "$MODEL_NAME" \
        --dtype float16 --max-batch-tokens 32768
    echo "[tfmx] Container '$INSTANCE_ID' (:$PORT) is started"
fi


# Kill all containers of TEI_IMAGE
# docker ps -q --filter "ancestor=$TEI_IMAGE" | xargs -r docker stop