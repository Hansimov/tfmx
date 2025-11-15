#!/usr/bin/env bash
set -euo pipefail

# envs
PORT=${PORT:-28888}
MODEL_NAME=${MODEL_NAME:-"Alibaba-NLP/gte-multilingual-base"}
INSTANCE_ID=${INSTANCE_ID:-"Alibaba-NLP--gte-multilingual-base"}
HF_TOKEN=${HF_TOKEN:-""}

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
        -u)
            HF_TOKEN="$2"
            shift 2
            ;;
        *)
            echo "Usage: $0 [-p PORT] [-m MODEL_NAME] [-id INSTANCE_ID] [-u HF_TOKEN]" >&2
            exit 1
            ;;
    esac
done

# paths
TFMX_DIR=${TFMX_DIR:-"$HOME/repos/tfmx"}
TFMX_SRC="$TFMX_DIR/src/tfmx"
CACHE_HF=${CACHE_HF:-".cache/huggingface"}
CACHE_HF_HUB=${CACHE_HF_HUB:-"$CACHE_HF/hub"}
HF_ENDPOINT="https://hf-mirror.com"

# patch config file to avoid redundant download or fix corrupted files
MODEL_NAME_DASH="$(printf '%s' "$MODEL_NAME" | sed 's,/,--,g')"
MODEL_SNAPSHOT_DIR=$(find "$HOME/$CACHE_HF_HUB" -type d -path "*/models--$MODEL_NAME_DASH/snapshots/*" -print -quit || true)
if [[ -n "${MODEL_SNAPSHOT_DIR:-}" ]]; then
    CONFIG_SENTFM_JSON="config_sentence_transformers.json"
    TARGET_SENTFM_CONFIG="$MODEL_SNAPSHOT_DIR/$CONFIG_SENTFM_JSON"
    SOURCE_SENTFM_CONFIG="$TFMX_SRC/$CONFIG_SENTFM_JSON"
    if [[ -f "$TARGET_SENTFM_CONFIG" ]]; then
        echo "[tfmx] Skip copy existed: '$TARGET_SENTFM_CONFIG'"
    else
        cp -v "$SOURCE_SENTFM_CONFIG" "$TARGET_SENTFM_CONFIG"
    fi

    CONFIG_JSON="config.json"
    TARGET_CONFIG="$MODEL_SNAPSHOT_DIR/$CONFIG_JSON"
    SOURCE_CONFIG="$TFMX_SRC/config_qwen3_embedding_06b.json"
    if [[ -f "$TARGET_CONFIG" ]]; then
        LAST_CHAR=$(tail -c 2 "$TARGET_CONFIG" | tr -d '[:space:]')
        if [[ "$LAST_CHAR" != "}" ]]; then
            echo "[tfmx] Corrupted: '$TARGET_CONFIG'"
            echo "[tfmx] Remove and patch ..."
            sudo rm -f "$TARGET_CONFIG"
            sudo cp -v "$SOURCE_CONFIG" "$TARGET_CONFIG"
        else
            echo "[tfmx] Skip copy existed: '$TARGET_CONFIG'"
        fi
    else
        sudo cp -v "$SOURCE_CONFIG" "$TARGET_CONFIG"
    fi
fi

# run docker
CONTAINER_EXISTS=$(docker ps -a --filter "name=^/${INSTANCE_ID}$" --format '{{.Names}}')
if [[ -n "$CONTAINER_EXISTS" ]]; then
    docker start "$INSTANCE_ID"
    echo "[tfmx] Container '$INSTANCE_ID' (:$PORT) is existed"
else
    # https://github.com/huggingface/text-embeddings-inference?tab=readme-ov-file#docker
    ROOT_CACHE_HF_HUB="/root/$CACHE_HF_HUB"
    ROOT_CACHE_HF="/root/$CACHE_HF"
    TFMX_DOCKER_DATA_DIR="$TFMX_DIR/data/docker_data"
    TEI_IMAGE=${TEI_IMAGE:-"ghcr.io/huggingface/text-embeddings-inference:89-1.8"}
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
        --hf-token "$HF_TOKEN" \
        --dtype float16 --max-batch-tokens 32768 --max-client-batch-size 100
    echo "[tfmx] Container '$INSTANCE_ID' (:$PORT) is started"
fi


# Kill all containers of TEI_IMAGE
# docker ps -q --filter "ancestor=$TEI_IMAGE" | xargs -r docker stop

# Clear cache in host if download corrupted files
# sudo rm -rf ~/.cache/huggingface/hub/models--Qwen--Qwen3-Embedding-0.6B