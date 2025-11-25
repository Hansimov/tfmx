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
if [[ -d "$HOME/repos/tfmx" ]]; then
    TFMX_DIR=${TFMX_DIR:-"$HOME/repos/tfmx"}
    TFMX_SRC="$TFMX_DIR/src/tfmx"
else
    TFMX_PIP=$(pip show tfmx 2>/dev/null | grep "Location:" | awk '{print $2}')
    if [[ -n "$TFMX_PIP" ]]; then
        TFMX_DIR=${TFMX_DIR:-"$TFMX_PIP/tfmx"}
        TFMX_SRC="$TFMX_DIR"
    else
        TFMX_DIR=${TFMX_DIR:-"$HOME/repos/tfmx"}
        TFMX_SRC="$TFMX_DIR/src/tfmx"
    fi
fi
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
    
    # determine image tag by GPU compute capability
    GPU_COMPUTE_CAP=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader | head -n1)
    case "$GPU_COMPUTE_CAP" in
        8.0) ARCH_TAG="1.8" ;;     # Ampere 80 (A100, A30)
        8.6) ARCH_TAG="86-1.8" ;;  # Ampere 86 (A10, A40, RTX 3080)
        8.9) ARCH_TAG="89-1.8" ;;  # Ada Lovelace (RTX 4090)
        *) ARCH_TAG="86-1.8" ;;    # Fallback to Ampere 86
    esac

    TEI_IMAGE_BY_ARCH="ghcr.io/huggingface/text-embeddings-inference:${ARCH_TAG}"
    TEI_IMAGE=${TEI_IMAGE:-"$TEI_IMAGE_BY_ARCH"}
    echo "[tfmx] Detected GPU compute capability: $GPU_COMPUTE_CAP, using image: $TEI_IMAGE"

    # pull image from mirror if not exists locally
    if ! docker image inspect "$TEI_IMAGE" &>/dev/null; then
        MIRROR_IMAGE="m.daocloud.io/$TEI_IMAGE"
        echo "[tfmx] Pulling image from mirror: $MIRROR_IMAGE"
        docker pull "$MIRROR_IMAGE"
        docker tag "$MIRROR_IMAGE" "$TEI_IMAGE"
        echo "[tfmx] Image tagged as: $TEI_IMAGE"
    fi

    mkdir -p "$TFMX_DOCKER_DATA_DIR"
    docker run --gpus all -d --name "$INSTANCE_ID" -p "$PORT:80" \
        -v "$HOME/$CACHE_HF":"$ROOT_CACHE_HF" \
        -v "$TFMX_DOCKER_DATA_DIR":/data \
        -e HF_ENDPOINT="$HF_ENDPOINT" \
        -e HF_HOME="$ROOT_CACHE_HF" \
        -e HF_HUB_CACHE="$ROOT_CACHE_HF_HUB" \
        -e HUGGINGFACE_HUB_CACHE="$ROOT_CACHE_HF_HUB" \
        "$TEI_IMAGE" \
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

# remove container
# docker rm -f "tei--Qwen--Qwen3-Embedding-0.6B"