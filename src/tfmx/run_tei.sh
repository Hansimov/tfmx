# # run this to avoid issue of stuck at downloading `config_sentence_transformers.json`
export OWNER_MODEL="Alibaba-NLP/gte-multilingual-base"
# replace '/' with '--'
export OWNER_MODEL_DASH="$(printf '%s' "$OWNER_MODEL" | sed 's,/,--,g')"
export CONFIG_SENTFM_JSON="config_sentence_transformers.json"
export TFMX_DIR="$HOME/repos/tfmx"
export CACHE_HF=".cache/huggingface"
export CACHE_HF_HUB="$CACHE_HF/hub"
export TEI_IMAGE="ghcr.io/huggingface/text-embeddings-inference:1.8"

export MODEL_SNAPSHOT_DIR=$(find "$HOME/$CACHE_HF_HUB" -type d -path "*/models--$OWNER_MODEL_DASH/snapshots/*" -print -quit)
cp -v "$TFMX_DIR/src/tfmx/$CONFIG_SENTFM_JSON" "$MODEL_SNAPSHOT_DIR/$CONFIG_SENTFM_JSON"

docker run --gpus all -p 28888:80 \
    -v "$HOME/$CACHE_HF":"/root/$CACHE_HF" \
    -v "$TFMX_DIR/data/docker_data":/data \
    -e HF_HOME="/root/$CACHE_HF" \
    -e HF_HUB_CACHE="/root/$CACHE_HF_HUB" \
    -e HUGGINGFACE_HUB_CACHE="/root/$CACHE_HF_HUB" \
    --pull always $TEI_IMAGE \
    --huggingface-hub-cache "/root/$CACHE_HF_HUB" \
    --model-id "$OWNER_MODEL" --dtype float16

# docker run --gpus all -p 28888:80 \
#     -v "$TFMX_DIR/data/docker_data":/data \
#     -e HF_ENDPOINT=https://hf-mirror.com \
#     --pull always $TEI_IMAGE \
#     --model-id $OWNER_MODEL --dtype float16

# docker exec -it <container_id> env | grep -i HUGGINGFACE
# docker ps -q --filter "ancestor=$TEI_IMAGE" | xargs -r docker stop