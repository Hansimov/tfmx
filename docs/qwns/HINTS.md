# QWN Hints & Troubleshooting

## Model Labels

- The Docker layer loads `cyankiwi/Qwen3.5-4B-AWQ-4bit`.
- The served model label is normalized to `4b:4bit` for routing stability.
- `qwn client models` and `GET /v1/models` will show the served labels, not necessarily the full Hugging Face repo.

## Runtime Image

- `qwn` auto-builds a local image `tfmx-vllm-openai:qwen3.5-v0.19.0` from `vllm/vllm-openai:v0.19.0`.
- This is required because only `transformers` main currently recognizes the `qwen3_5` architecture end-to-end for this model repo.
- The runtime image also installs a `transformers`-compatible `huggingface-hub>=1.5.0,<2.0`, because the upstream `v0.19.0` image ships an older hub package than `transformers` main now expects.
- The build now streams `docker build --progress=plain` output directly, so long downloads no longer look like a hang.
- Hugging Face downloads use `hf-mirror.com` and `pip` uses the USTC mirror by default.
- If your environment exposes a proxy through `QWN_PROXY`, `TFMX_QWN_PROXY`, or standard system proxy variables, `qwn` reuses it automatically for Docker build steps that need GitHub access.
- If you want to rebuild it manually, run:

```bash
docker rmi tfmx-vllm-openai:qwen3.5-v0.19.0
qwn compose up --gpu-configs "0:4b:4bit"
```

## Recommended Workflow

1. `qwn compose up --gpu-configs "0:4b:4bit"`
2. Wait for vLLM to finish loading
3. `qwn machine run -b`
4. `qwn client health`
5. `qwn benchmark run -E http://localhost:27800 -n 100`

## Troubleshooting

### No GPUs detected

- Verify `nvidia-smi`
- Verify Docker GPU runtime separately with an NVIDIA CUDA image
- If runtime mode fails, use `qwn compose up --mount-mode manual`

### Machine proxy does not start

- Check `qwn machine status`
- View daemon logs with `qwn machine logs`
- Remove stale files only if the process is confirmed dead:
  - `~/.cache/tfmx/qwn_machine.pid`
  - `~/.cache/tfmx/qwn_machine.log`

### Health endpoint shows zero healthy instances

- Confirm the containers are still loading models
- Check `qwn compose logs -f`
- Directly query a backend container:

```bash
curl http://localhost:27880/health
curl http://localhost:27880/v1/models
```

### Container discovery misses your deployment

Auto-discovery prefers container names starting with `qwn`. If you used a custom compose project name that drops that prefix, run the proxy with explicit endpoints or a matching regex:

```bash
qwn machine run -n my-custom-project
qwn machine run -e http://localhost:27880,http://localhost:27881
```