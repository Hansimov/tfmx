# QVL Module - Usage Guide

## CLI Reference

### qvl_compose

Manage Docker Compose deployments for vLLM Qwen3-VL services.

```bash
# Start services on all available GPUs
qvl_compose up

# Start with specific model
qvl_compose up -m "Qwen/Qwen3-VL-8B-Instruct"

# Start with GGUF quantization
qvl_compose up -m "Qwen/Qwen3-VL-8B-Instruct" -q gguf

# Start on specific GPUs
qvl_compose up -g "0,1"

# Custom port base (default: 29880)
qvl_compose up -p 29890

# Custom project name
qvl_compose up -j my-qvl

# Use HTTP proxy for model downloads
qvl_compose up --proxy http://127.0.0.1:11111

# Manual device mount mode (for systems without nvidia-runtime)
qvl_compose up --mount-mode manual

# Other compose operations
qvl_compose ps          # Container status
qvl_compose logs        # View logs
qvl_compose logs -f     # Follow logs
qvl_compose stop        # Stop containers (keep them)
qvl_compose start       # Start stopped containers
qvl_compose restart     # Restart containers
qvl_compose down        # Stop and remove containers
qvl_compose generate    # Generate compose YAML without starting
qvl_compose health      # Check GPU health
```

### qvl_machine

Start the load-balanced proxy server that distributes requests across vLLM instances.

```bash
# Start with auto-discovery of vLLM containers
qvl_machine run

# Start on specific port (default: 29800)
qvl_machine run -p 29800

# Filter containers by name pattern
qvl_machine run -n "qvl--qwen"

# Manual endpoint specification (skip auto-discovery)
qvl_machine run -e "http://localhost:29880,http://localhost:29881"

# Enable performance tracking
qvl_machine run --perf-track

# Discover instances without starting
qvl_machine discover

# Health check
qvl_machine health
```

### qvl_client

Single-machine client for direct interaction with a vLLM service.

```bash
# Health check
qvl_client health

# List available models
qvl_client models

# Simple text chat
qvl_client chat "Hello, how are you?"

# Chat with image
qvl_client chat -i photo.jpg "Describe this image"

# Connect to specific endpoint
qvl_client -e "http://localhost:29800" health
qvl_client -e "http://localhost:29880" chat "Hello"

# Chat with parameters
qvl_client chat --max-tokens 256 --temperature 0.7 "Tell me a story"
```

### qvl_clients

Multi-machine client for distributing requests across multiple machines.

```bash
# Health check across machines
qvl_clients health --endpoints http://host1:29800 http://host2:29800

# Get model info
qvl_clients info --endpoints http://host1:29800

# Interactive chat
qvl_clients chat --endpoints http://host1:29800 \
  --max-tokens 512 --temperature 0.7

# Generate text (with optional images)
qvl_clients generate --endpoints http://host1:29800 \
  --prompt "Describe this image" --images photo.jpg

# Quick inline benchmark
qvl_clients bench --endpoints http://host1:29800 --n 50 --max-tokens 64
```

### qvl_clients_stats

Same as `qvl_clients` but with verbose per-request logging—per-machine throughput, session summaries, and token counts.

```bash
qvl_clients_stats health --endpoints http://host1:29800
qvl_clients_stats generate --endpoints http://host1:29800 --prompt "Hello" -v
```

### qvl_benchmark

Dedicated benchmark runner with synthetic image+prompt generation.

```bash
# Basic benchmark (100 requests)
qvl_benchmark run -E "http://m1:29800,http://m2:29800" -n 100

# Text-only benchmark (no images)
qvl_benchmark run -E "http://m1:29800" -n 200 --text-only

# Custom max tokens + verbose + save results
qvl_benchmark run -E "http://m1:29800" -n 100 --max-tokens 256 -v -o results.json

# Health check
qvl_benchmark health -E "http://m1:29800"

# Generate sample prompts only (inspect what benchmark sends)
qvl_benchmark generate -n 20 --show
```

---

## Python API

### Single-Client Usage

```python
from tfmx.qvls import QVLClient, AsyncQVLClient, build_vision_messages

# Synchronous client
client = QVLClient(endpoint="http://localhost:29800")

# Health check
health = client.health()
print(health.status, health.healthy, health.total)

# List models
models = client.models()
print(models.models)

# Text-only chat
messages = [{"role": "user", "content": "What is 2+2?"}]
response = client.chat(messages, max_tokens=128)
print(response.text)
print(response.usage.completion_tokens)

# Vision chat (with local image)
messages = build_vision_messages(
    prompt="Describe this image in detail.",
    images=["photo.jpg", "https://example.com/image.png"],
    system_prompt="You are a helpful assistant."
)
response = client.chat(messages, max_tokens=512)
print(response.text)

# Convenience generate method
text = client.generate(
    prompt="What do you see?",
    images=["chart.png"],
    max_tokens=256,
    temperature=0.3,
)
print(text)
```

### Async Client

```python
import asyncio
from tfmx.qvls import AsyncQVLClient

async def main():
    client = AsyncQVLClient(endpoint="http://localhost:29800")
    try:
        health = await client.health()
        response = await client.chat(
            [{"role": "user", "content": "Hello!"}],
            max_tokens=64,
        )
        print(response.text)
    finally:
        await client.close()

asyncio.run(main())
```

### Multi-Machine Usage

```python
from tfmx.qvls import QVLClients

endpoints = ["http://host1:29800", "http://host2:29800"]
clients = QVLClients(endpoints=endpoints)

# Health across all machines
health = clients.health()
print(f"{health.healthy_machines}/{health.total_machines} machines healthy")

# Single request (routed to best machine)
response = clients.chat(
    messages=[{"role": "user", "content": "Hello"}],
    max_tokens=128,
)
print(response.text)

# Batch requests (distributed across machines)
requests = [
    {"messages": [{"role": "user", "content": f"Count to {i}"}]}
    for i in range(20)
]
responses = clients.chat_batch(requests, max_tokens=64)
for r in responses:
    print(r.text)

clients.close()
```

### Docker Compose Management

```python
from tfmx.qvls import QVLComposer

composer = QVLComposer()

# Bring up services
composer.up(
    model_name="Qwen/Qwen3-VL-8B-Instruct",
    quantization="gguf",
    gpu_indices="0,1",
)

# Check status
composer.ps()

# View logs
composer.logs(follow=True, tail=100)

# Tear down
composer.down()
```

### Benchmarking

```python
from tfmx.qvls import QVLBenchmark, QVLBenchImageGenerator

# Generate synthetic test data
gen = QVLBenchImageGenerator(seed=42)
items = gen.generate(count=50, img_size=(224, 224))
# Each item: {"prompt": "...", "image": "data:image/png;base64,..."}

# Text-only prompts
text_items = gen.generate_text_only(count=50)

# Mixed (some with images, some without)
mixed = gen.generate_mixed(count=100, image_ratio=0.5)

# Run benchmark
bench = QVLBenchmark(
    endpoints=["http://localhost:29800"],
    max_tokens=128,
    temperature=0.1,
    verbose=True,
)
metrics = bench.run(items)
print(f"Throughput: {metrics.requests_per_second:.2f} req/s")
print(f"Gen tokens/s: {metrics.gen_tokens_per_second:.2f}")
print(f"P50 latency: {metrics.latency_p50:.3f}s")
print(f"P99 latency: {metrics.latency_p99:.3f}s")
```

### Machine Server (Programmatic)

```python
from tfmx.qvls import QVLMachineServer, VLLMInstance

# Define instances manually
instances = [
    VLLMInstance(container_name="qvl-0", host="localhost", port=29880, gpu_id=0),
    VLLMInstance(container_name="qvl-1", host="localhost", port=29881, gpu_id=1),
]

server = QVLMachineServer(instances=instances, port=29800)
server.run()  # Starts uvicorn
```

---

## Workflow: End-to-End Deployment

```bash
# 1. Deploy vLLM containers (one per GPU)
qvl_compose up -m "Qwen/Qwen3-VL-8B-Instruct" -q gguf

# 2. Wait for containers to be healthy (~60-120s)
qvl_compose logs -f   # Watch for "Uvicorn running on ..."

# 3. Start load-balanced proxy
qvl_machine run

# 4. Verify everything works
qvl_client health
qvl_client chat "Hello, what can you do?"

# 5. Run benchmark
qvl_benchmark run -E "http://localhost:29800" -n 100

# 6. Use in production via multi-machine client
qvl_clients generate --endpoints http://host1:29800 http://host2:29800 \
  --prompt "Describe this image" --images photo.jpg
```

## Multi-Machine Deployment

On each machine:

```bash
# Machine 1 (host1)
qvl_compose up -m "Qwen/Qwen3-VL-8B-Instruct" -q gguf
qvl_machine run

# Machine 2 (host2)
qvl_compose up -m "Qwen/Qwen3-VL-8B-Instruct" -q gguf
qvl_machine run
```

From any machine:

```bash
qvl_clients health --endpoints http://host1:29800 http://host2:29800
qvl_benchmark run -E "http://host1:29800,http://host2:29800" -n 200
```
