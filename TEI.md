# TEI Client Usage Guide

High-performance clients for Text Embeddings Inference (TEI) services with multi-machine support and automatic load balancing.

## Installation

```sh
pip install tfmx --upgrade
```

## Command Line Usage

### Health Check

```sh
export TEI_EPS="http://localhost:28800,http://ai122:28800"
tei_clients -E $TEI_EPS health
```

### Embedding

```sh
tei_clients -E $TEI_EPS embed "Hello" "World"
```

### LSH (Locality Sensitive Hashing)

```sh
tei_clients -E $TEI_EPS lsh "Hello, world"
tei_clients -E $TEI_EPS lsh -b 2048 "Hello, world"
```

### Benchmark

```sh
# Run benchmark with 100k samples
tei_benchmark -E $TEI_EPS run -n 100000

# Auto-tune batch size
tei_benchmark -E $TEI_EPS tune

# Verbose output with results saved
tei_benchmark -E $TEI_EPS -v run -o results.json
```

## Python API Usage

### Single Machine Client

```python
from tfmx import TEIClient

# Create client
client = TEIClient("http://localhost:28800")

# Generate embeddings
texts = ["Hello, world", "Another text"]
embeddings = client.embed(texts)

# Generate LSH hashes
lsh_hashes = client.lsh(texts, bitn=2048)

# Check health
health = client.health()
print(f"Status: {health.status}, GPUs: {health.healthy}/{health.total}")

client.close()
```

### Multi-Machine Client (Production)

```python
from tfmx import TEIClients

endpoints = ["http://machine1:28800", "http://machine2:28800"]

# Create client (auto-loads config from tei_clients.config.json)
clients = TEIClients(endpoints)

# Check health
health = clients.health()
print(f"Healthy: {health.healthy_machines}/{health.total_machines} machines")
print(f"GPUs: {health.healthy_instances}/{health.total_instances}")

# Process data with auto batch sizing and pipeline scheduling
texts = ["text1", "text2", ..., "text100000"]
embeddings = clients.embed(texts)
lsh_hashes = clients.lsh(texts, bitn=2048)

clients.close()
```

### Context Manager (Recommended)

```python
from tfmx import TEIClients

endpoints = ["http://machine1:28800", "http://machine2:28800"]

with TEIClients(endpoints) as clients:
    embeddings = clients.embed(large_text_list)
    lsh_hashes = clients.lsh(large_text_list, bitn=2048)
```

### Multi-Machine Client with Stats (Testing)

```python
from tfmx import TEIClientsWithStats

endpoints = ["http://machine1:28800", "http://machine2:28800"]

# Enable verbose logging for progress tracking
clients = TEIClientsWithStats(endpoints, verbose=True)

# Process large dataset with progress logging
texts = ["text1", "text2", ..., "text100000"]
lsh_hashes = clients.lsh(texts, bitn=2048)

# Verbose output example:
# [machine1] Loaded config: batch_size=2000, max_concurrent=6
# [machine2] Loaded config: batch_size=2000, max_concurrent=6
# [  5%] 5000/100000 | machine1:800/s | machine2:1200/s | 2000/s
# [ 10%] 10000/100000 | machine1:850/s | machine2:1350/s | 2200/s
# [Pipeline] Complete: 100000 items, 50 batches, 45.5s, 2198/s

clients.close()
```

### Iterator Support (Memory Efficient)

```python
from tfmx import TEIClients

def text_generator():
    for i in range(1000000):
        yield f"Text number {i}"

with TEIClients(endpoints) as clients:
    results = clients.lsh_iter(
        text_generator(),
        total_hint=1000000,  # Optional: enables progress tracking
        bitn=2048
    )
    print(f"Processed {len(results)} items")
```

## Configuration

### Automatic Config Loading

Both `TEIClients` and `TEIClientsWithStats` automatically load optimal batch sizes and concurrency from `tei_clients.config.json`.

### Example `tei_clients.config.json`

```json
{
    "53eaad030a30": {
        "endpoints": [
            "http://localhost:28800"
        ],
        "machines": {
            "localhost:28800": {
                "optimal_batch_size": 2500,
                "optimal_max_concurrent": 2,
                "throughput": 1300,
                "instances": 2,
                "updated_at": "2026-01-15T22:03:33.344773"
            }
        }
    },
    "a2ddb1eae1bc": {
        "endpoints": [
            "http://ai122:28800"
        ],
        "machines": {
            "ai122:28800": {
                "optimal_batch_size": 2000,
                "optimal_max_concurrent": 8,
                "throughput": 2400,
                "instances": 8,
                "updated_at": "2026-01-15T22:03:33.344773"
            }
        }
    },
    "b775a741a567": {
        "endpoints": [
            "http://localhost:28800",
            "http://ai122:28800"
        ],
        "machines": {
            "localhost:28800": {
                "optimal_batch_size": 2000,
                "optimal_max_concurrent": 2,
                "throughput": 990,
                "instances": 2,
                "updated_at": "2026-01-16T21:54:10.945352"
            },
            "ai122:28800": {
                "optimal_batch_size": 2000,
                "optimal_max_concurrent": 8,
                "throughput": 2600,
                "instances": 8,
                "updated_at": "2026-01-16T21:54:38.278253"
            }
        }
    }
}
```

### Manual Configuration Tuning

```sh
# Auto-tune batch sizes for optimal performance
tei_benchmark -E $TEI_EPS tune --min-batch 500 --max-batch 3000 --step 250

# Results automatically saved to tei_clients.config.json
```

## Performance Features

- **Small batches (<10 items)**: Round-robin across machines
- **Large batches**: Async pipeline with optimal distribution
- **Progress tracking**: Every 5 seconds (with `TEIClientsWithStats`)
- **Auto load balancing**: Based on machine health and capacity
- **Auto batch sizing**: Each machine uses its optimal batch size from config

## API Reference

### TEIClients

```python
TEIClients(endpoints: list[str])
```

**Methods:**
- `health() -> ClientsHealthResponse`
- `embed(inputs, normalize=True, truncate=True) -> list[list[float]]`
- `lsh(inputs, bitn=2048, normalize=True, truncate=True) -> list[str]`
- `lsh_iter(inputs, total_hint=None, bitn=2048) -> list[str]`
- `info() -> list[InfoResponse]`
- `close()`

### TEIClientsWithStats

```python
TEIClientsWithStats(endpoints: list[str], verbose: bool = False)
```

Same methods as `TEIClients`. When `verbose=True`:
- Shows loaded configuration
- Progress updates every 5 seconds
- Per-machine and total throughput
- Completion statistics

## Troubleshooting

**No healthy machines:**
```python
health = clients.health()
if health.healthy_machines == 0:
    print("No healthy machines - check endpoints and TEI services")
```

**Low throughput:**
```sh
# Re-tune configuration
tei_benchmark -E $TEI_EPS tune

# Check performance with verbose output
tei_benchmark -E $TEI_EPS -v run -n 10000
```

**Memory issues:**
```python
# Use iterator for large datasets
results = clients.lsh_iter(data_generator(), total_hint=1000000)
```
