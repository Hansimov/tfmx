# QVL Deployment Scripts

Run these scripts in order for a full deployment + test cycle.

## Workflow A: Mixed Model Testing

```bash
# 1. Download all needed models (once)
bash runs/qvls/00_download_models.sh

# 2. Deploy 6 different model/quant combos
bash runs/qvls/01_deploy_mixed.sh

# 3. Wait ~120s, then health check
bash runs/qvls/03_health_check.sh

# 4. Start machine proxy (in a separate terminal)
bash runs/qvls/02_start_machine.sh

# 5. Run model comparison test
bash runs/qvls/04_test_mixed.sh

# 6. Cleanup
bash runs/qvls/99_cleanup.sh
```

## Workflow B: Scheduling Throughput Testing

```bash
# 1. Deploy uniform 4b-instruct on all 6 GPUs
bash runs/qvls/01_deploy_uniform.sh

# 2. Wait ~120s, then health check
bash runs/qvls/03_health_check.sh

# 3. Start machine proxy (in a separate terminal)
bash runs/qvls/02_start_machine.sh

# 4. Run scheduling throughput benchmark
bash runs/qvls/05_test_scheduling.sh

# 5. Cleanup
bash runs/qvls/99_cleanup.sh
```

## GPU Layout

| GPU | Mixed Deploy | Uniform Deploy |
|-----|-------------|----------------|
| 0 | 2b-instruct:4bit | 4b-instruct:4bit |
| 1 | 2b-thinking:4bit | 4b-instruct:4bit |
| 2 | 4b-instruct:4bit | 4b-instruct:4bit |
| 3 | 4b-thinking:4bit | 4b-instruct:4bit |
| 4 | 8b-instruct:4bit | 4b-instruct:4bit |
| 5 | 8b-thinking:4bit | 4b-instruct:4bit |
