# QWN Run Scripts

These scripts mirror the staged workflow already used by `runs/qvls/`, but target the unified `qwn` CLI.

## Workflow A: Single-GPU Smoke Test

```bash
bash runs/qwns/01_deploy_awq.sh
bash runs/qwns/03_health_check.sh
bash runs/qwns/02_start_machine.sh
bash runs/qwns/04_benchmark.sh
bash runs/qwns/99_cleanup.sh
```

## Workflow B: Uniform Multi-GPU Throughput Test

```bash
bash runs/qwns/01_deploy_uniform.sh
bash runs/qwns/03_health_check.sh
bash runs/qwns/02_start_machine.sh
bash runs/qwns/05_test_scheduling.sh
bash runs/qwns/99_cleanup.sh
```

## Notes

- The scripts use `qwn` when installed, otherwise they fall back to `python -m tfmx.qwns.cli`.
- The default qwn network config already applies the Hugging Face mirror and USTC PyPI mirror.
- Export `QWN_PROXY` only if you want to provide an explicit build proxy override.
- `02_start_machine.sh` starts the proxy in background mode.
- Benchmark results are written under `runs/qwns/results/`.