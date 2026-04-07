# TEI Run Scripts

These scripts provide the same staged workflow shape as `runs/qwns/`, but target the unified `tei` CLI.

## Workflow A: Default All-GPU Smoke Test

```bash
bash runs/teis/01_deploy_default.sh
bash runs/teis/02_start_machine.sh
bash runs/teis/03_health_check.sh
bash runs/teis/04_benchmark.sh
bash runs/teis/99_cleanup.sh
```

## Workflow B: Pipeline Throughput Test

```bash
bash runs/teis/01_deploy_default.sh
bash runs/teis/02_start_machine.sh
bash runs/teis/03_health_check.sh
bash runs/teis/05_test_scheduling.sh
bash runs/teis/99_cleanup.sh
```

## Notes

- The scripts use `tei` when installed, otherwise they fall back to `python -m tfmx.teis.cli`.
- By default, `01_deploy_default.sh` uses all GPUs currently visible in the VM.
- If you want a subset, export `TEI_DEPLOY_GPUS`, for example `TEI_DEPLOY_GPUS=0,1`.
- `02_start_machine.sh` launches `tei machine` in the background with `--on-conflict replace` and records its PID under `runs/teis/results/`.
- Benchmark results are written under `runs/teis/results/`.