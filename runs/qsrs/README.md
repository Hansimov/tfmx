# QSR Run Scripts

These scripts provide the same staged workflow shape as `runs/teis/` and `runs/qwns/`, but target the unified `qsr` CLI.

## Workflow A: Default All-GPU Smoke Test

```bash
bash runs/qsrs/01_deploy_default.sh
bash runs/qsrs/02_start_machine.sh
bash runs/qsrs/03_health_check.sh
bash runs/qsrs/04_benchmark.sh
bash runs/qsrs/99_cleanup.sh
```

## Workflow B: Scheduling Smoke Test

```bash
bash runs/qsrs/01_deploy_default.sh
bash runs/qsrs/02_start_machine.sh
bash runs/qsrs/03_health_check.sh
bash runs/qsrs/05_test_scheduling.sh
bash runs/qsrs/99_cleanup.sh
```

## Notes

- The scripts use `qsr` when installed, otherwise they fall back to `python -m tfmx.qsrs.cli`.
- By default, `01_deploy_default.sh` uses all GPUs currently visible in the VM.
- If you want a subset, export `QSR_DEPLOY_GPUS`, for example `QSR_DEPLOY_GPUS=0,1`.
- `02_start_machine.sh` launches `qsr machine` in the background with `--auto-start` and `--on-conflict replace`.
- Default sample audios come from the public Qwen3-ASR repo; override them with `QSR_SMOKE_AUDIO`, `QSR_BENCH_AUDIO`, `QSR_SCHED_AUDIO_A`, or `QSR_SCHED_AUDIO_B` if needed.
- `qsr machine` daemon logs live under `~/.cache/tfmx/qsr_machine.log`.
- Benchmark results are written under `runs/qsrs/results/`.