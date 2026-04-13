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

## Workflow C: Fast Wake/Resume

```bash
bash runs/qsrs/01_deploy_default.sh
bash runs/qsrs/98_sleep_backends.sh
bash runs/qsrs/02_start_machine.sh
```

## Workflow D: Mixed Soak

```bash
bash runs/qsrs/01_deploy_default.sh
bash runs/qsrs/02_start_machine.sh
bash runs/qsrs/06_soak_mixed.sh
```

## Notes

- The scripts use `qsr` when installed, otherwise they fall back to `python -m tfmx.qsrs.cli`.
- By default, `01_deploy_default.sh` uses all GPUs currently visible in the VM.
- If you want a subset, export `QSR_DEPLOY_GPUS`, for example `QSR_DEPLOY_GPUS=0,1`.
- Staged deploy/start scripts now enable sleep mode and try wake-first recovery by default. Export `QSR_ENABLE_SLEEP_MODE=0` only if you explicitly want the old cold-start-only behavior.
- Export `QSR_PROFILE_STARTUP=1` if you want `01_deploy_default.sh` to print a cold-start phase profile during deploy.
- `02_start_machine.sh` launches `qsr machine` in the background with `--auto-start` and `--on-conflict replace`.
- `02_start_machine.sh` now tries `qsr compose wake --wait-healthy` before falling back to the normal machine auto-start path, unless you opt out with `QSR_ENABLE_SLEEP_MODE=0`.
- `98_sleep_backends.sh` stops `qsr machine` and requests backend sleep so the next `02_start_machine.sh` can wake them quickly.
- `06_soak_mixed.sh` runs a mixed chat/transcribe soak against the machine endpoint and writes a JSON summary under `runs/qsrs/results/`.
- `python debugs/qsrs/profile_startup.py qsr-uniform--gpu0 ...` parses backend docker logs into internal cold-start phases such as model load, torch.compile, KV cache setup, graph capture, and first health.
- `qsr client transcribe-long -E http://127.0.0.1:27900 ./long.mp3 --json` is the current long-audio path for turning one long file into many silence-aware chunk requests and spreading them across idle GPUs.
- Default sample audios come from the public Qwen3-ASR repo; override them with `QSR_SMOKE_AUDIO`, `QSR_BENCH_AUDIO`, `QSR_SCHED_AUDIO_A`, or `QSR_SCHED_AUDIO_B` if needed.
- `06_soak_mixed.sh` accepts `QSR_SOAK_AUDIO_A`, `QSR_SOAK_AUDIO_B`, `QSR_SOAK_TRANSCRIBE_SAMPLES`, `QSR_SOAK_CHAT_SAMPLES`, `QSR_SOAK_MAX_WORKERS`, and `QSR_SOAK_MAX_TOKENS`.
- `qsr machine` daemon logs live under `~/.cache/tfmx/qsr_machine.log`.
- Benchmark results are written under `runs/qsrs/results/`.