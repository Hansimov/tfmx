# QWN Run Scripts

These scripts mirror the staged workflow already used by `runs/qvls/`, but target the unified `qwn` CLI.

## Workflow A: Single-GPU Smoke Test

```bash
bash runs/qwns/01_deploy_awq.sh
bash runs/qwns/02_start_machine.sh
bash runs/qwns/03_health_check.sh
bash runs/qwns/04_benchmark.sh
bash runs/qwns/99_cleanup.sh
```

## Workflow B: Uniform Multi-GPU Throughput Test

```bash
bash runs/qwns/01_deploy_uniform.sh
bash runs/qwns/02_start_machine.sh
bash runs/qwns/03_health_check.sh
bash runs/qwns/05_test_scheduling.sh
bash runs/qwns/99_cleanup.sh
```

## Workflow C: Fast Restart Without Cold Reinit

```bash
bash runs/qwns/01_deploy_uniform.sh
bash runs/qwns/02_start_machine.sh
bash runs/qwns/03_health_check.sh
bash runs/qwns/98_sleep_backends.sh
bash runs/qwns/02_start_machine.sh
bash runs/qwns/03_health_check.sh
```

默认会使用当前 VM 中全部可见 GPU；如需限制子集，可直接在 `01_deploy_uniform.sh` 的命令基础上改成 `qwn compose up --gpu-layout uniform-awq -g 0,1` 之类的显式列表。

## Notes

- The scripts use `qwn` when installed, otherwise they fall back to `python -m tfmx.qwns.cli`.
- The default qwn network config already applies the Hugging Face mirror and USTC PyPI mirror.
- Export `QWN_PROXY` only if you want to provide an explicit build proxy override.
- `01_deploy_uniform.sh` enables vLLM sleep mode by default; set `QWN_ENABLE_SLEEP_MODE=0` to opt out.
- `02_start_machine.sh` can wake sleeping backends and now also runs `qwn compose warmup --wait-healthy` by default, so each GPU pays its one-time first-request lazy init before user traffic arrives.
- Set `QWN_WARMUP_BACKENDS=0` if you explicitly want to skip that warmup step.
- `98_sleep_backends.sh` is the non-destructive stop path; `99_cleanup.sh` remains the fully destructive teardown path.
- Benchmark results are written under `runs/qwns/results/`.