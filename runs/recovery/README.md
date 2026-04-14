# Recovery Workflows

## TEI + QWN

这组脚本用于把本机的 TEI `28800` 与 QWN `27800` 一起重启，并补完最小可用的 live 验证。

## 默认脚本

```bash
bash runs/recovery/restart_tei_qwn.sh
```

默认行为：

- 默认使用当前 VM 中全部可见 GPU
- 先停止旧的 QWN machine 与已知 compose 项目，再清理 TEI compose
- 重新拉起 TEI 后端并等待 direct backend `/health`
- 以 `--on-conflict replace` 方式重启 TEI machine，并执行 3 轮 `240` 输入的 `/lsh` 校验
- 重新拉起 QWN uniform-awq 后端并等待 direct backend `/health`
- 以 `--on-conflict replace` 方式重启 QWN machine，并执行 `/v1/models` 与真实 chat completion 校验
- 默认再追加一轮轻量 `qwn benchmark`，输出写入 `runs/qwns/results/`

说明：脚本在后端预热阶段优先相信 direct backend HTTP `/health`，而不是 Docker healthcheck 状态，因为后者可能比真实服务可用性晚几分钟收敛。

## 常用覆盖参数

```bash
TEI_GPUS=0,1 \
QWN_GPUS=0,1 \
QWN_BENCH_SAMPLES=20 \
bash runs/recovery/restart_tei_qwn.sh
```

```bash
TEI_BENCH_SAMPLES=60000 \
QWN_BENCH_SAMPLES=80 \
QWN_BENCH_MAX_TOKENS=128 \
bash runs/recovery/restart_tei_qwn.sh
```

主要环境变量：

- `TEI_GPUS` / `QWN_GPUS`：要部署的 GPU 列表；默认会自动取当前 VM 中全部可见 GPU
- `TEI_LSH_INPUTS` / `TEI_LSH_ROUNDS`：TEI live 校验强度，默认 `240` 与 `3`
- `TEI_BENCH_SAMPLES`：若大于 `0`，脚本会额外执行 `tei benchmark run`
- `QWN_BENCH_SAMPLES`：若大于 `0`，脚本会额外执行 `qwn benchmark run`
- `QWN_BENCH_MAX_TOKENS`：QWN benchmark 的 `--max-tokens`
- `WAIT_TIMEOUT_SEC`：等待后端和 machine 变健康的超时时间，默认 `600`

## 产物位置

- 恢复日志与 TEI machine 启动日志：`runs/recovery/results/`
- QWN benchmark 结果：`runs/qwns/results/`
- TEI benchmark 结果：`runs/teis/results/`

## 适用场景

- 主机重启后，需要快速恢复 TEI + QWN 到当前 VM 中全部可见 GPU
- 想把“重启 + live 验证 + 轻量 benchmark”压缩成一个可复用入口
- 想在更高负载下复测两套服务共存时的稳定性

## TEI + QSR

若你现在要恢复的是 TEI `28800` 与 QSR `27900`，可直接执行：

```bash
bash runs/recovery/restart_tei_qsr.sh
```

默认行为：

- 默认使用当前全部可见 GPU，并允许 TEI / QSR 共存在同一批 GPU 上
- 先清理旧的 TEI/QSR machine 与 compose 部署
- 重新拉起 TEI backend + machine，并执行 `runs/teis/03_health_check.sh`
- 重新拉起 QSR backend + machine，并执行 `runs/qsrs/03_health_check.sh`

常用覆盖参数：

```bash
TEI_GPUS=0,1,2,3,4,5 \
QSR_GPUS=0,1,2,3,4,5 \
bash runs/recovery/restart_tei_qsr.sh
```

若你想让它在恢复完成后顺手跑 benchmark：

```bash
TEI_RUN_BENCHMARK=1 \
QSR_RUN_BENCHMARK=1 \
QSR_BENCH_SAMPLES=200 \
bash runs/recovery/restart_tei_qsr.sh
```

若你想专门测试 TEI + QSR 共存时的稳定性与吞吐，可直接执行：

```bash
bash runs/recovery/benchmark_tei_qsr_coexist.sh
```

默认会：

- 读取当前已经运行中的 TEI `28800` 与 QSR `27900`
- 先抓一次 pre-state 的 `/health` 与 `/info`
- 并发执行 TEI benchmark 与 QSR benchmark
- 再抓一次 post-state，并生成一份 Markdown 摘要

常用覆盖参数：

```bash
TEI_BENCH_SAMPLES=12000 \
QSR_BENCH_SAMPLES=1200 \
QSR_BENCH_MODE=transcribe \
bash runs/recovery/benchmark_tei_qsr_coexist.sh
```

主要产物：

- `runs/recovery/results/tei_qsr_coexist_*.md`
- `runs/recovery/results/tei_qsr_coexist_pre_*.json`
- `runs/recovery/results/tei_qsr_coexist_post_*.json`
- `runs/teis/results/coexist_*.json`
- `runs/qsrs/results/coexist_*.json`