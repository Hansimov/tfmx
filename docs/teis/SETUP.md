# TEI 配置与启动说明

## 概览

`tei` 是当前唯一的公开 CLI 入口，包含四类子命令：

- `tei compose`：生成并管理 Docker Compose 后端部署
- `tei machine`：启动单机多 GPU 聚合代理
- `tei client`：访问 `tei machine`、直接访问单个后端，或通过 `-E` 访问多个 machine
- `tei benchmark`：生成文本样本、压测、调优批量配置

当前默认 embedding 模型是：

- `Qwen/Qwen3-Embedding-0.6B`

默认端口布局：

- `tei machine`：`28800`
- TEI 后端：`28880+`

## 前置条件

- 已安装 Docker 与 NVIDIA Container Toolkit
- `nvidia-smi` 可以看到目标 GPU
- Python 3.10+
- 建议本机已有 Hugging Face 模型缓存，避免首次启动等待过久

## 安装

```bash
pip install -e .
tei --help
```

## 建议先准备的环境变量

```bash
export MODEL="Qwen/Qwen3-Embedding-0.6B"
export TEI_MACHINE_URL="http://$TEI_HOST:28800"
export TEI_PROXY_URL="http://$PROXY_HOST:$PROXY_PORT"
```

## 第一次启动前建议做的事

如果你已经把模型下载到 Hugging Face cache，建议先执行一次：

```bash
tei compose setup -m "$MODEL"
tei compose health
```

说明：

- `tei compose setup` 会在模型缓存里补齐 `sentence_*_config.json` 这类配置文件
- 这一步通常每个模型只需要做一次
- `tei compose health` 会先检查当前机器哪些 GPU 仍然健康，避免把坏卡也拉起来

## 快速开始

### 方案 A：让 `tei machine` 自动拉起后端

```bash
tei machine run --auto-start --perf-track --on-conflict replace
tei client health --port 28800
tei client embed --port 28800 "Hello" "World"
```

如果你只想让 auto-start 使用指定 GPU：

```bash
tei machine run --auto-start --compose-gpus "0,1" --on-conflict replace

# 或者显式走 per-GPU 配置语法
tei machine run --auto-start --compose-gpu-configs "0,1" --on-conflict replace
```

说明：

- `--auto-start` 会在没有发现现成后端时自动调用 `tei compose up`
- 启动时如果只有部分 GPU 后端恢复健康，`tei machine` 会先带着这些健康实例启动，而不是因为一张坏卡一直阻塞
- `--on-conflict replace` 适合替换旧的 `tei machine` 进程或占用 `28800` 的旧监听器

### 方案 B：手动启动 compose，再启动 machine

```bash
tei compose up -m "$MODEL"
tei machine run --perf-track --on-conflict replace
```

如果你要手动限制 GPU：

```bash
tei compose up --gpu-configs "0,1"
tei machine run --on-conflict replace
```

如果你处在受限网络环境：

```bash
tei compose up -m "$MODEL" --proxy "$TEI_PROXY_URL"
```

如果 NVIDIA runtime/NVML 当前不稳定：

```bash
tei compose up -m "$MODEL" --mount-mode manual
```

## 验证服务是否可用

```bash
tei client health --port 28800
tei client info --port 28800
tei client embed --port 28800 "Hello" "World"
tei client lsh --port 28800 "Hello, world"
```

## 端口布局

| 服务 | 端口 | 说明 |
| --- | --- | --- |
| `tei machine` | `28800` | 单机聚合代理 |
| TEI backend | `28880+` | 每张健康 GPU 一个后端 |

## 生命周期管理

Compose 层：

```bash
tei compose ps
tei compose logs -f
tei compose stop
tei compose start
tei compose restart
tei compose down
```

Machine 层：

- `tei machine` 现在支持内建后台 daemon 模式，可直接执行 `tei machine run --background --auto-start --on-conflict replace`
- daemon 的 PID / 日志默认写到 `~/.cache/tfmx/tei_machine.pid` 与 `~/.cache/tfmx/tei_machine.log`
- 如果是 TEI + QSR 这种依赖 GPU 与 Docker 的重资源组合，更推荐主机重启后按需执行 `bash runs/recovery/start_tei_qsr.sh`，而不是默认做开机自启
- 如果只是短期驻留单独的 TEI machine，再考虑放到 `tmux`、`screen` 或你现有的运维系统里
- 如果你担心旧进程占住 `28800`，直接在启动命令里加 `--on-conflict replace`