# QWN 使用说明

## 统一入口

`qwn` 是唯一的公开入口，所有能力都通过子命令暴露。

## `qwn compose`

```bash
qwn compose up
qwn compose up -g 0,1
qwn compose up --gpu-layout uniform-awq
qwn compose up --gpu-configs "0:4b:4bit,1:4b:4bit"
qwn compose generate -j qwn-awq --gpu-configs "0:4b:4bit"
qwn compose ps
qwn compose logs -f
qwn compose down
```

### 常用参数

- `--gpu-configs`：显式指定每张 GPU 的模型/量化配置
- `--gpu-layout uniform-awq`：对所有选中 GPU 应用同一套默认 AWQ 部署
- `--mount-mode manual`：直接挂载 `/dev/nvidia*`
- `--proxy`：覆盖默认 build 代理
- `--hf-endpoint`：覆盖默认 Hugging Face mirror，默认值为 `https://hf-mirror.com`
- `--pip-index-url`：覆盖默认 PyPI mirror，默认值为 `https://mirrors.ustc.edu.cn/pypi/simple`
- `--pip-trusted-host`：覆盖默认 PyPI trusted host，默认值为 `mirrors.ustc.edu.cn`
- 若环境里已经设置 `QWN_PROXY`、`TFMX_QWN_PROXY` 或系统代理，`qwn` 会自动复用
- `--project-name`：自定义 compose 项目名

## `qwn machine`

```bash
qwn machine run
qwn machine run -b
qwn machine run -e http://localhost:27880,http://localhost:27881
qwn machine discover
qwn machine health
qwn machine status
qwn machine logs --tail 200
qwn machine stop
qwn machine restart
```

### OpenAI 兼容接口

- `POST /v1/chat/completions`
- `GET /v1/models`
- `GET /health`
- `GET /info`
- `GET /metrics`
- `POST /chat`

### 调度观测

- `GET /info` 现在除了实例列表和累计统计外，还会返回顶层 `scheduler` 摘要，包含当前算法名、近期窗口长度、实例获取超时、最近健康刷新年龄、基础权重、实时生效权重，以及自动调权配置与最近窗口信号
- 每个实例的 `scheduler` 字段会暴露 `score`、`recent_requests`、`recent_failures`、`latency_ema_ms`、`ttft_ema_ms`、`tokens_per_second_ema`、`cooldown_remaining_sec` 等实时调度指标
- `GET /metrics` 会额外输出 Prometheus 文本格式指标，当前重点包括：`qwn_machine_scheduler_weight`、`qwn_machine_scheduler_signal`、`qwn_machine_instance_scheduler_score`，以及请求量、token、错误、failover、等待事件等运行时计数器
- `qwn client info` 会把这些关键字段整理成更易读的终端输出，例如 `active=已用/总槽位`、`score=...`、`lat=...`、`ttft=...`、`tokps=...`、`recent=...`，并直接展示当前 `weights`、`base_weights` 与 `tuning.signals`

## `qwn client`

```bash
qwn client health
qwn client models
qwn client info
qwn client chat "解释一下 AWQ 量化的用途"
qwn client chat -i image_a.png -i image_b.png "先看第一张图" "再比较第二张图"
qwn client chat "你好" --no-stream
qwn client chat "请展示详细推理过程" --thinking
qwn client generate --prompt "总结一下 qwn compose 的作用"
qwn client -e http://localhost:27880 models
```

`qwn client info` 现在除了基础实例信息外，还会显示调度器的实时画像，便于快速判断某张卡当前是否因为 GPU 压力、近期延迟、近期 tok/s 或冷却期而被降权。

### 流式输出说明

- `qwn client chat "..."` 现在默认以流式方式在终端打印增量文本
- 结束后会追加统计信息，格式类似：`[统计]: 首 0.5s | 总 8.3s | 103 tokens | 120.7 token/s`
- 每一项统计现在都会把整段值一起着色，便于快速对齐“首耗时 / 总耗时 / token 数 / token 速率”
- 若分钟数为 `0`，则耗时直接显示秒，例如：`[统计]: 首 0.5s | 总 8.3s | 64 tokens | 41.7 token/s`
- 若你需要旧的非流式行为，可以加 `--no-stream`
- 当前默认会显式关闭 Qwen thinking 模式，只输出最终回答；如需保留思维链输出，可加 `--thinking`
- `--thinking` 打开后，终端流式输出会在前后补上 `<thinking>` 与 `</thinking>` 包裹，便于肉眼区分思考段与正文
- 当前 `qwn client chat` 与 `qwn client generate` 的默认 `--max-tokens` 为 `8192`，但实际可生成上限仍受当前服务端 `--max-model-len=8192` 约束

### 多图多文本消息

- `qwn client chat` 支持重复传入 `-i/--image`，并支持多个文本片段
- 多个文本片段会与图片按顺序交错组成一个 OpenAI 兼容的 multimodal user message
- 图片参数既可以是本地文件路径，也可以是 URL 或 `data:` URI

## `qwn benchmark`

```bash
qwn benchmark health -E http://localhost:27800
qwn benchmark run -E http://localhost:27800 -n 100
qwn benchmark run -E http://localhost:27800 http://host2:27800 -n 100 -o bench.json
qwn benchmark run -E http://localhost:27800 -n 100 --no-ttft
qwn benchmark generate -n 5 --show
```

benchmark 使用 `tfmx.teis.benchtext.TEIBenchTextGenerator` 生成中文文本样本，因此不依赖 `data/prompts.txt` 里的视觉任务提示词。

- `qwn benchmark run` 现在默认通过流式请求额外统计 `ttft_sec`，并同时输出 `submitted/successful/failed/success_rate`
- `throughput.requests_per_second` 表示成功请求吞吐；`submitted_requests_per_second` 表示提交吞吐
- 若你只关心总吞吐，不想为 TTFT 走流式路径，可显式加 `--no-ttft`