# QSR 排障与经验说明

## 推荐流程

建议按下面顺序使用：

1. `qsr compose up --gpu-layout uniform`
2. `qsr machine run --auto-start -b --on-conflict replace`
3. `qsr client health`
4. `qsr client models`
5. `qsr client transcribe ./sample.wav`
6. `qsr benchmark run -E "$QSR_MACHINE_URL" -n 20 --audio ./sample.wav`

这套流程的目的不是“最少命令”，而是先确认后端容器能拉起，再启动聚合代理，最后才进入实际 ASR 请求与压测。

## 模型标签

- Docker 层当前默认加载的模型仓库是 `Qwen/Qwen3-ASR-0.6B`
- 对外稳定暴露的短标签是 `0.6b`
- `/models` 与 `/v1/models` 当前还会额外暴露兼容别名 `qwen3-asr-0.6b`、`qwen3-asr` 以及完整 HF repo 名
- 因此 `qsr client models` 看到的名称不一定只剩一个 repo 名，这是兼容不同 OpenAI 客户端的有意设计

## Runtime Image

- `qsr` 会自动基于 `vllm/vllm-openai:v0.19.0` 构建本地镜像 `tfmx-vllm-openai:qwen3-asr-v0.19.0`
- 构建时会固定安装 `vllm[audio]==0.19.0`、`qwen-asr==0.0.6` 与 `huggingface-hub>=0.34.0,<1.0`
- 这样做是为了避开上游 `qwen-asr[vllm]` 当前内置的 `vllm==0.14.0` 约束，防止 `pip` 在 Docker build 中回溯到需要源码编译 `xformers` 的旧版本链路
- Hugging Face 默认使用 `hf-mirror.com`，`pip` 默认使用 USTC mirror
- 若环境里存在 `QSR_PROXY_URL`、`TFMX_QSR_PROXY` 或系统代理，构建和容器内 runtime 会自动复用

手动重建示例：

```bash
docker rmi tfmx-vllm-openai:qwen3-asr-v0.19.0
qsr compose up --gpu-configs "0"
```

## 音频输入形式

`qsr client transcribe` 和 `qsr client chat --audio` 当前都支持三类输入：

- 本地文件路径
- `http://` 或 `https://` URL
- `data:` URI

如果你只是想做最稳定的最小链路验证，优先使用本地文件；这样可以避免把问题混到远端音频下载或代理链路里。

## `transcribe` 和 `chat` 的职责边界

- `transcribe`：最适合纯转写场景，直接走 `/v1/audio/transcriptions`
- `chat`：适合“转写 + 理解/总结/翻译/问答”这类需要文本提示词参与的场景

如果你只是排查模型是否能正确识别音频，优先从 `transcribe` 开始；如果要验证 OpenAI multimodal chat 兼容性，再切到 `chat`。

## `27900` 被旧进程占住

如果你看到端口冲突，不必先手动找 PID，直接使用：

```bash
qsr machine run --auto-start -b --on-conflict replace
```

补充说明：

- `report`：发现冲突就报错退出
- `replace`：先清掉旧 listener，再启动新的 machine daemon
- machine daemon 的状态、PID 和日志可以通过 `qsr machine status` / `qsr machine logs` 查看

## `/health` 显示 0 个 healthy 实例

建议按这个顺序排查：

```bash
qsr compose ps
qsr compose logs -f
qsr machine discover
qsr machine health
curl http://127.0.0.1:27900/health
curl http://127.0.0.1:27900/info
```

说明：

- 如果容器仍在加载模型，`qsr machine` 会暂时把它视为不可用
- 如果容器已经退出，优先看 `qsr compose logs -f`，再看宿主机 `nvidia-smi`
- 如果 `nvidia-smi` 自己已经报错，真正的问题在 GPU/驱动层，而不是 machine 层

## 自动发现不到容器

自动发现默认按容器名前缀 `qsr-` / `qsr_` 过滤。如果你用了自定义项目名，可显式指定：

```bash
qsr machine run -n 'my-qsr-project'
qsr machine run -e http://localhost:27980,http://localhost:27981
```

## 当前调度与可观测性

- 当前调度器是轻量级 `least_active_idle`，不是像 `qwn machine` 那样的复杂自适应调度器
- `/info` 会返回实例列表、每实例当前活跃请求数、最近健康检查延迟、累计请求统计和当前可用模型
- `/metrics` 当前暴露的是基础 Prometheus 指标，例如实例数、健康实例数、总请求数、错误数、failover 数和当前活跃请求数

因此如果你想解释“为什么这次路由到了某张卡”，当前最直接的判断依据还是 `/info` 里的 `active_requests` 与 `healthy` 状态，而不是更复杂的实时权重。

## 大 prompt / 大 `max_tokens` 被后端拒绝

`qsr client chat` 当前内置了一层自动回退：如果上游返回“请求 token 数超过上下文限制”，客户端会尝试自动收紧 `max_tokens` 后重试。

如果你希望手动避免这类问题，可直接显式收紧：

```bash
qsr client chat --audio ./sample.wav --max-tokens 256 "请转写并总结"
```

而 `transcribe` 路径本身不依赖 `max_tokens`，因此做纯 ASR 验证时通常更稳定。

## benchmark 口径

- `qsr benchmark run` 默认输出请求数、成功率、吞吐、输出字符数、延迟分位数
- `--mode chat` 时默认额外测量 TTFT；若只关心总吞吐，可加 `--no-ttft`
- 如果 benchmark 使用的是远端 URL 音频，那么结果会同时包含远端音频拉取链路的影响；若你要更纯粹地测模型/代理本身，建议改用本地音频文件

## 当前实现边界

- 当前 repo 内的默认 compose/脚本/文档都围绕 `Qwen/Qwen3-ASR-0.6B`
- 当前 `qsr compose` 没有额外的 `setup`、`warmup`、`sleep`、`wake` 子命令
- 当前 `qsr machine` 是单机聚合代理，不负责跨机器统一入口