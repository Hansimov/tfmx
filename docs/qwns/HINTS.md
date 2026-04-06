# QWN 排障与经验说明

## 模型标签

- Docker 层当前加载的模型仓库是 `cyankiwi/Qwen3.5-4B-AWQ-4bit`
- 对外统一暴露的模型名被规范成 `4b:4bit`
- `qwn client models` 和 `GET /v1/models` 展示的是服务标签，不一定是原始 Hugging Face repo 名称

## Runtime Image

- `qwn` 会自动基于 `vllm/vllm-openai:v0.19.0` 构建本地镜像 `tfmx-vllm-openai:qwen3.5-v0.19.0`
- 这是因为当前只有 `transformers` 主线能稳定识别这个模型的 `qwen3_5` 架构
- 镜像还会安装 `huggingface-hub>=1.5.0,<2.0`，用于匹配 `transformers` 主线的依赖要求
- `docker build --progress=plain` 已经默认开启，所以构建时不会再出现“看起来卡住但其实在下载”的情况
- Hugging Face 默认使用 `hf-mirror.com`，`pip` 默认使用 USTC mirror
- 若环境中存在 `QWN_PROXY`、`TFMX_QWN_PROXY` 或系统代理，构建镜像时会自动复用
- 如果共享 Hugging Face cache 已经包含模型和预处理配置，runtime 容器会自动打开 `HF_HUB_OFFLINE=1` 与 `TRANSFORMERS_OFFLINE=1`，避免多实例启动时因容器 DNS 抖动而反复访问远端

手动重建示例：

```bash
docker rmi tfmx-vllm-openai:qwen3.5-v0.19.0
qwn compose up --gpu-configs "0:4b:4bit"
```

## 当前显存为什么会到 13GB 左右

现在只起了一个 Docker 实例，并且只占用 GPU0，但 `gpustat` 中看到 `VLLM::EngineCore` 约 `13144 MB` 并不意外，主要原因不是单一的“模型权重太大”，而是以下几部分叠加：

1. 当前模型虽然以文本服务方式使用，但从日志看它被解析成 `Qwen3_5ForConditionalGeneration`
2. 启动日志里还出现了 `Qwen2VLImageProcessor` 和 `Multi-modal warmup completed`，说明它本质上不是一个极简纯文本 4B 模型路径，而是带多模态组件的条件生成架构
3. `vllm` 会预留大量 KV cache。当前日志里已经明确写出 `Available KV cache memory: 8.45 GiB`
4. 当前默认参数是 `--gpu-memory-utilization 0.72`、`--max-model-len 8192`、`--max-num-seqs 8`，这组参数会把并发能力配得比较激进
5. 启动阶段还有 `torch.compile`、CUDA graph capture、warmup 等额外开销，虽然日志里图池本身只有约 `0.07 GiB`，但初始化阶段会出现额外峰值

换句话说，约 `13GB` 的显存占用里，大头通常是：

- AWQ 模型权重与运行时基础开销
- 多模态相关组件
- 8GB 以上的 KV cache 预留

## 如何防止 OOM

如果这张卡还要和别的进程共享，或者你观察到 warmup 阶段容易爆显存，优先按下面顺序收紧：

1. 先降低 `--gpu-memory-utilization`
2. 再降低 `--max-model-len`
3. 最后降低 `--max-num-seqs`

比较稳妥的一组起步参数是：

```bash
qwn compose up \
  --gpu-configs "0:4b:4bit" \
  --gpu-memory-utilization 0.60 \
  --max-model-len 4096 \
  --max-num-seqs 4
```

如果还不够稳，可以继续收紧到：

```bash
qwn compose up \
  --gpu-configs "0:4b:4bit" \
  --gpu-memory-utilization 0.55 \
  --max-model-len 2048 \
  --max-num-seqs 2
```

实践建议：

- 给同卡其他进程至少留出 `2-4 GiB` 余量
- 若只是单用户交互，不要把 `max_num_seqs` 设得太高
- `8192` 的上下文长度不是免费能力，它会直接推高 KV cache 占用
- 当前模型路径偏多模态，如果目标是更低显存文本服务，应优先选择真正的纯文本 Qwen 3.5 模型仓库

## 推荐流程

1. `qwn compose up --gpu-configs "0:4b:4bit"`
2. 等待 vLLM 完成加载与 warmup
3. `qwn machine run -b`
4. `qwn client health`
5. `qwn client chat "你好"`
6. `qwn benchmark run -E http://localhost:27800 -n 100`

## CLI 流式输出

- `qwn client chat "..."` 现在默认以流式方式输出
- 默认会传入 `chat_template_kwargs.enable_thinking=false`，避免把模型思维链直接打印到终端
- 最后一行会附带统计信息，例如：`[统计]: 首 0.8s | 总 1min 12.4s | 512 tokens | 38.7 token/s`
- 若分钟数为 `0`，则只显示秒，例如：`elapsed: 8.3s | 41.7 token/s`
- 如需旧的非流式输出，显式加 `--no-stream`
- 如果你确实想观察 thinking 输出，再显式加 `--thinking`
- `--thinking` 模式下，CLI 会在终端流式输出前后补上 `<thinking>` 与 `</thinking>` 包裹；同时 `ttft` 现在按“首个非空白可见文本”计算，不再被纯换行或空格过早触发

## 与 TEI 共卡时的调度行为

- `qwn machine` 现在会周期性采样每张 GPU 的运行时利用率与显存占用，并把这些指标纳入实例选择
- 在没有外部负载时，5 个健康实例可以保持接近均匀的分发
- 当某张卡上的 TEI embedding 实例被持续打满时，`qwn machine` 会降低向该卡对应 LLM 实例分发新请求的优先级
- 这不是硬隔离，也不是跨服务统一调度器；它属于轻量级的“避让”策略，目标是减少同卡热点，而不是保证严格 QoS
- 当前机器上的一次实测结果是：5 个健康 `qwn` 实例在无干扰时，180 个请求平均落到每卡 36 个；在 GPU2 上持续施加 TEI embedding 压力后，GPU2 对应 `qwn` 实例只接到 24 个请求，而其他卡大致落在 36 到 40 个之间
- 2026-04-06 的另一轮实测中，`120` 个请求在无干扰时刚好均匀落到 `24 / 24 / 24 / 24 / 24`；当 GPU2 上的 TEI `/embed` 压力把 `utilization.gpu` 打到 `100%` 后，同样 `120` 个请求变成 `28 / 16 / 26 / 24 / 26`，说明当前路由逻辑会主动避让被 embedding 打热的 GPU2，但整体吞吐也会从 `28.4 req/s` 下降到 `23.0 req/s`

## 调度稳定性

- `qwn machine` 的健康检查与模型发现现在使用更短的单请求超时，避免单个后端卡死或 DNS/网络抖动时，把整轮健康刷新拖成慢探测
- 这不会改变正常实例的路由策略，但会让坏实例更快被判定为不可用，也让恢复探测更稳定

## 突发实例故障与 failover

- `qwn machine` 现在在请求转发层支持受控 failover：若实例在健康检查间隙里突然断连、超时，或在响应开始前返回 `5xx`，代理会立刻把它标记为不健康，并尝试切到其他健康实例
- 若流式响应已经开始输出，就不会再切到另一台实例，避免把两台实例的 token 流混在一起；此时会直接把错误回给客户端
- `qwn client info` 的 `stats` 里现在会包含 `total_failovers`，用于观察运行期到底发生了多少次实例级切换

## benchmark 口径

- `qwn benchmark run` 默认会额外统计 `ttft_sec`，同时区分 `submitted`、`successful`、`failed` 和 `success_rate`
- `throughput.requests_per_second` 现在代表成功请求吞吐，不再把失败请求静默算进“有效吞吐”里
- 如果只想做老式总吞吐压测，可以加 `--no-ttft`，跳过流式 TTFT 采样

## 常见问题

### 没检测到 GPU

- 先确认 `nvidia-smi`
- 单独用 NVIDIA CUDA 基础镜像验证 Docker GPU runtime
- 如果 runtime 模式异常，可尝试 `qwn compose up --mount-mode manual`

### `qwn machine` 起不来

- 先看 `qwn machine status`
- 再看 `qwn machine logs`
- 只有在确认进程已死的前提下，才手动删除：
  - `~/.cache/tfmx/qwn_machine.pid`
  - `~/.cache/tfmx/qwn_machine.log`

### `/health` 显示 0 个 healthy 实例

- 先确认容器是否仍在加载模型
- 查看 `qwn compose logs -f`
- 直接访问后端：

```bash
curl http://localhost:27880/health
curl http://localhost:27880/v1/models
```

### 自动发现不到容器

自动发现优先匹配容器名前缀为 `qwn` 的部署。如果你用了自定义项目名，可以显式指定：

```bash
qwn machine run -n my-custom-project
qwn machine run -e http://localhost:27880,http://localhost:27881
```