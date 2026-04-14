# TEI 排障与经验说明

## 推荐流程

建议按下面顺序使用：

1. `tei compose health`
2. `tei compose setup -m "$MODEL"`（首次或换模型时）
3. `tei machine run --auto-start --perf-track --on-conflict replace`
4. `tei client health --port 28800`
5. `tei client info --port 28800`
6. `tei benchmark tune -E "$TEI_MACHINE_URL"`

这套流程的目的不是“最少命令”，而是先确认 GPU 健康和模型缓存，再把 machine 拉起来，最后才进入压测或批量生产请求。

## `tei compose setup` 什么时候需要

- 首次使用某个 embedding 模型前，建议跑一次 `tei compose setup -m "$MODEL"`
- 如果你已经确认模型缓存中相关 `sentence_*_config.json` 文件都齐了，这一步可以跳过
- 这一步不是每次都要跑，通常一个模型跑一次就够

## GPU 掉卡 / `Unknown Error`

常见现象：

- `nvidia-smi -L` 报 `Unable to determine the device handle ... Unknown Error`
- `tei compose health` 只剩部分 GPU 健康
- `tei machine` 启动后只显示部分 healthy instances

建议排查命令：

```bash
nvidia-smi -L
tei compose health
tei machine discover
curl http://127.0.0.1:28800/health
curl http://127.0.0.1:28800/info
```

说明：

- `tei machine` 现在会跳过坏掉的后端，先使用剩余健康实例对外服务
- 但它不能替代宿主机 GPU/驱动层恢复；如果 `nvidia-smi` 本身已经报错，真正的问题不在 machine 层
- 如果 `nvidia-container-runtime` 因 NVML 异常而起不来，可以尝试 `tei compose up --mount-mode manual`

## `28800` 被旧进程占住

如果你看到端口冲突，不必先手动找 PID。直接用：

```bash
tei machine run --auto-start --on-conflict replace
```

补充说明：

- `report`：发现冲突就报错退出
- `replace`：先清掉旧监听器，再启动新的 machine
- `tei machine` 现在支持内建 daemon 管理，可直接用 `tei machine run --background --auto-start --on-conflict replace`
- daemon 的 PID / 日志默认在 `~/.cache/tfmx/tei_machine.pid` 与 `~/.cache/tfmx/tei_machine.log`
- 如果你要做开机自启，再把这条命令放进 `systemd`、Supervisor 或你现有的运维系统里

## OOM / 容量不足时怎么处理

当前 `tei machine` 在运行期遇到后端返回 OOM 或 capacity 错误时，会先自动递归拆小 batch 再重试。这意味着：

- 单次大请求不再轻易把所有健康后端都一起判死
- 第一次撞到容量上限时，日志会记录该实例当前学到的安全批量上限；后续请求会优先按这个上限预切分，不再每次都重复撞同样的过载点
- 如果拆到很小仍然失败，才说明问题已经不是简单的临时过载

如果仍然经常 OOM，建议按这个顺序收紧：

1. 降低外部请求批量
2. 降低 `tei machine -b/--batch-size`
3. 降低 `tei machine -m/--micro-batch-size`
4. 对 benchmark 先跑 `tune`，而不是直接用大 batch 长压

示例：

```bash
tei machine run -b 200 -m 50 --perf-track --on-conflict replace
tei benchmark tune -E "$TEI_MACHINE_URL" --min-batch 200 --max-batch 1500 --step 100
```

## Compose 和 machine 的职责边界

- `tei compose` 负责“起后端容器”
- `tei machine` 负责“把多个后端聚合起来并做负载均衡”
- `tei client -E` / `TEIClients` 负责“跨多个 machine 做更高一层的分发”

如果你想精确控制 GPU 和模型，优先手动 `tei compose up ...`；如果你只是想快速恢复一个能用的单机 embedding 服务，优先 `tei machine run --auto-start ...`。

## 什么时候应该直连后端

适合直连后端的场景：

- 只想验证某个 GPU 上单个容器是否活着
- 只想看 `/embed` 是否正常返回
- 排查 machine 聚合层之前，先缩小问题范围

可直接使用：

```bash
tei client -e "$TEI_BACKEND_A_URL" health
tei client -e "$TEI_BACKEND_A_URL" embed "Direct backend request"
```

而 `info`、`lsh`、`rerank` 这类聚合能力，应该优先打到 `tei machine`。

## benchmark 调优建议

- 刚开始不要直接用特别大的 batch，先跑 `tei benchmark tune`
- `-v/--verbose` 有助于看出到底是哪台 machine 或哪张卡拖慢了整体吞吐
- 如果当前 GPU 环境不稳定，优先先确认 `health` 和 `info`，再开始长时间压测

## 当前代码路径已经验证过的运行行为

- 旧 `28800` listener 可以通过 `--on-conflict replace` 直接替换
- 当部分 GPU 掉卡时，`tei machine` 仍可在剩余健康实例上启动
- 大批量 `/lsh` 请求在当前实现下会优先拆小 batch，而不是直接因为一次 OOM 把所有实例打成 unhealthy