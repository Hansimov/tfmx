# TEI Machine 调度器架构文档

## 概述

TEI Machine 使用多层调度架构来实现高效的 GPU 负载均衡和任务分配。本文档详细说明了调度系统的核心组件、使用路径和架构设计。

---

## 核心组件

### 1. `IdleFillingScheduler` (核心调度器类)

**位置**: `src/tfmx/tei_scheduler.py`

**作用**: 泛型调度器类，跟踪 workers（TEI 实例）的状态和性能指标

**主要功能**:
- 维护每个 worker 的状态（busy/idle）
- 记录和跟踪吞吐量历史（throughput tracking）
- 选择空闲 worker 进行任务分配
- 支持按吞吐量排序的 worker 选择策略

**关键属性**:
```python
class IdleFillingScheduler(Generic[W]):
    workers: list[W]                    # Worker 列表
    states: dict[str, WorkerState]      # Worker 状态映射
    max_batch_size: int                 # 最大批次大小
```

**核心方法**:
- `get_idle_workers()`: 获取所有空闲 workers
- `get_idle_workers_by_throughput()`: 按吞吐量排序获取空闲 workers
- `select_idle_worker()`: 选择一个空闲 worker
- `wait_for_idle_worker()`: 等待直到有 worker 空闲
- `update_workers()`: 更新 worker 列表
- `get_stats_summary()`: 获取统计摘要

---

### 2. `distribute_with_adaptive_pipeline` (自适应流水线调度算法)

**位置**: `src/tfmx/tei_scheduler.py`

**作用**: 当前使用的核心调度算法，针对异构 GPU 环境优化

**调度策略**:

#### Phase 1: 探测阶段 (Probing)
- 给每个 worker 分配小批次（probe_batch_size，默认 100）
- 快速测量每个 worker 的实际吞吐量
- 记录吞吐量数据用于后续分配

#### Phase 2: 自适应分配阶段 (Adaptive Distribution)
- 根据测量的吞吐量比例动态分配批次大小
- 快速 GPU 获得更大的批次
- 慢速 GPU 获得更小的批次
- 跨请求保留吞吐量历史，避免重复探测

**关键参数**:
```python
min_batch_size: int = 50      # 最小批次大小
max_batch_size: int = 500     # 最大批次大小（从 MAX_CLIENT_BATCH_SIZE）
probe_batch_size: int = 100   # 探测批次大小
```

**核心特性**:
- ✅ 消除轮次同步屏障（Round Barrier）
- ✅ 快速 worker 不等待慢速 worker
- ✅ 最大化异构 GPU 利用率
- ✅ 自适应批次大小调整
- ✅ 跨请求的吞吐量历史记录

---

## 架构层次

### 多层负载均衡架构

```
┌─────────────────────────────────────────────────────────────┐
│ User Request                                                  │
└────────────────────────┬────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────┐
│ tei_clients (多机器负载均衡)                                  │
│ - MachineScheduler (独立实现)                                │
│ - _TEIClientsPipeline                                         │
└────────────┬────────────────────────┬───────────────────────┘
             │                        │
             ▼                        ▼
┌────────────────────────┐  ┌────────────────────────┐
│ tei_machine_1          │  │ tei_machine_2          │
│ (单机器多GPU负载均衡)   │  │ (单机器多GPU负载均衡)   │
│                        │  │                        │
│ ┌────────────────────┐ │  │ ┌────────────────────┐ │
│ │ IdleFillingScheduler│ │  │ │ IdleFillingScheduler│ │
│ │        +            │ │  │ │        +            │ │
│ │ adaptive_pipeline   │ │  │ │ adaptive_pipeline   │ │
│ └────────┬───────────┘ │  │ └────────┬───────────┘ │
│          │             │  │          │             │
│    ┌─────┴─────┐       │  │    ┌─────┴─────┐       │
│    ▼           ▼       │  │    ▼           ▼       │
│ ┌──────┐  ┌──────┐    │  │ ┌──────┐  ┌──────┐    │
│ │GPU 0 │  │GPU 1 │    │  │ │GPU 0 │  │GPU 1 │    │
│ │TEI   │  │TEI   │    │  │ │TEI   │  │TEI   │    │
│ └──────┘  └──────┘    │  │ └──────┘  └──────┘    │
└────────────────────────┘  └────────────────────────┘
```

### 组件职责分层

| 层级 | 组件 | 职责 | 调度策略 |
|------|------|------|----------|
| L3 | `tei_clients` | 多机器间负载均衡 | 自定义 MachineScheduler + Pipeline |
| L2 | `tei_machine` | 单机器多 GPU 负载均衡 | IdleFillingScheduler + adaptive_pipeline |
| L1 | `TEI Instance` | 单 GPU 推理服务 | N/A (Docker 容器) |

---

## 主要使用路径

### 在 tei_machine 中的调用流程

```python
# 1. 初始化阶段
TEIMachineServer.__init__()
    └─ self.scheduler = IdleFillingScheduler(
           workers=instances,
           get_worker_id=lambda inst: inst.container_name,
           max_batch_size=batch_size,
       )

# 2. 请求处理阶段
TEIMachineServer.embed(request)  # 或 lsh(request)
    │
    ├─ 获取健康的实例列表
    │   └─ healthy = self.get_healthy_instances()
    │
    ├─ 使用锁序列化 GPU 访问
    │   └─ async with self._scheduler_lock:
    │
    └─ 调用分发函数
        └─ _distribute_with_scheduler(inputs, healthy, normalize, truncate)
            │
            ├─ 更新调度器的 worker 列表
            │   └─ self.scheduler.update_workers(instances)
            │
            ├─ 定义处理函数
            │   └─ async def process_on_instance(instance, chunk):
            │          └─ await self._send_embed_request(...)
            │
            └─ 调用自适应流水线算法
                └─ await distribute_with_adaptive_pipeline(
                       scheduler=self.scheduler,
                       inputs=inputs,
                       process_func=process_on_instance,
                       enable_perf_tracking=self.enable_perf_tracking,
                       perf_tracker=self.perf_tracker,
                       min_batch_size=MIN_BATCH_SIZE,
                       max_batch_size=MAX_BATCH_SIZE,
                       probe_batch_size=self.micro_batch_size,
                   )
```

---

## 详细调用链

### embed 端点完整调用链

```
HTTP POST /embed
    ↓
FastAPI Route Handler: TEIMachineServer.embed(request: EmbedRequest)
    ↓
1. 参数标准化
    inputs = [request.inputs] if isinstance(request.inputs, str) else request.inputs
    ↓
2. 获取健康实例
    healthy = self.get_healthy_instances()
    ↓
3. 获取调度锁（序列化 GPU 访问）
    async with self._scheduler_lock:
        ↓
    4. 调用内部分发方法
        embeddings = await self._distribute_with_scheduler(
            inputs, healthy, request.normalize, request.truncate
        )
        ↓
        5. 更新调度器 workers
            self.scheduler.update_workers(instances)
            ↓
        6. 定义异步处理函数
            async def process_on_instance(instance, chunk):
                └─ await self._send_embed_request(instance, chunk, normalize, truncate)
                    └─ async with self._client.post(...) as resp:
                        └─ return resp.json()
            ↓
        7. 调用自适应流水线调度
            embeddings, details = await distribute_with_adaptive_pipeline(...)
            ↓
            ┌─────────────────────────────────────────────────┐
            │ distribute_with_adaptive_pipeline 内部流程：     │
            │                                                  │
            │ 1. 计算吞吐量比例                                │
            │    └─ get_throughput_ratios()                   │
            │                                                  │
            │ 2. Phase 1: 探测阶段                            │
            │    └─ 给每个 worker 分配 probe_batch_size       │
            │                                                  │
            │ 3. Phase 2: 自适应分配                          │
            │    └─ 根据吞吐量比例动态分配批次                 │
            │                                                  │
            │ 4. 异步任务调度循环                              │
            │    while 还有输入 or 有待处理任务:               │
            │        ├─ 等待任务完成                           │
            │        ├─ 更新 worker 状态和吞吐量               │
            │        ├─ 选择空闲 worker                        │
            │        └─ 分配新批次                             │
            │                                                  │
            │ 5. 返回排序后的结果                              │
            │    └─ sort by start_idx                          │
            └─────────────────────────────────────────────────┘
            ↓
        8. 更新统计信息
            self.stats.requests_per_instance[instance_name] += 1
            ↓
        9. 返回 embeddings
    ↓
10. 返回 HTTP Response
```

---

## 调度器状态管理

### WorkerState 生命周期

```python
@dataclass
class WorkerState:
    busy: bool = False                    # 是否正在处理任务
    total_items: int = 0                  # 已处理的总项目数
    total_time: float = 0.0               # 总处理时间（秒）
    last_batch_size: int = 0              # 最后一个批次的大小
    last_latency: float = 0.0             # 最后一个批次的延迟
    
    @property
    def throughput(self) -> float:
        """计算吞吐量（items/second）"""
        return self.total_items / self.total_time if self.total_time > 0 else 0
```

### 状态转换流程

```
[IDLE] 空闲状态
    ↓ select_idle_worker()
[BUSY] 处理中
    ↓ mark_busy()
[PROCESSING] 实际处理
    ↓ process_func(worker, chunk)
[COMPLETE] 完成
    ↓ 更新吞吐量统计
    ↓ mark_idle()
[IDLE] 返回空闲状态
```

---

## 性能优化特性

### 1. 跨请求吞吐量记忆

```python
# WorkerState 保持在 IdleFillingScheduler 中
# 每次请求使用历史数据进行初始估计
def get_throughput_ratios() -> dict[str, float]:
    for w in scheduler.workers:
        state = scheduler.states.get(wid)
        if state and state.throughput > 0:
            throughputs[wid] = state.throughput  # 使用历史数据
```

**优势**:
- 第一次请求：探测阶段获取吞吐量
- 后续请求：直接使用历史吞吐量，减少探测开销
- 动态更新：每次请求后更新吞吐量数据

### 2. 自适应批次大小

```python
# 根据吞吐量比例动态计算批次大小
batch_size = max(
    min_batch_size,
    min(
        int(remaining_items * ratio),
        max_batch_size
    )
)
```

**效果**:
- 快速 GPU (高吞吐量): 获得接近 max_batch_size 的大批次
- 慢速 GPU (低吞吐量): 获得接近 min_batch_size 的小批次
- 避免慢速 GPU 成为瓶颈

### 3. 消除轮次同步屏障

**传统轮次调度的问题** (已废弃):
```python
# Round 1: 所有 workers 开始
[GPU0: 快速] ████████░░ (完成)
[GPU1: 慢速] ████████████████ (处理中...)
              ↑
              └─ GPU0 等待 GPU1 完成后才能开始 Round 2
```

**流水线调度的优势** (当前实现):
```python
# 无轮次概念，worker 空闲即可获取新任务
[GPU0: 快速] ████░░░░████░░░░████ (处理 3 个批次)
[GPU1: 慢速] ████████████████████ (处理 1 个批次)
              ↑
              └─ GPU0 不等待，持续处理新批次
```

---

## tei_client 和 tei_clients 的使用情况

### tei_client.py
- **不使用** tei_scheduler 中的任何调度函数
- 职责：简单的 HTTP 客户端
- 功能：向 tei_machine 或 TEI 容器发送请求
- 实现：使用 httpx 进行同步/异步 HTTP 调用

### tei_clients.py 和 tei_clients_core.py
- **不使用** tei_scheduler 中的调度函数
- 职责：多个 tei_machine 之间的负载均衡
- 使用独立的调度系统：
  - `MachineScheduler`: 不同于 `IdleFillingScheduler` 的独立实现
  - `_TEIClientsPipeline`: 自定义流水线实现
  - `IteratorBuffer`: 线程安全的迭代器缓冲
  - `MachineState`: 机器状态跟踪

**架构隔离**:
```
tei_scheduler.IdleFillingScheduler
    └─ 用于 GPU 级别调度（tei_machine 内部）

tei_clients_core.MachineScheduler
    └─ 用于机器级别调度（多个 tei_machine 之间）
```

---

## 配置参数

### TEIMachineServer 初始化参数

```python
TEIMachineServer(
    instances: list[TEIInstance],        # TEI 实例列表
    port: int = 28800,                   # 服务端口
    batch_size: int = 500,               # 最大批次大小 (MAX_CLIENT_BATCH_SIZE)
    micro_batch_size: int = 100,         # 探测批次大小
    timeout: float = 60.0,               # 请求超时（秒）
    use_gpu_lsh: bool = True,            # LSH 是否使用 GPU
    enable_perf_tracking: bool = False,  # 性能跟踪
)
```

### distribute_with_adaptive_pipeline 参数

```python
distribute_with_adaptive_pipeline(
    scheduler: IdleFillingScheduler,     # 调度器实例
    inputs: list[str],                   # 输入文本列表
    process_func: Callable,              # 异步处理函数
    enable_perf_tracking: bool = False,  # 性能跟踪
    perf_tracker: Optional[PerfTracker] = None,  # 性能跟踪器
    min_batch_size: int = 50,            # 最小批次大小
    max_batch_size: int = 500,           # 最大批次大小
    probe_batch_size: int = 100,         # 探测批次大小
)
```

---

## 命令行使用

### 启动 tei_machine 服务器

```bash
# 默认配置（自动发现 TEI 容器，使用自适应流水线）
tei_machine run

# 指定端口
tei_machine run -p 28800

# 启用性能跟踪
tei_machine run --perf-track

# 自定义批次大小
tei_machine run -b 500 -m 100

# 发现可用的 TEI 实例
tei_machine discover

# 检查所有实例的健康状态
tei_machine health
```

### 性能跟踪输出示例

```
[tei_machine] Performance tracking ENABLED
[tei_machine] Pipeline mode ENABLED (micro_batch_size=100)
[Pipeline] total=10000, micro_batches=100, micro_batch_size=100, workers=2
[Batch embed] n=500, time=234.5ms
```

---

## 最佳实践

### 1. 批次大小配置

- `min_batch_size`: 设置为较小值（50）以适应慢速 GPU
- `max_batch_size`: 设置为 TEI 容器的 `--max-client-batch-size` 值（500）
- `probe_batch_size`: 设置为中等值（100）用于快速探测

### 2. 性能跟踪

启用性能跟踪用于调试和优化：
```bash
tei_machine run --perf-track
```

查看详细的：
- 每个 worker 的吞吐量
- 批次分配策略
- 任务延迟统计

### 3. 异构 GPU 环境

在混合 GPU 环境中（如 RTX 3090 + RTX 4090）：
- 自适应流水线会自动检测性能差异
- 快速 GPU 自动获得更多工作负载
- 无需手动配置，系统自动优化

---

## 技术细节

### 并发控制

```python
# tei_machine 使用锁序列化 GPU 访问
async with self._scheduler_lock:
    embeddings = await self._distribute_with_scheduler(...)
```

**原因**: 多个并发请求竞争相同的 GPU 会导致严重的性能下降

### 异步任务管理

```python
# 使用 asyncio.Task 和 set 管理待处理任务
pending_tasks: set[asyncio.Task] = set()

# 主调度循环
while batch_index < len(micro_batches) or pending_tasks:
    if pending_tasks:
        done, pending_tasks = await asyncio.wait(
            pending_tasks,
            return_when=asyncio.FIRST_COMPLETED
        )
```

### 结果顺序保证

```python
# 使用 dict 按 start_idx 存储结果
results_map: dict[int, DistributionResult] = {}

# 最后按顺序组合
all_results = [results_map[k] for k in sorted(results_map.keys())]
```

---

## 历史演进

### 第 1 代: 轮次调度 (已废弃)
- 函数: `distribute_with_scheduler`
- 特点: 使用 `distribute_to_workers` 进行静态分配
- 问题: **轮次同步屏障导致 GPU 利用率低**
- 状态: 已从代码库中移除

### 第 2 代: 固定流水线 (已废弃)
- 函数: `distribute_with_pipeline`
- 特点: 消除轮次屏障，使用固定 `micro_batch_size`
- 改进: 快速 GPU 不再等待慢速 GPU
- 问题: 批次大小固定，不适应异构 GPU
- 状态: 已从代码库中移除

### 第 3 代: 自适应流水线 (当前使用) ✅
- 函数: `distribute_with_adaptive_pipeline`
- 特点: 动态测量吞吐量，自适应批次大小
- 优势: 
  - ✅ 异构 GPU 各得其所
  - ✅ 跨请求吞吐量记忆
  - ✅ 最优性能
- 状态: **生产环境使用**

---

## 相关文件

- `src/tfmx/tei_scheduler.py`: 调度器核心实现
- `src/tfmx/tei_machine.py`: TEI Machine 服务器
- `src/tfmx/tei_clients_core.py`: 多机器客户端调度器
- `src/tfmx/perf_tracker.py`: 性能跟踪工具
- `src/tfmx/__init__.py`: 包导出

---

## 总结

TEI Machine 的调度系统经过三代演进，当前使用的 **自适应流水线调度算法** (`distribute_with_adaptive_pipeline`) 结合 **IdleFillingScheduler** 实现了：

1. ✅ **高 GPU 利用率**: 消除轮次同步屏障
2. ✅ **异构 GPU 优化**: 自适应批次大小分配
3. ✅ **智能负载均衡**: 基于历史吞吐量的决策
4. ✅ **简洁的 API**: 单一入口点，自动优化

这个架构适用于生产环境中的多 GPU 异构集群，提供了最优的性能和灵活性。
