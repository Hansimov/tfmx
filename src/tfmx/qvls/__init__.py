"""tfmx.qvls - Qwen3-VL vision-language model serving module.

Deploys and manages Qwen3-VL models via vLLM in Docker containers,
with multi-GPU load balancing and multi-machine distribution.
"""

from .compose import QVLComposer, QVLComposeArgParser
from .compose import GPUInfo, GPUDetector
from .compose import ModelConfigManager, DockerImageManager, ComposeFileGenerator
from .compose import SUPPORTED_MODELS, GGUF_MODELS
from .client import QVLClient, AsyncQVLClient, QVLClientArgParser
from .client import HealthResponse, ModelInfo, ChatMessage, ChatResponse, ChatUsage
from .clients_core import (
    MachineState,
    MachineScheduler,
    ClientsHealthResponse,
)
from .clients import QVLClients
from .clients_stats import QVLClientsWithStats
from .clients_cli import QVLClientsArgParserBase, QVLClientsCLIBase
from .machine import QVLMachineServer, VLLMInstance, VLLMInstanceDiscovery
from .performance import (
    ExplorationConfig,
    PerformanceTracker,
    PerformanceMetrics,
)
from .scheduler import (
    WorkerState,
    IdleFillingScheduler,
    distribute_with_adaptive_pipeline,
)
from .perf_tracker import (
    PerfTracker,
    WorkerEvent,
    TaskRecord,
    RoundRecord,
    WorkerStats,
    RoundContext,
    TaskContext,
    get_global_tracker,
    reset_global_tracker,
)
from .benchmark import QVLBenchmark, BenchmarkMetrics
from .benchimgs import QVLBenchImageGenerator
