# Utils
from .utils.vectors import floats_to_bits, bits_to_hash, bits_dist, hash_dist
from .utils.vectors import bits_sim, hash_sim, dot_sim
from .utils.lsh import LSHConverter

# TEI
from .teis.compose import TEIComposer, TEIComposeArgParser
from .teis.compose import GPUInfo, GPUDetector
from .teis.compose import ModelConfigManager, DockerImageManager, ComposeFileGenerator
from .teis.client import TEIClient, AsyncTEIClient, TEIClientArgParser
from .teis.client import HealthResponse, InfoResponse, InstanceInfo, MachineStats
from .teis.clients_core import (
    MachineState,
    MachineScheduler,
    ClientsHealthResponse,
    IteratorBuffer,
)
from .teis.clients import TEIClients
from .teis.clients_stats import TEIClientsWithStats
from .teis.clients_cli import TEIClientsArgParserBase, TEIClientsCLIBase
from .teis.performance import (
    ExplorationConfig,
    PerformanceTracker,
    PerformanceMetrics,
    ExplorationState,
)
from .teis.scheduler import (
    WorkerState,
    IdleFillingScheduler,
    DistributionResult,
    distribute_with_adaptive_pipeline,
    MAX_CLIENT_BATCH_SIZE,
)
from .teis.perf_tracker import (
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
from .teis.machine import TEIMachineServer, TEIInstance, TEIMachineDaemon

# QWN
from .qwns import QWNBenchmark, BenchmarkMetrics as QWNBenchmarkMetrics
from .qwns import QWNClient, AsyncQWNClient, build_text_messages
from .qwns import QWNClients, QWNClientsWithStats
from .qwns import QWNComposer, QWNRouter, InstanceDescriptor as QWNInstanceDescriptor
from .qwns import QWNMachineServer, QWNInstance, QWNMachineDaemon

# QSR
from .qsrs import QSRBenchmark, ASRBenchmarkMetrics as QSRBenchmarkMetrics
from .qsrs import QSRClient, AsyncQSRClient, build_audio_messages
from .qsrs import QSRClients, QSRClientsWithStats
from .qsrs import QSRComposer, QSRRouter, InstanceDescriptor as QSRInstanceDescriptor
from .qsrs import QSRMachineServer, QSRInstance, QSRMachineDaemon

# GPU
from .gpus.fan import NvidiaSettingsParser, GPUFanController, GPUFanArgParser
from .gpus.pow import NvidiaSmiParser, GPUPowerController, GPUPowerArgParser
