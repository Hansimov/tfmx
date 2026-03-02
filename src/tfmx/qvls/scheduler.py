"""QVL Scheduler - Re-exports from teis.scheduler

The scheduler is generic (uses TypeVar) and works for any worker type.
QVL uses the same scheduling algorithm with different default parameters.
"""

from ..teis.scheduler import (
    WorkerState,
    IdleFillingScheduler,
    DistributionResult,
    distribute_with_adaptive_pipeline,
)

from .compose import MAX_CONCURRENT_REQUESTS

# Re-export with QVL defaults
__all__ = [
    "WorkerState",
    "IdleFillingScheduler",
    "DistributionResult",
    "distribute_with_adaptive_pipeline",
    "MAX_CONCURRENT_REQUESTS",
]
