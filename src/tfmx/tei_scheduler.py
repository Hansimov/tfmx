"""TEI Adaptive Pipeline Scheduler

This module provides an adaptive pipeline scheduler for TEI instances that
optimizes load balancing across heterogeneous GPUs.

Algorithm:
1. Each worker (TEI instance) processes batches asynchronously
2. Worker throughput is measured and tracked across requests
3. Batch sizes are dynamically adjusted based on worker performance
4. Fast workers get larger batches, slow workers get smaller batches
5. No round barriers - workers get new work immediately when idle

Key components:
- WorkerState: Tracks worker busy/idle state and throughput metrics
- IdleFillingScheduler: Manages worker state and idle worker selection
- distribute_with_adaptive_pipeline: Adaptive pipeline scheduling algorithm
- MAX_BATCH_SIZE: Maximum batch size from tei_compose (--max-client-batch-size)

Performance Features:
- Eliminates round barrier synchronization bottlenecks
- Cross-request throughput memory for faster convergence
- Adaptive batch sizing for heterogeneous GPU clusters
- Optional PerfTracker integration for detailed metrics
"""

import asyncio
import time

from dataclasses import dataclass, field
from typing import TypeVar, Generic, Callable, Optional, Any
from tclogger import logger, logstr

from .tei_compose import MAX_CLIENT_BATCH_SIZE
from .perf_tracker import PerfTracker, get_global_tracker


@dataclass
class WorkerState:
    """State tracking for a worker with adaptive performance metrics.

    Tracks:
    - busy/idle status
    - basic statistics (requests, items, errors)
    - throughput estimation for adaptive scheduling
    """

    worker_id: str

    # Busy/idle state
    busy: bool = False

    # Statistics
    total_requests: int = 0
    total_items: int = 0
    total_errors: int = 0
    total_latency: float = 0.0

    # Adaptive scheduling: throughput tracking (items per second)
    # Using EMA (Exponential Moving Average) for smooth estimation
    _throughput_ema: float = 0.0  # items/second
    _ema_alpha: float = 0.3  # EMA smoothing factor (higher = more responsive)
    _last_batch_size: int = 0
    _last_latency: float = 0.0

    @property
    def is_idle(self) -> bool:
        """Check if worker is idle."""
        return not self.busy

    @property
    def throughput(self) -> float:
        """Get estimated throughput in items/second."""
        return self._throughput_ema

    def mark_busy(self) -> None:
        """Mark worker as busy."""
        self.busy = True

    def mark_idle(self) -> None:
        """Mark worker as idle."""
        self.busy = False

    def record_success(self, latency: float, n_items: int) -> None:
        """Record a successful request and update throughput estimate."""
        self.total_requests += 1
        self.total_items += n_items
        self.total_latency += latency
        self._last_batch_size = n_items
        self._last_latency = latency

        # Update throughput EMA
        if latency > 0:
            current_throughput = n_items / latency  # items/second
            if self._throughput_ema == 0:
                # First measurement
                self._throughput_ema = current_throughput
            else:
                # EMA update
                self._throughput_ema = (
                    self._ema_alpha * current_throughput
                    + (1 - self._ema_alpha) * self._throughput_ema
                )

        self.mark_idle()

    def record_error(self) -> None:
        """Record a failed request."""
        self.total_requests += 1
        self.total_errors += 1
        self.mark_idle()

    def to_dict(self) -> dict:
        """Convert to dictionary for serialization."""
        avg_latency = (
            self.total_latency / self.total_requests if self.total_requests > 0 else 0
        )
        return {
            "worker_id": self.worker_id,
            "busy": self.busy,
            "total_requests": self.total_requests,
            "total_items": self.total_items,
            "total_errors": self.total_errors,
            "avg_latency_ms": avg_latency * 1000,
            "throughput": self._throughput_ema,
        }


W = TypeVar("W")  # Worker type (e.g., TEIInstance, TEIClient)


class IdleFillingScheduler(Generic[W]):
    """Simple idle-filling scheduler for distributing work across workers.

    Algorithm:
    1. Split incoming batch into chunks of MAX_BATCH_SIZE
    2. Assign each chunk to an idle worker
    3. Wait for workers to become available if all are busy
    4. Workers process one batch at a time (simple, predictable)

    This approach ensures:
    - All workers are utilized (no idle GPUs)
    - Simple and predictable behavior
    - No complex metrics or estimations needed
    """

    def __init__(
        self,
        workers: list[W],
        get_worker_id: Callable[[W], str],
        max_batch_size: int = MAX_CLIENT_BATCH_SIZE,
    ):
        """Initialize the scheduler.

        Args:
            workers: List of worker objects
            get_worker_id: Function to extract worker ID from worker object
            max_batch_size: Max batch size per worker (default: MAX_CLIENT_BATCH_SIZE)
        """
        self.workers = workers
        self.get_worker_id = get_worker_id
        self.max_batch_size = max_batch_size

        # Create state for each worker
        self.states: dict[str, WorkerState] = {}
        for w in workers:
            wid = get_worker_id(w)
            self.states[wid] = WorkerState(worker_id=wid)

        # Mapping from worker_id to worker object
        self._worker_map: dict[str, W] = {get_worker_id(w): w for w in workers}

        # Event to signal when a worker becomes idle
        self._idle_event = asyncio.Event()
        self._idle_event.set()  # Initially, all workers are idle

    def update_workers(self, workers: list[W]) -> None:
        """Update the worker list (e.g., after health check)."""
        self.workers = workers
        self._worker_map = {self.get_worker_id(w): w for w in workers}

        # Add state for new workers
        for w in workers:
            wid = self.get_worker_id(w)
            if wid not in self.states:
                self.states[wid] = WorkerState(worker_id=wid)

    def get_idle_workers(self) -> list[tuple[W, WorkerState]]:
        """Get list of idle workers with their states."""
        idle = []
        for w in self.workers:
            wid = self.get_worker_id(w)
            state = self.states.get(wid)
            if state and state.is_idle:
                idle.append((w, state))
        return idle

    def get_idle_workers_by_throughput(self) -> list[tuple[W, WorkerState]]:
        """Get list of idle workers sorted by throughput (highest first).

        This prioritizes high-throughput workers for task assignment,
        ensuring they get more work opportunities.
        """
        idle = self.get_idle_workers()
        # Sort by throughput descending (highest first)
        # Workers with no throughput data (0) go to the end
        idle.sort(key=lambda x: x[1].throughput, reverse=True)
        return idle

    def get_worker_by_id(self, worker_id: str) -> Optional[W]:
        """Get worker object by ID."""
        return self._worker_map.get(worker_id)

    def get_state(self, worker_id: str) -> Optional[WorkerState]:
        """Get worker state by ID."""
        return self.states.get(worker_id)

    def _signal_worker_idle(self) -> None:
        """Signal that a worker has become idle."""
        self._idle_event.set()

    async def wait_for_idle_worker(self, timeout: float = 60.0) -> bool:
        """Wait for a worker to become idle.

        Args:
            timeout: Maximum time to wait in seconds

        Returns:
            True if an idle worker is available, False if timeout
        """
        try:
            await asyncio.wait_for(self._idle_event.wait(), timeout=timeout)
            return True
        except asyncio.TimeoutError:
            return False

    def select_idle_worker(self) -> Optional[tuple[W, WorkerState]]:
        """Select an idle worker.

        Returns:
            Tuple of (worker, state) or None if no workers are idle
        """
        idle_workers = self.get_idle_workers()
        if not idle_workers:
            self._idle_event.clear()  # No idle workers, clear event
            return None

        # Return first idle worker (simple round-robin effect due to list order)
        return idle_workers[0]

    def get_stats_summary(self) -> dict:
        """Get summary of all worker stats."""
        return {wid: s.to_dict() for wid, s in self.states.items()}


@dataclass
class DistributionResult:
    """Result of distributing work to a worker."""

    worker_id: str
    start_idx: int
    end_idx: int
    result: Any = None
    error: Optional[Exception] = None
    latency: float = 0.0

    @property
    def success(self) -> bool:
        return self.error is None


async def distribute_with_adaptive_pipeline(
    scheduler: IdleFillingScheduler[W],
    inputs: list[str],
    process_func: Callable[[W, list[str]], Any],
    enable_perf_tracking: bool = False,
    perf_tracker: Optional[PerfTracker] = None,
    min_batch_size: int = 50,
    max_batch_size: int = 500,
    probe_batch_size: int = 100,
) -> tuple[list[Any], list[DistributionResult]]:
    """Distribute inputs using adaptive pipeline scheduling.

    This scheduler dynamically adjusts batch sizes based on worker performance:
    1. Uses small probe batches to quickly measure worker throughput
    2. Allocates remaining work proportionally to throughput
    3. Fast workers get larger batches, slow workers get smaller batches

    Key Strategy:
    - Phase 1 (Probe): Give each worker a small batch to measure throughput
    - Phase 2 (Adaptive): Distribute remaining work based on throughput ratios
    - Uses historical throughput data across requests for better initial estimates

    Args:
        scheduler: The idle-filling scheduler instance
        inputs: List of input texts to process
        process_func: Async function that processes inputs on a worker
        enable_perf_tracking: Enable detailed performance tracking
        perf_tracker: Custom PerfTracker instance
        min_batch_size: Minimum batch size (default: 50)
        max_batch_size: Maximum batch size (default: 500)
        probe_batch_size: Batch size for initial probing (default: 100)

    Returns:
        Tuple of (combined_results, distribution_details)
    """
    if not inputs:
        return [], []

    n_workers = len(scheduler.workers)
    if n_workers == 0:
        return [], []

    # Setup performance tracking
    tracker = None
    if enable_perf_tracking:
        tracker = perf_tracker or get_global_tracker()
        tracker.start_session(n_inputs=len(inputs), n_workers=n_workers)

    # Results storage
    results_map: dict[int, DistributionResult] = {}
    pending_tasks: set[asyncio.Task] = set()
    session_start = time.time()

    # Input queue tracking
    remaining_start = 0
    batch_counter = 0

    def get_throughput_ratios() -> dict[str, float]:
        """Get throughput ratio for each worker (sums to 1.0)."""
        throughputs = {}
        for w in scheduler.workers:
            wid = scheduler.get_worker_id(w)
            state = scheduler.states.get(wid)
            if state and state.throughput > 0:
                throughputs[wid] = state.throughput
            else:
                throughputs[wid] = 0

        total = sum(throughputs.values())
        if total > 0:
            return {wid: t / total for wid, t in throughputs.items()}
        else:
            # Equal distribution if no data
            return {wid: 1.0 / n_workers for wid in throughputs}

    async def process_batch(
        worker: W,
        state: WorkerState,
        chunk: list[str],
        start_idx: int,
        end_idx: int,
        batch_id: int,
    ) -> DistributionResult:
        """Process a batch on a worker."""
        task_start = time.time()

        try:
            result = await process_func(worker, chunk)
            latency = time.time() - task_start

            # Record success (updates throughput EMA)
            state.record_success(latency, len(chunk))
            scheduler._signal_worker_idle()

            if tracker:
                tracker.record_task(
                    worker_id=state.worker_id,
                    round_id=batch_id,
                    batch_size=len(chunk),
                    start_time=task_start,
                    end_time=time.time(),
                    queue_wait_time=0,
                    success=True,
                )

            return DistributionResult(
                worker_id=state.worker_id,
                start_idx=start_idx,
                end_idx=end_idx,
                result=result,
                latency=latency,
            )

        except Exception as e:
            latency = time.time() - task_start
            state.record_error()
            scheduler._signal_worker_idle()

            if tracker:
                tracker.record_task(
                    worker_id=state.worker_id,
                    round_id=batch_id,
                    batch_size=len(chunk),
                    start_time=task_start,
                    end_time=time.time(),
                    queue_wait_time=0,
                    success=False,
                    error=str(e),
                )

            return DistributionResult(
                worker_id=state.worker_id,
                start_idx=start_idx,
                end_idx=end_idx,
                error=e,
                latency=latency,
            )

    # ===== Adaptive Pipeline (No Barrier, Dynamic Batch Size) =====
    # Key insight: Don't use fixed probe_batch_size. Instead:
    # 1. Start with fair share based on total items and workers
    # 2. Adjust dynamically based on measured throughput
    # 3. Never wait for all workers - true pipeline

    def calculate_initial_batch_size(remaining: int, n_idle: int) -> int:
        """Calculate initial batch size when no throughput data available."""
        # Fair share among idle workers
        fair_share = remaining // max(n_idle, 1)
        # Use a reasonable portion (1/2) of fair share for initial probe
        # This leaves room for other workers while still being substantial
        initial = max(fair_share // 2, min_batch_size)
        return min(initial, max_batch_size, remaining)

    def calculate_adaptive_batch_size(
        state: WorkerState, remaining: int, n_idle: int
    ) -> int:
        """Calculate batch size based on worker's throughput.

        Strategy:
        - High-throughput workers get larger batches (proportional to their throughput ratio)
        - Use throughput-weighted fair share instead of simple equal division
        - This ensures fast GPUs process more data per batch, not just more batches
        """
        # Get throughput ratios (sums to 1.0)
        ratios = get_throughput_ratios()
        worker_ratio = ratios.get(state.worker_id, 1.0 / n_workers)

        # Calculate this worker's share based on throughput ratio
        # Higher throughput = larger share of remaining work
        target = int(remaining * worker_ratio)

        # For high-throughput workers, allow them to take more than simple fair share
        # But still leave some work for other idle workers
        # Use throughput-weighted fair share: if this worker has 20% throughput ratio
        # among 3 idle workers, it should get ~20%/(sum of idle ratios) of remaining
        idle_workers = scheduler.get_idle_workers()
        if idle_workers:
            # Sum of throughput ratios for currently idle workers
            idle_ratios_sum = sum(
                ratios.get(scheduler.get_worker_id(w), 1.0 / n_workers)
                for w, _ in idle_workers
            )
            if idle_ratios_sum > 0:
                # This worker's proportion among idle workers
                proportion_among_idle = worker_ratio / idle_ratios_sum
                # Allow up to this proportion of remaining work
                max_for_this_worker = int(remaining * proportion_among_idle)
                target = min(target, max_for_this_worker)

        # Clamp to bounds
        return max(min_batch_size, min(target, max_batch_size, remaining))

    while remaining_start < len(inputs) or pending_tasks:
        # Assign work to idle workers
        while remaining_start < len(inputs):
            # Use throughput-sorted selection for adaptive scheduling
            # High-throughput workers get priority and larger batches
            idle_workers = scheduler.get_idle_workers_by_throughput()
            if not idle_workers:
                break

            worker, state = idle_workers[0]
            remaining = len(inputs) - remaining_start
            n_idle = len(idle_workers)

            # Calculate batch size based on whether we have throughput data
            if state.throughput == 0:
                # No throughput data: use initial sizing
                batch_size = calculate_initial_batch_size(remaining, n_idle)
            else:
                # Have throughput data: use adaptive sizing
                batch_size = calculate_adaptive_batch_size(state, remaining, n_idle)

            start_idx = remaining_start
            end_idx = min(start_idx + batch_size, len(inputs))
            chunk = inputs[start_idx:end_idx]
            remaining_start = end_idx
            batch_counter += 1

            state.mark_busy()
            task = asyncio.create_task(
                process_batch(worker, state, chunk, start_idx, end_idx, batch_counter)
            )
            task._batch_start_idx = start_idx  # type: ignore
            pending_tasks.add(task)

        if not pending_tasks:
            break

        # Wait for at least one task to complete
        done, pending_tasks = await asyncio.wait(
            pending_tasks, return_when=asyncio.FIRST_COMPLETED
        )

        for task in done:
            try:
                result = task.result()
                results_map[result.start_idx] = result
            except Exception as e:
                start_idx = getattr(task, "_batch_start_idx", 0)
                results_map[start_idx] = DistributionResult(
                    worker_id="unknown",
                    start_idx=start_idx,
                    end_idx=start_idx,
                    error=e,
                    latency=0,
                )

    session_duration = time.time() - session_start

    if tracker:
        tracker.end_session()

    # Log stats
    if enable_perf_tracking:
        worker_stats = {}
        for result in results_map.values():
            wid = result.worker_id
            if wid not in worker_stats:
                worker_stats[wid] = {
                    "batches": 0,
                    "items": 0,
                    "total_time": 0,
                    "batch_sizes": [],
                }
            worker_stats[wid]["batches"] += 1
            worker_stats[wid]["items"] += result.end_idx - result.start_idx
            worker_stats[wid]["total_time"] += result.latency
            worker_stats[wid]["batch_sizes"].append(result.end_idx - result.start_idx)

        logger.mesg(f"[AdaptivePipeline] Completed in {session_duration*1000:.1f}ms")
        for wid, stats in worker_stats.items():
            avg_batch = (
                sum(stats["batch_sizes"]) / len(stats["batch_sizes"])
                if stats["batch_sizes"]
                else 0
            )
            state = scheduler.states.get(wid)
            throughput = state.throughput if state else 0
            ratio = get_throughput_ratios().get(wid, 0)
            logger.mesg(
                f"  {wid}: batches={stats['batches']}, "
                f"items={stats['items']} ({stats['items']*100//len(inputs)}%), "
                f"avg_batch={avg_batch:.0f}, "
                f"throughput={throughput:.0f}/s, "
                f"ratio={ratio:.1%}"
            )

    # Combine results
    all_results = [results_map[k] for k in sorted(results_map.keys())]

    failed = [r for r in all_results if not r.success]
    if failed:
        error_msgs = [f"{r.worker_id}: {r.error}" for r in failed[:3]]
        raise RuntimeError(
            f"AdaptivePipeline failed: {len(failed)}/{len(all_results)} batches. "
            f"Errors: {error_msgs}"
        )

    combined = []
    for r in all_results:
        if r.success and r.result is not None:
            if isinstance(r.result, list):
                combined.extend(r.result)
            else:
                combined.append(r.result)

    return combined, all_results
