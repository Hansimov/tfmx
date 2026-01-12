"""TEI Idle-Filling Scheduler

This module provides a simple "idle-filling" load balancing scheduler for TEI instances.
The algorithm is straightforward:

1. Each worker (TEI instance) can only process one batch at a time
2. When a batch arrives, it's split into chunks of MAX_BATCH_SIZE
3. Each chunk is assigned to an idle worker
4. Workers are marked busy while processing, idle when done
5. If no workers are idle, we wait for one to become available

Key concepts:
- WorkerState: Simple busy/idle state tracking for each worker
- IdleFillingScheduler: Distributes work to idle workers
- MAX_BATCH_SIZE: Fixed batch size from tei_compose (--max-client-batch-size)

Performance Tracking:
- Set enable_perf_tracking=True in distribute_with_scheduler for detailed analysis
- Use PerfTracker for identifying round barrier and utilization bottlenecks
"""

import asyncio
import time

from dataclasses import dataclass, field
from typing import TypeVar, Generic, Callable, Optional, Any
from tclogger import logger, logstr

from .tei_compose import MAX_CLIENT_BATCH_SIZE
from .perf_tracker import PerfTracker, get_global_tracker


def distribute_to_workers(
    inputs: list[str],
    n_workers: int,
    max_batch_size: int = MAX_CLIENT_BATCH_SIZE,
) -> list[tuple[list[str], int, int]]:
    """Distribute inputs across available workers optimally.

    Strategy:
    - If total inputs <= n_workers * max_batch_size: evenly distribute to all workers
    - If total inputs > n_workers * max_batch_size: give each worker max_batch_size

    Args:
        inputs: List of input texts
        n_workers: Number of available workers
        max_batch_size: Maximum batch size per worker (default: MAX_CLIENT_BATCH_SIZE)

    Returns:
        List of (chunk, start_idx, end_idx) tuples
    """
    if n_workers <= 0:
        return []

    n_inputs = len(inputs)
    if n_inputs == 0:
        return []

    total_capacity = n_workers * max_batch_size

    batches = []

    if n_inputs <= total_capacity:
        # Evenly distribute across all workers
        batch_size = (n_inputs + n_workers - 1) // n_workers  # Ceiling division
        start = 0
        for i in range(n_workers):
            if start >= n_inputs:
                break
            end = min(start + batch_size, n_inputs)
            chunk = inputs[start:end]
            batches.append((chunk, start, end))
            start = end
    else:
        # Give each worker the maximum batch size
        start = 0
        for i in range(n_workers):
            end = min(start + max_batch_size, n_inputs)
            chunk = inputs[start:end]
            batches.append((chunk, start, end))
            start = end

    return batches


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


async def distribute_with_scheduler(
    scheduler: IdleFillingScheduler[W],
    inputs: list[str],
    process_func: Callable[[W, list[str]], Any],
    enable_perf_tracking: bool = False,
    perf_tracker: Optional[PerfTracker] = None,
) -> tuple[list[Any], list[DistributionResult]]:
    """Distribute inputs across workers using idle-filling scheduling.

    This is the main entry point for async distribution with the scheduler.

    Algorithm:
    1. Get all idle workers
    2. Distribute inputs optimally across idle workers:
       - If inputs <= workers * max_batch_size: evenly distribute
       - If inputs > workers * max_batch_size: give each worker max_batch_size
    3. Process batches concurrently
    4. Repeat until all inputs processed

    ⚠️ KNOWN BOTTLENECK: Round Barrier Synchronization
    This algorithm waits for ALL tasks in a round to complete before starting
    the next round. This causes fast workers to sit idle while waiting for
    slow workers, leading to GPU underutilization.

    Args:
        scheduler: The idle-filling scheduler instance
        inputs: List of input texts to process
        process_func: Async function that processes inputs on a worker
            Signature: async def process_func(worker: W, inputs: list[str]) -> Any
        enable_perf_tracking: Enable detailed performance tracking (default: False)
        perf_tracker: Custom PerfTracker instance (uses global if None)

    Returns:
        Tuple of (combined_results, distribution_details)
    """
    if not inputs:
        return [], []

    # Setup performance tracking
    tracker = None
    if enable_perf_tracking:
        tracker = perf_tracker or get_global_tracker()
        tracker.start_session(n_inputs=len(inputs), n_workers=len(scheduler.workers))

    # Track all results
    all_results: list[DistributionResult] = []
    remaining_inputs = inputs
    processed_count = 0
    round_counter = 0

    async def process_batch(
        worker: W,
        state: WorkerState,
        chunk: list[str],
        start_idx: int,
        end_idx: int,
        round_id: int,
        round_start_time: float,
    ) -> DistributionResult:
        """Process a single batch on a worker."""
        start_time = time.time()
        queue_wait = start_time - round_start_time

        try:
            result = await process_func(worker, chunk)
            latency = time.time() - start_time

            # Record success and mark idle
            state.record_success(latency, len(chunk))
            scheduler._signal_worker_idle()

            # Record to perf tracker
            if tracker:
                tracker.record_task(
                    worker_id=state.worker_id,
                    round_id=round_id,
                    batch_size=len(chunk),
                    start_time=start_time,
                    end_time=time.time(),
                    queue_wait_time=queue_wait,
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
            latency = time.time() - start_time

            # Record error and mark idle
            state.record_error()
            scheduler._signal_worker_idle()

            # Record to perf tracker
            if tracker:
                tracker.record_task(
                    worker_id=state.worker_id,
                    round_id=round_id,
                    batch_size=len(chunk),
                    start_time=start_time,
                    end_time=time.time(),
                    queue_wait_time=queue_wait,
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

    # Process inputs in rounds until all done
    while remaining_inputs:
        # Get idle workers
        idle_workers = scheduler.get_idle_workers()

        if not idle_workers:
            # No idle workers, wait for at least one
            wait_start = time.time()
            has_idle = await scheduler.wait_for_idle_worker(timeout=60.0)
            wait_duration = time.time() - wait_start

            if enable_perf_tracking and wait_duration > 0.001:
                logger.warn(
                    f"[Scheduler] Waited {wait_duration*1000:.1f}ms for idle worker"
                )

            if not has_idle:
                # Timeout - fail remaining inputs
                for i in range(len(remaining_inputs)):
                    all_results.append(
                        DistributionResult(
                            worker_id="timeout",
                            start_idx=processed_count + i,
                            end_idx=processed_count + i + 1,
                            error=TimeoutError("No idle workers available"),
                        )
                    )
                break
            # Try again with newly idle workers
            continue

        round_counter += 1
        round_start_time = time.time()

        # Distribute remaining inputs to idle workers
        batches = distribute_to_workers(
            remaining_inputs,
            len(idle_workers),
            scheduler.max_batch_size,
        )

        # Log round info if tracking enabled
        if enable_perf_tracking:
            total_this_round = sum(len(chunk) for chunk, _, _ in batches)
            logger.mesg(
                f"[Round {round_counter}] "
                f"workers={logstr.file(len(idle_workers))}/{len(scheduler.workers)}, "
                f"items={logstr.mesg(total_this_round)}, "
                f"remaining={len(remaining_inputs)}"
            )

        # Create tasks for this round
        tasks = []
        for (chunk, rel_start, rel_end), (worker, state) in zip(batches, idle_workers):
            # Calculate absolute indices
            abs_start = processed_count + rel_start
            abs_end = processed_count + rel_end

            # Mark worker as busy
            state.mark_busy()

            # Create task
            task = asyncio.create_task(
                process_batch(
                    worker,
                    state,
                    chunk,
                    abs_start,
                    abs_end,
                    round_counter,
                    round_start_time,
                )
            )
            tasks.append(task)

        # ⚠️ BOTTLENECK: This is the round barrier!
        # We wait for ALL tasks to complete before proceeding
        # Fast workers sit idle while waiting for slow workers
        round_results = await asyncio.gather(*tasks, return_exceptions=True)
        round_end_time = time.time()
        round_duration = round_end_time - round_start_time

        # Collect round results and calculate metrics
        round_latencies = []
        for r in round_results:
            if isinstance(r, Exception):
                # Unexpected error in task
                all_results.append(
                    DistributionResult(
                        worker_id="unknown",
                        start_idx=0,
                        end_idx=0,
                        error=r,
                    )
                )
            else:
                all_results.append(r)
                round_latencies.append(r.latency)

        # Log round barrier analysis if tracking enabled
        if enable_perf_tracking and round_latencies:
            min_lat = min(round_latencies)
            max_lat = max(round_latencies)
            imbalance = max_lat / min_lat if min_lat > 0 else 1.0
            idle_waste = sum(round_duration - lat for lat in round_latencies)

            if imbalance > 1.2:  # More than 20% imbalance
                logger.warn(
                    f"[Round {round_counter}] "
                    f"duration={round_duration*1000:.1f}ms, "
                    f"latency_range=[{min_lat*1000:.1f}, {max_lat*1000:.1f}]ms, "
                    f"imbalance={logstr.warn(f'{imbalance:.2f}x')}, "
                    f"idle_waste={logstr.warn(f'{idle_waste*1000:.1f}ms')}"
                )

        # Record round to perf tracker
        if tracker:
            from .perf_tracker import RoundRecord

            round_record = RoundRecord(
                round_id=round_counter,
                start_time=round_start_time,
                end_time=round_end_time,
                n_workers_used=len(batches),
                n_workers_available=len(idle_workers),
                total_items=sum(len(chunk) for chunk, _, _ in batches),
            )
            tracker.record_round(round_record)

        # Update remaining inputs
        total_processed_this_round = sum(
            rel_end - rel_start for _, rel_start, rel_end in batches
        )
        remaining_inputs = remaining_inputs[total_processed_this_round:]
        processed_count += total_processed_this_round

    # End performance tracking session
    if tracker:
        tracker.end_session()

    # Sort results by start_idx
    all_results.sort(key=lambda r: r.start_idx)

    # Check for failures
    failed = [r for r in all_results if not r.success]
    if failed:
        # Extract error messages
        error_msgs = []
        for r in failed:
            error_msg = f"{r.worker_id}: {r.error}"
            error_msgs.append(error_msg)

        # Raise exception with details
        raise RuntimeError(
            f"Distribution failed: {len(failed)}/{len(all_results)} batches failed. "
            f"Errors: {error_msgs[:3]}"  # Show first 3 errors
        )

    # Combine results in order
    combined = []
    for r in all_results:
        if r.success and r.result is not None:
            if isinstance(r.result, list):
                combined.extend(r.result)
            else:
                combined.append(r.result)

    return combined, all_results


async def distribute_with_pipeline(
    scheduler: IdleFillingScheduler[W],
    inputs: list[str],
    process_func: Callable[[W, list[str]], Any],
    enable_perf_tracking: bool = False,
    perf_tracker: Optional[PerfTracker] = None,
    micro_batch_size: int = 50,
) -> tuple[list[Any], list[DistributionResult]]:
    """Distribute inputs using true pipeline scheduling (no round barriers).

    This scheduler eliminates the round barrier problem by:
    1. Splitting inputs into small micro-batches
    2. Immediately assigning new work to any worker that becomes idle
    3. Never waiting for all workers to finish before assigning more work

    This approach ensures:
    - Fast workers process more batches
    - No worker sits idle waiting for others
    - Maximum GPU utilization

    Args:
        scheduler: The idle-filling scheduler instance
        inputs: List of input texts to process
        process_func: Async function that processes inputs on a worker
        enable_perf_tracking: Enable detailed performance tracking
        perf_tracker: Custom PerfTracker instance
        micro_batch_size: Size of each micro-batch (default: 50)

    Returns:
        Tuple of (combined_results, distribution_details)
    """
    if not inputs:
        return [], []

    # Setup performance tracking
    tracker = None
    if enable_perf_tracking:
        tracker = perf_tracker or get_global_tracker()
        tracker.start_session(n_inputs=len(inputs), n_workers=len(scheduler.workers))

    # Split inputs into micro-batches
    micro_batches: list[tuple[list[str], int, int]] = []
    for i in range(0, len(inputs), micro_batch_size):
        end = min(i + micro_batch_size, len(inputs))
        micro_batches.append((inputs[i:end], i, end))

    if enable_perf_tracking:
        logger.mesg(
            f"[Pipeline] total={len(inputs)}, "
            f"micro_batches={len(micro_batches)}, "
            f"micro_batch_size={micro_batch_size}, "
            f"workers={len(scheduler.workers)}"
        )

    # Results storage (indexed by start position for ordering)
    results_map: dict[int, DistributionResult] = {}
    batch_index = 0
    pending_tasks: set[asyncio.Task] = set()
    session_start = time.time()

    async def process_micro_batch(
        worker: W,
        state: WorkerState,
        chunk: list[str],
        start_idx: int,
        end_idx: int,
        batch_id: int,
    ) -> DistributionResult:
        """Process a single micro-batch on a worker."""
        task_start = time.time()

        try:
            result = await process_func(worker, chunk)
            latency = time.time() - task_start

            # Record success
            state.record_success(latency, len(chunk))
            scheduler._signal_worker_idle()

            # Record to perf tracker
            if tracker:
                tracker.record_task(
                    worker_id=state.worker_id,
                    round_id=batch_id,  # Use batch_id as round_id for tracking
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

    # Main scheduling loop
    while batch_index < len(micro_batches) or pending_tasks:
        # Try to assign work to idle workers
        while batch_index < len(micro_batches):
            idle_workers = scheduler.get_idle_workers()
            if not idle_workers:
                break

            # Get next batch and an idle worker
            chunk, start_idx, end_idx = micro_batches[batch_index]
            worker, state = idle_workers[0]

            # Mark worker as busy and create task
            state.mark_busy()
            task = asyncio.create_task(
                process_micro_batch(
                    worker, state, chunk, start_idx, end_idx, batch_index
                )
            )
            # Store batch info in task for later retrieval
            task._batch_start_idx = start_idx  # type: ignore
            pending_tasks.add(task)
            batch_index += 1

        # If no pending tasks, we're done
        if not pending_tasks:
            break

        # Wait for at least one task to complete
        done, pending_tasks = await asyncio.wait(
            pending_tasks, return_when=asyncio.FIRST_COMPLETED
        )

        # Collect results from completed tasks
        for task in done:
            try:
                result = task.result()
                results_map[result.start_idx] = result
            except Exception as e:
                # This shouldn't happen as exceptions are caught in process_micro_batch
                start_idx = getattr(task, "_batch_start_idx", 0)
                results_map[start_idx] = DistributionResult(
                    worker_id="unknown",
                    start_idx=start_idx,
                    end_idx=start_idx,
                    error=e,
                    latency=0,
                )

    session_duration = time.time() - session_start

    # End performance tracking
    if tracker:
        tracker.end_session()

    # Log pipeline stats
    if enable_perf_tracking:
        # Calculate per-worker stats
        worker_stats = {}
        for result in results_map.values():
            wid = result.worker_id
            if wid not in worker_stats:
                worker_stats[wid] = {"batches": 0, "items": 0, "total_time": 0}
            worker_stats[wid]["batches"] += 1
            worker_stats[wid]["items"] += result.end_idx - result.start_idx
            worker_stats[wid]["total_time"] += result.latency

        logger.mesg(f"[Pipeline] Completed in {session_duration*1000:.1f}ms")
        for wid, stats in worker_stats.items():
            logger.mesg(
                f"  {wid}: batches={stats['batches']}, "
                f"items={stats['items']}, "
                f"avg_latency={stats['total_time']/stats['batches']*1000:.1f}ms"
            )

    # Sort results by start index and combine
    all_results = [results_map[k] for k in sorted(results_map.keys())]

    # Check for failures
    failed = [r for r in all_results if not r.success]
    if failed:
        error_msgs = [f"{r.worker_id}: {r.error}" for r in failed[:3]]
        raise RuntimeError(
            f"Pipeline failed: {len(failed)}/{len(all_results)} batches failed. "
            f"Errors: {error_msgs}"
        )

    # Combine results in order
    combined = []
    for r in all_results:
        if r.success and r.result is not None:
            if isinstance(r.result, list):
                combined.extend(r.result)
            else:
                combined.append(r.result)

    return combined, all_results


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
