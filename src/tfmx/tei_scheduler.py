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
"""

import asyncio
import time

from dataclasses import dataclass, field
from typing import TypeVar, Generic, Callable, Optional, Any
from tclogger import logger

from .tei_compose import MAX_CLIENT_BATCH_SIZE


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
    """Simple state tracking for a worker.

    Tracks:
    - busy/idle status
    - basic statistics (requests, items, errors)
    """

    worker_id: str

    # Busy/idle state
    busy: bool = False

    # Statistics
    total_requests: int = 0
    total_items: int = 0
    total_errors: int = 0
    total_latency: float = 0.0

    @property
    def is_idle(self) -> bool:
        """Check if worker is idle."""
        return not self.busy

    def mark_busy(self) -> None:
        """Mark worker as busy."""
        self.busy = True

    def mark_idle(self) -> None:
        """Mark worker as idle."""
        self.busy = False

    def record_success(self, latency: float, n_items: int) -> None:
        """Record a successful request."""
        self.total_requests += 1
        self.total_items += n_items
        self.total_latency += latency
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

    Args:
        scheduler: The idle-filling scheduler instance
        inputs: List of input texts to process
        process_func: Async function that processes inputs on a worker
            Signature: async def process_func(worker: W, inputs: list[str]) -> Any

    Returns:
        Tuple of (combined_results, distribution_details)
    """
    if not inputs:
        return [], []

    # Track all results
    all_results: list[DistributionResult] = []
    remaining_inputs = inputs
    processed_count = 0

    async def process_batch(
        worker: W,
        state: WorkerState,
        chunk: list[str],
        start_idx: int,
        end_idx: int,
    ) -> DistributionResult:
        """Process a single batch on a worker."""
        start_time = time.time()

        try:
            result = await process_func(worker, chunk)
            latency = time.time() - start_time

            # Record success and mark idle
            state.record_success(latency, len(chunk))
            scheduler._signal_worker_idle()

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
            has_idle = await scheduler.wait_for_idle_worker(timeout=60.0)
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

        # Distribute remaining inputs to idle workers
        batches = distribute_to_workers(
            remaining_inputs,
            len(idle_workers),
            scheduler.max_batch_size,
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
                process_batch(worker, state, chunk, abs_start, abs_end)
            )
            tasks.append(task)

        # Wait for this round to complete
        round_results = await asyncio.gather(*tasks, return_exceptions=True)
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

        # Update remaining inputs
        total_processed_this_round = sum(
            rel_end - rel_start for _, rel_start, rel_end in batches
        )
        remaining_inputs = remaining_inputs[total_processed_this_round:]
        processed_count += total_processed_this_round

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
