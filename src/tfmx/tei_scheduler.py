"""TEI Adaptive Scheduler

This module provides adaptive load balancing and scheduling for TEI instances.
It uses online metrics (latency, throughput, errors) to dynamically distribute
work across heterogeneous workers (GPUs/machines).

Key concepts:
- WorkerStats: Tracks per-worker metrics using EWMA (Exponential Weighted Moving Average)
- EFT (Estimated Finish Time): Predicts completion time for scheduling decisions
- Micro-batching: Splits large requests into smaller chunks for dynamic distribution
- Circuit breaker: Protects against cascading failures
"""

import time
import asyncio
import random

from dataclasses import dataclass, field
from typing import TypeVar, Generic, Callable, Optional, Any
from tclogger import logger


# EWMA decay factor (higher = more weight on recent samples)
EWMA_ALPHA = 0.3

# Micro-batch sizing
MICRO_BATCH_MIN = 4
MICRO_BATCH_MAX = 64
TARGET_BATCH_TIME = 0.5  # seconds

# Circuit breaker
ERROR_THRESHOLD = 3  # consecutive errors to trigger
COOLDOWN_SECONDS = 30.0
PROBE_INTERVAL = 10.0  # seconds between probes during cooldown

# Probing (epsilon-greedy exploration)
PROBE_PROBABILITY = 0.05  # 5% chance to probe non-optimal worker

# Inflight limits
MAX_INFLIGHT_PER_WORKER = 8


class EWMA:
    """Exponential Weighted Moving Average tracker."""

    def __init__(self, alpha: float = EWMA_ALPHA, initial: float = 0.0):
        """
        Args:
            alpha: Decay factor (0-1). Higher = more weight on recent values.
            initial: Initial value.
        """
        self.alpha = alpha
        self.value = initial
        self.count = 0

    def update(self, sample: float) -> float:
        """Update with a new sample and return the new average."""
        if self.count == 0:
            self.value = sample
        else:
            self.value = self.alpha * sample + (1 - self.alpha) * self.value
        self.count += 1
        return self.value

    def get(self) -> float:
        """Get the current EWMA value."""
        return self.value

    def reset(self, value: float = 0.0) -> None:
        """Reset the EWMA."""
        self.value = value
        self.count = 0


@dataclass
class WorkerStats:
    """Statistics for a single worker (GPU instance or machine endpoint).

    Tracks online metrics for adaptive scheduling decisions.
    """

    worker_id: str

    # Timing metrics (EWMA)
    rtt_ewma: EWMA = field(default_factory=lambda: EWMA(initial=0.1))
    per_item_ewma: EWMA = field(default_factory=lambda: EWMA(initial=0.01))
    per_char_ewma: EWMA = field(default_factory=lambda: EWMA(initial=0.0001))

    # Error tracking
    error_ewma: EWMA = field(default_factory=lambda: EWMA(alpha=0.5, initial=0.0))
    consecutive_errors: int = 0

    # Inflight tracking
    inflight: int = 0
    inflight_chars: int = 0

    # Circuit breaker state
    cooldown_until: float = 0.0
    last_probe_time: float = 0.0

    # Capacity hint (optional, e.g., healthy_instances for a machine)
    capacity_hint: int = 1

    # Cumulative stats
    total_requests: int = 0
    total_items: int = 0
    total_errors: int = 0
    total_latency: float = 0.0

    def is_available(self, now: Optional[float] = None) -> bool:
        """Check if worker is available (not in cooldown)."""
        if now is None:
            now = time.time()
        return now >= self.cooldown_until

    def is_in_cooldown(self, now: Optional[float] = None) -> bool:
        """Check if worker is in cooldown period."""
        return not self.is_available(now)

    def should_probe(self, now: Optional[float] = None) -> bool:
        """Check if we should send a probe during cooldown."""
        if now is None:
            now = time.time()
        if self.is_available(now):
            return False
        return (now - self.last_probe_time) >= PROBE_INTERVAL

    def enter_cooldown(self, duration: float = COOLDOWN_SECONDS) -> None:
        """Put worker into cooldown state."""
        now = time.time()
        self.cooldown_until = now + duration
        self.last_probe_time = now
        logger.warn(
            f"[scheduler] Worker {self.worker_id} entering cooldown for {duration:.1f}s"
        )

    def exit_cooldown(self) -> None:
        """Remove worker from cooldown state."""
        self.cooldown_until = 0.0
        self.consecutive_errors = 0
        logger.okay(f"[scheduler] Worker {self.worker_id} exiting cooldown")

    def record_success(self, latency: float, n_items: int, n_chars: int = 0) -> None:
        """Record a successful request completion."""
        self.total_requests += 1
        self.total_items += n_items
        self.total_latency += latency

        # Update timing EWMAs
        if n_items > 0:
            self.per_item_ewma.update(latency / n_items)
        if n_chars > 0:
            self.per_char_ewma.update(latency / n_chars)

        # RTT is the fixed overhead (estimate as latency minus processing time)
        # Simplified: use a fraction of latency for small batches
        if n_items <= 2:
            self.rtt_ewma.update(latency * 0.5)

        # Reset error state on success
        self.consecutive_errors = 0
        self.error_ewma.update(0.0)

        # Exit cooldown on success
        if self.is_in_cooldown():
            self.exit_cooldown()

    def record_error(self) -> None:
        """Record a failed request."""
        self.total_errors += 1
        self.consecutive_errors += 1
        self.error_ewma.update(1.0)

        # Trigger cooldown after threshold
        if self.consecutive_errors >= ERROR_THRESHOLD:
            # Exponential backoff
            backoff = COOLDOWN_SECONDS * (
                2 ** (self.consecutive_errors - ERROR_THRESHOLD)
            )
            backoff = min(backoff, 300.0)  # Cap at 5 minutes
            self.enter_cooldown(backoff)

    def begin_request(self, n_items: int, n_chars: int = 0) -> None:
        """Mark the start of a request."""
        self.inflight += 1
        self.inflight_chars += n_chars

    def end_request(self, n_chars: int = 0) -> None:
        """Mark the end of a request."""
        self.inflight = max(0, self.inflight - 1)
        self.inflight_chars = max(0, self.inflight_chars - n_chars)

    def estimate_finish_time(
        self,
        n_items: int,
        n_chars: int = 0,
        use_chars: bool = False,
    ) -> float:
        """Estimate the time to complete a batch of given size.

        Uses the EFT (Estimated Finish Time) model:
        EFT = queuing_delay + rtt + service_time + penalty

        Args:
            n_items: Number of items in the batch
            n_chars: Total characters in the batch (optional)
            use_chars: Whether to use per-char estimation (more accurate)

        Returns:
            Estimated completion time in seconds
        """
        # Service time estimation
        if use_chars and n_chars > 0 and self.per_char_ewma.count > 0:
            service_time = self.per_char_ewma.get() * n_chars
        else:
            service_time = self.per_item_ewma.get() * n_items

        # RTT (network/framework overhead)
        rtt = self.rtt_ewma.get()

        # Queuing delay: approximate as inflight * average_service_time
        avg_service = self.per_item_ewma.get() * max(n_items, 1)
        queuing_delay = self.inflight * avg_service

        # Error penalty: add extra time for unreliable workers
        error_penalty = self.error_ewma.get() * 2.0  # 2 seconds per unit error rate

        # Cooldown penalty: heavily penalize workers in cooldown
        if self.is_in_cooldown():
            cooldown_penalty = 1000.0  # Very high to avoid selection
        else:
            cooldown_penalty = 0.0

        return queuing_delay + rtt + service_time + error_penalty + cooldown_penalty

    def to_dict(self) -> dict:
        """Convert to dictionary for serialization."""
        return {
            "worker_id": self.worker_id,
            "rtt_ms": self.rtt_ewma.get() * 1000,
            "per_item_ms": self.per_item_ewma.get() * 1000,
            "error_rate": self.error_ewma.get(),
            "inflight": self.inflight,
            "in_cooldown": self.is_in_cooldown(),
            "total_requests": self.total_requests,
            "total_items": self.total_items,
            "total_errors": self.total_errors,
            "capacity_hint": self.capacity_hint,
        }


def calculate_micro_batch_size(
    total_items: int,
    total_chars: int,
    per_item_time: float,
    target_time: float = TARGET_BATCH_TIME,
    min_size: int = MICRO_BATCH_MIN,
    max_size: int = MICRO_BATCH_MAX,
) -> int:
    """Calculate optimal micro-batch size based on timing estimates.

    Args:
        total_items: Total number of items to process
        total_chars: Total characters (for better estimation)
        per_item_time: Estimated time per item (seconds)
        target_time: Target time per micro-batch (seconds)
        min_size: Minimum batch size
        max_size: Maximum batch size

    Returns:
        Recommended micro-batch size
    """
    if per_item_time <= 0:
        return max(min_size, min(max_size, total_items))

    # Calculate size to hit target time
    ideal_size = int(target_time / per_item_time)

    # Clamp to bounds
    size = max(min_size, min(max_size, ideal_size))

    # Don't exceed total
    size = min(size, total_items)

    return size


def split_into_micro_batches(
    inputs: list[str],
    batch_size: int,
) -> list[tuple[list[str], int, int]]:
    """Split inputs into micro-batches.

    Args:
        inputs: List of input texts
        batch_size: Target size for each micro-batch

    Returns:
        List of (chunk, start_idx, end_idx) tuples
    """
    batches = []
    n = len(inputs)
    start = 0

    while start < n:
        end = min(start + batch_size, n)
        chunk = inputs[start:end]
        batches.append((chunk, start, end))
        start = end

    return batches


W = TypeVar("W")  # Worker type (e.g., TEIInstance, TEIClient)


class AdaptiveScheduler(Generic[W]):
    """Adaptive scheduler for distributing work across heterogeneous workers.

    Uses EFT (Estimated Finish Time) based scheduling with:
    - Online metrics tracking (EWMA)
    - Micro-batching for dynamic distribution
    - Circuit breaker for fault tolerance
    - Epsilon-greedy probing for exploration

    Generic over worker type W (can be TEIInstance, TEIClient, etc.)
    """

    def __init__(
        self,
        workers: list[W],
        get_worker_id: Callable[[W], str],
        max_inflight_per_worker: int = MAX_INFLIGHT_PER_WORKER,
        probe_probability: float = PROBE_PROBABILITY,
        micro_batch_min: int = MICRO_BATCH_MIN,
        micro_batch_max: int = MICRO_BATCH_MAX,
        target_batch_time: float = TARGET_BATCH_TIME,
    ):
        """Initialize the scheduler.

        Args:
            workers: List of worker objects
            get_worker_id: Function to extract worker ID from worker object
            max_inflight_per_worker: Maximum concurrent requests per worker
            probe_probability: Probability of exploring non-optimal worker
            micro_batch_min: Minimum micro-batch size
            micro_batch_max: Maximum micro-batch size
            target_batch_time: Target time per micro-batch
        """
        self.workers = workers
        self.get_worker_id = get_worker_id
        self.max_inflight = max_inflight_per_worker
        self.probe_probability = probe_probability
        self.micro_batch_min = micro_batch_min
        self.micro_batch_max = micro_batch_max
        self.target_batch_time = target_batch_time

        # Create stats for each worker
        self.stats: dict[str, WorkerStats] = {}
        for w in workers:
            wid = get_worker_id(w)
            self.stats[wid] = WorkerStats(worker_id=wid)

        # Mapping from worker_id to worker object
        self._worker_map: dict[str, W] = {get_worker_id(w): w for w in workers}

    def update_workers(self, workers: list[W]) -> None:
        """Update the worker list (e.g., after health check)."""
        self.workers = workers
        self._worker_map = {self.get_worker_id(w): w for w in workers}

        # Add stats for new workers
        for w in workers:
            wid = self.get_worker_id(w)
            if wid not in self.stats:
                self.stats[wid] = WorkerStats(worker_id=wid)

    def set_capacity_hint(self, worker_id: str, capacity: int) -> None:
        """Set capacity hint for a worker (e.g., number of GPUs)."""
        if worker_id in self.stats:
            self.stats[worker_id].capacity_hint = capacity
            # Adjust initial estimates based on capacity
            # Higher capacity = lower per-item time estimate
            if self.stats[worker_id].per_item_ewma.count == 0:
                base_time = 0.01  # 10ms baseline
                self.stats[worker_id].per_item_ewma.value = base_time / capacity

    def get_available_workers(self) -> list[tuple[W, WorkerStats]]:
        """Get list of available workers with their stats."""
        now = time.time()
        available = []
        for w in self.workers:
            wid = self.get_worker_id(w)
            stats = self.stats.get(wid)
            if stats and stats.is_available(now):
                available.append((w, stats))
        return available

    def get_workers_for_probing(self) -> list[tuple[W, WorkerStats]]:
        """Get workers that should be probed (in cooldown but ready for probe)."""
        now = time.time()
        to_probe = []
        for w in self.workers:
            wid = self.get_worker_id(w)
            stats = self.stats.get(wid)
            if stats and stats.should_probe(now):
                to_probe.append((w, stats))
        return to_probe

    def select_worker_eft(
        self,
        n_items: int,
        n_chars: int = 0,
        use_chars: bool = False,
        exclude: Optional[set[str]] = None,
    ) -> Optional[tuple[W, WorkerStats]]:
        """Select the best worker using EFT (Estimated Finish Time).

        Args:
            n_items: Number of items in the batch
            n_chars: Total characters (optional)
            use_chars: Whether to use per-char estimation
            exclude: Set of worker IDs to exclude

        Returns:
            Tuple of (worker, stats) or None if no workers available
        """
        available = self.get_available_workers()
        if not available:
            return None

        # Filter excluded workers
        if exclude:
            available = [(w, s) for w, s in available if s.worker_id not in exclude]
            if not available:
                return None

        # Filter by inflight limit
        available = [(w, s) for w, s in available if s.inflight < self.max_inflight]
        if not available:
            # All workers at capacity, pick least loaded
            available = self.get_available_workers()
            if exclude:
                available = [(w, s) for w, s in available if s.worker_id not in exclude]

        if not available:
            return None

        # Epsilon-greedy exploration
        if random.random() < self.probe_probability and len(available) > 1:
            return random.choice(available)

        # Power of Two Choices: sample 2 workers, pick better one
        if len(available) >= 2:
            candidates = random.sample(available, 2)
            eft0 = candidates[0][1].estimate_finish_time(n_items, n_chars, use_chars)
            eft1 = candidates[1][1].estimate_finish_time(n_items, n_chars, use_chars)
            return candidates[0] if eft0 <= eft1 else candidates[1]

        # Only one available
        return available[0]

    def select_worker_least_loaded(
        self,
        exclude: Optional[set[str]] = None,
    ) -> Optional[tuple[W, WorkerStats]]:
        """Select worker with lowest inflight count (fallback strategy)."""
        available = self.get_available_workers()
        if exclude:
            available = [(w, s) for w, s in available if s.worker_id not in exclude]
        if not available:
            return None

        return min(available, key=lambda x: x[1].inflight)

    def calculate_micro_batch_size(self, total_items: int, total_chars: int = 0) -> int:
        """Calculate optimal micro-batch size based on current worker stats."""
        # Use average per-item time across all workers
        if not self.stats:
            return self.micro_batch_min

        per_item_times = [
            s.per_item_ewma.get()
            for s in self.stats.values()
            if s.per_item_ewma.count > 0
        ]
        if not per_item_times:
            return self.micro_batch_min

        avg_per_item = sum(per_item_times) / len(per_item_times)

        return calculate_micro_batch_size(
            total_items=total_items,
            total_chars=total_chars,
            per_item_time=avg_per_item,
            target_time=self.target_batch_time,
            min_size=self.micro_batch_min,
            max_size=self.micro_batch_max,
        )

    def get_stats_summary(self) -> dict:
        """Get summary of all worker stats."""
        return {wid: s.to_dict() for wid, s in self.stats.items()}


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
    scheduler: AdaptiveScheduler[W],
    inputs: list[str],
    process_func: Callable[[W, list[str]], Any],
    use_chars: bool = True,
) -> tuple[list[Any], list[DistributionResult]]:
    """Distribute inputs across workers using adaptive scheduling.

    This is the main entry point for async distribution with the scheduler.

    Args:
        scheduler: The adaptive scheduler instance
        inputs: List of input texts to process
        process_func: Async function that processes inputs on a worker
            Signature: async def process_func(worker: W, inputs: list[str]) -> Any
        use_chars: Whether to use character-based estimation

    Returns:
        Tuple of (combined_results, distribution_details)
    """
    if not inputs:
        return [], []

    n_inputs = len(inputs)
    total_chars = sum(len(s) for s in inputs)

    # Calculate micro-batch size
    batch_size = scheduler.calculate_micro_batch_size(n_inputs, total_chars)

    # Split into micro-batches
    micro_batches = split_into_micro_batches(inputs, batch_size)

    # Track results
    results: list[DistributionResult] = []
    pending_batches = list(micro_batches)  # Copy for modification

    # Process batches dynamically
    async def process_batch(
        worker: W,
        stats: WorkerStats,
        chunk: list[str],
        start_idx: int,
        end_idx: int,
    ) -> DistributionResult:
        """Process a single batch on a worker."""
        n_items = len(chunk)
        n_chars = sum(len(s) for s in chunk)

        # Mark request start
        stats.begin_request(n_items, n_chars)
        start_time = time.time()

        try:
            result = await process_func(worker, chunk)
            latency = time.time() - start_time

            # Record success
            stats.record_success(latency, n_items, n_chars)
            stats.end_request(n_chars)

            return DistributionResult(
                worker_id=stats.worker_id,
                start_idx=start_idx,
                end_idx=end_idx,
                result=result,
                latency=latency,
            )

        except Exception as e:
            latency = time.time() - start_time

            # Record error
            stats.record_error()
            stats.end_request(n_chars)

            return DistributionResult(
                worker_id=stats.worker_id,
                start_idx=start_idx,
                end_idx=end_idx,
                error=e,
                latency=latency,
            )

    # Create initial tasks for all batches
    tasks = []
    used_workers: set[str] = set()

    for chunk, start_idx, end_idx in pending_batches:
        n_items = len(chunk)
        n_chars = sum(len(s) for s in chunk)

        # Select best worker (can reuse workers for different batches)
        selection = scheduler.select_worker_eft(n_items, n_chars, use_chars)

        if selection is None:
            # No available workers, try least loaded
            selection = scheduler.select_worker_least_loaded()

        if selection is None:
            # Still no workers, record error
            results.append(
                DistributionResult(
                    worker_id="none",
                    start_idx=start_idx,
                    end_idx=end_idx,
                    error=RuntimeError("No available workers"),
                )
            )
            continue

        worker, stats = selection
        used_workers.add(stats.worker_id)

        task = process_batch(worker, stats, chunk, start_idx, end_idx)
        tasks.append(task)

    # Execute all tasks
    if tasks:
        batch_results = await asyncio.gather(*tasks, return_exceptions=True)
        for r in batch_results:
            if isinstance(r, Exception):
                # This shouldn't happen as we handle exceptions inside process_batch
                results.append(
                    DistributionResult(
                        worker_id="unknown",
                        start_idx=0,
                        end_idx=0,
                        error=r,
                    )
                )
            else:
                results.append(r)

    # Sort results by start_idx and combine
    results.sort(key=lambda r: r.start_idx)

    # Combine results in order
    combined = []
    for r in results:
        if r.success and r.result is not None:
            if isinstance(r.result, list):
                combined.extend(r.result)
            else:
                combined.append(r.result)
        else:
            # For failed batches, add placeholder (empty results)
            chunk_size = r.end_idx - r.start_idx
            combined.extend([None] * chunk_size)

    return combined, results
