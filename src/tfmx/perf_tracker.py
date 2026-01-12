"""Performance Tracking and Analysis Module

This module provides tools for tracking, analyzing, and visualizing performance
metrics to help identify bottlenecks in the TEI multi-GPU scheduling system.

Key metrics tracked:
- Worker utilization (busy vs idle time)
- Batch processing latency per worker
- Queue wait times
- Round-by-round scheduling efficiency
- GPU idle gaps (time between task completion and new task assignment)
"""

import asyncio
import time
import statistics
from dataclasses import dataclass, field
from typing import Optional, Any, Callable
from collections import defaultdict
from tclogger import logger, logstr


@dataclass
class WorkerEvent:
    """Represents a single event in a worker's timeline."""

    worker_id: str
    event_type: str  # "task_start", "task_end", "idle_start", "idle_end"
    timestamp: float
    batch_size: int = 0
    round_id: int = 0
    metadata: dict = field(default_factory=dict)


@dataclass
class TaskRecord:
    """Record of a single task execution."""

    worker_id: str
    round_id: int
    batch_size: int
    start_time: float
    end_time: float
    queue_wait_time: float = 0.0  # Time waiting before task started
    success: bool = True
    error: Optional[str] = None

    @property
    def latency(self) -> float:
        """Task execution latency in seconds."""
        return self.end_time - self.start_time

    @property
    def throughput(self) -> float:
        """Items per second for this task."""
        if self.latency > 0:
            return self.batch_size / self.latency
        return 0.0


@dataclass
class RoundRecord:
    """Record of a single scheduling round."""

    round_id: int
    start_time: float
    end_time: float
    n_workers_used: int
    n_workers_available: int
    total_items: int
    tasks: list[TaskRecord] = field(default_factory=list)

    @property
    def duration(self) -> float:
        """Round duration in seconds."""
        return self.end_time - self.start_time

    @property
    def worker_utilization(self) -> float:
        """Fraction of available workers used."""
        if self.n_workers_available > 0:
            return self.n_workers_used / self.n_workers_available
        return 0.0

    @property
    def avg_task_latency(self) -> float:
        """Average task latency in this round."""
        if self.tasks:
            return statistics.mean(t.latency for t in self.tasks)
        return 0.0

    @property
    def min_task_latency(self) -> float:
        """Minimum task latency in this round."""
        if self.tasks:
            return min(t.latency for t in self.tasks)
        return 0.0

    @property
    def max_task_latency(self) -> float:
        """Maximum task latency in this round."""
        if self.tasks:
            return max(t.latency for t in self.tasks)
        return 0.0

    @property
    def latency_imbalance(self) -> float:
        """Ratio of max to min latency (indicates GPU speed variance)."""
        if self.min_task_latency > 0:
            return self.max_task_latency / self.min_task_latency
        return 1.0

    @property
    def idle_time_wasted(self) -> float:
        """
        Time wasted by fast workers waiting for slow workers.
        This is the core metric for identifying the "round barrier" problem.
        """
        if not self.tasks:
            return 0.0
        # Sum of (round_duration - task_latency) for each task
        return sum(self.duration - t.latency for t in self.tasks)


@dataclass
class WorkerStats:
    """Accumulated statistics for a single worker."""

    worker_id: str
    total_tasks: int = 0
    total_items: int = 0
    total_busy_time: float = 0.0
    total_idle_time: float = 0.0
    total_queue_wait: float = 0.0
    task_latencies: list[float] = field(default_factory=list)

    @property
    def avg_latency(self) -> float:
        if self.task_latencies:
            return statistics.mean(self.task_latencies)
        return 0.0

    @property
    def utilization(self) -> float:
        """Fraction of time spent busy vs total time."""
        total = self.total_busy_time + self.total_idle_time
        if total > 0:
            return self.total_busy_time / total
        return 0.0

    @property
    def throughput(self) -> float:
        """Items per second."""
        if self.total_busy_time > 0:
            return self.total_items / self.total_busy_time
        return 0.0


class PerfTracker:
    """
    Performance tracker for TEI scheduling operations.

    Tracks detailed timing information to identify:
    1. GPU utilization imbalances
    2. Round barrier synchronization overhead
    3. Task distribution efficiency
    4. Queue wait times

    Usage:
        tracker = PerfTracker()
        tracker.start_session(n_inputs=10000, n_workers=4)

        # In scheduler:
        with tracker.track_round(round_id, n_workers) as round_ctx:
            for worker, batch in assignments:
                with round_ctx.track_task(worker_id, len(batch)) as task_ctx:
                    result = await process(batch)

        tracker.end_session()
        tracker.print_analysis()
    """

    def __init__(self, name: str = "default", verbose: bool = True):
        self.name = name
        self.verbose = verbose

        # Session state
        self.session_start: Optional[float] = None
        self.session_end: Optional[float] = None
        self.n_inputs: int = 0
        self.n_workers: int = 0

        # Records
        self.events: list[WorkerEvent] = []
        self.rounds: list[RoundRecord] = []
        self.tasks: list[TaskRecord] = []

        # Worker tracking
        self.worker_stats: dict[str, WorkerStats] = {}
        self.worker_last_end: dict[str, float] = {}  # For tracking idle gaps

        # Current round context
        self._current_round: Optional[RoundRecord] = None
        self._round_counter: int = 0

    def reset(self) -> None:
        """Reset all tracking data."""
        self.session_start = None
        self.session_end = None
        self.n_inputs = 0
        self.n_workers = 0
        self.events.clear()
        self.rounds.clear()
        self.tasks.clear()
        self.worker_stats.clear()
        self.worker_last_end.clear()
        self._current_round = None
        self._round_counter = 0

    def start_session(self, n_inputs: int, n_workers: int) -> None:
        """Start a new tracking session."""
        self.reset()
        self.session_start = time.time()
        self.n_inputs = n_inputs
        self.n_workers = n_workers

        if self.verbose:
            logger.note(
                f"[PerfTracker:{self.name}] Session started: "
                f"inputs={n_inputs}, workers={n_workers}"
            )

    def end_session(self) -> None:
        """End the current tracking session."""
        self.session_end = time.time()

        if self.verbose:
            duration = self.session_end - self.session_start
            logger.note(
                f"[PerfTracker:{self.name}] Session ended: " f"duration={duration:.3f}s"
            )

    @property
    def session_duration(self) -> float:
        """Total session duration in seconds."""
        if self.session_start is None:
            return 0.0
        end = self.session_end or time.time()
        return end - self.session_start

    def _ensure_worker_stats(self, worker_id: str) -> WorkerStats:
        """Get or create worker stats."""
        if worker_id not in self.worker_stats:
            self.worker_stats[worker_id] = WorkerStats(worker_id=worker_id)
        return self.worker_stats[worker_id]

    def track_round(self, n_workers_available: int) -> "RoundContext":
        """
        Context manager for tracking a scheduling round.

        Args:
            n_workers_available: Number of workers available this round

        Returns:
            RoundContext for tracking tasks within this round
        """
        self._round_counter += 1
        return RoundContext(
            tracker=self,
            round_id=self._round_counter,
            n_workers_available=n_workers_available,
        )

    def record_task(
        self,
        worker_id: str,
        round_id: int,
        batch_size: int,
        start_time: float,
        end_time: float,
        queue_wait_time: float = 0.0,
        success: bool = True,
        error: Optional[str] = None,
    ) -> TaskRecord:
        """Record a completed task."""
        task = TaskRecord(
            worker_id=worker_id,
            round_id=round_id,
            batch_size=batch_size,
            start_time=start_time,
            end_time=end_time,
            queue_wait_time=queue_wait_time,
            success=success,
            error=error,
        )
        self.tasks.append(task)

        # Update worker stats
        stats = self._ensure_worker_stats(worker_id)
        stats.total_tasks += 1
        stats.total_items += batch_size
        stats.total_busy_time += task.latency
        stats.total_queue_wait += queue_wait_time
        stats.task_latencies.append(task.latency)

        # Track idle gap
        if worker_id in self.worker_last_end:
            idle_gap = start_time - self.worker_last_end[worker_id]
            stats.total_idle_time += idle_gap
        self.worker_last_end[worker_id] = end_time

        # Add to current round if active
        if self._current_round and self._current_round.round_id == round_id:
            self._current_round.tasks.append(task)

        return task

    def record_round(self, round_record: RoundRecord) -> None:
        """Record a completed round.

        If the round has no tasks, we'll try to associate tasks from self.tasks
        that have the same round_id.
        """
        # If round has no tasks, try to find and associate them
        if not round_record.tasks:
            round_record.tasks = [
                t for t in self.tasks if t.round_id == round_record.round_id
            ]

        self.rounds.append(round_record)

    def get_analysis(self) -> dict:
        """
        Generate comprehensive performance analysis.

        Returns dict with:
        - summary: Overall metrics
        - worker_analysis: Per-worker breakdown
        - round_analysis: Per-round breakdown
        - bottleneck_analysis: Identified bottlenecks
        """
        if not self.tasks:
            return {"error": "No tasks recorded"}

        analysis = {
            "summary": self._analyze_summary(),
            "worker_analysis": self._analyze_workers(),
            "round_analysis": self._analyze_rounds(),
            "bottleneck_analysis": self._analyze_bottlenecks(),
        }

        return analysis

    def _analyze_summary(self) -> dict:
        """Generate summary statistics."""
        total_items = sum(t.batch_size for t in self.tasks)
        total_latency = sum(t.latency for t in self.tasks)
        successful_tasks = [t for t in self.tasks if t.success]

        return {
            "session_duration_s": self.session_duration,
            "total_inputs": self.n_inputs,
            "total_workers": self.n_workers,
            "total_tasks": len(self.tasks),
            "successful_tasks": len(successful_tasks),
            "total_rounds": len(self.rounds),
            "throughput_items_per_sec": (
                total_items / self.session_duration if self.session_duration > 0 else 0
            ),
            "avg_task_latency_ms": (
                total_latency / len(self.tasks) * 1000 if self.tasks else 0
            ),
        }

    def _analyze_workers(self) -> dict:
        """Generate per-worker analysis."""
        worker_data = {}

        for wid, stats in self.worker_stats.items():
            worker_data[wid] = {
                "total_tasks": stats.total_tasks,
                "total_items": stats.total_items,
                "utilization_pct": stats.utilization * 100,
                "avg_latency_ms": stats.avg_latency * 1000,
                "throughput_items_per_sec": stats.throughput,
                "total_idle_time_s": stats.total_idle_time,
                "total_queue_wait_s": stats.total_queue_wait,
            }

        return worker_data

    def _analyze_rounds(self) -> dict:
        """Generate per-round analysis."""
        if not self.rounds:
            return {"error": "No rounds recorded"}

        round_data = []
        for r in self.rounds:
            round_data.append(
                {
                    "round_id": r.round_id,
                    "duration_ms": r.duration * 1000,
                    "workers_used": r.n_workers_used,
                    "workers_available": r.n_workers_available,
                    "total_items": r.total_items,
                    "utilization_pct": r.worker_utilization * 100,
                    "latency_imbalance": r.latency_imbalance,
                    "idle_time_wasted_s": r.idle_time_wasted,
                    "min_task_latency_ms": r.min_task_latency * 1000,
                    "max_task_latency_ms": r.max_task_latency * 1000,
                }
            )

        # Aggregate round stats
        durations = [r.duration for r in self.rounds]
        imbalances = [r.latency_imbalance for r in self.rounds]
        idle_wasted = [r.idle_time_wasted for r in self.rounds]

        return {
            "rounds": round_data,
            "aggregate": {
                "avg_round_duration_ms": statistics.mean(durations) * 1000,
                "avg_latency_imbalance": statistics.mean(imbalances),
                "total_idle_time_wasted_s": sum(idle_wasted),
                "avg_idle_time_wasted_per_round_s": statistics.mean(idle_wasted),
            },
        }

    def _analyze_bottlenecks(self) -> dict:
        """
        Identify and quantify performance bottlenecks.

        Key bottlenecks analyzed:
        1. Round barrier synchronization: Fast workers waiting for slow ones
        2. Worker imbalance: Uneven distribution of work
        3. Queue wait time: Time spent waiting for assignment
        """
        bottlenecks = {}

        # 1. Round barrier analysis
        if self.rounds:
            total_idle_wasted = sum(r.idle_time_wasted for r in self.rounds)
            avg_imbalance = statistics.mean(r.latency_imbalance for r in self.rounds)

            bottlenecks["round_barrier"] = {
                "description": "Time fast workers spend waiting for slow workers at round boundaries",
                "total_time_wasted_s": total_idle_wasted,
                "pct_of_session": (
                    total_idle_wasted / self.session_duration * 100
                    if self.session_duration > 0
                    else 0
                ),
                "avg_latency_imbalance_ratio": avg_imbalance,
                "severity": self._classify_severity(avg_imbalance, 1.0, 1.5, 2.0),
            }

        # 2. Worker utilization imbalance
        if self.worker_stats:
            utilizations = [s.utilization for s in self.worker_stats.values()]
            util_range = max(utilizations) - min(utilizations)
            avg_util = statistics.mean(utilizations)

            bottlenecks["worker_imbalance"] = {
                "description": "Difference in utilization across workers",
                "utilization_range_pct": util_range * 100,
                "avg_utilization_pct": avg_util * 100,
                "min_utilization_pct": min(utilizations) * 100,
                "max_utilization_pct": max(utilizations) * 100,
                "severity": self._classify_severity(util_range, 0.1, 0.2, 0.3),
            }

            # Per-worker idle time
            idle_times = {
                wid: s.total_idle_time for wid, s in self.worker_stats.items()
            }
            total_idle = sum(idle_times.values())

            bottlenecks["worker_idle_time"] = {
                "description": "Total time workers spent idle between tasks",
                "total_idle_time_s": total_idle,
                "pct_of_session": (
                    total_idle / (self.session_duration * self.n_workers) * 100
                    if self.session_duration > 0 and self.n_workers > 0
                    else 0
                ),
                "per_worker": idle_times,
            }

        # 3. Queue wait analysis
        if self.tasks:
            queue_waits = [t.queue_wait_time for t in self.tasks]
            total_queue_wait = sum(queue_waits)

            bottlenecks["queue_wait"] = {
                "description": "Time tasks spent waiting to be assigned",
                "total_queue_wait_s": total_queue_wait,
                "avg_queue_wait_ms": statistics.mean(queue_waits) * 1000,
                "max_queue_wait_ms": max(queue_waits) * 1000,
            }

        return bottlenecks

    def _classify_severity(
        self, value: float, low: float, medium: float, high: float
    ) -> str:
        """Classify severity based on thresholds."""
        if value < low:
            return "low"
        elif value < medium:
            return "medium"
        elif value < high:
            return "high"
        else:
            return "critical"

    def print_analysis(self) -> None:
        """Print formatted analysis to console."""
        analysis = self.get_analysis()

        if "error" in analysis:
            logger.warn(f"Analysis error: {analysis['error']}")
            return

        # Summary
        summary = analysis["summary"]
        logger.note("=" * 80)
        logger.note(f"[PerfTracker:{self.name}] Performance Analysis")
        logger.note("=" * 80)

        logger.mesg(f"\n📊 Summary:")
        logger.mesg(f"  Session duration: {summary['session_duration_s']:.3f}s")
        logger.mesg(f"  Total inputs: {summary['total_inputs']}")
        logger.mesg(f"  Total workers: {summary['total_workers']}")
        logger.mesg(
            f"  Total tasks: {summary['total_tasks']} (success: {summary['successful_tasks']})"
        )
        logger.mesg(f"  Total rounds: {summary['total_rounds']}")
        throughput_str = f"{summary['throughput_items_per_sec']:.1f}"
        logger.mesg(f"  Throughput: {logstr.file(throughput_str)} items/s")
        logger.mesg(f"  Avg task latency: {summary['avg_task_latency_ms']:.1f}ms")

        # Round analysis (detailed)
        round_analysis = analysis.get("round_analysis", {})
        if "rounds" in round_analysis and round_analysis["rounds"]:
            logger.mesg(f"\n📋 Round Details:")
            for rd in round_analysis["rounds"]:
                imbalance = rd["latency_imbalance"]
                idle_wasted_ms = rd["idle_time_wasted_s"] * 1000

                # Color code based on severity
                if imbalance > 1.5:
                    imb_str = logstr.warn(f"{imbalance:.2f}x")
                    waste_str = logstr.warn(f"{idle_wasted_ms:.1f}ms")
                elif imbalance > 1.2:
                    imb_str = logstr.mesg(f"{imbalance:.2f}x")
                    waste_str = logstr.mesg(f"{idle_wasted_ms:.1f}ms")
                else:
                    imb_str = logstr.okay(f"{imbalance:.2f}x")
                    waste_str = logstr.okay(f"{idle_wasted_ms:.1f}ms")

                logger.mesg(
                    f"  Round {rd['round_id']}: "
                    f"workers={rd['workers_used']}, "
                    f"items={rd['total_items']}, "
                    f"latency=[{rd['min_task_latency_ms']:.0f}, {rd['max_task_latency_ms']:.0f}]ms, "
                    f"imbalance={imb_str}, "
                    f"idle_waste={waste_str}"
                )

        # Worker analysis
        worker_analysis = analysis["worker_analysis"]
        logger.mesg(f"\n👷 Worker Analysis:")
        for wid, data in worker_analysis.items():
            util_str = f"{data['utilization_pct']:.1f}%"
            if data["utilization_pct"] < 70:
                util_str = logstr.warn(util_str)
            elif data["utilization_pct"] < 90:
                util_str = logstr.mesg(util_str)
            else:
                util_str = logstr.okay(util_str)

            logger.mesg(f"  {wid}:")
            logger.mesg(
                f"    Tasks: {data['total_tasks']}, Items: {data['total_items']}"
            )
            logger.mesg(f"    Utilization: {util_str}")
            logger.mesg(f"    Avg latency: {data['avg_latency_ms']:.1f}ms")
            logger.mesg(f"    Idle time: {data['total_idle_time_s']:.3f}s")

        # Bottleneck analysis
        bottlenecks = analysis["bottleneck_analysis"]
        logger.mesg(f"\n🔍 Bottleneck Analysis:")

        if "round_barrier" in bottlenecks:
            rb = bottlenecks["round_barrier"]
            severity_str = rb["severity"]
            if severity_str in ("high", "critical"):
                severity_str = logstr.warn(severity_str.upper())
            elif severity_str == "medium":
                severity_str = logstr.mesg(severity_str)
            else:
                severity_str = logstr.okay(severity_str)

            logger.mesg(
                f"\n  🚧 Round Barrier Synchronization (Severity: {severity_str})"
            )
            logger.mesg(f"     {rb['description']}")
            logger.mesg(
                f"     Time wasted: {rb['total_time_wasted_s']:.3f}s ({rb['pct_of_session']:.1f}% of session)"
            )
            logger.mesg(
                f"     Avg latency imbalance: {rb['avg_latency_imbalance_ratio']:.2f}x"
            )

        if "worker_imbalance" in bottlenecks:
            wi = bottlenecks["worker_imbalance"]
            severity_str = wi["severity"]
            if severity_str in ("high", "critical"):
                severity_str = logstr.warn(severity_str.upper())
            elif severity_str == "medium":
                severity_str = logstr.mesg(severity_str)
            else:
                severity_str = logstr.okay(severity_str)

            logger.mesg(
                f"\n  ⚖️  Worker Utilization Imbalance (Severity: {severity_str})"
            )
            logger.mesg(f"     {wi['description']}")
            logger.mesg(f"     Utilization range: {wi['utilization_range_pct']:.1f}%")
            logger.mesg(
                f"     Min: {wi['min_utilization_pct']:.1f}%, Max: {wi['max_utilization_pct']:.1f}%"
            )

        if "worker_idle_time" in bottlenecks:
            wit = bottlenecks["worker_idle_time"]
            logger.mesg(f"\n  ⏱️  Worker Idle Time")
            logger.mesg(f"     {wit['description']}")
            logger.mesg(
                f"     Total idle: {wit['total_idle_time_s']:.3f}s ({wit['pct_of_session']:.1f}% of worker-time)"
            )

        logger.note("=" * 80)

    def get_timeline_data(self) -> list[dict]:
        """
        Get timeline data for visualization.

        Returns list of events suitable for timeline visualization.
        """
        timeline = []

        for task in self.tasks:
            timeline.append(
                {
                    "worker_id": task.worker_id,
                    "type": "task",
                    "start": task.start_time - self.session_start,
                    "end": task.end_time - self.session_start,
                    "duration": task.latency,
                    "batch_size": task.batch_size,
                    "round_id": task.round_id,
                }
            )

        return sorted(timeline, key=lambda x: x["start"])

    def export_json(self) -> dict:
        """Export all data as JSON-serializable dict."""
        return {
            "name": self.name,
            "session": {
                "start": self.session_start,
                "end": self.session_end,
                "duration": self.session_duration,
                "n_inputs": self.n_inputs,
                "n_workers": self.n_workers,
            },
            "analysis": self.get_analysis(),
            "timeline": self.get_timeline_data(),
        }


class RoundContext:
    """Context manager for tracking a single scheduling round."""

    def __init__(
        self,
        tracker: PerfTracker,
        round_id: int,
        n_workers_available: int,
    ):
        self.tracker = tracker
        self.round_id = round_id
        self.n_workers_available = n_workers_available
        self.start_time: Optional[float] = None
        self.end_time: Optional[float] = None
        self.tasks: list[TaskRecord] = []
        self.n_workers_used = 0
        self.total_items = 0

    def __enter__(self) -> "RoundContext":
        self.start_time = time.time()

        # Create round record
        self._round = RoundRecord(
            round_id=self.round_id,
            start_time=self.start_time,
            end_time=0,  # Will be set on exit
            n_workers_used=0,
            n_workers_available=self.n_workers_available,
            total_items=0,
        )
        self.tracker._current_round = self._round

        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.end_time = time.time()

        # Update round record
        self._round.end_time = self.end_time
        self._round.n_workers_used = self.n_workers_used
        self._round.total_items = self.total_items

        # Record the round
        self.tracker.record_round(self._round)
        self.tracker._current_round = None

    def track_task(self, worker_id: str, batch_size: int) -> "TaskContext":
        """
        Create a context manager for tracking a single task.

        Args:
            worker_id: ID of the worker executing this task
            batch_size: Number of items in this batch

        Returns:
            TaskContext for tracking the task
        """
        self.n_workers_used += 1
        self.total_items += batch_size

        return TaskContext(
            round_ctx=self,
            worker_id=worker_id,
            batch_size=batch_size,
        )


class TaskContext:
    """Context manager for tracking a single task."""

    def __init__(
        self,
        round_ctx: RoundContext,
        worker_id: str,
        batch_size: int,
    ):
        self.round_ctx = round_ctx
        self.worker_id = worker_id
        self.batch_size = batch_size
        self.start_time: Optional[float] = None
        self.end_time: Optional[float] = None
        self.queue_wait_time: float = 0.0
        self.success: bool = True
        self.error: Optional[str] = None

    def __enter__(self) -> "TaskContext":
        self.start_time = time.time()

        # Calculate queue wait (time since round started)
        if self.round_ctx.start_time:
            self.queue_wait_time = self.start_time - self.round_ctx.start_time

        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.end_time = time.time()

        if exc_type is not None:
            self.success = False
            self.error = str(exc_val)

        # Record the task
        self.round_ctx.tracker.record_task(
            worker_id=self.worker_id,
            round_id=self.round_ctx.round_id,
            batch_size=self.batch_size,
            start_time=self.start_time,
            end_time=self.end_time,
            queue_wait_time=self.queue_wait_time,
            success=self.success,
            error=self.error,
        )

    def mark_error(self, error: str) -> None:
        """Mark this task as failed with an error message."""
        self.success = False
        self.error = error


# Global tracker instance for convenience
_global_tracker: Optional[PerfTracker] = None


def get_global_tracker() -> PerfTracker:
    """Get or create the global performance tracker."""
    global _global_tracker
    if _global_tracker is None:
        _global_tracker = PerfTracker(name="global", verbose=True)
    return _global_tracker


def reset_global_tracker() -> None:
    """Reset the global performance tracker."""
    global _global_tracker
    if _global_tracker:
        _global_tracker.reset()
