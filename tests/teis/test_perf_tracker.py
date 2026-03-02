"""Tests for tfmx.teis.perf_tracker module"""

import pytest
import time

from tfmx.teis.perf_tracker import (
    WorkerEvent,
    TaskRecord,
    RoundRecord,
    WorkerStats,
    PerfTracker,
    get_global_tracker,
    reset_global_tracker,
)


class TestWorkerEvent:
    """Test WorkerEvent dataclass"""

    def test_creation(self):
        event = WorkerEvent(
            worker_id="w0",
            event_type="task_start",
            timestamp=1.0,
            batch_size=10,
        )
        assert event.worker_id == "w0"
        assert event.event_type == "task_start"
        assert event.timestamp == 1.0
        assert event.batch_size == 10

    def test_defaults(self):
        event = WorkerEvent(worker_id="w0", event_type="idle_start", timestamp=0.0)
        assert event.batch_size == 0


class TestTaskRecord:
    """Test TaskRecord computed properties"""

    def test_latency(self):
        record = TaskRecord(
            worker_id="w0",
            round_id=0,
            batch_size=100,
            start_time=1.0,
            end_time=2.0,
        )
        assert record.latency == pytest.approx(1.0)

    def test_throughput(self):
        record = TaskRecord(
            worker_id="w0",
            round_id=0,
            batch_size=100,
            start_time=1.0,
            end_time=2.0,
        )
        assert record.throughput == pytest.approx(100.0)

    def test_zero_latency_throughput(self):
        record = TaskRecord(
            worker_id="w0",
            round_id=0,
            batch_size=100,
            start_time=1.0,
            end_time=1.0,
        )
        assert record.latency == 0.0
        assert record.throughput == 0.0


class TestRoundRecord:
    """Test RoundRecord computed properties"""

    def test_duration(self):
        record = RoundRecord(
            round_id=0,
            start_time=1.0,
            end_time=3.0,
            n_workers_used=2,
            n_workers_available=4,
            total_items=200,
            tasks=[],
        )
        assert record.duration == pytest.approx(2.0)

    def test_with_tasks(self):
        tasks = [
            TaskRecord(
                worker_id="w0", round_id=0, batch_size=100, start_time=1.0, end_time=1.5
            ),
            TaskRecord(
                worker_id="w1", round_id=0, batch_size=100, start_time=1.0, end_time=2.0
            ),
        ]
        record = RoundRecord(
            round_id=0,
            start_time=1.0,
            end_time=2.0,
            n_workers_used=2,
            n_workers_available=2,
            total_items=200,
            tasks=tasks,
        )
        assert record.avg_task_latency == pytest.approx(0.75)
        assert record.min_task_latency == pytest.approx(0.5)
        assert record.max_task_latency == pytest.approx(1.0)


class TestPerfTracker:
    """Test PerfTracker basic operations"""

    def test_creation(self):
        tracker = PerfTracker(name="test")
        assert tracker.name == "test"

    def test_reset(self):
        tracker = PerfTracker(name="test")
        tracker.reset()
        # Should not raise

    def test_session_lifecycle(self):
        tracker = PerfTracker(name="test")
        tracker.start_session(n_inputs=100, n_workers=2)
        tracker.end_session()
        assert tracker.session_duration >= 0


class TestGlobalTracker:
    """Test global tracker management"""

    def test_get_global_tracker(self):
        tracker = get_global_tracker()
        assert tracker is not None
        assert isinstance(tracker, PerfTracker)

    def test_reset_global_tracker(self):
        reset_global_tracker()
        tracker = get_global_tracker()
        assert tracker is not None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
