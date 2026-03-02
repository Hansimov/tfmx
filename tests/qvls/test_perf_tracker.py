"""Tests for tfmx.qvls.perf_tracker module (re-exports from teis)"""

import pytest

from tfmx.qvls.perf_tracker import (
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


class TestPerfTrackerImports:
    """Verify that re-exported classes are usable."""

    def test_perf_tracker_init(self):
        tracker = PerfTracker(name="test_qvl", verbose=False)
        assert tracker.name == "test_qvl"

    def test_global_tracker(self):
        reset_global_tracker()
        tracker = get_global_tracker()
        assert tracker is not None

    def test_worker_event_fields(self):
        # WorkerEvent is a dataclass with event_type field, not an enum
        import time

        event = WorkerEvent(
            worker_id="w0",
            event_type="task_start",
            timestamp=time.time(),
        )
        assert event.worker_id == "w0"
        assert event.event_type == "task_start"

    def test_round_record(self):
        import time

        t = time.time()
        record = RoundRecord(
            round_id=0,
            start_time=t,
            end_time=t + 1.0,
            n_workers_used=2,
            n_workers_available=3,
            total_items=100,
        )
        assert record.round_id == 0
        assert record.duration == pytest.approx(1.0, abs=0.01)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
