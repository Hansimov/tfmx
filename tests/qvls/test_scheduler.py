"""Tests for tfmx.qvls.scheduler module (re-exports from teis)"""

import pytest

from tfmx.qvls.scheduler import (
    WorkerState,
    IdleFillingScheduler,
    DistributionResult,
)


class TestWorkerState:
    """Test WorkerState dataclass."""

    def test_default_state(self):
        state = WorkerState(worker_id="w0")
        assert state.is_idle is True
        assert state.total_requests == 0
        assert state.total_items == 0
        assert state.total_errors == 0

    def test_mark_busy(self):
        state = WorkerState(worker_id="w0")
        state.mark_busy()
        assert state.is_idle is False

    def test_mark_idle(self):
        state = WorkerState(worker_id="w0")
        state.mark_busy()
        state.mark_idle()
        assert state.is_idle is True

    def test_record_success(self):
        state = WorkerState(worker_id="w0")
        state.record_success(latency=1.0, n_items=50)
        assert state.total_requests == 1
        assert state.total_items == 50
        assert state.throughput > 0

    def test_record_error(self):
        state = WorkerState(worker_id="w0")
        state.record_error()
        assert state.total_errors == 1


class TestIdleFillingScheduler:
    """Test IdleFillingScheduler."""

    def _make_scheduler(self, n_workers=3):
        workers = [f"worker_{i}" for i in range(n_workers)]
        scheduler = IdleFillingScheduler(
            workers=workers,
            get_worker_id=lambda w: w,
            max_batch_size=8,
        )
        return scheduler, workers

    def test_init(self):
        scheduler, workers = self._make_scheduler(3)
        assert len(scheduler.workers) == 3

    def test_all_idle_initially(self):
        scheduler, workers = self._make_scheduler(3)
        idle = scheduler.get_idle_workers()
        assert len(idle) == 3

    def test_mark_busy_reduces_idle(self):
        scheduler, workers = self._make_scheduler(3)
        state = scheduler.get_state(workers[0])
        state.mark_busy()
        idle = scheduler.get_idle_workers()
        assert len(idle) == 2

    def test_select_idle_worker(self):
        scheduler, workers = self._make_scheduler(3)
        result = scheduler.select_idle_worker()
        assert result is not None

    def test_no_idle_workers(self):
        scheduler, workers = self._make_scheduler(2)
        for w in workers:
            scheduler.get_state(w).mark_busy()
        result = scheduler.select_idle_worker()
        assert result is None


class TestDistributionResult:
    """Test DistributionResult dataclass."""

    def test_success(self):
        result = DistributionResult(
            worker_id="w0",
            start_idx=0,
            end_idx=3,
            result=[1, 2, 3],
            error=None,
        )
        assert result.success is True

    def test_failure(self):
        result = DistributionResult(
            worker_id="w0",
            start_idx=0,
            end_idx=0,
            result=None,
            error=Exception("fail"),
        )
        assert result.success is False


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
