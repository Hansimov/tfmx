"""Tests for tfmx.teis.scheduler module"""

import pytest
import asyncio

from tfmx.teis.scheduler import (
    WorkerState,
    IdleFillingScheduler,
    DistributionResult,
)


class TestWorkerState:
    """Test WorkerState dataclass"""

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
        state.record_success(latency=1.0, n_items=100)
        assert state.total_requests == 1
        assert state.total_items == 100
        assert state.throughput > 0

    def test_record_error(self):
        state = WorkerState(worker_id="w0")
        state.record_error()
        assert state.total_errors == 1

    def test_throughput_ema(self):
        """Throughput should use EMA (exponential moving average)"""
        state = WorkerState(worker_id="w0")
        state.record_success(latency=1.0, n_items=100)  # 100/s
        tp1 = state.throughput
        state.record_success(latency=1.0, n_items=200)  # 200/s
        tp2 = state.throughput
        # EMA should increase but not jump to 200
        assert tp2 > tp1
        assert tp2 < 200

    def test_to_dict(self):
        state = WorkerState(worker_id="w0")
        d = state.to_dict()
        assert isinstance(d, dict)
        assert "worker_id" in d


class TestIdleFillingScheduler:
    """Test IdleFillingScheduler with fake workers"""

    def _make_scheduler(self, n_workers=3):
        workers = [f"worker_{i}" for i in range(n_workers)]
        scheduler = IdleFillingScheduler(
            workers=workers,
            get_worker_id=lambda w: w,
            max_batch_size=300,
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
        worker, state = result
        assert worker in workers

    def test_no_idle_workers(self):
        scheduler, workers = self._make_scheduler(2)
        for w in workers:
            scheduler.get_state(w).mark_busy()
        result = scheduler.select_idle_worker()
        assert result is None

    def test_get_worker_by_id(self):
        scheduler, workers = self._make_scheduler(3)
        w = scheduler.get_worker_by_id("worker_1")
        assert w == "worker_1"

    def test_get_worker_by_id_not_found(self):
        scheduler, workers = self._make_scheduler(3)
        w = scheduler.get_worker_by_id("nonexistent")
        assert w is None

    def test_update_workers(self):
        scheduler, workers = self._make_scheduler(2)
        new_workers = ["worker_0", "worker_1", "worker_2"]
        scheduler.update_workers(new_workers)
        assert len(scheduler.workers) == 3

    def test_get_stats_summary(self):
        scheduler, workers = self._make_scheduler(3)
        stats = scheduler.get_stats_summary()
        assert isinstance(stats, dict)
        # stats is a dict of worker_id -> worker_stats
        assert len(stats) == 3
        assert "worker_0" in stats

    def test_idle_workers_by_throughput(self):
        scheduler, workers = self._make_scheduler(3)
        # Record different throughputs
        s0 = scheduler.get_state(workers[0])
        s1 = scheduler.get_state(workers[1])
        s2 = scheduler.get_state(workers[2])
        s0.record_success(1.0, 100)  # 100/s
        s1.record_success(1.0, 200)  # 200/s
        s2.record_success(1.0, 50)  # 50/s
        s0.mark_idle()
        s1.mark_idle()
        s2.mark_idle()

        by_tp = scheduler.get_idle_workers_by_throughput()
        # Should be sorted by throughput descending
        tps = [state.throughput for _, state in by_tp]
        assert tps == sorted(tps, reverse=True)


class TestDistributionResult:
    """Test DistributionResult dataclass"""

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
            error=Exception("Something went wrong"),
        )
        assert result.success is False


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
