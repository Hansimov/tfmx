"""Tests for tfmx.qvls.clients_core module"""

import asyncio
import time
import pytest
from unittest.mock import patch, MagicMock, AsyncMock, PropertyMock

from tfmx.qvls.clients_core import (
    MachineState,
    MachineScheduler,
    ClientsHealthResponse,
    HealthRecoveryManager,
)
from tfmx.qvls.compose import MAX_CONCURRENT_REQUESTS


# ── MachineState ─────────────────────────────────────────────────────


class TestMachineState:
    """Test MachineState dataclass."""

    def _make(self, **kw) -> MachineState:
        defaults = dict(
            endpoint="http://localhost:29800",
            client=MagicMock(),
        )
        defaults.update(kw)
        return MachineState(**defaults)

    def test_defaults(self):
        m = self._make()
        assert m.healthy is False
        assert m._active_requests == 0
        assert m._max_concurrent == MAX_CONCURRENT_REQUESTS
        assert m._consecutive_failures == 0

    def test_is_idle_no_requests(self):
        m = self._make()
        assert m.is_idle is True

    def test_is_idle_at_max(self):
        m = self._make()
        m._active_requests = MAX_CONCURRENT_REQUESTS
        assert m.is_idle is False

    def test_active_requests_property(self):
        m = self._make()
        m._active_requests = 3
        assert m.active_requests == 3

    def test_available_slots(self):
        m = self._make()
        m._active_requests = 2
        assert m.available_slots == MAX_CONCURRENT_REQUESTS - 2

    def test_available_slots_negative_clamped(self):
        m = self._make()
        m._active_requests = MAX_CONCURRENT_REQUESTS + 5
        assert m.available_slots == 0

    def test_weight_healthy(self):
        m = self._make(healthy=True)
        m.healthy_instances = 6
        assert m.weight == 6

    def test_weight_unhealthy(self):
        m = self._make(healthy=False)
        m.healthy_instances = 6
        assert m.weight == 0

    def test_capacity_from_config(self):
        m = self._make(healthy=True)
        m._config_throughput = 100.0
        m._config_instances = 6
        m.healthy_instances = 6
        assert m.capacity == 100.0

    def test_capacity_scaled(self):
        m = self._make(healthy=True)
        m._config_throughput = 100.0
        m._config_instances = 6
        m.healthy_instances = 3
        assert m.capacity == 50.0  # 100 * 3/6

    def test_capacity_from_max_concurrent(self):
        m = self._make(healthy=True)
        # No config, falls back to max_concurrent
        assert m.capacity == float(MAX_CONCURRENT_REQUESTS)

    def test_capacity_unhealthy_zero(self):
        m = self._make(healthy=False)
        assert m.capacity == 0.0

    def test_mark_busy(self):
        m = self._make()
        m.mark_busy()
        assert m._active_requests == 1
        m.mark_busy()
        assert m._active_requests == 2

    def test_mark_idle(self):
        m = self._make()
        m._active_requests = 3
        m.mark_idle()
        assert m._active_requests == 2

    def test_mark_idle_floor_zero(self):
        m = self._make()
        m._active_requests = 0
        m.mark_idle()
        assert m._active_requests == 0

    def test_record_failure(self):
        m = self._make()
        m.record_failure()
        assert m._consecutive_failures == 1
        assert m._last_failure_time > 0

    def test_record_success(self):
        m = self._make()
        m._consecutive_failures = 5
        m.record_success()
        assert m._consecutive_failures == 0


# ── MachineScheduler ─────────────────────────────────────────────────


class TestMachineScheduler:
    """Test MachineScheduler request distribution."""

    def _make_machines(self, n=3, healthy=True) -> list[MachineState]:
        return [
            MachineState(
                endpoint=f"http://m{i}:29800",
                client=MagicMock(),
                healthy=healthy,
            )
            for i in range(n)
        ]

    def test_get_healthy_machines(self):
        machines = self._make_machines(3)
        machines[1].healthy = False
        sched = MachineScheduler(machines)
        healthy = sched.get_healthy_machines()
        assert len(healthy) == 2
        assert machines[1] not in healthy

    def test_get_idle_machine(self):
        machines = self._make_machines(2)
        sched = MachineScheduler(machines)
        m = sched.get_idle_machine()
        assert m is not None

    def test_get_idle_machine_most_slots(self):
        machines = self._make_machines(2)
        machines[0]._active_requests = 3
        machines[1]._active_requests = 1
        sched = MachineScheduler(machines)
        m = sched.get_idle_machine()
        assert m.endpoint == machines[1].endpoint

    def test_get_idle_machine_none_idle(self):
        machines = self._make_machines(2)
        for m in machines:
            m._active_requests = MAX_CONCURRENT_REQUESTS
        sched = MachineScheduler(machines)
        assert sched.get_idle_machine() is None

    def test_get_idle_machine_all_unhealthy(self):
        machines = self._make_machines(2, healthy=False)
        sched = MachineScheduler(machines)
        assert sched.get_idle_machine() is None

    def test_signal_idle(self):
        machines = self._make_machines(1)
        sched = MachineScheduler(machines)
        sched._idle_event.clear()
        sched.signal_idle()
        assert sched._idle_event.is_set()

    def test_get_total_capacity(self):
        machines = self._make_machines(3)
        sched = MachineScheduler(machines)
        # No config → capacity = MAX_CONCURRENT_REQUESTS per machine
        total = sched.get_total_capacity()
        assert total == float(MAX_CONCURRENT_REQUESTS) * 3

    def test_get_total_capacity_custom_healthy(self):
        machines = self._make_machines(3)
        machines[2].healthy = False
        sched = MachineScheduler(machines)
        total = sched.get_total_capacity(healthy=[machines[0]])
        assert total == float(MAX_CONCURRENT_REQUESTS)


# ── ClientsHealthResponse ────────────────────────────────────────────


class TestClientsHealthResponse:
    """Test ClientsHealthResponse."""

    def test_from_machines_healthy(self):
        machines = [
            MagicMock(healthy=True, healthy_instances=3, total_instances=3),
            MagicMock(healthy=True, healthy_instances=6, total_instances=6),
        ]
        resp = ClientsHealthResponse.from_machines(machines)
        assert resp.status == "healthy"
        assert resp.healthy_machines == 2
        assert resp.total_machines == 2
        assert resp.healthy_instances == 9
        assert resp.total_instances == 9

    def test_from_machines_none_healthy(self):
        machines = [
            MagicMock(healthy=False, healthy_instances=0, total_instances=3),
        ]
        resp = ClientsHealthResponse.from_machines(machines)
        assert resp.status == "unhealthy"
        assert resp.healthy_machines == 0

    def test_from_machines_partial(self):
        machines = [
            MagicMock(healthy=True, healthy_instances=6, total_instances=6),
            MagicMock(healthy=False, healthy_instances=0, total_instances=6),
        ]
        resp = ClientsHealthResponse.from_machines(machines)
        assert resp.status == "healthy"
        assert resp.healthy_machines == 1
        assert resp.healthy_instances == 6
        assert resp.total_instances == 12


# ── HealthRecoveryManager ────────────────────────────────────────────


class TestHealthRecoveryManager:
    """Test HealthRecoveryManager logic."""

    def _make_machine(self, healthy=True) -> MachineState:
        return MachineState(
            endpoint="http://localhost:29800",
            client=MagicMock(),
            healthy=healthy,
        )

    def test_get_unhealthy_machines(self):
        m1 = self._make_machine(healthy=True)
        m2 = self._make_machine(healthy=False)
        mgr = HealthRecoveryManager([m1, m2], health_check_fn=MagicMock())
        unhealthy = mgr.get_unhealthy_machines()
        assert len(unhealthy) == 1
        assert unhealthy[0] is m2

    def test_get_unhealthy_machines_none(self):
        m1 = self._make_machine(healthy=True)
        mgr = HealthRecoveryManager([m1], health_check_fn=MagicMock())
        assert mgr.get_unhealthy_machines() == []

    def test_constants(self):
        assert HealthRecoveryManager.RECOVERY_CHECK_INTERVAL == 10.0
        assert HealthRecoveryManager.MAX_FAILURES_BEFORE_BACKOFF == 3
        assert HealthRecoveryManager.BACKOFF_MULTIPLIER == 2.0
