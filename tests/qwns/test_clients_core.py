"""Tests for tfmx.qwns.clients_core."""

from unittest.mock import MagicMock

from tfmx.qwns.clients_core import ClientsHealthResponse
from tfmx.qwns.clients_core import MachineScheduler
from tfmx.qwns.clients_core import MachineState
from tfmx.qwns.compose import MAX_CONCURRENT_REQUESTS


class TestMachineState:
    def _make(self, **kwargs) -> MachineState:
        params = {
            "endpoint": "http://localhost:27800",
            "client": MagicMock(),
        }
        params.update(kwargs)
        return MachineState(**params)

    def test_capacity_healthy(self):
        machine = self._make(healthy=True)
        assert machine.capacity == float(MAX_CONCURRENT_REQUESTS)

    def test_mark_busy_idle(self):
        machine = self._make()
        machine.mark_busy()
        machine.mark_idle()
        assert machine.active_requests == 0


class TestMachineScheduler:
    def test_get_idle_machine(self):
        machines = [
            MachineState(endpoint="http://m0", client=MagicMock(), healthy=True),
            MachineState(endpoint="http://m1", client=MagicMock(), healthy=True),
        ]
        scheduler = MachineScheduler(machines)
        assert scheduler.get_idle_machine() is not None


class TestClientsHealthResponse:
    def test_from_machines(self):
        machines = [
            MagicMock(healthy=True, healthy_instances=1, total_instances=1),
            MagicMock(healthy=False, healthy_instances=0, total_instances=1),
        ]
        response = ClientsHealthResponse.from_machines(machines)
        assert response.healthy_machines == 1
        assert response.total_instances == 2
