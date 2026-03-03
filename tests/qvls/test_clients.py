"""Tests for tfmx.qvls.clients module"""

import pytest
from unittest.mock import patch, MagicMock, PropertyMock

from tfmx.qvls.clients import QVLClients
from tfmx.qvls.clients_core import MachineState, _QVLClientsBase


class TestQVLClientsInit:
    """Test QVLClients initialization."""

    @patch.object(_QVLClientsBase, "_load_config")
    def test_single_endpoint(self, mock_config):
        clients = QVLClients(["http://host1:29800"])
        assert len(clients.machines) == 1
        assert clients.machines[0].endpoint == "http://host1:29800"

    @patch.object(_QVLClientsBase, "_load_config")
    def test_multiple_endpoints(self, mock_config):
        endpoints = ["http://host1:29800", "http://host2:29800", "http://host3:29800"]
        clients = QVLClients(endpoints)
        assert len(clients.machines) == 3

    @patch.object(_QVLClientsBase, "_load_config")
    def test_repr(self, mock_config):
        clients = QVLClients(["http://host1:29800"])
        r = repr(clients)
        assert "QVLClients" in r
        assert "machines=1" in r

    @patch.object(_QVLClientsBase, "_load_config")
    def test_pipeline_created(self, mock_config):
        clients = QVLClients(["http://host1:29800"])
        assert clients._pipeline is not None


class TestQVLClientsHealthResponse:
    """Test health checking via QVLClients."""

    @patch.object(_QVLClientsBase, "_load_config")
    def test_health_returns_response(self, mock_config):
        """health() returns a ClientsHealthResponse."""
        clients = QVLClients(["http://localhost:29800"])
        # Mock the machine health check
        clients.machines[0].healthy = True
        clients.machines[0].healthy_instances = 4
        clients.machines[0].total_instances = 4
        resp = clients.health()
        assert resp.total_machines == 1
        assert resp.status in ("healthy", "degraded", "unhealthy")

    @patch.object(_QVLClientsBase, "_load_config")
    def test_health_all_unhealthy(self, mock_config):
        clients = QVLClients(["http://h1:29800", "http://h2:29800"])
        for m in clients.machines:
            m.healthy = False
        resp = clients.health()
        assert resp.healthy_machines == 0
        assert resp.status == "unhealthy"


class TestQVLClientsContextManager:
    """Test context manager protocol."""

    @patch.object(_QVLClientsBase, "_load_config")
    def test_enter_exit(self, mock_config):
        clients = QVLClients(["http://host1:29800"])
        with clients as c:
            assert c is clients


class TestQVLClientsMachineScheduler:
    """Test machine scheduler integration."""

    @patch.object(_QVLClientsBase, "_load_config")
    def test_scheduler_initialized(self, mock_config):
        clients = QVLClients(["http://h1:29800", "http://h2:29800"])
        assert clients.machine_scheduler is not None
        assert len(clients.machine_scheduler.machines) == 2

    @patch.object(_QVLClientsBase, "_load_config")
    def test_scheduler_get_idle(self, mock_config):
        clients = QVLClients(["http://h1:29800"])
        clients.machines[0].healthy = True
        idle = clients.machine_scheduler.get_idle_machine()
        # Should return the only machine since it's idle
        assert idle is not None
        assert idle.endpoint == "http://h1:29800"

    @patch.object(_QVLClientsBase, "_load_config")
    def test_scheduler_no_healthy(self, mock_config):
        clients = QVLClients(["http://h1:29800"])
        clients.machines[0].healthy = False
        idle = clients.machine_scheduler.get_idle_machine()
        assert idle is None
