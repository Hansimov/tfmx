"""Tests for tfmx.qvls.clients_stats module"""

import pytest
from io import StringIO
from unittest.mock import patch, MagicMock

from tfmx.qvls.clients_stats import QVLClientsWithStats, _log
from tfmx.qvls.clients_core import _QVLClientsBase, ClientsHealthResponse
import tfmx.qvls.clients_stats as clients_stats_mod


@pytest.fixture
def capture_log(monkeypatch):
    """Capture _log output by monkeypatching the module-level function."""
    captured = []

    def _fake_log(msg, **kwargs):
        captured.append(msg)

    monkeypatch.setattr(clients_stats_mod, "_log", _fake_log)
    return captured


class TestLog:
    """Test the _log helper."""

    def test_log_to_stderr(self):
        buf = StringIO()
        _log("test message", file=buf)
        assert "test message" in buf.getvalue()

    def test_log_custom_file(self):
        buf = StringIO()
        _log("custom output", file=buf)
        assert "custom output" in buf.getvalue()


class TestQVLClientsWithStatsInit:
    """Test QVLClientsWithStats initialization."""

    @patch.object(_QVLClientsBase, "_load_config")
    def test_verbose_default_true(self, mock_config):
        c = QVLClientsWithStats(["http://host1:29800"])
        assert c._verbose is True

    @patch.object(_QVLClientsBase, "_load_config")
    def test_verbose_explicit_false(self, mock_config):
        c = QVLClientsWithStats(["http://host1:29800"], verbose=False)
        assert c._verbose is False

    @patch.object(_QVLClientsBase, "_load_config")
    def test_repr(self, mock_config):
        c = QVLClientsWithStats(["http://host1:29800"])
        r = repr(c)
        assert "QVLClientsWithStats" in r
        assert "machines=1" in r

    @patch.object(_QVLClientsBase, "_load_config")
    def test_pipeline_has_callbacks(self, mock_config):
        c = QVLClientsWithStats(["http://host1:29800"])
        assert c._pipeline.on_progress is not None
        assert c._pipeline.on_complete is not None


class TestOnProgress:
    """Test the _on_progress callback."""

    @patch.object(_QVLClientsBase, "_load_config")
    def test_progress_logged_when_verbose(self, mock_config, capture_log):
        c = QVLClientsWithStats(["http://host1:29800"], verbose=True)
        c._on_progress(completed=50, total=100, elapsed=5.0, machine_stats={})
        output = "\n".join(capture_log)
        assert "50/100" in output
        assert "50%" in output
        assert "req/s" in output

    @patch.object(_QVLClientsBase, "_load_config")
    def test_progress_silent_when_not_verbose(self, mock_config, capture_log):
        c = QVLClientsWithStats(["http://host1:29800"], verbose=False)
        c._on_progress(completed=50, total=100, elapsed=5.0, machine_stats={})
        assert len(capture_log) == 0

    @patch.object(_QVLClientsBase, "_load_config")
    def test_progress_with_machine_stats(self, mock_config, capture_log):
        c = QVLClientsWithStats(["http://host1:29800"], verbose=True)
        stats = {"m1": {"host": "host1", "items": 25}}
        c._on_progress(completed=25, total=100, elapsed=2.5, machine_stats=stats)
        output = "\n".join(capture_log)
        assert "host1" in output

    @patch.object(_QVLClientsBase, "_load_config")
    def test_progress_zero_elapsed(self, mock_config, capture_log):
        """Handles zero elapsed time without division error."""
        c = QVLClientsWithStats(["http://host1:29800"], verbose=True)
        c._on_progress(completed=0, total=100, elapsed=0.0, machine_stats={})
        output = "\n".join(capture_log)
        assert "0/100" in output


class TestOnComplete:
    """Test the _on_complete callback."""

    @patch.object(_QVLClientsBase, "_load_config")
    def test_complete_logged_when_verbose(self, mock_config, capture_log):
        c = QVLClientsWithStats(["http://host1:29800"], verbose=True)
        c._on_complete(total_responses=90, total_requests=100, total_time=10.0)
        output = "\n".join(capture_log)
        assert "90/100" in output
        assert "10.00s" in output
        assert "req/s" in output

    @patch.object(_QVLClientsBase, "_load_config")
    def test_complete_silent_when_not_verbose(self, mock_config, capture_log):
        c = QVLClientsWithStats(["http://host1:29800"], verbose=False)
        c._on_complete(total_responses=90, total_requests=100, total_time=10.0)
        assert len(capture_log) == 0

    @patch.object(_QVLClientsBase, "_load_config")
    def test_complete_zero_responses(self, mock_config, capture_log):
        c = QVLClientsWithStats(["http://host1:29800"], verbose=True)
        c._on_complete(total_responses=0, total_requests=10, total_time=1.0)
        output = "\n".join(capture_log)
        assert "0/10" in output


class TestRefreshHealth:
    """Test verbose health refresh."""

    @patch.object(_QVLClientsBase, "_load_config")
    def test_refresh_health_verbose(self, mock_config, capture_log):
        c = QVLClientsWithStats(["http://host1:29800"], verbose=True)
        c.machines[0].healthy = True
        c.machines[0].healthy_instances = 4
        c.machines[0].total_instances = 4
        resp = c.refresh_health()
        output = "\n".join(capture_log)
        assert "Checking health" in output
        assert "host1" in output

    @patch.object(_QVLClientsBase, "_load_config")
    def test_refresh_health_silent(self, mock_config, capture_log):
        c = QVLClientsWithStats(["http://host1:29800"], verbose=False)
        c.machines[0].healthy = True
        resp = c.refresh_health()
        assert len(capture_log) == 0
