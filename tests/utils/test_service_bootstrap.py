"""Tests for backend startup helpers."""

from types import SimpleNamespace
from unittest.mock import patch

from tfmx.utils.service_bootstrap import PortConflict
from tfmx.utils.service_bootstrap import handle_port_conflicts
from tfmx.utils.service_bootstrap import wait_for_available_backend_instances


class _FakeHttpClient:
    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False

    def get(self, url):
        if url.endswith(":27880/health"):
            return SimpleNamespace(status_code=200)
        raise RuntimeError("backend unavailable")


class TestWaitForAvailableBackendInstances:
    @patch("tfmx.utils.service_bootstrap.time.sleep", return_value=None)
    @patch("tfmx.utils.service_bootstrap.time.monotonic", side_effect=[0.0] * 8)
    @patch(
        "tfmx.utils.service_bootstrap.httpx.Client",
        return_value=_FakeHttpClient(),
    )
    def test_returns_partially_healthy_instances(
        self,
        _mock_client,
        _mock_monotonic,
        _mock_sleep,
    ):
        instances = [
            SimpleNamespace(
                endpoint="http://localhost:27880",
                health_url="http://localhost:27880/health",
                healthy=False,
            ),
            SimpleNamespace(
                endpoint="http://localhost:27881",
                health_url="http://localhost:27881/health",
                healthy=False,
            ),
        ]

        returned = wait_for_available_backend_instances(
            lambda: instances,
            timeout_sec=10.0,
            poll_interval_sec=1.0,
            settle_sec=0.0,
            label="[test]",
        )

        assert returned == instances
        assert instances[0].healthy is True
        assert instances[1].healthy is False


class TestHandlePortConflicts:
    @patch("tfmx.utils.service_bootstrap.find_port_conflicts")
    def test_report_policy_rejects_existing_listener(self, mock_find):
        mock_find.return_value = [
            PortConflict(port=27800, pid=1234, process_name="python")
        ]

        handled = handle_port_conflicts(27800, policy="report", label="[test]")

        assert handled is False

    @patch("tfmx.utils.service_bootstrap.terminate_process", return_value=True)
    @patch("tfmx.utils.service_bootstrap.find_port_conflicts")
    def test_replace_policy_terminates_listener(self, mock_find, mock_terminate):
        mock_find.side_effect = [
            [PortConflict(port=27800, pid=1234, process_name="python")],
            [],
        ]

        handled = handle_port_conflicts(27800, policy="replace", label="[test]")

        assert handled is True
        mock_terminate.assert_called_once_with(1234, label="[test]")
