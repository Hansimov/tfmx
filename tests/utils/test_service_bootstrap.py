"""Tests for backend startup helpers."""

from types import SimpleNamespace
from unittest.mock import patch

from tfmx.utils.service_bootstrap import PortConflict
from tfmx.utils.service_bootstrap import docker_status_to_health
from tfmx.utils.service_bootstrap import wait_for_healthy_docker_containers
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

    @patch("tfmx.utils.service_bootstrap.time.monotonic", side_effect=[0.0] * 6)
    def test_prefers_docker_health_metadata_when_available(self, _mock_monotonic):
        instances = [
            SimpleNamespace(
                endpoint="http://localhost:27880",
                health_url="http://localhost:27880/health",
                healthy=False,
                docker_health=True,
            ),
        ]

        returned = wait_for_available_backend_instances(
            lambda: instances,
            timeout_sec=1.0,
            poll_interval_sec=1.0,
            settle_sec=0.0,
            label="[test]",
        )

        assert returned == instances
        assert instances[0].healthy is True


class TestDockerHealthHelpers:
    def test_docker_status_to_health(self):
        assert docker_status_to_health("Up 10 seconds (healthy)") is True
        assert docker_status_to_health("Up 5 seconds (health: starting)") is None
        assert docker_status_to_health("Exited (1) 2 seconds ago") is False
        assert docker_status_to_health("Up 1 minute") is None

    @patch("tfmx.utils.service_bootstrap.time.sleep", return_value=None)
    @patch(
        "tfmx.utils.service_bootstrap.get_docker_container_statuses",
        side_effect=[
            {"qwn--gpu0": "Up 2 seconds (health: starting)"},
            {"qwn--gpu0": "Up 8 seconds (healthy)"},
        ],
    )
    @patch(
        "tfmx.utils.service_bootstrap.time.monotonic",
        side_effect=[0.0, 0.0, 1.0, 1.0],
    )
    def test_wait_for_healthy_docker_containers(
        self,
        _mock_monotonic,
        _mock_statuses,
        _mock_sleep,
    ):
        assert (
            wait_for_healthy_docker_containers(
                ["qwn--gpu0"],
                timeout_sec=10.0,
                poll_interval_sec=1.0,
                label="[test]",
            )
            is True
        )


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
