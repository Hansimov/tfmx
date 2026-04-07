"""Tests for qwn machine auto-start helpers."""

from argparse import Namespace
from unittest.mock import patch

from tfmx.qwns.machine import QWNInstance, QWNMachineServer, discover_instances
from tfmx.qwns.machine import run_from_args


class TestQWNMachineAutoStart:
    @patch("tfmx.utils.service_bootstrap.wait_for_available_backend_instances")
    @patch("tfmx.qwns.machine.QWNComposer")
    @patch("tfmx.qwns.machine.QWNInstanceDiscovery.discover")
    def test_discover_instances_auto_starts_backends(
        self,
        mock_discover,
        mock_composer,
        mock_wait_for_available,
    ):
        discovered_instances = [
            QWNInstance(
                container_name="qwn--gpu0",
                host="localhost",
                port=27880,
                gpu_id=0,
            )
        ]
        mock_discover.return_value = []
        mock_wait_for_available.return_value = discovered_instances
        composer = mock_composer.return_value
        composer.get_backend_endpoints.return_value = ["http://localhost:27880"]

        args = Namespace(
            endpoints=None,
            name_pattern=None,
            auto_start=True,
            startup_timeout=15.0,
            startup_poll_interval=1.0,
            compose_model_name=None,
            compose_port=None,
            compose_project_name=None,
            compose_gpus=None,
            compose_gpu_layout=None,
            compose_gpu_configs=None,
        )

        instances = discover_instances(args)

        mock_composer.assert_called_once_with()
        composer.up.assert_called_once_with()
        composer.wait_for_healthy_backends.assert_not_called()
        mock_wait_for_available.assert_called_once()
        assert mock_wait_for_available.call_args.kwargs == {
            "timeout_sec": 15.0,
            "poll_interval_sec": 1.0,
            "settle_sec": 10.0,
            "label": "[qwn_machine]",
        }
        assert [instance.port for instance in instances] == [27880]

    def test_machine_server_merges_newly_discovered_instances(self):
        first = QWNInstance(
            container_name="qwn--gpu0",
            host="localhost",
            port=27880,
            gpu_id=0,
        )
        second = QWNInstance(
            container_name="qwn--gpu1",
            host="localhost",
            port=27881,
            gpu_id=1,
        )

        server = QWNMachineServer(instances=[first], port=27800, timeout=1.0)

        changed, added_instances = server._merge_discovered_instances([first, second])

        assert changed is True
        assert [instance.container_name for instance in added_instances] == [
            "qwn--gpu1"
        ]
        assert [instance.gpu_id for instance in server.instances] == [0, 1]

    @patch("tfmx.qwns.machine.handle_port_conflicts", return_value=True)
    @patch("tfmx.qwns.machine.discover_instances")
    @patch("tfmx.qwns.machine.QWNMachineDaemon")
    def test_run_from_args_background_replace_existing_daemon(
        self,
        mock_daemon_cls,
        mock_discover_instances,
        mock_handle_conflicts,
    ):
        instance = QWNInstance(
            container_name="qwn--gpu0",
            host="localhost",
            port=27880,
            gpu_id=0,
        )
        mock_discover_instances.return_value = [instance]
        daemon = mock_daemon_cls.return_value
        daemon.is_running.return_value = True
        daemon.stop.return_value = True

        args = Namespace(
            action="run",
            background=True,
            on_conflict="replace",
            port=27800,
            timeout=1.0,
            endpoints=None,
            name_pattern=None,
        )

        run_from_args(args)

        daemon.stop.assert_called_once_with()
        mock_handle_conflicts.assert_called_once_with(
            27800,
            policy="replace",
            label="[qwn_machine]",
        )
        daemon.start_background.assert_called_once()
