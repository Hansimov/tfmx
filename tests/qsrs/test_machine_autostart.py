"""Tests for qsr machine auto-start helpers."""

from argparse import Namespace
from unittest.mock import patch

from tfmx.qsrs.machine import QSRInstance, QSRMachineServer, discover_instances
from tfmx.qsrs.machine import run_from_args


class TestQSRMachineAutoStart:
    @patch("tfmx.utils.service_bootstrap.wait_for_available_backend_instances")
    @patch("tfmx.qsrs.machine.QSRComposer")
    @patch("tfmx.qsrs.machine.QSRInstanceDiscovery.discover")
    def test_discover_instances_auto_starts_backends(
        self,
        mock_discover,
        mock_composer,
        mock_wait_for_available,
    ):
        discovered_instances = [
            QSRInstance(
                container_name="qsr--gpu0",
                host="localhost",
                port=27980,
                gpu_id=0,
            )
        ]
        mock_discover.return_value = []
        mock_wait_for_available.return_value = discovered_instances
        composer = mock_composer.return_value
        composer.get_backend_endpoints.return_value = ["http://localhost:27980"]

        args = Namespace(
            endpoints=None,
            name_pattern=None,
            auto_start=True,
            startup_timeout=12.0,
            startup_poll_interval=0.5,
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
            "timeout_sec": 12.0,
            "poll_interval_sec": 0.5,
            "settle_sec": 2.0,
            "label": "[qsr_machine]",
        }
        assert [instance.port for instance in instances] == [27980]

    def test_machine_server_merges_newly_discovered_instances(self):
        first = QSRInstance(
            container_name="qsr--gpu0",
            host="localhost",
            port=27980,
            gpu_id=0,
        )
        second = QSRInstance(
            container_name="qsr--gpu1",
            host="localhost",
            port=27981,
            gpu_id=1,
        )

        server = QSRMachineServer(instances=[first], port=27900, timeout=1.0)

        changed, added_instances = server._merge_discovered_instances([first, second])

        assert changed is True
        assert [instance.container_name for instance in added_instances] == [
            "qsr--gpu1"
        ]
        assert [instance.gpu_id for instance in server.instances] == [0, 1]

    @patch("tfmx.qsrs.machine.handle_port_conflicts", return_value=True)
    @patch("tfmx.qsrs.machine.discover_instances")
    @patch("tfmx.qsrs.machine.QSRMachineDaemon")
    def test_run_from_args_background_replace_existing_daemon(
        self,
        mock_daemon_cls,
        mock_discover_instances,
        mock_handle_conflicts,
    ):
        instance = QSRInstance(
            container_name="qsr--gpu0",
            host="localhost",
            port=27980,
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
            port=27900,
            timeout=1.0,
            endpoints=None,
            name_pattern=None,
            auto_start=False,
        )

        run_from_args(args)

        daemon.stop.assert_called_once_with()
        mock_handle_conflicts.assert_called_once_with(
            27900,
            policy="replace",
            label="[qsr_machine]",
        )
        daemon.start_background.assert_called_once()
