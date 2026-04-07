"""Tests for qwn machine auto-start helpers."""

from argparse import Namespace
from unittest.mock import patch

from tfmx.qwns.machine import QWNInstance, discover_instances


class TestQWNMachineAutoStart:
    @patch("tfmx.qwns.machine.QWNComposer")
    @patch("tfmx.qwns.machine.QWNInstanceDiscovery.discover")
    def test_discover_instances_auto_starts_backends(
        self,
        mock_discover,
        mock_composer,
    ):
        mock_discover.side_effect = [
            [],
            [
                QWNInstance(
                    container_name="qwn--gpu0",
                    host="localhost",
                    port=27880,
                    gpu_id=0,
                )
            ],
        ]
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
        composer.wait_for_healthy_backends.assert_called_once_with(
            timeout_sec=15.0,
            poll_interval_sec=1.0,
            label="[qwn_machine]",
        )
        assert [instance.port for instance in instances] == [27880]
