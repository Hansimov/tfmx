"""Tests for tfmx.teis.machine CLI helpers."""

from argparse import Namespace
from unittest.mock import patch

from tfmx.teis.machine import TEIInstance, discover_instances


class TestTEIMachineAutoStart:
    @patch("tfmx.teis.machine.TEIComposer")
    @patch("tfmx.teis.machine.TEIInstanceDiscovery.discover")
    def test_discover_instances_auto_starts_backends(
        self,
        mock_discover,
        mock_composer,
    ):
        mock_discover.side_effect = [
            [],
            [
                TEIInstance(
                    container_name="tei--gpu0",
                    host="localhost",
                    port=28880,
                    gpu_id=0,
                )
            ],
        ]
        composer = mock_composer.return_value
        composer.get_backend_endpoints.return_value = ["http://localhost:28880"]

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
        )

        instances = discover_instances(args)

        mock_composer.assert_called_once_with()
        composer.up.assert_called_once_with()
        composer.wait_for_healthy_backends.assert_called_once_with(
            timeout_sec=12.0,
            poll_interval_sec=0.5,
            label="[tei_machine]",
        )
        assert [instance.port for instance in instances] == [28880]
