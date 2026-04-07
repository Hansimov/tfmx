"""Tests for tfmx.teis.machine CLI helpers."""

import asyncio
import httpx
import orjson

from argparse import Namespace
from unittest.mock import patch

import numpy as np

from tfmx.teis.machine import TEIInstance, TEIMachineServer, discover_instances
from tfmx.teis.machine import run_from_args


class TestTEIMachineAutoStart:
    @patch("tfmx.utils.service_bootstrap.wait_for_available_backend_instances")
    @patch("tfmx.teis.machine.TEIComposer")
    @patch("tfmx.teis.machine.TEIInstanceDiscovery.discover")
    def test_discover_instances_auto_starts_backends(
        self,
        mock_discover,
        mock_composer,
        mock_wait_for_available,
    ):
        discovered_instances = [
            TEIInstance(
                container_name="tei--gpu0",
                host="localhost",
                port=28880,
                gpu_id=0,
            )
        ]
        mock_discover.return_value = []
        mock_wait_for_available.return_value = discovered_instances
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
            "settle_sec": 10.0,
            "label": "[tei_machine]",
        }
        assert [instance.port for instance in instances] == [28880]

    def test_machine_server_merges_newly_discovered_instances(self):
        first = TEIInstance(
            container_name="tei--gpu0",
            host="localhost",
            port=28880,
            gpu_id=0,
        )
        second = TEIInstance(
            container_name="tei--gpu1",
            host="localhost",
            port=28881,
            gpu_id=1,
        )

        server = TEIMachineServer(instances=[first], port=28800, timeout=1.0)

        changed, added_instances = server._merge_discovered_instances([first, second])

        assert changed is True
        assert [instance.container_name for instance in added_instances] == [
            "tei--gpu1"
        ]
        assert [instance.gpu_id for instance in server.instances] == [0, 1]

    def test_run_embeddings_with_failover_retries_after_runtime_failure(self):
        first = TEIInstance(
            container_name="tei--gpu0",
            host="localhost",
            port=28880,
            gpu_id=0,
            healthy=True,
        )
        second = TEIInstance(
            container_name="tei--gpu2",
            host="localhost",
            port=28882,
            gpu_id=2,
            healthy=True,
        )
        server = TEIMachineServer(instances=[first, second], port=28800, timeout=1.0)

        call_sizes: list[int] = []

        async def flaky_distribute(inputs, instances, normalize, truncate):
            call_sizes.append(len(instances))
            if len(call_sizes) == 1:
                server._mark_instance_unhealthy(first, "backend oom")
                raise RuntimeError("Instance 28880 error: backend oom")
            return np.array([[1.0, 2.0]], dtype=np.float32)

        server._distribute_with_scheduler_np = flaky_distribute  # type: ignore[method-assign]

        result = asyncio.run(
            server._run_embeddings_with_failover(["hello"], True, True)
        )

        assert call_sizes == [2, 1]
        assert first.healthy is False
        assert second.healthy is True
        assert result.shape == (1, 2)

    def test_send_embed_request_splits_capacity_errors_without_marking_unhealthy(self):
        instance = TEIInstance(
            container_name="tei--gpu0",
            host="localhost",
            port=28880,
            gpu_id=0,
            healthy=True,
        )
        server = TEIMachineServer(instances=[instance], port=28800, timeout=1.0)

        class _FakeClient:
            async def post(self, _url, json):
                request = httpx.Request("POST", "http://localhost:28880/embed")
                if len(json["inputs"]) > 1:
                    return httpx.Response(
                        500,
                        json={
                            "error": "CUDA_ERROR_OUT_OF_MEMORY",
                            "error_type": "Backend",
                        },
                        request=request,
                    )
                return httpx.Response(
                    200,
                    content=orjson.dumps([[float(len(json["inputs"][0]))]]),
                    request=request,
                )

        server._client = _FakeClient()

        result = asyncio.run(
            server._send_embed_request_np(instance, ["aa", "bbbb"], True, True)
        )

        assert instance.healthy is True
        assert result.shape == (2, 1)
        assert result[:, 0].tolist() == [2.0, 4.0]

    @patch("tfmx.teis.machine.handle_port_conflicts", return_value=False)
    @patch("tfmx.teis.machine.discover_instances")
    @patch("tfmx.teis.machine.TEIMachineServer")
    def test_run_from_args_stops_when_conflict_reported(
        self,
        mock_server_cls,
        mock_discover_instances,
        mock_handle_conflicts,
    ):
        mock_discover_instances.return_value = [
            TEIInstance(
                container_name="tei--gpu0",
                host="localhost",
                port=28880,
                gpu_id=0,
            )
        ]
        args = Namespace(
            action="run",
            port=28800,
            timeout=1.0,
            batch_size=32,
            micro_batch_size=8,
            no_gpu_lsh=False,
            perf_track=False,
            endpoints=None,
            name_pattern=None,
            on_conflict="report",
        )

        run_from_args(args)

        mock_handle_conflicts.assert_called_once_with(
            28800,
            policy="report",
            label="[tei_machine]",
        )
        mock_server_cls.assert_not_called()
