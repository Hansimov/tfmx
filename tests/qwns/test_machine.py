"""Tests for tfmx.qwns.machine."""

import asyncio
import httpx
import orjson
import pytest

from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

from fastapi import HTTPException

from tfmx.qwns.machine import ModelInfo
from tfmx.qwns.machine import ModelsResponse
from tfmx.qwns.machine import ChatCompletionRequest
from tfmx.qwns.machine import QWNInstance
from tfmx.qwns.machine import QWNInstanceDiscovery
from tfmx.qwns.machine import QWNMachineDaemon
from tfmx.qwns.machine import QWNMachineServer
from tfmx.qwns.machine import QWNStatsData


class TestQWNInstance:
    def _make(self, **kwargs) -> QWNInstance:
        params = {
            "container_name": "qwn-multi--gpu0",
            "host": "localhost",
            "port": 27880,
            "gpu_id": 0,
        }
        params.update(kwargs)
        return QWNInstance(**params)

    def test_endpoint(self):
        instance = self._make()
        assert instance.endpoint == "http://localhost:27880"

    def test_available_slots(self):
        instance = self._make()
        assert instance.available_slots > 0

    def test_to_info(self):
        instance = self._make(healthy=True, model_name="4b:4bit")
        info = instance.to_info()
        assert info.healthy is True
        assert info.model_name == "4b:4bit"


class TestQWNStatsData:
    def test_to_model(self):
        stats = QWNStatsData(
            total_requests=3,
            total_tokens=50,
            total_errors=1,
            total_failovers=2,
            active_requests=2,
        )
        model = stats.to_model()
        assert model.total_requests == 3
        assert model.total_failovers == 2
        assert model.active_requests == 2


class TestQWNInstanceDiscovery:
    @patch("subprocess.run")
    def test_discover(self, mock_run):
        mock_run.return_value = MagicMock(
            returncode=0,
            stdout="qwn-multi--gpu0|vllm/vllm-openai:latest|0.0.0.0:27880->8000/tcp\n",
        )
        instances = QWNInstanceDiscovery.discover()
        assert len(instances) == 1
        assert instances[0].gpu_id == 0

    def test_from_endpoints(self):
        instances = QWNInstanceDiscovery.from_endpoints(
            ["http://localhost:27880", "27881"]
        )
        assert len(instances) == 2
        assert instances[0].port == 27880
        assert instances[1].port == 27881


class TestQWNMachineServer:
    def test_chat_completion_request_accepts_multimodal_content(self):
        request = ChatCompletionRequest(
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "先看这张图"},
                        {
                            "type": "image_url",
                            "image_url": {"url": "data:image/png;base64,AAA"},
                        },
                        {"type": "text", "text": "再回答"},
                    ],
                }
            ]
        )
        assert request.messages[0].content[1].type == "image_url"

    def test_get_model_label(self):
        instance = QWNInstance(
            container_name="qwn-multi--gpu0",
            host="localhost",
            port=27880,
            gpu_id=0,
            model_name="4b:4bit",
        )
        server = QWNMachineServer(instances=[instance])
        assert server._get_model_label(instance) == "4b:4bit"

    def test_rewrite_model_in_body(self):
        instance = QWNInstance(
            container_name="qwn-multi--gpu0",
            host="localhost",
            port=27880,
            model_name="4b:4bit",
        )
        server = QWNMachineServer(instances=[instance])
        body = b'{"model":"4b","messages":[{"role":"user","content":"hello"}]}'
        rewritten = server._rewrite_model_in_body(body, instance)
        assert b"4b:4bit" in rewritten

    def test_get_idle_instance_prefers_lower_gpu_pressure(self):
        busy_gpu = QWNInstance(
            container_name="qwn-multi--gpu0",
            host="localhost",
            port=27880,
            gpu_id=0,
            healthy=True,
        )
        cool_gpu = QWNInstance(
            container_name="qwn-multi--gpu2",
            host="localhost",
            port=27882,
            gpu_id=2,
            healthy=True,
        )
        busy_gpu._gpu_utilization_pct = 95.0
        busy_gpu._gpu_memory_used_mib = 18000.0
        busy_gpu._gpu_memory_total_mib = 20480.0
        cool_gpu._gpu_utilization_pct = 12.0
        cool_gpu._gpu_memory_used_mib = 14000.0
        cool_gpu._gpu_memory_total_mib = 20480.0

        server = QWNMachineServer(instances=[busy_gpu, cool_gpu])

        chosen = server._get_idle_instance()
        assert chosen is cool_gpu

    def test_models_endpoint(self):
        instance = QWNInstance(
            container_name="qwn-multi--gpu0",
            host="localhost",
            port=27880,
            healthy=True,
            model_name="4b:4bit",
        )
        server = QWNMachineServer(instances=[instance])
        result = asyncio.run(server.models())
        assert isinstance(result, ModelsResponse)
        assert result.data[0].id == "4b:4bit"

    def test_acquire_instance_does_not_fallback_to_wrong_requested_model(self):
        primary = QWNInstance(
            container_name="qwn-multi--gpu0",
            host="localhost",
            port=27880,
            healthy=True,
            model_name="4b:4bit",
        )
        alternate = QWNInstance(
            container_name="qwn-multi--gpu2",
            host="localhost",
            port=27882,
            healthy=True,
            model_name="8b:4bit",
        )
        server = QWNMachineServer(instances=[primary, alternate])
        server._build_router()

        with pytest.raises(HTTPException):
            asyncio.run(server._acquire_instance("32b:4bit"))

    def test_forward_chat_fails_over_to_next_healthy_instance(self):
        primary = QWNInstance(
            container_name="qwn-multi--gpu0",
            host="localhost",
            port=27880,
            healthy=True,
            model_name="4b:4bit",
        )
        alternate = QWNInstance(
            container_name="qwn-multi--gpu2",
            host="localhost",
            port=27882,
            healthy=True,
            model_name="4b:4bit",
        )
        server = QWNMachineServer(instances=[primary, alternate])
        server._build_router()
        server._client = AsyncMock()
        server._client.post.side_effect = [
            httpx.ConnectError("boom"),
            MagicMock(
                status_code=200,
                content=orjson.dumps(
                    {
                        "id": "ok",
                        "object": "chat.completion",
                        "model": "4b:4bit",
                        "choices": [
                            {
                                "index": 0,
                                "message": {"role": "assistant", "content": "ok"},
                                "finish_reason": "stop",
                            }
                        ],
                        "usage": {
                            "prompt_tokens": 10,
                            "completion_tokens": 3,
                            "total_tokens": 13,
                        },
                    }
                ),
            ),
        ]

        result = asyncio.run(
            server._forward_chat(b'{"messages":[{"role":"user","content":"hello"}]}')
        )

        assert result.model == "4b:4bit"
        assert primary.healthy is False
        assert alternate.healthy is True
        assert server.stats.total_failovers == 1

    def test_forward_stream_fails_over_before_first_chunk(self):
        class FakeStreamResponse:
            def __init__(self, status_code=200, lines=None, enter_error=None):
                self.status_code = status_code
                self._lines = lines or []
                self._enter_error = enter_error

            async def __aenter__(self):
                if self._enter_error is not None:
                    raise self._enter_error
                return self

            async def __aexit__(self, exc_type, exc, tb):
                return False

            async def aread(self):
                return b"upstream error"

            async def aiter_lines(self):
                for line in self._lines:
                    yield line

        class FakeAsyncClient:
            def __init__(self, responses):
                self._responses = list(responses)

            def stream(self, *args, **kwargs):
                return self._responses.pop(0)

        primary = QWNInstance(
            container_name="qwn-multi--gpu0",
            host="localhost",
            port=27880,
            healthy=True,
            model_name="4b:4bit",
        )
        alternate = QWNInstance(
            container_name="qwn-multi--gpu2",
            host="localhost",
            port=27882,
            healthy=True,
            model_name="4b:4bit",
        )
        server = QWNMachineServer(instances=[primary, alternate])
        server._build_router()
        server._client = FakeAsyncClient(
            [
                FakeStreamResponse(enter_error=httpx.ConnectError("boom")),
                FakeStreamResponse(
                    lines=[
                        'data: {"choices":[{"delta":{"content":"ok"}}]}',
                        "data: [DONE]",
                    ]
                ),
            ]
        )

        response = asyncio.run(
            server._forward_stream(b'{"messages":[{"role":"user","content":"hello"}]}')
        )

        async def collect_stream():
            chunks = []
            async for chunk in response.body_iterator:
                chunks.append(chunk)
            return chunks

        chunks = asyncio.run(collect_stream())

        assert any('"model":"4b:4bit"' in chunk for chunk in chunks)
        assert primary.healthy is False
        assert server.stats.total_failovers == 1

    @patch.object(
        QWNMachineServer, "_refresh_gpu_runtime_metrics", new_callable=AsyncMock
    )
    @patch.object(QWNMachineServer, "_discover_instance_models", new_callable=AsyncMock)
    @patch.object(QWNMachineServer, "_check_instance_health", new_callable=AsyncMock)
    def test_health_check_all_rebuilds_router_for_newly_healthy_instances(
        self,
        mock_check_health,
        mock_discover_models,
        mock_refresh_gpu,
    ):
        instance = QWNInstance(
            container_name="qwn-multi--gpu0",
            host="localhost",
            port=27880,
            healthy=True,
        )
        server = QWNMachineServer(instances=[instance])
        server._client = MagicMock()

        with patch.object(server, "_build_router") as mock_build_router:
            asyncio.run(server.health_check_all())

        mock_refresh_gpu.assert_awaited_once()
        mock_discover_models.assert_awaited_once()
        mock_build_router.assert_called_once()


class TestQWNMachineDaemon:
    def test_get_pid_missing(self, tmp_path):
        daemon = QWNMachineDaemon(
            pid_file=tmp_path / "qwn.pid",
            log_file=tmp_path / "qwn.log",
        )
        assert daemon.get_pid() is None

    @patch("subprocess.Popen")
    def test_start_background_strips_cli_machine_arg(self, mock_popen, tmp_path):
        proc = MagicMock(pid=4321)
        mock_popen.return_value = proc
        daemon = QWNMachineDaemon(
            pid_file=tmp_path / "qwn.pid",
            log_file=tmp_path / "qwn.log",
        )

        daemon.start_background(["cli.py", "machine", "run", "-b"])

        cmd = mock_popen.call_args.args[0]
        assert cmd[1:] == ["-m", "tfmx.qwns.machine", "run"]
        assert daemon.pid_file.read_text().strip() == "4321"
