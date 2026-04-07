"""Tests for tfmx.qwns.machine."""

import asyncio
import httpx
import orjson
import pytest
import time

from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

from fastapi import HTTPException

from tfmx.qwns.machine import ModelInfo
from tfmx.qwns.machine import ModelsResponse
from tfmx.qwns.machine import ChatCompletionRequest
from tfmx.qwns.machine import ChatCompletionResponse
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
        assert info.available_slots >= 0


class TestQWNStatsData:
    def test_to_model(self):
        stats = QWNStatsData(
            total_requests=3,
            total_tokens=50,
            total_errors=1,
            total_failovers=2,
            active_requests=2,
        )
        stats.record_wait(42.0)
        model = stats.to_model()
        assert model.total_requests == 3
        assert model.total_failovers == 2
        assert model.active_requests == 2
        assert model.total_wait_events == 1
        assert model.avg_wait_time_ms == 42.0


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
        assert request.max_tokens is None

    def test_chat_completions_omits_max_tokens_when_not_provided(self):
        server = QWNMachineServer(instances=[])
        server._forward_chat = AsyncMock(return_value=ChatCompletionResponse())

        request = ChatCompletionRequest(
            messages=[{"role": "user", "content": "hello"}],
        )
        asyncio.run(server.chat_completions(request))

        body = server._forward_chat.call_args.args[0]
        payload = orjson.loads(body)
        assert "max_tokens" not in payload

    def test_chat_completions_preserves_explicit_max_tokens(self):
        server = QWNMachineServer(instances=[])
        server._forward_chat = AsyncMock(return_value=ChatCompletionResponse())

        request = ChatCompletionRequest(
            messages=[{"role": "user", "content": "hello"}],
            max_tokens=256,
        )
        asyncio.run(server.chat_completions(request))

        body = server._forward_chat.call_args.args[0]
        payload = orjson.loads(body)
        assert payload["max_tokens"] == 256

    def test_chat_form_omits_max_tokens_when_not_provided(self):
        server = QWNMachineServer(instances=[])
        server._forward_chat = AsyncMock(return_value=ChatCompletionResponse())

        asyncio.run(
            server.chat_form(
                text="hello",
                system_prompt="",
                model="",
                max_tokens=None,
                temperature=0.7,
                top_p=0.9,
            )
        )

        body = server._forward_chat.call_args.args[0]
        payload = orjson.loads(body)
        assert "max_tokens" not in payload

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

    def test_rewrite_model_in_body_normalizes_thinking_payload(self):
        instance = QWNInstance(
            container_name="qwn-multi--gpu0",
            host="localhost",
            port=27880,
            model_name="4b:4bit",
        )
        server = QWNMachineServer(instances=[instance])
        body = b'{"model":"4b","messages":[{"role":"user","content":"hello"}],"thinking":{"type":"disabled"}}'

        rewritten = server._rewrite_model_in_body(body, instance)
        payload = orjson.loads(rewritten)

        assert payload["model"] == "4b:4bit"
        assert payload["chat_template_kwargs"]["enable_thinking"] is False
        assert "thinking" not in payload

    def test_rewrite_model_in_body_defaults_thinking_to_disabled(self):
        instance = QWNInstance(
            container_name="qwn-multi--gpu0",
            host="localhost",
            port=27880,
            model_name="4b:4bit",
            quant_level="4bit",
        )
        server = QWNMachineServer(instances=[instance])
        body = b'{"model":"qwen3.5-4b-awq-4bit","messages":[{"role":"user","content":"hello"}]}'

        rewritten = server._rewrite_model_in_body(body, instance)
        payload = orjson.loads(rewritten)

        assert payload["model"] == "4b:4bit"
        assert payload["chat_template_kwargs"]["enable_thinking"] is False

    def test_alias_model_request_routes_to_existing_instance(self):
        instance = QWNInstance(
            container_name="qwn-multi--gpu0",
            host="localhost",
            port=27880,
            gpu_id=0,
            healthy=True,
            model_name="4b:4bit",
            quant_level="4bit",
        )
        server = QWNMachineServer(instances=[instance])
        server._build_router()

        candidates = server._get_candidate_instances("qwen3.5-4b-awq-4bit")

        assert [candidate.container_name for candidate in candidates] == [
            "qwn-multi--gpu0"
        ]

    def test_blank_model_requests_use_stable_default_model(self):
        default_a = QWNInstance(
            container_name="qwn-multi--gpu0",
            host="localhost",
            port=27880,
            gpu_id=0,
            healthy=True,
            model_name="Qwen/Qwen3.5-4B-AWQ-4bit",
            quant_level="4bit",
        )
        default_b = QWNInstance(
            container_name="qwn-multi--gpu2",
            host="localhost",
            port=27882,
            gpu_id=2,
            healthy=True,
            model_name="Qwen/Qwen3.5-4B-AWQ-4bit",
            quant_level="4bit",
        )
        other = QWNInstance(
            container_name="qwn-multi--gpu3",
            host="localhost",
            port=27883,
            gpu_id=3,
            healthy=True,
            model_name="Qwen/Qwen3-8B",
        )

        server = QWNMachineServer(instances=[default_a, default_b, other])
        server._build_router()

        candidates = server._get_candidate_instances()

        assert {instance.container_name for instance in candidates} == {
            "qwn-multi--gpu0",
            "qwn-multi--gpu2",
        }
        assert server._resolve_requested_model_field("") == "4b:4bit"

    def test_app_exposes_openai_compat_aliases(self):
        server = QWNMachineServer(instances=[])
        route_paths = {route.path for route in server.app.routes}

        assert "/models" in route_paths
        assert "/v1/models" in route_paths
        assert "/chat/completions" in route_paths
        assert "/v1/chat/completions" in route_paths

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

    def test_get_idle_instance_penalizes_recent_failures(self):
        now = time.monotonic()
        unstable = QWNInstance(
            container_name="qwn-multi--gpu0",
            host="localhost",
            port=27880,
            gpu_id=0,
            healthy=True,
        )
        stable = QWNInstance(
            container_name="qwn-multi--gpu2",
            host="localhost",
            port=27882,
            gpu_id=2,
            healthy=True,
        )
        unstable.telemetry.record_dispatch(now=now)
        unstable.telemetry.record_failure("boom", now=now)
        stable.telemetry.record_dispatch(now=now)
        stable.telemetry.record_success(latency_ms=900.0, now=now)

        server = QWNMachineServer(instances=[unstable, stable])

        chosen = server._get_idle_instance()
        assert chosen is stable

    def test_get_idle_instance_penalizes_high_recent_latency(self):
        now = time.monotonic()
        slow = QWNInstance(
            container_name="qwn-multi--gpu0",
            host="localhost",
            port=27880,
            gpu_id=0,
            healthy=True,
        )
        fast = QWNInstance(
            container_name="qwn-multi--gpu2",
            host="localhost",
            port=27882,
            gpu_id=2,
            healthy=True,
        )
        slow.telemetry.record_dispatch(now=now)
        slow.telemetry.record_success(latency_ms=6800.0, now=now)
        fast.telemetry.record_dispatch(now=now)
        fast.telemetry.record_success(latency_ms=900.0, now=now)

        server = QWNMachineServer(instances=[slow, fast])

        chosen = server._get_idle_instance()
        assert chosen is fast

    def test_get_idle_instance_prefers_higher_observed_throughput(self):
        now = time.monotonic()
        slower = QWNInstance(
            container_name="qwn-multi--gpu0",
            host="localhost",
            port=27880,
            gpu_id=0,
            healthy=True,
        )
        faster = QWNInstance(
            container_name="qwn-multi--gpu2",
            host="localhost",
            port=27882,
            gpu_id=2,
            healthy=True,
        )
        slower.telemetry.record_dispatch(now=now)
        slower.telemetry.record_success(
            latency_ms=1200.0,
            completion_tokens=48,
            now=now,
        )
        faster.telemetry.record_dispatch(now=now)
        faster.telemetry.record_success(
            latency_ms=600.0,
            completion_tokens=48,
            now=now,
        )

        server = QWNMachineServer(instances=[slower, faster])

        chosen = server._get_idle_instance()
        assert chosen is faster

    def test_models_endpoint(self):
        instance = QWNInstance(
            container_name="qwn-multi--gpu0",
            host="localhost",
            port=27880,
            healthy=True,
            model_name="4b:4bit",
            quant_level="4bit",
        )
        server = QWNMachineServer(instances=[instance])
        result = asyncio.run(server.models())
        assert isinstance(result, ModelsResponse)
        assert [item.id for item in result.data] == [
            "qwen3.5-4b-awq-4bit",
            "qwen3.5-4b",
            "4b:4bit",
        ]

    def test_check_instance_health_uses_short_timeout(self):
        instance = QWNInstance(
            container_name="qwn-multi--gpu0",
            host="localhost",
            port=27880,
            healthy=True,
        )
        server = QWNMachineServer(instances=[instance])
        response = MagicMock(status_code=200)
        server._client = AsyncMock()
        server._client.get.return_value = response

        assert asyncio.run(server._check_instance_health(instance)) is True
        timeout = server._client.get.call_args.kwargs["timeout"]
        assert timeout.connect == 5.0

    def test_discover_instance_models_uses_short_timeout(self):
        instance = QWNInstance(
            container_name="qwn-multi--gpu0",
            host="localhost",
            port=27880,
            healthy=True,
        )
        server = QWNMachineServer(instances=[instance])
        response = MagicMock(status_code=200)
        response.json.return_value = {"data": [{"id": "4b:4bit"}]}
        server._client = AsyncMock()
        server._client.get.return_value = response

        asyncio.run(server._discover_instance_models())

        timeout = server._client.get.call_args.kwargs["timeout"]
        assert timeout.connect == 10.0
        assert instance.model_name == "4b:4bit"

    def test_acquire_instance_waits_for_capacity_release(self):
        instance = QWNInstance(
            container_name="qwn-multi--gpu0",
            host="localhost",
            port=27880,
            healthy=True,
        )
        instance._active_requests = 8
        server = QWNMachineServer(instances=[instance])

        async def scenario():
            async def release_later():
                await asyncio.sleep(0.05)
                instance._active_requests = 7
                await server._notify_capacity_changed()

            release_task = asyncio.create_task(release_later())
            try:
                reserved, _, _ = await server._acquire_instance()
            finally:
                await release_task
            return reserved

        chosen = asyncio.run(scenario())
        assert chosen is instance
        assert server.stats.total_wait_events == 1

    def test_info_includes_scheduler_snapshot(self):
        now = time.monotonic()
        instance = QWNInstance(
            container_name="qwn-multi--gpu0",
            host="localhost",
            port=27880,
            healthy=True,
            model_name="4b:4bit",
        )
        instance.telemetry.record_dispatch(now=now)
        instance.telemetry.record_success(
            latency_ms=1200.0,
            ttft_ms=450.0,
            completion_tokens=32,
            now=now,
        )
        server = QWNMachineServer(instances=[instance])
        server._last_health_refresh_monotonic = now
        server._last_gpu_refresh_monotonic = now

        result = asyncio.run(server.info())

        assert result.scheduler.algorithm == "adaptive_pressure_v2"
        assert result.instances[0].scheduler.latency_ema_ms == 1200.0
        assert result.instances[0].active_requests == 0

    def test_metrics_payload_includes_scheduler_and_instance_metrics(self):
        now = time.monotonic()
        instance = QWNInstance(
            container_name="qwn-multi--gpu0",
            host="localhost",
            port=27880,
            healthy=True,
            model_name="4b:4bit",
            gpu_id=0,
        )
        instance._gpu_utilization_pct = 33.0
        instance._gpu_memory_used_mib = 10240.0
        instance._gpu_memory_total_mib = 20480.0
        instance.telemetry.record_dispatch(now=now)
        instance.telemetry.record_success(
            latency_ms=800.0,
            ttft_ms=240.0,
            completion_tokens=32,
            now=now,
        )
        server = QWNMachineServer(instances=[instance])
        server.stats.total_requests = 3
        server.stats.total_tokens = 128
        server.stats.requests_per_instance[instance.container_name] = 3
        server._last_health_refresh_monotonic = now
        server._last_gpu_refresh_monotonic = now

        payload = server._build_metrics_payload()

        assert "qwn_machine_requests_total 3" in payload
        assert (
            'qwn_machine_scheduler_weight{component="load",kind="effective"}' in payload
        )
        assert "qwn_machine_instance_scheduler_score{" in payload
        assert "qwn_machine_instance_tokens_per_second_ema{" in payload

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
