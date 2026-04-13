"""Tests for tfmx.qsrs.machine."""

import asyncio
import orjson

from fastapi.responses import JSONResponse
from unittest.mock import AsyncMock, MagicMock, patch

from tfmx.qsrs.machine import ChatCompletionRequest
from tfmx.qsrs.machine import ChatCompletionResponse
from tfmx.qsrs.machine import QSRInstance
from tfmx.qsrs.machine import QSRInstanceDiscovery
from tfmx.qsrs.machine import QSRMachineServer
from tfmx.qsrs.machine import QSRStatsData


class TestQSRInstance:
    def _make(self, **kwargs) -> QSRInstance:
        params = {
            "container_name": "qsr-multi--gpu0",
            "host": "localhost",
            "port": 27980,
            "gpu_id": 0,
        }
        params.update(kwargs)
        return QSRInstance(**params)

    def test_endpoint(self):
        instance = self._make()
        assert instance.endpoint == "http://localhost:27980"

    def test_available_slots(self):
        instance = self._make()
        assert instance.available_slots > 0

    def test_to_info(self):
        instance = self._make(
            healthy=True,
            sleeping=True,
            model_name="qwen3-asr-0.6b",
        )
        info = instance.to_info()
        assert info.healthy is True
        assert info.sleeping is True
        assert info.model_name == "qwen3-asr-0.6b"
        assert info.available_slots >= 0


class TestQSRStatsData:
    def test_to_model(self):
        stats = QSRStatsData(
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


class TestQSRInstanceDiscovery:
    @patch("subprocess.run")
    def test_discover(self, mock_run):
        mock_run.return_value = MagicMock(
            returncode=0,
            stdout="qsr-multi--gpu0|vllm/vllm-openai:latest|0.0.0.0:27980->8000/tcp|Up 3 seconds (healthy)\n",
        )
        instances = QSRInstanceDiscovery.discover()
        assert len(instances) == 1
        assert instances[0].gpu_id == 0
        assert instances[0].docker_health is True

    def test_from_endpoints(self):
        instances = QSRInstanceDiscovery.from_endpoints(
            ["http://localhost:27980", "27981"]
        )
        assert len(instances) == 2
        assert instances[0].port == 27980
        assert instances[1].port == 27981


class TestQSRMachineServer:
    def test_chat_completion_request_accepts_audio_content(self):
        request = ChatCompletionRequest(
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "请转写"},
                        {
                            "type": "audio_url",
                            "audio_url": {"url": "data:audio/wav;base64,AAA"},
                        },
                    ],
                }
            ]
        )
        assert request.messages[0].content[1].type == "audio_url"
        assert request.max_tokens is None

    def test_chat_completions_omits_max_tokens_when_not_provided(self):
        server = QSRMachineServer(instances=[])
        server._forward_chat = AsyncMock(return_value=ChatCompletionResponse())

        request = ChatCompletionRequest(
            messages=[{"role": "user", "content": "hello"}],
        )
        asyncio.run(server.chat_completions(request))

        body = server._forward_chat.call_args.args[0]
        payload = orjson.loads(body)
        assert "max_tokens" not in payload

    def test_chat_completions_preserves_explicit_max_tokens(self):
        server = QSRMachineServer(instances=[])
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
        server = QSRMachineServer(instances=[])
        server._forward_chat = AsyncMock(return_value=ChatCompletionResponse())

        asyncio.run(
            server.chat_form(
                text="hello",
                system_prompt="",
                model="",
                max_tokens=None,
                temperature=0.0,
                top_p=1.0,
            )
        )

        body = server._forward_chat.call_args.args[0]
        payload = orjson.loads(body)
        assert "max_tokens" not in payload

    def test_rewrite_model_in_body(self):
        instance = QSRInstance(
            container_name="qsr-multi--gpu0",
            host="localhost",
            port=27980,
            model_name="qwen3-asr-0.6b",
        )
        server = QSRMachineServer(instances=[instance])
        body = b'{"model":"0.6b","messages":[{"role":"user","content":"hello"}]}'
        rewritten = server._rewrite_model_in_body(body, instance)
        assert b"qwen3-asr-0.6b" in rewritten

    def test_app_exposes_openai_compat_aliases(self):
        server = QSRMachineServer(instances=[])
        route_paths = {route.path for route in server.app.routes}

        assert "/models" in route_paths
        assert "/v1/models" in route_paths
        assert "/chat/completions" in route_paths
        assert "/v1/chat/completions" in route_paths
        assert "/audio/transcriptions" in route_paths
        assert "/v1/audio/transcriptions" in route_paths

    def test_check_instance_health_marks_sleeping_instance_unhealthy(self):
        instance = QSRInstance(
            container_name="qsr-multi--gpu0",
            host="localhost",
            port=27980,
            healthy=True,
        )
        server = QSRMachineServer(instances=[instance])
        health_response = MagicMock(status_code=200)
        sleep_response = MagicMock(status_code=200)
        sleep_response.json.return_value = {"is_sleeping": True}
        server._client = AsyncMock()
        server._client.get.side_effect = [health_response, sleep_response]

        assert asyncio.run(server._check_instance_health(instance)) is False
        assert instance.healthy is False
        assert instance.sleeping is True

    @patch(
        "tfmx.qsrs.machine.load_backend_sleep_states",
        return_value={"http://localhost:27980": True},
    )
    def test_check_instance_health_uses_docker_health_sleep_state_without_http(
        self,
        _mock_sleep_state,
    ):
        instance = QSRInstance(
            container_name="qsr-multi--gpu0",
            host="localhost",
            port=27980,
            healthy=False,
            docker_health=True,
            docker_status="Up 3 seconds (healthy)",
        )
        server = QSRMachineServer(instances=[instance])
        server._client = AsyncMock()

        assert asyncio.run(server._check_instance_health(instance)) is False
        assert instance.sleeping is True
        server._client.get.assert_not_called()

    def test_select_idle_instance_round_robins_when_load_ties(self):
        first_instance = QSRInstance(
            container_name="qsr-uniform--gpu0",
            host="localhost",
            port=27980,
            gpu_id=0,
            healthy=True,
        )
        second_instance = QSRInstance(
            container_name="qsr-uniform--gpu1",
            host="localhost",
            port=27981,
            gpu_id=1,
            healthy=True,
        )
        server = QSRMachineServer(instances=[first_instance, second_instance])

        selected_a, has_candidates_a = server._select_idle_instance()
        selected_b, has_candidates_b = server._select_idle_instance()

        assert has_candidates_a is True
        assert has_candidates_b is True
        assert selected_a is first_instance
        assert selected_b is second_instance

    def test_forward_transcription_uses_sync_bridge(self):
        instance = QSRInstance(
            container_name="qsr-multi--gpu0",
            host="localhost",
            port=27980,
            gpu_id=0,
            healthy=True,
            model_name="0.6b",
        )
        server = QSRMachineServer(instances=[instance])
        server._acquire_instance = AsyncMock(return_value=instance)
        server._release_instance = AsyncMock()
        server._post_transcription_sync = MagicMock(
            return_value=(
                200,
                {"content-type": "application/json"},
                b'{"text":"ok"}',
            )
        )

        response = asyncio.run(
            server._forward_transcription(
                filename="sample.wav",
                payload=b"RIFF1234WAVEfmt ",
                content_type="audio/wav",
                response_format="json",
            )
        )

        assert isinstance(response, JSONResponse)
        assert response.body == b'{"text":"ok"}'

        call = server._post_transcription_sync.call_args
        assert call.args[0] == "http://localhost:27980/v1/audio/transcriptions"
        multipart_fields = call.args[1]
        assert any(
            field[0] == "model" and field[1] == (None, "0.6b")
            for field in multipart_fields
        )
        assert any(field[0] == "file" for field in multipart_fields)

    def test_post_transcription_sync_reuses_persistent_client(self):
        server = QSRMachineServer(instances=[])
        response = MagicMock(
            status_code=200,
            headers={"content-type": "application/json"},
            content=b"{}",
        )
        client = MagicMock()
        client.post.return_value = response
        server._transcription_clients = {
            "http://localhost:27980/v1/audio/transcriptions": client
        }

        status_code, headers, content = server._post_transcription_sync(
            "http://localhost:27980/v1/audio/transcriptions",
            [("model", (None, "0.6b"))],
        )

        assert status_code == 200
        assert headers == {"content-type": "application/json"}
        assert content == b"{}"
        client.post.assert_called_once()

    def test_handle_retryable_instance_error_preserves_healthy_backend(self):
        instance = QSRInstance(
            container_name="qsr-multi--gpu0",
            host="localhost",
            port=27980,
            gpu_id=0,
            healthy=True,
        )
        server = QSRMachineServer(instances=[instance])
        server._probe_instance_http_health = AsyncMock(return_value=True)
        server._mark_instance_unhealthy = MagicMock()

        asyncio.run(
            server._handle_retryable_instance_error(
                instance,
                OSError(9, "Bad file descriptor"),
                reset_transcription_client=True,
            )
        )

        server._probe_instance_http_health.assert_awaited_once_with(instance)
        server._mark_instance_unhealthy.assert_not_called()

    def test_handle_retryable_upstream_status_preserves_healthy_backend(self):
        instance = QSRInstance(
            container_name="qsr-multi--gpu0",
            host="localhost",
            port=27980,
            gpu_id=0,
            healthy=True,
        )
        server = QSRMachineServer(instances=[instance])
        server._request_instance_maintenance = AsyncMock(return_value=True)
        server._probe_instance_http_health = AsyncMock(return_value=True)
        server._mark_instance_unhealthy = MagicMock()

        asyncio.run(
            server._handle_retryable_upstream_status(
                instance,
                status_code=500,
                detail={"message": "Internal server error"},
                reset_mm_cache=True,
            )
        )

        server._request_instance_maintenance.assert_awaited_once_with(
            instance,
            "/reset_mm_cache",
        )
        server._probe_instance_http_health.assert_awaited_once_with(instance)
        server._mark_instance_unhealthy.assert_not_called()
