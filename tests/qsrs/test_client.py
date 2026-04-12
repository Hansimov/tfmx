"""Tests for tfmx.qsrs.client."""

import asyncio

from pathlib import Path
from unittest.mock import MagicMock, patch

import httpx

from tfmx.qsrs.client import ChatChoice
from tfmx.qsrs.client import ChatMessage
from tfmx.qsrs.client import ChatResponse
from tfmx.qsrs.client import ChatUsage
from tfmx.qsrs.client import AsyncQSRClient
from tfmx.qsrs.client import InfoResponse
from tfmx.qsrs.client import ModelInfo
from tfmx.qsrs.client import QSRClient
from tfmx.qsrs.client import StreamChatResult
from tfmx.qsrs.client import _load_audio_upload
from tfmx.qsrs.client import build_audio_messages
from tfmx.qsrs.client import build_text_messages
from tfmx.qsrs.client import format_elapsed_time
from tfmx.qsrs.client import format_stream_stats_line
from tfmx.qsrs.client import join_prompt_texts
from tfmx.qsrs.client import normalize_audio_url


class TestBuildTextMessages:
    def test_prompt_only(self):
        messages = build_text_messages("Hello")
        assert messages == [{"role": "user", "content": "Hello"}]

    def test_with_system_prompt(self):
        messages = build_text_messages("Hello", system_prompt="Be helpful")
        assert messages[0]["role"] == "system"
        assert messages[1]["role"] == "user"


class TestAudioMessages:
    def test_join_prompt_texts(self):
        assert join_prompt_texts(["Hello", "World"]) == "Hello\n\nWorld"

    def test_build_audio_messages_interleaves_audio(self):
        messages = build_audio_messages(
            texts=["请先转写", "再总结"],
            audios=["https://example.com/a.wav", "https://example.com/b.wav"],
            system_prompt="你是一个语音助手",
        )
        assert messages[0]["role"] == "system"
        assert messages[1]["content"][0]["type"] == "text"
        assert messages[1]["content"][1]["type"] == "audio_url"
        assert messages[1]["content"][2]["text"] == "再总结"

    def test_normalize_audio_url_local_file(self, tmp_path):
        audio_path = tmp_path / "sample.wav"
        audio_path.write_bytes(b"RIFF1234")
        normalized = normalize_audio_url(str(audio_path))
        assert normalized.startswith("data:audio/")


class TestFormatHelpers:
    def test_seconds_only(self):
        assert format_elapsed_time(8.34) == "8.3s"

    def test_includes_ttft_and_rate(self):
        stats = format_stream_stats_line(
            StreamChatResult(
                text="你好",
                usage=ChatUsage(completion_tokens=12, total_tokens=20),
                elapsed_sec=3.0,
                first_token_latency_sec=0.8,
            )
        )
        assert "[统计]:" in stats
        assert "tokens" in stats
        assert "token/s" in stats


class TestChatResponse:
    def test_text_property(self):
        response = ChatResponse(
            id="chat-1",
            model="0.6b",
            choices=[
                ChatChoice(
                    index=0,
                    message=ChatMessage(role="assistant", content="Hi"),
                    finish_reason="stop",
                )
            ],
            usage=ChatUsage(total_tokens=10),
        )
        assert response.text == "Hi"


class TestInfoResponse:
    def test_from_dict(self):
        info = InfoResponse.from_dict(
            {
                "port": 27900,
                "instances": [
                    {
                        "name": "qsr--gpu0",
                        "endpoint": "http://localhost:27980",
                        "healthy": True,
                        "active_requests": 1,
                        "available_slots": 7,
                        "scheduler": {
                            "score": 0.42,
                            "recent_requests": 10,
                            "latency_ema_ms": 1234.5,
                        },
                    }
                ],
                "stats": {"total_requests": 1, "total_wait_events": 2},
                "available_models": ["qwen3-asr-0.6b"],
                "scheduler": {
                    "algorithm": "least_active_idle",
                    "acquire_timeout_sec": 5.0,
                },
            }
        )
        assert info.port == 27900
        assert len(info.instances) == 1
        assert info.available_models == ["qwen3-asr-0.6b"]
        assert info.instances[0].scheduler.score == 0.42
        assert info.stats.total_wait_events == 2


class TestQSRClient:
    def test_custom_endpoint_with_v1_suffix_normalizes_root(self):
        client = QSRClient(endpoint="http://myhost:29999/v1")
        assert client.endpoint == "http://myhost:29999"
        assert client.chat_endpoint == "http://myhost:29999/v1/chat/completions"
        assert client.models_endpoint == "http://myhost:29999/v1/models"
        client.close()

    def test_models_fallback_to_alias_endpoint(self):
        client = QSRClient(endpoint="http://localhost:27900")

        missing = MagicMock()
        missing.status_code = 404
        missing.raise_for_status.side_effect = httpx.HTTPStatusError(
            "not found",
            request=MagicMock(),
            response=missing,
        )

        ok = MagicMock()
        ok.raise_for_status.return_value = None
        ok.json.return_value = {
            "object": "list",
            "data": [{"id": "qwen3-asr-0.6b"}],
        }

        def get_side_effect(url, *args, **kwargs):
            if url.endswith("/v1/models"):
                return missing
            return ok

        client.client.get = MagicMock(side_effect=get_side_effect)

        result = client.models()

        assert result.models == ["qwen3-asr-0.6b"]
        assert client._cached_default_model == "qwen3-asr-0.6b"
        client.close()

    def test_chat_resolves_default_model_when_omitted(self):
        client = QSRClient(endpoint="http://localhost:27900")

        models_response = MagicMock()
        models_response.raise_for_status.return_value = None
        models_response.json.return_value = {
            "object": "list",
            "data": [{"id": "qwen3-asr-0.6b"}],
        }

        chat_response = MagicMock()
        chat_response.raise_for_status.return_value = None
        chat_response.json.return_value = {
            "id": "ok",
            "model": "qwen3-asr-0.6b",
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": "ok"},
                    "finish_reason": "stop",
                }
            ],
            "usage": {"prompt_tokens": 3, "completion_tokens": 1, "total_tokens": 4},
        }

        seen_requests: list[tuple[str, dict]] = []

        def get_side_effect(url, *args, **kwargs):
            return models_response

        def post_side_effect(url, *args, **kwargs):
            seen_requests.append((url, kwargs["json"]))
            return chat_response

        client.client.get = MagicMock(side_effect=get_side_effect)
        client.client.post = MagicMock(side_effect=post_side_effect)

        response = client.chat(messages=[{"role": "user", "content": "hello"}])

        assert response.model == "qwen3-asr-0.6b"
        assert seen_requests[0][1]["model"] == "qwen3-asr-0.6b"
        client.close()

    def test_transcribe_builds_valid_multipart_request(self, tmp_path):
        audio_path = tmp_path / "sample.wav"
        audio_path.write_bytes(b"RIFF1234WAVEfmt ")

        captured = {}

        def handler(request: httpx.Request) -> httpx.Response:
            body = request.read()
            captured["body"] = body
            captured["content_type"] = request.headers.get("content-type", "")
            return httpx.Response(
                200,
                json={"text": "ok", "language": "zh"},
                headers={"content-type": "application/json"},
            )

        client = QSRClient(endpoint="http://localhost:27900")
        client.client.close()
        client.client = httpx.Client(transport=httpx.MockTransport(handler))
        client._cached_default_model = "qwen3-asr-0.6b"

        response = client.transcribe(
            audio=str(audio_path),
            response_format="json",
            timestamp_granularities=["segment", "word"],
        )

        assert response.text == "ok"
        assert "multipart/form-data" in captured["content_type"]
        assert b'name="model"' in captured["body"]
        assert b"qwen3-asr-0.6b" in captured["body"]
        assert b'name="timestamp_granularities[]"' in captured["body"]
        assert b'filename="sample.wav"' in captured["body"]
        client.close()

    def test_load_audio_upload_caches_remote_downloads(self):
        url = "https://example.com/cache-audio.wav"
        response = MagicMock()
        response.raise_for_status.return_value = None
        response.content = b"RIFF1234WAVEfmt "
        response.headers = {"content-type": "audio/wav"}

        with patch("tfmx.qsrs.client.httpx.get", return_value=response) as mock_get:
            first = _load_audio_upload(url)
            second = _load_audio_upload(url)

        assert first == second
        assert mock_get.call_count == 1


class TestAsyncQSRClient:
    def test_resolve_model_fetches_models_once_for_concurrent_requests(self):
        client = AsyncQSRClient(endpoint="http://localhost:27900")
        call_count = 0

        async def fake_models() -> ModelInfo:
            nonlocal call_count
            call_count += 1
            await asyncio.sleep(0)
            return ModelInfo(models=["qwen3-asr-0.6b"])

        async def run_test() -> None:
            with patch.object(client, "models", side_effect=fake_models):
                resolved = await asyncio.gather(
                    *[client._resolve_model("") for _ in range(8)]
                )
            assert resolved == ["qwen3-asr-0.6b"] * 8
            assert call_count == 1
            await client.close()

        asyncio.run(run_test())
