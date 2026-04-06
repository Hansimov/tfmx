"""Tests for tfmx.qwns.client."""

import httpx
import pytest

from unittest.mock import MagicMock, patch

from tfmx.qwns.client import AsyncQWNClient
from tfmx.qwns.client import ChatChoice
from tfmx.qwns.client import ChatMessage
from tfmx.qwns.client import ChatResponse
from tfmx.qwns.client import ChatUsage
from tfmx.qwns.client import InfoResponse
from tfmx.qwns.client import ModelInfo
from tfmx.qwns.client import QWNClient
from tfmx.qwns.client import StreamChatResult
from tfmx.qwns.client import build_multimodal_messages
from tfmx.qwns.client import build_text_messages
from tfmx.qwns.client import DEFAULT_MAX_TOKENS
from tfmx.qwns.client import format_elapsed_time
from tfmx.qwns.client import format_stream_stats_line
from tfmx.qwns.client import join_prompt_texts
from tfmx.qwns.client import _get_retry_max_tokens
from tfmx.qwns.client import _normalize_error_detail


class TestBuildTextMessages:
    def test_prompt_only(self):
        messages = build_text_messages("Hello")
        assert messages == [{"role": "user", "content": "Hello"}]

    def test_with_system_prompt(self):
        messages = build_text_messages("Hello", system_prompt="Be helpful")
        assert messages[0]["role"] == "system"
        assert messages[1]["role"] == "user"


class TestMultimodalMessages:
    def test_join_prompt_texts(self):
        assert join_prompt_texts(["Hello", "World"]) == "Hello\n\nWorld"

    def test_build_multimodal_messages_interleaves_images(self):
        messages = build_multimodal_messages(
            texts=["先看第一张图", "再比较第二张图"],
            images=[
                "data:image/png;base64,AAA",
                "data:image/png;base64,BBB",
            ],
            system_prompt="你是一个视觉助手",
        )
        assert messages[0]["role"] == "system"
        assert messages[1]["content"][0]["type"] == "text"
        assert messages[1]["content"][1]["type"] == "image_url"
        assert messages[1]["content"][2]["text"] == "再比较第二张图"
        assert messages[1]["content"][3]["image_url"]["url"].endswith("BBB")


class TestFormatElapsedTime:
    def test_seconds_only(self):
        assert format_elapsed_time(8.34) == "8.3s"

    def test_minutes_and_seconds(self):
        assert format_elapsed_time(75.25) == "1min 15.2s"


class TestFormatStreamStatsLine:
    def test_includes_ttft_and_rate(self):
        stats = format_stream_stats_line(
            StreamChatResult(
                text="你好",
                usage=ChatUsage(completion_tokens=12, total_tokens=20),
                elapsed_sec=3.0,
                first_token_latency_sec=0.8,
            )
        )
        assert "[stats]" in stats
        assert "elapsed=" in stats
        assert "ttft=" in stats
        assert "token/s" in stats


class TestErrorNormalization:
    def test_normalizes_nested_json_error_message(self):
        detail = _normalize_error_detail(
            {
                "message": (
                    '{"error":{"message":"max_tokens=8192 cannot be greater than '
                    'max_total_tokens=4096"}}'
                )
            }
        )
        assert detail == "max_tokens=8192 cannot be greater than max_total_tokens=4096"

    def test_extracts_retry_limit_from_error_message(self):
        detail = (
            "max_tokens=8192 cannot be greater than max_model_len=max_total_tokens=4096. "
            "Please request fewer output tokens."
        )
        assert _get_retry_max_tokens(detail, 8192) == 4096

    def test_extracts_retry_limit_from_prompt_context_error(self):
        detail = (
            "This model's maximum context length is 4096 tokens. However, you requested "
            "4096 output tokens and your prompt contains 110 characters (more than 0 "
            "characters, which is the upper bound for 0 input tokens)."
        )
        assert _get_retry_max_tokens(detail, 4096) == 3986


class TestChatResponse:
    def test_text_property(self):
        response = ChatResponse(
            id="chat-1",
            model="4b:4bit",
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
                "port": 27800,
                "instances": [
                    {
                        "name": "qwn--gpu0",
                        "endpoint": "http://localhost:27880",
                        "healthy": True,
                    }
                ],
                "stats": {"total_requests": 1},
                "available_models": ["4b:4bit"],
            }
        )
        assert info.port == 27800
        assert len(info.instances) == 1
        assert info.available_models == ["4b:4bit"]


class TestQWNClient:
    def test_default_endpoint(self):
        client = QWNClient()
        assert "27800" in client.endpoint
        client.close()

    def test_custom_endpoint(self):
        client = QWNClient(endpoint="http://myhost:29999")
        assert client.endpoint == "http://myhost:29999"
        client.close()

    def test_stream_chat_collects_text_and_usage(self):
        client = QWNClient(endpoint="http://localhost:27800")

        stream_response = MagicMock()
        stream_response.iter_lines.return_value = iter(
            [
                'data: {"choices":[{"delta":{"reasoning_content":"先想一下"}}]}',
                'data: {"choices":[{"delta":{"content":"你"}}]}',
                'data: {"choices":[{"delta":{"content":"好"}}]}',
                'data: {"choices":[],"usage":{"prompt_tokens":3,"completion_tokens":2,"total_tokens":5}}',
                "data: [DONE]",
            ]
        )
        stream_response.raise_for_status.return_value = None

        stream_context = MagicMock()
        stream_context.__enter__.return_value = stream_response
        stream_context.__exit__.return_value = None
        client.client.stream = MagicMock(return_value=stream_context)

        chunks: list[str] = []
        result = client.stream_chat(
            messages=build_text_messages("你好"),
            on_text=chunks.append,
        )

        assert result.text == "你好"
        assert chunks == ["你", "好"]
        assert result.usage.total_tokens == 5
        assert result.first_token_latency_sec > 0
        stream_kwargs = client.client.stream.call_args.kwargs
        assert stream_kwargs["json"]["chat_template_kwargs"] == {
            "enable_thinking": False
        }
        client.close()

    def test_stream_chat_ignores_whitespace_for_ttft(self):
        client = QWNClient(endpoint="http://localhost:27800")

        stream_response = MagicMock()
        stream_response.iter_lines.return_value = iter(
            [
                'data: {"choices":[{"delta":{"content":"\\n\\n"}}]}',
                'data: {"choices":[{"delta":{"content":"你好"}}]}',
                "data: [DONE]",
            ]
        )
        stream_response.raise_for_status.return_value = None

        stream_context = MagicMock()
        stream_context.__enter__.return_value = stream_response
        stream_context.__exit__.return_value = None
        client.client.stream = MagicMock(return_value=stream_context)

        with patch(
            "tfmx.qwns.client.time.perf_counter",
            side_effect=[10.0, 11.2, 11.5],
        ):
            result = client.stream_chat(messages=build_text_messages("你好"))

        assert result.first_token_latency_sec == pytest.approx(1.2)
        client.close()

    def test_stream_chat_can_enable_thinking(self):
        client = QWNClient(endpoint="http://localhost:27800")

        stream_response = MagicMock()
        stream_response.iter_lines.return_value = iter(["data: [DONE]"])
        stream_response.raise_for_status.return_value = None

        stream_context = MagicMock()
        stream_context.__enter__.return_value = stream_response
        stream_context.__exit__.return_value = None
        client.client.stream = MagicMock(return_value=stream_context)

        client.stream_chat(
            messages=build_text_messages("你好"),
            enable_thinking=True,
        )

        stream_kwargs = client.client.stream.call_args.kwargs
        assert stream_kwargs["json"]["chat_template_kwargs"] == {
            "enable_thinking": True
        }
        client.close()

    def test_stream_chat_can_render_thinking_tags(self):
        client = QWNClient(endpoint="http://localhost:27800")

        stream_response = MagicMock()
        stream_response.iter_lines.return_value = iter(
            [
                'data: {"choices":[{"delta":{"reasoning_content":"先想一下"}}]}',
                'data: {"choices":[{"delta":{"content":"答案"}}]}',
                "data: [DONE]",
            ]
        )
        stream_response.raise_for_status.return_value = None

        stream_context = MagicMock()
        stream_context.__enter__.return_value = stream_response
        stream_context.__exit__.return_value = None
        client.client.stream = MagicMock(return_value=stream_context)

        chunks: list[str] = []
        result = client.stream_chat(
            messages=build_text_messages("你好"),
            on_text=chunks.append,
            enable_thinking=True,
            include_thinking_tags=True,
        )

        assert result.text.startswith("<thinking>\n")
        assert "</thinking>" in result.text
        assert chunks[0] == "<thinking>\n"
        assert any("答案" in chunk for chunk in chunks)
        client.close()

    def test_stream_chat_retries_when_machine_returns_sse_max_token_error(self):
        client = QWNClient(endpoint="http://localhost:27800")
        seen_max_tokens: list[int] = []

        first_response = MagicMock()
        first_response.iter_lines.return_value = iter(
            [
                'data: {"error":{"message":"{\\"error\\":{\\"message\\":\\"max_tokens=8192 cannot be greater than max_total_tokens=4096\\"}}"}}',
                "data: [DONE]",
            ]
        )
        first_response.raise_for_status.return_value = None

        second_response = MagicMock()
        second_response.iter_lines.return_value = iter(
            [
                'data: {"choices":[{"delta":{"content":"你好"}}]}',
                'data: {"choices":[],"usage":{"prompt_tokens":3,"completion_tokens":2,"total_tokens":5}}',
                "data: [DONE]",
            ]
        )
        second_response.raise_for_status.return_value = None

        first_context = MagicMock()
        first_context.__enter__.return_value = first_response
        first_context.__exit__.return_value = None

        second_context = MagicMock()
        second_context.__enter__.return_value = second_response
        second_context.__exit__.return_value = None

        def stream_side_effect(*args, **kwargs):
            seen_max_tokens.append(kwargs["json"]["max_tokens"])
            if len(seen_max_tokens) == 1:
                return first_context
            return second_context

        client.client.stream = MagicMock(side_effect=stream_side_effect)

        result = client.stream_chat(
            messages=build_text_messages("你好"),
            enable_thinking=True,
            include_thinking_tags=True,
        )

        assert result.text.startswith("<thinking>\n")
        assert "你好" in result.text
        assert seen_max_tokens == [8192, 4096]
        client.close()

    def test_stream_chat_retries_twice_for_context_window_error(self):
        client = QWNClient(endpoint="http://localhost:27800")
        seen_max_tokens: list[int] = []

        first_response = MagicMock()
        first_response.iter_lines.return_value = iter(
            [
                'data: {"error":{"message":"max_tokens=8192 cannot be greater than max_total_tokens=4096"}}',
                "data: [DONE]",
            ]
        )
        first_response.raise_for_status.return_value = None

        second_response = MagicMock()
        second_response.iter_lines.return_value = iter(
            [
                'data: {"error":{"message":"This model\'s maximum context length is 4096 tokens. However, you requested 4096 output tokens and your prompt contains 110 characters (more than 0 characters, which is the upper bound for 0 input tokens)."}}',
                "data: [DONE]",
            ]
        )
        second_response.raise_for_status.return_value = None

        third_response = MagicMock()
        third_response.iter_lines.return_value = iter(
            [
                'data: {"choices":[{"delta":{"content":"你好"}}]}',
                'data: {"choices":[],"usage":{"prompt_tokens":3,"completion_tokens":2,"total_tokens":5}}',
                "data: [DONE]",
            ]
        )
        third_response.raise_for_status.return_value = None

        first_context = MagicMock()
        first_context.__enter__.return_value = first_response
        first_context.__exit__.return_value = None

        second_context = MagicMock()
        second_context.__enter__.return_value = second_response
        second_context.__exit__.return_value = None

        third_context = MagicMock()
        third_context.__enter__.return_value = third_response
        third_context.__exit__.return_value = None

        def stream_side_effect(*args, **kwargs):
            seen_max_tokens.append(kwargs["json"]["max_tokens"])
            if len(seen_max_tokens) == 1:
                return first_context
            if len(seen_max_tokens) == 2:
                return second_context
            return third_context

        client.client.stream = MagicMock(side_effect=stream_side_effect)

        result = client.stream_chat(messages=build_text_messages("你好"))

        assert result.text == "你好"
        assert seen_max_tokens == [8192, 4096, 3986]
        client.close()

    def test_chat_retries_when_http_error_reports_max_token_limit(self):
        client = QWNClient(endpoint="http://localhost:27800")
        seen_max_tokens: list[int] = []

        first_response = MagicMock()
        first_response.json.return_value = {
            "error": {
                "message": "max_tokens=8192 cannot be greater than max_total_tokens=4096"
            }
        }
        first_response.request = MagicMock()
        first_response.status_code = 400
        first_response.raise_for_status.side_effect = httpx.HTTPStatusError(
            "bad request",
            request=MagicMock(),
            response=first_response,
        )

        second_response = MagicMock()
        second_response.raise_for_status.return_value = None
        second_response.json.return_value = {
            "id": "ok",
            "model": "4b:4bit",
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": "ok"},
                    "finish_reason": "stop",
                }
            ],
            "usage": {"prompt_tokens": 3, "completion_tokens": 1, "total_tokens": 4},
        }

        def post_side_effect(*args, **kwargs):
            seen_max_tokens.append(kwargs["json"]["max_tokens"])
            if len(seen_max_tokens) == 1:
                return first_response
            return second_response

        client.client.post = MagicMock(side_effect=post_side_effect)

        result = client.chat(messages=build_text_messages("你好"))

        assert result.text == "ok"
        assert seen_max_tokens == [8192, 4096]
        client.close()

    def test_default_max_tokens_matches_model_len(self):
        client = QWNClient(endpoint="http://localhost:27800")

        stream_response = MagicMock()
        stream_response.iter_lines.return_value = iter(["data: [DONE]"])
        stream_response.raise_for_status.return_value = None

        stream_context = MagicMock()
        stream_context.__enter__.return_value = stream_response
        stream_context.__exit__.return_value = None
        client.client.stream = MagicMock(return_value=stream_context)

        client.stream_chat(messages=build_text_messages("你好"))

        stream_kwargs = client.client.stream.call_args.kwargs
        assert stream_kwargs["json"]["max_tokens"] == DEFAULT_MAX_TOKENS
        client.close()


class TestAsyncQWNClient:
    def test_reset(self):
        client = AsyncQWNClient(endpoint="http://localhost:27800")
        client.reset()
        assert client._client is None
