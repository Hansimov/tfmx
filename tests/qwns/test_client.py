"""Tests for tfmx.qwns.client."""

from unittest.mock import MagicMock

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
from tfmx.qwns.client import format_elapsed_time
from tfmx.qwns.client import format_stream_stats_line
from tfmx.qwns.client import join_prompt_texts


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
        assert "elapsed=" in stats
        assert "ttft=" in stats
        assert "token/s" in stats


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


class TestAsyncQWNClient:
    def test_reset(self):
        client = AsyncQWNClient(endpoint="http://localhost:27800")
        client.reset()
        assert client._client is None
