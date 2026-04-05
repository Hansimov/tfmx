"""Tests for tfmx.qwns.client."""

from tfmx.qwns.client import AsyncQWNClient
from tfmx.qwns.client import ChatChoice
from tfmx.qwns.client import ChatMessage
from tfmx.qwns.client import ChatResponse
from tfmx.qwns.client import ChatUsage
from tfmx.qwns.client import InfoResponse
from tfmx.qwns.client import ModelInfo
from tfmx.qwns.client import QWNClient
from tfmx.qwns.client import build_text_messages


class TestBuildTextMessages:
    def test_prompt_only(self):
        messages = build_text_messages("Hello")
        assert messages == [{"role": "user", "content": "Hello"}]

    def test_with_system_prompt(self):
        messages = build_text_messages("Hello", system_prompt="Be helpful")
        assert messages[0]["role"] == "system"
        assert messages[1]["role"] == "user"


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


class TestAsyncQWNClient:
    def test_reset(self):
        client = AsyncQWNClient(endpoint="http://localhost:27800")
        client.reset()
        assert client._client is None
