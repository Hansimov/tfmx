"""Tests for tfmx.qvls.client module"""

import pytest
from unittest.mock import patch, MagicMock
from dataclasses import asdict

from tfmx.qvls.client import (
    QVLClient,
    AsyncQVLClient,
    HealthResponse,
    ModelInfo,
    ChatMessage,
    ChatUsage,
    ChatChoice,
    ChatResponse,
    InstanceInfo,
    build_vision_messages,
    _encode_image_to_base64,
)


class TestHealthResponse:
    """Test HealthResponse dataclass."""

    def test_creation(self):
        hr = HealthResponse(status="healthy", healthy=3, total=4)
        assert hr.status == "healthy"
        assert hr.healthy == 3
        assert hr.total == 4


class TestChatMessage:
    """Test ChatMessage dataclass."""

    def test_text_message(self):
        msg = ChatMessage(role="user", content="Hello")
        assert msg.role == "user"
        assert msg.content == "Hello"

    def test_multimodal_message(self):
        content = [
            {"type": "text", "text": "Describe this"},
            {"type": "image_url", "image_url": {"url": "data:image/png;base64,abc"}},
        ]
        msg = ChatMessage(role="user", content=content)
        assert isinstance(msg.content, list)
        assert len(msg.content) == 2


class TestChatUsage:
    """Test ChatUsage dataclass."""

    def test_creation(self):
        usage = ChatUsage(prompt_tokens=10, completion_tokens=20, total_tokens=30)
        assert usage.prompt_tokens == 10
        assert usage.completion_tokens == 20
        assert usage.total_tokens == 30


class TestChatChoice:
    """Test ChatChoice dataclass."""

    def test_creation(self):
        msg = ChatMessage(role="assistant", content="Hi there")
        choice = ChatChoice(index=0, message=msg, finish_reason="stop")
        assert choice.index == 0
        assert choice.message.content == "Hi there"


class TestChatResponse:
    """Test ChatResponse dataclass."""

    def test_text_property(self):
        msg = ChatMessage(role="assistant", content="Generated text")
        choice = ChatChoice(index=0, message=msg, finish_reason="stop")
        usage = ChatUsage(prompt_tokens=5, completion_tokens=3, total_tokens=8)
        resp = ChatResponse(
            id="chatcmpl-123",
            choices=[choice],
            usage=usage,
            model="test-model",
            created=1234567890,
        )
        assert resp.text == "Generated text"
        assert resp.usage.total_tokens == 8

    def test_text_empty_choices(self):
        usage = ChatUsage(prompt_tokens=0, completion_tokens=0, total_tokens=0)
        resp = ChatResponse(
            id="chatcmpl-empty",
            choices=[],
            usage=usage,
            model="test-model",
            created=0,
        )
        assert resp.text == ""


class TestBuildVisionMessages:
    """Test build_vision_messages helper."""

    def test_text_only(self):
        messages = build_vision_messages(prompt="Hello")
        assert len(messages) == 1
        assert messages[0]["role"] == "user"
        # build_vision_messages always uses content_parts format
        content = messages[0]["content"]
        assert isinstance(content, list)
        text_parts = [p for p in content if p.get("type") == "text"]
        assert len(text_parts) == 1
        assert text_parts[0]["text"] == "Hello"

    def test_with_system_prompt(self):
        messages = build_vision_messages(prompt="Hello", system_prompt="Be helpful")
        assert len(messages) == 2
        assert messages[0]["role"] == "system"
        assert messages[0]["content"] == "Be helpful"
        assert messages[1]["role"] == "user"

    def test_with_images_url(self):
        messages = build_vision_messages(
            prompt="Describe this",
            images=["https://example.com/img.jpg"],
        )
        assert len(messages) == 1
        user_msg = messages[0]
        assert isinstance(user_msg["content"], list)

        # Should have image_url and text parts
        types = [p["type"] for p in user_msg["content"]]
        assert "image_url" in types
        assert "text" in types

    def test_with_images_base64(self):
        messages = build_vision_messages(
            prompt="Describe",
            images=["data:image/png;base64,abc123"],
        )
        user_msg = messages[0]
        img_part = next(p for p in user_msg["content"] if p["type"] == "image_url")
        assert img_part["image_url"]["url"] == "data:image/png;base64,abc123"

    def test_multiple_images(self):
        messages = build_vision_messages(
            prompt="Compare these",
            images=[
                "https://example.com/img1.jpg",
                "https://example.com/img2.jpg",
            ],
        )
        user_msg = messages[0]
        img_parts = [p for p in user_msg["content"] if p["type"] == "image_url"]
        assert len(img_parts) == 2


class TestEncodeImage:
    """Test _encode_image_to_base64 (local file encoder only)."""

    def test_file_not_found(self):
        with pytest.raises(FileNotFoundError):
            _encode_image_to_base64("/nonexistent/image.png")

    def test_encode_real_file(self, tmp_path):
        # Create a small test file
        img_file = tmp_path / "test.png"
        img_file.write_bytes(b"\x89PNG\r\n\x1a\n" + b"\x00" * 20)
        result = _encode_image_to_base64(str(img_file))
        assert result.startswith("data:image/png;base64,")


class TestInstanceInfo:
    """Test InstanceInfo dataclass."""

    def test_creation(self):
        info = InstanceInfo(
            name="qvl--gpu0",
            endpoint="http://localhost:29880",
            gpu_id=0,
            healthy=True,
        )
        assert info.name == "qvl--gpu0"
        assert info.healthy is True

    def test_model_fields(self):
        info = InstanceInfo(
            name="qvl--gpu0",
            endpoint="http://localhost:29880",
            gpu_id=0,
            healthy=True,
            model_name="Qwen/Qwen3-VL-8B-Instruct",
            quant_method="gguf",
            quant_level="Q4_K_M",
            model_label="8B-Instruct:Q4_K_M",
        )
        assert info.model_name == "Qwen/Qwen3-VL-8B-Instruct"
        assert info.quant_method == "gguf"
        assert info.quant_level == "Q4_K_M"
        assert info.model_label == "8B-Instruct:Q4_K_M"

    def test_from_dict(self):
        data = {
            "name": "qvl--gpu0",
            "endpoint": "http://localhost:29880",
            "gpu_id": 0,
            "healthy": True,
            "model_name": "Qwen/Qwen3-VL-8B-Instruct",
            "quant_method": "gguf",
            "quant_level": "Q4_K_M",
            "model_label": "8B-Instruct:Q4_K_M",
        }
        info = InstanceInfo.from_dict(data)
        assert info.model_name == "Qwen/Qwen3-VL-8B-Instruct"
        assert info.model_label == "8B-Instruct:Q4_K_M"

    def test_from_dict_missing_model_fields(self):
        data = {
            "name": "qvl--gpu0",
            "endpoint": "http://localhost:29880",
            "gpu_id": 0,
            "healthy": True,
        }
        info = InstanceInfo.from_dict(data)
        assert info.model_name == ""
        assert info.quant_method == ""
        assert info.quant_level == ""
        assert info.model_label == ""


class TestQVLClient:
    """Test QVLClient (sync)."""

    def test_init_default(self):
        client = QVLClient()
        assert "localhost" in client.endpoint or "127.0.0.1" in client.endpoint

    def test_init_custom_endpoint(self):
        client = QVLClient(endpoint="http://myhost:29880")
        assert client.endpoint == "http://myhost:29880"
        client.close()

    def test_close(self):
        client = QVLClient()
        client.close()

    def test_context_manager(self):
        with QVLClient() as client:
            assert client is not None


class TestAsyncQVLClient:
    """Test AsyncQVLClient basics (no actual HTTP)."""

    def test_init(self):
        client = AsyncQVLClient(endpoint="http://localhost:29880")
        assert client.endpoint == "http://localhost:29880"

    def test_reset(self):
        client = AsyncQVLClient(endpoint="http://localhost:29880")
        client.reset()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
