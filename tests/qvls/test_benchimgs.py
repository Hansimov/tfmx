"""Tests for tfmx.qvls.benchimgs module"""

import pytest
import base64

from tfmx.qvls.benchimgs import (
    QVLBenchImageGenerator,
    _generate_synthetic_image_bytes,
    _image_bytes_to_data_url,
    _minimal_png,
    VL_PROMPTS,
    TEXT_PROMPTS,
    CN_TEXT_PROMPTS,
)


class TestMinimalPng:
    """Test fallback PNG generator."""

    def test_generates_bytes(self):
        data = _minimal_png(size=4)
        assert isinstance(data, bytes)
        assert len(data) > 0

    def test_png_signature(self):
        data = _minimal_png(size=4)
        assert data[:4] == b"\x89PNG"

    def test_different_sizes(self):
        small = _minimal_png(size=2)
        large = _minimal_png(size=16)
        assert len(large) > len(small)


class TestSyntheticImageBytes:
    """Test synthetic image generation."""

    def test_generates_bytes(self):
        data = _generate_synthetic_image_bytes(size=32)
        assert isinstance(data, bytes)
        assert len(data) > 0

    def test_different_sizes(self):
        small = _generate_synthetic_image_bytes(size=32)
        large = _generate_synthetic_image_bytes(size=128)
        # Larger images should generally produce more bytes
        assert isinstance(small, bytes)
        assert isinstance(large, bytes)

    def test_deterministic_with_seed(self):
        import random

        rng1 = random.Random(42)
        rng2 = random.Random(42)
        img1 = _generate_synthetic_image_bytes(size=32, rng=rng1)
        img2 = _generate_synthetic_image_bytes(size=32, rng=rng2)
        assert img1 == img2


class TestImageBytesToDataUrl:
    """Test data URL conversion."""

    def test_png(self):
        data = b"\x89PNG\r\n\x1a\n"
        url = _image_bytes_to_data_url(data, "image/png")
        assert url.startswith("data:image/png;base64,")

    def test_base64_valid(self):
        data = b"test image data"
        url = _image_bytes_to_data_url(data)
        _, b64_part = url.split(",", 1)
        decoded = base64.b64decode(b64_part)
        assert decoded == data


class TestQVLBenchImageGenerator:
    """Test QVLBenchImageGenerator."""

    def test_generate_with_images(self):
        gen = QVLBenchImageGenerator(seed=42)
        samples = gen.generate(count=5, img_size=32)
        assert len(samples) == 5
        for s in samples:
            assert "messages" in s
            msgs = s["messages"]
            assert len(msgs) >= 1
            user_msg = msgs[0]
            assert user_msg["role"] == "user"
            assert isinstance(user_msg["content"], list)

    def test_generate_has_image_and_text(self):
        gen = QVLBenchImageGenerator(seed=42)
        samples = gen.generate(count=1, img_size=32)
        content = samples[0]["messages"][0]["content"]
        types = [p["type"] for p in content]
        assert "image_url" in types
        assert "text" in types

    def test_generate_text_only(self):
        gen = QVLBenchImageGenerator(seed=42)
        samples = gen.generate_text_only(count=5)
        assert len(samples) == 5
        for s in samples:
            msgs = s["messages"]
            assert isinstance(msgs[0]["content"], str)

    def test_generate_mixed(self):
        gen = QVLBenchImageGenerator(seed=42)
        samples = gen.generate_mixed(count=10, image_ratio=0.5, img_size=32)
        assert len(samples) == 10
        # Should have a mix of image and text-only
        has_image = sum(
            1 for s in samples if isinstance(s["messages"][0]["content"], list)
        )
        has_text = sum(
            1 for s in samples if isinstance(s["messages"][0]["content"], str)
        )
        assert has_image > 0
        assert has_text > 0

    def test_deterministic(self):
        gen1 = QVLBenchImageGenerator(seed=123)
        gen2 = QVLBenchImageGenerator(seed=123)
        s1 = gen1.generate_text_only(count=5)
        s2 = gen2.generate_text_only(count=5)
        for a, b in zip(s1, s2):
            assert a["messages"][0]["content"] == b["messages"][0]["content"]


class TestPromptTemplates:
    """Test prompt template collections."""

    def test_vl_prompts_not_empty(self):
        assert len(VL_PROMPTS) > 0

    def test_text_prompts_not_empty(self):
        assert len(TEXT_PROMPTS) > 0

    def test_cn_text_prompts_not_empty(self):
        assert len(CN_TEXT_PROMPTS) > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
