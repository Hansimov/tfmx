"""Tests for tfmx.qvls.benchimgs module"""

import pytest
import base64
from pathlib import Path

from tfmx.qvls.benchimgs import (
    QVLBenchImageGenerator,
    _generate_synthetic_image_bytes,
    _image_bytes_to_data_url,
    _image_to_data_url,
    _minimal_png,
    load_local_images,
    download_benchmark_images,
    VL_PROMPTS,
    CN_VL_PROMPTS,
    TEXT_PROMPTS,
    CN_TEXT_PROMPTS,
    DATA_DIR,
    BENCH_IMAGES_DIR,
    IMAGE_EXTENSIONS,
    HF_IMAGE_DATASETS,
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

    def test_cn_vl_prompts_not_empty(self):
        assert len(CN_VL_PROMPTS) > 0

    def test_text_prompts_not_empty(self):
        assert len(TEXT_PROMPTS) > 0

    def test_cn_text_prompts_not_empty(self):
        assert len(CN_TEXT_PROMPTS) > 0

    def test_all_prompts_are_strings(self):
        for prompt in VL_PROMPTS + CN_VL_PROMPTS + TEXT_PROMPTS + CN_TEXT_PROMPTS:
            assert isinstance(prompt, str)
            assert len(prompt) > 0


class TestDataConstants:
    """Test data directory constants."""

    def test_data_dir_is_path(self):
        assert isinstance(DATA_DIR, Path)

    def test_bench_images_dir_is_path(self):
        assert isinstance(BENCH_IMAGES_DIR, Path)

    def test_bench_images_under_data(self):
        assert str(BENCH_IMAGES_DIR).startswith(str(DATA_DIR))

    def test_image_extensions(self):
        assert ".jpg" in IMAGE_EXTENSIONS
        assert ".png" in IMAGE_EXTENSIONS
        assert ".jpeg" in IMAGE_EXTENSIONS

    def test_hf_datasets_config(self):
        assert len(HF_IMAGE_DATASETS) > 0
        for cfg in HF_IMAGE_DATASETS:
            assert "name" in cfg
            assert "repo" in cfg
            assert "split" in cfg
            assert "image_key" in cfg
            assert "max_samples" in cfg


class TestLoadLocalImages:
    """Test load_local_images function."""

    def test_nonexistent_dir(self):
        images = load_local_images(Path("/nonexistent/dir"))
        assert images == []

    def test_empty_dir(self, tmp_path):
        images = load_local_images(tmp_path)
        assert images == []

    def test_loads_images(self, tmp_path):
        (tmp_path / "img1.jpg").write_bytes(b"fake jpg")
        (tmp_path / "img2.png").write_bytes(b"fake png")
        (tmp_path / "not_image.txt").write_text("text")
        images = load_local_images(tmp_path)
        assert len(images) == 2

    def test_max_images(self, tmp_path):
        for i in range(10):
            (tmp_path / f"img_{i:03d}.jpg").write_bytes(b"fake")
        images = load_local_images(tmp_path, max_images=5)
        assert len(images) == 5

    def test_sorted_output(self, tmp_path):
        (tmp_path / "c.jpg").write_bytes(b"fake")
        (tmp_path / "a.jpg").write_bytes(b"fake")
        (tmp_path / "b.jpg").write_bytes(b"fake")
        images = load_local_images(tmp_path)
        names = [p.name for p in images]
        assert names == sorted(names)


class TestImageToDataUrl:
    """Test _image_to_data_url function."""

    def test_jpg_file(self, tmp_path):
        img_path = tmp_path / "test.jpg"
        img_path.write_bytes(b"\xff\xd8\xff\xe0test")
        url = _image_to_data_url(img_path)
        assert url.startswith("data:image/jpeg;base64,")

    def test_png_file(self, tmp_path):
        img_path = tmp_path / "test.png"
        img_path.write_bytes(b"\x89PNG\r\n\x1a\n")
        url = _image_to_data_url(img_path)
        assert url.startswith("data:image/png;base64,")

    def test_roundtrip(self, tmp_path):
        data = b"test image content"
        img_path = tmp_path / "test.png"
        img_path.write_bytes(data)
        url = _image_to_data_url(img_path)
        _, b64_part = url.split(",", 1)
        decoded = base64.b64decode(b64_part)
        assert decoded == data


class TestQVLBenchImageGeneratorWithLocalImages:
    """Test QVLBenchImageGenerator with local images."""

    def test_init_custom_dir(self, tmp_path):
        gen = QVLBenchImageGenerator(seed=42, image_dir=tmp_path)
        assert gen.image_dir == tmp_path

    def test_has_local_images_false(self, tmp_path):
        gen = QVLBenchImageGenerator(seed=42, image_dir=tmp_path)
        assert gen.has_local_images is False
        assert gen.local_image_count == 0

    def test_has_local_images_true(self, tmp_path):
        (tmp_path / "img.jpg").write_bytes(b"fake jpg")
        gen = QVLBenchImageGenerator(seed=42, image_dir=tmp_path)
        assert gen.has_local_images is True
        assert gen.local_image_count == 1

    def test_generate_uses_local_images(self, tmp_path):
        # Create a tiny valid image file
        png_data = _minimal_png(4)
        (tmp_path / "img.png").write_bytes(png_data)
        gen = QVLBenchImageGenerator(seed=42, image_dir=tmp_path)
        samples = gen.generate(count=3, img_size=32)
        assert len(samples) == 3
        # Check image data URL is from local file
        content = samples[0]["messages"][0]["content"]
        img_urls = [p for p in content if p.get("type") == "image_url"]
        assert len(img_urls) == 1


class TestDownloadBenchmarkImages:
    """Test download_benchmark_images function."""

    def test_missing_datasets_package(self, tmp_path, monkeypatch):
        """Test graceful handling when datasets package is not installed."""
        import builtins

        original_import = builtins.__import__

        def mock_import(name, *args, **kwargs):
            if name == "datasets":
                raise ImportError("No module named 'datasets'")
            return original_import(name, *args, **kwargs)

        monkeypatch.setattr(builtins, "__import__", mock_import)
        count = download_benchmark_images(output_dir=tmp_path, max_images=5)
        assert count == 0

    def test_unknown_dataset(self, tmp_path):
        count = download_benchmark_images(
            output_dir=tmp_path,
            max_images=5,
            dataset_name="nonexistent_dataset",
        )
        assert count == 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
