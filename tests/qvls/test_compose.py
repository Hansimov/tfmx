"""Tests for tfmx.qvls.compose module"""

import pytest
import json
from pathlib import Path

from tfmx.qvls.compose import (
    GPUInfo,
    GPUDetector,
    ModelConfigManager,
    DockerImageManager,
    ComposeFileGenerator,
    QVLComposer,
    SUPPORTED_MODELS,
    GGUF_MODELS,
    SERVER_PORT,
    MACHINE_PORT,
    MAX_CONCURRENT_REQUESTS,
    MAX_MODEL_LEN,
    MAX_NUM_SEQS,
    VLLM_IMAGE,
    VLLM_INTERNAL_PORT,
    MODEL_NAME,
    QUANT_RECOMMENDATIONS,
    CACHE_HF_HUB,
)


class TestConstants:
    """Test module-level constants."""

    def test_port_constants(self):
        assert SERVER_PORT == 29880
        assert MACHINE_PORT == 29800
        assert VLLM_INTERNAL_PORT == 8000

    def test_model_constants(self):
        assert MAX_CONCURRENT_REQUESTS > 0
        assert MAX_MODEL_LEN > 0
        assert MAX_NUM_SEQS > 0

    def test_supported_models(self):
        assert len(SUPPORTED_MODELS) >= 6
        for model_id, info in SUPPORTED_MODELS.items():
            assert "size" in info
            assert "type" in info
            assert info["type"] in ("instruct", "thinking")

    def test_gguf_models(self):
        assert len(GGUF_MODELS) >= 3
        for gguf_id, base_id in GGUF_MODELS.items():
            assert "unsloth" in gguf_id
            assert base_id in SUPPORTED_MODELS

    def test_default_model_is_supported(self):
        assert MODEL_NAME in SUPPORTED_MODELS

    def test_quant_recommendations(self):
        assert len(QUANT_RECOMMENDATIONS) > 0
        for key, rec in QUANT_RECOMMENDATIONS.items():
            assert "recommended_quant" in rec
            assert "min_vram_gb" in rec


class TestGPUInfo:
    """Test GPUInfo class."""

    def test_creation(self):
        gpu = GPUInfo(index=0, compute_cap="8.9")
        assert gpu.index == 0
        assert gpu.compute_cap == "8.9"

    def test_arch_name(self):
        gpu = GPUInfo(index=0, compute_cap="8.9")
        assert "RTX 40" in gpu.arch_name

    def test_image(self):
        gpu = GPUInfo(index=0, compute_cap="8.6")
        assert "vllm" in gpu.image


class TestModelConfigManager:
    """Test ModelConfigManager."""

    def test_init(self):
        mgr = ModelConfigManager()
        assert mgr.cache_hf_hub == CACHE_HF_HUB

    def test_get_model_config_supported(self):
        mgr = ModelConfigManager()
        config = mgr.get_model_config(MODEL_NAME)
        assert "size" in config
        assert "type" in config

    def test_get_model_config_gguf(self):
        mgr = ModelConfigManager()
        config = mgr.get_model_config("unsloth/Qwen3-VL-8B-Instruct-GGUF")
        assert config.get("gguf") is True

    def test_get_model_config_unknown(self):
        mgr = ModelConfigManager()
        config = mgr.get_model_config("nonexistent/model")
        assert config["size"] == "unknown"

    def test_quantization_recommendation(self):
        mgr = ModelConfigManager()
        rec = mgr.get_quantization_recommendation(MODEL_NAME)
        assert rec in ("none", "gguf", "bitsandbytes", "awq")


class TestDockerImageManager:
    """Test DockerImageManager (static methods)."""

    def test_class_exists(self):
        # DockerImageManager uses static methods for ensure_image
        assert hasattr(DockerImageManager, "ensure_image")


class TestComposeFileGenerator:
    """Test ComposeFileGenerator."""

    def test_generate_single_gpu(self):
        from pathlib import Path

        gpu = GPUInfo(index=0, compute_cap="8.9")
        gen = ComposeFileGenerator(
            gpus=[gpu],
            model_name="Qwen/Qwen3-VL-8B-Instruct",
            port=SERVER_PORT,
            project_name="test-qvl",
            data_dir=Path("/tmp/test-qvl"),
        )
        compose = gen.generate()
        assert isinstance(compose, str)
        assert "services" in compose
        assert "gpu0" in compose

    def test_generate_multi_gpu(self):
        from pathlib import Path

        gpus = [GPUInfo(index=i, compute_cap="8.9") for i in range(3)]
        gen = ComposeFileGenerator(
            gpus=gpus,
            model_name="Qwen/Qwen3-VL-8B-Instruct",
            port=SERVER_PORT,
            project_name="test-qvl",
            data_dir=Path("/tmp/test-qvl"),
        )
        compose = gen.generate()
        assert "gpu0" in compose
        assert "gpu1" in compose
        assert "gpu2" in compose


class TestQVLComposer:
    """Test QVLComposer main class."""

    def test_init(self):
        composer = QVLComposer()
        assert composer is not None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
