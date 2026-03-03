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
    GpuModelConfig,
    parse_gpu_configs,
    SUPPORTED_MODELS,
    AWQ_MODELS,
    MODEL_SHORTCUTS,
    MODEL_SHORTCUT_REV,
    AWQ_REPO_MAP,
    AWQ_QUANT_LEVELS,
    DEFAULT_QUANT_METHOD,
    DEFAULT_QUANT_LEVEL,
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
    normalize_model_key,
    resolve_model_name,
    resolve_quant_level,
    get_model_shortcut,
    get_display_shortcut,
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

    def test_awq_models(self):
        assert len(AWQ_MODELS) >= 6
        for awq_id, base_id in AWQ_MODELS.items():
            assert "cyankiwi" in awq_id
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

    def test_get_model_config_awq(self):
        mgr = ModelConfigManager()
        config = mgr.get_model_config("cyankiwi/Qwen3-VL-8B-Instruct-AWQ-4bit")
        assert config.get("awq") is True

    def test_get_model_config_unknown(self):
        mgr = ModelConfigManager()
        config = mgr.get_model_config("nonexistent/model")
        assert config["size"] == "unknown"

    def test_quantization_recommendation(self):
        mgr = ModelConfigManager()
        rec = mgr.get_quantization_recommendation(MODEL_NAME)
        assert rec in ("none", "awq", "bitsandbytes")


class TestDockerImageManager:
    """Test DockerImageManager (static methods)."""

    def test_class_exists(self):
        assert hasattr(DockerImageManager, "ensure_image")


class TestComposeFileGenerator:
    """Test ComposeFileGenerator."""

    def test_generate_single_gpu(self):
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

    def test_no_sitecustomize_mount(self):
        """AWQ does not need sitecustomize monkey-patch."""
        gpu = GPUInfo(index=0, compute_cap="8.9")
        gen = ComposeFileGenerator(
            gpus=[gpu],
            model_name="Qwen/Qwen3-VL-8B-Instruct",
            port=SERVER_PORT,
            project_name="test-qvl",
            data_dir=Path("/tmp/test-qvl"),
        )
        compose = gen.generate()
        assert "sitecustomize" not in compose
        assert "PYTHONPATH" not in compose


class TestQVLComposer:
    """Test QVLComposer main class."""

    def test_init(self):
        composer = QVLComposer()
        assert composer is not None


class TestModelShortcuts:
    """Test model shortcut mappings."""

    def test_shortcuts_cover_all_models(self):
        assert len(MODEL_SHORTCUTS) >= 6
        assert "2b-instruct" in MODEL_SHORTCUTS
        assert "4b-thinking" in MODEL_SHORTCUTS
        assert "8b-instruct" in MODEL_SHORTCUTS

    def test_shortcuts_resolve_to_supported(self):
        for shortcut, full_name in MODEL_SHORTCUTS.items():
            assert (
                full_name in SUPPORTED_MODELS
            ), f"{shortcut} -> {full_name} not in SUPPORTED_MODELS"

    def test_reverse_mapping(self):
        for shortcut, full_name in MODEL_SHORTCUTS.items():
            assert MODEL_SHORTCUT_REV[full_name] == shortcut


class TestAWQMappings:
    """Test AWQ repo mappings."""

    def test_awq_repo_map_covers_all_shortcuts(self):
        for shortcut in MODEL_SHORTCUTS:
            assert (shortcut, "4bit") in AWQ_REPO_MAP or (
                shortcut, "8bit"
            ) in AWQ_REPO_MAP, f"{shortcut} not in AWQ_REPO_MAP"

    def test_awq_repo_map_values(self):
        for key, repo in AWQ_REPO_MAP.items():
            assert "cyankiwi" in repo
            assert "AWQ" in repo

    def test_awq_quant_levels(self):
        assert "4bit" in AWQ_QUANT_LEVELS
        assert "8bit" in AWQ_QUANT_LEVELS
        assert len(AWQ_QUANT_LEVELS) == 2

    def test_default_awq_constants(self):
        assert DEFAULT_QUANT_METHOD == "awq"
        assert DEFAULT_QUANT_LEVEL == "4bit"


class TestGpuModelConfig:
    """Test GpuModelConfig dataclass."""

    def test_default_config(self):
        gc = GpuModelConfig(gpu_id=0)
        assert gc.gpu_id == 0
        assert gc.model_name == MODEL_NAME
        assert gc.quant_method == DEFAULT_QUANT_METHOD
        assert gc.quant_level == DEFAULT_QUANT_LEVEL

    def test_model_shortcut(self):
        gc = GpuModelConfig(gpu_id=0, model_name="Qwen/Qwen3-VL-8B-Instruct")
        assert gc.model_shortcut == "8b-instruct"

    def test_model_shortcut_unknown(self):
        gc = GpuModelConfig(gpu_id=0, model_name="unknown/model")
        assert gc.model_shortcut == "model"

    def test_awq_repo(self):
        gc = GpuModelConfig(
            gpu_id=0,
            model_name="Qwen/Qwen3-VL-8B-Instruct",
            quant_method="awq",
            quant_level="4bit",
        )
        assert gc.awq_repo == "cyankiwi/Qwen3-VL-8B-Instruct-AWQ-4bit"

    def test_awq_repo_8bit(self):
        gc = GpuModelConfig(
            gpu_id=0,
            model_name="Qwen/Qwen3-VL-8B-Instruct",
            quant_method="awq",
            quant_level="8bit",
        )
        assert gc.awq_repo == "cyankiwi/Qwen3-VL-8B-Instruct-AWQ-8bit"

    def test_awq_repo_none_for_non_awq(self):
        gc = GpuModelConfig(gpu_id=0, quant_method="none")
        assert gc.awq_repo is None

    def test_vllm_model_arg_awq(self):
        gc = GpuModelConfig(
            gpu_id=0,
            model_name="Qwen/Qwen3-VL-8B-Instruct",
            quant_method="awq",
            quant_level="4bit",
        )
        assert gc.vllm_model_arg == "cyankiwi/Qwen3-VL-8B-Instruct-AWQ-4bit"

    def test_vllm_model_arg_awq_8bit(self):
        gc = GpuModelConfig(
            gpu_id=0,
            model_name="Qwen/Qwen3-VL-8B-Instruct",
            quant_method="awq",
            quant_level="8bit",
        )
        assert gc.vllm_model_arg == "cyankiwi/Qwen3-VL-8B-Instruct-AWQ-8bit"

    def test_vllm_model_arg_non_awq(self):
        gc = GpuModelConfig(
            gpu_id=0,
            model_name="Qwen/Qwen3-VL-8B-Instruct",
            quant_method="none",
        )
        assert gc.vllm_model_arg == "Qwen/Qwen3-VL-8B-Instruct"

    def test_label(self):
        gc = GpuModelConfig(
            gpu_id=0,
            model_name="Qwen/Qwen3-VL-8B-Instruct",
            quant_level="4bit",
        )
        assert gc.label == "8b-instruct:4bit"

    def test_label_no_quant(self):
        gc = GpuModelConfig(
            gpu_id=0,
            model_name="Qwen/Qwen3-VL-8B-Instruct",
            quant_level="",
        )
        assert gc.label == "8b-instruct"

    def test_to_dict(self):
        gc = GpuModelConfig(gpu_id=0)
        d = gc.to_dict()
        assert "gpu_id" in d
        assert "model_name" in d
        assert "model_shortcut" in d
        assert "quant_method" in d
        assert "quant_level" in d
        assert "awq_repo" in d


class TestParseGpuConfigs:
    """Test parse_gpu_configs function."""

    def test_single_config(self):
        configs = parse_gpu_configs("0:8B-Instruct:4bit")
        assert len(configs) == 1
        assert configs[0].gpu_id == 0
        assert configs[0].model_name == "Qwen/Qwen3-VL-8B-Instruct"
        assert configs[0].quant_level == "4bit"

    def test_multiple_configs(self):
        configs = parse_gpu_configs(
            "0:2B-Instruct:4bit,1:4B-Instruct:4bit,2:8B-Instruct:8bit"
        )
        assert len(configs) == 3
        assert configs[0].model_name == "Qwen/Qwen3-VL-2B-Instruct"
        assert configs[1].model_name == "Qwen/Qwen3-VL-4B-Instruct"
        assert configs[2].quant_level == "8bit"

    def test_default_quant(self):
        configs = parse_gpu_configs("0:8B-Instruct")
        assert len(configs) == 1
        assert configs[0].quant_level == DEFAULT_QUANT_LEVEL

    def test_full_model_name(self):
        configs = parse_gpu_configs("0:Qwen/Qwen3-VL-8B-Instruct:8bit")
        assert len(configs) == 1
        assert configs[0].model_name == "Qwen/Qwen3-VL-8B-Instruct"

    def test_empty_string(self):
        configs = parse_gpu_configs("")
        assert len(configs) == 0

    def test_invalid_format_raises(self):
        with pytest.raises(ValueError, match="Invalid config"):
            parse_gpu_configs("invalid")

    def test_six_gpu_deployment(self):
        config_str = (
            "0:2B-Instruct:4bit,"
            "1:4B-Instruct:4bit,"
            "2:8B-Instruct:4bit,"
            "3:4B-Thinking:4bit,"
            "4:8B-Instruct:8bit,"
            "5:8B-Thinking:8bit"
        )
        configs = parse_gpu_configs(config_str)
        assert len(configs) == 6
        assert configs[3].model_name == "Qwen/Qwen3-VL-4B-Thinking"
        assert configs[4].quant_level == "8bit"
        assert configs[5].model_name == "Qwen/Qwen3-VL-8B-Thinking"


class TestComposeWithGpuConfigs:
    """Test ComposeFileGenerator with per-GPU configs."""

    def test_generate_with_gpu_configs(self):
        gpus = [GPUInfo(index=i, compute_cap="8.9") for i in range(2)]
        configs = [
            GpuModelConfig(
                gpu_id=0,
                model_name="Qwen/Qwen3-VL-2B-Instruct",
                quant_method="awq",
                quant_level="4bit",
            ),
            GpuModelConfig(
                gpu_id=1,
                model_name="Qwen/Qwen3-VL-8B-Instruct",
                quant_method="awq",
                quant_level="8bit",
            ),
        ]
        gen = ComposeFileGenerator(
            gpus=gpus,
            model_name=MODEL_NAME,
            port=SERVER_PORT,
            project_name="test-multi",
            data_dir=Path("/tmp/test-multi"),
            gpu_configs=configs,
        )
        compose = gen.generate()
        assert "gpu0" in compose
        assert "gpu1" in compose
        assert "cyankiwi/Qwen3-VL-2B-Instruct-AWQ-4bit" in compose
        assert "cyankiwi/Qwen3-VL-8B-Instruct-AWQ-8bit" in compose
        assert "--quantization" in compose
        assert "awq" in compose
        assert "--tokenizer" not in compose
        assert "--hf-config-path" not in compose

    def test_header_shows_per_gpu_info(self):
        gpus = [GPUInfo(index=0, compute_cap="8.9")]
        configs = [
            GpuModelConfig(
                gpu_id=0,
                model_name="Qwen/Qwen3-VL-8B-Instruct",
                quant_method="awq",
                quant_level="4bit",
            ),
        ]
        gen = ComposeFileGenerator(
            gpus=gpus,
            model_name=MODEL_NAME,
            port=SERVER_PORT,
            project_name="test-configured",
            data_dir=Path("/tmp/test"),
            gpu_configs=configs,
        )
        compose = gen.generate()
        assert "Per-GPU configs" in compose
        assert "8b-instruct:4bit" in compose


class TestCaseInsensitiveHelpers:
    """Test case-insensitive model/quant helper functions."""

    def test_normalize_model_key(self):
        assert normalize_model_key("8B-Instruct") == "8b-instruct"
        assert normalize_model_key("4bit") == "4bit"
        assert (
            normalize_model_key("Qwen/Qwen3-VL-8B-Instruct")
            == "qwen/qwen3-vl-8b-instruct"
        )
        assert normalize_model_key("") == ""
        assert normalize_model_key("  8B-Instruct  ") == "8b-instruct"

    def test_resolve_model_name_from_shortcut(self):
        assert resolve_model_name("8b-instruct") == "Qwen/Qwen3-VL-8B-Instruct"
        assert resolve_model_name("8B-Instruct") == "Qwen/Qwen3-VL-8B-Instruct"
        assert resolve_model_name("8B-INSTRUCT") == "Qwen/Qwen3-VL-8B-Instruct"
        assert resolve_model_name("2b-thinking") == "Qwen/Qwen3-VL-2B-Thinking"

    def test_resolve_model_name_full(self):
        assert (
            resolve_model_name("Qwen/Qwen3-VL-8B-Instruct")
            == "Qwen/Qwen3-VL-8B-Instruct"
        )
        assert (
            resolve_model_name("qwen/qwen3-vl-8b-instruct")
            == "Qwen/Qwen3-VL-8B-Instruct"
        )

    def test_resolve_model_name_unknown(self):
        assert resolve_model_name("unknown/model") == "unknown/model"

    def test_resolve_quant_level(self):
        assert resolve_quant_level("4bit") == "4bit"
        assert resolve_quant_level("4BIT") == "4bit"
        assert resolve_quant_level("8bit") == "8bit"

    def test_get_model_shortcut(self):
        assert get_model_shortcut("Qwen/Qwen3-VL-8B-Instruct") == "8b-instruct"
        assert get_model_shortcut("Qwen/Qwen3-VL-2B-Thinking") == "2b-thinking"

    def test_get_display_shortcut(self):
        assert get_display_shortcut("8b-instruct") == "8B-Instruct"
        assert get_display_shortcut("2b-thinking") == "2B-Thinking"
        assert get_display_shortcut("8B-Instruct") == "8B-Instruct"

    def test_model_shortcuts_all_lowercase_keys(self):
        for key in MODEL_SHORTCUTS:
            assert (
                key == key.lower()
            ), f"MODEL_SHORTCUTS key '{key}' should be lowercase"

    def test_awq_quant_levels_all_lowercase(self):
        for level in AWQ_QUANT_LEVELS:
            assert level == level.lower()

    def test_parse_gpu_configs_case_insensitive(self):
        configs1 = parse_gpu_configs("0:8B-Instruct:4bit")
        configs2 = parse_gpu_configs("0:8b-instruct:4bit")
        configs3 = parse_gpu_configs("0:8B-INSTRUCT:4BIT")
        for cfg in [configs1[0], configs2[0], configs3[0]]:
            assert cfg.model_name == "Qwen/Qwen3-VL-8B-Instruct"
            assert cfg.quant_level == "4bit"

    def test_gpu_model_config_normalizes_on_init(self):
        gc = GpuModelConfig(
            gpu_id=0,
            model_name="8B-Instruct",
            quant_method="AWQ",
            quant_level="4BIT",
        )
        assert gc.model_name == "Qwen/Qwen3-VL-8B-Instruct"
        assert gc.quant_method == "awq"
        assert gc.quant_level == "4bit"

    def test_gpu_model_config_display_shortcut(self):
        gc = GpuModelConfig(gpu_id=0, model_name="Qwen/Qwen3-VL-8B-Instruct")
        assert gc.display_shortcut == "8B-Instruct"

    def test_gpu_model_config_display_label(self):
        gc = GpuModelConfig(
            gpu_id=0,
            model_name="Qwen/Qwen3-VL-8B-Instruct",
            quant_level="4bit",
        )
        assert gc.display_label == "8B-Instruct:4BIT"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
