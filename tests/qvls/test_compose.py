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
    GGUF_MODELS,
    MODEL_SHORTCUTS,
    MODEL_SHORTCUT_REV,
    GGUF_REPO_MAP,
    GGUF_FILES,
    DEFAULT_QUANT_METHOD,
    DEFAULT_QUANT_LEVEL,
    DEFAULT_GGUF_REPO,
    DEFAULT_GGUF_FILE,
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


class TestModelShortcuts:
    """Test model shortcut mappings."""

    def test_shortcuts_cover_all_models(self):
        assert len(MODEL_SHORTCUTS) >= 6
        assert "2B-Instruct" in MODEL_SHORTCUTS
        assert "4B-Thinking" in MODEL_SHORTCUTS
        assert "8B-Instruct" in MODEL_SHORTCUTS

    def test_shortcuts_resolve_to_supported(self):
        for shortcut, full_name in MODEL_SHORTCUTS.items():
            assert (
                full_name in SUPPORTED_MODELS
            ), f"{shortcut} -> {full_name} not in SUPPORTED_MODELS"

    def test_reverse_mapping(self):
        for shortcut, full_name in MODEL_SHORTCUTS.items():
            assert MODEL_SHORTCUT_REV[full_name] == shortcut


class TestGGUFMappings:
    """Test GGUF repo and file mappings."""

    def test_gguf_repo_map_covers_supported(self):
        for full_name in MODEL_SHORTCUTS.values():
            assert full_name in GGUF_REPO_MAP, f"{full_name} not in GGUF_REPO_MAP"

    def test_gguf_repo_map_values(self):
        for base, repo in GGUF_REPO_MAP.items():
            assert "unsloth" in repo
            assert "GGUF" in repo

    def test_gguf_files_for_all_shortcuts(self):
        for shortcut in MODEL_SHORTCUTS:
            assert shortcut in GGUF_FILES, f"{shortcut} not in GGUF_FILES"
            files = GGUF_FILES[shortcut]
            assert "Q4_K_M" in files
            assert "Q8_0" in files

    def test_gguf_file_naming(self):
        for shortcut, files in GGUF_FILES.items():
            for quant, filename in files.items():
                assert shortcut in filename
                assert quant in filename
                assert filename.endswith(".gguf")

    def test_default_gguf_constants(self):
        assert DEFAULT_QUANT_METHOD == "gguf"
        assert DEFAULT_QUANT_LEVEL == "Q4_K_M"
        assert "unsloth" in DEFAULT_GGUF_REPO
        assert DEFAULT_GGUF_FILE.endswith(".gguf")


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
        assert gc.model_shortcut == "8B-Instruct"

    def test_model_shortcut_unknown(self):
        gc = GpuModelConfig(gpu_id=0, model_name="unknown/model")
        assert gc.model_shortcut == "model"

    def test_gguf_repo(self):
        gc = GpuModelConfig(
            gpu_id=0,
            model_name="Qwen/Qwen3-VL-8B-Instruct",
            quant_method="gguf",
        )
        assert gc.gguf_repo == "unsloth/Qwen3-VL-8B-Instruct-GGUF"

    def test_gguf_repo_none_for_non_gguf(self):
        gc = GpuModelConfig(gpu_id=0, quant_method="none")
        assert gc.gguf_repo is None

    def test_gguf_file(self):
        gc = GpuModelConfig(
            gpu_id=0,
            model_name="Qwen/Qwen3-VL-8B-Instruct",
            quant_method="gguf",
            quant_level="Q4_K_M",
        )
        assert gc.gguf_file == "Qwen3-VL-8B-Instruct-Q4_K_M.gguf"

    def test_vllm_model_arg_gguf(self):
        gc = GpuModelConfig(
            gpu_id=0,
            model_name="Qwen/Qwen3-VL-8B-Instruct",
            quant_method="gguf",
        )
        assert gc.vllm_model_arg == "unsloth/Qwen3-VL-8B-Instruct-GGUF"

    def test_vllm_model_arg_non_gguf(self):
        gc = GpuModelConfig(
            gpu_id=0,
            model_name="Qwen/Qwen3-VL-8B-Instruct",
            quant_method="none",
        )
        assert gc.vllm_model_arg == "Qwen/Qwen3-VL-8B-Instruct"

    def test_vllm_tokenizer_arg_gguf(self):
        gc = GpuModelConfig(
            gpu_id=0,
            model_name="Qwen/Qwen3-VL-8B-Instruct",
            quant_method="gguf",
        )
        assert gc.vllm_tokenizer_arg == "Qwen/Qwen3-VL-8B-Instruct"

    def test_vllm_tokenizer_arg_non_gguf(self):
        gc = GpuModelConfig(
            gpu_id=0,
            model_name="Qwen/Qwen3-VL-8B-Instruct",
            quant_method="none",
        )
        assert gc.vllm_tokenizer_arg is None

    def test_label(self):
        gc = GpuModelConfig(
            gpu_id=0,
            model_name="Qwen/Qwen3-VL-8B-Instruct",
            quant_level="Q4_K_M",
        )
        assert gc.label == "8B-Instruct:Q4_K_M"

    def test_label_no_quant(self):
        gc = GpuModelConfig(
            gpu_id=0,
            model_name="Qwen/Qwen3-VL-8B-Instruct",
            quant_level="",
        )
        assert gc.label == "8B-Instruct"

    def test_to_dict(self):
        gc = GpuModelConfig(gpu_id=0)
        d = gc.to_dict()
        assert "gpu_id" in d
        assert "model_name" in d
        assert "model_shortcut" in d
        assert "quant_method" in d
        assert "quant_level" in d
        assert "gguf_repo" in d
        assert "gguf_file" in d


class TestParseGpuConfigs:
    """Test parse_gpu_configs function."""

    def test_single_config(self):
        configs = parse_gpu_configs("0:8B-Instruct:Q4_K_M")
        assert len(configs) == 1
        assert configs[0].gpu_id == 0
        assert configs[0].model_name == "Qwen/Qwen3-VL-8B-Instruct"
        assert configs[0].quant_level == "Q4_K_M"

    def test_multiple_configs(self):
        configs = parse_gpu_configs(
            "0:2B-Instruct:Q4_K_M,1:4B-Instruct:Q4_K_M,2:8B-Instruct:Q8_0"
        )
        assert len(configs) == 3
        assert configs[0].model_name == "Qwen/Qwen3-VL-2B-Instruct"
        assert configs[1].model_name == "Qwen/Qwen3-VL-4B-Instruct"
        assert configs[2].quant_level == "Q8_0"

    def test_default_quant(self):
        configs = parse_gpu_configs("0:8B-Instruct")
        assert len(configs) == 1
        assert configs[0].quant_level == DEFAULT_QUANT_LEVEL

    def test_full_model_name(self):
        configs = parse_gpu_configs("0:Qwen/Qwen3-VL-8B-Instruct:Q8_0")
        assert len(configs) == 1
        # Full name is used as-is since it's not in shortcuts
        assert configs[0].model_name == "Qwen/Qwen3-VL-8B-Instruct"

    def test_empty_string(self):
        configs = parse_gpu_configs("")
        assert len(configs) == 0

    def test_invalid_format_raises(self):
        with pytest.raises(ValueError, match="Invalid config"):
            parse_gpu_configs("invalid")

    def test_six_gpu_deployment(self):
        config_str = (
            "0:2B-Instruct:Q4_K_M,"
            "1:4B-Instruct:Q4_K_M,"
            "2:8B-Instruct:Q4_K_M,"
            "3:4B-Thinking:Q4_K_M,"
            "4:8B-Instruct:Q8_0,"
            "5:8B-Thinking:Q4_K_M"
        )
        configs = parse_gpu_configs(config_str)
        assert len(configs) == 6
        assert configs[3].model_name == "Qwen/Qwen3-VL-4B-Thinking"
        assert configs[4].quant_level == "Q8_0"
        assert configs[5].model_name == "Qwen/Qwen3-VL-8B-Thinking"


class TestComposeWithGpuConfigs:
    """Test ComposeFileGenerator with per-GPU configs."""

    def test_generate_with_gpu_configs(self):
        gpus = [GPUInfo(index=i, compute_cap="8.9") for i in range(2)]
        configs = [
            GpuModelConfig(
                gpu_id=0,
                model_name="Qwen/Qwen3-VL-2B-Instruct",
                quant_method="gguf",
                quant_level="Q4_K_M",
            ),
            GpuModelConfig(
                gpu_id=1,
                model_name="Qwen/Qwen3-VL-8B-Instruct",
                quant_method="gguf",
                quant_level="Q8_0",
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
        # Should have different models
        assert "unsloth/Qwen3-VL-2B-Instruct-GGUF" in compose
        assert "unsloth/Qwen3-VL-8B-Instruct-GGUF" in compose
        # Should have tokenizer args for GGUF
        assert "--tokenizer" in compose
        assert "Qwen/Qwen3-VL-2B-Instruct" in compose

    def test_header_shows_per_gpu_info(self):
        gpus = [GPUInfo(index=0, compute_cap="8.9")]
        configs = [
            GpuModelConfig(
                gpu_id=0,
                model_name="Qwen/Qwen3-VL-8B-Instruct",
                quant_method="gguf",
                quant_level="Q4_K_M",
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
        assert "8B-Instruct:Q4_K_M" in compose


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
