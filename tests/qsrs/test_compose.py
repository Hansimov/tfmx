"""Tests for tfmx.qsrs.compose."""

from pathlib import Path

import pytest

from tfmx.qsrs.compose import ComposeFileGenerator
from tfmx.qsrs.compose import GPUInfo
from tfmx.qsrs.compose import GPU_MEMORY_UTILIZATION
from tfmx.qsrs.compose import GPU_LAYOUT_UNIFORM
from tfmx.qsrs.compose import GpuModelConfig
from tfmx.qsrs.compose import HEALTHCHECK_INTERVAL
from tfmx.qsrs.compose import HEALTHCHECK_START_PERIOD
from tfmx.qsrs.compose import HEALTHCHECK_TCP_PROBE
from tfmx.qsrs.compose import MACHINE_PORT
from tfmx.qsrs.compose import MAX_MODEL_LEN
from tfmx.qsrs.compose import MAX_NUM_SEQS
from tfmx.qsrs.compose import MODEL_NAME
from tfmx.qsrs.compose import MODEL_SHORTCUTS
from tfmx.qsrs.compose import QWEN_ASR_HF_HUB_SPEC
from tfmx.qsrs.compose import QWEN_ASR_SPEC
from tfmx.qsrs.compose import VLLM_AUDIO_SPEC
from tfmx.qsrs.compose import SERVER_PORT
from tfmx.qsrs.compose import SUPPORTED_MODELS
from tfmx.qsrs.compose import VLLM_BASE_IMAGE
from tfmx.qsrs.compose import VLLM_INTERNAL_PORT
from tfmx.qsrs.compose import build_gpu_configs_for_layout
from tfmx.qsrs.compose import get_model_shortcut
from tfmx.qsrs.compose import infer_gpu_ids
from tfmx.qsrs.compose import parse_gpu_configs


class TestConstants:
    def test_port_constants(self):
        assert SERVER_PORT == 27980
        assert MACHINE_PORT == 27900
        assert VLLM_INTERNAL_PORT == 8000

    def test_versioned_base_image(self):
        assert VLLM_BASE_IMAGE.endswith(":v0.19.0")

    def test_runtime_dependency_pins(self):
        assert VLLM_AUDIO_SPEC == "vllm[audio]==0.19.0"
        assert QWEN_ASR_SPEC == "qwen-asr==0.0.6"
        assert QWEN_ASR_HF_HUB_SPEC == "huggingface-hub>=0.34.0,<1.0"
        assert "[vllm]" not in QWEN_ASR_SPEC

    def test_default_model(self):
        assert MODEL_NAME in SUPPORTED_MODELS
        assert MODEL_SHORTCUTS["0.6b"] == MODEL_NAME

    def test_runtime_defaults_are_tuned_for_asr(self):
        assert MAX_MODEL_LEN == 4096
        assert MAX_NUM_SEQS == 8
        assert GPU_MEMORY_UTILIZATION == 0.35


class TestModelShortcuts:
    def test_get_model_shortcut_from_default_model(self):
        assert get_model_shortcut(MODEL_NAME) == "0.6b"

    def test_get_model_shortcut_from_label(self):
        assert get_model_shortcut("qwen3-asr-0.6b") == "0.6b"


class TestGpuModelConfig:
    def test_default_config(self):
        config = GpuModelConfig(gpu_id=0)
        assert config.gpu_id == 0
        assert config.model_name == MODEL_NAME
        assert config.served_model_name == "0.6b"


class TestParseGpuConfigs:
    def test_parse_gpu_only_config_uses_defaults(self):
        configs = parse_gpu_configs("0,2")
        assert [config.gpu_id for config in configs] == [0, 2]
        assert [config.label for config in configs] == ["0.6b", "0.6b"]

    def test_parse_single_config(self):
        configs = parse_gpu_configs("0:Qwen/Qwen3-ASR-0.6B")
        assert len(configs) == 1
        assert configs[0].gpu_id == 0
        assert configs[0].model_name == MODEL_NAME

    def test_duplicate_gpu_ids_raise(self):
        with pytest.raises(ValueError, match="Duplicate GPU ID"):
            parse_gpu_configs("0,0")


class TestGpuLayouts:
    def test_infer_gpu_ids_from_configs(self):
        configs = parse_gpu_configs("0,2")
        assert infer_gpu_ids(None, configs) == "0,2"

    def test_build_uniform_layout(self):
        configs = build_gpu_configs_for_layout(
            GPU_LAYOUT_UNIFORM,
            [GPUInfo(index=0, compute_cap="8.6"), GPUInfo(index=1, compute_cap="8.6")],
        )
        assert [config.label for config in configs] == ["0.6b", "0.6b"]


class TestComposeFileGenerator:
    def test_generate_single_gpu(self):
        generator = ComposeFileGenerator(
            gpus=[GPUInfo(index=0, compute_cap="8.9")],
            model_name=MODEL_NAME,
            port=SERVER_PORT,
            project_name="qsr-test",
            data_dir=Path("/tmp/qsr-test"),
        )
        compose_text = generator.generate()
        assert "qsr-gpu0" in compose_text
        assert "--served-model-name" in compose_text
        assert MODEL_NAME in compose_text
        assert "PIP_INDEX_URL=https://mirrors.ustc.edu.cn/pypi/simple" in compose_text
        assert "HF_ENDPOINT=https://hf-mirror.com" in compose_text
        assert "CUDA_VISIBLE_DEVICES=0" in compose_text
        assert "NVIDIA_VISIBLE_DEVICES=0" in compose_text
        assert "${HOME}/.cache/vllm:/root/.cache/vllm" in compose_text
        assert "--max-model-len 4096" in compose_text
        assert "--max-num-seqs 8" in compose_text
        assert "--gpu-memory-utilization 0.35" in compose_text
        assert "CMD-SHELL" in compose_text
        assert HEALTHCHECK_TCP_PROBE in compose_text
        assert f"interval: {HEALTHCHECK_INTERVAL}" in compose_text
        assert f"start_period: {HEALTHCHECK_START_PERIOD}" in compose_text
