"""Tests for tfmx.qwns.compose."""

from pathlib import Path
from unittest.mock import patch

import pytest

from tfmx.qwns.compose import AWQ_MODELS
from tfmx.qwns.compose import AWQ_QUANT_LEVELS
from tfmx.qwns.compose import ComposeFileGenerator
from tfmx.qwns.compose import DEFAULT_AWQ_MODEL
from tfmx.qwns.compose import DEFAULT_QUANT_LEVEL
from tfmx.qwns.compose import DEFAULT_QUANT_METHOD
from tfmx.qwns.compose import GPUInfo
from tfmx.qwns.compose import GPU_LAYOUT_UNIFORM_AWQ
from tfmx.qwns.compose import GpuModelConfig
from tfmx.qwns.compose import MACHINE_PORT
from tfmx.qwns.compose import MODEL_NAME
from tfmx.qwns.compose import MODEL_SHORTCUTS
from tfmx.qwns.compose import QWNComposer
from tfmx.qwns.compose import SERVER_PORT
from tfmx.qwns.compose import SUPPORTED_MODELS
from tfmx.qwns.compose import VLLM_INTERNAL_PORT
from tfmx.qwns.compose import VLLM_BASE_IMAGE
from tfmx.qwns.compose import build_gpu_configs_for_layout
from tfmx.qwns.compose import get_model_shortcut
from tfmx.qwns.compose import infer_gpu_ids
from tfmx.qwns.compose import parse_gpu_configs
from tfmx.qwns.networking import QWNNetworkConfig


class TestConstants:
    def test_port_constants(self):
        assert SERVER_PORT == 27880
        assert MACHINE_PORT == 27800
        assert VLLM_INTERNAL_PORT == 8000

    def test_versioned_base_image(self):
        assert VLLM_BASE_IMAGE.endswith(":v0.19.0")

    def test_default_model(self):
        assert MODEL_NAME in SUPPORTED_MODELS
        assert DEFAULT_AWQ_MODEL in AWQ_MODELS
        assert DEFAULT_QUANT_LEVEL in AWQ_QUANT_LEVELS


class TestModelShortcuts:
    def test_shortcuts_cover_default_model(self):
        assert MODEL_SHORTCUTS["4b"] == MODEL_NAME

    def test_get_model_shortcut_from_awq_repo(self):
        assert get_model_shortcut(DEFAULT_AWQ_MODEL) == "4b"

    def test_get_model_shortcut_from_label(self):
        assert get_model_shortcut("4b:4bit") == "4b"


class TestGpuModelConfig:
    def test_default_config(self):
        config = GpuModelConfig(gpu_id=0)
        assert config.gpu_id == 0
        assert config.quant_method == DEFAULT_QUANT_METHOD
        assert config.quant_level == DEFAULT_QUANT_LEVEL
        assert config.vllm_model_arg == DEFAULT_AWQ_MODEL

    def test_served_model_name(self):
        config = GpuModelConfig(gpu_id=0)
        assert config.served_model_name == "4b:4bit"


class TestParseGpuConfigs:
    def test_parse_single_config(self):
        configs = parse_gpu_configs("0:4b:4bit")
        assert len(configs) == 1
        assert configs[0].gpu_id == 0
        assert configs[0].label == "4b:4bit"

    def test_parse_case_insensitive_model(self):
        configs = parse_gpu_configs("0:4B:4BIT")
        assert configs[0].label == "4b:4bit"

    def test_duplicate_gpu_ids_raise(self):
        with pytest.raises(ValueError, match="Duplicate GPU ID"):
            parse_gpu_configs("0:4b:4bit,0:4b:4bit")


class TestGpuLayouts:
    def test_infer_gpu_ids_from_configs(self):
        configs = parse_gpu_configs("0:4b:4bit,2:4b:4bit")
        assert infer_gpu_ids(None, configs) == "0,2"

    def test_build_uniform_awq_layout(self):
        configs = build_gpu_configs_for_layout(
            GPU_LAYOUT_UNIFORM_AWQ,
            [GPUInfo(index=0, compute_cap="8.6"), GPUInfo(index=1, compute_cap="8.6")],
        )
        assert [config.label for config in configs] == ["4b:4bit", "4b:4bit"]


class TestComposeFileGenerator:
    def test_generate_single_gpu(self):
        generator = ComposeFileGenerator(
            gpus=[GPUInfo(index=0, compute_cap="8.9")],
            model_name=MODEL_NAME,
            port=SERVER_PORT,
            project_name="qwn-test",
            data_dir=Path("/tmp/qwn-test"),
        )
        compose_text = generator.generate()
        assert "qwn-gpu0" in compose_text
        assert "--served-model-name" in compose_text
        assert "--reasoning-parser qwen3" in compose_text
        assert DEFAULT_AWQ_MODEL in compose_text
        assert "PIP_INDEX_URL=https://mirrors.ustc.edu.cn/pypi/simple" in compose_text
        assert "HF_ENDPOINT=https://hf-mirror.com" in compose_text

    def test_generate_single_gpu_with_local_proxy(self):
        local_proxy = "http://localhost:" + "18080"
        generator = ComposeFileGenerator(
            gpus=[GPUInfo(index=0, compute_cap="8.9")],
            model_name=MODEL_NAME,
            port=SERVER_PORT,
            project_name="qwn-test",
            data_dir=Path("/tmp/qwn-test"),
            http_proxy=local_proxy,
        )
        compose_text = generator.generate()
        assert "HTTP_PROXY=" not in compose_text
        assert "host.docker.internal:host-gateway" not in compose_text

    def test_generate_single_gpu_with_remote_proxy(self):
        generator = ComposeFileGenerator(
            gpus=[GPUInfo(index=0, compute_cap="8.9")],
            model_name=MODEL_NAME,
            port=SERVER_PORT,
            project_name="qwn-test",
            data_dir=Path("/tmp/qwn-test"),
            http_proxy="http://proxy.internal:8080",
        )
        compose_text = generator.generate()
        assert "HTTP_PROXY=http://proxy.internal:8080" in compose_text

    def test_generate_single_gpu_with_network_config(self):
        generator = ComposeFileGenerator(
            gpus=[GPUInfo(index=0, compute_cap="8.9")],
            model_name=MODEL_NAME,
            port=SERVER_PORT,
            project_name="qwn-test",
            data_dir=Path("/tmp/qwn-test"),
            network_config=QWNNetworkConfig.from_overrides(
                proxy="http://proxy.internal:8080",
                hf_endpoint="https://hf-mirror.example",
                pip_index_url="https://mirror.example/pypi/simple",
                pip_trusted_host="mirror.example",
            ),
        )
        compose_text = generator.generate()
        assert "HF_ENDPOINT=https://hf-mirror.example" in compose_text
        assert "PIP_INDEX_URL=https://mirror.example/pypi/simple" in compose_text

    @patch("tfmx.qwns.compose.ModelConfigManager.get_model_snapshot_dir")
    def test_generate_enables_runtime_offline_when_model_cached(
        self, mock_snapshot_dir, tmp_path
    ):
        snapshot_dir = tmp_path / "snapshot"
        snapshot_dir.mkdir()
        (snapshot_dir / "config.json").write_text("{}")
        (snapshot_dir / "preprocessor_config.json").write_text("{}")
        mock_snapshot_dir.return_value = snapshot_dir

        generator = ComposeFileGenerator(
            gpus=[GPUInfo(index=0, compute_cap="8.9")],
            model_name=MODEL_NAME,
            port=SERVER_PORT,
            project_name="qwn-test",
            data_dir=Path("/tmp/qwn-test"),
        )

        compose_text = generator.generate()
        assert "HF_HUB_OFFLINE=1" in compose_text
        assert "TRANSFORMERS_OFFLINE=1" in compose_text

    @patch("tfmx.qwns.compose.ModelConfigManager.get_model_snapshot_dir")
    def test_generate_keeps_runtime_online_without_cache(self, mock_snapshot_dir):
        mock_snapshot_dir.return_value = None

        generator = ComposeFileGenerator(
            gpus=[GPUInfo(index=0, compute_cap="8.9")],
            model_name=MODEL_NAME,
            port=SERVER_PORT,
            project_name="qwn-test",
            data_dir=Path("/tmp/qwn-test"),
        )

        compose_text = generator.generate()
        assert "HF_HUB_OFFLINE=1" not in compose_text
        assert "TRANSFORMERS_OFFLINE=1" not in compose_text


class TestQWNComposer:
    @patch("tfmx.qwns.compose.GPUDetector.detect")
    def test_init(self, mock_detect):
        mock_detect.return_value = [GPUInfo(index=0, compute_cap="8.6")]
        composer = QWNComposer(project_name="qwn-test")
        assert composer.project_name == "qwn-test"

    @patch("tfmx.qwns.compose.GPUDetector.detect")
    def test_gpu_configs_limit_detected_gpus(self, mock_detect):
        mock_detect.return_value = [GPUInfo(index=0, compute_cap="8.6")]
        composer = QWNComposer(gpu_configs=parse_gpu_configs("0:4b:4bit"))
        mock_detect.assert_called_once_with("0")
        assert [gpu.index for gpu in composer.gpus] == [0]

    @patch("tfmx.qwns.compose.GPUDetector.detect")
    def test_gpu_layout_builds_configs(self, mock_detect):
        mock_detect.return_value = [
            GPUInfo(index=0, compute_cap="8.6"),
            GPUInfo(index=1, compute_cap="8.6"),
        ]
        composer = QWNComposer(gpu_layout=GPU_LAYOUT_UNIFORM_AWQ)
        assert [config.label for config in composer.gpu_configs] == [
            "4b:4bit",
            "4b:4bit",
        ]
