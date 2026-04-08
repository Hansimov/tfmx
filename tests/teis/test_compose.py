"""Tests for tfmx.teis.compose."""

from pathlib import Path
from unittest.mock import patch

import pytest

from tests.proxy_test_values import PLACEHOLDER_HOST_PROXY_URL
from tfmx.teis.compose import ComposeFileGenerator
from tfmx.teis.compose import GPUInfo
from tfmx.teis.compose import GpuModelConfig
from tfmx.teis.compose import HEALTHCHECK_TCP_PROBE_TEMPLATE
from tfmx.teis.compose import MODEL_NAME
from tfmx.teis.compose import SERVER_PORT
from tfmx.teis.compose import TEIComposer
from tfmx.teis.compose import infer_gpu_ids
from tfmx.teis.compose import parse_gpu_configs


class TestParseGpuConfigs:
    def test_parse_gpu_only_config_uses_default_model(self):
        configs = parse_gpu_configs("0,2")
        assert [config.gpu_id for config in configs] == [0, 2]
        assert [config.model_name for config in configs] == [MODEL_NAME, MODEL_NAME]

    def test_duplicate_gpu_ids_raise(self):
        with pytest.raises(ValueError, match="Duplicate GPU ID"):
            parse_gpu_configs("0,0")


class TestInferGpuIds:
    def test_infer_gpu_ids_from_configs(self):
        configs = [GpuModelConfig(gpu_id=0), GpuModelConfig(gpu_id=3)]
        assert infer_gpu_ids(None, configs) == "0,3"


class TestComposeFileGenerator:
    def test_generate_uses_per_gpu_model(self):
        generator = ComposeFileGenerator(
            gpus=[GPUInfo(index=0, compute_cap="8.6")],
            model_name=MODEL_NAME,
            port=SERVER_PORT,
            project_name="tei-test",
            data_dir=Path("/tmp/tei-test"),
            gpu_configs=parse_gpu_configs("0:Alibaba-NLP/gte-multilingual-base"),
        )

        compose_text = generator.generate()
        assert "Alibaba-NLP/gte-multilingual-base" in compose_text

    def test_generate_manual_mode_preserves_common_environment(self):
        generator = ComposeFileGenerator(
            gpus=[GPUInfo(index=0, compute_cap="8.6")],
            model_name=MODEL_NAME,
            port=SERVER_PORT,
            project_name="tei-test",
            data_dir=Path("/tmp/tei-test"),
        )

        compose_text = generator.generate()
        assert "HF_ENDPOINT=https://hf-mirror.com" in compose_text
        assert "CUDA_VISIBLE_DEVICES=0" in compose_text
        assert "NVIDIA_VISIBLE_DEVICES=0" in compose_text
        assert "    environment:" not in compose_text
        assert "CMD-SHELL" in compose_text
        assert HEALTHCHECK_TCP_PROBE_TEMPLATE.format(port=80) in compose_text

    def test_generate_host_mode_healthcheck_uses_bound_port(self):
        generator = ComposeFileGenerator(
            gpus=[GPUInfo(index=0, compute_cap="8.6")],
            model_name=MODEL_NAME,
            port=SERVER_PORT,
            project_name="tei-test",
            data_dir=Path("/tmp/tei-test"),
            http_proxy=PLACEHOLDER_HOST_PROXY_URL,
        )

        compose_text = generator.generate()

        assert "network_mode: host" in compose_text
        assert HEALTHCHECK_TCP_PROBE_TEMPLATE.format(port=SERVER_PORT) in compose_text


class TestTEIComposer:
    @patch("tfmx.teis.compose.GPUDetector.detect")
    def test_gpu_configs_limit_detected_gpus(self, mock_detect):
        mock_detect.return_value = [GPUInfo(index=0, compute_cap="8.6")]

        composer = TEIComposer(gpu_configs=parse_gpu_configs("0"))

        mock_detect.assert_called_once_with("0")
        assert composer.gpu_ids == "0"

    @patch("tfmx.teis.compose.wait_for_healthy_docker_containers", return_value=True)
    @patch("tfmx.teis.compose.GPUDetector.detect")
    def test_wait_for_healthy_backends_uses_docker_health(
        self,
        mock_detect,
        mock_wait,
    ):
        mock_detect.return_value = [GPUInfo(index=0, compute_cap="8.6")]

        composer = TEIComposer(project_name="tei-test")

        assert composer.wait_for_healthy_backends(label="[test]") is True
        mock_wait.assert_called_once_with(
            ["tei-test--gpu0"],
            timeout_sec=300.0,
            poll_interval_sec=5.0,
            label="[test]",
        )
