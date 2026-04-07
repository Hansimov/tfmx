"""Tests for tfmx.teis.compose."""

from pathlib import Path
from unittest.mock import patch

import pytest

from tfmx.teis.compose import ComposeFileGenerator
from tfmx.teis.compose import GPUInfo
from tfmx.teis.compose import GpuModelConfig
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


class TestTEIComposer:
    @patch("tfmx.teis.compose.GPUDetector.detect")
    def test_gpu_configs_limit_detected_gpus(self, mock_detect):
        mock_detect.return_value = [GPUInfo(index=0, compute_cap="8.6")]

        composer = TEIComposer(gpu_configs=parse_gpu_configs("0"))

        mock_detect.assert_called_once_with("0")
        assert composer.gpu_ids == "0"
