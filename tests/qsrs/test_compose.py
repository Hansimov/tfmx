"""Tests for tfmx.qsrs.compose."""

from argparse import Namespace
from pathlib import Path
from unittest.mock import patch

import pytest

from tfmx.qsrs import compose
from tfmx.qsrs.compose import ComposeFileGenerator
from tfmx.qsrs.compose import CUDAGRAPH_MODE_CHOICES
from tfmx.qsrs.compose import DEFAULT_CUDAGRAPH_MODE
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
from tfmx.qsrs.compose import build_compilation_config


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
        assert DEFAULT_CUDAGRAPH_MODE is None


class TestCompilationConfig:
    def test_build_compilation_config_default(self):
        assert "NONE" in CUDAGRAPH_MODE_CHOICES
        assert build_compilation_config() is None

    def test_build_compilation_config_can_be_disabled(self):
        assert build_compilation_config(None) is None

    def test_build_compilation_config_with_override(self):
        assert build_compilation_config("NONE") == {"cudagraph_mode": "NONE"}


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

    def test_generate_single_gpu_can_enable_startup_shortcuts(self):
        generator = ComposeFileGenerator(
            gpus=[GPUInfo(index=0, compute_cap="8.9")],
            model_name=MODEL_NAME,
            port=SERVER_PORT,
            project_name="qsr-test",
            data_dir=Path("/tmp/qsr-test"),
            skip_mm_profiling=True,
            cudagraph_mode="NONE",
        )
        compose_text = generator.generate()

        assert "--skip-mm-profiling" in compose_text
        assert 'cudagraph_mode":"NONE"' in compose_text


class TestQSRComposerWarmup:
    def test_build_embedded_warmup_audio(self):
        filename, payload, mime_type = compose.QSRComposer._load_warmup_audio(None)

        assert filename == "qsr-warmup.wav"
        assert mime_type == "audio/wav"
        assert payload.startswith(b"RIFF")
        assert b"WAVE" in payload[:16]

    @patch.object(compose.QSRComposer, "wait_for_ready_backends", return_value=True)
    @patch("tfmx.qsrs.compose.GPUDetector.detect")
    def test_warmup_targets_running_backends(
        self,
        mock_detect,
        mock_wait_for_ready,
    ):
        mock_detect.return_value = [
            GPUInfo(index=0, compute_cap="8.6"),
            GPUInfo(index=1, compute_cap="8.6"),
        ]
        composer_obj = compose.QSRComposer(
            project_name="qsr-uniform",
            gpu_ids="0,1",
        )
        with patch.object(
            composer_obj,
            "_discover_running_backend_targets",
            side_effect=[
                [
                    ("qsr-uniform--gpu0", "http://localhost:27980"),
                    ("qsr-uniform--gpu1", "http://localhost:27981"),
                ],
                [
                    ("qsr-uniform--gpu0", "http://localhost:27980"),
                    ("qsr-uniform--gpu1", "http://localhost:27981"),
                ],
            ],
        ), patch.object(
            compose.QSRComposer,
            "_load_warmup_audio",
            return_value=("sample.wav", b"RIFF1234WAVEfmt ", "audio/wav"),
        ), patch.object(
            composer_obj,
            "_warmup_endpoint",
            side_effect=[(True, "ok"), (True, "ok")],
        ) as mock_warmup:
            composer_obj.warmup(
                audio="./sample.wav",
                wait_timeout_sec=12.0,
                poll_interval_sec=0.5,
                request_timeout_sec=34.0,
            )

        mock_wait_for_ready.assert_called_once_with(
            timeout_sec=12.0,
            poll_interval_sec=0.5,
            request_timeout_sec=1.0,
            label="[qsr]",
        )
        assert mock_warmup.call_count == 2
        assert mock_warmup.call_args_list[0].args == ("http://localhost:27980",)
        assert mock_warmup.call_args_list[0].kwargs == {
            "audio_upload": ("sample.wav", b"RIFF1234WAVEfmt ", "audio/wav"),
            "request_timeout_sec": 34.0,
        }

    @patch("tfmx.qsrs.compose.QSRComposer")
    def test_run_from_args_dispatches_warmup(self, mock_composer_cls):
        args = Namespace(
            compose_action="warmup",
            model_name=MODEL_NAME,
            port=SERVER_PORT,
            project_name=None,
            gpus=None,
            hf_token=None,
            mount_mode="manual",
            proxy=None,
            hf_endpoint=None,
            pip_index_url=None,
            pip_trusted_host=None,
            max_model_len=MAX_MODEL_LEN,
            max_num_seqs=MAX_NUM_SEQS,
            gpu_memory_utilization=GPU_MEMORY_UTILIZATION,
            skip_mm_profiling=True,
            gpu_layout=None,
            gpu_configs=None,
            audio="./sample.wav",
            wait_timeout=45.0,
            poll_interval=0.75,
            request_timeout=60.0,
        )

        compose.run_from_args(args)

        mock_composer_cls.return_value.warmup.assert_called_once_with(
            audio="./sample.wav",
            wait_timeout_sec=45.0,
            poll_interval_sec=0.75,
            request_timeout_sec=60.0,
        )

    @patch("tfmx.qsrs.compose.QSRComposer")
    def test_run_from_args_dispatches_up_with_default_warmup(self, mock_composer_cls):
        args = Namespace(
            compose_action="up",
            model_name=MODEL_NAME,
            port=SERVER_PORT,
            project_name=None,
            gpus=None,
            hf_token=None,
            mount_mode="manual",
            proxy=None,
            hf_endpoint=None,
            pip_index_url=None,
            pip_trusted_host=None,
            max_model_len=MAX_MODEL_LEN,
            max_num_seqs=MAX_NUM_SEQS,
            gpu_memory_utilization=GPU_MEMORY_UTILIZATION,
            skip_mm_profiling=True,
            gpu_layout=None,
            gpu_configs=None,
            skip_warmup=False,
            warmup_audio="./sample.wav",
            wait_timeout=45.0,
            poll_interval=0.75,
            request_timeout=60.0,
        )

        compose.run_from_args(args)

        mock_composer_cls.return_value.up.assert_called_once_with(
            skip_warmup=False,
            warmup_audio="./sample.wav",
            wait_timeout_sec=45.0,
            poll_interval_sec=0.75,
            request_timeout_sec=60.0,
        )
