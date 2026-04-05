"""Cross-module integration tests for tfmx.qwns."""

from tfmx.qwns.compose import DEFAULT_AWQ_MODEL
from tfmx.qwns.compose import GPUInfo
from tfmx.qwns.compose import GPU_LAYOUT_UNIFORM_AWQ
from tfmx.qwns.compose import GpuModelConfig
from tfmx.qwns.compose import build_gpu_configs_for_layout
from tfmx.qwns.compose import get_model_shortcut
from tfmx.qwns.compose import parse_gpu_configs
from tfmx.qwns.router import InstanceDescriptor
from tfmx.qwns.router import QWNRouter


class TestComposeRouterIntegration:
    def test_awq_model_shortcut(self):
        assert get_model_shortcut(DEFAULT_AWQ_MODEL) == "4b"

    def test_gpu_config_routable(self):
        configs = parse_gpu_configs("0:4b:4bit")
        router = QWNRouter()
        for config in configs:
            router.register(
                InstanceDescriptor(
                    model_name=config.served_model_name,
                    quant_method=config.quant_method,
                    quant_level=config.quant_level,
                    endpoint=f"http://localhost:{27880 + config.gpu_id}",
                )
            )
        assert router.route_from_model_field("4b:4bit") is not None

    def test_gpu_model_config_served_name(self):
        config = GpuModelConfig(gpu_id=0)
        assert config.served_model_name == "4b:4bit"

    def test_uniform_layout_routable(self):
        configs = build_gpu_configs_for_layout(
            GPU_LAYOUT_UNIFORM_AWQ,
            [GPUInfo(index=0, compute_cap="8.6"), GPUInfo(index=1, compute_cap="8.6")],
        )
        router = QWNRouter()
        for config in configs:
            router.register(
                InstanceDescriptor(
                    model_name=config.served_model_name,
                    quant_method=config.quant_method,
                    quant_level=config.quant_level,
                    endpoint=f"http://localhost:{27880 + config.gpu_id}",
                )
            )
        assert router.route_from_model_field("4b:4bit") is not None
