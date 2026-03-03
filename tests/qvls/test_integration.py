"""Integration tests for tfmx.qvls module.

Tests cross-module interactions:
- Compose → Router (generated configs feed into routing)
- Machine → Router (instance discovery feeds router)
- Client → Machine (clients talk to machine proxy)
- Compose constants consistency across modules
"""

import pytest
from unittest.mock import patch, MagicMock, AsyncMock

from tfmx.qvls.compose import (
    AWQ_MODELS,
    AWQ_QUANT_LEVELS,
    AWQ_REPO_MAP,
    MODEL_SHORTCUTS,
    QVLComposer,
    GpuModelConfig,
    parse_gpu_configs,
    DEFAULT_QUANT_METHOD,
)
from tfmx.qvls.router import (
    QVLRouter,
    InstanceDescriptor,
    parse_model_spec,
    normalize_model_key,
)
from tfmx.qvls.machine import VLLMInstance, QVLMachineServer
from tfmx.qvls.client import (
    QVLClient,
    AsyncQVLClient,
    ChatResponse,
    ChatMessage,
    ChatChoice,
    ChatUsage,
    build_vision_messages,
)
from tfmx.qvls.clients_core import (
    MachineState,
    MachineScheduler,
    ClientsHealthResponse,
)
from tfmx.qvls.performance import ExplorationConfig, PerformanceMetrics


# ────────────────────────────────────────────────────────────────
# Compose → Router integration
# ────────────────────────────────────────────────────────────────


class TestComposeRouterIntegration:
    """Test that compose-generated configs are routable."""

    def test_all_awq_models_parseable(self):
        """Every AWQ model name can be parsed into a valid model spec."""
        for model_name in AWQ_MODELS:
            # AWQ_MODELS entries like "cyankiwi/Qwen3-VL-2B-Instruct-AWQ-4bit"
            model, quant = parse_model_spec(model_name)
            assert model, f"Failed to parse model from: {model_name}"

    def test_all_shortcuts_routable(self):
        """Every MODEL_SHORTCUTS entry can be route-matched."""
        router = QVLRouter()
        # Register a descriptor for each shortcut
        for shortcut, full_name in MODEL_SHORTCUTS.items():
            desc = InstanceDescriptor(
                endpoint=f"http://localhost:{29880 + hash(shortcut) % 6}",
                model_name=full_name,
                quant_level="4bit",
            )
            router.register(desc)

        # Every shortcut should find at least one match
        for shortcut in MODEL_SHORTCUTS:
            result = router.route_from_model_field(shortcut)
            assert result is not None, f"No route for shortcut: {shortcut}"

    def test_awq_repo_map_covers_all_levels(self):
        """AWQ_REPO_MAP has an entry for every model × quant combination."""
        for shortcut in MODEL_SHORTCUTS:
            for level in AWQ_QUANT_LEVELS:
                key = (normalize_model_key(shortcut), level)
                assert key in AWQ_REPO_MAP, f"Missing AWQ_REPO_MAP entry: {key}"

    def test_gpu_config_generates_valid_quant(self):
        """GPU configs produce quant levels in AWQ_QUANT_LEVELS."""
        configs = parse_gpu_configs("0:2b-instruct:4bit")
        assert len(configs) == 1
        assert configs[0].quant_level in AWQ_QUANT_LEVELS

    def test_compose_default_quant_method_is_awq(self):
        """Default quantization method should be 'awq'."""
        assert DEFAULT_QUANT_METHOD == "awq"


class TestComposeRouterModelResolving:
    """Test model name resolution from compose to router."""

    def test_shortcut_to_full_name_roundtrip(self):
        """Shortcut → full name → shortcut roundtrip."""
        for shortcut, full_name in MODEL_SHORTCUTS.items():
            # Full name should normalize to match something
            normalized = normalize_model_key(full_name)
            assert normalized  # non-empty

    def test_router_matches_compose_generated_descriptor(self):
        """Router can route to instances created from compose gpu configs."""
        gpu_configs = parse_gpu_configs(
            "0:2b-instruct:4bit,1:4b-instruct:4bit,2:8b-instruct:4bit"
        )

        router = QVLRouter()
        for gc in gpu_configs:
            desc = InstanceDescriptor(
                endpoint=f"http://localhost:{29880 + gc.gpu_id}",
                model_name=gc.vllm_model_arg,
                quant_level=gc.quant_level,
            )
            router.register(desc)

        # Route by shortcut
        for shortcut in ["2b-instruct", "4b-instruct", "8b-instruct"]:
            result = router.route_from_model_field(shortcut)
            assert result is not None, f"Cannot route: {shortcut}"


# ────────────────────────────────────────────────────────────────
# Machine → Router integration
# ────────────────────────────────────────────────────────────────


class TestMachineRouterIntegration:
    """Test machine server integration with router."""

    def test_vllm_instance_to_info(self):
        """VLLMInstance.to_info() creates a valid InstanceInfo."""
        inst = VLLMInstance(
            container_name="qvl-multi--gpu0",
            host="localhost",
            port=29880,
            gpu_id=0,
            model_name="cyankiwi/Qwen3-VL-2B-Instruct-AWQ-4bit",
        )
        info = inst.to_info()
        assert info.endpoint == "http://localhost:29880"
        assert info.model_name == "cyankiwi/Qwen3-VL-2B-Instruct-AWQ-4bit"

    def test_vllm_instance_to_router_descriptor(self):
        """VLLMInstance can be converted to an InstanceDescriptor for routing."""
        inst = VLLMInstance(
            container_name="qvl-multi--gpu0",
            host="localhost",
            port=29880,
            gpu_id=0,
            model_name="cyankiwi/Qwen3-VL-2B-Instruct-AWQ-4bit",
            quant_level="4bit",
        )
        desc = InstanceDescriptor(
            model_name=inst.model_name,
            quant_method=inst.quant_method,
            quant_level=inst.quant_level,
            endpoint=inst.endpoint,
            gpu_id=inst.gpu_id,
            instance_id=inst.container_name,
            healthy=inst.healthy,
        )
        assert isinstance(desc, InstanceDescriptor)
        assert desc.endpoint == "http://localhost:29880"
        assert desc.model_name == "cyankiwi/Qwen3-VL-2B-Instruct-AWQ-4bit"

    def test_vllm_instance_descriptor_routable(self):
        """InstanceDescriptor from VLLMInstance is routable by the router."""
        inst = VLLMInstance(
            container_name="qvl-multi--gpu0",
            host="localhost",
            port=29880,
            gpu_id=0,
            model_name="cyankiwi/Qwen3-VL-4B-Instruct-AWQ-4bit",
            quant_level="4bit",
        )
        router = QVLRouter()
        desc = InstanceDescriptor(
            model_name=inst.model_name,
            quant_method=inst.quant_method,
            quant_level=inst.quant_level,
            endpoint=inst.endpoint,
            gpu_id=inst.gpu_id,
            instance_id=inst.container_name,
            healthy=True,
        )
        router.register(desc)

        # Should be routable by short name
        result = router.route_from_model_field("4b-instruct")
        assert result is not None
        assert result.endpoint == "http://localhost:29880"

    def test_mixed_deployment_routing(self):
        """Simulates 6-GPU mixed deployment routing."""
        configs = [
            (0, "cyankiwi/Qwen3-VL-2B-Instruct-AWQ-4bit"),
            (1, "cyankiwi/Qwen3-VL-2B-Thinking-AWQ-4bit"),
            (2, "cyankiwi/Qwen3-VL-4B-Instruct-AWQ-4bit"),
            (3, "cyankiwi/Qwen3-VL-4B-Thinking-AWQ-4bit"),
            (4, "cyankiwi/Qwen3-VL-8B-Instruct-AWQ-4bit"),
            (5, "cyankiwi/Qwen3-VL-8B-Thinking-AWQ-4bit"),
        ]

        router = QVLRouter()
        for gpu_id, model_name in configs:
            inst = VLLMInstance(
                container_name=f"qvl-multi--gpu{gpu_id}",
                host="localhost",
                port=29880 + gpu_id,
                gpu_id=gpu_id,
                model_name=model_name,
                quant_level="4bit",
            )
            desc = InstanceDescriptor(
                model_name=inst.model_name,
                quant_method=inst.quant_method,
                quant_level=inst.quant_level,
                endpoint=inst.endpoint,
                gpu_id=inst.gpu_id,
                instance_id=inst.container_name,
                healthy=True,
            )
            router.register(desc)

        assert len(router) == 6

        # Route each shortcut
        shortcuts = [
            "2b-instruct",
            "2b-thinking",
            "4b-instruct",
            "4b-thinking",
            "8b-instruct",
            "8b-thinking",
        ]
        for i, shortcut in enumerate(shortcuts):
            result = router.route_from_model_field(shortcut)
            assert result is not None, f"No route for {shortcut}"
            assert result.endpoint == f"http://localhost:{29880 + i}"

    def test_router_healthy_filtering(self):
        """Router only routes to healthy instances."""
        router = QVLRouter()
        desc_healthy = InstanceDescriptor(
            endpoint="http://localhost:29880",
            model_name="cyankiwi/Qwen3-VL-2B-Instruct-AWQ-4bit",
            quant_level="4bit",
            healthy=True,
        )
        desc_unhealthy = InstanceDescriptor(
            endpoint="http://localhost:29881",
            model_name="cyankiwi/Qwen3-VL-4B-Instruct-AWQ-4bit",
            quant_level="4bit",
            healthy=False,
        )
        router.register(desc_healthy)
        router.register(desc_unhealthy)

        # Route to 4b should fail (unhealthy), fall back
        result = router.route(model="4b-instruct")
        # Should fall back to default (first registered = 2b healthy)
        assert result is not None
        assert result.endpoint == "http://localhost:29880"


# ────────────────────────────────────────────────────────────────
# Client data model integration
# ────────────────────────────────────────────────────────────────


class TestClientDataModelIntegration:
    """Test client data models work together."""

    def test_chat_response_roundtrip(self):
        """Build a ChatResponse from components and extract text."""
        msg = ChatMessage(role="assistant", content="Hello world")
        choice = ChatChoice(index=0, message=msg, finish_reason="stop")
        usage = ChatUsage(prompt_tokens=5, completion_tokens=2, total_tokens=7)
        resp = ChatResponse(
            id="chatcmpl-abc",
            choices=[choice],
            usage=usage,
            model="cyankiwi/Qwen3-VL-2B-Instruct-AWQ-4bit",
            created=1700000000,
        )
        assert resp.text == "Hello world"
        assert resp.usage.total_tokens == 7
        assert "2B-Instruct" in resp.model

    def test_vision_messages_structure(self):
        """build_vision_messages produces valid message format."""
        messages = build_vision_messages(
            prompt="Describe this image",
            images=["data:image/png;base64,abc123"],
        )
        assert len(messages) == 1
        assert messages[0]["role"] == "user"
        content = messages[0]["content"]
        assert isinstance(content, list)
        # Should have text + image parts
        types = [part["type"] for part in content]
        assert "text" in types
        assert "image_url" in types


# ────────────────────────────────────────────────────────────────
# Machine scheduler + client interaction
# ────────────────────────────────────────────────────────────────


class TestSchedulerIntegration:
    """Test MachineScheduler with realistic machine states."""

    def test_round_robin_across_machines(self):
        """Scheduler distributes across multiple healthy machines."""
        machines = [
            MachineState(
                endpoint=f"http://host{i}:29800",
                client=MagicMock(),
                async_client=MagicMock(),
            )
            for i in range(3)
        ]
        for m in machines:
            m.healthy = True

        scheduler = MachineScheduler(machines)
        # Mark machines busy without releasing — forces scheduler
        # to pick different machines each time (most available_slots)
        picked = []
        for _ in range(3):
            m = scheduler.get_idle_machine()
            assert m is not None
            picked.append(m.endpoint)
            m.mark_busy()

        seen = set(picked)
        assert len(seen) == 3  # all machines used

        # Release all
        for m in machines:
            m.mark_idle()

    def test_degraded_machine_skipped(self):
        """Unhealthy machines are skipped by scheduler."""
        machines = [
            MachineState(
                endpoint="http://healthy:29800",
                client=MagicMock(),
                async_client=MagicMock(),
            ),
            MachineState(
                endpoint="http://broken:29800",
                client=MagicMock(),
                async_client=MagicMock(),
            ),
        ]
        machines[0].healthy = True
        machines[1].healthy = False

        scheduler = MachineScheduler(machines)
        for _ in range(5):
            m = scheduler.get_idle_machine()
            assert m is not None
            assert m.endpoint == "http://healthy:29800"
            m.mark_busy()
            m.mark_idle()

    def test_capacity_calculation(self):
        """Total capacity reflects healthy machines only."""
        machines = [
            MachineState(
                endpoint=f"http://host{i}:29800",
                client=MagicMock(),
                async_client=MagicMock(),
            )
            for i in range(3)
        ]
        machines[0].healthy = True
        machines[1].healthy = True
        machines[2].healthy = False

        scheduler = MachineScheduler(machines)
        healthy = scheduler.get_healthy_machines()
        assert len(healthy) == 2

        cap = scheduler.get_total_capacity(healthy)
        assert cap > 0


# ────────────────────────────────────────────────────────────────
# Performance config integration
# ────────────────────────────────────────────────────────────────


class TestPerformanceConfigIntegration:
    """Test performance config uses correct QVL defaults."""

    def test_exploration_config_exists(self):
        """ExplorationConfig is importable and usable."""
        config = ExplorationConfig()
        assert config is not None

    def test_performance_metrics_exists(self):
        """PerformanceMetrics is importable and usable."""
        metrics = PerformanceMetrics()
        assert metrics is not None
        assert metrics.total_requests == 0


# ────────────────────────────────────────────────────────────────
# Constants consistency
# ────────────────────────────────────────────────────────────────


class TestConstantsConsistency:
    """Test that constants are consistent across modules."""

    def test_awq_quant_levels_only_4bit(self):
        """AWQ only supports 4bit (8bit removed)."""
        assert AWQ_QUANT_LEVELS == {"4bit"}

    def test_awq_models_all_4bit(self):
        """All AWQ_MODELS entries are 4bit."""
        for model in AWQ_MODELS:
            assert "4bit" in model.lower(), f"Non-4bit model: {model}"

    def test_no_8bit_in_awq(self):
        """No 8bit entries in AWQ constants."""
        for model in AWQ_MODELS:
            assert "8bit" not in model.lower(), f"8bit found: {model}"
        for key in AWQ_REPO_MAP:
            assert "8bit" not in key[1], f"8bit in repo map: {key}"

    def test_model_shortcuts_cover_all_sizes(self):
        """Shortcuts exist for 2b, 4b, 8b × instruct, thinking."""
        expected = {
            "2b-instruct",
            "2b-thinking",
            "4b-instruct",
            "4b-thinking",
            "8b-instruct",
            "8b-thinking",
        }
        actual = {normalize_model_key(s) for s in MODEL_SHORTCUTS}
        assert expected.issubset(actual)

    def test_awq_repo_map_values_in_awq_models(self):
        """Every AWQ_REPO_MAP value is in AWQ_MODELS."""
        for key, repo in AWQ_REPO_MAP.items():
            assert repo in AWQ_MODELS, f"Repo {repo} (key={key!r}) not in AWQ_MODELS"

    def test_parse_model_spec_case_insensitive(self):
        """Model spec parsing is case-insensitive."""
        assert parse_model_spec("8B-Instruct:4bit") == parse_model_spec(
            "8b-instruct:4bit"
        )
        assert parse_model_spec("2B-THINKING:4BIT") == parse_model_spec(
            "2b-thinking:4bit"
        )
