"""Tests for tfmx.qvls.router module"""

import pytest

from tfmx.qvls.router import (
    QVLRouter,
    InstanceDescriptor,
    parse_model_spec,
)
from tfmx.qvls.compose import MODEL_SHORTCUTS, MODEL_SHORTCUT_REV


class TestInstanceDescriptor:
    """Test InstanceDescriptor dataclass."""

    def test_creation(self):
        desc = InstanceDescriptor(
            model_name="Qwen/Qwen3-VL-8B-Instruct",
            quant_method="awq",
            quant_level="4bit",
            endpoint="http://localhost:29880",
            gpu_id=0,
            instance_id="qvl--gpu0",
        )
        assert desc.model_name == "Qwen/Qwen3-VL-8B-Instruct"
        assert desc.quant_method == "awq"
        assert desc.quant_level == "4bit"
        assert desc.endpoint == "http://localhost:29880"
        assert desc.gpu_id == 0

    def test_model_shortcut(self):
        desc = InstanceDescriptor(model_name="Qwen/Qwen3-VL-8B-Instruct")
        assert desc.model_shortcut == "8b-instruct"

    def test_model_shortcut_unknown(self):
        desc = InstanceDescriptor(model_name="unknown/model")
        assert desc.model_shortcut == "model"

    def test_model_shortcut_empty(self):
        desc = InstanceDescriptor(model_name="")
        assert desc.model_shortcut == ""

    def test_label_with_quant(self):
        desc = InstanceDescriptor(
            model_name="Qwen/Qwen3-VL-8B-Instruct",
            quant_level="4bit",
        )
        assert desc.label == "8b-instruct:4bit"

    def test_label_without_quant(self):
        desc = InstanceDescriptor(model_name="Qwen/Qwen3-VL-8B-Instruct")
        assert desc.label == "8b-instruct"

    def test_matches_any(self):
        desc = InstanceDescriptor(model_name="Qwen/Qwen3-VL-8B-Instruct")
        assert desc.matches() is True

    def test_matches_model_full_name(self):
        desc = InstanceDescriptor(model_name="Qwen/Qwen3-VL-8B-Instruct")
        assert desc.matches(model="Qwen/Qwen3-VL-8B-Instruct") is True
        assert desc.matches(model="Qwen/Qwen3-VL-4B-Instruct") is False

    def test_matches_model_shortcut(self):
        desc = InstanceDescriptor(model_name="Qwen/Qwen3-VL-8B-Instruct")
        assert desc.matches(model="8B-Instruct") is True
        assert desc.matches(model="4B-Instruct") is False

    def test_matches_model_case_insensitive(self):
        desc = InstanceDescriptor(model_name="Qwen/Qwen3-VL-8B-Instruct")
        assert desc.matches(model="8b-instruct") is True
        assert desc.matches(model="qwen/qwen3-vl-8b-instruct") is True

    def test_matches_quant(self):
        desc = InstanceDescriptor(
            model_name="Qwen/Qwen3-VL-8B-Instruct",
            quant_level="4bit",
        )
        assert desc.matches(quant="4bit") is True
        assert desc.matches(quant="8bit") is False

    def test_matches_quant_case_insensitive(self):
        desc = InstanceDescriptor(
            model_name="Qwen/Qwen3-VL-8B-Instruct",
            quant_level="4bit",
        )
        assert desc.matches(quant="4BIT") is True

    def test_matches_model_and_quant(self):
        desc = InstanceDescriptor(
            model_name="Qwen/Qwen3-VL-8B-Instruct",
            quant_level="4bit",
        )
        assert desc.matches(model="8B-Instruct", quant="4bit") is True
        assert desc.matches(model="8B-Instruct", quant="8bit") is False
        assert desc.matches(model="4B-Instruct", quant="4bit") is False

    def test_to_dict(self):
        desc = InstanceDescriptor(
            model_name="Qwen/Qwen3-VL-8B-Instruct",
            quant_method="awq",
            quant_level="4bit",
            endpoint="http://localhost:29880",
            gpu_id=0,
            instance_id="qvl--gpu0",
        )
        d = desc.to_dict()
        assert d["model_name"] == "Qwen/Qwen3-VL-8B-Instruct"
        assert d["model_shortcut"] == "8b-instruct"
        assert d["quant_method"] == "awq"
        assert d["quant_level"] == "4bit"
        assert d["endpoint"] == "http://localhost:29880"
        assert d["gpu_id"] == 0
        assert d["instance_id"] == "qvl--gpu0"
        assert d["label"] == "8b-instruct:4bit"
        assert "healthy" in d

    def test_healthy_default(self):
        desc = InstanceDescriptor()
        assert desc.healthy is True


class TestParseModelSpec:
    """Test parse_model_spec function."""

    def test_empty_string(self):
        model, quant = parse_model_spec("")
        assert model == ""
        assert quant == ""

    def test_model_only(self):
        model, quant = parse_model_spec("8B-Instruct")
        assert model == "8b-instruct"
        assert quant == ""

    def test_full_model_name(self):
        model, quant = parse_model_spec("Qwen/Qwen3-VL-8B-Instruct")
        assert model == "qwen/qwen3-vl-8b-instruct"
        assert quant == ""

    def test_model_with_quant(self):
        model, quant = parse_model_spec("8B-Instruct:4bit")
        assert model == "8b-instruct"
        assert quant == "4bit"

    def test_full_name_with_quant(self):
        model, quant = parse_model_spec("Qwen/Qwen3-VL-8B-Instruct:8bit")
        assert model == "qwen/qwen3-vl-8b-instruct"
        assert quant == "8bit"

    def test_quant_levels(self):
        for level in ["4bit", "8bit", "4BIT", "8BIT"]:
            model, quant = parse_model_spec(f"8B-Instruct:{level}")
            assert quant == level.lower()

    def test_non_awq_quant_not_parsed(self):
        model, quant = parse_model_spec("model:custom_quant")
        assert model == "model:custom_quant"
        assert quant == ""


class TestQVLRouter:
    """Test QVLRouter class."""

    def _make_descriptors(self) -> list[InstanceDescriptor]:
        """Create a test set of instance descriptors."""
        return [
            InstanceDescriptor(
                model_name="Qwen/Qwen3-VL-2B-Instruct",
                quant_method="awq",
                quant_level="4bit",
                endpoint="http://localhost:29880",
                gpu_id=0,
                instance_id="qvl--gpu0",
            ),
            InstanceDescriptor(
                model_name="Qwen/Qwen3-VL-4B-Instruct",
                quant_method="awq",
                quant_level="4bit",
                endpoint="http://localhost:29881",
                gpu_id=1,
                instance_id="qvl--gpu1",
            ),
            InstanceDescriptor(
                model_name="Qwen/Qwen3-VL-8B-Instruct",
                quant_method="awq",
                quant_level="4bit",
                endpoint="http://localhost:29882",
                gpu_id=2,
                instance_id="qvl--gpu2",
            ),
            InstanceDescriptor(
                model_name="Qwen/Qwen3-VL-8B-Instruct",
                quant_method="awq",
                quant_level="8bit",
                endpoint="http://localhost:29883",
                gpu_id=3,
                instance_id="qvl--gpu3",
            ),
        ]

    def test_empty_router(self):
        router = QVLRouter()
        assert len(router) == 0
        assert router.default_instance is None
        assert router.route() is None

    def test_register(self):
        router = QVLRouter()
        desc = InstanceDescriptor(
            model_name="Qwen/Qwen3-VL-8B-Instruct",
            endpoint="http://localhost:29880",
        )
        router.register(desc)
        assert len(router) == 1
        assert router.default_instance == desc

    def test_register_multiple(self):
        router = QVLRouter()
        descs = self._make_descriptors()
        for d in descs:
            router.register(d)
        assert len(router) == 4
        assert router.default_instance == descs[0]

    def test_find_instances_all(self):
        router = QVLRouter()
        for d in self._make_descriptors():
            router.register(d)
        found = router.find_instances()
        assert len(found) == 4

    def test_find_instances_by_model(self):
        router = QVLRouter()
        for d in self._make_descriptors():
            router.register(d)
        found = router.find_instances(model="8B-Instruct")
        assert len(found) == 2
        for f in found:
            assert f.model_name == "Qwen/Qwen3-VL-8B-Instruct"

    def test_find_instances_by_quant(self):
        router = QVLRouter()
        for d in self._make_descriptors():
            router.register(d)
        found = router.find_instances(quant="8bit")
        assert len(found) == 1
        assert found[0].quant_level == "8bit"

    def test_find_instances_by_model_and_quant(self):
        router = QVLRouter()
        for d in self._make_descriptors():
            router.register(d)
        found = router.find_instances(model="8B-Instruct", quant="4bit")
        assert len(found) == 1
        assert found[0].gpu_id == 2

    def test_route_default(self):
        router = QVLRouter()
        descs = self._make_descriptors()
        for d in descs:
            router.register(d)
        result = router.route()
        assert result == descs[0]

    def test_route_by_model(self):
        router = QVLRouter()
        for d in self._make_descriptors():
            router.register(d)
        result = router.route(model="4B-Instruct")
        assert result.model_name == "Qwen/Qwen3-VL-4B-Instruct"

    def test_route_by_model_and_quant(self):
        router = QVLRouter()
        for d in self._make_descriptors():
            router.register(d)
        result = router.route(model="8B-Instruct", quant="8bit")
        assert result.gpu_id == 3

    def test_route_no_match_falls_back(self):
        router = QVLRouter()
        descs = self._make_descriptors()
        for d in descs:
            router.register(d)
        result = router.route(model="nonexistent")
        assert result is not None

    def test_route_unhealthy_skipped(self):
        router = QVLRouter()
        desc = InstanceDescriptor(
            model_name="Qwen/Qwen3-VL-8B-Instruct",
            endpoint="http://localhost:29880",
            healthy=False,
        )
        router.register(desc)
        result = router.route(model="8B-Instruct")
        assert result is None

    def test_route_from_model_field(self):
        router = QVLRouter()
        for d in self._make_descriptors():
            router.register(d)
        result = router.route_from_model_field("8B-Instruct:8bit")
        assert result.gpu_id == 3

    def test_route_from_model_field_no_quant(self):
        router = QVLRouter()
        for d in self._make_descriptors():
            router.register(d)
        result = router.route_from_model_field("2B-Instruct")
        assert result.model_name == "Qwen/Qwen3-VL-2B-Instruct"

    def test_route_from_model_field_empty(self):
        router = QVLRouter()
        descs = self._make_descriptors()
        for d in descs:
            router.register(d)
        result = router.route_from_model_field("")
        assert result == descs[0]

    def test_set_default(self):
        router = QVLRouter()
        descs = self._make_descriptors()
        for d in descs:
            router.register(d)
        assert router.default_instance == descs[0]
        success = router.set_default(model="8B-Instruct", quant="8bit")
        assert success is True
        assert router.default_instance.gpu_id == 3

    def test_set_default_no_match(self):
        router = QVLRouter()
        for d in self._make_descriptors():
            router.register(d)
        success = router.set_default(model="nonexistent")
        assert success is False

    def test_healthy_instances(self):
        router = QVLRouter()
        descs = self._make_descriptors()
        descs[1].healthy = False
        for d in descs:
            router.register(d)
        assert len(router.healthy_instances) == 3

    def test_get_all_info(self):
        router = QVLRouter()
        for d in self._make_descriptors():
            router.register(d)
        info = router.get_all_info()
        assert len(info) == 4
        assert all(isinstance(i, dict) for i in info)
        assert "model_name" in info[0]
        assert "label" in info[0]

    def test_get_available_models(self):
        router = QVLRouter()
        for d in self._make_descriptors():
            router.register(d)
        models = router.get_available_models()
        assert "2b-instruct:4bit" in models
        assert "8b-instruct:8bit" in models
        assert len(models) == 4  # 2B/4B/8B 4bit + 8B 8bit

    def test_repr(self):
        router = QVLRouter()
        for d in self._make_descriptors():
            router.register(d)
        r = repr(router)
        assert "QVLRouter" in r
        assert "instances=4" in r


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
