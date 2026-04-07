"""Model-aware routing helpers for QWN machine."""

from dataclasses import dataclass

from .compose import AWQ_QUANT_LEVELS
from .compose import get_model_shortcut, normalize_model_key, resolve_model_name


@dataclass
class InstanceDescriptor:
    model_name: str = ""
    quant_method: str = ""
    quant_level: str = ""
    endpoint: str = ""
    gpu_id: int | None = None
    instance_id: str = ""
    healthy: bool = True

    @property
    def model_shortcut(self) -> str:
        return get_model_shortcut(self.model_name)

    @property
    def label(self) -> str:
        shortcut = self.model_shortcut
        if self.quant_level:
            return f"{shortcut}:{self.quant_level}"
        return shortcut

    def matches(self, model: str = "", quant: str = "") -> bool:
        if model:
            requested = normalize_model_key(model)
            requested_shortcut = normalize_model_key(
                get_model_shortcut(resolve_model_name(model))
            )
            requested_model_name = normalize_model_key(resolve_model_name(model))
            model_name = normalize_model_key(self.model_name)
            model_shortcut = normalize_model_key(self.model_shortcut)
            if not {
                requested,
                requested_shortcut,
                requested_model_name,
            }.intersection({model_name, model_shortcut}):
                return False

        if quant and normalize_model_key(quant) != normalize_model_key(
            self.quant_level
        ):
            return False

        return True

    def to_dict(self) -> dict:
        return {
            "model_name": self.model_name,
            "model_shortcut": self.model_shortcut,
            "quant_method": self.quant_method,
            "quant_level": self.quant_level,
            "endpoint": self.endpoint,
            "gpu_id": self.gpu_id,
            "instance_id": self.instance_id,
            "healthy": self.healthy,
            "label": self.label,
        }


def parse_model_spec(model_field: str) -> tuple[str, str]:
    if not model_field:
        return ("", "")

    normalized = normalize_model_key(model_field)
    if ":" in normalized:
        model_name, maybe_quant = normalized.rsplit(":", 1)
        if maybe_quant in AWQ_QUANT_LEVELS:
            return (model_name, maybe_quant)
    return (normalized, "")


class QWNRouter:
    def __init__(self):
        self._instances: list[InstanceDescriptor] = []
        self._default_instance: InstanceDescriptor | None = None

    def register(self, descriptor: InstanceDescriptor) -> None:
        self._instances.append(descriptor)
        if self._default_instance is None:
            self._default_instance = descriptor

    @property
    def instances(self) -> list[InstanceDescriptor]:
        return list(self._instances)

    @property
    def healthy_instances(self) -> list[InstanceDescriptor]:
        return [instance for instance in self._instances if instance.healthy]

    def find_instances(
        self, model: str = "", quant: str = ""
    ) -> list[InstanceDescriptor]:
        if not model and not quant:
            return list(self._instances)
        return [
            instance for instance in self._instances if instance.matches(model, quant)
        ]

    def route(self, model: str = "", quant: str = "") -> InstanceDescriptor | None:
        if not model and not quant:
            if self._default_instance and self._default_instance.healthy:
                return self._default_instance
            healthy = self.healthy_instances
            return healthy[0] if healthy else None

        matches = [
            instance
            for instance in self._instances
            if instance.healthy and instance.matches(model, quant)
        ]
        if matches:
            return matches[0]

        if quant and model:
            matches = [
                instance
                for instance in self._instances
                if instance.healthy and instance.matches(model, "")
            ]
            if matches:
                return matches[0]

        if self._default_instance and self._default_instance.healthy:
            return self._default_instance

        healthy = self.healthy_instances
        return healthy[0] if healthy else None

    def route_from_model_field(self, model_field: str) -> InstanceDescriptor | None:
        model, quant = parse_model_spec(model_field)
        return self.route(model=model, quant=quant)

    def get_available_models(self) -> list[str]:
        labels = {instance.label for instance in self._instances if instance.healthy}
        return sorted(label for label in labels if label)

    def get_all_info(self) -> list[dict]:
        return [instance.to_dict() for instance in self._instances]

    def __len__(self) -> int:
        return len(self._instances)

    def __repr__(self) -> str:
        healthy_count = sum(1 for instance in self._instances if instance.healthy)
        return f"QWNRouter(instances={len(self._instances)}, healthy={healthy_count})"
