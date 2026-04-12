"""Model-aware routing helpers for QSR machine."""

from dataclasses import dataclass

from .compose import get_model_shortcut, normalize_model_key, resolve_model_name


@dataclass
class InstanceDescriptor:
    model_name: str = ""
    endpoint: str = ""
    gpu_id: int | None = None
    instance_id: str = ""
    healthy: bool = True

    @property
    def model_shortcut(self) -> str:
        return get_model_shortcut(self.model_name)

    @property
    def label(self) -> str:
        return self.model_shortcut or self.model_name

    def matches(self, model: str = "") -> bool:
        if not model:
            return True

        requested_name = normalize_model_key(resolve_model_name(model))
        requested_shortcut = normalize_model_key(get_model_shortcut(model))
        requested_value = normalize_model_key(model)
        current_name = normalize_model_key(self.model_name)
        current_shortcut = normalize_model_key(self.model_shortcut)

        return bool(
            {requested_name, requested_shortcut, requested_value}.intersection(
                {current_name, current_shortcut}
            )
        )

    def to_dict(self) -> dict:
        return {
            "model_name": self.model_name,
            "model_shortcut": self.model_shortcut,
            "endpoint": self.endpoint,
            "gpu_id": self.gpu_id,
            "instance_id": self.instance_id,
            "healthy": self.healthy,
            "label": self.label,
        }


def parse_model_spec(model_field: str) -> str:
    return normalize_model_key(model_field)


class QSRRouter:
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

    def find_instances(self, model: str = "") -> list[InstanceDescriptor]:
        if not model:
            return list(self._instances)
        return [instance for instance in self._instances if instance.matches(model)]

    def route(self, model: str = "") -> InstanceDescriptor | None:
        if not model:
            if self._default_instance and self._default_instance.healthy:
                return self._default_instance
            healthy = self.healthy_instances
            return healthy[0] if healthy else None

        matches = [
            instance
            for instance in self._instances
            if instance.healthy and instance.matches(model)
        ]
        if matches:
            return matches[0]

        if self._default_instance and self._default_instance.healthy:
            return self._default_instance
        healthy = self.healthy_instances
        return healthy[0] if healthy else None

    def route_from_model_field(self, model_field: str) -> InstanceDescriptor | None:
        return self.route(model=parse_model_spec(model_field))

    def get_available_models(self) -> list[str]:
        labels = {instance.label for instance in self._instances if instance.healthy}
        return sorted(label for label in labels if label)

    def get_all_info(self) -> list[dict]:
        return [instance.to_dict() for instance in self._instances]

    def __len__(self) -> int:
        return len(self._instances)

    def __repr__(self) -> str:
        healthy_count = sum(1 for instance in self._instances if instance.healthy)
        return f"QSRRouter(instances={len(self._instances)}, healthy={healthy_count})"
