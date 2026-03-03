"""QVL Router - Model/Quant-aware request routing.

Routes chat completion requests to the appropriate vLLM instance
based on model name and quantization level. Used by QVLMachineServer
for multi-model deployments where different GPUs run different models.

Usage:
    router = QVLRouter()
    router.register(InstanceDescriptor(
        model_name="Qwen/Qwen3-VL-8B-Instruct",
        quant_method="awq", quant_level="4bit",
        endpoint="http://localhost:29880", gpu_id=0,
    ))
    match = router.route(model="8B-Instruct", quant="4bit")
"""

from dataclasses import dataclass, field
from typing import Optional

from .compose import MODEL_SHORTCUTS, MODEL_SHORTCUT_REV, AWQ_QUANT_LEVELS
from .compose import normalize_model_key, get_model_shortcut, get_display_shortcut


@dataclass
class InstanceDescriptor:
    """Describes a model/quant instance for routing."""

    model_name: str = ""  # Full HF name, e.g., "Qwen/Qwen3-VL-8B-Instruct"
    quant_method: str = ""  # "awq", "bitsandbytes", "none"
    quant_level: str = ""  # "4bit", etc.
    endpoint: str = ""  # http://host:port
    gpu_id: int | None = None
    instance_id: str = ""  # Container name or unique ID
    healthy: bool = True

    @property
    def model_shortcut(self) -> str:
        """Get lowercase model shortcut from full name."""
        return get_model_shortcut(self.model_name)

    @property
    def label(self) -> str:
        """Human-readable label (lowercase)."""
        shortcut = self.model_shortcut
        if self.quant_level:
            return f"{shortcut}:{self.quant_level}"
        return shortcut

    def matches(self, model: str = "", quant: str = "") -> bool:
        """Check if this instance matches the given model/quant filter.

        Case-insensitive matching for both model and quant.

        Args:
            model: Model filter (shortcut like "8b-instruct", full name, or empty)
            quant: Quant level filter (e.g., "4bit", or empty)

        Returns:
            True if this instance matches the filter
        """
        if model:
            model_lower = normalize_model_key(model)
            my_name_lower = self.model_name.lower()
            my_shortcut_lower = self.model_shortcut.lower()
            if model_lower != my_name_lower and model_lower != my_shortcut_lower:
                return False
        if quant:
            if normalize_model_key(quant) != normalize_model_key(self.quant_level):
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
    """Parse a model specification that may include quant info.

    Case-insensitive. All outputs are lowercase.

    Formats:
        "8b-instruct"              -> ("8b-instruct", "")
        "8B-Instruct"              -> ("8b-instruct", "")
        "8b-instruct:4bit"         -> ("8b-instruct", "4bit")
        "8B-Instruct:4bit"         -> ("8b-instruct", "4bit")
        "Qwen/Qwen3-VL-8B-Instruct:4bit" -> ("qwen/qwen3-vl-8b-instruct", "4bit")
        ""                         -> ("", "")

    Returns:
        (model, quant) tuple, both lowercase
    """
    if not model_field:
        return ("", "")

    # Check for "model:quant" format
    if ":" in model_field:
        parts = model_field.rsplit(":", 1)
        if normalize_model_key(parts[1]) in AWQ_QUANT_LEVELS:
            return (normalize_model_key(parts[0]), normalize_model_key(parts[1]))

    return (normalize_model_key(model_field), "")


class QVLRouter:
    """Routes requests to appropriate model/quant instances.

    Maintains a registry of instances and selects the best match
    based on model name and quantization level in the request.
    """

    def __init__(self):
        self._instances: list[InstanceDescriptor] = []
        self._default_instance: InstanceDescriptor | None = None

    def register(self, descriptor: InstanceDescriptor) -> None:
        """Register an instance with its model/quant info."""
        self._instances.append(descriptor)
        if self._default_instance is None:
            self._default_instance = descriptor

    def set_default(self, model: str = "", quant: str = "") -> bool:
        """Set the default instance for unspecified requests.

        Returns:
            True if a matching instance was found and set as default
        """
        matching = self.find_instances(model, quant)
        if matching:
            self._default_instance = matching[0]
            return True
        return False

    @property
    def default_instance(self) -> InstanceDescriptor | None:
        return self._default_instance

    @property
    def instances(self) -> list[InstanceDescriptor]:
        return list(self._instances)

    @property
    def healthy_instances(self) -> list[InstanceDescriptor]:
        return [inst for inst in self._instances if inst.healthy]

    def find_instances(
        self, model: str = "", quant: str = ""
    ) -> list[InstanceDescriptor]:
        """Find all instances matching the model/quant filter.

        Args:
            model: Model filter (shortcut, full name, or empty for all)
            quant: Quant level filter (e.g., "4bit", or empty for all)

        Returns:
            List of matching InstanceDescriptor objects
        """
        if not model and not quant:
            return list(self._instances)
        return [inst for inst in self._instances if inst.matches(model, quant)]

    def route(self, model: str = "", quant: str = "") -> InstanceDescriptor | None:
        """Route a request to the best matching instance.

        Strategy:
        1. If model/quant specified, find exact match
        2. If no match, try model-only match
        3. If still no match, try default instance
        4. Return None if nothing matches

        Args:
            model: Model name or shortcut from request
            quant: Quantization level from request

        Returns:
            Best matching InstanceDescriptor, or None
        """
        if not model and not quant:
            return self._default_instance

        # Try exact match (model + quant)
        matching = [
            inst
            for inst in self._instances
            if inst.healthy and inst.matches(model, quant)
        ]
        if matching:
            # Prefer the one with most available capacity (first match for now)
            return matching[0]

        # Try model-only match if quant was specified
        if quant and model:
            model_only = [
                inst
                for inst in self._instances
                if inst.healthy and inst.matches(model, "")
            ]
            if model_only:
                return model_only[0]

        # Fall back to default
        if self._default_instance and self._default_instance.healthy:
            return self._default_instance

        # Last resort: any healthy instance
        healthy = self.healthy_instances
        return healthy[0] if healthy else None

    def route_from_model_field(self, model_field: str) -> InstanceDescriptor | None:
        """Route based on a model field that may contain quant info.

        Parses "8B-Instruct:4bit" format and routes accordingly.
        """
        model, quant = parse_model_spec(model_field)
        return self.route(model=model, quant=quant)

    def get_all_info(self) -> list[dict]:
        """Get info about all registered instances."""
        return [inst.to_dict() for inst in self._instances]

    def get_available_models(self) -> list[str]:
        """Get list of unique model labels available."""
        labels = set()
        for inst in self._instances:
            if inst.healthy:
                labels.add(inst.label)
        return sorted(labels)

    def __len__(self) -> int:
        return len(self._instances)

    def __repr__(self) -> str:
        healthy = sum(1 for i in self._instances if i.healthy)
        models = len(set(i.label for i in self._instances))
        return f"QVLRouter(instances={len(self._instances)}, healthy={healthy}, models={models})"
