"""Runtime GPU telemetry helpers for QWN services."""

import subprocess

from dataclasses import dataclass
from typing import Iterable


@dataclass
class GPURuntimeStats:
    index: int
    utilization_gpu_pct: float = 0.0
    memory_used_mib: float = 0.0
    memory_total_mib: float = 0.0

    @property
    def memory_utilization_pct(self) -> float:
        if self.memory_total_mib <= 0:
            return 0.0
        return self.memory_used_mib / self.memory_total_mib * 100.0


def _parse_float(raw: str) -> float:
    value = raw.strip()
    if not value or value == "[Not Supported]":
        return 0.0
    try:
        return float(value)
    except ValueError:
        return 0.0


def query_gpu_runtime_stats(
    gpu_ids: Iterable[int] | None = None,
    timeout: float = 5.0,
) -> dict[int, GPURuntimeStats]:
    """Return runtime load snapshots keyed by GPU index.

    Some hosts return partial stdout together with a non-zero exit code when one GPU is
    unhealthy. We still parse the healthy GPU lines in that case.
    """

    requested = {int(gpu_id) for gpu_id in gpu_ids} if gpu_ids is not None else None
    result = subprocess.run(
        [
            "nvidia-smi",
            "--query-gpu=index,utilization.gpu,memory.used,memory.total",
            "--format=csv,noheader,nounits",
        ],
        capture_output=True,
        text=True,
        timeout=timeout,
    )

    snapshots: dict[int, GPURuntimeStats] = {}
    for line in result.stdout.splitlines():
        if not line.strip():
            continue
        parts = [part.strip() for part in line.split(",")]
        if len(parts) < 4:
            continue
        try:
            index = int(parts[0])
        except ValueError:
            continue
        if requested is not None and index not in requested:
            continue
        snapshots[index] = GPURuntimeStats(
            index=index,
            utilization_gpu_pct=_parse_float(parts[1]),
            memory_used_mib=_parse_float(parts[2]),
            memory_total_mib=_parse_float(parts[3]),
        )

    return snapshots
