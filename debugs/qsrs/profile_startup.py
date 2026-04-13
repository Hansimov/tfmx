#!/usr/bin/env python3
"""Parse QSR backend docker logs into internal cold-start phases."""

import argparse
import json
import re
import subprocess

from dataclasses import dataclass, field
from datetime import datetime
from typing import Iterable, Optional


PHASE_PATTERNS = [
    ("model_load_start", re.compile(r"Starting to load model ", re.IGNORECASE)),
    ("weights_loaded", re.compile(r"Loading weights took ", re.IGNORECASE)),
    ("model_loaded", re.compile(r"Model loading took ", re.IGNORECASE)),
    (
        "encoder_cache_profile_start",
        re.compile(r"Encoder cache will be initialized", re.IGNORECASE),
    ),
    (
        "compile_start",
        re.compile(r"Using cache directory: .*torch_compile", re.IGNORECASE),
    ),
    (
        "compile_transform_done",
        re.compile(r"Dynamo bytecode transform time:", re.IGNORECASE),
    ),
    ("compile_done", re.compile(r"torch\.compile took ", re.IGNORECASE)),
    (
        "initial_profile_done",
        re.compile(r"Initial profiling/warmup run took ", re.IGNORECASE),
    ),
    ("kv_cache_ready", re.compile(r"GPU KV cache size:", re.IGNORECASE)),
    (
        "graph_capture_done",
        re.compile(r"Graph capturing finished in ", re.IGNORECASE),
    ),
    (
        "engine_init_done",
        re.compile(
            r"init engine \(profile, create kv cache, warmup model\) took ",
            re.IGNORECASE,
        ),
    ),
    (
        "routes_ready",
        re.compile(r"Available routes are:", re.IGNORECASE),
    ),
    (
        "server_start",
        re.compile(r"Starting vLLM server on http://", re.IGNORECASE),
    ),
    (
        "app_startup_complete",
        re.compile(r"Application startup complete\.", re.IGNORECASE),
    ),
    (
        "first_health_200",
        re.compile(r'"GET /health HTTP/1\.1" 200 OK', re.IGNORECASE),
    ),
    (
        "first_warmup_200",
        re.compile(r'"POST /v1/audio/transcriptions HTTP/1\.1" 200 OK', re.IGNORECASE),
    ),
]

WARNING_PATTERNS = {
    "mm_cache_assertions": re.compile(
        r"Expected a cached item for mm_hash=", re.IGNORECASE
    ),
    "tokenizer_regex_warning": re.compile(r"incorrect regex pattern", re.IGNORECASE),
    "development_endpoints_warning": re.compile(
        r"Development endpoints are enabled", re.IGNORECASE
    ),
}

TIMESTAMP_RE = re.compile(r"^(?P<ts>\d{4}-\d{2}-\d{2}T[^ ]+Z)\s+(?P<msg>.*)$")


def parse_timestamp(value: str) -> datetime:
    return datetime.fromisoformat(value.replace("Z", "+00:00"))


def format_seconds(value: float | None) -> str:
    if value is None:
        return "n/a"
    return f"{value:.2f}s"


@dataclass
class ContainerStartupSummary:
    container: str
    first_timestamp: str = ""
    last_timestamp: str = ""
    phases: dict[str, str] = field(default_factory=dict)
    phase_elapsed_sec: dict[str, float] = field(default_factory=dict)
    inter_phase_elapsed_sec: dict[str, float] = field(default_factory=dict)
    warning_counts: dict[str, int] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "container": self.container,
            "first_timestamp": self.first_timestamp,
            "last_timestamp": self.last_timestamp,
            "phases": self.phases,
            "phase_elapsed_sec": {
                key: round(value, 3) for key, value in self.phase_elapsed_sec.items()
            },
            "inter_phase_elapsed_sec": {
                key: round(value, 3)
                for key, value in self.inter_phase_elapsed_sec.items()
            },
            "warning_counts": self.warning_counts,
        }


def parse_log_lines(container: str, lines: Iterable[str]) -> ContainerStartupSummary:
    summary = ContainerStartupSummary(container=container)
    first_dt: Optional[datetime] = None
    phase_dt: dict[str, datetime] = {}
    warning_counts = {name: 0 for name in WARNING_PATTERNS}

    for raw_line in lines:
        match = TIMESTAMP_RE.match(raw_line.rstrip())
        if not match:
            continue
        timestamp_text = match.group("ts")
        message = match.group("msg")
        dt = parse_timestamp(timestamp_text)

        if first_dt is None:
            first_dt = dt
            summary.first_timestamp = timestamp_text
        summary.last_timestamp = timestamp_text

        for warning_name, pattern in WARNING_PATTERNS.items():
            if pattern.search(message):
                warning_counts[warning_name] += 1

        for phase_name, pattern in PHASE_PATTERNS:
            if phase_name in phase_dt:
                continue
            if pattern.search(message):
                phase_dt[phase_name] = dt
                summary.phases[phase_name] = timestamp_text

    summary.warning_counts = warning_counts

    if first_dt is None:
        return summary

    for phase_name, dt in phase_dt.items():
        summary.phase_elapsed_sec[phase_name] = (dt - first_dt).total_seconds()

    phase_sequence = [phase_name for phase_name, _pattern in PHASE_PATTERNS]
    previous_name = None
    previous_dt = None
    for phase_name in phase_sequence:
        dt = phase_dt.get(phase_name)
        if dt is None:
            continue
        if previous_name is not None and previous_dt is not None:
            summary.inter_phase_elapsed_sec[f"{previous_name}->{phase_name}"] = (
                dt - previous_dt
            ).total_seconds()
        previous_name = phase_name
        previous_dt = dt

    return summary


def discover_qsr_containers(project_prefix: str | None = None) -> list[str]:
    result = subprocess.run(
        "docker ps -a --format '{{.Names}}'",
        shell=True,
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        return []

    names = [line.strip() for line in result.stdout.splitlines() if line.strip()]
    filtered = [
        name for name in names if name.startswith("qsr-") or name.startswith("qsr_")
    ]
    if project_prefix:
        filtered = [name for name in filtered if name.startswith(project_prefix)]
    return sorted(filtered)


def load_container_logs(container: str) -> list[str]:
    result = subprocess.run(
        ["docker", "logs", "--timestamps", container],
        capture_output=True,
        text=True,
        check=False,
    )
    combined = ""
    if result.stdout:
        combined += result.stdout
    if result.stderr:
        combined += result.stderr
    return combined.splitlines()


def print_summary(summary: ContainerStartupSummary) -> None:
    print(f"=== {summary.container} ===")
    if not summary.first_timestamp:
        print("no timestamped log lines found")
        return
    print(f"first_timestamp: {summary.first_timestamp}")
    print(f"last_timestamp:  {summary.last_timestamp}")
    if summary.phase_elapsed_sec:
        print("phase_elapsed_sec:")
        for phase_name in summary.phase_elapsed_sec:
            print(
                f"  {phase_name}: {format_seconds(summary.phase_elapsed_sec[phase_name])}"
            )
    if summary.inter_phase_elapsed_sec:
        print("inter_phase_elapsed_sec:")
        for phase_name, elapsed in summary.inter_phase_elapsed_sec.items():
            print(f"  {phase_name}: {format_seconds(elapsed)}")
    print("warning_counts:")
    for warning_name, count in summary.warning_counts.items():
        print(f"  {warning_name}: {count}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Profile QSR backend startup logs")
    parser.add_argument(
        "containers",
        nargs="*",
        help="Container names to inspect; defaults to discovered QSR containers",
    )
    parser.add_argument(
        "--project-prefix",
        default=None,
        help="Optional prefix used when auto-discovering QSR containers",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Emit machine-readable JSON instead of human-readable text",
    )
    parser.add_argument(
        "-o",
        "--output",
        default=None,
        help="Optional path to write the JSON summary",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    containers = args.containers or discover_qsr_containers(args.project_prefix)
    if not containers:
        raise SystemExit("No QSR containers found")

    summaries = [
        parse_log_lines(container, load_container_logs(container))
        for container in containers
    ]
    payload = [summary.to_dict() for summary in summaries]

    if args.output:
        with open(args.output, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, ensure_ascii=False)
            handle.write("\n")

    if args.json:
        print(json.dumps(payload, indent=2, ensure_ascii=False))
        return

    for summary in summaries:
        print_summary(summary)


if __name__ == "__main__":
    main()
