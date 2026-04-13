#!/usr/bin/env python3
"""Run a mixed chat/transcribe soak test against a QSR machine endpoint."""

import argparse
import json
import random
import subprocess
import threading
import time

from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from tfmx.qsrs.client import InfoResponse, QSRClient, build_audio_messages


MM_CACHE_ERROR_PATTERN = "Expected a cached item for mm_hash="


@dataclass
class WorkItem:
    mode: str
    audio: str
    prompt: str
    max_tokens: int


def percentile(values: list[float], ratio: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = min(len(ordered) - 1, int(len(ordered) * ratio))
    return ordered[index]


def client_info(endpoint: str) -> InfoResponse:
    with QSRClient(endpoint=endpoint, verbose=False) as client:
        return client.info()


def machine_requests_snapshot(info: InfoResponse) -> dict[str, int]:
    return dict(info.stats.requests_per_instance)


def subtract_counts(after: dict[str, int], before: dict[str, int]) -> dict[str, int]:
    keys = sorted(set(before) | set(after))
    deltas: dict[str, int] = {}
    for key in keys:
        deltas[key] = after.get(key, 0) - before.get(key, 0)
    return deltas


def count_backend_mm_cache_assertions(project_prefix: str) -> dict[str, int]:
    result = subprocess.run(
        "docker ps -a --format '{{.Names}}'",
        shell=True,
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        return {}

    counts: dict[str, int] = {}
    for container in [
        line.strip() for line in result.stdout.splitlines() if line.strip()
    ]:
        if not container.startswith(project_prefix):
            continue
        log_result = subprocess.run(
            ["docker", "logs", container],
            capture_output=True,
            text=True,
            check=False,
        )
        combined = (log_result.stdout or "") + (log_result.stderr or "")
        counts[container] = combined.count(MM_CACHE_ERROR_PATTERN)
    return counts


def make_workload(
    audio_inputs: list[str],
    *,
    num_transcribe: int,
    num_chat: int,
    prompt: str,
    max_tokens: int,
    seed: int,
) -> list[WorkItem]:
    items: list[WorkItem] = []
    if not audio_inputs:
        raise ValueError("At least one audio input is required")

    for index in range(num_transcribe):
        items.append(
            WorkItem(
                mode="transcribe",
                audio=audio_inputs[index % len(audio_inputs)],
                prompt=prompt,
                max_tokens=max_tokens,
            )
        )
    for index in range(num_chat):
        items.append(
            WorkItem(
                mode="chat",
                audio=audio_inputs[index % len(audio_inputs)],
                prompt=prompt,
                max_tokens=max_tokens,
            )
        )

    rng = random.Random(seed)
    rng.shuffle(items)
    return items


def make_thread_client(endpoint: str):
    local = threading.local()

    def get_client() -> QSRClient:
        client = getattr(local, "client", None)
        if client is None:
            client = QSRClient(endpoint=endpoint, verbose=False)
            local.client = client
        return client

    return get_client


def run_workload(
    endpoint: str,
    workload: list[WorkItem],
    *,
    max_workers: int,
) -> dict:
    get_client = make_thread_client(endpoint)
    latencies = {"chat": [], "transcribe": []}
    successes = {"chat": 0, "transcribe": 0}
    failures = {"chat": 0, "transcribe": 0}
    output_chars = {"chat": 0, "transcribe": 0}
    errors: list[dict[str, str]] = []

    def execute(item: WorkItem) -> tuple[str, float, bool, int, str]:
        client = get_client()
        started_at = time.perf_counter()
        try:
            if item.mode == "transcribe":
                response = client.transcribe(audio=item.audio, response_format="json")
                text = response.text or ""
            else:
                response = client.chat(
                    messages=build_audio_messages(
                        texts=[item.prompt],
                        audios=[item.audio],
                    ),
                    max_tokens=item.max_tokens,
                    temperature=0.0,
                    top_p=1.0,
                )
                text = response.text or ""
            return item.mode, time.perf_counter() - started_at, True, len(text), ""
        except Exception as exc:
            return item.mode, time.perf_counter() - started_at, False, 0, str(exc)

    started_at = time.perf_counter()
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(execute, item) for item in workload]
        for future in as_completed(futures):
            mode, latency_sec, succeeded, text_len, error_text = future.result()
            latencies[mode].append(latency_sec)
            if succeeded:
                successes[mode] += 1
                output_chars[mode] += text_len
            else:
                failures[mode] += 1
                if len(errors) < 20:
                    errors.append({"mode": mode, "error": error_text})

    total_time = time.perf_counter() - started_at
    total_successes = sum(successes.values())
    total_failures = sum(failures.values())
    return {
        "total_time_sec": round(total_time, 3),
        "submitted": len(workload),
        "successful": total_successes,
        "failed": total_failures,
        "success_rate": round(total_successes / len(workload), 4) if workload else 0.0,
        "requests_per_second": (
            round(total_successes / total_time, 2) if total_time > 0 else 0.0
        ),
        "mode_breakdown": {
            mode: {
                "submitted": successes[mode] + failures[mode],
                "successful": successes[mode],
                "failed": failures[mode],
                "output_chars": output_chars[mode],
                "latency_avg_sec": (
                    round(sum(latencies[mode]) / len(latencies[mode]), 4)
                    if latencies[mode]
                    else 0.0
                ),
                "latency_p50_sec": round(percentile(latencies[mode], 0.50), 4),
                "latency_p95_sec": round(percentile(latencies[mode], 0.95), 4),
                "latency_p99_sec": round(percentile(latencies[mode], 0.99), 4),
            }
            for mode in ("transcribe", "chat")
        },
        "sample_errors": errors,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a mixed QSR soak test")
    parser.add_argument(
        "-E",
        "--endpoint",
        default="http://127.0.0.1:27900",
        help="QSR machine endpoint",
    )
    parser.add_argument(
        "--audio",
        action="append",
        default=[],
        help="Audio file path or URL; repeat to provide multiple samples",
    )
    parser.add_argument("--num-transcribe", type=int, default=1200)
    parser.add_argument("--num-chat", type=int, default=400)
    parser.add_argument("--max-workers", type=int, default=32)
    parser.add_argument("--max-tokens", type=int, default=128)
    parser.add_argument(
        "--prompt",
        default="请先转写音频，再用一句话概括语种和主要内容。",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--project-prefix",
        default="qsr-uniform--gpu",
        help="Prefix used when counting backend multimodal cache assertions",
    )
    parser.add_argument("-o", "--output", default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.audio:
        raise SystemExit("At least one --audio input is required")

    before_info = client_info(args.endpoint)
    before_requests = machine_requests_snapshot(before_info)
    before_mm_cache = count_backend_mm_cache_assertions(args.project_prefix)

    workload = make_workload(
        args.audio,
        num_transcribe=args.num_transcribe,
        num_chat=args.num_chat,
        prompt=args.prompt,
        max_tokens=args.max_tokens,
        seed=args.seed,
    )
    result = run_workload(
        args.endpoint,
        workload,
        max_workers=args.max_workers,
    )

    after_info = client_info(args.endpoint)
    after_requests = machine_requests_snapshot(after_info)
    after_mm_cache = count_backend_mm_cache_assertions(args.project_prefix)

    payload = {
        "config": {
            "endpoint": args.endpoint,
            "audio": args.audio,
            "num_transcribe": args.num_transcribe,
            "num_chat": args.num_chat,
            "max_workers": args.max_workers,
            "max_tokens": args.max_tokens,
            "seed": args.seed,
        },
        "result": result,
        "machine_delta": {
            "total_requests": after_info.stats.total_requests
            - before_info.stats.total_requests,
            "total_errors": after_info.stats.total_errors
            - before_info.stats.total_errors,
            "total_failovers": after_info.stats.total_failovers
            - before_info.stats.total_failovers,
            "total_wait_events": after_info.stats.total_wait_events
            - before_info.stats.total_wait_events,
            "requests_per_instance": subtract_counts(after_requests, before_requests),
        },
        "backend_mm_cache_assertions_delta": subtract_counts(
            after_mm_cache, before_mm_cache
        ),
    }

    if args.output:
        output_path = Path(args.output)
        output_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n")

    print(json.dumps(payload, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
