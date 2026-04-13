#!/usr/bin/env python3
"""Profile QSR long-audio latency and throughput with duration-aware metrics."""

import argparse
import json
import threading
import time
import sys

from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from tfmx.qsrs.client import InfoResponse, QSRClient


DEFAULT_BLUX_SRC = Path("/home/asimov/repos/blux/src")


def percentile(values: list[float], ratio: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = min(len(ordered) - 1, int(len(ordered) * ratio))
    return ordered[index]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Benchmark QSR latency and audio-throughput on long samples"
    )
    parser.add_argument(
        "-E",
        "--endpoint",
        default="http://127.0.0.1:27900",
        help="QSR machine or backend endpoint",
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=None,
        help="Manifest created by download_bili_audio.py",
    )
    parser.add_argument(
        "--audio",
        action="append",
        default=[],
        help="Local audio file path; repeat to add more samples",
    )
    parser.add_argument(
        "--repeats",
        type=int,
        default=2,
        help="How many times to replay the full audio set in the batch run",
    )
    parser.add_argument(
        "--max-workers",
        type=int,
        default=16,
        help="Concurrent transcription workers for the batch run",
    )
    parser.add_argument(
        "--timeout-sec",
        type=float,
        default=900.0,
        help="Per-request timeout to tolerate long audio inference",
    )
    parser.add_argument(
        "--blux-src",
        type=Path,
        default=DEFAULT_BLUX_SRC,
        help="Path containing the blux package source tree for local media probing",
    )
    parser.add_argument("--language", type=str, default=None)
    parser.add_argument("--prompt", type=str, default=None)
    parser.add_argument("-o", "--output", default=None)
    return parser.parse_args()


def ensure_blux_import(blux_src: Path) -> None:
    resolved = blux_src.expanduser().resolve()
    if not resolved.exists():
        raise SystemExit(f"blux source path not found: {resolved}")
    sys.path.insert(0, str(resolved))


def load_manifest_audio(manifest_path: Path) -> list[str]:
    payload = json.loads(manifest_path.read_text())
    audio_paths: list[str] = []
    for item in payload:
        qsr_input_path = item.get("qsr_input_path")
        if qsr_input_path:
            audio_paths.append(qsr_input_path)
            continue
        for audio in item.get("audio", []):
            path = audio.get("path")
            if path:
                audio_paths.append(path)
    return audio_paths


def gather_audio_inputs(args: argparse.Namespace) -> list[str]:
    inputs = list(args.audio)
    if args.manifest is not None:
        inputs.extend(load_manifest_audio(args.manifest))

    deduped: list[str] = []
    seen: set[str] = set()
    for item in inputs:
        normalized = str(Path(item).expanduser())
        if normalized in seen:
            continue
        seen.add(normalized)
        deduped.append(normalized)
    if not deduped:
        raise SystemExit("At least one audio sample is required")
    return deduped


def load_local_audio_metadata(audio_inputs: list[str]) -> dict[str, dict]:
    from blux.convert import probe_media

    metadata: dict[str, dict] = {}
    for audio in audio_inputs:
        probe = probe_media(audio)
        metadata[audio] = probe.to_dict()
    return metadata


def client_info(endpoint: str, timeout_sec: float) -> InfoResponse:
    with QSRClient(endpoint=endpoint, verbose=False, timeout_sec=timeout_sec) as client:
        return client.info()


def machine_requests_snapshot(info: InfoResponse) -> dict[str, int]:
    return dict(info.stats.requests_per_instance)


def subtract_counts(after: dict[str, int], before: dict[str, int]) -> dict[str, int]:
    keys = sorted(set(before) | set(after))
    return {key: after.get(key, 0) - before.get(key, 0) for key in keys}


def probe_audio_samples(
    endpoint: str,
    audio_inputs: list[str],
    local_audio_metadata: dict[str, dict],
    *,
    timeout_sec: float,
    language: str | None,
    prompt: str | None,
) -> tuple[list[dict], dict[str, float]]:
    rows: list[dict] = []
    durations: dict[str, float] = {}
    with QSRClient(endpoint=endpoint, verbose=False, timeout_sec=timeout_sec) as client:
        for audio in audio_inputs:
            path = Path(audio)
            local_probe = local_audio_metadata.get(audio, {})
            started_at = time.perf_counter()
            response = client.transcribe(
                audio=audio,
                language=language,
                prompt=prompt,
                response_format="json",
            )
            latency_sec = time.perf_counter() - started_at
            local_duration_sec = float(local_probe.get("duration_sec") or 0.0)
            duration_sec = float(response.duration or local_duration_sec or 0.0)
            durations[audio] = duration_sec
            rows.append(
                {
                    "audio": audio,
                    "name": path.name,
                    "size_mib": round(path.stat().st_size / (1024 * 1024), 3),
                    "local_format": local_probe.get("format_name", ""),
                    "local_codec": local_probe.get("codec_name", ""),
                    "local_duration_sec": round(local_duration_sec, 3),
                    "audio_duration_sec": round(duration_sec, 3),
                    "latency_sec": round(latency_sec, 3),
                    "real_time_factor": (
                        round(latency_sec / duration_sec, 4)
                        if duration_sec > 0
                        else None
                    ),
                    "text_chars": len(response.text or ""),
                    "language": response.language,
                }
            )
    return rows, durations


def make_thread_client(endpoint: str, timeout_sec: float):
    local = threading.local()

    def get_client() -> QSRClient:
        client = getattr(local, "client", None)
        if client is None:
            client = QSRClient(
                endpoint=endpoint, verbose=False, timeout_sec=timeout_sec
            )
            local.client = client
        return client

    return get_client


def run_batch(
    endpoint: str,
    audio_inputs: list[str],
    duration_by_audio: dict[str, float],
    *,
    repeats: int,
    max_workers: int,
    timeout_sec: float,
    language: str | None,
    prompt: str | None,
) -> dict:
    workload = [audio for _ in range(max(1, repeats)) for audio in audio_inputs]
    get_client = make_thread_client(endpoint, timeout_sec)
    latencies: list[float] = []
    total_audio_duration_sec = 0.0
    total_output_chars = 0
    failures = 0
    errors: list[dict[str, str]] = []
    per_audio: dict[str, dict[str, float | int | str | None]] = {}

    def execute(audio: str) -> tuple[str, float, bool, float, int, str]:
        client = get_client()
        started_at = time.perf_counter()
        try:
            response = client.transcribe(
                audio=audio,
                language=language,
                prompt=prompt,
                response_format="json",
            )
            latency_sec = time.perf_counter() - started_at
            duration_sec = float(
                response.duration or duration_by_audio.get(audio, 0.0) or 0.0
            )
            return audio, latency_sec, True, duration_sec, len(response.text or ""), ""
        except Exception as exc:
            return audio, time.perf_counter() - started_at, False, 0.0, 0, str(exc)

    started_at = time.perf_counter()
    with ThreadPoolExecutor(max_workers=max(1, max_workers)) as executor:
        futures = [executor.submit(execute, audio) for audio in workload]
        for future in as_completed(futures):
            audio, latency_sec, succeeded, duration_sec, text_chars, error_text = (
                future.result()
            )
            latencies.append(latency_sec)
            stats = per_audio.setdefault(
                audio,
                {
                    "audio": audio,
                    "submitted": 0,
                    "successful": 0,
                    "failed": 0,
                    "audio_duration_sec": round(duration_by_audio.get(audio, 0.0), 3),
                    "latencies_sec": [],
                    "output_chars": 0,
                },
            )
            stats["submitted"] += 1
            stats["latencies_sec"].append(latency_sec)
            if succeeded:
                stats["successful"] += 1
                stats["output_chars"] += text_chars
                total_audio_duration_sec += duration_sec
                total_output_chars += text_chars
            else:
                stats["failed"] += 1
                failures += 1
                if len(errors) < 20:
                    errors.append({"audio": audio, "error": error_text})

    total_time_sec = time.perf_counter() - started_at
    successful = len(workload) - failures
    per_audio_payload = []
    for audio, stats in sorted(per_audio.items()):
        sample_latencies = stats.pop("latencies_sec")
        stats["latency_avg_sec"] = round(
            sum(sample_latencies) / len(sample_latencies), 4
        )
        stats["latency_p50_sec"] = round(percentile(sample_latencies, 0.50), 4)
        stats["latency_p95_sec"] = round(percentile(sample_latencies, 0.95), 4)
        audio_duration_sec = float(stats.get("audio_duration_sec", 0.0) or 0.0)
        latency_avg_sec = float(stats["latency_avg_sec"])
        stats["real_time_factor"] = (
            round(latency_avg_sec / audio_duration_sec, 4)
            if audio_duration_sec > 0
            else None
        )
        per_audio_payload.append(stats)

    return {
        "submitted": len(workload),
        "successful": successful,
        "failed": failures,
        "success_rate": round(successful / len(workload), 4) if workload else 0.0,
        "total_time_sec": round(total_time_sec, 3),
        "requests_per_second": (
            round(successful / total_time_sec, 3) if total_time_sec > 0 else 0.0
        ),
        "audio_seconds_per_second": (
            round(total_audio_duration_sec / total_time_sec, 3)
            if total_time_sec > 0
            else 0.0
        ),
        "aggregate_real_time_factor": (
            round(total_time_sec / total_audio_duration_sec, 4)
            if total_audio_duration_sec > 0
            else None
        ),
        "total_audio_duration_sec": round(total_audio_duration_sec, 3),
        "total_output_chars": total_output_chars,
        "latency_sec": {
            "avg": round(sum(latencies) / len(latencies), 4) if latencies else 0.0,
            "p50": round(percentile(latencies, 0.50), 4),
            "p95": round(percentile(latencies, 0.95), 4),
            "p99": round(percentile(latencies, 0.99), 4),
        },
        "per_audio": per_audio_payload,
        "sample_errors": errors,
    }


def main() -> None:
    args = parse_args()
    ensure_blux_import(args.blux_src)
    audio_inputs = gather_audio_inputs(args)
    local_audio_metadata = load_local_audio_metadata(audio_inputs)

    before_info = client_info(args.endpoint, args.timeout_sec)
    before_requests = machine_requests_snapshot(before_info)

    probe_rows, duration_by_audio = probe_audio_samples(
        args.endpoint,
        audio_inputs,
        local_audio_metadata,
        timeout_sec=args.timeout_sec,
        language=args.language,
        prompt=args.prompt,
    )
    batch_result = run_batch(
        args.endpoint,
        audio_inputs,
        duration_by_audio,
        repeats=args.repeats,
        max_workers=args.max_workers,
        timeout_sec=args.timeout_sec,
        language=args.language,
        prompt=args.prompt,
    )

    after_info = client_info(args.endpoint, args.timeout_sec)
    after_requests = machine_requests_snapshot(after_info)
    payload = {
        "config": {
            "endpoint": args.endpoint,
            "audio": audio_inputs,
            "repeats": args.repeats,
            "max_workers": args.max_workers,
            "timeout_sec": args.timeout_sec,
            "language": args.language,
            "prompt": args.prompt,
        },
        "probe": {
            "samples": probe_rows,
            "total_audio_duration_sec": round(sum(duration_by_audio.values()), 3),
            "avg_real_time_factor": round(
                sum(
                    row["real_time_factor"]
                    for row in probe_rows
                    if row["real_time_factor"] is not None
                )
                / max(
                    1,
                    sum(1 for row in probe_rows if row["real_time_factor"] is not None),
                ),
                4,
            ),
        },
        "batch": batch_result,
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
    }

    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n")

    print(json.dumps(payload, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
