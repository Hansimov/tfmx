from __future__ import annotations

"""Long-audio chunk planning, scheduling, and stitching for QSR."""

import json
import math
import threading
import shutil
import subprocess
import tempfile
import time

from collections import Counter
from concurrent.futures import FIRST_COMPLETED, Future, ThreadPoolExecutor, wait
from dataclasses import dataclass, field
from difflib import SequenceMatcher
from pathlib import Path
from typing import TYPE_CHECKING, Optional

import httpx


if TYPE_CHECKING:
    from .client import InfoResponse


FFMPEG_REQUIRED_MESSAGE = (
    "ffmpeg and ffprobe are required for qsr long-audio transcription. "
    "Install them first, then retry."
)
_SILENCE_START_TOKEN = "silence_start:"
_SILENCE_END_TOKEN = "silence_end:"
_BOUNDARY_DELIMITERS = frozenset("。！？!?；;\n")
_REPETITION_DELIMITERS = frozenset(".。！？!?；;，,\n")
_SIMILAR_BOUNDARY_MIN_CHARS = 12
_SIMILAR_BOUNDARY_MIN_RATIO = 0.82
_SIMILAR_BOUNDARY_MAX_FRAGMENTS = 2
_SIMILAR_BOUNDARY_MAX_PREFIX_SKIP = 1
_SIMILAR_BOUNDARY_MAX_PREFIX_SKIP_CHARS = 10
_FUZZY_NORMALIZED_OVERLAP_MIN_CHARS = 18
_FUZZY_NORMALIZED_OVERLAP_MIN_RATIO = 0.9
_PATHOLOGICAL_REPETITION_MIN_TOTAL_CHARS = 64
_PATHOLOGICAL_REPETITION_MIN_FRAGMENT_CHARS = 1
_PATHOLOGICAL_REPETITION_MAX_FRAGMENT_CHARS = 80
_PATHOLOGICAL_REPETITION_MIN_OCCURRENCES = 8
_PATHOLOGICAL_REPETITION_MIN_COVERAGE = 0.33
_PATHOLOGICAL_REPETITION_MIN_CONSECUTIVE = 4
_PATHOLOGICAL_REPETITION_MIN_CONSECUTIVE_COVERAGE = 0.2
_PATHOLOGICAL_REPETITION_MIN_TOKEN_CHARS = 1
_PATHOLOGICAL_REPETITION_MAX_TOKEN_CHARS = 24
_PATHOLOGICAL_REPETITION_MIN_TOKEN_OCCURRENCES = 16
_PATHOLOGICAL_REPETITION_MIN_TOKEN_COVERAGE = 0.18
_PATHOLOGICAL_REPETITION_MIN_TOKEN_CONSECUTIVE = 8
_PATHOLOGICAL_REPETITION_MIN_TOKEN_CONSECUTIVE_COVERAGE = 0.12
_QUALITY_REPAIR_MAX_PASSES = 4
_QUALITY_REPAIR_ACCEPT_MIN_NORMALIZED_CHARS = 96
_QUALITY_REPAIR_ACCEPT_MAX_NORMALIZED_CHARS = 256
_QUALITY_REPAIR_ACCEPT_CHARS_PER_SEC = 3.0
_TARGETED_EDGE_RESPLIT_MAX_FRAGMENT_CHARS = 8
_TARGETED_EDGE_RESPLIT_MAX_EDGE_SKIP_FRAGMENTS = 32
_TARGETED_EDGE_RESPLIT_MIN_PREFIX_FRAGMENTS_FOR_SUFFIX = 16
_TARGETED_EDGE_RESPLIT_MIN_RUN_COUNT = 32
_TARGETED_EDGE_RESPLIT_MIN_COVERAGE = 0.45
_QUALITY_SPLIT_MAX_DEPTH = 3
_QUALITY_SPLIT_MIN_CHILD_SEC = 8.0
_QUALITY_SPLIT_OVERLAP_SEC = 1.5
_VERBOSE_JSON_CAPABILITY_CACHE: dict[str, bool] = {}
_VERBOSE_JSON_CAPABILITY_LOCK = threading.Lock()


def _require_binary(name: str) -> str:
    path = shutil.which(name)
    if not path:
        raise RuntimeError(FFMPEG_REQUIRED_MESSAGE)
    return path


def _run_command(command: list[str]) -> subprocess.CompletedProcess[str]:
    try:
        return subprocess.run(
            command,
            capture_output=True,
            text=True,
            check=True,
        )
    except subprocess.CalledProcessError as exc:
        detail = (exc.stderr or exc.stdout or "").strip()
        raise RuntimeError(detail or "Command failed") from exc


@dataclass
class AudioMetadata:
    path: str
    duration_sec: float
    sample_rate: int | None = None
    channels: int | None = None
    bit_rate: int | None = None
    size_bytes: int = 0

    def to_dict(self) -> dict:
        return {
            "path": self.path,
            "duration_sec": round(self.duration_sec, 3),
            "sample_rate": self.sample_rate,
            "channels": self.channels,
            "bit_rate": self.bit_rate,
            "size_bytes": self.size_bytes,
        }


@dataclass
class SilenceRegion:
    start_sec: float
    end_sec: float

    @property
    def duration_sec(self) -> float:
        return max(0.0, self.end_sec - self.start_sec)

    @property
    def center_sec(self) -> float:
        return self.start_sec + self.duration_sec / 2.0

    def to_dict(self) -> dict:
        return {
            "start_sec": round(self.start_sec, 3),
            "end_sec": round(self.end_sec, 3),
            "duration_sec": round(self.duration_sec, 3),
        }


@dataclass
class AudioChunk:
    index: int
    start_sec: float
    end_sec: float
    keep_start_sec: float = 0.0
    keep_end_sec: float = 0.0
    output_path: str = ""

    @property
    def duration_sec(self) -> float:
        return max(0.0, self.end_sec - self.start_sec)

    def to_dict(self) -> dict:
        return {
            "index": self.index,
            "start_sec": round(self.start_sec, 3),
            "end_sec": round(self.end_sec, 3),
            "duration_sec": round(self.duration_sec, 3),
            "keep_start_sec": round(self.keep_start_sec, 3),
            "keep_end_sec": round(self.keep_end_sec, 3),
            "output_path": self.output_path,
        }


@dataclass(frozen=True)
class QualityFallbackTrace:
    pathological_repetition_detected: bool = False
    repair_attempted: bool = False
    repair_only_accepted: bool = False
    split_fallback_used: bool = False
    post_split_repair_applied: bool = False
    split_child_count: int = 0
    max_split_depth: int = 0
    repair_passes_applied: int = 0
    total_transcribe_requests: int = 1

    def to_dict(self) -> dict:
        return {
            "pathological_repetition_detected": self.pathological_repetition_detected,
            "repair_attempted": self.repair_attempted,
            "repair_only_accepted": self.repair_only_accepted,
            "split_fallback_used": self.split_fallback_used,
            "post_split_repair_applied": self.post_split_repair_applied,
            "split_child_count": self.split_child_count,
            "max_split_depth": self.max_split_depth,
            "repair_passes_applied": self.repair_passes_applied,
            "total_transcribe_requests": self.total_transcribe_requests,
        }


@dataclass
class SchedulingDiagnostics:
    dispatch_cycle_count: int = 0
    forced_machine_refresh_count: int = 0
    forced_machine_refresh_hit_count: int = 0
    completion_refill_trigger_count: int = 0
    completion_refill_hit_count: int = 0
    completion_refill_dispatched_chunk_count: int = 0
    idle_poll_wait_count: int = 0
    wait_timeout_count: int = 0

    def to_dict(self) -> dict:
        return {
            "dispatch_cycle_count": self.dispatch_cycle_count,
            "forced_machine_refresh_count": self.forced_machine_refresh_count,
            "forced_machine_refresh_hit_count": self.forced_machine_refresh_hit_count,
            "completion_refill_trigger_count": self.completion_refill_trigger_count,
            "completion_refill_hit_count": self.completion_refill_hit_count,
            "completion_refill_dispatched_chunk_count": self.completion_refill_dispatched_chunk_count,
            "idle_poll_wait_count": self.idle_poll_wait_count,
            "wait_timeout_count": self.wait_timeout_count,
        }


@dataclass
class ChunkTranscriptionResult:
    chunk_index: int
    start_sec: float
    end_sec: float
    keep_start_sec: float
    keep_end_sec: float
    latency_sec: float
    attempt: int
    text: str
    language: str
    duration_sec: float | None
    segments: list[dict] = field(default_factory=list)
    output_path: str = ""
    quality_fallback: QualityFallbackTrace = field(default_factory=QualityFallbackTrace)

    def to_dict(self) -> dict:
        return {
            "chunk_index": self.chunk_index,
            "start_sec": round(self.start_sec, 3),
            "end_sec": round(self.end_sec, 3),
            "keep_start_sec": round(self.keep_start_sec, 3),
            "keep_end_sec": round(self.keep_end_sec, 3),
            "latency_sec": round(self.latency_sec, 3),
            "attempt": self.attempt,
            "text": self.text,
            "language": self.language,
            "duration_sec": (
                round(self.duration_sec, 3) if self.duration_sec is not None else None
            ),
            "segments": self.segments,
            "output_path": self.output_path,
            "quality_fallback": self.quality_fallback.to_dict(),
        }


@dataclass
class LongAudioTranscriptionResult:
    audio: str
    metadata: AudioMetadata
    chunks: list[AudioChunk]
    silence_regions: list[SilenceRegion]
    chunk_results: list[ChunkTranscriptionResult]
    total_time_sec: float
    max_parallel_chunks: int
    endpoint: str
    machine_snapshot: dict | None
    text: str
    language: str
    scheduling_diagnostics: SchedulingDiagnostics = field(
        default_factory=SchedulingDiagnostics
    )
    merged_segments: list[dict] = field(default_factory=list)
    chunk_response_format: str = "json"
    stitching_mode: str = "text_overlap"

    def _quality_fallback_summary(self) -> dict:
        if not self.chunk_results:
            return {
                "chunk_count": 0,
                "pathological_chunks": 0,
                "repair_only_accepted_chunks": 0,
                "split_fallback_chunks": 0,
                "post_split_repair_chunks": 0,
                "max_split_depth": 0,
                "repair_passes_applied": 0,
                "total_transcribe_requests": 0,
                "extra_transcribe_requests": 0,
            }

        traces = [result.quality_fallback for result in self.chunk_results]
        total_transcribe_requests = sum(
            trace.total_transcribe_requests for trace in traces
        )
        return {
            "chunk_count": len(traces),
            "pathological_chunks": sum(
                trace.pathological_repetition_detected for trace in traces
            ),
            "repair_only_accepted_chunks": sum(
                trace.repair_only_accepted for trace in traces
            ),
            "split_fallback_chunks": sum(trace.split_fallback_used for trace in traces),
            "post_split_repair_chunks": sum(
                trace.post_split_repair_applied for trace in traces
            ),
            "max_split_depth": max(trace.max_split_depth for trace in traces),
            "repair_passes_applied": sum(
                trace.repair_passes_applied for trace in traces
            ),
            "total_transcribe_requests": total_transcribe_requests,
            "extra_transcribe_requests": max(
                0,
                total_transcribe_requests - len(traces),
            ),
        }

    def to_dict(self) -> dict:
        total_audio_duration_sec = sum(chunk.duration_sec for chunk in self.chunks)
        return {
            "audio": self.audio,
            "endpoint": self.endpoint,
            "metadata": self.metadata.to_dict(),
            "planning": {
                "chunks": [chunk.to_dict() for chunk in self.chunks],
                "silence_regions": [
                    region.to_dict() for region in self.silence_regions
                ],
            },
            "scheduling": {
                "max_parallel_chunks": self.max_parallel_chunks,
                "total_time_sec": round(self.total_time_sec, 3),
                "audio_seconds_per_second": (
                    round(total_audio_duration_sec / self.total_time_sec, 3)
                    if self.total_time_sec > 0
                    else 0.0
                ),
                "aggregate_real_time_factor": (
                    round(
                        self.total_time_sec / total_audio_duration_sec,
                        4,
                    )
                    if total_audio_duration_sec > 0
                    else None
                ),
                "diagnostics": self.scheduling_diagnostics.to_dict(),
            },
            "machine_snapshot": self.machine_snapshot,
            "transcription": {
                "text": self.text,
                "language": self.language,
                "duration": round(self.metadata.duration_sec, 3),
                "chunk_response_format": self.chunk_response_format,
                "stitching_mode": self.stitching_mode,
                "quality_fallback_summary": self._quality_fallback_summary(),
                "segments": self.merged_segments,
                "chunks": [result.to_dict() for result in self.chunk_results],
            },
        }


@dataclass
class LongAudioTranscriptionConfig:
    model: str = ""
    language: str | None = None
    prompt: str | None = None
    transcription_response_format: str = "auto"
    timeout_sec: float = 900.0
    target_chunk_sec: float = 60.0
    min_chunk_sec: float = 35.0
    max_chunk_sec: float = 90.0
    overlap_sec: float = 4.0
    search_window_sec: float = 12.0
    min_silence_sec: float = 0.35
    silence_noise_db: float = -32.0
    idle_poll_interval_sec: float = 0.2
    machine_info_refresh_sec: float = 0.2
    max_parallel_chunks: int | None = None
    per_instance_parallelism_cap: int | None = 4
    max_chunk_retries: int = 2
    keep_chunks: bool = False
    work_dir: str | None = None


@dataclass(frozen=True)
class ChunkRequestPlan:
    response_format: str = "json"
    timestamp_granularities: tuple[str, ...] = ()


_JSON_CHUNK_REQUEST_PLAN = ChunkRequestPlan()
_VERBOSE_JSON_CHUNK_REQUEST_PLAN = ChunkRequestPlan(
    response_format="verbose_json",
    timestamp_granularities=("segment",),
)


def probe_audio(audio_path: str) -> AudioMetadata:
    ffprobe = _require_binary("ffprobe")
    path = Path(audio_path).expanduser().resolve()
    result = _run_command(
        [
            ffprobe,
            "-v",
            "quiet",
            "-print_format",
            "json",
            "-show_format",
            "-show_streams",
            str(path),
        ]
    )
    payload = json.loads(result.stdout)
    format_payload = payload.get("format", {})
    audio_stream = next(
        (
            stream
            for stream in payload.get("streams", [])
            if stream.get("codec_type") == "audio"
        ),
        {},
    )
    duration_sec = float(
        format_payload.get("duration") or audio_stream.get("duration") or 0.0
    )
    bit_rate = format_payload.get("bit_rate") or audio_stream.get("bit_rate")
    sample_rate = audio_stream.get("sample_rate")
    channels = audio_stream.get("channels")
    return AudioMetadata(
        path=str(path),
        duration_sec=duration_sec,
        sample_rate=int(sample_rate) if sample_rate else None,
        channels=int(channels) if channels else None,
        bit_rate=int(bit_rate) if bit_rate else None,
        size_bytes=path.stat().st_size,
    )


def detect_silence_regions(
    audio_path: str,
    *,
    min_silence_sec: float,
    silence_noise_db: float,
) -> list[SilenceRegion]:
    ffmpeg = _require_binary("ffmpeg")
    result = _run_command(
        [
            ffmpeg,
            "-hide_banner",
            "-nostats",
            "-i",
            str(Path(audio_path).expanduser().resolve()),
            "-af",
            f"silencedetect=n={silence_noise_db}dB:d={min_silence_sec}",
            "-f",
            "null",
            "-",
        ]
    )
    lines = (result.stderr or result.stdout or "").splitlines()
    silence_start: float | None = None
    regions: list[SilenceRegion] = []
    for line in lines:
        if _SILENCE_START_TOKEN in line:
            silence_start = float(line.split(_SILENCE_START_TOKEN, 1)[1].strip())
        elif _SILENCE_END_TOKEN in line and silence_start is not None:
            value = line.split(_SILENCE_END_TOKEN, 1)[1].split("|", 1)[0].strip()
            silence_end = float(value)
            if silence_end > silence_start:
                regions.append(SilenceRegion(silence_start, silence_end))
            silence_start = None
    return regions


def _pick_cut_point(
    silence_regions: list[SilenceRegion],
    *,
    segment_start_sec: float,
    total_duration_sec: float,
    target_chunk_sec: float,
    min_chunk_sec: float,
    max_chunk_sec: float,
    search_window_sec: float,
) -> float:
    min_end_sec = min(total_duration_sec, segment_start_sec + min_chunk_sec)
    ideal_end_sec = min(total_duration_sec, segment_start_sec + target_chunk_sec)
    max_end_sec = min(total_duration_sec, segment_start_sec + max_chunk_sec)

    candidate_regions = [
        region
        for region in silence_regions
        if min_end_sec <= region.center_sec <= max_end_sec
    ]
    nearby_regions = [
        region
        for region in candidate_regions
        if abs(region.center_sec - ideal_end_sec) <= search_window_sec
    ]
    if nearby_regions:
        return min(
            nearby_regions,
            key=lambda region: (
                abs(region.center_sec - ideal_end_sec),
                -region.duration_sec,
            ),
        ).center_sec
    if candidate_regions:
        return min(
            candidate_regions,
            key=lambda region: (
                abs(region.center_sec - ideal_end_sec),
                -region.duration_sec,
            ),
        ).center_sec
    return max(min_end_sec, min(ideal_end_sec, max_end_sec))


def plan_audio_chunks(
    duration_sec: float,
    *,
    silence_regions: list[SilenceRegion],
    target_chunk_sec: float,
    min_chunk_sec: float,
    max_chunk_sec: float,
    overlap_sec: float,
    search_window_sec: float,
) -> list[AudioChunk]:
    if duration_sec <= 0:
        raise ValueError("Audio duration must be positive")
    if min_chunk_sec <= 0 or target_chunk_sec <= 0 or max_chunk_sec <= 0:
        raise ValueError("Chunk durations must be positive")
    if not (min_chunk_sec <= target_chunk_sec <= max_chunk_sec):
        raise ValueError("Chunk durations must satisfy min <= target <= max")
    if overlap_sec < 0 or overlap_sec >= min_chunk_sec:
        raise ValueError("Overlap must be non-negative and smaller than min_chunk")

    chunks: list[AudioChunk] = []
    segment_start_sec = 0.0
    index = 0
    min_progress_sec = max(1.0, overlap_sec + 1.0)

    while segment_start_sec < duration_sec:
        remaining_sec = duration_sec - segment_start_sec
        if remaining_sec <= max_chunk_sec:
            segment_end_sec = duration_sec
        else:
            segment_end_sec = _pick_cut_point(
                silence_regions,
                segment_start_sec=segment_start_sec,
                total_duration_sec=duration_sec,
                target_chunk_sec=target_chunk_sec,
                min_chunk_sec=min_chunk_sec,
                max_chunk_sec=max_chunk_sec,
                search_window_sec=search_window_sec,
            )
            if duration_sec - segment_end_sec < min_chunk_sec * 0.65:
                segment_end_sec = duration_sec

        segment_end_sec = min(
            duration_sec, max(segment_start_sec + min_progress_sec, segment_end_sec)
        )
        chunks.append(
            AudioChunk(
                index=index,
                start_sec=segment_start_sec,
                end_sec=segment_end_sec,
            )
        )
        index += 1
        if segment_end_sec >= duration_sec:
            break
        next_start_sec = max(0.0, segment_end_sec - overlap_sec)
        if next_start_sec <= segment_start_sec:
            next_start_sec = min(duration_sec, segment_start_sec + min_progress_sec)
        segment_start_sec = next_start_sec

    for idx, chunk in enumerate(chunks):
        previous_end = chunks[idx - 1].end_sec if idx > 0 else chunk.start_sec
        next_start = (
            chunks[idx + 1].start_sec if idx + 1 < len(chunks) else chunk.end_sec
        )
        chunk.keep_start_sec = (
            chunk.start_sec if idx == 0 else (previous_end + chunk.start_sec) / 2.0
        )
        chunk.keep_end_sec = (
            chunk.end_sec
            if idx + 1 == len(chunks)
            else (chunk.end_sec + next_start) / 2.0
        )
    return chunks


def extract_audio_chunk(
    audio_path: str,
    chunk: AudioChunk,
    *,
    output_dir: str,
) -> AudioChunk:
    ffmpeg = _require_binary("ffmpeg")
    output_root = Path(output_dir)
    output_root.mkdir(parents=True, exist_ok=True)
    chunk_path = output_root / f"chunk_{chunk.index:04d}.wav"
    if chunk.output_path and Path(chunk.output_path).exists():
        return chunk
    duration_sec = max(0.01, chunk.duration_sec)
    _run_command(
        [
            ffmpeg,
            "-hide_banner",
            "-loglevel",
            "error",
            "-y",
            "-ss",
            f"{chunk.start_sec:.3f}",
            "-t",
            f"{duration_sec:.3f}",
            "-i",
            str(Path(audio_path).expanduser().resolve()),
            "-vn",
            "-ac",
            "1",
            "-ar",
            "16000",
            "-c:a",
            "pcm_s16le",
            str(chunk_path),
        ]
    )
    chunk.output_path = str(chunk_path)
    return chunk


def extract_audio_chunks(
    audio_path: str,
    chunks: list[AudioChunk],
    *,
    output_dir: str,
) -> list[AudioChunk]:
    for chunk in chunks:
        extract_audio_chunk(audio_path, chunk, output_dir=output_dir)
    return chunks


def _normalize_boundary_text(text: str) -> tuple[str, list[int]]:
    normalized_chars: list[str] = []
    index_map: list[int] = []
    for index, char in enumerate(text):
        if not char.isalnum():
            continue
        normalized_chars.append(char.casefold())
        index_map.append(index)
    return "".join(normalized_chars), index_map


def _split_text_fragments(
    text: str,
    *,
    delimiters: frozenset[str],
) -> list[tuple[int, int, str]]:
    fragments: list[tuple[int, int, str]] = []
    start = 0
    index = 0

    while index < len(text):
        if text[index] in delimiters:
            end = index + 1
            while end < len(text) and text[end] in delimiters:
                end += 1
            fragment = text[start:end].strip()
            if fragment:
                fragments.append((start, end, fragment))
            start = end
            while start < len(text) and text[start].isspace():
                start += 1
            index = start
            continue
        index += 1

    tail_fragment = text[start:].strip()
    if tail_fragment:
        fragments.append((start, len(text), tail_fragment))
    return fragments


def _split_boundary_fragments(text: str) -> list[tuple[int, int, str]]:
    return _split_text_fragments(text, delimiters=_BOUNDARY_DELIMITERS)


def _split_word_tokens(text: str) -> list[str]:
    tokens: list[str] = []
    current: list[str] = []

    for char in text:
        if char.isalnum():
            current.append(char.casefold())
            continue
        if current:
            tokens.append("".join(current))
            current = []

    if current:
        tokens.append("".join(current))
    return tokens


def _extract_digit_runs(text: str) -> list[str]:
    runs: list[str] = []
    current: list[str] = []

    for char in text:
        if char.isdigit():
            current.append(char)
            continue
        if current:
            runs.append("".join(current))
            current = []

    if current:
        runs.append("".join(current))
    return runs


def _has_pathological_repetition(text: str) -> bool:
    normalized_text, _ = _normalize_boundary_text(text)
    if len(normalized_text) < _PATHOLOGICAL_REPETITION_MIN_TOTAL_CHARS:
        return False

    fragment_counts: Counter[str] = Counter()
    fragment_runs: dict[str, int] = {}
    previous_fragment = ""
    current_run = 0

    for _start, _end, fragment in _split_text_fragments(
        text,
        delimiters=_REPETITION_DELIMITERS,
    ):
        normalized_fragment, _ = _normalize_boundary_text(fragment)
        if not (
            _PATHOLOGICAL_REPETITION_MIN_FRAGMENT_CHARS
            <= len(normalized_fragment)
            <= _PATHOLOGICAL_REPETITION_MAX_FRAGMENT_CHARS
        ):
            if previous_fragment:
                fragment_runs[previous_fragment] = max(
                    fragment_runs.get(previous_fragment, 0),
                    current_run,
                )
            previous_fragment = ""
            current_run = 0
            continue

        fragment_counts[normalized_fragment] += 1
        if normalized_fragment == previous_fragment:
            current_run += 1
        else:
            if previous_fragment:
                fragment_runs[previous_fragment] = max(
                    fragment_runs.get(previous_fragment, 0),
                    current_run,
                )
            previous_fragment = normalized_fragment
            current_run = 1

    if previous_fragment:
        fragment_runs[previous_fragment] = max(
            fragment_runs.get(previous_fragment, 0),
            current_run,
        )

    total_chars = len(normalized_text)
    for fragment, count in fragment_counts.items():
        coverage = (count * len(fragment)) / total_chars
        if (
            count >= _PATHOLOGICAL_REPETITION_MIN_OCCURRENCES
            and coverage >= _PATHOLOGICAL_REPETITION_MIN_COVERAGE
        ):
            return True
        if (
            fragment_runs.get(fragment, 0) >= _PATHOLOGICAL_REPETITION_MIN_CONSECUTIVE
            and coverage >= _PATHOLOGICAL_REPETITION_MIN_CONSECUTIVE_COVERAGE
        ):
            return True

    token_counts: Counter[str] = Counter()
    token_runs: dict[str, int] = {}
    previous_token = ""
    current_run = 0
    for token in _split_word_tokens(text):
        if not (
            _PATHOLOGICAL_REPETITION_MIN_TOKEN_CHARS
            <= len(token)
            <= _PATHOLOGICAL_REPETITION_MAX_TOKEN_CHARS
        ):
            if previous_token:
                token_runs[previous_token] = max(
                    token_runs.get(previous_token, 0),
                    current_run,
                )
            previous_token = ""
            current_run = 0
            continue

        token_counts[token] += 1
        if token == previous_token:
            current_run += 1
        else:
            if previous_token:
                token_runs[previous_token] = max(
                    token_runs.get(previous_token, 0),
                    current_run,
                )
            previous_token = token
            current_run = 1

    if previous_token:
        token_runs[previous_token] = max(
            token_runs.get(previous_token, 0),
            current_run,
        )

    for token, count in token_counts.items():
        coverage = (count * len(token)) / total_chars
        if (
            count >= _PATHOLOGICAL_REPETITION_MIN_TOKEN_OCCURRENCES
            and coverage >= _PATHOLOGICAL_REPETITION_MIN_TOKEN_COVERAGE
        ):
            return True
        if (
            token_runs.get(token, 0) >= _PATHOLOGICAL_REPETITION_MIN_TOKEN_CONSECUTIVE
            and coverage >= _PATHOLOGICAL_REPETITION_MIN_TOKEN_CONSECUTIVE_COVERAGE
        ):
            return True
    return False


def _are_similar_repetition_fragments(left: str, right: str) -> bool:
    if not left or not right:
        return False
    left_normalized, _ = _normalize_boundary_text(left)
    right_normalized, _ = _normalize_boundary_text(right)
    shorter = min(len(left_normalized), len(right_normalized))
    longer = max(len(left_normalized), len(right_normalized))
    if shorter / longer < 0.65:
        return False
    if (
        left_normalized == right_normalized
        or left_normalized.endswith(right_normalized)
        or right_normalized.endswith(left_normalized)
    ):
        return True
    left_digit_runs = _extract_digit_runs(left_normalized)
    right_digit_runs = _extract_digit_runs(right_normalized)
    if (left_digit_runs or right_digit_runs) and left_digit_runs != right_digit_runs:
        return False

    left_tokens = _split_word_tokens(left_normalized)
    right_tokens = _split_word_tokens(right_normalized)
    if len(left_tokens) >= 2 and len(right_tokens) >= 2:
        if Counter(left_tokens) != Counter(right_tokens):
            return False

    return (
        SequenceMatcher(
            None,
            left_normalized,
            right_normalized,
            autojunk=False,
        ).ratio()
        >= _SIMILAR_BOUNDARY_MIN_RATIO
    )


def _collapse_repeated_word_runs(text: str) -> str:
    words = text.split()
    if len(words) < 4:
        return text

    collapsed: list[str] = []
    previous_word = ""
    for word in words:
        normalized_word, _ = _normalize_boundary_text(word)
        if normalized_word and normalized_word == previous_word:
            continue
        collapsed.append(word)
        previous_word = normalized_word
    return " ".join(collapsed)


def _join_repetition_fragments(fragments: list[str]) -> str:
    if not fragments:
        return ""

    joined: list[str] = []
    for fragment in fragments:
        if not fragment:
            continue
        if joined:
            previous = joined[-1]
            if (
                previous
                and not previous[-1].isspace()
                and not fragment[0].isspace()
                and (
                    (previous[-1].isalnum() and fragment[0].isalnum())
                    or (previous[-1] in ".!?;:," and fragment[0].isalnum())
                )
            ):
                joined.append(" ")
        joined.append(fragment)
    return "".join(joined).strip()


def _collapse_similar_fragment_runs(text: str) -> str:
    fragments = [
        (
            text[start:end].strip(),
            _normalize_boundary_text(fragment)[0],
        )
        for start, end, fragment in _split_text_fragments(
            text,
            delimiters=_REPETITION_DELIMITERS,
        )
        if text[start:end].strip()
    ]
    if len(fragments) < 2:
        return text

    collapsed: list[str] = []
    index = 0
    while index < len(fragments):
        matched = False
        max_width = min(4, (len(fragments) - index) // 2)
        for width in range(max_width, 0, -1):
            pattern = fragments[index : index + width]
            if not all(normalized for _raw, normalized in pattern):
                continue

            scan_index = index + width
            repeat_count = 1
            while scan_index + width <= len(fragments):
                candidate = fragments[scan_index : scan_index + width]
                if all(
                    _are_similar_repetition_fragments(
                        pattern[item_index][1],
                        candidate[item_index][1],
                    )
                    for item_index in range(width)
                ):
                    repeat_count += 1
                    scan_index += width
                    continue
                break

            if repeat_count >= 2:
                collapsed.extend(raw for raw, _normalized in pattern)
                index = scan_index
                matched = True
                break

        if matched:
            continue

        collapsed.append(fragments[index][0])
        index += 1
    return _join_repetition_fragments(collapsed)


def _repair_pathological_repetition(text: str) -> str:
    repaired = _collapse_repeated_word_runs(text)
    repaired = _collapse_similar_fragment_runs(repaired)
    return repaired.strip() or text


def _apply_pathological_repetition_repair(text: str) -> str:
    repaired, _ = _apply_pathological_repetition_repair_with_stats(text)
    return repaired


def _apply_pathological_repetition_repair_with_stats(text: str) -> tuple[str, int]:
    repaired = text
    repair_passes = 0
    for _ in range(_QUALITY_REPAIR_MAX_PASSES):
        updated = _repair_pathological_repetition(repaired)
        if updated == repaired:
            break
        repaired = updated
        repair_passes += 1
    return repaired, repair_passes


def _should_accept_repaired_chunk_text(
    text: str,
    *,
    duration_sec: float,
) -> bool:
    normalized_text, _ = _normalize_boundary_text(text)
    if not normalized_text:
        return False

    min_chars = max(
        _QUALITY_REPAIR_ACCEPT_MIN_NORMALIZED_CHARS,
        min(
            _QUALITY_REPAIR_ACCEPT_MAX_NORMALIZED_CHARS,
            math.ceil(max(duration_sec, 0.0) * _QUALITY_REPAIR_ACCEPT_CHARS_PER_SEC),
        ),
    )
    return len(normalized_text) >= min_chars


def _targeted_edge_resplit_side(text: str) -> str | None:
    normalized_text, _ = _normalize_boundary_text(text)
    if len(normalized_text) < _PATHOLOGICAL_REPETITION_MIN_TOTAL_CHARS:
        return None

    fragments = [
        _normalize_boundary_text(fragment)[0]
        for _start, _end, fragment in _split_text_fragments(
            text,
            delimiters=_REPETITION_DELIMITERS,
        )
    ]
    if not fragments:
        return None

    best_side: str | None = None
    best_score = (0.0, 0, 0)
    current_fragment = fragments[0]
    current_start = 0
    current_run = 1

    def consider_run(fragment: str, start_index: int, run_count: int) -> None:
        nonlocal best_side, best_score
        if not (
            _PATHOLOGICAL_REPETITION_MIN_FRAGMENT_CHARS
            <= len(fragment)
            <= _TARGETED_EDGE_RESPLIT_MAX_FRAGMENT_CHARS
        ):
            return

        coverage = (run_count * len(fragment)) / len(normalized_text)
        if (
            run_count < _TARGETED_EDGE_RESPLIT_MIN_RUN_COUNT
            or coverage < _TARGETED_EDGE_RESPLIT_MIN_COVERAGE
        ):
            return

        suffix_skip = len(fragments) - (start_index + run_count)
        candidate_sides: list[tuple[str, int]] = []
        if start_index == 0:
            candidate_sides.append(("prefix", start_index))
        if (
            suffix_skip <= _TARGETED_EDGE_RESPLIT_MAX_EDGE_SKIP_FRAGMENTS
            and start_index >= _TARGETED_EDGE_RESPLIT_MIN_PREFIX_FRAGMENTS_FOR_SUFFIX
        ):
            candidate_sides.append(("suffix", suffix_skip))

        for side, edge_skip in candidate_sides:
            score = (coverage, run_count, -edge_skip)
            if score > best_score:
                best_side = side
                best_score = score

    for index in range(1, len(fragments) + 1):
        if index < len(fragments) and fragments[index] == current_fragment:
            current_run += 1
            continue
        consider_run(current_fragment, current_start, current_run)
        if index < len(fragments):
            current_fragment = fragments[index]
            current_start = index
            current_run = 1
    return best_side


def _find_similar_fragment_overlap_cut(left_text: str, right_text: str) -> int | None:
    left_fragments = _split_boundary_fragments(left_text)
    right_fragments = _split_boundary_fragments(right_text)
    if not left_fragments or not right_fragments:
        return None

    max_width = min(
        len(left_fragments),
        len(right_fragments),
        _SIMILAR_BOUNDARY_MAX_FRAGMENTS,
    )
    for width in range(max_width, 0, -1):
        left_candidate = "".join(
            fragment for _start, _end, fragment in left_fragments[-width:]
        )
        left_normalized, _ = _normalize_boundary_text(left_candidate)
        if len(left_normalized) < _SIMILAR_BOUNDARY_MIN_CHARS:
            continue

        max_prefix_skip = min(
            _SIMILAR_BOUNDARY_MAX_PREFIX_SKIP,
            len(right_fragments) - width,
        )
        for prefix_skip in range(0, max_prefix_skip + 1):
            skipped_prefix = "".join(
                fragment for _start, _end, fragment in right_fragments[:prefix_skip]
            )
            skipped_normalized, _ = _normalize_boundary_text(skipped_prefix)
            if len(skipped_normalized) > _SIMILAR_BOUNDARY_MAX_PREFIX_SKIP_CHARS:
                continue

            right_candidate = "".join(
                fragment
                for _start, _end, fragment in right_fragments[
                    prefix_skip : prefix_skip + width
                ]
            )
            right_normalized, _ = _normalize_boundary_text(right_candidate)
            if len(right_normalized) < _SIMILAR_BOUNDARY_MIN_CHARS:
                continue

            shorter = min(len(left_normalized), len(right_normalized))
            longer = max(len(left_normalized), len(right_normalized))
            if shorter / longer < 0.65:
                continue

            if (
                left_normalized == right_normalized
                or left_normalized.endswith(right_normalized)
                or right_normalized.endswith(left_normalized)
            ):
                return right_fragments[prefix_skip + width - 1][1]

            ratio = SequenceMatcher(
                None,
                left_normalized,
                right_normalized,
                autojunk=False,
            ).ratio()
            if ratio >= _SIMILAR_BOUNDARY_MIN_RATIO:
                return right_fragments[prefix_skip + width - 1][1]
    return None


def _find_normalized_overlap_cut(
    left_text: str,
    right_text: str,
    *,
    max_chars: int,
) -> int | None:
    left_normalized, _left_map = _normalize_boundary_text(left_text)
    right_normalized, right_map = _normalize_boundary_text(right_text)
    max_overlap = min(len(left_normalized), len(right_normalized), max_chars)
    if max_overlap <= 0:
        return None

    for width in range(max_overlap, 0, -1):
        if left_normalized[-width:] == right_normalized[:width]:
            return right_map[width - 1] + 1

    if max_overlap < 2:
        return None

    left_window = left_normalized[-max_overlap:]
    right_window = right_normalized[:max_overlap]
    match = SequenceMatcher(
        None,
        left_window,
        right_window,
        autojunk=False,
    ).find_longest_match(0, len(left_window), 0, len(right_window))
    if match.size < 2:
        return None
    if match.a + match.size != len(left_window) or match.b != 0:
        return None
    return right_map[match.size - 1] + 1


def _find_fuzzy_normalized_overlap_cut(
    left_text: str,
    right_text: str,
    *,
    max_chars: int,
) -> int | None:
    left_normalized, _left_map = _normalize_boundary_text(left_text)
    right_normalized, right_map = _normalize_boundary_text(right_text)
    max_overlap = min(len(left_normalized), len(right_normalized), max_chars)
    if max_overlap < _FUZZY_NORMALIZED_OVERLAP_MIN_CHARS:
        return None

    for width in range(max_overlap, _FUZZY_NORMALIZED_OVERLAP_MIN_CHARS - 1, -1):
        ratio = SequenceMatcher(
            None,
            left_normalized[-width:],
            right_normalized[:width],
            autojunk=False,
        ).ratio()
        if ratio >= _FUZZY_NORMALIZED_OVERLAP_MIN_RATIO:
            return right_map[width - 1] + 1
    return None


def _strip_duplicate_boundary(left: str, right: str, *, max_chars: int = 64) -> str:
    left_text = left.rstrip()
    right_text = right.lstrip()
    max_overlap = min(len(left_text), len(right_text), max_chars)
    for width in range(max_overlap, 0, -1):
        if left_text[-width:] == right_text[:width]:
            return left_text + right_text[width:]
    fuzzy_cut = _find_fuzzy_normalized_overlap_cut(
        left_text,
        right_text,
        max_chars=max_chars,
    )
    if fuzzy_cut is not None:
        return left_text + right_text[fuzzy_cut:]
    fragment_cut = _find_similar_fragment_overlap_cut(left_text, right_text)
    if fragment_cut is not None:
        return left_text + right_text[fragment_cut:]
    normalized_cut = _find_normalized_overlap_cut(
        left_text,
        right_text,
        max_chars=max_chars,
    )
    if normalized_cut is not None:
        return left_text + right_text[normalized_cut:]
    return left_text + right_text


def merge_transcribed_chunks(
    chunks: list[AudioChunk],
    results: list[ChunkTranscriptionResult],
) -> tuple[str, str, list[dict]]:
    ordered_results = sorted(results, key=lambda item: item.chunk_index)
    merged_segments: list[dict] = []
    languages = [result.language for result in ordered_results if result.language]

    for result in ordered_results:
        if not result.segments:
            continue
        for segment in result.segments:
            start = float(segment.get("start", 0.0) or 0.0) + result.start_sec
            end = float(segment.get("end", 0.0) or 0.0) + result.start_sec
            midpoint = (start + end) / 2.0
            if midpoint < result.keep_start_sec or midpoint > result.keep_end_sec:
                continue
            merged_segment = dict(segment)
            merged_segment["start"] = round(start, 3)
            merged_segment["end"] = round(end, 3)
            if merged_segments:
                previous_segment = merged_segments[-1]
                previous_text = str(previous_segment.get("text", "")).strip()
                current_text = str(merged_segment.get("text", "")).strip()
                previous_end = float(previous_segment.get("end", 0.0) or 0.0)
                if (
                    previous_text
                    and current_text
                    and previous_text == current_text
                    and start <= previous_end
                ):
                    continue
            merged_segments.append(merged_segment)

    if merged_segments:
        merged_text = "".join(
            str(segment.get("text", ""))
            for segment in merged_segments
            if segment.get("text")
        ).strip()
    else:
        merged_text = ""
        for result in ordered_results:
            text = result.text.strip()
            if not text:
                continue
            if not merged_text:
                merged_text = text
                continue
            merged_text = _strip_duplicate_boundary(merged_text, text)

    return merged_text, (languages[0] if languages else ""), merged_segments


def count_idle_healthy_instances(info: "InfoResponse") -> int:
    return sum(
        1
        for instance in info.instances
        if instance.healthy
        and not getattr(instance, "sleeping", False)
        and instance.active_requests == 0
    )


def count_available_healthy_slots(info: "InfoResponse") -> int:
    return sum(
        max(0, instance.available_slots)
        for instance in info.instances
        if instance.healthy and not getattr(instance, "sleeping", False)
    )


def count_schedulable_healthy_slots(
    info: "InfoResponse",
    *,
    per_instance_parallelism_cap: int | None = None,
) -> int:
    total_slots = 0
    for instance in info.instances:
        if not instance.healthy or getattr(instance, "sleeping", False):
            continue
        available_slots = max(0, instance.available_slots)
        if (
            per_instance_parallelism_cap is not None
            and per_instance_parallelism_cap > 0
        ):
            available_slots = min(
                available_slots,
                max(0, per_instance_parallelism_cap - max(0, instance.active_requests)),
            )
        total_slots += available_slots
    return total_slots


def machine_snapshot(
    info: "InfoResponse",
    *,
    per_instance_parallelism_cap: int | None = None,
) -> dict:
    return {
        "healthy_instances": sum(1 for instance in info.instances if instance.healthy),
        "idle_healthy_instances": count_idle_healthy_instances(info),
        "available_healthy_slots": count_available_healthy_slots(info),
        "schedulable_healthy_slots": count_schedulable_healthy_slots(
            info,
            per_instance_parallelism_cap=per_instance_parallelism_cap,
        ),
        "per_instance_parallelism_cap": per_instance_parallelism_cap,
        "instances": [
            {
                "name": instance.name,
                "endpoint": instance.endpoint,
                "healthy": instance.healthy,
                "sleeping": instance.sleeping,
                "active_requests": instance.active_requests,
                "available_slots": instance.available_slots,
            }
            for instance in info.instances
        ],
        "stats": info.stats.__dict__,
    }


class LongAudioTranscriber:
    def __init__(
        self, endpoint: str, config: LongAudioTranscriptionConfig | None = None
    ):
        self.endpoint = endpoint.rstrip("/")
        self.config = config or LongAudioTranscriptionConfig()

    def _machine_info(self) -> "InfoResponse" | None:
        try:
            from .client import QSRClient

            with QSRClient(
                endpoint=self.endpoint,
                verbose=False,
                timeout_sec=self.config.timeout_sec,
            ) as client:
                return client.info()
        except Exception:
            return None

    def _capability_cache_key(self) -> str:
        return f"{self.endpoint}|{self.config.model.strip()}"

    def _get_cached_verbose_json_support(self) -> bool | None:
        with _VERBOSE_JSON_CAPABILITY_LOCK:
            return _VERBOSE_JSON_CAPABILITY_CACHE.get(self._capability_cache_key())

    def _set_cached_verbose_json_support(self, supported: bool) -> None:
        with _VERBOSE_JSON_CAPABILITY_LOCK:
            _VERBOSE_JSON_CAPABILITY_CACHE[self._capability_cache_key()] = supported

    def _normalized_transcription_response_format(self) -> str:
        normalized = self.config.transcription_response_format.strip().lower()
        if normalized in {"json", "verbose_json"}:
            return normalized
        return "auto"

    def _select_probe_chunk(self, chunks: list[AudioChunk]) -> AudioChunk | None:
        if not chunks:
            return None
        return min(chunks, key=lambda chunk: (chunk.duration_sec, chunk.index))

    def _transcribe_chunk_with_plan(
        self,
        audio_path: str,
        work_dir: str,
        chunk: AudioChunk,
        attempt: int,
        request_plan: ChunkRequestPlan,
    ) -> ChunkTranscriptionResult:
        started_at = time.perf_counter()
        extract_audio_chunk(audio_path, chunk, output_dir=work_dir)
        from .client import QSRClient

        with QSRClient(
            endpoint=self.endpoint,
            verbose=False,
            timeout_sec=self.config.timeout_sec,
        ) as client:
            response = client.transcribe(
                audio=chunk.output_path,
                model=self.config.model,
                language=self.config.language,
                prompt=self.config.prompt,
                response_format=request_plan.response_format,
                timestamp_granularities=(
                    list(request_plan.timestamp_granularities)
                    if request_plan.timestamp_granularities
                    else None
                ),
            )
        return ChunkTranscriptionResult(
            chunk_index=chunk.index,
            start_sec=chunk.start_sec,
            end_sec=chunk.end_sec,
            keep_start_sec=chunk.keep_start_sec,
            keep_end_sec=chunk.keep_end_sec,
            latency_sec=time.perf_counter() - started_at,
            attempt=attempt,
            text=response.text,
            language=response.language,
            duration_sec=response.duration,
            segments=response.segments or [],
            output_path=chunk.output_path,
        )

    def _resolve_chunk_request_plan(
        self,
        audio_path: str,
        work_dir: str,
        probe_chunk: AudioChunk | None,
    ) -> tuple[ChunkRequestPlan, ChunkTranscriptionResult | None]:
        requested_format = self._normalized_transcription_response_format()
        if requested_format == "json" or probe_chunk is None:
            return _JSON_CHUNK_REQUEST_PLAN, None

        cached_support = self._get_cached_verbose_json_support()
        if cached_support is True:
            return _VERBOSE_JSON_CHUNK_REQUEST_PLAN, None
        if cached_support is False:
            return _JSON_CHUNK_REQUEST_PLAN, None

        try:
            probe_result = self._transcribe_chunk_with_plan(
                audio_path,
                work_dir,
                probe_chunk,
                1,
                _VERBOSE_JSON_CHUNK_REQUEST_PLAN,
            )
        except httpx.HTTPStatusError as exc:
            if exc.response is not None and exc.response.status_code == 400:
                self._set_cached_verbose_json_support(False)
                return _JSON_CHUNK_REQUEST_PLAN, None
            raise

        if probe_result.segments:
            self._set_cached_verbose_json_support(True)
            return _VERBOSE_JSON_CHUNK_REQUEST_PLAN, probe_result

        self._set_cached_verbose_json_support(False)
        return _JSON_CHUNK_REQUEST_PLAN, probe_result

    def _transcribe_chunk(
        self,
        audio_path: str,
        work_dir: str,
        chunk: AudioChunk,
        attempt: int,
        request_plan: ChunkRequestPlan,
    ) -> ChunkTranscriptionResult:
        return self._transcribe_chunk_with_quality_fallback(
            audio_path,
            work_dir,
            chunk,
            attempt,
            request_plan,
        )

    def _split_chunk_for_quality_fallback(self, chunk: AudioChunk) -> list[AudioChunk]:
        if chunk.duration_sec < _QUALITY_SPLIT_MIN_CHILD_SEC * 2:
            return []

        boundary_sec = chunk.start_sec + chunk.duration_sec / 2.0
        overlap_sec = min(
            _QUALITY_SPLIT_OVERLAP_SEC,
            max(0.5, self.config.overlap_sec / 2.0),
        )
        left_end_sec = min(chunk.end_sec, boundary_sec + overlap_sec / 2.0)
        right_start_sec = max(chunk.start_sec, boundary_sec - overlap_sec / 2.0)
        if right_start_sec >= left_end_sec:
            left_end_sec = boundary_sec
            right_start_sec = boundary_sec

        return [
            AudioChunk(
                index=0,
                start_sec=chunk.start_sec,
                end_sec=left_end_sec,
                keep_start_sec=chunk.start_sec,
                keep_end_sec=boundary_sec,
            ),
            AudioChunk(
                index=1,
                start_sec=right_start_sec,
                end_sec=chunk.end_sec,
                keep_start_sec=boundary_sec,
                keep_end_sec=chunk.end_sec,
            ),
        ]

    def _plan_quality_fallback_chunks(
        self,
        chunk: AudioChunk,
        text: str,
        *,
        split_depth: int,
    ) -> list[tuple[AudioChunk, int]]:
        child_chunks = self._split_chunk_for_quality_fallback(chunk)
        if len(child_chunks) < 2:
            return []

        planned_chunks: list[tuple[AudioChunk, int]] = [
            (child_chunk, split_depth + 1) for child_chunk in child_chunks
        ]
        if split_depth + 2 > _QUALITY_SPLIT_MAX_DEPTH:
            return planned_chunks

        targeted_side = _targeted_edge_resplit_side(text)
        if targeted_side is None:
            return planned_chunks

        child_index = 0 if targeted_side == "prefix" else 1
        resplit_children = self._split_chunk_for_quality_fallback(
            child_chunks[child_index]
        )
        if len(resplit_children) < 2:
            return planned_chunks

        if targeted_side == "prefix":
            return [
                (resplit_children[0], split_depth + 2),
                (resplit_children[1], split_depth + 2),
                (child_chunks[1], split_depth + 1),
            ]
        return [
            (child_chunks[0], split_depth + 1),
            (resplit_children[0], split_depth + 2),
            (resplit_children[1], split_depth + 2),
        ]

    def _transcribe_chunk_with_quality_fallback(
        self,
        audio_path: str,
        work_dir: str,
        chunk: AudioChunk,
        attempt: int,
        request_plan: ChunkRequestPlan,
        *,
        split_depth: int = 0,
    ) -> ChunkTranscriptionResult:
        started_at = time.perf_counter()
        result = self._transcribe_chunk_with_plan(
            audio_path,
            work_dir,
            chunk,
            attempt,
            request_plan,
        )
        pathological_repetition_detected = _has_pathological_repetition(result.text)
        if result.segments or not pathological_repetition_detected:
            return result

        repaired_text, direct_repair_passes = (
            _apply_pathological_repetition_repair_with_stats(result.text)
        )
        if (
            repaired_text != result.text
            and not _has_pathological_repetition(repaired_text)
            and _should_accept_repaired_chunk_text(
                repaired_text,
                duration_sec=result.duration_sec or chunk.duration_sec,
            )
        ):
            return ChunkTranscriptionResult(
                chunk_index=chunk.index,
                start_sec=chunk.start_sec,
                end_sec=chunk.end_sec,
                keep_start_sec=chunk.keep_start_sec,
                keep_end_sec=chunk.keep_end_sec,
                latency_sec=time.perf_counter() - started_at,
                attempt=attempt,
                text=repaired_text,
                language=result.language,
                duration_sec=result.duration_sec,
                segments=result.segments,
                output_path=result.output_path,
                quality_fallback=QualityFallbackTrace(
                    pathological_repetition_detected=pathological_repetition_detected,
                    repair_attempted=True,
                    repair_only_accepted=True,
                    split_fallback_used=False,
                    post_split_repair_applied=False,
                    split_child_count=0,
                    max_split_depth=split_depth,
                    repair_passes_applied=direct_repair_passes,
                    total_transcribe_requests=1,
                ),
            )

        final_text = result.text
        final_language = result.language
        final_segments = result.segments
        split_fallback_used = False
        split_child_count = 0
        max_split_depth = split_depth
        total_transcribe_requests = 1
        repair_passes_applied = direct_repair_passes
        post_split_repair_applied = False
        if split_depth < _QUALITY_SPLIT_MAX_DEPTH:
            planned_chunks = self._plan_quality_fallback_chunks(
                chunk,
                result.text,
                split_depth=split_depth,
            )
            if planned_chunks:
                split_fallback_used = True
                split_child_count = len(planned_chunks)
                fallback_chunks = [
                    child_chunk for child_chunk, _depth in planned_chunks
                ]
                child_results = [
                    self._transcribe_chunk_with_quality_fallback(
                        audio_path,
                        work_dir,
                        child_chunk,
                        attempt,
                        request_plan,
                        split_depth=child_depth,
                    )
                    for child_chunk, child_depth in planned_chunks
                ]
                total_transcribe_requests += sum(
                    child_result.quality_fallback.total_transcribe_requests
                    for child_result in child_results
                )
                repair_passes_applied += sum(
                    child_result.quality_fallback.repair_passes_applied
                    for child_result in child_results
                )
                max_split_depth = max(
                    split_depth + 1,
                    max(child_depth for _child_chunk, child_depth in planned_chunks),
                    max(
                        child_result.quality_fallback.max_split_depth
                        for child_result in child_results
                    ),
                )
                final_text, final_language, final_segments = merge_transcribed_chunks(
                    fallback_chunks,
                    child_results,
                )
                if not final_text:
                    final_text = result.text
                if not final_language:
                    final_language = result.language
        if split_fallback_used:
            final_text, post_split_repair_passes = (
                _apply_pathological_repetition_repair_with_stats(final_text)
            )
            repair_passes_applied += post_split_repair_passes
            post_split_repair_applied = post_split_repair_passes > 0
        else:
            final_text = repaired_text

        return ChunkTranscriptionResult(
            chunk_index=chunk.index,
            start_sec=chunk.start_sec,
            end_sec=chunk.end_sec,
            keep_start_sec=chunk.keep_start_sec,
            keep_end_sec=chunk.keep_end_sec,
            latency_sec=time.perf_counter() - started_at,
            attempt=attempt,
            text=final_text,
            language=final_language,
            duration_sec=result.duration_sec,
            segments=final_segments,
            output_path=result.output_path,
            quality_fallback=QualityFallbackTrace(
                pathological_repetition_detected=pathological_repetition_detected,
                repair_attempted=True,
                repair_only_accepted=False,
                split_fallback_used=split_fallback_used,
                post_split_repair_applied=post_split_repair_applied,
                split_child_count=split_child_count,
                max_split_depth=max_split_depth,
                repair_passes_applied=repair_passes_applied,
                total_transcribe_requests=total_transcribe_requests,
            ),
        )

    def _dispatch_capacity(self, info: "InfoResponse" | None) -> int:
        if info is None:
            return 1
        healthy_instances = sum(
            1
            for instance in info.instances
            if instance.healthy and not getattr(instance, "sleeping", False)
        )
        schedulable_slots = count_schedulable_healthy_slots(
            info,
            per_instance_parallelism_cap=self.config.per_instance_parallelism_cap,
        )
        return max(healthy_instances, schedulable_slots)

    def _effective_parallelism(
        self, info: "InfoResponse" | None, chunk_count: int
    ) -> int:
        configured = self.config.max_parallel_chunks
        if configured is not None and configured > 0:
            return min(configured, chunk_count)
        if info is None:
            return 1
        return min(max(1, self._dispatch_capacity(info)), chunk_count)

    def _executor_parallelism_limit(
        self,
        info: "InfoResponse" | None,
        chunk_count: int,
    ) -> int:
        configured = self.config.max_parallel_chunks
        if configured is not None and configured > 0:
            return min(configured, chunk_count)
        if info is None:
            return 1
        if (
            self.config.per_instance_parallelism_cap is not None
            and self.config.per_instance_parallelism_cap > 0
        ):
            healthy_instances = sum(
                1
                for instance in info.instances
                if instance.healthy and not getattr(instance, "sleeping", False)
            )
            if healthy_instances > 0:
                return min(
                    healthy_instances * self.config.per_instance_parallelism_cap,
                    chunk_count,
                )
        return min(max(1, count_available_healthy_slots(info)), chunk_count)

    def _dispatch_pending_chunks(
        self,
        *,
        executor,
        metadata: AudioMetadata,
        work_dir: Path,
        pending: list[AudioChunk],
        running: dict[Future, tuple[AudioChunk, int]],
        attempts: dict[int, int],
        request_plan: ChunkRequestPlan,
        executor_parallelism_limit: int,
        chunk_count: int,
        get_latest_info,
        diagnostics: SchedulingDiagnostics | None = None,
        initial_info: "InfoResponse" | None = None,
        initial_info_was_forced: bool = False,
    ) -> int:
        dispatched = 0
        if diagnostics is not None:
            diagnostics.dispatch_cycle_count += 1
        use_initial_info = initial_info_was_forced or initial_info is not None
        latest_info_override = initial_info
        latest_info_override_was_forced = initial_info_was_forced
        while pending:
            if use_initial_info:
                current_info = latest_info_override
                current_info_was_forced = latest_info_override_was_forced
                use_initial_info = False
                latest_info_override = None
                latest_info_override_was_forced = False
            else:
                current_info = get_latest_info()
                current_info_was_forced = False
            current_parallelism_target = min(
                executor_parallelism_limit,
                self._effective_parallelism(current_info, chunk_count),
            )
            if len(running) >= current_parallelism_target:
                break

            latest_info = current_info
            dispatch_capacity = (
                min(
                    current_parallelism_target,
                    self._dispatch_capacity(latest_info),
                )
                if latest_info is not None
                else current_parallelism_target
            )
            allowed_new = max(
                0,
                min(
                    dispatch_capacity - len(running),
                    len(pending),
                ),
            )
            if current_info_was_forced and diagnostics is not None and allowed_new > 0:
                diagnostics.forced_machine_refresh_hit_count += 1
            if (
                allowed_new == 0
                and latest_info is not None
                and len(running) < current_parallelism_target
            ):
                if diagnostics is not None:
                    diagnostics.forced_machine_refresh_count += 1
                latest_info = get_latest_info(force=True)
                dispatch_capacity = min(
                    current_parallelism_target,
                    self._dispatch_capacity(latest_info),
                )
                allowed_new = max(
                    0,
                    min(
                        dispatch_capacity - len(running),
                        len(pending),
                    ),
                )
                if diagnostics is not None and allowed_new > 0:
                    diagnostics.forced_machine_refresh_hit_count += 1
            if allowed_new == 0:
                if not running:
                    allowed_new = 1
                else:
                    break

            for _ in range(allowed_new):
                chunk = pending.pop(0)
                attempts[chunk.index] += 1
                future = executor.submit(
                    self._transcribe_chunk,
                    metadata.path,
                    str(work_dir),
                    chunk,
                    attempts[chunk.index],
                    request_plan,
                )
                running[future] = (chunk, attempts[chunk.index])
                dispatched += 1
        return dispatched

    def transcribe(self, audio: str) -> LongAudioTranscriptionResult:
        metadata = probe_audio(audio)
        silence_regions = detect_silence_regions(
            metadata.path,
            min_silence_sec=self.config.min_silence_sec,
            silence_noise_db=self.config.silence_noise_db,
        )
        chunks = plan_audio_chunks(
            metadata.duration_sec,
            silence_regions=silence_regions,
            target_chunk_sec=self.config.target_chunk_sec,
            min_chunk_sec=self.config.min_chunk_sec,
            max_chunk_sec=self.config.max_chunk_sec,
            overlap_sec=self.config.overlap_sec,
            search_window_sec=self.config.search_window_sec,
        )

        temp_dir_cm = None
        if self.config.work_dir:
            work_dir = Path(self.config.work_dir).expanduser().resolve()
            work_dir.mkdir(parents=True, exist_ok=True)
        else:
            temp_dir_cm = tempfile.TemporaryDirectory(prefix="qsr-long-audio-")
            work_dir = Path(temp_dir_cm.name)

        started_at = time.perf_counter()
        try:
            info = self._machine_info()
            latest_info = info
            latest_info_refreshed_at = time.monotonic()
            scheduling_diagnostics = SchedulingDiagnostics()

            def get_latest_info(force: bool = False) -> "InfoResponse" | None:
                nonlocal latest_info, latest_info_refreshed_at
                refresh_interval = max(0.0, self.config.machine_info_refresh_sec)
                now = time.monotonic()
                if (
                    force
                    or latest_info is None
                    or refresh_interval == 0.0
                    or now - latest_info_refreshed_at >= refresh_interval
                ):
                    latest_info = self._machine_info()
                    latest_info_refreshed_at = now
                return latest_info

            effective_parallelism = self._effective_parallelism(info, len(chunks))
            executor_parallelism_limit = self._executor_parallelism_limit(
                info,
                len(chunks),
            )
            pending = sorted(
                chunks, key=lambda chunk: (-chunk.duration_sec, chunk.start_sec)
            )
            running: dict[Future, tuple[AudioChunk, int]] = {}
            results: list[ChunkTranscriptionResult] = []
            attempts = {chunk.index: 0 for chunk in chunks}
            request_plan, probe_result = self._resolve_chunk_request_plan(
                metadata.path,
                str(work_dir),
                self._select_probe_chunk(chunks),
            )
            if probe_result is not None:
                results.append(probe_result)
                attempts[probe_result.chunk_index] = probe_result.attempt
                pending = [
                    chunk
                    for chunk in pending
                    if chunk.index != probe_result.chunk_index
                ]

            with ThreadPoolExecutor(
                max_workers=max(1, executor_parallelism_limit)
            ) as executor:
                while pending or running:
                    self._dispatch_pending_chunks(
                        executor=executor,
                        metadata=metadata,
                        work_dir=work_dir,
                        pending=pending,
                        running=running,
                        attempts=attempts,
                        request_plan=request_plan,
                        executor_parallelism_limit=executor_parallelism_limit,
                        chunk_count=len(chunks),
                        get_latest_info=get_latest_info,
                        diagnostics=scheduling_diagnostics,
                    )

                    if not running:
                        scheduling_diagnostics.idle_poll_wait_count += 1
                        time.sleep(self.config.idle_poll_interval_sec)
                        continue

                    done, _pending = wait(
                        set(running.keys()),
                        timeout=self.config.idle_poll_interval_sec,
                        return_when=FIRST_COMPLETED,
                    )
                    if not done:
                        scheduling_diagnostics.wait_timeout_count += 1
                        continue

                    for future in done:
                        chunk, attempt = running.pop(future)
                        try:
                            results.append(future.result())
                        except Exception:
                            if attempt < self.config.max_chunk_retries:
                                pending.append(chunk)
                                pending.sort(
                                    key=lambda item: (
                                        -item.duration_sec,
                                        item.start_sec,
                                    )
                                )
                                continue
                            raise
                    if pending:
                        scheduling_diagnostics.completion_refill_trigger_count += 1
                        scheduling_diagnostics.forced_machine_refresh_count += 1
                        completion_dispatched = self._dispatch_pending_chunks(
                            executor=executor,
                            metadata=metadata,
                            work_dir=work_dir,
                            pending=pending,
                            running=running,
                            attempts=attempts,
                            request_plan=request_plan,
                            executor_parallelism_limit=executor_parallelism_limit,
                            chunk_count=len(chunks),
                            get_latest_info=get_latest_info,
                            diagnostics=scheduling_diagnostics,
                            initial_info=get_latest_info(force=True),
                            initial_info_was_forced=True,
                        )
                        if completion_dispatched > 0:
                            scheduling_diagnostics.completion_refill_hit_count += 1
                            scheduling_diagnostics.completion_refill_dispatched_chunk_count += (
                                completion_dispatched
                            )

            total_time_sec = time.perf_counter() - started_at
            merged_text, language, merged_segments = merge_transcribed_chunks(
                chunks, results
            )
            chunk_response_format = (
                "verbose_json" if any(result.segments for result in results) else "json"
            )
            snapshot = (
                machine_snapshot(
                    get_latest_info(),
                    per_instance_parallelism_cap=self.config.per_instance_parallelism_cap,
                )
                if get_latest_info() is not None
                else None
            )
            return LongAudioTranscriptionResult(
                audio=metadata.path,
                metadata=metadata,
                chunks=chunks,
                silence_regions=silence_regions,
                chunk_results=sorted(results, key=lambda item: item.chunk_index),
                total_time_sec=total_time_sec,
                max_parallel_chunks=effective_parallelism,
                endpoint=self.endpoint,
                machine_snapshot=snapshot,
                scheduling_diagnostics=scheduling_diagnostics,
                text=merged_text,
                language=language,
                merged_segments=merged_segments,
                chunk_response_format=chunk_response_format,
                stitching_mode=("segments" if merged_segments else "text_overlap"),
            )
        finally:
            if temp_dir_cm is not None and not self.config.keep_chunks:
                temp_dir_cm.cleanup()
