from __future__ import annotations

"""Long-audio chunk planning, scheduling, and stitching for QSR."""

import json
import math
import shutil
import subprocess
import tempfile
import time

from concurrent.futures import FIRST_COMPLETED, Future, ThreadPoolExecutor, wait
from dataclasses import dataclass, field
from difflib import SequenceMatcher
from pathlib import Path
from typing import TYPE_CHECKING, Optional


if TYPE_CHECKING:
    from .client import InfoResponse


FFMPEG_REQUIRED_MESSAGE = (
    "ffmpeg and ffprobe are required for qsr long-audio transcription. "
    "Install them first, then retry."
)
_SILENCE_START_TOKEN = "silence_start:"
_SILENCE_END_TOKEN = "silence_end:"


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
    merged_segments: list[dict] = field(default_factory=list)

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
            },
            "machine_snapshot": self.machine_snapshot,
            "transcription": {
                "text": self.text,
                "language": self.language,
                "duration": round(self.metadata.duration_sec, 3),
                "segments": self.merged_segments,
                "chunks": [result.to_dict() for result in self.chunk_results],
            },
        }


@dataclass
class LongAudioTranscriptionConfig:
    model: str = ""
    language: str | None = None
    prompt: str | None = None
    timeout_sec: float = 900.0
    target_chunk_sec: float = 60.0
    min_chunk_sec: float = 35.0
    max_chunk_sec: float = 90.0
    overlap_sec: float = 4.0
    search_window_sec: float = 12.0
    min_silence_sec: float = 0.35
    silence_noise_db: float = -32.0
    idle_poll_interval_sec: float = 1.0
    max_parallel_chunks: int | None = None
    per_instance_parallelism_cap: int | None = 3
    max_chunk_retries: int = 2
    keep_chunks: bool = False
    work_dir: str | None = None


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


def _strip_duplicate_boundary(left: str, right: str, *, max_chars: int = 64) -> str:
    left_text = left.rstrip()
    right_text = right.lstrip()
    max_overlap = min(len(left_text), len(right_text), max_chars)
    for width in range(max_overlap, 0, -1):
        if left_text[-width:] == right_text[:width]:
            return left_text + right_text[width:]
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

    def _transcribe_chunk(
        self,
        audio_path: str,
        work_dir: str,
        chunk: AudioChunk,
        attempt: int,
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
                response_format="json",
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
            segments=[],
            output_path=chunk.output_path,
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
            effective_parallelism = self._effective_parallelism(info, len(chunks))
            pending = sorted(
                chunks, key=lambda chunk: (-chunk.duration_sec, chunk.start_sec)
            )
            running: dict[Future, tuple[AudioChunk, int]] = {}
            results: list[ChunkTranscriptionResult] = []
            attempts = {chunk.index: 0 for chunk in chunks}

            with ThreadPoolExecutor(
                max_workers=max(1, effective_parallelism)
            ) as executor:
                while pending or running:
                    if pending and len(running) < effective_parallelism:
                        latest_info = self._machine_info()
                        dispatch_capacity = (
                            min(
                                effective_parallelism,
                                self._dispatch_capacity(latest_info),
                            )
                            if latest_info is not None
                            else effective_parallelism
                        )
                        allowed_new = max(
                            0,
                            min(
                                dispatch_capacity - len(running),
                                len(pending),
                            ),
                        )
                        if allowed_new == 0 and not running and pending:
                            allowed_new = 1
                        for _ in range(allowed_new):
                            chunk = pending.pop(0)
                            attempts[chunk.index] += 1
                            future = executor.submit(
                                self._transcribe_chunk,
                                metadata.path,
                                str(work_dir),
                                chunk,
                                attempts[chunk.index],
                            )
                            running[future] = (chunk, attempts[chunk.index])

                    if not running:
                        time.sleep(self.config.idle_poll_interval_sec)
                        continue

                    done, _pending = wait(
                        set(running.keys()),
                        timeout=self.config.idle_poll_interval_sec,
                        return_when=FIRST_COMPLETED,
                    )
                    if not done:
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

            total_time_sec = time.perf_counter() - started_at
            merged_text, language, merged_segments = merge_transcribed_chunks(
                chunks, results
            )
            snapshot = (
                machine_snapshot(
                    info,
                    per_instance_parallelism_cap=self.config.per_instance_parallelism_cap,
                )
                if info is not None
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
                text=merged_text,
                language=language,
                merged_segments=merged_segments,
            )
        finally:
            if temp_dir_cm is not None and not self.config.keep_chunks:
                temp_dir_cm.cleanup()
