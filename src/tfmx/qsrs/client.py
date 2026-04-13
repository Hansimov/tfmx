"""QSR client helpers for ASR chat and transcription endpoints."""

import argparse
import asyncio
import base64
import mimetypes
import re
import time

import httpx
import orjson

from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from tclogger import dict_to_lines, logger, logstr
from typing import Callable, Optional
from urllib.parse import urlsplit, urlunsplit

from .compose import MACHINE_PORT as PORT, MAX_MODEL_LEN
from .long_audio import LongAudioTranscriber, LongAudioTranscriptionConfig


HOST = "localhost"
DEFAULT_MAX_TOKENS = 1024
DEFAULT_REQUEST_TIMEOUT_SEC = 120.0
_MAX_TOKENS_LIMIT_PATTERNS = (
    re.compile(r"max_total_tokens\s*=\s*(\d+)", re.IGNORECASE),
    re.compile(r"max_model_len(?:=max_total_tokens)?\s*=\s*(\d+)", re.IGNORECASE),
    re.compile(r"maximum context length is\s*(\d+)\s*tokens", re.IGNORECASE),
)
_REQUESTED_TOKENS_PATTERN = re.compile(
    r"requested\s+(\d+)\s+tokens\s*\((\d+)\s+in the messages,\s*(\d+)\s+in the completion\)",
    re.IGNORECASE,
)
_PROMPT_CHARS_PATTERN = re.compile(
    r"prompt contains\s*(\d+)\s+characters",
    re.IGNORECASE,
)
_ENDPOINT_ROUTE_SUFFIXES = (
    "/v1/chat/completions",
    "/chat/completions",
    "/v1/audio/transcriptions",
    "/audio/transcriptions",
    "/v1/models",
    "/models",
    "/health",
    "/info",
    "/metrics",
)


def _normalize_service_endpoint_root(endpoint: str) -> str:
    normalized = endpoint.rstrip("/")
    parsed = urlsplit(normalized)
    path = parsed.path or ""

    for suffix in _ENDPOINT_ROUTE_SUFFIXES:
        if path.endswith(suffix):
            path = path[: -len(suffix)]
            break
    else:
        if path.endswith("/v1"):
            path = path[: -len("/v1")]

    normalized_path = path.rstrip("/")
    return urlunsplit(
        (
            parsed.scheme,
            parsed.netloc,
            normalized_path,
            parsed.query,
            parsed.fragment,
        )
    ).rstrip("/")


def _join_endpoint_route(endpoint: str, route: str) -> str:
    return f"{endpoint}{route}"


def _extract_message_text(value: object) -> str:
    return "".join(_iter_text_parts(value))


@dataclass
class HealthResponse:
    status: str
    healthy: int
    total: int

    @classmethod
    def from_dict(cls, data: dict | str) -> "HealthResponse":
        if isinstance(data, str):
            return cls(status="healthy", healthy=1, total=1)
        return cls(
            status=data.get("status", "unknown"),
            healthy=data.get("healthy", 0),
            total=data.get("total", 0),
        )


@dataclass
class ModelInfo:
    models: list[str]

    @classmethod
    def from_dict(cls, data: dict) -> "ModelInfo":
        return cls(models=[item.get("id", "") for item in data.get("data", [])])


@dataclass
class ChatMessage:
    role: str
    content: object

    def to_dict(self) -> dict:
        return {"role": self.role, "content": self.content}

    @classmethod
    def from_dict(cls, data: dict) -> "ChatMessage":
        return cls(
            role=data.get("role", "assistant"),
            content=data.get("content", ""),
        )


@dataclass
class ChatUsage:
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0

    @classmethod
    def from_dict(cls, data: dict) -> "ChatUsage":
        return cls(
            prompt_tokens=data.get("prompt_tokens", 0),
            completion_tokens=data.get("completion_tokens", 0),
            total_tokens=data.get("total_tokens", 0),
        )


@dataclass
class ChatChoice:
    index: int
    message: ChatMessage
    finish_reason: str

    @classmethod
    def from_dict(cls, data: dict) -> "ChatChoice":
        return cls(
            index=data.get("index", 0),
            message=ChatMessage.from_dict(data.get("message", {})),
            finish_reason=data.get("finish_reason", "stop"),
        )


@dataclass
class ChatResponse:
    id: str
    model: str
    choices: list[ChatChoice]
    usage: ChatUsage
    created: int = 0

    @classmethod
    def from_dict(cls, data: dict) -> "ChatResponse":
        return cls(
            id=data.get("id", ""),
            model=data.get("model", ""),
            choices=[
                ChatChoice.from_dict(choice) for choice in data.get("choices", [])
            ],
            usage=ChatUsage.from_dict(data.get("usage", {})),
            created=data.get("created", 0),
        )

    @property
    def text(self) -> str:
        if not self.choices:
            return ""
        return _extract_message_text(self.choices[0].message.content)


@dataclass
class StreamChatResult:
    text: str = ""
    usage: ChatUsage = field(default_factory=ChatUsage)
    elapsed_sec: float = 0.0
    first_token_latency_sec: float = 0.0

    @property
    def token_per_second(self) -> float:
        if self.elapsed_sec <= 0:
            return 0.0
        token_count = self.usage.completion_tokens or self.usage.total_tokens
        return token_count / self.elapsed_sec if token_count > 0 else 0.0


@dataclass
class InstanceSchedulerInfo:
    score: float | None = None
    recent_requests: int = 0
    recent_successes: int = 0
    recent_failures: int = 0
    success_rate: float | None = None
    latency_ema_ms: float | None = None
    ttft_ema_ms: float | None = None
    tokens_per_second_ema: float | None = None
    cooldown_remaining_sec: float = 0.0
    consecutive_failures: int = 0
    score_components: dict[str, float] = field(default_factory=dict)
    last_error: str = ""

    @classmethod
    def from_dict(cls, data: dict) -> "InstanceSchedulerInfo":
        return cls(
            score=data.get("score"),
            recent_requests=data.get("recent_requests", 0),
            recent_successes=data.get("recent_successes", 0),
            recent_failures=data.get("recent_failures", 0),
            success_rate=data.get("success_rate"),
            latency_ema_ms=data.get("latency_ema_ms"),
            ttft_ema_ms=data.get("ttft_ema_ms"),
            tokens_per_second_ema=data.get("tokens_per_second_ema"),
            cooldown_remaining_sec=data.get("cooldown_remaining_sec", 0.0),
            consecutive_failures=data.get("consecutive_failures", 0),
            score_components=data.get("score_components", {}),
            last_error=data.get("last_error", ""),
        )


@dataclass
class InstanceInfo:
    name: str
    endpoint: str
    gpu_id: Optional[int]
    healthy: bool
    sleeping: bool = False
    model_name: str = ""
    model_label: str = ""
    active_requests: int = 0
    available_slots: int = 0
    gpu_utilization_pct: float | None = None
    gpu_memory_used_mib: float | None = None
    gpu_memory_total_mib: float | None = None
    routing_pressure: float | None = None
    scheduler: InstanceSchedulerInfo = field(default_factory=InstanceSchedulerInfo)

    @classmethod
    def from_dict(cls, data: dict) -> "InstanceInfo":
        return cls(
            name=data.get("name", ""),
            endpoint=data.get("endpoint", ""),
            gpu_id=data.get("gpu_id"),
            healthy=data.get("healthy", False),
            sleeping=data.get("sleeping", False),
            model_name=data.get("model_name", ""),
            model_label=data.get("model_label", ""),
            active_requests=data.get("active_requests", 0),
            available_slots=data.get("available_slots", 0),
            gpu_utilization_pct=data.get("gpu_utilization_pct"),
            gpu_memory_used_mib=data.get("gpu_memory_used_mib"),
            gpu_memory_total_mib=data.get("gpu_memory_total_mib"),
            routing_pressure=data.get("routing_pressure"),
            scheduler=InstanceSchedulerInfo.from_dict(data.get("scheduler", {})),
        )


@dataclass
class MachineStats:
    total_requests: int = 0
    total_tokens: int = 0
    total_errors: int = 0
    total_failovers: int = 0
    active_requests: int = 0
    requests_per_instance: dict[str, int] = field(default_factory=dict)
    total_wait_events: int = 0
    avg_wait_time_ms: float | None = None
    max_wait_time_ms: float | None = None

    @classmethod
    def from_dict(cls, data: dict) -> "MachineStats":
        return cls(
            total_requests=data.get("total_requests", 0),
            total_tokens=data.get("total_tokens", 0),
            total_errors=data.get("total_errors", 0),
            total_failovers=data.get("total_failovers", 0),
            active_requests=data.get("active_requests", 0),
            requests_per_instance=data.get("requests_per_instance", {}),
            total_wait_events=data.get("total_wait_events", 0),
            avg_wait_time_ms=data.get("avg_wait_time_ms"),
            max_wait_time_ms=data.get("max_wait_time_ms"),
        )


@dataclass
class SchedulerTuningInfo:
    enabled: bool = False
    target_latency_ms: float = 0.0
    target_ttft_ms: float = 0.0
    hot_gpu_pressure: float = 0.0
    saturation_active_ratio: float = 0.0
    throughput_cv_threshold: float = 0.0
    failure_rate_threshold: float = 0.0
    request_cv_threshold: float = 0.0
    signals: dict[str, float] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, data: dict) -> "SchedulerTuningInfo":
        return cls(
            enabled=data.get("enabled", False),
            target_latency_ms=data.get("target_latency_ms", 0.0),
            target_ttft_ms=data.get("target_ttft_ms", 0.0),
            hot_gpu_pressure=data.get("hot_gpu_pressure", 0.0),
            saturation_active_ratio=data.get("saturation_active_ratio", 0.0),
            throughput_cv_threshold=data.get("throughput_cv_threshold", 0.0),
            failure_rate_threshold=data.get("failure_rate_threshold", 0.0),
            request_cv_threshold=data.get("request_cv_threshold", 0.0),
            signals=data.get("signals", {}),
        )

    def to_dict(self) -> dict:
        return {
            "enabled": self.enabled,
            "target_latency_ms": self.target_latency_ms,
            "target_ttft_ms": self.target_ttft_ms,
            "hot_gpu_pressure": self.hot_gpu_pressure,
            "saturation_active_ratio": self.saturation_active_ratio,
            "throughput_cv_threshold": self.throughput_cv_threshold,
            "failure_rate_threshold": self.failure_rate_threshold,
            "request_cv_threshold": self.request_cv_threshold,
            "signals": self.signals,
        }


@dataclass
class SchedulerInfo:
    algorithm: str = ""
    recent_window_sec: float = 0.0
    acquire_timeout_sec: float = 0.0
    last_health_refresh_age_sec: float | None = None
    last_gpu_refresh_age_sec: float | None = None
    weights: dict[str, float] = field(default_factory=dict)
    base_weights: dict[str, float] = field(default_factory=dict)
    tuning: SchedulerTuningInfo = field(default_factory=SchedulerTuningInfo)

    @classmethod
    def from_dict(cls, data: dict) -> "SchedulerInfo":
        return cls(
            algorithm=data.get("algorithm", ""),
            recent_window_sec=data.get("recent_window_sec", 0.0),
            acquire_timeout_sec=data.get("acquire_timeout_sec", 0.0),
            last_health_refresh_age_sec=data.get("last_health_refresh_age_sec"),
            last_gpu_refresh_age_sec=data.get("last_gpu_refresh_age_sec"),
            weights=data.get("weights", {}),
            base_weights=data.get("base_weights", {}),
            tuning=SchedulerTuningInfo.from_dict(data.get("tuning", {})),
        )

    def to_dict(self) -> dict:
        return {
            "algorithm": self.algorithm,
            "recent_window_sec": self.recent_window_sec,
            "acquire_timeout_sec": self.acquire_timeout_sec,
            "last_health_refresh_age_sec": self.last_health_refresh_age_sec,
            "last_gpu_refresh_age_sec": self.last_gpu_refresh_age_sec,
            "weights": self.weights,
            "base_weights": self.base_weights,
            "tuning": self.tuning.to_dict(),
        }


@dataclass
class InfoResponse:
    port: int
    instances: list[InstanceInfo] = field(default_factory=list)
    stats: MachineStats = field(default_factory=MachineStats)
    available_models: list[str] = field(default_factory=list)
    scheduler: SchedulerInfo = field(default_factory=SchedulerInfo)

    @classmethod
    def from_dict(cls, data: dict) -> "InfoResponse":
        return cls(
            port=data.get("port", 0),
            instances=[
                InstanceInfo.from_dict(item) for item in data.get("instances", [])
            ],
            stats=MachineStats.from_dict(data.get("stats", {})),
            available_models=data.get("available_models", []),
            scheduler=SchedulerInfo.from_dict(data.get("scheduler", {})),
        )


@dataclass
class TranscriptionResponse:
    text: str = ""
    language: str = ""
    duration: float | None = None
    segments: list[dict] = field(default_factory=list)
    raw: object = None

    @classmethod
    def from_dict(cls, data: dict | str) -> "TranscriptionResponse":
        if isinstance(data, str):
            return cls(text=data, raw=data)
        return cls(
            text=str(data.get("text", "")),
            language=str(data.get("language", "")),
            duration=data.get("duration"),
            segments=data.get("segments", []) or [],
            raw=data,
        )

    def to_dict(self) -> dict:
        if isinstance(self.raw, dict):
            return self.raw
        return {
            "text": self.text,
            "language": self.language,
            "duration": self.duration,
            "segments": self.segments,
        }


@dataclass
class _StreamRenderState:
    started_at: float
    on_text: Callable[[str], None] | None = None
    text_fragments: list[str] = field(default_factory=list)
    first_token_latency_sec: float = 0.0

    def emit_chunk(self, chunk: dict) -> None:
        fragment_text = _extract_stream_text(chunk)
        if not fragment_text:
            return
        if self.first_token_latency_sec <= 0 and fragment_text.strip():
            self.first_token_latency_sec = time.perf_counter() - self.started_at
        self.text_fragments.append(fragment_text)
        if self.on_text:
            self.on_text(fragment_text)


def build_text_messages(
    prompt: str,
    system_prompt: str | None = None,
) -> list[dict]:
    messages: list[dict] = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": prompt})
    return messages


def join_prompt_texts(texts: list[str] | tuple[str, ...] | str) -> str:
    if isinstance(texts, str):
        return texts.strip()
    return "\n\n".join(part.strip() for part in texts if part and part.strip())


def _encode_audio_to_data_url(audio_path: str) -> str:
    path = Path(audio_path)
    if not path.exists():
        raise FileNotFoundError(f"Audio not found: {audio_path}")
    return _encode_local_audio_data_url_cached(*_get_local_audio_cache_identity(path))


def normalize_audio_url(audio: str) -> str:
    if audio.startswith(("http://", "https://", "data:")):
        return audio
    return _encode_audio_to_data_url(audio)


def build_audio_messages(
    texts: list[str],
    audios: list[str] | None = None,
    system_prompt: str | None = None,
) -> list[dict]:
    normalized_texts = [text.strip() for text in texts if text and text.strip()]
    normalized_audios = [normalize_audio_url(audio) for audio in audios or []]

    if not normalized_audios:
        return build_text_messages(join_prompt_texts(normalized_texts), system_prompt)

    content_parts: list[dict] = []
    total_parts = max(len(normalized_texts), len(normalized_audios))
    for index in range(total_parts):
        if index < len(normalized_texts):
            content_parts.append({"type": "text", "text": normalized_texts[index]})
        if index < len(normalized_audios):
            content_parts.append(
                {
                    "type": "audio_url",
                    "audio_url": {"url": normalized_audios[index]},
                }
            )

    messages: list[dict] = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": content_parts})
    return messages


def _build_chat_payload(
    messages: list[dict],
    model: str = "",
    max_tokens: int = DEFAULT_MAX_TOKENS,
    temperature: float = 0.0,
    top_p: float = 1.0,
    stream: bool = False,
) -> dict:
    payload = {
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "top_p": top_p,
        "stream": stream,
    }
    if model:
        payload["model"] = model
    return payload


def format_elapsed_time(elapsed_sec: float) -> str:
    total_seconds = max(0.0, elapsed_sec)
    minutes = int(total_seconds // 60)
    seconds = total_seconds - minutes * 60
    if minutes > 0:
        return f"{minutes}min {seconds:.1f}s"
    return f"{seconds:.1f}s"


def format_stream_stats_line(result: StreamChatResult) -> str:
    first_token_value = (
        format_elapsed_time(result.first_token_latency_sec)
        if result.first_token_latency_sec > 0
        else "n/a"
    )
    total_value = format_elapsed_time(result.elapsed_sec)
    first_token_text = (
        logstr.mesg(f"首 {first_token_value}")
        if result.first_token_latency_sec > 0
        else logstr.warn(f"首 {first_token_value}")
    )
    total_text = logstr.okay(f"总 {total_value}")
    token_count = result.usage.completion_tokens or result.usage.total_tokens
    token_text = logstr.mesg(f"{token_count} tokens")
    throughput_value = f"{result.token_per_second:.1f} token/s"
    throughput_text = (
        logstr.okay(throughput_value)
        if result.token_per_second > 0
        else logstr.mesg(throughput_value)
    )
    return (
        f"{logstr.mesg('[统计]:')} {first_token_text} | "
        f"{total_text} | {token_text} | {throughput_text}"
    )


def _iter_text_parts(content: object) -> list[str]:
    if isinstance(content, str):
        return [content]
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict) and item.get("type") == "text":
                text = item.get("text")
                if isinstance(text, str):
                    parts.append(text)
        return parts
    if isinstance(content, dict):
        text = content.get("text")
        if isinstance(text, str):
            return [text]
    return []


def _extract_stream_text(chunk: dict) -> str:
    fragments: list[str] = []
    for choice in chunk.get("choices", []):
        delta = choice.get("delta") or choice.get("message") or {}
        fragments.extend(_iter_text_parts(delta.get("content")))
    return "".join(fragments)


def _normalize_error_detail(detail: object) -> str:
    if isinstance(detail, dict):
        if "error" in detail:
            return _normalize_error_detail(detail["error"])
        if "message" in detail:
            return _normalize_error_detail(detail["message"])
        if "detail" in detail:
            return _normalize_error_detail(detail["detail"])
        try:
            return orjson.dumps(detail).decode("utf-8")
        except Exception:
            return str(detail)

    if isinstance(detail, bytes):
        detail = detail.decode("utf-8", errors="replace")

    if isinstance(detail, str):
        stripped = detail.strip()
        if stripped.startswith("{") and stripped.endswith("}"):
            try:
                payload = orjson.loads(stripped)
                return _normalize_error_detail(payload)
            except Exception:
                pass
        return detail

    return str(detail)


def _get_retry_max_tokens(detail: str, current_max_tokens: int) -> int | None:
    if current_max_tokens <= 1:
        return None

    max_total_tokens: int | None = None
    for pattern in _MAX_TOKENS_LIMIT_PATTERNS:
        match = pattern.search(detail)
        if match:
            max_total_tokens = int(match.group(1))
            break

    requested_match = _REQUESTED_TOKENS_PATTERN.search(detail)
    if requested_match and max_total_tokens is not None:
        prompt_tokens = int(requested_match.group(2))
        capped = max(1, max_total_tokens - prompt_tokens)
        if capped < current_max_tokens:
            return capped

    prompt_chars_match = _PROMPT_CHARS_PATTERN.search(detail)
    if (
        max_total_tokens is not None
        and prompt_chars_match
        and "output tokens" in detail.lower()
    ):
        max_context_tokens = max_total_tokens
        prompt_chars = int(prompt_chars_match.group(1))
        capped = max(1, max_context_tokens - prompt_chars)
        if capped < current_max_tokens:
            return capped

    if max_total_tokens is not None and max_total_tokens < current_max_tokens:
        return max_total_tokens

    return None


def _guess_upload_name_and_type(
    source: str, content_type: str | None = None
) -> tuple[str, str]:
    parsed = urlsplit(source)
    filename = Path(parsed.path).name if parsed.path else "audio.wav"
    if not filename:
        filename = "audio.wav"
    mime_type = content_type or mimetypes.guess_type(filename)[0] or "audio/wav"
    return filename, mime_type


def _get_local_audio_cache_identity(path: Path) -> tuple[str, int, int]:
    resolved_path = path.resolve()
    stat = resolved_path.stat()
    return str(resolved_path), stat.st_mtime_ns, stat.st_size


@lru_cache(maxsize=128)
def _read_local_audio_upload_cached(
    resolved_path: str,
    modified_time_ns: int,
    file_size: int,
) -> tuple[str, bytes, str]:
    del modified_time_ns
    del file_size
    path = Path(resolved_path)
    mime_type = mimetypes.guess_type(path.name)[0] or "audio/wav"
    return path.name, path.read_bytes(), mime_type


@lru_cache(maxsize=128)
def _encode_local_audio_data_url_cached(
    resolved_path: str,
    modified_time_ns: int,
    file_size: int,
) -> str:
    del modified_time_ns
    del file_size
    path = Path(resolved_path)
    mime_type = mimetypes.guess_type(path.name)[0] or "audio/wav"
    data = base64.b64encode(path.read_bytes()).decode("utf-8")
    return f"data:{mime_type};base64,{data}"


@lru_cache(maxsize=64)
def _download_audio_upload_cached(source: str) -> tuple[str, bytes, str]:
    response = httpx.get(source, timeout=httpx.Timeout(120.0), follow_redirects=True)
    response.raise_for_status()
    filename, mime_type = _guess_upload_name_and_type(
        source,
        response.headers.get("content-type"),
    )
    return filename, response.content, mime_type


def _load_audio_upload(source: str) -> tuple[str, bytes, str]:
    if source.startswith("data:"):
        match = re.match(r"^data:([^;]+);base64,(.*)$", source, re.DOTALL)
        if not match:
            raise ValueError("Unsupported audio data URL")
        mime_type = match.group(1)
        payload = base64.b64decode(match.group(2))
        filename = f"upload.{mime_type.split('/')[-1] or 'bin'}"
        return filename, payload, mime_type

    if source.startswith(("http://", "https://")):
        return _download_audio_upload_cached(source)

    path = Path(source)
    if not path.exists():
        raise FileNotFoundError(f"Audio not found: {source}")
    return _read_local_audio_upload_cached(*_get_local_audio_cache_identity(path))


def _build_transcription_multipart_fields(
    *,
    filename: str,
    payload: bytes,
    mime_type: str,
    model: str = "",
    language: str | None = None,
    prompt: str | None = None,
    response_format: str = "json",
    temperature: float | None = None,
    timestamp_granularities: list[str] | None = None,
    extra_fields: dict[str, object] | None = None,
) -> list[tuple[str, object]]:
    files: list[tuple[str, object]] = []

    def add_field(name: str, value: object) -> None:
        if value is None:
            return
        files.append((name, (None, str(value))))

    if model:
        add_field("model", model)
    if language:
        add_field("language", language)
    if prompt:
        add_field("prompt", prompt)
    if response_format:
        add_field("response_format", response_format)
    if temperature is not None:
        add_field("temperature", temperature)
    for item in timestamp_granularities or []:
        add_field("timestamp_granularities[]", item)
    for key, value in (extra_fields or {}).items():
        if isinstance(value, list):
            for item in value:
                add_field(key, item)
            continue
        add_field(key, value)

    files.append(("file", (filename, payload, mime_type)))
    return files


def _dump_json(payload: object) -> str:
    return orjson.dumps(payload, option=orjson.OPT_INDENT_2).decode("utf-8")


class QSRClient:
    def __init__(
        self,
        endpoint: str | None = None,
        host: str = HOST,
        port: int = PORT,
        verbose: bool = False,
        timeout_sec: float = DEFAULT_REQUEST_TIMEOUT_SEC,
    ):
        raw_endpoint = endpoint.rstrip("/") if endpoint else f"http://{host}:{port}"
        self.endpoint = _normalize_service_endpoint_root(raw_endpoint)
        self.chat_endpoint = _join_endpoint_route(self.endpoint, "/v1/chat/completions")
        self.chat_alias_endpoint = _join_endpoint_route(
            self.endpoint, "/chat/completions"
        )
        self.transcriptions_endpoint = _join_endpoint_route(
            self.endpoint, "/v1/audio/transcriptions"
        )
        self.transcriptions_alias_endpoint = _join_endpoint_route(
            self.endpoint, "/audio/transcriptions"
        )
        self.models_endpoint = _join_endpoint_route(self.endpoint, "/v1/models")
        self.models_alias_endpoint = _join_endpoint_route(self.endpoint, "/models")
        self.health_endpoint = _join_endpoint_route(self.endpoint, "/health")
        self.info_endpoint = _join_endpoint_route(self.endpoint, "/info")
        self.verbose = verbose
        self.timeout_sec = float(timeout_sec)
        self.client = httpx.Client(timeout=httpx.Timeout(self.timeout_sec))
        self._cached_models: list[str] | None = None
        self._cached_default_model: str = ""

    def close(self) -> None:
        if self.client is not None:
            self.client.close()
            self.client = None

    def __enter__(self) -> "QSRClient":
        return self

    def __exit__(self, *args) -> None:
        self.close()

    def _log_fail(self, action: str, error: Exception | str) -> None:
        if self.verbose:
            logger.warn(f"× QSR {action} error: {error}")

    def _log_okay(self, action: str, message: str) -> None:
        if self.verbose:
            logger.okay(f"✓ QSR {action}: {message}")

    def _extract_error_detail(self, exc: httpx.HTTPStatusError) -> str:
        try:
            payload = exc.response.json()
            if "error" in payload:
                return _normalize_error_detail(payload["error"])
            if "detail" in payload:
                return _normalize_error_detail(payload["detail"])
            return _normalize_error_detail(payload)
        except Exception:
            return str(exc)

    def _maybe_retry_with_capped_max_tokens(
        self,
        payload: dict,
        detail: str,
        action: str,
        attempt: int,
    ) -> bool:
        retry_max_tokens = _get_retry_max_tokens(
            detail, int(payload.get("max_tokens", 0) or 0)
        )
        if retry_max_tokens is None or attempt >= 2:
            return False
        self._log_fail(
            action,
            (
                f"requested max_tokens={payload.get('max_tokens')} exceeds backend limit; "
                f"retrying with max_tokens={retry_max_tokens}"
            ),
        )
        payload["max_tokens"] = retry_max_tokens
        return True

    def health(self) -> HealthResponse:
        try:
            response = self.client.get(self.health_endpoint)
            response.raise_for_status()
            try:
                payload = response.json()
            except Exception:
                payload = {"status": "healthy", "healthy": 1, "total": 1}
            result = HealthResponse.from_dict(payload)
            self._log_okay("health", f"status={result.status}")
            return result
        except httpx.HTTPStatusError as exc:
            try:
                payload = exc.response.json()
                if "detail" in payload and isinstance(payload["detail"], dict):
                    return HealthResponse.from_dict(payload["detail"])
            except Exception:
                pass
            self._log_fail("health", exc)
            raise
        except Exception as exc:
            self._log_fail("health", exc)
            raise

    def log_machine_health(self) -> None:
        health = self.health()
        logger.mesg(f"* Healthy: {logstr.okay(health.healthy)}/{health.total}")

    def is_healthy(self) -> bool:
        try:
            health = self.health()
            return health.status == "healthy" or health.healthy > 0
        except Exception:
            return False

    def _cache_models(self, models: list[str]) -> None:
        self._cached_models = list(models)
        if models:
            self._cached_default_model = models[0]

    def get_default_model(self) -> str:
        return self._resolve_model("")

    def _resolve_model(self, model: str = "") -> str:
        explicit_model = (model or "").strip()
        if explicit_model:
            return explicit_model
        if self._cached_default_model:
            return self._cached_default_model
        try:
            result = self.models()
        except Exception:
            return ""
        if result.models:
            self._cached_default_model = result.models[0]
            return self._cached_default_model
        return ""

    def models(self) -> ModelInfo:
        last_exc: Exception | None = None
        candidate_urls = [self.models_endpoint, self.models_alias_endpoint]
        seen_urls: set[str] = set()
        for url in candidate_urls:
            if url in seen_urls:
                continue
            seen_urls.add(url)
            try:
                response = self.client.get(url)
                response.raise_for_status()
                result = ModelInfo.from_dict(response.json())
                self._cache_models(result.models)
                self._log_okay("models", f"count={len(result.models)}")
                return result
            except httpx.HTTPStatusError as exc:
                last_exc = exc
                if exc.response.status_code in (404, 405):
                    continue
                self._log_fail("models", exc)
                raise
            except Exception as exc:
                last_exc = exc
                self._log_fail("models", exc)
                raise

        if last_exc is not None:
            self._log_fail("models", last_exc)
            raise last_exc
        raise RuntimeError("No models endpoint available")

    def info(self) -> InfoResponse:
        try:
            response = self.client.get(self.info_endpoint)
            response.raise_for_status()
            result = InfoResponse.from_dict(response.json())
            self._log_okay("info", f"instances={len(result.instances)}")
            return result
        except Exception as exc:
            self._log_fail("info", exc)
            raise

    def chat(
        self,
        messages: list[dict],
        model: str = "",
        max_tokens: int = DEFAULT_MAX_TOKENS,
        temperature: float = 0.0,
        top_p: float = 1.0,
        stream: bool = False,
    ) -> ChatResponse:
        resolved_model = self._resolve_model(model)
        payload = _build_chat_payload(
            messages=messages,
            model=resolved_model,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            stream=stream,
        )

        for attempt in range(3):
            try:
                response = self.client.post(self.chat_endpoint, json=payload)
                response.raise_for_status()
                result = ChatResponse.from_dict(response.json())
                self._log_okay("chat", f"tokens={result.usage.total_tokens}")
                return result
            except httpx.HTTPStatusError as exc:
                detail = self._extract_error_detail(exc)
                if self._maybe_retry_with_capped_max_tokens(
                    payload, detail, action="chat", attempt=attempt
                ):
                    continue
                self._log_fail("chat", detail)
                raise ValueError(f"Chat failed: {detail}") from exc
            except Exception as exc:
                self._log_fail("chat", exc)
                raise

        raise RuntimeError("unreachable")

    def stream_chat(
        self,
        messages: list[dict],
        model: str = "",
        max_tokens: int = DEFAULT_MAX_TOKENS,
        temperature: float = 0.0,
        top_p: float = 1.0,
        on_text: Callable[[str], None] | None = None,
    ) -> StreamChatResult:
        resolved_model = self._resolve_model(model)
        payload = _build_chat_payload(
            messages=messages,
            model=resolved_model,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            stream=True,
        )
        payload["stream_options"] = {"include_usage": True}
        for attempt in range(3):
            text_fragments: list[str] = []
            usage = ChatUsage()
            started_at = time.perf_counter()
            render_state = _StreamRenderState(
                started_at=started_at,
                on_text=on_text,
                text_fragments=text_fragments,
            )
            try:
                with self.client.stream(
                    "POST", self.chat_endpoint, json=payload
                ) as response:
                    response.raise_for_status()
                    for line in response.iter_lines():
                        if not line:
                            continue
                        if isinstance(line, bytes):
                            line = line.decode("utf-8", errors="replace")
                        if not line.startswith("data: "):
                            continue
                        data = line[6:].strip()
                        if data == "[DONE]":
                            break
                        chunk = orjson.loads(data)
                        render_state.emit_chunk(chunk)
                        chunk_usage = chunk.get("usage")
                        if chunk_usage:
                            usage = ChatUsage.from_dict(chunk_usage)
                result = StreamChatResult(
                    text="".join(text_fragments),
                    usage=usage,
                    elapsed_sec=time.perf_counter() - started_at,
                    first_token_latency_sec=render_state.first_token_latency_sec,
                )
                self._log_okay("stream_chat", f"tokens={result.usage.total_tokens}")
                return result
            except httpx.HTTPStatusError as exc:
                detail = self._extract_error_detail(exc)
                if self._maybe_retry_with_capped_max_tokens(
                    payload, detail, action="stream_chat", attempt=attempt
                ):
                    continue
                self._log_fail("stream_chat", detail)
                raise ValueError(f"Stream chat failed: {detail}") from exc
            except Exception as exc:
                self._log_fail("stream_chat", exc)
                raise

        raise RuntimeError("unreachable")

    def transcribe(
        self,
        audio: str,
        model: str = "",
        language: str | None = None,
        prompt: str | None = None,
        response_format: str = "json",
        temperature: float | None = None,
        timestamp_granularities: list[str] | None = None,
        extra_form_fields: dict[str, object] | None = None,
    ) -> TranscriptionResponse:
        resolved_model = self._resolve_model(model)
        filename, payload, mime_type = _load_audio_upload(audio)
        response = self.client.post(
            self.transcriptions_endpoint,
            files=_build_transcription_multipart_fields(
                filename=filename,
                payload=payload,
                mime_type=mime_type,
                model=resolved_model,
                language=language,
                prompt=prompt,
                response_format=response_format,
                temperature=temperature,
                timestamp_granularities=timestamp_granularities,
                extra_fields=extra_form_fields,
            ),
        )
        response.raise_for_status()
        content_type = response.headers.get("content-type", "").lower()
        if "application/json" in content_type:
            result = TranscriptionResponse.from_dict(response.json())
        else:
            result = TranscriptionResponse.from_dict(response.text)
        self._log_okay("transcribe", f"chars={len(result.text)}")
        return result


class AsyncQSRClient:
    def __init__(
        self,
        endpoint: str | None = None,
        host: str = HOST,
        port: int = PORT,
        verbose: bool = False,
        timeout_sec: float = DEFAULT_REQUEST_TIMEOUT_SEC,
    ):
        raw_endpoint = endpoint.rstrip("/") if endpoint else f"http://{host}:{port}"
        self.endpoint = _normalize_service_endpoint_root(raw_endpoint)
        self.chat_endpoint = _join_endpoint_route(self.endpoint, "/v1/chat/completions")
        self.transcriptions_endpoint = _join_endpoint_route(
            self.endpoint, "/v1/audio/transcriptions"
        )
        self.health_endpoint = _join_endpoint_route(self.endpoint, "/health")
        self.models_endpoint = _join_endpoint_route(self.endpoint, "/v1/models")
        self.models_alias_endpoint = _join_endpoint_route(self.endpoint, "/models")
        self.verbose = verbose
        self.timeout_sec = float(timeout_sec)
        self.client = httpx.AsyncClient(timeout=httpx.Timeout(self.timeout_sec))
        self._cached_default_model: str = ""
        self._model_resolve_lock: asyncio.Lock | None = None

    def reset(self) -> None:
        return

    async def close(self) -> None:
        if self.client is not None:
            await self.client.aclose()
            self.client = None

    async def health(self) -> HealthResponse:
        response = await self.client.get(self.health_endpoint)
        response.raise_for_status()
        try:
            payload = response.json()
        except Exception:
            payload = {"status": "healthy", "healthy": 1, "total": 1}
        return HealthResponse.from_dict(payload)

    async def models(self) -> ModelInfo:
        for url in (self.models_endpoint, self.models_alias_endpoint):
            response = await self.client.get(url)
            if response.status_code in (404, 405):
                continue
            response.raise_for_status()
            result = ModelInfo.from_dict(response.json())
            if result.models:
                self._cached_default_model = result.models[0]
            return result
        raise RuntimeError("No models endpoint available")

    def _get_model_resolve_lock(self) -> asyncio.Lock:
        if self._model_resolve_lock is None:
            self._model_resolve_lock = asyncio.Lock()
        return self._model_resolve_lock

    async def _resolve_model(self, model: str = "") -> str:
        explicit_model = (model or "").strip()
        if explicit_model:
            return explicit_model
        if self._cached_default_model:
            return self._cached_default_model

        async with self._get_model_resolve_lock():
            if self._cached_default_model:
                return self._cached_default_model
            result = await self.models()
            if result.models:
                self._cached_default_model = result.models[0]
            return self._cached_default_model

    async def chat(
        self,
        messages: list[dict],
        model: str = "",
        max_tokens: int = DEFAULT_MAX_TOKENS,
        temperature: float = 0.0,
        top_p: float = 1.0,
    ) -> ChatResponse:
        resolved_model = await self._resolve_model(model)
        payload = _build_chat_payload(
            messages=messages,
            model=resolved_model,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
        )
        response = await self.client.post(self.chat_endpoint, json=payload)
        response.raise_for_status()
        return ChatResponse.from_dict(response.json())

    async def stream_chat(
        self,
        messages: list[dict],
        model: str = "",
        max_tokens: int = DEFAULT_MAX_TOKENS,
        temperature: float = 0.0,
        top_p: float = 1.0,
    ) -> StreamChatResult:
        resolved_model = await self._resolve_model(model)
        payload = _build_chat_payload(
            messages=messages,
            model=resolved_model,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            stream=True,
        )
        payload["stream_options"] = {"include_usage": True}
        text_fragments: list[str] = []
        usage = ChatUsage()
        started_at = time.perf_counter()
        render_state = _StreamRenderState(
            started_at=started_at,
            text_fragments=text_fragments,
        )
        async with self.client.stream(
            "POST", self.chat_endpoint, json=payload
        ) as response:
            response.raise_for_status()
            async for line in response.aiter_lines():
                if not line or not line.startswith("data: "):
                    continue
                data = line[6:].strip()
                if data == "[DONE]":
                    break
                chunk = orjson.loads(data)
                render_state.emit_chunk(chunk)
                chunk_usage = chunk.get("usage")
                if chunk_usage:
                    usage = ChatUsage.from_dict(chunk_usage)
        return StreamChatResult(
            text="".join(text_fragments),
            usage=usage,
            elapsed_sec=time.perf_counter() - started_at,
            first_token_latency_sec=render_state.first_token_latency_sec,
        )

    async def transcribe(
        self,
        audio: str,
        model: str = "",
        language: str | None = None,
        prompt: str | None = None,
        response_format: str = "json",
        temperature: float | None = None,
        timestamp_granularities: list[str] | None = None,
        extra_form_fields: dict[str, object] | None = None,
    ) -> TranscriptionResponse:
        resolved_model = await self._resolve_model(model)
        filename, payload, mime_type = await asyncio.to_thread(
            _load_audio_upload, audio
        )
        response = await self.client.post(
            self.transcriptions_endpoint,
            files=_build_transcription_multipart_fields(
                filename=filename,
                payload=payload,
                mime_type=mime_type,
                model=resolved_model,
                language=language,
                prompt=prompt,
                response_format=response_format,
                temperature=temperature,
                timestamp_granularities=timestamp_granularities,
                extra_fields=extra_form_fields,
            ),
        )
        response.raise_for_status()
        content_type = response.headers.get("content-type", "").lower()
        if "application/json" in content_type:
            return TranscriptionResponse.from_dict(response.json())
        return TranscriptionResponse.from_dict(response.text)


CLI_EPILOG = """
Examples:
  qsr client health
  qsr client models
  qsr client info
  qsr client transcribe ./sample.wav
  qsr client transcribe ./meeting.mp3 --long-audio-mode auto --json
    qsr client transcribe-long ./meeting.mp3 --json
  qsr client transcribe https://example.com/sample.wav --response-format text
  qsr client chat --audio ./sample.wav "请转写为简体中文"
  qsr client chat --audio ./sample.wav --no-stream --json
"""


def _add_long_audio_args(
    parser: argparse.ArgumentParser,
    *,
    include_mode: bool,
    include_output: bool,
    include_work_dir: bool,
) -> None:
    if include_mode:
        parser.add_argument(
            "--long-audio-mode",
            choices=["off", "auto", "force"],
            default="off",
            help=(
                "Offload silence-aware long-audio chunking to qsr machine; "
                "requires a machine endpoint"
            ),
        )
        parser.add_argument(
            "--long-audio-min-duration-sec",
            type=float,
            default=120.0,
            help="Minimum duration for --long-audio-mode auto before chunking is enabled",
        )

    parser.add_argument(
        "--target-chunk-sec",
        type=float,
        default=60.0,
        help="Target chunk length before silence-aware adjustment",
    )
    parser.add_argument(
        "--min-chunk-sec",
        type=float,
        default=35.0,
        help="Minimum chunk length",
    )
    parser.add_argument(
        "--max-chunk-sec",
        type=float,
        default=90.0,
        help="Maximum chunk length before forcing a cut",
    )
    parser.add_argument(
        "--overlap-sec",
        type=float,
        default=4.0,
        help="Chunk overlap in seconds to reduce transcript loss at boundaries",
    )
    parser.add_argument(
        "--search-window-sec",
        type=float,
        default=12.0,
        help="Silence search window around the target boundary",
    )
    parser.add_argument(
        "--min-silence-sec",
        type=float,
        default=0.35,
        help="Minimum silence duration used to propose chunk boundaries",
    )
    parser.add_argument(
        "--silence-noise-db",
        type=float,
        default=-32.0,
        help="Silence threshold in dB for ffmpeg silencedetect",
    )
    parser.add_argument(
        "--idle-poll-interval-sec",
        type=float,
        default=1.0,
        help="Polling interval while refreshing long-audio dispatch capacity",
    )
    parser.add_argument(
        "--max-parallel-chunks",
        type=int,
        default=None,
        help="Optional cap on total in-flight long-audio chunk requests",
    )
    parser.add_argument(
        "--per-instance-parallelism-cap",
        type=int,
        default=4,
        help="Soft cap on concurrently in-flight long-audio chunks per healthy backend instance",
    )
    parser.add_argument(
        "--max-chunk-retries",
        type=int,
        default=2,
        help="How many times to retry a failed chunk before aborting",
    )

    if include_work_dir:
        parser.add_argument(
            "--work-dir",
            type=str,
            default=None,
            help="Directory to store extracted chunk files",
        )
        parser.add_argument(
            "--keep-chunks",
            action="store_true",
            help="Keep extracted chunk files after the job completes",
        )

    if include_output:
        parser.add_argument("-o", "--output", type=str, default=None)


def _build_machine_long_audio_fields(
    args: argparse.Namespace,
) -> dict[str, object] | None:
    mode = getattr(args, "long_audio_mode", "off")
    if mode == "off":
        return None
    return {
        "long_audio_mode": mode,
        "long_audio_min_duration_sec": getattr(
            args, "long_audio_min_duration_sec", 120.0
        ),
        "target_chunk_sec": args.target_chunk_sec,
        "min_chunk_sec": args.min_chunk_sec,
        "max_chunk_sec": args.max_chunk_sec,
        "overlap_sec": args.overlap_sec,
        "search_window_sec": args.search_window_sec,
        "min_silence_sec": args.min_silence_sec,
        "silence_noise_db": args.silence_noise_db,
        "idle_poll_interval_sec": args.idle_poll_interval_sec,
        "max_parallel_chunks": args.max_parallel_chunks,
        "per_instance_parallelism_cap": args.per_instance_parallelism_cap,
        "max_chunk_retries": args.max_chunk_retries,
    }


def configure_parser(parser: argparse.ArgumentParser) -> None:
    parser.description = "Talk to a QSR machine or direct ASR backend"
    parser.formatter_class = argparse.RawDescriptionHelpFormatter
    parser.epilog = CLI_EPILOG

    subparsers = parser.add_subparsers(dest="client_action", required=True)

    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("-E", "--endpoint", type=str, default=None)
    common.add_argument("-m", "--model", type=str, default="")
    common.add_argument(
        "--timeout-sec",
        type=float,
        default=DEFAULT_REQUEST_TIMEOUT_SEC,
        help="HTTP timeout in seconds for chat/transcribe/model requests",
    )
    common.add_argument("-v", "--verbose", action="store_true")

    for action in ("health", "models", "info"):
        subparsers.add_parser(
            action, parents=[common], help=f"Show {action} information"
        )

    chat_parser = subparsers.add_parser(
        "chat", parents=[common], help="Send chat/audio request"
    )
    chat_parser.add_argument("text", nargs="*", help="Optional prompt text")
    chat_parser.add_argument("-a", "--audio", action="append", default=[])
    chat_parser.add_argument("-s", "--system-prompt", type=str, default="")
    chat_parser.add_argument("--max-tokens", type=int, default=DEFAULT_MAX_TOKENS)
    chat_parser.add_argument("--temperature", type=float, default=0.0)
    chat_parser.add_argument("--top-p", type=float, default=1.0)
    chat_parser.add_argument("--no-stream", action="store_true")
    chat_parser.add_argument("--json", action="store_true")

    transcribe_parser = subparsers.add_parser(
        "transcribe",
        parents=[common],
        help="Call OpenAI-compatible transcription endpoint",
    )
    transcribe_parser.add_argument(
        "audio", nargs="+", help="Audio file path, URL, or data URL"
    )
    transcribe_parser.add_argument("--language", type=str, default=None)
    transcribe_parser.add_argument("--prompt", type=str, default=None)
    transcribe_parser.add_argument(
        "--response-format",
        choices=["json", "text", "verbose_json", "srt", "vtt"],
        default="json",
    )
    transcribe_parser.add_argument("--temperature", type=float, default=None)
    transcribe_parser.add_argument("--timestamp-granularities", nargs="*", default=None)
    _add_long_audio_args(
        transcribe_parser,
        include_mode=True,
        include_output=False,
        include_work_dir=False,
    )
    transcribe_parser.add_argument("--json", action="store_true")

    transcribe_long_parser = subparsers.add_parser(
        "transcribe-long",
        parents=[common],
        help="Split a long audio file into silence-aware chunks and schedule them across machine capacity",
    )
    transcribe_long_parser.add_argument("audio", help="Long audio file path")
    transcribe_long_parser.add_argument("--language", type=str, default=None)
    transcribe_long_parser.add_argument("--prompt", type=str, default=None)
    _add_long_audio_args(
        transcribe_long_parser,
        include_mode=False,
        include_output=True,
        include_work_dir=True,
    )
    transcribe_long_parser.add_argument("--json", action="store_true")


def run_from_args(args: argparse.Namespace) -> None:
    with QSRClient(
        endpoint=args.endpoint,
        verbose=args.verbose,
        timeout_sec=getattr(args, "timeout_sec", DEFAULT_REQUEST_TIMEOUT_SEC),
    ) as client:
        if args.client_action == "health":
            health = client.health()
            print(
                _dump_json(
                    {
                        "status": health.status,
                        "healthy": health.healthy,
                        "total": health.total,
                    }
                )
            )
            return

        if args.client_action == "models":
            models = client.models()
            print(_dump_json({"models": models.models}))
            return

        if args.client_action == "info":
            info = client.info()
            print(
                _dump_json(
                    info.__dict__
                    | {
                        "instances": [
                            instance.__dict__
                            | {"scheduler": instance.scheduler.__dict__}
                            for instance in info.instances
                        ],
                        "stats": info.stats.__dict__,
                        "scheduler": info.scheduler.to_dict(),
                    }
                )
            )
            return

        if args.client_action == "chat":
            messages = build_audio_messages(
                texts=args.text,
                audios=args.audio,
                system_prompt=args.system_prompt or None,
            )
            if args.json or args.no_stream:
                response = client.chat(
                    messages=messages,
                    model=args.model,
                    max_tokens=args.max_tokens,
                    temperature=args.temperature,
                    top_p=args.top_p,
                )
                if args.json:
                    payload = {
                        "id": response.id,
                        "model": response.model,
                        "choices": [
                            {
                                "index": choice.index,
                                "message": choice.message.to_dict(),
                                "finish_reason": choice.finish_reason,
                            }
                            for choice in response.choices
                        ],
                        "usage": response.usage.__dict__,
                    }
                    print(_dump_json(payload))
                else:
                    print(response.text)
                    print(
                        dict_to_lines(
                            {
                                "prompt_tokens": response.usage.prompt_tokens,
                                "completion_tokens": response.usage.completion_tokens,
                                "total_tokens": response.usage.total_tokens,
                            }
                        )
                    )
                return

            result = client.stream_chat(
                messages=messages,
                model=args.model,
                max_tokens=args.max_tokens,
                temperature=args.temperature,
                top_p=args.top_p,
                on_text=lambda text: print(text, end="", flush=True),
            )
            if result.text and not result.text.endswith("\n"):
                print()
            print(format_stream_stats_line(result))
            return

        if args.client_action == "transcribe":
            extra_form_fields = _build_machine_long_audio_fields(args)
            if extra_form_fields is not None:
                try:
                    client.info()
                except Exception as exc:
                    raise ValueError(
                        "Machine-side long-audio mode requires a qsr machine endpoint; "
                        "use qsr client transcribe-long for direct backend URLs"
                    ) from exc
            multiple = len(args.audio) > 1
            for index, audio in enumerate(args.audio):
                result = client.transcribe(
                    audio=audio,
                    model=args.model,
                    language=args.language,
                    prompt=args.prompt,
                    response_format=args.response_format,
                    temperature=args.temperature,
                    timestamp_granularities=args.timestamp_granularities,
                    extra_form_fields=extra_form_fields,
                )
                if multiple:
                    print(f"===== [{index + 1}/{len(args.audio)}] {audio} =====")
                if args.json or args.response_format in {"json", "verbose_json"}:
                    print(_dump_json(result.to_dict()))
                else:
                    print(result.text)
            return

        if args.client_action == "transcribe-long":
            transcriber = LongAudioTranscriber(
                endpoint=client.endpoint,
                config=LongAudioTranscriptionConfig(
                    model=args.model,
                    language=args.language,
                    prompt=args.prompt,
                    timeout_sec=getattr(
                        args, "timeout_sec", DEFAULT_REQUEST_TIMEOUT_SEC
                    ),
                    target_chunk_sec=args.target_chunk_sec,
                    min_chunk_sec=args.min_chunk_sec,
                    max_chunk_sec=args.max_chunk_sec,
                    overlap_sec=args.overlap_sec,
                    search_window_sec=args.search_window_sec,
                    min_silence_sec=args.min_silence_sec,
                    silence_noise_db=args.silence_noise_db,
                    idle_poll_interval_sec=args.idle_poll_interval_sec,
                    max_parallel_chunks=args.max_parallel_chunks,
                    per_instance_parallelism_cap=args.per_instance_parallelism_cap,
                    max_chunk_retries=args.max_chunk_retries,
                    keep_chunks=args.keep_chunks,
                    work_dir=args.work_dir,
                ),
            )
            result = transcriber.transcribe(args.audio)
            payload = result.to_dict()
            if args.output:
                output_path = Path(args.output)
                output_path.parent.mkdir(parents=True, exist_ok=True)
                output_path.write_text(_dump_json(payload) + "\n")
            if args.json:
                print(_dump_json(payload))
            else:
                print(result.text)
            return

        raise ValueError(f"Unknown client action: {args.client_action}")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser()
    configure_parser(parser)
    args = parser.parse_args(argv)
    run_from_args(args)


if __name__ == "__main__":
    main()
