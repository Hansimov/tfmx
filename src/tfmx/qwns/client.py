"""QWN client helpers for single-endpoint text chat services."""

import argparse
import base64
import re
import time

import httpx
import orjson

from dataclasses import dataclass, field
from pathlib import Path
from tclogger import dict_to_lines, logger, logstr
from typing import Callable, Optional

from .compose import MACHINE_PORT as PORT, MAX_MODEL_LEN


HOST = "localhost"
DEFAULT_MAX_TOKENS = MAX_MODEL_LEN
THINKING_OPEN_TAG = "<thinking>\n"
THINKING_CLOSE_TAG = "\n</thinking>\n"
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
        return self.choices[0].message.content


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
class _StreamRenderState:
    started_at: float
    on_text: Callable[[str], None] | None = None
    enable_thinking: bool = False
    include_thinking_tags: bool = False
    text_fragments: list[str] = field(default_factory=list)
    first_token_latency_sec: float = 0.0
    _thinking_tag_open: bool = False
    _saw_reasoning: bool = False

    def begin(self) -> None:
        return

    def finish(self) -> None:
        if self._thinking_tag_open:
            self._close_thinking_tag()

    def emit_chunk(self, chunk: dict) -> None:
        for fragment_kind, fragment_text in _extract_stream_fragments(
            chunk,
            include_reasoning=self.enable_thinking,
        ):
            if not fragment_text:
                continue
            if (
                self.enable_thinking
                and self.include_thinking_tags
                and not self._thinking_tag_open
            ):
                self._open_thinking_tag()
            if (
                fragment_kind == "content"
                and self._thinking_tag_open
                and self._saw_reasoning
            ):
                self._close_thinking_tag()
            if fragment_kind == "reasoning":
                self._saw_reasoning = True
                if (
                    self.enable_thinking
                    and self.include_thinking_tags
                    and not self._thinking_tag_open
                ):
                    self._open_thinking_tag()
            self._emit(fragment_text)

    def _emit(self, text: str, synthetic: bool = False) -> None:
        if not text:
            return
        self.text_fragments.append(text)
        if not synthetic and self.first_token_latency_sec <= 0 and text.strip():
            self.first_token_latency_sec = time.perf_counter() - self.started_at
        if self.on_text:
            self.on_text(text)

    def _open_thinking_tag(self) -> None:
        if self._thinking_tag_open:
            return
        self._emit(THINKING_OPEN_TAG, synthetic=True)
        self._thinking_tag_open = True

    def _close_thinking_tag(self) -> None:
        if not self._thinking_tag_open:
            return
        self._emit(THINKING_CLOSE_TAG, synthetic=True)
        self._thinking_tag_open = False


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
    model_name: str = ""
    quant_method: str = ""
    quant_level: str = ""
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
            model_name=data.get("model_name", ""),
            quant_method=data.get("quant_method", ""),
            quant_level=data.get("quant_level", ""),
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


def _encode_image_to_data_url(image_path: str) -> str:
    path = Path(image_path)
    if not path.exists():
        raise FileNotFoundError(f"Image not found: {image_path}")

    mime_map = {
        ".jpg": "image/jpeg",
        ".jpeg": "image/jpeg",
        ".png": "image/png",
        ".gif": "image/gif",
        ".webp": "image/webp",
        ".bmp": "image/bmp",
    }
    mime_type = mime_map.get(path.suffix.lower(), "image/png")
    data = base64.b64encode(path.read_bytes()).decode("utf-8")
    return f"data:{mime_type};base64,{data}"


def normalize_image_url(image: str) -> str:
    if image.startswith(("http://", "https://", "data:")):
        return image
    return _encode_image_to_data_url(image)


def build_multimodal_messages(
    texts: list[str],
    images: list[str] | None = None,
    system_prompt: str | None = None,
) -> list[dict]:
    normalized_texts = [text.strip() for text in texts if text and text.strip()]
    normalized_images = [normalize_image_url(image) for image in images or []]

    if not normalized_images:
        return build_text_messages(join_prompt_texts(normalized_texts), system_prompt)

    content_parts: list[dict] = []
    total_parts = max(len(normalized_texts), len(normalized_images))
    for index in range(total_parts):
        if index < len(normalized_texts):
            content_parts.append({"type": "text", "text": normalized_texts[index]})
        if index < len(normalized_images):
            content_parts.append(
                {
                    "type": "image_url",
                    "image_url": {"url": normalized_images[index]},
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
    temperature: float = 0.7,
    top_p: float = 0.9,
    stream: bool = False,
    enable_thinking: bool = False,
) -> dict:
    payload = {
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "top_p": top_p,
        "stream": stream,
        "chat_template_kwargs": {"enable_thinking": enable_thinking},
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


def _extract_stream_fragments(
    chunk: dict,
    include_reasoning: bool = False,
) -> list[tuple[str, str]]:
    fragments: list[str] = []
    for choice in chunk.get("choices", []):
        delta = choice.get("delta") or choice.get("message") or {}
        typed_fragments: list[tuple[str, str]] = []
        if include_reasoning:
            typed_fragments.extend(
                ("reasoning", text)
                for text in _iter_text_parts(delta.get("reasoning_content"))
            )
            typed_fragments.extend(
                ("reasoning", text) for text in _iter_text_parts(delta.get("reasoning"))
            )
        typed_fragments.extend(
            ("content", text) for text in _iter_text_parts(delta.get("content"))
        )
        fragments.extend(typed_fragments)
    return fragments


def _extract_stream_text(chunk: dict, include_reasoning: bool = False) -> str:
    return "".join(
        text
        for _, text in _extract_stream_fragments(
            chunk,
            include_reasoning=include_reasoning,
        )
    )


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


class QWNClient:
    def __init__(
        self,
        endpoint: str | None = None,
        host: str = HOST,
        port: int = PORT,
        verbose: bool = False,
    ):
        self.endpoint = endpoint.rstrip("/") if endpoint else f"http://{host}:{port}"
        self.verbose = verbose
        self.client = httpx.Client(timeout=httpx.Timeout(120.0))

    def close(self) -> None:
        if self.client is not None:
            self.client.close()
            self.client = None

    def __enter__(self) -> "QWNClient":
        return self

    def __exit__(self, *args) -> None:
        self.close()

    def _log_fail(self, action: str, error: Exception | str) -> None:
        if self.verbose:
            logger.warn(f"× QWN {action} error: {error}")

    def _log_okay(self, action: str, message: str) -> None:
        if self.verbose:
            logger.okay(f"✓ QWN {action}: {message}")

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
            detail,
            int(payload.get("max_tokens", 0) or 0),
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
            response = self.client.get(f"{self.endpoint}/health")
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

    def models(self) -> ModelInfo:
        try:
            response = self.client.get(f"{self.endpoint}/v1/models")
            response.raise_for_status()
            result = ModelInfo.from_dict(response.json())
            self._log_okay("models", f"count={len(result.models)}")
            return result
        except Exception as exc:
            self._log_fail("models", exc)
            raise

    def info(self) -> InfoResponse:
        try:
            response = self.client.get(f"{self.endpoint}/info")
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
        temperature: float = 0.7,
        top_p: float = 0.9,
        stream: bool = False,
        enable_thinking: bool = False,
    ) -> ChatResponse:
        payload = _build_chat_payload(
            messages=messages,
            model=model,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            stream=stream,
            enable_thinking=enable_thinking,
        )

        for attempt in range(3):
            try:
                response = self.client.post(
                    f"{self.endpoint}/v1/chat/completions",
                    json=payload,
                )
                response.raise_for_status()
                result = ChatResponse.from_dict(response.json())
                self._log_okay("chat", f"tokens={result.usage.total_tokens}")
                return result
            except httpx.HTTPStatusError as exc:
                detail = self._extract_error_detail(exc)
                if self._maybe_retry_with_capped_max_tokens(
                    payload,
                    detail,
                    action="chat",
                    attempt=attempt,
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
        temperature: float = 0.7,
        top_p: float = 0.9,
        on_text: Callable[[str], None] | None = None,
        enable_thinking: bool = False,
        include_thinking_tags: bool = False,
    ) -> StreamChatResult:
        payload = _build_chat_payload(
            messages=messages,
            model=model,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            stream=True,
            enable_thinking=enable_thinking,
        )
        payload["stream_options"] = {"include_usage": True}
        for attempt in range(3):
            text_fragments: list[str] = []
            usage = ChatUsage()
            started_at = time.perf_counter()
            render_state = _StreamRenderState(
                started_at=started_at,
                on_text=on_text,
                enable_thinking=enable_thinking,
                include_thinking_tags=include_thinking_tags,
                text_fragments=text_fragments,
            )
            try:
                with self.client.stream(
                    "POST",
                    f"{self.endpoint}/v1/chat/completions",
                    json=payload,
                ) as response:
                    response.raise_for_status()
                    render_state.begin()
                    retry_requested = False
                    for line in response.iter_lines():
                        if not line or not line.startswith("data: "):
                            continue
                        data = line[6:].strip()
                        if data == "[DONE]":
                            break
                        chunk = orjson.loads(data)
                        if "error" in chunk:
                            detail = _normalize_error_detail(chunk["error"])
                            if (
                                not text_fragments
                                and self._maybe_retry_with_capped_max_tokens(
                                    payload,
                                    detail,
                                    action="stream_chat",
                                    attempt=attempt,
                                )
                            ):
                                retry_requested = True
                                break
                            raise ValueError(f"Chat failed: {detail}")

                        render_state.emit_chunk(chunk)

                        chunk_usage = chunk.get("usage")
                        if isinstance(chunk_usage, dict):
                            usage = ChatUsage.from_dict(chunk_usage)

                if retry_requested:
                    continue

                render_state.finish()
                elapsed_sec = time.perf_counter() - started_at
                result = StreamChatResult(
                    text="".join(text_fragments),
                    usage=usage,
                    elapsed_sec=elapsed_sec,
                    first_token_latency_sec=render_state.first_token_latency_sec,
                )
                self._log_okay("stream_chat", f"tokens={result.usage.total_tokens}")
                return result
            except httpx.HTTPStatusError as exc:
                detail = self._extract_error_detail(exc)
                if self._maybe_retry_with_capped_max_tokens(
                    payload,
                    detail,
                    action="stream_chat",
                    attempt=attempt,
                ):
                    continue
                self._log_fail("stream_chat", detail)
                raise ValueError(f"Chat failed: {detail}") from exc
            except Exception as exc:
                self._log_fail("stream_chat", exc)
                raise

        raise RuntimeError("unreachable")

    def generate(
        self,
        prompt: str,
        system_prompt: str | None = None,
        max_tokens: int = DEFAULT_MAX_TOKENS,
        temperature: float = 0.7,
        top_p: float = 0.9,
        model: str = "",
        enable_thinking: bool = False,
    ) -> str:
        messages = build_text_messages(prompt=prompt, system_prompt=system_prompt)
        response = self.chat(
            messages=messages,
            model=model,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            enable_thinking=enable_thinking,
        )
        return response.text


class AsyncQWNClient:
    def __init__(
        self,
        endpoint: str | None = None,
        host: str = HOST,
        port: int = PORT,
        verbose: bool = False,
    ):
        self.endpoint = endpoint.rstrip("/") if endpoint else f"http://{host}:{port}"
        self.verbose = verbose
        self._client: httpx.AsyncClient | None = None

    async def _get_client(self) -> httpx.AsyncClient:
        if self._client is None:
            self._client = httpx.AsyncClient(
                timeout=httpx.Timeout(120.0),
                limits=httpx.Limits(max_keepalive_connections=10, max_connections=20),
            )
        return self._client

    async def close(self) -> None:
        if self._client is not None:
            await self._client.aclose()
            self._client = None

    def reset(self) -> None:
        self._client = None

    async def __aenter__(self) -> "AsyncQWNClient":
        return self

    async def __aexit__(self, *args) -> None:
        await self.close()

    def _log_fail(self, action: str, error: Exception | str) -> None:
        if self.verbose:
            logger.warn(f"× AsyncQWN {action} error: {error}")

    def _log_okay(self, action: str, message: str) -> None:
        if self.verbose:
            logger.okay(f"✓ AsyncQWN {action}: {message}")

    def _extract_error_detail(self, exc: httpx.HTTPStatusError) -> str:
        try:
            payload = exc.response.json()
            if "error" in payload:
                return _normalize_error_detail(payload["error"])
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
            detail,
            int(payload.get("max_tokens", 0) or 0),
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

    async def health(self) -> HealthResponse:
        client = await self._get_client()
        try:
            response = await client.get(f"{self.endpoint}/health")
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

    async def chat(
        self,
        messages: list[dict],
        model: str = "",
        max_tokens: int = DEFAULT_MAX_TOKENS,
        temperature: float = 0.7,
        top_p: float = 0.9,
        stream: bool = False,
        enable_thinking: bool = False,
    ) -> ChatResponse:
        payload = _build_chat_payload(
            messages=messages,
            model=model,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            stream=stream,
            enable_thinking=enable_thinking,
        )

        client = await self._get_client()
        for attempt in range(3):
            try:
                response = await client.post(
                    f"{self.endpoint}/v1/chat/completions",
                    content=orjson.dumps(payload),
                    headers={"Content-Type": "application/json"},
                )
                response.raise_for_status()
                result = ChatResponse.from_dict(orjson.loads(response.content))
                self._log_okay("chat", f"tokens={result.usage.total_tokens}")
                return result
            except httpx.HTTPStatusError as exc:
                detail = self._extract_error_detail(exc)
                if self._maybe_retry_with_capped_max_tokens(
                    payload,
                    detail,
                    action="chat",
                    attempt=attempt,
                ):
                    continue
                self._log_fail("chat", detail)
                raise ValueError(f"Chat failed: {detail}") from exc
            except Exception as exc:
                self._log_fail("chat", exc)
                raise

        raise RuntimeError("unreachable")

    async def stream_chat(
        self,
        messages: list[dict],
        model: str = "",
        max_tokens: int = DEFAULT_MAX_TOKENS,
        temperature: float = 0.7,
        top_p: float = 0.9,
        on_text: Callable[[str], None] | None = None,
        enable_thinking: bool = False,
        include_thinking_tags: bool = False,
    ) -> StreamChatResult:
        payload = _build_chat_payload(
            messages=messages,
            model=model,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            stream=True,
            enable_thinking=enable_thinking,
        )
        payload["stream_options"] = {"include_usage": True}

        client = await self._get_client()
        for attempt in range(3):
            text_fragments: list[str] = []
            usage = ChatUsage()
            started_at = time.perf_counter()
            render_state = _StreamRenderState(
                started_at=started_at,
                on_text=on_text,
                enable_thinking=enable_thinking,
                include_thinking_tags=include_thinking_tags,
                text_fragments=text_fragments,
            )
            try:
                async with client.stream(
                    "POST",
                    f"{self.endpoint}/v1/chat/completions",
                    content=orjson.dumps(payload),
                    headers={"Content-Type": "application/json"},
                ) as response:
                    response.raise_for_status()
                    render_state.begin()
                    retry_requested = False
                    async for line in response.aiter_lines():
                        if not line or not line.startswith("data: "):
                            continue
                        data = line[6:].strip()
                        if data == "[DONE]":
                            break
                        chunk = orjson.loads(data)
                        if "error" in chunk:
                            detail = _normalize_error_detail(chunk["error"])
                            if (
                                not text_fragments
                                and self._maybe_retry_with_capped_max_tokens(
                                    payload,
                                    detail,
                                    action="stream_chat",
                                    attempt=attempt,
                                )
                            ):
                                retry_requested = True
                                break
                            raise ValueError(f"Chat failed: {detail}")

                        render_state.emit_chunk(chunk)

                        chunk_usage = chunk.get("usage")
                        if isinstance(chunk_usage, dict):
                            usage = ChatUsage.from_dict(chunk_usage)

                if retry_requested:
                    continue

                render_state.finish()
                elapsed_sec = time.perf_counter() - started_at
                result = StreamChatResult(
                    text="".join(text_fragments),
                    usage=usage,
                    elapsed_sec=elapsed_sec,
                    first_token_latency_sec=render_state.first_token_latency_sec,
                )
                self._log_okay("stream_chat", f"tokens={result.usage.total_tokens}")
                return result
            except httpx.HTTPStatusError as exc:
                detail = self._extract_error_detail(exc)
                if self._maybe_retry_with_capped_max_tokens(
                    payload,
                    detail,
                    action="stream_chat",
                    attempt=attempt,
                ):
                    continue
                self._log_fail("stream_chat", detail)
                raise ValueError(f"Chat failed: {detail}") from exc
            except Exception as exc:
                self._log_fail("stream_chat", exc)
                raise

        raise RuntimeError("unreachable")

    async def generate(
        self,
        prompt: str,
        system_prompt: str | None = None,
        max_tokens: int = DEFAULT_MAX_TOKENS,
        temperature: float = 0.7,
        top_p: float = 0.9,
        model: str = "",
        enable_thinking: bool = False,
    ) -> str:
        messages = build_text_messages(prompt=prompt, system_prompt=system_prompt)
        response = await self.chat(
            messages=messages,
            model=model,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            enable_thinking=enable_thinking,
        )
        return response.text


CLI_EPILOG = """
Examples:
  qwn client health
  qwn client models
    qwn client chat "你好，介绍一下自己"
    qwn client chat -i photo_a.png -i photo_b.png "先看第一张图" "再比较第二张图"
    qwn client chat "你好" --no-stream
  qwn client generate --prompt "请总结一下 Docker 部署步骤"
  qwn client -e http://localhost:27880 models
"""


def configure_parser(parser: argparse.ArgumentParser) -> None:
    parser.description = "Connect to QWN machine or direct vLLM endpoints"
    parser.formatter_class = argparse.RawDescriptionHelpFormatter
    parser.epilog = CLI_EPILOG

    parser.add_argument(
        "-e", "--endpoint", type=str, default=None, help="Full endpoint URL"
    )
    parser.add_argument(
        "-p", "--port", type=int, default=PORT, help=f"Server port (default: {PORT})"
    )
    parser.add_argument(
        "-H", "--host", type=str, default=HOST, help=f"Server host (default: {HOST})"
    )
    parser.add_argument(
        "-v", "--verbose", action="store_true", help="Enable verbose output"
    )

    subparsers = parser.add_subparsers(dest="client_action", required=True)
    subparsers.add_parser("health", help="Check service health")
    subparsers.add_parser("models", help="List available models")
    subparsers.add_parser("info", help="Show machine info")

    chat_parser = subparsers.add_parser("chat", help="Send a chat completion")
    chat_parser.add_argument("text", nargs="+", help="Prompt text segment(s)")
    chat_parser.add_argument("--model", default="", help="Model label")
    chat_parser.add_argument("--max-tokens", type=int, default=DEFAULT_MAX_TOKENS)
    chat_parser.add_argument("--temperature", type=float, default=0.7)
    chat_parser.add_argument("--top-p", type=float, default=0.9)
    chat_parser.add_argument("-s", "--system", default=None, help="System prompt")
    chat_parser.add_argument(
        "-i",
        "--image",
        action="append",
        default=[],
        help="Image path, URL, or data URI. Repeat for multi-image prompts.",
    )
    chat_parser.add_argument(
        "--thinking",
        action="store_true",
        help="Enable model thinking-mode output",
    )
    chat_parser.add_argument(
        "--no-stream",
        action="store_true",
        help="Disable streaming terminal output",
    )

    generate_parser = subparsers.add_parser(
        "generate", help="Generate text from a prompt"
    )
    generate_parser.add_argument("--prompt", required=True)
    generate_parser.add_argument("--model", default="", help="Model label")
    generate_parser.add_argument("--max-tokens", type=int, default=DEFAULT_MAX_TOKENS)
    generate_parser.add_argument("--temperature", type=float, default=0.7)
    generate_parser.add_argument("--top-p", type=float, default=0.9)
    generate_parser.add_argument("-s", "--system", default=None, help="System prompt")
    generate_parser.add_argument(
        "--thinking",
        action="store_true",
        help="Enable model thinking-mode output",
    )


def run_from_args(args: argparse.Namespace) -> None:
    client = QWNClient(
        endpoint=args.endpoint,
        host=args.host,
        port=args.port,
        verbose=args.verbose,
    )

    try:
        if args.client_action == "health":
            client.log_machine_health()
        elif args.client_action == "models":
            info = client.models()
            logger.note("Available models:")
            for model in info.models:
                logger.mesg(f"  - {model}")
        elif args.client_action == "info":
            info = client.info()
            logger.note(f"Port: {info.port}")
            logger.note("Instances:")
            for instance in info.instances:
                details = [instance.model_label] if instance.model_label else []
                if instance.gpu_id is not None:
                    details.append(f"GPU{instance.gpu_id}")
                details.append(
                    f"active={instance.active_requests}/{instance.active_requests + instance.available_slots}"
                )
                if instance.gpu_utilization_pct is not None:
                    details.append(f"util={instance.gpu_utilization_pct:.0f}%")
                if (
                    instance.gpu_memory_used_mib is not None
                    and instance.gpu_memory_total_mib is not None
                    and instance.gpu_memory_total_mib > 0
                ):
                    details.append(
                        f"mem={instance.gpu_memory_used_mib:.0f}/{instance.gpu_memory_total_mib:.0f}MiB"
                    )
                if instance.routing_pressure is not None:
                    details.append(f"pressure={instance.routing_pressure:.2f}")
                if instance.scheduler.score is not None:
                    details.append(f"score={instance.scheduler.score:.2f}")
                if instance.scheduler.latency_ema_ms is not None:
                    details.append(f"lat={instance.scheduler.latency_ema_ms:.0f}ms")
                if instance.scheduler.ttft_ema_ms is not None:
                    details.append(f"ttft={instance.scheduler.ttft_ema_ms:.0f}ms")
                if instance.scheduler.tokens_per_second_ema is not None:
                    details.append(
                        f"tokps={instance.scheduler.tokens_per_second_ema:.1f}"
                    )
                if instance.scheduler.recent_requests > 0:
                    details.append(f"recent={instance.scheduler.recent_requests}")
                if instance.scheduler.recent_failures > 0:
                    details.append(f"fail={instance.scheduler.recent_failures}")
                if instance.scheduler.cooldown_remaining_sec > 0:
                    details.append(
                        f"cooldown={instance.scheduler.cooldown_remaining_sec:.1f}s"
                    )
                detail_text = ", ".join(details)
                logger.mesg(f"  - {instance.name}: {instance.endpoint} ({detail_text})")
                if instance.scheduler.last_error:
                    logger.warn(f"    last_error: {instance.scheduler.last_error}")
            logger.note("Stats:")
            for line in dict_to_lines(info.stats.__dict__).splitlines():
                logger.mesg(f"  {line}")
            logger.note("Scheduler:")
            for line in dict_to_lines(info.scheduler.to_dict()).splitlines():
                logger.mesg(f"  {line}")
        elif args.client_action == "chat":
            messages = build_multimodal_messages(
                texts=args.text,
                images=args.image,
                system_prompt=args.system,
            )
            if args.no_stream:
                result = client.chat(
                    messages=messages,
                    max_tokens=args.max_tokens,
                    temperature=args.temperature,
                    top_p=args.top_p,
                    model=args.model,
                    enable_thinking=args.thinking,
                )
                print(result.text)
            else:
                stream_result = client.stream_chat(
                    messages=messages,
                    model=args.model,
                    max_tokens=args.max_tokens,
                    temperature=args.temperature,
                    top_p=args.top_p,
                    on_text=lambda chunk: print(chunk, end="", flush=True),
                    enable_thinking=args.thinking,
                    include_thinking_tags=args.thinking,
                )
                if stream_result.text and not stream_result.text.endswith("\n"):
                    print()
                print(format_stream_stats_line(stream_result))
        elif args.client_action == "generate":
            result = client.generate(
                prompt=args.prompt,
                system_prompt=args.system,
                max_tokens=args.max_tokens,
                temperature=args.temperature,
                top_p=args.top_p,
                model=args.model,
                enable_thinking=args.thinking,
            )
            print(result)
        else:
            raise ValueError(f"Unknown client action: {args.client_action}")
    except httpx.ConnectError as exc:
        logger.warn(f"× Connection failed: {exc}")
        logger.hint(f"  Is the QWN service running at {client.endpoint}?")
    finally:
        client.close()


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser()
    configure_parser(parser)
    args = parser.parse_args(argv)
    run_from_args(args)


if __name__ == "__main__":
    main()
