"""Reusable network and mirror settings for QWN builds and runtime."""

import os

from dataclasses import dataclass
from urllib.request import getproxies


DEFAULT_HF_ENDPOINT = "https://hf-mirror.com"
DEFAULT_PIP_INDEX_URL = "https://mirrors.ustc.edu.cn/pypi/simple"
DEFAULT_PIP_TRUSTED_HOST = "mirrors.ustc.edu.cn"
DOCKER_HOST_ALIAS = "host.docker.internal"
DEFAULT_NO_PROXY = ("localhost", "127.0.0.1", DOCKER_HOST_ALIAS)
EXPLICIT_PROXY_ENV_KEYS = (
    "TFMX_QWN_PROXY",
    "QWN_PROXY",
    "TFMX_HTTP_PROXY",
)


def _read_env(*keys: str) -> str | None:
    for key in keys:
        value = os.getenv(key)
        if value is None:
            continue
        normalized = value.strip()
        return normalized or None
    return None


def is_loopback_proxy(proxy: str | None) -> bool:
    if not proxy:
        return False
    normalized = proxy.strip().lower()
    return "://127.0.0.1" in normalized or "://localhost" in normalized


def detect_default_proxy(proxies: dict[str, str] | None = None) -> str | None:
    explicit = _read_env(*EXPLICIT_PROXY_ENV_KEYS)
    if explicit is not None:
        return explicit

    proxy_map = proxies if proxies is not None else getproxies()
    for key in ("https", "http", "all"):
        value = proxy_map.get(key)
        if not value:
            continue
        normalized = value.strip()
        if normalized:
            return normalized
    return None


def _split_csv(value: str | None) -> tuple[str, ...]:
    if not value:
        return ()
    return tuple(item.strip() for item in value.split(",") if item.strip())


@dataclass(frozen=True)
class QWNNetworkConfig:
    hf_endpoint: str = DEFAULT_HF_ENDPOINT
    http_proxy: str | None = None
    https_proxy: str | None = None
    pip_index_url: str = DEFAULT_PIP_INDEX_URL
    pip_trusted_host: str = DEFAULT_PIP_TRUSTED_HOST
    no_proxy: tuple[str, ...] = DEFAULT_NO_PROXY

    @classmethod
    def from_overrides(
        cls,
        proxy: str | None = None,
        http_proxy: str | None = None,
        https_proxy: str | None = None,
        hf_endpoint: str | None = None,
        pip_index_url: str | None = None,
        pip_trusted_host: str | None = None,
        no_proxy: str | tuple[str, ...] | None = None,
    ) -> "QWNNetworkConfig":
        resolved_proxy = proxy if proxy is not None else detect_default_proxy()
        resolved_http_proxy = http_proxy if http_proxy is not None else resolved_proxy
        resolved_https_proxy = (
            https_proxy if https_proxy is not None else resolved_http_proxy
        )

        resolved_hf_endpoint = (
            hf_endpoint
            or _read_env(
                "TFMX_QWN_HF_ENDPOINT",
                "QWN_HF_ENDPOINT",
            )
            or DEFAULT_HF_ENDPOINT
        )
        resolved_pip_index_url = (
            pip_index_url
            or _read_env(
                "TFMX_QWN_PIP_INDEX_URL",
                "QWN_PIP_INDEX_URL",
                "PIP_INDEX_URL",
            )
            or DEFAULT_PIP_INDEX_URL
        )
        resolved_pip_trusted_host = (
            pip_trusted_host
            or _read_env(
                "TFMX_QWN_PIP_TRUSTED_HOST",
                "QWN_PIP_TRUSTED_HOST",
                "PIP_TRUSTED_HOST",
            )
            or DEFAULT_PIP_TRUSTED_HOST
        )

        if isinstance(no_proxy, str):
            no_proxy_values = _split_csv(no_proxy)
        elif isinstance(no_proxy, tuple):
            no_proxy_values = no_proxy
        else:
            no_proxy_values = _split_csv(
                _read_env(
                    "TFMX_QWN_NO_PROXY",
                    "QWN_NO_PROXY",
                    "NO_PROXY",
                )
            )

        merged_no_proxy = []
        for item in (*DEFAULT_NO_PROXY, *no_proxy_values):
            if item and item not in merged_no_proxy:
                merged_no_proxy.append(item)

        return cls(
            hf_endpoint=resolved_hf_endpoint,
            http_proxy=resolved_http_proxy,
            https_proxy=resolved_https_proxy,
            pip_index_url=resolved_pip_index_url,
            pip_trusted_host=resolved_pip_trusted_host,
            no_proxy=tuple(merged_no_proxy),
        )

    @property
    def build_proxy(self) -> str | None:
        return self.https_proxy or self.http_proxy

    @property
    def runtime_http_proxy(self) -> str | None:
        if is_loopback_proxy(self.http_proxy):
            return None
        return self.http_proxy

    @property
    def runtime_https_proxy(self) -> str | None:
        if is_loopback_proxy(self.https_proxy):
            return None
        return self.https_proxy

    @property
    def no_proxy_csv(self) -> str:
        return ",".join(self.no_proxy)

    @property
    def use_host_network_for_build(self) -> bool:
        return is_loopback_proxy(self.build_proxy)

    def docker_build_args(self) -> list[str]:
        args = [
            "--build-arg",
            f"HF_ENDPOINT={self.hf_endpoint}",
            "--build-arg",
            f"PIP_INDEX_URL={self.pip_index_url}",
            "--build-arg",
            f"PIP_TRUSTED_HOST={self.pip_trusted_host}",
            "--build-arg",
            f"NO_PROXY={self.no_proxy_csv}",
            "--build-arg",
            f"no_proxy={self.no_proxy_csv}",
        ]

        if self.http_proxy:
            args.extend(["--build-arg", f"HTTP_PROXY={self.http_proxy}"])
            args.extend(["--build-arg", f"http_proxy={self.http_proxy}"])
        if self.https_proxy:
            args.extend(["--build-arg", f"HTTPS_PROXY={self.https_proxy}"])
            args.extend(["--build-arg", f"https_proxy={self.https_proxy}"])
        return args
