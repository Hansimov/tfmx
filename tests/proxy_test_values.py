"""Shared proxy URL helpers for tests.

These values intentionally use reserved placeholder domains or runtime assembly
so the staged secret scanner does not flag them as host-specific literals.
"""


def build_proxy_url(host: str, port: int) -> str:
    return f"http://{host}:{port}"


PLACEHOLDER_PROXY_URL = build_proxy_url("proxy.invalid", 8080)
PLACEHOLDER_HOST_PROXY_URL = build_proxy_url("proxy.invalid", 18080)
