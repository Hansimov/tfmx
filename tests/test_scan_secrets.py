"""Tests for the git hook secret scanner."""

import importlib.util
import sys

from pathlib import Path


HOOK_PATH = Path(__file__).resolve().parent.parent / ".githooks" / "scan_secrets.py"
SPEC = importlib.util.spec_from_file_location("scan_secrets", HOOK_PATH)
SCAN_SECRETS = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules[SPEC.name] = SCAN_SECRETS
SPEC.loader.exec_module(SCAN_SECRETS)


def test_scan_text_detects_proxy_loopback_url():
    local_proxy_line = "HTTP_PROXY=http://localhost:" + "18080"
    findings = SCAN_SECRETS.scan_text("demo.txt", local_proxy_line)
    assert findings
    assert findings[0].rule_name == "proxy-loopback-url"


def test_scan_text_detects_bare_host_url():
    host_url = "endpoint=http://worker" + "42:8080"
    findings = SCAN_SECRETS.scan_text("demo.txt", host_url)
    assert findings
    assert findings[0].rule_name == "bare-host-url"


def test_scan_text_allows_generic_placeholders():
    findings = SCAN_SECRETS.scan_text(
        "demo.txt",
        'qwn compose up --proxy "$QWN_PROXY"',
    )
    assert findings == []
