#!/usr/bin/env python3
"""Scan staged content for host- and proxy-like sensitive literals."""

from __future__ import annotations

import argparse
import re
import subprocess
import sys

from dataclasses import dataclass
from pathlib import Path


IGNORE_MARKER = "scan-secrets: ignore"


@dataclass(frozen=True)
class Finding:
    path: str
    line_number: int
    rule_name: str
    line: str


@dataclass(frozen=True)
class Rule:
    name: str
    pattern: re.Pattern[str]
    message: str


RULES: tuple[Rule, ...] = (
    Rule(
        name="proxy-loopback-url",
        pattern=re.compile(
            r"(?i)\b(?:https?_proxy|proxy)\b[^\n#]{0,120}(?:https?://)?(?:127\.0\.0\.1|localhost):\d{2,5}\b"
        ),
        message="explicit loopback proxy URL",
    ),
    Rule(
        name="bare-host-url",
        pattern=re.compile(
            r"(?i)https?://(?!localhost\b)(?!127(?:\.\d+){3}\b)(?![a-z0-9.-]+\.[a-z]{2,}\b)([a-z][a-z0-9_-]*\d{2,4})(?::\d{2,5})?\b"
        ),
        message="URL with bare internal host name",
    ),
)


def _git(args: list[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["git", *args],
        text=True,
        capture_output=True,
        check=False,
    )


def get_staged_files() -> list[str]:
    result = _git(["diff", "--cached", "--name-only", "--diff-filter=ACMR"])
    if result.returncode != 0:
        raise RuntimeError(result.stderr.strip() or "failed to list staged files")
    return [line.strip() for line in result.stdout.splitlines() if line.strip()]


def read_staged_text(path: str) -> str | None:
    result = _git(["show", f":{path}"])
    if result.returncode != 0:
        return None
    if "\x00" in result.stdout:
        return None
    return result.stdout


def read_worktree_text(path: str) -> str | None:
    try:
        text = Path(path).read_text()
    except (OSError, UnicodeDecodeError):
        return None
    if "\x00" in text:
        return None
    return text


def scan_text(path: str, text: str) -> list[Finding]:
    findings: list[Finding] = []
    for line_number, raw_line in enumerate(text.splitlines(), start=1):
        if IGNORE_MARKER in raw_line:
            continue
        for rule in RULES:
            if not rule.pattern.search(raw_line):
                continue
            findings.append(
                Finding(
                    path=path,
                    line_number=line_number,
                    rule_name=rule.name,
                    line=raw_line.strip(),
                )
            )
    return findings


def scan_staged_files(paths: list[str]) -> list[Finding]:
    findings: list[Finding] = []
    for path in paths:
        if path.startswith(".pytest_cache/"):
            continue
        text = read_staged_text(path)
        if text is None:
            continue
        findings.extend(scan_text(path, text))
    return findings


def scan_paths(paths: list[str]) -> list[Finding]:
    findings: list[Finding] = []
    for path in paths:
        text = read_worktree_text(path)
        if text is None:
            continue
        findings.extend(scan_text(path, text))
    return findings


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Scan staged files for sensitive host or proxy literals.",
    )
    parser.add_argument("--staged", action="store_true", help="Scan staged files")
    parser.add_argument("paths", nargs="*", help="Optional explicit file paths")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    paths = args.paths
    if args.staged or not paths:
        paths = get_staged_files()
        findings = scan_staged_files(paths)
    else:
        findings = scan_paths(paths)
    if not findings:
        return 0

    print("Potential sensitive literals detected:", file=sys.stderr)
    for finding in findings:
        print(
            f"- {finding.path}:{finding.line_number}: {finding.rule_name}: {finding.line}",
            file=sys.stderr,
        )
    print(
        "Use environment variables or generic placeholders instead of host-specific or proxy-specific literals.",
        file=sys.stderr,
    )
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
