"""Shared background daemon management for machine CLIs."""

import os
import signal
import subprocess
import sys
import time

from pathlib import Path
from typing import Sequence

from tclogger import logger


class ManagedMachineDaemon:
    def __init__(
        self,
        *,
        label: str,
        module_name: str,
        daemon_env_flag: str,
        pid_file: Path,
        log_file: Path,
        background_flags: tuple[str, ...],
    ):
        self.label = label
        self.module_name = module_name
        self.daemon_env_flag = daemon_env_flag
        self.pid_file = pid_file
        self.log_file = log_file
        self.background_flags = background_flags
        self._ensure_cache_dir()

    def _ensure_cache_dir(self) -> None:
        self.pid_file.parent.mkdir(parents=True, exist_ok=True)

    def get_pid(self) -> int | None:
        if not self.pid_file.exists():
            return None
        try:
            pid = int(self.pid_file.read_text().strip())
            os.kill(pid, 0)
            return pid
        except (ValueError, ProcessLookupError, PermissionError):
            self.pid_file.unlink(missing_ok=True)
            return None

    def is_running(self) -> bool:
        return self.get_pid() is not None

    def write_pid(self) -> None:
        self.pid_file.write_text(str(os.getpid()))

    def remove_pid(self) -> None:
        self.pid_file.unlink(missing_ok=True)

    def is_daemon_process(self) -> bool:
        return os.environ.get(self.daemon_env_flag) == "1"

    def _normalize_background_argv(self, argv: Sequence[str]) -> list[str]:
        normalized_args = list(argv[1:]) if argv else []
        if normalized_args and normalized_args[0] == "machine":
            normalized_args = normalized_args[1:]

        cleaned_args: list[str] = []
        for arg in normalized_args:
            if arg in self.background_flags:
                continue
            cleaned_args.append(arg)
        return cleaned_args

    def start_background(self, argv: Sequence[str]) -> None:
        cmd = [sys.executable, "-m", self.module_name]
        cmd.extend(self._normalize_background_argv(argv))

        with self.log_file.open("a", encoding="utf-8") as log_handle:
            proc = subprocess.Popen(
                cmd,
                stdout=log_handle,
                stderr=subprocess.STDOUT,
                start_new_session=True,
                env={**os.environ, self.daemon_env_flag: "1"},
            )

        self.pid_file.write_text(str(proc.pid))
        logger.okay(f"{self.label} Started in background (PID {proc.pid})")
        logger.mesg(f"  Log: {self.log_file}")
        logger.mesg(f"  PID: {self.pid_file}")

    def stop(self) -> bool:
        pid = self.get_pid()
        if pid is None:
            logger.mesg(f"{self.label} No running daemon found")
            return False

        logger.mesg(f"{self.label} Stopping daemon (PID {pid})...")
        try:
            os.kill(pid, signal.SIGTERM)
            for _ in range(100):
                try:
                    os.kill(pid, 0)
                    time.sleep(0.1)
                except ProcessLookupError:
                    break
            else:
                logger.warn(f"{self.label} Force killing PID {pid}")
                os.kill(pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        except PermissionError:
            logger.warn(f"× {self.label} Permission denied killing PID {pid}")
            return False

        self.remove_pid()
        logger.okay(f"{self.label} Daemon stopped")
        return True

    def status(self) -> None:
        pid = self.get_pid()
        if pid is None:
            logger.mesg(f"{self.label} Status: not running")
            return

        logger.okay(f"{self.label} Status: running (PID {pid})")
        logger.mesg(f"  PID file: {self.pid_file}")
        logger.mesg(f"  Log file: {self.log_file}")
        if self.log_file.exists():
            size = self.log_file.stat().st_size
            if size < 1024:
                logger.mesg(f"  Log size: {size} B")
            elif size < 1024 * 1024:
                logger.mesg(f"  Log size: {size / 1024:.1f} KB")
            else:
                logger.mesg(f"  Log size: {size / 1024 / 1024:.1f} MB")

    def show_logs(self, *, follow: bool = False, tail: int = 50) -> None:
        if not self.log_file.exists():
            logger.mesg(f"{self.label} No log file found")
            return

        if follow:
            try:
                subprocess.run(["tail", "-f", "-n", str(tail), str(self.log_file)])
            except KeyboardInterrupt:
                pass
            return

        result = subprocess.run(
            ["tail", "-n", str(tail), str(self.log_file)],
            capture_output=True,
            text=True,
        )
        if result.stdout:
            print(result.stdout, end="")
        else:
            logger.mesg(f"{self.label} Log file is empty")
