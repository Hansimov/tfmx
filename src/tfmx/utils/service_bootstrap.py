import os
import re
import signal
import subprocess
import time

from dataclasses import dataclass

import httpx

from tclogger import logger


@dataclass
class PortConflict:
    port: int
    pid: int | None = None
    process_name: str = ""


def find_port_conflicts(port: int) -> list[PortConflict]:
    try:
        result = subprocess.run(
            ["ss", "-ltnp", f"sport = :{port}"],
            capture_output=True,
            text=True,
            timeout=5,
        )
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return []

    if result.returncode != 0 or not result.stdout.strip():
        return []

    conflicts: list[PortConflict] = []
    seen_pids: set[int | None] = set()
    lines = [line for line in result.stdout.splitlines() if line.strip()]
    if len(lines) <= 1:
        return []

    for line in lines[1:]:
        matches = re.findall(r'\("([^\"]+)",pid=(\d+),fd=\d+\)', line)
        if matches:
            for process_name, pid_text in matches:
                pid = int(pid_text)
                if pid in seen_pids:
                    continue
                seen_pids.add(pid)
                conflicts.append(
                    PortConflict(
                        port=port,
                        pid=pid,
                        process_name=process_name,
                    )
                )
            continue

        if None not in seen_pids:
            seen_pids.add(None)
            conflicts.append(PortConflict(port=port))

    return conflicts


def terminate_process(pid: int, *, label: str, timeout_sec: float = 5.0) -> bool:
    try:
        os.kill(pid, signal.SIGTERM)
        deadline = time.monotonic() + timeout_sec
        while time.monotonic() < deadline:
            try:
                os.kill(pid, 0)
            except ProcessLookupError:
                return True
            time.sleep(0.1)

        logger.warn(f"{label} Force killing conflicting PID {pid}")
        os.kill(pid, signal.SIGKILL)
        return True
    except ProcessLookupError:
        return True
    except PermissionError:
        logger.warn(f"× {label} Permission denied killing conflicting PID {pid}")
        return False


def handle_port_conflicts(
    port: int,
    *,
    policy: str,
    label: str,
) -> bool:
    conflicts = find_port_conflicts(port)
    if not conflicts:
        return True

    descriptions = []
    for conflict in conflicts:
        if conflict.pid is None:
            descriptions.append("unknown listener")
        elif conflict.process_name:
            descriptions.append(f"{conflict.process_name}(pid={conflict.pid})")
        else:
            descriptions.append(f"pid={conflict.pid}")
    detail = ", ".join(descriptions)

    if policy == "report":
        logger.warn(f"× {label} Port {port} is already in use: {detail}")
        return False

    logger.warn(f"{label} Replacing conflicting listener(s) on port {port}: {detail}")
    for conflict in conflicts:
        if conflict.pid is None:
            logger.warn(f"× {label} Cannot replace unknown listener on port {port}")
            return False
        if not terminate_process(conflict.pid, label=label):
            return False

    remaining = find_port_conflicts(port)
    if remaining:
        logger.warn(f"× {label} Port {port} is still occupied after replace attempt")
        return False

    logger.okay(f"{label} Cleared conflicting listener(s) on port {port}")
    return True


def wait_for_healthy_http_endpoints(
    endpoints: list[str],
    *,
    timeout_sec: float,
    poll_interval_sec: float,
    health_path: str = "/health",
    label: str = "[tfmx]",
) -> bool:
    normalized_endpoints = [endpoint.rstrip("/") for endpoint in endpoints if endpoint]
    if not normalized_endpoints:
        logger.warn(f"× {label} No backend endpoints available for health wait")
        return False

    pending = set(normalized_endpoints)
    deadline = time.monotonic() + timeout_sec

    logger.mesg(
        f"{label} Waiting for {len(pending)} backend endpoint(s) to become healthy"
    )

    with httpx.Client(timeout=httpx.Timeout(5.0)) as client:
        while pending and time.monotonic() < deadline:
            for endpoint in list(pending):
                try:
                    response = client.get(f"{endpoint}{health_path}")
                    if response.status_code == 200:
                        pending.remove(endpoint)
                except Exception:
                    continue

            if pending:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    break
                time.sleep(min(poll_interval_sec, remaining))

    if pending:
        missing = ", ".join(sorted(pending))
        logger.warn(f"× {label} Timed out waiting for healthy backends: {missing}")
        return False

    logger.okay(f"{label} All backend endpoints are healthy")
    return True


def docker_status_to_health(status: str) -> bool | None:
    normalized = (status or "").strip().lower()
    if not normalized:
        return None
    if "unhealthy" in normalized:
        return False
    if "health: starting" in normalized:
        return False
    if "(healthy)" in normalized or normalized == "healthy":
        return True
    if normalized.startswith("up "):
        return None
    return False


def get_docker_container_statuses(container_names: list[str]) -> dict[str, str]:
    normalized_names = [
        name.strip() for name in container_names if name and name.strip()
    ]
    if not normalized_names:
        return {}

    try:
        result = subprocess.run(
            ["docker", "ps", "-a", "--format", "{{.Names}}|{{.Status}}"],
            capture_output=True,
            text=True,
            timeout=10,
        )
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return {}

    if result.returncode != 0 or not result.stdout.strip():
        return {}

    requested = set(normalized_names)
    statuses: dict[str, str] = {}
    for line in result.stdout.splitlines():
        if not line.strip() or "|" not in line:
            continue
        container_name, status = line.split("|", 1)
        container_name = container_name.strip()
        if container_name in requested:
            statuses[container_name] = status.strip()
    return statuses


def wait_for_healthy_docker_containers(
    container_names: list[str],
    *,
    timeout_sec: float,
    poll_interval_sec: float,
    label: str = "[tfmx]",
) -> bool:
    normalized_names = [
        name.strip() for name in container_names if name and name.strip()
    ]
    if not normalized_names:
        logger.warn(f"× {label} No backend containers available for health wait")
        return False

    pending = set(normalized_names)
    deadline = time.monotonic() + timeout_sec

    logger.mesg(
        f"{label} Waiting for {len(pending)} backend container(s) to become healthy"
    )

    last_statuses: dict[str, str] = {}
    while pending and time.monotonic() < deadline:
        last_statuses = get_docker_container_statuses(normalized_names)
        for container_name in list(pending):
            if docker_status_to_health(last_statuses.get(container_name, "")) is True:
                pending.remove(container_name)

        if pending:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                break
            time.sleep(min(poll_interval_sec, remaining))

    if pending:
        missing = []
        for container_name in sorted(pending):
            status = last_statuses.get(container_name)
            if status:
                missing.append(f"{container_name} ({status})")
            else:
                missing.append(container_name)
        logger.warn(
            f"× {label} Timed out waiting for healthy containers: {', '.join(missing)}"
        )
        return False

    logger.okay(f"{label} All backend containers are healthy")
    return True


def wait_for_available_backend_instances(
    rediscover,
    *,
    timeout_sec: float,
    poll_interval_sec: float,
    settle_sec: float,
    health_path: str = "/health",
    label: str = "[tfmx]",
) -> list:
    deadline = time.monotonic() + timeout_sec
    last_instances: list = []
    last_discovered_endpoints: set[str] = set()
    last_change_monotonic = time.monotonic()

    logger.mesg(f"{label} Waiting for discovered backend instances to become healthy")

    with httpx.Client(timeout=httpx.Timeout(5.0)) as client:
        while time.monotonic() < deadline:
            now = time.monotonic()
            instances = rediscover() or []
            last_instances = instances

            current_endpoints = {
                instance.endpoint.rstrip("/")
                for instance in instances
                if getattr(instance, "endpoint", "")
            }
            if current_endpoints != last_discovered_endpoints:
                last_discovered_endpoints = current_endpoints
                last_change_monotonic = now
                if current_endpoints:
                    logger.mesg(
                        f"{label} Discovered {len(current_endpoints)} backend instance(s)"
                    )

            healthy_count = 0
            for instance in instances:
                endpoint = getattr(instance, "endpoint", "").rstrip("/")
                health_url = getattr(instance, "health_url", "")
                url = health_url or (f"{endpoint}{health_path}" if endpoint else "")
                docker_health = getattr(instance, "docker_health", None)
                is_healthy = False
                if docker_health is not None:
                    is_healthy = bool(docker_health)
                elif url:
                    try:
                        response = client.get(url)
                        is_healthy = response.status_code == 200
                    except Exception:
                        is_healthy = False
                if hasattr(instance, "healthy"):
                    instance.healthy = is_healthy
                if is_healthy:
                    healthy_count += 1

            if healthy_count:
                all_discovered_healthy = healthy_count == len(instances)
                settled = (now - last_change_monotonic) >= settle_sec
                if all_discovered_healthy or settled:
                    if all_discovered_healthy:
                        logger.okay(
                            f"{label} {healthy_count} backend instance(s) are healthy"
                        )
                    else:
                        logger.note(
                            f"{label} Proceeding with {healthy_count}/{len(instances)} healthy backend instance(s); unavailable instances will be skipped"
                        )
                    return instances

            remaining = deadline - time.monotonic()
            if remaining <= 0:
                break
            time.sleep(min(poll_interval_sec, remaining))

    if last_instances:
        healthy_count = sum(
            1 for instance in last_instances if getattr(instance, "healthy", False)
        )
        if healthy_count:
            logger.note(
                f"{label} Startup wait timed out; proceeding with {healthy_count}/{len(last_instances)} healthy backend instance(s)"
            )
        else:
            logger.warn(f"× {label} Timed out waiting for healthy backend instances")
        return last_instances

    logger.warn(f"× {label} Timed out waiting for backend instances to appear")
    return []


def ensure_backend_instances(
    instances: list,
    *,
    enabled: bool,
    manual_endpoints: bool,
    service_label: str,
    compose_factory,
    rediscover,
    timeout_sec: float,
    poll_interval_sec: float,
    allow_partial: bool = False,
    settle_sec: float = 10.0,
) -> list:
    if instances or manual_endpoints or not enabled:
        return instances

    logger.note(
        f"{service_label} No running backend instances found; starting compose deployment"
    )

    composer = compose_factory()
    endpoints = composer.get_backend_endpoints()
    if not endpoints:
        logger.warn(f"× {service_label} Auto-start skipped: no healthy GPUs available")
        return instances

    composer.up()
    if allow_partial:
        refreshed_instances = wait_for_available_backend_instances(
            rediscover,
            timeout_sec=timeout_sec,
            poll_interval_sec=poll_interval_sec,
            settle_sec=settle_sec,
            label=service_label,
        )
    else:
        composer.wait_for_healthy_backends(
            timeout_sec=timeout_sec,
            poll_interval_sec=poll_interval_sec,
            label=service_label,
        )
        refreshed_instances = rediscover()

    if refreshed_instances:
        logger.okay(
            f"{service_label} Auto-start discovered {len(refreshed_instances)} backend instance(s)"
        )
    else:
        logger.warn(f"× {service_label} No backends discovered after auto-start")
    return refreshed_instances
