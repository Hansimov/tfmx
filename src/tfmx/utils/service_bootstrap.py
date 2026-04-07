import time

import httpx

from tclogger import logger


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
