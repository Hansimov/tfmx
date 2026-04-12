#!/usr/bin/env python3
"""Small local debug helper for inspecting generated QSR compose output."""

from pathlib import Path

from tfmx.qsrs.compose import GPUInfo, QSRComposer, parse_gpu_configs


def main() -> None:
    composer = QSRComposer(
        project_name="qsr-debug",
        gpu_configs=parse_gpu_configs("0"),
        compose_dir=Path("/tmp/tfmx-qsr-debug"),
    )
    composer.gpus = [GPUInfo(index=0, compute_cap="8.9")]
    compose_file = composer.generate_compose_file()
    print(compose_file)
    print(compose_file.read_text()[:1600])


if __name__ == "__main__":
    main()
