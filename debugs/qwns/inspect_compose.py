"""Small local debug helper for inspecting generated QWN compose output."""

from pathlib import Path

from tfmx.qwns.compose import GPUInfo, QWNComposer, parse_gpu_configs


def main() -> None:
    composer = QWNComposer(
        project_name="qwn-debug",
        gpu_configs=parse_gpu_configs("0:4b:4bit"),
        compose_dir=Path("/tmp/tfmx-qwn-debug"),
    )
    composer.gpus = [GPUInfo(index=0, compute_cap="8.9")]
    compose_file = composer.generate_compose_file()
    print(compose_file)
    print(compose_file.read_text()[:1200])


if __name__ == "__main__":
    main()
