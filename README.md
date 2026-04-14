# tfmx

![](https://img.shields.io/pypi/v/tfmx?label=tfmx&color=blue&cacheSeconds=60)

Utilities and runbooks for GPU control plus TEI, QSR, QWN, and QVL deployment workflows.

## Install

```sh
pip install tfmx --upgrade
```

## Recommended Entry Point

对当前服务器上最常见的 TEI + QSR 全链路恢复，推荐直接执行：

```bash
bash runs/recovery/start_tei_qsr.sh
```

详细说明见 [runs/recovery/README.md](runs/recovery/README.md)。

## Documentation Index

This README is intentionally kept as a navigation page. Concrete commands, staged workflows, and operational examples live in the module documents below.

| Module | Setup | Usage | Hints |
| --- | --- | --- | --- |
| TEI | [docs/teis/SETUP.md](docs/teis/SETUP.md) | [docs/teis/USAGE.md](docs/teis/USAGE.md) | [docs/teis/HINTS.md](docs/teis/HINTS.md) |
| QSR | [docs/qsrs/SETUP.md](docs/qsrs/SETUP.md) | [docs/qsrs/USAGE.md](docs/qsrs/USAGE.md) | [docs/qsrs/HINTS.md](docs/qsrs/HINTS.md) |
| QWN | [docs/qwns/SETUP.md](docs/qwns/SETUP.md) | [docs/qwns/USAGE.md](docs/qwns/USAGE.md) | [docs/qwns/HINTS.md](docs/qwns/HINTS.md) |
| QVL | [docs/qvls/SETUP.md](docs/qvls/SETUP.md) | [docs/qvls/USAGE.md](docs/qvls/USAGE.md) | [docs/qvls/HINTS.md](docs/qvls/HINTS.md) |

## Operational Runbooks

- [runs/recovery/README.md](runs/recovery/README.md): recommended TEI + QSR full-chain start, stop, and coexistence benchmark entrypoints
- [runs/teis/README.md](runs/teis/README.md): staged TEI deploy, health check, benchmark, and cleanup workflow
- [runs/qsrs/README.md](runs/qsrs/README.md): staged QSR deploy, health check, benchmark, soak, and cleanup workflow
- [runs/qwns/README.md](runs/qwns/README.md): staged QWN deploy, health check, benchmark, and cleanup workflow
- [runs/qvls/README.md](runs/qvls/README.md): staged QVL deploy and test workflow
- [runs/systemd/README.md](runs/systemd/README.md): optional `systemd --user` utilities, not the default path for heavy GPU services

## Repository Layout

- [docs](docs): module-specific setup, usage, and hints
- [runs](runs): staged scripts, recovery flows, and service runbooks
- [src/tfmx](src/tfmx): Python package source code
- [tests](tests): automated test suite

## Notes

- GPU control utilities are part of the package, but their concrete usage should be documented in module or future dedicated GPU docs rather than duplicated here.
- If you are starting from an operational task instead of an API task, begin with the relevant document under [docs](docs) or [runs](runs).