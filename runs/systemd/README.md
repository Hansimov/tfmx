# Optional User Systemd Units

对于 TEI `28800` 与 QSR `27900` 这类依赖 GPU 与 Docker 的重资源服务，默认不建议用 `systemd --user` 做开机自启。更推荐的主路径是：主机重启后按需手动执行 [runs/recovery/start_tei_qsr.sh](../recovery/start_tei_qsr.sh)。

这里保留的 `systemd` 文件只作为可选工具，不再作为默认推荐方案。

## 推荐主路径

```bash
bash runs/recovery/start_tei_qsr.sh
```

配套入口：

- 停止清理：`bash runs/recovery/stop_tei_qsr.sh`
- 共存 benchmark：`bash runs/recovery/benchmark_tei_qsr_coexist.sh`

## 如果之前装过 user units，先卸载

```bash
bash runs/systemd/uninstall_tei_qsr_user_services.sh
```

这会：

- `disable --now` 已安装的 `tfmx-tei-machine.service` 与 `tfmx-qsr-machine.service`
- 删除 `~/.config/systemd/user/` 下对应 unit 文件
- 清掉 `TEI_SERVICE_GPUS` / `QSR_SERVICE_GPUS` 的 user 环境覆盖

## 可选安装

```bash
bash runs/systemd/install_tei_qsr_user_services.sh
```

默认只会渲染 unit 文件并 `daemon-reload`，不会自动 enable。

如果你仍然明确要启用 user units：

```bash
ENABLE_USER_UNITS=1 bash runs/systemd/install_tei_qsr_user_services.sh
```

如果你还要安装后立刻启动：

```bash
ENABLE_USER_UNITS=1 START_USER_UNITS=1 bash runs/systemd/install_tei_qsr_user_services.sh
```

## 手动管理命令

```bash
systemctl --user start tfmx-tei-machine.service tfmx-qsr-machine.service
systemctl --user status tfmx-tei-machine.service tfmx-qsr-machine.service
journalctl --user -u tfmx-tei-machine.service -f
journalctl --user -u tfmx-qsr-machine.service -f
systemctl --user stop tfmx-tei-machine.service tfmx-qsr-machine.service
```

## GPU 选择

默认行为：

- 若未显式设置环境变量，会在启动时自动检测当前全部可见 GPU
- 只排除不健康 GPU，不会因为同卡上已经有另一套服务在跑就自动排除

若你想手动限制 GPU 子集，可在启动前设置 user service 环境：

```bash
systemctl --user set-environment TEI_SERVICE_GPUS=0,1
systemctl --user set-environment QSR_SERVICE_GPUS=0,1
systemctl --user restart tfmx-tei-machine.service tfmx-qsr-machine.service
```