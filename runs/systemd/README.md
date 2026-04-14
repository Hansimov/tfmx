# TEI + QSR User Systemd Services

这组文件用于把 TEI `28800` 与 QSR `27900` 安装成 `systemd --user` 长期服务。

## 安装

```bash
bash runs/systemd/install_tei_qsr_user_services.sh
```

安装脚本会把模板渲染到：

- `~/.config/systemd/user/tfmx-tei-machine.service`
- `~/.config/systemd/user/tfmx-qsr-machine.service`

并自动执行：

- `systemctl --user daemon-reload`
- `systemctl --user enable ...`

## 启动

```bash
systemctl --user start tfmx-tei-machine.service
systemctl --user start tfmx-qsr-machine.service
```

或者：

```bash
systemctl --user start tfmx-tei-machine.service tfmx-qsr-machine.service
```

## 常用命令

```bash
systemctl --user status tfmx-tei-machine.service
systemctl --user status tfmx-qsr-machine.service
journalctl --user -u tfmx-tei-machine.service -f
journalctl --user -u tfmx-qsr-machine.service -f
systemctl --user restart tfmx-tei-machine.service
systemctl --user restart tfmx-qsr-machine.service
systemctl --user stop tfmx-tei-machine.service tfmx-qsr-machine.service
```

## GPU 选择

默认行为：

- 若未显式设置环境变量，会在启动时自动检测当前全部可见 GPU
- 只排除不健康 GPU，不会因为同卡上已经有另一套服务在跑就自动排除

若你想手动限制 GPU 子集，可在启动前设置 user service 环境：

```bash
systemctl --user set-environment TEI_SERVICE_GPUS=0,1,2,3,4,5
systemctl --user set-environment QSR_SERVICE_GPUS=0,1,2,3,4,5
systemctl --user restart tfmx-tei-machine.service tfmx-qsr-machine.service
```

清除覆盖：

```bash
systemctl --user unset-environment TEI_SERVICE_GPUS QSR_SERVICE_GPUS
```

## 开机自启

当前机器若 `loginctl show-user $USER -p Linger` 返回 `Linger=yes`，则 `systemctl --user enable` 后可在开机时自动拉起。