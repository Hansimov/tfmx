# tfmx

![](https://img.shields.io/pypi/v/tfmx?label=tfmx&color=blue&cacheSeconds=60)

## Install

```sh
pip install tfmx --upgrade
```

## Commands

Set GPU control state:

```sh
gpu_fan -cs a:1
```

Set GPU power limit:

```sh
# M-X GPU-0/1
gpu_pow -pm a:1 && gpu_pow -pl "0:160;1:200"

# M-A GPU-0/1
gpu_pow -pm a:1 && gpu_pow -pl "0,1:160"
```

Set GPU fan speed:

```sh
gpu_fan -cs a:1 && gpu_fan -fs a:100
```

Set GPU monitor with curve:

```sh
# gpu_mon -c "a:30-50/50-65/60-80/75-100;3,7:25-100" -s
```

Run tei compose and machine:

```sh
tei machine run --auto-start --perf-track --on-conflict replace
```

Run tei benchmark:

```sh
tei benchmark run -E "http://localhost:28800" -n 100000
```

Run qwn compose, machine, and benchmark:

```sh
export QWN_MACHINE_URL="http://$QWN_HOST:27800"

qwn compose up --gpu-configs "0"
qwn compose up --gpu-layout uniform-awq
qwn machine run --auto-start -b --on-conflict replace
qwn client chat "你好，请做个自我介绍"
qwn benchmark run -E "$QWN_MACHINE_URL" -n 100
```

Run qsr compose, machine, and benchmark:

```sh
export QSR_MACHINE_URL="http://$QSR_HOST:27900"

qsr compose up --gpu-configs "0"
qsr machine run --auto-start -b --on-conflict replace
qsr client transcribe ./sample.wav
qsr client chat --audio ./sample.wav "请转写为简体中文"
qsr benchmark run -E "$QSR_MACHINE_URL" -n 20 --audio ./sample.wav
```

## Staged Run Scripts

For repeatable repo-local workflows, use the staged script directories:

```sh
bash runs/teis/01_deploy_default.sh
bash runs/teis/02_start_machine.sh
bash runs/teis/03_health_check.sh
bash runs/qwns/01_deploy_uniform.sh
bash runs/qwns/02_start_machine.sh
bash runs/qwns/03_health_check.sh
bash runs/qsrs/01_deploy_default.sh
bash runs/qsrs/02_start_machine.sh
bash runs/qsrs/03_health_check.sh
bash runs/recovery/restart_tei_qwn.sh
```

- `runs/teis/README.md`: staged TEI deploy, health, benchmark, cleanup
- `runs/qwns/README.md`: staged QWN deploy, health, benchmark, cleanup
- `runs/qsrs/README.md`: staged QSR deploy, health, benchmark, cleanup
- `runs/recovery/README.md`: joint TEI + QWN recovery and live validation