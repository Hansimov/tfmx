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
tei machine run --auto-start --perf-track
```

Run tei benchmark:

```sh
tei benchmark run -E "http://localhost:28800" -n 100000
```

Run qwn compose, machine, and benchmark:

```sh
export QWN_MACHINE_URL="http://$QWN_HOST:27800"

qwn compose up --gpu-configs "0:4b:4bit"
qwn compose up --gpu-layout uniform-awq
qwn machine run --auto-start -b
qwn client chat "你好，请做个自我介绍"
qwn benchmark run -E "$QWN_MACHINE_URL" -n 100
```