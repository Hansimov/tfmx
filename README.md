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
gpu_pow -pm a:1 && gpu_pow -pl "0:160;1:240"

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
tei_compose up && sleep 45 && tei_machine run --perf-track
```

Run tei benchmark:

```sh
tei_benchmark -E "http://localhost:28800" -n 1000000 run
```