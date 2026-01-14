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
gpu_pow -pl a:240
gpu_pow -pl 0:240
```

Set GPU monitor:

```sh
gpu_mon -c a:30-50/50-65/60-80/75-100 -s
```