import argparse

from tclogger import logger, shell_cmd, log_error
from time import sleep
from typing import Union

MIN_FAN_PERCENT = 30
MAX_FAN_PERCENT = 100

NV_SETTINGS = "DISPLAY=:0 nvidia-settings"
GREP_GPU = "grep -e 'gpu:'"
GREP_FAN = "grep -e 'fan:'"


def parse_gpu_idx(idx: Union[str, int]) -> int:
    try:
        gpu_idx = int(idx)
        return gpu_idx
    except Exception as e:
        log_error(f"× Invalid gpu idx input: {idx}")
        return None


def parse_fan_idx(idx: Union[str, int]) -> int:
    try:
        fan_idx = int(idx)
        return fan_idx
    except Exception as e:
        log_error(f"× Invalid fan idx input: {idx}")
        return None


def parse_fan_percent(fan_percent: int) -> int:
    """GPUTargetFanSpeed: 0 ~ 100"""
    try:
        fan_percent = int(fan_percent)
    except Exception as e:
        log_error(f"× Invalid GPUTargetFanSpeed input: {fan_percent}")
        return None
    if not (0 <= fan_percent <= 100):
        log_error(f"× Invalid GPUTargetFanSpeed input: {fan_percent}")
        return None
    return int(min(max(fan_percent, MIN_FAN_PERCENT), MAX_FAN_PERCENT))


def parse_fan_control_state(state: Union[str, int]) -> int:
    """GPUFanControlState: 0, 1"""
    if state not in [0, 1, "0", "1"]:
        log_error(f"× Invalid GPUFanControlState input: {state}")
        return None
    return int(state)


def get_gpus() -> str:
    """Get GPU list"""
    cmd = f"{NV_SETTINGS} -q gpus | {GREP_GPU}"
    output: str = shell_cmd(cmd, getoutput=True, showcmd=True)
    logger.okay(output)
    return output


def get_gpu_core_temp(gpu_idx: int = 0, verbose: bool = False) -> str:
    if verbose:
        cmd = f"{NV_SETTINGS} -q '[gpu:{gpu_idx}]/GPUCoreTemp' | {GREP_GPU}"
    else:
        cmd = f"{NV_SETTINGS} -q '[gpu:{gpu_idx}]/GPUCoreTemp' -t"
    output: str = shell_cmd(cmd, getoutput=True, showcmd=True)
    logger.okay(output)
    return output


def get_gpu_fan_control_state(gpu_idx: int = 0, verbose: bool = False) -> str:
    """Get GPU fan control state"""
    if verbose:
        cmd = f"{NV_SETTINGS} -q '[gpu:{gpu_idx}]/GPUFanControlState' | {GREP_GPU}"
    else:
        cmd = f"{NV_SETTINGS} -q '[gpu:{gpu_idx}]/GPUFanControlState' -t"
    output: str = shell_cmd(cmd, getoutput=True, showcmd=True)
    logger.okay(output)
    return output


def set_gpu_fan_control_state(gpu_idx: int = 0, state: int = 1):
    """Set GPU fan control state"""
    cmd = f"{NV_SETTINGS} -a '[gpu:{gpu_idx}]/GPUFanControlState={state}' | {GREP_GPU}"
    output: str = shell_cmd(cmd, getoutput=True, showcmd=True)
    logger.okay(output)


def get_gpu_fan_speed_percent(fan_idx: int = 0, verbose: bool = False) -> str:
    """Get GPU fan speed percentage"""
    if verbose:
        cmd = f"{NV_SETTINGS} -q '[fan:{fan_idx}]/GPUCurrentFanSpeed' | {GREP_FAN}"
    else:
        cmd = f"{NV_SETTINGS} -q '[fan:{fan_idx}]/GPUCurrentFanSpeed' -t"
    output: str = shell_cmd(cmd, getoutput=True, showcmd=True)
    logger.okay(output)
    return output


def set_gpu_fan_speed_percent(fan_idx: int, fan_percent: int):
    """Set GPU fan speed percentage"""
    fan_percent = parse_fan_percent(fan_percent)
    cmd = f"{NV_SETTINGS} -a '[fan:{fan_idx}]/GPUTargetFanSpeed={fan_percent}' | {GREP_FAN}"
    output: str = shell_cmd(cmd, getoutput=True, showcmd=True)
    logger.okay(output)


class GPUFanArgParser(argparse.ArgumentParser):
    def __init__(self):
        super().__init__(description="GPU Fan Control")
        # info args
        self.add_argument("-gs", "--gpus", action="store_true")
        self.add_argument("-gt", "--gpu-temp", action="store_true")
        self.add_argument("-fs", "--fan-speed", action="store_true")  # support set
        self.add_argument("-cs", "--control-state", action="store_true")  # support set
        # idx args
        self.add_argument("-gi", "--gpu-idx", type=str)
        self.add_argument("-fi", "--fan-idx", type=str)
        # set args
        self.add_argument("-s", "--set", type=int)
        # verbose args
        self.add_argument("-v", "--verbose", action="store_true")
        self.args, _ = self.parse_known_args()


def control_gpu_fan():
    args = GPUFanArgParser().args
    verbose = args.verbose

    if args.gpus:
        get_gpus()

    if args.gpu_temp:
        gpu_idx = parse_gpu_idx(args.gpu_idx)
        get_gpu_core_temp(gpu_idx, verbose=verbose)

    if args.control_state:
        gpu_idx = parse_gpu_idx(args.gpu_idx)
        if args.set:
            value = parse_fan_control_state(args.set)
            set_gpu_fan_control_state(gpu_idx, value)
        else:
            get_gpu_fan_control_state(gpu_idx, verbose=verbose)

    if args.fan_speed:
        fan_idx = parse_fan_idx(args.fan_idx)
        if args.set is not None:
            fan_percent = parse_fan_percent(args.set)
            set_gpu_fan_speed_percent(fan_idx, fan_percent)
        else:
            get_gpu_fan_speed_percent(fan_idx, verbose=verbose)


if __name__ == "__main__":
    control_gpu_fan()

    # Case: Get GPUs list
    # python -m tfmx.gpu_fan -gs

    # Case: Get GPU0 core temperature
    # python -m tfmx.gpu_fan -gt -gi 0
    # python -m tfmx.gpu_fan -gt -gi 0 -v

    # Case: Get/Set GPU0 fan control state
    # python -m tfmx.gpu_fan -cs -gi 0
    # python -m tfmx.gpu_fan -cs -gi 0 -v
    # python -m tfmx.gpu_fan -cs -gi 0 -s 1

    # Case: Get/Set Fan0 speed percentage
    # python -m tfmx.gpu_fan -fs -fi 0
    # python -m tfmx.gpu_fan -fs -fi 0 -v
    # python -m tfmx.gpu_fan -fs -fi 0 -s 50
