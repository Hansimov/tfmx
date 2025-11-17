import argparse

from tclogger import logger, shell_cmd, log_error
from time import sleep
from typing import Union

MIN_FAN_PERCENT = 30
MAX_FAN_PERCENT = 100

NV_SETTINGS = "DISPLAY=:0 nvidia-settings"
GREP_GPU = "grep -e 'gpu:'"
GREP_FAN = "grep -e 'fan:'"


def norm_fan_percent(fan_percent: int) -> int:
    """GPUTargetFanSpeed: 0 ~ 100"""
    return int(min(max(fan_percent, MIN_FAN_PERCENT), MAX_FAN_PERCENT))


def norm_fan_control_state(state: Union[str, int]) -> int:
    """GPUFanControlState: 0, 1"""
    if state not in [0, 1, "0", "1"]:
        log_error(f"× Invalid GPUFanControlState input: {state}")
    return int(state)


def get_gpus() -> str:
    """Get GPU list"""
    cmd = f"{NV_SETTINGS} -q gpus | {GREP_GPU}"
    output: str = shell_cmd(cmd, getoutput=True, showcmd=True)
    logger.okay(output)
    return output


def get_gpu_core_temp(gpu_idx: int = 0) -> str:
    # cmd = f"{NV_SETTINGS} -q '[gpu:{gpu_idx}]/GPUCoreTemp' | {GREP_GPU}"
    cmd = f"{NV_SETTINGS} -q '[gpu:{gpu_idx}]/GPUCoreTemp' -t"
    output: str = shell_cmd(cmd, getoutput=True, showcmd=True)
    logger.okay(output)
    return output


def get_gpu_fan_control_state(gpu_idx: int = 0) -> str:
    """Get GPU fan control state"""
    # cmd = f"{NV_SETTINGS} -q '[gpu:{gpu_idx}]/GPUFanControlState' | {GREP_GPU}"
    cmd = f"{NV_SETTINGS} -q '[gpu:{gpu_idx}]/GPUFanControlState' -t"
    output: str = shell_cmd(cmd, getoutput=True, showcmd=True)
    logger.okay(output)
    return output


def set_gpu_fan_control_state(gpu_idx: int = 0, state: int = 1):
    """Set GPU fan control state"""
    cmd = f"{NV_SETTINGS} -a '[gpu:{gpu_idx}]/GPUFanControlState={state}' | {GREP_GPU}"
    output: str = shell_cmd(cmd, getoutput=True, showcmd=True)
    logger.okay(output)


def get_gpu_fan_speed_percent(fan_idx: int = 0) -> str:
    """Get GPU fan speed percentage"""
    # cmd = f"{NV_SETTINGS} -q '[fan:{fan_idx}]/GPUCurrentFanSpeed' | {GREP_FAN}"
    cmd = f"{NV_SETTINGS} -q '[fan:{fan_idx}]/GPUCurrentFanSpeed' -t"
    output: str = shell_cmd(cmd, getoutput=True, showcmd=True)
    logger.okay(output)
    return output


def set_gpu_fan_speed_percent(fan_idx: int, fan_percent: int):
    """Set GPU fan speed percentage"""
    fan_percent = norm_fan_percent(fan_percent)
    cmd = f"{NV_SETTINGS} -a '[fan:{fan_idx}]/GPUTargetFanSpeed={fan_percent}' | {GREP_FAN}"
    output: str = shell_cmd(cmd, getoutput=True, showcmd=True)
    logger.okay(output)


class GPUFanArgParser(argparse.ArgumentParser):
    def __init__(self):
        super().__init__(description="GPU Fan Control")
        # op args
        self.add_argument("-g", "--get", action="store_true")
        self.add_argument("-s", "--set", action="store_true")
        # idx args
        self.add_argument("-gi", "--gpu-idx", type=str)
        self.add_argument("-fi", "--fan-idx", type=str)
        # info args
        self.add_argument("-gs", "--gpus", type=int)
        self.add_argument("-gt", "--gpu-temp", type=int)
        self.add_argument("-fs", "--fan-speed", type=int)
        self.add_argument("-cs", "--control-state", type=int)
        self.args, _ = self.parse_known_args()


def main():
    args = GPUFanArgParser().args
    # get_gpus()
    # get_gpu_core_temp(gpu_idx=0)
    # get_gpu_fan_control_state(gpu_idx=0)
    # set_gpu_fan_control_state(gpu_idx=0)
    # set_gpu_fan_speed_percent(fan_idx=0, fan_percent=40)
    # sleep(2)
    # get_gpu_fan_speed_percent(fan_idx=0)


if __name__ == "__main__":
    main()

    # python -m tfmx.gpu_fan
