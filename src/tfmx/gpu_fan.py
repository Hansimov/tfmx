import argparse

from tclogger import logger, shell_cmd, log_error
from typing import Union, Literal

MIN_FAN_PERCENT = 0
MAX_FAN_PERCENT = 100

NV_SETTINGS = "DISPLAY=:0 nvidia-settings"
GREP_GPU = "grep -Ei 'gpu:'"
GREP_FAN = "grep -Ei 'fan:'"
GREP_GPU_OR_FAN = "grep -Ei '(gpu:|fan:)'"


def parse_idx(idx: Union[str, int]) -> int:
    try:
        idx = int(idx)
        return idx
    except Exception as e:
        log_error(f"× Invalid idx: {idx}")
        return None


def parse_val(val: Union[str, int]) -> int:
    try:
        val = int(val)
        return val
    except Exception as e:
        log_error(f"× Invalid val: {val}")
        return None


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


def parse_control_state(state: Union[str, int]) -> int:
    """GPUFanControlState: 0, 1"""
    if state not in [0, 1, "0", "1"]:
        log_error(f"× Invalid GPUFanControlState input: {state}")
        return None
    return int(state)


def parse_fan_speed(fan_speed: int) -> int:
    """GPUTargetFanSpeed: 0 ~ 100"""
    try:
        fan_speed = int(fan_speed)
    except Exception as e:
        log_error(f"× Invalid GPUTargetFanSpeed input: {fan_speed}")
        return None
    if not (0 <= fan_speed <= 100):
        log_error(f"× Invalid GPUTargetFanSpeed input: {fan_speed}")
        return None
    return int(min(max(fan_speed, MIN_FAN_PERCENT), MAX_FAN_PERCENT))


OpKeyType = Literal["gpus", "control_state", "core_temp", "fan_speed"]
NvKeyType = Literal["gpus", "GPUFanControlState", "GPUCoreTemp", "GPUCurrentFanSpeed"]
DeviceType = Literal["gpu", "fan"]
OpsType = list[tuple[Literal["set", "get"], int, Union[int, None]]]

OP_NV_KEYS = {
    "gpus": "gpus",
    "core_temp": "GPUCoreTemp",
    "control_state": "GPUFanControlState",
    "fan_speed": "GPUCurrentFanSpeed",
}
OP_DEVICES = {
    "gpus": "gpu",
    "core_temp": "gpu",
    "control_state": "gpu",
    "fan_speed": "fan",
}


class NvidiaSettingsParser:
    def idx_vals_to_ops(self, s: str) -> OpsType:
        """Usages:
        * "-fs 0":    get fan 0 speed
        * "-fs 0,1":  get fan 0 and 1 speed
        * "-fs a":    get all fans speed
        * "-fs 0:40":          set fan 0 speed to 40%
        * "-fs 0,1:50":        set fan 0 and 1 speed to 50%
        * "-fs 0,1:50;2,3:80": set fan 0 and 1 speed to 50%, set fan 2 and 3 speed to 80%
        * "-fs a:70":          set all fans speed to 70%
        * "-cs 0":   get gpu 0 fan control state
        * "-cs 0,1": get gpu 0 and 1 fan control state
        * "-cs a":   get all gpu fan control state
        * "-cs 0:1":         set gpu 0 fan control state to 1
        * "-cs 0,1:1":       set gpu 0 and 1 fan control state to 1
        * "-cs 0,1:1;2,3:0": set gpu 0 and 1 fan control state to 1, set gpu 2 and 3 fan control state to 0
        * "-cs a:1":         set all gpu fan control state to 1

        Syntax:
        * ";": sep <idx>:<val> groups
        * ",": sep idxs
        * ":": sep <idx> and <val>
        """
        ops: OpsType = []
        idx_vals = s.split(";")
        for idx_val in idx_vals:
            idx, val = idx_val.split(":")
            if idx.lower().startswith("a"):
                idx = "a"
            else:
                idx = parse_idx(idx)
            if val == "":
                op, val = "get", None
            else:
                op, val = "set", parse_val(val)
            ops.append((op, idx, val))
        return ops

    def ops_to_nv_args(self, ops: OpsType, op_key: OpKeyType) -> list[str]:
        nv_args: list[str] = []
        nv_key: NvKeyType = OP_NV_KEYS[op_key]
        dv_key: DeviceType = OP_DEVICES[op_key]
        for op, idx, val in ops:
            if idx == "a":
                if op == "get":
                    nv_arg = f"-q '{nv_key}'"
                else:  # set
                    nv_arg = f"-a '{nv_key}={val}'"
            else:  # idx
                if op == "get":
                    nv_arg = f"-q '[{dv_key}:{idx}]/{nv_key}'"
                else:  # set
                    nv_arg = f"-a '[{dv_key}:{idx}]/{nv_key}={val}'"
            nv_args.append(nv_arg)
        return nv_args


class GPUFanController:
    def __init__(self, verbose: bool = False, terse: bool = False):
        self.verbose = verbose
        self.terse = terse

    def get_gpus(self) -> str:
        """Get GPU list"""
        get_s = "-q gpus"
        cmd = f"{NV_SETTINGS} {get_s} | {GREP_GPU}"
        output: str = shell_cmd(cmd, getoutput=True, showcmd=self.verbose)
        logger.okay(output, verbose=self.verbose)
        return output

    def get_core_temp(self, gpu_idx: int) -> str:
        get_s = f"-q '[gpu:{gpu_idx}]/GPUCoreTemp'"
        if self.terse:
            cmd = f"{NV_SETTINGS} {get_s} -t"
        else:
            cmd = f"{NV_SETTINGS} {get_s} | {GREP_GPU}"
        output: str = shell_cmd(cmd, getoutput=True, showcmd=self.verbose)
        logger.okay(output, verbose=self.verbose)
        return output

    def get_control_state(self, gpu_idx: int) -> str:
        """Get GPU fan control state"""
        get_s = f"-q '[gpu:{gpu_idx}]/GPUFanControlState'"
        if self.terse:
            cmd = f"{NV_SETTINGS} {get_s} -t"
        else:
            cmd = f"{NV_SETTINGS} {get_s} | {GREP_GPU}"
        output: str = shell_cmd(cmd, getoutput=True, showcmd=self.verbose)
        logger.okay(output, verbose=self.verbose)
        return output

    def set_control_state(self, gpu_idx: int, control_state: int):
        """Set GPU fan control state"""
        set_s = f"-a '[gpu:{gpu_idx}]/GPUFanControlState={control_state}'"
        cmd = f"{NV_SETTINGS} {set_s} | {GREP_GPU}"
        output: str = shell_cmd(cmd, getoutput=True, showcmd=self.verbose)
        logger.okay(output, verbose=self.verbose)

    def get_fan_speed(self, fan_idx: int) -> str:
        """Get GPU fan speed percentage"""
        get_s = f"-q '[fan:{fan_idx}]/GPUCurrentFanSpeed'"
        if self.terse:
            cmd = f"{NV_SETTINGS} {get_s} -t"
        else:
            cmd = f"{NV_SETTINGS} {get_s} | {GREP_FAN}"
        output: str = shell_cmd(cmd, getoutput=True, showcmd=self.verbose)
        logger.okay(output, verbose=self.verbose)
        return output

    def set_fan_speed(self, fan_idx: int, fan_speed: int):
        """Set GPU fan speed percentage"""
        set_s = f"-a '[fan:{fan_idx}]/GPUTargetFanSpeed={fan_speed}'"
        cmd = f"{NV_SETTINGS} {set_s} | {GREP_FAN}"
        output: str = shell_cmd(cmd, getoutput=True, showcmd=self.verbose)
        logger.okay(output, verbose=self.verbose)


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
        # log args
        self.add_argument("-q", "--quiet", action="store_true")
        self.add_argument("-t", "--terse", action="store_true")
        self.args, _ = self.parse_known_args()


def control_gpu_fan():
    args = GPUFanArgParser().args
    c = GPUFanController(verbose=not args.quiet, terse=args.terse)

    if args.gpus:
        c.get_gpus()

    if args.gpu_temp:
        gpu_idx = parse_gpu_idx(args.gpu_idx)
        c.get_core_temp(gpu_idx)

    if args.control_state:
        gpu_idx = parse_gpu_idx(args.gpu_idx)
        if args.set:
            value = parse_control_state(args.set)
            c.set_control_state(gpu_idx, value)
        else:
            c.get_control_state(gpu_idx)

    if args.fan_speed:
        fan_idx = parse_fan_idx(args.fan_idx)
        if args.set is not None:
            fan_speed = parse_fan_speed(args.set)
            c.set_fan_speed(fan_idx, fan_speed)
        else:
            c.get_fan_speed(fan_idx)


if __name__ == "__main__":
    control_gpu_fan()

    # Case: Get GPUs list
    # python -m tfmx.gpu_fan -gs

    # Case: Get GPU0 core temperature
    # python -m tfmx.gpu_fan -gt -gi 0
    # python -m tfmx.gpu_fan -gt -gi 0 -q
    # python -m tfmx.gpu_fan -gt -gi 0 -t

    # Case: Get/Set GPU0 fan control state
    # python -m tfmx.gpu_fan -cs -gi 0
    # python -m tfmx.gpu_fan -cs -gi 0 -q
    # python -m tfmx.gpu_fan -cs -gi 0 -t
    # python -m tfmx.gpu_fan -cs -gi 0 -s 1

    # Case: Get/Set Fan0 speed percentage
    # python -m tfmx.gpu_fan -fs -fi 0
    # python -m tfmx.gpu_fan -fs -fi 0 -q
    # python -m tfmx.gpu_fan -fs -fi 0 -t
    # python -m tfmx.gpu_fan -fs -fi 0 -s 50
