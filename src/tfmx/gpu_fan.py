import argparse

from tclogger import logger, shell_cmd, log_error
from typing import Union, Literal

from tfmx.gpu_ctl import (
    MIN_FAN_PERCENT,
    MAX_FAN_PERCENT,
    get_nv_settings_cmd,
    check_nv_permission,
    is_none_or_empty,
    is_str_and_all,
    parse_idx,
)

GREP_GPU = "grep -Ei 'gpu:'"
GREP_FAN = "grep -Ei 'fan:'"
GREP_GPU_OR_FAN = "grep -Ei '(gpu:|fan:)'"

OpKeyType = Literal["gpus", "control_state", "core_temp", "fan_speed"]
NvKeyType = Literal[
    "gpus",
    "GPUFanControlState",
    "GPUCoreTemp",
    "GPUCurrentFanSpeed",
    "GPUTargetFanSpeed",
]
DeviceType = Literal["gpu", "fan"]
OpsType = list[tuple[OpKeyType, Literal["set", "get"], str, Union[str, None]]]
NO_IDX_OP_KEYS = ["gpus"]
NO_VAL_OP_KEYS = ["gpus", "core_temp"]

OP_NV_GET_KEYS = {
    "gpus": "gpus",
    "core_temp": "GPUCoreTemp",
    "control_state": "GPUFanControlState",
    "fan_speed": "GPUCurrentFanSpeed",
}
OP_NV_SET_KEYS = {
    "gpus": "gpus",
    "core_temp": "GPUCoreTemp",
    "control_state": "GPUFanControlState",
    "fan_speed": "GPUTargetFanSpeed",
}
OP_DEVICES = {
    "gpus": "gpu",
    "core_temp": "gpu",
    "control_state": "gpu",
    "fan_speed": "fan",
}


def is_op_key_has_no_idx(op_key: str) -> bool:
    """op_key should not have idx"""
    return op_key in NO_IDX_OP_KEYS


def is_op_key_has_no_val(op_key: str) -> bool:
    """op_key should not have val"""
    return op_key in NO_VAL_OP_KEYS


def parse_val(val: Union[str, int]) -> Union[int, None]:
    if is_none_or_empty(val):
        return None
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


def parse_control_state(control_state: Union[str, int]) -> Union[int, None]:
    """GPUFanControlState: 0, 1"""
    if is_none_or_empty(control_state):
        return None
    if control_state not in [0, 1, "0", "1"]:
        log_error(f"× Invalid control_state: {control_state}")
        return None
    return int(control_state)


def parse_fan_speed(fan_speed: int) -> int:
    """GPUTargetFanSpeed: 0 ~ 100"""
    if is_none_or_empty(fan_speed):
        return None
    try:
        fan_speed = int(fan_speed)
    except Exception as e:
        log_error(f"× Invalid fan_speed: {fan_speed}")
        return None
    if not (0 <= fan_speed <= 100):
        log_error(f"× Invalid fan_speed: {fan_speed}")
        return None
    return int(min(max(fan_speed, MIN_FAN_PERCENT), MAX_FAN_PERCENT))


def parse_val_by_op_key(val: str, op_key: OpKeyType) -> Union[int, None]:
    if op_key == "fan_speed":
        return parse_fan_speed(val)
    elif op_key == "control_state":
        return parse_control_state(val)
    else:
        return parse_val(val)


class NvidiaSettingsParser:
    def key_idx_val_to_ops(self, op_key: OpKeyType, idx_val_str: str) -> OpsType:
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

        Return:
        * list of (op_key, op:"get"/"set", idx:str, val:str)
        """
        if is_op_key_has_no_idx(op_key):
            return [(op_key, "get", None, None)]
        ops: OpsType = []
        idx_vals = idx_val_str.split(";")
        for idx_val in idx_vals:
            sep_res = idx_val.split(":", maxsplit=1)
            idx_str = sep_res[0]
            if len(sep_res) == 1 or is_op_key_has_no_val(op_key):
                op, val = "get", None
            else:
                op, val = "set", sep_res[1]
            idxs = idx_str.split(",")
            for idx in idxs:
                ops.append((op_key, op, idx, val))
        return ops

    def ops_to_nv_args(self, ops: OpsType) -> list[str]:
        nv_args: list[str] = []
        for op_key, op, idx, val in ops:
            if op == "get":
                nv_key: NvKeyType = OP_NV_GET_KEYS[op_key]
            else:  # set
                nv_key: NvKeyType = OP_NV_SET_KEYS[op_key]
            dv_key: DeviceType = OP_DEVICES[op_key]
            if is_op_key_has_no_idx(op_key) or is_str_and_all(idx):
                key_str = f"{nv_key}"
            else:
                idx = parse_idx(idx)
                key_str = f"[{dv_key}:{idx}]/{nv_key}"
            if op == "get":
                nv_arg = f"-q '{key_str}'"
            else:  # set
                val = parse_val_by_op_key(val, op_key)
                nv_arg = f"-a '{key_str}={val}'"
            nv_args.append(nv_arg)
        return nv_args


class GPUFanController:
    def __init__(self, verbose: bool = False, terse: bool = False):
        self.verbose = verbose
        self.terse = terse

    def get_gpus(self) -> str:
        """Get GPU list"""
        cmd = get_nv_settings_cmd("-q gpus", f"| {GREP_GPU}")
        output: str = shell_cmd(cmd, getoutput=True, showcmd=self.verbose)
        logger.okay(output, verbose=self.verbose)
        return output

    def get_core_temp(self, gpu_idx: int) -> str:
        nv_args = f"-q '[gpu:{gpu_idx}]/GPUCoreTemp'"
        if self.terse:
            cmd = get_nv_settings_cmd(f"{nv_args} -t")
        else:
            cmd = get_nv_settings_cmd(nv_args, f"| {GREP_GPU}")
        output: str = shell_cmd(cmd, getoutput=True, showcmd=self.verbose)
        logger.okay(output, verbose=self.verbose)
        return output

    def get_control_state(self, gpu_idx: int) -> str:
        """Get GPU fan control state"""
        nv_args = f"-q '[gpu:{gpu_idx}]/GPUFanControlState'"
        if self.terse:
            cmd = get_nv_settings_cmd(f"{nv_args} -t")
        else:
            cmd = get_nv_settings_cmd(nv_args, f"| {GREP_GPU}")
        output: str = shell_cmd(cmd, getoutput=True, showcmd=self.verbose)
        logger.okay(output, verbose=self.verbose)
        return output

    def set_control_state(self, gpu_idx: int, control_state: int):
        """Set GPU fan control state"""
        nv_args = f"-a '[gpu:{gpu_idx}]/GPUFanControlState={control_state}'"
        cmd = get_nv_settings_cmd(nv_args, f"| {GREP_GPU}")
        output: str = shell_cmd(cmd, getoutput=True, showcmd=self.verbose)
        logger.okay(output, verbose=self.verbose)

    def get_fan_speed(self, fan_idx: int) -> str:
        """Get GPU fan speed percentage"""
        nv_args = f"-q '[fan:{fan_idx}]/GPUCurrentFanSpeed'"
        if self.terse:
            cmd = get_nv_settings_cmd(f"{nv_args} -t")
        else:
            cmd = get_nv_settings_cmd(nv_args, f"| {GREP_FAN}")
        output: str = shell_cmd(cmd, getoutput=True, showcmd=self.verbose)
        logger.okay(output, verbose=self.verbose)
        return output

    def set_fan_speed(self, fan_idx: int, fan_speed: int):
        """Set GPU fan speed percentage"""
        nv_args = f"-a '[fan:{fan_idx}]/GPUTargetFanSpeed={fan_speed}'"
        cmd = get_nv_settings_cmd(nv_args, f"| {GREP_FAN}")
        output: str = shell_cmd(cmd, getoutput=True, showcmd=self.verbose)
        logger.okay(output, verbose=self.verbose)

    def execute(self, nv_args: list[str]):
        nv_args_str = " ".join(nv_args)
        if self.terse:
            cmd = get_nv_settings_cmd(f"{nv_args_str} -t")
        else:
            cmd = get_nv_settings_cmd(nv_args_str, f"| {GREP_GPU_OR_FAN}")
        output: str = shell_cmd(cmd, getoutput=True, showcmd=self.verbose)
        logger.okay(output, verbose=self.verbose)


class GPUFanArgParser(argparse.ArgumentParser):
    def __init__(self):
        super().__init__(description="GPU Fan Control")
        # op_key args
        self.add_argument("-gs", "--gpus", action="store_true")
        self.add_argument("-gt", "--gpu-temp", type=str)
        self.add_argument("-cs", "--control-state", type=str)
        self.add_argument("-fs", "--fan-speed", type=str)
        # log args
        self.add_argument("-q", "--quiet", action="store_true")
        self.add_argument("-t", "--terse", action="store_true")
        self.args, _ = self.parse_known_args()


def main():
    args = GPUFanArgParser().args
    # Auto-detect permission requirements (will use sudo if needed)
    check_nv_permission()
    c = GPUFanController(verbose=not args.quiet, terse=args.terse)
    p = NvidiaSettingsParser()
    kivs = []
    if args.gpus:
        kivs.append(("gpus", ""))
    if args.gpu_temp:
        kivs.append(("core_temp", args.gpu_temp))
    if args.control_state is not None:
        kivs.append(("control_state", args.control_state))
    if args.fan_speed is not None:
        kivs.append(("fan_speed", args.fan_speed))
    for op_key, idx_val_str in kivs:
        ops = p.key_idx_val_to_ops(op_key, idx_val_str)
    nv_args = p.ops_to_nv_args(ops)
    c.execute(nv_args)


if __name__ == "__main__":
    main()

    # Case: Get GPUs list
    # gpu_fan -gs

    # Case: Get GPU0 core temperature
    # gpu_fan -gt 0
    # gpu_fan -gt 0,1
    # gpu_fan -gt a
    # gpu_fan -gt 0 -q
    # gpu_fan -gt 0 -t

    # Case: Get/Set GPU0 fan control state
    # gpu_fan -cs 0
    # gpu_fan -cs a
    # gpu_fan -cs a:1
    # gpu_fan -cs 0:1
    # gpu_fan -cs 0 -q
    # gpu_fan -cs 0 -t

    # Case: Get/Set Fan0 speed percentage
    # gpu_fan -fs 0
    # gpu_fan -fs 0 -q
    # gpu_fan -fs 0 -t
    # gpu_fan -fs 0,1:50
    # gpu_fan -fs a:80
    # gpu_fan -fs "0,1:35;2,3:30"
