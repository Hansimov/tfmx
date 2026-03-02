import argparse
import os

from tclogger import logger, shell_cmd, log_error
from typing import Union, Literal

MIN_POWER_LIMIT = 100
MAX_POWER_LIMIT = 500

NVIDIA_SMI = "nvidia-smi"

OpKeyType = Literal["persistence_mode", "power_limit"]
OpsType = list[tuple[OpKeyType, Literal["set", "get"], str, Union[str, None]]]
NO_IDX_OP_KEYS = []
NO_VAL_OP_KEYS = []


def is_none_or_empty(val: Union[str, None]) -> bool:
    """val is None or empty"""
    return val is None or val.strip() == ""


def is_str_and_all(idx: str) -> bool:
    """idx starts with 'a'"""
    if isinstance(idx, str) and idx.strip().lower().startswith("a"):
        return True
    return False


def is_op_key_has_no_idx(op_key: str) -> bool:
    """op_key should not have idx"""
    return op_key in NO_IDX_OP_KEYS


def is_op_key_has_no_val(op_key: str) -> bool:
    """op_key should not have val"""
    return op_key in NO_VAL_OP_KEYS


def parse_idx(idx: Union[str, int]) -> int:
    try:
        idx = int(idx)
        return idx
    except Exception as e:
        log_error(f"× Invalid idx: {idx}")
        return None


def parse_val(val: Union[str, int]) -> Union[int, None]:
    if is_none_or_empty(val):
        return None
    try:
        val = int(val)
        return val
    except Exception as e:
        log_error(f"× Invalid val: {val}")
        return None


def parse_persistence_mode(mode: Union[str, int]) -> Union[int, None]:
    """Persistence Mode: 0 (Disabled), 1 (Enabled)"""
    if is_none_or_empty(mode):
        return None
    if mode not in [0, 1, "0", "1"]:
        log_error(f"× Invalid persistence_mode: {mode}")
        return None
    return int(mode)


def parse_power_limit(power_limit: Union[str, int]) -> Union[int, None]:
    """Power Limit in Watts"""
    if is_none_or_empty(power_limit):
        return None
    try:
        power_limit = int(power_limit)
    except Exception as e:
        log_error(f"× Invalid power_limit: {power_limit}")
        return None
    if not (MIN_POWER_LIMIT <= power_limit <= MAX_POWER_LIMIT):
        log_error(
            f"× Invalid power_limit: {power_limit}, must be {MIN_POWER_LIMIT}~{MAX_POWER_LIMIT}"
        )
        return None
    return power_limit


def parse_val_by_op_key(val: str, op_key: OpKeyType) -> Union[int, None]:
    if op_key == "persistence_mode":
        return parse_persistence_mode(val)
    elif op_key == "power_limit":
        return parse_power_limit(val)
    else:
        return parse_val(val)


def build_sudo_cmd(cmd: str) -> str:
    """Build a command with sudo, using SUDOPASS if available.
    Uses $SUDOPASS environment variable to avoid interactive password prompt.
    If SUDOPASS is not set, uses regular sudo (prompts for password).
    """
    sudopass = os.environ.get("SUDOPASS", "")
    if sudopass:
        escaped_cmd = cmd.replace("'", "'\\''")
        return f"bash -c 'echo \"$SUDOPASS\" | sudo -S {escaped_cmd} 2>/dev/null'"
    return f"sudo {cmd}"


class NvidiaSmiParser:
    def key_idx_val_to_ops(self, op_key: OpKeyType, idx_val_str: str) -> OpsType:
        """Usages:
        * "-pm 0":    get gpu 0 persistence mode
        * "-pm 0,1":  get gpu 0 and 1 persistence mode
        * "-pm a":    get all gpus persistence mode
        * "-pm 0:1":          set gpu 0 persistence mode to 1 (Enabled)
        * "-pm 0,1:1":        set gpu 0 and 1 persistence mode to 1
        * "-pm 0,1:1;2,3:0":  set gpu 0 and 1 persistence mode to 1, set gpu 2 and 3 to 0
        * "-pm a:1":          set all gpus persistence mode to 1
        * "-pl 0":    get gpu 0 power limit
        * "-pl 0,1":  get gpu 0 and 1 power limit
        * "-pl a":    get all gpus power limit
        * "-pl 0:290":         set gpu 0 power limit to 290W
        * "-pl 0,1:290":       set gpu 0 and 1 power limit to 290W
        * "-pl 0,1:290;2,3:280": set gpu 0 and 1 power limit to 290W, set gpu 2 and 3 to 280W
        * "-pl a:290":         set all gpus power limit to 290W

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


class GPUPowerController:
    def __init__(self, verbose: bool = False, terse: bool = False):
        self.verbose = verbose
        self.terse = terse

    def get_persistence_mode(self, gpu_idx: Union[int, str] = None) -> str:
        """Get GPU persistence mode"""
        if gpu_idx is None or is_str_and_all(gpu_idx):
            idx_s = ""
        else:
            idx_s = f" -i {gpu_idx}"
        if self.terse:
            cmd = f"{NVIDIA_SMI}{idx_s} --query-gpu=index,persistence_mode --format=csv,noheader"
        else:
            cmd = f"{NVIDIA_SMI}{idx_s} --query-gpu=index,name,persistence_mode --format=csv"
        output: str = shell_cmd(cmd, getoutput=True, showcmd=self.verbose)
        logger.okay(output, verbose=self.verbose)
        return output

    def set_persistence_mode(self, gpu_idx: Union[int, str], mode: int) -> str:
        """Set GPU persistence mode"""
        if is_str_and_all(gpu_idx):
            idx_s = ""
        else:
            idx_s = f" -i {gpu_idx}"
        base_cmd = f"{NVIDIA_SMI}{idx_s} -pm {mode}"
        cmd = build_sudo_cmd(base_cmd)
        output: str = shell_cmd(cmd, getoutput=True, showcmd=self.verbose)
        logger.okay(output, verbose=self.verbose)
        return output

    def get_power_limit(self, gpu_idx: Union[int, str] = None) -> str:
        """Get GPU power limit"""
        if gpu_idx is None or is_str_and_all(gpu_idx):
            idx_s = ""
        else:
            idx_s = f" -i {gpu_idx}"
        if self.terse:
            cmd = f"{NVIDIA_SMI}{idx_s} --query-gpu=index,power.limit --format=csv,noheader"
        else:
            cmd = f"{NVIDIA_SMI}{idx_s} --query-gpu=index,name,power.limit,power.min_limit,power.max_limit --format=csv"
        output: str = shell_cmd(cmd, getoutput=True, showcmd=self.verbose)
        logger.okay(output, verbose=self.verbose)
        return output

    def set_power_limit(self, gpu_idx: Union[int, str], power_limit: int) -> str:
        """Set GPU power limit in Watts"""
        if is_str_and_all(gpu_idx):
            idx_s = ""
        else:
            idx_s = f" -i {gpu_idx}"
        base_cmd = f"{NVIDIA_SMI}{idx_s} -pl {power_limit}"
        cmd = build_sudo_cmd(base_cmd)
        output: str = shell_cmd(cmd, getoutput=True, showcmd=self.verbose)
        logger.okay(output, verbose=self.verbose)
        return output

    def execute_ops(self, ops: OpsType):
        """Execute operations"""
        for op_key, op, idx, val in ops:
            if op_key == "persistence_mode":
                if op == "get":
                    self.get_persistence_mode(idx)
                else:  # set
                    mode = parse_persistence_mode(val)
                    if mode is not None:
                        self.set_persistence_mode(idx, mode)
            elif op_key == "power_limit":
                if op == "get":
                    self.get_power_limit(idx)
                else:  # set
                    power_limit = parse_power_limit(val)
                    if power_limit is not None:
                        self.set_power_limit(idx, power_limit)


class GPUPowerArgParser(argparse.ArgumentParser):
    def __init__(self):
        super().__init__(description="GPU Power Control")
        # op_key args
        self.add_argument("-pm", "--persistence-mode", type=str)
        self.add_argument("-pl", "--power-limit", type=str)
        # log args
        self.add_argument("-q", "--quiet", action="store_true")
        self.add_argument("-t", "--terse", action="store_true")
        self.args, _ = self.parse_known_args()


def main():
    args = GPUPowerArgParser().args
    c = GPUPowerController(verbose=not args.quiet, terse=args.terse)
    p = NvidiaSmiParser()
    kivs = []
    if args.persistence_mode is not None:
        kivs.append(("persistence_mode", args.persistence_mode))
    if args.power_limit is not None:
        kivs.append(("power_limit", args.power_limit))
    for op_key, idx_val_str in kivs:
        ops = p.key_idx_val_to_ops(op_key, idx_val_str)
        c.execute_ops(ops)


if __name__ == "__main__":
    main()

    # Case: Get/Set GPU persistence mode
    # gpu_pow -pm 0
    # gpu_pow -pm a
    # gpu_pow -pm "0,1"
    # gpu_pow -pm 0:1
    # gpu_pow -pm a:1
    # gpu_pow -pm "0,1:1;2,3:0"
    # gpu_pow -pm 0 -q
    # gpu_pow -pm 0 -t

    # Case: Get/Set GPU power limit
    # gpu_pow -pl 0
    # gpu_pow -pl a
    # gpu_pow -pl "0,1"
    # gpu_pow -pl 0:290
    # gpu_pow -pl a:290
    # gpu_pow -pl "0,1:290;2,3:280"
    # gpu_pow -pl 0 -q
    # gpu_pow -pl 0 -t
