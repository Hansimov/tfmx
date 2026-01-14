"""
GPU Control Utilities - Shared code for gpu_fan and gpu_mon
"""

import os
import glob

from tclogger import logger, shell_cmd, log_error
from typing import Union

# Constants
MIN_FAN_PERCENT = 0
MAX_FAN_PERCENT = 100

NV_SMI = "nvidia-smi"

# Global flag to use sudo for nvidia-settings
_USE_SUDO = False
_PERMISSION_CHECKED = False
_HAS_PERMISSION = False
_X_DISPLAY = None  # Cached X display


def get_x_display() -> str:
    """Get the correct X display for nvidia-settings.
    Tries to detect the actual display used by the local X server.
    """
    global _X_DISPLAY
    if _X_DISPLAY is not None:
        return _X_DISPLAY

    # Try to find display from /tmp/.X*-lock files
    lock_files = glob.glob("/tmp/.X*-lock")
    for lock_file in lock_files:
        try:
            # Extract display number from filename like /tmp/.X0-lock
            display_num = lock_file.replace("/tmp/.X", "").replace("-lock", "")
            _X_DISPLAY = f":{display_num}"
            return _X_DISPLAY
        except Exception:
            pass

    # Try common displays
    for display in [":0", ":1", ":2"]:
        _X_DISPLAY = display
        return _X_DISPLAY

    _X_DISPLAY = ":0"
    return _X_DISPLAY


def get_nv_settings_base() -> str:
    """Get nvidia-settings command with correct display"""
    display = get_x_display()
    return f"nvidia-settings --ctrl-display={display}"


def get_xauthority_path() -> str:
    """Get the correct XAUTHORITY path.
    In some environments (like PVE VMs), X server runs as root or via GDM.
    Try multiple possible locations in order of likelihood.
    """
    uid = os.getuid()

    # Try /run/user based paths first (GDM uses this)
    runtime_xauth = f"/run/user/{uid}/gdm/Xauthority"
    if os.path.exists(runtime_xauth):
        return runtime_xauth

    # Check environment variable
    xauth = os.environ.get("XAUTHORITY")
    if xauth and os.path.exists(xauth):
        return xauth

    # Try current user's home
    user_xauth = os.path.expanduser("~/.Xauthority")
    if os.path.exists(user_xauth):
        return user_xauth

    # Try root's Xauthority (common in some setups)
    if os.path.exists("/root/.Xauthority"):
        return "/root/.Xauthority"

    # Fallback
    return runtime_xauth


def build_sudo_cmd(cmd: str) -> str:
    """Build a command with sudo, using SUDOPASS if available.
    Uses $SUDOPASS environment variable to avoid interactive password prompt.
    Also handles X11 permissions by preserving DISPLAY and XAUTHORITY.
    """
    sudopass = os.environ.get("SUDOPASS", "")
    xauth = get_xauthority_path()
    display = get_x_display()

    # Set DISPLAY and XAUTHORITY environment variables
    env_vars = f"DISPLAY={display} XAUTHORITY={xauth}"

    if sudopass:
        # Wrap in bash -c to handle pipe correctly
        escaped_cmd = cmd.replace("'", "'\\''")
        return f"bash -c 'echo \"$SUDOPASS\" | sudo -S env {env_vars} {escaped_cmd} 2>/dev/null'"
    # For interactive sudo
    return f"sudo env {env_vars} {cmd}"


def get_nv_settings_cmd(nv_args: str = "", suffix: str = "") -> str:
    """Get nvidia-settings command with sudo if needed.

    Args:
        nv_args: Arguments to pass to nvidia-settings
        suffix: Additional command suffix (e.g., "| grep xxx")

    Returns:
        Complete command string ready for shell execution

    Note:
        Always sets DISPLAY and XAUTHORITY for X11 access.
        When using sudo, uses build_sudo_cmd for proper env handling.
    """
    nv_settings = get_nv_settings_base()
    base_cmd = f"{nv_settings} {nv_args}".strip()
    if suffix:
        base_cmd = f"{base_cmd} {suffix}"
    if _USE_SUDO:
        return build_sudo_cmd(base_cmd)
    # For non-sudo, still need DISPLAY and XAUTHORITY
    xauth = get_xauthority_path()
    display = get_x_display()
    return f"DISPLAY={display} XAUTHORITY={xauth} {base_cmd}"


def set_use_sudo(use_sudo: bool):
    """Set whether to use sudo for nvidia-settings"""
    global _USE_SUDO
    _USE_SUDO = use_sudo


def is_none_or_empty(val: Union[str, None]) -> bool:
    """val is None or empty"""
    return val is None or (isinstance(val, str) and val.strip() == "")


def is_str_and_all(idx: Union[str, int]) -> bool:
    """idx starts with 'a'"""
    if isinstance(idx, str) and idx.strip().lower().startswith("a"):
        return True
    return False


def parse_idx(idx: Union[str, int]) -> int:
    """Parse index string to int"""
    try:
        return int(idx)
    except Exception:
        log_error(f"× Invalid idx: {idx}")
        return None


def check_x_server() -> bool:
    """Check if X server is available on :0"""
    xauth = get_xauthority_path()
    # Try to query X server
    cmd = f"DISPLAY=:0 XAUTHORITY={xauth} xdpyinfo >/dev/null 2>&1"
    result = os.system(cmd)
    return result == 0


def check_nv_permission() -> bool:
    """Check if we have permission to control fans.
    Try without sudo first, then with sudo if needed.
    Returns True if we have permission (with or without sudo).

    Tests actual set operation to ensure write permission.
    """
    global _USE_SUDO, _PERMISSION_CHECKED, _HAS_PERMISSION

    if _PERMISSION_CHECKED:
        return _HAS_PERMISSION

    _PERMISSION_CHECKED = True

    def test_permission(use_sudo: bool) -> bool:
        """Test if we can actually set fan control state"""
        xauth = get_xauthority_path()
        display = get_x_display()
        nv_settings = get_nv_settings_base()

        def build_cmd(nv_args: str) -> str:
            base_cmd = f"{nv_settings} {nv_args}"
            if use_sudo:
                return build_sudo_cmd(base_cmd)
            # For non-sudo, still need DISPLAY and XAUTHORITY
            return f"DISPLAY={display} XAUTHORITY={xauth} {base_cmd}"

        # First get current state
        query_cmd = build_cmd("-q '[gpu:0]/GPUFanControlState' -t")
        query_output = shell_cmd(query_cmd, getoutput=True, showcmd=False)

        # Check for X server / display errors
        if (
            "control display" in query_output.lower()
            or "authorization" in query_output.lower()
        ):
            return False
        if "error" in query_output.lower() or "permission" in query_output.lower():
            return False

        # Try to set the same state (no actual change, just test permission)
        try:
            current_state = int(query_output.strip())
        except Exception:
            current_state = 0

        set_cmd = build_cmd(f"-a '[gpu:0]/GPUFanControlState={current_state}'")
        set_output = shell_cmd(set_cmd, getoutput=True, showcmd=False)

        # Check if set was successful
        if "assigned" in set_output.lower():
            return True
        if "error" in set_output.lower() or "permission" in set_output.lower():
            return False
        return True

    # First try without sudo
    if test_permission(use_sudo=False):
        _HAS_PERMISSION = True
        return True

    # Try with sudo
    logger.warn("  nvidia-settings requires elevated permissions, trying with sudo...")
    if test_permission(use_sudo=True):
        _USE_SUDO = True
        _HAS_PERMISSION = True
        logger.success("  Using sudo for nvidia-settings")
        return True

    # Check if X server is the problem
    if not check_x_server():
        logger.warn(
            "× X server not accessible on DISPLAY=:0\n"
            "  Fan control requires X server. Please ensure:\n"
            "  1. X server is running (check with: ps aux | grep X)\n"
            "  2. Try starting X: sudo X :0 &\n"
            "  3. Or use a display manager (lightdm, gdm, etc.)"
        )
    else:
        logger.warn(
            "× No permission to control GPU fans.\n"
            "  Please check NVIDIA driver settings and Coolbits configuration.\n"
            '  You may need to add \'Option "Coolbits" "12"\' to xorg.conf'
        )
    _HAS_PERMISSION = False
    return False


class GPUControllerBase:
    """Base class for GPU control operations"""

    def __init__(self, verbose: bool = True):
        self.verbose = verbose
        self._gpu_count = None
        self._fan_count = None
        self._fans_per_gpu = 2  # Most GPUs have 2 fans

    def check_permission(self) -> bool:
        """Check if we have permission to control fans"""
        return check_nv_permission()

    def get_gpu_count(self) -> int:
        """Get number of GPUs"""
        if self._gpu_count is not None:
            return self._gpu_count
        cmd = f"{NV_SMI} --query-gpu=count --format=csv,noheader | head -1"
        output = shell_cmd(cmd, getoutput=True, showcmd=False)
        try:
            self._gpu_count = int(output.strip())
        except Exception:
            self._gpu_count = 0
        return self._gpu_count

    def get_fan_count(self) -> int:
        """Get total number of fans"""
        if self._fan_count is not None:
            return self._fan_count
        # Query nvidia-settings for fan count
        cmd = get_nv_settings_cmd("-q fans", "| grep -c '\\[fan:'")
        output = shell_cmd(cmd, getoutput=True, showcmd=False)
        try:
            self._fan_count = int(output.strip())
        except Exception:
            # Fallback: assume 2 fans per GPU
            self._fan_count = self.get_gpu_count() * self._fans_per_gpu
        return self._fan_count

    def get_fan_indices_for_gpu(self, gpu_idx: int) -> list[int]:
        """Get fan indices for a specific GPU.
        Assumes fans are evenly distributed across GPUs.
        E.g., GPU 0 -> [fan:0, fan:1], GPU 1 -> [fan:2, fan:3], etc.
        """
        fan_count = self.get_fan_count()
        gpu_count = self.get_gpu_count()
        if gpu_count == 0:
            return []
        fans_per_gpu = fan_count // gpu_count
        if fans_per_gpu == 0:
            fans_per_gpu = 1
        start_fan = gpu_idx * fans_per_gpu
        end_fan = min(start_fan + fans_per_gpu, fan_count)
        return list(range(start_fan, end_fan))

    def get_gpu_temp(self, gpu_idx: int) -> int:
        """Get GPU core temperature"""
        cmd = f"{NV_SMI} -i {gpu_idx} --query-gpu=temperature.gpu --format=csv,noheader"
        output = shell_cmd(cmd, getoutput=True, showcmd=False)
        try:
            return int(output.strip())
        except Exception:
            return None

    def get_all_gpu_temps(self) -> dict[int, int]:
        """Get all GPU temperatures"""
        temps = {}
        for i in range(self.get_gpu_count()):
            temp = self.get_gpu_temp(i)
            if temp is not None:
                temps[i] = temp
        return temps

    def get_fan_speed(self, fan_idx: int) -> int:
        """Get current fan speed"""
        cmd = get_nv_settings_cmd(f"-q '[fan:{fan_idx}]/GPUCurrentFanSpeed' -t")
        output = shell_cmd(cmd, getoutput=True, showcmd=False)
        try:
            return int(output.strip())
        except Exception:
            return None

    def get_control_state(self, gpu_idx: int) -> int:
        """Get GPU fan control state (0=auto, 1=manual)"""
        cmd = get_nv_settings_cmd(f"-q '[gpu:{gpu_idx}]/GPUFanControlState' -t")
        output = shell_cmd(cmd, getoutput=True, showcmd=False)
        try:
            return int(output.strip())
        except Exception:
            return None

    def set_control_state(self, gpu_idx: Union[int, str], state: int) -> bool:
        """Set GPU fan control state"""
        if is_str_and_all(gpu_idx):
            cmd = get_nv_settings_cmd(f"-a 'GPUFanControlState={state}'")
        else:
            cmd = get_nv_settings_cmd(
                f"-a '[gpu:{gpu_idx}]/GPUFanControlState={state}'"
            )
        output = shell_cmd(cmd, getoutput=True, showcmd=False)
        success = "assigned" in output.lower() or "error" not in output.lower()
        if not success and self.verbose:
            logger.warn(f"  Failed to set control state for GPU {gpu_idx}")
        return success

    def set_fan_speed(self, fan_idx: Union[int, str], speed: int) -> bool:
        """Set fan speed for a single fan"""
        speed = min(max(speed, MIN_FAN_PERCENT), MAX_FAN_PERCENT)
        if is_str_and_all(fan_idx):
            cmd = get_nv_settings_cmd(f"-a 'GPUTargetFanSpeed={speed}'")
        else:
            cmd = get_nv_settings_cmd(f"-a '[fan:{fan_idx}]/GPUTargetFanSpeed={speed}'")
        output = shell_cmd(cmd, getoutput=True, showcmd=False)
        success = "assigned" in output.lower() or "error" not in output.lower()
        if not success and self.verbose:
            logger.warn(f"  Failed to set fan speed for fan {fan_idx}")
        return success

    def set_gpu_fan_speed(self, gpu_idx: Union[int, str], speed: int) -> bool:
        """Set fan speed for all fans of a GPU"""
        speed = min(max(speed, MIN_FAN_PERCENT), MAX_FAN_PERCENT)
        if is_str_and_all(gpu_idx):
            # Set all fans
            return self.set_fan_speed("a", speed)
        else:
            # Set fans for specific GPU
            fan_indices = self.get_fan_indices_for_gpu(int(gpu_idx))
            success = True
            for fan_idx in fan_indices:
                if not self.set_fan_speed(fan_idx, speed):
                    success = False
            return success

    def set_auto_control(self, gpu_idx: Union[int, str]) -> bool:
        """Reset to automatic fan control"""
        return self.set_control_state(gpu_idx, 0)

    def set_manual_control(self, gpu_idx: Union[int, str]) -> bool:
        """Enable manual fan control"""
        return self.set_control_state(gpu_idx, 1)
