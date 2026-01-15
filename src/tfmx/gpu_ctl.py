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

# Global state for sudo and X11
_USE_SUDO = False
_PERMISSION_CHECKED = False
_X_DISPLAY = None  # Cached X display


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


def _test_display_with_nvidia(display: str) -> bool:
    """Test if a display can access NVIDIA GPU via nvidia-settings.
    Returns True if the display has valid NVIDIA GPU access.
    """
    xauth = get_xauthority_path()
    # Quick test: query GPU count - if it returns a number, display works
    cmd = f"DISPLAY={display} XAUTHORITY={xauth} nvidia-settings --ctrl-display={display} -q gpus 2>/dev/null | grep -c '\\[gpu:'"
    output = shell_cmd(cmd, getoutput=True, showcmd=False)
    try:
        gpu_count = int(output.strip())
        return gpu_count > 0
    except Exception:
        return False


def _get_candidate_displays() -> list[str]:
    """Get list of candidate X displays, sorted by priority.
    Excludes virtual displays (like :99 used by Xvfb) and prioritizes
    physical displays.
    """
    candidates = []

    # Priority 1: Environment variable DISPLAY (if set and looks valid)
    env_display = os.environ.get("DISPLAY", "")
    if env_display and env_display.startswith(":"):
        try:
            display_num = int(env_display.split(":")[1].split(".")[0])
            # Only trust low-numbered displays from env (avoid Xvfb :99, etc.)
            if display_num < 10:
                candidates.append(env_display.split(".")[0])  # Remove screen number
        except Exception:
            pass

    # Priority 2: Lock files, but only for low-numbered displays
    lock_files = sorted(glob.glob("/tmp/.X*-lock"))
    for lock_file in lock_files:
        try:
            display_num_str = lock_file.replace("/tmp/.X", "").replace("-lock", "")
            display_num = int(display_num_str)
            # Skip high-numbered displays (likely Xvfb, VNC, etc.)
            # Physical X servers typically use :0, :1, :2
            if display_num >= 10:
                continue
            display = f":{display_num}"
            if display not in candidates:
                candidates.append(display)
        except Exception:
            pass

    # Priority 3: Common default displays
    for display in [":0", ":1", ":2"]:
        if display not in candidates:
            candidates.append(display)

    return candidates


def get_x_display() -> str:
    """Get the correct X display for nvidia-settings.
    Tries to detect the actual display used by the local X server
    by testing each candidate display for NVIDIA GPU access.
    """
    global _X_DISPLAY
    if _X_DISPLAY is not None:
        return _X_DISPLAY

    candidates = _get_candidate_displays()

    # Test each candidate display
    for display in candidates:
        if _test_display_with_nvidia(display):
            _X_DISPLAY = display
            return _X_DISPLAY

    # Fallback: return first candidate even if test failed
    # (might work with sudo)
    _X_DISPLAY = candidates[0] if candidates else ":0"
    return _X_DISPLAY


def get_nv_settings_base() -> str:
    """Get nvidia-settings command with correct display"""
    display = get_x_display()
    return f"nvidia-settings --ctrl-display={display}"


def build_sudo_cmd(cmd: str) -> str:
    """Build a command with sudo, using SUDOPASS if available.
    Uses $SUDOPASS environment variable to avoid interactive password prompt.
    Also handles X11 permissions by preserving DISPLAY and XAUTHORITY.
    """
    sudopass = os.environ.get("SUDOPASS", "")
    xauth = get_xauthority_path()
    display = get_x_display()
    env_vars = f"DISPLAY={display} XAUTHORITY={xauth}"

    if sudopass:
        escaped_cmd = cmd.replace("'", "'\\''")
        return f"bash -c 'echo \"$SUDOPASS\" | sudo -S env {env_vars} {escaped_cmd} 2>/dev/null'"
    return f"sudo env {env_vars} {cmd}"


def get_nv_settings_cmd(nv_args: str = "", suffix: str = "") -> str:
    """Get nvidia-settings command with proper DISPLAY and XAUTHORITY.
    Uses sudo if _USE_SUDO is True (auto-detected by check_nv_permission).

    Args:
        nv_args: Arguments to pass to nvidia-settings
        suffix: Additional command suffix (e.g., "| grep xxx")

    Returns:
        Complete command string ready for shell execution
    """
    nv_settings = get_nv_settings_base()
    base_cmd = f"{nv_settings} {nv_args}".strip()
    if suffix:
        base_cmd = f"{base_cmd} {suffix}"
    if _USE_SUDO:
        return build_sudo_cmd(base_cmd)
    xauth = get_xauthority_path()
    display = get_x_display()
    return f"DISPLAY={display} XAUTHORITY={xauth} {base_cmd}"


def set_use_sudo(use_sudo: bool):
    """Manually set whether to use sudo for nvidia-settings"""
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
    """Check if X server is available"""
    xauth = get_xauthority_path()
    display = get_x_display()
    cmd = f"DISPLAY={display} XAUTHORITY={xauth} xdpyinfo >/dev/null 2>&1"
    result = os.system(cmd)
    return result == 0


def check_nv_permission() -> bool:
    """Check if we have permission to control fans.
    Auto-detects whether sudo is needed by testing actual set operation.
    Returns True if we have permission (with or without sudo).
    """
    global _USE_SUDO, _PERMISSION_CHECKED, _X_DISPLAY

    if _PERMISSION_CHECKED:
        return True

    display = get_x_display()
    xauth = get_xauthority_path()
    nv_settings = get_nv_settings_base()

    def build_test_cmd(nv_args: str, use_sudo: bool) -> str:
        base_cmd = f"{nv_settings} {nv_args}"
        if use_sudo:
            return build_sudo_cmd(base_cmd)
        return f"DISPLAY={display} XAUTHORITY={xauth} {base_cmd}"

    def test_write_permission(use_sudo: bool) -> bool:
        """Test if we can actually set fan control state"""
        # First get current state
        query_cmd = build_test_cmd("-q '[gpu:0]/GPUFanControlState' -t", use_sudo)
        query_output = shell_cmd(query_cmd, getoutput=True, showcmd=False)

        # Check for X11 errors
        if "authorization" in query_output.lower():
            return False
        if "control display" in query_output.lower():
            return False
        # "Bad handle" means wrong display or no GPU access
        if "bad handle" in query_output.lower():
            return False

        try:
            current_state = int(query_output.strip())
        except Exception:
            current_state = 0

        # Try to set the same state (no actual change, just test permission)
        set_cmd = build_test_cmd(
            f"-a '[gpu:0]/GPUFanControlState={current_state}'", use_sudo
        )
        set_output = shell_cmd(set_cmd, getoutput=True, showcmd=False)

        # Check if set was successful
        if "assigned" in set_output.lower():
            return True
        if "not permitted" in set_output.lower():
            return False
        if "error" in set_output.lower():
            return False
        if "bad handle" in set_output.lower():
            return False
        return True

    def try_alternative_displays_with_sudo() -> bool:
        """Try alternative displays with sudo if main display failed."""
        global _X_DISPLAY
        candidates = _get_candidate_displays()
        for alt_display in candidates:
            if alt_display == display:
                continue
            # Test this display with sudo
            alt_nv_settings = f"nvidia-settings --ctrl-display={alt_display}"
            test_cmd = build_sudo_cmd(
                f"{alt_nv_settings} -q '[gpu:0]/GPUFanControlState' -t"
            )
            output = shell_cmd(test_cmd, getoutput=True, showcmd=False)
            if "bad handle" not in output.lower() and "error" not in output.lower():
                try:
                    int(output.strip())
                    # This display works!
                    _X_DISPLAY = alt_display
                    logger.hint(f"  Found working display: {alt_display}")
                    return True
                except Exception:
                    pass
        return False

    # First try without sudo
    if test_write_permission(use_sudo=False):
        _PERMISSION_CHECKED = True
        return True

    # Try with sudo
    logger.hint("  nvidia-settings requires elevated permissions, trying with sudo...")
    if test_write_permission(use_sudo=True):
        _USE_SUDO = True
        _PERMISSION_CHECKED = True
        logger.success("  Using sudo for nvidia-settings")
        return True

    # Current display doesn't work, try alternatives with sudo
    logger.hint(f"  Display {display} not working, trying alternatives...")
    if try_alternative_displays_with_sudo():
        _USE_SUDO = True
        _PERMISSION_CHECKED = True
        logger.success(f"  Using sudo for nvidia-settings with display {_X_DISPLAY}")
        return True

    # Permission check failed
    if not check_x_server():
        logger.warn(
            f"× X server not accessible on DISPLAY={display}\n"
            "  Fan control requires X server. Please ensure:\n"
            "  1. X server is running (check with: ps aux | grep X)\n"
            "  2. DISPLAY and XAUTHORITY are set correctly\n"
            "  3. If using Xvfb/VNC, ensure a physical X server is also running"
        )
    else:
        logger.warn(
            "× No permission to control GPU fans.\n"
            "  Please check NVIDIA driver settings and Coolbits configuration.\n"
            '  You may need to add \'Option "Coolbits" "12"\' to xorg.conf'
        )
    return False


class GPUControllerBase:
    """Base class for GPU control operations"""

    def __init__(self, verbose: bool = True):
        self.verbose = verbose
        self._gpu_count = None
        self._gpu_indices = None  # List of available GPU indices
        self._fan_count = None
        self._fans_per_gpu = 2  # Most GPUs have 2 fans

    def check_permission(self) -> bool:
        """Check if we have permission to control fans"""
        return check_nv_permission()

    def get_gpu_indices(self) -> list[int]:
        """Get list of available GPU indices.
        Handles cases where some GPUs have errors (e.g., GPU6 offline).
        Returns actual GPU indices that are accessible.
        """
        if self._gpu_indices is not None:
            return self._gpu_indices

        # Parse nvidia-smi --list-gpus output to get actual GPU indices
        # Example output: "GPU 0: NVIDIA GeForce RTX 3080 (UUID: GPU-xxx)"
        cmd = f"{NV_SMI} --list-gpus 2>/dev/null"
        output = shell_cmd(cmd, getoutput=True, showcmd=False)
        indices = []
        try:
            for line in output.strip().split("\n"):
                line = line.strip()
                if line.startswith("GPU "):
                    # Extract GPU index from "GPU X: ..."
                    idx_str = line.split(":")[0].replace("GPU ", "").strip()
                    if idx_str.isdigit():
                        indices.append(int(idx_str))
        except Exception:
            pass

        if not indices:
            # Fallback: assume sequential indices from 0 to count-1
            count = self.get_gpu_count()
            indices = list(range(count))

        self._gpu_indices = sorted(indices)
        return self._gpu_indices

    def get_gpu_count(self) -> int:
        """Get number of GPUs.
        Handles cases where some GPUs have errors by filtering output lines.
        """
        if self._gpu_count is not None:
            return self._gpu_count
        # Use --list-gpus which is more robust - counts actual GPU lines
        cmd = f"{NV_SMI} --list-gpus 2>/dev/null | grep -c 'GPU [0-9]'"
        output = shell_cmd(cmd, getoutput=True, showcmd=False)
        try:
            self._gpu_count = int(output.strip())
        except Exception:
            # Fallback: try query-gpu but filter for numeric lines only
            cmd = f"{NV_SMI} --query-gpu=count --format=csv,noheader 2>/dev/null"
            output = shell_cmd(cmd, getoutput=True, showcmd=False)
            try:
                # Find first line that is a valid integer
                for line in output.strip().split("\n"):
                    line = line.strip()
                    if line.isdigit():
                        self._gpu_count = int(line)
                        break
                else:
                    self._gpu_count = 0
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
