import argparse
import json
import os
import signal
import time

from pathlib import Path
from tclogger import logger, shell_cmd, log_error
from typing import Union

MIN_FAN_PERCENT = 0
MAX_FAN_PERCENT = 100
DEFAULT_CONFIG_FILE = "gpu_mon.config.json"
DEFAULT_INTERVAL = 1.0

NV_SETTINGS = "DISPLAY=:0 nvidia-settings"
NV_SMI = "nvidia-smi"


def get_script_dir() -> Path:
    """Get the directory of current script"""
    return Path(__file__).parent


def is_none_or_empty(val: Union[str, None]) -> bool:
    """val is None or empty"""
    return val is None or (isinstance(val, str) and val.strip() == "")


def is_str_and_all(idx: str) -> bool:
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


def parse_curve_points(curve_str: str) -> Union[list[tuple[int, int]], None]:
    """Parse curve string to list of (temp, speed) tuples.
    Format: "50-80/75-100" means temp>=50°C->80%, temp>=75°C->100%
    """
    if is_none_or_empty(curve_str):
        return None
    try:
        points = []
        for point_str in curve_str.split("/"):
            temp_str, speed_str = point_str.split("-")
            temp = int(temp_str)
            speed = int(speed_str)
            if not (0 <= temp <= 120):
                log_error(f"× Invalid temp in curve: {temp}")
                return None
            if not (0 <= speed <= 100):
                log_error(f"× Invalid speed in curve: {speed}")
                return None
            points.append((temp, speed))
        # Sort by temperature
        points.sort(key=lambda x: x[0])
        return points
    except Exception as e:
        log_error(f"× Invalid curve format: {curve_str}")
        return None


def curve_points_to_str(points: list[tuple[int, int]]) -> str:
    """Convert curve points to string format"""
    if not points:
        return "auto"
    return "/".join([f"{t}-{s}" for t, s in points])


def curve_points_to_display(points: list[tuple[int, int]]) -> str:
    """Convert curve points to display string"""
    if not points:
        return "auto"
    return ", ".join([f"{t}°C->{s}%" for t, s in points])


def calculate_fan_speed(temp: int, curve_points: list[tuple[int, int]]) -> int:
    """Calculate fan speed based on temperature and curve points.
    Uses linear interpolation between points.
    """
    if not curve_points:
        return None  # auto mode

    # Sort by temp
    points = sorted(curve_points, key=lambda x: x[0])

    # Below lowest point
    if temp <= points[0][0]:
        return points[0][1]

    # Above highest point
    if temp >= points[-1][0]:
        return points[-1][1]

    # Find two points to interpolate between
    for i in range(len(points) - 1):
        t1, s1 = points[i]
        t2, s2 = points[i + 1]
        if t1 <= temp <= t2:
            # Linear interpolation
            if t2 == t1:
                return s1
            ratio = (temp - t1) / (t2 - t1)
            speed = int(s1 + ratio * (s2 - s1))
            return min(max(speed, MIN_FAN_PERCENT), MAX_FAN_PERCENT)

    return points[-1][1]


class GPUMonConfig:
    """Manage GPU monitor configuration"""

    def __init__(self, config_file: str = None):
        if config_file:
            self.config_path = Path(config_file)
            if not self.config_path.is_absolute():
                self.config_path = get_script_dir() / config_file
        else:
            self.config_path = get_script_dir() / DEFAULT_CONFIG_FILE
        self.config = {"curves": {}, "interval": DEFAULT_INTERVAL}

    def load(self) -> bool:
        """Load config from file"""
        if not self.config_path.exists():
            logger.warn(f"  Config file not found: {self.config_path}")
            return False
        try:
            with open(self.config_path, "r") as f:
                self.config = json.load(f)
            logger.success(f"  Loaded config from: {self.config_path}")
            return True
        except Exception as e:
            log_error(f"× Failed to load config: {e}")
            return False

    def save(self) -> bool:
        """Save config to file"""
        try:
            with open(self.config_path, "w") as f:
                json.dump(self.config, f, indent=2)
            logger.success(f"  Saved config to: {self.config_path}")
            return True
        except Exception as e:
            log_error(f"× Failed to save config: {e}")
            return False

    def get_curve(self, gpu_idx: int) -> list[tuple[int, int]]:
        """Get curve for specific GPU"""
        curve_str = self.config.get("curves", {}).get(str(gpu_idx), None)
        if curve_str:
            return parse_curve_points(curve_str)
        return None

    def set_curve(self, gpu_idx: int, curve_points: list[tuple[int, int]]):
        """Set curve for specific GPU"""
        if "curves" not in self.config:
            self.config["curves"] = {}
        if curve_points:
            self.config["curves"][str(gpu_idx)] = curve_points_to_str(curve_points)
        else:
            # Remove curve (auto mode)
            self.config["curves"].pop(str(gpu_idx), None)

    def get_interval(self) -> float:
        """Get monitoring interval"""
        return self.config.get("interval", DEFAULT_INTERVAL)

    def set_interval(self, interval: float):
        """Set monitoring interval"""
        self.config["interval"] = interval


class GPUController:
    """Control GPU fan via nvidia-settings"""

    def __init__(self, verbose: bool = True):
        self.verbose = verbose
        self._gpu_count = None

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
        cmd = f"{NV_SETTINGS} -q '[fan:{fan_idx}]/GPUCurrentFanSpeed' -t"
        output = shell_cmd(cmd, getoutput=True, showcmd=False)
        try:
            return int(output.strip())
        except Exception:
            return None

    def get_control_state(self, gpu_idx: int) -> int:
        """Get GPU fan control state (0=auto, 1=manual)"""
        cmd = f"{NV_SETTINGS} -q '[gpu:{gpu_idx}]/GPUFanControlState' -t"
        output = shell_cmd(cmd, getoutput=True, showcmd=False)
        try:
            return int(output.strip())
        except Exception:
            return None

    def set_control_state(self, gpu_idx: Union[int, str], state: int) -> bool:
        """Set GPU fan control state"""
        if is_str_and_all(gpu_idx):
            cmd = f"{NV_SETTINGS} -a 'GPUFanControlState={state}'"
        else:
            cmd = f"{NV_SETTINGS} -a '[gpu:{gpu_idx}]/GPUFanControlState={state}'"
        # output = shell_cmd(cmd, getoutput=True, showcmd=self.verbose)
        output = shell_cmd(cmd, getoutput=True, showcmd=False)
        return "assigned" in output.lower() or "error" not in output.lower()

    def set_fan_speed(self, fan_idx: Union[int, str], speed: int) -> bool:
        """Set fan speed"""
        speed = min(max(speed, MIN_FAN_PERCENT), MAX_FAN_PERCENT)
        if is_str_and_all(fan_idx):
            cmd = f"{NV_SETTINGS} -a 'GPUTargetFanSpeed={speed}'"
        else:
            cmd = f"{NV_SETTINGS} -a '[fan:{fan_idx}]/GPUTargetFanSpeed={speed}'"
        # output = shell_cmd(cmd, getoutput=True, showcmd=self.verbose)
        output = shell_cmd(cmd, getoutput=True, showcmd=False)
        return "assigned" in output.lower() or "error" not in output.lower()

    def set_auto_control(self, gpu_idx: Union[int, str]) -> bool:
        """Reset to automatic fan control"""
        return self.set_control_state(gpu_idx, 0)

    def set_manual_control(self, gpu_idx: Union[int, str]) -> bool:
        """Enable manual fan control"""
        return self.set_control_state(gpu_idx, 1)


class GPUMonitor:
    """Monitor GPU temperature and control fan speed based on curve"""

    def __init__(
        self,
        config: GPUMonConfig,
        controller: GPUController,
        verbose: bool = True,
    ):
        self.config = config
        self.controller = controller
        self.verbose = verbose
        self.running = False
        self._last_speeds = {}  # Track last set speeds to avoid redundant commands

    def display_curve(self, gpu_idx: Union[int, str]):
        """Display curve settings for GPU(s)"""
        gpu_count = self.controller.get_gpu_count()
        if gpu_count == 0:
            log_error("× No GPU found")
            return

        if is_str_and_all(gpu_idx):
            indices = list(range(gpu_count))
        else:
            indices = [parse_idx(gpu_idx)]

        for idx in indices:
            if idx is None or idx >= gpu_count:
                continue
            curve = self.config.get_curve(idx)
            temp = self.controller.get_gpu_temp(idx)
            fan_speed = self.controller.get_fan_speed(idx)
            ctrl_state = self.controller.get_control_state(idx)
            ctrl_str = "manual" if ctrl_state == 1 else "auto"

            logger.mesg(f"  GPU {idx}:")
            logger.mesg(f"    Temperature: {temp}°C")
            logger.mesg(f"    Fan Speed:   {fan_speed}%")
            logger.mesg(f"    Control:     {ctrl_str}")
            logger.mesg(f"    Curve:       {curve_points_to_display(curve)}")

    def set_curve(
        self,
        gpu_idx: Union[int, str],
        curve_points: list[tuple[int, int]],
    ):
        """Set curve for GPU(s)"""
        gpu_count = self.controller.get_gpu_count()
        if gpu_count == 0:
            log_error("× No GPU found")
            return

        if is_str_and_all(gpu_idx):
            indices = list(range(gpu_count))
        else:
            indices = [parse_idx(gpu_idx)]

        for idx in indices:
            if idx is None or idx >= gpu_count:
                continue
            self.config.set_curve(idx, curve_points)
            if curve_points:
                logger.success(
                    f"  GPU {idx}: Set curve to {curve_points_to_display(curve_points)}"
                )
            else:
                logger.success(f"  GPU {idx}: Set to auto control")

    def reset_to_auto(self, gpu_idx: Union[int, str]):
        """Reset GPU(s) to automatic control"""
        gpu_count = self.controller.get_gpu_count()
        if gpu_count == 0:
            log_error("× No GPU found")
            return

        if is_str_and_all(gpu_idx):
            indices = list(range(gpu_count))
        else:
            indices = [parse_idx(gpu_idx)]

        for idx in indices:
            if idx is None or idx >= gpu_count:
                continue
            self.config.set_curve(idx, None)
            self.controller.set_auto_control(idx)
            self._last_speeds.pop(idx, None)
            logger.success(f"  GPU {idx}: Reset to automatic control")

    def apply_curve_once(self, gpu_idx: int):
        """Apply curve for single GPU based on current temperature"""
        curve = self.config.get_curve(gpu_idx)
        if not curve:
            return  # auto mode, skip

        temp = self.controller.get_gpu_temp(gpu_idx)
        if temp is None:
            return

        target_speed = calculate_fan_speed(temp, curve)
        if target_speed is None:
            return

        # Only set if speed changed
        last_speed = self._last_speeds.get(gpu_idx)
        if last_speed != target_speed:
            # Ensure manual control is enabled
            self.controller.set_manual_control(gpu_idx)
            self.controller.set_fan_speed(gpu_idx, target_speed)
            self._last_speeds[gpu_idx] = target_speed
            if self.verbose:
                logger.mesg(f"  GPU {gpu_idx}: {temp}°C -> {target_speed}%")

    def run_loop(self):
        """Run monitoring loop"""
        self.running = True
        interval = self.config.get_interval()
        gpu_count = self.controller.get_gpu_count()

        logger.note(f"  Starting GPU monitor (interval: {interval}s)")
        logger.note(f"  Press Ctrl+C to stop")

        # Setup signal handler for graceful shutdown
        def signal_handler(signum, frame):
            logger.warn("\n  Stopping monitor...")
            self.running = False

        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)

        try:
            while self.running:
                for gpu_idx in range(gpu_count):
                    if not self.running:
                        break
                    self.apply_curve_once(gpu_idx)
                if self.running:
                    time.sleep(interval)
        except KeyboardInterrupt:
            pass
        finally:
            logger.note("  Monitor stopped")


class GPUMonArgParser(argparse.ArgumentParser):
    def __init__(self):
        super().__init__(
            description="GPU Fan Monitor - Control fan speed based on temperature curve",
            formatter_class=argparse.RawDescriptionHelpFormatter,
            epilog="""
Examples:
  gpu_mon -c 0                  # Show GPU 0 curve settings
  gpu_mon -c a                  # Show all GPUs curve settings
  gpu_mon -c 0:50-80/75-100     # Set GPU 0 curve: 50°C->80%, 75°C->100%
  gpu_mon -c a:50-80/75-100 -s  # Set all GPUs curve and save config
  gpu_mon -c a:file             # Load curve from config file
  gpu_mon -c 0:auto             # Reset GPU 0 to automatic control
  gpu_mon -c a:auto -s          # Reset all GPUs to auto and save

Curve format:
  <temp1>-<speed1>/<temp2>-<speed2>/...
  Example: 50-60/70-80/85-100 means:
    - Below 50°C: 60% fan speed
    - 50-70°C: interpolate between 60-80%
    - 70-85°C: interpolate between 80-100%
    - Above 85°C: 100% fan speed
""",
        )
        self.add_argument(
            "-c",
            "--curve",
            type=str,
            help="Get/Set temperature-speed curve. Format: <gpu_idx>:<curve> or <gpu_idx> to view",
        )
        self.add_argument(
            "-s",
            "--save",
            action="store_true",
            help="Save configuration to file",
        )
        self.add_argument(
            "-f",
            "--file",
            type=str,
            default=DEFAULT_CONFIG_FILE,
            help=f"Config file path (default: {DEFAULT_CONFIG_FILE})",
        )
        self.add_argument(
            "-t",
            "--interval",
            type=float,
            default=None,
            help=f"Monitoring interval in seconds (default: {DEFAULT_INTERVAL})",
        )
        self.add_argument(
            "-q",
            "--quiet",
            action="store_true",
            help="Quiet mode, less output",
        )
        self.args, _ = self.parse_known_args()


def parse_curve_arg(curve_arg: str) -> tuple[str, str]:
    """Parse curve argument into (gpu_idx, curve_value).
    Returns (gpu_idx, None) for get operation.
    Returns (gpu_idx, curve_str) for set operation.
    """
    if ":" not in curve_arg:
        # Get operation: just gpu index
        return (curve_arg.strip(), None)

    parts = curve_arg.split(":", maxsplit=1)
    gpu_idx = parts[0].strip()
    curve_val = parts[1].strip() if len(parts) > 1 else None
    return (gpu_idx, curve_val)


def main():
    args = GPUMonArgParser().args

    # Show help if no arguments
    if args.curve is None:
        GPUMonArgParser().print_help()
        return

    # Initialize components
    config = GPUMonConfig(args.file)
    controller = GPUController(verbose=not args.quiet)
    monitor = GPUMonitor(config, controller, verbose=not args.quiet)

    # Set interval if provided
    if args.interval is not None:
        config.set_interval(args.interval)

    # Parse curve argument
    gpu_idx, curve_val = parse_curve_arg(args.curve)

    # Handle different operations
    if curve_val is None:
        # Get operation: display current curve
        config.load()  # Try to load existing config
        monitor.display_curve(gpu_idx)

    elif curve_val.lower() == "auto":
        # Reset to automatic control
        monitor.reset_to_auto(gpu_idx)
        if args.save:
            config.save()

    elif curve_val.lower() == "file":
        # Load from config file
        if not config.load():
            log_error("× Cannot load config file")
            return
        # Start monitoring loop
        monitor.run_loop()

    else:
        # Set curve
        curve_points = parse_curve_points(curve_val)
        if curve_points is None:
            log_error(f"× Invalid curve format: {curve_val}")
            return

        # Parse multiple GPU indices
        if is_str_and_all(gpu_idx):
            monitor.set_curve(gpu_idx, curve_points)
        else:
            for idx in gpu_idx.split(","):
                idx = idx.strip()
                if idx:
                    monitor.set_curve(idx, curve_points)

        if args.save:
            config.save()

        # Start monitoring loop
        monitor.run_loop()


if __name__ == "__main__":
    main()

    # Usage examples:
    # gpu_mon                     # Show help
    # gpu_mon -c 0                # Show GPU 0 curve settings
    # gpu_mon -c a                # Show all GPUs curve settings
    # gpu_mon -c "0,1"            # Show GPU 0,1 curve settings
    # gpu_mon -c 0:50-80/75-100   # Set GPU 0 curve and start monitoring
    # gpu_mon -c a:50-80/75-100 -s -t 0.5  # Set all GPUs, save config, 0.5s interval
    # gpu_mon -c a:file           # Load curve from file and start monitoring
    # gpu_mon -c 0:auto           # Reset GPU 0 to automatic control
    # gpu_mon -c a:auto -s        # Reset all GPUs to auto and save config
    # gpu_mon -c a:30-0/40-30/50-65/60-80/75-100 -s
