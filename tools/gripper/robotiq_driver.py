from __future__ import annotations

import socket
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import Any


@dataclass
class GripperStatus:
    connected: bool
    active: bool
    last_command: str | None
    speed: int
    force: int
    ready: bool
    fault: str | None = None


class RobotiqGripper:
    def __init__(
        self,
        robot_ip: str,
        port: int = 30002,
        script_path: str = "Gripper.script",
        timeout: float = 5.0,
    ) -> None:
        self.robot_ip = robot_ip
        self.port = port
        self.script_path = Path(script_path)
        self.timeout = timeout

        self.default_speed = 100
        self.default_force = 100
        self.active = False
        self.last_command: str | None = None
        self.fault: str | None = None

        if not self.script_path.exists():
            raise FileNotFoundError(f"Missing template: {self.script_path}")

        self.template = self.script_path.read_text(encoding="utf-8", errors="ignore")

    def _is_robot_reachable(self) -> bool:
        """Quick TCP probe to check if robot port is reachable."""
        try:
            s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            s.settimeout(0.5)
            s.connect((self.robot_ip, self.port))
            s.close()
            return True
        except Exception:
            return False

    def _clamp(self, value: int, low: int, high: int) -> int:
        return max(low, min(high, int(value)))

    def _send_script(self, script: str) -> None:
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.settimeout(self.timeout)
                s.connect((self.robot_ip, self.port))
                s.sendall(script.encode("utf-8"))
        except Exception as e:
            self.fault = str(e)
            raise

    def _replace_action_block(self, action_block: str) -> str:
        old_block = "  rq_activate_and_wait()\n  rq_move_and_wait(58)\n  \nend"
        new_block = f"{action_block.rstrip()}\n  \nend"

        if old_block in self.template:
            return self.template.replace(old_block, new_block, 1)

        marker = "\nend"
        idx = self.template.rfind(marker)
        if idx == -1:
            raise RuntimeError("Could not find final 'end' in Gripper.script")

        return self.template[:idx] + "\n" + action_block.rstrip() + "\n  " + self.template[idx:]

    def _run(self, commands: str, label: str) -> dict:
        script = self._replace_action_block(commands)
        self._send_script(script)
        self.last_command = label
        self.fault = None
        return self.get_status()

    def set_speed(self, speed: int) -> dict:
        self.default_speed = self._clamp(speed, 0, 100)
        return self.get_status()

    def set_force(self, force: int) -> dict:
        self.default_force = self._clamp(force, 0, 100)
        return self.get_status()

    def activate(self) -> dict:
        result = self._run(
            "  rq_reset()\n  sleep(0.5)\n  rq_activate_and_wait()",
            "activate",
        )
        self.active = True
        return result

    def reset(self) -> dict:
        result = self._run("  rq_reset()", "reset")
        self.active = False
        return result

    def stop(self) -> dict:
        return self._run("  rq_stop()", "stop")

    def open(self, speed: int | None = None, force: int | None = None) -> dict:
        s = self.default_speed if speed is None else self._clamp(speed, 0, 100)
        f = self.default_force if force is None else self._clamp(force, 0, 100)
        result = self._run(
            f"  rq_set_speed_norm({s})\n  rq_set_force_norm({f})\n  rq_open_and_wait()",
            "open",
        )
        result["applied_speed"] = s
        result["applied_force"] = f
        return result

    def close(self, speed: int | None = None, force: int | None = None) -> dict:
        s = self.default_speed if speed is None else self._clamp(speed, 0, 100)
        f = self.default_force if force is None else self._clamp(force, 0, 100)
        result = self._run(
            f"  rq_set_speed_norm({s})\n  rq_set_force_norm({f})\n  rq_close_and_wait()",
            "close",
        )
        result["applied_speed"] = s
        result["applied_force"] = f
        return result

    def move(self, position: int, speed: int | None = None, force: int | None = None) -> dict:
        p = self._clamp(position, 0, 100)
        s = self.default_speed if speed is None else self._clamp(speed, 0, 100)
        f = self.default_force if force is None else self._clamp(force, 0, 100)
        result = self._run(
            f"  rq_set_speed_norm({s})\n  rq_set_force_norm({f})\n  rq_move_and_wait_norm({p})",
            f"move({p})",
        )
        result["applied_speed"] = s
        result["applied_force"] = f
        return result

    def get_status(self) -> dict:
        status = GripperStatus(
            connected=self._is_robot_reachable(),
            active=self.active,
            last_command=self.last_command,
            speed=self.default_speed,
            force=self.default_force,
            ready=self.fault is None,
            fault=self.fault,
        )
        return asdict(status)