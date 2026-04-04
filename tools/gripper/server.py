from __future__ import annotations

import os
import socket
import threading
import time
from mcp.server.fastmcp import FastMCP
from robotiq_driver import RobotiqGripper

ROBOT_IP = os.getenv("ROBOT_IP", "192.168.1.2")
ROBOT_PORT = int(os.getenv("ROBOT_PORT", "30002"))
GRIPPER_SCRIPT = os.getenv("GRIPPER_SCRIPT", "Gripper.script")
DASHBOARD_PORT = 29999

mcp = FastMCP("Robotiq Gripper MCP", host="0.0.0.0", port=8001)

gripper = RobotiqGripper(
    robot_ip=ROBOT_IP,
    port=ROBOT_PORT,
    script_path=GRIPPER_SCRIPT,
)


def _restart_external_control(wait: float) -> None:
    """
    Restore External Control in a background daemon thread.
    Does NOT block the caller — gripper tools return immediately.
    Logs failures but does not raise.
    """
    def _recover() -> None:
        time.sleep(wait)
        try:
            s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            s.settimeout(5.0)
            s.connect((ROBOT_IP, DASHBOARD_PORT))
            s.recv(1024)           # discard the dashboard banner
            s.send(b"play\n")
            time.sleep(1)
            response = s.recv(1024).decode(errors="replace").strip()
            s.close()
            if "error" in response.lower():
                print(f"[gripper] dashboard play failed: {response!r}", flush=True)
            else:
                print(f"[gripper] external control restored: {response!r}", flush=True)
        except Exception as e:
            print(f"[gripper] external control recovery failed: {e}", flush=True)

    threading.Thread(target=_recover, daemon=True).start()


def _to_result(status: dict, extra: dict | None = None) -> dict:
    """
    Normalize a gripper status dict into a consistent result.
    Adds "ok" field (True when ready=True and no fault) for consistency
    with the arm MCP which uses "ok": bool on all tool results.
    """
    result = {"ok": bool(status.get("ready", False)), **status}
    if extra:
        result.update(extra)
    return result


@mcp.tool()
def activate() -> dict:
    """Activate the gripper (full initialization: reset + activate + wait)."""
    try:
        result = gripper.activate()
        _restart_external_control(wait=5.0)
        return _to_result(result)
    except Exception as e:
        return {"ok": False, "error": type(e).__name__, "message": str(e)}


@mcp.tool()
def reset() -> dict:
    """Reset the gripper."""
    try:
        result = gripper.reset()
        _restart_external_control(wait=2.0)
        return _to_result(result)
    except Exception as e:
        return {"ok": False, "error": type(e).__name__, "message": str(e)}


@mcp.tool()
def stop() -> dict:
    """Stop the gripper motion immediately."""
    try:
        result = gripper.stop()
        _restart_external_control(wait=1.0)
        return _to_result(result)
    except Exception as e:
        return {"ok": False, "error": type(e).__name__, "message": str(e)}


@mcp.tool()
def open(speed: int | None = None, force: int | None = None) -> dict:
    """Open the gripper. Speed and force are 0..100."""
    try:
        result = gripper.open(speed=speed, force=force)
        _restart_external_control(wait=3.0)
        return _to_result(result)
    except Exception as e:
        return {"ok": False, "error": type(e).__name__, "message": str(e)}


@mcp.tool()
def close(speed: int | None = None, force: int | None = None) -> dict:
    """Close the gripper. Speed and force are 0..100."""
    try:
        result = gripper.close(speed=speed, force=force)
        _restart_external_control(wait=3.0)
        return _to_result(result)
    except Exception as e:
        return {"ok": False, "error": type(e).__name__, "message": str(e)}


@mcp.tool()
def move(position: int, speed: int | None = None, force: int | None = None) -> dict:
    """Move the gripper to a normalized position 0..100 (0=open, 100=closed)."""
    try:
        result = gripper.move(position=position, speed=speed, force=force)
        _restart_external_control(wait=4.0)
        return _to_result(result)
    except Exception as e:
        return {"ok": False, "error": type(e).__name__, "message": str(e)}


@mcp.tool()
def set_speed(speed: int) -> dict:
    """Set default speed 0..100."""
    try:
        result = gripper.set_speed(speed)
        return _to_result(result)
    except Exception as e:
        return {"ok": False, "error": type(e).__name__, "message": str(e)}


@mcp.tool()
def set_force(force: int) -> dict:
    """Set default force 0..100."""
    try:
        result = gripper.set_force(force)
        return _to_result(result)
    except Exception as e:
        return {"ok": False, "error": type(e).__name__, "message": str(e)}


@mcp.tool()
def get_status() -> dict:
    """Return current gripper status."""
    try:
        result = gripper.get_status()
        return _to_result(result)
    except Exception as e:
        return {"ok": False, "error": type(e).__name__, "message": str(e)}


if __name__ == "__main__":
    mcp.run(transport="streamable-http")
