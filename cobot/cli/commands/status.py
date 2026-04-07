from __future__ import annotations

from cobot.cli.session import CobotSession
from cobot.cli.renderer import console, print_status_table


async def cmd_status(session: CobotSession) -> None:
    if session.deps is None:
        console.print("[red]No robot connected.[/red]")
        return
    arm = session.deps.robotic_deps.arm_mcp

    # get_robot_status may be unavailable (dashboard service error on some setups).
    # Always fetch joint states and TCP pose which are reliable ROS topics.
    status = await arm.call_tool("get_robot_status", {})
    joints_result = await arm.call_tool("get_joint_states", {})
    tcp_result = await arm.call_tool("get_tcp_pose", {})

    if not status.get("ok"):
        # Compose a useful status from the tools that work
        status = {
            "ok": False,
            "robot_ip": session.active_robot.ip,
            "has_joint_state": joints_result.get("ok", False),
            "headless_mode": "—",
            "motion": {},
            "_warning": status.get("message", "get_robot_status unavailable"),
        }

    if joints_result.get("ok"):
        status["joints"] = joints_result.get("joints")
    if tcp_result.get("ok"):
        pos = tcp_result.get("position_m", {})
        status["tcp_position_m"] = pos

    print_status_table(status)
