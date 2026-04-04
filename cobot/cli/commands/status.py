from __future__ import annotations

from cobot.cli.session import CobotSession
from cobot.cli.renderer import console, print_status_table


async def cmd_status(session: CobotSession) -> None:
    if session.deps is None:
        console.print("[red]No robot connected.[/red]")
        return
    status = await session.deps.scenario_deps.arm_mcp.call_tool("get_robot_status", {})
    print_status_table(status)
