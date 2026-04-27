from __future__ import annotations

from cobot.cli.session import CobotSession
from cobot.cli.renderer import console


async def cmd_calibrate(session: CobotSession, check_only: bool = False, dry_run: bool = False) -> None:
    """Run hand-eye calibration using the active MCP connection."""
    if session.deps is None:
        console.print("[red]No robot connected — use /reconnect first.[/red]")
        return

    arm = session.deps.robotic_deps.arm_mcp
    cfg = session.cfg

    # Inject Google API key from config into env so calibrate_handeye can find it
    import os
    google_key = getattr(cfg.agents.vision, "google_api_key", "") or os.environ.get("GOOGLE_API_KEY", "")
    if google_key:
        os.environ["GOOGLE_API_KEY"] = google_key

    from cobot.agents.vision_agent.calibrate import calibrate_handeye
    from cobot.agents.vision_agent.camera import CALIB_FILE

    console.print(f"  [dim]Starting hand-eye calibration (check_only={check_only}, dry_run={dry_run})…[/dim]")

    result = await calibrate_handeye(
        arm_mcp_client=arm,
        calib_file=CALIB_FILE,
        check_only=check_only,
        dry_run=dry_run,
    )

    if result["success"]:
        console.print(f"  [green]✓ {result['message']}[/green]")
        if "mean_err_mm" in result:
            console.print(f"    mean error: {result['mean_err_mm']:.1f}mm   max: {result['max_err_mm']:.1f}mm   points: {result['n_points']}")
    else:
        console.print(f"  [red]✗ {result['message']}[/red]")
