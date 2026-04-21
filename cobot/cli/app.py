"""
cobot CLI entry point.

Commands:
  cobot              Start the interactive REPL
  cobot status       Print robot status and exit
  cobot run <name>   Execute a scenario directly
  cobot ingest       Ingest robot manuals into RAG knowledge base (Phase 6)
"""

from __future__ import annotations

import asyncio

import typer
from rich.console import Console

from cobot.config import load_config
from cobot.cli.session import CobotSession

app = typer.Typer(
    name="cobot-claw",
    help="cobot-claw — UR robot agent CLI",
    no_args_is_help=False,
    invoke_without_command=True,
)
console = Console()


def _build_session(config: str, robot: str | None) -> CobotSession:
    cfg = load_config(config)
    active_robot = cfg.default_robot
    if robot:
        matches = [r for r in cfg.fleet if r.name == robot]
        if not matches:
            console.print(f"[red]Robot '{robot}' not found in fleet.[/red]")
            raise typer.Exit(1)
        active_robot = matches[0]
    return CobotSession(cfg=cfg, active_robot=active_robot)


class _OfflineMCP:
    """Stub used in place of any MCP that failed to connect.

    Returns a clean dict so the LLM can respond naturally instead of
    seeing a 'Not connected' exception.
    """
    def __init__(self, url: str) -> None:
        self._url = url

    async def call_tool(self, name: str, args: dict | None = None) -> dict:
        return {"ok": False, "message": f"MCP server offline ({self._url}). Use /reconnect to retry."}

    async def list_tools(self):
        raise ConnectionError(f"MCP server offline ({self._url}). Use /reconnect to retry.")


async def _connect_session(session: CobotSession) -> None:
    """Connect MCP clients and wire up agent deps.

    Each MCP is connected independently.  Offline endpoints are replaced
    with _OfflineMCP so the LLM gets a clean error message instead of
    an unhandled exception.  config_deps is None until Phase 6.
    """
    from cobot.mcp.client import MCPClient
    from cobot.mcp.perception_contract import PerceptionClient
    from cobot.scenarios.executor import ScenarioExecutor
    from cobot.agents.deps import RoboticAgentDeps
    from cobot.agents.orchestrator import OrchestratorDeps

    robot = session.active_robot

    _arm_client = MCPClient(robot.arm_mcp_url)
    try:
        await _arm_client.connect()
        arm: object = _arm_client
        session.mcp_status["arm"] = True
    except Exception as e:
        arm = _OfflineMCP(robot.arm_mcp_url)
        session.mcp_status["arm"] = False
        console.print(f"  [yellow]⚠[/yellow]  Arm MCP unavailable ({robot.arm_mcp_url}): {e}")

    tool_mcps: dict = {}
    if robot.gripper_mcp_url:
        _gripper_client = MCPClient(robot.gripper_mcp_url)
        try:
            await _gripper_client.connect()
            tool_mcps["gripper"] = _gripper_client
            session.mcp_status["gripper"] = True
        except Exception as e:
            tool_mcps["gripper"] = _OfflineMCP(robot.gripper_mcp_url)
            session.mcp_status["gripper"] = False
            console.print(f"  [yellow]⚠[/yellow]  Gripper MCP unavailable ({robot.gripper_mcp_url}): {e}")

    perception = PerceptionClient()

    from cobot.agents.deps import ConfigAgentDeps
    from cobot.agents.vision_agent import VisionAgentDeps
    from cobot.rag.pipeline import RAGPipeline

    rag = RAGPipeline(db_path=session.cfg.rag.db_path, cfg=session.cfg)
    config_deps = ConfigAgentDeps(
        arm_mcp=arm,  # type: ignore[arg-type]
        rag_pipeline=rag,
        scenarios_store_path=session.cfg.scenarios.store_path,
        cfg=session.cfg,
    )

    vision_deps = VisionAgentDeps(cfg=session.cfg)
    session.mcp_status["vision"] = True

    async def _vision_servo_fn(description: str) -> dict | None:
        import asyncio
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, vision_deps.locate_sync, description)

    executor = ScenarioExecutor(arm, tool_mcps, perception, vision_fn=_vision_servo_fn)  # type: ignore[arg-type]

    session.deps = OrchestratorDeps(
        robotic_deps=RoboticAgentDeps(arm, tool_mcps, executor, perception, session.cfg),  # type: ignore[arg-type]
        cfg=session.cfg,
        config_deps=config_deps,
        vision_deps=vision_deps,
    )


@app.callback()
def main(
    ctx: typer.Context,
    robot: str = typer.Option(None, "--robot", "-r", help="Robot name from fleet config"),
    config: str = typer.Option("cobot.yaml", "--config", "-c", help="Path to cobot.yaml"),
    agent: str = typer.Option("orchestrator", "--agent", "-a", help="Starting agent"),
) -> None:
    """Start the interactive cobot REPL."""
    if ctx.invoked_subcommand is not None:
        return
    session = _build_session(config, robot)
    session.active_agent = agent

    from cobot.observability import setup as obs_setup
    obs_setup(session.cfg)

    async def _run() -> None:
        try:
            await _connect_session(session)
        except Exception as e:
            console.print(f"  [yellow]⚠[/yellow]  Could not connect to robot MCP: {e}")
            console.print("  [dim]Starting in disconnected mode — natural language commands unavailable.[/dim]\n")
        from cobot.cli.repl import run_repl
        await run_repl(session)

    asyncio.run(_run())


@app.command()
def run(
    scenario: str = typer.Argument(..., help="Scenario name"),
    dry_run: bool = typer.Option(False, "--dry-run", help="Preview steps without executing"),
    config: str = typer.Option("cobot.yaml", "--config", "-c"),
    robot: str = typer.Option(None, "--robot", "-r"),
) -> None:
    """Execute a scenario directly without entering the REPL."""
    session = _build_session(config, robot)

    async def _run() -> None:
        await _connect_session(session)
        from cobot.cli.commands.scenario import cmd_run
        await cmd_run(session, scenario, dry_run=dry_run)

    asyncio.run(_run())


@app.command()
def status(
    config: str = typer.Option("cobot.yaml", "--config", "-c"),
    robot: str = typer.Option(None, "--robot", "-r"),
) -> None:
    """Print robot status and exit."""
    session = _build_session(config, robot)

    async def _run() -> None:
        await _connect_session(session)
        from cobot.cli.commands.status import cmd_status
        await cmd_status(session)

    asyncio.run(_run())


@app.command()
def ingest(
    path: str = typer.Option("./manuals", "--path", help="Directory of robot manual PDFs"),
    config: str = typer.Option("cobot.yaml", "--config", "-c"),
) -> None:
    """Ingest robot manual PDFs into the RAG knowledge base (requires [rag] extras)."""
    from cobot.rag.ingest import main as ingest_main
    import sys
    sys.argv = ["cobot-claw ingest", "--path", path, "--config", config]
    ingest_main()


if __name__ == "__main__":
    app()
