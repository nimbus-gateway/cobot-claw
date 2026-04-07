from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from rich.align import Align
from rich.columns import Columns
from rich.console import Console, Group
from rich.live import Live
from rich.markdown import Markdown
from rich.panel import Panel
from rich.rule import Rule
from rich.spinner import Spinner
from rich.table import Table
from rich.text import Text

if TYPE_CHECKING:
    from cobot.config import CobotConfig, RobotConfig

console = Console()


# ---------------------------------------------------------------------------
# Welcome screen
# ---------------------------------------------------------------------------

_ROBOT_ART = """\
  ╔════╗
  ║ ██ ║
╔═╩════╩═╗
║        ║
║        ║
╚══╦══╦══╝
   ║  ║
  ═╩══╩═"""

_PROVIDER_ICONS = {
    "anthropic": ("🟣", "Anthropic"),
    "openai":    ("🟢", "OpenAI"),
    "local":     ("🔵", "Local (Ollama)"),
}


def print_welcome(
    cfg: "CobotConfig",
    robot: "RobotConfig",
    connected: bool,
    recent_scenarios: list[str] | None = None,
    mcp_status: dict[str, bool] | None = None,
) -> None:
    cwd = Path.cwd()
    mcp_status = mcp_status or {}
    provider = cfg.llm.provider.lower()
    provider_icon, provider_label = _PROVIDER_ICONS.get(provider, ("⚪", provider))

    # ── Left panel ───────────────────────────────────────────────────────────
    # Title
    title = Text(justify="center")
    title.append("cobot-claw", style="bold cyan")
    title.append("  v0.1", style="dim")

    # Robot art — centered
    art = Align(Text(_ROBOT_ART, style="cyan"), align="center")

    # Robot info table (centered via table column alignment)
    info = Table(show_header=False, box=None, padding=(0, 1), show_edge=False)
    info.add_column(justify="right", style="dim")
    info.add_column(justify="left")
    info.add_row("name",  Text(robot.name, style="bold white"))
    info.add_row("type",  Text(robot.model_type.upper(), style="white"))
    info.add_row("ip",    Text(robot.ip, style="dim"))

    if connected:
        conn_text = Text("● connected", style="bold green")
    else:
        conn_text = Text("✗ disconnected", style="bold red")
    info.add_row("", conn_text)

    # MCP endpoints
    def _mcp_icon(key: str) -> Text:
        state = mcp_status.get(key)
        if state is True:
            return Text("✓", style="green")
        if state is False:
            return Text("✗", style="red")
        return Text("?", style="dim")   # not yet attempted

    mcp = Table(show_header=False, box=None, padding=(0, 1), show_edge=False)
    mcp.add_column(justify="right", style="dim")
    mcp.add_column(justify="left", style="dim")
    mcp.add_column(justify="left")
    mcp.add_row("arm", robot.arm_mcp_url, _mcp_icon("arm"))
    if robot.gripper_mcp_url:
        mcp.add_row("gripper", robot.gripper_mcp_url, _mcp_icon("gripper"))

    # Provider / model status
    llm_info = Table(show_header=False, box=None, padding=(0, 1), show_edge=False)
    llm_info.add_column(justify="right", style="dim")
    llm_info.add_column(justify="left", style="dim")
    llm_info.add_row("provider", f"{provider_icon}  {provider_label}")
    llm_info.add_row("model", cfg.agents.orchestrator.model)

    footer = Text(f"\n{cwd}", style="dim", justify="center")

    left = Panel(
        Group(
            Text(""),
            title,
            Text(""),
            art,
            Text(""),
            Rule(style="dim"),
            Text(""),
            Align(info, align="center"),
            Text(""),
            Rule(style="dim"),
            Text(""),
            Align(mcp, align="center"),
            Text(""),
            Rule(style="dim"),
            Text(""),
            Align(llm_info, align="center"),
            footer,
        ),
        expand=True,
        border_style="dim",
    )

    # ── Right panel ──────────────────────────────────────────────────────────
    _AGENT_DESCRIPTIONS = [
        ("orchestrator", cfg.agents.orchestrator.model,
         "Routes commands, delegates to scenario or config agent"),
        ("robotic",      cfg.agents.scenario.model,
         "All robot interactions — status, moves, tools, scenarios"),
        ("safety",       cfg.agents.safety.model,
         "Monitors for unsafe states and blocks dangerous moves"),
        ("config",       cfg.agents.config.model,
         "Manages configuration and manual queries  [dim](phase 6)[/dim]"),
    ]

    agents_table = Table(show_header=False, box=None, padding=(0, 2), show_edge=False)
    agents_table.add_column(style="cyan",    no_wrap=True)
    agents_table.add_column(style="dim",     no_wrap=True)
    agents_table.add_column(style="dim italic")
    for name, model, desc in _AGENT_DESCRIPTIONS:
        agents_table.add_row(name, model, desc)

    recent_header = Text("\nRecent Scenarios\n", style="bold")
    if recent_scenarios:
        recent_rows = Table(show_header=False, box=None, padding=(0, 2), show_edge=False)
        recent_rows.add_column(style="cyan")
        for s in recent_scenarios[:5]:
            recent_rows.add_row(s)
        recent_body = recent_rows
    else:
        recent_body = Text("  No recent scenarios", style="dim")

    tips_header = Text("\nCommands\n", style="bold")
    tips = Table(show_header=False, box=None, padding=(0, 1), show_edge=False)
    tips.add_column(style="cyan",  no_wrap=True)
    tips.add_column(style="dim")
    tips.add_row("/status",    "robot state")
    tips.add_row("/scenarios", "list saved scenarios")
    tips.add_row("/run",       "<name> [--dry]  execute scenario")
    tips.add_row("/help",      "all commands")

    hint = Text("\nOr just type naturally — the agent will figure it out.", style="dim italic")

    right = Panel(
        Group(
            Text("Agents\n", style="bold"),
            agents_table,
            recent_header,
            recent_body,
            tips_header,
            tips,
            hint,
        ),
        expand=True,
        border_style="dim",
    )

    console.print(Columns([left, right], expand=True))
    console.print()


# ---------------------------------------------------------------------------
# Tool call / result display
# ---------------------------------------------------------------------------

def print_tool_call(tool_name: str, args: dict) -> None:
    args_str = ", ".join(f"{k}={repr(v)}" for k, v in args.items())
    console.print(f"  [dim]▸[/dim] [cyan]{tool_name}[/cyan][dim]({args_str})[/dim]")


def print_tool_result(ok: bool, summary: str) -> None:
    icon = "✓" if ok else "✗"
    color = "green" if ok else "red"
    console.print(f"  [{color}]{icon}[/{color}] [dim]{summary[:120]}[/dim]")


def print_agent_response(text: str) -> None:
    console.print(Markdown(text))


def print_error(message: str) -> None:
    console.print(f"  [red]✗[/red] {message}")


def print_hint(message: str) -> None:
    console.print(f"  [dim]{message}[/dim]")


# ---------------------------------------------------------------------------
# Status table
# ---------------------------------------------------------------------------

def print_status_table(status: dict) -> None:
    table = Table(show_header=False, box=None, padding=(0, 2))
    table.add_row("[dim]robot ip[/dim]", str(status.get("robot_ip", "—")))
    table.add_row("[dim]has joint state[/dim]", str(status.get("has_joint_state", "—")))
    table.add_row("[dim]headless mode[/dim]", str(status.get("headless_mode", "—")))
    motion = status.get("motion", {})
    table.add_row("[dim]motion complete[/dim]", str(motion.get("complete", "—")))

    joints = status.get("joints")
    if joints:
        joint_str = "  ".join(
            f"{j['joint'].replace('_joint', '')}={j['value_deg']:.1f}°"
            for j in joints
        )
        table.add_row("[dim]joints[/dim]", joint_str)

    tcp = status.get("tcp_position_m")
    if tcp:
        table.add_row(
            "[dim]tcp position[/dim]",
            f"x={tcp.get('x', 0):.3f}  y={tcp.get('y', 0):.3f}  z={tcp.get('z', 0):.3f} m",
        )

    if warning := status.get("_warning"):
        table.add_row("[dim yellow]warning[/dim yellow]", f"[yellow]{warning}[/yellow]")
    console.print(table)


# ---------------------------------------------------------------------------
# Spinner context manager
# ---------------------------------------------------------------------------

def spinner(text: str) -> Live:
    return Live(Spinner("dots", text=f"[dim]{text}[/dim]"), console=console, refresh_per_second=10)
