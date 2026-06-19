from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from rich.align import Align
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

_LOGO = """\
 ██████╗ ██████╗ ██████╗  ██████╗ ████████╗     ██████╗██╗      █████╗ ██╗    ██╗
██╔════╝██╔═══██╗██╔══██╗██╔═══██╗╚══██╔══╝    ██╔════╝██║     ██╔══██╗██║    ██║
██║     ██║   ██║██████╔╝██║   ██║   ██║       ██║     ██║     ███████║██║ █╗ ██║
██║     ██║   ██║██╔══██╗██║   ██║   ██║       ██║     ██║     ██╔══██║██║███╗██║
╚██████╗╚██████╔╝██████╔╝╚██████╔╝   ██║       ╚██████╗███████╗██║  ██║╚███╔███╔╝
 ╚═════╝ ╚═════╝ ╚═════╝  ╚═════╝   ╚═╝        ╚═════╝╚══════╝╚═╝  ╚═╝ ╚══╝╚══╝ """

_PROVIDER_ICONS = {
    "anthropic": ("🟣", "Anthropic"),
    "openai":    ("🟢", "OpenAI"),
    "local":     ("🔵", "Local (vLLM)"),
}


def print_welcome(
    cfg: "CobotConfig",
    robot: "RobotConfig",
    connected: bool,
    mcp_status: dict[str, bool] | None = None,
    recent_scenarios: list[str] | None = None,  # unused — kept for call-site compat
) -> None:
    cwd = Path.cwd()
    mcp_status = mcp_status or {}
    provider = cfg.llm.provider.lower()
    provider_icon, provider_label = _PROVIDER_ICONS.get(provider, ("⚪", provider))

    # ── Logo header (full width, centered) ───────────────────────────────────
    console.print()
    console.print(Align(Text(_LOGO, style="bold cyan", justify="center"), align="center"))
    console.print()

    # ── Left panel ───────────────────────────────────────────────────────────
    # Robot info table
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
        return Text("?", style="dim")

    mcp = Table(show_header=False, box=None, padding=(0, 1), show_edge=False)
    mcp.add_column(justify="right", style="dim")
    mcp.add_column(justify="left", style="dim")
    mcp.add_column(justify="left")
    mcp.add_row("arm", robot.arm_mcp_url, _mcp_icon("arm"))
    if robot.gripper_mcp_url:
        mcp.add_row("gripper", robot.gripper_mcp_url, _mcp_icon("gripper"))

    # Provider / model
    llm_info = Table(show_header=False, box=None, padding=(0, 1), show_edge=False)
    llm_info.add_column(justify="right", style="dim")
    llm_info.add_column(justify="left", style="dim")
    llm_info.add_row("provider", f"{provider_icon}  {provider_label}")
    llm_info.add_row("model", cfg.agents.orchestrator.model)

    footer = Text(f"\n{cwd}", style="dim", justify="center")

    left = Panel(
        Group(
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
         "Routes commands, delegates to specialist agents"),
        ("robotic",      cfg.agents.scenario.model,
         "Robot interactions — status, moves, tools, scenarios"),
        ("vision",       cfg.agents.vision.model,
         "3D camera perception and object manipulation"),
        ("config",       cfg.agents.config.model,
         "Manual Q&A, teach-by-demo, scenario authoring"),
        ("artistic",     cfg.agents.artistic.model,
         "Choreography, gesture, theatre, ritual  [dim]/artistic[/dim]"),
    ]

    agents_table = Table(show_header=False, box=None, padding=(0, 2), show_edge=False)
    agents_table.add_column(style="cyan",    no_wrap=True)
    agents_table.add_column(style="dim",     no_wrap=True)
    agents_table.add_column(style="dim italic")
    for name, model, desc in _AGENT_DESCRIPTIONS:
        agents_table.add_row(name, model, desc)

    tips_header = Text("\nCommands\n", style="bold")
    tips = Table(show_header=False, box=None, padding=(0, 1), show_edge=False)
    tips.add_column(style="cyan",  no_wrap=True)
    tips.add_column(style="dim")
    tips.add_row("/status",    "robot state")
    tips.add_row("/run",       "<name> [--dry]  execute scenario")
    tips.add_row("/teach",     "jog joints, record waypoints")
    tips.add_row("/artistic",  "enter artistic mode")
    tips.add_row("/help",      "all commands")

    hint = Text("\nOr just type naturally — the agent will figure it out.", style="dim italic")

    right = Panel(
        Group(
            Text("Agents\n", style="bold"),
            agents_table,
            tips_header,
            tips,
            hint,
        ),
        expand=True,
        border_style="dim",
    )

    grid = Table.grid(expand=True)
    grid.add_column(ratio=1)
    grid.add_column(ratio=2)
    grid.add_row(left, right)
    console.print(grid)
    console.print()


# ---------------------------------------------------------------------------
# Tool call / result display
# ---------------------------------------------------------------------------

# Orchestrator-level tools — everything else is treated as nested (depth 1)
_TOP_LEVEL_TOOLS = {"run_vision_task", "run_robotic_task", "run_config_task"}


def _tool_depth(tool_name: str) -> int:
    return 0 if tool_name in _TOP_LEVEL_TOOLS else 1


def print_reasoning(text: str, max_chars: int = 400) -> None:
    """Display model reasoning / thinking text in a muted style."""
    truncated = text[:max_chars] + ("…" if len(text) > max_chars else "")
    from rich.padding import Padding
    console.print(
        Padding(
            Panel(
                Text(truncated, style="dim italic"),
                border_style="dim",
                padding=(0, 1),
                expand=False,
            ),
            (0, 0, 0, 2),
        )
    )


def print_tool_call(tool_name: str, args: dict) -> None:
    depth = _tool_depth(tool_name)
    if depth == 0:
        # Top-level orchestrator call — prominent, full cyan
        console.print(
            f"\n  [bold cyan]▶  {tool_name}[/bold cyan]"
            f"[dim]({', '.join(f'{k}={repr(v)}' for k,v in args.items())})[/dim]"
        )
    else:
        # Nested agent tool call — indented, blue
        console.print(
            f"     [blue]▸[/blue] [cyan]{tool_name}[/cyan]"
            f"[dim]({', '.join(f'{k}={repr(v)}' for k,v in args.items())})[/dim]"
        )


def print_tool_result(ok: bool, summary: str, tool_name: str = "") -> None:
    depth  = _tool_depth(tool_name) if tool_name else 1
    icon   = "✓" if ok else "✗"
    color  = "green" if ok else "red"
    indent = "  " if depth == 0 else "     "
    body   = summary[:160]
    if ok:
        console.print(f"{indent}[bold {color}]{icon}[/bold {color}] {body}")
    else:
        console.print(f"{indent}[bold {color}]{icon}[/bold {color}] [dim]{body}[/dim]")


def print_user_message(text: str) -> None:
    console.print(f"\n  [bold white]you[/bold white]  [white]{text}[/white]\n")


def print_agent_response(text: str) -> None:
    from rich.padding import Padding
    console.print()
    console.print(Rule(style="dim"))
    console.print(
        Padding(Markdown(text), (0, 2)),
    )
    console.print(Rule(style="dim"))
    console.print()


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
