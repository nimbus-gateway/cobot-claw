"""
/artistic — ArtisticAgent sub-REPL.

Opens a dedicated interaction loop where the arm is treated as a performer.
The main REPL is suspended until the user types /done.

Commands within artistic mode:
  /observe    Ask the vision agent to describe the current workspace
  /narrate    Ask the agent to explain its last artistic choice
  /skills     List active skills and their descriptions
  /done       Return to the main REPL
"""

from __future__ import annotations

import asyncio
from pathlib import Path

from prompt_toolkit import PromptSession
from prompt_toolkit.formatted_text import HTML
from prompt_toolkit.history import FileHistory

from cobot.cli.renderer import console, print_error

_ARTISTIC_PROMPT = HTML(
    "<ansiyellow><b>✦</b></ansiyellow> <ansiwhite>&gt;</ansiwhite> "
)

_WELCOME = """\

  [bold yellow]✦  Artistic Mode[/bold yellow]

  The arm is a performer. Movement is expression.
  Type naturally — describe what you want to happen, not how to make it happen.

  [dim]Active skills:[/dim]  {skills}

  [dim]/observe[/dim]  see the workspace   [dim]/narrate[/dim]  hear the agent's reasoning
  [dim]/skills[/dim]   list active skills  [dim]/done[/dim]     return to main REPL
"""

_HELP = """
  [bold]Artistic mode commands[/bold]
    [cyan]/observe[/cyan]      Observe the current workspace via camera
    [cyan]/narrate[/cyan]      Agent explains its last artistic choice
    [cyan]/skills[/cyan]       Show active skills
    [cyan]/done[/cyan]         Exit artistic mode, return to main REPL
"""


def _print_skills(cfg) -> None:
    import re
    skills_path = Path(cfg.agents.artistic.skills_path)
    for name in cfg.agents.artistic.active_skills:
        path = skills_path / f"{name}.md"
        description = ""
        if path.exists():
            raw = path.read_text(encoding="utf-8")
            m = re.search(r"^description:\s*(.+)$", raw, re.MULTILINE)
            if m:
                description = m.group(1).strip()
        console.print(f"  [yellow]✦[/yellow]  [bold]{name}[/bold]  [dim]{description}[/dim]")


async def cmd_artistic(session) -> None:
    from cobot.agents.artistic_agent import build_artistic_agent, ArtisticAgentDeps
    from pydantic_ai.messages import ModelMessage

    cfg = session.cfg

    if session.deps is None:
        print_error("No robot connected — artistic mode requires an active connection.")
        return

    skills_display = ", ".join(cfg.agents.artistic.active_skills) or "none configured"
    console.print(_WELCOME.format(skills=skills_display))

    artistic_deps = ArtisticAgentDeps(
        arm_mcp=session.deps.robotic_deps.arm_mcp,
        tool_mcps=session.deps.robotic_deps.tool_mcps,
        vision_deps=session.deps.vision_deps,
        cfg=cfg,
    )

    agent = build_artistic_agent(cfg)
    message_history: list[ModelMessage] = []

    history_path = Path.home() / ".cobot_artistic_history"
    prompt_session: PromptSession = PromptSession(
        history=FileHistory(str(history_path)),
        multiline=False,
    )

    while True:
        try:
            user_input = await prompt_session.prompt_async(_ARTISTIC_PROMPT)
        except KeyboardInterrupt:
            continue
        except EOFError:
            break

        text = user_input.strip()
        if not text:
            continue

        match text.lower():
            case "/done" | "/exit":
                console.print("  [dim]Returning to main REPL.[/dim]\n")
                break
            case "/help" | "?":
                console.print(_HELP)
                continue
            case "/skills":
                _print_skills(cfg)
                continue
            case "/observe":
                text = "Observe the current workspace and describe what you see — who is present, where they are, and the spatial arrangement."
            case "/narrate":
                text = "Narrate your most recent artistic choice: what you did, why you chose it, and what it was intended to express."

        await _run_artistic(text, agent, artistic_deps, message_history)


async def _run_artistic(
    text: str,
    agent,
    deps,
    message_history: list,
) -> None:
    from rich.live import Live
    from rich.spinner import Spinner

    live = Live(
        Spinner("dots", text="[dim]composing…[/dim]"),
        console=console,
        refresh_per_second=10,
        transient=True,
    )
    live.start()

    try:
        result = await agent.run(
            text,
            deps=deps,
            message_history=message_history or None,
        )
        message_history.clear()
        message_history.extend(result.all_messages())

        live.stop()
        console.print()
        output = result.output
        console.print(f"  [italic]{output.summary}[/italic]")
        if output.error:
            print_error(output.error)
        console.print()

    except asyncio.CancelledError:
        live.stop()
        console.print("\n  [yellow]Stopped.[/yellow]\n")
    except Exception as e:
        live.stop()
        print_error(str(e))
