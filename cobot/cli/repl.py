"""
Interactive REPL for cobot-claw.

Claude Code-style interface:
- Welcome screen with robot info, agent table, and tips panels
- Simple "> " prompt for natural language and slash commands
- Tool calls/results streamed inline as the agent works
- "? for shortcuts" bottom hint
- Works in disconnected mode — NLP still available, robot tools return offline notice
"""

from __future__ import annotations

import asyncio
from pathlib import Path

from prompt_toolkit import PromptSession
from prompt_toolkit.formatted_text import HTML
from prompt_toolkit.history import FileHistory
from prompt_toolkit.key_binding import KeyBindings

from cobot.cli.commands.scenario import cmd_run, cmd_scenarios
from cobot.cli.commands.status import cmd_status
from cobot.cli.completer import CobotCompleter
from cobot.cli.renderer import (
    console,
    print_agent_response,
    print_error,
    print_tool_call,
    print_tool_result,
    print_welcome,
)
from cobot.cli.session import CobotSession


# ---------------------------------------------------------------------------
# Offline stub deps (used when MCP servers are unreachable)
# ---------------------------------------------------------------------------

class _OfflineMCP:
    """Stub MCP client that returns a clean 'offline' response for every tool call."""

    async def call_tool(self, name: str, args: dict | None = None) -> dict:
        return {"ok": False, "message": "Robot offline — MCP servers not reachable."}

    async def list_tools(self):
        raise ConnectionError("Robot offline — MCP servers not reachable.")


def _make_offline_deps(session: CobotSession):
    from cobot.agents.deps import ScenarioAgentDeps
    from cobot.agents.orchestrator import OrchestratorDeps
    from cobot.mcp.perception_contract import PerceptionClient
    from cobot.scenarios.executor import ScenarioExecutor

    arm = _OfflineMCP()
    perception = PerceptionClient(enabled=False)
    executor = ScenarioExecutor(arm, {}, perception)  # type: ignore[arg-type]
    scenario_deps = ScenarioAgentDeps(arm, {}, executor, perception, session.cfg)  # type: ignore[arg-type]
    return OrchestratorDeps(scenario_deps=scenario_deps, cfg=session.cfg)


# ---------------------------------------------------------------------------
# Prompt
# ---------------------------------------------------------------------------

def _prompt_text() -> HTML:
    return HTML("<ansiwhite><b>&gt;</b></ansiwhite> ")


def _build_bindings() -> KeyBindings:
    kb = KeyBindings()

    @kb.add("c-c")
    def _(event):
        event.current_buffer.reset()

    return kb


# ---------------------------------------------------------------------------
# Slash commands
# ---------------------------------------------------------------------------

_HELP_TEXT = """
[bold]Slash commands[/bold]
  [cyan]/status[/cyan]              Robot state and motion info
  [cyan]/scenarios[/cyan]           List saved scenarios
  [cyan]/run[/cyan] [dim]<name>[/dim]          Execute a scenario
  [cyan]/run[/cyan] [dim]<name> --dry[/dim]    Preview steps without moving
  [cyan]/reconnect[/cyan]           Reconnect to robot MCP servers
  [cyan]/clear[/cyan]               Clear the screen
  [cyan]/exit[/cyan]                Quit

[bold]Natural language examples[/bold]
  [dim]move to home position[/dim]
  [dim]run the pick and place sequence[/dim]
  [dim]what is the current robot state?[/dim]
  [dim]open the gripper slowly[/dim]
"""


async def _handle_slash_command(text: str, session: CobotSession) -> None:
    parts = text.strip().split()
    cmd = parts[0].lower()

    match cmd:
        case "/help" | "?":
            console.print(_HELP_TEXT)

        case "/status":
            await cmd_status(session)

        case "/scenarios":
            await cmd_scenarios(session)

        case "/run":
            if len(parts) < 2:
                print_error("Usage: /run <scenario_name> [--dry]")
            else:
                dry = "--dry" in parts
                await cmd_run(session, parts[1], dry_run=dry)

        case "/reconnect":
            from cobot.cli.app import _connect_session
            console.print("  [dim]Reconnecting to MCP servers…[/dim]")
            try:
                await _connect_session(session)
                status = "connected" if session.robot_connected else "partially connected"
                console.print(f"  [green]✓[/green] {status}")
            except Exception as e:
                print_error(f"Reconnect failed: {e}")

        case "/clear":
            console.clear()

        case "/exit":
            raise SystemExit(0)

        case _:
            print_error(f"Unknown command: {cmd}  —  type [cyan]/help[/cyan] for a list")


# ---------------------------------------------------------------------------
# Agent message handler
# ---------------------------------------------------------------------------

async def _handle_agent_message(text: str, session: CobotSession) -> None:
    from cobot.agents.orchestrator import build_orchestrator
    from rich.live import Live
    from rich.spinner import Spinner

    agent = build_orchestrator(session.cfg)
    deps = session.deps or _make_offline_deps(session)
    console.print()

    import json

    try:
        live = Live(
            Spinner("dots", text="[dim]thinking…[/dim]"),
            console=console,
            refresh_per_second=10,
            transient=True,
        )
        live.start()
        first_node = True

        async with agent.iter(text, deps=deps) as run:
            async for node in run:
                if first_node:
                    live.stop()
                    first_node = False

                if hasattr(node, "model_response"):
                    for part in node.model_response.parts:
                        if hasattr(part, "args_as_dict"):
                            # new tool call → show a fresh spinner while it runs
                            live = Live(
                                Spinner("dots", text=f"[dim]{part.tool_name}…[/dim]"),
                                console=console,
                                refresh_per_second=10,
                                transient=True,
                            )
                            live.start()
                            print_tool_call(part.tool_name, part.args_as_dict())
                elif hasattr(node, "request"):
                    for part in node.request.parts:
                        if hasattr(part, "tool_name") and hasattr(part, "content") and not hasattr(part, "args_as_dict"):
                            live.stop()
                            content = getattr(part, "content", "")
                            ok = True
                            if isinstance(content, str):
                                try:
                                    result_dict = json.loads(content)
                                    ok = result_dict.get("ok", True) if isinstance(result_dict, dict) else True
                                except Exception:
                                    pass
                            print_tool_result(ok, str(content)[:120])
                            # spinner for next thinking phase
                            live = Live(
                                Spinner("dots", text="[dim]thinking…[/dim]"),
                                console=console,
                                refresh_per_second=10,
                                transient=True,
                            )
                            live.start()

        # ensure spinner is stopped before printing final response
        try:
            live.stop()
        except Exception:
            pass

        output = run.result.output
        console.print()
        if isinstance(output, str):
            print_agent_response(output)
        else:
            print_agent_response(str(output))
        console.print()

    except Exception as e:
        try:
            live.stop()
        except Exception:
            pass
        print_error(str(e))


# ---------------------------------------------------------------------------
# REPL loop
# ---------------------------------------------------------------------------

async def run_repl(session: CobotSession) -> None:
    store = Path(session.cfg.scenarios.store_path)
    recent = [f.stem for f in sorted(store.glob("*.yaml"), key=lambda f: f.stat().st_mtime, reverse=True)]

    print_welcome(
        cfg=session.cfg,
        robot=session.active_robot,
        connected=session.robot_connected,
        recent_scenarios=recent,
        mcp_status=session.mcp_status,
    )

    history_path = Path.home() / ".cobot_history"
    prompt_session: PromptSession = PromptSession(
        history=FileHistory(str(history_path)),
        completer=CobotCompleter(session),
        key_bindings=_build_bindings(),
        multiline=False,
    )

    console.print(" [dim]? for shortcuts[/dim]    [dim]/help for all commands[/dim]\n")

    while True:
        try:
            user_input = await prompt_session.prompt_async(_prompt_text())
        except KeyboardInterrupt:
            continue
        except EOFError:
            break

        text = user_input.strip()
        if not text:
            continue

        if text == "?":
            text = "/help"

        session.history.append({"role": "user", "content": text})

        if text.startswith("/") or text == "?":
            await _handle_slash_command(text, session)
        else:
            await _handle_agent_message(text, session)
