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
import sys
import threading
import time as _time
from pathlib import Path

from prompt_toolkit import PromptSession
from prompt_toolkit.formatted_text import HTML
from prompt_toolkit.history import FileHistory
from prompt_toolkit.key_binding import KeyBindings

from cobot.cli.commands.artistic import cmd_artistic
from cobot.cli.commands.teach import cmd_teach
from cobot.cli.commands.scenario import cmd_run, cmd_scenarios
from cobot.cli.commands.status import cmd_status
from cobot.cli.commands.calibrate import cmd_calibrate
from cobot.cli.completer import CobotCompleter
from cobot.cli.renderer import (
    console,
    print_agent_response,
    print_error,
    print_reasoning as _print_reasoning,
    print_tool_call,
    print_tool_result,
    print_user_message,
    print_welcome,
)
from cobot.cli.session import CobotSession

try:
    import logfire as _logfire
except ImportError:
    _logfire = None  # type: ignore[assignment]


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
    from cobot.agents.deps import RoboticAgentDeps
    from cobot.agents.orchestrator import OrchestratorDeps
    from cobot.mcp.perception_contract import PerceptionClient
    from cobot.scenarios.executor import ScenarioExecutor

    arm = _OfflineMCP()
    perception = PerceptionClient(enabled=False)
    executor = ScenarioExecutor(arm, {}, perception)  # type: ignore[arg-type]
    robotic_deps = RoboticAgentDeps(arm, {}, executor, perception, session.cfg)  # type: ignore[arg-type]
    return OrchestratorDeps(robotic_deps=robotic_deps, cfg=session.cfg)


# ---------------------------------------------------------------------------
# Prompt
# ---------------------------------------------------------------------------

def _prompt_text() -> HTML:
    return HTML("<ansiwhite><b>&gt;</b></ansiwhite> ")


def _build_bindings(session: CobotSession | None = None) -> KeyBindings:
    kb = KeyBindings()

    @kb.add("c-c")
    def _(event):
        event.current_buffer.reset()

    voice = getattr(session, "voice", None)
    if voice is not None:
        keys = [k for k in (voice.cfg.key, voice.cfg.alt_key) if k]

        def _voice_key(event):
            # Second press while listening or transcribing: finish now and
            # send the command as if Enter was pressed.
            if voice.state in ("listening", "transcribing"):
                voice.send_now = True
                voice.cancel.set()
                event.app.invalidate()
                return
            voice.send_now = False
            event.app.create_background_task(_voice_to_prompt(event.app, voice, session))

        for k in keys:
            kb.add(k)(_voice_key)

    return kb


# ---------------------------------------------------------------------------
# Voice input (cobot/cli/voice.py)
# ---------------------------------------------------------------------------

async def _voice_stop_arm(session: CobotSession) -> str:
    """Brake the arm directly, bypassing the agents: stop_motion (stopj), or
    hold_position on arm servers without it."""
    arm = session.deps.robotic_deps.arm_mcp if session.deps else None
    if arm is None:
        return "no arm connection"
    last = ""
    for tool in ("stop_motion", "hold_position"):
        try:
            res = await arm.call_tool(tool, {})
        except Exception as e:  # noqa: BLE001 — e.g. tool not found on older servers
            last = f"{tool} failed: {e}"
            continue
        if res.get("ok"):
            return "arm stopped"
        last = f"{tool} failed: {res.get('message')}"
    return last


async def _voice_to_prompt(app, voice, session: CobotSession) -> None:
    """Record one spoken command and put the transcript in the prompt."""
    from cobot.cli.voice import is_stop_command

    if not voice.ready:
        voice.message = "speech model still loading…" if voice.state == "loading" else voice.message
        app.invalidate()
        return
    loop = asyncio.get_running_loop()
    voice.message = ""
    text = ""
    try:
        voice.state = "listening"
        app.invalidate()
        def _recorded() -> None:          # called from the recording thread
            voice.state = "transcribing"
            loop.call_soon_threadsafe(app.invalidate)

        text = await loop.run_in_executor(None, voice.listen_and_transcribe, _recorded)
        if text is None:
            voice.message = "didn't hear anything"
            return
    except Exception as e:  # noqa: BLE001
        voice.message = f"voice error: {e}"
        return
    finally:
        voice.state = "ready"
        app.invalidate()

    if not text:
        voice.message = "couldn't make out any words"
    elif is_stop_command(text):
        voice.message = f"heard “{text}”: {await _voice_stop_arm(session)}"
    else:
        buf = app.current_buffer
        buf.text = text
        buf.cursor_position = len(text)
        if voice.cfg.auto_send or voice.send_now:
            buf.validate_and_handle()
        else:
            voice.message = "check it, then Enter to send"
    app.invalidate()


# ---------------------------------------------------------------------------
# Slash commands
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------
# Manual ingest
# ---------------------------------------------------------------------------

async def _cmd_ingest(session: CobotSession, path: str) -> None:
    """Index robot manuals into the RAG knowledge base."""
    from rich.live import Live
    from rich.spinner import Spinner

    if session.deps is None or session.deps.config_deps is None:
        print_error("Config Agent not available.")
        return

    live = Live(
        Spinner("dots", text=f"[dim]Indexing {path}…[/dim]"),
        console=console,
        refresh_per_second=10,
        transient=True,
    )
    live.start()
    try:
        result = await session.deps.config_deps.rag_pipeline.ingest(path)
        live.stop()
        if result.get("ok"):
            console.print(
                f"\n  [green]✓[/green]  Indexed [bold]{result['indexed']}[/bold] document(s) "
                f"into [dim]{result['db_path']}[/dim]\n"
            )
        else:
            print_error(result.get("message", "Ingest failed."))
    except Exception as e:
        live.stop()
        print_error(f"Ingest failed: {e}")


# ---------------------------------------------------------------------------
# Slash commands
# ---------------------------------------------------------------------------

async def _cmd_vision_debug(session: CobotSession, arg: str) -> None:
    """Start or stop the live OpenCV camera debug window."""
    from cobot.camera.stream import CameraStream

    currently_on = session.camera_stream is not None and session.camera_stream._debug_enabled

    want_on = {"on": True, "off": False, "toggle": not currently_on}.get(arg, not currently_on)

    if want_on:
        if session.camera_stream is None:
            fps = 10
            session.camera_stream = CameraStream(fps=fps)
            session.camera_stream.start()
            # Push stream reference into live vision deps if session is connected
            if session.deps and session.deps.vision_deps:
                session.deps.vision_deps.camera_stream = session.camera_stream
            console.print(f"  [dim]Camera stream started at {fps}fps[/dim]")
        session.camera_stream.enable_debug_window()
        console.print("  [green]✓[/green]  Vision debug window opened  [dim](press Q in window to close)[/dim]")
    else:
        if session.camera_stream is not None:
            session.camera_stream.stop()
            session.camera_stream = None
            if session.deps and session.deps.vision_deps:
                session.deps.vision_deps.camera_stream = None
            console.print("  [yellow]Vision debug stopped — camera released[/yellow]")
        else:
            console.print("  [dim]Vision debug is not active[/dim]")


async def _cmd_reload_config(session: CobotSession) -> None:
    """Reload cobot.yaml and reconnect MCP servers."""
    from cobot.config import load_config
    from cobot.cli.app import _connect_session

    try:
        new_cfg = load_config(session.config_path)
    except Exception as e:
        print_error(f"Failed to load {session.config_path}: {e}")
        return

    session.cfg = new_cfg
    session.active_robot = new_cfg.default_robot
    session.deps = None
    session.mcp_status = {}

    console.print(f"  [dim]Config reloaded from {session.config_path} — reconnecting…[/dim]")
    try:
        await _connect_session(session)
        status = "connected" if session.robot_connected else "partially connected"
        console.print(f"  [green]✓[/green] Config reloaded and {status}")
    except Exception as e:
        print_error(f"Reconnect failed after reload: {e}")


_HELP_TEXT = """
[bold]Slash commands[/bold]
  [cyan]/status[/cyan]              Robot state and motion info
  [cyan]/scenarios[/cyan]           List saved scenarios
  [cyan]/run[/cyan] [dim]<name>[/dim]          Execute a scenario
  [cyan]/run[/cyan] [dim]<name> --dry[/dim]    Preview steps without moving
  [cyan]/teach[/cyan] [dim][step][/dim]         Keyboard-jog joints, record waypoints + gripper actions
  [cyan]/calibrate[/cyan]            Run hand-eye calibration (moves arm through 4 poses)
  [cyan]/calibrate --check[/cyan]   Report current calibration error without saving
  [cyan]/ingest[/cyan] [dim][path][/dim]       Index robot manuals into RAG knowledge base
  [cyan]/artistic[/cyan]             Enter artistic mode — choreography, gesture, theatre, ritual
  [cyan]/vision-debug[/cyan] [dim][on|off][/dim]  Toggle live OpenCV camera window with detection overlays
  [cyan]/reload-config[/cyan]        Reload cobot.yaml and reconnect MCP servers
  [cyan]/reconnect[/cyan]           Reconnect to robot MCP servers
  [cyan]/clear[/cyan]               Clear the screen
  [cyan]/exit[/cyan]                Quit

[bold]Natural language examples[/bold]
  [dim]move to home position[/dim]
  [dim]run the pick and place sequence[/dim]
  [dim]what is the current robot state?[/dim]
  [dim]open the gripper slowly[/dim]
  [dim]what is the max payload for the ur3e?[/dim]
  [dim]how do I configure the gripper force?[/dim]
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

        case "/calibrate":
            check_only = "--check" in parts
            dry_run    = "--dry" in parts
            await cmd_calibrate(session, check_only=check_only, dry_run=dry_run)

        case "/teach":
            step = float(parts[1]) if len(parts) > 1 else 0.02
            await cmd_teach(session, step=step)

        case "/ingest":
            path = parts[1] if len(parts) > 1 else "./manuals"
            await _cmd_ingest(session, path)

        case "/vision-debug":
            arg = parts[1].lower() if len(parts) > 1 else "toggle"
            await _cmd_vision_debug(session, arg)

        case "/artistic":
            await cmd_artistic(session)

        case "/reload-config":
            await _cmd_reload_config(session)

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
            session.message_history = []

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
    print_user_message(text)

    import json

    live = Live(
        Spinner("dots", text="[dim]thinking…[/dim]"),
        console=console,
        refresh_per_second=10,
        transient=True,
    )
    live.start()

    try:
        async with agent.iter(text, deps=deps, message_history=session.message_history or None) as run:
            async for node in run:
                if hasattr(node, "model_response"):
                    parts = node.model_response.parts
                    has_tool_calls = any(hasattr(p, "args_as_dict") for p in parts)

                    for part in parts:
                        # Extended thinking block (Claude extended thinking)
                        if hasattr(part, "thinking"):
                            live.stop()
                            _print_reasoning(str(part.thinking))
                            live = Live(
                                Spinner("dots", text="[dim]thinking…[/dim]"),
                                console=console,
                                refresh_per_second=10,
                                transient=True,
                            )
                            live.start()

                        # Text alongside tool calls = inline reasoning before action
                        elif hasattr(part, "content") and isinstance(part.content, str) and has_tool_calls:
                            reasoning = part.content.strip()
                            if reasoning:
                                live.stop()
                                _print_reasoning(reasoning)
                                live = Live(
                                    Spinner("dots", text="[dim]thinking…[/dim]"),
                                    console=console,
                                    refresh_per_second=10,
                                    transient=True,
                                )
                                live.start()

                        # Tool call
                        elif hasattr(part, "args_as_dict"):
                            live.stop()
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
                            print_tool_result(ok, str(content)[:160], tool_name=part.tool_name)
                            live = Live(
                                Spinner("dots", text="[dim]thinking…[/dim]"),
                                console=console,
                                refresh_per_second=10,
                                transient=True,
                            )
                            live.start()

        output = run.result.output
        session.message_history = run.result.all_messages()
        console.print()
        if isinstance(output, str):
            print_agent_response(output)
        else:
            print_agent_response(str(output))
        console.print()

    except asyncio.CancelledError:
        raise   # let _run_cancellable handle the message
    except Exception as e:
        print_error(str(e))
    finally:
        try:
            live.stop()
        except Exception:
            pass


# ---------------------------------------------------------------------------
# ESC-to-cancel support
# ---------------------------------------------------------------------------

def _watch_for_escape(cancel_event: asyncio.Event, loop: asyncio.AbstractEventLoop) -> None:
    """Background thread: set cancel_event when ESC is pressed."""
    try:
        if sys.platform == "win32":
            import msvcrt
            while not cancel_event.is_set():
                if msvcrt.kbhit():
                    ch = msvcrt.getwch()
                    if ch == "\x1b":
                        loop.call_soon_threadsafe(cancel_event.set)
                        return
                _time.sleep(0.05)
        else:
            import select
            import termios
            import tty
            fd = sys.stdin.fileno()
            old = termios.tcgetattr(fd)
            try:
                tty.setraw(fd)
                while not cancel_event.is_set():
                    r, _, _ = select.select([sys.stdin], [], [], 0.05)
                    if r:
                        ch = sys.stdin.read(1)
                        if ch == "\x1b":
                            loop.call_soon_threadsafe(cancel_event.set)
                            return
            finally:
                termios.tcsetattr(fd, termios.TCSADRAIN, old)
    except Exception:
        pass


async def _run_cancellable(text: str, session: CobotSession) -> None:
    """Run _handle_agent_message with ESC-to-cancel support."""
    loop = asyncio.get_running_loop()
    cancel_event = asyncio.Event()

    console.print("  [dim]Press ESC to stop[/dim]")

    # Spoken "stop": cancel the task and hold the arm, without the agents.
    voice = getattr(session, "voice", None)
    stop_listener_done = threading.Event()
    hold_tasks: list[asyncio.Task] = []
    if voice is not None and voice.ready and voice.cfg.stop_while_running:
        def _spoken_stop(heard: str) -> None:
            console.print(f"\n  [red]Heard “{heard}”: stopping the arm[/red]")
            hold_tasks.append(asyncio.create_task(_voice_stop_arm(session)))
            cancel_event.set()

        threading.Thread(
            target=voice.listen_for_stop,
            args=(stop_listener_done, lambda heard: loop.call_soon_threadsafe(_spoken_stop, heard)),
            daemon=True,
            name="voice-stop",
        ).start()
        console.print("  [dim]…or say “stop”[/dim]")

    watcher = threading.Thread(
        target=_watch_for_escape,
        args=(cancel_event, loop),
        daemon=True,
    )
    watcher.start()

    task = asyncio.create_task(_handle_agent_message(text, session))

    async def _wait_for_cancel():
        await cancel_event.wait()
        task.cancel()

    cancel_task = asyncio.create_task(_wait_for_cancel())

    try:
        await task
    except asyncio.CancelledError:
        console.print("\n  [yellow]Stopped.[/yellow]\n")
    finally:
        stop_listener_done.set()
        for t in hold_tasks:
            console.print(f"  [dim]{await t}[/dim]")
        cancel_event.set()   # stop watcher thread
        cancel_task.cancel()
        try:
            await cancel_task
        except asyncio.CancelledError:
            pass


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

    if session.cfg.voice.enabled and session.voice is None:
        try:
            from cobot.cli.voice import VoiceInput
            session.voice = VoiceInput(session.cfg.voice)
            session.voice.preload()
        except ImportError:
            console.print(' [yellow]⚠[/yellow]  voice.enabled is set but faster-whisper isn\'t installed: '
                          'pip install -e ".[voice]"')

    history_path = Path.home() / ".cobot_history"
    voice_toolbar = {}
    if session.voice is not None:
        voice_toolbar = {
            "bottom_toolbar": lambda: session.voice.toolbar(),
            "rprompt": lambda: HTML(session.voice.rprompt()),   # on the prompt line itself
            "refresh_interval": 0.5,
        }
    prompt_session: PromptSession = PromptSession(
        history=FileHistory(str(history_path)),
        completer=CobotCompleter(session),
        key_bindings=_build_bindings(session),
        **voice_toolbar,
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

        if _logfire and not text.startswith("/"):
            _logfire.info(
                "repl.user_utterance",
                text=text,
                word_count=len(text.split()),
                history_turn_index=len(session.history) - 1,
                session_id=session.session_id,
            )

        if text.startswith("/") or text == "?":
            await _handle_slash_command(text, session)
        else:
            await _run_cancellable(text, session)
