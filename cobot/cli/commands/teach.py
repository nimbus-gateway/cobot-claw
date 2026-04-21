"""
/teach — keyboard teach mode.

Jog joints with the keyboard and record a sequence of arm positions and
gripper actions.  Each keypress calls jog_joint() on the MCP server — no
LLM, no agent overhead.

Key bindings
------------
  Q / A   →  Joint 0  (base)         +/-
  W / S   →  Joint 1  (shoulder)     +/-
  E / D   →  Joint 2  (elbow)        +/-
  R / F   →  Joint 3  (wrist 1)      +/-
  T / G   →  Joint 4  (wrist 2)      +/-
  Y / H   →  Joint 5  (wrist 3)      +/-

  +  / =  →  increase step size
  -       →  decrease step size
  SPACE   →  record joint waypoint
  O       →  gripper open  (recorded as step)
  C       →  gripper close (recorded as step)
  X / ESC →  exit teach mode
"""

from __future__ import annotations

import asyncio
import math
import sys
import threading
import time
from pathlib import Path
from typing import TYPE_CHECKING

import yaml
from rich.live import Live
from rich.table import Table
from rich.text import Text
from rich.panel import Panel
from rich.console import Group

from cobot.cli.renderer import console

if TYPE_CHECKING:
    from cobot.cli.session import CobotSession


# ──────────────────────────────────────────────────────────────────────────────
# Constants
# ──────────────────────────────────────────────────────────────────────────────

JOINT_NAMES = ["base", "shoulder", "elbow", "wrist_1", "wrist_2", "wrist_3"]
KEY_PAIRS   = ["Q/A", "W/S", "E/D", "R/F", "T/G", "Y/H"]

KEY_MAP: dict[str, tuple[int, int]] = {
    "q": (0, +1), "a": (0, -1),
    "w": (1, +1), "s": (1, -1),
    "e": (2, +1), "d": (2, -1),
    "r": (3, +1), "f": (3, -1),
    "t": (4, +1), "g": (4, -1),
    "y": (5, +1), "h": (5, -1),
}

STEP_MIN = 0.005
STEP_MAX = 0.50
STEP_INC = 0.005
DEFAULT_STEP  = 0.02
DEFAULT_SPEED = 0.30
DEFAULT_ACCEL = 0.50

GRIPPER_OPEN_PARAMS  = {"speed": 100, "force": 100}
GRIPPER_CLOSE_PARAMS = {"speed": 100, "force": 100}


# ──────────────────────────────────────────────────────────────────────────────
# Raw key reader (cross-platform, non-blocking)
# ──────────────────────────────────────────────────────────────────────────────

def _read_key_win() -> str | None:
    import msvcrt
    if msvcrt.kbhit():
        ch = msvcrt.getwch()
        if ch in ("\x00", "\xe0"):
            msvcrt.getwch()
            return None
        return ch
    return None


def _read_key_unix() -> str | None:
    import select, tty, termios
    fd = sys.stdin.fileno()
    old = termios.tcgetattr(fd)
    try:
        tty.setraw(fd)
        r, _, _ = select.select([sys.stdin], [], [], 0.0)
        if r:
            return sys.stdin.read(1)
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old)
    return None


def _get_key() -> str | None:
    if sys.platform == "win32":
        return _read_key_win()
    return _read_key_unix()


# ──────────────────────────────────────────────────────────────────────────────
# Display helpers
# ──────────────────────────────────────────────────────────────────────────────

def _step_label(step: dict) -> str:
    """One-line summary of a recorded step."""
    if step["type"] == "move_joints":
        j = step["joints"]
        short = f"[{', '.join(f'{v:.3f}' for v in j)}]"
        return f"[yellow]wp[/yellow]  {short}"
    if step["type"] == "tool_action":
        action = step["action"]
        color = "green" if action == "open" else "magenta"
        return f"[{color}]gripper {action}[/{color}]"
    return step["type"]


def _render(
    joints: list[float] | None,
    step: float,
    speed: float,
    status: str,
    status_ok: bool,
    steps: list[dict],
    active_joint: int | None,
) -> Group:

    # Joint table
    tbl = Table.grid(padding=(0, 2))
    tbl.add_column(width=3)
    tbl.add_column(width=10)
    tbl.add_column(width=10, justify="right")
    tbl.add_column(width=9,  justify="right")
    tbl.add_column(width=5)

    for i, name in enumerate(JOINT_NAMES):
        active = (i == active_joint)
        prefix = "[bold cyan]►[/bold cyan]" if active else " "
        if joints:
            rad_str = f"{joints[i]:+.5f}"
            deg_str = f"{math.degrees(joints[i]):+.1f}°"
            color   = "bold green" if active else "white"
        else:
            rad_str = "?.??????"
            deg_str = "?.?°"
            color   = "dim"
        tbl.add_row(
            prefix,
            f"[{color}]J{i} {name}[/{color}]",
            f"[{color}]{rad_str}[/{color}]",
            f"[{color}]{deg_str}[/{color}]",
            f"[dim]{KEY_PAIRS[i]}[/dim]",
        )

    status_color = "green" if status_ok else "red"
    status_line  = Text(f" {status}", style=status_color)

    info_line = Text(
        f" step: {step:.4f} rad ({math.degrees(step):.2f}°)   "
        f"speed: {speed:.2f} rad/s   [+/-] adjust",
        style="dim",
    )

    # Recorded steps (last 4)
    step_lines: list[Text] = []
    if steps:
        step_lines.append(Text(f" {len(steps)} step(s) recorded", style="cyan"))
        for idx, s in enumerate(steps[-4:], max(1, len(steps) - 3)):
            step_lines.append(Text(f"  #{idx} {_step_label(s)}", style="dim"))

    help_line = Text(
        " SPACE=record waypoint   O=gripper open   C=gripper close   X/ESC=exit",
        style="dim",
    )

    panel = Panel(
        Group(tbl, info_line, status_line, *step_lines, help_line),
        title="[bold cyan] Teach Mode [/bold cyan]",
        border_style="cyan",
    )
    return Group(panel)


# ──────────────────────────────────────────────────────────────────────────────
# Steps → YAML scenario
# ──────────────────────────────────────────────────────────────────────────────

def _save_scenario(name: str, steps: list[dict], store_path: str) -> Path:
    scenario = {
        "name": name,
        "description": f"Recorded via /teach ({len(steps)} steps)",
        "robot": "ur3e",
        "steps": steps,
    }
    path = Path(store_path) / f"{name}.yaml"
    path.write_text(yaml.dump(scenario, sort_keys=False, allow_unicode=True))
    return path


# ──────────────────────────────────────────────────────────────────────────────
# Main teach command
# ──────────────────────────────────────────────────────────────────────────────

async def cmd_teach(
    session: "CobotSession",
    step: float = DEFAULT_STEP,
    speed: float = DEFAULT_SPEED,
    accel: float = DEFAULT_ACCEL,
) -> None:
    if session.deps is None:
        console.print("[red]No robot connection.[/red]")
        return

    deps = session.deps.robotic_deps
    arm  = deps.arm_mcp
    # Gripper MCP — may be None if not connected
    gripper = deps.tool_mcps.get("gripper")

    recorded_steps: list[dict] = []
    joints: list[float] | None = None
    status    = "Ready — jog with keys, SPACE to record, O/C for gripper"
    status_ok = True
    active_joint: int | None = None

    # Fetch initial joint state
    try:
        result = await arm.call_tool("rtde_get_joints", {})
        if result.get("ok"):
            joints = result["joints_rad"]
    except Exception:
        pass

    # Activate gripper so its communication daemon is running on the robot
    if gripper is not None:
        console.print("[dim]  Activating gripper…[/dim]")
        try:
            r = await gripper.call_tool("activate", {})
            if r.get("ok"):
                console.print("[green]  Gripper ready.[/green]")
            else:
                console.print(f"[yellow]  Gripper: {r.get('message', r)}[/yellow]")
        except Exception as e:
            console.print(f"[yellow]  Gripper activation failed: {e}[/yellow]")

    console.print()
    console.print(Panel(
        "[bold]How to teach a scenario:[/bold]\n\n"
        "  1. Use [cyan]Q/A W/S E/D R/F T/G Y/H[/cyan] to jog each joint +/-\n"
        "  2. Press [cyan]+[/cyan] / [cyan]-[/cyan] to increase or decrease the step size\n"
        "  3. Press [cyan]SPACE[/cyan] to record the current arm position as a waypoint\n"
        "  4. Press [cyan]O[/cyan] to open the gripper  (recorded + sent immediately)\n"
        "  5. Press [cyan]C[/cyan] to close the gripper (recorded + sent immediately)\n"
        "  6. Press [cyan]X[/cyan] or [cyan]ESC[/cyan] when done — you will be asked for a scenario name\n\n"
        "[dim]Tip: jog to each key position, record it, then add a gripper action — "
        "the saved YAML will replay exactly what you taught.[/dim]",
        title="[bold cyan] Teach Mode [/bold cyan]",
        border_style="cyan",
    ))
    console.print()
    await asyncio.sleep(0.3)

    with Live(
        _render(joints, step, speed, status, status_ok, recorded_steps, active_joint),
        console=console,
        refresh_per_second=20,
        transient=False,
    ) as live:

        loop = asyncio.get_running_loop()
        key_queue: asyncio.Queue[str] = asyncio.Queue()

        def _poll_keys() -> None:
            while True:
                ch = _get_key()
                if ch:
                    asyncio.run_coroutine_threadsafe(key_queue.put(ch), loop)
                time.sleep(0.03)

        threading.Thread(target=_poll_keys, daemon=True).start()

        running = True
        while running:
            await asyncio.sleep(0.05)

            while not key_queue.empty():
                try:
                    ch = key_queue.get_nowait()
                except asyncio.QueueEmpty:
                    break

                ch_lower = ch.lower()

                # Quit
                if ch in ("\x1b", "x", "X"):
                    running = False
                    break

                # Step adjust
                if ch in ("+", "="):
                    step = min(STEP_MAX, round(step + STEP_INC, 4))
                    status = f"Step → {step:.4f} rad ({math.degrees(step):.2f}°)"
                    status_ok = True
                    active_joint = None
                    continue

                if ch == "-":
                    step = max(STEP_MIN, round(step - STEP_INC, 4))
                    status = f"Step → {step:.4f} rad ({math.degrees(step):.2f}°)"
                    status_ok = True
                    active_joint = None
                    continue

                # Record joint waypoint
                if ch == " ":
                    if joints:
                        n = sum(1 for s in recorded_steps if s["type"] == "move_joints") + 1
                        recorded_steps.append({
                            "type":         "move_joints",
                            "joints":       [round(v, 7) for v in joints],
                            "speed":        speed,
                            "accel":        accel,
                            "blend_radius": 0.0,
                        })
                        status = f"Recorded wp{n}: {[round(v,3) for v in joints]}"
                        status_ok = True
                    else:
                        status = "No joint data — cannot record"
                        status_ok = False
                    active_joint = None
                    continue

                # Gripper open
                if ch_lower == "o":
                    recorded_steps.append({
                        "type":            "tool_action",
                        "tool":            "gripper",
                        "action":          "open",
                        "params":          GRIPPER_OPEN_PARAMS,
                        "arm_recovery_sec": 3.0,
                    })
                    status = "Recorded: gripper open"
                    status_ok = True
                    active_joint = None
                    # Also send immediately so the physical gripper opens
                    if gripper is not None:
                        try:
                            await gripper.call_tool("open", GRIPPER_OPEN_PARAMS)
                        except Exception as e:
                            status = f"Gripper open failed: {e}"
                            status_ok = False
                    continue

                # Gripper close
                if ch_lower == "c":
                    recorded_steps.append({
                        "type":            "tool_action",
                        "tool":            "gripper",
                        "action":          "close",
                        "params":          GRIPPER_CLOSE_PARAMS,
                        "arm_recovery_sec": 3.0,
                    })
                    status = "Recorded: gripper close"
                    status_ok = True
                    active_joint = None
                    if gripper is not None:
                        try:
                            await gripper.call_tool("close", GRIPPER_CLOSE_PARAMS)
                        except Exception as e:
                            status = f"Gripper close failed: {e}"
                            status_ok = False
                    continue

                # Jog key
                if ch_lower in KEY_MAP:
                    joint_idx, direction = KEY_MAP[ch_lower]
                    active_joint = joint_idx
                    try:
                        result = await arm.call_tool("jog_joint", {
                            "joint_index": joint_idx,
                            "direction":   direction,
                            "step":        step,
                            "speed":       speed,
                            "accel":       accel,
                        })
                        if result.get("ok"):
                            joints    = result["joints_rad"]
                            status    = (
                                f"J{joint_idx} {JOINT_NAMES[joint_idx]} "
                                f"{'▲' if direction > 0 else '▼'} "
                                f"→ {result['new_value_rad']:+.5f} rad"
                            )
                            status_ok = True
                        else:
                            status    = result.get("message", "Move failed")
                            status_ok = False
                            active_joint = None
                    except Exception as e:
                        status    = f"MCP error: {e}"
                        status_ok = False
                        active_joint = None

            live.update(_render(joints, step, speed, status, status_ok, recorded_steps, active_joint))

    # ── Post-teach: show summary and offer to save ────────────────────────────
    console.print()
    if not recorded_steps:
        console.print("[dim]  No steps recorded.[/dim]\n")
        return

    console.print(f"[cyan]  {len(recorded_steps)} step(s) recorded:[/cyan]")
    for i, s in enumerate(recorded_steps, 1):
        if s["type"] == "move_joints":
            console.print(f"  [dim]#{i}[/dim] [yellow]waypoint[/yellow]  "
                          f"[dim]{[round(v,4) for v in s['joints']]}[/dim]")
        elif s["type"] == "tool_action":
            action = s["action"]
            color = "green" if action == "open" else "magenta"
            console.print(f"  [dim]#{i}[/dim] [{color}]gripper {action}[/{color}]")

    console.print()
    console.print("[dim]  Save as scenario? Enter a name (or press Enter to discard):[/dim]")
    try:
        name = input("  scenario name > ").strip()
    except (EOFError, KeyboardInterrupt):
        name = ""

    if name:
        path = _save_scenario(name, recorded_steps, session.cfg.scenarios.store_path)
        console.print(f"\n  [green]✓[/green]  Saved [bold]{name}[/bold] → [dim]{path}[/dim]\n")
    else:
        console.print("  [dim]Discarded.[/dim]\n")
