"""
/jog — in-REPL keyboard jog command.

Drops into jog mode directly inside the CLI.  Each keypress calls
jog_joint() on the MCP server — no LLM, no agent overhead, ~5ms latency.

Waypoints can be recorded with SPACE and saved as a scenario on exit.

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
  SPACE   →  record waypoint
  X / ESC →  exit jog mode
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
STEP_MAX = 0.10
STEP_INC = 0.005
DEFAULT_STEP  = 0.02
DEFAULT_SPEED = 0.30
DEFAULT_ACCEL = 0.50


# ──────────────────────────────────────────────────────────────────────────────
# Raw key reader (cross-platform, non-blocking)
# ──────────────────────────────────────────────────────────────────────────────

def _read_key_win() -> str | None:
    import msvcrt
    if msvcrt.kbhit():
        ch = msvcrt.getwch()
        if ch in ("\x00", "\xe0"):   # special key prefix — discard next
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

def _render(
    joints: list[float] | None,
    step: float,
    speed: float,
    status: str,
    status_ok: bool,
    waypoints: list[dict],
    active_joint: int | None,
) -> Group:

    # Joint table
    tbl = Table.grid(padding=(0, 2))
    tbl.add_column(width=3)   # index
    tbl.add_column(width=10)  # name
    tbl.add_column(width=10, justify="right")  # rad
    tbl.add_column(width=9,  justify="right")  # deg
    tbl.add_column(width=5)   # keys

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

    # Status line
    status_color = "green" if status_ok else "red"
    status_line  = Text(f" {status}", style=status_color)

    # Step / speed line
    info_line = Text(
        f" step: {step:.4f} rad ({math.degrees(step):.2f}°)   "
        f"speed: {speed:.2f} rad/s   [+/-] adjust",
        style="dim",
    )

    # Waypoints
    wp_lines: list[Text] = []
    if waypoints:
        wp_lines.append(Text(f" {len(waypoints)} waypoint(s) recorded", style="cyan"))
        for i, wp in enumerate(waypoints[-3:], max(1, len(waypoints) - 2)):
            j = wp["joints"]
            short = f"[{', '.join(f'{v:.4f}' for v in j)}]"
            wp_lines.append(Text(f"  #{i} {wp.get('label','wp')}  {short}", style="yellow"))

    help_line = Text(
        " SPACE=record waypoint   X/ESC=exit jog",
        style="dim",
    )

    panel = Panel(
        Group(tbl, info_line, status_line, *wp_lines, help_line),
        title="[bold cyan] Jog Mode [/bold cyan]",
        border_style="cyan",
    )
    return Group(panel)


# ──────────────────────────────────────────────────────────────────────────────
# Waypoint → YAML scenario
# ──────────────────────────────────────────────────────────────────────────────

def _save_scenario(name: str, waypoints: list[dict], store_path: str) -> Path:
    steps = []
    for wp in waypoints:
        step: dict = {
            "type": "move_joints",
            "joints": [round(v, 7) for v in wp["joints"]],
            "speed":  wp.get("speed",  DEFAULT_SPEED),
            "accel":  wp.get("accel",  DEFAULT_ACCEL),
            "blend_radius": 0.0,
        }
        steps.append(step)
        if wp.get("wait_sec", 0) > 0:
            steps.append({"type": "wait", "duration_sec": wp["wait_sec"]})

    scenario = {
        "name": name,
        "description": f"Recorded via /jog ({len(waypoints)} waypoints)",
        "robot": "ur3e",
        "steps": steps,
    }

    path = Path(store_path) / f"{name}.yaml"
    path.write_text(yaml.dump(scenario, sort_keys=False, allow_unicode=True))
    return path


# ──────────────────────────────────────────────────────────────────────────────
# Main jog command
# ──────────────────────────────────────────────────────────────────────────────

async def cmd_jog(
    session: "CobotSession",
    step: float = DEFAULT_STEP,
    speed: float = DEFAULT_SPEED,
    accel: float = DEFAULT_ACCEL,
) -> None:
    if session.deps is None:
        console.print("[red]No robot connection.[/red]")
        return

    arm = session.deps.robotic_deps.arm_mcp
    waypoints: list[dict] = []
    joints: list[float] | None = None
    status    = "Ready — move the arm with keys"
    status_ok = True
    active_joint: int | None = None

    # Fetch initial joint state
    try:
        result = await arm.call_tool("rtde_get_joints", {})
        if result.get("ok"):
            joints = result["joints_rad"]
    except Exception:
        pass

    console.print("\n[dim]  Press X or ESC to exit jog mode[/dim]")
    # Small pause so the user sees the message before Live takes over
    await asyncio.sleep(0.3)

    with Live(
        _render(joints, step, speed, status, status_ok, waypoints, active_joint),
        console=console,
        refresh_per_second=20,
        transient=False,
    ) as live:

        loop = asyncio.get_running_loop()

        # Key poller runs in background thread, puts keys on a queue
        key_queue: asyncio.Queue[str] = asyncio.Queue()

        def _poll_keys() -> None:
            while True:
                ch = _get_key()
                if ch:
                    asyncio.run_coroutine_threadsafe(key_queue.put(ch), loop)
                time.sleep(0.03)   # ~33Hz poll

        key_thread = threading.Thread(target=_poll_keys, daemon=True)
        key_thread.start()

        running = True
        while running:
            # Refresh display every ~50ms even without input
            await asyncio.sleep(0.05)

            # Drain all pending keys
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

                # Record waypoint
                if ch == " ":
                    if joints:
                        label = f"wp{len(waypoints) + 1}"
                        waypoints.append({
                            "label":  label,
                            "joints": list(joints),
                            "speed":  speed,
                            "accel":  accel,
                            "wait_sec": 0.0,
                        })
                        status = f"Recorded {label}: {[round(v,3) for v in joints]}"
                        status_ok = True
                    else:
                        status = "No joint data — cannot record"
                        status_ok = False
                    active_joint = None
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

            live.update(_render(joints, step, speed, status, status_ok, waypoints, active_joint))

    # ── Post-jog: offer to save waypoints as scenario ─────────────────────────
    console.print()
    if not waypoints:
        console.print("[dim]  No waypoints recorded.[/dim]\n")
        return

    console.print(f"[cyan]  {len(waypoints)} waypoint(s) recorded.[/cyan]")
    for i, wp in enumerate(waypoints, 1):
        console.print(f"  [dim]#{i}[/dim] [yellow]{wp['label']}[/yellow]  "
                      f"[dim]{[round(v,4) for v in wp['joints']]}[/dim]")

    console.print()
    console.print("[dim]  Save as scenario? Enter a name (or press Enter to discard):[/dim]")
    try:
        name = input("  scenario name > ").strip()
    except (EOFError, KeyboardInterrupt):
        name = ""

    if name:
        path = _save_scenario(name, waypoints, session.cfg.scenarios.store_path)
        console.print(f"\n  [green]✓[/green]  Saved [bold]{name}[/bold] → [dim]{path}[/dim]\n")
    else:
        console.print("  [dim]Discarded.[/dim]\n")
