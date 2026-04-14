#!/usr/bin/env python3
"""
UR3e Keyboard Jog Tool
======================
Jog individual joints using keyboard keys.
Uses RTDE (port 30004) for live joint reading and sends incremental
movej() commands to port 30002.  No ROS, no External Control needed.

Key bindings
------------
  Q / A   →  Joint 0  (base)         +/-
  W / S   →  Joint 1  (shoulder)     +/-
  E / D   →  Joint 2  (elbow)        +/-
  R / F   →  Joint 3  (wrist 1)      +/-
  T / G   →  Joint 4  (wrist 2)      +/-
  Y / H   →  Joint 5  (wrist 3)      +/-

  +  /  =  →  increase step size
  -        →  decrease step size
  SPACE    →  snapshot — print YAML-ready joint line
  0        →  go to home position (press twice to confirm)
  ESC / X  →  quit

Usage
-----
  python3 jog.py [--ip ROBOT_IP] [--step 0.02] [--speed 0.3]
"""

from __future__ import annotations

import argparse
import math
import os
import socket
import struct
import sys
import threading
import time


# ──────────────────────────────────────────────────────────────────────────────
# Config
# ──────────────────────────────────────────────────────────────────────────────

DEFAULT_IP  = os.getenv("ROBOT_IP", "192.168.1.2")
RTDE_PORT   = 30004
SCRIPT_PORT = 30002

JOINT_NAMES = [
    "base (J0)",
    "shoulder (J1)",
    "elbow (J2)",
    "wrist_1 (J3)",
    "wrist_2 (J4)",
    "wrist_3 (J5)",
]

JOINT_LIMITS = [
    (-6.0,  6.0),
    (-6.0,  6.0),
    (-3.0,  3.0),
    (-6.0,  6.0),
    (-6.0,  6.0),
    (-6.0,  6.0),
]

SOFT_LIMIT_MARGIN = 0.3   # rad

STEP_MIN = 0.005
STEP_MAX = 0.10
STEP_INC = 0.005

HOME_JOINTS = [-3.16018, -0.6864, -1.4132, -2.0002, 1.63811, 3.17233]

KEY_MAP: dict[str, tuple[int, int]] = {
    "q": (0, +1), "a": (0, -1),
    "w": (1, +1), "s": (1, -1),
    "e": (2, +1), "d": (2, -1),
    "r": (3, +1), "f": (3, -1),
    "t": (4, +1), "g": (4, -1),
    "y": (5, +1), "h": (5, -1),
}


# ──────────────────────────────────────────────────────────────────────────────
# RTDE passive reader
# ──────────────────────────────────────────────────────────────────────────────

_RTDE_PROTOCOL_VERSION              = 2
_RTDE_REQUEST_PROTOCOL_VERSION      = 86
_RTDE_DATA_PACKAGE                  = 85
_RTDE_CONTROL_PACKAGE_SETUP_OUTPUTS = 79
_RTDE_CONTROL_PACKAGE_START         = 83


def _rtde_pack(ptype: int, payload: bytes = b"") -> bytes:
    return struct.pack(">HB", 3 + len(payload), ptype) + payload


def _rtde_recv(sock: socket.socket) -> tuple[int, bytes]:
    header = b""
    while len(header) < 3:
        chunk = sock.recv(3 - len(header))
        if not chunk:
            raise ConnectionError("RTDE closed")
        header += chunk
    size, ptype = struct.unpack(">HB", header)
    payload = b""
    remaining = size - 3
    while len(payload) < remaining:
        chunk = sock.recv(remaining - len(payload))
        if not chunk:
            raise ConnectionError("RTDE closed")
        payload += chunk
    return ptype, payload


class RTDEReader:
    def __init__(self, ip: str) -> None:
        self._ip    = ip
        self._lock  = threading.Lock()
        self._joints: list[float] | None = None
        self._running = False

    def start(self) -> None:
        self._running = True
        threading.Thread(target=self._loop, daemon=True, name="rtde").start()

    def stop(self) -> None:
        self._running = False

    @property
    def joints(self) -> list[float] | None:
        with self._lock:
            return list(self._joints) if self._joints is not None else None

    @property
    def connected(self) -> bool:
        with self._lock:
            return self._joints is not None

    def _loop(self) -> None:
        while self._running:
            try:
                self._run()
            except Exception:
                pass
            if self._running:
                with self._lock:
                    self._joints = None
                time.sleep(2.0)

    def _run(self) -> None:
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(5.0)
        sock.connect((self._ip, RTDE_PORT))
        sock.settimeout(10.0)
        try:
            sock.sendall(_rtde_pack(_RTDE_REQUEST_PROTOCOL_VERSION,
                                    struct.pack(">H", _RTDE_PROTOCOL_VERSION)))
            ptype, payload = _rtde_recv(sock)
            if ptype != _RTDE_REQUEST_PROTOCOL_VERSION or not payload or payload[0] != 1:
                raise RuntimeError("Protocol rejected")

            recipe = b"actual_q"
            sock.sendall(_rtde_pack(_RTDE_CONTROL_PACKAGE_SETUP_OUTPUTS,
                                    struct.pack(">d", 25.0) + recipe))
            ptype, payload = _rtde_recv(sock)
            if ptype != _RTDE_CONTROL_PACKAGE_SETUP_OUTPUTS:
                raise RuntimeError("Setup failed")
            if b"NOT_FOUND" in payload:
                raise RuntimeError("actual_q not found")

            sock.sendall(_rtde_pack(_RTDE_CONTROL_PACKAGE_START))
            ptype, payload = _rtde_recv(sock)
            if ptype != _RTDE_CONTROL_PACKAGE_START or not payload or payload[0] != 1:
                raise RuntimeError("Start failed")

            sock.settimeout(3.0)
            while self._running:
                ptype, payload = _rtde_recv(sock)
                if ptype != _RTDE_DATA_PACKAGE:
                    continue
                if len(payload) < 1 + 6 * 8:
                    continue
                values = struct.unpack_from(">6d", payload, 1)
                with self._lock:
                    self._joints = list(values)
        finally:
            sock.close()


# ──────────────────────────────────────────────────────────────────────────────
# Robot commands
# ──────────────────────────────────────────────────────────────────────────────

def send_movej(ip: str, joints: list[float], speed: float, accel: float) -> None:
    j = joints
    script = (
        f"movej([{j[0]:.6f},{j[1]:.6f},{j[2]:.6f},"
        f"{j[3]:.6f},{j[4]:.6f},{j[5]:.6f}],"
        f"a={accel:.3f},v={speed:.3f})\n"
    )
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.settimeout(3.0)
    sock.connect((ip, SCRIPT_PORT))
    sock.sendall(script.encode())
    sock.close()


def check_limits(joints: list[float]) -> str | None:
    for i, (val, (lo, hi)) in enumerate(zip(joints, JOINT_LIMITS)):
        margin = min(val - lo, hi - val)
        if margin < SOFT_LIMIT_MARGIN:
            if val <= lo + SOFT_LIMIT_MARGIN:
                return f"Joint {i} ({JOINT_NAMES[i]}) too close to lower limit ({lo:.2f} rad)"
            else:
                return f"Joint {i} ({JOINT_NAMES[i]}) too close to upper limit ({hi:.2f} rad)"
    return None


# ──────────────────────────────────────────────────────────────────────────────
# Terminal UI
# ──────────────────────────────────────────────────────────────────────────────

def run_jog(ip: str, step_rad: float, speed: float, accel: float) -> None:
    import curses

    rtde = RTDEReader(ip)
    rtde.start()

    snapshots:  list[list[float]] = []
    step        = step_rad
    status_msg  = "Ready"
    status_err  = False

    def ui(stdscr) -> None:
        nonlocal step, status_msg, status_err

        curses.curs_set(0)
        stdscr.nodelay(True)
        stdscr.timeout(50)

        curses.start_color()
        curses.use_default_colors()
        curses.init_pair(1, curses.COLOR_GREEN,  -1)
        curses.init_pair(2, curses.COLOR_RED,    -1)
        curses.init_pair(3, curses.COLOR_YELLOW, -1)
        curses.init_pair(4, curses.COLOR_CYAN,   -1)
        curses.init_pair(5, curses.COLOR_WHITE,  -1)

        GREEN  = curses.color_pair(1) | curses.A_BOLD
        RED    = curses.color_pair(2) | curses.A_BOLD
        YELLOW = curses.color_pair(3)
        CYAN   = curses.color_pair(4) | curses.A_BOLD
        DIM    = curses.color_pair(5) | curses.A_DIM

        home_pending = False

        while True:
            stdscr.erase()
            h, w = stdscr.getmaxyx()
            row = 0

            title = " UR3e Keyboard Jog "
            stdscr.addstr(row, max(0, (w - len(title)) // 2), title, CYAN)
            row += 1
            stdscr.addstr(row, 0, "─" * min(w - 1, 62), DIM)
            row += 1

            if rtde.connected:
                stdscr.addstr(row, 0, f"  RTDE {ip}:{RTDE_PORT}  ● connected", GREEN)
            else:
                stdscr.addstr(row, 0, f"  RTDE {ip}:{RTDE_PORT}  ○ connecting…", RED)
            row += 2

            joints = rtde.joints
            stdscr.addstr(row, 0, f"  {'Joint':<22}{'Rad':>10}  {'Deg':>9}  Keys", DIM)
            row += 1

            key_pairs = ["Q/A", "W/S", "E/D", "R/F", "T/G", "Y/H"]
            for i, name in enumerate(JOINT_NAMES):
                val_str = f"{joints[i]:+.5f}" if joints else "  ?.??????"
                deg_str = f"{math.degrees(joints[i]):+.2f}°" if joints else "  ?.??°"
                lo, hi  = JOINT_LIMITS[i]
                margin  = min(joints[i] - lo, hi - joints[i]) if joints else 999
                color   = RED if margin < SOFT_LIMIT_MARGIN else (YELLOW if margin < 0.8 else curses.color_pair(5))
                stdscr.addstr(row, 0,
                    f"  J{i}  {name:<18}{val_str:>10}  {deg_str:>9}  {key_pairs[i]}", color)
                row += 1

            row += 1
            stdscr.addstr(row, 0,
                f"  Step: {step:.4f} rad ({math.degrees(step):.2f}°)   "
                f"Speed: {speed:.2f} rad/s   [+/-] step", DIM)
            row += 1

            color = RED if status_err else (YELLOW if home_pending else GREEN)
            stdscr.addstr(row, 0, f"  {status_msg}", color)
            row += 2

            if snapshots:
                stdscr.addstr(row, 0, "  Snapshots:", CYAN)
                row += 1
                for s in snapshots[-4:]:
                    line = f"  joints: [{', '.join(f'{v:.7f}' for v in s)}]"
                    stdscr.addstr(row, 0, line[:w-1], YELLOW)
                    row += 1
                row += 1

            stdscr.addstr(min(row, h-2), 0,
                "  SPACE=snapshot  0=home(x2)  ESC/X=quit", DIM)
            stdscr.refresh()

            try:
                key = stdscr.getkey()
            except curses.error:
                continue

            key_lower = key.lower() if len(key) == 1 else key

            if key in ("\x1b", "x"):
                break

            if not rtde.connected:
                status_msg = "Not connected"
                status_err = True
                continue

            joints = rtde.joints
            if joints is None:
                status_msg = "No joint data from RTDE"
                status_err = True
                continue

            if key in ("+", "="):
                step = min(STEP_MAX, round(step + STEP_INC, 4))
                status_msg = f"Step → {step:.4f} rad  ({math.degrees(step):.2f}°)"
                status_err = False
                home_pending = False
                continue

            if key == "-":
                step = max(STEP_MIN, round(step - STEP_INC, 4))
                status_msg = f"Step → {step:.4f} rad  ({math.degrees(step):.2f}°)"
                status_err = False
                home_pending = False
                continue

            if key == " ":
                snapshots.append(list(joints))
                status_msg = f"Snapshot #{len(snapshots)} saved"
                status_err = False
                home_pending = False
                print(f"\n# Snapshot {len(snapshots)}")
                print(f"joints: [{', '.join(f'{v:.7f}' for v in joints)}]")
                continue

            if key == "0":
                if home_pending:
                    try:
                        send_movej(ip, HOME_JOINTS, speed=0.5, accel=0.8)
                        status_msg = "Moving to home…"
                        status_err = False
                    except Exception as e:
                        status_msg = f"Home failed: {e}"
                        status_err = True
                    home_pending = False
                else:
                    status_msg = "Press 0 again to confirm home move"
                    status_err = False
                    home_pending = True
                continue

            home_pending = False

            if key_lower in KEY_MAP:
                joint_idx, direction = KEY_MAP[key_lower]
                target = list(joints)
                target[joint_idx] += direction * step

                err = check_limits(target)
                if err:
                    status_msg = f"BLOCKED: {err}"
                    status_err = True
                    continue

                try:
                    send_movej(ip, target, speed=speed, accel=accel)
                    status_msg = (
                        f"J{joint_idx} {'▲' if direction > 0 else '▼'} "
                        f"{step:.4f} rad → {target[joint_idx]:+.5f}"
                    )
                    status_err = False
                except Exception as e:
                    status_msg = f"Send error: {e}"
                    status_err = True

    try:
        curses.wrapper(ui)
    finally:
        rtde.stop()

    if snapshots:
        print("\n" + "═" * 60)
        print("  All snapshots (YAML-ready):")
        print("═" * 60)
        for i, s in enumerate(snapshots, 1):
            print(f"\n  # Snapshot {i}")
            print(f"  joints: [{', '.join(f'{v:.7f}' for v in s)}]")
        print()


# ──────────────────────────────────────────────────────────────────────────────
# Entry point
# ──────────────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="UR3e keyboard jog tool")
    parser.add_argument("--ip",    default=DEFAULT_IP,  help="Robot IP (default: %(default)s)")
    parser.add_argument("--step",  default=0.02, type=float, help="Step size in rad (default: 0.02)")
    parser.add_argument("--speed", default=0.30, type=float, help="movej speed in rad/s (default: 0.30)")
    parser.add_argument("--accel", default=0.50, type=float, help="movej accel in rad/s² (default: 0.50)")
    args = parser.parse_args()

    if args.step < STEP_MIN or args.step > STEP_MAX:
        print(f"--step must be between {STEP_MIN} and {STEP_MAX} rad")
        sys.exit(1)

    print(f"Connecting to {args.ip}…  (Ctrl-C to abort)")
    run_jog(ip=args.ip, step_rad=args.step, speed=args.speed, accel=args.accel)


if __name__ == "__main__":
    main()
