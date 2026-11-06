#!/usr/bin/env python3
"""
Robotiq gripper emulator for URSim.

On a real UR controller, the Robotiq URCap runs a driver that listens on
127.0.0.1:63352. Gripper.script talks to that port (SET ACT 1, GET STA, ...).
URSim has no gripper hardware, so this script stands in for the driver: it
answers the same protocol and simulates a Hand-E opening and closing.

Run it *inside* the URSim container (scripts/start_ursim_demo.sh does this):

    docker cp tools/gripper/robotiq_emulator.py ursim:/tmp/
    docker exec -d ursim python3 /tmp/robotiq_emulator.py

Options (environment variables):
    EMU_PORT        port to listen on (default 63352)
    EMU_OBJECT_POS  if set (0-255), closing stops there and reports an object
                    detected, as if a part were in the jaws. The scene server
                    (tools/sim_scene) overrides this at runtime with SET SOB.
    EMU_STATUS_FILE file the current state is written to as JSON after every
                    change (default /tmp/robotiq_emulator.json)

Protocol, as used by Gripper.script:
    SET <VAR> <int> [<VAR> <int> ...]\n   -> "ack"
    GET POS\n  (socket_get_var)            -> "POS <int>\n"
    GET STA | GET OBJ | GET PRE | GET FLT  -> bare zero-padded digits
                                              (1, 1, 3 and 2 chars)
    GET VER | FWV | SNU | PCO | NCY | DST  -> "<VAR> <text>\n"
"""
from __future__ import annotations

import json
import os
import re
import socket
import threading
import time

PORT = int(os.getenv("EMU_PORT", "63352"))
OBJECT_POS = os.getenv("EMU_OBJECT_POS")
OBJECT_POS = int(OBJECT_POS) if OBJECT_POS not in (None, "") else None
STATUS_FILE = os.getenv("EMU_STATUS_FILE", "/tmp/robotiq_emulator.json")

FULL_STROKE_SEC_AT_MAX_SPEED = 0.5   # Hand-E: ~150 mm/s over 50 mm
FULL_STROKE_SEC_AT_MIN_SPEED = 2.5
ACTIVATION_SEC = 0.5

TEXT_REPLIES = {
    "VER": "DCU-2.0.0",
    "FWV": "GD1-1.4.6",
    "SNU": "EMULATOR",
    "PCO": "0",
    "NCY": "0",
    "DST": "0",
}

# Tokenise the incoming byte stream. SET commands end with a newline
# (socket_set_var); GET commands may not (socket_send_string), so a GET is
# complete as soon as its three-letter variable name has arrived.
CMD_RE = re.compile(rb"SET((?:\s+[A-Z]{3}\s+-?\d+)+)\s*\n|GET\s+([A-Z]{3})\n?")


class Gripper:
    def __init__(self) -> None:
        self.lock = threading.Lock()
        self.act = 0
        self.gto = 0
        self.atr = 0
        self.ard = 0
        self.spe = 255
        self.for_ = 150
        self.pre = 0          # position request (echo)
        self.pos = 0          # current position
        self.sid = 9
        self.act_since: float | None = None
        # t0, start, target, duration, stopped_on_object
        self.motion: tuple[float, int, int, float, bool] | None = None
        self.obj = 0
        self.sim_obj_pos: int | None = None   # set at runtime with SET SOB (scene server)
        self._write_status()

    # ── state ────────────────────────────────────────────────────────────────
    def _activated(self) -> bool:
        return (
            self.act == 1
            and self.act_since is not None
            and time.monotonic() - self.act_since >= ACTIVATION_SEC
        )

    def _update(self) -> None:
        """Advance the simulated motion."""
        if not self.motion:
            return
        t0, start, target, duration, on_object = self.motion
        frac = 1.0 if duration <= 0 else min(1.0, (time.monotonic() - t0) / duration)
        self.pos = round(start + (target - start) * frac)
        if frac >= 1.0:
            self.pos = target
            self.motion = None
            self.obj = 2 if on_object else 3
            self._write_status()

    def _start_motion(self) -> None:
        if not self._activated():
            return
        target = self.pre
        obj_pos = self.sim_obj_pos if self.sim_obj_pos is not None else OBJECT_POS
        on_object = obj_pos is not None and target > obj_pos >= self.pos
        if on_object:
            target = obj_pos               # jaws stop on the part
        speed_frac = self.spe / 255
        full = FULL_STROKE_SEC_AT_MIN_SPEED - (
            FULL_STROKE_SEC_AT_MIN_SPEED - FULL_STROKE_SEC_AT_MAX_SPEED
        ) * speed_frac
        duration = full * abs(target - self.pos) / 255
        self.obj = 0
        self.motion = (time.monotonic(), self.pos, target, duration, on_object)
        log(f"moving {self.pos} -> {target} ({duration:.2f}s)")
        self._update()

    def _write_status(self) -> None:
        try:
            with open(STATUS_FILE, "w") as f:
                json.dump(self.snapshot(), f)
        except OSError:
            pass

    def snapshot(self) -> dict:
        return {
            "activated": self._activated(),
            "position": self.pos,
            "position_pct": round(self.pos * 100 / 255),
            "requested": self.pre,
            "object_detected": self.obj in (1, 2),
            "moving": self.motion is not None,
            "speed": self.spe,
            "force": self.for_,
        }

    # ── protocol ─────────────────────────────────────────────────────────────
    def set_vars(self, pairs: list[tuple[str, int]]) -> None:
        with self.lock:
            self._update()
            for name, value in pairs:
                if name == "ACT":
                    if value == 1 and self.act == 0:
                        self.act_since = time.monotonic()
                        log("activating")
                    elif value == 0:
                        self.act_since = None
                        self.motion = None
                        self.obj = 0
                    self.act = value
                elif name == "GTO":
                    self.gto = value
                    if value == 0:
                        self.motion = None
                elif name == "POS":
                    self.pre = max(0, min(255, value))
                elif name == "SPE":
                    self.spe = max(0, min(255, value))
                elif name == "FOR":
                    self.for_ = max(0, min(255, value))
                elif name == "ATR":
                    self.atr = value
                elif name == "ARD":
                    self.ard = value
                elif name == "SID":
                    self.sid = value
                elif name == "SOB":
                    # Not a Robotiq register: the scene server sets where the
                    # jaws would meet an object (-1 = nothing between them).
                    self.sim_obj_pos = value if value >= 0 else None
            if any(n in ("GTO", "POS") for n, _ in pairs) and self.gto == 1:
                self._start_motion()
            self._write_status()

    def get_var(self, name: str) -> bytes:
        with self.lock:
            self._update()
            if name == "STA":
                if self.act == 0:
                    sta = 0
                elif self._activated():
                    sta = 3
                else:
                    sta = 1
                return str(sta).encode()
            if name == "OBJ":
                return str(self.obj if self.gto == 1 else 0).encode()
            if name == "PRE":
                return f"{self.pre:03d}".encode()
            if name == "FLT":
                return b"00"
            if name in ("POS", "ACT", "GTO", "SPE", "FOR"):
                value = {
                    "POS": self.pos, "ACT": self.act, "GTO": self.gto,
                    "SPE": self.spe, "FOR": self.for_,
                }[name]
                return f"{name} {value}\n".encode()
            if name in TEXT_REPLIES:
                return f"{name} {TEXT_REPLIES[name]}\n".encode()
            return f"{name} 0\n".encode()


GRIPPER = Gripper()


def log(msg: str) -> None:
    print(f"[robotiq-emulator] {msg}", flush=True)


def handle(conn: socket.socket, addr) -> None:
    buf = b""
    with conn:
        while True:
            try:
                data = conn.recv(4096)
            except OSError:
                return
            if not data:
                return
            buf += data
            while True:
                m = CMD_RE.search(buf)
                if not m:
                    # keep the buffer bounded if junk arrives
                    if len(buf) > 4096:
                        buf = b""
                    break
                buf = buf[m.end():]
                if m.group(1) is not None:
                    tokens = m.group(1).split()
                    pairs = [
                        (tokens[i].decode(), int(tokens[i + 1]))
                        for i in range(0, len(tokens) - 1, 2)
                    ]
                    GRIPPER.set_vars(pairs)
                    reply = b"ack"
                else:
                    reply = GRIPPER.get_var(m.group(2).decode())
                try:
                    conn.sendall(reply)
                except OSError:
                    return


def main() -> None:
    srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    srv.bind(("0.0.0.0", PORT))
    srv.listen(8)
    log(f"listening on :{PORT} (object at {OBJECT_POS if OBJECT_POS is not None else 'none'})")
    while True:
        conn, addr = srv.accept()
        threading.Thread(target=handle, args=(conn, addr), daemon=True).start()


if __name__ == "__main__":
    main()
