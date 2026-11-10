"""What the robot is doing and has done this session.

Tools append events while a task runs (inspected part 2: FAIL, picked it,
placed it in the reject bin, ...). The REPL prints each event as it happens,
the observer agent answers questions from the log while a task is still
running, and the orchestrator reads it afterwards to answer follow-ups
("why was part 2 rejected?").
"""

from __future__ import annotations

import collections
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Callable

# Event kinds:
#   task     a request started or ended
#   step     an arm or gripper step (picked X, placed X in Y)
#   result   an outcome worth reporting (inspection result, order packed)
#   problem  something failed
#   qa       a question asked during a task, and its answer


@dataclass
class TaskEvent:
    t: float
    kind: str
    text: str
    data: dict[str, Any] = field(default_factory=dict)

    def as_dict(self, start: float | None = None) -> dict[str, Any]:
        d = {"time": time.strftime("%H:%M:%S", time.localtime(self.t)), "kind": self.kind, "text": self.text}
        if start is not None:
            d["seconds_into_task"] = round(self.t - start, 1)
        if self.data:
            d["data"] = self.data
        return d


class TaskLog:
    def __init__(self, maxlen: int = 1000) -> None:
        self.events: collections.deque[TaskEvent] = collections.deque(maxlen=maxlen)
        self.task: str | None = None          # request being worked on, None when idle
        self.started: float | None = None
        self.last_task: str | None = None
        self.listeners: list[Callable[[TaskEvent], None]] = []
        self._lock = threading.Lock()

    @property
    def running(self) -> bool:
        return self.task is not None

    def add(self, kind: str, text: str, **data: Any) -> TaskEvent:
        ev = TaskEvent(time.time(), kind, text, data)
        with self._lock:
            self.events.append(ev)
            listeners = list(self.listeners)
        for fn in listeners:
            try:
                fn(ev)
            except Exception:  # noqa: BLE001 — a display problem must not break the task
                pass
        return ev

    def begin(self, request: str) -> None:
        self.task, self.started = request, time.time()
        self.add("task", f"started: {request}")

    def end(self, outcome: str) -> None:
        if self.task is None:
            return
        took = time.time() - (self.started or time.time())
        self.add("task", f"{outcome} after {took:.0f} s: {self.task}")
        self.last_task, self.task, self.started = self.task, None, None

    def snapshot(self, last_n: int = 60) -> dict[str, Any]:
        """Recent events plus what is running now, for an agent to read."""
        with self._lock:
            events = list(self.events)[-last_n:]
        return {
            "running_task": self.task,
            "running_for_s": round(time.time() - self.started, 1) if self.started else None,
            "previous_task": self.last_task,
            "events": [e.as_dict(self.started) for e in events],
        }


def log_event(deps: Any, kind: str, text: str, **data: Any) -> None:
    """Add an event if `deps` carries a task log (tools work without one)."""
    log = getattr(deps, "task_log", None)
    if log is not None:
        log.add(kind, text, **data)
