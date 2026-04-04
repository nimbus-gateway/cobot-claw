from __future__ import annotations

from pathlib import Path

from prompt_toolkit.completion import Completer, Completion

from cobot.cli.session import CobotSession

SLASH_COMMANDS = [
    "/help", "/status", "/poses", "/scenarios",
    "/run", "/teach", "/agent", "/model",
    "/fleet", "/config", "/history", "/clear", "/exit",
]

AGENT_NAMES = ["orchestrator", "scenario", "config", "safety"]


class CobotCompleter(Completer):
    def __init__(self, session: CobotSession) -> None:
        self.session = session

    def get_completions(self, document, complete_event):
        text = document.text_before_cursor

        if text.startswith("/run "):
            prefix = text[5:]
            store = Path(self.session.cfg.scenarios.store_path)
            for f in store.glob("*.yaml"):
                if f.stem.startswith(prefix):
                    yield Completion(f.stem, start_position=-len(prefix))

        elif text.startswith("/agent "):
            prefix = text[7:]
            for name in AGENT_NAMES:
                if name.startswith(prefix):
                    yield Completion(name, start_position=-len(prefix))

        elif text.startswith("/model "):
            # /model <agent> <model_name>
            parts = text[7:].split()
            if len(parts) <= 1:
                prefix = parts[0] if parts else ""
                for name in AGENT_NAMES:
                    if name.startswith(prefix):
                        yield Completion(name, start_position=-len(prefix))

        elif text.startswith("/"):
            for cmd in SLASH_COMMANDS:
                if cmd.startswith(text):
                    yield Completion(cmd, start_position=-len(text))
