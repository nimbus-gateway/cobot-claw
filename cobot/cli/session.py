from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from cobot.config import CobotConfig, RobotConfig
from cobot.tasklog import TaskLog
from cobot.agents.orchestrator import OrchestratorDeps
from pydantic_ai.messages import ModelMessage

if TYPE_CHECKING:
    from cobot.camera.stream import CameraStream


@dataclass
class CobotSession:
    """
    One CLI session = one active robot.

    Fleet config allows switching the active robot between sessions (or using
    ``cobot --robot <name>``). Simultaneous multi-robot orchestration is a future
    capability — for now the executor and agents always target active_robot.
    """

    cfg: CobotConfig
    active_robot: RobotConfig
    config_path: str = "cobot.yaml"
    active_agent: str = "orchestrator"
    session_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    camera_stream: "CameraStream | None" = None  # set by /vision-debug; shared with monitor agent
    deps: OrchestratorDeps | None = None
    history: list[dict] = field(default_factory=list)
    message_history: list[ModelMessage] = field(default_factory=list)
    mcp_status: dict[str, bool] = field(default_factory=dict)  # {"arm": True, "gripper": False, ...}
    voice: object | None = None  # cobot.cli.voice.VoiceInput when voice.enabled
    task_log: TaskLog = field(default_factory=TaskLog)   # what the robot is doing / has done
    current_task: object | None = None   # asyncio.Task of the request being worked on
    task_status: str = ""                # short status for the bottom toolbar

    @property
    def robot_connected(self) -> bool:
        """True if at least the arm MCP connected successfully."""
        return self.mcp_status.get("arm", False)

    # Backward-compat: some internal code still uses .scenario_deps
    @property
    def scenario_deps(self):
        return self.deps.robotic_deps if self.deps else None
