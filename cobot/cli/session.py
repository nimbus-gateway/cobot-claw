from __future__ import annotations

from dataclasses import dataclass, field

from cobot.config import CobotConfig, RobotConfig
from cobot.agents.orchestrator import OrchestratorDeps
from pydantic_ai.messages import ModelMessage


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
    active_agent: str = "orchestrator"
    deps: OrchestratorDeps | None = None
    history: list[dict] = field(default_factory=list)
    message_history: list[ModelMessage] = field(default_factory=list)
    mcp_status: dict[str, bool] = field(default_factory=dict)  # {"arm": True, "gripper": False, ...}

    @property
    def robot_connected(self) -> bool:
        """True if at least the arm MCP connected successfully."""
        return self.mcp_status.get("arm", False)

    # Backward-compat: some internal code still uses .scenario_deps
    @property
    def scenario_deps(self):
        return self.deps.robotic_deps if self.deps else None
