from __future__ import annotations

from dataclasses import dataclass

from cobot.config import CobotConfig
from cobot.mcp.client import MCPClient


@dataclass
class ArtisticAgentDeps:
    arm_mcp: MCPClient
    tool_mcps: dict[str, MCPClient]     # whatever end-effectors are connected
    vision_deps: object | None          # VisionAgentDeps — typed as object to avoid circular import
    cfg: CobotConfig
