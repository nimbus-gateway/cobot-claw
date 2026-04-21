from __future__ import annotations

from dataclasses import dataclass

from cobot.config import CobotConfig
from cobot.mcp.client import MCPClient
from cobot.mcp.perception_contract import PerceptionClient
from cobot.scenarios.executor import ScenarioExecutor
from cobot.agents.vision_agent.agent import VisionAgentDeps  # noqa: F401 — re-export


@dataclass
class RoboticAgentDeps:
    arm_mcp: MCPClient
    tool_mcps: dict[str, MCPClient]
    executor: ScenarioExecutor
    perception: PerceptionClient
    cfg: CobotConfig


# Backward-compat alias
ScenarioAgentDeps = RoboticAgentDeps


@dataclass
class ConfigAgentDeps:
    arm_mcp: MCPClient
    rag_pipeline: "RAGPipeline"  # noqa: F821 — imported at runtime to avoid circular dep
    scenarios_store_path: str
    cfg: CobotConfig
