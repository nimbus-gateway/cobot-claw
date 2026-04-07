"""
Orchestrator Agent

Top-level agent. All CLI input goes here first.
Delegates to robotic_agent for all robot movements, status queries, and tool
actions. Delegates to config_agent for manual Q&A and teach mode.

The orchestrator never calls MCP tools directly — tool discovery and execution
is the sole responsibility of the Robotic Agent, which adapts to whatever arm
model and end-effectors are connected.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from pydantic_ai import Agent, RunContext

from cobot.config import CobotConfig
from cobot.llm import make_model
from cobot.agents.deps import RoboticAgentDeps, ConfigAgentDeps


@dataclass
class OrchestratorDeps:
    robotic_deps: RoboticAgentDeps
    cfg: CobotConfig
    config_deps: ConfigAgentDeps | None = None

    # Backward-compat: allow old code that passes scenario_deps= keyword
    @classmethod
    def _from_legacy(cls, scenario_deps: RoboticAgentDeps, cfg: CobotConfig, config_deps=None):
        return cls(robotic_deps=scenario_deps, cfg=cfg, config_deps=config_deps)


def build_orchestrator(cfg: CobotConfig) -> Agent:
    from cobot.agents.robotic_agent import build_robotic_agent
    from cobot.agents.config_agent import build_config_agent

    _robotic_agent = build_robotic_agent(cfg)
    _config_agent = build_config_agent(cfg)

    agent: Agent[OrchestratorDeps, str] = Agent(
        make_model(cfg.agents.orchestrator.model, cfg),
        deps_type=OrchestratorDeps,
        output_type=str,
        system_prompt=(
            "You orchestrate a UR robot system. "
            "For ALL robot interactions — status queries, movements, tool control, "
            "scenario execution — delegate to run_robotic_task. "
            "For questions about robot specs, safety, error codes, payload limits, "
            "configuration, manuals, or teach-by-demo — delegate to run_config_task. "
            "Never call robot hardware directly."
        ),
    )

    @agent.tool
    async def run_robotic_task(
        ctx: RunContext[OrchestratorDeps],
        task_description: str,
    ) -> str:
        """Delegate any robot task to the Robotic Agent — status, movements, tool actions, scenarios."""
        result = await _robotic_agent.run(task_description, deps=ctx.deps.robotic_deps)
        return str(result.output)

    @agent.tool
    async def run_config_task(
        ctx: RunContext[OrchestratorDeps],
        task_description: str,
    ) -> str:
        """Delegate to the Config Agent for manual Q&A, configuration, setup, or teach-by-demo."""
        if ctx.deps.config_deps is None:
            return "Config Agent not yet available. Install [rag] extras and configure Phase 6."
        result = await _config_agent.run(task_description, deps=ctx.deps.config_deps)
        return result.output

    return agent
