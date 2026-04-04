"""
Orchestrator Agent

Top-level agent. All CLI input goes here first.
Delegates to scenario_agent or config_agent as appropriate.
Holds the fleet context (which robots and tools are connected).
"""

from __future__ import annotations

from dataclasses import dataclass, field

from pydantic_ai import Agent, RunContext

from cobot.config import CobotConfig
from cobot.llm import make_model
from cobot.agents.deps import ScenarioAgentDeps, ConfigAgentDeps


@dataclass
class OrchestratorDeps:
    scenario_deps: ScenarioAgentDeps
    cfg: CobotConfig
    config_deps: ConfigAgentDeps | None = None  # None until Phase 6 (Config Agent + RAG)


def build_orchestrator(cfg: CobotConfig) -> Agent:
    # Build sub-agents lazily to avoid circular imports
    from cobot.agents.scenario import build_scenario_agent
    from cobot.agents.config_agent import build_config_agent

    _scenario_agent = build_scenario_agent(cfg)
    _config_agent = build_config_agent(cfg)

    agent: Agent[OrchestratorDeps, str] = Agent(
        make_model(cfg.agents.orchestrator.model, cfg),
        deps_type=OrchestratorDeps,
        output_type=str,
        system_prompt=(
            "You orchestrate a UR robot system. "
            "For robot movements and scenario execution, delegate to run_scenario_task. "
            "For configuration, setup, manuals, and teach-by-demo, delegate to run_config_task. "
            "For status queries, call get_robot_status directly. "
            "Always confirm dangerous operations before executing."
        ),
    )

    @agent.tool
    async def get_robot_status(ctx: RunContext[OrchestratorDeps]) -> dict:
        """Get arm controller status (joint state, motion, ROS readiness)."""
        return await ctx.deps.scenario_deps.arm_mcp.call_tool("get_robot_status", {})

    @agent.tool
    async def get_tool_mcp_status(ctx: RunContext[OrchestratorDeps]) -> dict:
        """
        Check connectivity of each attached tool MCP (gripper, vacuum, etc.).
        Returns a dict mapping tool name → {ok, message}.
        Use this when the user asks about a specific tool (gripper, vacuum, etc.).
        """
        results: dict = {}
        for name, mcp in ctx.deps.scenario_deps.tool_mcps.items():
            try:
                tools = await mcp.list_tools()
                results[name] = {"ok": True, "available_actions": [t.name for t in tools]}
            except Exception as e:
                results[name] = {"ok": False, "message": str(e)}
        if not results:
            return {"ok": False, "message": "No tool MCPs configured."}
        return results

    @agent.tool
    async def run_scenario_task(
        ctx: RunContext[OrchestratorDeps],
        task_description: str,
    ) -> str:
        """Delegate a movement or scenario execution task to the Scenario Agent."""
        result = await _scenario_agent.run(task_description, deps=ctx.deps.scenario_deps)
        return str(result.data)

    @agent.tool
    async def run_config_task(
        ctx: RunContext[OrchestratorDeps],
        task_description: str,
    ) -> str:
        """Delegate a configuration, manual query, or teach task to the Config Agent (Phase 6+)."""
        if ctx.deps.config_deps is None:
            return "Config Agent not yet available. Install [rag] extras and configure Phase 6."
        result = await _config_agent.run(task_description, deps=ctx.deps.config_deps)
        return result.data

    return agent
