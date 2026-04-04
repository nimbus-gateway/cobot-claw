"""
Scenario Agent

Handles natural-language task requests that map to scenario execution.
Uses the deterministic ScenarioExecutor for predefined scenarios.
Falls back to direct MCP tool calls for ad-hoc joint/pose commands.

End-effector tools are discovered dynamically from tool_mcps at runtime —
no code changes are needed when the robot's attached tools change.
"""

from __future__ import annotations

from pathlib import Path

from pydantic import BaseModel
from pydantic_ai import Agent, RunContext

from cobot.agents.deps import ScenarioAgentDeps
from cobot.config import CobotConfig
from cobot.llm import make_model


class ScenarioResult(BaseModel):
    success: bool
    scenario_name: str | None
    steps_completed: int
    error: str | None
    summary: str


def build_scenario_agent(cfg: CobotConfig) -> Agent:
    agent: Agent[ScenarioAgentDeps, ScenarioResult] = Agent(
        make_model(cfg.agents.scenario.model, cfg),
        deps_type=ScenarioAgentDeps,
        output_type=ScenarioResult,
        system_prompt=(
            "You control a UR robot arm. "
            "For named scenarios use run_scenario. "
            "For direct moves use the arm tools. "
            "For end-effectors, call list_tool_mcps first to discover what is attached, "
            "then use call_tool_action to operate it. "
            "Always check robot status before moving. "
            "Return a ScenarioResult when done."
        ),
    )

    @agent.tool
    async def get_robot_status(ctx: RunContext[ScenarioAgentDeps]) -> dict:
        """Get current robot state, joint positions, and motion status."""
        return await ctx.deps.arm_mcp.call_tool("get_robot_status", {})

    @agent.tool
    async def list_scenarios(ctx: RunContext[ScenarioAgentDeps]) -> list[str]:
        """List available scenario names that can be run."""
        store = Path(ctx.deps.cfg.scenarios.store_path)
        return [f.stem for f in store.glob("*.yaml")]

    @agent.tool
    async def run_scenario(
        ctx: RunContext[ScenarioAgentDeps],
        name: str,
        dry_run: bool = False,
    ) -> dict:
        """Execute a named scenario. Use dry_run=True to preview steps without moving."""
        store = Path(ctx.deps.cfg.scenarios.store_path)
        path = store / f"{name}.yaml"
        if not path.exists():
            return {"ok": False, "message": f"Scenario '{name}' not found"}
        scenario = ctx.deps.executor.load(path)
        result = await ctx.deps.executor.run(scenario, dry_run=dry_run)
        return {
            "ok": result.success,
            "steps_completed": result.steps_completed,
            "total_steps": result.total_steps,
            "error": result.error,
        }

    @agent.tool
    async def move_to_pose(
        ctx: RunContext[ScenarioAgentDeps],
        pose_name: str,
        duration_sec: float = 5.0,
    ) -> dict:
        """Move arm to a named pose (home, retract, reach_forward, etc.)."""
        return await ctx.deps.arm_mcp.call_tool("move_named_pose", {
            "name": pose_name,
            "duration_sec": duration_sec,
        })

    @agent.tool
    async def list_tool_mcps(ctx: RunContext[ScenarioAgentDeps]) -> dict:
        """
        List all end-effector tool MCPs attached to this robot.

        Returns a mapping of tool name → list of available actions.
        Call this before using call_tool_action so you know what tools and
        actions are available on the current robot configuration.
        """
        result: dict[str, list[str]] = {}
        for tool_name, mcp in ctx.deps.tool_mcps.items():
            try:
                tools = await mcp.list_tools()
                result[tool_name] = [t.name for t in tools]
            except Exception as e:
                result[tool_name] = [f"<error listing tools: {e}>"]
        return result

    @agent.tool
    async def call_tool_action(
        ctx: RunContext[ScenarioAgentDeps],
        tool: str,
        action: str,
        params: dict | None = None,
    ) -> dict:
        """
        Call an action on an attached end-effector tool MCP.

        Args:
            tool:   Tool MCP name as returned by list_tool_mcps (e.g. "gripper", "vacuum").
            action: Action name as returned by list_tool_mcps (e.g. "open", "close", "activate").
            params: Optional parameters for the action (tool-specific).
        """
        if tool not in ctx.deps.tool_mcps:
            available = list(ctx.deps.tool_mcps)
            return {
                "ok": False,
                "message": f"Tool '{tool}' not registered. Available: {available}",
            }
        return await ctx.deps.tool_mcps[tool].call_tool(action, params or {})

    return agent
