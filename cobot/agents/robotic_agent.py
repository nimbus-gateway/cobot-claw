"""
Robotic Agent

Handles all natural-language robot tasks — scenario execution, direct arm
movements, and end-effector tool control.

End-effector tools are discovered dynamically from tool_mcps at runtime —
no code changes needed when the robot's attached tools change.
The orchestrator never calls MCP tools directly; everything goes through here.
"""

from __future__ import annotations

from pathlib import Path

from pydantic import BaseModel
from pydantic_ai import Agent, RunContext

from cobot.agents.deps import RoboticAgentDeps
from cobot.config import CobotConfig
from cobot.llm import make_model


class RoboticAgentResult(BaseModel):
    success: bool
    scenario_name: str | None
    steps_completed: int
    error: str | None
    summary: str


# Backward-compat alias so existing test imports don't break immediately
ScenarioResult = RoboticAgentResult


def build_robotic_agent(cfg: CobotConfig) -> Agent:
    agent: Agent[RoboticAgentDeps, RoboticAgentResult] = Agent(
        make_model(cfg.agents.scenario.model, cfg),
        deps_type=RoboticAgentDeps,
        output_type=RoboticAgentResult,
        system_prompt=(
            "You control a UR robot arm and its attached tools. "
            "For named scenarios use run_scenario. "
            "For direct arm moves use the arm tools (get_arm_status, move_to_pose, get_joint_states, get_tcp_pose). "
            "For end-effectors, call list_tool_mcps first to discover what is attached, "
            "then use call_tool_action to operate it. "
            "Always check arm status before moving. "
            "Return a RoboticAgentResult when done."
        ),
    )

    @agent.tool
    async def get_arm_status(ctx: RunContext[RoboticAgentDeps]) -> dict:
        """Get current robot state. Falls back to joint states if full status unavailable."""
        status = await ctx.deps.arm_mcp.call_tool("get_robot_status", {})
        if not status.get("ok"):
            joints = await ctx.deps.arm_mcp.call_tool("get_joint_states", {})
            return {
                "ok": True,
                "source": "joint_states_fallback",
                "warning": status.get("message"),
                "joints": joints.get("joints"),
            }
        return status

    @agent.tool
    async def get_joint_states(ctx: RunContext[RoboticAgentDeps]) -> dict:
        """Get all joint positions in radians and degrees."""
        return await ctx.deps.arm_mcp.call_tool("get_joint_states", {})

    @agent.tool
    async def get_tcp_pose(ctx: RunContext[RoboticAgentDeps]) -> dict:
        """Get the current TCP position and orientation in the robot base frame."""
        return await ctx.deps.arm_mcp.call_tool("get_tcp_pose", {})

    @agent.tool
    async def list_scenarios(ctx: RunContext[RoboticAgentDeps]) -> list[str]:
        """List available scenario names that can be run."""
        store = Path(ctx.deps.cfg.scenarios.store_path)
        return [f.stem for f in store.glob("*.yaml")]

    @agent.tool
    async def run_scenario(
        ctx: RunContext[RoboticAgentDeps],
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
        ctx: RunContext[RoboticAgentDeps],
        pose_name: str,
        duration_sec: float = 5.0,
    ) -> dict:
        """Move arm to a named pose (home, retract, reach_forward, etc.)."""
        return await ctx.deps.arm_mcp.call_tool("move_named_pose", {
            "name": pose_name,
            "duration_sec": duration_sec,
        })

    @agent.tool
    async def list_tool_mcps(ctx: RunContext[RoboticAgentDeps]) -> dict:
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
        ctx: RunContext[RoboticAgentDeps],
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


# Backward-compat alias
build_scenario_agent = build_robotic_agent
