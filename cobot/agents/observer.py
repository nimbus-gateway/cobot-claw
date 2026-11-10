"""Observer agent: answers questions while the robot is busy with a task.

It reads the task log (what the robot has done so far), the simulated scene
and the arm's state. It has no tools that move anything, so questions typed
mid-task can't interfere with the running task.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from pydantic_ai import Agent, RunContext

from cobot.config import CobotConfig
from cobot.llm import make_model

_PROMPT = (
    "You answer the operator's questions while a UR3e robot cell is busy with a task.\n"
    "You can only look, not act: read the task log (get_task_log), the scene (get_scene) and the arm\n"
    "state (get_arm_state). Answer from what they return, in one to three short sentences of plain text\n"
    "(no markdown: it prints in a terminal).\n"
    "- 'What's wrong with that one / the previous part' means the most recent part in the log.\n"
    "- Use the names from the log and scene (e.g. 'part 2', 'reject bin'), and say what was found and where\n"
    "  it went or is going. A part counts as put away once the log says it was dropped in a container.\n"
    "- If the log doesn't say, say so; don't guess.\n"
    "- If the operator asks the robot to DO something (move, pick, pack, skip, change the plan), say you\n"
    "  can't while the task runs: they can wait for it to finish, or type 'stop' (or press Esc) to cancel it.\n"
)


@dataclass
class ObserverDeps:
    task_log: Any
    sim_scene: Any = None
    arm_mcp: Any = None


def build_observer(cfg: CobotConfig) -> Agent:
    obs = getattr(cfg.agents, "observer", None)
    model = obs.model if obs else cfg.agents.safety.model   # a small, fast model is enough
    agent: Agent[ObserverDeps, str] = Agent(
        make_model(model, cfg),
        deps_type=ObserverDeps,
        output_type=str,
        system_prompt=(obs.system_prompt if obs and obs.system_prompt else _PROMPT),
    )

    @agent.tool
    async def get_task_log(ctx: RunContext[ObserverDeps], last_n: int = 60) -> dict:
        """What the robot is doing now and the recent events of this session:
        steps (picked, placed), results (inspection PASS/FAIL with defect,
        packed items) and problems, oldest first."""
        return ctx.deps.task_log.snapshot(last_n)

    @agent.tool
    async def get_scene(ctx: RunContext[ObserverDeps]) -> dict:
        """Objects and containers in the simulated cell, where each object is,
        inspection results already revealed, and order status (packing)."""
        if ctx.deps.sim_scene is None:
            return {"ok": False, "message": "No simulated scene in this session"}
        try:
            return await ctx.deps.sim_scene.call_tool("list_objects", {})
        except Exception as e:  # noqa: BLE001
            return {"ok": False, "message": f"Scene unavailable: {e}"}

    @agent.tool
    async def get_arm_state(ctx: RunContext[ObserverDeps]) -> dict:
        """The arm's joint angles and status (read-only)."""
        if ctx.deps.arm_mcp is None:
            return {"ok": False, "message": "Arm not connected"}
        try:
            return await ctx.deps.arm_mcp.call_tool("get_joint_states", {})
        except Exception as e:  # noqa: BLE001
            return {"ok": False, "message": f"Arm state unavailable: {e}"}

    return agent
