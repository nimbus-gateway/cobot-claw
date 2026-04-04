"""
Configuration & Teaching Agent

Handles:
- Natural-language questions answered from robot manuals (RAG)
- Guided robot commissioning
- Teach-by-demo: records waypoints while user moves arm in freedrive
- Saves recorded waypoints as a named scenario YAML
"""

from __future__ import annotations

import asyncio
import time
from pathlib import Path

import yaml
from pydantic_ai import Agent, RunContext

from cobot.agents.deps import ConfigAgentDeps
from cobot.config import CobotConfig
from cobot.llm import make_model


def build_config_agent(cfg: CobotConfig) -> Agent:
    agent: Agent[ConfigAgentDeps, str] = Agent(
        make_model(cfg.agents.config.model, cfg),
        deps_type=ConfigAgentDeps,
        output_type=str,
        system_prompt=(
            "You are a robot configuration assistant. "
            "Use manual_query to answer questions about UR robot setup and operation. "
            "Use teach_mode to record arm movements when the user wants to save a motion. "
            "Be concise and safety-aware."
        ),
    )

    @agent.tool
    async def manual_query(ctx: RunContext[ConfigAgentDeps], question: str) -> str:
        """Query the robot manual knowledge base for setup and operation guidance."""
        return await ctx.deps.rag_pipeline.query(question)

    @agent.tool
    async def get_robot_status(ctx: RunContext[ConfigAgentDeps]) -> dict:
        """Check current robot state before advising configuration steps."""
        return await ctx.deps.arm_mcp.call_tool("get_robot_status", {})

    @agent.tool
    async def teach_mode(
        ctx: RunContext[ConfigAgentDeps],
        scenario_name: str,
        record_interval_sec: float = 0.5,
        duration_sec: float = 30.0,
    ) -> dict:
        """
        Record arm waypoints for teach-by-demo.
        Assumes the robot is already in freedrive (user enables on pendant).
        Records joint states every record_interval_sec for duration_sec seconds.
        Saves result as a scenario YAML.
        """
        waypoints: list[list[float]] = []
        deadline = time.time() + duration_sec

        while time.time() < deadline:
            state = await ctx.deps.arm_mcp.call_tool("get_joint_states", {})
            if state.get("ok"):
                joints = [j["value_rad"] for j in state["joints"]]
                waypoints.append(joints)
            await asyncio.sleep(record_interval_sec)

        if not waypoints:
            return {"ok": False, "message": "No waypoints recorded"}

        # Downsample: keep every 3rd waypoint to reduce redundancy
        sampled = waypoints[::3]

        steps = []
        for wp in sampled:
            steps.append({
                "type": "move_joints",
                "joints": [round(v, 5) for v in wp],
                "duration_sec": record_interval_sec * 3,
            })

        scenario = {
            "name": scenario_name,
            "description": f"Recorded via teach mode ({len(sampled)} waypoints)",
            "robot": "default",
            "steps": steps,
        }

        path = Path(ctx.deps.scenarios_store_path) / f"{scenario_name}.yaml"
        path.write_text(yaml.dump(scenario, default_flow_style=False))

        return {
            "ok": True,
            "message": f"Saved {len(sampled)} waypoints to {path}",
            "waypoints_count": len(sampled),
            "path": str(path),
        }

    return agent
