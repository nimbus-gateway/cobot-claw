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
from cobot.agents.deps import RoboticAgentDeps, ConfigAgentDeps, VisionAgentDeps


@dataclass
class OrchestratorDeps:
    robotic_deps: RoboticAgentDeps
    cfg: CobotConfig
    config_deps: ConfigAgentDeps | None = None
    vision_deps: VisionAgentDeps | None = None

    # Backward-compat: allow old code that passes scenario_deps= keyword
    @classmethod
    def _from_legacy(cls, scenario_deps: RoboticAgentDeps, cfg: CobotConfig, config_deps=None):
        return cls(robotic_deps=scenario_deps, cfg=cfg, config_deps=config_deps)


def build_orchestrator(cfg: CobotConfig) -> Agent:
    from cobot.agents.robotic_agent import build_robotic_agent
    from cobot.agents.config_agent import build_config_agent
    from cobot.agents.vision_agent import build_vision_agent

    _robotic_agent = build_robotic_agent(cfg)
    _config_agent = build_config_agent(cfg)
    _vision_agent = build_vision_agent(cfg)

    agent: Agent[OrchestratorDeps, str] = Agent(
        make_model(cfg.agents.orchestrator.model, cfg),
        deps_type=OrchestratorDeps,
        output_type=str,
        retries=1,
        system_prompt=(
            "You orchestrate a UR robot system.\n\n"
            "Routing rules — follow these EXACTLY, one tool call per request:\n\n"
            "• User says 'pick', 'grab', 'lift', 'get' an object (e.g. 'pick the bottle') "
            "→ run_vision_task with 'pick <object>'. "
            "The Vision Agent detects the object AND moves the arm directly. "
            "Do NOT call run_robotic_task separately for this.\n\n"
            "• CREATING a scenario (user says 'create scenario', 'generate scenario', 'save') "
            "→ run_vision_task with the pick and place descriptions.\n\n"
            "• Robot status, movements, tool control, running an EXISTING named scenario "
            "→ run_robotic_task\n\n"
            "• 'Point to X', 'show me X', 'hover over X', 'indicate X' "
            "→ run_vision_task with 'point to <object>'. "
            "The Vision Agent moves the arm above the object without picking it.\n\n"
            "• Visual perception only (where is X, what do you see, workspace check) "
            "→ run_vision_task\n\n"
            "• Robot manuals, specs, safety questions, teach-by-demo "
            "→ run_config_task\n\n"
            "Never call robot hardware or camera directly.\n"
            "After a tool returns its result, immediately return that result as your "
            "final response. Do not call any tool more than once per user request."
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
    async def run_vision_task(
        ctx: RunContext[OrchestratorDeps],
        task_description: str,
    ) -> str:
        """Delegate to the Vision Agent — object detection, 3D location, workspace check, AND creating new pick-and-place scenarios."""
        result = await _vision_agent.run(task_description, deps=ctx.deps.vision_deps)
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
