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
    task_log: object | None = None   # cobot.tasklog.TaskLog

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

    _DEFAULT_ORCHESTRATOR_PROMPT = (
        "You orchestrate a UR robot system.\n\n"
        "Routing rules — follow these EXACTLY, one tool call per request:\n\n"
        "• 'Pick X' / 'grab X' / 'lift X' (no destination) "
        "→ run_vision_task with 'pick <object>'.\n\n"
        "• 'Pick X and put/place/drop/move it to/in/on Y' "
        "→ run_vision_task with 'pick <object> and place it on <target>'.\n\n"
        "• 'Point to X', 'show me X', 'hover over X' "
        "→ run_vision_task with 'point to <object>'.\n\n"
        "• 'Where is X', 'what do you see', 'workspace check' "
        "→ run_vision_task\n\n"
        "• 'Inspect X', 'sort the parts', 'quality check', 'move the good/bad parts', 'reset the scene', "
        "'pack X into Y', 'pack the orders', 'what orders are open' "
        "→ run_vision_task with the request as given\n\n"
        "• Robot status, movements, tool control, running an EXISTING named scenario "
        "→ run_robotic_task\n\n"
        "• Create/save/teach a scenario, robot manuals, specs, safety questions "
        "→ run_config_task\n\n"
        "• Questions about what the robot did or found ('why was part 2 rejected?', 'which parts "
        "passed?', 'what happened?', 'what's in shipping box 1?') → get_task_log, and answer from it. "
        "Don't re-run a robot task to answer a question.\n\n"
        "Never call robot hardware or camera directly.\n"
        "After a tool returns its result, immediately return that result as your "
        "final response. Do not call any tool more than once per user request."
    )

    agent: Agent[OrchestratorDeps, str] = Agent(
        make_model(cfg.agents.orchestrator.model, cfg),
        deps_type=OrchestratorDeps,
        output_type=str,
        retries=1,
        system_prompt=cfg.agents.orchestrator.system_prompt or _DEFAULT_ORCHESTRATOR_PROMPT,
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
    async def get_task_log(ctx: RunContext[OrchestratorDeps], last_n: int = 80) -> dict:
        """What the robot did this session, step by step: inspection results
        (PASS/FAIL and the defect), parts picked and where they were placed,
        items packed, problems, and any task still running."""
        if ctx.deps.task_log is None:
            return {"ok": False, "message": "No task log in this session"}
        return ctx.deps.task_log.snapshot(last_n)

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
