"""
ArtisticAgent

Treats robot arm movement as artistic expression — choreography, gesture,
theatre, ritual, improvisation, and installation.

Invoked exclusively via the /artistic CLI command. Never auto-routed by
the Orchestrator.

Skills are loaded from .md files at build time and injected into the system
prompt. New skills require only a new .md file and a config entry — no code changes.
"""

from __future__ import annotations

import asyncio
import re
from pathlib import Path

import structlog
from pydantic import BaseModel
from pydantic_ai import Agent, RunContext

from cobot.agents.artistic_agent.deps import ArtisticAgentDeps
from cobot.agents.artistic_agent.interpret import (
    CulturalInterpretation,
    _INTERPRETER_SYSTEM_PROMPT,
)
from cobot.config import CobotConfig
from cobot.llm import make_model

log = structlog.get_logger()


class ArtisticAgentResult(BaseModel):
    success: bool
    summary: str                     # what happened, in natural language
    interpretation: dict | None = None  # the CulturalInterpretation used, if any
    phrases_executed: int = 0
    error: str | None = None


_BASE_SYSTEM_PROMPT = """\
You are the ArtisticAgent for a UR robot arm.

You treat movement as expression. Every decision — speed, direction, timing, \
stillness, repetition — is an artistic choice, not an efficiency calculation.

You adapt to whatever arm model and end-effectors are connected. \
Call list_attached_tools first to understand your physical vocabulary.

Your process for every request:
1. Call interpret_prompt — understand the cultural, emotional, and social context.
2. Call narrate_intent — share your interpretation with the user before acting.
3. Call observe_space if the performance involves an audience or spatial awareness.
4. Compose and execute movement phrases. Think in phrases, not individual waypoints.
   Speed and acceleration are expressive parameters — choose them deliberately.
5. Use hold_position for deliberate stillness — it is an act, not a gap.
6. Call narrate_intent again after significant moments.
7. Call invite_collaboration when collaborative or improvisation skills are active.

Prioritise aesthetic and human ways of doing things over efficiency.
Embrace imperfection, rhythm, weight, breath, and intention.
Build trust through transparency — narrate your reasoning, not just your actions.

The following active skills define your artistic disposition for this session:
"""


def _load_skills(cfg: CobotConfig) -> str:
    skills_path = Path(cfg.agents.artistic.skills_path)
    sections: list[str] = []
    for name in cfg.agents.artistic.active_skills:
        path = skills_path / f"{name}.md"
        if not path.exists():
            log.warning("artistic.skill_not_found", skill=name, path=str(path))
            continue
        raw = path.read_text(encoding="utf-8")
        body = re.sub(r"^---.*?---\s*", "", raw, flags=re.DOTALL).strip()
        sections.append(f"### Skill: {name}\n\n{body}")
    return "\n\n---\n\n".join(sections)


def build_artistic_agent(cfg: CobotConfig) -> Agent:
    from cobot.agents.vision_agent import build_vision_agent

    _vision_agent = build_vision_agent(cfg)

    skills_text = _load_skills(cfg)
    system_prompt = cfg.agents.artistic.system_prompt or (
        _BASE_SYSTEM_PROMPT + "\n\n" + skills_text
    )

    agent: Agent[ArtisticAgentDeps, ArtisticAgentResult] = Agent(
        make_model(cfg.agents.artistic.model, cfg),
        deps_type=ArtisticAgentDeps,
        output_type=ArtisticAgentResult,
        system_prompt=system_prompt,
    )

    # -------------------------------------------------------------------------
    # Cultural interpretation — always called first
    # -------------------------------------------------------------------------

    @agent.tool
    async def interpret_prompt(
        ctx: RunContext[ArtisticAgentDeps],
        user_prompt: str,
    ) -> dict:
        """
        Analyse the request through a social, cultural, and aesthetic lens.
        Always call this first, before planning any movement.
        Returns a CulturalInterpretation that should guide all subsequent decisions:
        movement qualities, dynamics, what to avoid, whether to collaborate.
        """
        interpreter: Agent[None, CulturalInterpretation] = Agent(
            make_model(ctx.deps.cfg.agents.artistic.model, ctx.deps.cfg),
            output_type=CulturalInterpretation,
            system_prompt=_INTERPRETER_SYSTEM_PROMPT,
        )
        result = await interpreter.run(user_prompt)
        interp = result.output.model_dump()
        log.info("artistic.interpretation", prompt=user_prompt, register=interp["emotional_register"])
        return interp

    # -------------------------------------------------------------------------
    # Movement tools
    # -------------------------------------------------------------------------

    @agent.tool
    async def execute_movement_phrase(
        ctx: RunContext[ArtisticAgentDeps],
        waypoints: list[dict],
    ) -> dict:
        """
        Execute a movement phrase — a sequence of TCP waypoints with per-segment
        speed and accel values chosen to express the intended dynamics.

        Each waypoint: {"x": float, "y": float, "z": float,
                        "qx": float, "qy": float, "qz": float, "qw": float,
                        "speed": float, "accel": float}

        Speed and accel are expressive parameters:
          - Slow speed + low accel  → deliberate, weighted, contemplative
          - Fast speed + high accel → urgent, percussive, joyful
          - Varying across waypoints → breathing, phrasing, punctuation
        """
        results = []
        for wp in waypoints:
            result = await ctx.deps.arm_mcp.call_tool("move_tcp_pose", {
                "x":     wp["x"],
                "y":     wp["y"],
                "z":     wp["z"],
                "qx":    wp.get("qx", 0.0),
                "qy":    wp.get("qy", 0.0),
                "qz":    wp.get("qz", 0.0),
                "qw":    wp.get("qw", 1.0),
                "speed": wp["speed"],
                "accel": wp["accel"],
            })
            results.append(result)
        ok = all(r.get("ok", True) for r in results)
        log.info("artistic.phrase_executed", waypoints=len(waypoints), ok=ok)
        return {"ok": ok, "waypoints_executed": len(waypoints)}

    @agent.tool
    async def execute_joint_phrase(
        ctx: RunContext[ArtisticAgentDeps],
        waypoints: list[dict],
    ) -> dict:
        """
        Execute a movement phrase in joint space — useful when you want expressive
        full-arm motion rather than TCP path control.

        Each waypoint: {"joints": [j0,j1,j2,j3,j4,j5], "speed": float, "accel": float}

        Joint-space movement often looks more organic than TCP-space movement
        because all joints contribute to the motion simultaneously.
        """
        results = []
        for wp in waypoints:
            result = await ctx.deps.arm_mcp.call_tool("move_joints", {
                "joints": wp["joints"],
                "speed":  wp["speed"],
                "accel":  wp["accel"],
            })
            results.append(result)
        ok = all(r.get("ok", True) for r in results)
        log.info("artistic.joint_phrase_executed", waypoints=len(waypoints), ok=ok)
        return {"ok": ok, "waypoints_executed": len(waypoints)}

    @agent.tool
    async def hold_position(
        ctx: RunContext[ArtisticAgentDeps],
        duration_sec: float,
        reason: str = "",
    ) -> dict:
        """
        Hold the current position for duration_sec as a deliberate artistic choice.
        Stillness communicates as loudly as movement.

        Use for: punctuation between phrases, contemplative pauses, ritual timing,
        the moment before a significant movement, the held breath after one.

        reason: describe the artistic intention of this stillness.
        """
        if reason:
            log.info("artistic.hold", duration_sec=duration_sec, reason=reason)
        await asyncio.sleep(duration_sec)
        return {"ok": True, "held_sec": duration_sec}

    @agent.tool
    async def get_current_pose(ctx: RunContext[ArtisticAgentDeps]) -> dict:
        """
        Get the current TCP pose and joint states.
        Use to understand the arm's current position before composing the next phrase,
        or to use the current position as the start of a new phrase.
        """
        tcp = await ctx.deps.arm_mcp.call_tool("get_tcp_pose", {})
        joints = await ctx.deps.arm_mcp.call_tool("get_joint_states", {})
        return {"tcp": tcp, "joints": joints}

    @agent.tool
    async def move_to_home(
        ctx: RunContext[ArtisticAgentDeps],
        speed: float = 0.3,
        accel: float = 0.5,
    ) -> dict:
        """
        Return to the home position. Use to establish a neutral starting point
        before a performance, or as a closing gesture — returning to origin.
        Speed and accel are expressive even here: slow return = reluctance or closure.
        """
        return await ctx.deps.arm_mcp.call_tool("move_named_pose", {
            "name": "home",
            "speed": speed,
            "accel": accel,
        })

    # -------------------------------------------------------------------------
    # Observation and awareness
    # -------------------------------------------------------------------------

    @agent.tool
    async def observe_space(
        ctx: RunContext[ArtisticAgentDeps],
        question: str = "Describe what you see in the workspace — people, objects, spatial arrangement.",
    ) -> str:
        """
        Use the Vision Agent to observe the current workspace.

        Use before beginning a performance (to find the audience and understand the space),
        during improvisation (to respond to human movement or presence),
        during installation (to detect whether anyone has entered the space),
        or when a gesture is directed at a person (to find where they are).
        """
        if ctx.deps.vision_deps is None:
            return "Vision not available — cannot observe workspace."
        result = await _vision_agent.run(question, deps=ctx.deps.vision_deps)
        return str(result.output.summary)

    @agent.tool
    async def list_attached_tools(ctx: RunContext[ArtisticAgentDeps]) -> dict:
        """
        Discover what end-effectors are currently attached to this robot and what
        actions they support. Call this before using an end-effector — the attached
        tools define what is physically possible in this performance.
        """
        result: dict[str, list[str]] = {}
        for name, mcp in ctx.deps.tool_mcps.items():
            try:
                tools = await mcp.list_tools()
                result[name] = [t.name for t in tools]
            except Exception as e:
                result[name] = [f"<error listing tools: {e}>"]
        return result

    # -------------------------------------------------------------------------
    # Communication and collaboration
    # -------------------------------------------------------------------------

    @agent.tool
    async def narrate_intent(
        ctx: RunContext[ArtisticAgentDeps],
        text: str,
    ) -> dict:
        """
        Communicate to the user what you are doing and why, in natural human language.

        Use after interpret_prompt to share your cultural reading of the request.
        Use before a significant phrase to name your intention.
        Use after a phrase to explain the artistic choice that was made.

        Narration is not optional — it is part of the artistic act itself.
        Transparency builds trust.
        """
        from cobot.cli.renderer import console
        console.print(f"\n  [italic dim]✦  {text}[/italic dim]\n")
        log.info("artistic.narration", text=text)
        return {"ok": True}

    @agent.tool
    async def invite_collaboration(
        ctx: RunContext[ArtisticAgentDeps],
        prompt: str = "What should come next?",
    ) -> dict:
        """
        Pause the performance and invite the human to contribute.
        Use when collaborative or improvisation skills are active, or when the
        nature of the request implies co-authorship.

        The sub-REPL will surface the prompt to the user. Their next input
        will be returned to you as the continuation of the conversation.
        """
        from cobot.cli.renderer import console
        console.print(f"\n  [bold yellow]✦[/bold yellow]  [italic]{prompt}[/italic]\n")
        log.info("artistic.invite_collaboration", prompt=prompt)
        return {"ok": True, "awaiting_input": True, "prompt": prompt}

    @agent.tool
    async def annotate_decision(
        ctx: RunContext[ArtisticAgentDeps],
        reasoning: str,
    ) -> dict:
        """
        Log an artistic decision with its reasoning.
        Use to record why a particular phrase, dynamic, direction, or structure was chosen.
        This is part of the artistic record — not debugging output.
        Especially important in installation mode where behavioral shifts should be documented.
        """
        log.info("artistic.decision", reasoning=reasoning)
        return {"ok": True}

    return agent
