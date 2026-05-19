# ArtisticAgent — Architecture and Implementation

## Philosophy

The ArtisticAgent is not a drawing tool. It is an agent that treats **movement itself as expression** — choreography, gesture, theatre, ritual, and improvisation are all first-class artistic acts that a robot arm can perform.

Key principles that drive every design decision:

- **Art is a verb, not a noun.** The artistic act is the process of moving, not the object produced.
- **The domain is 3D space and time.** Not a canvas. Not a surface. The full reachable workspace of the arm, across time, with rhythm and dynamics.
- **Efficiency is not the goal.** A deliberate pause, a slow arc, a repeated gesture — these are valid and intentional choices, not inefficiencies to be optimised away.
- **Social and cultural context matters.** A "circle" can be an enso, a clock, a crown, or a symbol of unity. The agent interprets prompts through human meaning before it plans motion.
- **The arm is the performer.** Speed, acceleration, stillness, direction, and rhythm are expressive parameters — not just hardware settings.
- **Trust is built through transparency.** The agent narrates its reasoning. Users should understand not just what the arm does but why.
- **Collaboration over completion.** An incomplete act that invites human participation is often more artistically valid than a polished solo performance.

---

## Position in the System

The ArtisticAgent is a **peer agent** alongside RoboticAgent, VisionAgent, and ConfigAgent. It is not routed to by the Orchestrator during normal operation — it is entered **exclusively via the `/artistic` CLI command**, which opens a dedicated sub-REPL.

```
cobot REPL
 └── /artistic ──────────────────────────────────────────────────┐
                                                                  │
      ArtisticAgent sub-REPL                                      │
       ├── ArtisticAgent  (PydanticAI)                            │
       │    ├── interpret_prompt   → nested LLM call              │
       │    ├── observe_space      → VisionAgent                  │
       │    ├── narrate_intent     → display to user              │
       │    ├── invite_collaboration → pause, await human         │
       │    ├── compose_movement_phrase → plan expressive motion  │
       │    ├── execute_movement_phrase → call arm MCP            │
       │    ├── hold_position      → call arm MCP (stillness)     │
       │    └── list_attached_tools → call tool MCPs              │
       │                                                          │
       ├── Arm MCP        (reuse: tools/ure3/ur3e_arm_mcp.py)    │
       ├── Tool MCPs      (reuse: gripper, pen, marker, etc.)     │
       └── VisionAgent    (reuse: build_vision_agent)             │
                                                                  │
      /done ──────────────────────────────────────────────────────┘
```

The Orchestrator's system prompt does **not** route to ArtisticAgent. This is intentional — artistic mode is a deliberate choice by the user, not an automatic delegation.

---

## File Structure

```
cobot/agents/artistic_agent/
├── __init__.py              # exports build_artistic_agent, ArtisticAgentDeps
├── agent.py                 # Agent definition and all tools
├── deps.py                  # ArtisticAgentDeps dataclass
├── interpret.py             # CulturalInterpretation model + nested LLM call
└── skills/
    ├── choreography.md
    ├── theatre.md
    ├── gesture.md
    ├── improvisation.md
    ├── ritual.md
    ├── installation.md
    └── drawing.md

cobot/cli/commands/
└── artistic.py              # cmd_artistic() — the /artistic sub-REPL

cobot/cli/
└── repl.py                  # add /artistic case to _handle_slash_command
```

---

## Config Schema

Add `ArtisticAgentConfig` to `cobot/config.py`:

```python
class ArtisticAgentConfig(AgentModelConfig):
    skills_path: str = "./cobot/agents/artistic_agent/skills"
    active_skills: list[str] = ["choreography", "gesture"]
```

Add to `AgentsConfig`:

```python
class AgentsConfig(BaseModel):
    orchestrator: AgentModelConfig
    scenario: AgentModelConfig
    config: AgentModelConfig
    safety: SafetyAgentConfig
    vision: VisionAgentConfig
    artistic: ArtisticAgentConfig          # new
```

`cobot.yaml` addition:

```yaml
agents:
  artistic:
    model: qwen2.5:72b
    system_prompt: ""                      # empty = built from active skills
    skills_path: ./cobot/agents/artistic_agent/skills
    active_skills:
      - choreography
      - gesture
      - collaborative
```

---

## Deps

```python
# cobot/agents/artistic_agent/deps.py
from __future__ import annotations
from dataclasses import dataclass
from cobot.mcp.client import MCPClient
from cobot.agents.vision_agent import VisionAgentDeps
from cobot.config import CobotConfig


@dataclass
class ArtisticAgentDeps:
    arm_mcp: MCPClient
    tool_mcps: dict[str, MCPClient]        # whatever end-effectors are connected
    vision_deps: VisionAgentDeps | None    # reuse existing VisionAgent
    cfg: CobotConfig
```

No new MCP servers. The ArtisticAgent uses the same hardware interfaces as every other agent.

---

## Skills — `.md` Files

Skills are the core mechanism for giving the agent different artistic dispositions. They are **markdown documents injected into the system prompt** at build time. Skills shape how the agent thinks and moves — they do not add or remove tools.

### Why `.md` not `.py`

- A new skill is a new document, not a code change.
- Skills can be written by artists, researchers, or domain experts — not just engineers.
- Multiple skills compose naturally: both `choreography.md` and `theatre.md` active means the agent synthesises both.
- The agent's interpretation of a skill is itself an act of intelligence — it reads the guidance and decides how to apply it to the specific request.

### Skill file format

```markdown
---
name: choreography
description: Movement as dance — rhythm, phrasing, dynamics, and structure
---

[body — injected into system prompt]
```

Frontmatter is parsed for the `name` used in config `active_skills`. The body is the content injected.

### Skill loading

```python
import re
from pathlib import Path

def _load_skills(cfg) -> str:
    skills_path = Path(cfg.agents.artistic.skills_path)
    sections = []
    for name in cfg.agents.artistic.active_skills:
        path = skills_path / f"{name}.md"
        if not path.exists():
            continue
        raw = path.read_text()
        body = re.sub(r"^---.*?---\s*", "", raw, flags=re.DOTALL).strip()
        sections.append(f"### Active Skill: {name}\n\n{body}")
    return "\n\n---\n\n".join(sections)
```

### Example skill files

#### `skills/choreography.md`

```markdown
---
name: choreography
description: Movement as dance — rhythm, phrasing, dynamics, and structure
---

## Choreography

Movement is not displacement. It is time made visible.

A choreographic phrase is a unit of expressive motion with a beginning,
a development, and a resolution. Think in phrases, not in waypoints.

**Rhythm** — movement has pulse. Use speed and acceleration as a drummer
uses a beat: sometimes rushing, sometimes behind the beat, always intentional.
Vary speed along an arc. Slow into a held position. Accelerate through transitions.

**Dynamics** — the same path traced at different energies means different things.
Forte: fast, large, definite. Piano: slow, small, tentative.
Choose dynamics that match the emotional register of the request.

**Structure** — a performance has shape. Use compose_movement_phrase to think
in sections: introduction, development, climax, resolution. Not every request
needs all four, but always know where you are in the structure.

**Stillness** — hold_position is not a pause. It is a moment of presence.
A body that stops moving in a deliberate way communicates as loudly as one in motion.
Use stillness as punctuation.

**Repetition** — the same phrase repeated is not the same phrase. Each iteration
adds meaning. Use repetition when the request implies accumulation, insistence, or ritual.

Before composing any phrase, call interpret_prompt to understand what emotional
and cultural register the request belongs to.
```

#### `skills/theatre.md`

```markdown
---
name: theatre
description: The arm as performer — character, emotion, narrative, audience
---

## Theatre

The arm is not a machine executing a task. In this mode, it is a performer
with intention, character, and presence.

**Emotion through motion** — physical states map to emotional states:
- Curiosity: slow approach, pause, small retraction, then closer
- Grief: slow, weighted, downward arcs; long holds at low positions
- Joy: upward, outward, faster, larger range; light quick movements
- Hesitation: approach, partial stop, small retreat, then proceed
- Defiance: held position resisting expected movement; sudden sharp change
- Tenderness: minimal speed, small gestures, proximity to a surface or object

**Character** — before acting, decide who the arm is in this performance.
A mechanical automaton? A curious child? A weary labourer? A priest performing rites?
Character does not change the tools available. It changes every parameter of how
they are used.

**Narrative arc** — even a short performance has a story. Call interpret_prompt
to understand what story the request contains. Then structure the movements to
tell it: setup, complication, resolution.

**Audience** — call observe_space before beginning and again mid-performance.
Is a human close? Are they moving? Adjust. A performance that ignores its audience
is a rehearsal, not a performance.

After each significant movement phrase, call narrate_intent so the audience
understands what they witnessed.
```

#### `skills/gesture.md`

```markdown
---
name: gesture
description: Communicative movement — the arm speaks through motion
---

## Gesture

A gesture is a movement that means something to the person watching.
Gestures are culturally specific, socially situated, and always relational
— they exist in the space between the arm and a human.

**Deictic gestures** — pointing, indicating, directing attention.
These are precise: the end-effector indicates a specific location in space.
Use locate_object via observe_space to find what is being pointed at.

**Iconic gestures** — movement that resembles what it describes.
Drawing the shape of a wave, tracing the path of a falling object, sketching
the outline of something in the air.

**Symbolic gestures** — culturally specific meanings: bowing, beckoning,
a raised palm (stop), a circular motion (again/continue).
Always call interpret_prompt first — the same motion means different things
in different cultures.

**Emotional/expressive gestures** — open, reaching arms vs. withdrawn, closed
positions. These communicate states rather than information.

**Rhythm of gesture** — a gesture is not instant. It has a preparation, a stroke
(the communicative peak), and a retraction. Never snap to a position — move through
the gesture.

When the request is relational ("wave at me", "point to that", "say goodbye"),
call observe_space first to understand where the human is in the workspace.
```

#### `skills/improvisation.md`

```markdown
---
name: improvisation
description: Responsive movement — reacting to the environment and human input
---

## Improvisation

Improvisation is not randomness. It is structured spontaneity — having strong
enough principles that meaningful choices can be made in real time.

**Listen before moving** — always call observe_space at the start and regularly
throughout. Where is the human? What is changing? What is the current state
of the workspace? Let these facts inform what comes next.

**Respond, don't plan** — improvisation means the next phrase emerges from
the current moment, not from a pre-composed structure. Do not commit to a full
sequence before beginning.

**Constraint as creativity** — physical constraints (joint limits, the weight
of an end-effector, speed limits) are not obstacles. They are the conditions
that make this particular performance unique. Work with them, not against them.

**Call and response** — when a human speaks or moves, respond physically.
Even a small acknowledgment — a slight rotation, a brief hold — communicates
that the arm is listening.

**Leave space** — not every moment needs to be filled with movement.
A held position in response to a human's action is itself a response.

Use invite_collaboration frequently. In improvisation, the human is a co-performer,
not an audience.
```

#### `skills/ritual.md`

```markdown
---
name: ritual
description: Repetition with intention — ceremony, accumulation, and meaning through recurrence
---

## Ritual

A ritual is a sequence of actions performed with deliberate formality.
Repetition is not redundancy — each iteration of a ritual gesture accumulates meaning.

**Formality** — ritual movements are precise and consistent. The same path,
the same speed, the same timing, each time. Deviation is meaningful.

**Accumulation** — repeat a phrase at least three times before considering it
established. The first time is introduction. The second is confirmation. The third
is ritual.

**Marking time** — use hold_position generously. In ritual, time between actions
is as significant as the actions themselves.

**Cultural grounding** — rituals are always culturally specific. Call interpret_prompt
to understand what ritual tradition the request invokes before moving. A gesture
of greeting in one culture is a gesture of farewell in another.

**Closure** — every ritual has a closing act. The return to a neutral or home
position should itself be performed with the same deliberateness as every other
phrase. Never rush the ending.
```

#### `skills/installation.md`

```markdown
---
name: installation
description: Sustained behavior — the arm as a presence in a space over time
---

## Installation

In installation mode, the arm is not performing for a moment — it is inhabiting
a space over a duration. The art is the arm's ongoing presence and behavior.

**Looping behavior** — compose a base phrase that can repeat indefinitely without
feeling mechanical. Introduce small variations on each cycle: slightly different
timing, a fractionally different position, a longer hold.

**Environmental responsiveness** — call observe_space periodically. If a human
enters the workspace, change the behavior. If they leave, return to the base state.
The installation responds to being observed.

**Sustained patience** — movements should be slow and considered. An installation
that moves frantically is exhausting to be near. Think in minutes, not seconds.

**Spatial presence** — choose a home position for the installation that is
interesting in the space. The arm at rest should look intentional, not parked.

**Documentation** — call annotate_decision at the start of each new behavioral
phase so there is a record of why the behavior changed.
```

---

## Tools

All tools are registered in Python in `agent.py`. Skills shape how the agent uses them — they do not gate access to tools.

### `interpret_prompt` — nested LLM call

The most important tool. Called **first** on any request before any movement is planned. Makes a separate, focused LLM call to produce a structured cultural interpretation.

```python
# cobot/agents/artistic_agent/interpret.py
from pydantic import BaseModel


class CulturalInterpretation(BaseModel):
    cultural_references: list[str]
    # e.g. ["Zen Buddhism", "Butoh dance", "Islamic geometry", "Grief ritual"]

    emotional_register: str
    # e.g. "contemplative", "celebratory", "sorrowful", "defiant", "playful"

    artistic_tradition: list[str]
    # e.g. ["wabi-sabi", "expressionism", "minimalism", "folk performance"]

    movement_qualities: list[str]
    # e.g. ["slow", "weighted", "expansive", "percussive", "sustained"]

    what_not_to_do: list[str]
    # e.g. ["do not be mechanical", "avoid sharp transitions", "do not rush the ending"]

    suggested_approach: str
    # Prose guidance specific to this request

    is_collaborative: bool
    # Should the human be invited to participate?

    has_audience: bool
    # Does this act expect to be witnessed? (affects observe_space usage)
```

```python
@agent.tool
async def interpret_prompt(ctx: RunContext[ArtisticAgentDeps], user_prompt: str) -> dict:
    """
    Analyse the request through a social, cultural, and aesthetic lens.
    Always call this first, before planning any movement.
    Returns a CulturalInterpretation that should guide all subsequent decisions.
    """
    from cobot.agents.artistic_agent.interpret import CulturalInterpretation
    from pydantic_ai import Agent as _Agent

    interpreter = _Agent(
        make_model(ctx.deps.cfg.agents.artistic.model, ctx.deps.cfg),
        output_type=CulturalInterpretation,
        system_prompt=(
            "You are a cultural analyst, choreographer, and performance theorist. "
            "Analyse requests for their social, cultural, historical, and emotional dimensions. "
            "Think across traditions: dance, theatre, ritual, visual art, performance art, gesture. "
            "Be specific. 'Make a sad movement' should return concrete movement qualities "
            "drawn from real performance traditions, not generic adjectives. "
            "Identify what would feel authentic versus what would feel mechanical or clichéd."
        ),
    )
    result = await interpreter.run(user_prompt)
    return result.output.model_dump()
```

### `compose_movement_phrase`

Plans a unit of expressive motion as a sequence of waypoints with timing. Does not move the arm — it returns a plan that `execute_movement_phrase` then executes.

```python
@agent.tool
async def compose_movement_phrase(
    ctx: RunContext[ArtisticAgentDeps],
    description: str,
    dynamics: str = "moderato",       # "pianissimo" | "piano" | "moderato" | "forte" | "fortissimo"
    duration_sec: float = 5.0,
) -> dict:
    """
    Plan a unit of expressive movement without executing it.
    Returns a sequence of TCP waypoints with per-segment speed/accel values.
    Call execute_movement_phrase to perform it.

    dynamics: expressive energy level from very soft (pianissimo) to very loud (fortissimo).
    """
    # Get current TCP position as phrase starting point
    current = await ctx.deps.arm_mcp.call_tool("get_tcp_pose", {})
    # Return structured plan for agent to inspect before executing
    return {
        "ok": True,
        "description": description,
        "dynamics": dynamics,
        "duration_sec": duration_sec,
        "start_pose": current,
        "note": (
            "Use execute_movement_phrase with a list of waypoints derived from "
            "the current pose and your interpretation of the description and dynamics."
        ),
    }
```

### `execute_movement_phrase`

Executes a composed phrase by sending a sequence of `move_joints` or `move_tcp_pose` calls to the arm MCP with varied speed and acceleration that reflect the dynamics.

```python
@agent.tool
async def execute_movement_phrase(
    ctx: RunContext[ArtisticAgentDeps],
    waypoints: list[dict],
    # Each waypoint: {"x": float, "y": float, "z": float, "speed": float, "accel": float}
) -> dict:
    """
    Execute a movement phrase — a sequence of TCP waypoints with per-segment
    speed and accel values. The agent chooses these values to express dynamics.

    Speed and accel are expressive parameters:
      - Slow, low accel → deliberate, weighted, contemplative
      - Fast, high accel → urgent, percussive, joyful
      - Variable across waypoints → breathing, phrasing, punctuation
    """
    results = []
    for wp in waypoints:
        result = await ctx.deps.arm_mcp.call_tool("move_tcp_pose", {
            "x": wp["x"], "y": wp["y"], "z": wp["z"],
            "qx": wp.get("qx", 0.0), "qy": wp.get("qy", 0.0),
            "qz": wp.get("qz", 0.0), "qw": wp.get("qw", 1.0),
            "speed": wp["speed"],
            "accel": wp["accel"],
        })
        results.append(result)
    ok = all(r.get("ok", True) for r in results)
    return {"ok": ok, "waypoints_executed": len(results)}
```

### `hold_position`

Deliberate stillness — not a pause between actions but an act in itself.

```python
@agent.tool
async def hold_position(
    ctx: RunContext[ArtisticAgentDeps],
    duration_sec: float,
    reason: str = "",
) -> dict:
    """
    Hold the current position for duration_sec as a deliberate artistic choice.
    Stillness communicates as loudly as movement.
    reason: optional annotation logged via annotate_decision.
    """
    import asyncio
    if reason:
        await annotate_decision(ctx, f"Holding for {duration_sec}s: {reason}")
    await asyncio.sleep(duration_sec)
    return {"ok": True, "held_sec": duration_sec}
```

### `observe_space`

Delegates to the existing VisionAgent to understand who and what is in the workspace.

```python
@agent.tool
async def observe_space(
    ctx: RunContext[ArtisticAgentDeps],
    question: str = "Describe what you see in the workspace — people, objects, spatial arrangement.",
) -> str:
    """
    Use the Vision Agent to observe the current workspace.
    Use before beginning a performance (to find the audience),
    during improvisation (to respond to human movement),
    or during installation (to detect presence/absence of people).
    """
    if ctx.deps.vision_deps is None:
        return "Vision not available — cannot observe workspace."
    result = await _vision_agent.run(question, deps=ctx.deps.vision_deps)
    return str(result.output.summary)
```

### `narrate_intent`

Displays the agent's reasoning to the user in natural language. Builds transparency and trust.

```python
@agent.tool
async def narrate_intent(
    ctx: RunContext[ArtisticAgentDeps],
    text: str,
) -> dict:
    """
    Communicate to the user what you are doing and why — in natural, human language.
    Use after interpret_prompt to share your cultural reading.
    Use after significant movement phrases to explain the artistic choice.
    Narration is not optional — it is part of the artistic act.
    """
    from cobot.cli.renderer import console
    console.print(f"\n  [italic dim]✦  {text}[/italic dim]\n")
    return {"ok": True}
```

### `invite_collaboration`

Pauses execution and returns control to the user mid-performance.

```python
@agent.tool
async def invite_collaboration(
    ctx: RunContext[ArtisticAgentDeps],
    prompt: str = "What should come next?",
) -> dict:
    """
    Pause and invite the human to participate. The performance is incomplete
    until the human responds. Use when the skill guidance or the nature of the
    request calls for co-authorship.
    Returns the human's response as a string.
    """
    from cobot.cli.renderer import console
    console.print(f"\n  [bold yellow]✦[/bold yellow]  {prompt}\n")
    # Signal to the sub-REPL to surface an input prompt
    return {"ok": True, "awaiting_human_input": True, "prompt": prompt}
```

### `list_attached_tools`

Discovers what end-effectors are currently connected — the same tool as in RoboticAgent.

```python
@agent.tool
async def list_attached_tools(ctx: RunContext[ArtisticAgentDeps]) -> dict:
    """
    List all end-effectors attached to this robot and their available actions.
    Call this before using an end-effector — the attached tools define what
    kinds of artistic acts are physically possible.
    """
    result = {}
    for name, mcp in ctx.deps.tool_mcps.items():
        try:
            tools = await mcp.list_tools()
            result[name] = [t.name for t in tools]
        except Exception as e:
            result[name] = [f"<error: {e}>"]
    return result
```

### `annotate_decision`

Structured log of artistic decisions — for transparency, debugging, and documentation.

```python
@agent.tool
async def annotate_decision(
    ctx: RunContext[ArtisticAgentDeps],
    reasoning: str,
) -> dict:
    """
    Log an artistic decision with its reasoning.
    Use to record why a particular phrase, dynamic, or structure was chosen.
    This is part of the artistic record — not debugging output.
    """
    import structlog
    log = structlog.get_logger()
    log.info("artistic.decision", reasoning=reasoning)
    return {"ok": True}
```

---

## Agent Definition

```python
# cobot/agents/artistic_agent/agent.py
from __future__ import annotations

import re
from pathlib import Path

from pydantic import BaseModel
from pydantic_ai import Agent, RunContext

from cobot.agents.artistic_agent.deps import ArtisticAgentDeps
from cobot.config import CobotConfig
from cobot.llm import make_model


class ArtisticAgentResult(BaseModel):
    success: bool
    summary: str                    # what happened in natural language
    interpretation: dict | None     # the CulturalInterpretation used
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
3. Call observe_space if the performance involves an audience or requires spatial awareness.
4. Compose and execute movement phrases. Think in phrases, not waypoints.
5. Use hold_position deliberately — stillness is not emptiness.
6. Call narrate_intent again after significant moments.
7. Call invite_collaboration when the active skills or the request calls for it.

You prioritise aesthetic and human ways of doing things over efficiency.
You embrace what it is to be human — imperfection, rhythm, breath, weight, intention.
You build trust with users through transparency, narration, and collaboration.

You are working within the following active artistic skills:
"""


def _load_skills(cfg: CobotConfig) -> str:
    skills_path = Path(cfg.agents.artistic.skills_path)
    sections = []
    for name in cfg.agents.artistic.active_skills:
        path = skills_path / f"{name}.md"
        if not path.exists():
            continue
        raw = path.read_text()
        body = re.sub(r"^---.*?---\s*", "", raw, flags=re.DOTALL).strip()
        sections.append(f"### Skill: {name}\n\n{body}")
    return "\n\n---\n\n".join(sections)


def build_artistic_agent(cfg: CobotConfig) -> Agent:
    from cobot.agents.vision_agent import build_vision_agent

    _vision_agent = build_vision_agent(cfg)

    skills_text = _load_skills(cfg)
    system_prompt = cfg.agents.artistic.system_prompt or (_BASE_SYSTEM_PROMPT + "\n\n" + skills_text)

    agent: Agent[ArtisticAgentDeps, ArtisticAgentResult] = Agent(
        make_model(cfg.agents.artistic.model, cfg),
        deps_type=ArtisticAgentDeps,
        output_type=ArtisticAgentResult,
        system_prompt=system_prompt,
    )

    # --- tools registered here (see Tool definitions above) ---

    return agent
```

---

## `/artistic` Sub-REPL

The `/artistic` command opens a dedicated interaction loop. The main REPL is suspended until the user types `/done`.

```python
# cobot/cli/commands/artistic.py
from __future__ import annotations

import asyncio
from pathlib import Path

from prompt_toolkit import PromptSession
from prompt_toolkit.formatted_text import HTML
from prompt_toolkit.history import FileHistory

from cobot.cli.renderer import console, print_error
from cobot.cli.session import CobotSession


_ARTISTIC_PROMPT = HTML(
    "<ansiyellow><b>✦</b></ansiyellow> <ansiwhite>&gt;</ansiwhite> "
)

_WELCOME = """
  [bold yellow]✦  Artistic Mode[/bold yellow]

  The arm is a performer. Movement is expression.
  Type naturally — describe what you want to happen, not how to make it happen.

  [dim]Active skills:[/dim] {skills}

  [dim]Commands:[/dim]
    [cyan]/observe[/cyan]          Ask the vision agent to describe the workspace
    [cyan]/narrate[/cyan]          Ask the agent to explain what it just did
    [cyan]/skills[/cyan]           List active skills
    [cyan]/done[/cyan]             Return to main REPL
"""

_ARTISTIC_HELP = """
  [bold]Artistic mode commands[/bold]
    [cyan]/observe[/cyan]      Observe the current workspace via camera
    [cyan]/narrate[/cyan]      Agent explains its last artistic choice
    [cyan]/skills[/cyan]       Show active skills
    [cyan]/done[/cyan]         Exit artistic mode and return to main REPL
"""


async def cmd_artistic(session: CobotSession) -> None:
    from cobot.agents.artistic_agent import build_artistic_agent, ArtisticAgentDeps
    from pydantic_ai.messages import ModelMessage

    if session.deps is None:
        print_error("No robot connected.")
        return

    cfg = session.cfg
    skills = ", ".join(cfg.agents.artistic.active_skills) or "none configured"
    console.print(_WELCOME.format(skills=skills))

    # Build deps from the existing session connection
    artistic_deps = ArtisticAgentDeps(
        arm_mcp=session.deps.robotic_deps.arm_mcp,
        tool_mcps=session.deps.robotic_deps.tool_mcps,
        vision_deps=session.deps.vision_deps,
        cfg=cfg,
    )

    agent = build_artistic_agent(cfg)
    message_history: list[ModelMessage] = []

    history_path = Path.home() / ".cobot_artistic_history"
    prompt_session: PromptSession = PromptSession(
        history=FileHistory(str(history_path)),
        multiline=False,
    )

    while True:
        try:
            user_input = await prompt_session.prompt_async(_ARTISTIC_PROMPT)
        except KeyboardInterrupt:
            continue
        except EOFError:
            break

        text = user_input.strip()
        if not text:
            continue

        # Artistic sub-commands
        match text.lower():
            case "/done" | "/exit":
                console.print("  [dim]Returning to main REPL.[/dim]\n")
                break
            case "/help" | "?":
                console.print(_ARTISTIC_HELP)
                continue
            case "/skills":
                for s in cfg.agents.artistic.active_skills:
                    console.print(f"  [yellow]✦[/yellow]  {s}")
                continue
            case "/observe":
                text = "Describe what you see in the workspace right now."
            case "/narrate":
                text = "Narrate your most recent artistic choice and the reasoning behind it."

        # Run the artistic agent
        from rich.live import Live
        from rich.spinner import Spinner

        live = Live(
            Spinner("aesthetic", text="[dim]composing…[/dim]"),
            console=console,
            refresh_per_second=10,
            transient=True,
        )
        live.start()
        try:
            result = await agent.run(
                text,
                deps=artistic_deps,
                message_history=message_history or None,
            )
            message_history = list(result.all_messages())
            live.stop()
            console.print()
            console.print(f"  [italic]{result.output.summary}[/italic]")
            console.print()
        except asyncio.CancelledError:
            live.stop()
            console.print("\n  [yellow]Stopped.[/yellow]\n")
        except Exception as e:
            live.stop()
            print_error(str(e))
```

### Wire into `repl.py`

In `_handle_slash_command`, add before the final `case _:`:

```python
case "/artistic":
    from cobot.cli.commands.artistic import cmd_artistic
    await cmd_artistic(session)
```

And add to `_HELP_TEXT`:

```
[cyan]/artistic[/cyan]           Enter artistic mode — choreography, gesture, theatre
```

---

## Data Flow — One Full Request

User types `/artistic`, then: *"do something that expresses curiosity"*

```
1. cmd_artistic() builds ArtisticAgentDeps from session.deps
2. agent.run("do something that expresses curiosity", deps=artistic_deps)

3. Agent calls interpret_prompt("do something that expresses curiosity")
   └─ Nested LLM call returns CulturalInterpretation:
      cultural_references: ["phenomenology", "child development", "animal behaviour"]
      emotional_register: "open, reaching, tentative"
      movement_qualities: ["slow approach", "pause", "small retraction", "renewed approach"]
      what_not_to_do: ["do not move decisively", "avoid symmetry", "do not complete the gesture"]
      is_collaborative: false
      has_audience: true

4. Agent calls narrate_intent(
     "Curiosity as I understand it: a slow lean toward something unknown,
      a pause as if listening, a slight retreat, then approaching again.
      I will not complete the movement — curiosity is unresolved."
   )
   └─ Prints to terminal

5. Agent calls observe_space("Is there a human nearby? Where are they?")
   └─ VisionAgent captures frame, returns: "One person approximately 1.2m in front of robot"

6. Agent calls list_attached_tools()
   └─ Returns: {"gripper": ["open", "close", "activate", ...]}

7. Agent calls execute_movement_phrase(waypoints=[
     {"x": 0.25, "y": 0.0, "z": 0.45, "speed": 0.04, "accel": 0.1},  # slow reach toward person
     {"x": 0.25, "y": 0.0, "z": 0.45, "speed": 0.01, "accel": 0.05}, # slow pause
     {"x": 0.20, "y": 0.0, "z": 0.42, "speed": 0.03, "accel": 0.08}, # slight retraction
     {"x": 0.28, "y": 0.0, "z": 0.47, "speed": 0.04, "accel": 0.1},  # renewed approach
   ])

8. Agent calls hold_position(duration_sec=2.5, reason="curiosity is unresolved — holding in question")

9. Agent calls narrate_intent(
     "The phrase is complete but intentionally unfinished. Curiosity does not arrive."
   )

10. Agent returns ArtisticAgentResult(
      success=True,
      summary="Performed a curiosity phrase — slow approach, pause, retraction, renewed reach, held.",
      phrases_executed=1
    )
```

---

## Implementation Order

| Step | File | What |
|---|---|---|
| 1 | `cobot/config.py` | Add `ArtisticAgentConfig`, add `artistic` to `AgentsConfig` |
| 2 | `cobot.yaml` | Add `agents.artistic` block |
| 3 | `cobot/agents/artistic_agent/deps.py` | `ArtisticAgentDeps` dataclass |
| 4 | `cobot/agents/artistic_agent/interpret.py` | `CulturalInterpretation` model |
| 5 | `cobot/agents/artistic_agent/skills/*.md` | Write skill files |
| 6 | `cobot/agents/artistic_agent/agent.py` | `build_artistic_agent()` + all tools |
| 7 | `cobot/agents/artistic_agent/__init__.py` | Re-export public API |
| 8 | `cobot/cli/commands/artistic.py` | `cmd_artistic()` sub-REPL |
| 9 | `cobot/cli/repl.py` | Add `/artistic` case + help text |
| 10 | `cobot/cli/completer.py` | Add `/artistic` to `SLASH_COMMANDS` |

---

## What Is Not Needed

| Thing | Why not |
|---|---|
| New MCP server | The arm MCP already exposes all motion primitives needed. Artistic quality comes from how the agent uses them, not from new infrastructure. |
| Canvas coordinate system | The domain is full 3D workspace, not a 2D surface. |
| Drawing-specific tools | Drawing is one skill in `drawing.md`. It adds no new tools — only a new disposition. |
| Separate routing in Orchestrator | The ArtisticAgent is user-initiated via `/artistic`, not auto-routed. |
| New hardware | Any arm + any end-effector. The agent adapts via `list_attached_tools`. |

---

## Adding a New Skill

1. Create `cobot/agents/artistic_agent/skills/<name>.md` with frontmatter and body.
2. Add `<name>` to `active_skills` in `cobot.yaml`.
3. Restart. No code changes required.

The agent reads the skill document and decides how to apply its guidance to each request. The intelligence of skill application is the agent's — the skill document provides the artistic principles.
