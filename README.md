# cobot-claw

Agentic CLI for controlling UR robots via natural language — powered by local or cloud LLMs, FastMCP tool servers, and a Claude Code-style terminal interface.

---

## Preview

![cobot-claw welcome screen](docs/claw_welcome.png)

---

## Overview

cobot-claw lets you command UR robot arms and end-effector tools (grippers, vacuum pumps, and more) through plain English — no programming required. Just type what you want the robot to do and the system figures out how to do it.

Under the hood, cobot-claw is a **multi-agent AI system**. Think of it as a small team of specialists, each responsible for a different part of the job:

- **Orchestrator** — the manager. Reads your request and decides which specialist to hand it to.
- **Scenario Agent** — the operator. Translates your instruction into precise robot movements and tool actions, then executes them step by step.
- **Safety Monitor** — the supervisor. Watches the robot continuously and intervenes if anything looks dangerous.
- **Config Agent** — the technician. Answers questions from robot manuals and handles setup tasks *(coming in Phase 6)*.

Each agent is powered by a large language model (LLM). cobot-claw works with **local models** running on your own hardware via [Ollama](https://ollama.com) — keeping your data private — as well as **cloud APIs** (OpenAI, Anthropic) when you need more capability. Switching between them is a one-line change in the config file.

The robot hardware is connected through lightweight **MCP servers** — small network services that expose the arm and tools as callable actions. The CLI works even when the robot is offline: the AI still responds, and you can `/reconnect` once the hardware is ready.

---

## Requirements

- Python 3.11+
- A UR robot accessible on the network (or just the gripper MCP in stub mode)
- One of:
  - [Ollama](https://ollama.com) or vLLM for local inference
  - OpenAI API key
  - Anthropic API key

---

## Installation

```bash
git clone <repo>
cd cobot-claw

# Core install
pip install -e .

# With RAG support (Phase 6+)
pip install -e ".[rag]"

# With dev/test tools
pip install -e ".[dev]"
```

---

## Configuration

Copy the example and edit before first run:

```bash
cp cobot.yaml.example cobot.yaml
```

```yaml
llm:
  # provider options: local (Ollama) | openai | anthropic
  provider: local
  base_url: http://localhost:11434/v1   # only used for provider: local
  api_key: ""                           # or set OPENAI_API_KEY / ANTHROPIC_API_KEY

fleet:
  - name: ur3e
    model_type: ur3e
    ip: 192.168.1.2
    arm_mcp_url: http://localhost:8765/mcp
    gripper_mcp_url: http://localhost:8001/mcp   # omit if no end-effector

agents:
  orchestrator:
    model: qwen2.5:72b      # or gpt-4o, claude-sonnet-4-6, etc.
  scenario:
    model: qwen2.5:14b
  config:
    model: qwen2.5:14b
  safety:
    model: qwen2.5:7b
```

### Switching to OpenAI or Anthropic

```yaml
# OpenAI
llm:
  provider: openai
  api_key: sk-...      # or export OPENAI_API_KEY
agents:
  orchestrator:
    model: gpt-4o

# Anthropic
llm:
  provider: anthropic
  api_key: sk-ant-...  # or export ANTHROPIC_API_KEY
agents:
  orchestrator:
    model: claude-sonnet-4-6
  safety:
    model: claude-haiku-4-5-20251001
```

Full schema reference: [cobot/config.py](cobot/config.py)

---

## CLI

```bash
cobot-claw                        # start interactive REPL
cobot-claw --robot ur5e           # select a specific robot from fleet
cobot-claw status                 # print robot status and exit
cobot-claw run pick_place         # execute a scenario directly
cobot-claw run pick_place --dry-run  # preview steps without moving
```

### REPL Slash Commands

| Command | Description |
|---|---|
| `/status` | Robot arm state, joint state, motion info |
| `/scenarios` | List saved scenarios |
| `/run <name>` | Execute a scenario |
| `/run <name> --dry` | Preview scenario steps without moving |
| `/reconnect` | Reconnect to robot MCP servers |
| `/clear` | Clear the screen |
| `/help` or `?` | Show all commands |
| `/exit` | Quit |

### Natural Language Examples

```
> move to home position
> run the pick and place sequence
> is the gripper working?
> what is the current robot state?
> open the gripper slowly
```

The CLI shows an animated spinner while waiting for the LLM and while tool calls execute. Tool calls and results stream inline as the agent works.

### Disconnected Mode

The CLI starts even if MCP servers are unreachable. Natural language commands still work — tool calls return a clean offline message so the LLM can respond meaningfully. Use `/reconnect` to reconnect once the hardware is ready.

---

## Project Structure

```
cobot-claw/
├── cobot.yaml                        # Fleet, LLM, agent config
├── pyproject.toml
├── docs/
│   └── claw_welcome.png              # Welcome screen preview
├── cobot/
│   ├── config.py                     # Pydantic config schema + loader
│   ├── llm.py                        # LLM client factory (local / OpenAI / Anthropic)
│   ├── logging.py                    # structlog setup
│   ├── mcp/
│   │   ├── client.py                 # Async MCPClient (streamable-http)
│   │   └── perception_contract.py    # PerceptionClient stub
│   ├── scenarios/
│   │   ├── schema.py                 # Pydantic step types + discriminated union
│   │   ├── executor.py               # Deterministic scenario executor
│   │   └── store/
│   │       └── pick_place.yaml       # Example pick-and-place scenario
│   ├── agents/
│   │   ├── deps.py                   # ScenarioAgentDeps, ConfigAgentDeps
│   │   ├── scenario.py               # Scenario Agent (dynamic tool discovery)
│   │   ├── config_agent.py           # Config Agent (RAG + teach mode)
│   │   └── orchestrator.py           # Orchestrator Agent + OrchestratorDeps
│   ├── cli/
│   │   ├── app.py                    # Typer entry point (cobot-claw command)
│   │   ├── repl.py                   # Interactive REPL loop + agent streaming
│   │   ├── renderer.py               # Rich welcome screen, tool call display
│   │   ├── session.py                # CobotSession dataclass
│   │   ├── completer.py              # Tab completion for slash commands + scenarios
│   │   └── commands/
│   │       ├── status.py             # /status command
│   │       └── scenario.py           # /run, /scenarios commands
│   └── rag/
│       └── pipeline.py               # RAGPipeline stub (Phase 6 replaces this)
├── tools/
│   ├── ure3/
│   │   └── ur3e_arm_mcp.py           # UR3e arm MCP server (FastMCP + ROS2)
│   └── gripper/
│       ├── server.py                 # Robotiq gripper MCP server
│       └── robotiq_driver.py
├── tests/                            # pytest suite
└── scripts/
    ├── check_phase1.py
    └── check_phase2.py
```

---

## MCP Servers

Both servers are standalone — start them independently of the CLI.

### Arm MCP (requires ROS 2 + UR driver)

```bash
# On Linux / WSL2
source /opt/ros/jazzy/setup.bash
export ROS_DOMAIN_ID=42
python tools/ure3/ur3e_arm_mcp.py
# Listening at http://0.0.0.0:8765/mcp
```

Key tools: `get_robot_status`, `get_joint_states`, `move_joints`, `move_named_pose`, `move_tcp_pose`, `hold_position`, `recover_external_control`

### Gripper MCP (Robotiq)

```bash
ROBOT_IP=192.168.1.2 python tools/gripper/server.py
# Listening at http://0.0.0.0:8001/mcp
```

Key tools: `activate`, `open`, `close`, `move`, `stop`, `get_status`, `set_speed`, `set_force`

The Scenario Agent discovers these actions automatically via `list_tool_mcps` — replace or add any end-effector MCP without touching agent code.

---

## Scenarios

Predefined scenarios are YAML files in `cobot/scenarios/store/`. The executor runs them deterministically — no LLM is invoked during execution.

```yaml
# cobot/scenarios/store/pick_place.yaml
name: pick-place-box
robot: ur3e
steps:
  - type: move_named_pose
    pose: home
  - type: move_tcp_pose
    x: 0.30
    y: 0.10
    z: 0.20
    perception:
      locate: "red_box"
    fallback_position:
      x: 0.30
      y: 0.10
      z: 0.20
  - type: tool_action
    tool: gripper
    action: close
    params: {speed: 50, force: 60}
    arm_recovery_sec: 6.0         # wait for UR External Control to restore
  - type: move_named_pose
    pose: home
```

Supported step types: `move_named_pose`, `move_joints`, `move_tcp_pose`, `tool_action`, `wait`, `check_motion_complete`, `repeat`.

`arm_recovery_sec` on `tool_action` steps is critical when using a Robotiq gripper — the gripper script interrupts UR External Control, and the executor polls `get_robot_status` until `ros_ready=True` before continuing.

---

## Verification

```bash
# Phase 1 — config, schema, LLM client (no robot needed)
python scripts/check_phase1.py

# Phase 2 — MCP client + perception stub (no robot needed)
python scripts/check_phase2.py

# Phase 2 with live MCP servers
python scripts/check_phase2.py --live

# Full test suite
pytest
pytest -v
```

---

## Implementation Phases

| Phase | Deliverable | Status |
|---|---|---|
| 1 | Config schema, LLM client, logging | **Done** |
| 2 | MCP client wrapper, perception contract stub | **Done** |
| 3 | Scenario schema, executor, example YAML, RAG stub | **Done** |
| 4 | Scenario Agent + Config Agent + Orchestrator Agent | **Done** |
| 5 | CLI — REPL, slash commands, streaming output | **Done** |
| 6 | Config Agent RAG + `/teach` (real ChromaDB/LlamaIndex) | Later |
| 7 | Safety Monitor | Future |
| 8 | Perception Agent + sensor MCPs | Future |

Full implementation details: [IMPLEMENTATION_PLAN.md](IMPLEMENTATION_PLAN.md)

---

## Tech Stack

| Layer | Library |
|---|---|
| Agent framework | [PydanticAI](https://ai.pydantic.dev) |
| LLM providers | OpenAI SDK (local/OpenAI/Anthropic via provider config) |
| MCP servers | [FastMCP](https://github.com/jlowin/fastmcp) |
| Config schema | Pydantic v2 |
| Scenario schema | Pydantic v2 discriminated union |
| CLI | Typer + Rich + prompt_toolkit |
| RAG (Phase 6) | LlamaIndex + ChromaDB |
| Logging | structlog |
| Tests | pytest + pytest-asyncio |
