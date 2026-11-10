# cobot-claw

[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](LICENSE)

Agentic CLI for controlling UR robots via natural language — powered by local or cloud LLMs, FastMCP tool servers, and a Claude Code-style terminal interface.

---

## Preview

![cobot-claw welcome screen](docs/claw_welcome.png)

---

## Architecture

![cobot-claw high-level architecture](docs/cobot-claw-architecture.png)

> Full draw.io source: [docs/architecture.drawio](docs/architecture.drawio)

---

## Overview

cobot-claw lets you command UR robot arms and end-effector tools (grippers, vacuum pumps, and more) through plain English — no programming required. Just type what you want the robot to do and the system figures out how to do it.

Under the hood, cobot-claw is a **multi-agent AI system**. Think of it as a small team of specialists, each responsible for a different part of the job:

- **Orchestrator** — the manager. Reads your request and decides which specialist to hand it to. Never calls robot hardware directly.
- **Robotic Agent** — the operator. Handles all robot interactions — status queries, movements, tool control, and scenario execution. Discovers attached tools dynamically so no code changes are needed when hardware changes.
- **Vision Agent** — the eyes. Uses an Intel RealSense camera and a vision LLM (Gemini ER or any OpenAI-compatible vision endpoint) to locate objects in 3D space, compute inverse kinematics, and execute pick, pick-and-place, and point-to operations directly — no scenario YAML needed.
- **Safety Monitor** — the supervisor. Watches the robot continuously and intervenes if anything looks dangerous.
- **Config Agent** — the technician. Answers questions from robot manuals and handles teach-by-demo scenario recording.
- **Artistic Agent** — the performer. Treats robot arm movement as artistic expression — choreography, gesture, theatre, ritual, improvisation, and installation. Invoked via `/artistic`; never auto-routed by the Orchestrator.

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
  config:
    model: qwen2.5:14b
  safety:
    model: qwen2.5:7b
  vision:
    model: gemini-robotics-er-1.6-preview
    detector: gemini              # "gemini" | "openai_vision"
    google_api_key: ""            # or set GOOGLE_API_KEY env var
    ik_solver: robot              # "robot" = use arm MCP IK | "local" = analytical
    pick_x_offset_m: 0.0          # tune to correct systematic grasp X error
    pick_y_offset_m: 0.0          # tune to correct systematic grasp Y error
  artistic:
    model: claude-sonnet-4-6
    active_skills:               # any subset of the available skills
      - choreography
      - gesture
    skills_path: ./cobot/agents/artistic_agent/skills
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

## Running

### One-command startup (recommended)

`start.sh` launches the UR ROS 2 driver, both MCP servers, and cobot-claw in the correct order with a single command:

```bash
chmod +x start.sh

./start.sh                    # uses cobot.yaml by default
./start.sh cobot.local.yaml   # pass a specific config file
```

What it does:
1. Starts the UR ROS 2 driver in the background (waits 8 s for it to come up)
2. Starts the arm MCP server in the background (ROS sourced, `ROS_DOMAIN_ID=42`)
3. Starts the gripper MCP server in the background
4. Waits 4 s for MCP servers to be ready
5. Launches `cobot-claw` in the foreground

Background process logs are written to `ur_driver.log`, `arm_mcp.log`, and `gripper_mcp.log` in the repo root. `Ctrl+C` shuts everything down cleanly.

### Manual startup

If you prefer separate terminals:

**Terminal 1 — UR ROS 2 driver:**
```bash
source /opt/ros/jazzy/setup.bash
ros2 launch ur_robot_driver ur_control.launch.py ur_type:=ur3e robot_ip:=192.168.1.2 headless_mode:=true
```

**Terminal 2 — Arm MCP server:**
```bash
source /opt/ros/jazzy/setup.bash
export ROS_DOMAIN_ID=42
python tools/ure3/ure3_arm_mcp.py
# Listening at http://0.0.0.0:8765/mcp
```

**Terminal 3 — Gripper MCP server:**
```bash
ROBOT_IP=192.168.1.2 python tools/gripper/server.py
# Listening at http://0.0.0.0:8001/mcp
```

**Terminal 4 — cobot-claw:**
```bash
cobot-claw --config cobot.yaml
```

### CLI flags

```bash
cobot-claw                            # start interactive REPL (uses cobot.yaml)
cobot-claw --config cobot.local.yaml  # use a specific config file
cobot-claw --robot ur5e               # select a specific robot from fleet
cobot-claw status                     # print robot status and exit
cobot-claw run pick_place             # execute a scenario directly
cobot-claw run pick_place --dry-run   # preview steps without moving
```

### Running against URSim (no real robot)

> **Just want to run the demo?** See [HOWTORUN-DEMO.md](HOWTORUN-DEMO.md) for short step-by-step instructions and how to load each scenario.

You can run the whole stack against a simulated UR3e in Docker. You need Docker, ROS 2 Jazzy with `ur_robot_driver`, and a `cobot.yaml` whose robot `ip` is `192.168.56.101`.

```bash
python3 -m venv .venv
.venv/bin/pip install -c requirements.lock.txt -e ".[obs,dev]"   # pinned: newer anthropic/mcp/openai majors break it
cp cobot.example.yaml cobot.yaml   # set ip: 192.168.56.101 and your API key
scripts/start_ursim_demo.sh        # add --no-app to start only the sim stack, --rviz for RViz
```

The script starts URSim and powers on the arm. It then starts a Robotiq gripper emulator, the ROS 2 driver, the arm and gripper MCP servers, the simulated scene server, RViz (with `--rviz`) and cobot-claw. Exiting cobot-claw or pressing Ctrl+C stops everything except the URSim container. To start the parts one by one instead, see [Starting each part manually](#starting-each-part-manually).

To watch the simulated robot, open the teach pendant at
<http://192.168.56.101:6080/vnc.html?autoconnect=true&resize=scale> and choose the **Move** tab.
The pendant only draws the bare arm. To see the gripper as well, start with `--rviz`, or run `tools/viz/start_rviz.sh` while the stack is up. This opens RViz with the arm following URSim and a Robotiq 2F-85 model whose fingers follow the gripper emulator. It needs the `robotiq_description` package, which it looks for in `~/robotiq-gripper/install` (override with `ROBOTIQ_WS`). The cell's real gripper is a Hand-E, but the 2F-85 is the closest model available.

- URSim is pinned to 5.22.2. Newer images (5.26+) crash `ur_client_library` 2.6.
- URSim has no gripper hardware, and the Robotiq URCap can't detect one there. Instead, [robotiq_emulator.py](tools/gripper/robotiq_emulator.py) runs inside the container on the URCap's port (63352). `Gripper.script` and the gripper MCP server work unchanged against it. Don't also install the Robotiq URCap in the sim: it takes the same port.
- Watch the gripper with `docker exec ursim cat /tmp/robotiq_emulator.json`. To simulate grasping a part, start with `EMU_OBJECT_POS=200`: closing then stops at 200 and reports an object detected.
- As on the real robot, a gripper command replaces the arm's control program. In the sim, the gripper server restores it through the arm MCP server's `recover_external_control` before replying (`ARM_MCP_URL`). Avoid the pendant's Installation tab while the stack runs, because it also stops the program.
- Stop the simulator with `docker stop ursim`.

#### Starting each part manually

This is what `scripts/start_ursim_demo.sh` does, one terminal per part. It's useful for watching each part's output or restarting just one. If the script is already running, stop it first (`pkill -f start_ursim_demo.sh`), or the servers below fail with "address already in use".

Run this at the start of every new terminal (all but cobot-claw need ROS):

```bash
source /opt/ros/jazzy/setup.bash && export ROS_DOMAIN_ID=42 && cd /path/to/cobot-claw
```

1. **Simulator (URSim).** Start the container. Skip this if `docker ps` already lists `ursim`.

   ```bash
   ros2 run ur_client_library start_ursim.sh -m ur3e -v 5.22.2 -p ~/.ursim/e-series/ur3e-5.22/programs -d
   ```

   After about 30 s, power on the arm and release the brakes through the dashboard server. You can also use the button at the bottom left of the pendant at <http://192.168.56.101:6080/vnc.html>.

   ```bash
   for c in "close popup" "power on"; do (echo "$c"; sleep 1) | nc -w 3 192.168.56.101 29999 | tail -1; done; sleep 5; (echo "brake release"; sleep 1) | nc -w 3 192.168.56.101 29999 | tail -1
   ```

   It's ready when this prints `Robotmode: RUNNING`:

   ```bash
   (echo robotmode; sleep 1) | nc -w 3 192.168.56.101 29999 | tail -1
   ```

2. **Gripper emulator.** It runs inside the container, so repeat this after every simulator restart.

   ```bash
   docker cp tools/gripper/robotiq_emulator.py ursim:/tmp/robotiq_emulator.py && docker exec -d ursim sh -c 'python3 /tmp/robotiq_emulator.py > /tmp/robotiq_emulator.log 2>&1'
   ```

3. **ROS 2 driver** (terminal A, leave running). Wait for `Robot connected to reverse interface. Ready to receive control commands.`

   ```bash
   ros2 launch ur_robot_driver ur_control.launch.py ur_type:=ur3e robot_ip:=192.168.56.101 headless_mode:=true launch_rviz:=false
   ```

4. **Arm MCP server** (terminal B). `HOME_JOINTS` is the sim's home in front of the robot. Leave it out to use the real cell's home, which is behind the robot.

   ```bash
   cd tools/ure3 && ROBOT_IP=192.168.56.101 HOME_JOINTS=0.552,-1.248,-1.612,-1.853,1.571,3.693 python3 ur3e_arm_mcp.py
   ```

5. **Gripper MCP server** (terminal C). `ARM_MCP_URL` makes each gripper command restore arm control before returning (see the notes below).

   ```bash
   cd tools/gripper && ROBOT_IP=192.168.56.101 ARM_MCP_URL=http://127.0.0.1:8765/mcp ../../.venv/bin/python server.py
   ```

6. **Scene server** (terminal D). This provides the objects to pick when `detector: sim`.

   ```bash
   python3 tools/sim_scene/scene_server.py
   ```

7. **3D view** (terminal E). RViz shows the arm, gripper, table and objects. The simulator's own view is the **Move** tab at <http://192.168.56.101:6080/vnc.html?autoconnect=true&resize=scale>.

   ```bash
   tools/viz/start_rviz.sh
   ```

8. **cobot-claw** (terminal F, no ROS setup needed). Type a request, or press Ctrl+Space / F9 to speak one if voice is enabled.

   ```bash
   cd /path/to/cobot-claw && .venv/bin/cobot-claw -c cobot.yaml
   ```

Order matters:

- Start the driver (3) before the arm server (4).
- The scene server (6) needs the driver and the emulator (2).
- cobot-claw connects to the servers when it starts, so start it last, and restart it if you start a server later.

To stop, press Ctrl+C in each terminal in reverse order, then run `docker stop ursim` if you also want to stop the simulator. Ctrl+C on `ros2 launch` sometimes leaves `robot_state_publisher` running, which publishes stale arm positions. Clean it up with `pkill -f robot_state_publisher`.

#### Simulated objects: pick and place in the sim

URSim has no camera, so the stack also runs a scene server, [scene_server.py](tools/sim_scene/scene_server.py). It loads the objects and containers in [scene.yaml](tools/sim_scene/scene.yaml), draws them in the RViz view, and tracks grasps. An object attaches when the gripper closes on it and drops onto the table or into a container when the gripper opens. To use it, set this in `cobot.yaml`:

```yaml
agents:
  vision:
    detector: sim        # objects come from tools/sim_scene instead of the camera + Gemini
    ik_solver: local     # robot IK sends a script that stops arm control
```

Then ask, for example, "what objects are on the table?" or "pick up the blue cylinder and put it in the grey bin".

- Only detection is simulated. Objects are found by name, colour or shape from the scene file, and the moves follow the camera-based pick and place tools ([sim.py](cobot/agents/vision_agent/sim.py)). Camera detection and hand-eye calibration aren't exercised.
- There's no physics: objects don't tip, slide or collide, and the arm can pass through them.
- Edit `scene.yaml` to change the layout. To reload it, restart the stack or call the scene server's `reset_scene` tool.
- In the sim, the gripper server waits for each gripper script to finish and restores arm control through the arm MCP server (`ARM_MCP_URL`) before returning. Otherwise the next arm move races the gripper. Without `ARM_MCP_URL`, the server keeps its original behaviour.

#### Scenes: quality sorting cell

Other layouts live in [tools/sim_scene/scenes/](tools/sim_scene/scenes/). Load one at startup with `--scene`:

```bash
scripts/start_ursim_demo.sh --rviz --scene quality_sorting
```

**Quality sorting** ([quality_sorting.yaml](tools/sim_scene/scenes/quality_sorting.yaml)): six machined housings (part 1–6) sit in a blue incoming tray at the end of an infeed conveyor. An inspection camera checks each part, and the arm sorts it into the green pass bin or the red reject bin. Two parts are defective: a dark scratch shows on their top face.

- **"Sort the parts in the incoming tray"** runs the whole job: for each part, inspect, pick, place in the pass or reject bin. Parts are spread over the bin floor. The reply lists what passed and what was rejected, and why. It takes a few minutes for six parts.
- One step at a time: "inspect part 2", "put part 4 in the pass bin", "move part 5 to the reject bin".
- Each part's result is hidden until it is inspected, as on a real line. In RViz an inspected part's label turns green (PASS) or red (FAIL).
- **"Reset the scene"** opens the gripper, puts every part back in the tray and clears the results.
- The conveyor and camera are drawn for context only (`fixtures:` in the scene file): the arm can pass through them.
- Inspection, sorting and reset are vision-agent tools (`inspect_object`, `sort_parts`, `reset_scene`), available only with `detector: sim`.

#### Scenes: packing cell

```bash
scripts/start_ursim_demo.sh --rviz --scene packing
```

**Packing** ([packing.yaml](tools/sim_scene/scenes/packing.yaml)): six product packs (red, blue and yellow, two of each) arrive on an infeed conveyor into an accumulation tray. Two open shipping boxes wait on either side of the arm, one per customer order:

| Order | Ships in | Wants |
|---|---|---|
| order 1 | shipping box 1 | 1 red, 1 blue, 1 yellow pack |
| order 2 | shipping box 2 | 1 red, 1 blue pack |

- **"Pack the orders"**: the agent reads the open orders, chooses which packs fill each one, and packs them box by box. Packs sit side by side on the box floor. One yellow pack is left over.
- Also: "what orders are open?", "pack a red pack and a blue pack into shipping box 2", "fill order 1".
- Here the agent does the planning: the scene server reports each order with what's packed and what's missing, and the agent picks the products. A product already in a shipping box is never taken for another order.
- The conveyor, open box flaps and tape dispenser are drawn for context only.
- Packing is the `pack_items` vision-agent tool (sim only); orders are listed by `capture_scene`.

#### Your own scenes

Copy a file in `scenes/`. Objects can start inside a container: give a position inside it and the object rests on its floor. Add `quality: pass|fail` and `defect:` to make a part inspectable, and `orders:` (box and products) for packing.

#### Voice commands

cobot-claw can take spoken commands. Speech-to-text runs locally with [faster-whisper](https://github.com/SYSTRAN/faster-whisper) on the CPU, and the audio never leaves the machine. Install and enable it:

```bash
.venv/bin/pip install -c requirements.lock.txt -e ".[voice]"
```

```yaml
voice:
  enabled: true
```

- **At the prompt, press F9 or Ctrl+Space and speak.** A red **● listening** label appears at the right end of the prompt line. Then either:
  - **pause**: the text appears in the prompt for you to check; press Enter to send it, or edit it first; or
  - **press the key again**: recording stops and the command is sent at once, as if you'd pressed Enter. This also works while it says "transcribing".

  On laptops whose top row sends media keys (e.g. Dell with Fn Lock off), F9 never reaches the terminal: use Ctrl+Space, or toggle Fn Lock (Fn+Esc on Dells). Misheard words become arm motion, so sending without checking (the second key press, or `auto_send: true` for every command) is best kept for commands you've rehearsed. The status line at the bottom shows what's happening.
- **While a task runs, say "stop"** (or "halt", "freeze"). The task is cancelled and the arm brakes with the arm server's `stop_motion` tool (URScript `stopj`), without going through the agents. It takes about 1.5–3 s from the start of the word. This is a convenience, not a safety function: use the E-stop.
- **Models:** `distil-small.en` for commands and `base.en` for listening for "stop". `base.en` was too error-prone for commands: in tests it turned "Close the gripper" into "Place the gripper". `distil-small.en` was as accurate as `small.en` on the test commands and slightly faster. Both download once (~0.6 GB) to the Hugging Face cache.
- **Speed:** the text appears about 2.3 s after you stop speaking (it was 3.2 s), on this 12-core laptop CPU. A command ends after a 0.6 s pause (`voice.silence_sec`). Transcription starts early, at the first 0.25 s pause (`voice.early_start_sec`); if you keep talking that result is dropped and the whole command is transcribed. Most of the remaining time is the model itself: Whisper always processes a 30 s window, however short the command. Thread count, the voice filter and the vocabulary hint made no measurable difference.
- **Microphone:** audio comes from `arecord` (ALSA/PipeWire), so no PortAudio is needed. The mic stays open while cobot-claw runs with voice enabled: from cold, this laptop's mic gives about 0.5 s of full-scale noise before it settles. Audio is only kept and transcribed after F9, or during a task for the stop listener. Set `voice.device` to choose a microphone (`arecord -L`), and `voice.min_level` for noisy rooms.
- **Vocabulary:** `voice.vocabulary` biases recognition toward the cell's words. Add your object names there.

#### Arm motion notes (sim and real robot)

These changes are in shared code, so they also affect the real robot:

- **IK picks natural arm configurations.** The local IK solver ranks its candidates mostly by shoulder, elbow and wrist 1–2, and counts base rotation and wrist-3 roll much less. Before this, about a quarter of reachable targets got contorted solutions (shoulder at or below horizontal, wrist folded back), which made the gripper sweep into the arm. Its final ±360° adjustment also respects joint limits now.
- **`move_named_pose` waits for the pose** (as `move_joints` does). Every arm or gripper command is a URScript program on port 30002 that replaces the running one. A "home" move that returned at once was cancelled by the next command, so picks started from wherever the arm happened to be.
- **Named poses take the shortest way round.** "Home" is reached with the ±360° equivalent nearest the current joints. If the base has been wound past about ±210°, it unwinds on the way home, with the arm raised (`SHORTEST_JOINT_PATH=0` to disable). Explicit joint targets are sent unchanged.
- **New `stop_motion` tool** on the arm server brakes the arm with URScript `stopj`. `hold_position` moves back to the joint angles it read while the arm was still moving: in a test it crept 26° back over 5 s after "stopping".
- **Sim home in front of the robot.** The real cell's home is behind the robot (base −181°), so every pick-and-place swung the base around. The launcher gives the arm server a sim home 0.35 m above the table at (0.25, 0, 0), facing the scene, through `HOME_JOINTS`. Set `SIM_HOME_JOINTS` to change it, or leave it empty to use the real home. In a recorded rotate-and-two-picks test, this cut joint travel by a third (2,120° to 1,430°) and time from 96 s to 64 s.

---

## CLI

### REPL Slash Commands

| Command | Description |
|---|---|
| `/status` | Robot arm state, joint state, motion info |
| `/scenarios` | List saved scenarios |
| `/run <name>` | Execute a scenario |
| `/run <name> --dry` | Preview scenario steps without moving |
| `/teach <name>` | Record arm movements and save as scenario |
| `/teach <name> --dur 60` | Teach mode with custom duration (seconds) |
| `/calibrate` | Run hand-eye calibration (moves arm through 4 poses, saves result) |
| `/calibrate --check` | Report current calibration error without saving |
| `/artistic` | Enter artistic mode — choreography, gesture, theatre, ritual |
| `/vision-debug [on\|off]` | Toggle live OpenCV camera window with detection overlays |
| `/reload-config` | Reload cobot.yaml and reconnect MCP servers |
| `/ingest [path]` | Index robot manuals into RAG knowledge base |
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

> pick the white block
> pick the bottle and place it on the brown box
> point to the red cup
> what do you see on the table?
```

The CLI shows an animated spinner while waiting for the LLM and while tool calls execute. Tool calls and results stream inline as the agent works.

### Disconnected Mode

The CLI starts even if MCP servers are unreachable. Natural language commands still work — tool calls return a clean offline message so the LLM can respond meaningfully. Use `/reconnect` to reconnect once the hardware is ready.

---

## MCP Servers

Both servers are standalone — `start.sh` manages them automatically, or start them independently.

### Arm MCP (requires ROS 2 + UR driver)

```bash
source /opt/ros/jazzy/setup.bash
export ROS_DOMAIN_ID=42
python tools/ure3/ure3_arm_mcp.py
# Listening at http://0.0.0.0:8765/mcp
```

Key tools: `get_robot_status`, `get_joint_states`, `move_joints`, `move_named_pose`, `move_tcp_pose`, `hold_position`, `recover_external_control`

### Gripper MCP (Robotiq)

```bash
ROBOT_IP=192.168.1.2 python tools/gripper/server.py
# Listening at http://0.0.0.0:8001/mcp
```

Key tools: `activate`, `open`, `close`, `move`, `stop`, `get_status`, `set_speed`, `set_force`

The gripper MCP communicates via URScript directly to TCP port 30002 — no ROS dependency. Sending URScript stops UR External Control; call `recover_external_control` on the arm MCP after any gripper command before issuing arm commands.

The Robotic Agent discovers these actions automatically via `list_tool_mcps` — replace or add any end-effector MCP without touching agent code.

---

## Vision and Hand-Eye Calibration

The Vision Agent uses an Intel RealSense camera (accessed directly via `pyrealsense2`) and a vision LLM for object detection. Before use, run hand-eye calibration to align the camera frame with the robot's TCP frame.

```
> /calibrate          # moves arm through 4 poses, computes and saves calibration
> /calibrate --check  # report reprojection error for the existing calibration
```

Calibration data is stored in `cobot/agents/vision_agent/calibration.json`. After calibration succeeds:

```
> pick the white block
> pick the bottle and place it on the brown box
> point to the red cup
> what do you see on the table?
```

Toggle the live camera debug window at any time:
```
> /vision-debug on
> /vision-debug off
```

---

## Artistic Agent

The Artistic Agent treats robot arm movement as artistic expression. It is invoked via `/artistic` and never auto-routed by the Orchestrator.

```
> /artistic
> perform a slow, contemplative greeting
> improvise a response to my presence
> perform a ritual of welcome
```

Skills define the agent's artistic disposition for a session. Available skills:

| Skill | Description |
|---|---|
| `choreography` | Structured movement sequences with compositional intent |
| `gesture` | Communicative movement directed at a person or object |
| `theatre` | Narrative arc, character, dramatic timing |
| `ritual` | Repetition, ceremony, symbolic action |
| `improvisation` | Emergent movement in response to the environment |
| `installation` | Continuous, ambient movement for unattended spaces |
| `collaborative` | Co-authorship — pauses to invite human contribution |

Configure active skills in `cobot.yaml`:

```yaml
agents:
  artistic:
    model: claude-sonnet-4-6
    active_skills:
      - choreography
      - gesture
      - improvisation
```

New skills require only a new `.md` file in `cobot/agents/artistic_agent/skills/` and a config entry — no code changes.

The Artistic Agent uses the Vision Agent (`observe_space`) to detect audience presence and respond to spatial context during improvisation and installation performances.

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

## Manual Q&A and Teach Mode

Ask natural language questions about your robot, or record arm movements directly from the REPL.

### Setup

```bash
pip install -e ".[rag]"

# Place PDF/TXT/MD manuals in ./manuals, then index them:
cobot-claw ingest --path ./manuals

# Or inside the REPL:
> /ingest ./manuals
```

### Manual Q&A

Once indexed, ask questions in plain English:

```
> what is the max payload for the ur3e?
> how do I set the TCP offset?
> what does error code 53 mean?
```

The Config Agent searches the indexed manuals and returns grounded answers.

### Teach Mode

Record waypoints while the robot arm is in **freedrive**:

```
> /teach my_motion
> /teach pick_sequence --dur 60
```

The REPL prompts you to enable freedrive on the pendant, then records joint states for the specified duration (default 30 s). The motion is automatically saved as a scenario YAML in `cobot/scenarios/store/` — ready to run with `/run my_motion`.

You can also ask conversationally:

```
> record a new picking motion called box_pick
> teach me a new scenario for 45 seconds
```

---

## Project Structure

```
cobot-claw/
├── cobot.yaml                        # Fleet, LLM, agent config
├── start.sh                          # One-command launcher (driver + MCPs + CLI)
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
│   │   └── store/                    # Saved scenario YAMLs
│   ├── agents/
│   │   ├── deps.py                   # Shared deps dataclasses
│   │   ├── robotic_agent.py          # Robotic Agent (dynamic tool discovery)
│   │   ├── config_agent.py           # Config Agent (RAG + teach mode)
│   │   ├── orchestrator.py           # Orchestrator Agent
│   │   ├── vision_agent/
│   │   │   ├── agent.py              # Vision Agent tools (pick, pick-and-place, point)
│   │   │   ├── camera.py             # RealSense capture + depth utilities
│   │   │   ├── calibrate.py          # Hand-eye calibration (4-pose routine)
│   │   │   ├── detector.py           # VisionDetector protocol + Gemini/OpenAI backends
│   │   │   ├── kinematics.py         # UR3e analytical IK + robot MCP IK with fallback
│   │   │   ├── scenario.py           # Vision-guided scenario YAML generator
│   │   │   └── calibration.json      # Saved hand-eye calibration
│   │   └── artistic_agent/
│   │       ├── agent.py              # Artistic Agent (movement as expression)
│   │       ├── deps.py               # ArtisticAgentDeps
│   │       ├── interpret.py          # Cultural/emotional interpretation sub-agent
│   │       └── skills/               # Skill definition .md files
│   │           ├── choreography.md
│   │           ├── gesture.md
│   │           ├── theatre.md
│   │           ├── ritual.md
│   │           ├── improvisation.md
│   │           ├── installation.md
│   │           └── collaborative.md
│   ├── cli/
│   │   ├── app.py                    # Typer entry point (cobot-claw command)
│   │   ├── repl.py                   # Interactive REPL loop + agent streaming
│   │   ├── renderer.py               # Rich welcome screen, tool call display
│   │   ├── session.py                # CobotSession dataclass
│   │   ├── completer.py              # Tab completion for slash commands + scenarios
│   │   └── commands/
│   │       ├── status.py             # /status command
│   │       ├── scenario.py           # /run, /scenarios commands
│   │       ├── calibrate.py          # /calibrate command
│   │       └── artistic.py           # /artistic command
│   └── rag/
│       └── pipeline.py               # RAGPipeline (ChromaDB + LlamaIndex)
├── tools/
│   ├── ure3/
│   │   └── ure3_arm_mcp.py           # UR3e arm MCP server (FastMCP + ROS2)
│   └── gripper/
│       ├── server.py                 # Robotiq gripper MCP server
│       ├── robotiq_driver.py         # Low-level URScript driver
│       └── Gripper.script            # URScript template
├── tests/                            # pytest suite
└── scripts/
    ├── check_phase1.py
    └── check_phase2.py
```

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
| 4 | Robotic Agent + Config Agent + Orchestrator Agent | **Done** |
| 5 | CLI — REPL, slash commands, streaming output | **Done** |
| 6 | Config Agent RAG + `/teach` (real ChromaDB/LlamaIndex) | **Done** |
| 7 | Vision Agent — RealSense + Gemini ER + analytical IK + pick/place | **Done** |
| 8 | Artistic Agent — movement as expression, skill system, cultural interpretation | **Done** |
| 9 | Safety Monitor | Future |

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
| RAG | LlamaIndex + ChromaDB |
| Vision / detection | Gemini Robotics ER · OpenAI-compatible vision endpoints |
| Depth sensing | Intel RealSense (pyrealsense2) |
| Robot kinematics | Closed-form analytical IK + UR controller MCP IK |
| Logging | structlog |
| Tests | pytest + pytest-asyncio |

---

## Demo

![Move and Home](demo/stages/stage_move_and_home.gif)

![Point to Object](demo/stages/point.gif)

![Pick and Drop](demo/stages/pick.gif)
