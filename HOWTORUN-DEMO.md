# How to run the demo

Run cobot-claw against a simulated UR3e (URSim) with a 3D view, and pick a
scenario: pick and place, quality sorting, or packing. No real robot needed.

For more detail on the setup, see
[Running against URSim](README.md#running-against-ursim-no-real-robot) in the README.

## 1. One-time setup

You need Docker, ROS 2 Jazzy with `ur_robot_driver`, and the Robotiq gripper
model in `~/robotiq-gripper/install` (for the 3D view).

From the repo folder:

```bash
python3 -m venv .venv
.venv/bin/pip install -c requirements.lock.txt -e ".[obs,dev,voice]"
cp cobot.example.yaml cobot.yaml
```

In `cobot.yaml`, set your API key and these settings:

```yaml
fleet:
  - name: ur3e
    model_type: ur3e
    ip: 192.168.56.101                       # URSim
    arm_mcp_url: http://localhost:8765/mcp
    gripper_mcp_url: http://localhost:8001/mcp

agents:
  vision:
    detector: sim        # objects come from the simulated scene, not a camera
    ik_solver: local

voice:
  enabled: true          # optional: speak commands
```

The first start downloads the URSim image and, with voice on, the speech
models (about 0.6 GB). Later starts take under a minute.

## 2. Start the demo

Pick a scenario and run its command from the repo folder:

| Scenario | Command |
|---|---|
| Pick and place (blocks, a cylinder and a bin) | `scripts/start_ursim_demo.sh --rviz` |
| Quality sorting | `scripts/start_ursim_demo.sh --rviz --scene quality_sorting` |
| Packing | `scripts/start_ursim_demo.sh --rviz --scene packing` |

This starts the simulator, the gripper, the arm and scene servers, the 3D
view (RViz), and then the cobot-claw prompt in your terminal.

- **3D view:** RViz opens by itself and shows the arm, the gripper and the
  scene. Leave out `--rviz` if you don't want it.
- **Teach pendant** (optional): open
  <http://192.168.56.101:6080/vnc.html?autoconnect=true&resize=scale> in a browser.

To change scenario, stop the demo (below) and start it again with another
`--scene`.

### With a local model (Gemma 4) instead of Anthropic

`cobot-local.yaml` runs every agent on a Gemma 4 model served by vLLM at
`http://172.31.0.105:8000/v1`. It needs no API key. Add `-c cobot-local.yaml`
to any start command:

```bash
scripts/start_ursim_demo.sh --rviz --scene quality_sorting -c cobot-local.yaml
```

Before starting, check that the model server is up. This should list `gemma4`:

```bash
curl -s http://172.31.0.105:8000/v1/models
```

In `cobot-local.yaml`:

- `llm.base_url` is the server address, and `model: gemma4` is the model
  name it serves. Change these to use another vLLM or Ollama server.
- `llm.extra_body` turns Gemma's thinking off, so each reply takes about 1 s
  instead of about 5 s. Delete it to let the model think.
- The rest (robot, scene, voice) is the same as `cobot.yaml`.

The vLLM server needs tool calling turned on (`--enable-auto-tool-choice`
with a tool-call parser) and a reasoning parser, or the model's thinking
shows up in its answers.

## 3. What to say

Type at the `>` prompt and press Enter, or speak (see Voice below).

**Pick and place**
- "What objects are on the table?"
- "Pick up the red block and put it in the grey bin"
- "Point to the blue cylinder"
- "Go home"

**Quality sorting:** six parts in a blue tray, two of them defective.
- "Sort the parts in the incoming tray": inspects each part and puts it in
  the green pass bin or the red reject bin. Takes about 2.5 minutes.
- "Inspect part 2", "Put part 4 in the pass bin"
- Inspected parts get a green PASS or red FAIL label in RViz.

**Packing:** six product packs and two customer orders.
- "What orders are open?"
- "Pack the orders": the agent chooses the packs for each order and packs
  them into the two shipping boxes. Takes about 2 minutes.
- "Pack a red pack and a blue pack into shipping box 2"

**Any scenario**
- "Reset the scene": puts every object back where it started.

### While the robot is working

The prompt stays open during a task. Each step prints as it happens
(✓ part 1: PASS, ✗ part 2: FAIL, crack across the top face, → part 2 sorted
into reject bin, …), and the bottom line shows what the agent is doing.

- **Ask about it at any time**: "what's wrong with the part you just put
  away?", "how many are done?", "what's in shipping box 1?". A separate
  observer agent answers from what the robot has done so far. It can look
  but not act, so asking never disturbs the task.
- **Asking it to do something else** while a task runs ("pick up the red
  block") gets a reply that it's busy. Wait for the task, or cancel it.
- **Cancel**: type `stop`, press **Esc** or **Ctrl+C**, or say "stop". The
  task is cancelled and the arm brakes.
- **Afterwards**, the main agent can answer from the same record: "why was
  part 5 rejected?", "which parts passed?".

## 4. Voice

With `voice.enabled: true`:

1. At the prompt, press **Ctrl+Space** (or F9). A red **● listening** label appears.
2. Speak the command.
3. Either **pause**: the text appears in the prompt to check, then press Enter;
   or press **Ctrl+Space again**: the command is sent at once.

While the arm is moving, say **"stop"** to stop it. This is a convenience,
not a safety stop.

If F9 does nothing, your laptop's F-keys are sending media keys: use
Ctrl+Space, or press Fn+Esc.

## 5. Stop the demo

Type `/exit` at the prompt (or press Ctrl+D). That stops everything except
the URSim container, which keeps running so the next start is faster. To
stop it too:

```bash
docker stop ursim
```

If a start fails with "Port … already in use", an earlier run is still up:

```bash
pkill -INT -f start_ursim_demo.sh; pkill -f robot_state_publisher
```

## Running cobot-claw on another computer

Start only the simulator part on this machine:

```bash
scripts/start_ursim_demo.sh --rviz --no-app --scene packing
```

On the other computer, use the same `cobot.yaml` settings but with this
machine's IP address instead of `localhost` (for example
`http://172.31.0.77:8765/mcp`), and add
`sim_scene_url: http://<this machine's IP>:8010/mcp` under `agents.vision`.
Then run:

```bash
.venv/bin/cobot-claw -c cobot.yaml -r ur3e
```

## Troubleshooting

| Problem | Fix |
|---|---|
| "Robot 'ur3e' not found in fleet" | The name after `-r` must match `fleet: - name:` in `cobot.yaml`. |
| The arm stops responding | Stop the demo and start it again. |
| An object isn't where the agent expects | Say "reset the scene". |
| No 🎙 line at the bottom of the prompt | Check `voice.enabled: true` and that the voice extra is installed (`pip install -e ".[voice]"`). |
| Logs | `logs/ursim_*.log` and `logs/rviz.log` in the repo folder. |
