# gripper_mcp — Robotiq Hand-E Gripper MCP Server

MCP server for controlling the Robotiq Hand-E gripper attached to the UR3e. Exposes open, close, move, and status tools over MCP.

- **Transport:** Streamable HTTP
- **Endpoint:** `http://0.0.0.0:8001/mcp`

---

## How it communicates

The gripper is controlled by sending URScript directly to the robot over **TCP port 30002**. This approach is used because the Robotiq URCap does not support remote control — publishing via the ROS 2 `/urscript_interface/script_command` topic does not activate the gripper when operating remotely.

### Important: External Control interruption

Sending URScript to port 30002 replaces whatever program is currently running on the robot, which means **External Control (used by the arm MCP server) is stopped** after every gripper command.

The arm MCP server (`arm_mcp`) exposes a `recover_external_control` tool that restores External Control. Call this after any gripper operation before issuing arm commands, or instruct Claude to do so in your system prompt.

---

## Prerequisites

- Python 3.10+
- Robot reachable at `192.168.1.2`
- `Gripper.script` present in the same directory as `server.py`
- Robotiq Hand-E gripper physically connected to the UR3e tool flange and wired via the tool I/O

No ROS dependency — communicates directly with the robot over TCP.

---

## Installation

```bash
pip install fastmcp
```

---

## Running

```bash
cd UR3e/gripper_mcp
python3 server.py
```

Expected output: server starts silently and listens on port `8001`.

### Environment variables

| Variable | Default | Description |
|----------|---------|-------------|
| `ROBOT_IP` | `192.168.1.2` | Robot controller IP |
| `ROBOT_PORT` | `30002` | URScript primary port |
| `GRIPPER_SCRIPT` | `Gripper.script` | Path to the URScript template |

---

## Tools

| Tool | Parameters | Description |
|------|-----------|-------------|
| `activate` | — | Initialize the gripper. Call once per session before any motion. Takes ~4–5 seconds. |
| `open` | `speed` (0–100), `force` (0–100) | Fully open the gripper |
| `close` | `speed` (0–100), `force` (0–100) | Fully close the gripper |
| `move` | `position` (0–100), `speed`, `force` | Move to a normalized position: 0 = fully open, 100 = fully closed |
| `stop` | — | Stop gripper motion immediately |
| `reset` | — | Reset gripper fault state |
| `set_speed` | `speed` (0–100) | Set the default speed for subsequent commands |
| `set_force` | `force` (0–100) | Set the default force for subsequent commands |
| `get_status` | — | Return current gripper state (active, last command, speed, force, fault) |

`set_speed`, `set_force`, and `get_status` do not send URScript and do not affect External Control.

---

## Typical session sequence

```
1. activate           — initialize gripper (once per session)
2. open               — open before approaching object
3. [arm moves to pick position]
4. close              — grip object
5. recover_external_control  (on arm_mcp) — restore arm control
6. [arm moves to place position]
7. open               — release object
8. recover_external_control  (on arm_mcp) — restore arm control
```

---

## Files

| File | Description |
|------|-------------|
| `server.py` | FastMCP server, tool definitions |
| `robotiq_driver.py` | Low-level driver: builds and sends URScript to port 30002 |
| `Gripper.script` | URScript template with Robotiq `rq_` functions |
| `requirements.txt` | Python dependencies |

---

## Claude Desktop config (Windows)

```json
{
  "mcpServers": {
    "ur3e-gripper": {
      "command": "npx",
      "args": ["mcp-remote", "http://<ubuntu-ip>:8001/mcp", "--allow-http"]
    }
  }
}
```

---

## Troubleshooting

| Symptom | Cause | Fix |
|---------|-------|-----|
| `Connection refused` on activate | Wrong robot IP or port 30002 blocked | Confirm `ROBOT_IP` and that the robot is powered on |
| Gripper does not move after activate | Gripper not activated yet or fault state | Call `reset` then `activate` again |
| `Missing template: Gripper.script` | Script not in working directory | Run `server.py` from inside `gripper_mcp/` or set `GRIPPER_SCRIPT` env var to the full path |
| Arm unresponsive after gripper command | External Control was stopped by the URScript | Call `recover_external_control` on the arm MCP server |
| Port 8001 already in use | Previous server still running | `pkill -f "python3 server.py"` |
