#!/usr/bin/env bash
# Run cobot-claw against a simulated UR3e (URSim in Docker) instead of a real robot.
#
# Starts, in order:
#   1. URSim e-Series UR3e container at 192.168.56.101 (powered on, brakes released)
#   2. Robotiq gripper emulator inside the container (tools/gripper/robotiq_emulator.py)
#   3. UR ROS 2 driver (ur_robot_driver) connected to the simulator
#   4. Arm MCP server (tools/ure3/ur3e_arm_mcp.py) on :8765
#   5. Gripper MCP server (tools/gripper/server.py) on :8001
#   6. cobot-claw REPL (skip with --no-app)
#
# View the simulated robot / teach pendant in a browser:
#   http://192.168.56.101:6080/vnc.html?autoconnect=true&resize=scale
#
# Usage:
#   scripts/start_ursim_demo.sh [--no-app] [--rviz] [-c path/to/cobot.yaml]
#
# Ctrl-C (or exiting cobot-claw) stops the driver and the MCP servers.
# The URSim container keeps running; stop it with:  docker stop ursim
#
# Notes:
#   - URSim is pinned to 5.22.2: the newest images (5.26+) send packets that
#     ur_client_library 2.6 can't parse, and the driver crashes on startup.
#   - URSim has no gripper hardware. The emulator answers the Robotiq URCap's
#     socket protocol on port 63352, so Gripper.script runs unchanged.
#     Set EMU_OBJECT_POS=0-255 to make closing stop on a simulated part.
#     Gripper state: docker exec ursim cat /tmp/robotiq_emulator.json

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CONFIG="$ROOT/cobot.yaml"
RUN_APP=true
RVIZ=false

while [[ $# -gt 0 ]]; do
  case "$1" in
    --no-app) RUN_APP=false ;;
    --rviz)   RVIZ=true ;;
    -c)       CONFIG="$2"; shift ;;
    -h|--help) sed -n '2,27p' "$0"; exit 0 ;;
    *) echo "Unknown option: $1" >&2; exit 1 ;;
  esac
  shift
done

ROS_DISTRO_DIR="${ROS_DISTRO_DIR:-/opt/ros/jazzy}"
ROS_DOMAIN_ID="${ROS_DOMAIN_ID:-42}"
URSIM_VERSION="${URSIM_VERSION:-5.22.2}"
SIM_IP=192.168.56.101
PROGRAMS_DIR="$HOME/.ursim/e-series/ur3e-$URSIM_VERSION/programs"
LOG_DIR="$ROOT/logs"
mkdir -p "$LOG_DIR" "$PROGRAMS_DIR"

log() { echo "[ursim-demo] $*"; }
dash() { (echo "$1"; sleep 1) | nc -w 3 "$SIM_IP" 29999 2>/dev/null | tail -n +2; }

[[ -f "$CONFIG" ]] || { echo "Config not found: $CONFIG (copy cobot.example.yaml to cobot.yaml)" >&2; exit 1; }
[[ -x "$ROOT/.venv/bin/cobot-claw" ]] || { echo "No .venv — see 'Running against URSim' in README.md" >&2; exit 1; }

# ROS setup scripts reference unset variables, so relax -u while sourcing.
set +u; source "$ROS_DISTRO_DIR/setup.bash"; set -u
export ROS_DOMAIN_ID

PIDS=()
cleanup() {
  [[ ${#PIDS[@]} -gt 0 ]] || return 0
  log "Stopping driver and MCP servers…"
  kill "${PIDS[@]}" 2>/dev/null || true
  wait "${PIDS[@]}" 2>/dev/null || true
}
trap cleanup EXIT

# ── 1. URSim ─────────────────────────────────────────────────────────────────
if docker ps --format '{{.Names}}' | grep -qx ursim; then
  log "URSim container already running"
else
  log "Starting URSim $URSIM_VERSION (UR3e)…"
  ros2 run ur_client_library start_ursim.sh -m ur3e -v "$URSIM_VERSION" -p "$PROGRAMS_DIR" -d >/dev/null
fi

log "Waiting for the simulator's dashboard server…"
for _ in $(seq 1 60); do
  dash robotmode | grep -qE 'POWER_OFF|IDLE|RUNNING' && break
  sleep 2
done

if ! dash robotmode | grep -q RUNNING; then
  log "Powering on and releasing brakes…"
  dash "close popup" >/dev/null
  dash "power on" >/dev/null
  for _ in $(seq 1 15); do dash robotmode | grep -q IDLE && break; sleep 2; done
  dash "brake release" >/dev/null
  for _ in $(seq 1 15); do dash robotmode | grep -q RUNNING && break; sleep 2; done
fi
log "$(dash robotmode)"

# ── 2. Robotiq gripper emulator ──────────────────────────────────────────────
if docker exec ursim pgrep -f robotiq_emulator.py >/dev/null 2>&1; then
  log "Gripper emulator already running"
else
  log "Starting Robotiq gripper emulator in the simulator…"
  docker cp "$ROOT/tools/gripper/robotiq_emulator.py" ursim:/tmp/robotiq_emulator.py
  docker exec -d -e EMU_OBJECT_POS="${EMU_OBJECT_POS:-}" ursim \
    sh -c 'python3 /tmp/robotiq_emulator.py > /tmp/robotiq_emulator.log 2>&1'
  sleep 1
  docker exec ursim pgrep -f robotiq_emulator.py >/dev/null \
    || { log "Gripper emulator failed:"; docker exec ursim cat /tmp/robotiq_emulator.log; exit 1; }
fi

# ── 3. UR ROS 2 driver ───────────────────────────────────────────────────────
log "Starting UR ROS 2 driver (log: logs/ursim_driver.log)…"
ros2 launch ur_robot_driver ur_control.launch.py \
  ur_type:=ur3e robot_ip:="$SIM_IP" headless_mode:=true launch_rviz:="$RVIZ" \
  > "$LOG_DIR/ursim_driver.log" 2>&1 &
PIDS+=($!)

for _ in $(seq 1 60); do
  grep -q 'Ready to receive control commands' "$LOG_DIR/ursim_driver.log" && break
  if grep -qE 'FATAL|process has died' "$LOG_DIR/ursim_driver.log"; then
    log "Driver failed — see logs/ursim_driver.log"; exit 1
  fi
  sleep 1
done
log "Driver connected to the simulator"

# ── 4. Arm MCP server ────────────────────────────────────────────────────────
if ss -ltn | grep -q ':8765 '; then
  log "Port 8765 already in use — is another arm MCP server running?"; exit 1
fi
log "Starting arm MCP server on :8765 (log: logs/ursim_arm_mcp.log)…"
( cd "$ROOT/tools/ure3" && ROBOT_IP="$SIM_IP" exec python3 ur3e_arm_mcp.py ) \
  > "$LOG_DIR/ursim_arm_mcp.log" 2>&1 &
PIDS+=($!)
for _ in $(seq 1 30); do ss -ltn | grep -q ':8765 ' && break; sleep 1; done

# ── 5. Gripper MCP server ────────────────────────────────────────────────────
if ss -ltn | grep -q ':8001 '; then
  log "Port 8001 already in use — is another gripper MCP server running?"; exit 1
fi
log "Starting gripper MCP server on :8001 (log: logs/ursim_gripper_mcp.log)…"
( cd "$ROOT/tools/gripper" && ROBOT_IP="$SIM_IP" exec "$ROOT/.venv/bin/python" server.py ) \
  > "$LOG_DIR/ursim_gripper_mcp.log" 2>&1 &
PIDS+=($!)
for _ in $(seq 1 30); do ss -ltn | grep -q ':8001 ' && break; sleep 1; done

log "Ready. Teach pendant: http://$SIM_IP:6080/vnc.html?autoconnect=true&resize=scale"

# ── 6. cobot-claw ────────────────────────────────────────────────────────────
if $RUN_APP; then
  cd "$ROOT"
  "$ROOT/.venv/bin/cobot-claw" -c "$CONFIG" -r ur3e
else
  log "Sim stack running (--no-app). Press Ctrl-C to stop."
  wait
fi
