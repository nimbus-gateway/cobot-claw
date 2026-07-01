#!/usr/bin/env bash
# Start UR ROS2 driver, arm MCP, gripper MCP, then cobot-claw.
# Usage: ./start.sh [path/to/cobot.yaml]

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONFIG="${1:-$SCRIPT_DIR/cobot.yaml}"

ROS_SETUP=/opt/ros/jazzy/setup.bash
DOMAIN_ID=42
ROBOT_IP=192.168.1.2

# ── helpers ────────────────────────────────────────────────────────────────
log() { echo "[start.sh] $*"; }

cleanup() {
    log "Shutting down background processes…"
    kill "${PIDS[@]}" 2>/dev/null || true
    wait "${PIDS[@]}" 2>/dev/null || true
}
trap cleanup EXIT

PIDS=()

# ── 1. UR ROS2 driver ──────────────────────────────────────────────────────
log "Starting UR ROS2 driver (background)…"
bash -c "
  source $ROS_SETUP
  export ROS_DOMAIN_ID=$DOMAIN_ID
  ros2 launch ur_robot_driver ur_control.launch.py \
    ur_type:=ur3e robot_ip:=$ROBOT_IP headless_mode:=true
" &>> "$SCRIPT_DIR/ur_driver.log" &
PIDS+=($!)
log "  PID ${PIDS[-1]} — logs: ur_driver.log"

log "Waiting 8s for ROS2 driver to come up…"
sleep 8

# ── 2. Arm MCP server ──────────────────────────────────────────────────────
log "Starting arm MCP server (background)…"
bash -c "
  source $ROS_SETUP
  export ROS_DOMAIN_ID=$DOMAIN_ID
  cd $SCRIPT_DIR/tools/ure3
  python ure3_arm_mcp.py
" &>> "$SCRIPT_DIR/arm_mcp.log" &
PIDS+=($!)
log "  PID ${PIDS[-1]} — logs: arm_mcp.log"

# ── 3. Gripper MCP server ──────────────────────────────────────────────────
log "Starting gripper MCP server (background)…"
bash -c "
  cd $SCRIPT_DIR/tools/gripper
  ROBOT_IP=$ROBOT_IP python server.py
" &>> "$SCRIPT_DIR/gripper_mcp.log" &
PIDS+=($!)
log "  PID ${PIDS[-1]} — logs: gripper_mcp.log"

log "Waiting 4s for MCP servers to come up…"
sleep 4

# ── 4. cobot-claw ──────────────────────────────────────────────────────────
log "Starting cobot-claw (foreground) with config: $CONFIG"
cobot-claw --config "$CONFIG"
