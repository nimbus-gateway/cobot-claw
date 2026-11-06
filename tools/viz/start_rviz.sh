#!/usr/bin/env bash
# Show the simulated UR3e with a Robotiq gripper in RViz.
#
# The arm follows URSim (the UR driver's /robot_description and joint states).
# The gripper is a Robotiq 2F-85 model whose fingers follow the gripper
# emulator (tools/gripper/robotiq_emulator.py) through gripper_state_bridge.py.
#
# Needs the sim stack running (scripts/start_ursim_demo.sh) and the
# robotiq_description package (default: ~/robotiq-gripper/install).
#
# Usage: tools/viz/start_rviz.sh

set -eo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROS_DISTRO_DIR="${ROS_DISTRO_DIR:-/opt/ros/jazzy}"
ROBOTIQ_WS="${ROBOTIQ_WS:-$HOME/robotiq-gripper/install}"
export ROS_DOMAIN_ID="${ROS_DOMAIN_ID:-42}"

source "$ROS_DISTRO_DIR/setup.bash"
[[ -f "$ROBOTIQ_WS/setup.bash" ]] || {
  echo "robotiq_description not found at $ROBOTIQ_WS (set ROBOTIQ_WS)" >&2; exit 1; }
source "$ROBOTIQ_WS/setup.bash"

PIDS=()
kill_tree() {
  local child
  for child in $(pgrep -P "$1" 2>/dev/null); do kill_tree "$child"; done
  kill "$1" 2>/dev/null || true
}
cleanup() { local p; for p in "${PIDS[@]}"; do kill_tree "$p"; done; wait "${PIDS[@]}" 2>/dev/null || true; }
trap cleanup EXIT

URDF="$(xacro "$HERE/gripper_on_ur.urdf.xacro")"

ros2 run robot_state_publisher robot_state_publisher \
  --ros-args -r __ns:=/gripper -r __node:=gripper_state_publisher -p robot_description:="$URDF" &
PIDS+=($!)

python3 "$HERE/gripper_state_bridge.py" --ros-args -r __ns:=/gripper &
PIDS+=($!)

rviz2 -d "$HERE/ursim_gripper.rviz" &
PIDS+=($!)

# Wait in the background so a SIGTERM from the launcher runs cleanup at once.
trap 'exit 0' TERM INT
wait "${PIDS[-1]}"
