#!/usr/bin/env python3
"""
UR3e Arm MCP Server (FastMCP + Streamable HTTP)

What this server provides:
- Direct joint-space movement
- Named pose movement (optionally via home)
- Joint limit reporting
- TCP pose/orientation reporting from TF
- External Control recovery after interruptions
- Dashboard speed control
- Better joint-state ordering and motion completion checks

Recommended robot launch:
  ros2 launch ur_robot_driver ur_control.launch.py \
    ur_type:=ur3e robot_ip:=192.168.1.2 headless_mode:=true

Run:
  source /opt/ros/jazzy/setup.bash
  export ROS_DOMAIN_ID=42
  python3 ur3e_arm_mcp.py

MCP endpoint:
  http://0.0.0.0:8765/mcp
"""

from __future__ import annotations

import functools
import math
import os
import threading
import time
from typing import Any

import rclpy
from rclpy.node import Node
from rclpy.time import Time
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from sensor_msgs.msg import JointState
from builtin_interfaces.msg import Duration
from std_srvs.srv import Trigger
from geometry_msgs.msg import TransformStamped
import tf2_ros

from mcp.server.fastmcp import FastMCP

try:
    from ur_dashboard_msgs.srv import GetRobotMode, GetSafetyMode, IsProgramRunning, IsInRemoteControl
    HAVE_UR_DASHBOARD_MSGS = True
except Exception:
    HAVE_UR_DASHBOARD_MSGS = False


# ------------------------------------------------------------------------------
# Config
# ------------------------------------------------------------------------------

SERVER_HOST = os.getenv("MCP_HOST", "0.0.0.0")
SERVER_PORT = int(os.getenv("MCP_PORT", "8765"))

ROBOT_IP = os.getenv("ROBOT_IP", "192.168.1.2")
HEADLESS_MODE = os.getenv("HEADLESS_MODE", "true").lower() in {"1", "true", "yes"}

BASE_FRAME = os.getenv("BASE_FRAME", "base")
TCP_FRAME = os.getenv("TCP_FRAME", "tool0")

DEFAULT_MOVE_DURATION = float(os.getenv("DEFAULT_MOVE_DURATION", "5.0"))

JOINTS = [
    "shoulder_pan_joint",
    "shoulder_lift_joint",
    "elbow_joint",
    "wrist_1_joint",
    "wrist_2_joint",
    "wrist_3_joint",
]

# Conservative software-side limits; adjust if needed.
JOINT_LIMITS = {
    "shoulder_pan_joint": (-6.0, 6.0),
    "shoulder_lift_joint": (-6.0, 6.0),
    "elbow_joint": (-3.0, 3.0),
    "wrist_1_joint": (-6.0, 6.0),
    "wrist_2_joint": (-6.0, 6.0),
    "wrist_3_joint": (-6.0, 6.0),
}

WARN_MARGIN_RAD = 0.8
BLOCK_MARGIN_RAD = 0.3

POSES = {
    "home":          [0.0,    -1.5708, 0.0,    -1.5708, 0.0,    0.0],
    "retract":       [0.0,    -2.5,    2.0,    -1.0,    0.0,    0.0],
    "reach_forward": [0.0,    -1.5708, 1.5708, -1.5708, 0.0,    0.0],
    "point_down":    [0.0,    -1.5708, 1.5708, -1.5708, 1.5708, 0.0],
    "look_left":     [1.5708, -1.5708, 0.0,    -1.5708, 0.0,    0.0],
    "look_right":    [-1.5708,-1.5708, 0.0,    -1.5708, 0.0,    0.0],
    "look_back":     [2.8,    -1.5708, 0.0,    -1.5708, 0.0,    0.0],
}


# ------------------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------------------

def rad_to_deg(x: float) -> float:
    return x * 180.0 / math.pi


def clamp(value: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, value))


def quat_to_rpy(x: float, y: float, z: float, w: float) -> dict[str, float]:
    # Standard quaternion -> roll/pitch/yaw
    sinr_cosp = 2.0 * (w * x + y * z)
    cosr_cosp = 1.0 - 2.0 * (x * x + y * y)
    roll = math.atan2(sinr_cosp, cosr_cosp)

    sinp = 2.0 * (w * y - z * x)
    if abs(sinp) >= 1.0:
        pitch = math.copysign(math.pi / 2.0, sinp)
    else:
        pitch = math.asin(sinp)

    siny_cosp = 2.0 * (w * z + x * y)
    cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
    yaw = math.atan2(siny_cosp, cosy_cosp)

    return {
        "roll_rad": roll,
        "pitch_rad": pitch,
        "yaw_rad": yaw,
        "roll_deg": rad_to_deg(roll),
        "pitch_deg": rad_to_deg(pitch),
        "yaw_deg": rad_to_deg(yaw),
    }


def check_joint_vector_limits(joints: list[float]) -> dict[str, Any]:
    issues: list[dict[str, Any]] = []
    blocked = False

    for name, value in zip(JOINTS, joints):
        lo, hi = JOINT_LIMITS[name]
        margin = min(value - lo, hi - value)

        if margin < BLOCK_MARGIN_RAD:
            blocked = True
            status = "BLOCKED"
        elif margin < WARN_MARGIN_RAD:
            status = "WARNING"
        else:
            continue

        issues.append({
            "joint": name,
            "value_rad": round(value, 4),
            "value_deg": round(rad_to_deg(value), 1),
            "limit_low_rad": lo,
            "limit_high_rad": hi,
            "margin_rad": round(margin, 4),
            "status": status,
        })

    return {"blocked": blocked, "issues": issues}


# ------------------------------------------------------------------------------
# ROS node
# ------------------------------------------------------------------------------

class UR3eMCPNode(Node):
    def __init__(self) -> None:
        super().__init__("ur3e_mcp_node")

        self._lock = threading.Lock()
        self.current_joints: list[float] | None = None
        self.last_target_joints: list[float] | None = None
        self.last_command_time: float | None = None
        self.last_command_duration: float | None = None

        self.trajectory_pub = self.create_publisher(
            JointTrajectory,
            "/scaled_joint_trajectory_controller/joint_trajectory",
            10,
        )

        self.create_subscription(
            JointState,
            "/joint_states",
            self._on_joint_state,
            10,
        )

        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)

        self.play_client = self.create_client(Trigger, "/dashboard_client/play")
        self.resend_client = self.create_client(Trigger, "/io_and_status_controller/resend_robot_program")

        if HAVE_UR_DASHBOARD_MSGS:
            self.robot_mode_client = self.create_client(GetRobotMode, "/dashboard_client/get_robot_mode")
            self.safety_mode_client = self.create_client(GetSafetyMode, "/dashboard_client/get_safety_mode")
            self.program_running_client = self.create_client(IsProgramRunning, "/dashboard_client/program_running")
            self.remote_control_client = self.create_client(IsInRemoteControl, "/dashboard_client/is_in_remote_control")
        else:
            self.robot_mode_client = None
            self.safety_mode_client = None
            self.program_running_client = None
            self.remote_control_client = None

    def _on_joint_state(self, msg: JointState) -> None:
        name_to_pos = dict(zip(msg.name, msg.position))
        ordered: list[float] = []

        for joint in JOINTS:
            if joint not in name_to_pos:
                self.get_logger().warning(
                    f"joint_state incomplete: '{joint}' missing. "
                    f"Got: {list(name_to_pos.keys())}"
                )
                return
            ordered.append(float(name_to_pos[joint]))

        with self._lock:
            self.current_joints = ordered

    def get_current_joints(self) -> list[float] | None:
        with self._lock:
            return list(self.current_joints) if self.current_joints is not None else None

    def send_joint_trajectory(self, joints: list[float], duration_sec: float) -> None:
        traj = JointTrajectory()
        traj.joint_names = JOINTS

        pt = JointTrajectoryPoint()
        pt.positions = [float(x) for x in joints]
        pt.velocities = [0.0] * 6
        pt.accelerations = [0.0] * 6

        sec = int(duration_sec)
        nanosec = int((duration_sec - sec) * 1e9)
        pt.time_from_start = Duration(sec=sec, nanosec=nanosec)

        traj.points = [pt]
        self.trajectory_pub.publish(traj)

        with self._lock:
            self.last_target_joints = list(joints)
            self.last_command_time = time.time()
            self.last_command_duration = duration_sec

    def is_motion_complete(self, tolerance_rad: float = 0.03) -> dict[str, Any]:
        with self._lock:
            current = list(self.current_joints) if self.current_joints is not None else None
            target = list(self.last_target_joints) if self.last_target_joints is not None else None
            t0 = self.last_command_time
            duration = self.last_command_duration

        if current is None:
            return {"ok": False, "reason": "no_joint_state"}
        if target is None:
            return {"ok": True, "complete": True, "reason": "no_target"}

        diffs = [abs(c - t) for c, t in zip(current, target)]
        max_error = max(diffs)

        elapsed = None if t0 is None else time.time() - t0
        nominal_done = (elapsed is not None and duration is not None and elapsed >= duration)

        return {
            "ok": True,
            "complete": max_error <= tolerance_rad,
            "nominal_time_elapsed": nominal_done,
            "max_error_rad": round(max_error, 5),
            "joint_errors_rad": [round(d, 5) for d in diffs],
            "target_joints": [round(x, 5) for x in target],
            "current_joints": [round(x, 5) for x in current],
        }

    def get_tcp_transform(self) -> TransformStamped:
        return self.tf_buffer.lookup_transform(
            BASE_FRAME,
            TCP_FRAME,
            Time(),
            timeout=rclpy.duration.Duration(seconds=0.5),
        )

    def _call_trigger(self, client, timeout_sec: float = 3.0) -> dict[str, Any]:
        if client is None:
            return {"ok": False, "message": "service client unavailable"}

        if not client.wait_for_service(timeout_sec=timeout_sec):
            return {"ok": False, "message": "service not available"}

        future = client.call_async(Trigger.Request())
        start = time.time()
        while not future.done():
            if time.time() - start > timeout_sec:
                return {"ok": False, "message": "service call timeout"}
            time.sleep(0.05)

        try:
            resp = future.result()
            return {"ok": bool(resp.success), "message": resp.message}
        except Exception as e:
            return {"ok": False, "message": str(e)}

    def _call_simple_service(self, client, request, timeout_sec: float = 3.0) -> dict[str, Any]:
        if client is None:
            return {"ok": False, "message": "service client unavailable"}

        if not client.wait_for_service(timeout_sec=timeout_sec):
            return {"ok": False, "message": "service not available"}

        future = client.call_async(request)
        start = time.time()
        while not future.done():
            if time.time() - start > timeout_sec:
                return {"ok": False, "message": "service call timeout"}
            time.sleep(0.05)

        try:
            resp = future.result()
            return {"ok": True, "response": resp}
        except Exception as e:
            return {"ok": False, "message": str(e)}

    def recover_external_control(self) -> dict[str, Any]:
        if HEADLESS_MODE:
            result = self._call_trigger(self.resend_client)
            result["method"] = "resend_robot_program"
            return result

        result = self._call_trigger(self.play_client)
        result["method"] = "dashboard_play"
        return result

    def get_dashboard_status(self) -> dict[str, Any]:
        status: dict[str, Any] = {}

        if HAVE_UR_DASHBOARD_MSGS:
            # Robot mode — .robot_mode is a ROS2 msg object; extract .val (int)
            rm = self._call_simple_service(self.robot_mode_client, GetRobotMode.Request())
            if rm["ok"]:
                resp = rm["response"]
                raw_mode = getattr(resp, "robot_mode", None)
                status["robot_mode"] = int(getattr(raw_mode, "val", raw_mode)) if raw_mode is not None else None
                status["robot_mode_label"] = getattr(resp, "answer", None)

            sm = self._call_simple_service(self.safety_mode_client, GetSafetyMode.Request())
            if sm["ok"]:
                resp = sm["response"]
                raw_safety = getattr(resp, "safety_mode", None)
                status["safety_mode"] = int(getattr(raw_safety, "val", raw_safety)) if raw_safety is not None else None
                status["safety_mode_label"] = getattr(resp, "answer", None)

            pr = self._call_simple_service(self.program_running_client, IsProgramRunning.Request())
            if pr["ok"]:
                resp = pr["response"]
                status["program_running"] = bool(getattr(resp, "program_running", None))

            rc = self._call_simple_service(self.remote_control_client, IsInRemoteControl.Request())
            if rc["ok"]:
                resp = rc["response"]
                status["in_remote_control"] = bool(getattr(resp, "in_remote_control", None))

        return status


robot: UR3eMCPNode | None = None
_ros_ready = threading.Event()


def ros_spin_thread() -> None:
    global robot
    rclpy.init()
    robot = UR3eMCPNode()
    _ros_ready.set()
    rclpy.spin(robot)


def require_robot() -> UR3eMCPNode:
    if robot is None:
        raise RuntimeError("ROS node not initialized — is ur_robot_driver running?")
    return robot


def safe_tool(fn):
    """
    Decorator for @mcp.tool() functions.
    Catches any unhandled exception and returns {"ok": False, "error": ..., "message": ...}
    instead of propagating a stacktrace to the MCP client.
    """
    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        try:
            return fn(*args, **kwargs)
        except Exception as e:
            return {
                "ok": False,
                "error": type(e).__name__,
                "message": str(e),
            }
    return wrapper


# ------------------------------------------------------------------------------
# FastMCP
# ------------------------------------------------------------------------------

mcp = FastMCP(
    "ur3e-arm",
    host=SERVER_HOST,
    port=SERVER_PORT,
    instructions=(
        "This server controls a UR3e arm over ROS 2. "
        "Use get_robot_status, get_joint_states, and get_joint_limits_status to assess state. "
        "Use move_joints for direct joint-space control. "
        "Use move_named_pose for convenience. "
        "Use recover_external_control after gripper or custom-script interruptions. "
        "Do not assume moves require home unless via_home is explicitly requested."
    ),
)


@mcp.tool()
@safe_tool
def get_robot_status() -> dict[str, Any]:
    """
    Return current high-level robot state, joint availability, dashboard status,
    and motion completion info.
    """
    r = require_robot()
    joints = r.get_current_joints()
    status = {
        "ros_ready": True,
        "robot_ip": ROBOT_IP,
        "headless_mode": HEADLESS_MODE,
        "has_joint_state": joints is not None,
        "joint_state": [round(x, 5) for x in joints] if joints else None,
        "motion": r.is_motion_complete(),
        "dashboard": r.get_dashboard_status(),
        "frames": {
            "base_frame": BASE_FRAME,
            "tcp_frame": TCP_FRAME,
        },
    }
    return status


@mcp.tool()
@safe_tool
def get_joint_states() -> dict[str, Any]:
    """
    Return all joints in correct UR order with radians, degrees, and limit margins.
    """
    r = require_robot()
    joints = r.get_current_joints()
    if joints is None:
        return {"ok": False, "message": "No joint state available"}

    entries = []
    for name, value in zip(JOINTS, joints):
        lo, hi = JOINT_LIMITS[name]
        margin = min(value - lo, hi - value)
        entries.append({
            "joint": name,
            "value_rad": round(value, 5),
            "value_deg": round(rad_to_deg(value), 3),
            "limit_low_rad": lo,
            "limit_high_rad": hi,
            "margin_rad": round(margin, 5),
        })

    return {"ok": True, "joints": entries}


@mcp.tool()
@safe_tool
def get_joint_limits_status() -> dict[str, Any]:
    """
    Report whether any current joints are near or beyond configured software limits.
    """
    r = require_robot()
    joints = r.get_current_joints()
    if joints is None:
        return {"ok": False, "message": "No joint state available"}

    result = check_joint_vector_limits(joints)
    result["ok"] = True
    result["current_joints"] = [round(x, 5) for x in joints]
    return result


@mcp.tool()
@safe_tool
def get_tcp_pose() -> dict[str, Any]:
    """
    Return TCP pose from TF using BASE_FRAME -> TCP_FRAME.
    Includes position, quaternion, and RPY.
    """
    r = require_robot()
    try:
        tf = r.get_tcp_transform()
    except Exception as e:
        return {
            "ok": False,
            "message": f"Could not get TF from {BASE_FRAME} to {TCP_FRAME}: {e}",
        }

    t = tf.transform.translation
    q = tf.transform.rotation
    rpy = quat_to_rpy(q.x, q.y, q.z, q.w)

    return {
        "ok": True,
        "base_frame": BASE_FRAME,
        "tcp_frame": TCP_FRAME,
        "position_m": {
            "x": t.x,
            "y": t.y,
            "z": t.z,
        },
        "orientation_quaternion": {
            "x": q.x,
            "y": q.y,
            "z": q.z,
            "w": q.w,
        },
        "orientation_rpy": rpy,
    }


@mcp.tool()
@safe_tool
def is_motion_complete(tolerance_rad: float = 0.03) -> dict[str, Any]:
    """
    Check whether the last commanded motion has converged to target within tolerance.
    """
    r = require_robot()
    return r.is_motion_complete(tolerance_rad=tolerance_rad)


@mcp.tool()
@safe_tool
def move_joints(joints: list[float], duration_sec: float = DEFAULT_MOVE_DURATION) -> dict[str, Any]:
    """
    Move directly in joint space.
    This does not force via-home or midpoint logic.
    """
    r = require_robot()

    if len(joints) != 6:
        return {"ok": False, "message": "Exactly 6 joint values are required"}

    # Validate target vector against configured limits
    target_check = check_joint_vector_limits(joints)
    if target_check["blocked"]:
        return {
            "ok": False,
            "message": "Target joint vector exceeds configured software limits",
            "issues": target_check["issues"],
        }

    duration_sec = max(0.2, float(duration_sec))
    r.send_joint_trajectory([float(x) for x in joints], duration_sec)

    return {
        "ok": True,
        "message": "Joint trajectory sent",
        "duration_sec": duration_sec,
        "target_joints_rad": [round(float(x), 5) for x in joints],
    }


@mcp.tool()
@safe_tool
def move_named_pose(name: str, duration_sec: float = DEFAULT_MOVE_DURATION, via_home: bool = False) -> dict[str, Any]:
    """
    Move to a named pose. Optionally go via home first.
    """
    r = require_robot()

    if name not in POSES:
        return {"ok": False, "message": f"Unknown pose '{name}'", "available_poses": list(POSES.keys())}

    if via_home and name != "home":
        r.send_joint_trajectory(POSES["home"], max(0.2, duration_sec))
        # Poll for completion instead of a single blocking sleep.
        # Cap at duration_sec + 2s to handle slower moves without hanging.
        deadline = time.time() + duration_sec + 2.0
        while time.time() < deadline:
            time.sleep(0.1)
            if r.is_motion_complete().get("complete"):
                break

    r.send_joint_trajectory(POSES[name], max(0.2, duration_sec))

    return {
        "ok": True,
        "message": "Named pose command sent",
        "pose": name,
        "via_home": via_home,
        "duration_sec": duration_sec,
        "target_joints_rad": [round(x, 5) for x in POSES[name]],
    }


@mcp.tool()
@safe_tool
def go_home(duration_sec: float = DEFAULT_MOVE_DURATION) -> dict[str, Any]:
    """
    Convenience move to home. Not forced anywhere else.
    """
    return move_named_pose("home", duration_sec=duration_sec, via_home=False)


@mcp.tool()
@safe_tool
def unwind_joint(joint_index: int, target_rad: float = 0.0, duration_sec: float = DEFAULT_MOVE_DURATION) -> dict[str, Any]:
    """
    Move one joint only, keeping the others fixed.
    Useful for recovery from winding.
    """
    r = require_robot()
    current = r.get_current_joints()
    if current is None:
        return {"ok": False, "message": "No joint state available"}

    if not 0 <= joint_index < 6:
        return {"ok": False, "message": "joint_index must be between 0 and 5"}

    target = list(current)
    target[joint_index] = float(target_rad)

    target_check = check_joint_vector_limits(target)
    if target_check["blocked"]:
        return {
            "ok": False,
            "message": "Requested unwind target exceeds configured software limits",
            "issues": target_check["issues"],
        }

    r.send_joint_trajectory(target, max(0.2, duration_sec))
    return {
        "ok": True,
        "message": "Single-joint unwind sent",
        "joint": JOINTS[joint_index],
        "joint_index": joint_index,
        "target_rad": float(target_rad),
        "duration_sec": duration_sec,
    }


@mcp.tool()
@safe_tool
def hold_position(duration_sec: float = 0.5) -> dict[str, Any]:
    """
    Command the robot to hold its current position by re-sending current joints.
    This is not an emergency stop.
    """
    r = require_robot()
    current = r.get_current_joints()
    if current is None:
        return {"ok": False, "message": "No joint state available"}

    r.send_joint_trajectory(current, max(0.2, duration_sec))
    return {
        "ok": True,
        "message": "Hold-position command sent",
        "duration_sec": duration_sec,
        "current_joints_rad": [round(x, 5) for x in current],
    }


@mcp.tool()
@safe_tool
def recover_external_control() -> dict[str, Any]:
    """
    Recover arm control after External Control was interrupted.
    In headless mode this uses resend_robot_program.
    Otherwise it uses dashboard play.
    """
    r = require_robot()
    result = r.recover_external_control()
    result["headless_mode"] = HEADLESS_MODE
    return result


@mcp.tool()
@safe_tool
def list_named_poses() -> dict[str, Any]:
    """
    List all available named poses.
    """
    return {"ok": True, "poses": POSES}


@mcp.tool()
@safe_tool
def move_tcp_pose(
    x: float, y: float, z: float,
    qx: float, qy: float, qz: float, qw: float,
    speed: float = 0.05,
    accel: float = 0.3,
) -> dict[str, Any]:
    """
    Move the TCP to a Cartesian pose using URScript movel() sent directly to
    the robot controller on port 30002.  The robot's internal IK resolves joints.

    x, y, z   – target TCP position in robot base frame (metres)
    qx,qy,qz,qw – target orientation as quaternion
    speed     – linear speed m/s (default 0.05 m/s – slow/safe)
    accel     – linear acceleration m/s² (default 0.3)
    """
    import socket

    # Quaternion → axis-angle rotation vector for URScript
    n = math.sqrt(qx*qx + qy*qy + qz*qz + qw*qw)
    qx, qy, qz, qw = qx/n, qy/n, qz/n, qw/n
    if qw < 0:
        qx, qy, qz, qw = -qx, -qy, -qz, -qw
    angle = 2.0 * math.acos(min(1.0, max(-1.0, qw)))
    s = math.sqrt(max(0.0, 1.0 - qw*qw))
    if s < 1e-6:
        rx, ry, rz = 0.0, 0.0, float(angle)
    else:
        rx = qx / s * angle
        ry = qy / s * angle
        rz = qz / s * angle

    script = (
        f"movel(p[{x:.6f},{y:.6f},{z:.6f},{rx:.6f},{ry:.6f},{rz:.6f}],"
        f"a={accel:.3f},v={speed:.3f})\n"
    )

    try:
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(5.0)
        sock.connect((ROBOT_IP, 30002))
        sock.sendall(script.encode("utf-8"))
        sock.close()
        return {
            "ok": True,
            "message": "movel command sent",
            "target_position_m": {"x": x, "y": y, "z": z},
            "rotation_vector": {"rx": rx, "ry": ry, "rz": rz},
            "speed_m_s": speed,
            "script": script.strip(),
        }
    except Exception as e:
        return {"ok": False, "message": str(e)}


@mcp.tool()
@safe_tool
def set_speed(percent: int) -> dict[str, Any]:
    """
    Set robot speed slider via dashboard raw socket on port 29999.
    """
    import socket

    percent = int(clamp(percent, 1, 100))
    try:
        s = socket.socket()
        s.settimeout(3.0)
        s.connect((ROBOT_IP, 29999))
        _ = s.recv(1024)
        s.send(f"speed {percent}\n".encode())
        resp = s.recv(1024).decode().strip()
        s.close()
        return {"ok": True, "percent": percent, "response": resp}
    except Exception as e:
        return {"ok": False, "message": str(e)}


@mcp.custom_route("/health", methods=["GET"])
async def health_check(request):
    return {
        "ok": True,
        "server": "ur3e-arm",
        "ros_ready": _ros_ready.is_set(),
        "robot_connected": robot is not None and robot.current_joints is not None if robot else False,
    }


def main() -> None:
    ros_thread = threading.Thread(target=ros_spin_thread, daemon=True)
    ros_thread.start()

    _ros_ready.wait(timeout=10.0)

    print("Waiting for robot joint states...", flush=True)
    for _ in range(50):
        time.sleep(0.2)
        if robot is not None and robot.current_joints is not None:
            print("Robot connected.", flush=True)
            break
    else:
        print("WARNING: No joint states yet. Is UR driver running?", flush=True)

    print(f"Starting MCP HTTP server on http://{SERVER_HOST}:{SERVER_PORT}/mcp", flush=True)
    mcp.run(transport="streamable-http")


if __name__ == "__main__":
    main()