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
from sensor_msgs.msg import JointState
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

RTDE_PORT = 30004

POSES = {
    "home":          [-3.16018, -0.6864, -1.4132, -2.0002, 1.63811, 3.17233],
    "retract":       [0.0,    -2.5,    2.0,    -1.0,    0.0,    0.0],
    "reach_forward": [0.0,    -1.5708, 1.5708, -1.5708, 0.0,    0.0],
    "point_down":    [0.0,    -1.5708, 1.5708, -1.5708, 1.5708, 0.0],
    "look_left":     [1.5708, -1.5708, 0.0,    -1.5708, 0.0,    0.0],
    "look_right":    [-1.5708,-1.5708, 0.0,    -1.5708, 0.0,    0.0],
    "look_back":     [2.8,    -1.5708, 0.0,    -1.5708, 0.0,    0.0],
}

# Optional override of the home pose: six comma-separated joint angles (rad).
# The URSim demo uses this for a home above the table in front of the robot.
# The real cell's home (base -181°, behind the robot) made every pick-and-place
# swing the base around. Unset = the home above.
if os.getenv("HOME_JOINTS"):
    _home = [float(v) for v in os.environ["HOME_JOINTS"].split(",")]
    if len(_home) != 6:
        raise SystemExit("HOME_JOINTS needs exactly 6 comma-separated joint angles (rad)")
    POSES["home"] = _home


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


# UR joints turn about ±360°, so a named pose such as "home" (base -181°)
# can be reached with joint angles that differ by 2π, and movej interpolates
# the raw angles. With SHORTEST_JOINT_PATH on (default), move_named_pose
# shifts each joint by ±2π to the equivalent nearest the arm's current angles
# (inside JOINT_LIMITS), so going home from +75° turns 104° instead of 256°.
# For the base, equivalents within ±(π + 0.5) are preferred: if the base has
# been wound up, it unwinds on the way to the named pose, with the arm raised,
# and not on the next IK move down near the table (IK returns base angles in
# that range). Other joints just take the nearest equivalent.
# Explicit joint targets (move_joints, move_path, unwind_joint) are sent
# unchanged: the caller chose those angles and the direction of travel.
SHORTEST_JOINT_PATH = os.getenv("SHORTEST_JOINT_PATH", "1").lower() not in {"0", "false", "no"}


def shortest_equivalent(target: list[float], reference: list[float] | None) -> list[float]:
    """Shift each target angle by multiples of 2π to the value nearest `reference`."""
    if not SHORTEST_JOINT_PATH or reference is None or len(reference) != len(target):
        return [float(x) for x in target]
    out = []
    for name, t, ref in zip(JOINTS, target, reference):
        lo, hi = JOINT_LIMITS[name]
        cands = [float(t) + 2 * math.pi * k for k in (0, -1, 1, -2, 2)]
        cands = [c for c in cands if lo <= c <= hi] or [float(t)]
        unwound = [c for c in cands if abs(c) <= math.pi + 0.5] if name == "shoulder_pan_joint" else []
        out.append(min(unwound or cands, key=lambda c: abs(c - ref)))
    return out


def _wait_until_reached(target: list[float], timeout_sec: float, tolerance_rad: float) -> bool:
    """Block until every joint is within tolerance of target. False on timeout."""
    deadline = time.time() + timeout_sec
    while time.time() < deadline:
        time.sleep(0.1)
        current = _current_joints()
        if current is not None and max(abs(c - t) for c, t in zip(current, target)) <= tolerance_rad:
            return True
    return False


def _current_joints() -> list[float] | None:
    r = robot
    if r is not None:
        cur = r.get_current_joints()
        if cur is not None:
            return cur
    if rtde is not None and rtde.connected:
        return rtde.joints
    return None


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
# RTDE passive reader (port 30004) — works without External Control
# ------------------------------------------------------------------------------
# Implements just enough of the RTDE handshake to subscribe to:
#   actual_q        – actual joint positions [6 x double]
#   actual_TCP_pose – actual TCP pose [6 x double] (x,y,z,rx,ry,rz)
# No ur-rtde pip package required — pure stdlib sockets.

import struct

# RTDE packet types
_RTDE_REQUEST_PROTOCOL_VERSION = 86   # 'V'
_RTDE_GET_URCONTROL_VERSION    = 118  # 'v'
_RTDE_TEXT_MESSAGE             = 77   # 'M'
_RTDE_DATA_PACKAGE             = 85   # 'U'
_RTDE_CONTROL_PACKAGE_SETUP_OUTPUTS  = 79   # 'O'
_RTDE_CONTROL_PACKAGE_START    = 83   # 'S'
_RTDE_CONTROL_PACKAGE_PAUSE    = 80   # 'P'

_RTDE_PROTOCOL_VERSION = 2


def _rtde_pack(ptype: int, payload: bytes = b"") -> bytes:
    return struct.pack(">HB", 3 + len(payload), ptype) + payload


def _rtde_recv(sock) -> tuple[int, bytes]:
    """Read one RTDE packet; return (type, payload)."""
    header = b""
    while len(header) < 3:
        chunk = sock.recv(3 - len(header))
        if not chunk:
            raise ConnectionError("RTDE connection closed")
        header += chunk
    size, ptype = struct.unpack(">HB", header)
    payload = b""
    remaining = size - 3
    while len(payload) < remaining:
        chunk = sock.recv(remaining - len(payload))
        if not chunk:
            raise ConnectionError("RTDE connection closed")
        payload += chunk
    return ptype, payload


class RTDEReader:
    """
    Passive RTDE subscriber.  Connects to port 30004 and continuously
    reads actual_q and actual_TCP_pose.  Works while any UR program is
    running — does NOT require External Control.

    Thread-safe; self-reconnects on drop.
    """

    # actual_q[6] + actual_TCP_pose[6] + output_double_register_0..5[6] = 18 doubles
    VARIABLES = [
        "actual_q", "actual_TCP_pose",
        "output_double_register_0", "output_double_register_1",
        "output_double_register_2", "output_double_register_3",
        "output_double_register_4", "output_double_register_5",
    ]

    def __init__(self, robot_ip: str, frequency: float = 10.0) -> None:
        self._ip = robot_ip
        self._frequency = frequency
        self._lock = threading.Lock()
        self._joints: list[float] | None = None
        self._tcp: list[float] | None = None
        self._registers: list[float] = [0.0] * 6   # output_double_register_0..5
        self._running = False
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        self._running = True
        self._thread = threading.Thread(target=self._loop, daemon=True, name="rtde-reader")
        self._thread.start()

    def stop(self) -> None:
        self._running = False

    @property
    def joints(self) -> list[float] | None:
        with self._lock:
            return list(self._joints) if self._joints is not None else None

    @property
    def tcp(self) -> list[float] | None:
        with self._lock:
            return list(self._tcp) if self._tcp is not None else None

    @property
    def registers(self) -> list[float]:
        with self._lock:
            return list(self._registers)

    @property
    def connected(self) -> bool:
        with self._lock:
            return self._joints is not None

    def _loop(self) -> None:
        import socket as _socket
        while self._running:
            try:
                self._connect_and_read()
            except Exception:
                pass
            if self._running:
                with self._lock:
                    self._joints = None
                    self._tcp = None
                time.sleep(2.0)   # back-off before reconnect

    def _connect_and_read(self) -> None:
        import socket as _socket
        sock = _socket.socket(_socket.AF_INET, _socket.SOCK_STREAM)
        sock.settimeout(5.0)
        sock.connect((self._ip, RTDE_PORT))
        sock.settimeout(10.0)

        try:
            # 1. Negotiate protocol version
            sock.sendall(_rtde_pack(_RTDE_REQUEST_PROTOCOL_VERSION,
                                    struct.pack(">H", _RTDE_PROTOCOL_VERSION)))
            ptype, payload = _rtde_recv(sock)
            if ptype != _RTDE_REQUEST_PROTOCOL_VERSION or not payload or payload[0] != 1:
                raise RuntimeError("RTDE protocol version rejected")

            # 2. Setup output subscription
            recipe = ",".join(self.VARIABLES).encode()
            freq_payload = struct.pack(">d", self._frequency) + recipe
            sock.sendall(_rtde_pack(_RTDE_CONTROL_PACKAGE_SETUP_OUTPUTS, freq_payload))
            ptype, payload = _rtde_recv(sock)
            if ptype != _RTDE_CONTROL_PACKAGE_SETUP_OUTPUTS:
                raise RuntimeError("RTDE setup outputs failed")
            # payload[0] is recipe_id; rest is comma-separated type names
            recipe_id = payload[0]
            types_str = payload[1:].decode(errors="replace")
            if "NOT_FOUND" in types_str:
                raise RuntimeError(f"RTDE variable not found: {types_str}")

            # 3. Start streaming
            sock.sendall(_rtde_pack(_RTDE_CONTROL_PACKAGE_START))
            ptype, payload = _rtde_recv(sock)
            if ptype != _RTDE_CONTROL_PACKAGE_START or not payload or payload[0] != 1:
                raise RuntimeError("RTDE start failed")

            # 4. Read data packets
            sock.settimeout(5.0)
            while self._running:
                ptype, payload = _rtde_recv(sock)
                if ptype != _RTDE_DATA_PACKAGE:
                    continue
                # payload: 1 byte recipe_id + 18 doubles (actual_q + actual_TCP_pose + 6 registers)
                if len(payload) < 1 + 18 * 8:
                    continue
                values = struct.unpack_from(">18d", payload, 1)
                with self._lock:
                    self._joints    = list(values[:6])
                    self._tcp       = list(values[6:12])
                    self._registers = list(values[12:18])

        finally:
            sock.close()


# Global RTDE reader — started alongside ROS in main()
rtde: RTDEReader | None = None


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

    def is_motion_complete(self, tolerance_rad: float = 0.03) -> dict[str, Any]:
        with self._lock:
            current = list(self.current_joints) if self.current_joints is not None else None
            target = list(self.last_target_joints) if self.last_target_joints is not None else None
            t0 = self.last_command_time

        if current is None:
            return {"ok": False, "reason": "no_joint_state"}
        if target is None:
            return {"ok": True, "complete": True, "reason": "no_target"}

        diffs = [abs(c - t) for c, t in zip(current, target)]
        max_error = max(diffs)
        elapsed = None if t0 is None else round(time.time() - t0, 2)

        return {
            "ok": True,
            "complete": max_error <= tolerance_rad,
            "elapsed_sec": elapsed,
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
    Falls back to RTDE if ROS/External Control is not running.
    """
    r = require_robot()
    joints = r.get_current_joints()
    source = "ros"

    if joints is None and rtde is not None and rtde.connected:
        joints = rtde.joints
        source = "rtde"

    if joints is None:
        return {"ok": False, "message": "No joint state available (ROS and RTDE both unavailable)"}

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

    return {"ok": True, "source": source, "joints": entries}


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
def move_joints(
    joints: list[float],
    speed: float = 1.05,
    accel: float = 1.4,
    blend_radius: float = 0.0,
    wait_for_completion: bool = True,
    completion_timeout_sec: float = 30.0,
    completion_tolerance_rad: float = 0.03,
) -> dict[str, Any]:
    """
    Move directly in joint space using URScript movej() sent to port 30002.

    Sends the command directly to the robot's internal trajectory planner —
    does NOT use the ROS JointTrajectory controller, which conflicts with
    ExternalControl's servo stream and causes velocity-limit faults.

    joints                   – 6 joint angles in radians [base, shoulder, elbow, w1, w2, w3]
    speed                    – joint speed rad/s  (default 1.05, UR default is 1.05)
    accel                    – joint accel rad/s² (default 1.4,  UR default is 1.4)
    blend_radius             – blend radius in metres (default 0.0 = stop at waypoint).
                               Use >0 for transit waypoints so the arm flows smoothly
                               between consecutive moves without stopping.
    wait_for_completion      – if True (default), block until the arm reaches the target
                               before returning. Set False only when using blend_radius>0
                               in a manually-sequenced path.
    completion_timeout_sec   – how long to wait before giving up (default 30 s)
    completion_tolerance_rad – joint convergence threshold in rad (default 0.03)
    """
    if len(joints) != 6:
        return {"ok": False, "message": "Exactly 6 joint values are required"}

    target_check = check_joint_vector_limits(joints)
    if target_check["blocked"]:
        return {
            "ok": False,
            "message": "Target joint vector exceeds configured software limits",
            "issues": target_check["issues"],
        }

    j = [float(x) for x in joints]
    try:
        script = _send_movej(j, speed=speed, accel=accel, blend_radius=blend_radius)
    except Exception as e:
        return {"ok": False, "message": str(e)}

    if wait_for_completion and blend_radius == 0.0:
        deadline = time.time() + completion_timeout_sec
        while time.time() < deadline:
            time.sleep(0.1)
            # Prefer ROS joint state; fall back to RTDE
            current: list[float] | None = None
            r = robot
            if r is not None and r.current_joints is not None:
                with r._lock:
                    current = list(r.current_joints)
            elif rtde is not None and rtde.connected:
                current = rtde.joints
            if current is not None:
                diffs = [abs(c - t) for c, t in zip(current, j)]
                if max(diffs) <= completion_tolerance_rad:
                    break
        else:
            return {"ok": False, "message": f"Motion did not complete within {completion_timeout_sec}s"}

    return {
        "ok": True,
        "message": "movej complete" if wait_for_completion else "movej command sent",
        "target_joints_rad": [round(x, 5) for x in j],
        "speed_rad_s": speed,
        "accel_rad_s2": accel,
        "blend_radius_m": blend_radius,
        "script": script.strip(),
    }


@mcp.tool()
@safe_tool
def move_path(
    waypoints: list[dict[str, Any]],
) -> dict[str, Any]:
    """
    Execute a sequence of movej waypoints as a single URScript program.

    This is the key to smooth motion — the UR controller sees all waypoints
    at once and can blend between them using the blend radius (r parameter).
    Individual move_joints calls cannot blend because each is a separate script.

    waypoints – list of dicts, each with:
      joints: list[float]  – 6 joint angles in radians
      speed:  float        – rad/s (default 1.047)
      accel:  float        – rad/s² (default 1.396)
      blend_radius: float  – metres (default 0.0). Last waypoint must be 0.0.
    """
    import socket

    if not waypoints:
        return {"ok": False, "message": "No waypoints provided"}

    # Validate all waypoints first
    for i, wp in enumerate(waypoints):
        joints = wp.get("joints", [])
        if len(joints) != 6:
            return {"ok": False, "message": f"Waypoint {i}: exactly 6 joint values required"}
        check = check_joint_vector_limits(joints)
        if check["blocked"]:
            return {"ok": False, "message": f"Waypoint {i} exceeds limits", "issues": check["issues"]}

    # Force last waypoint blend_radius to 0 (UR requires it)
    waypoints[-1]["blend_radius"] = 0.0

    # Build a single URScript program with all waypoints
    lines = ["def move_path():"]
    for wp in waypoints:
        j = [float(x) for x in wp["joints"]]
        speed = float(wp.get("speed", 1.047))
        accel = float(wp.get("accel", 1.396))
        blend = float(wp.get("blend_radius", 0.0))
        r_part = f",r={blend:.4f}" if blend > 0 else ""
        lines.append(
            f"  movej([{j[0]:.6f},{j[1]:.6f},{j[2]:.6f},{j[3]:.6f},{j[4]:.6f},{j[5]:.6f}],"
            f"a={accel:.3f},v={speed:.3f}{r_part})"
        )
    lines.append("end")
    lines.append("")
    script = "\n".join(lines)

    try:
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(5.0)
        sock.connect((ROBOT_IP, 30002))
        sock.sendall(script.encode("utf-8"))
        sock.close()
    except Exception as e:
        return {"ok": False, "message": str(e)}

    # Track the final waypoint for is_motion_complete
    last_j = [float(x) for x in waypoints[-1]["joints"]]
    r = robot
    if r is not None:
        with r._lock:
            r.last_target_joints = last_j
            r.last_command_time = time.time()

    return {
        "ok": True,
        "message": f"move_path sent ({len(waypoints)} waypoints)",
        "waypoint_count": len(waypoints),
        "script_lines": len(lines),
    }


def _send_movej(joints: list[float], speed: float = 1.05, accel: float = 1.4, blend_radius: float = 0.0) -> str:
    """Internal helper: send URScript movej() to port 30002 and record target for is_motion_complete.

    blend_radius – blend radius in metres. >0 lets the arm blend into the next
    motion without decelerating to zero (smooth multi-waypoint paths).
    Use 0.0 for final pick/place positions where the arm must stop precisely.
    """
    import socket
    j = [float(x) for x in joints]
    r_part = f",r={blend_radius:.4f}" if blend_radius > 0 else ""
    script = (
        f"movej([{j[0]:.6f},{j[1]:.6f},{j[2]:.6f},{j[3]:.6f},{j[4]:.6f},{j[5]:.6f}],"
        f"a={accel:.3f},v={speed:.3f}{r_part})\n"
    )
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.settimeout(5.0)
    sock.connect((ROBOT_IP, 30002))
    sock.sendall(script.encode("utf-8"))
    sock.close()

    # Update motion tracking so is_motion_complete() reflects the latest command
    r = robot
    if r is not None:
        with r._lock:
            r.last_target_joints = j
            r.last_command_time = time.time()

    return script


@mcp.tool()
@safe_tool
def move_named_pose(
    name: str,
    speed: float = 1.05,
    accel: float = 1.4,
    via_home: bool = False,
    wait_for_completion: bool = True,
    completion_timeout_sec: float = 30.0,
    completion_tolerance_rad: float = 0.03,
) -> dict[str, Any]:
    """
    Move to a named pose using URScript movej() via port 30002.
    Optionally go via home first.

    By default, blocks until the pose is reached, like move_joints. Each
    command sent to port 30002 replaces the running program, so returning
    early let the next command (e.g. a gripper action) cancel this move.
    """
    if name not in POSES:
        return {"ok": False, "message": f"Unknown pose '{name}'", "available_poses": list(POSES.keys())}

    if via_home and name != "home":
        _send_movej(shortest_equivalent(POSES["home"], _current_joints()), speed=speed, accel=accel)
        # Wait for home to complete before moving to target
        deadline = time.time() + 15.0
        r = require_robot()
        while time.time() < deadline:
            time.sleep(0.2)
            if r.is_motion_complete().get("complete"):
                break

    target = shortest_equivalent(POSES[name], _current_joints())
    script = _send_movej(target, speed=speed, accel=accel)

    if wait_for_completion and not _wait_until_reached(
        target, completion_timeout_sec, completion_tolerance_rad
    ):
        return {"ok": False, "message": f"Motion to '{name}' did not complete within {completion_timeout_sec}s"}

    return {
        "ok": True,
        "message": "movej complete" if wait_for_completion else "movej command sent",
        "pose": name,
        "via_home": via_home,
        "target_joints_rad": [round(x, 5) for x in target],
        "speed_rad_s": speed,
        "script": script.strip(),
    }


@mcp.tool()
@safe_tool
def go_home(speed: float = 1.05, accel: float = 1.4) -> dict[str, Any]:
    """
    Convenience move to home pose.
    """
    return move_named_pose("home", speed=speed, accel=accel, via_home=False)


@mcp.tool()
@safe_tool
def unwind_joint(
    joint_index: int,
    target_rad: float = 0.0,
    speed: float = 0.5,
    accel: float = 0.5,
) -> dict[str, Any]:
    """
    Move one joint to target_rad, keeping the others fixed.
    Useful for recovery from winding. Uses slow speed by default.
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

    _send_movej(target, speed=speed, accel=accel)
    return {
        "ok": True,
        "message": "Single-joint unwind sent via movej",
        "joint": JOINTS[joint_index],
        "joint_index": joint_index,
        "target_rad": float(target_rad),
        "speed_rad_s": speed,
    }


@mcp.tool()
@safe_tool
def hold_position() -> dict[str, Any]:
    """
    Hold current position by sending a movej to the current joint angles.
    This is not an emergency stop.
    """
    r = require_robot()
    current = r.get_current_joints()
    if current is None:
        return {"ok": False, "message": "No joint state available"}

    _send_movej(current, speed=0.1, accel=0.1)
    return {
        "ok": True,
        "message": "Hold-position movej sent",
        "current_joints_rad": [round(x, 5) for x in current],
    }


@mcp.tool()
@safe_tool
def stop_motion(decel_rad_s2: float = 3.0) -> dict[str, Any]:
    """
    Stop the arm where it is: send URScript stopj() to brake all joints.
    Unlike hold_position, which moves back to the joint angles read while the
    arm was still moving, this doesn't move the arm again. Not an emergency stop.
    """
    import socket
    script = f"stopj({float(decel_rad_s2):.3f})\n"
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.settimeout(5.0)
    sock.connect((ROBOT_IP, 30002))
    sock.sendall(script.encode("utf-8"))
    sock.close()
    return {"ok": True, "message": "stopj sent", "decel_rad_s2": float(decel_rad_s2)}


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

    x, y, z      – target TCP position in robot base frame (metres)
    qx,qy,qz,qw  – target orientation as quaternion
    speed        – linear speed m/s (default 0.05 m/s – slow/safe)
    accel        – linear acceleration m/s² (default 0.3)
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
def get_inverse_kin(
    x: float, y: float, z: float,
    rx: float = 0.0, ry: float = 3.14159, rz: float = 0.0,
    q_near: list[float] | None = None,
) -> dict[str, Any]:
    """
    Ask the UR3e controller to solve IK for a TCP pose without moving.

    Sends URScript to port 30002 that calls get_inverse_kin() and writes each
    joint to output_double_register_0..5. Reads those registers back via a
    one-shot RTDE subscription — no motion, no parsing of binary primary packets.

    x, y, z    – target TCP position in robot base frame (metres)
    rx, ry, rz – rotation vector (axis * angle). Default [0, π, 0] = tool down.
    q_near     – 6-joint seed for configuration preference. Defaults to current
                 joint state so the solver picks the nearest configuration.
    """
    import socket as _socket

    # Use current joints as seed if none provided
    if q_near is None:
        joints_now: list[float] | None = None
        if rtde is not None and rtde.connected:
            joints_now = rtde.joints
        if joints_now is None and robot is not None:
            joints_now = robot.get_current_joints()
        q_near = joints_now or [0.0, -1.5708, 1.5708, -1.5708, -1.5708, 0.0]

    q_str = ",".join(f"{v:.6f}" for v in q_near)

    # URScript: compute IK, write each joint to an output double register
    script = (
        "def get_ik_result():\n"
        f"  local pose = p[{x:.6f},{y:.6f},{z:.6f},{rx:.6f},{ry:.6f},{rz:.6f}]\n"
        f"  local qn = [{q_str}]\n"
        "  local q = get_inverse_kin(pose, qnear=qn)\n"
        "  write_output_float_register(0, q[0])\n"
        "  write_output_float_register(1, q[1])\n"
        "  write_output_float_register(2, q[2])\n"
        "  write_output_float_register(3, q[3])\n"
        "  write_output_float_register(4, q[4])\n"
        "  write_output_float_register(5, q[5])\n"
        "end\n"
        "get_ik_result()\n"
    )

    if rtde is None or not rtde.connected:
        return {"ok": False, "message": "RTDE not connected"}

    # Send script via port 30002
    try:
        sock = _socket.socket(_socket.AF_INET, _socket.SOCK_STREAM)
        sock.settimeout(5.0)
        sock.connect((ROBOT_IP, 30002))
        sock.sendall(script.encode("utf-8"))
        sock.close()
    except Exception as e:
        return {"ok": False, "message": f"Failed to send script: {e}"}

    # Wait for URScript to execute and registers to update via the running RTDEReader
    time.sleep(0.5)
    result_joints = rtde.registers

    # Detect all-zero result (IK failed / unreachable)
    if all(abs(j) < 1e-9 for j in result_joints):
        return {"ok": False, "message": "Robot IK returned all zeros — pose likely unreachable"}

    wrapped = [round(((j + math.pi) % (2 * math.pi)) - math.pi, 6) for j in result_joints]
    return {
        "ok": True,
        "joints_rad": wrapped,
        "joints_deg": [round(math.degrees(j), 3) for j in wrapped],
        "tcp_pose": {"x": x, "y": y, "z": z, "rx": rx, "ry": ry, "rz": rz},
        "note": "Solved by robot controller's get_inverse_kin(). Use move_joints to execute.",
    }


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


@mcp.tool()
@safe_tool
def rtde_get_joints() -> dict[str, Any]:
    """
    Read actual joint positions directly via RTDE (port 30004).

    Works WITHOUT External Control running — use this while running a native
    UR program from the teach pendant to capture actual joint values at any
    waypoint (pause the program, then call this tool).

    Returns joint angles in radians in UR order:
      [base, shoulder, elbow, wrist_1, wrist_2, wrist_3]
    """
    if rtde is None or not rtde.connected:
        return {"ok": False, "message": "RTDE not connected — is the robot powered on and reachable?"}

    joints = rtde.joints
    if joints is None:
        return {"ok": False, "message": "No RTDE data yet"}

    entries = []
    for name, value in zip(JOINTS, joints):
        entries.append({
            "joint": name,
            "value_rad": round(value, 7),
            "value_deg": round(rad_to_deg(value), 4),
        })

    return {
        "ok": True,
        "source": "rtde",
        "joints": entries,
        "joints_rad": [round(v, 7) for v in joints],
        "note": "These are the robot's actual joint positions — safe to copy into YAML waypoints",
    }


@mcp.tool()
@safe_tool
def rtde_get_tcp() -> dict[str, Any]:
    """
    Read actual TCP pose directly via RTDE (port 30004).

    Works WITHOUT External Control running.
    Returns TCP pose as [x, y, z, rx, ry, rz] in robot base frame
    where rx/ry/rz is a rotation vector (axis * angle).
    """
    if rtde is None or not rtde.connected:
        return {"ok": False, "message": "RTDE not connected — is the robot powered on and reachable?"}

    tcp = rtde.tcp
    if tcp is None:
        return {"ok": False, "message": "No RTDE data yet"}

    return {
        "ok": True,
        "source": "rtde",
        "tcp_pose": {
            "x":  round(tcp[0], 6),
            "y":  round(tcp[1], 6),
            "z":  round(tcp[2], 6),
            "rx": round(tcp[3], 6),
            "ry": round(tcp[4], 6),
            "rz": round(tcp[5], 6),
        },
        "note": "Rotation vector (axis*angle). Matches UR script _p[x,y,z,rx,ry,rz] format.",
    }


@mcp.tool()
@safe_tool
def jog_joint(
    joint_index: int,
    direction: int,
    step: float = 0.02,
    speed: float = 0.3,
    accel: float = 0.5,
) -> dict[str, Any]:
    """
    Jog a single joint by one step in the given direction.

    Reads current joint positions via RTDE, applies the delta, checks soft
    limits, and sends movej().  Designed for rapid repeated calls from a
    keyboard jog interface — no LLM involved.

    joint_index  – 0=base, 1=shoulder, 2=elbow, 3=wrist_1, 4=wrist_2, 5=wrist_3
    direction    – +1 or -1
    step         – step size in radians (default 0.02)
    speed        – movej speed in rad/s (default 0.3)
    accel        – movej accel in rad/s² (default 0.5)
    """
    import socket as _socket

    if not 0 <= joint_index <= 5:
        return {"ok": False, "message": "joint_index must be 0–5"}
    if direction not in (1, -1):
        return {"ok": False, "message": "direction must be +1 or -1"}
    if not 0.001 <= step <= 0.50:
        return {"ok": False, "message": "step must be between 0.001 and 0.50 rad"}

    # Read current joints — prefer RTDE (works without External Control)
    joints: list[float] | None = None
    source = "none"
    if rtde is not None and rtde.connected:
        joints = rtde.joints
        source = "rtde"
    if joints is None and robot is not None:
        joints = robot.get_current_joints()
        source = "ros"
    if joints is None:
        return {"ok": False, "message": "No joint data available (RTDE and ROS both unavailable)"}

    target = list(joints)
    target[joint_index] += direction * step

    # Soft limit check
    limit_check = check_joint_vector_limits(target)
    if limit_check["blocked"]:
        return {
            "ok": False,
            "message": f"Joint {joint_index} soft limit reached — move blocked",
            "issues": limit_check["issues"],
            "current_joints": [round(x, 5) for x in joints],
        }

    # Send movej
    j = target
    script = (
        f"movej([{j[0]:.6f},{j[1]:.6f},{j[2]:.6f},"
        f"{j[3]:.6f},{j[4]:.6f},{j[5]:.6f}],"
        f"a={accel:.3f},v={speed:.3f})\n"
    )
    sock = _socket.socket(_socket.AF_INET, _socket.SOCK_STREAM)
    sock.settimeout(3.0)
    sock.connect((ROBOT_IP, 30002))
    sock.sendall(script.encode())
    sock.close()

    # Update motion tracking
    r = robot
    if r is not None:
        with r._lock:
            r.last_target_joints = target
            r.last_command_time = time.time()

    return {
        "ok": True,
        "joint_index": joint_index,
        "joint_name": JOINTS[joint_index],
        "direction": direction,
        "step_rad": step,
        "new_value_rad": round(target[joint_index], 6),
        "joints_rad": [round(x, 6) for x in target],
        "source": source,
    }


@mcp.custom_route("/health", methods=["GET"])
async def health_check(request):
    return {
        "ok": True,
        "server": "ur3e-arm",
        "ros_ready": _ros_ready.is_set(),
        "robot_connected": robot is not None and robot.current_joints is not None if robot else False,
    }


def main() -> None:
    global rtde

    # Start RTDE passive reader — works independently of ROS/External Control
    rtde = RTDEReader(robot_ip=ROBOT_IP, frequency=10.0)
    rtde.start()
    print(f"RTDE reader started (connecting to {ROBOT_IP}:{RTDE_PORT})...", flush=True)

    ros_thread = threading.Thread(target=ros_spin_thread, daemon=True)
    ros_thread.start()

    _ros_ready.wait(timeout=10.0)

    print("Waiting for robot joint states...", flush=True)
    for _ in range(50):
        time.sleep(0.2)
        ros_ok = robot is not None and robot.current_joints is not None
        rtde_ok = rtde.connected
        if ros_ok or rtde_ok:
            src = []
            if ros_ok:  src.append("ROS")
            if rtde_ok: src.append("RTDE")
            print(f"Robot connected via: {', '.join(src)}", flush=True)
            break
    else:
        print("WARNING: No joint states yet from ROS or RTDE. Is the robot powered on?", flush=True)

    print(f"Starting MCP HTTP server on http://{SERVER_HOST}:{SERVER_PORT}/mcp", flush=True)
    mcp.run(transport="streamable-http")


if __name__ == "__main__":
    main()