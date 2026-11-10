"""UR3e DH forward kinematics and closed-form analytical IK (8-solution).

The analytical solver derives all solutions exactly from the DH geometry —
no numerical optimisation, no seed coverage gaps, instant execution.

Algorithm:
  1. theta1  (2 solutions) — from wrist-centre XY position
  2. theta5  (2 solutions) — from TCP approach-vector orientation
  3. theta6  (determined)  — from remaining orientation components
  4. theta2, theta3 (2 solutions) — 2R planar sub-problem on T14
  5. theta4  (determined)  — from residual rotation in T14

Total: up to 8 candidate configurations per target pose.
"""

from __future__ import annotations

import math
import os

import numpy as np

try:
    import logfire as _logfire
except ImportError:
    _logfire = None  # type: ignore[assignment]

APPROACH_Z_OFFSET = 0.12   # metres above object for pre-grasp approach
PICK_Z_OFFSET     = 0.02   # metres above detected surface for grasp
LIFT_Z_OFFSET     = 0.18   # metres above object after grasping

# DH parameters: (d, a, alpha) — theta is the joint variable
_DH = [
    (0.15185,  0.0,      math.pi / 2),
    (0.0,     -0.24355,  0.0),
    (0.0,     -0.2132,   0.0),
    (0.13105,  0.0,      math.pi / 2),
    (0.08535,  0.0,     -math.pi / 2),
    (0.0921,   0.0,      0.0),
]

# Joint limits (rad), matching the arm MCP server's software limits.
_JOINT_LIMITS = [(-6.0, 6.0), (-6.0, 6.0), (-3.0, 3.0), (-6.0, 6.0), (-6.0, 6.0), (-6.0, 6.0)]

# Weights for comparing IK candidates with the seed. Turning the base or
# rolling the last wrist joint doesn't change the arm's shape, so differences
# there count much less. Equal weights made the solver prefer contorted
# configurations (shoulder below horizontal, wrist folded back) just to save
# base rotation: 25% of reachable targets with the pick seeds, 4% after.
_SEED_WEIGHTS = (0.05, 1.0, 1.0, 1.0, 1.0, 0.2)

# Reference config used to prefer "natural" arm orientation when no seed given
_PICK_SEED       = [ 1.151908, -1.467556, -1.403377, -1.746785,  1.567782, 3.172426]
_PICK_SEED_LEFT  = [-1.151908, -1.467556, -1.403377, -1.746785,  1.567782, 3.172426]
_HOME_SEED       = [-3.16018,  -0.6864,   -1.4132,   -2.0002,    1.63811,  3.17233]


def _auto_seed(y: float, q_seed: list[float] | None) -> list[float]:
    """Return q_seed if given, else pick a side-appropriate default seed.

    Using the robot's current joint state as IK seed (the default when no
    q_near is passed) causes intermittent failures: after a move that left
    the arm in a non-standard pose, the robot IK finds a 'close' solution
    for the new target that requires extreme wrist compensation.  Always
    seeding from a known-good reference configuration avoids this.
    """
    if q_seed is not None:
        return q_seed
    return _PICK_SEED_LEFT if y > 0.05 else _PICK_SEED


def _dh_matrix(d: float, a: float, alpha: float, theta: float) -> np.ndarray:
    ct, st = math.cos(theta), math.sin(theta)
    ca, sa = math.cos(alpha), math.sin(alpha)
    return np.array([
        [ct, -st * ca,  st * sa, a * ct],
        [st,  ct * ca, -ct * sa, a * st],
        [0,   sa,       ca,      d     ],
        [0,   0,        0,       1     ],
    ])


def ur3e_fk(joints: list[float]) -> np.ndarray:
    """UR3e forward kinematics → 4×4 TCP transform in robot base frame."""
    T = np.eye(4)
    for i, (d, a, alpha) in enumerate(_DH):
        T = T @ _dh_matrix(d, a, alpha, joints[i])
    return T


def _tool_down_T(x: float, y: float, z: float, roll: float) -> np.ndarray:
    """4×4 target with tool Z pointing down, rotated by roll about that Z."""
    cr, sr = math.cos(roll), math.sin(roll)
    # x-col=[cr,sr,0]  y-col=[sr,-cr,0]  z-col=[0,0,-1]  (valid rotation, det=+1)
    return np.array([
        [cr,  sr,  0.0,  x],
        [sr, -cr,  0.0,  y],
        [0.0, 0.0, -1.0, z],
        [0.0, 0.0,  0.0, 1.0],
    ])


def _ur3e_ik_all(T_target: np.ndarray) -> list[list[float]]:
    """Return all analytical IK solutions (up to 8) for the given 4×4 target."""
    d1 = _DH[0][0]; d4 = _DH[3][0]; d5 = _DH[4][0]; d6 = _DH[5][0]
    a2 = _DH[1][1]; a3 = _DH[2][1]

    px, py = T_target[0, 3], T_target[1, 3]
    ax, ay = T_target[0, 2], T_target[1, 2]   # approach vector (z-col of R)
    nx, ny = T_target[0, 0], T_target[1, 0]   # normal    (x-col of R)
    ox, oy = T_target[0, 1], T_target[1, 1]   # orient    (y-col of R)

    # ── theta1 (2 solutions) ──────────────────────────────────────────────────
    wx = px - d6 * ax
    wy = py - d6 * ay
    r_wxy = math.hypot(wx, wy)
    if r_wxy < abs(d4):
        return []

    disc = math.sqrt(r_wxy ** 2 - d4 ** 2)

    solutions: list[list[float]] = []

    for sign1 in (+1, -1):
        theta1 = math.atan2(wy, wx) + math.atan2(d4, sign1 * disc)
        c1, s1 = math.cos(theta1), math.sin(theta1)

        # ── theta5 (2 solutions) ─────────────────────────────────────────────
        cos5 = max(-1.0, min(1.0, s1 * ax - c1 * ay))

        for sign5 in (+1, -1):
            theta5 = sign5 * math.acos(cos5)
            s5 = math.sin(theta5)

            # ── theta6 ───────────────────────────────────────────────────────
            if abs(s5) < 1e-8:
                theta6 = 0.0   # wrist singularity — free variable
            else:
                c6 =  (s1 * nx - c1 * ny) / s5
                s6 = -(s1 * ox - c1 * oy) / s5
                theta6 = math.atan2(s6, c6)

            # ── T14 = inv(T01) * T_target * inv(T45 * T56) ──────────────────
            T01 = _dh_matrix(d1, 0.0,  math.pi / 2, theta1)
            T45 = _dh_matrix(d5, 0.0, -math.pi / 2, theta5)
            T56 = _dh_matrix(d6, 0.0,  0.0,         theta6)

            T16 = np.linalg.inv(T01) @ T_target
            T14 = T16 @ np.linalg.inv(T45 @ T56)

            px4, py4 = T14[0, 3], T14[1, 3]

            # ── theta3  (2R planar elbow) ─────────────────────────────────────
            r2   = px4 ** 2 + py4 ** 2
            c3v  = (r2 - a2 ** 2 - a3 ** 2) / (2.0 * a2 * a3)
            if abs(c3v) > 1.0 + 1e-8:
                continue
            c3v = max(-1.0, min(1.0, c3v))

            for sign3 in (+1, -1):
                s3v    = sign3 * math.sqrt(max(0.0, 1.0 - c3v ** 2))
                theta3 = math.atan2(s3v, c3v)

                # ── theta2 ───────────────────────────────────────────────────
                theta2 = math.atan2(py4, px4) - math.atan2(a3 * s3v, a2 + a3 * c3v)

                # ── theta4  (from residual R14) ───────────────────────────────
                # R1234 = [[c234,0,s234],[s234,0,-c234],[0,1,0]]
                # → t234 = atan2(R14[1,0], R14[0,0])
                t234   = math.atan2(T14[1, 0], T14[0, 0])
                theta4 = t234 - theta2 - theta3

                solutions.append([theta1, theta2, theta3, theta4, theta5, theta6])

    return solutions


def ur3e_ik(x: float, y: float, z: float,
            q_seed: list[float] | None = None) -> list[float] | None:
    """
    Closed-form analytical IK for UR3e (tool pointing down, any tool roll).

    Enumerates all 8 kinematic configurations for four canonical tool-roll
    orientations (0°, 90°, 180°, 270°) → up to 32 exact candidates.
    Candidates are filtered by position error < 10 mm, then ranked by:
      • distance to q_seed (if provided) — keeps same arm configuration
      • otherwise position error alone
    """
    p_target = np.array([x, y, z])
    _PI = math.pi
    q_seed = _auto_seed(y, q_seed)

    candidates: list[tuple[float, float, list[float]]] = []

    for roll in (0.0, _PI / 2, _PI, 3 * _PI / 2):
        T = _tool_down_T(x, y, z, roll)
        for sol in _ur3e_ik_all(T):
            T_fk  = ur3e_fk(sol)
            p_err = float(np.linalg.norm(T_fk[:3, 3] - p_target))
            if p_err > 0.010:
                continue
            seed_dist = sum(
                w * (((a - b + _PI) % (2 * _PI) - _PI) ** 2)
                for w, a, b in zip(_SEED_WEIGHTS, sol, q_seed)
            )
            candidates.append((seed_dist, p_err, sol))

    if not candidates:
        print(f"  [IK] No solution for ({x:.3f},{y:.3f},{z:.3f}) — "
              "position may be out of reach.")
        return None

    candidates.sort(key=lambda c: (c[0], c[1]))
    best = candidates[0][2]
    wrapped = [((j + _PI) % (2 * _PI)) - _PI for j in best]
    # After normalising to [-π, π], nudge each joint ±2π toward the seed so
    # the arm stays in the same wrist/elbow branch it was already in, but
    # never outside the joint limits (a wound seed used to give e.g. -7.24).
    if q_seed:
        for i in range(len(wrapped)):
            lo, hi = _JOINT_LIMITS[i]
            for offset in (2 * _PI, -2 * _PI):
                cand = wrapped[i] + offset
                if lo <= cand <= hi and abs(cand - q_seed[i]) < abs(wrapped[i] - q_seed[i]):
                    wrapped[i] = cand
    return [round(j, 6) for j in wrapped]


def robot_ik(x: float, y: float, z: float,
             arm_mcp_url: str,
             q_seed: list[float] | None = None,
             rx: float = 0.0, ry: float = math.pi, rz: float = 0.0) -> list[float] | None:
    """
    IK via the robot controller's get_inverse_kin (MCP tool).
    Falls back to local analytical IK if the MCP call fails.
    """
    import urllib.request
    import json as _json
    import struct as _struct

    # MCP session handshake + tool call
    base_url = arm_mcp_url.rstrip("/")
    headers  = {"Content-Type": "application/json", "Accept": "application/json, text/event-stream"}

    def _post(payload: dict, session_id: str | None = None) -> tuple[str | None, dict]:
        req_headers = dict(headers)
        if session_id:
            req_headers["mcp-session-id"] = session_id
        data = _json.dumps(payload).encode()
        req  = urllib.request.Request(base_url, data=data, headers=req_headers, method="POST")
        with urllib.request.urlopen(req, timeout=10) as resp:
            sid = resp.headers.get("mcp-session-id")
            raw = resp.read().decode()
        # Parse SSE data line
        for line in raw.splitlines():
            if line.startswith("data:"):
                return sid, _json.loads(line[5:].strip())
        return sid, {}

    try:
        # 1. Initialize session
        sid, _ = _post({"jsonrpc": "2.0", "method": "initialize",
                        "params": {"protocolVersion": "2024-11-05",
                                   "capabilities": {},
                                   "clientInfo": {"name": "ik-client", "version": "1.0"}},
                        "id": 0})

        # 2. Call get_inverse_kin
        args: dict = {"x": x, "y": y, "z": z, "rx": rx, "ry": ry, "rz": rz}
        if q_seed:
            args["q_near"] = q_seed
        _, resp = _post({"jsonrpc": "2.0", "method": "tools/call",
                         "params": {"name": "get_inverse_kin", "arguments": args},
                         "id": 1}, session_id=sid)

        result = resp.get("result", {}).get("structuredContent", {})
        if result.get("ok") and result.get("joints_rad"):
            joints = result["joints_rad"]
            _ik_trace(f"    [robot IK] joints={[round(j,4) for j in joints]}")
            return joints

        print(f"    [robot IK] failed: {result.get('message','unknown')} — falling back to local")
    except Exception as e:
        print(f"    [robot IK] error: {e} — falling back to local")

    return ur3e_ik(x, y, z, q_seed)


_ROBOT_IK_ERR_THRESHOLD = 0.050   # metres — robot IK results above this fall back to local

# Per-move IK lines ("approach (x, y, z) → joints=[…]") flood the REPL while
# a task's progress is printing; show them with COBOT_IK_VERBOSE=1. IK
# failures and fallbacks always print.
_IK_VERBOSE = os.getenv("COBOT_IK_VERBOSE", "") not in ("", "0", "false")


def _ik_trace(msg: str) -> None:
    if _IK_VERBOSE:
        print(msg)

def get_joints_for_pose(name: str, x: float, y: float, z: float,
                        q_seed: list[float] | None = None,
                        arm_mcp_url: str | None = None) -> list[float] | None:
    """Solve IK for a named pose. Uses robot MCP if arm_mcp_url provided, else local.
    If robot IK error exceeds threshold, falls back to local analytical IK."""
    _ik_trace(f"  {name}  ({x:.3f}, {y:.3f}, {z:.3f})")
    q_seed = _auto_seed(y, q_seed)
    solver_used = "local"
    robot_ik_fallback = False

    if arm_mcp_url:
        solver_used = "robot"
        joints = robot_ik(x, y, z, arm_mcp_url, q_seed)
        if joints:
            T   = ur3e_fk(joints)
            err = float(np.linalg.norm(T[:3, 3] - np.array([x, y, z])))
            if err > _ROBOT_IK_ERR_THRESHOLD:
                print(f"    [robot IK] err={err*1000:.1f}mm > threshold, falling back to local IK")
                robot_ik_fallback = True
                solver_used = "local"
                joints = ur3e_ik(x, y, z, q_seed)
            else:
                _ik_trace(f"    [robot IK] joints={[round(j, 4) for j in joints]}  "
                          f"err={err * 1000:.1f}mm  z_tool={T[:3, 2].round(3).tolist()}")
                if _logfire:
                    _logfire.info("ik.solved", pose_name=name, x=x, y=y, z=z,
                                  solver_used=solver_used, fk_error_mm=round(err * 1000, 1),
                                  robot_ik_fallback=False)
                return joints
    else:
        joints = ur3e_ik(x, y, z, q_seed)

    if joints:
        T   = ur3e_fk(joints)
        err = float(np.linalg.norm(T[:3, 3] - np.array([x, y, z])))
        _ik_trace(f"    → joints={[round(j, 4) for j in joints]}  "
                  f"err={err * 1000:.1f}mm  z_tool={T[:3, 2].round(3).tolist()}")
        if _logfire:
            _logfire.info("ik.solved", pose_name=name, x=x, y=y, z=z,
                          solver_used=solver_used, fk_error_mm=round(err * 1000, 1),
                          robot_ik_fallback=robot_ik_fallback)
    else:
        if _logfire:
            _logfire.warning("ik.no_solution", pose_name=name, x=x, y=y, z=z,
                             solver_used=solver_used)
    return joints
