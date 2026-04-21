"""UR3e DH forward kinematics and numerical IK (scipy L-BFGS-B)."""

from __future__ import annotations

import math
import sys

import numpy as np

APPROACH_Z_OFFSET = 0.12   # metres above object for pre-grasp approach
PICK_Z_OFFSET     = 0.04   # metres above detected surface for grasp

# DH parameters: (d, a, alpha) — theta is the joint variable
_DH = [
    (0.15185,  0.0,      math.pi / 2),
    (0.0,     -0.24355,  0.0),
    (0.0,     -0.2132,   0.0),
    (0.13105,  0.0,      math.pi / 2),
    (0.08535,  0.0,     -math.pi / 2),
    (0.0921,   0.0,      0.0),
]

# Known-working configuration seeds
_PICK_SEED      = [1.151908, -1.467556, -1.403377, -1.746785,  1.567782, 3.172426]
_HOME_SEED      = [-3.16018, -0.6864,   -1.4132,   -2.0002,    1.63811,  3.17233]
_FAR_REACH_SEED = [ 1.5631,  -2.8851,    0.4758,   -2.2817,    1.517,    0.0]    # y≈0.48m


def _dh_matrix(d: float, a: float, alpha: float, theta: float) -> np.ndarray:
    ct, st = math.cos(theta), math.sin(theta)
    ca, sa = math.cos(alpha), math.sin(alpha)
    return np.array([
        [ct, -st * ca,  st * sa, a * ct],
        [st,  ct * ca, -ct * sa, a * st],
        [0,   sa,       ca,      d],
        [0,   0,        0,       1],
    ])


def ur3e_fk(joints: list[float]) -> np.ndarray:
    """UR3e forward kinematics → 4×4 TCP transform in robot base frame."""
    T = np.eye(4)
    for i, (d, a, alpha) in enumerate(_DH):
        T = T @ _dh_matrix(d, a, alpha, joints[i])
    return T


def _generate_seeds(x: float, y: float, z: float,
                    q_hint: list[float] | None) -> list[list[float]]:
    """Dense grid seeds over theta1 (every 20°) × several elbow combos."""
    seeds = []
    for t1_deg in range(-180, 180, 20):
        t1 = math.radians(t1_deg)
        for t2 in [-0.8, -1.2, -1.6, -2.0, -2.4, -2.8]:
            for t3 in [-0.8, -1.2, -1.6, -2.0, 0.2, 0.5, 0.8]:
                seeds.append([t1, t2, t3, -1.5, math.pi / 2, 0.0])
    if q_hint:
        seeds.insert(0, q_hint[:])
    seeds.insert(0, _FAR_REACH_SEED[:])
    seeds.insert(0, _PICK_SEED[:])
    seeds.insert(0, _HOME_SEED[:])
    return seeds


def _perturb_seeds(q_seed: list[float]) -> list[list[float]]:
    """Perturbations around a known-good seed — keeps the same arm config."""
    seeds = [q_seed[:]]
    ranges = [
        [-0.05, 0.05, -0.15, 0.15],
        [-0.1, 0.1, -0.2, 0.2, -0.4, 0.4],
        [-0.1, 0.1, -0.2, 0.2, -0.4, 0.4],
        [-0.1, 0.1, -0.2, 0.2, -0.4, 0.4],
        [-0.05, 0.05],
        [-0.05, 0.05],
    ]
    for i, deltas in enumerate(ranges):
        for delta in deltas:
            s = q_seed[:]
            s[i] += delta
            seeds.append(s)
    for ds in [-0.2, -0.4]:
        for de in [0.2, 0.4]:
            s = q_seed[:]
            s[1] += ds
            s[2] += de
            s[3] += ds
            seeds.append(s)
    return seeds


def ur3e_ik(x: float, y: float, z: float,
            q_seed: list[float] | None = None) -> list[float] | None:
    """
    Numerical IK for UR3e using scipy L-BFGS-B with joint bounds ±2π.

    If q_seed is provided only perturbations of that seed are tried so the arm
    stays in the same kinematic configuration (no config flips).
    Without q_seed a dense grid over all shoulder-pan angles is used.
    """
    try:
        from scipy.optimize import minimize
    except ImportError:
        print("ERROR: scipy not installed.  Run:  pip install scipy")
        sys.exit(1)

    p_target = np.array([x, y, z])
    z_down   = np.array([0.0, 0.0, -1.0])
    reg_w    = 1.0 if q_seed else 0.5
    bounds   = [(-2 * math.pi, 2 * math.pi)] * 6

    def cost(q: np.ndarray, seed: list[float]) -> float:
        T  = ur3e_fk(q.tolist())
        dp = T[:3, 3] - p_target
        dq = q - np.array(seed)
        return (float(np.dot(dp, dp)) * 2000.0
                + float(np.dot(T[:3, 2] - z_down, T[:3, 2] - z_down)) * 30.0
                + float(np.dot(dq, dq)) * reg_w)

    seeds     = _perturb_seeds(q_seed) if q_seed else _generate_seeds(x, y, z, None)
    best_q, best_err = None, 1e9

    for seed in seeds:
        res = minimize(lambda q: cost(q, seed), seed, method="L-BFGS-B",
                       bounds=bounds,
                       options={"maxiter": 3000, "ftol": 1e-15, "gtol": 1e-10})
        T   = ur3e_fk(res.x.tolist())
        err = float(np.linalg.norm(T[:3, 3] - p_target))
        if err < best_err:
            best_err = err
            best_q   = res.x.tolist()
        if best_err < 0.001:
            break

    if best_err < 0.010:
        wrapped = [((j + math.pi) % (2 * math.pi)) - math.pi for j in best_q]
        return [round(j, 6) for j in wrapped]

    print(f"  [IK] No solution (pos_err={best_err * 1000:.1f}mm). "
          f"Position ({x:.3f},{y:.3f},{z:.3f}) may be out of reach.")
    return None


def get_joints_for_pose(name: str, x: float, y: float, z: float,
                        q_seed: list[float] | None = None) -> list[float] | None:
    print(f"  {name}  ({x:.3f}, {y:.3f}, {z:.3f})")
    joints = ur3e_ik(x, y, z, q_seed)
    if joints:
        T   = ur3e_fk(joints)
        err = float(np.linalg.norm(T[:3, 3] - np.array([x, y, z])))
        print(f"    → joints={[round(j, 4) for j in joints]}  "
              f"err={err * 1000:.1f}mm  z_tool={T[:3, 2].round(3).tolist()}")
    return joints
