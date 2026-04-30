"""Hand-eye calibration: SVD least-squares solve for T_base_cam."""

from __future__ import annotations

import asyncio
import json
import os
import shutil

import cv2
import numpy as np
from google import genai

from cobot.agents.vision_agent.camera import (
    capture_realsense, encode_jpeg, depth_at_pixel, pixel_to_camera_xyz,
    CALIB_FILE,
)
from cobot.agents.vision_agent.gemini import gemini_locate_gripper
ARM_MCP_URL = os.getenv("ARM_MCP_URL", "http://172.31.0.77:8765/mcp")

CALIB_MOVE_SPEED = 0.2
CALIB_MOVE_ACCEL = 0.3
CALIB_SETTLE_SEC = 1.5

CALIB_JOINTS = [
    ("pose_A", [1.151908, -1.467556, -1.403377, -1.746785, 1.567782, 3.172426]),
    ("pose_B", [-3.16018, -0.6864,   -1.4132,   -2.0002,   1.63811,  3.17233]),
    ("pose_C", [0.8,      -1.2,      -1.5,      -1.8,      1.5708,   3.14]),
    ("pose_D", [1.5,      -1.5,      -1.2,      -1.8,      1.5708,   3.14]),
]


def _solve_T_base_cam(robot_pts: np.ndarray, camera_pts: np.ndarray) -> np.ndarray:
    """SVD least-squares rigid-body fit: base_pt = R @ cam_pt + t"""
    assert robot_pts.shape == camera_pts.shape and robot_pts.shape[0] >= 3
    mu_r = robot_pts.mean(axis=0)
    mu_c = camera_pts.mean(axis=0)
    H    = (camera_pts - mu_c).T @ (robot_pts - mu_r)
    U, _, Vt = np.linalg.svd(H)
    R = Vt.T @ U.T
    if np.linalg.det(R) < 0:
        Vt[-1, :] *= -1
        R = Vt.T @ U.T
    t      = mu_r - R @ mu_c
    T      = np.eye(4)
    T[:3, :3] = R
    T[:3, 3]  = t
    return T


async def calibrate_handeye(
    arm_mcp_url: str = ARM_MCP_URL,
    calib_file: str = CALIB_FILE,
    check_only: bool = False,
    dry_run: bool = False,
    arm_mcp_client=None,
) -> dict:
    """
    Run or verify hand-eye calibration.

    Moves the robot to CALIB_JOINTS poses, captures RealSense frames, asks
    Gemini ER to locate the gripper tip, and SVD-solves T_base_cam.

    Args:
        arm_mcp_url:    Streamable-HTTP URL of the arm MCP server (used when
                        arm_mcp_client is None — standalone script mode).
        calib_file:     Path to calibration.json to read/update.
        check_only:     If True, report errors without saving.
        dry_run:        If True, skip robot movement (camera + Gemini only).
        arm_mcp_client: Pre-connected MCPClient (used from CLI /calibrate).

    Returns dict with keys: success, n_points, mean_err_mm, max_err_mm,
    T_base_cam, message.
    """
    google_key = os.environ.get("GOOGLE_API_KEY")
    if not google_key:
        return {"success": False, "message": "GOOGLE_API_KEY not set"}

    gemini = genai.Client(api_key=google_key)

    if arm_mcp_client is not None:
        # Called from CLI — reuse the existing MCPClient connection
        async def call_mcp(tool: str, args: dict) -> dict:
            return await arm_mcp_client.call_tool(tool, args)

        return await _run_calibration(call_mcp, calib_file, check_only, dry_run, gemini)

    # Standalone script mode — open a fresh MCP connection
    from mcp import ClientSession
    from mcp.client.streamable_http import streamable_http_client

    async with streamable_http_client(arm_mcp_url) as (r, w, _):
        async with ClientSession(r, w) as session:
            await session.initialize()
            print(f"[calib] Connected to arm MCP: {arm_mcp_url}")

            async def call_mcp(tool: str, args: dict) -> dict:
                result = await session.call_tool(tool, args)
                if result.content and result.content[0].type == "text":
                    try:
                        return json.loads(result.content[0].text)
                    except Exception:
                        return {"raw": result.content[0].text}
                if result.structuredContent:
                    return dict(result.structuredContent)
                return {"ok": False, "message": "empty response"}

            return await _run_calibration(call_mcp, calib_file, check_only, dry_run, gemini)


async def _run_calibration(call_mcp, calib_file: str, check_only: bool, dry_run: bool, gemini) -> dict:
    """Core calibration logic — shared between CLI and standalone script modes."""
    with open(calib_file) as f:
        calib = json.load(f)
    T_old = np.array(calib["T_base_cam"])

    print(f"[calib] Loaded: {calib_file}")
    print(f"  Current translation: {T_old[:3, 3].round(4).tolist()}\n")

    robot_pts:  list[np.ndarray] = []
    camera_pts: list[np.ndarray] = []

    for name, joints in CALIB_JOINTS:
        print(f"── {name} {'(dry run)' if dry_run else ''}")

        if not dry_run:
            print(f"  Moving to joints: {[round(j, 3) for j in joints]}")
            result = await call_mcp("move_joints", {
                "joints": joints, "speed": CALIB_MOVE_SPEED,
                "accel": CALIB_MOVE_ACCEL, "wait": True,
            })
            print(f"  move_joints → {result}")
            print(f"  Settling {CALIB_SETTLE_SEC}s...")
            await asyncio.sleep(CALIB_SETTLE_SEC)

        tcp = await call_mcp("get_tcp_pose", {})
        print(f"  TCP from robot: {tcp}")
        if not tcp.get("ok"):
            print(f"  ERROR: get_tcp_pose failed — {tcp.get('message', 'unknown error')}")
            print("  Hint: set BASE_FRAME env var on the arm MCP server to match your ROS2 TF tree (e.g. base_link, ur3e_base)")
            continue
        pos      = tcp["position_m"]
        tcp_base = np.array([pos["x"], pos["y"], pos["z"]])

        print("  Capturing RealSense frame...")
        color, depth, ci, ds = capture_realsense()

        print("  Asking Gemini ER to locate gripper...")
        pix = gemini_locate_gripper(gemini, encode_jpeg(color))
        if pix is None:
            print("  WARNING: Gemini could not find gripper — skipping\n")
            cv2.imwrite(f"calib_{name}_NOFIND.jpg", color)
            continue

        px, py = pix
        d_m    = depth_at_pixel(depth, ci, px, py, ds)
        if d_m < 0.05:
            print(f"  WARNING: depth invalid ({d_m:.3f}m) — skipping\n")
            cv2.imwrite(f"calib_{name}_NODEPTH.jpg", color)
            continue

        cam_xyz = pixel_to_camera_xyz(ci, px, py, d_m)[:3]
        mapped  = (T_old @ np.append(cam_xyz, 1.0))[:3]
        err     = np.linalg.norm(mapped - tcp_base)

        print(f"  Gripper pixel: ({px}, {py})   depth: {d_m:.3f}m")
        print(f"  Camera-frame XYZ:   {cam_xyz.round(4).tolist()}")
        print(f"  Robot-reported TCP: {tcp_base.round(4).tolist()}")
        print(f"  Old-calib mapped:   {mapped.round(4).tolist()}")
        print(f"  Calibration error (old): {err * 1000:.1f} mm\n")

        robot_pts.append(tcp_base)
        camera_pts.append(cam_xyz)

        ann = color.copy()
        cv2.circle(ann, (px, py), 14, (0, 255, 0), 3)
        cv2.putText(ann, f"{name}  {d_m:.2f}m", (px + 14, py),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        cv2.imwrite(f"calib_{name}.jpg", ann)

    n = len(robot_pts)
    print(f"\nCollected {n} valid correspondences")

    if n < 3:
        return {"success": False, "n_points": n,
                "message": "Need at least 3 valid poses to solve T_base_cam."}

    if check_only:
        return {"success": True, "n_points": n,
                "message": "Check-only mode — calibration not updated."}

    R_pts = np.array(robot_pts)
    C_pts = np.array(camera_pts)
    T_new = _solve_T_base_cam(R_pts, C_pts)

    residuals  = [np.linalg.norm((T_new @ np.append(c, 1.0))[:3] - r)
                  for r, c in zip(R_pts, C_pts)]
    mean_err   = float(np.mean(residuals)) * 1000
    max_err    = float(np.max(residuals))  * 1000

    print(f"\nNew T_base_cam:\n{T_new.round(6)}")
    print(f"Residuals: mean={mean_err:.1f}mm  max={max_err:.1f}mm")

    backup = calib_file.replace(".json", "_backup.json")
    shutil.copy(calib_file, backup)
    print(f"Backed up old calibration → {backup}")

    calib["T_base_cam"]         = T_new.tolist()
    calib["calibrated_at"]      = __import__("datetime").datetime.utcnow().isoformat() + "Z"
    calib["calibration_method"] = "handeye_gemini_er"
    calib["n_points"]           = n

    with open(calib_file, "w") as f:
        json.dump(calib, f, indent=2)
    print(f"Saved new calibration → {calib_file}")

    return {
        "success":     True,
        "n_points":    n,
        "mean_err_mm": round(mean_err, 2),
        "max_err_mm":  round(max_err,  2),
        "T_base_cam":  T_new.tolist(),
        "message":     f"Calibration updated. Mean error {mean_err:.1f}mm.",
    }


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Hand-eye calibration for cobot-claw")
    parser.add_argument("--arm-mcp-url", default=ARM_MCP_URL)
    parser.add_argument("--check-only", action="store_true", help="Report errors without saving")
    parser.add_argument("--dry-run",    action="store_true", help="Skip robot movement")
    args = parser.parse_args()

    result = asyncio.run(calibrate_handeye(
        arm_mcp_url=args.arm_mcp_url,
        check_only=args.check_only,
        dry_run=args.dry_run,
    ))
    print(f"\nResult: {result['message']}")
    if not result["success"]:
        raise SystemExit(1)
