"""
pick_agent.py — Standalone vision-guided pick-and-place agent.

Uses:
  - Gemini Robotics-ER 1.6  → find objects by natural language description
  - RealSense D455           → depth at detected pixel → 3D camera coords
  - calibration.json         → camera → robot base frame transform
  - Arm MCP (UR3e)           → move TCP, get status
  - Gripper MCP (Robotiq)    → open / close

Usage:
    python pick_agent.py
    python pick_agent.py --pick "water bottle" --place "left side of table"
    python pick_agent.py --dry-run   # plan only, no robot movement

Requirements:
    GOOGLE_API_KEY   — Gemini API key
    ANTHROPIC_API_KEY or set in env
    RealSense D455 connected via USB 3
    Arm MCP running at http://172.31.0.77:8765/mcp
    Gripper MCP running at http://172.31.0.77:8001/mcp
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
import yaml
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import pyrealsense2 as rs
from google import genai
from google.genai import types as gtypes
from pydantic import BaseModel
from pydantic_ai import Agent, RunContext
from pydantic_ai.models.anthropic import AnthropicModel

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

ARM_MCP_URL     = os.getenv("ARM_MCP_URL",     "http://172.31.0.77:8765/mcp")
GRIPPER_MCP_URL = os.getenv("GRIPPER_MCP_URL", "http://172.31.0.77:8001/mcp")
CALIB_FILE      = os.getenv("CALIB_FILE",
    str(Path(__file__).parent.parent.parent / "tools/realsense/calibration.json"))
APPROACH_Z_OFFSET = 0.12   # metres above object for pre-grasp approach
PICK_Z_OFFSET     = 0.01   # metres above marker surface for grasp

# ---------------------------------------------------------------------------
# Camera helpers
# ---------------------------------------------------------------------------

def capture_realsense():
    """Return (color_bgr, depth_uint16, intrinsics, depth_scale)."""
    pipe = rs.pipeline()
    cfg = rs.config()
    cfg.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
    cfg.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
    profile = pipe.start(cfg)
    align = rs.align(rs.stream.color)
    for _ in range(30):
        pipe.wait_for_frames()
    frames = align.process(pipe.wait_for_frames())
    color = np.asanyarray(frames.get_color_frame().get_data()).copy()
    depth = np.asanyarray(frames.get_depth_frame().get_data()).copy()
    ci = profile.get_stream(rs.stream.color).as_video_stream_profile().get_intrinsics()
    depth_scale = profile.get_device().first_depth_sensor().get_depth_scale()
    pipe.stop()
    return color, depth, ci, depth_scale


def encode_jpeg(img_bgr, quality=90):
    _, buf = cv2.imencode(".jpg", img_bgr, [cv2.IMWRITE_JPEG_QUALITY, quality])
    return buf.tobytes()


def depth_at_pixel(depth, ci, px, py, depth_scale, radius=4):
    """Median depth in a patch (metres)."""
    h, w = depth.shape
    x1, x2 = max(0, px-radius), min(w, px+radius+1)
    y1, y2 = max(0, py-radius), min(h, py+radius+1)
    patch = depth[y1:y2, x1:x2]
    valid = patch[patch > 0]
    return float(np.median(valid)) * depth_scale if len(valid) else 0.0


def pixel_to_camera_xyz(ci, px, py, depth_m):
    """Back-project pixel + depth to camera frame XYZ."""
    x = (px - ci.ppx) * depth_m / ci.fx
    y = (py - ci.ppy) * depth_m / ci.fy
    return np.array([x, y, depth_m, 1.0])


def camera_to_base(xyz_cam: np.ndarray, T_base_cam: np.ndarray) -> dict:
    """Transform 4-vector from camera frame to robot base frame."""
    p = T_base_cam @ xyz_cam
    return {"x": round(float(p[0]), 4),
            "y": round(float(p[1]), 4),
            "z": round(float(p[2]), 4)}


# ---------------------------------------------------------------------------
# Calibration
# ---------------------------------------------------------------------------

def load_calibration(path: str) -> np.ndarray:
    with open(path) as f:
        d = json.load(f)
    T = np.array(d["T_base_cam"], dtype=np.float64)
    print(f"[calib] Loaded from {path}")
    return T


# ---------------------------------------------------------------------------
# Gemini ER helpers
# ---------------------------------------------------------------------------

def gemini_locate(client: genai.Client, image_bytes: bytes, description: str):
    """Ask Gemini ER to locate an object. Returns normalised [y, x] or None."""
    prompt = f"""
Point to: {description}
Return ONLY a JSON array (no text, no code fences):
[{{"point": [y, x], "label": "..."}}]
Points normalised 0-1000. Return [] if not found.
"""
    r = client.models.generate_content(
        model="gemini-robotics-er-1.6-preview",
        contents=[
            gtypes.Part.from_bytes(data=image_bytes, mime_type="image/jpeg"),
            prompt,
        ],
        config=gtypes.GenerateContentConfig(
            temperature=1.0,
            thinking_config=gtypes.ThinkingConfig(thinking_budget=0),
        ),
    )
    raw = r.text.strip().strip("```json").strip("```").strip()
    detections = json.loads(raw)
    return detections[0] if detections else None


# ---------------------------------------------------------------------------
# Agent deps and result
# ---------------------------------------------------------------------------

@dataclass
class PickAgentDeps:
    gemini: genai.Client
    T_base_cam: np.ndarray
    dry_run: bool = False
    # Orientation quaternion read from robot at startup — preserved for all moves
    qx: float = 1.0
    qy: float = 0.0
    qz: float = 0.0
    qw: float = 0.0


class PickResult(BaseModel):
    success: bool
    summary: str
    pick_position: dict | None = None
    place_position: dict | None = None
    steps_executed: int = 0
    error: str | None = None


# ---------------------------------------------------------------------------
# MCP helpers (direct async calls without the full MCPClient wrapper)
# ---------------------------------------------------------------------------

async def call_mcp(url: str, tool: str, args: dict) -> dict:
    from mcp import ClientSession
    from mcp.client.streamable_http import streamable_http_client
    async with streamable_http_client(url) as (r, w, _):
        async with ClientSession(r, w) as s:
            await s.initialize()
            result = await s.call_tool(tool, args)
            if result.content and result.content[0].type == "text":
                try:
                    return json.loads(result.content[0].text)
                except Exception:
                    return {"raw": result.content[0].text}
            if result.structuredContent:
                return dict(result.structuredContent)
            return {"ok": False, "message": "empty response"}


# ---------------------------------------------------------------------------
# Build agent
# ---------------------------------------------------------------------------

def load_anthropic_key() -> str:
    """Read API key from cobot.yaml if ANTHROPIC_API_KEY env var is not set."""
    key = os.environ.get("ANTHROPIC_API_KEY", "")
    if key:
        return key
    calib_dir = Path(__file__).parent.parent.parent
    yaml_path = calib_dir / "cobot.yaml"
    if yaml_path.exists():
        import yaml
        with open(yaml_path) as f:
            cfg = yaml.safe_load(f)
        key = cfg.get("llm", {}).get("api_key", "")
        if key:
            os.environ["ANTHROPIC_API_KEY"] = key
            return key
    raise RuntimeError(
        "No Anthropic API key found. Set ANTHROPIC_API_KEY or add it to cobot.yaml"
    )


def build_pick_agent() -> Agent:
    load_anthropic_key()
    model = AnthropicModel("claude-sonnet-4-6")

    agent: Agent[PickAgentDeps, PickResult] = Agent(
        model,
        deps_type=PickAgentDeps,
        output_type=PickResult,
        system_prompt=(
            "You are a pick-and-place robot controller.\n"
            "Workflow:\n"
            "1. capture_and_locate(pick_description) — find pick object, get 3D pose\n"
            "2. capture_and_locate(place_description) — find place target, get 3D pose\n"
            "3. get_arm_status() — confirm robot is ready\n"
            "4. open_gripper()\n"
            "5. move_to_approach(x, y, z) — move to 12cm ABOVE the pick object\n"
            "6. move_to_pose(x, y, z) — descend to pick object\n"
            "7. close_gripper()\n"
            "8. move_to_approach(x, y, z) — lift 12cm up\n"
            "9. move_to_approach(place_x, place_y, place_z) — move above place target\n"
            "10. move_to_pose(place_x, place_y, place_z) — descend to place\n"
            "11. open_gripper()\n"
            "12. move_home()\n\n"
            "In dry_run mode: plan and describe steps but do NOT call move or gripper tools.\n"
            "Always check arm status before moving. Stop if any move fails.\n"
            "Return a PickResult when done."
        ),
    )

    @agent.tool
    async def capture_and_locate(
        ctx: RunContext[PickAgentDeps],
        description: str,
    ) -> dict:
        """
        Capture a live frame, ask Gemini ER to locate the described object,
        and return its 3D position in robot base frame.

        Args:
            description: Natural language description e.g. "water bottle", "blue block"
        """
        print(f"\n[vision] Locating: '{description}'")
        color, depth, ci, depth_scale = capture_realsense()
        h, w = color.shape[:2]
        det = gemini_locate(ctx.deps.gemini, encode_jpeg(color), description)

        if det is None:
            return {"found": False, "description": description,
                    "error": f"'{description}' not found in scene"}

        ny, nx = det["point"]
        px = int(nx / 1000 * w)
        py = int(ny / 1000 * h)
        d_m = depth_at_pixel(depth, ci, px, py, depth_scale)

        if d_m == 0:
            return {"found": False, "description": description,
                    "error": "Object detected but depth reading is zero — may be out of D455 range"}

        xyz_cam = pixel_to_camera_xyz(ci, px, py, d_m)
        pos_base = camera_to_base(xyz_cam, ctx.deps.T_base_cam)

        print(f"[vision]   label={det['label']}  pixel=({px},{py})  "
              f"depth={d_m:.3f}m  base={pos_base}")

        # Save annotated frame
        annotated = color.copy()
        cv2.circle(annotated, (px, py), 12, (0, 255, 0), -1)
        cv2.putText(annotated, f"{det['label']} {d_m:.2f}m",
                    (px+12, py), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0,255,0), 2)
        cv2.imwrite(f"locate_{description[:20].replace(' ','_')}.jpg", annotated)

        return {
            "found": True,
            "description": description,
            "label": det["label"],
            "pixel": [px, py],
            "depth_m": round(d_m, 4),
            "position_base_m": pos_base,
            "approach_pose_m": {
                "x": pos_base["x"],
                "y": pos_base["y"],
                "z": round(pos_base["z"] + APPROACH_Z_OFFSET, 4),
            },
        }

    @agent.tool
    async def get_arm_status(ctx: RunContext[PickAgentDeps]) -> dict:
        """Get current robot arm status and TCP position."""
        if ctx.deps.dry_run:
            return {"ok": True, "dry_run": True, "message": "Dry run — skipping status check"}
        status = await call_mcp(ARM_MCP_URL, "get_robot_status", {})
        tcp = await call_mcp(ARM_MCP_URL, "get_tcp_pose", {})
        return {"status": status, "tcp": tcp}

    @agent.tool
    async def move_to_approach(
        ctx: RunContext[PickAgentDeps],
        x: float, y: float, z: float,
        speed: float = 0.15,
    ) -> dict:
        """
        Move TCP to an approach position (typically 12cm above target).
        Safe for larger movements — arm moves at reduced speed.
        """
        if ctx.deps.dry_run:
            print(f"[DRY RUN] move_to_approach({x:.3f}, {y:.3f}, {z:.3f})")
            return {"ok": True, "dry_run": True, "position": {"x": x, "y": y, "z": z}}
        print(f"[arm] move_to_approach  x={x:.3f} y={y:.3f} z={z:.3f}")
        return await call_mcp(ARM_MCP_URL, "move_tcp_pose", {
            "x": x, "y": y, "z": z,
            "qx": ctx.deps.qx, "qy": ctx.deps.qy,
            "qz": ctx.deps.qz, "qw": ctx.deps.qw,
            "speed": speed, "accel": 0.3,
        })

    @agent.tool
    async def move_to_pose(
        ctx: RunContext[PickAgentDeps],
        x: float, y: float, z: float,
        speed: float = 0.05,
    ) -> dict:
        """
        Move TCP to an exact pick or place position.
        Slow, precise movement — use for final approach to object.
        """
        if ctx.deps.dry_run:
            print(f"[DRY RUN] move_to_pose({x:.3f}, {y:.3f}, {z:.3f})")
            return {"ok": True, "dry_run": True, "position": {"x": x, "y": y, "z": z}}
        print(f"[arm] move_to_pose      x={x:.3f} y={y:.3f} z={z:.3f}")
        return await call_mcp(ARM_MCP_URL, "move_tcp_pose", {
            "x": x, "y": y, "z": z,
            "qx": ctx.deps.qx, "qy": ctx.deps.qy,
            "qz": ctx.deps.qz, "qw": ctx.deps.qw,
            "speed": speed, "accel": 0.1,
        })

    @agent.tool
    async def move_home(ctx: RunContext[PickAgentDeps]) -> dict:
        """Return the robot arm to the home position."""
        if ctx.deps.dry_run:
            print("[DRY RUN] move_home()")
            return {"ok": True, "dry_run": True}
        print("[arm] move_home")
        return await call_mcp(ARM_MCP_URL, "move_named_pose",
                               {"name": "home", "speed": 0.5, "accel": 0.8})

    @agent.tool
    async def open_gripper(ctx: RunContext[PickAgentDeps]) -> dict:
        """Open the Robotiq gripper fully."""
        if ctx.deps.dry_run:
            print("[DRY RUN] open_gripper()")
            return {"ok": True, "dry_run": True}
        print("[gripper] open")
        return await call_mcp(GRIPPER_MCP_URL, "open", {})

    @agent.tool
    async def close_gripper(ctx: RunContext[PickAgentDeps]) -> dict:
        """Close the Robotiq gripper to grasp the object."""
        if ctx.deps.dry_run:
            print("[DRY RUN] close_gripper()")
            return {"ok": True, "dry_run": True}
        print("[gripper] close")
        return await call_mcp(GRIPPER_MCP_URL, "close", {})

    return agent


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

async def main():
    parser = argparse.ArgumentParser(description="Vision-guided pick-and-place agent")
    parser.add_argument("--pick",  default="object on the right", help="Object to pick")
    parser.add_argument("--place", default="object on the left",  help="Where to place it")
    parser.add_argument("--dry-run", action="store_true", help="Plan only, no robot movement")
    args = parser.parse_args()

    google_key = os.environ.get("GOOGLE_API_KEY")
    if not google_key:
        print("ERROR: GOOGLE_API_KEY not set.")
        sys.exit(1)

    print("=" * 60)
    print("Vision-Guided Pick-and-Place Agent")
    print(f"  Pick  : {args.pick}")
    print(f"  Place : {args.place}")
    print(f"  Mode  : {'DRY RUN' if args.dry_run else 'LIVE'}")
    print("=" * 60)

    T_base_cam = load_calibration(CALIB_FILE)
    gemini_client = genai.Client(api_key=google_key)

    # Read current TCP orientation from robot — reuse it for all moves so IK always has a valid solution
    qx, qy, qz, qw = 1.0, 0.0, 0.0, 0.0
    if not args.dry_run:
        try:
            tcp = await call_mcp(ARM_MCP_URL, "get_tcp_pose", {})
            q = tcp.get("orientation_quaternion", {})
            if q and all(k in q for k in ("x", "y", "z", "w")):
                qx, qy, qz, qw = q["x"], q["y"], q["z"], q["w"]
                print(f"[arm] TCP orientation  qx={qx:.4f} qy={qy:.4f} qz={qz:.4f} qw={qw:.4f}")
            else:
                print("[arm] Could not read TCP orientation — using default downward quaternion")
        except Exception as e:
            print(f"[arm] TCP pose read failed ({e}) — using default quaternion")

    deps = PickAgentDeps(
        gemini=gemini_client,
        T_base_cam=T_base_cam,
        dry_run=args.dry_run,
        qx=qx, qy=qy, qz=qz, qw=qw,
    )

    agent = build_pick_agent()
    task = (
        f"Pick up the '{args.pick}' and place it at the '{args.place}'. "
        f"{'This is a dry run — describe the plan without moving.' if args.dry_run else 'Execute the full pick-and-place sequence.'}"
    )

    print(f"\nTask: {task}\n")
    result = await agent.run(task, deps=deps)
    output = result.output

    print("\n" + "=" * 60)
    print("RESULT")
    print("=" * 60)
    print(f"Success       : {output.success}")
    print(f"Steps executed: {output.steps_executed}")
    print(f"Summary       : {output.summary}")
    if output.pick_position:
        print(f"Pick position : {output.pick_position}")
    if output.place_position:
        print(f"Place position: {output.place_position}")
    if output.error:
        print(f"Error         : {output.error}")


if __name__ == "__main__":
    asyncio.run(main())
