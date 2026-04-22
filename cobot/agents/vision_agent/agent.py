"""Pydantic-AI Vision Agent — direct RealSense + Gemini ER tools (no MCP)."""

from __future__ import annotations

import os
from dataclasses import dataclass, field

from pathlib import Path

import cv2
import numpy as np
from google import genai
from pydantic import BaseModel
from pydantic_ai import Agent, RunContext

from cobot.config import CobotConfig
from cobot.llm import make_model
from cobot.agents.vision_agent.camera import (
    load_calibration, capture_realsense, encode_jpeg,
    depth_at_pixel, depth_at_box, pixel_to_camera_xyz, camera_to_base,
    CALIB_FILE,
)
from cobot.agents.vision_agent.gemini import gemini_locate
from cobot.agents.vision_agent.kinematics import APPROACH_Z_OFFSET



@dataclass
class VisionAgentDeps:
    cfg: CobotConfig
    calib_file: str = field(default_factory=lambda: CALIB_FILE)
    gemini_client: genai.Client | None = None
    arm_mcp: object | None = None       # MCPClient — injected at session startup
    gripper_mcp: object | None = None   # MCPClient — injected at session startup
    _T_base_cam: np.ndarray | None = field(default=None, repr=False)
    # Cached frame from the last capture_scene call — reused by locate_object
    _frame: tuple | None = field(default=None, repr=False)   # (color, depth, ci, ds)
    _annotated: np.ndarray | None = field(default=None, repr=False)

    # Debug images are written here
    debug_image_path: str = field(
        default_factory=lambda: str(Path(__file__).parent / "vision_scene.jpg")
    )

    @property
    def arm_mcp_url(self) -> str | None:
        """Return arm MCP URL if ik_solver == 'robot', else None (use local IK)."""
        solver = getattr(self.cfg.agents.vision, "ik_solver", "local")
        if solver == "robot":
            return self.cfg.fleet[0].arm_mcp_url
        return None

    def __post_init__(self):
        if self.gemini_client is None:
            key = (getattr(self.cfg.agents.vision, "google_api_key", "") or
                   os.environ.get("GOOGLE_API_KEY", ""))
            if key:
                self.gemini_client = genai.Client(api_key=key)

    def get_T_base_cam(self) -> np.ndarray:
        if self._T_base_cam is None:
            self._T_base_cam = load_calibration(self.calib_file)
        return self._T_base_cam

    def set_frame(self, color, depth, ci, ds) -> None:
        self._frame = (color, depth, ci, ds)
        self._annotated = color.copy()
        cv2.imwrite(self.debug_image_path, color)

    def get_frame(self):
        """Return cached frame, capturing a fresh one if needed."""
        if self._frame is None:
            color, depth, ci, ds = capture_realsense()
            self.set_frame(color, depth, ci, ds)
        return self._frame

    def locate_sync(self, description: str) -> dict | None:
        """Capture a fresh frame, locate object, return base-frame {x,y,z} or None."""
        if self.gemini_client is None:
            return None
        from cobot.agents.vision_agent.camera import (
            capture_realsense, encode_jpeg, depth_at_pixel, depth_at_box,
            pixel_to_camera_xyz, camera_to_base,
        )
        from cobot.agents.vision_agent.gemini import gemini_locate
        T = self.get_T_base_cam()
        color, depth, ci, ds = capture_realsense()
        det = gemini_locate(self.gemini_client, encode_jpeg(color), description)
        if det is None:
            return None
        h, w = color.shape[:2]
        ny, nx = det["point"]
        px = int(nx / 1000 * w)
        py = int(ny / 1000 * h)
        box = det.get("box_2d")
        if box:
            ymin, xmin, ymax, xmax = box
            x1 = int(xmin / 1000 * w); y1 = int(ymin / 1000 * h)
            x2 = int(xmax / 1000 * w); y2 = int(ymax / 1000 * h)
            d_m = depth_at_box(depth, ds, x1, y1, x2, y2)
        else:
            d_m = depth_at_pixel(depth, ci, px, py, ds)
        if d_m < 0.05:
            return None
        xyz = pixel_to_camera_xyz(ci, px, py, d_m)
        return camera_to_base(xyz, T)

    def annotate(self, px: int, py: int, label: str, d_m: float) -> None:
        """Draw a detection point on the accumulated annotated image."""
        if self._annotated is None:
            return
        cv2.circle(self._annotated, (px, py), 10, (0, 255, 0), -1)
        cv2.putText(self._annotated, f"{label} {d_m:.2f}m", (px + 12, py),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        cv2.imwrite(self.debug_image_path, self._annotated)


class VisionAgentResult(BaseModel):
    success: bool
    summary: str
    object_label: str | None = None
    position_base_m: dict | None = None    # {x, y, z} in robot base frame
    approach_pose_m: dict | None = None    # APPROACH_Z_OFFSET above object
    depth_m: float | None = None
    workspace_clear: bool | None = None
    error: str | None = None


def build_vision_agent(cfg: CobotConfig) -> Agent:
    agent: Agent[VisionAgentDeps, VisionAgentResult] = Agent(
        make_model(cfg.agents.vision.model, cfg),
        deps_type=VisionAgentDeps,
        output_type=VisionAgentResult,
        system_prompt=(
            "You are the Vision Agent for a UR3e robot pick-and-place cell.\n"
            "You have direct access to a RealSense D455 camera and Gemini ER.\n\n"
            "Tool selection guide:\n"
            "- User says 'pick', 'grab', 'lift', 'get' an object → call pick_object(). "
            "This detects the object, computes IK, and moves the arm directly. No YAML file.\n"
            "- User says 'point to', 'show me', 'indicate', 'hover over' an object → call point_to_object(). "
            "Moves the gripper above the object without picking it.\n"
            "- User says 'create a scenario' or 'save a pick-and-place' → call create_pick_scenario() or create_lift_scenario().\n"
            "- User says 'where is X' or 'find X' → call capture_scene() then locate_object().\n"
            "- User says 'is workspace clear' → call check_workspace_clear().\n\n"
            "Return a VisionAgentResult with all relevant fields populated."
        ),
    )

    @agent.tool
    async def capture_scene(ctx: RunContext[VisionAgentDeps]) -> dict:
        """
        Capture a fresh RGB-D frame from the RealSense camera.
        Saves the raw image to vision_scene.jpg for inspection.
        Call this once — subsequent locate_object calls reuse this frame.
        """
        color, depth, ci, ds = capture_realsense()
        ctx.deps.set_frame(color, depth, ci, ds)
        valid = (depth * ds)[depth > 0]
        return {
            "width": color.shape[1],
            "height": color.shape[0],
            "depth_min_m": round(float(valid.min()), 3) if len(valid) else None,
            "depth_max_m": round(float(valid.max()), 3) if len(valid) else None,
            "depth_mean_m": round(float(valid.mean()), 3) if len(valid) else None,
            "saved_to": ctx.deps.debug_image_path,
        }

    @agent.tool
    async def locate_object(
        ctx: RunContext[VisionAgentDeps],
        description: str,
    ) -> dict:
        """
        Locate an object by natural-language description using Gemini ER +
        RealSense depth. Reuses the last captured frame. Annotates detections
        onto vision_scene.jpg so you can see what was found.

        Args:
            description: e.g. "water bottle", "red block", "right side of table"
        """
        if ctx.deps.gemini_client is None:
            return {"success": False, "error": "GOOGLE_API_KEY not set"}

        T = ctx.deps.get_T_base_cam()
        color, depth, ci, ds = ctx.deps.get_frame()
        h, w = color.shape[:2]

        det = gemini_locate(ctx.deps.gemini_client, encode_jpeg(color), description)
        if det is None:
            return {"success": False, "error": f"Could not locate '{description}'"}

        ny, nx = det["point"]
        px = int(nx / 1000 * w)
        py = int(ny / 1000 * h)
        box = det.get("box_2d")
        if box:
            ymin, xmin, ymax, xmax = box
            x1 = int(xmin / 1000 * w); y1 = int(ymin / 1000 * h)
            x2 = int(xmax / 1000 * w); y2 = int(ymax / 1000 * h)
            d_m = depth_at_box(depth, ds, x1, y1, x2, y2)
            if d_m < 0.05:
                d_m = depth_at_pixel(depth, ci, px, py, ds)
        else:
            d_m = depth_at_pixel(depth, ci, px, py, ds)

        if d_m < 0.05:
            return {"success": False, "error": f"No depth reading at detected pixel ({px},{py})"}

        xyz      = pixel_to_camera_xyz(ci, px, py, d_m)
        base_pos = camera_to_base(xyz, T)
        approach = {
            "x": base_pos["x"],
            "y": base_pos["y"],
            "z": round(base_pos["z"] + APPROACH_Z_OFFSET, 4),
        }

        label = det.get("label", description)
        ctx.deps.annotate(px, py, label, d_m)

        return {
            "success":         True,
            "label":           label,
            "pixel":           [px, py],
            "depth_m":         round(d_m, 4),
            "position_base_m": base_pos,
            "approach_pose_m": approach,
        }

    @agent.tool
    async def create_pick_scenario(
        ctx: RunContext[VisionAgentDeps],
        pick_description: str,
        place_description: str,
        scenario_name: str = "vision_pick",
    ) -> dict:
        """
        Generate a complete pick-and-place scenario YAML by detecting both objects
        live with Gemini ER + RealSense, computing joint angles, and writing the
        scenario to the scenarios store.

        Use this whenever the user wants to create a new scenario for an object
        at its current position — do NOT call run_scenario after this unless the
        user explicitly asks to execute it.

        Args:
            pick_description:  Natural-language description of the pick object,
                               e.g. "bottle on the right side of the table"
            place_description: Natural-language description of the place target,
                               e.g. "left side of the table"
            scenario_name:     Filename for the scenario (without .yaml).
        """
        from cobot.agents.vision_agent.scenario import generate_scenario
        result = generate_scenario(
            pick=pick_description,
            place=place_description,
            name=scenario_name,
            calib_file=ctx.deps.calib_file,
            gemini_client=ctx.deps.gemini_client,
            cfg=ctx.deps.cfg,
            arm_mcp_url=ctx.deps.arm_mcp_url,
        )
        return result

    @agent.tool
    async def create_lift_scenario(
        ctx: RunContext[VisionAgentDeps],
        pick_description: str,
        lift_height_m: float = 0.15,
        scenario_name: str = "vision_lift",
    ) -> dict:
        """
        Generate a scenario that picks an object and lifts it straight up.
        Use this when the user wants to pick/lift/grab an object without moving
        it to a different location.

        Args:
            pick_description: Natural-language description of the object to pick,
                              e.g. "bottle on the right side of the table"
            lift_height_m:    How high to lift above the pick position (default 0.15m).
            scenario_name:    Filename for the scenario (without .yaml).
        """
        from cobot.agents.vision_agent.scenario import generate_scenario
        from cobot.agents.vision_agent.camera import (
            capture_realsense, encode_jpeg, depth_at_pixel, depth_at_box,
            pixel_to_camera_xyz, camera_to_base,
        )
        from cobot.agents.vision_agent.gemini import gemini_locate

        if ctx.deps.gemini_client is None:
            return {"success": False, "error": "GOOGLE_API_KEY not set"}

        T = ctx.deps.get_T_base_cam()
        color, depth, ci, ds = capture_realsense()
        h, w = color.shape[:2]

        det = gemini_locate(ctx.deps.gemini_client, encode_jpeg(color), pick_description)
        if det is None:
            return {"success": False, "error": f"Could not locate '{pick_description}'"}

        ny, nx = det["point"]
        px = int(nx / 1000 * w)
        py = int(ny / 1000 * h)
        box = det.get("box_2d")
        if box:
            ymin, xmin, ymax, xmax = box
            x1 = int(xmin / 1000 * w); y1 = int(ymin / 1000 * h)
            x2 = int(xmax / 1000 * w); y2 = int(ymax / 1000 * h)
            d_m = depth_at_box(depth, ds, x1, y1, x2, y2)
            if d_m < 0.05:
                d_m = depth_at_pixel(depth, ci, px, py, ds)
        else:
            d_m = depth_at_pixel(depth, ci, px, py, ds)

        if d_m < 0.05:
            return {"success": False, "error": "No depth reading at detected pixel"}

        xyz = pixel_to_camera_xyz(ci, px, py, d_m)
        pick_base = camera_to_base(xyz, T)

        place_xyz = {
            "x": pick_base["x"],
            "y": pick_base["y"],
            "z": round(pick_base["z"] + lift_height_m, 4),
        }

        result = generate_scenario(
            pick=pick_description,
            place=f"lift {lift_height_m}m above pick",
            name=scenario_name,
            calib_file=ctx.deps.calib_file,
            gemini_client=ctx.deps.gemini_client,
            cfg=ctx.deps.cfg,
            place_xyz=place_xyz,
            arm_mcp_url=ctx.deps.arm_mcp_url,
        )
        return result

    @agent.tool
    async def pick_object(
        ctx: RunContext[VisionAgentDeps],
        description: str,
        lift_height_m: float = 0.15,
    ) -> dict:
        """
        Detect an object with Gemini ER + RealSense, compute IK, and execute a
        pick-and-lift motion directly — no YAML scenario file is created.

        Use this when the user says "pick", "grab", "lift", or "get" an object.

        Steps: home → open gripper → approach → descend → close gripper → lift.

        Args:
            description:   What to pick, e.g. "bottle on the table".
            lift_height_m: How high to lift above the pick z (default 0.15 m).
        """
        from cobot.agents.vision_agent.camera import (
            capture_realsense, encode_jpeg, depth_at_pixel, depth_at_box,
            pixel_to_camera_xyz, camera_to_base,
        )
        from cobot.agents.vision_agent.gemini import gemini_locate
        from cobot.agents.vision_agent.kinematics import (
            get_joints_for_pose, APPROACH_Z_OFFSET, PICK_Z_OFFSET, _PICK_SEED,
        )

        if ctx.deps.gemini_client is None:
            return {"success": False, "error": "GOOGLE_API_KEY not set"}
        if ctx.deps.arm_mcp is None:
            return {"success": False, "error": "arm_mcp not wired — cannot move robot"}

        arm     = ctx.deps.arm_mcp
        gripper = ctx.deps.gripper_mcp
        T       = ctx.deps.get_T_base_cam()

        # 1. Detect object
        color, depth, ci, ds = capture_realsense()
        h, w = color.shape[:2]
        det = gemini_locate(ctx.deps.gemini_client, encode_jpeg(color), description)
        if det is None:
            return {"success": False, "error": f"Could not locate '{description}'"}

        ny, nx = det["point"]
        px = int(nx / 1000 * w); py = int(ny / 1000 * h)
        box = det.get("box_2d")
        if box:
            ymin, xmin, ymax, xmax = box
            x1 = int(xmin/1000*w); y1 = int(ymin/1000*h)
            x2 = int(xmax/1000*w); y2 = int(ymax/1000*h)
            d_m = depth_at_box(depth, ds, x1, y1, x2, y2)
            if d_m < 0.05:
                d_m = depth_at_pixel(depth, ci, px, py, ds)
        else:
            d_m = depth_at_pixel(depth, ci, px, py, ds)

        if d_m < 0.05:
            return {"success": False, "error": "No depth reading at detected pixel"}

        xyz      = pixel_to_camera_xyz(ci, px, py, d_m)
        base_pos = camera_to_base(xyz, T)
        bx, by, bz = base_pos["x"], base_pos["y"], base_pos["z"]

        approach_z = round(bz + APPROACH_Z_OFFSET, 4)
        grasp_z    = round(bz + PICK_Z_OFFSET,     4)
        lift_z     = round(bz + lift_height_m,     4)

        # 2. IK for approach only (coarse estimate)
        mcp_url = ctx.deps.arm_mcp_url
        j_approach = get_joints_for_pose("approach", bx, by, approach_z, _PICK_SEED, arm_mcp_url=mcp_url)
        if j_approach is None:
            return {"success": False, "error": f"IK failed for approach ({bx:.3f},{by:.3f},{approach_z:.3f})"}

        # 3. Move home + open gripper, then move to approach
        await arm.call_tool("move_named_pose", {"name": "home", "speed": 0.5, "accel": 0.8})

        if gripper:
            await gripper.call_tool("move", {"position": 0, "speed": 100, "force": 100})

        await arm.call_tool("move_joints", {"joints": j_approach, "speed": 0.3, "accel": 0.5})

        # 4. Re-detect from approach position for accurate grasp/lift IK
        print("  [pick] re-detecting from approach position …")
        color2, depth2, ci2, ds2 = capture_realsense()
        h2, w2 = color2.shape[:2]
        det2 = gemini_locate(ctx.deps.gemini_client, encode_jpeg(color2), description)
        if det2 is not None:
            ny2, nx2 = det2["point"]
            px2 = int(nx2 / 1000 * w2); py2 = int(ny2 / 1000 * h2)
            box2 = det2.get("box_2d")
            if box2:
                ymin2, xmin2, ymax2, xmax2 = box2
                x1_2 = int(xmin2/1000*w2); y1_2 = int(ymin2/1000*h2)
                x2_2 = int(xmax2/1000*w2); y2_2 = int(ymax2/1000*h2)
                # Horizontal center of box → centers gripper over object
                # Top third of box → samples depth from top surface, not object body
                px2 = (x1_2 + x2_2) // 2
                py2 = y1_2 + (y2_2 - y1_2) // 3
                d2  = depth_at_box(depth2, ds2, x1_2, y1_2, x2_2, y1_2 + (y2_2 - y1_2) // 2)
                if d2 < 0.05:
                    d2 = depth_at_pixel(depth2, ci2, px2, py2, ds2)
            else:
                d2 = depth_at_pixel(depth2, ci2, px2, py2, ds2)
            if d2 >= 0.05:
                xyz2   = pixel_to_camera_xyz(ci2, px2, py2, d2)
                base2  = camera_to_base(xyz2, T)
                bx, by, bz = base2["x"], base2["y"], base2["z"]
                x_off = ctx.deps.cfg.agents.vision.pick_x_offset_m
                y_off = ctx.deps.cfg.agents.vision.pick_y_offset_m
                print(f"  [pick] applying offsets: x={x_off:+.4f}  y={y_off:+.4f}")
                bx += x_off
                by += y_off
                grasp_z = round(bz + PICK_Z_OFFSET, 4)
                lift_z  = round(bz + lift_height_m, 4)
                print(f"  [pick] refined position (after offset): ({bx:.3f},{by:.3f},{bz:.3f})")
            else:
                print("  [pick] re-detect depth invalid, using initial estimate")
        else:
            print("  [pick] re-detect failed, using initial estimate")

        # Grasp and lift use local analytical IK for precision — robot IK is only
        # used for the large approach transit where exact accuracy is less critical.
        refined_approach_z = round(bz + APPROACH_Z_OFFSET, 4)
        j_approach2 = get_joints_for_pose("approach2", bx, by, refined_approach_z, j_approach, arm_mcp_url=None)
        # Seed grasp/lift from j_approach2 — that is where the arm will actually be sitting.
        # Using j_approach here caused the IK to find solutions far from the refined pose.
        grasp_seed = j_approach2 if j_approach2 is not None else j_approach
        j_grasp = get_joints_for_pose("grasp", bx, by, grasp_z, grasp_seed, arm_mcp_url=None)
        if j_grasp is None:
            return {"success": False, "error": f"IK failed for grasp ({bx:.3f},{by:.3f},{grasp_z:.3f})"}
        j_lift  = get_joints_for_pose("lift",  bx, by, lift_z,  grasp_seed, arm_mcp_url=None)
        if j_lift is None:
            return {"success": False, "error": f"IK failed for lift ({bx:.3f},{by:.3f},{lift_z:.3f})"}

        # 5. Re-approach at refined XY (corrects lateral error), then descend straight down
        if j_approach2 is not None:
            await arm.call_tool("move_joints", {"joints": j_approach2, "speed": 0.2, "accel": 0.4})
        await arm.call_tool("move_joints", {"joints": j_grasp, "speed": 0.1, "accel": 0.2})

        if gripper:
            await gripper.call_tool("move", {"position": 255, "speed": 100, "force": 150})

        await arm.call_tool("move_joints", {"joints": j_lift, "speed": 0.2, "accel": 0.4})

        return {
            "success":     True,
            "label":       det.get("label", description),
            "position_m":  {"x": bx, "y": by, "z": bz},
            "depth_m":     round(d_m, 4),
            "summary":     f"Picked '{det.get('label', description)}' at ({bx:.3f},{by:.3f},{bz:.3f}), lifted to z={lift_z:.3f}",
        }

    @agent.tool
    async def point_to_object(
        ctx: RunContext[VisionAgentDeps],
        description: str,
        hover_height_m: float = 0.12,
    ) -> dict:
        """
        Detect an object and move the gripper to hover directly above it — does NOT pick or grip.

        Use when the user says "point to", "show me", "indicate", or "where is" combined with
        a physical move request.

        Args:
            description:    What to point at, e.g. "bottle", "red cup".
            hover_height_m: Height above the detected object surface to hover (default 0.12 m).
        """
        from cobot.agents.vision_agent.camera import (
            capture_realsense, encode_jpeg, depth_at_pixel, depth_at_box,
            pixel_to_camera_xyz, camera_to_base,
        )
        from cobot.agents.vision_agent.gemini import gemini_locate
        from cobot.agents.vision_agent.kinematics import get_joints_for_pose, _PICK_SEED

        if ctx.deps.gemini_client is None:
            return {"success": False, "error": "GOOGLE_API_KEY not set"}
        if ctx.deps.arm_mcp is None:
            return {"success": False, "error": "arm_mcp not wired — cannot move robot"}

        arm = ctx.deps.arm_mcp
        T   = ctx.deps.get_T_base_cam()

        color, depth, ci, ds = capture_realsense()
        h, w = color.shape[:2]
        det = gemini_locate(ctx.deps.gemini_client, encode_jpeg(color), description)
        if det is None:
            return {"success": False, "error": f"Could not locate '{description}'"}

        ny, nx = det["point"]
        px = int(nx / 1000 * w); py = int(ny / 1000 * h)
        box = det.get("box_2d")
        if box:
            ymin, xmin, ymax, xmax = box
            x1 = int(xmin/1000*w); y1 = int(ymin/1000*h)
            x2 = int(xmax/1000*w); y2 = int(ymax/1000*h)
            d_m = depth_at_box(depth, ds, x1, y1, x2, y2)
            if d_m < 0.05:
                d_m = depth_at_pixel(depth, ci, px, py, ds)
        else:
            d_m = depth_at_pixel(depth, ci, px, py, ds)

        if d_m < 0.05:
            return {"success": False, "error": "No depth reading at detected pixel"}

        xyz      = pixel_to_camera_xyz(ci, px, py, d_m)
        base_pos = camera_to_base(xyz, T)
        bx, by, bz = base_pos["x"], base_pos["y"], base_pos["z"]
        hover_z  = round(bz + hover_height_m, 4)

        mcp_url  = ctx.deps.arm_mcp_url
        j_hover  = get_joints_for_pose("hover", bx, by, hover_z, _PICK_SEED, arm_mcp_url=None)
        if j_hover is None:
            return {"success": False, "error": f"IK failed for hover ({bx:.3f},{by:.3f},{hover_z:.3f})"}

        await arm.call_tool("move_named_pose", {"name": "home", "speed": 0.5, "accel": 0.8})
        await arm.call_tool("move_joints", {"joints": j_hover, "speed": 0.3, "accel": 0.5})

        return {
            "success":    True,
            "label":      det.get("label", description),
            "position_m": base_pos,
            "hover_z":    hover_z,
            "summary":    f"Pointing at '{det.get('label', description)}' at ({bx:.3f},{by:.3f},{bz:.3f}), hovering at z={hover_z:.3f}",
        }

    @agent.tool
    async def check_workspace_clear(ctx: RunContext[VisionAgentDeps]) -> dict:
        """
        Ask Gemini ER whether the robot workspace is clear of unexpected obstacles.
        Returns a safety assessment before large arm movements.
        """
        if ctx.deps.gemini_client is None:
            return {"success": False, "error": "GOOGLE_API_KEY not set"}

        color, _, _, _ = capture_realsense()
        det = gemini_locate(
            ctx.deps.gemini_client,
            encode_jpeg(color),
            "any obstacle or object in the robot arm's movement path",
        )
        clear = det is None
        return {
            "workspace_clear": clear,
            "message": "No obstacles detected." if clear else f"Possible obstacle: {det}",
        }

    return agent
