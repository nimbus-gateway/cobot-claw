"""Pydantic-AI Vision Agent — direct RealSense + detector tools (no MCP)."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

try:
    import logfire as _logfire
except ImportError:
    _logfire = None  # type: ignore[assignment]

import cv2
import numpy as np
from pydantic import BaseModel
from pydantic_ai import Agent, RunContext

from cobot.config import CobotConfig
from cobot.llm import make_model
from cobot.agents.vision_agent.camera import (
    load_calibration, capture_realsense, encode_jpeg,
    depth_at_pixel, depth_at_box, pixel_to_camera_xyz, camera_to_base,
    CALIB_FILE,
)
from cobot.agents.vision_agent.detector import VisionDetector, make_detector
from cobot.agents.vision_agent.kinematics import APPROACH_Z_OFFSET


@dataclass
class VisionAgentDeps:
    cfg: CobotConfig
    calib_file: str = field(default_factory=lambda: CALIB_FILE)
    detector: VisionDetector | None = None
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
        if self.detector is None:
            self.detector = make_detector(self.cfg)

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
        if self.detector is None:
            return None
        from cobot.agents.vision_agent.camera import (
            capture_realsense, encode_jpeg, depth_at_pixel, depth_at_box,
            pixel_to_camera_xyz, camera_to_base,
        )
        T = self.get_T_base_cam()
        color, depth, ci, ds = capture_realsense()
        det = self.detector.locate(encode_jpeg(color), description)
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


_DEFAULT_VISION_PROMPT = (
    "You are the Vision Agent for a UR3e robot pick-and-place cell.\n"
    "You have direct access to a RealSense D455 camera and Gemini ER.\n\n"
    "Tool selection guide:\n"
    "- User says 'pick', 'grab', 'lift', 'get' an object (no destination) → call pick_object().\n"
    "- User says 'pick X and put/drop/place/move it to/in/on Y' → call pick_and_place_object().\n"
    "- User says 'point to', 'show me', 'indicate', 'hover over' an object → call point_to_object().\n"
    "- User says 'where is X', 'find X', 'what do you see' → call capture_scene() then locate_object().\n"
    "- User says 'is workspace clear' → call check_workspace_clear().\n"
    "Never generate YAML scenario files — that is the Config Agent's responsibility.\n\n"
    "Return a VisionAgentResult with all relevant fields populated."
)


def build_vision_agent(cfg: CobotConfig) -> Agent:
    agent: Agent[VisionAgentDeps, VisionAgentResult] = Agent(
        make_model(cfg.agents.vision.model, cfg),
        deps_type=VisionAgentDeps,
        output_type=VisionAgentResult,
        system_prompt=cfg.agents.vision.system_prompt or _DEFAULT_VISION_PROMPT,
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
        if ctx.deps.detector is None:
            return {"success": False, "error": "GOOGLE_API_KEY not set"}

        T = ctx.deps.get_T_base_cam()
        color, depth, ci, ds = ctx.deps.get_frame()
        h, w = color.shape[:2]

        det = ctx.deps.detector.locate( encode_jpeg(color), description)
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
    async def pick_and_place_object(
        ctx: RunContext[VisionAgentDeps],
        pick_description: str,
        place_description: str,
        drop_height_m: float = 0.05,
    ) -> dict:
        """
        Detect two objects, pick the first and place it onto/into the second —
        no YAML scenario is created. Executes the motion directly.

        Use when the user says "pick X and put/drop/place it on/in Y",
        "move X to Y", "transfer X to Y".

        Args:
            pick_description:  Object to pick, e.g. "white block".
            place_description: Target location, e.g. "transparent box", "left side of table".
            drop_height_m:     Height above detected place surface to release (default 0.05m).
        """
        from cobot.agents.vision_agent.camera import (
            capture_realsense, encode_jpeg, depth_at_pixel, depth_at_box,
            pixel_to_camera_xyz, camera_to_base,
        )
        from cobot.agents.vision_agent.kinematics import (
            get_joints_for_pose, APPROACH_Z_OFFSET, PICK_Z_OFFSET, LIFT_Z_OFFSET, _PICK_SEED,
        )

        if ctx.deps.detector is None:
            return {"success": False, "error": "Vision detector not configured"}
        if ctx.deps.arm_mcp is None:
            return {"success": False, "error": "arm_mcp not wired — cannot move robot"}

        arm     = ctx.deps.arm_mcp
        gripper = ctx.deps.gripper_mcp
        T       = ctx.deps.get_T_base_cam()
        mcp_url = ctx.deps.arm_mcp_url
        x_off   = ctx.deps.cfg.agents.vision.pick_x_offset_m
        y_off   = ctx.deps.cfg.agents.vision.pick_y_offset_m

        def _detect_xyz(description: str) -> dict | None:
            color, depth, ci, ds = capture_realsense()
            h, w = color.shape[:2]
            det = ctx.deps.detector.locate(encode_jpeg(color), description)
            if det is None:
                return None
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
                return None
            xyz = pixel_to_camera_xyz(ci, px, py, d_m)
            return camera_to_base(xyz, T)

        # 1. Detect pick target
        pick_pos = _detect_xyz(pick_description)
        if pick_pos is None:
            if _logfire:
                _logfire.warning("vision.detection_failed", object=pick_description, role="pick")
            return {"success": False, "error": f"Could not locate pick target '{pick_description}'"}
        px_b = pick_pos["x"] + x_off
        py_b = pick_pos["y"] + y_off
        pz_b = pick_pos["z"]
        print(f"  [pick_and_place] pick: ({px_b:.3f},{py_b:.3f},{pz_b:.3f})")
        if _logfire:
            _logfire.info("vision.detection", object=pick_description, role="pick",
                          base_x=round(px_b, 4), base_y=round(py_b, 4), base_z=round(pz_b, 4),
                          x_offset_applied=x_off, y_offset_applied=y_off)

        # 2. Detect place target
        place_pos = _detect_xyz(place_description)
        if place_pos is None:
            if _logfire:
                _logfire.warning("vision.detection_failed", object=place_description, role="place")
            return {"success": False, "error": f"Could not locate place target '{place_description}'"}
        plx = place_pos["x"]
        ply = place_pos["y"]
        plz = place_pos["z"]
        print(f"  [pick_and_place] place: ({plx:.3f},{ply:.3f},{plz:.3f})")
        if _logfire:
            _logfire.info("vision.detection", object=place_description, role="place",
                          base_x=round(plx, 4), base_y=round(ply, 4), base_z=round(plz, 4))

        from cobot.agents.vision_agent.kinematics import PICK_Z_OFFSET

        # 3. IK for coarse pick approach only — grasp/lift IK computed after re-detection
        j_pick_approach = get_joints_for_pose("pick_approach", px_b, py_b, round(pz_b + APPROACH_Z_OFFSET, 4), _PICK_SEED, arm_mcp_url=mcp_url)
        if j_pick_approach is None:
            return {"success": False, "error": "IK failed for pick approach"}

        # --- PICK PHASE ---
        await arm.call_tool("move_named_pose", {"name": "home", "speed": 0.5, "accel": 0.8})
        if gripper:
            await gripper.call_tool("open", {"speed": 100, "force": 100})

        await arm.call_tool("move_joints", {"joints": j_pick_approach, "speed": 0.3, "accel": 0.5})

        # Re-detect pick from approach for accurate grasp/lift Z
        print("  [pick_and_place] re-detecting pick from approach position …")
        color_r, depth_r, ci_r, ds_r = capture_realsense()
        h_r, w_r = color_r.shape[:2]
        det_r = ctx.deps.detector.locate(encode_jpeg(color_r), pick_description)
        if det_r is not None:
            ny_r, nx_r = det_r["point"]
            px_r = int(nx_r / 1000 * w_r); py_r = int(ny_r / 1000 * h_r)
            box_r = det_r.get("box_2d")
            if box_r:
                ymin_r, xmin_r, ymax_r, xmax_r = box_r
                x1_r = int(xmin_r/1000*w_r); y1_r = int(ymin_r/1000*h_r)
                x2_r = int(xmax_r/1000*w_r); y2_r = int(ymax_r/1000*h_r)
                px_r = (x1_r + x2_r) // 2
                py_r = y1_r + (y2_r - y1_r) // 3
                d_r  = depth_at_box(depth_r, ds_r, x1_r, y1_r, x2_r, y1_r + (y2_r - y1_r) // 2)
                if d_r < 0.05:
                    d_r = depth_at_pixel(depth_r, ci_r, px_r, py_r, ds_r)
            else:
                d_r = depth_at_pixel(depth_r, ci_r, px_r, py_r, ds_r)
            if d_r >= 0.05:
                xyz_r  = pixel_to_camera_xyz(ci_r, px_r, py_r, d_r)
                base_r = camera_to_base(xyz_r, T)
                px_b = base_r["x"] + x_off
                py_b = base_r["y"] + y_off
                pz_b = base_r["z"]
                print(f"  [pick_and_place] refined pick: ({px_b:.3f},{py_b:.3f},{pz_b:.3f})")
            else:
                print("  [pick_and_place] re-detect depth invalid, using initial estimate")
        else:
            print("  [pick_and_place] re-detect failed, using initial estimate")

        j_approach2 = get_joints_for_pose("pick_approach2", px_b, py_b, round(pz_b + APPROACH_Z_OFFSET, 4), j_pick_approach, arm_mcp_url=None)
        grasp_seed = j_approach2 if j_approach2 is not None else j_pick_approach
        j_pick_grasp = get_joints_for_pose("pick_grasp", px_b, py_b, round(pz_b + PICK_Z_OFFSET, 4), grasp_seed, arm_mcp_url=None)
        if j_pick_grasp is None:
            return {"success": False, "error": f"IK failed for pick grasp ({px_b:.3f},{py_b:.3f})"}
        j_pick_lift = get_joints_for_pose("pick_lift", px_b, py_b, round(pz_b + LIFT_Z_OFFSET, 4), grasp_seed, arm_mcp_url=None)
        if j_pick_lift is None:
            return {"success": False, "error": "IK failed for pick lift"}

        if j_approach2 is not None:
            await arm.call_tool("move_joints", {"joints": j_approach2,  "speed": 0.2, "accel": 0.4})
        await arm.call_tool("move_joints", {"joints": j_pick_grasp,    "speed": 0.1, "accel": 0.2})
        if gripper:
            await gripper.call_tool("close", {"speed": 100, "force": 100})
        await arm.call_tool("move_joints", {"joints": j_pick_lift,     "speed": 0.2, "accel": 0.4})

        # --- PLACE PHASE: re-detect from lift position, then move directly to place ---
        print("  [pick_and_place] re-detecting place target from lift position …")
        color_p, depth_p, ci_p, ds_p = capture_realsense()
        h_p, w_p = color_p.shape[:2]
        det_p = ctx.deps.detector.locate(encode_jpeg(color_p), place_description)
        if det_p is None:
            return {"success": False, "error": f"Could not re-locate place target '{place_description}' from lift position"}
        ny_p, nx_p = det_p["point"]
        px_p = int(nx_p / 1000 * w_p); py_p = int(ny_p / 1000 * h_p)
        box_p = det_p.get("box_2d")
        if box_p:
            ymin_p, xmin_p, ymax_p, xmax_p = box_p
            x1_p = int(xmin_p/1000*w_p); y1_p = int(ymin_p/1000*h_p)
            x2_p = int(xmax_p/1000*w_p); y2_p = int(ymax_p/1000*h_p)
            px_p = (x1_p + x2_p) // 2
            py_p = (y1_p + y2_p) // 2
            d_p  = depth_at_box(depth_p, ds_p, x1_p, y1_p, x2_p, y2_p)
            if d_p < 0.05:
                d_p = depth_at_pixel(depth_p, ci_p, px_p, py_p, ds_p)
        else:
            d_p = depth_at_pixel(depth_p, ci_p, px_p, py_p, ds_p)
        if d_p < 0.05:
            return {"success": False, "error": "No depth for place target from home position"}
        xyz_p  = pixel_to_camera_xyz(ci_p, px_p, py_p, d_p)
        base_p = camera_to_base(xyz_p, T)
        plx = base_p["x"]; ply = base_p["y"]; plz = base_p["z"]
        print(f"  [pick_and_place] place from lift: ({plx:.3f},{ply:.3f},{plz:.3f})")

        j_place_approach = get_joints_for_pose("place_approach", plx, ply, round(plz + APPROACH_Z_OFFSET, 4), j_pick_lift, arm_mcp_url=mcp_url)
        if j_place_approach is None:
            return {"success": False, "error": "IK failed for place approach"}
        min_drop_z = ctx.deps.cfg.agents.vision.min_drop_z_m
        computed_drop_z = plz + drop_height_m
        drop_z = max(computed_drop_z, min_drop_z)
        if _logfire:
            _logfire.info("vision.drop_z",
                          computed_m=round(computed_drop_z, 4),
                          applied_m=round(drop_z, 4),
                          clamped=drop_z > computed_drop_z,
                          min_drop_z_m=min_drop_z)
        j_place_drop = get_joints_for_pose("place_drop", plx, ply, round(drop_z, 4), j_place_approach, arm_mcp_url=None)
        if j_place_drop is None:
            return {"success": False, "error": "IK failed for place drop"}

        await arm.call_tool("move_joints", {"joints": j_place_approach, "speed": 0.3, "accel": 0.5})
        await arm.call_tool("move_joints", {"joints": j_place_drop,     "speed": 0.1, "accel": 0.2})
        if gripper:
            await gripper.call_tool("open", {"speed": 100, "force": 100})
        await arm.call_tool("move_joints", {"joints": j_place_approach, "speed": 0.2, "accel": 0.4})
        await arm.call_tool("move_named_pose", {"name": "home", "speed": 0.5, "accel": 0.8})

        return {
            "success": True,
            "summary": f"Picked '{pick_description}' at ({px_b:.3f},{py_b:.3f},{pz_b:.3f}) and placed onto '{place_description}' at ({plx:.3f},{ply:.3f},{plz:.3f})",
        }

    @agent.tool
    async def pick_object(
        ctx: RunContext[VisionAgentDeps],
        description: str,
    ) -> dict:
        """
        Detect an object with Gemini ER + RealSense, compute IK, and execute a
        pick-and-lift motion directly — no YAML scenario file is created.

        Use this when the user says "pick", "grab", "lift", or "get" an object.

        Steps: home → open gripper → approach → descend → close gripper → lift.

        Args:
            description: What to pick, e.g. "bottle on the table".
        """
        from cobot.agents.vision_agent.camera import (
            capture_realsense, encode_jpeg, depth_at_pixel, depth_at_box,
            pixel_to_camera_xyz, camera_to_base,
        )
        from cobot.agents.vision_agent.kinematics import (
            get_joints_for_pose, APPROACH_Z_OFFSET, PICK_Z_OFFSET, LIFT_Z_OFFSET, _PICK_SEED,
        )

        if ctx.deps.detector is None:
            return {"success": False, "error": "GOOGLE_API_KEY not set"}
        if ctx.deps.arm_mcp is None:
            return {"success": False, "error": "arm_mcp not wired — cannot move robot"}

        arm     = ctx.deps.arm_mcp
        gripper = ctx.deps.gripper_mcp
        T       = ctx.deps.get_T_base_cam()

        # 1. Detect object
        color, depth, ci, ds = capture_realsense()
        h, w = color.shape[:2]
        det = ctx.deps.detector.locate( encode_jpeg(color), description)
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
        lift_z     = round(bz + LIFT_Z_OFFSET,      4)

        # 2. IK for approach only (coarse estimate)
        mcp_url = ctx.deps.arm_mcp_url
        j_approach = get_joints_for_pose("approach", bx, by, approach_z, _PICK_SEED, arm_mcp_url=mcp_url)
        if j_approach is None:
            return {"success": False, "error": f"IK failed for approach ({bx:.3f},{by:.3f},{approach_z:.3f})"}

        # 3. Move home + activate/open gripper, then move to approach
        await arm.call_tool("move_named_pose", {"name": "home", "speed": 0.5, "accel": 0.8})

        if gripper:
            await gripper.call_tool("open", {"speed": 100, "force": 100})

        await arm.call_tool("move_joints", {"joints": j_approach, "speed": 0.3, "accel": 0.5})

        # 4. Re-detect from approach position for accurate grasp/lift IK
        print("  [pick] re-detecting from approach position …")
        color2, depth2, ci2, ds2 = capture_realsense()
        h2, w2 = color2.shape[:2]
        det2 = ctx.deps.detector.locate( encode_jpeg(color2), description)
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
                grasp_z = round(bz + PICK_Z_OFFSET,  4)
                lift_z  = round(bz + LIFT_Z_OFFSET,  4)
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
            await gripper.call_tool("close", {"speed": 100, "force": 100})

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
        from cobot.agents.vision_agent.kinematics import get_joints_for_pose, _PICK_SEED

        if ctx.deps.detector is None:
            return {"success": False, "error": "GOOGLE_API_KEY not set"}
        if ctx.deps.arm_mcp is None:
            return {"success": False, "error": "arm_mcp not wired — cannot move robot"}

        arm = ctx.deps.arm_mcp
        T   = ctx.deps.get_T_base_cam()

        color, depth, ci, ds = capture_realsense()
        h, w = color.shape[:2]
        det = ctx.deps.detector.locate( encode_jpeg(color), description)
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
        if ctx.deps.detector is None:
            return {"success": False, "error": "GOOGLE_API_KEY not set"}

        color, _, _, _ = capture_realsense()
        det = ctx.deps.detector.locate(
            encode_jpeg(color),
            "any obstacle or object in the robot arm's movement path",
        )
        clear = det is None
        return {
            "workspace_clear": clear,
            "message": "No obstacles detected." if clear else f"Possible obstacle: {det}",
        }

    return agent
