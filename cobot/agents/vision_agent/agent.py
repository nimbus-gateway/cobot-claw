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
    _T_base_cam: np.ndarray | None = field(default=None, repr=False)
    # Cached frame from the last capture_scene call — reused by locate_object
    _frame: tuple | None = field(default=None, repr=False)   # (color, depth, ci, ds)
    _annotated: np.ndarray | None = field(default=None, repr=False)

    # Debug images are written here
    debug_image_path: str = field(
        default_factory=lambda: str(Path(__file__).parent / "vision_scene.jpg")
    )

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
            "Workflow for a pick/place request:\n"
            "1. Call capture_scene() to get a fresh camera frame.\n"
            "2. Call locate_object(description) to find the target in 3D.\n"
            "3. Return position_base_m (robot base frame) and approach_pose_m.\n\n"
            "For workspace safety checks call check_workspace_clear().\n"
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
        )
        return result

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
