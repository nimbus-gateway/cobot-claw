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
from cobot.agents.vision_agent import sim as _sim


@dataclass
class VisionAgentDeps:
    cfg: CobotConfig
    calib_file: str = field(default_factory=lambda: CALIB_FILE)
    detector: VisionDetector | None = None
    arm_mcp: object | None = None       # MCPClient — injected at session startup
    gripper_mcp: object | None = None   # MCPClient — injected at session startup
    camera_stream: object | None = None  # CameraStream — set when /vision-debug is active
    sim_scene: object | None = None     # MCPClient for tools/sim_scene — set when detector == "sim"
    task_log: object | None = None      # cobot.tasklog.TaskLog — what the robot is doing, shown live
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
        """Return cached frame. Uses stream if active, otherwise captures directly."""
        if self._frame is None:
            color, depth, ci, ds = self._capture()
            self.set_frame(color, depth, ci, ds)
        return self._frame

    def capture_fresh_frame(self):
        """Always capture a new frame, bypassing the cache."""
        color, depth, ci, ds = self._capture()
        self.set_frame(color, depth, ci, ds)
        return self._frame

    def _capture(self) -> tuple:
        """Single capture entry point — uses stream when active, otherwise direct."""
        if self.camera_stream is not None:
            return self.camera_stream.get_latest_frame()
        return capture_realsense()

    def locate_sync(self, description: str) -> dict | None:
        """Capture a fresh frame, locate object, return base-frame {x,y,z} or None."""
        if self.detector is None or self.sim_scene is not None:
            return None
        from cobot.agents.vision_agent.camera import (
            encode_jpeg, depth_at_pixel, depth_at_box,
            pixel_to_camera_xyz, camera_to_base,
        )
        T = self.get_T_base_cam()
        color, depth, ci, ds = self._capture()
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

    def annotate(self, px: int, py: int, label: str, d_m: float,
                 box: tuple | None = None) -> None:
        """Draw a detection on the local annotated image and on the stream window."""
        if self._annotated is None:
            return
        if box:
            x1, y1, x2, y2 = box
            cv2.rectangle(self._annotated, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.circle(self._annotated, (px, py), 10, (0, 255, 0), -1)
        cv2.putText(self._annotated, f"{label} {d_m:.2f}m", (px + 12, py),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        cv2.imwrite(self.debug_image_path, self._annotated)
        if self.camera_stream is not None:
            self.camera_stream.overlay_detection(label, px, py, d_m, box)


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
    "- User says 'now place it on Y', 'put it in Y', 'drop it on Y' (object already held) → call place_object().\n"
    "- User says 'pick X and put/drop/place/move it to/in/on Y' → call pick_object() THEN call place_object(). Two separate calls.\n"
    "- User says 'point to', 'show me', 'indicate', 'hover over' an object → call point_to_object().\n"
    "- User says 'where is X', 'find X' → call locate_object() once.\n"
    "- User says 'what do you see', 'what is on the table', 'workspace check' → call check_workspace_clear() once.\n"
    "- User says 'inspect X', 'check X for defects', 'is X good' → call inspect_object() once.\n"
    "- User says 'sort the parts', 'run quality sorting', 'inspect and sort' → call sort_parts() ONCE. It inspects,\n"
    "  picks and places every part itself; do not call pick/place/inspect for the same parts.\n"
    "- User says 'reset the scene', 'put the parts back' → call reset_scene() once.\n"
    "- Packing: 'pack X and Y into box Z' → call pack_items() once with those items and the box.\n"
    "  'Pack the orders', 'fill order 1', 'what orders are open' → call capture_scene() first: it lists\n"
    "  the orders (what each needs and what is missing) and every product with where it is. Choose\n"
    "  products that are NOT already in a shipping box, by their exact names, then call pack_items()\n"
    "  once per order with exactly the missing products. Don't take a product another order needs.\n"
    "- Do NOT call capture_scene() — it is called automatically by locate_object() (except for orders, see above).\n"
    "Never generate YAML scenario files — that is the Config Agent's responsibility.\n"
    "If tool results say simulated=True you are driving a simulated cell (URSim): objects come\n"
    "from a scene file, not the camera. Use the same tools; the object names from the scene\n"
    "(e.g. 'red block', 'grey bin') work best as descriptions.\n\n"
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
        if ctx.deps.sim_scene is not None:
            return await _sim.sim_capture_scene(ctx.deps)
        color, depth, ci, ds = ctx.deps._capture()
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
        if ctx.deps.sim_scene is not None:
            return await _sim.sim_locate_object(ctx.deps, description)
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
        if ctx.deps.sim_scene is not None:
            return await _sim.sim_pick_object(ctx.deps, description)
        from cobot.agents.vision_agent.camera import (
            encode_jpeg, depth_at_pixel, depth_at_box,
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
        color, depth, ci, ds = ctx.deps._capture()
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
        color2, depth2, ci2, ds2 = ctx.deps._capture()
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
    async def place_object(
        ctx: RunContext[VisionAgentDeps],
        place_description: str,
    ) -> dict:
        """
        Place the currently held object onto a target location detected by the camera.

        Call this AFTER pick_object() — the arm must already be in lift position holding
        an object. Detects the place target from the current arm position, moves above it,
        releases the gripper, then returns home.

        Use when the user says "now place it on X", "put it in Y", "drop it on Z" as a
        follow-up after a pick.

        Args:
            place_description: Target location, e.g. "cardboard box", "left side of table".
        """
        if ctx.deps.sim_scene is not None:
            return await _sim.sim_place_object(ctx.deps, ctx.deps.cfg, place_description)
        from cobot.agents.vision_agent.camera import (
            encode_jpeg, depth_at_pixel, depth_at_box_top,
            pixel_to_camera_xyz, camera_to_base,
        )
        from cobot.agents.vision_agent.kinematics import (
            get_joints_for_pose, APPROACH_Z_OFFSET,
        )

        if ctx.deps.detector is None:
            return {"success": False, "error": "Vision detector not configured"}
        if ctx.deps.arm_mcp is None:
            return {"success": False, "error": "arm_mcp not wired — cannot move robot"}

        arm     = ctx.deps.arm_mcp
        gripper = ctx.deps.gripper_mcp
        T       = ctx.deps.get_T_base_cam()
        mcp_url = ctx.deps.arm_mcp_url

        # Detect place target from current (lift) position
        print(f"  [place] detecting place target '{place_description}' from current position …")
        color, depth, ci, ds = ctx.deps._capture()
        h, w = color.shape[:2]
        det = ctx.deps.detector.locate(encode_jpeg(color), place_description)
        if det is None:
            return {"success": False, "error": f"Could not locate place target '{place_description}'"}

        ny, nx = det["point"]
        px = int(nx / 1000 * w); py = int(ny / 1000 * h)
        box = det.get("box_2d")
        if box:
            ymin, xmin, ymax, xmax = box
            x1 = int(xmin/1000*w); y1 = int(ymin/1000*h)
            x2 = int(xmax/1000*w); y2 = int(ymax/1000*h)
            top_fraction = 0.3
            strip_h = max(1, int((y2 - y1) * top_fraction))
            # Use horizontal center of box, but vertical center of top strip
            # so back-projection lands on the top surface, not the front face
            px = (x1 + x2) // 2
            py = y1 + strip_h // 2
            d_m = depth_at_box_top(depth, ds, x1, y1, x2, y2, top_fraction)
            if d_m < 0.05:
                d_m = depth_at_pixel(depth, ci, px, py, ds)
        else:
            d_m = depth_at_pixel(depth, ci, px, py, ds)

        if d_m < 0.05:
            return {"success": False, "error": "No depth reading for place target"}

        depth_offset = ctx.deps.cfg.agents.vision.place_depth_offset_m
        if depth_offset:
            d_m = max(0.05, d_m - depth_offset)

        xyz      = pixel_to_camera_xyz(ci, px, py, d_m)
        base_pos = camera_to_base(xyz, T)
        plx, ply, plz = base_pos["x"], base_pos["y"], base_pos["z"]
        print(f"  [place] target at ({plx:.3f},{ply:.3f},{plz:.3f})")

        if _logfire:
            _logfire.info("vision.detection", object=place_description, role="place",
                          base_x=round(plx, 4), base_y=round(ply, 4), base_z=round(plz, 4))

        # Use current joint state as IK seed — arm is already in lift position after pick,
        # so this gives the solver the correct configuration family for the transit.
        current_joints: list[float] | None = None
        try:
            js = await arm.call_tool("get_joint_states", {})
            raw = js.get("joints_rad") or js.get("joints")
            if raw and len(raw) >= 6:
                current_joints = [e["value_rad"] if isinstance(e, dict) else e for e in raw]
        except Exception:
            pass

        # IK for approach — release from approach height, no descent
        place_approach_z = round(max(plz + APPROACH_Z_OFFSET, ctx.deps.cfg.agents.vision.min_drop_z_m), 4)

        if _logfire:
            _logfire.info("vision.drop_z", applied_m=place_approach_z,
                          clamped=place_approach_z > plz + APPROACH_Z_OFFSET,
                          min_drop_z_m=ctx.deps.cfg.agents.vision.min_drop_z_m)

        j_place_approach = get_joints_for_pose("place_approach", plx, ply, place_approach_z,
                                               q_seed=current_joints, arm_mcp_url=mcp_url)
        if j_place_approach is None:
            return {"success": False, "error": "IK failed for place approach"}

        # Execute place — open gripper at approach height, no descend
        await arm.call_tool("move_joints", {"joints": j_place_approach, "speed": 0.3, "accel": 0.5})
        if gripper:
            await gripper.call_tool("open", {"speed": 100, "force": 100})
        await arm.call_tool("move_joints", {"joints": j_place_approach, "speed": 0.2, "accel": 0.4})
        await arm.call_tool("move_named_pose", {"name": "home", "speed": 0.5, "accel": 0.8})

        return {
            "success": True,
            "summary": f"Placed object onto '{place_description}' at ({plx:.3f},{ply:.3f},{plz:.3f}), released at z={place_approach_z}m",
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
        if ctx.deps.sim_scene is not None:
            return await _sim.sim_point_to_object(ctx.deps, description, hover_height_m)
        from cobot.agents.vision_agent.camera import (
            encode_jpeg, depth_at_pixel, depth_at_box,
            pixel_to_camera_xyz, camera_to_base,
        )
        from cobot.agents.vision_agent.kinematics import get_joints_for_pose, _PICK_SEED

        if ctx.deps.detector is None:
            return {"success": False, "error": "GOOGLE_API_KEY not set"}
        if ctx.deps.arm_mcp is None:
            return {"success": False, "error": "arm_mcp not wired — cannot move robot"}

        arm = ctx.deps.arm_mcp
        T   = ctx.deps.get_T_base_cam()

        color, depth, ci, ds = ctx.deps._capture()
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
    async def inspect_object(ctx: RunContext[VisionAgentDeps], description: str) -> dict:
        """
        Quality-inspect one part with the cell's inspection camera: returns
        result "pass" or "fail" and the defect found. Simulated cell only.

        Args:
            description: the part, e.g. 'part 3'.
        """
        if ctx.deps.sim_scene is not None:
            return await _sim.sim_inspect_object(ctx.deps, description)
        return {"success": False,
                "error": "Quality inspection is only available in the simulated cell (detector: sim)"}

    @agent.tool
    async def sort_parts(
        ctx: RunContext[VisionAgentDeps],
        source: str = "incoming tray",
        pass_target: str = "pass bin",
        fail_target: str = "reject bin",
    ) -> dict:
        """
        Quality sorting: inspect every part in `source`, then pick it and place
        it in `pass_target` if it passes or `fail_target` if it fails, one part
        at a time. Call once; it handles all the parts. Simulated cell only.

        Args:
            source: container holding the parts to sort (default 'incoming tray').
            pass_target: container for good parts (default 'pass bin').
            fail_target: container for defective parts (default 'reject bin').
        """
        if ctx.deps.sim_scene is not None:
            return await _sim.sim_sort_parts(ctx.deps, ctx.deps.cfg, source, pass_target, fail_target)
        return {"success": False,
                "error": "Quality sorting is only available in the simulated cell (detector: sim)"}

    @agent.tool
    async def pack_items(ctx: RunContext[VisionAgentDeps], items: list[str], container: str) -> dict:
        """
        Packing: pick each item and place it in `container` (e.g. 'shipping box 1'),
        side by side on its floor. Call once per box with all its items.
        Simulated cell only.

        Args:
            items: exact product names, e.g. ['red pack 1', 'blue pack 2'].
            container: the box to pack them into.
        """
        if ctx.deps.sim_scene is not None:
            return await _sim.sim_pack_items(ctx.deps, ctx.deps.cfg, items, container)
        return {"success": False, "error": "Packing is only available in the simulated cell (detector: sim)"}

    @agent.tool
    async def reset_scene(ctx: RunContext[VisionAgentDeps]) -> dict:
        """
        Simulated cell only: put every object back where the scene starts and
        clear inspection results. Opens the gripper first so nothing stays held.
        """
        if ctx.deps.sim_scene is None:
            return {"success": False, "error": "There is no simulated scene to reset (detector is not 'sim')"}
        if ctx.deps.gripper_mcp is not None:
            await ctx.deps.gripper_mcp.call_tool("open", {"speed": 100, "force": 100})
        res = await ctx.deps.sim_scene.call_tool("reset_scene", {})
        return {"success": bool(res.get("ok")), "simulated": True,
                "summary": "Scene reset: " + ", ".join(res.get("items", []))}

    @agent.tool
    async def check_workspace_clear(ctx: RunContext[VisionAgentDeps]) -> dict:
        """
        Ask Gemini ER whether the robot workspace is clear of unexpected obstacles.
        Returns a safety assessment before large arm movements.
        """
        if ctx.deps.sim_scene is not None:
            return await _sim.sim_check_workspace(ctx.deps)
        if ctx.deps.detector is None:
            return {"success": False, "error": "GOOGLE_API_KEY not set"}

        color, _, _, _ = ctx.deps._capture()
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
