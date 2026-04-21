"""Vision-guided scenario generator: Gemini ER + RealSense → move_joints YAML."""

from __future__ import annotations

import os
from pathlib import Path

import cv2
import numpy as np
import yaml as _yaml
from google import genai

from cobot.agents.vision_agent.camera import (
    load_calibration, capture_realsense, encode_jpeg,
    depth_at_pixel, depth_at_box, pixel_to_camera_xyz, camera_to_base,
    CALIB_FILE,
)
from cobot.agents.vision_agent.gemini import gemini_locate
from cobot.agents.vision_agent.kinematics import (
    ur3e_fk, get_joints_for_pose,
    APPROACH_Z_OFFSET, PICK_Z_OFFSET, _PICK_SEED,
)

_REPO_ROOT     = Path(__file__).parent.parent.parent.parent
SCENARIO_STORE = Path(os.getenv("SCENARIO_STORE", str(_REPO_ROOT / "cobot/scenarios/store")))


def generate_scenario(
    pick: str,
    place: str,
    name: str = "vision_pick",
    speed: float = 0.3,
    accel: float = 0.5,
    calib_file: str = CALIB_FILE,
    scenario_store: Path = SCENARIO_STORE,
    save_debug_image: bool = True,
    gemini_client=None,
    cfg=None,
) -> dict:
    """
    Locate objects with Gemini ER + RealSense, compute joint angles with local
    UR3e IK, and write a move_joints scenario YAML to scenario_store.

    Args:
        pick:            Natural-language description of the pick object.
        place:           Natural-language description of the place target.
        name:            Scenario filename (without .yaml).
        speed / accel:   Default joint speed/accel for transit moves.
        calib_file:      Path to calibration.json.
        scenario_store:  Directory to write the scenario YAML.
        save_debug_image: Write <name>_vision.jpg with annotated detections.

    Returns dict with keys: success, scenario_name, out_path, pick_base,
    place_base, message.
    """
    if gemini_client is None:
        google_key = (getattr(getattr(cfg, "agents", None) and getattr(cfg.agents, "vision", None), "google_api_key", "") or
                      os.environ.get("GOOGLE_API_KEY", "")) if cfg is not None else os.environ.get("GOOGLE_API_KEY", "")
        if not google_key:
            return {"success": False, "message": "google_api_key not set in cobot.yaml (agents.vision.google_api_key)"}
        gemini_client = genai.Client(api_key=google_key)

    print(f"\nGenerating scenario: pick '{pick}' → place '{place}'")
    print("=" * 60)

    T_base_cam = load_calibration(calib_file)
    gemini     = gemini_client

    print("\n[vision] Capturing frame...")
    color, depth, ci, depth_scale = capture_realsense()
    h, w = color.shape[:2]

    print(f"[vision] Locating: '{pick}'")
    pick_det = gemini_locate(gemini, encode_jpeg(color), pick)
    if not pick_det:
        return {"success": False, "message": f"Could not locate '{pick}'"}

    print(f"[vision] Locating: '{place}'")
    place_det = gemini_locate(gemini, encode_jpeg(color), place)
    if not place_det:
        return {"success": False, "message": f"Could not locate '{place}'"}

    def det_to_base(det: dict, fallback_depth_m: float | None = None):
        ny, nx = det["point"]
        px = int(nx / 1000 * w)
        py = int(ny / 1000 * h)
        box = det.get("box_2d")
        if box:
            ymin, xmin, ymax, xmax = box
            x1 = int(xmin / 1000 * w); y1 = int(ymin / 1000 * h)
            x2 = int(xmax / 1000 * w); y2 = int(ymax / 1000 * h)
            d_m = depth_at_box(depth, depth_scale, x1, y1, x2, y2)
            if d_m >= 0.05:
                print(f"    [depth] box region median depth={d_m:.3f}m")
            else:
                d_m = depth_at_pixel(depth, ci, px, py, depth_scale)
        else:
            d_m = depth_at_pixel(depth, ci, px, py, depth_scale)
        if d_m < 0.05:
            if fallback_depth_m and fallback_depth_m > 0.05:
                d_m = fallback_depth_m
                print(f"    [depth] no reading at ({px},{py}), using fallback {d_m:.3f}m")
            else:
                print(f"    [depth] WARNING: no depth at ({px},{py}) and no fallback")
        xyz = pixel_to_camera_xyz(ci, px, py, d_m)
        return camera_to_base(xyz, T_base_cam), px, py, d_m

    pick_base,  pick_px,  pick_py,  pick_d  = det_to_base(pick_det)
    place_base, place_px, place_py, place_d = det_to_base(place_det, fallback_depth_m=pick_d)

    print(f"\n  Pick  '{pick}':  depth={pick_d:.3f}m  base={pick_base}")
    print(f"  Place '{place}': depth={place_d:.3f}m  base={place_base}")

    if save_debug_image:
        ann = color.copy()
        cv2.circle(ann, (pick_px, pick_py), 12, (0, 255, 0), -1)
        cv2.putText(ann, f"PICK {pick_d:.2f}m", (pick_px + 12, pick_py),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        cv2.circle(ann, (place_px, place_py), 12, (0, 0, 255), -1)
        cv2.putText(ann, f"PLACE {place_d:.2f}m", (place_px + 12, place_py),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        cv2.imwrite(f"{name}_vision.jpg", ann)
        print(f"  Saved {name}_vision.jpg")

    print("\n[IK] Computing joint angles (local solver)...")
    _T = ur3e_fk(_PICK_SEED)
    print(f"  [debug] FK(PICK_SEED) = {_T[:3, 3].round(4).tolist()}")

    pick_x,  pick_y  = pick_base["x"],  pick_base["y"]
    place_x, place_y = place_base["x"], place_base["y"]
    pick_z_approach  = round(pick_base["z"]  + APPROACH_Z_OFFSET, 4)
    pick_z_grasp     = round(pick_base["z"]  + PICK_Z_OFFSET,     4)
    place_z_approach = round(place_base["z"] + APPROACH_Z_OFFSET, 4)
    place_z_drop     = round(place_base["z"] + PICK_Z_OFFSET,     4)

    def ik(label: str, x: float, y: float, z: float,
           q_seed: list[float] | None = None) -> list[float]:
        j = get_joints_for_pose(label, x, y, z, q_seed)
        if j is None:
            raise RuntimeError(f"IK failed for '{label}' at ({x:.3f},{y:.3f},{z:.3f})")
        return j

    try:
        j_pick_approach  = ik("pick_approach",  pick_x,  pick_y,  pick_z_approach)
        j_pick_grasp     = ik("pick_grasp",     pick_x,  pick_y,  pick_z_grasp,    q_seed=j_pick_approach)
        j_place_approach = ik("place_approach", place_x, place_y, place_z_approach)
        j_place_drop     = ik("place_drop",     place_x, place_y, place_z_drop,    q_seed=j_place_approach)
    except RuntimeError as e:
        return {"success": False, "message": str(e)}

    def jstep(joints, spd=None, acc=None, blend=0.0):
        return {"type": "move_joints", "joints": joints,
                "speed": spd or speed, "accel": acc or accel,
                "blend_radius": blend}

    steps = [
        {"type": "move_named_pose", "pose": "home", "speed": 0.5, "accel": 0.8},
        {"type": "tool_action", "tool": "gripper", "action": "open",
         "params": {"speed": 100, "force": 100}, "arm_recovery_sec": 3.0},
        jstep(j_pick_approach,  spd=0.4,  acc=0.6),
        {
            "type": "visual_servo",
            "pick_description": pick,
            "approach_z": pick_z_approach,
            "grasp_z": pick_z_grasp,
            "fallback_joints": j_pick_grasp,
            "speed": 0.05,
            "accel": 0.2,
            "threshold_mm": 2.0,
        },
        {"type": "tool_action", "tool": "gripper", "action": "close",
         "params": {"speed": 100, "force": 100}, "arm_recovery_sec": 3.0},
        jstep(j_pick_approach,  spd=0.3,  acc=0.5),
        jstep(j_place_approach, spd=0.4,  acc=0.6),
        jstep(j_place_drop,     spd=0.15, acc=0.3),
        {"type": "tool_action", "tool": "gripper", "action": "open",
         "params": {"speed": 100, "force": 100}, "arm_recovery_sec": 3.0},
        jstep(j_place_approach, spd=0.3,  acc=0.5),
        {"type": "move_named_pose", "pose": "home", "speed": 0.5, "accel": 0.8},
    ]

    scenario = {
        "name": name,
        "description": (
            f"Vision pick-and-place: '{pick}' \u2192 '{place}'. "
            f"pick={pick_base}  place={place_base}"
        ),
        "robot": "ur3e",
        "steps": steps,
    }

    out_path = scenario_store / f"{name}.yaml"
    scenario_store.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        _yaml.dump(scenario, f, default_flow_style=False, sort_keys=False)

    print(f"\n[done] Saved → {out_path}")
    print(f"       Dry run:  cobot run {name} --dry-run")
    print(f"       Live run: cobot run {name}")

    return {
        "success":       True,
        "scenario_name": name,
        "out_path":      str(out_path),
        "pick_base":     pick_base,
        "place_base":    place_base,
        "message":       f"Scenario saved to {out_path}",
    }
