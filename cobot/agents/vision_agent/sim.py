"""Simulated vision for URSim demos (agents.vision.detector: sim).

Instead of a RealSense camera and Gemini, objects come from the simulated
scene server (tools/sim_scene/scene_server.py), which knows where every
object is. The motion sequences mirror the camera-based tools in agent.py
(same IK calls, poses, speeds and order). Only detection is replaced, along
with the camera-only re-detect step during a pick.

Height conventions: the camera path gives the top surface of an object and
the flange is sent PICK_Z_OFFSET above it. In the simulation the gripper's
length is known (the scene server reports grasp_depth_m: flange to finger
pads), so heights are converted to put the finger pads on the object.
"""

from __future__ import annotations

from typing import Any

from cobot.agents.vision_agent.kinematics import (
    APPROACH_Z_OFFSET,
    LIFT_Z_OFFSET,
    PICK_Z_OFFSET,
    _PICK_SEED,
    get_joints_for_pose,
)

# Grip tall objects this far below their top instead of at their centre.
_GRIP_BELOW_TOP_M = 0.025


class MoveFailed(Exception):
    pass


async def _move(arm, tool: str, args: dict[str, Any]) -> None:
    """Call an arm tool and raise MoveFailed if it reports failure (the
    camera-based tools ignore these results)."""
    res = await arm.call_tool(tool, args)
    if isinstance(res, dict) and res.get("ok") is False:
        raise MoveFailed(f"{tool} failed: {res.get('message', res)}")


async def sim_locate(deps, description: str) -> dict[str, Any]:
    """Ask the scene server for an object. Returns its reply (ok, name, center_m, ...)."""
    try:
        return await deps.sim_scene.call_tool("locate_object", {"description": description})
    except Exception as e:  # scene server offline or MCP error
        return {"ok": False, "message": f"Simulated scene unavailable: {e}"}


def _pick_reference_z(obj: dict[str, Any]) -> float:
    """Value for the pick code's 'detected surface' z, so that the flange at
    bz + PICK_Z_OFFSET puts the finger pads at the grip height."""
    grip_z = max(obj["center_m"]["z"], obj["top_z_m"] - _GRIP_BELOW_TOP_M)
    return grip_z + obj["grasp_depth_m"] - PICK_Z_OFFSET


def _above_top_z(obj: dict[str, Any]) -> float:
    """Flange height with the finger pads level with the top of `obj`."""
    return obj["top_z_m"] + obj["grasp_depth_m"]


async def sim_capture_scene(deps) -> dict[str, Any]:
    res = await deps.sim_scene.call_tool("list_objects", {})
    items = res.get("items", []) if isinstance(res, dict) else []
    return {
        "simulated": True,
        "objects": [f"{i['name']} ({i['kind']})" for i in items],
    }


async def sim_locate_object(deps, description: str) -> dict[str, Any]:
    obj = await sim_locate(deps, description)
    if not obj.get("ok"):
        return {"success": False, "error": obj.get("message", f"Could not locate '{description}'"),
                "available": obj.get("available")}
    c = obj["center_m"]
    return {
        "success": True,
        "label": obj["name"],
        "simulated": True,
        "position_base_m": c,
        "top_z_m": obj["top_z_m"],
        "size_m": obj["size_m"],
        "held": obj["held"],
        "approach_pose_m": {"x": c["x"], "y": c["y"],
                            "z": round(_above_top_z(obj) + APPROACH_Z_OFFSET, 4)},
    }


async def sim_pick_object(deps, description: str) -> dict[str, Any]:
    try:
        return await _sim_pick_object_impl(deps, description)
    except MoveFailed as e:
        return {"success": False, "simulated": True, "error": str(e)}


async def _sim_pick_object_impl(deps, description: str) -> dict[str, Any]:
    if deps.arm_mcp is None:
        return {"success": False, "error": "arm_mcp not wired — cannot move robot"}
    obj = await sim_locate(deps, description)
    if not obj.get("ok"):
        return {"success": False, "error": obj.get("message", f"Could not locate '{description}'"),
                "available": obj.get("available")}
    if obj.get("kind") != "object":
        return {"success": False, "error": f"'{obj['name']}' is a container and can't be picked"}

    arm, gripper = deps.arm_mcp, deps.gripper_mcp
    bx, by = obj["center_m"]["x"], obj["center_m"]["y"]
    bz = _pick_reference_z(obj)
    approach_z = round(bz + APPROACH_Z_OFFSET, 4)
    grasp_z = round(bz + PICK_Z_OFFSET, 4)
    lift_z = round(bz + LIFT_Z_OFFSET, 4)

    j_approach = get_joints_for_pose("approach", bx, by, approach_z, _PICK_SEED,
                                     arm_mcp_url=deps.arm_mcp_url)
    if j_approach is None:
        return {"success": False, "error": f"IK failed for approach ({bx:.3f},{by:.3f},{approach_z:.3f})"}
    j_grasp = get_joints_for_pose("grasp", bx, by, grasp_z, j_approach, arm_mcp_url=None)
    if j_grasp is None:
        return {"success": False, "error": f"IK failed for grasp ({bx:.3f},{by:.3f},{grasp_z:.3f})"}
    j_lift = get_joints_for_pose("lift", bx, by, lift_z, j_approach, arm_mcp_url=None)
    if j_lift is None:
        return {"success": False, "error": f"IK failed for lift ({bx:.3f},{by:.3f},{lift_z:.3f})"}

    await _move(arm, "move_named_pose", {"name": "home", "speed": 0.5, "accel": 0.8})
    if gripper:
        await gripper.call_tool("open", {"speed": 100, "force": 100})
    await _move(arm, "move_joints", {"joints": j_approach, "speed": 0.3, "accel": 0.5})
    await _move(arm, "move_joints", {"joints": j_grasp, "speed": 0.1, "accel": 0.2})
    if gripper:
        await gripper.call_tool("close", {"speed": 100, "force": 100})
    await _move(arm, "move_joints", {"joints": j_lift, "speed": 0.2, "accel": 0.4})

    after = await sim_locate(deps, obj["name"])
    held = bool(after.get("held"))
    return {
        "success": held,
        "label": obj["name"],
        "simulated": True,
        "position_m": obj["center_m"],
        "summary": (f"Picked '{obj['name']}' at ({bx:.3f},{by:.3f}) and lifted it"
                    if held else
                    f"Moved to '{obj['name']}' and closed the gripper, but the object was not grasped"),
    }


async def sim_place_object(deps, cfg, place_description: str) -> dict[str, Any]:
    try:
        return await _sim_place_object_impl(deps, cfg, place_description)
    except MoveFailed as e:
        return {"success": False, "simulated": True, "error": str(e)}


async def _sim_place_object_impl(deps, cfg, place_description: str) -> dict[str, Any]:
    if deps.arm_mcp is None:
        return {"success": False, "error": "arm_mcp not wired — cannot move robot"}
    target = await sim_locate(deps, place_description)
    if not target.get("ok"):
        return {"success": False,
                "error": target.get("message", f"Could not locate place target '{place_description}'"),
                "available": target.get("available")}

    arm, gripper = deps.arm_mcp, deps.gripper_mcp
    plx, ply = target["center_m"]["x"], target["center_m"]["y"]
    plz = _above_top_z(target)

    current_joints: list[float] | None = None
    try:
        js = await arm.call_tool("get_joint_states", {})
        raw = js.get("joints_rad") or js.get("joints")
        if raw and len(raw) >= 6:
            current_joints = [e["value_rad"] if isinstance(e, dict) else e for e in raw]
    except Exception:
        pass

    place_approach_z = round(max(plz + APPROACH_Z_OFFSET, cfg.agents.vision.min_drop_z_m), 4)
    j_place = get_joints_for_pose("place_approach", plx, ply, place_approach_z,
                                  q_seed=current_joints, arm_mcp_url=deps.arm_mcp_url)
    if j_place is None:
        return {"success": False, "error": "IK failed for place approach"}

    await _move(arm, "move_joints", {"joints": j_place, "speed": 0.3, "accel": 0.5})
    if gripper:
        await gripper.call_tool("open", {"speed": 100, "force": 100})
    await _move(arm, "move_joints", {"joints": j_place, "speed": 0.2, "accel": 0.4})
    await _move(arm, "move_named_pose", {"name": "home", "speed": 0.5, "accel": 0.8})
    return {
        "success": True,
        "simulated": True,
        "summary": f"Placed object onto '{target['name']}' at ({plx:.3f},{ply:.3f}), released at z={place_approach_z}m",
    }


async def sim_point_to_object(deps, description: str, hover_height_m: float) -> dict[str, Any]:
    try:
        return await _sim_point_to_object_impl(deps, description, hover_height_m)
    except MoveFailed as e:
        return {"success": False, "simulated": True, "error": str(e)}


async def _sim_point_to_object_impl(deps, description: str, hover_height_m: float) -> dict[str, Any]:
    if deps.arm_mcp is None:
        return {"success": False, "error": "arm_mcp not wired — cannot move robot"}
    obj = await sim_locate(deps, description)
    if not obj.get("ok"):
        return {"success": False, "error": obj.get("message", f"Could not locate '{description}'"),
                "available": obj.get("available")}
    bx, by = obj["center_m"]["x"], obj["center_m"]["y"]
    hover_z = round(_above_top_z(obj) + hover_height_m, 4)
    j_hover = get_joints_for_pose("hover", bx, by, hover_z, _PICK_SEED, arm_mcp_url=None)
    if j_hover is None:
        return {"success": False, "error": f"IK failed for hover ({bx:.3f},{by:.3f},{hover_z:.3f})"}
    await _move(deps.arm_mcp, "move_named_pose", {"name": "home", "speed": 0.5, "accel": 0.8})
    await _move(deps.arm_mcp, "move_joints", {"joints": j_hover, "speed": 0.3, "accel": 0.5})
    return {
        "success": True,
        "simulated": True,
        "label": obj["name"],
        "position_m": obj["center_m"],
        "hover_z": hover_z,
        "summary": f"Pointing at '{obj['name']}', fingertips {hover_height_m:.2f} m above it",
    }


async def sim_check_workspace(deps) -> dict[str, Any]:
    res = await deps.sim_scene.call_tool("list_objects", {})
    items = res.get("items", []) if isinstance(res, dict) else []
    return {
        "workspace_clear": True,
        "simulated": True,
        "message": "Simulated scene. Objects: " + ", ".join(
            f"{i['name']} at ({i['center_m']['x']:.2f}, {i['center_m']['y']:.2f})" for i in items),
    }
