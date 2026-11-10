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

from cobot.tasklog import log_event
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
    out: dict[str, Any] = {
        "simulated": True,
        "scene": res.get("scene", "") if isinstance(res, dict) else "",
        "objects": [f"{i['name']} ({i['kind']}"
                    + (f", in {i['in_container']}" if i.get("in_container") else "") + ")"
                    for i in items],
    }
    if isinstance(res, dict) and res.get("orders"):
        out["orders"] = res["orders"]
    return out


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

    log_event(deps, "step", f"picking {obj['name']}", object=obj["name"])
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
    if held:
        log_event(deps, "step", f"picked {obj['name']}", object=obj["name"])
    else:
        log_event(deps, "problem", f"closed the gripper on {obj['name']} but didn't grasp it", object=obj["name"])
    return {
        "success": held,
        "label": obj["name"],
        "simulated": True,
        "position_m": obj["center_m"],
        "summary": (f"Picked '{obj['name']}' at ({bx:.3f},{by:.3f}) and lifted it"
                    if held else
                    f"Moved to '{obj['name']}' and closed the gripper, but the object was not grasped"),
    }


async def sim_place_object(deps, cfg, place_description: str,
                           offset_xy: tuple[float, float] = (0.0, 0.0)) -> dict[str, Any]:
    try:
        return await _sim_place_object_impl(deps, cfg, place_description, offset_xy)
    except MoveFailed as e:
        return {"success": False, "simulated": True, "error": str(e)}


async def _sim_place_object_impl(deps, cfg, place_description: str,
                                 offset_xy: tuple[float, float] = (0.0, 0.0)) -> dict[str, Any]:
    if deps.arm_mcp is None:
        return {"success": False, "error": "arm_mcp not wired — cannot move robot"}
    target = await sim_locate(deps, place_description)
    if not target.get("ok"):
        return {"success": False,
                "error": target.get("message", f"Could not locate place target '{place_description}'"),
                "available": target.get("available")}

    arm, gripper = deps.arm_mcp, deps.gripper_mcp
    plx = target["center_m"]["x"] + offset_xy[0]
    ply = target["center_m"]["y"] + offset_xy[1]
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
    log_event(deps, "step", f"dropped it in {target['name']}", target=target["name"])
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


async def sim_inspect_object(deps, description: str) -> dict[str, Any]:
    """Quality inspection by the simulated cell's inspection camera."""
    try:
        res = await deps.sim_scene.call_tool("inspect_object", {"description": description})
    except Exception as e:
        return {"success": False, "error": f"Simulated scene unavailable: {e}"}
    if not res.get("ok"):
        return {"success": False, "error": res.get("message"), "available": res.get("available")}
    if not res.get("inspectable"):
        return {"success": False, "simulated": True, "label": res["name"], "error": res.get("message")}
    verdict = res["result"]
    if verdict == "pass":
        log_event(deps, "result", f"{res['name']}: PASS", part=res["name"], result="pass")
    else:
        log_event(deps, "result", f"{res['name']}: FAIL, {res.get('defect') or 'defect found'}",
                  part=res["name"], result="fail", defect=res.get("defect"))
    return {
        "success": True,
        "simulated": True,
        "label": res["name"],
        "result": verdict,
        "defect": res.get("defect"),
        "summary": f"'{res['name']}' {'passed' if verdict == 'pass' else 'FAILED'} inspection"
                   + (f": {res['defect']}" if res.get("defect") else ""),
    }


async def sim_sort_parts(deps, cfg, source: str, pass_target: str, fail_target: str) -> dict[str, Any]:
    """Inspect every part in `source` and move it to `pass_target` or
    `fail_target`. One part at a time: inspect, pick, place."""
    src = await sim_locate(deps, source)
    if not src.get("ok"):
        return {"success": False, "error": src.get("message", f"Could not locate '{source}'"),
                "available": src.get("available")}
    for t in (pass_target, fail_target):
        loc = await sim_locate(deps, t)
        if not loc.get("ok") or loc.get("kind") != "container":
            return {"success": False, "error": f"'{t}' is not a container in the scene",
                    "available": loc.get("available")}

    listing = await deps.sim_scene.call_tool("list_objects", {})
    parts = [i["name"] for i in listing.get("items", [])
             if i["kind"] == "object" and i.get("in_container") == src["name"]]
    if not parts:
        return {"success": True, "simulated": True, "sorted": [],
                "summary": f"No parts in '{src['name']}' to sort"}

    log_event(deps, "step", f"sorting {len(parts)} parts from {src['name']}: {', '.join(parts)}")
    # Spread parts over the bin floor instead of stacking them on one spot.
    slots = [(-0.03, -0.03), (0.03, -0.03), (-0.03, 0.03), (0.03, 0.03), (0.0, 0.0)]
    used = {pass_target: 0, fail_target: 0}
    sorted_parts, problems = [], []
    for name in parts:
        insp = await sim_inspect_object(deps, name)
        if not insp.get("success"):
            problems.append(f"{name}: inspection failed ({insp.get('error')})")
            continue
        target = pass_target if insp["result"] == "pass" else fail_target
        pick = await sim_pick_object(deps, name)
        if not pick.get("success"):
            problems.append(f"{name}: pick failed ({pick.get('error') or pick.get('summary')})")
            break                       # the arm may be holding nothing or something: stop here
        offset = slots[used[target] % len(slots)]
        used[target] += 1
        place = await sim_place_object(deps, cfg, target, offset)
        if not place.get("success"):
            problems.append(f"{name}: place failed ({place.get('error')})")
            break
        sorted_parts.append({"part": name, "result": insp["result"], "defect": insp.get("defect"),
                             "moved_to": target})
        log_event(deps, "result", f"{name} sorted into {target} ({len(sorted_parts)} of {len(parts)} done)",
                  part=name, moved_to=target)

    n_pass = sum(p["result"] == "pass" for p in sorted_parts)
    n_fail = len(sorted_parts) - n_pass
    summary = (f"Sorted {len(sorted_parts)} of {len(parts)} parts from '{src['name']}': "
               f"{n_pass} passed → {pass_target}, {n_fail} rejected → {fail_target}")
    rejects = [f"{p['part']} ({p['defect']})" for p in sorted_parts if p["result"] != "pass"]
    if rejects:
        summary += ". Rejected: " + "; ".join(rejects)
    if problems:
        summary += ". Problems: " + "; ".join(problems)
        for p in problems:
            log_event(deps, "problem", p)
    return {"success": not problems, "simulated": True, "sorted": sorted_parts,
            "problems": problems, "summary": summary}


def _box_slots(box: dict[str, Any], footprint: tuple[float, float], gap: float = 0.01) -> list[tuple[float, float]]:
    """Offsets from the box centre for a grid of items side by side on its floor."""
    wall = 0.005
    inner_x = box["size_m"][0] - 2 * wall
    inner_y = box["size_m"][1] - 2 * wall
    fx, fy = footprint
    cols = max(1, int((inner_x + gap) // (fx + gap)))
    rows = max(1, int((inner_y + gap) // (fy + gap)))
    xs = [(i - (cols - 1) / 2) * (fx + gap) for i in range(cols)]
    ys = [(j - (rows - 1) / 2) * (fy + gap) for j in range(rows)]
    return [(x, y) for y in ys for x in xs]


async def sim_pack_items(deps, cfg, items: list[str], container: str) -> dict[str, Any]:
    """Pick each item and place it in `container`, side by side in a grid."""
    box = await sim_locate(deps, container)
    if not box.get("ok") or box.get("kind") != "container":
        return {"success": False, "error": f"'{container}' is not a container in the scene",
                "available": box.get("available")}
    listing = await deps.sim_scene.call_tool("list_objects", {})
    objs = {i["name"]: i for i in listing.get("items", []) if i["kind"] == "object"}
    # Products already in this box or in any order's box are taken.
    taken_in = {box["name"]} | {o["ship_in"] for o in listing.get("orders", [])}

    # Resolve each description to a distinct item that isn't packed anywhere yet.
    chosen: list[str] = []
    for d in items:
        if d in objs:
            name = d
        else:
            words = set(d.lower().split())
            cands = [n for n, i in objs.items() if words <= set(n.lower().split())
                     and n not in chosen and i.get("in_container") not in taken_in]
            name = cands[0] if cands else (await sim_locate(deps, d)).get("name")
        if not name or name not in objs:
            return {"success": False, "error": f"No product matches '{d}'", "available": list(objs)}
        if name in chosen:
            return {"success": False, "error": f"'{name}' was listed twice"}
        chosen.append(name)

    first = objs[chosen[0]]
    fp = (first["size_m"][0], first["size_m"][1])
    slots = _box_slots(box, fp)
    used = sum(1 for i in objs.values() if i.get("in_container") == box["name"])
    if used + len(chosen) > len(slots):
        return {"success": False,
                "error": f"'{box['name']}' has room for {len(slots) - used} more, asked for {len(chosen)}"}

    log_event(deps, "step", f"packing {', '.join(chosen)} into {box['name']}")
    packed, problems = [], []
    for name in chosen:
        if objs[name].get("in_container") == box["name"]:
            packed.append(name)                 # already there
            continue
        pick = await sim_pick_object(deps, name)
        if not pick.get("success"):
            problems.append(f"{name}: pick failed ({pick.get('error') or pick.get('summary')})")
            break
        place = await sim_place_object(deps, cfg, box["name"], slots[used])
        used += 1
        if not place.get("success"):
            problems.append(f"{name}: place failed ({place.get('error')})")
            break
        packed.append(name)
        log_event(deps, "result", f"{name} packed into {box['name']}", item=name, box=box["name"])

    for p in problems:
        log_event(deps, "problem", p)
    summary = f"Packed {len(packed)} of {len(chosen)} into '{box['name']}': " + ", ".join(packed)
    if problems:
        summary += ". Problems: " + "; ".join(problems)
    return {"success": not problems, "simulated": True, "packed": packed,
            "problems": problems, "summary": summary}
