"""Quality sorting in the simulated cell (cobot/agents/vision_agent/sim.py),
with fake scene, arm and gripper servers. No URSim or ROS needed."""
import asyncio
from pathlib import Path
from types import SimpleNamespace

import yaml

from cobot.agents.vision_agent import sim

SCENE = Path(__file__).resolve().parents[1] / "tools/sim_scene/scenes/quality_sorting.yaml"


class FakeScene:
    """Answers the scene server's tools from the quality_sorting scene file.
    A pick followed by a place moves the part into the target container."""

    def __init__(self):
        cfg = yaml.safe_load(SCENE.read_text())
        self.containers = {c["name"]: c for c in cfg["containers"]}
        self.parts = {o["name"]: dict(o, location="incoming tray", inspected=False) for o in cfg["objects"]}
        self.held = None

    def _find(self, d):
        d = d.lower()
        for n in list(self.parts) + list(self.containers):
            if n in d or d in n:
                return n
        return None

    def _describe(self, n):
        if n in self.containers:
            c = self.containers[n]
            return {"ok": True, "name": n, "kind": "container", "center_m": {"x": c["position_m"][0], "y": c["position_m"][1], "z": c["size_m"][2] / 2},
                    "top_z_m": c["size_m"][2], "size_m": c["size_m"], "held": False, "grasp_depth_m": 0.14}
        p = self.parts[n]
        return {"ok": True, "name": n, "kind": "object", "center_m": {"x": p["position_m"][0], "y": p["position_m"][1], "z": 0.021},
                "top_z_m": 0.036, "size_m": p["size_m"], "held": self.held == n, "grasp_depth_m": 0.14,
                "in_container": None if self.held == n else p["location"]}

    async def call_tool(self, tool, args):
        if tool == "list_objects":
            return {"ok": True, "items": [self._describe(n) for n in list(self.containers) + list(self.parts)]}
        n = self._find(args["description"])
        if n is None:
            return {"ok": False, "message": "no match", "available": []}
        if tool == "locate_object":
            return self._describe(n)
        if tool == "inspect_object":
            p = self.parts[n]
            p["inspected"] = True
            return {"ok": True, "name": n, "inspectable": True, "result": p["quality"], "defect": p.get("defect")}
        raise AssertionError(tool)


class FakeArm:
    def __init__(self, scene):
        self.scene, self.moves = scene, 0

    async def call_tool(self, tool, args):
        self.moves += 1
        if tool == "get_joint_states":
            return {"joints": [2.0, -1.3, -1.9, -1.4, 1.57, 3.6]}
        return {"ok": True}


class FakeGripper:
    def __init__(self, scene):
        self.scene = scene
        self.target = None

    async def call_tool(self, tool, args):
        return {"ok": True}


def _deps():
    scene = FakeScene()
    deps = SimpleNamespace(sim_scene=scene, arm_mcp=FakeArm(scene), gripper_mcp=FakeGripper(scene), arm_mcp_url=None)
    # Make "held" true after a close, and move the part on place.
    real_pick, real_place = sim._sim_pick_object_impl, sim._sim_place_object_impl

    async def pick(d, description):
        scene.held = scene._find(description)
        return await real_pick(d, description)

    async def place(d, cfg, target, offset=(0.0, 0.0)):
        res = await real_place(d, cfg, target, offset)
        scene.parts[scene.held]["location"] = scene._find(target)
        scene.held = None
        return res

    sim._sim_pick_object_impl, sim._sim_place_object_impl = pick, place
    return deps, scene, (real_pick, real_place)


def test_sort_parts_moves_each_part_to_the_right_bin():
    deps, scene, (real_pick, real_place) = _deps()
    cfg = SimpleNamespace(agents=SimpleNamespace(vision=SimpleNamespace(min_drop_z_m=0.30)))
    try:
        res = asyncio.run(sim.sim_sort_parts(deps, cfg, "incoming tray", "pass bin", "reject bin"))
    finally:
        sim._sim_pick_object_impl, sim._sim_place_object_impl = real_pick, real_place
    assert res["success"], res
    assert len(res["sorted"]) == 6
    for name, p in scene.parts.items():
        assert p["inspected"]
        assert p["location"] == ("pass bin" if p["quality"] == "pass" else "reject bin"), name
    assert "2 rejected" in res["summary"] and "part 2 (crack" in res["summary"]


def test_quality_hidden_until_inspected_in_scene_file():
    cfg = yaml.safe_load(SCENE.read_text())
    qualities = [o["quality"] for o in cfg["objects"]]
    assert qualities.count("fail") == 2 and qualities.count("pass") == 4
    assert all(o.get("defect") for o in cfg["objects"] if o["quality"] == "fail")
