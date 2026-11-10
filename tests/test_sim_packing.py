"""Packing in the simulated cell (cobot/agents/vision_agent/sim.py) with a
fake scene server; arm moves are stubbed. No URSim or ROS needed."""
import asyncio
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml

from cobot.agents.vision_agent import sim

SCENE = Path(__file__).resolve().parents[1] / "tools/sim_scene/scenes/packing.yaml"


class FakeScene:
    def __init__(self):
        cfg = yaml.safe_load(SCENE.read_text())
        self.boxes = {c["name"]: c for c in cfg["containers"]}
        self.packs = {o["name"]: dict(o, location="accumulation tray") for o in cfg["objects"]}
        self.orders = cfg["orders"]

    def _item(self, n):
        if n in self.boxes:
            c = self.boxes[n]
            return {"ok": True, "name": n, "kind": "container", "size_m": c["size_m"],
                    "center_m": {"x": c["position_m"][0], "y": c["position_m"][1], "z": 0.035}}
        p = self.packs[n]
        return {"ok": True, "name": n, "kind": "object", "size_m": p["size_m"], "in_container": p["location"],
                "center_m": {"x": p["position_m"][0], "y": p["position_m"][1], "z": 0.0235}}

    async def call_tool(self, tool, args):
        if tool == "list_objects":
            return {"ok": True, "items": [self._item(n) for n in list(self.boxes) + list(self.packs)],
                    "orders": [{"order": o["name"], "ship_in": o["ship_in"]} for o in self.orders]}
        if tool == "locate_object":
            d = args["description"].lower()
            for n in list(self.boxes) + list(self.packs):
                if n in d or d in n:
                    return self._item(n)
            return {"ok": False, "message": "no match", "available": []}
        raise AssertionError(tool)


@pytest.fixture
def packing(monkeypatch):
    scene = FakeScene()
    placed = []

    async def pick(deps, name):
        deps.held = name
        return {"success": True}

    async def place(deps, cfg, box, offset=(0.0, 0.0)):
        scene.packs[deps.held]["location"] = box
        placed.append((deps.held, box, offset))
        return {"success": True}

    monkeypatch.setattr(sim, "sim_pick_object", pick)
    monkeypatch.setattr(sim, "sim_place_object", place)
    deps = SimpleNamespace(sim_scene=scene, held=None)
    return deps, scene, placed


def _pack(deps, items, box):
    return asyncio.run(sim.sim_pack_items(deps, None, items, box))


def test_packs_named_items_into_distinct_slots(packing):
    deps, scene, placed = packing
    res = _pack(deps, ["red pack 1", "blue pack 2", "yellow pack 1"], "shipping box 1")
    assert res["success"], res
    assert [p[0] for p in placed] == ["red pack 1", "blue pack 2", "yellow pack 1"]
    assert len({p[2] for p in placed}) == 3                     # side by side, not stacked
    assert all(scene.packs[n]["location"] == "shipping box 1" for n in res["packed"])


def test_generic_product_names_pick_different_unpacked_items(packing):
    deps, scene, placed = packing
    _pack(deps, ["red pack 1"], "shipping box 1")
    res = _pack(deps, ["red pack", "blue pack"], "shipping box 2")
    assert res["success"], res
    assert res["packed"] == ["red pack 2", "blue pack 1"]        # red pack 1 is already shipped
    assert placed[1][2] != placed[2][2]


def test_next_order_continues_after_existing_slots(packing):
    deps, scene, placed = packing
    _pack(deps, ["red pack 1", "blue pack 1"], "shipping box 1")
    _pack(deps, ["yellow pack 1"], "shipping box 1")
    assert len({p[2] for p in placed}) == 3


def test_refuses_more_than_fits(packing):
    deps, _, placed = packing
    res = _pack(deps, ["red pack 1", "red pack 2", "blue pack 1", "blue pack 2", "yellow pack 1"],
                "shipping box 1")
    assert not res["success"] and "room for 4" in res["error"] and not placed


def test_unknown_box(packing):
    deps, _, _ = packing
    assert not _pack(deps, ["red pack 1"], "pallet")["success"]
