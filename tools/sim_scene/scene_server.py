#!/usr/bin/env python3
"""
Simulated pick-and-place scene for URSim.

Gives cobot-claw objects to pick when there's no camera:

- Draws a table, the objects and containers from scene.yaml in RViz
  (MarkerArray on /sim_scene/markers, frame "base").
- Tracks grasps: when the gripper closes with an object between the finger
  pads, the object attaches to the gripper and follows the arm. When the
  gripper opens, the object drops straight down onto the table or into the
  container below it.
- Tells the Robotiq emulator about the object (SET SOB <pos>), so closing
  stops at the object's width and reports "object detected", like a real
  grasp.
- Serves MCP tools (list_objects, locate_object, inspect_object,
  reset_scene) that cobot-claw's vision agent uses in place of the camera
  when agents.vision.detector is "sim".
- Quality inspection: objects can carry a hidden quality result (pass, or
  fail with a defect). inspect_object reveals it, as an inspection camera
  would, and RViz then labels the part PASS or FAIL.
- Fixtures: equipment drawn for context (conveyor, camera, machine). They
  can't be picked and objects don't rest on them.
- Orders (packing scenes): which products go into which box. list_objects
  reports each order with what is already packed and what is missing.

No physics: objects don't tip, slide or collide.

Run with the system ROS Python (needs rclpy, tf2_ros, mcp, pyyaml):

    source /opt/ros/jazzy/setup.bash
    export ROS_DOMAIN_ID=42
    python3 tools/sim_scene/scene_server.py

Environment:
    SCENE_FILE   scene layout: a path, or a name in scenes/ (default: scene.yaml next to this file)
    SCENE_PORT   MCP port (default 8010)
    EMU_HOST     gripper emulator host (default 192.168.56.101)
    EMU_PORT     gripper emulator port (default 63352)

Coordinates are metres in the UR controller's base frame ("base" in ROS).
"""
from __future__ import annotations

import copy
import math
import os
import re
import socket
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import rclpy
import yaml
from builtin_interfaces.msg import Duration
from geometry_msgs.msg import Point
from mcp.server.fastmcp import FastMCP
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from rclpy.time import Time
from tf2_ros import Buffer, TransformListener
from visualization_msgs.msg import Marker, MarkerArray

HERE = Path(__file__).resolve().parent


def _scene_path(value: str) -> Path:
    """A path, or a scene name such as "quality_sorting" (scenes/<name>.yaml)."""
    p = Path(value)
    if p.exists():
        return p
    named = HERE / "scenes" / f"{value.removesuffix('.yaml')}.yaml"
    return named if named.exists() else p


SCENE_FILE = _scene_path(os.getenv("SCENE_FILE", str(HERE / "scene.yaml")))
SCENE_PORT = int(os.getenv("SCENE_PORT", "8010"))
EMU_HOST = os.getenv("EMU_HOST", "192.168.56.101")
EMU_PORT = int(os.getenv("EMU_PORT", "63352"))
FRAME = "base"
RATE_HZ = 20.0

COLORS = {
    "red": (0.85, 0.15, 0.15), "green": (0.15, 0.7, 0.25), "blue": (0.15, 0.35, 0.85),
    "yellow": (0.95, 0.8, 0.1), "orange": (0.95, 0.5, 0.1), "purple": (0.55, 0.25, 0.75),
    "white": (0.95, 0.95, 0.95), "black": (0.1, 0.1, 0.1), "grey": (0.55, 0.55, 0.58),
    "gray": (0.55, 0.55, 0.58), "brown": (0.5, 0.33, 0.2),
}
SHAPE_WORDS = {"box": ["block", "cube", "brick"], "cylinder": ["cylinder", "can", "tube", "bottle"]}
WORD_RE = r"[a-z0-9]+"           # digits count, so "part 3" picks part 3
STOPWORDS = {"the", "a", "an", "on", "in", "into", "of", "to", "and", "please", "object", "thing"}


# ── scene model ───────────────────────────────────────────────────────────────
@dataclass
class Item:
    name: str
    kind: str                     # "object" | "container"
    shape: str                    # "box" | "cylinder"
    size: list[float]             # box [x,y,z]; cylinder [d,h]; container [x,y,h]
    color: str
    x: float
    y: float
    z: float                      # centre height
    yaw: float = 0.0
    wall: float = 0.006
    aliases: list[str] = field(default_factory=list)
    held: bool = False
    quality: str | None = None    # hidden result: "pass" | "fail" (None = not an inspected part)
    defect: str = ""              # what an inspection finds on a failed part
    inspected: bool = False

    @property
    def height(self) -> float:
        return self.size[1] if self.shape == "cylinder" else self.size[2]

    @property
    def footprint(self) -> tuple[float, float]:
        return (self.size[0], self.size[0]) if self.shape == "cylinder" else (self.size[0], self.size[1])

    @property
    def grip_width(self) -> float:
        return min(self.footprint)

    def words(self) -> set[str]:
        w = set(re.findall(WORD_RE, self.name.lower()))
        w.add(self.color.lower())
        for a in self.aliases:
            w.update(re.findall(WORD_RE, a.lower()))
        if self.kind == "object":
            w.update(SHAPE_WORDS.get(self.shape, []))
        return w

    def describe(self, grasp_depth: float) -> dict[str, Any]:
        top = self.z + self.height / 2
        d = {
            "name": self.name,
            "kind": self.kind,
            "shape": self.shape,
            "color": self.color,
            "size_m": [round(v, 4) for v in self.size],
            "center_m": {"x": round(self.x, 4), "y": round(self.y, 4), "z": round(self.z, 4)},
            "top_z_m": round(top, 4),
            "held": self.held,
            "grasp_depth_m": grasp_depth,
        }
        if self.kind == "container":
            d["floor_z_m"] = round(self.z - self.height / 2 + self.wall, 4)
        if self.quality is not None:
            d["inspected"] = self.inspected
            if self.inspected:          # hidden until inspected, as on a real line
                d["quality"] = self.quality
                if self.defect:
                    d["defect"] = self.defect
        return d


class Scene:
    def __init__(self, path: Path) -> None:
        self.path = path
        self.lock = threading.RLock()
        self.load()

    def load(self) -> None:
        cfg = yaml.safe_load(self.path.read_text())
        g = cfg.get("gripper", {})
        t = cfg.get("table", {})
        with self.lock:
            self.grasp_depth = float(g.get("grasp_depth_m", 0.14))
            self.stroke = float(g.get("stroke_m", 0.085))
            self.capture_xy = float(g.get("capture_xy_m", 0.03))
            self.table_size = t.get("size_m", [1.0, 1.0])
            self.table_center = t.get("center_m", [0.0, 0.0])
            self.title = cfg.get("title", "")
            self.items: list[Item] = []
            for c in cfg.get("containers", []):
                size = [float(v) for v in c["size_m"]]
                self.items.append(Item(
                    name=c["name"], kind="container", shape="box", size=size,
                    color=c.get("color", "grey"), x=float(c["position_m"][0]),
                    y=float(c["position_m"][1]), z=size[2] / 2,
                    wall=float(c.get("wall_m", 0.006)), aliases=c.get("aliases", []),
                ))
            for o in cfg.get("objects", []):
                shape = o.get("shape", "box")
                size = [float(v) for v in o["size_m"]]
                h = size[1] if shape == "cylinder" else size[2]
                x, y = float(o["position_m"][0]), float(o["position_m"][1])
                q = o.get("quality")
                self.items.append(Item(
                    name=o["name"], kind="object", shape=shape, size=size,
                    color=o.get("color", "grey"), x=x, y=y,
                    z=self.support_z(x, y) + h / 2,   # objects may start inside a tray
                    yaw=float(o.get("yaw", 0.0)), aliases=o.get("aliases", []),
                    quality=str(q).lower() if q is not None else None,
                    defect=o.get("defect", ""),
                ))
            # Fixtures: drawn only. Each is a list of boxes/cylinders.
            self.fixtures: list[dict] = cfg.get("fixtures", [])
            self.orders: list[dict] = cfg.get("orders", [])
            self.held: Item | None = None
            self.held_rel: np.ndarray | None = None

    # gripper position (0 open … 255 closed) at which the jaws touch `item`
    def stop_pos(self, item: Item) -> int | None:
        w = item.grip_width
        if w >= self.stroke:
            return None
        return max(0, min(250, round(255 * (1 - w / self.stroke))))

    def match(self, description: str) -> tuple[Item | None, list[str]]:
        tokens = {t for t in re.findall(WORD_RE, description.lower()) if t not in STOPWORDS}
        best, best_score = None, 0.0
        with self.lock:
            for it in self.items:
                score = float(len(tokens & it.words()))
                if it.name.lower() in description.lower():
                    score += 2
                if score > best_score:
                    best, best_score = it, score
            names = [i.name for i in self.items]
        return best, names

    def container_at(self, x: float, y: float) -> "Item | None":
        for c in self.items:
            if c.kind != "container":
                continue
            hx, hy = c.size[0] / 2 - c.wall, c.size[1] / 2 - c.wall
            if abs(x - c.x) <= hx and abs(y - c.y) <= hy:
                return c
        return None

    def support_z(self, x: float, y: float) -> float:
        """Height an object dropped at (x, y) comes to rest on."""
        for c in self.items:
            if c.kind != "container":
                continue
            hx, hy = c.size[0] / 2 - c.wall, c.size[1] / 2 - c.wall
            if abs(x - c.x) <= hx and abs(y - c.y) <= hy:
                return c.z - c.height / 2 + c.wall
        return 0.0


# ── helpers ───────────────────────────────────────────────────────────────────
def pose_matrix(x: float, y: float, z: float, yaw: float) -> np.ndarray:
    c, s = math.cos(yaw), math.sin(yaw)
    return np.array([[c, -s, 0, x], [s, c, 0, y], [0, 0, 1, z], [0, 0, 0, 1]], dtype=float)


def tf_to_matrix(t) -> np.ndarray:
    q = t.transform.rotation
    x, y, z, w = q.x, q.y, q.z, q.w
    r = np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ])
    m = np.eye(4)
    m[:3, :3] = r
    tr = t.transform.translation
    m[:3, 3] = [tr.x, tr.y, tr.z]
    return m


class EmulatorLink:
    """Minimal client for the Robotiq emulator's socket protocol."""

    def __init__(self, host: str, port: int) -> None:
        self.addr = (host, port)
        self.sock: socket.socket | None = None

    def _tx(self, payload: bytes, nbytes: int = 64) -> str | None:
        try:
            if self.sock is None:
                self.sock = socket.create_connection(self.addr, timeout=0.5)
            self.sock.sendall(payload)
            return self.sock.recv(nbytes).decode(errors="replace")
        except OSError:
            if self.sock is not None:
                self.sock.close()
            self.sock = None
            return None

    def pos(self) -> int | None:
        r = self._tx(b"GET POS\n")
        parts = r.split() if r else []
        return int(parts[1]) if len(parts) >= 2 and parts[0] == "POS" else None

    def obj(self) -> int | None:
        r = self._tx(b"GET OBJ", 1)
        return int(r) if r and r.isdigit() else None

    def set_object_stop(self, pos: int) -> bool:
        return self._tx(f"SET SOB {pos}\n".encode(), 3) == "ack"


# ── ROS node ──────────────────────────────────────────────────────────────────
class SceneNode(Node):
    def __init__(self, scene: Scene) -> None:
        super().__init__("sim_scene")
        self.scene = scene
        self.emu = EmulatorLink(EMU_HOST, EMU_PORT)
        self.tf_buffer = Buffer()
        # Own thread for TF: sharing the node's callback queue with the update
        # timer let tool0 lag up to ~2 s, so carried objects dropped off-target.
        self.tf_listener = TransformListener(self.tf_buffer, self, spin_thread=True)
        self.pub = self.create_publisher(MarkerArray, "/sim_scene/markers", 10)
        self.sent_stop: int | None = None
        self.release_below: int | None = None
        self.warned = False
        self.create_timer(1.0 / RATE_HZ, self.tick)

    def grasp_point(self) -> tuple[np.ndarray, np.ndarray] | None:
        try:
            t = self.tf_buffer.lookup_transform(FRAME, "tool0", Time())
        except Exception:
            return None
        age = (self.get_clock().now() - Time.from_msg(t.header.stamp)).nanoseconds / 1e9
        if age > 0.3:
            return None   # stale: skip this tick rather than act on an old pose
        T = tf_to_matrix(t)
        g = T @ np.array([0, 0, self.scene.grasp_depth, 1.0])
        return T, g[:3]

    def candidate(self, g: np.ndarray) -> Item | None:
        best, best_d = None, None
        for it in self.scene.items:
            if it.kind != "object" or it.held:
                continue
            dxy = math.hypot(g[0] - it.x, g[1] - it.y)
            if dxy > self.scene.capture_xy or abs(g[2] - it.z) > it.height / 2 + 0.01:
                continue
            if best_d is None or dxy < best_d:
                best, best_d = it, dxy
        return best

    def tick(self) -> None:
        s = self.scene
        gp = self.grasp_point()
        with s.lock:
            if gp is not None:
                T, g = gp
                if s.held is None:
                    cand = self.candidate(g)
                    stop = s.stop_pos(cand) if cand else None
                    want = stop if stop is not None else -1
                    if want != self.sent_stop and self.emu.set_object_stop(want):
                        self.sent_stop = want
                    if cand is not None and stop is not None and self.emu.obj() == 2:
                        s.held, cand.held = cand, True
                        s.held_rel = np.linalg.inv(T) @ pose_matrix(cand.x, cand.y, cand.z, cand.yaw)
                        self.release_below = stop - 15
                        self.get_logger().info(f"grasped '{cand.name}'")
                else:
                    it = s.held
                    P = T @ s.held_rel
                    it.x, it.y, it.z = P[0, 3], P[1, 3], P[2, 3]
                    it.yaw = math.atan2(P[1, 0], P[0, 0])
                    pos = self.emu.pos()
                    if pos is not None and self.release_below is not None and pos < self.release_below:
                        it.z = s.support_z(it.x, it.y) + it.height / 2
                        it.held = False
                        s.held = None
                        s.held_rel = None
                        self.get_logger().info(
                            f"released '{it.name}' at ({it.x:.3f}, {it.y:.3f}, {it.z:.3f})")
            elif not self.warned:
                self.get_logger().warn("waiting for TF base -> tool0 (is the UR driver running?)")
                self.warned = True
            self.pub.publish(self.markers())

    def markers(self) -> MarkerArray:
        s = self.scene
        out = MarkerArray()
        stamp = self.get_clock().now().to_msg()
        life = Duration(sec=1)

        def mk(mid: int, mtype: int, ns: str) -> Marker:
            m = Marker()
            m.header.frame_id = FRAME
            m.header.stamp = stamp
            m.ns, m.id, m.type, m.action = ns, mid, mtype, Marker.ADD
            m.lifetime = life
            m.pose.orientation.w = 1.0
            return m

        def color(m: Marker, name: str, a: float = 1.0) -> None:
            r, g, b = COLORS.get(name.lower(), COLORS["grey"])
            m.color.r, m.color.g, m.color.b, m.color.a = r, g, b, a

        def set_yaw(m: Marker, yaw: float) -> None:
            m.pose.orientation.z = math.sin(yaw / 2)
            m.pose.orientation.w = math.cos(yaw / 2)

        t = mk(0, Marker.CUBE, "table")
        t.pose.position.x, t.pose.position.y = s.table_center
        t.pose.position.z = -0.011
        t.scale.x, t.scale.y, t.scale.z = s.table_size[0], s.table_size[1], 0.02
        t.color.r, t.color.g, t.color.b, t.color.a = 0.72, 0.6, 0.45, 1.0
        out.markers.append(t)

        mid = 1
        for it in s.items:
            if it.kind == "object":
                m = mk(mid, Marker.CUBE if it.shape == "box" else Marker.CYLINDER, "objects")
                m.pose.position.x, m.pose.position.y, m.pose.position.z = it.x, it.y, it.z
                set_yaw(m, it.yaw)
                if it.shape == "box":
                    m.scale.x, m.scale.y, m.scale.z = it.size
                else:
                    m.scale.x = m.scale.y = it.size[0]
                    m.scale.z = it.size[1]
                color(m, it.color)
                out.markers.append(m)
                mid += 1
                if it.quality == "fail":
                    # The flaw itself: a dark scratch across the top face.
                    d = mk(mid, Marker.CUBE, "defects")
                    d.pose.position.x, d.pose.position.y = it.x, it.y
                    d.pose.position.z = it.z + it.height / 2 + 0.0008
                    set_yaw(d, it.yaw + 0.6)
                    d.scale.x, d.scale.y, d.scale.z = it.grip_width * 0.8, 0.003, 0.0015
                    color(d, "black")
                    out.markers.append(d)
                    mid += 1
            else:
                sx, sy, h = it.size
                w = it.wall
                z0 = it.z - h / 2
                parts = [  # (dx, dy, dz, size)
                    (0, 0, z0 + w / 2, (sx, sy, w)),
                    (0, sy / 2 - w / 2, it.z, (sx, w, h)),
                    (0, -sy / 2 + w / 2, it.z, (sx, w, h)),
                    (sx / 2 - w / 2, 0, it.z, (w, sy, h)),
                    (-sx / 2 + w / 2, 0, it.z, (w, sy, h)),
                ]
                for dx, dy, z, size in parts:
                    m = mk(mid, Marker.CUBE, "containers")
                    m.pose.position.x, m.pose.position.y, m.pose.position.z = it.x + dx, it.y + dy, z
                    m.scale.x, m.scale.y, m.scale.z = size
                    color(m, it.color, 0.85)
                    out.markers.append(m)
                    mid += 1
            label = mk(mid, Marker.TEXT_VIEW_FACING, "labels")
            label.pose.position.x, label.pose.position.y = it.x, it.y
            label.pose.position.z = it.z + it.height / 2 + 0.03
            label.scale.z = 0.022
            label.text = it.name + (" (held)" if it.held else "")
            label.color.r = label.color.g = label.color.b = label.color.a = 1.0
            if it.inspected and it.quality:
                label.text += "  PASS" if it.quality == "pass" else "  FAIL"
                color(label, "green" if it.quality == "pass" else "red")
            out.markers.append(label)
            mid += 1
        for fx in s.fixtures:
            for part in fx.get("parts", []):
                shape = part.get("shape", "box")
                m = mk(mid, Marker.CYLINDER if shape == "cylinder" else Marker.CUBE, "fixtures")
                m.pose.position.x, m.pose.position.y, m.pose.position.z = part["center_m"]
                set_yaw(m, float(part.get("yaw", 0.0)))
                sz = part["size_m"]
                if shape == "cylinder":
                    m.scale.x = m.scale.y = sz[0]
                    m.scale.z = sz[1]
                else:
                    m.scale.x, m.scale.y, m.scale.z = sz
                color(m, part.get("color", fx.get("color", "grey")), float(part.get("alpha", 1.0)))
                out.markers.append(m)
                mid += 1
            if fx.get("label_m"):
                label = mk(mid, Marker.TEXT_VIEW_FACING, "labels")
                label.pose.position.x, label.pose.position.y, label.pose.position.z = fx["label_m"]
                label.scale.z = 0.022
                label.text = fx.get("name", "")
                label.color.r = label.color.g = label.color.b = 0.8
                label.color.a = 1.0
                out.markers.append(label)
                mid += 1
        return out


# ── MCP server ────────────────────────────────────────────────────────────────
SCENE = Scene(SCENE_FILE)
mcp = FastMCP("Simulated scene MCP", host="0.0.0.0", port=SCENE_PORT)


@mcp.tool()
def list_objects() -> dict[str, Any]:
    """List every object and container in the simulated scene with its position
    (metres, UR base frame), size, colour and whether the gripper is holding it."""
    with SCENE.lock:
        return {
            "ok": True,
            "frame": FRAME,
            "scene": SCENE.title,
            "items": [_describe(it) for it in SCENE.items],
            **({"orders": _order_status()} if SCENE.orders else {}),
        }


def _order_status() -> list[dict[str, Any]]:
    """Each order with the products already in its box and those still missing.
    A product counts if its name contains the order's product name
    ("red pack 2" counts as a "red pack")."""
    out = []
    for o in SCENE.orders:
        box = o["ship_in"]
        inside = [it.name for it in SCENE.items if it.kind == "object" and not it.held
                  and (c := SCENE.container_at(it.x, it.y)) is not None and c.name == box]
        missing = {}
        for product, qty in o.get("items", {}).items():
            have = sum(product.lower() in n.lower() for n in inside)
            if have < int(qty):
                missing[product] = int(qty) - have
        out.append({"order": o["name"], "ship_in": box, "wanted": o.get("items", {}),
                    "packed": inside, "missing": missing, "complete": not missing})
    return out


def _describe(it: Item) -> dict[str, Any]:
    d = it.describe(SCENE.grasp_depth)
    if it.kind == "object" and not it.held:
        c = SCENE.container_at(it.x, it.y)
        d["in_container"] = c.name if c else None
    return d


@mcp.tool()
def locate_object(description: str) -> dict[str, Any]:
    """Find the object or container that best matches a description such as
    "red block", "the blue can" or "grey bin". Returns its centre and top
    height in the UR base frame."""
    item, names = SCENE.match(description)
    if item is None:
        return {"ok": False, "message": f"No object matches '{description}'", "available": names}
    with SCENE.lock:
        return {"ok": True, **_describe(item)}


@mcp.tool()
def inspect_object(description: str) -> dict[str, Any]:
    """Inspect a part with the cell's inspection camera and report whether it
    passes quality control, and the defect if it fails. Works on a part
    lying in the scene or held by the gripper."""
    item, names = SCENE.match(description)
    if item is None:
        return {"ok": False, "message": f"No object matches '{description}'", "available": names}
    with SCENE.lock:
        if item.kind != "object":
            return {"ok": False, "message": f"'{item.name}' is a container, not a part"}
        if item.quality is None:
            return {"ok": True, "name": item.name, "inspectable": False,
                    "message": f"'{item.name}' has no quality-control result in this scene"}
        item.inspected = True
        return {"ok": True, "name": item.name, "inspectable": True,
                "result": item.quality, "defect": item.defect or None}


@mcp.tool()
def reset_scene() -> dict[str, Any]:
    """Reload the scene file and put every object back in its starting position
    (inspection results are cleared too).
    Open the gripper first if it is holding something."""
    SCENE.load()
    return {"ok": True, "items": [i.name for i in SCENE.items]}


def main() -> None:
    rclpy.init()
    node = SceneNode(SCENE)
    executor = MultiThreadedExecutor()
    executor.add_node(node)
    threading.Thread(target=executor.spin, daemon=True).start()
    print(f"[sim-scene] {SCENE.title or SCENE_FILE.stem}: {len(SCENE.items)} items from {SCENE_FILE}; "
          f"MCP on :{SCENE_PORT}", flush=True)
    try:
        mcp.run(transport="streamable-http")
    finally:
        executor.shutdown()
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
