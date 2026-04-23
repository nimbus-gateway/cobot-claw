# Perception for cobot-claw

Inspired by [PhyAgentOS](https://github.com/PhyAgentOS/PhyAgentOS).

The goal is to give the robotic agent semantic understanding of its workspace —
not raw pixels, but a structured **scene graph** of objects, their positions,
and their spatial relationships. The agent queries this graph via a tool, the
same way it queries joint states or arm status.

---

## Architecture overview

```
Camera (RGBD or RGB)
        │
        ▼
Vision Model  ←─── runs on host machine (or DGX)
(Gemma 4 / SAM / YOLO)
        │  structured detections
        │  [{label, center, confidence, relations}]
        ▼
SegmentationPipeline   — normalises detection format
        │
        ▼
FusionPipeline         — builds nodes + spatial edges
        │                  • explicit edges from detection relations
        │                  • auto CLOSE_TO edges (objects within 0.5 m)
        ▼
SceneGraph  {nodes[], edges[]}
        │
        ▼
PerceptionService.tick()   — merges with robot pose + map
        │
        ▼
PerceptionClient (in-memory)   — agent queries this
        │
        ▼
Robotic Agent  →  query_scene_graph tool
```

The robot streams **sensor data only** — no inference runs on the robot hardware.
All vision processing runs on the host machine (or offloaded to the DGX).

---

## Scene graph format

### Node

```json
{
  "id": "bottle_01",
  "class": "bottle",
  "confidence": 0.94,
  "center": {"x": 0.32, "y": 0.10, "z": 0.75},
  "size":   {"x": 0.08, "y": 0.08, "z": 0.22},
  "frame": "base_link",
  "track_id": "track_bottle_01",
  "last_seen_at": "2026-04-16T10:00:00Z"
}
```

### Edge

```json
{"source": "bottle_01", "relation": "ON",       "target": "table_01",  "confidence": 0.97}
{"source": "bottle_01", "relation": "CLOSE_TO",  "target": "cup_01",    "confidence": 0.75}
```

Relations are either:
- **explicit** — returned by the vision model (`ON`, `IN`, `NEXT_TO`, …)
- **auto** — `CLOSE_TO` inserted by FusionPipeline when two object centres are ≤ 0.5 m apart

---

## Vision model options

| Model | Where it runs | How detections are produced |
|-------|--------------|----------------------------|
| **Gemma 4** (recommended) | DGX via vLLM | Send base64 frame + prompt asking for JSON detections |
| **SAM3 / Grounding DINO** | Host subprocess | Text-prompted segmentation → bbox + mask → 3D via depth |
| **YOLO** | Host or edge device | Class + bbox → 3D via depth unprojection |
| **Color mask** | Host (CPU) | Fallback — no model needed, works on known object colours |

### Gemma 4 prompt pattern

```python
DETECTION_PROMPT = """
Analyse this image of a robot workspace.
Return a JSON array of detected objects. Each object:
{
  "label": "<object class>",
  "confidence": <0.0-1.0>,
  "center": {"x": <metres from camera>, "y": <metres left/right>, "z": <metres up/down>},
  "relations": [{"relation": "ON|IN|NEXT_TO", "target": "<other label>", "confidence": <0-1>}]
}
Return only the JSON array, no prose.
"""
```

Send the frame to `http://172.31.0.94:8000/v1/chat/completions` with `image_url`
content type. Parse the response and feed into `SegmentationPipeline.process()`.

---

## Depth / 3D position

Two strategies depending on available hardware:

| Sensor | Strategy |
|--------|---------|
| RGBD camera (RealSense, ZED) | Sample depth image at detection centre pixel → unproject to 3D |
| RGB only | Estimate distance from bounding box area using a reference object size |

Unprojection formula (pinhole model):
```python
x_cam = (cx - camera_cx) * depth / camera_fx
y_cam = (cy - camera_cy) * depth / camera_fy
z_cam = depth
```

For the UR3e workspace a RealSense D435 mounted overhead or wrist-mounted gives
reliable depth at 0.2–1.5 m range.

---

## Integration with cobot-claw

### 1. PerceptionService

Create `cobot/perception/service.py` mirroring PhyAgentOS:

```python
class PerceptionService:
    def tick(self, *, rgb: np.ndarray, depth_m: np.ndarray | None,
             robot_pose: dict | None) -> SceneGraph:
        detections = self.vision.detect(rgb, depth_m)   # calls VLM or SAM
        return self.fusion.process(detections)
```

### 2. PerceptionClient (agent-facing)

The agent never calls `PerceptionService` directly. It uses a thin
`PerceptionClient` that holds the latest scene graph in memory and exposes a
`locate_object(label)` method. The existing stub in
`cobot/mcp/perception_contract.py` is the right place to fill in.

### 3. Agent tool

Add `query_scene_graph` to the robotic agent:

```python
@agent.tool
async def query_scene_graph(
    ctx: RunContext[RoboticAgentDeps],
    target_class: str,
) -> dict:
    """Find objects in the workspace by class label.
    Returns a list of scene graph nodes with 3D positions."""
    return await ctx.deps.perception.locate_object(target_class)
```

### 4. Camera feed loop

A background task feeds frames into `PerceptionService.tick()` at ~1 Hz:

```python
async def perception_loop(camera, perception_service, perception_client):
    while True:
        rgb, depth = await camera.capture()
        scene_graph = perception_service.tick(rgb=rgb, depth_m=depth)
        perception_client.update(scene_graph)
        await asyncio.sleep(1.0)
```

---

## Graceful degradation

The system should work without perception:

| State | Behaviour |
|-------|-----------|
| No camera connected | `perception_client.locate_object()` returns `{"found": False}` |
| VLM unavailable | Fall back to colour-mask detector if configured |
| Object not found | Agent reports object not visible, asks user to confirm placement |

This matches the existing `PerceptionClient` stub behaviour — the current cobot
already handles `{"found": False}` gracefully in scenario execution.

---

## Physical setup recommendation for UR3e

- **Camera**: Intel RealSense D435i (RGBD, USB3, compact)
- **Mount**: overhead on a post above the workspace, looking down at ~45°
  gives full table coverage without wrist-mount cable management issues
- **Frame**: calibrate camera-to-base_link transform once using a known
  calibration board; store in `cobot.yaml` under `perception.camera_tf`
- **Connection**: USB3 to the host PC running cobot-claw

With this setup the scene graph `center` coordinates are directly in the robot's
base frame, so waypoints from perception output can be passed straight to
`move_joints` or `move_tcp_pose` without additional transforms.

---

## Files to create (implementation order)

1. `cobot/perception/vision.py` — VLM / SAM / colour-mask detector interface
2. `cobot/perception/fusion.py` — build scene graph from detections (port from PhyAgentOS)
3. `cobot/perception/service.py` — orchestrate vision + fusion + pose
4. `cobot/perception/client.py` — in-memory state, `locate_object()` method
5. `cobot/agents/robotic_agent.py` — add `query_scene_graph` tool
6. `cobot/cli/app.py` — start perception loop alongside MCP clients
