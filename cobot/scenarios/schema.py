from __future__ import annotations

from typing import Annotated, Any, Literal

from pydantic import BaseModel, Field


class PerceptionQuery(BaseModel):
    locate: str | None = None           # object label to locate
    estimate_pose: str | None = None    # object label for full pose


class FallbackPosition(BaseModel):
    x: float
    y: float
    z: float
    qx: float = 0.0
    qy: float = 0.0
    qz: float = 0.0
    qw: float = 1.0


# --- Step types ---

class MoveJointsStep(BaseModel):
    type: Literal["move_joints"]
    joints: Annotated[list[float], Field(min_length=6, max_length=6)]
    speed: float = 1.05        # rad/s
    accel: float = 1.4         # rad/s²
    blend_radius: float = 0.0  # metres — >0 blends into next motion without stopping


class MoveNamedPoseStep(BaseModel):
    type: Literal["move_named_pose"]
    pose: str
    speed: float = 1.05
    accel: float = 1.4
    via_home: bool = False


class MoveTCPStep(BaseModel):
    type: Literal["move_tcp_pose"]
    x: float
    y: float
    z: float
    qx: float = 0.0
    qy: float = 0.0
    qz: float = 0.0
    qw: float = 1.0
    speed: float = 0.05
    accel: float = 0.3
    perception: PerceptionQuery | None = None       # resolved at runtime if available
    fallback_position: FallbackPosition | None = None


class ToolActionStep(BaseModel):
    type: Literal["tool_action"]
    tool: str                    # "gripper" — maps to registered tool MCP
    action: str                  # MCP tool name to call
    params: dict[str, Any] = {}
    arm_recovery_sec: float = 0.0  # >0 for gripper actions that drop UR External Control


class WaitStep(BaseModel):
    type: Literal["wait"]
    duration_sec: float


class PathWaypoint(BaseModel):
    joints: Annotated[list[float], Field(min_length=6, max_length=6)]
    speed: float = 1.047
    accel: float = 1.396
    blend_radius: float = 0.0


class MovePathStep(BaseModel):
    """Send multiple waypoints as a single URScript for smooth blended motion."""
    type: Literal["move_path"]
    waypoints: list[PathWaypoint]


class CheckMotionStep(BaseModel):
    type: Literal["check_motion_complete"]
    timeout_sec: float = 10.0
    tolerance_rad: float = 0.03


class RepeatStep(BaseModel):
    type: Literal["repeat"]
    count: int
    steps: list[ScenarioStep]


class VisualServoStep(BaseModel):
    """Re-detect object at approach height, correct XY, then descend to grasp.

    Falls back to fallback_joints if vision is unavailable or detection fails.
    """
    type: Literal["visual_servo"]
    pick_description: str      # natural-language label passed to Gemini ER
    approach_z: float          # Z (base frame) to hold while correcting XY
    grasp_z: float             # Z (base frame) to descend to for grasp
    fallback_joints: Annotated[list[float], Field(min_length=6, max_length=6)]
    speed: float = 0.05
    accel: float = 0.2
    threshold_mm: float = 2.0  # skip XY correction if offset < this


ScenarioStep = Annotated[
    MoveJointsStep
    | MoveNamedPoseStep
    | MoveTCPStep
    | MovePathStep
    | ToolActionStep
    | WaitStep
    | CheckMotionStep
    | RepeatStep
    | VisualServoStep,
    Field(discriminator="type"),
]

# Pydantic needs the forward ref resolved after all types are defined
RepeatStep.model_rebuild()


class Scenario(BaseModel):
    name: str
    description: str = ""
    robot: str = "default"          # matches fleet[].name in cobot.yaml
    steps: list[ScenarioStep]
