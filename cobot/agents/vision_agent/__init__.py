"""
Vision Agent package.

Public API re-exported from submodules for backward compatibility:
    from cobot.agents.vision_agent import build_vision_agent, VisionAgentDeps, ...
    from cobot.agents.vision_agent import generate_scenario, calibrate_handeye
"""

from cobot.agents.vision_agent.kinematics import (
    ur3e_fk,
    ur3e_ik,
    get_joints_for_pose,
    _PICK_SEED,
    _HOME_SEED,
    APPROACH_Z_OFFSET,
    PICK_Z_OFFSET,
)
from cobot.agents.vision_agent.camera import (
    load_calibration,
    capture_realsense,
    encode_jpeg,
    depth_at_pixel,
    pixel_to_camera_xyz,
    camera_to_base,
    CALIB_FILE,
)
from cobot.agents.vision_agent.gemini import (
    gemini_locate,
    gemini_locate_gripper,
)
from cobot.agents.vision_agent.calibrate import calibrate_handeye
from cobot.agents.vision_agent.scenario import generate_scenario
from cobot.agents.vision_agent.agent import (
    VisionAgentDeps,
    VisionAgentResult,
    build_vision_agent,
)

__all__ = [
    "build_vision_agent",
    "VisionAgentDeps",
    "VisionAgentResult",
    "generate_scenario",
    "calibrate_handeye",
    "ur3e_fk",
    "ur3e_ik",
    "get_joints_for_pose",
    "CALIB_FILE",
    "load_calibration",
    "capture_realsense",
    "encode_jpeg",
    "depth_at_pixel",
    "pixel_to_camera_xyz",
    "camera_to_base",
    "gemini_locate",
    "gemini_locate_gripper",
    "APPROACH_Z_OFFSET",
    "PICK_Z_OFFSET",
    "_PICK_SEED",
    "_HOME_SEED",
]
