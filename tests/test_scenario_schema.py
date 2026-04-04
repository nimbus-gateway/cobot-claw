"""
Tests for cobot/scenarios/schema.py
"""

import pytest
from pydantic import ValidationError

from cobot.scenarios.schema import (
    Scenario,
    MoveJointsStep,
    MoveNamedPoseStep,
    MoveTCPStep,
    ToolActionStep,
    WaitStep,
    CheckMotionStep,
    RepeatStep,
    FallbackPosition,
    PerceptionQuery,
)


class TestMoveJointsStep:
    def test_valid(self):
        s = MoveJointsStep(type="move_joints", joints=[0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        assert s.duration_sec == 5.0

    def test_too_few_joints(self):
        with pytest.raises(ValidationError):
            MoveJointsStep(type="move_joints", joints=[0.0, 0.0])

    def test_too_many_joints(self):
        with pytest.raises(ValidationError):
            MoveJointsStep(type="move_joints", joints=[0.0] * 7)


class TestMoveNamedPoseStep:
    def test_defaults(self):
        s = MoveNamedPoseStep(type="move_named_pose", pose="home")
        assert s.via_home is False
        assert s.duration_sec == 5.0


class TestMoveTCPStep:
    def test_minimal(self):
        s = MoveTCPStep(type="move_tcp_pose", x=0.3, y=0.1, z=0.2)
        assert s.qw == 1.0
        assert s.perception is None
        assert s.fallback_position is None

    def test_with_perception_and_fallback(self):
        s = MoveTCPStep(
            type="move_tcp_pose",
            x=0.3, y=0.1, z=0.2,
            perception=PerceptionQuery(locate="red_box"),
            fallback_position=FallbackPosition(x=0.3, y=0.1, z=0.2),
        )
        assert s.perception.locate == "red_box"
        assert s.fallback_position.x == 0.3


class TestToolActionStep:
    def test_defaults(self):
        s = ToolActionStep(type="tool_action", tool="gripper", action="close")
        assert s.params == {}
        assert s.arm_recovery_sec == 0.0

    def test_arm_recovery_sec(self):
        s = ToolActionStep(
            type="tool_action", tool="gripper", action="close",
            params={"speed": 50, "force": 60},
            arm_recovery_sec=6.0,
        )
        assert s.arm_recovery_sec == 6.0


class TestWaitStep:
    def test_valid(self):
        s = WaitStep(type="wait", duration_sec=2.5)
        assert s.duration_sec == 2.5


class TestCheckMotionStep:
    def test_defaults(self):
        s = CheckMotionStep(type="check_motion_complete")
        assert s.timeout_sec == 10.0
        assert s.tolerance_rad == 0.03


class TestRepeatStep:
    def test_nested_steps(self):
        rs = RepeatStep(
            type="repeat",
            count=3,
            steps=[WaitStep(type="wait", duration_sec=0.1)],
        )
        assert rs.count == 3
        assert len(rs.steps) == 1
        assert rs.steps[0].type == "wait"

    def test_empty_steps(self):
        rs = RepeatStep(type="repeat", count=2, steps=[])
        assert rs.steps == []


class TestScenario:
    def test_parse_full(self):
        data = {
            "name": "test",
            "description": "test scenario",
            "robot": "ur3e",
            "steps": [
                {"type": "move_named_pose", "pose": "home"},
                {"type": "tool_action", "tool": "gripper", "action": "close",
                 "params": {"speed": 50}, "arm_recovery_sec": 6.0},
                {"type": "wait", "duration_sec": 1.0},
                {"type": "repeat", "count": 2, "steps": [
                    {"type": "wait", "duration_sec": 0.5},
                ]},
            ],
        }
        s = Scenario(**data)
        assert s.name == "test"
        assert len(s.steps) == 4
        assert s.steps[1].arm_recovery_sec == 6.0  # type: ignore[attr-defined]

    def test_default_robot(self):
        s = Scenario(name="x", steps=[])
        assert s.robot == "default"

    def test_invalid_step_type(self):
        with pytest.raises(ValidationError):
            Scenario(name="x", steps=[{"type": "invalid_step"}])
