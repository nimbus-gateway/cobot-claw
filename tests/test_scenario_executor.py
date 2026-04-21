"""
Tests for cobot/scenarios/executor.py
"""

import asyncio
import pytest
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

from cobot.scenarios.schema import (
    Scenario, WaitStep, MoveJointsStep, MoveNamedPoseStep,
    ToolActionStep, RepeatStep, CheckMotionStep, MoveTCPStep,
)
from cobot.scenarios.executor import ScenarioExecutor, StepResult, ExecutionResult
from cobot.mcp.perception_contract import PerceptionClient


def _make_executor(arm_responses: dict | None = None) -> tuple[ScenarioExecutor, MagicMock]:
    """Return executor + mock arm. arm_responses maps tool name -> return value."""
    arm = MagicMock()
    responses = arm_responses or {}

    async def call_tool(name, args):
        if name in responses:
            return responses[name]
        return {"ok": True}

    arm.call_tool = call_tool
    perception = PerceptionClient()
    executor = ScenarioExecutor(arm, {}, perception)
    return executor, arm


class TestStepResult:
    def test_to_dict(self):
        r = StepResult(0, "wait", True, {"ok": True}, 1.2345)
        d = r.to_dict()
        assert d["index"] == 0
        assert d["step_type"] == "wait"
        assert d["ok"] is True
        assert d["elapsed_sec"] == 1.234


class TestDryRun:
    async def test_dry_run_all_steps_pass(self):
        executor, _ = _make_executor()
        scenario = Scenario(name="test", steps=[
            WaitStep(type="wait", duration_sec=0.0),
            MoveNamedPoseStep(type="move_named_pose", pose="home"),
        ])
        result = await executor.run(scenario, dry_run=True)
        assert result.success is True
        assert result.steps_completed == 2
        assert all(r.response == {"dry_run": True} for r in result.step_results)


class TestWaitStep:
    async def test_wait_returns_ok(self):
        executor, _ = _make_executor()
        scenario = Scenario(name="test", steps=[WaitStep(type="wait", duration_sec=0.0)])
        result = await executor.run(scenario)
        assert result.success is True
        assert result.step_results[0].ok is True


class TestMoveJointsStep:
    async def test_calls_arm_tool(self):
        arm = MagicMock()
        calls = []

        async def call_tool(name, args):
            calls.append((name, args))
            return {"ok": True}

        arm.call_tool = call_tool
        executor = ScenarioExecutor(arm, {}, PerceptionClient())
        scenario = Scenario(name="test", steps=[
            MoveJointsStep(type="move_joints", joints=[0.0]*6, duration_sec=3.0),
        ])
        result = await executor.run(scenario)
        assert result.success is True
        assert calls[0][0] == "move_joints"
        assert calls[0][1]["joints"] == [0.0]*6
        assert calls[0][1]["duration_sec"] == 3.0


class TestToolActionStep:
    async def test_tool_not_registered(self):
        executor, _ = _make_executor()
        scenario = Scenario(name="test", steps=[
            ToolActionStep(type="tool_action", tool="gripper", action="close"),
        ])
        result = await executor.run(scenario)
        assert result.success is False
        assert "not registered" in result.error

    async def test_tool_action_no_recovery(self):
        arm = MagicMock()
        gripper = MagicMock()
        arm_calls = []
        gripper_calls = []

        async def arm_call(name, args):
            arm_calls.append(name)
            return {"ok": True}

        async def gripper_call(name, args):
            gripper_calls.append(name)
            return {"ok": True}

        arm.call_tool = arm_call
        gripper.call_tool = gripper_call
        executor = ScenarioExecutor(arm, {"gripper": gripper}, PerceptionClient())
        scenario = Scenario(name="test", steps=[
            ToolActionStep(type="tool_action", tool="gripper", action="close",
                           arm_recovery_sec=0.0),
        ])
        result = await executor.run(scenario)
        assert result.success is True
        assert gripper_calls == ["close"]
        assert arm_calls == []  # no recovery poll when arm_recovery_sec=0

    async def test_tool_action_with_recovery_polls_arm(self):
        arm = MagicMock()
        gripper = MagicMock()
        arm_calls = []

        async def arm_call(name, args):
            arm_calls.append(name)
            return {"ok": True, "ros_ready": True}

        async def gripper_call(name, args):
            return {"ok": True}

        arm.call_tool = arm_call
        gripper.call_tool = gripper_call
        executor = ScenarioExecutor(arm, {"gripper": gripper}, PerceptionClient())
        scenario = Scenario(name="test", steps=[
            ToolActionStep(type="tool_action", tool="gripper", action="close",
                           arm_recovery_sec=2.0),
        ])
        result = await executor.run(scenario)
        assert result.success is True
        assert "get_robot_status" in arm_calls  # recovery polling happened


class TestRepeatStep:
    async def test_repeat_embeds_child_results(self):
        executor, _ = _make_executor()
        scenario = Scenario(name="test", steps=[
            RepeatStep(type="repeat", count=3,
                       steps=[WaitStep(type="wait", duration_sec=0.0)]),
        ])
        result = await executor.run(scenario)
        assert result.success is True
        rep = result.step_results[0]
        assert rep.ok is True
        assert len(rep.response["iterations"]) == 3
        for it in rep.response["iterations"]:
            assert len(it["steps"]) == 1
            assert it["steps"][0]["ok"] is True

    async def test_repeat_stops_on_failure_with_trace(self):
        call_count = 0

        arm = MagicMock()

        async def call_tool(name, args):
            nonlocal call_count
            call_count += 1
            if call_count >= 2:
                return {"ok": False, "message": "arm error"}
            return {"ok": True}

        arm.call_tool = call_tool
        executor = ScenarioExecutor(arm, {}, PerceptionClient())
        scenario = Scenario(name="test", steps=[
            RepeatStep(type="repeat", count=5,
                       steps=[MoveNamedPoseStep(type="move_named_pose", pose="home")]),
        ])
        result = await executor.run(scenario)
        assert result.success is False
        rep = result.step_results[0]
        assert rep.response["failed_iteration"] == 1
        # iterations up to and including the failure are recorded
        assert len(rep.response["iterations"]) == 2


class TestCheckMotionStep:
    async def test_motion_complete(self):
        arm = MagicMock()

        async def call_tool(name, args):
            if name == "is_motion_complete":
                return {"complete": True}
            return {"ok": True}

        arm.call_tool = call_tool
        executor = ScenarioExecutor(arm, {}, PerceptionClient())
        scenario = Scenario(name="test", steps=[
            CheckMotionStep(type="check_motion_complete", timeout_sec=2.0),
        ])
        result = await executor.run(scenario)
        assert result.success is True

    async def test_motion_timeout(self):
        arm = MagicMock()

        async def call_tool(name, args):
            if name == "is_motion_complete":
                return {"complete": False}
            return {"ok": True}

        arm.call_tool = call_tool
        executor = ScenarioExecutor(arm, {}, PerceptionClient())
        scenario = Scenario(name="test", steps=[
            CheckMotionStep(type="check_motion_complete", timeout_sec=0.1),
        ])
        result = await executor.run(scenario)
        assert result.success is False
        assert "timeout" in result.step_results[0].response["message"]


class TestWaypointOrdering:
    """Ensure multi-waypoint scenarios call move_joints in the correct order
    and do not skip any steps (regression for fire-and-forget bug)."""

    async def test_all_waypoints_called_in_order(self):
        arm = MagicMock()
        calls = []

        async def call_tool(name, args):
            calls.append((name, dict(args)))
            return {"ok": True}

        arm.call_tool = call_tool
        executor = ScenarioExecutor(arm, {}, PerceptionClient())

        waypoints = [
            [0.1, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.2, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.3, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.4, 0.0, 0.0, 0.0, 0.0, 0.0],
        ]
        scenario = Scenario(name="test", steps=[
            MoveJointsStep(type="move_joints", joints=j, duration_sec=1.0)
            for j in waypoints
        ])

        result = await executor.run(scenario)

        assert result.success is True
        assert result.steps_completed == 4

        move_calls = [c for c in calls if c[0] == "move_joints"]
        assert len(move_calls) == 4
        for i, mc in enumerate(move_calls):
            assert mc[1]["joints"] == waypoints[i], (
                f"Waypoint {i} mismatch: expected {waypoints[i]}, got {mc[1]['joints']}"
            )

    async def test_waypoints_not_skipped_when_earlier_one_is_slow(self):
        """Simulate a slow first move — subsequent waypoints must still execute."""
        arm = MagicMock()
        calls = []

        async def call_tool(name, args):
            calls.append((name, dict(args)))
            return {"ok": True}

        arm.call_tool = call_tool
        executor = ScenarioExecutor(arm, {}, PerceptionClient())

        scenario = Scenario(name="test", steps=[
            MoveJointsStep(type="move_joints", joints=[float(i)] + [0.0]*5, duration_sec=0.5)
            for i in range(3)
        ])

        result = await executor.run(scenario)

        assert result.success is True
        assert result.steps_completed == 3
        move_calls = [c for c in calls if c[0] == "move_joints"]
        assert len(move_calls) == 3
        # Verify joints are in increasing order (not collapsed to last)
        base_joints = [mc[1]["joints"][0] for mc in move_calls]
        assert base_joints == [0.0, 1.0, 2.0]

    async def test_joint_and_tool_steps_interleaved(self):
        """Scenario with move_joints and tool_action steps executes all in order."""
        arm = MagicMock()
        gripper = MagicMock()
        arm_calls = []
        gripper_calls = []

        async def arm_call(name, args):
            arm_calls.append(name)
            return {"ok": True}

        async def gripper_call(name, args):
            gripper_calls.append(name)
            return {"ok": True}

        arm.call_tool = arm_call
        gripper.call_tool = gripper_call
        executor = ScenarioExecutor(arm, {"gripper": gripper}, PerceptionClient())

        scenario = Scenario(name="test", steps=[
            MoveJointsStep(type="move_joints", joints=[0.0]*6, duration_sec=1.0),
            ToolActionStep(type="tool_action", tool="gripper", action="close", arm_recovery_sec=0.0),
            MoveJointsStep(type="move_joints", joints=[0.5]*6, duration_sec=1.0),
            ToolActionStep(type="tool_action", tool="gripper", action="open", arm_recovery_sec=0.0),
        ])

        result = await executor.run(scenario)

        assert result.success is True
        assert result.steps_completed == 4
        assert arm_calls.count("move_joints") == 2
        assert gripper_calls == ["close", "open"]


class TestLoadYaml:
    def test_load_pick_place(self):
        path = Path("cobot/scenarios/store/pick_place.yaml")
        scenario = ScenarioExecutor.load(path)
        assert scenario.name == "pick-place-box"
        assert scenario.robot == "ur3e"
        # gripper close step should have arm_recovery_sec set
        gripper_steps = [s for s in scenario.steps if s.type == "tool_action"]
        assert all(s.arm_recovery_sec > 0 for s in gripper_steps)


class TestStepFailurePropagates:
    async def test_failed_step_stops_execution(self):
        arm = MagicMock()
        call_count = 0

        async def call_tool(name, args):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return {"ok": False, "message": "connection lost"}
            return {"ok": True}

        arm.call_tool = call_tool
        executor = ScenarioExecutor(arm, {}, PerceptionClient())
        scenario = Scenario(name="test", steps=[
            MoveNamedPoseStep(type="move_named_pose", pose="home"),
            WaitStep(type="wait", duration_sec=0.0),
        ])
        result = await executor.run(scenario)
        assert result.success is False
        assert result.steps_completed == 0
        assert call_count == 1  # second step never called
