"""
Deterministic scenario executor.

Runs each step in order, calling arm/gripper MCP tools directly.
Perception queries are resolved via PerceptionClient — falls back to
scenario-defined coordinates when perception is unavailable.

No LLM is invoked during execution of predefined scenarios.
"""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass
from pathlib import Path

import yaml
import structlog

from cobot.scenarios.schema import (
    Scenario, ScenarioStep,
    MoveJointsStep, MoveNamedPoseStep, MoveTCPStep, MovePathStep,
    ToolActionStep, WaitStep, CheckMotionStep, RepeatStep,
)
from cobot.mcp.client import MCPClient
from cobot.mcp.perception_contract import PerceptionClient

logger = structlog.get_logger()


@dataclass
class StepResult:
    step_index: int
    step_type: str
    ok: bool
    response: dict
    duration_sec: float

    def to_dict(self) -> dict:
        return {
            "index": self.step_index,
            "step_type": self.step_type,
            "ok": self.ok,
            "response": self.response,
            "elapsed_sec": round(self.duration_sec, 3),
        }


@dataclass
class ExecutionResult:
    scenario_name: str
    success: bool
    steps_completed: int
    total_steps: int
    error: str | None
    step_results: list[StepResult]


class ScenarioExecutor:
    def __init__(
        self,
        arm_mcp: MCPClient,
        tool_mcps: dict[str, MCPClient],   # e.g. {"gripper": MCPClient(...)}
        perception: PerceptionClient,
    ):
        self.arm = arm_mcp
        self.tools = tool_mcps
        self.perception = perception

    @staticmethod
    def load(path: str | Path) -> Scenario:
        raw = yaml.safe_load(Path(path).read_text())
        return Scenario(**raw)

    async def run(self, scenario: Scenario, dry_run: bool = False) -> ExecutionResult:
        logger.info("executor.start", scenario=scenario.name, dry_run=dry_run)
        results: list[StepResult] = []

        ok, error = await self._run_steps(scenario.steps, results, dry_run, offset=0)

        return ExecutionResult(
            scenario_name=scenario.name,
            success=ok,
            steps_completed=len([r for r in results if r.ok]),
            total_steps=len(results),
            error=error,
            step_results=results,
        )

    async def _run_steps(
        self,
        steps: list[ScenarioStep],
        results: list[StepResult],
        dry_run: bool,
        offset: int,
    ) -> tuple[bool, str | None]:
        for i, step in enumerate(steps):
            idx = offset + i
            t0 = time.time()

            if dry_run:
                logger.info("executor.dry_run_step", index=idx, type=step.type)
                results.append(StepResult(idx, step.type, True, {"dry_run": True}, 0.0))
                continue

            # RepeatStep is handled inline to preserve per-iteration child step results
            if isinstance(step, RepeatStep):
                all_iter_results: list[dict] = []
                for iteration in range(step.count):
                    iter_results: list[StepResult] = []
                    ok, error = await self._run_steps(step.steps, iter_results, dry_run, offset=0)
                    all_iter_results.append({
                        "iteration": iteration,
                        "steps": [r.to_dict() for r in iter_results],
                    })
                    if not ok:
                        response = {
                            "ok": False,
                            "message": error,
                            "failed_iteration": iteration,
                            "iterations": all_iter_results,
                        }
                        results.append(StepResult(idx, step.type, False, response, time.time() - t0))
                        return False, f"Step {idx} (repeat) failed on iteration {iteration}: {error}"
                response = {"ok": True, "iterations": all_iter_results}
                results.append(StepResult(idx, step.type, True, response, time.time() - t0))
                continue

            try:
                response = await self._execute_step(step)
            except Exception as e:
                results.append(StepResult(idx, step.type, False, {"error": str(e)}, time.time() - t0))
                return False, f"Step {idx} ({step.type}) raised: {e}"

            ok = response.get("ok", True)
            results.append(StepResult(idx, step.type, ok, response, time.time() - t0))

            if not ok:
                return False, f"Step {idx} ({step.type}) failed: {response.get('message', '')}"

        return True, None

    async def _execute_step(self, step: ScenarioStep) -> dict:
        match step:
            case MoveJointsStep():
                return await self.arm.call_tool("move_joints", {
                    "joints": step.joints,
                    "speed": step.speed,
                    "accel": step.accel,
                    "blend_radius": step.blend_radius,
                })

            case MovePathStep():
                return await self.arm.call_tool("move_path", {
                    "waypoints": [
                        {
                            "joints": wp.joints,
                            "speed": wp.speed,
                            "accel": wp.accel,
                            "blend_radius": wp.blend_radius,
                        }
                        for wp in step.waypoints
                    ],
                })

            case MoveNamedPoseStep():
                return await self.arm.call_tool("move_named_pose", {
                    "name": step.pose,
                    "speed": step.speed,
                    "accel": step.accel,
                    "via_home": step.via_home,
                })

            case MoveTCPStep():
                pos = await self._resolve_tcp_position(step)
                return await self.arm.call_tool("move_tcp_pose", {
                    "x": pos["x"], "y": pos["y"], "z": pos["z"],
                    "qx": pos.get("qx", step.qx),
                    "qy": pos.get("qy", step.qy),
                    "qz": pos.get("qz", step.qz),
                    "qw": pos.get("qw", step.qw),
                    "speed": step.speed,
                    "accel": step.accel,
                })

            case ToolActionStep():
                if step.tool not in self.tools:
                    return {"ok": False, "message": f"Tool '{step.tool}' not registered"}
                result = await self.tools[step.tool].call_tool(step.action, step.params)
                if step.arm_recovery_sec > 0:
                    await self._wait_for_arm_recovery(step.arm_recovery_sec)
                return result

            case WaitStep():
                await asyncio.sleep(step.duration_sec)
                return {"ok": True}

            case CheckMotionStep():
                deadline = time.time() + step.timeout_sec
                while time.time() < deadline:
                    result = await self.arm.call_tool("is_motion_complete",
                                                      {"tolerance_rad": step.tolerance_rad})
                    if result.get("complete"):
                        return {"ok": True, "result": result}
                    await asyncio.sleep(0.2)
                return {"ok": False, "message": "Motion completion timeout"}

            case _:
                return {"ok": False, "message": f"Unknown step type: {step.type}"}

    async def _resolve_tcp_position(self, step: MoveTCPStep) -> dict:
        """Use perception if available, otherwise fall back to step coordinates."""
        if step.perception and step.perception.locate:
            result = await self.perception.locate_object(step.perception.locate)
            if result and result.get("found"):
                return result   # has x, y, z keys
            logger.warning("executor.perception_miss", label=step.perception.locate,
                           fallback=step.fallback_position is not None)

        if step.fallback_position:
            fb = step.fallback_position
            return {"x": fb.x, "y": fb.y, "z": fb.z,
                    "qx": fb.qx, "qy": fb.qy, "qz": fb.qz, "qw": fb.qw}

        return {"x": step.x, "y": step.y, "z": step.z}

    async def _wait_for_arm_recovery(self, timeout_sec: float) -> None:
        """Restore UR External Control after a gripper/tool action interrupts it.

        Strategy:
          1. Wait a short settling time for the tool script to finish.
          2. Actively call recover_external_control (resend_robot_program in
             headless mode, dashboard play otherwise).
          3. Poll get_joint_states — a successful response confirms ROS and the
             arm controller are alive (does not require get_robot_status, which
             may be unavailable due to dashboard service issues).
          4. Raise RuntimeError on timeout so the executor aborts the scenario
             rather than sending a move command to an uncontrolled arm.
        """
        SETTLE_SEC = 1.5   # give the tool script time to finish before recovery

        await asyncio.sleep(SETTLE_SEC)

        # Actively trigger External Control recovery
        try:
            await self.arm.call_tool("recover_external_control", {})
        except Exception as e:
            logger.warning("executor.recover_external_control_failed", error=str(e))

        deadline = time.time() + max(0.0, timeout_sec - SETTLE_SEC)
        while time.time() < deadline:
            await asyncio.sleep(0.5)
            try:
                result = await self.arm.call_tool("get_joint_states", {})
                if result.get("ok"):
                    return   # arm controller is responding — safe to continue
            except Exception:
                pass  # still recovering — keep polling

        # Timeout: do NOT continue — a move command now would cause a Protective Stop
        raise RuntimeError(
            f"Arm did not recover External Control within {timeout_sec}s. "
            "Scenario aborted to prevent unsafe motion. "
            "Check the teach pendant and use /reconnect to restore."
        )
