"""
Tests for cobot/cli/ — session, renderer, completer, commands.

No REPL loop or live MCP servers needed. All async tests use mock deps.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest

from cobot.agents.deps import RoboticAgentDeps
from cobot.agents.orchestrator import OrchestratorDeps
from cobot.cli.commands.scenario import cmd_run, cmd_scenarios
from cobot.cli.commands.status import cmd_status
from cobot.cli.completer import CobotCompleter, SLASH_COMMANDS
from cobot.cli.session import CobotSession
from cobot.config import (
    AgentModelConfig, AgentsConfig, CobotConfig, LLMConfig,
    PerceptionAgentConfig, RAGConfig, RobotConfig, SafetyAgentConfig,
    ScenariosConfig,
)
from cobot.mcp.perception_contract import PerceptionClient
from cobot.scenarios.executor import ScenarioExecutor


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_cfg(store_path: str = "./cobot/scenarios/store") -> CobotConfig:
    return CobotConfig(
        llm=LLMConfig(base_url="http://localhost:11434/v1"),
        fleet=[RobotConfig(name="ur3e", model_type="UR3e", ip="192.168.1.100",
                           arm_mcp_url="http://localhost:8765/mcp",
                           gripper_mcp_url="http://localhost:8001/mcp")],
        agents=AgentsConfig(
            orchestrator=AgentModelConfig(model="qwen2.5:72b"),
            scenario=AgentModelConfig(model="qwen2.5:14b"),
            config=AgentModelConfig(model="qwen2.5:14b"),
            safety=SafetyAgentConfig(model="qwen2.5:7b"),
            perception=PerceptionAgentConfig(model="llava:13b"),
        ),
        scenarios=ScenariosConfig(store_path=store_path),
        rag=RAGConfig(),
    )


def _make_mock_mcp(responses: dict | None = None) -> MagicMock:
    mcp = MagicMock()
    resp = responses or {}

    async def call_tool(name, args):
        return resp.get(name, {"ok": True})

    mcp.call_tool = call_tool
    return mcp


def _make_session(store_path: str = "./cobot/scenarios/store",
                  arm_responses: dict | None = None,
                  connected: bool = True) -> CobotSession:
    cfg = _make_cfg(store_path)
    session = CobotSession(cfg=cfg, active_robot=cfg.default_robot)
    if connected:
        arm = _make_mock_mcp(arm_responses)
        perception = PerceptionClient()
        executor = ScenarioExecutor(arm, {}, perception)
        scenario_deps = RoboticAgentDeps(arm, {}, executor, perception, cfg)
        session.deps = OrchestratorDeps(robotic_deps=scenario_deps, cfg=cfg)
        session.mcp_status["arm"] = True
    return session


# ---------------------------------------------------------------------------
# CobotSession
# ---------------------------------------------------------------------------

class TestCobotSession:
    def test_robot_connected_true_when_deps_set(self):
        session = _make_session(connected=True)
        assert session.robot_connected is True

    def test_robot_connected_false_when_no_deps(self):
        session = _make_session(connected=False)
        assert session.robot_connected is False

    def test_default_active_agent(self):
        session = _make_session()
        assert session.active_agent == "orchestrator"

    def test_history_starts_empty(self):
        session = _make_session()
        assert session.history == []


# ---------------------------------------------------------------------------
# CobotCompleter
# ---------------------------------------------------------------------------

class TestCobotCompleter:
    def _completer(self, store_path="./cobot/scenarios/store") -> CobotCompleter:
        session = _make_session(store_path=store_path)
        return CobotCompleter(session)

    def _completions(self, completer: CobotCompleter, text: str) -> list[str]:
        from prompt_toolkit.document import Document
        doc = Document(text, cursor_position=len(text))
        return [c.text for c in completer.get_completions(doc, None)]

    def test_slash_prefix_completes_commands(self):
        c = self._completer()
        results = self._completions(c, "/st")
        assert "/status" in results

    def test_slash_prefix_completes_all_on_single_slash(self):
        c = self._completer()
        results = self._completions(c, "/")
        for cmd in SLASH_COMMANDS:
            assert cmd in results

    def test_run_completes_scenario_names(self, tmp_path):
        (tmp_path / "pick_place.yaml").write_text("name: x\nsteps: []\n")
        (tmp_path / "calibrate.yaml").write_text("name: x\nsteps: []\n")
        c = self._completer(store_path=str(tmp_path))
        results = self._completions(c, "/run p")
        assert "pick_place" in results
        assert "calibrate" not in results

    def test_agent_completes_agent_names(self):
        c = self._completer()
        results = self._completions(c, "/agent o")
        assert "orchestrator" in results

    def test_no_completions_for_plain_text(self):
        c = self._completer()
        results = self._completions(c, "move the robot")
        assert results == []


# ---------------------------------------------------------------------------
# cmd_status
# ---------------------------------------------------------------------------

class TestCmdStatus:
    async def test_prints_error_when_not_connected(self, capsys):
        session = _make_session(connected=False)
        # Just check it doesn't raise; rich prints to its own console
        await cmd_status(session)

    async def test_calls_arm_get_robot_status(self):
        calls = []

        async def call_tool(name, args):
            calls.append(name)
            return {"robot_ip": "192.168.1.2", "has_joint_state": True,
                    "headless_mode": True, "motion": {"complete": True}}

        cfg = _make_cfg()
        arm = MagicMock()
        arm.call_tool = call_tool
        perception = PerceptionClient()
        executor = ScenarioExecutor(arm, {}, perception)
        scenario_deps = RoboticAgentDeps(arm, {}, executor, perception, cfg)
        session = CobotSession(cfg=cfg, active_robot=cfg.default_robot)
        session.deps = OrchestratorDeps(robotic_deps=scenario_deps, cfg=cfg)

        await cmd_status(session)
        assert "get_robot_status" in calls


# ---------------------------------------------------------------------------
# cmd_run / cmd_scenarios
# ---------------------------------------------------------------------------

class TestCmdRun:
    async def test_not_connected_prints_error(self):
        session = _make_session(connected=False)
        await cmd_run(session, "pick_place")  # should not raise

    async def test_missing_scenario_prints_error(self, tmp_path):
        session = _make_session(store_path=str(tmp_path))
        await cmd_run(session, "nonexistent")  # should not raise

    async def test_runs_scenario_dry_run(self, tmp_path):
        (tmp_path / "test_scenario.yaml").write_text(
            "name: test_scenario\nsteps:\n  - type: wait\n    duration_sec: 0.0\n"
        )
        session = _make_session(store_path=str(tmp_path))
        await cmd_run(session, "test_scenario", dry_run=True)

    async def test_runs_scenario_with_arm_failure(self, tmp_path):
        (tmp_path / "fail_scenario.yaml").write_text(
            "name: fail_scenario\nsteps:\n"
            "  - type: move_named_pose\n    pose: home\n"
        )
        arm = _make_mock_mcp({"move_named_pose": {"ok": False, "message": "arm error"}})
        cfg = _make_cfg(str(tmp_path))
        perception = PerceptionClient()
        executor = ScenarioExecutor(arm, {}, perception)
        scenario_deps = RoboticAgentDeps(arm, {}, executor, perception, cfg)
        session = CobotSession(cfg=cfg, active_robot=cfg.default_robot)
        session.deps = OrchestratorDeps(robotic_deps=scenario_deps, cfg=cfg)
        # Should print failure without raising
        await cmd_run(session, "fail_scenario")


class TestCmdScenarios:
    async def test_lists_yaml_files(self, tmp_path):
        (tmp_path / "alpha.yaml").write_text("name: alpha\nsteps: []\n")
        (tmp_path / "beta.yaml").write_text("name: beta\nsteps: []\n")
        session = _make_session(store_path=str(tmp_path))
        await cmd_scenarios(session)  # should not raise

    async def test_empty_store_prints_message(self, tmp_path):
        session = _make_session(store_path=str(tmp_path))
        await cmd_scenarios(session)  # should not raise
