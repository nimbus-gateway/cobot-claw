"""
Tests for cobot/agents/ — deps, robotic agent tools, orchestrator deps.

These tests exercise tool logic in isolation using mock MCPClients and a
minimal CobotConfig. No LLM calls are made — we call tool functions directly
via the agent's _function_tools registry.
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest

from cobot.agents.deps import RoboticAgentDeps, ConfigAgentDeps
from cobot.agents.orchestrator import OrchestratorDeps, build_orchestrator
from cobot.agents.robotic_agent import build_robotic_agent, RoboticAgentResult
from cobot.agents.config_agent import build_config_agent
from cobot.config import (
    CobotConfig, LLMConfig, RobotConfig, AgentsConfig,
    AgentModelConfig, SafetyAgentConfig, PerceptionAgentConfig,
    ScenariosConfig, RAGConfig,
)
from cobot.mcp.perception_contract import PerceptionClient
from cobot.scenarios.executor import ScenarioExecutor
from cobot.rag.pipeline import RAGPipeline


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
    arm = MagicMock()
    resp = responses or {}

    async def call_tool(name, args):
        return resp.get(name, {"ok": True})

    arm.call_tool = call_tool
    return arm


def _make_robotic_deps(
    arm_responses: dict | None = None,
    gripper_responses: dict | None = None,
    store_path: str = "./cobot/scenarios/store",
) -> RoboticAgentDeps:
    cfg = _make_cfg(store_path)
    arm = _make_mock_mcp(arm_responses)
    gripper = _make_mock_mcp(gripper_responses)
    perception = PerceptionClient()
    executor = ScenarioExecutor(arm, {"gripper": gripper}, perception)
    return RoboticAgentDeps(arm, {"gripper": gripper}, executor, perception, cfg)


def _get_tool(agent, name: str):
    """Return a tool's function from agent._function_toolset.tools by name."""
    tools = agent._function_toolset.tools
    if name in tools:
        return tools[name].function
    raise KeyError(f"Tool '{name}' not found in agent. Available: {list(tools)}")


# ---------------------------------------------------------------------------
# RoboticAgentDeps
# ---------------------------------------------------------------------------

class TestRoboticAgentDeps:
    def test_construction(self):
        deps = _make_robotic_deps()
        assert deps.arm_mcp is not None
        assert "gripper" in deps.tool_mcps
        assert deps.executor is not None
        assert deps.perception is not None
        assert deps.cfg is not None


# ---------------------------------------------------------------------------
# OrchestratorDeps
# ---------------------------------------------------------------------------

class TestOrchestratorDeps:
    def test_config_deps_defaults_none(self):
        cfg = _make_cfg()
        scenario_deps = _make_robotic_deps()
        od = OrchestratorDeps(robotic_deps=scenario_deps, cfg=cfg)
        assert od.config_deps is None

    def test_config_deps_can_be_set(self):
        cfg = _make_cfg()
        scenario_deps = _make_robotic_deps()
        arm = _make_mock_mcp()
        rag = RAGPipeline()
        config_deps = ConfigAgentDeps(arm_mcp=arm, rag_pipeline=rag,
                                      scenarios_store_path="./store", cfg=cfg)
        od = OrchestratorDeps(robotic_deps=scenario_deps, cfg=cfg, config_deps=config_deps)
        assert od.config_deps is not None


# ---------------------------------------------------------------------------
# RAGPipeline stub
# ---------------------------------------------------------------------------

class TestRAGPipelineStub:
    async def test_query_returns_message(self):
        rag = RAGPipeline()
        msg = await rag.query("How do I enable freedrive?")
        assert "RAG not available" in msg

    async def test_ingest_returns_not_ok(self):
        rag = RAGPipeline()
        result = await rag.ingest("./manuals")
        assert result["ok"] is False


# ---------------------------------------------------------------------------
# Scenario Agent tool functions (called directly, no LLM)
# ---------------------------------------------------------------------------

class TestRoboticAgentTools:
    def _agent_and_deps(self, arm_responses=None, store_path="./cobot/scenarios/store"):
        cfg = _make_cfg(store_path)
        agent = build_robotic_agent(cfg)
        deps = _make_robotic_deps(arm_responses=arm_responses, store_path=store_path)
        return agent, deps


    async def test_get_arm_status(self):
        agent, deps = self._agent_and_deps(arm_responses={"get_robot_status": {"ok": True, "state": "idle"}})
        fn = _get_tool(agent, "get_arm_status")
        ctx = MagicMock()
        ctx.deps = deps
        result = await fn(ctx)
        assert result["state"] == "idle"

    async def test_list_scenarios(self, tmp_path):
        (tmp_path / "pick_place.yaml").write_text("name: pick_place\nsteps: []\n")
        (tmp_path / "calibrate.yaml").write_text("name: calibrate\nsteps: []\n")
        agent, deps = self._agent_and_deps(store_path=str(tmp_path))
        fn = _get_tool(agent, "list_scenarios")
        ctx = MagicMock()
        ctx.deps = deps
        result = await fn(ctx)
        assert set(result) == {"pick_place", "calibrate"}

    async def test_run_scenario_not_found(self, tmp_path):
        agent, deps = self._agent_and_deps(store_path=str(tmp_path))
        fn = _get_tool(agent, "run_scenario")
        ctx = MagicMock()
        ctx.deps = deps
        result = await fn(ctx, name="nonexistent")
        assert result["ok"] is False
        assert "not found" in result["message"]

    async def test_run_scenario_dry_run(self, tmp_path):
        (tmp_path / "test_scenario.yaml").write_text(
            "name: test_scenario\nsteps:\n  - type: wait\n    duration_sec: 0.0\n"
        )
        agent, deps = self._agent_and_deps(store_path=str(tmp_path))
        fn = _get_tool(agent, "run_scenario")
        ctx = MagicMock()
        ctx.deps = deps
        result = await fn(ctx, name="test_scenario", dry_run=True)
        assert result["ok"] is True
        assert result["steps_completed"] == 1

    async def test_list_tool_mcps_no_tools(self):
        agent, deps = self._agent_and_deps()
        # Override tool_mcps to empty
        deps = RoboticAgentDeps(deps.arm_mcp, {}, deps.executor, deps.perception, deps.cfg)
        fn = _get_tool(agent, "list_tool_mcps")
        ctx = MagicMock()
        ctx.deps = deps
        result = await fn(ctx)
        assert result == {}

    async def test_list_tool_mcps_returns_action_names(self):
        from mcp.types import Tool as MCPTool
        cfg = _make_cfg()
        agent = build_robotic_agent(cfg)
        arm = _make_mock_mcp()

        gripper_mcp = MagicMock()
        vacuum_mcp = MagicMock()

        async def gripper_list_tools():
            return [MCPTool(name="open", description="", inputSchema={}),
                    MCPTool(name="close", description="", inputSchema={})]

        async def vacuum_list_tools():
            return [MCPTool(name="activate", description="", inputSchema={}),
                    MCPTool(name="deactivate", description="", inputSchema={})]

        gripper_mcp.list_tools = gripper_list_tools
        vacuum_mcp.list_tools = vacuum_list_tools

        perception = PerceptionClient()
        executor = ScenarioExecutor(arm, {"gripper": gripper_mcp, "vacuum": vacuum_mcp}, perception)
        deps = RoboticAgentDeps(arm, {"gripper": gripper_mcp, "vacuum": vacuum_mcp}, executor, perception, cfg)
        fn = _get_tool(agent, "list_tool_mcps")
        ctx = MagicMock()
        ctx.deps = deps
        result = await fn(ctx)
        assert set(result["gripper"]) == {"open", "close"}
        assert set(result["vacuum"]) == {"activate", "deactivate"}

    async def test_call_tool_action_unknown_tool(self):
        agent, deps = self._agent_and_deps()
        fn = _get_tool(agent, "call_tool_action")
        ctx = MagicMock()
        ctx.deps = deps
        result = await fn(ctx, tool="laser", action="fire")
        assert result["ok"] is False
        assert "laser" in result["message"]

    async def test_call_tool_action_dispatches_to_mcp(self):
        calls = []

        async def gripper_call(name, args):
            calls.append((name, args))
            return {"ok": True, "applied_speed": 30}

        cfg = _make_cfg()
        agent = build_robotic_agent(cfg)
        arm = _make_mock_mcp()
        gripper = MagicMock()
        gripper.call_tool = gripper_call
        perception = PerceptionClient()
        executor = ScenarioExecutor(arm, {"gripper": gripper}, perception)
        deps = RoboticAgentDeps(arm, {"gripper": gripper}, executor, perception, cfg)
        fn = _get_tool(agent, "call_tool_action")
        ctx = MagicMock()
        ctx.deps = deps
        result = await fn(ctx, tool="gripper", action="open", params={"speed": 30, "force": 10})
        assert result["ok"] is True
        assert calls[0] == ("open", {"speed": 30, "force": 10})

    async def test_call_tool_action_vacuum_tool(self):
        """Demonstrates tool-agnosticism: vacuum pump works without any code change."""
        calls = []

        async def vacuum_call(name, args):
            calls.append((name, args))
            return {"ok": True, "suction_kpa": 60}

        cfg = _make_cfg()
        agent = build_robotic_agent(cfg)
        arm = _make_mock_mcp()
        vacuum = MagicMock()
        vacuum.call_tool = vacuum_call
        perception = PerceptionClient()
        executor = ScenarioExecutor(arm, {"vacuum": vacuum}, perception)
        deps = RoboticAgentDeps(arm, {"vacuum": vacuum}, executor, perception, cfg)
        fn = _get_tool(agent, "call_tool_action")
        ctx = MagicMock()
        ctx.deps = deps
        result = await fn(ctx, tool="vacuum", action="activate", params={"pressure_kpa": 60})
        assert result["suction_kpa"] == 60
        assert calls[0] == ("activate", {"pressure_kpa": 60})

    async def test_call_tool_action_no_params(self):
        """params defaults to {} when not provided."""
        calls = []

        async def gripper_call(name, args):
            calls.append((name, args))
            return {"ok": True}

        cfg = _make_cfg()
        agent = build_robotic_agent(cfg)
        arm = _make_mock_mcp()
        gripper = MagicMock()
        gripper.call_tool = gripper_call
        perception = PerceptionClient()
        executor = ScenarioExecutor(arm, {"gripper": gripper}, perception)
        deps = RoboticAgentDeps(arm, {"gripper": gripper}, executor, perception, cfg)
        fn = _get_tool(agent, "call_tool_action")
        ctx = MagicMock()
        ctx.deps = deps
        result = await fn(ctx, tool="gripper", action="get_status")
        assert calls[0] == ("get_status", {})

    async def test_move_to_pose(self):
        calls = []

        async def arm_call(name, args):
            calls.append((name, args))
            return {"ok": True}

        cfg = _make_cfg()
        agent = build_robotic_agent(cfg)
        arm = MagicMock()
        arm.call_tool = arm_call
        perception = PerceptionClient()
        executor = ScenarioExecutor(arm, {}, perception)
        deps = RoboticAgentDeps(arm, {}, executor, perception, cfg)
        fn = _get_tool(agent, "move_to_pose")
        ctx = MagicMock()
        ctx.deps = deps
        result = await fn(ctx, pose_name="home", duration_sec=3.0)
        assert calls[0] == ("move_named_pose", {"name": "home", "duration_sec": 3.0})


# ---------------------------------------------------------------------------
# Config Agent tool functions
# ---------------------------------------------------------------------------

class TestConfigAgentTools:
    def _agent_and_deps(self) -> tuple:
        cfg = _make_cfg()
        agent = build_config_agent(cfg)
        arm = _make_mock_mcp({"get_robot_status": {"ok": True, "state": "idle"}})
        rag = RAGPipeline()
        deps = ConfigAgentDeps(arm_mcp=arm, rag_pipeline=rag,
                               scenarios_store_path="./store", cfg=cfg)
        return agent, deps

    async def test_manual_query_stub(self):
        agent, deps = self._agent_and_deps()
        fn = _get_tool(agent, "manual_query")
        ctx = MagicMock()
        ctx.deps = deps
        result = await fn(ctx, question="How do I enable freedrive?")
        assert "RAG not available" in result

    async def test_get_robot_status(self):
        agent, deps = self._agent_and_deps()
        fn = _get_tool(agent, "get_robot_status")
        ctx = MagicMock()
        ctx.deps = deps
        result = await fn(ctx)
        assert result["state"] == "idle"


# ---------------------------------------------------------------------------
# Orchestrator — config_deps guard
# ---------------------------------------------------------------------------

class TestOrchestratorTools:
    async def test_run_config_task_returns_unavailable_when_no_config_deps(self):
        cfg = _make_cfg()
        agent = build_orchestrator(cfg)
        scenario_deps = _make_robotic_deps()
        od = OrchestratorDeps(robotic_deps=scenario_deps, cfg=cfg, config_deps=None)
        fn = _get_tool(agent, "run_config_task")
        ctx = MagicMock()
        ctx.deps = od
        result = await fn(ctx, task_description="what is freedrive?")
        assert "not yet available" in result

    async def test_run_robotic_task_tool_exists(self):
        """run_robotic_task is the sole robot-facing tool on the orchestrator."""
        cfg = _make_cfg()
        agent = build_orchestrator(cfg)
        tools = list(agent._function_toolset.tools)
        assert "run_robotic_task" in tools
        assert "get_robot_status" not in tools
        assert "get_tool_mcp_status" not in tools
