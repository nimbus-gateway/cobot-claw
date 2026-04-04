"""Tests for cobot.config — schema validation and load_config."""
from __future__ import annotations

import textwrap
from pathlib import Path

import pytest
import yaml

from cobot.config import (
    AgentModelConfig,
    AgentsConfig,
    CobotConfig,
    LLMConfig,
    PerceptionAgentConfig,
    RAGConfig,
    RobotConfig,
    SafetyAgentConfig,
    ScenariosConfig,
    load_config,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _write_yaml(tmp_path: Path, data: dict) -> Path:
    p = tmp_path / "cobot.yaml"
    p.write_text(yaml.dump(data))
    return p


def _minimal_config() -> dict:
    return {
        "llm": {"base_url": "http://localhost:11434/v1", "api_key": "local"},
        "fleet": [
            {
                "name": "ur3e",
                "model_type": "ur3e",
                "ip": "192.168.1.2",
                "arm_mcp_url": "http://localhost:8765/mcp",
                "gripper_mcp_url": "http://localhost:8001/mcp",
            }
        ],
        "agents": {
            "orchestrator": {"model": "qwen2.5:72b"},
            "scenario": {"model": "qwen2.5:14b"},
            "config": {"model": "qwen2.5:72b"},
            "safety": {"model": "qwen2.5:7b", "poll_interval_sec": 1.0, "alert_threshold": "WARNING"},
            "perception": {"enabled": False, "endpoint": None, "model": "qwen2.5:14b"},
        },
        "scenarios": {"store_path": "./cobot/scenarios/store"},
        "rag": {"manuals_path": "./manuals", "db_path": "./chroma_db", "chunk_size": 512, "chunk_overlap": 64},
    }


# ---------------------------------------------------------------------------
# LLMConfig
# ---------------------------------------------------------------------------

class TestLLMConfig:
    def test_required_base_url(self):
        cfg = LLMConfig(base_url="http://localhost:11434/v1")
        assert cfg.base_url == "http://localhost:11434/v1"

    def test_default_api_key(self):
        cfg = LLMConfig(base_url="http://localhost:11434/v1")
        assert cfg.api_key == "local"

    def test_custom_api_key(self):
        cfg = LLMConfig(base_url="http://host/v1", api_key="sk-secret")
        assert cfg.api_key == "sk-secret"

    def test_missing_base_url_raises(self):
        with pytest.raises(Exception):
            LLMConfig()  # type: ignore[call-arg]


# ---------------------------------------------------------------------------
# RobotConfig
# ---------------------------------------------------------------------------

class TestRobotConfig:
    def test_basic(self):
        r = RobotConfig(name="ur3e", model_type="ur3e", ip="192.168.1.2",
                        arm_mcp_url="http://localhost:8765/mcp")
        assert r.name == "ur3e"
        assert r.gripper_mcp_url is None

    def test_with_gripper(self):
        r = RobotConfig(name="ur3e", model_type="ur3e", ip="192.168.1.2",
                        arm_mcp_url="http://localhost:8765/mcp",
                        gripper_mcp_url="http://localhost:8001/mcp")
        assert r.gripper_mcp_url == "http://localhost:8001/mcp"


# ---------------------------------------------------------------------------
# SafetyAgentConfig
# ---------------------------------------------------------------------------

class TestSafetyAgentConfig:
    def test_valid_warning(self):
        cfg = SafetyAgentConfig(model="qwen2.5:7b", alert_threshold="WARNING")
        assert cfg.alert_threshold == "WARNING"

    def test_valid_blocked(self):
        cfg = SafetyAgentConfig(model="qwen2.5:7b", alert_threshold="BLOCKED")
        assert cfg.alert_threshold == "BLOCKED"

    def test_invalid_threshold_raises(self):
        with pytest.raises(Exception):
            SafetyAgentConfig(model="qwen2.5:7b", alert_threshold="CRITICAL")

    def test_defaults(self):
        cfg = SafetyAgentConfig(model="qwen2.5:7b")
        assert cfg.poll_interval_sec == 1.0
        assert cfg.alert_threshold == "WARNING"


# ---------------------------------------------------------------------------
# PerceptionAgentConfig
# ---------------------------------------------------------------------------

class TestPerceptionAgentConfig:
    def test_disabled_by_default(self):
        cfg = PerceptionAgentConfig(model="qwen2.5:14b")
        assert cfg.enabled is False
        assert cfg.endpoint is None

    def test_enabled_with_endpoint(self):
        cfg = PerceptionAgentConfig(model="qwen2.5:14b", enabled=True,
                                     endpoint="http://localhost:8767/mcp")
        assert cfg.enabled is True
        assert cfg.endpoint == "http://localhost:8767/mcp"


# ---------------------------------------------------------------------------
# CobotConfig helpers
# ---------------------------------------------------------------------------

class TestCobotConfigHelpers:
    def _make_cfg(self) -> CobotConfig:
        return CobotConfig(**_minimal_config())

    def test_default_robot_returns_first(self):
        cfg = self._make_cfg()
        assert cfg.default_robot.name == "ur3e"

    def test_robot_lookup_by_name(self):
        cfg = self._make_cfg()
        r = cfg.robot("ur3e")
        assert r.ip == "192.168.1.2"

    def test_robot_lookup_missing_raises(self):
        cfg = self._make_cfg()
        with pytest.raises(KeyError, match="ur5e"):
            cfg.robot("ur5e")

    def test_default_robot_empty_fleet_raises(self):
        data = _minimal_config()
        data["fleet"] = []
        cfg = CobotConfig(**data)
        with pytest.raises(ValueError, match="empty"):
            _ = cfg.default_robot

    def test_multi_robot_fleet(self):
        data = _minimal_config()
        data["fleet"].append({
            "name": "ur5e",
            "model_type": "ur5e",
            "ip": "192.168.1.3",
            "arm_mcp_url": "http://localhost:8766/mcp",
        })
        cfg = CobotConfig(**data)
        assert len(cfg.fleet) == 2
        assert cfg.robot("ur5e").ip == "192.168.1.3"


# ---------------------------------------------------------------------------
# load_config
# ---------------------------------------------------------------------------

class TestLoadConfig:
    def test_loads_valid_file(self, tmp_path: Path):
        p = _write_yaml(tmp_path, _minimal_config())
        cfg = load_config(p)
        assert isinstance(cfg, CobotConfig)
        assert cfg.llm.base_url == "http://localhost:11434/v1"
        assert cfg.fleet[0].name == "ur3e"
        assert cfg.agents.orchestrator.model == "qwen2.5:72b"
        assert cfg.agents.scenario.model == "qwen2.5:14b"
        assert cfg.agents.safety.poll_interval_sec == 1.0
        assert cfg.agents.perception.enabled is False
        assert cfg.scenarios.store_path == "./cobot/scenarios/store"
        assert cfg.rag.chunk_size == 512

    def test_missing_file_raises(self, tmp_path: Path):
        with pytest.raises(FileNotFoundError, match="not found"):
            load_config(tmp_path / "missing.yaml")

    def test_invalid_schema_raises(self, tmp_path: Path):
        p = tmp_path / "cobot.yaml"
        p.write_text("not_a_valid_key: true\n")
        with pytest.raises(Exception):
            load_config(p)

    def test_loads_project_cobot_yaml(self):
        """Validate the real cobot.yaml in the project root is well-formed."""
        cfg = load_config("cobot.yaml")
        assert cfg.llm.base_url
        assert len(cfg.fleet) >= 1
        assert cfg.fleet[0].arm_mcp_url

    def test_gripper_url_optional(self, tmp_path: Path):
        data = _minimal_config()
        data["fleet"][0].pop("gripper_mcp_url")
        p = _write_yaml(tmp_path, data)
        cfg = load_config(p)
        assert cfg.fleet[0].gripper_mcp_url is None

    def test_rag_defaults(self, tmp_path: Path):
        data = _minimal_config()
        data["rag"] = {}
        p = _write_yaml(tmp_path, data)
        cfg = load_config(p)
        assert cfg.rag.chunk_size == 512
        assert cfg.rag.chunk_overlap == 64

    def test_scenarios_default_store_path(self, tmp_path: Path):
        data = _minimal_config()
        data["scenarios"] = {}
        p = _write_yaml(tmp_path, data)
        cfg = load_config(p)
        assert cfg.scenarios.store_path == "./cobot/scenarios/store"
