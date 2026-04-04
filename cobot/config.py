from __future__ import annotations

from pathlib import Path

import yaml
from pydantic import BaseModel, field_validator


class LLMConfig(BaseModel):
    provider: str = "local"   # "local" (Ollama) | "openai" | "anthropic"
    base_url: str = "http://localhost:11434/v1"
    api_key: str = ""         # leave blank to use OPENAI_API_KEY / ANTHROPIC_API_KEY env vars


class RobotConfig(BaseModel):
    name: str
    model_type: str
    ip: str
    arm_mcp_url: str
    gripper_mcp_url: str | None = None


class AgentModelConfig(BaseModel):
    model: str


class SafetyAgentConfig(AgentModelConfig):
    poll_interval_sec: float = 1.0
    alert_threshold: str = "WARNING"

    @field_validator("alert_threshold")
    @classmethod
    def validate_threshold(cls, v: str) -> str:
        if v not in {"WARNING", "BLOCKED"}:
            raise ValueError("alert_threshold must be 'WARNING' or 'BLOCKED'")
        return v


class PerceptionAgentConfig(AgentModelConfig):
    enabled: bool = False
    endpoint: str | None = None


class AgentsConfig(BaseModel):
    orchestrator: AgentModelConfig
    scenario: AgentModelConfig
    config: AgentModelConfig
    safety: SafetyAgentConfig
    perception: PerceptionAgentConfig


class ScenariosConfig(BaseModel):
    store_path: str = "./cobot/scenarios/store"


class RAGConfig(BaseModel):
    manuals_path: str = "./manuals"
    db_path: str = "./chroma_db"
    chunk_size: int = 512
    chunk_overlap: int = 64


class CobotConfig(BaseModel):
    llm: LLMConfig
    fleet: list[RobotConfig]
    agents: AgentsConfig
    scenarios: ScenariosConfig
    rag: RAGConfig

    def robot(self, name: str) -> RobotConfig:
        """Look up a robot by name. Raises KeyError if not found."""
        for r in self.fleet:
            if r.name == name:
                return r
        raise KeyError(f"Robot '{name}' not in fleet. Available: {[r.name for r in self.fleet]}")

    @property
    def default_robot(self) -> RobotConfig:
        """Return the first robot in the fleet."""
        if not self.fleet:
            raise ValueError("Fleet is empty — add at least one robot to cobot.yaml")
        return self.fleet[0]


def load_config(path: str | Path = "cobot.yaml") -> CobotConfig:
    """Load and validate cobot.yaml. Raises on missing file or invalid schema."""
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Config file not found: {p.resolve()}")
    raw = yaml.safe_load(p.read_text())
    return CobotConfig(**raw)
