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
    system_prompt: str = ""   # override agent system prompt; empty = use built-in default


class SafetyAgentConfig(AgentModelConfig):
    poll_interval_sec: float = 1.0
    alert_threshold: str = "WARNING"

    @field_validator("alert_threshold")
    @classmethod
    def validate_threshold(cls, v: str) -> str:
        if v not in {"WARNING", "BLOCKED"}:
            raise ValueError("alert_threshold must be 'WARNING' or 'BLOCKED'")
        return v


class VisionAgentConfig(AgentModelConfig):
    detector: str = "gemini"       # "gemini" | "openai_vision" (Ollama / any OpenAI-compatible endpoint)
    google_api_key: str = ""       # required when detector=gemini; leave blank to use GOOGLE_API_KEY env var
    ik_solver: str = "robot"       # "robot" = robot MCP get_inverse_kin, "local" = analytical solver
    pick_x_offset_m: float = 0.0   # correction offset applied to detected X position (tune to fix systematic grasp error)
    pick_y_offset_m: float = 0.0   # correction offset applied to detected Y position
    min_drop_z_m: float = 0.30     # minimum absolute Z (metres) for item release — prevents collision with table clutter


class AgentsConfig(BaseModel):
    orchestrator: AgentModelConfig
    scenario: AgentModelConfig
    config: AgentModelConfig
    safety: SafetyAgentConfig
    vision: VisionAgentConfig


class ScenariosConfig(BaseModel):
    store_path: str = "./cobot/scenarios/store"


class RAGConfig(BaseModel):
    manuals_path: str = "./manuals"
    db_path: str = "./chroma_db"
    chunk_size: int = 512
    chunk_overlap: int = 64


class ObservabilityConfig(BaseModel):
    enabled: bool = False
    token: str = ""   # Logfire project token — or set LOGFIRE_TOKEN env var


class CobotConfig(BaseModel):
    llm: LLMConfig
    fleet: list[RobotConfig]
    agents: AgentsConfig
    scenarios: ScenariosConfig
    rag: RAGConfig
    observability: ObservabilityConfig = ObservabilityConfig()

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
