"""Tests for cobot.llm — client and model factories."""
from __future__ import annotations

import pytest
from openai import AsyncOpenAI
from pydantic_ai.models.openai import OpenAIChatModel

from cobot.config import CobotConfig, load_config
from cobot.llm import make_model, make_openai_client


def _cfg() -> CobotConfig:
    return load_config("cobot.yaml")


class TestMakeOpenAIClient:
    def test_returns_async_openai(self):
        client = make_openai_client(_cfg())
        assert isinstance(client, AsyncOpenAI)

    def test_base_url_set(self):
        cfg = _cfg()
        client = make_openai_client(cfg)
        assert str(client.base_url).rstrip("/") == cfg.llm.base_url.rstrip("/")

    def test_api_key_set(self):
        cfg = _cfg()
        client = make_openai_client(cfg)
        assert client.api_key == cfg.llm.api_key

    def test_different_base_urls_produce_different_clients(self):
        from cobot.config import LLMConfig, CobotConfig
        cfg1 = _cfg()
        cfg2 = _cfg().model_copy(
            update={"llm": LLMConfig(base_url="http://otherhost:11434/v1", api_key="local")}
        )
        c1 = make_openai_client(cfg1)
        c2 = make_openai_client(cfg2)
        assert str(c1.base_url) != str(c2.base_url)


class TestMakeModel:
    def test_returns_openai_model(self):
        model = make_model("qwen2.5:14b", _cfg())
        assert isinstance(model, OpenAIChatModel)

    def test_model_name_stored(self):
        model = make_model("qwen2.5:14b", _cfg())
        assert model.model_name == "qwen2.5:14b"

    def test_different_model_names(self):
        cfg = _cfg()
        m1 = make_model("qwen2.5:14b", cfg)
        m2 = make_model("qwen2.5:72b", cfg)
        assert m1.model_name != m2.model_name

    def test_orchestrator_model_from_config(self):
        cfg = _cfg()
        model = make_model(cfg.agents.orchestrator.model, cfg)
        assert model.model_name == cfg.agents.orchestrator.model

    def test_scenario_model_from_config(self):
        cfg = _cfg()
        model = make_model(cfg.agents.scenario.model, cfg)
        assert model.model_name == cfg.agents.scenario.model
