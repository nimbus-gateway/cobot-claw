from __future__ import annotations

import os

from cobot.config import CobotConfig


def make_model(model_name: str, cfg: CobotConfig):
    """
    Return a PydanticAI model for the given provider:

      provider: local      → Ollama (OpenAI-compatible, custom base_url)
      provider: openai     → OpenAI API  (uses OPENAI_API_KEY or api_key)
      provider: anthropic  → Anthropic API (uses ANTHROPIC_API_KEY or api_key)

    Switch providers by changing `llm.provider` in cobot.yaml.
    """
    provider = cfg.llm.provider.lower()

    if provider == "anthropic":
        from pydantic_ai.models.anthropic import AnthropicModel
        from pydantic_ai.providers.anthropic import AnthropicProvider
        api_key = cfg.llm.api_key or os.environ.get("ANTHROPIC_API_KEY", "")
        if not api_key:
            raise ValueError(
                "Anthropic provider requires an API key. "
                "Set llm.api_key in cobot.yaml or export ANTHROPIC_API_KEY."
            )
        return AnthropicModel(model_name, provider=AnthropicProvider(api_key=api_key))

    # Both "openai" and "local" use OpenAIProvider — only base_url differs
    from openai import AsyncOpenAI
    from pydantic_ai.models.openai import OpenAIChatModel
    from pydantic_ai.providers.openai import OpenAIProvider

    if provider == "openai":
        api_key = cfg.llm.api_key or os.environ.get("OPENAI_API_KEY", "")
        if not api_key:
            raise ValueError(
                "OpenAI provider requires an API key. "
                "Set llm.api_key in cobot.yaml or export OPENAI_API_KEY."
            )
        client = AsyncOpenAI(api_key=api_key)
    else:
        # local / Ollama
        api_key = cfg.llm.api_key or "local"
        client = AsyncOpenAI(base_url=cfg.llm.base_url, api_key=api_key)

    provider_obj = OpenAIProvider(openai_client=client)
    return OpenAIChatModel(model_name, provider=provider_obj)


def make_openai_client(cfg: CobotConfig) -> "AsyncOpenAI":
    """Return a raw AsyncOpenAI client (used outside pydantic-ai, e.g. embeddings)."""
    from openai import AsyncOpenAI
    api_key = cfg.llm.api_key or os.environ.get("OPENAI_API_KEY", "local")
    return AsyncOpenAI(base_url=cfg.llm.base_url, api_key=api_key)
