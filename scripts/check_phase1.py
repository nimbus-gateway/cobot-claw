#!/usr/bin/env python3
"""
Phase 1 smoke check — run this to verify the foundation is working.

Usage:
  python scripts/check_phase1.py

No robot or LLM server needed. All checks are offline.
"""

import sys
import traceback

PASS = "\033[92mPASS\033[0m"
FAIL = "\033[91mFAIL\033[0m"
HEAD = "\033[1m"
RESET = "\033[0m"

results = []


def check(label: str):
    """Decorator — runs the function, records pass/fail."""
    def decorator(fn):
        try:
            fn()
            results.append((True, label))
            print(f"  {PASS}  {label}")
        except Exception as e:
            results.append((False, label))
            print(f"  {FAIL}  {label}")
            print(f"       {e}")
        return fn
    return decorator


# ---------------------------------------------------------------------------
print(f"\n{HEAD}Phase 1 — Foundation{RESET}\n")
# ---------------------------------------------------------------------------

# --- config loading ---------------------------------------------------------
print("config.py")

@check("load_config() reads cobot.yaml without error")
def _():
    from cobot.config import load_config
    load_config("cobot.yaml")

@check("fleet has at least one robot")
def _():
    from cobot.config import load_config
    cfg = load_config("cobot.yaml")
    assert len(cfg.fleet) >= 1, "fleet is empty"

@check("default_robot returns first robot")
def _():
    from cobot.config import load_config
    cfg = load_config("cobot.yaml")
    r = cfg.default_robot
    print(f"       -> {r.name}  ip={r.ip}  arm={r.arm_mcp_url}", end="")

@check("robot lookup by name works")
def _():
    from cobot.config import load_config
    cfg = load_config("cobot.yaml")
    r = cfg.robot(cfg.fleet[0].name)
    assert r is not None

@check("LLM base_url is set")
def _():
    from cobot.config import load_config
    cfg = load_config("cobot.yaml")
    assert cfg.llm.base_url.startswith("http"), f"unexpected base_url: {cfg.llm.base_url}"
    print(f"       -> {cfg.llm.base_url}", end="")

@check("agent models are configured")
def _():
    from cobot.config import load_config
    cfg = load_config("cobot.yaml")
    assert cfg.agents.orchestrator.model
    assert cfg.agents.scenario.model
    print(f"       -> orchestrator={cfg.agents.orchestrator.model}  scenario={cfg.agents.scenario.model}", end="")

@check("safety threshold validator rejects bad value")
def _():
    from cobot.config import SafetyAgentConfig
    try:
        SafetyAgentConfig(model="x", alert_threshold="INVALID")
        raise AssertionError("should have raised")
    except Exception as e:
        if "INVALID" in str(e) or "alert_threshold" in str(e) or "WARNING" in str(e):
            pass  # expected validation error
        else:
            raise

@check("missing config file raises FileNotFoundError")
def _():
    from cobot.config import load_config
    try:
        load_config("nonexistent.yaml")
        raise AssertionError("should have raised")
    except FileNotFoundError:
        pass


# --- llm factory ------------------------------------------------------------
print("\nllm.py")

@check("make_openai_client() returns AsyncOpenAI instance")
def _():
    from openai import AsyncOpenAI
    from cobot.config import load_config
    from cobot.llm import make_openai_client
    cfg = load_config("cobot.yaml")
    client = make_openai_client(cfg)
    assert isinstance(client, AsyncOpenAI)
    print(f"       -> base_url={client.base_url}", end="")

@check("make_model() returns OpenAIChatModel instance")
def _():
    from pydantic_ai.models.openai import OpenAIChatModel
    from cobot.config import load_config
    from cobot.llm import make_model
    cfg = load_config("cobot.yaml")
    model = make_model(cfg.agents.scenario.model, cfg)
    assert isinstance(model, OpenAIChatModel)
    print(f"       -> model_name={model.model_name}", end="")

@check("make_model() uses model name from config")
def _():
    from cobot.config import load_config
    from cobot.llm import make_model
    cfg = load_config("cobot.yaml")
    m = make_model(cfg.agents.orchestrator.model, cfg)
    assert m.model_name == cfg.agents.orchestrator.model


# --- logging ----------------------------------------------------------------
print("\nlogging.py")

@check("configure_logging() runs without error")
def _():
    from cobot.logging import configure_logging
    configure_logging("INFO")

@check("get_logger() returns a usable logger")
def _():
    from cobot.logging import configure_logging, get_logger
    configure_logging("WARNING")   # suppress output during check
    log = get_logger("smoke")
    log.info("smoke check", phase=1)

@check("logger accepts structured key-value pairs")
def _():
    from cobot.logging import get_logger
    log = get_logger("smoke")
    log.warning("test event", component="phase1", status="ok")


# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------
passed = sum(1 for ok, _ in results if ok)
failed = sum(1 for ok, _ in results if not ok)

print(f"\n{'-' * 50}")
print(f"  {HEAD}{passed} passed  {failed} failed{RESET}  ({len(results)} checks)")

if failed:
    print("\n  Failed checks:")
    for ok, label in results:
        if not ok:
            print(f"    {FAIL}  {label}")
    print()
    sys.exit(1)
else:
    print(f"\n  {PASS}  Phase 1 foundation is working.\n")
