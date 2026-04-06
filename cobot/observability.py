"""
Logfire observability setup for cobot-claw.

Instruments pydantic-ai agents so every LLM call, tool invocation,
input, output, and token usage is visible in the Logfire UI.

Install extras before using:
    pip install -e ".[obs]"

Then either:
  - Set observability.token in cobot.yaml, or
  - Export LOGFIRE_TOKEN=<your-token>

If logfire is not installed or observability is disabled, this module
is a no-op — the CLI never crashes.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from cobot.config import CobotConfig

logger = logging.getLogger(__name__)


def setup(cfg: "CobotConfig") -> None:
    """Configure Logfire if enabled in cobot.yaml. Safe to call unconditionally."""
    if not cfg.observability.enabled:
        return

    try:
        import logfire
    except ImportError:
        logger.warning(
            "observability.enabled=true but logfire is not installed. "
            "Run: pip install -e \".[obs]\""
        )
        return

    import os

    token = cfg.observability.token or os.environ.get("LOGFIRE_TOKEN") or None

    logfire.configure(
        token=token,
        service_name="cobot-claw",
        send_to_logfire=token is not None,  # local-only mode if no token
    )

    logfire.instrument_pydantic_ai()

    logger.info(
        "logfire.configured",
        send_to_logfire=token is not None,
    )
