"""
MCP client wrapper for Cobot-Claw.

MCPClient manages a single connection to a FastMCP server (streamable-http).
Agents use it to list and call tools without knowing the transport details.

Usage:
    client = MCPClient("http://localhost:8765/mcp")
    await client.connect()
    tools = await client.list_tools()
    result = await client.call_tool("get_robot_status", {})
    await client.disconnect()

Or as an async context manager:
    async with MCPClient("http://localhost:8765/mcp") as client:
        result = await client.call_tool("get_robot_status", {})
"""

from __future__ import annotations

import json
import time
from types import TracebackType
from typing import Any

from mcp import ClientSession
from mcp.client.streamable_http import streamable_http_client
from mcp.types import Tool

from cobot.logging import get_logger

try:
    import logfire as _logfire
except ImportError:
    _logfire = None  # type: ignore[assignment]

logger = get_logger("cobot.mcp.client")


class MCPClientError(Exception):
    """Raised when an MCP call fails at the transport or protocol level."""


class MCPClient:
    """
    Async client for a single FastMCP server.

    Lifecycle:
      - Call connect() before using list_tools() or call_tool().
      - Call disconnect() when done, or use as an async context manager.

    The client does not automatically reconnect on failure — callers are
    responsible for reconnecting if needed.
    """

    def __init__(self, url: str) -> None:
        self.url = url
        self._session: ClientSession | None = None
        self._transport_cm = None
        self._session_cm = None

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def connect(self) -> None:
        """
        Open connection and initialize MCP session.

        The streamablehttp_client transport must stay alive for the duration
        of the session, so we manage it via a background task that holds both
        open and signals readiness through an asyncio.Event.
        """
        if self._session is not None:
            return  # already connected

        import asyncio

        ready = asyncio.Event()
        self._connect_error: BaseException | None = None
        self._disconnect_event = asyncio.Event()

        async def _run() -> None:
            try:
                async with streamable_http_client(self.url) as (read, write, _):
                    async with ClientSession(read, write) as session:
                        await session.initialize()
                        self._session = session
                        ready.set()
                        # Hold the connection open until disconnect() is called
                        await self._disconnect_event.wait()
            except BaseException as e:
                # Catches both plain exceptions and ExceptionGroups (anyio TaskGroup)
                self._connect_error = e
                ready.set()

        self._bg_task = asyncio.create_task(_run())
        await ready.wait()

        if self._connect_error is not None:
            err = self._connect_error
            self._connect_error = None
            raise MCPClientError(f"Failed to connect to {self.url}: {err}") from err

        logger.info("mcp.connected", url=self.url)

    async def disconnect(self) -> None:
        """Signal the background task to close the session and transport."""
        if hasattr(self, "_disconnect_event") and self._disconnect_event is not None:
            self._disconnect_event.set()

        if hasattr(self, "_bg_task") and self._bg_task is not None:
            try:
                await self._bg_task
            except Exception:
                pass
            self._bg_task = None

        self._session = None
        logger.info("mcp.disconnected", url=self.url)

    async def __aenter__(self) -> "MCPClient":
        await self.connect()
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> None:
        await self.disconnect()

    # ------------------------------------------------------------------
    # Tools
    # ------------------------------------------------------------------

    def _require_session(self) -> ClientSession:
        if self._session is None:
            raise MCPClientError(
                f"Not connected to {self.url}. Call connect() first."
            )
        return self._session

    async def list_tools(self) -> list[Tool]:
        """Return all tools exposed by this MCP server."""
        session = self._require_session()
        result = await session.list_tools()
        return result.tools

    async def call_tool(self, name: str, arguments: dict[str, Any]) -> dict[str, Any]:
        """
        Call a tool by name and return its result as a dict.

        Result resolution order:
          1. structuredContent (if present) — already a dict, no parsing needed
          2. First TextContent block parsed as JSON
          3. Raw text wrapped in {"raw": ...} if JSON parse fails
          4. Empty dict if no content at all
        """
        session = self._require_session()
        _t0 = time.perf_counter()

        try:
            result = await session.call_tool(name, arguments)
        except Exception as e:
            if _logfire:
                _logfire.error(
                    "mcp.call_tool",
                    tool_name=name,
                    arguments=arguments,
                    ok=False,
                    latency_ms=round((time.perf_counter() - _t0) * 1000, 1),
                    error=str(e),
                    mcp_url=self.url,
                )
            raise MCPClientError(f"Tool call '{name}' failed: {e}") from e

        # structuredContent is preferred — clean dict, no parsing needed
        if result.structuredContent is not None:
            payload = result.structuredContent
        elif not result.content:
            logger.warning("mcp.empty_result", tool=name)
            payload = {}
        else:
            text_blocks = [b.text for b in result.content if hasattr(b, "text")]
            if not text_blocks:
                payload = {}
            else:
                text = text_blocks[0]
                try:
                    payload = json.loads(text)
                except json.JSONDecodeError:
                    logger.warning("mcp.json_parse_failed", tool=name, text=text[:120])
                    payload = {"raw": text}

        if _logfire:
            _logfire.info(
                "mcp.call_tool",
                tool_name=name,
                arguments=arguments,
                ok=bool(payload.get("ok", True)),
                latency_ms=round((time.perf_counter() - _t0) * 1000, 1),
                mcp_url=self.url,
            )
        return payload

    # ------------------------------------------------------------------
    # PydanticAI / OpenAI tool schema conversion
    # ------------------------------------------------------------------

    def to_openai_tools(self, tools: list[Tool]) -> list[dict[str, Any]]:
        """
        Convert an MCP tool list to OpenAI function-calling format.
        Pass the result directly to the `tools` parameter of chat completions.
        """
        return [
            {
                "type": "function",
                "function": {
                    "name": t.name,
                    "description": t.description or "",
                    "parameters": t.inputSchema,
                },
            }
            for t in tools
        ]

    # ------------------------------------------------------------------
    # Repr
    # ------------------------------------------------------------------

    @property
    def connected(self) -> bool:
        return self._session is not None

    def __repr__(self) -> str:
        state = "connected" if self.connected else "disconnected"
        return f"MCPClient(url={self.url!r}, state={state!r})"
