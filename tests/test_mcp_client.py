"""
Tests for cobot.mcp.client.MCPClient

Unit tests run without any MCP server.
Integration tests require a live server — they are skipped automatically
if the server is unreachable (mark with @pytest.mark.integration to run).

Run only unit tests:
    pytest tests/test_mcp_client.py -m "not integration"

Run all including integration (servers must be running):
    pytest tests/test_mcp_client.py
"""
from __future__ import annotations

import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from cobot.mcp.client import MCPClient, MCPClientError


# ---------------------------------------------------------------------------
# Unit tests — no server needed
# ---------------------------------------------------------------------------

class TestMCPClientInit:
    def test_url_stored(self):
        c = MCPClient("http://localhost:8765/mcp")
        assert c.url == "http://localhost:8765/mcp"

    def test_initially_disconnected(self):
        c = MCPClient("http://localhost:8765/mcp")
        assert c.connected is False
        assert c._session is None

    def test_repr_disconnected(self):
        c = MCPClient("http://localhost:8765/mcp")
        r = repr(c)
        assert "disconnected" in r
        assert "8765" in r

    def test_repr_connected(self):
        c = MCPClient("http://localhost:8765/mcp")
        c._session = MagicMock()  # fake session
        assert "connected" in repr(c)


class TestRequireSession:
    def test_raises_when_not_connected(self):
        c = MCPClient("http://localhost:8765/mcp")
        with pytest.raises(MCPClientError, match="Not connected"):
            c._require_session()

    def test_returns_session_when_connected(self):
        c = MCPClient("http://localhost:8765/mcp")
        mock_session = MagicMock()
        c._session = mock_session
        assert c._require_session() is mock_session


class TestCallToolResultParsing:
    """Test _result parsing logic without a real server by mocking the session."""

    def _client_with_mock_session(self) -> tuple[MCPClient, AsyncMock]:
        c = MCPClient("http://localhost:8765/mcp")
        mock_session = AsyncMock()
        c._session = mock_session
        return c, mock_session

    def _make_result(self, text: str | None = None, structured: dict | None = None):
        """Build a fake CallToolResult."""
        result = MagicMock()
        result.structuredContent = structured

        if text is not None:
            content_block = MagicMock()
            content_block.text = text
            result.content = [content_block]
        else:
            result.content = []

        return result

    @pytest.mark.asyncio
    async def test_structured_content_takes_priority(self):
        c, session = self._client_with_mock_session()
        session.call_tool.return_value = self._make_result(
            text='{"ignored": true}',
            structured={"ok": True, "value": 42},
        )
        result = await c.call_tool("some_tool", {})
        assert result == {"ok": True, "value": 42}

    @pytest.mark.asyncio
    async def test_text_content_parsed_as_json(self):
        c, session = self._client_with_mock_session()
        session.call_tool.return_value = self._make_result(
            text='{"ok": true, "joint_state": [0.1, 0.2]}',
            structured=None,
        )
        result = await c.call_tool("get_joint_states", {})
        assert result["ok"] is True
        assert result["joint_state"] == [0.1, 0.2]

    @pytest.mark.asyncio
    async def test_invalid_json_returns_raw(self):
        c, session = self._client_with_mock_session()
        session.call_tool.return_value = self._make_result(
            text="not json at all",
            structured=None,
        )
        result = await c.call_tool("some_tool", {})
        assert "raw" in result
        assert result["raw"] == "not json at all"

    @pytest.mark.asyncio
    async def test_empty_content_returns_empty_dict(self):
        c, session = self._client_with_mock_session()
        result = MagicMock()
        result.structuredContent = None
        result.content = []
        session.call_tool.return_value = result
        out = await c.call_tool("some_tool", {})
        assert out == {}

    @pytest.mark.asyncio
    async def test_tool_call_exception_raises_mcp_error(self):
        c, session = self._client_with_mock_session()
        session.call_tool.side_effect = RuntimeError("connection lost")
        with pytest.raises(MCPClientError, match="connection lost"):
            await c.call_tool("some_tool", {})


class TestToOpenAITools:
    def test_converts_tool_list(self):
        c = MCPClient("http://localhost:8765/mcp")

        tool = MagicMock()
        tool.name = "move_joints"
        tool.description = "Move arm joints"
        tool.inputSchema = {
            "type": "object",
            "properties": {"joints": {"type": "array"}},
            "required": ["joints"],
        }

        openai_tools = c.to_openai_tools([tool])

        assert len(openai_tools) == 1
        assert openai_tools[0]["type"] == "function"
        assert openai_tools[0]["function"]["name"] == "move_joints"
        assert openai_tools[0]["function"]["description"] == "Move arm joints"
        assert "joints" in openai_tools[0]["function"]["parameters"]["properties"]

    def test_empty_description_becomes_empty_string(self):
        c = MCPClient("http://localhost:8765/mcp")
        tool = MagicMock()
        tool.name = "test"
        tool.description = None
        tool.inputSchema = {"type": "object", "properties": {}}
        result = c.to_openai_tools([tool])
        assert result[0]["function"]["description"] == ""

    def test_empty_tool_list(self):
        c = MCPClient("http://localhost:8765/mcp")
        assert c.to_openai_tools([]) == []

    def test_multiple_tools(self):
        c = MCPClient("http://localhost:8765/mcp")
        tools = []
        for name in ["open", "close", "move", "get_status"]:
            t = MagicMock()
            t.name = name
            t.description = f"Gripper {name}"
            t.inputSchema = {"type": "object", "properties": {}}
            tools.append(t)

        result = c.to_openai_tools(tools)
        assert len(result) == 4
        assert [r["function"]["name"] for r in result] == ["open", "close", "move", "get_status"]


class TestConnectFailure:
    @pytest.mark.asyncio
    async def test_connect_failure_raises_mcp_error(self):
        c = MCPClient("http://localhost:19999/mcp")  # nothing listening here
        with pytest.raises(MCPClientError, match="Failed to connect"):
            await c.connect()

    @pytest.mark.asyncio
    async def test_session_remains_none_after_failed_connect(self):
        c = MCPClient("http://localhost:19999/mcp")
        try:
            await c.connect()
        except MCPClientError:
            pass
        assert c._session is None
        assert c.connected is False

    @pytest.mark.asyncio
    async def test_list_tools_without_connect_raises(self):
        c = MCPClient("http://localhost:8765/mcp")
        with pytest.raises(MCPClientError, match="Not connected"):
            await c.list_tools()

    @pytest.mark.asyncio
    async def test_call_tool_without_connect_raises(self):
        c = MCPClient("http://localhost:8765/mcp")
        with pytest.raises(MCPClientError, match="Not connected"):
            await c.call_tool("get_robot_status", {})


# ---------------------------------------------------------------------------
# Integration tests — require live MCP servers
# Run: pytest tests/test_mcp_client.py -m integration
# ---------------------------------------------------------------------------

def _server_reachable(url: str) -> bool:
    import urllib.request
    try:
        urllib.request.urlopen(url.replace("/mcp", "/health"), timeout=1)
        return True
    except Exception:
        return False


@pytest.mark.integration
class TestArmMCPIntegration:
    ARM_URL = "http://localhost:8765/mcp"

    @pytest.fixture(autouse=True)
    def skip_if_offline(self):
        if not _server_reachable(self.ARM_URL):
            pytest.skip("Arm MCP server not reachable — start tools/ure3/ur3e_arm_mcp.py")

    @pytest.mark.asyncio
    async def test_connect_and_list_tools(self):
        async with MCPClient(self.ARM_URL) as c:
            tools = await c.list_tools()
            names = [t.name for t in tools]
            assert "get_robot_status" in names
            assert "move_joints" in names
            assert "get_joint_states" in names

    @pytest.mark.asyncio
    async def test_get_robot_status(self):
        async with MCPClient(self.ARM_URL) as c:
            result = await c.call_tool("get_robot_status", {})
            assert isinstance(result, dict)
            assert "ros_ready" in result

    @pytest.mark.asyncio
    async def test_to_openai_tools_format(self):
        async with MCPClient(self.ARM_URL) as c:
            tools = await c.list_tools()
            openai_tools = c.to_openai_tools(tools)
            for t in openai_tools:
                assert t["type"] == "function"
                assert "name" in t["function"]
                assert "parameters" in t["function"]


@pytest.mark.integration
class TestGripperMCPIntegration:
    GRIPPER_URL = "http://localhost:8001/mcp"

    @pytest.fixture(autouse=True)
    def skip_if_offline(self):
        if not _server_reachable(self.GRIPPER_URL):
            pytest.skip("Gripper MCP server not reachable — start tools/gripper/server.py")

    @pytest.mark.asyncio
    async def test_connect_and_list_tools(self):
        async with MCPClient(self.GRIPPER_URL) as c:
            tools = await c.list_tools()
            names = [t.name for t in tools]
            assert "open" in names
            assert "close" in names
            assert "get_status" in names

    @pytest.mark.asyncio
    async def test_get_status(self):
        async with MCPClient(self.GRIPPER_URL) as c:
            result = await c.call_tool("get_status", {})
            assert isinstance(result, dict)
