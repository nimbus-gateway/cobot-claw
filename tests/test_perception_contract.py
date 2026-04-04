"""Tests for cobot.mcp.perception_contract.PerceptionClient."""
from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from cobot.mcp.perception_contract import PerceptionClient


def _mock_mcp_client(responses: dict) -> MagicMock:
    """Build a mock MCPClient that returns preset responses per tool name."""
    client = MagicMock()

    async def call_tool(name: str, arguments: dict):
        return responses.get(name, {})

    client.call_tool = call_tool
    return client


# ---------------------------------------------------------------------------
# Disabled (default) — all methods return None
# ---------------------------------------------------------------------------

class TestPerceptionClientDisabled:
    def test_disabled_by_default(self):
        p = PerceptionClient()
        assert p.available is False

    def test_disabled_when_no_client(self):
        p = PerceptionClient(mcp_client=None, enabled=True)
        assert p.available is False

    def test_disabled_when_enabled_false(self):
        mock = MagicMock()
        p = PerceptionClient(mcp_client=mock, enabled=False)
        assert p.available is False

    @pytest.mark.asyncio
    async def test_locate_object_returns_none(self):
        p = PerceptionClient()
        assert await p.locate_object("red_box") is None

    @pytest.mark.asyncio
    async def test_estimate_object_pose_returns_none(self):
        p = PerceptionClient()
        assert await p.estimate_object_pose("red_box") is None

    @pytest.mark.asyncio
    async def test_is_workspace_clear_returns_none(self):
        p = PerceptionClient()
        assert await p.is_workspace_clear() is None

    @pytest.mark.asyncio
    async def test_get_scene_objects_returns_none(self):
        p = PerceptionClient()
        assert await p.get_scene_objects() is None

    def test_repr_shows_unavailable(self):
        p = PerceptionClient()
        assert "False" in repr(p)


# ---------------------------------------------------------------------------
# Enabled — delegates to MCP client
# ---------------------------------------------------------------------------

class TestPerceptionClientEnabled:
    def _enabled(self, responses: dict) -> PerceptionClient:
        mock = _mock_mcp_client(responses)
        return PerceptionClient(mcp_client=mock, enabled=True)

    def test_available_when_enabled_with_client(self):
        p = self._enabled({})
        assert p.available is True

    @pytest.mark.asyncio
    async def test_locate_object_delegates(self):
        p = self._enabled({
            "locate_object": {"found": True, "x": 0.3, "y": 0.1, "z": 0.15, "confidence": 0.92}
        })
        result = await p.locate_object("red_box")
        assert result is not None
        assert result["found"] is True
        assert result["x"] == 0.3

    @pytest.mark.asyncio
    async def test_locate_object_not_found(self):
        p = self._enabled({
            "locate_object": {"found": False, "x": 0.0, "y": 0.0, "z": 0.0, "confidence": 0.0}
        })
        result = await p.locate_object("blue_cube")
        assert result["found"] is False

    @pytest.mark.asyncio
    async def test_estimate_object_pose_delegates(self):
        p = self._enabled({
            "estimate_object_pose": {
                "found": True,
                "position": {"x": 0.3, "y": 0.1, "z": 0.15},
                "orientation": {"qx": 0.0, "qy": 0.0, "qz": 0.0, "qw": 1.0},
                "confidence": 0.88,
            }
        })
        result = await p.estimate_object_pose("red_box")
        assert result["found"] is True
        assert "position" in result
        assert "orientation" in result

    @pytest.mark.asyncio
    async def test_is_workspace_clear_delegates(self):
        p = self._enabled({
            "is_workspace_clear": {"clear": True, "obstacles": []}
        })
        result = await p.is_workspace_clear()
        assert result["clear"] is True
        assert result["obstacles"] == []

    @pytest.mark.asyncio
    async def test_is_workspace_clear_with_obstacle(self):
        p = self._enabled({
            "is_workspace_clear": {
                "clear": False,
                "obstacles": [{"label": "unknown_object", "position": {"x": 0.2, "y": 0.0, "z": 0.1}}]
            }
        })
        result = await p.is_workspace_clear()
        assert result["clear"] is False
        assert len(result["obstacles"]) == 1

    @pytest.mark.asyncio
    async def test_get_scene_objects_delegates(self):
        p = self._enabled({
            "get_scene_objects": {
                "objects": [
                    {"label": "red_box", "position": {"x": 0.3, "y": 0.1, "z": 0.1}, "confidence": 0.9},
                    {"label": "blue_cup", "position": {"x": 0.1, "y": 0.2, "z": 0.1}, "confidence": 0.7},
                ]
            }
        })
        result = await p.get_scene_objects()
        assert len(result["objects"]) == 2
        labels = [o["label"] for o in result["objects"]]
        assert "red_box" in labels
        assert "blue_cup" in labels

    @pytest.mark.asyncio
    async def test_get_scene_objects_empty_scene(self):
        p = self._enabled({"get_scene_objects": {"objects": []}})
        result = await p.get_scene_objects()
        assert result["objects"] == []

    def test_repr_shows_available(self):
        p = self._enabled({})
        assert "True" in repr(p)
