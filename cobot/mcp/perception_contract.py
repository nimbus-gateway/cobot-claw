"""
Perception MCP Contract
=======================
Specification for any future perception MCP server in the Cobot-Claw system.

Any perception MCP server MUST expose the four tools documented below with
these exact names, parameter types, and return shapes.

This file is the authoritative interface spec. It also contains PerceptionClient,
a null-safe wrapper used by the Scenario Executor. When perception is disabled
(the default), all methods return None and execution falls back to
scenario-defined coordinates — no errors, no broken scenarios.

Enabling perception (Phase 8):
  1. Deploy a perception MCP server satisfying this contract.
  2. Set perception.enabled: true and perception.endpoint in cobot.yaml.
  3. Wire up a real MCPClient in cli/app.py._connect_session().
  No other code changes needed.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from cobot.mcp.client import MCPClient


# ---------------------------------------------------------------------------
# Required tool signatures
# (documentation only — implemented by the perception MCP server, not here)
# ---------------------------------------------------------------------------
#
# @mcp.tool()
# def locate_object(label: str) -> dict:
#     """
#     Find a named object in the workspace.
#     Returns position in robot base frame (metres).
#
#     Return shape:
#       {
#         "found": bool,
#         "x": float,
#         "y": float,
#         "z": float,
#         "confidence": float        # 0.0 – 1.0
#       }
#     """
#
# @mcp.tool()
# def estimate_object_pose(label: str) -> dict:
#     """
#     Full 6-DOF pose of a named object in robot base frame.
#
#     Return shape:
#       {
#         "found": bool,
#         "position":    {"x": float, "y": float, "z": float},
#         "orientation": {"qx": float, "qy": float, "qz": float, "qw": float},
#         "confidence":  float
#       }
#     """
#
# @mcp.tool()
# def is_workspace_clear() -> dict:
#     """
#     Check for unexpected obstacles in the robot workspace.
#
#     Return shape:
#       {
#         "clear": bool,
#         "obstacles": [{"label": str, "position": {"x": float, "y": float, "z": float}}]
#       }
#     """
#
# @mcp.tool()
# def get_scene_objects() -> dict:
#     """
#     All detected objects currently visible in the scene.
#
#     Return shape:
#       {
#         "objects": [
#           {"label": str, "position": {"x": float, "y": float, "z": float}, "confidence": float}
#         ]
#       }
#     """


# ---------------------------------------------------------------------------
# Null-safe client wrapper
# ---------------------------------------------------------------------------

class PerceptionClient:
    """
    Wraps a perception MCP session with null-safe fallback.

    When perception is disabled (enabled=False or no mcp_client provided),
    all query methods return None. The Scenario Executor treats None as
    "perception unavailable" and falls back to scenario-defined coordinates.

    This means scenarios work identically with or without perception —
    they just become more accurate when perception is available.
    """

    def __init__(
        self,
        mcp_client: "MCPClient | None" = None,
        enabled: bool = False,
    ) -> None:
        self._client = mcp_client
        self.available = enabled and mcp_client is not None

    async def locate_object(self, label: str) -> dict[str, Any] | None:
        """
        Find a named object. Returns position dict or None if unavailable.

        Result when available: {"found": bool, "x": float, "y": float, "z": float, "confidence": float}
        """
        if not self.available:
            return None
        return await self._client.call_tool("locate_object", {"label": label})

    async def estimate_object_pose(self, label: str) -> dict[str, Any] | None:
        """
        Get full 6-DOF pose of a named object. Returns pose dict or None.

        Result when available:
          {"found": bool, "position": {...}, "orientation": {...}, "confidence": float}
        """
        if not self.available:
            return None
        return await self._client.call_tool("estimate_object_pose", {"label": label})

    async def is_workspace_clear(self) -> dict[str, Any] | None:
        """
        Check for obstacles. Returns clearance dict or None if unavailable.

        Result when available: {"clear": bool, "obstacles": [...]}
        """
        if not self.available:
            return None
        return await self._client.call_tool("is_workspace_clear", {})

    async def get_scene_objects(self) -> dict[str, Any] | None:
        """
        Get all detected objects. Returns scene dict or None if unavailable.

        Result when available: {"objects": [{"label": str, "position": {...}, "confidence": float}]}
        """
        if not self.available:
            return None
        return await self._client.call_tool("get_scene_objects", {})

    def __repr__(self) -> str:
        return f"PerceptionClient(available={self.available})"
