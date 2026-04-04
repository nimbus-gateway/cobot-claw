#!/usr/bin/env python3
"""
Phase 2 smoke check — MCP client wrapper and perception contract.

Usage:
  python scripts/check_phase2.py              # offline checks only
  python scripts/check_phase2.py --live       # also test live MCP servers

Live checks require:
  python tools/ure3/ur3e_arm_mcp.py           # port 8765
  python tools/gripper/server.py              # port 8001
"""

import asyncio
import sys

PASS = "\033[92mPASS\033[0m"
FAIL = "\033[91mFAIL\033[0m"
SKIP = "\033[93mSKIP\033[0m"
HEAD = "\033[1m"
RESET = "\033[0m"

results = []
live_mode = "--live" in sys.argv


def check(label: str, skip_reason: str | None = None):
    def decorator(fn):
        if skip_reason:
            results.append(("skip", label))
            print(f"  {SKIP}  {label}")
            print(f"       ({skip_reason})")
            return fn
        try:
            if asyncio.iscoroutinefunction(fn):
                asyncio.run(fn())
            else:
                fn()
            results.append((True, label))
            print(f"  {PASS}  {label}")
        except Exception as e:
            results.append((False, label))
            print(f"  {FAIL}  {label}")
            print(f"       {e}")
        return fn
    return decorator


def server_reachable(url: str) -> bool:
    import urllib.request
    try:
        urllib.request.urlopen(url.replace("/mcp", "/health"), timeout=1)
        return True
    except Exception:
        return False


# ---------------------------------------------------------------------------
print(f"\n{HEAD}Phase 2 - MCP Layer{RESET}\n")
# ---------------------------------------------------------------------------

print("Imports")

@check("MCPClient imports without error")
def _():
    from cobot.mcp.client import MCPClient, MCPClientError

@check("PerceptionClient imports without error")
def _():
    from cobot.mcp.perception_contract import PerceptionClient

# ---------------------------------------------------------------------------
print("\nMCPClient — offline")

@check("MCPClient initialises with URL")
def _():
    from cobot.mcp.client import MCPClient
    c = MCPClient("http://localhost:8765/mcp")
    assert c.url == "http://localhost:8765/mcp"
    assert c.connected is False
    print(f"       -> {c}", end="")

@check("connect() to unreachable server raises MCPClientError")
async def _():
    from cobot.mcp.client import MCPClient, MCPClientError
    c = MCPClient("http://localhost:19999/mcp")
    try:
        await c.connect()
        raise AssertionError("should have raised")
    except MCPClientError:
        pass
    assert c.connected is False

@check("call_tool() before connect() raises MCPClientError")
async def _():
    from cobot.mcp.client import MCPClient, MCPClientError
    c = MCPClient("http://localhost:8765/mcp")
    try:
        await c.call_tool("get_robot_status", {})
        raise AssertionError("should have raised")
    except MCPClientError as e:
        assert "Not connected" in str(e)

@check("to_openai_tools() converts tool schema correctly")
def _():
    from unittest.mock import MagicMock
    from cobot.mcp.client import MCPClient
    c = MCPClient("http://localhost:8765/mcp")
    tool = MagicMock()
    tool.name = "move_joints"
    tool.description = "Move joints"
    tool.inputSchema = {"type": "object", "properties": {"joints": {"type": "array"}}}
    result = c.to_openai_tools([tool])
    assert result[0]["type"] == "function"
    assert result[0]["function"]["name"] == "move_joints"
    print(f"       -> {len(result)} tool(s) converted", end="")

# ---------------------------------------------------------------------------
print("\nPerceptionClient — offline")

@check("disabled by default (no client, enabled=False)")
async def _():
    from cobot.mcp.perception_contract import PerceptionClient
    p = PerceptionClient()
    assert p.available is False
    assert await p.locate_object("box") is None
    assert await p.estimate_object_pose("box") is None
    assert await p.is_workspace_clear() is None
    assert await p.get_scene_objects() is None
    print(f"       -> {p}", end="")

@check("disabled if client provided but enabled=False")
async def _():
    from unittest.mock import MagicMock
    from cobot.mcp.perception_contract import PerceptionClient
    p = PerceptionClient(mcp_client=MagicMock(), enabled=False)
    assert p.available is False
    assert await p.locate_object("box") is None

@check("disabled if enabled=True but no client")
async def _():
    from cobot.mcp.perception_contract import PerceptionClient
    p = PerceptionClient(mcp_client=None, enabled=True)
    assert p.available is False

@check("enabled when client + enabled=True, delegates calls")
async def _():
    from unittest.mock import MagicMock
    from cobot.mcp.perception_contract import PerceptionClient

    mock = MagicMock()
    async def call_tool(name, args):
        if name == "locate_object":
            return {"found": True, "x": 0.3, "y": 0.1, "z": 0.15, "confidence": 0.9}
        return {}
    mock.call_tool = call_tool

    p = PerceptionClient(mcp_client=mock, enabled=True)
    assert p.available is True
    result = await p.locate_object("red_box")
    assert result["found"] is True
    assert result["x"] == 0.3
    print(f"       -> located red_box at x={result['x']}", end="")

# ---------------------------------------------------------------------------
print("\nMCPClient — live server tests")
arm_url = "http://localhost:8765/mcp"
gripper_url = "http://localhost:8001/mcp"
arm_live = live_mode and server_reachable(arm_url)
gripper_live = live_mode and server_reachable(gripper_url)

arm_skip = None if arm_live else (
    "pass --live and start tools/ure3/ur3e_arm_mcp.py" if live_mode
    else "run with --live to test against live server"
)
gripper_skip = None if gripper_live else (
    "pass --live and start tools/gripper/server.py" if live_mode
    else "run with --live to test against live server"
)

@check("arm MCP: connect and list tools", skip_reason=arm_skip)
async def _():
    from cobot.mcp.client import MCPClient
    async with MCPClient(arm_url) as c:
        tools = await c.list_tools()
        names = [t.name for t in tools]
        assert "get_robot_status" in names
        assert "move_joints" in names
        print(f"       -> {len(tools)} tools: {names[:4]}...", end="")

@check("arm MCP: call get_robot_status", skip_reason=arm_skip)
async def _():
    from cobot.mcp.client import MCPClient
    async with MCPClient(arm_url) as c:
        result = await c.call_tool("get_robot_status", {})
        assert isinstance(result, dict)
        assert "ros_ready" in result
        print(f"       -> ros_ready={result['ros_ready']}  has_joint_state={result.get('has_joint_state')}", end="")

@check("gripper MCP: connect and list tools", skip_reason=gripper_skip)
async def _():
    from cobot.mcp.client import MCPClient
    async with MCPClient(gripper_url) as c:
        tools = await c.list_tools()
        names = [t.name for t in tools]
        assert "open" in names
        assert "close" in names
        assert "get_status" in names
        print(f"       -> {len(tools)} tools: {names}", end="")

@check("gripper MCP: call get_status", skip_reason=gripper_skip)
async def _():
    from cobot.mcp.client import MCPClient
    async with MCPClient(gripper_url) as c:
        result = await c.call_tool("get_status", {})
        assert isinstance(result, dict)
        print(f"       -> {result}", end="")

# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------
passed = sum(1 for ok, _ in results if ok is True)
failed = sum(1 for ok, _ in results if ok is False)
skipped = sum(1 for ok, _ in results if ok == "skip")

print(f"\n{'-' * 50}")
print(f"  {HEAD}{passed} passed  {failed} failed  {skipped} skipped{RESET}  ({len(results)} checks)")

if failed:
    print("\n  Failed checks:")
    for ok, label in results:
        if ok is False:
            print(f"    {FAIL}  {label}")
    print()
    sys.exit(1)
else:
    print(f"\n  {PASS}  Phase 2 MCP layer is working.\n")
