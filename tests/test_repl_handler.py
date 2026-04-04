"""
Tests for cobot/cli/repl.py — _handle_agent_message node iteration.

These tests guard against the class of bug where pydantic-ai node types are
misidentified, causing AttributeError (e.g. calling args_as_dict() on a
ToolReturnPart) or silently swapping tool-call and tool-result display.
"""

from __future__ import annotations

import json
from contextlib import asynccontextmanager
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from cobot.cli.repl import _handle_agent_message, _make_offline_deps
from tests.test_cli import _make_session


# ---------------------------------------------------------------------------
# Fake pydantic-ai node types
# ---------------------------------------------------------------------------

class _ToolCallPart:
    """Mimics pydantic_ai.messages.ToolCallPart — lives in model_response."""
    def __init__(self, tool_name: str, args: dict):
        self.tool_name = tool_name
        self._args = args

    def args_as_dict(self) -> dict:
        return self._args


class _TextPart:
    """Mimics pydantic_ai.messages.TextPart — lives in model_response."""
    def __init__(self, content: str):
        self.content = content


class _ToolReturnPart:
    """
    Mimics pydantic_ai.messages.ToolReturnPart — lives in the *request*
    sent back to the model after a tool call.  It has tool_name and content
    but NO args_as_dict — calling args_as_dict() on it is the bug we guard.
    """
    def __init__(self, tool_name: str, content: str):
        self.tool_name = tool_name
        self.content = content
        # Deliberately no args_as_dict method


class _UserPromptPart:
    """Mimics pydantic_ai.messages.UserPromptPart — also in request parts."""
    def __init__(self, content: str):
        self.content = content


class _ModelResponseNode:
    def __init__(self, parts):
        self.model_response = MagicMock()
        self.model_response.parts = parts


class _ModelRequestNode:
    def __init__(self, parts):
        self.request = MagicMock()
        self.request.parts = parts


# ---------------------------------------------------------------------------
# Mock agent run
# ---------------------------------------------------------------------------

def _mock_agent(nodes: list, output: str = "Done."):
    """Return a mock agent whose iter() yields the given nodes."""
    result = MagicMock()
    result.output = output

    @asynccontextmanager
    async def _iter(*args, **kwargs):
        class _Run:
            def __aiter__(self_inner):
                return aiter(nodes)
            result = MagicMock()
        run = _Run()
        run.result = result
        yield run

    agent = MagicMock()
    agent.iter = _iter
    return agent


async def aiter(items):
    for item in items:
        yield item


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

def _run_handler(nodes, output="Done.", connected=False):
    session = _make_session(connected=connected)

    agent = _mock_agent(nodes, output)
    with patch("cobot.agents.orchestrator.build_orchestrator", return_value=agent), \
         patch("cobot.cli.repl.print_tool_call") as mock_call, \
         patch("cobot.cli.repl.print_tool_result") as mock_result, \
         patch("cobot.cli.repl.print_agent_response") as mock_response, \
         patch("cobot.cli.repl.print_error") as mock_error, \
         patch("cobot.cli.repl.console"):
        import asyncio
        asyncio.get_event_loop().run_until_complete(
            _handle_agent_message("test input", session)
        )
    return mock_call, mock_result, mock_response, mock_error


# ---------------------------------------------------------------------------
# Tool call display
# ---------------------------------------------------------------------------

class TestToolCallDisplay:
    def test_tool_call_part_triggers_print_tool_call(self):
        """ToolCallPart in model_response → print_tool_call with name + args."""
        nodes = [
            _ModelResponseNode([_ToolCallPart("get_robot_status", {})]),
        ]
        mock_call, _, _, mock_error = _run_handler(nodes)
        mock_call.assert_called_once_with("get_robot_status", {})
        mock_error.assert_not_called()

    def test_tool_call_args_passed_correctly(self):
        nodes = [
            _ModelResponseNode([_ToolCallPart("move_named_pose", {"pose": "home"})]),
        ]
        mock_call, _, _, _ = _run_handler(nodes)
        mock_call.assert_called_once_with("move_named_pose", {"pose": "home"})

    def test_text_part_does_not_trigger_tool_call(self):
        """TextPart in model_response is not mistaken for a tool call."""
        nodes = [
            _ModelResponseNode([_TextPart("Hello there!")]),
        ]
        mock_call, _, _, _ = _run_handler(nodes)
        mock_call.assert_not_called()

    def test_multiple_tool_calls_in_one_response(self):
        nodes = [
            _ModelResponseNode([
                _ToolCallPart("list_tool_mcps", {}),
                _ToolCallPart("call_tool_action", {"tool": "gripper", "action": "open"}),
            ]),
        ]
        mock_call, _, _, _ = _run_handler(nodes)
        assert mock_call.call_count == 2


# ---------------------------------------------------------------------------
# Tool result display
# ---------------------------------------------------------------------------

class TestToolResultDisplay:
    def test_tool_return_part_triggers_print_tool_result(self):
        """ToolReturnPart in request → print_tool_result, no AttributeError."""
        nodes = [
            _ModelRequestNode([_ToolReturnPart("get_robot_status", '{"ok": true}')]),
        ]
        _, mock_result, _, mock_error = _run_handler(nodes)
        mock_result.assert_called_once()
        mock_error.assert_not_called()

    def test_tool_return_no_args_as_dict_does_not_crash(self):
        """
        Regression: the original bug called args_as_dict() on ToolReturnPart.
        Confirm no AttributeError is raised.
        """
        part = _ToolReturnPart("get_robot_status", '{"ok": true}')
        assert not hasattr(part, "args_as_dict"), "test setup: ToolReturnPart must not have args_as_dict"

        nodes = [_ModelRequestNode([part])]
        _, mock_result, _, mock_error = _run_handler(nodes)
        # Should display result, not an error about missing attribute
        mock_result.assert_called_once()
        mock_error.assert_not_called()

    def test_ok_true_result_passes_true_to_print_tool_result(self):
        content = json.dumps({"ok": True, "robot_ip": "192.168.1.2"})
        nodes = [_ModelRequestNode([_ToolReturnPart("get_robot_status", content)])]
        _, mock_result, _, _ = _run_handler(nodes)
        ok_arg = mock_result.call_args[0][0]
        assert ok_arg is True

    def test_ok_false_result_passes_false_to_print_tool_result(self):
        content = json.dumps({"ok": False, "message": "arm error"})
        nodes = [_ModelRequestNode([_ToolReturnPart("move_named_pose", content)])]
        _, mock_result, _, _ = _run_handler(nodes)
        ok_arg = mock_result.call_args[0][0]
        assert ok_arg is False

    def test_non_json_result_defaults_ok_to_true(self):
        nodes = [_ModelRequestNode([_ToolReturnPart("some_tool", "plain text result")])]
        _, mock_result, _, _ = _run_handler(nodes)
        ok_arg = mock_result.call_args[0][0]
        assert ok_arg is True

    def test_user_prompt_part_in_request_is_ignored(self):
        """UserPromptPart in request.parts must not be treated as a tool result."""
        nodes = [_ModelRequestNode([_UserPromptPart("move to home")])]
        _, mock_result, _, _ = _run_handler(nodes)
        mock_result.assert_not_called()


# ---------------------------------------------------------------------------
# Full round-trip: tool call → tool result
# ---------------------------------------------------------------------------

class TestRoundTrip:
    def test_call_then_result_both_displayed(self):
        """A typical agent turn: model_response with ToolCallPart, then request with ToolReturnPart."""
        nodes = [
            _ModelResponseNode([_ToolCallPart("get_robot_status", {})]),
            _ModelRequestNode([_ToolReturnPart("get_robot_status", '{"ok": true, "robot_ip": "x"}')]),
        ]
        mock_call, mock_result, _, mock_error = _run_handler(nodes, output="The robot is connected.")
        mock_call.assert_called_once_with("get_robot_status", {})
        mock_result.assert_called_once()
        mock_error.assert_not_called()

    def test_agent_final_response_displayed(self):
        nodes = [
            _ModelResponseNode([_TextPart("Here is the status.")]),
        ]
        _, _, mock_response, _ = _run_handler(nodes, output="Here is the status.")
        mock_response.assert_called_once_with("Here is the status.")

    def test_exception_in_agent_shows_error(self):
        """If the agent raises, print_error is called instead of crashing."""
        session = _make_session(connected=False)

        @asynccontextmanager
        async def _bad_iter(*args, **kwargs):
            raise RuntimeError("LLM call failed")
            yield  # make it a generator

        agent = MagicMock()
        agent.iter = _bad_iter

        with patch("cobot.agents.orchestrator.build_orchestrator", return_value=agent), \
             patch("cobot.cli.repl.print_error") as mock_error, \
             patch("cobot.cli.repl.console"):
            import asyncio
            asyncio.get_event_loop().run_until_complete(
                _handle_agent_message("test", session)
            )
        mock_error.assert_called_once()
        assert "LLM call failed" in mock_error.call_args[0][0]


# ---------------------------------------------------------------------------
# Offline / disconnected mode
# ---------------------------------------------------------------------------

class TestOfflineMode:
    def test_offline_deps_created_when_session_deps_none(self):
        """When session.deps is None the handler builds offline stub deps, not crash."""
        session = _make_session(connected=False)
        assert session.deps is None

        nodes = []
        agent = _mock_agent(nodes, output="I'm offline.")
        called_with_deps = []

        @asynccontextmanager
        async def _capturing_iter(*args, **kwargs):
            called_with_deps.append(kwargs.get("deps") or (args[1] if len(args) > 1 else None))
            class _Run:
                async def __aiter__(self_inner):
                    return
                    yield
                result = MagicMock()
                result.output = "offline response"
            yield _Run()

        capture_agent = MagicMock()
        capture_agent.iter = _capturing_iter

        with patch("cobot.agents.orchestrator.build_orchestrator", return_value=capture_agent), \
             patch("cobot.cli.repl.print_agent_response"), \
             patch("cobot.cli.repl.console"):
            import asyncio
            asyncio.get_event_loop().run_until_complete(
                _handle_agent_message("hi", session)
            )
        # iter was called (agent ran) even though session.deps was None
        assert len(called_with_deps) == 1

    def test_make_offline_deps_arm_returns_offline_message(self):
        session = _make_session(connected=False)
        deps = _make_offline_deps(session)
        import asyncio
        result = asyncio.get_event_loop().run_until_complete(
            deps.scenario_deps.arm_mcp.call_tool("move_named_pose", {"pose": "home"})
        )
        assert result["ok"] is False
        assert "offline" in result["message"].lower()

    def test_make_offline_deps_list_tools_raises(self):
        session = _make_session(connected=False)
        deps = _make_offline_deps(session)
        import asyncio
        with pytest.raises((ConnectionError, Exception)):
            asyncio.get_event_loop().run_until_complete(
                deps.scenario_deps.arm_mcp.list_tools()
            )
