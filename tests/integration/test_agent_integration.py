"""Integration tests: AgentLoop with real StatusCommentManager and tool wiring.

These tests use the real AgentLoop, StatusCommentManager, and formatter code.
Only the LLM client and container exec are mocked. This verifies:
- Tool-calling loop drives multiple rounds correctly
- Status comments are posted and updated in real-time via API tracker
- Stuck detection, blocked commands, and fallback all work end-to-end
- Extra tools (smart_search) integrate with the agent loop
- Todo markers flow from exec output → status comment
"""

from __future__ import annotations

import json
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest

from forge_bot.agent import AgentLoop
from forge_bot.config import Settings
from forge_bot.status.formatter import TodoItem
from forge_bot.status.manager import StatusCommentManager
from forge_bot.tools.base import BaseTool, ToolParameter, ToolResult

from .conftest import (
    FakeApiTracker,
    FakeExecResult,
    make_llm_text_response,
    make_llm_tool_response,
    make_mock_container,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


class FakeSearchTool(BaseTool):
    """Fake smart_search tool for integration tests."""

    name = "smart_search"
    description = "Search the codebase."
    parameters = [
        ToolParameter(name="query", type="string", description="Search query"),
    ]

    def __init__(self, answer: str = "Found relevant code in auth.py.") -> None:
        self._answer = answer

    async def execute(self, **kwargs: object) -> ToolResult:
        return ToolResult(tool_name=self.name, success=True, content=self._answer)


def _make_real_status(api_tracker: FakeApiTracker) -> StatusCommentManager:
    """Build a real StatusCommentManager backed by the FakeApiTracker."""
    # We need a thin wrapper around GenericForgeClient.call
    api = MagicMock()
    api.call = AsyncMock(side_effect=api_tracker.call)
    return StatusCommentManager(api, "org", "myapp", 7)


def _make_agent(
    llm_responses: list[Any],
    container: MagicMock,
    settings: Settings,
    status: StatusCommentManager | None = None,
    extra_tools: list[BaseTool] | None = None,
) -> AgentLoop:
    """Build an AgentLoop with mocked LLM but real everything else."""
    llm = AsyncMock()
    llm.chat_with_tools = AsyncMock(side_effect=llm_responses)
    llm.chat = AsyncMock(return_value="Fallback answer.")

    agent = AgentLoop(
        llm,
        container,
        settings,
        status=status,
        extra_tools=extra_tools,
    )
    # Bypass container.exec for notes reading — keep exec call counts clean.
    agent._read_notes = AsyncMock(return_value="")
    return agent


# ---------------------------------------------------------------------------
# Tests: agent + real status manager
# ---------------------------------------------------------------------------


class TestAgentWithStatusManager:
    """Agent loop updates real StatusCommentManager via API tracker."""

    async def test_single_tool_call_records_status(
        self,
        integration_settings: Settings,
        api_tracker: FakeApiTracker,
    ) -> None:
        """One exec round → status gets phase update + tool call record."""
        status = _make_real_status(api_tracker)
        await status.post_initial_status()

        container = make_mock_container({
            "ls": FakeExecResult(stdout="README.md\nsrc/\ntests/"),
        })

        agent = _make_agent(
            llm_responses=[
                make_llm_tool_response(["ls /workspace"]),
                make_llm_text_response("The repo has 3 top-level entries."),
            ],
            container=container,
            settings=integration_settings,
            status=status,
        )

        reply = await agent.run("You are a bot.", "What files are in this repo?")

        assert reply == "The repo has 3 top-level entries."

        # Verify API calls: initial post + at least one edit (phase update)
        post_calls = api_tracker.calls_for("post_issue_comment")
        edit_calls = api_tracker.calls_for("edit_issue_comment")
        assert len(post_calls) >= 1  # initial status
        assert len(edit_calls) >= 1  # at least one status update

        # The status should have recorded one tool call
        assert len(status.tool_calls) == 1
        assert status.tool_calls[0].tool_name == "execute"
        assert status.tool_calls[0].success is True

    async def test_multi_round_tool_calls(
        self,
        integration_settings: Settings,
        api_tracker: FakeApiTracker,
    ) -> None:
        """Multiple exec rounds accumulate tool call records in status."""
        status = _make_real_status(api_tracker)
        await status.post_initial_status()

        container = make_mock_container({
            "grep": FakeExecResult(stdout="auth.py:10: def login()"),
            "cat": FakeExecResult(stdout="def login():\n    return jwt.encode(...)"),
        })

        agent = _make_agent(
            llm_responses=[
                make_llm_tool_response(["grep -rn 'login' ."]),
                make_llm_tool_response(["cat auth.py"]),
                make_llm_text_response("The login function uses JWT."),
            ],
            container=container,
            settings=integration_settings,
            status=status,
        )

        reply = await agent.run("You are a bot.", "How does login work?")

        assert reply == "The login function uses JWT."
        assert len(status.tool_calls) == 2
        assert status.tool_calls[0].arguments_summary  # not empty
        assert status.tool_calls[1].arguments_summary

    async def test_blocked_command_recorded_without_exec(
        self,
        integration_settings: Settings,
        api_tracker: FakeApiTracker,
    ) -> None:
        """Blocked commands don't execute but the loop continues."""
        status = _make_real_status(api_tracker)
        await status.post_initial_status()

        container = make_mock_container({})

        agent = _make_agent(
            llm_responses=[
                make_llm_tool_response(["rm -rf /"]),
                make_llm_text_response("I won't delete anything."),
            ],
            container=container,
            settings=integration_settings,
            status=status,
        )

        reply = await agent.run("You are a bot.", "Delete everything")

        assert "won't" in reply.lower() or "delete" in reply.lower()
        container.exec.assert_not_awaited()

    async def test_llm_failure_fallback(
        self,
        integration_settings: Settings,
        api_tracker: FakeApiTracker,
    ) -> None:
        """When LLM tool-calling fails, falls back to simple chat."""
        status = _make_real_status(api_tracker)
        await status.post_initial_status()

        container = make_mock_container({})

        llm = AsyncMock()
        llm.chat_with_tools = AsyncMock(side_effect=Exception("LLM timeout"))
        llm.chat = AsyncMock(return_value="I can still help without tools.")

        agent = AgentLoop(
            llm, container, integration_settings, status=status
        )
        agent._read_notes = AsyncMock(return_value="")

        reply = await agent.run("System prompt", "Help me")
        assert reply == "I can still help without tools."
        llm.chat.assert_awaited_once()


class TestAgentWithExtraTools:
    """Agent loop dispatches to extra tools (smart_search)."""

    async def test_smart_search_dispatched(
        self,
        integration_settings: Settings,
        api_tracker: FakeApiTracker,
    ) -> None:
        """Extra tool calls are dispatched and recorded in status."""
        status = _make_real_status(api_tracker)
        await status.post_initial_status()

        container = make_mock_container({})
        search_tool = FakeSearchTool(answer="auth.py implements JWT login.")

        # LLM calls smart_search, then responds
        from types import SimpleNamespace

        search_tc = SimpleNamespace(
            id="call_search_0",
            function=SimpleNamespace(
                name="smart_search",
                arguments=json.dumps({"query": "auth module"}),
            ),
        )
        search_msg = SimpleNamespace(content=None, tool_calls=[search_tc])
        search_msg.model_dump = lambda: {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {
                    "id": "call_search_0",
                    "type": "function",
                    "function": {
                        "name": "smart_search",
                        "arguments": json.dumps({"query": "auth module"}),
                    },
                }
            ],
        }
        search_response = SimpleNamespace(choices=[SimpleNamespace(message=search_msg)])

        agent = _make_agent(
            llm_responses=[
                search_response,
                make_llm_text_response("Auth uses JWT per auth.py."),
            ],
            container=container,
            settings=integration_settings,
            status=status,
            extra_tools=[search_tool],
        )

        reply = await agent.run("System prompt", "Explain auth")

        assert "JWT" in reply
        assert len(status.tool_calls) == 1
        assert status.tool_calls[0].tool_name == "smart_search"
        assert status.tool_calls[0].success is True


class TestAgentTodoMarkers:
    """FORGE_TODO markers in exec output update the status todo list."""

    async def test_todo_set_from_exec_output(
        self,
        integration_settings: Settings,
        api_tracker: FakeApiTracker,
    ) -> None:
        """FORGE_TODO:set markers create the todo list."""
        status = _make_real_status(api_tracker)
        await status.post_initial_status()

        container = make_mock_container({
            "echo": FakeExecResult(stdout='FORGE_TODO:set:["Clone repo","Run tests","Analyze"]'),
        })

        agent = _make_agent(
            llm_responses=[
                make_llm_tool_response(["echo 'FORGE_TODO:set:[\"Clone repo\",\"Run tests\",\"Analyze\"]'"]),
                make_llm_text_response("All done."),
            ],
            container=container,
            settings=integration_settings,
            status=status,
        )

        await agent.run("System prompt", "Set up todos")

        assert len(status.todos) == 3
        assert status.todos[0].text == "Clone repo"
        assert status.todos[1].text == "Run tests"
        assert status.todos[2].text == "Analyze"

    async def test_todo_check_marks_done(
        self,
        integration_settings: Settings,
        api_tracker: FakeApiTracker,
    ) -> None:
        """FORGE_TODO:check markers mark items as done."""
        status = _make_real_status(api_tracker)
        await status.post_initial_status()
        # Pre-populate todos
        await status.update_todos([
            TodoItem("Clone repo"),
            TodoItem("Run tests"),
        ])

        container = make_mock_container({
            "echo": FakeExecResult(stdout="FORGE_TODO:check:Clone repo"),
        })

        agent = _make_agent(
            llm_responses=[
                make_llm_tool_response(["echo 'FORGE_TODO:check:Clone repo'"]),
                make_llm_text_response("Cloned."),
            ],
            container=container,
            settings=integration_settings,
            status=status,
        )

        await agent.run("System prompt", "Clone the repo")

        todos = status.todos
        done_items = [t for t in todos if t.done]
        assert len(done_items) >= 1
        assert any(t.text == "Clone repo" for t in done_items)


class TestAgentStuckDetection:
    """Stuck detection breaks the loop and forces a final response."""

    async def test_stuck_agent_produces_final_answer(
        self,
        integration_settings: Settings,
    ) -> None:
        """After repeated stale rounds, agent forces a final response."""
        container = make_mock_container({})

        # The loop will call chat_with_tools for:
        #   rounds 0..N (tool calls) + 1 call from _force_final (no tools).
        # Use a side_effect function so _force_final gets a text response.
        tool_resp = make_llm_tool_response(["echo hi"])
        final_resp = make_llm_text_response("Forced final answer.")
        call_count = 0

        def smart_side_effect(**kwargs):
            nonlocal call_count
            call_count += 1
            # When tools=[] it's the _force_final call
            if not kwargs.get("tools"):
                return final_resp
            return tool_resp

        llm = AsyncMock()
        llm.chat_with_tools = AsyncMock(side_effect=smart_side_effect)
        llm.chat = AsyncMock(return_value="Fallback")

        agent = AgentLoop(llm, container, integration_settings)
        agent._read_notes = AsyncMock(return_value="")

        reply = await agent.run("System prompt", "stuck test")

        # Should have terminated early and produced an answer
        assert isinstance(reply, str)
        assert len(reply) > 0
        assert "Forced final answer" in reply
        # Should not have used all 10 rounds (loop + force_final)
        assert llm.chat_with_tools.await_count < 10
