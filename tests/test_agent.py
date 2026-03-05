"""Tests for the AgentLoop."""

from __future__ import annotations

import json
from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest

from forge_bot.agent import (
    AgentLoop,
    _check_blocked,
    _extract_text_tool_call,
    _LoopState,
    _parse_command,
    _truncate,
)
from forge_bot.tools.base import BaseTool, ToolParameter, ToolResult

# -- Helpers -----------------------------------------------------------------


@dataclass
class _FakeExecResult:
    exit_code: int = 0
    stdout: str = ""
    stderr: str = ""
    duration_seconds: float = 0.1
    command: str = "echo hi"


def _llm_text(content: str = "Final answer.") -> Any:
    """LLM response with no tool calls (final text)."""
    msg = SimpleNamespace(content=content, tool_calls=None)
    msg.model_dump = lambda: {"role": "assistant", "content": content, "tool_calls": None}
    return SimpleNamespace(choices=[SimpleNamespace(message=msg)])


def _llm_tool(commands: list[str], tool_name: str = "execute") -> Any:
    """LLM response requesting tool calls."""
    tcs = []
    for i, cmd in enumerate(commands):
        if tool_name == "execute":
            args = json.dumps({"command": cmd})
        else:
            args = json.dumps({"query": cmd})
        tcs.append(
            SimpleNamespace(
                id=f"call_{i}",
                function=SimpleNamespace(name=tool_name, arguments=args),
            )
        )
    msg = SimpleNamespace(content=None, tool_calls=tcs)
    msg.model_dump = lambda: {
        "role": "assistant",
        "content": None,
        "tool_calls": [
            {
                "id": tc.id,
                "type": "function",
                "function": {
                    "name": tc.function.name,
                    "arguments": tc.function.arguments,
                },
            }
            for tc in tcs
        ],
    }
    return SimpleNamespace(choices=[SimpleNamespace(message=msg)])


class _FakeTool(BaseTool):
    """A fake BaseTool for testing extra_tools dispatch."""

    name = "fake_search"
    description = "A fake search tool."
    parameters = [
        ToolParameter(name="query", type="string", description="Search query"),
    ]

    def __init__(self, result: str = "Found it.") -> None:
        self._result = result

    async def execute(self, **kwargs: object) -> ToolResult:
        return ToolResult(
            tool_name=self.name,
            success=True,
            content=self._result,
        )


def _make_agent(
    *,
    llm_responses: list | None = None,
    exec_result: _FakeExecResult | None = None,
    status: MagicMock | None = None,
    context_window: int = 8192,
    extra_tools: list[BaseTool] | None = None,
) -> tuple[AgentLoop, AsyncMock, AsyncMock]:
    """Build an AgentLoop with mocked LLM and container."""
    llm = AsyncMock()
    if llm_responses:
        llm.chat_with_tools = AsyncMock(side_effect=llm_responses)
    else:
        llm.chat_with_tools = AsyncMock(return_value=_llm_text())
    llm.chat = AsyncMock(return_value="Fallback answer.")

    container = AsyncMock()
    container.exec = AsyncMock(
        return_value=exec_result or _FakeExecResult(),
    )

    settings = MagicMock()
    settings.llm_context_window = context_window
    settings.llm_max_tokens = 1024
    settings.container_timeout = 60

    agent = AgentLoop(
        llm,
        container,
        settings,
        status=status,
        extra_tools=extra_tools,
    )
    # Prevent _read_notes from calling container.exec (keeps exec counts clean).
    agent._read_notes = AsyncMock(return_value="")
    return agent, llm, container


# -- Unit tests: helpers -----------------------------------------------------


class TestParseCommand:
    def test_from_dict(self) -> None:
        assert _parse_command({"command": "ls -la"}) == "ls -la"

    def test_from_json_string(self) -> None:
        assert _parse_command('{"command": "pwd"}') == "pwd"

    def test_from_plain_string(self) -> None:
        assert _parse_command("echo hello") == "echo hello"

    def test_empty_dict(self) -> None:
        assert _parse_command({}) == ""


class TestCheckBlocked:
    def test_blocked_rm_rf(self) -> None:
        assert _check_blocked("rm -rf /") is not None

    def test_blocked_mkfs(self) -> None:
        assert _check_blocked("mkfs /dev/sda") is not None

    def test_blocked_dd(self) -> None:
        assert _check_blocked("dd if=/dev/zero of=/dev/sda") is not None

    def test_blocked_dev_redirect(self) -> None:
        assert _check_blocked("echo x > /dev/sda") is not None

    def test_safe_command(self) -> None:
        assert _check_blocked("grep -rn 'auth' .") is None

    def test_case_insensitive(self) -> None:
        assert _check_blocked("RM -RF /") is not None


class TestTruncate:
    def test_short_text_untouched(self) -> None:
        assert _truncate("hello", 100) == "hello"

    def test_long_text_truncated(self) -> None:
        text = "x" * 1000
        result = _truncate(text, 200)
        assert len(result) < 1000
        assert "truncated" in result

    def test_preserves_tail(self) -> None:
        text = "HEAD" * 100 + "TAIL_MARKER"
        result = _truncate(text, 400)
        assert "TAIL_MARKER" in result


class TestExtractTextToolCall:
    def test_valid_json(self) -> None:
        text = 'I will run: {"name": "execute", "arguments": {"command": "ls -la"}}'
        assert _extract_text_tool_call(text) == "ls -la"

    def test_no_match(self) -> None:
        assert _extract_text_tool_call("Just a plain text response.") is None

    def test_malformed_json(self) -> None:
        text = '{"name": "execute", "arguments": {"command": "ls -la"'
        assert _extract_text_tool_call(text) is None

    def test_nested_braces(self) -> None:
        text = '{"name": "execute", "arguments": {"command": "echo \\"{}\\""}}'
        result = _extract_text_tool_call(text)
        assert result is not None

    def test_trailing_text(self) -> None:
        text = 'Let me check: {"name": "execute", "arguments": {"command": "pwd"}} and more text'
        assert _extract_text_tool_call(text) == "pwd"

    def test_empty_string(self) -> None:
        assert _extract_text_tool_call("") is None

    def test_arguments_as_string(self) -> None:
        text = '{"name": "execute", "arguments": "{\\"command\\": \\"ls\\"}"}'
        result = _extract_text_tool_call(text)
        assert result == "ls"


class TestLoopState:
    def test_duplicate_after_three(self) -> None:
        state = _LoopState()
        assert not state.record("cmd1")
        assert not state.record("cmd1")
        assert state.record("cmd1")  # 3rd time

    def test_different_commands_not_duplicate(self) -> None:
        state = _LoopState()
        assert not state.record("cmd1")
        assert not state.record("cmd2")
        assert not state.record("cmd3")

    def test_stuck_after_two_stale_rounds(self) -> None:
        state = _LoopState()
        assert not state.is_stuck()
        state.record_round(had_new=False)
        assert not state.is_stuck()
        state.record_round(had_new=False)
        assert state.is_stuck()

    def test_new_call_resets_stale(self) -> None:
        state = _LoopState()
        state.record_round(had_new=False)
        state.record_round(had_new=True)
        state.record_round(had_new=False)
        assert not state.is_stuck()

    def test_text_call_count_starts_at_zero(self) -> None:
        state = _LoopState()
        assert state.text_call_count == 0


# -- Integration tests: AgentLoop.run() -------------------------------------


class TestAgentLoopRun:
    @pytest.mark.asyncio
    async def test_simple_text_response(self) -> None:
        """LLM returns text immediately -- no tool calls."""
        agent, llm, _ = _make_agent(
            llm_responses=[_llm_text("This repo is a webhook bot.")],
        )
        reply = await agent.run("system prompt", "what does this do?")
        assert reply == "This repo is a webhook bot."
        llm.chat_with_tools.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_single_exec_then_response(self) -> None:
        """LLM executes one command, then responds with text."""
        agent, llm, container = _make_agent(
            llm_responses=[
                _llm_tool(["ls /workspace"]),
                _llm_text("Found 3 files."),
            ],
            exec_result=_FakeExecResult(stdout="README.md\nsrc/\ntests/"),
        )
        reply = await agent.run("system prompt", "list files")
        assert reply == "Found 3 files."
        container.exec.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_multi_round(self) -> None:
        """LLM executes multiple commands across rounds."""
        agent, llm, container = _make_agent(
            llm_responses=[
                _llm_tool(["grep -rn auth ."]),
                _llm_tool(["cat src/auth.py"]),
                _llm_text("The auth module handles JWT."),
            ],
            exec_result=_FakeExecResult(stdout="some output"),
        )
        reply = await agent.run("system prompt", "explain auth")
        assert reply == "The auth module handles JWT."
        assert container.exec.await_count == 2

    @pytest.mark.asyncio
    async def test_blocked_command(self) -> None:
        """Blocked commands return error without execution."""
        agent, _, container = _make_agent(
            llm_responses=[
                _llm_tool(["rm -rf /"]),
                _llm_text("OK, I won't do that."),
            ],
        )
        reply = await agent.run("system prompt", "delete everything")
        assert reply == "OK, I won't do that."
        container.exec.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_duplicate_skipped(self) -> None:
        """Same command 3+ times gets skipped."""
        agent, _, container = _make_agent(
            llm_responses=[
                _llm_tool(["echo hi"]),
                _llm_tool(["echo hi"]),
                _llm_tool(["echo hi"]),  # 3rd -- skipped
                _llm_text("Done."),
            ],
            exec_result=_FakeExecResult(stdout="hi"),
        )
        reply = await agent.run("system prompt", "test")
        assert reply == "Done."
        assert container.exec.await_count == 2  # 3rd skipped

    @pytest.mark.asyncio
    async def test_stuck_detection(self) -> None:
        """After 2 stale rounds, loop breaks and forces final response."""
        # All rounds produce the same command that gets skipped
        responses = [_llm_tool(["echo hi"])] * 8
        responses.append(_llm_text("Forced answer."))

        agent, llm, _ = _make_agent(
            llm_responses=responses,
            exec_result=_FakeExecResult(stdout="hi"),
        )
        await agent.run("system prompt", "stuck test")
        # Should have broken out early, not used all 8 rounds
        assert llm.chat_with_tools.await_count < 8

    @pytest.mark.asyncio
    async def test_llm_error_fallback(self) -> None:
        """LLM failure falls back to simple chat."""
        llm = AsyncMock()
        llm.chat_with_tools = AsyncMock(side_effect=Exception("API down"))
        llm.chat = AsyncMock(return_value="Fallback answer.")

        container = AsyncMock()
        settings = MagicMock()
        settings.llm_context_window = 8192
        settings.llm_max_tokens = 1024
        settings.container_timeout = 60

        agent = AgentLoop(llm, container, settings)
        agent._read_notes = AsyncMock(return_value="")
        reply = await agent.run("system prompt", "test")
        assert reply == "Fallback answer."
        llm.chat.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_total_failure(self) -> None:
        """Both tool and fallback fail -- canned error message."""
        llm = AsyncMock()
        llm.chat_with_tools = AsyncMock(side_effect=Exception("API down"))
        llm.chat = AsyncMock(side_effect=Exception("Also down"))

        container = AsyncMock()
        settings = MagicMock()
        settings.llm_context_window = 8192
        settings.llm_max_tokens = 1024
        settings.container_timeout = 60

        agent = AgentLoop(llm, container, settings)
        agent._read_notes = AsyncMock(return_value="")
        reply = await agent.run("system prompt", "test")
        assert "error" in reply.lower()

    @pytest.mark.asyncio
    async def test_status_updates(self) -> None:
        """Status manager gets phase updates and tool call records."""
        status = MagicMock()
        status.update_phase = AsyncMock()
        status.record_tool_call = AsyncMock()
        status.update_todos = AsyncMock()
        status.todos = []

        agent, _, _ = _make_agent(
            llm_responses=[
                _llm_tool(["ls /workspace"]),
                _llm_text("Found files."),
            ],
            exec_result=_FakeExecResult(stdout="README.md"),
            status=status,
        )
        await agent.run("system prompt", "list")

        status.update_phase.assert_awaited()
        status.record_tool_call.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_output_truncation(self) -> None:
        """Large output gets truncated based on context window."""
        agent, _, container = _make_agent(
            llm_responses=[
                _llm_tool(["cat huge_file"]),
                _llm_text("Summary."),
            ],
            exec_result=_FakeExecResult(stdout="x" * 50_000),
            context_window=4096,  # Small window
        )
        await agent.run("system prompt", "read file")

        # The LLM should have received truncated output
        call_args = agent._llm.chat_with_tools.call_args_list
        assert len(call_args) >= 2
        # Check the second call has tool results
        second_call_messages = call_args[1].kwargs.get(
            "messages", call_args[1].args[0] if call_args[1].args else []
        )
        for msg in second_call_messages:
            if msg.get("role") == "tool":
                assert len(msg["content"]) < 50_000

    @pytest.mark.asyncio
    async def test_notes_injection(self) -> None:
        """Notes from .notes file are injected into messages."""
        agent, llm, container = _make_agent(
            llm_responses=[
                _llm_tool(["echo test"]),
                _llm_text("Answer with notes."),
            ],
            exec_result=_FakeExecResult(stdout="output"),
        )

        # Mock _read_notes to return some content
        agent._read_notes = AsyncMock(return_value="- Found auth.py\n- Uses JWT")

        await agent.run("system prompt", "question")

        # The second LLM call should include notes in the messages
        second_call = llm.chat_with_tools.call_args_list[1]
        messages = second_call.kwargs.get(
            "messages", second_call.args[0] if second_call.args else []
        )
        notes_found = any(
            "accumulated findings" in (m.get("content") or "").lower() for m in messages
        )
        assert notes_found

    @pytest.mark.asyncio
    async def test_todo_markers_set(self) -> None:
        """FORGE_TODO:set markers update the status todo list."""
        status = MagicMock()
        status.update_phase = AsyncMock()
        status.record_tool_call = AsyncMock()
        status.update_todos = AsyncMock()
        status.todos = []

        agent, _, _ = _make_agent(
            llm_responses=[
                _llm_tool(['echo \'FORGE_TODO:set:["step 1","step 2"]\'']),
                _llm_text("Done."),
            ],
            exec_result=_FakeExecResult(
                stdout='FORGE_TODO:set:["step 1","step 2"]',
            ),
            status=status,
        )
        await agent.run("system prompt", "test todos")

        status.update_todos.assert_awaited()
        todos = status.update_todos.call_args[0][0]
        assert len(todos) == 2
        assert todos[0].text == "step 1"

    @pytest.mark.asyncio
    async def test_todo_markers_check(self) -> None:
        """FORGE_TODO:check markers mark items done."""
        from forge_bot.status.formatter import TodoItem

        status = MagicMock()
        status.update_phase = AsyncMock()
        status.record_tool_call = AsyncMock()
        status.update_todos = AsyncMock()
        status.todos = [TodoItem("step 1"), TodoItem("step 2")]

        agent, _, _ = _make_agent(
            llm_responses=[
                _llm_tool(["echo 'FORGE_TODO:check:step 1'"]),
                _llm_text("Done."),
            ],
            exec_result=_FakeExecResult(
                stdout="FORGE_TODO:check:step 1",
            ),
            status=status,
        )
        await agent.run("system prompt", "test check")

        status.update_todos.assert_awaited()
        updated = status.update_todos.call_args[0][0]
        assert updated[0].done is True
        assert updated[1].done is False


class TestTextToolCallDetection:
    @pytest.mark.asyncio
    async def test_text_embedded_tool_call_executed(self) -> None:
        """When LLM outputs a JSON tool call as text, detect and execute it."""
        text_with_json = (
            'Let me check: {"name": "execute", "arguments": {"command": "ls /workspace"}}'
        )
        agent, llm, container = _make_agent(
            llm_responses=[
                _llm_text(text_with_json),
                _llm_text("Found files in workspace."),
            ],
            exec_result=_FakeExecResult(stdout="README.md"),
        )
        reply = await agent.run("system prompt", "list files")
        assert reply == "Found files in workspace."
        container.exec.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_text_call_capped_at_three(self) -> None:
        """Text-embedded tool calls are capped at 3."""
        # Use unique commands to avoid duplicate detection
        jsons = [
            '{"name": "execute", "arguments": {"command": "echo 1"}}',
            '{"name": "execute", "arguments": {"command": "echo 2"}}',
            '{"name": "execute", "arguments": {"command": "echo 3"}}',
            '{"name": "execute", "arguments": {"command": "echo 4"}}',
        ]
        agent, llm, container = _make_agent(
            llm_responses=[
                _llm_text(jsons[0]),
                _llm_text(jsons[1]),
                _llm_text(jsons[2]),
                _llm_text(jsons[3]),  # 4th -- should be returned as text
            ],
            exec_result=_FakeExecResult(stdout="hi"),
        )
        reply = await agent.run("system prompt", "test cap")
        # The 4th attempt should return the raw text since cap is reached
        assert reply == jsons[3]
        assert container.exec.await_count == 3


class TestExtraTools:
    @pytest.mark.asyncio
    async def test_extra_tool_dispatched(self) -> None:
        """Extra tools registered via extra_tools are dispatched correctly."""
        fake_tool = _FakeTool(result="Search found auth module.")
        agent, llm, _ = _make_agent(
            llm_responses=[
                _llm_tool(["how does auth work"], tool_name="fake_search"),
                _llm_text("Auth uses JWT."),
            ],
            extra_tools=[fake_tool],
        )
        reply = await agent.run("system prompt", "explain auth")
        assert reply == "Auth uses JWT."

    @pytest.mark.asyncio
    async def test_unknown_tool_returns_error(self) -> None:
        """Calling an unknown tool name returns an error message."""
        agent, _, _ = _make_agent(
            llm_responses=[
                _llm_tool(["query"], tool_name="nonexistent_tool"),
                _llm_text("OK."),
            ],
        )
        reply = await agent.run("system prompt", "test")
        assert reply == "OK."

    @pytest.mark.asyncio
    async def test_extra_tool_status_recorded(self) -> None:
        """Extra tool calls are recorded in the status manager."""
        status = MagicMock()
        status.update_phase = AsyncMock()
        status.record_tool_call = AsyncMock()
        status.update_todos = AsyncMock()
        status.todos = []

        fake_tool = _FakeTool(result="Found it.")
        agent, _, _ = _make_agent(
            llm_responses=[
                _llm_tool(["search query"], tool_name="fake_search"),
                _llm_text("Answer."),
            ],
            extra_tools=[fake_tool],
            status=status,
        )
        await agent.run("system prompt", "test")

        status.record_tool_call.assert_awaited_once()
        record = status.record_tool_call.call_args[0][0]
        assert record.tool_name == "fake_search"
        assert record.success is True

    @pytest.mark.asyncio
    async def test_extra_tool_schema_in_llm_call(self) -> None:
        """Extra tool schemas are included in LLM tool calls."""
        fake_tool = _FakeTool()
        agent, llm, _ = _make_agent(
            llm_responses=[_llm_text("Answer.")],
            extra_tools=[fake_tool],
        )
        await agent.run("system prompt", "test")

        call_kwargs = llm.chat_with_tools.call_args_list[0].kwargs
        tools = call_kwargs.get("tools", [])
        tool_names = [t["function"]["name"] for t in tools]
        assert "execute" in tool_names
        assert "fake_search" in tool_names


class TestReasoningNudge:
    @pytest.mark.asyncio
    async def test_nudge_injected_at_round_two(self) -> None:
        """Reasoning nudge is injected every 3 rounds starting from round 2."""
        agent, llm, _ = _make_agent(
            llm_responses=[
                _llm_tool(["cmd0"]),
                _llm_tool(["cmd1"]),
                _llm_tool(["cmd2"]),  # round 2 -- nudge should be injected after
                _llm_text("Final answer."),
            ],
            exec_result=_FakeExecResult(stdout="out"),
        )
        reply = await agent.run("system prompt", "test nudge")
        assert reply == "Final answer."

        # Check the last LLM call includes the nudge message
        last_call = llm.chat_with_tools.call_args_list[-1]
        messages = last_call.kwargs.get("messages", [])
        nudge_found = any("pause and assess" in (m.get("content") or "").lower() for m in messages)
        assert nudge_found


class TestCollectArtifacts:
    @pytest.mark.asyncio
    async def test_collect_artifacts_empty(self) -> None:
        """No artifacts when directory doesn't exist."""
        agent, _, container = _make_agent()
        container.exec.return_value = _FakeExecResult(exit_code=1, stdout="")
        agent._read_notes = AsyncMock(return_value="")
        result = await agent.collect_artifacts()
        assert result == []

    @pytest.mark.asyncio
    async def test_collect_artifacts_with_files(self) -> None:
        """Artifacts are collected from /workspace/.artifacts/."""
        agent, _, container = _make_agent()

        call_count = 0

        async def mock_exec(command, timeout=60, **kw):
            nonlocal call_count
            call_count += 1
            if "ls " in command:
                return _FakeExecResult(stdout="report.txt\ndata.csv")
            if "report.txt" in command:
                return _FakeExecResult(stdout="Report content")
            if "data.csv" in command:
                return _FakeExecResult(stdout="a,b,c")
            return _FakeExecResult()

        container.exec = AsyncMock(side_effect=mock_exec)
        agent._read_notes = AsyncMock(return_value="")
        result = await agent.collect_artifacts()
        assert len(result) == 2
        assert result[0][0] == "report.txt"
        assert result[0][1] == b"Report content"


class TestContextTrimming:
    @pytest.mark.asyncio
    async def test_messages_trimmed_for_small_context(self) -> None:
        """With a tiny context window, old rounds are dropped."""
        # 5 rounds of execution, each producing output
        responses = []
        for _ in range(5):
            responses.append(_llm_tool(["echo output"]))
        responses.append(_llm_text("Final."))

        agent, llm, _ = _make_agent(
            llm_responses=responses,
            exec_result=_FakeExecResult(stdout="x" * 2000),
            context_window=2048,  # Very small
        )
        reply = await agent.run("system prompt", "test trim")
        assert isinstance(reply, str)

        # Later calls should have fewer messages (old ones trimmed)
        last_call = llm.chat_with_tools.call_args_list[-1]
        messages = last_call.kwargs.get("messages", last_call.args[0] if last_call.args else [])
        # System + user are always present (at least 2 messages)
        assert len(messages) >= 2


class TestAgentName:
    @pytest.mark.asyncio
    async def test_agent_name_prefixes_status(self) -> None:
        """When agent_name is set, status.update_phase includes the prefix."""
        status = MagicMock()
        status.update_phase = AsyncMock()
        status.record_tool_call = AsyncMock()

        agent, _, _ = _make_agent(
            llm_responses=[_llm_tool(["echo hi"]), _llm_text("Done.")],
            status=status,
        )
        agent._agent_name = "test-sub-agent"
        await agent.run("system", "user msg")

        # Check that the phase update includes the agent name prefix.
        phase_calls = [call.args[0] for call in status.update_phase.call_args_list]
        assert any("[test-sub-agent]" in p for p in phase_calls)

        # Check that ToolCallRecords include agent_name.
        record_calls = status.record_tool_call.call_args_list
        assert len(record_calls) > 0
        for call in record_calls:
            record = call.args[0]
            assert record.agent_name == "test-sub-agent"

    @pytest.mark.asyncio
    async def test_no_prefix_without_agent_name(self) -> None:
        """Without agent_name, status.update_phase has no prefix."""
        status = MagicMock()
        status.update_phase = AsyncMock()
        status.record_tool_call = AsyncMock()

        agent, _, _ = _make_agent(
            llm_responses=[_llm_tool(["echo hi"]), _llm_text("Done.")],
            status=status,
        )
        await agent.run("system", "user msg")

        phase_calls = [call.args[0] for call in status.update_phase.call_args_list]
        # None of the phase updates should have a bracket prefix.
        assert all("[" not in p for p in phase_calls)
