"""Tests for the status comment system."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from forge_bot.status.formatter import (
    TodoItem,
    ToolCallRecord,
    abbreviate,
    format_todo_list,
    format_tool_call,
    format_tool_calls_section,
)
from forge_bot.status.manager import StatusCommentManager


# -- formatter tests --


class TestFormatTodoList:
    def test_empty(self) -> None:
        assert format_todo_list([]) == ""

    def test_all_pending(self) -> None:
        todos = [TodoItem("Fetch files"), TodoItem("Run tests")]
        result = format_todo_list(todos)
        assert "- [ ] Fetch files" in result
        assert "- [ ] Run tests" in result

    def test_mixed(self) -> None:
        todos = [TodoItem("Fetch files", done=True), TodoItem("Run tests")]
        result = format_todo_list(todos)
        assert "- [x] Fetch files" in result
        assert "- [ ] Run tests" in result

    def test_all_done(self) -> None:
        todos = [TodoItem("Done", done=True)]
        result = format_todo_list(todos)
        assert "- [x] Done" in result


class TestAbbreviate:
    def test_short_text(self) -> None:
        assert abbreviate("hello", 100) == "hello"

    def test_exact_limit(self) -> None:
        text = "x" * 100
        assert abbreviate(text, 100) == text

    def test_truncated(self) -> None:
        text = "x" * 200
        result = abbreviate(text, 100)
        assert len(result) == 100
        assert result.endswith("...")


class TestFormatToolCall:
    def test_success(self) -> None:
        record = ToolCallRecord(
            tool_name="exec",
            arguments_summary="echo hello",
            result_summary="hello",
            success=True,
            duration_seconds=0.5,
        )
        result = format_tool_call(record)
        assert "[+]" in result
        assert "`exec(echo hello)`" in result
        assert "(0.5s)" in result
        assert "> hello" in result

    def test_failure(self) -> None:
        record = ToolCallRecord(
            tool_name="api_call",
            arguments_summary="get_file_content",
            result_summary="404 Not Found",
            success=False,
            duration_seconds=1.2,
        )
        result = format_tool_call(record)
        assert "[x]" in result
        assert "api_call" in result

    def test_empty_result(self) -> None:
        record = ToolCallRecord(
            tool_name="todo",
            arguments_summary="set items",
            result_summary="",
            success=True,
            duration_seconds=0.1,
        )
        result = format_tool_call(record)
        assert ">" not in result  # No result summary line

    def test_long_args_abbreviated(self) -> None:
        record = ToolCallRecord(
            tool_name="exec",
            arguments_summary="a" * 200,
            result_summary="ok",
            success=True,
            duration_seconds=1.0,
        )
        result = format_tool_call(record)
        assert "..." in result


class TestFormatToolCallsSection:
    def test_empty(self) -> None:
        assert format_tool_calls_section([]) == ""

    def test_single(self) -> None:
        records = [
            ToolCallRecord("exec", "ls", "file.py", True, 0.2)
        ]
        result = format_tool_calls_section(records)
        assert "<details>" in result
        assert "Tool calls (1)" in result
        assert "</details>" in result

    def test_max_shown(self) -> None:
        records = [
            ToolCallRecord(f"tool{i}", f"arg{i}", f"result{i}", True, 0.1)
            for i in range(15)
        ]
        result = format_tool_calls_section(records, max_shown=5)
        assert "Tool calls (15)" in result
        # Only last 5 should be shown
        assert "tool10" in result
        assert "tool14" in result
        # Earlier ones should not be shown
        assert "tool0" not in result


# -- manager tests --


class TestStatusCommentManager:
    def _make_mock_api(self) -> MagicMock:
        """Create a mock GenericForgeClient."""
        api = MagicMock()
        # call() needs to be async
        api.call = AsyncMock()
        api.call.return_value = {"id": 42, "body": "status"}
        return api

    @pytest.mark.asyncio
    async def test_post_initial_status(self) -> None:
        api = self._make_mock_api()
        mgr = StatusCommentManager(api, "owner", "repo", 5)

        await mgr.post_initial_status()

        api.call.assert_called_once()
        call_args = api.call.call_args
        assert call_args[0][0] == "post_issue_comment"
        assert call_args[1]["owner"] == "owner"
        assert call_args[1]["repo"] == "repo"
        assert call_args[1]["index"] == 5
        assert "Starting..." in call_args[1]["body"]
        assert mgr.status_comment_id == 42

    @pytest.mark.asyncio
    async def test_update_phase(self) -> None:
        api = self._make_mock_api()
        mgr = StatusCommentManager(api, "owner", "repo", 5)
        await mgr.post_initial_status()

        api.call.reset_mock()
        await mgr.update_phase("Analyzing code...")

        api.call.assert_called_once()
        call_args = api.call.call_args
        assert call_args[0][0] == "edit_issue_comment"
        assert "Analyzing code..." in call_args[1]["body"]

    @pytest.mark.asyncio
    async def test_update_todos(self) -> None:
        api = self._make_mock_api()
        mgr = StatusCommentManager(api, "owner", "repo", 5)
        await mgr.post_initial_status()

        api.call.reset_mock()
        await mgr.update_todos([
            TodoItem("Fetch files", done=True),
            TodoItem("Run tests"),
        ])

        call_args = api.call.call_args
        body = call_args[1]["body"]
        assert "- [x] Fetch files" in body
        assert "- [ ] Run tests" in body
        assert mgr.todos[0].done is True

    @pytest.mark.asyncio
    async def test_record_tool_call(self) -> None:
        api = self._make_mock_api()
        mgr = StatusCommentManager(api, "owner", "repo", 5)
        await mgr.post_initial_status()

        api.call.reset_mock()
        await mgr.record_tool_call(ToolCallRecord(
            tool_name="exec",
            arguments_summary="git log",
            result_summary="abc123 initial commit",
            success=True,
            duration_seconds=1.5,
        ))

        call_args = api.call.call_args
        body = call_args[1]["body"]
        assert "exec" in body
        assert "Tool calls (1)" in body
        assert len(mgr.tool_calls) == 1

    @pytest.mark.asyncio
    async def test_post_response(self) -> None:
        api = self._make_mock_api()
        mgr = StatusCommentManager(api, "owner", "repo", 5)

        await mgr.post_response("Here is my analysis...")

        call_args = api.call.call_args
        assert call_args[0][0] == "post_issue_comment"
        assert call_args[1]["body"] == "Here is my analysis..."

    @pytest.mark.asyncio
    async def test_finalize_status(self) -> None:
        api = self._make_mock_api()
        mgr = StatusCommentManager(api, "owner", "repo", 5)
        await mgr.post_initial_status()

        api.call.reset_mock()
        await mgr.finalize_status("Done")

        call_args = api.call.call_args
        assert "Done" in call_args[1]["body"]

    @pytest.mark.asyncio
    async def test_update_without_post_is_noop(self) -> None:
        api = self._make_mock_api()
        mgr = StatusCommentManager(api, "owner", "repo", 5)

        # Never posted initial status, so no comment_id
        await mgr.update_phase("something")
        # Should not have called edit
        api.call.assert_not_called()

    @pytest.mark.asyncio
    async def test_update_failure_logged_not_raised(self) -> None:
        api = self._make_mock_api()
        mgr = StatusCommentManager(api, "owner", "repo", 5)
        await mgr.post_initial_status()

        # Make edit calls fail
        api.call.side_effect = Exception("network error")
        # Should not raise
        await mgr.update_phase("crashed?")

    @pytest.mark.asyncio
    async def test_char_limit_enforced(self) -> None:
        api = self._make_mock_api()
        mgr = StatusCommentManager(api, "owner", "repo", 5)
        await mgr.post_initial_status()

        # Add many tool calls to exceed limit
        for i in range(100):
            mgr._tool_calls.append(ToolCallRecord(
                tool_name=f"tool_{i}",
                arguments_summary="x" * 100,
                result_summary="y" * 100,
                success=True,
                duration_seconds=0.1,
            ))

        api.call.reset_mock()
        await mgr._update_status_comment()

        body = api.call.call_args[1]["body"]
        assert len(body) <= 4000 + 20  # limit + trim marker
