"""Integration tests: StatusCommentManager + formatter end-to-end.

Tests the real StatusCommentManager posting and editing comments via the
API tracker, with real markdown rendering from the formatter module.
Verifies the two-comment system (status + response) works correctly.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from forge_bot.status.formatter import TodoItem, ToolCallRecord
from forge_bot.status.manager import StatusCommentManager

from .conftest import FakeApiTracker


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_status(api_tracker: FakeApiTracker) -> StatusCommentManager:
    """Build a real StatusCommentManager backed by FakeApiTracker."""
    api = MagicMock()
    api.call = AsyncMock(side_effect=api_tracker.call)
    return StatusCommentManager(api, "org", "myapp", 7)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestTwoCommentSystem:
    """Verify the status + response comment lifecycle."""

    async def test_initial_status_posted(
        self,
        api_tracker: FakeApiTracker,
    ) -> None:
        """post_initial_status() creates a comment with 'Starting...' state."""
        status = _make_status(api_tracker)
        await status.post_initial_status()

        posts = api_tracker.calls_for("post_issue_comment")
        assert len(posts) == 1
        body = posts[0]["body"]
        assert "Starting" in body
        assert status.status_comment_id is not None

    async def test_phase_updates_edit_status(
        self,
        api_tracker: FakeApiTracker,
    ) -> None:
        """update_phase() edits the existing status comment."""
        status = _make_status(api_tracker)
        await status.post_initial_status()
        await status.update_phase("Cloning repository...")

        edits = api_tracker.calls_for("edit_issue_comment")
        assert len(edits) >= 1
        assert "Cloning repository" in edits[-1]["body"]

    async def test_response_is_separate_comment(
        self,
        api_tracker: FakeApiTracker,
    ) -> None:
        """post_response() creates a new comment (not an edit)."""
        status = _make_status(api_tracker)
        await status.post_initial_status()
        await status.post_response("Here is my analysis of the code.")

        posts = api_tracker.calls_for("post_issue_comment")
        assert len(posts) == 2  # status + response
        assert "analysis" in posts[1]["body"]

    async def test_finalize_edits_status(
        self,
        api_tracker: FakeApiTracker,
    ) -> None:
        """finalize_status() updates the status comment to 'Done'."""
        status = _make_status(api_tracker)
        await status.post_initial_status()
        await status.finalize_status("Done")

        edits = api_tracker.calls_for("edit_issue_comment")
        assert any("Done" in e.get("body", "") for e in edits)


class TestTodoRendering:
    """Todo items are rendered in the status comment markdown."""

    async def test_todos_appear_in_status(
        self,
        api_tracker: FakeApiTracker,
    ) -> None:
        """After update_todos(), status comment shows checkboxes."""
        status = _make_status(api_tracker)
        await status.post_initial_status()
        await status.update_todos([
            TodoItem("Clone repository"),
            TodoItem("Run tests"),
            TodoItem("Analyze results"),
        ])

        edits = api_tracker.calls_for("edit_issue_comment")
        assert len(edits) >= 1
        body = edits[-1]["body"]
        assert "- [ ] Clone repository" in body
        assert "- [ ] Run tests" in body
        assert "- [ ] Analyze results" in body

    async def test_checked_todos_rendered(
        self,
        api_tracker: FakeApiTracker,
    ) -> None:
        """Completed todos show [x] checkboxes."""
        status = _make_status(api_tracker)
        await status.post_initial_status()
        await status.update_todos([
            TodoItem("Clone repository", done=True),
            TodoItem("Run tests", done=False),
        ])

        edits = api_tracker.calls_for("edit_issue_comment")
        body = edits[-1]["body"]
        assert "- [x] Clone repository" in body
        assert "- [ ] Run tests" in body


class TestToolCallRendering:
    """Tool call history is rendered in the status comment."""

    async def test_tool_calls_in_collapsible_section(
        self,
        api_tracker: FakeApiTracker,
    ) -> None:
        """Tool calls appear in a collapsible <details> section."""
        status = _make_status(api_tracker)
        await status.post_initial_status()

        await status.record_tool_call(ToolCallRecord(
            tool_name="execute",
            arguments_summary="ls /workspace",
            result_summary="README.md src/ tests/",
            success=True,
            duration_seconds=0.3,
        ))

        edits = api_tracker.calls_for("edit_issue_comment")
        body = edits[-1]["body"]
        assert "<details>" in body
        assert "Tool calls (1)" in body
        assert "execute" in body
        assert "ls /workspace" in body

    async def test_multiple_tool_calls_accumulated(
        self,
        api_tracker: FakeApiTracker,
    ) -> None:
        """Multiple tool calls are all rendered."""
        status = _make_status(api_tracker)
        await status.post_initial_status()

        for i, (cmd, result) in enumerate([
            ("ls /workspace", "README.md"),
            ("cat README.md", "# MyApp"),
            ("grep -rn login .", "auth.py:5: def login()"),
        ]):
            await status.record_tool_call(ToolCallRecord(
                tool_name="execute",
                arguments_summary=cmd,
                result_summary=result,
                success=True,
                duration_seconds=0.1 * (i + 1),
            ))

        edits = api_tracker.calls_for("edit_issue_comment")
        body = edits[-1]["body"]
        assert "Tool calls (3)" in body
        assert "ls /workspace" in body
        assert "cat README.md" in body
        assert "grep -rn login" in body

    async def test_failed_tool_call_marked(
        self,
        api_tracker: FakeApiTracker,
    ) -> None:
        """Failed tool calls are marked with [x] icon."""
        status = _make_status(api_tracker)
        await status.post_initial_status()

        await status.record_tool_call(ToolCallRecord(
            tool_name="execute",
            arguments_summary="python bad_script.py",
            result_summary="SyntaxError: invalid syntax",
            success=False,
            duration_seconds=0.5,
        ))

        edits = api_tracker.calls_for("edit_issue_comment")
        body = edits[-1]["body"]
        assert "[x]" in body
        assert "bad_script.py" in body


class TestFullStatusLifecycle:
    """Test the complete status comment lifecycle as it happens during a real event."""

    async def test_full_lifecycle(
        self,
        api_tracker: FakeApiTracker,
    ) -> None:
        """Simulate a full event: status → phases → todos → tools → response → done."""
        status = _make_status(api_tracker)

        # 1. Post initial status
        await status.post_initial_status()
        assert status.status_comment_id is not None

        # 2. Update phase
        await status.update_phase("Starting workspace...")

        # 3. Set todos
        await status.update_todos([
            TodoItem("Clone repo"),
            TodoItem("Search for auth"),
            TodoItem("Write answer"),
        ])

        # 4. Record tool calls
        await status.record_tool_call(ToolCallRecord(
            tool_name="execute",
            arguments_summary="git clone ...",
            result_summary="Cloned successfully",
            success=True,
            duration_seconds=2.5,
        ))

        # 5. Check off a todo
        todos = status.todos
        todos[0] = TodoItem("Clone repo", done=True)
        await status.update_todos(todos)

        # 6. More tool calls
        await status.record_tool_call(ToolCallRecord(
            tool_name="execute",
            arguments_summary="grep -rn auth .",
            result_summary="auth.py:10: class AuthManager",
            success=True,
            duration_seconds=0.8,
        ))

        # 7. Post response
        result = await status.post_response(
            "The auth module uses JWT tokens managed by AuthManager."
        )
        assert result.get("id") is not None

        # 8. Finalize
        await status.finalize_status("Done")

        # Verify the full sequence of API calls
        posts = api_tracker.calls_for("post_issue_comment")
        edits = api_tracker.calls_for("edit_issue_comment")

        # 2 posts: initial status + response
        assert len(posts) == 2

        # Multiple edits for phase, todos, tool calls, finalize
        assert len(edits) >= 5

        # Final status has "Done" and checked todo
        final_status = edits[-1]["body"]
        assert "Done" in final_status
        assert "- [x] Clone repo" in final_status
        assert "Tool calls (2)" in final_status

        # Response comment has the actual answer
        assert "JWT tokens" in posts[1]["body"]
