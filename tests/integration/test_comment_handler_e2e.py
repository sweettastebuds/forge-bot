"""End-to-end integration: IssueCommentHandler.

Tests the full flow from IssueCommentHandler.handle() through:
- Real StatusCommentManager (posts + edits via API tracker)
- Real AgentLoop (tool-calling loop with mocked LLM + container)
- Real template rendering (agent_system.j2)
- Conversation context fetching
- Artifact collection

Only external boundaries are mocked: Docker, LLM HTTP API, Gitea HTTP API.
"""

from __future__ import annotations

import json
from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from forge_bot.config import Settings
from forge_bot.handlers.issue_comment import IssueCommentHandler
from forge_bot.models import IssueCommentEvent

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


def _make_handler(
    api_tracker: FakeApiTracker,
    settings: Settings,
    llm_responses: list[Any],
) -> tuple[IssueCommentHandler, AsyncMock]:
    """Build an IssueCommentHandler with a tracked API and scripted LLM."""
    api = MagicMock()
    api.call = AsyncMock(side_effect=api_tracker.call)

    llm = AsyncMock()
    llm.chat_with_tools = AsyncMock(side_effect=llm_responses)
    llm.chat = AsyncMock(return_value="Fallback answer.")

    handler = IssueCommentHandler(
        api_client=api,
        llm_client=llm,
        settings=settings,
        bot_username="forge-bot",
    )
    return handler, llm


def _make_event(
    body: str = "@forge-bot explain auth",
    issue_number: int = 7,
    issue_body: str = "I need help understanding the auth flow.",
) -> IssueCommentEvent:
    return IssueCommentEvent.model_validate({
        "action": "created",
        "comment": {
            "id": 200,
            "body": body,
            "user": {"id": 10, "login": "alice"},
        },
        "issue": {
            "number": issue_number,
            "title": "Auth module question",
            "body": issue_body,
            "pull_request": None,
            "assignees": [],
        },
        "is_pull": False,
        "repository": {
            "full_name": "org/myapp",
            "clone_url": "https://gitea.example.com/org/myapp.git",
            "default_branch": "main",
        },
        "sender": {"id": 10, "login": "alice"},
    })


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestIssueCommentE2E:
    """Full handle() flow with real internal wiring."""

    async def test_happy_path_posts_response(
        self,
        integration_settings: Settings,
        api_tracker: FakeApiTracker,
    ) -> None:
        """LLM explores, then responds — status + response comments posted."""
        container = make_mock_container({
            "ls": FakeExecResult(stdout="auth.py\nmodels.py\nroutes.py"),
            "cat": FakeExecResult(stdout="def login():\n    return jwt.encode(payload)"),
            ".git": FakeExecResult(stdout="ok"),
            ".notes": FakeExecResult(exit_code=1, stdout=""),
            ".artifacts": FakeExecResult(exit_code=1, stdout=""),
        })

        handler, llm = _make_handler(
            api_tracker,
            integration_settings,
            llm_responses=[
                make_llm_tool_response(["ls /workspace"]),
                make_llm_tool_response(["cat auth.py"]),
                make_llm_text_response(
                    "The auth module in `auth.py` uses JWT tokens. "
                    "The `login()` function encodes a payload with `jwt.encode()`."
                ),
            ],
        )

        event = _make_event()

        with (
            patch(
                "forge_bot.handlers.issue_comment.ContainerManager",
                return_value=container,
            ),
            patch("forge_bot.handlers.issue_comment.SmartRetriever"),
            patch("forge_bot.handlers.issue_comment.RetrievalTool"),
        ):
            await handler.handle(event)

        # Container lifecycle
        container.create.assert_awaited_once()
        container.destroy.assert_awaited_once()

        # A status comment was posted initially
        status_posts = api_tracker.calls_for("post_issue_comment")
        assert len(status_posts) >= 2  # 1 status + 1 response

        # Status edits happened (phase updates, tool calls)
        edits = api_tracker.calls_for("edit_issue_comment")
        assert len(edits) >= 1

        # The final response comment contains the LLM answer
        response_bodies = [
            kw["body"]
            for name, kw in api_tracker.calls
            if name == "post_issue_comment" and "JWT" in kw.get("body", "")
        ]
        assert len(response_bodies) >= 1
        assert "jwt.encode()" in response_bodies[0].lower() or "JWT" in response_bodies[0]

    async def test_container_failure_posts_error(
        self,
        integration_settings: Settings,
        api_tracker: FakeApiTracker,
    ) -> None:
        """When container.create() fails, an error response is posted."""
        container = make_mock_container({})
        container.create = AsyncMock(side_effect=RuntimeError("Docker unavailable"))

        handler, _ = _make_handler(
            api_tracker,
            integration_settings,
            llm_responses=[make_llm_text_response("Unreachable")],
        )

        event = _make_event()

        with (
            patch(
                "forge_bot.handlers.issue_comment.ContainerManager",
                return_value=container,
            ),
            patch("forge_bot.handlers.issue_comment.SmartRetriever"),
            patch("forge_bot.handlers.issue_comment.RetrievalTool"),
        ):
            await handler.handle(event)

        # Container destroy always called (finally block)
        container.destroy.assert_awaited_once()

        # Error response was posted
        posted = api_tracker.posted_comments
        assert any("error" in body.lower() for body in posted)

    async def test_conversation_context_included(
        self,
        integration_settings: Settings,
        api_tracker: FakeApiTracker,
    ) -> None:
        """Prior issue comments are fetched and included in the user message."""
        # Set up API to return existing comments
        api_tracker.set_response("get_issue_comments", [
            {"id": 100, "user": {"login": "alice"}, "body": "I tried the login endpoint"},
            {"id": 150, "user": {"login": "bob"}, "body": "Same issue here"},
            {"id": 200, "user": {"login": "alice"}, "body": "@forge-bot explain auth"},
        ])

        container = make_mock_container({
            ".notes": FakeExecResult(exit_code=1, stdout=""),
            ".artifacts": FakeExecResult(exit_code=1, stdout=""),
        })

        captured_user_msg = None

        original_run = None

        class CapturingAgent:
            def __init__(self, *args, **kwargs):
                pass

            async def run(self, system_prompt, user_message):
                nonlocal captured_user_msg
                captured_user_msg = user_message
                return "Here's what I found."

            async def collect_artifacts(self):
                return []

        handler, _ = _make_handler(
            api_tracker,
            integration_settings,
            llm_responses=[],
        )

        event = _make_event(issue_body="Login endpoint returns 500")

        with (
            patch(
                "forge_bot.handlers.issue_comment.ContainerManager",
                return_value=container,
            ),
            patch(
                "forge_bot.handlers.issue_comment.AgentLoop",
                CapturingAgent,
            ),
            patch("forge_bot.handlers.issue_comment.SmartRetriever"),
            patch("forge_bot.handlers.issue_comment.RetrievalTool"),
        ):
            await handler.handle(event)

        # User message should include conversation context
        assert captured_user_msg is not None
        assert "Conversation so far" in captured_user_msg
        assert "Login endpoint returns 500" in captured_user_msg
        assert "I tried the login endpoint" in captured_user_msg
        # The triggering comment (id=200) should be excluded from context
        # but included as "Current request"
        assert "Current request" in captured_user_msg

    async def test_artifacts_uploaded(
        self,
        integration_settings: Settings,
        api_tracker: FakeApiTracker,
    ) -> None:
        """Artifacts collected by the agent are uploaded as attachments."""

        class ArtifactAgent:
            def __init__(self, *args, **kwargs):
                pass

            async def run(self, system_prompt, user_message):
                return "Here's a report."

            async def collect_artifacts(self):
                return [("report.txt", b"Report content")]

        container = make_mock_container({})

        handler, _ = _make_handler(
            api_tracker,
            integration_settings,
            llm_responses=[],
        )

        event = _make_event()

        with (
            patch(
                "forge_bot.handlers.issue_comment.ContainerManager",
                return_value=container,
            ),
            patch(
                "forge_bot.handlers.issue_comment.AgentLoop",
                ArtifactAgent,
            ),
            patch("forge_bot.handlers.issue_comment.SmartRetriever"),
            patch("forge_bot.handlers.issue_comment.RetrievalTool"),
        ):
            await handler.handle(event)

        # Check that upload_comment_attachment was called
        upload_calls = api_tracker.calls_for("upload_comment_attachment")
        assert len(upload_calls) == 1
        assert upload_calls[0]["attachment"] == ("report.txt", b"Report content")

    async def test_status_comment_finalized(
        self,
        integration_settings: Settings,
        api_tracker: FakeApiTracker,
    ) -> None:
        """Status comment is finalized with 'Done' after successful processing."""

        class SimpleAgent:
            def __init__(self, *args, **kwargs):
                pass

            async def run(self, system_prompt, user_message):
                return "Answer."

            async def collect_artifacts(self):
                return []

        container = make_mock_container({})

        handler, _ = _make_handler(
            api_tracker,
            integration_settings,
            llm_responses=[],
        )

        event = _make_event()

        with (
            patch(
                "forge_bot.handlers.issue_comment.ContainerManager",
                return_value=container,
            ),
            patch(
                "forge_bot.handlers.issue_comment.AgentLoop",
                SimpleAgent,
            ),
            patch("forge_bot.handlers.issue_comment.SmartRetriever"),
            patch("forge_bot.handlers.issue_comment.RetrievalTool"),
        ):
            await handler.handle(event)

        # The last edit to the status comment should contain "Done"
        edits = api_tracker.edited_comments
        assert len(edits) >= 1
        assert "Done" in edits[-1]

    async def test_system_prompt_contains_event_metadata(
        self,
        integration_settings: Settings,
        api_tracker: FakeApiTracker,
    ) -> None:
        """System prompt rendered from template includes issue metadata."""
        captured_system_prompt = None

        class CapturingAgent:
            def __init__(self, *args, **kwargs):
                pass

            async def run(self, system_prompt, user_message):
                nonlocal captured_system_prompt
                captured_system_prompt = system_prompt
                return "Answer."

            async def collect_artifacts(self):
                return []

        container = make_mock_container({})

        handler, _ = _make_handler(
            api_tracker,
            integration_settings,
            llm_responses=[],
        )

        event = _make_event()

        with (
            patch(
                "forge_bot.handlers.issue_comment.ContainerManager",
                return_value=container,
            ),
            patch(
                "forge_bot.handlers.issue_comment.AgentLoop",
                CapturingAgent,
            ),
            patch("forge_bot.handlers.issue_comment.SmartRetriever"),
            patch("forge_bot.handlers.issue_comment.RetrievalTool"),
        ):
            await handler.handle(event)

        assert captured_system_prompt is not None
        assert "org/myapp" in captured_system_prompt
        assert "#7" in captured_system_prompt or "7" in captured_system_prompt
        assert "Auth module question" in captured_system_prompt
        assert "forge-bot" in captured_system_prompt
