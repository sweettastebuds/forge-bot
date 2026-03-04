"""End-to-end integration: PullRequestHandler.

Tests the full flow from PullRequestHandler.handle() through:
- Real StatusCommentManager (posts + edits via API tracker)
- Real AgentLoop (tool-calling loop with mocked LLM + container)
- Real template rendering (agent_pr_review.j2)
- Artifact collection
- Error recovery

Only external boundaries are mocked: Docker, LLM HTTP API, Gitea HTTP API.
"""

from __future__ import annotations

from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from forge_bot.config import Settings
from forge_bot.handlers.pull_request import PullRequestHandler
from forge_bot.models import PullRequestEvent

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
) -> tuple[PullRequestHandler, AsyncMock]:
    api = MagicMock()
    api.call = AsyncMock(side_effect=api_tracker.call)

    llm = AsyncMock()
    llm.chat_with_tools = AsyncMock(side_effect=llm_responses)
    llm.chat = AsyncMock(return_value="Fallback review.")

    handler = PullRequestHandler(
        api_client=api,
        llm_client=llm,
        settings=settings,
        bot_username="forge-bot",
    )
    return handler, llm


def _make_pr_event() -> PullRequestEvent:
    return PullRequestEvent.model_validate({
        "action": "opened",
        "number": 42,
        "pull_request": {
            "id": 42,
            "number": 42,
            "title": "Add login endpoint",
            "body": "Implements JWT-based authentication.",
            "state": "open",
            "user": {"id": 10, "login": "alice"},
            "head": {"ref": "feature/login", "sha": "aaa111"},
            "base": {"ref": "main", "sha": "bbb222"},
            "assignees": [],
            "requested_reviewers": [],
        },
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


class TestPRHandlerE2E:
    """Full handle() flow with real internal wiring."""

    async def test_happy_path_reviews_pr(
        self,
        integration_settings: Settings,
        api_tracker: FakeApiTracker,
    ) -> None:
        """LLM clones, diffs, and reviews — status + review posted."""
        container = make_mock_container({
            "git diff": FakeExecResult(
                stdout=(
                    "diff --git a/auth.py b/auth.py\n"
                    "+def login(user, password):\n"
                    "+    return jwt.encode({'sub': user})\n"
                )
            ),
            "cat": FakeExecResult(stdout="def login(user, password):\n    return jwt.encode(...)"),
            ".notes": FakeExecResult(exit_code=1, stdout=""),
            ".artifacts": FakeExecResult(exit_code=1, stdout=""),
        })

        handler, llm = _make_handler(
            api_tracker,
            integration_settings,
            llm_responses=[
                make_llm_tool_response([
                    "git diff origin/main..origin/feature/login"
                ]),
                make_llm_tool_response(["cat auth.py"]),
                make_llm_text_response(
                    "## Review\n\n"
                    "💡 **Nit**: Consider adding input validation for the password parameter.\n\n"
                    "Overall the changes look good. JWT implementation is correct."
                ),
            ],
        )

        event = _make_pr_event()

        with (
            patch(
                "forge_bot.handlers.pull_request.ContainerManager",
                return_value=container,
            ),
            patch("forge_bot.handlers.pull_request.SmartRetriever"),
            patch("forge_bot.handlers.pull_request.RetrievalTool"),
        ):
            await handler.handle(event)

        # Container lifecycle
        container.create.assert_awaited_once()
        container.destroy.assert_awaited_once()

        # Status comment posted and edited
        posts = api_tracker.calls_for("post_issue_comment")
        assert len(posts) >= 2  # status + response
        edits = api_tracker.calls_for("edit_issue_comment")
        assert len(edits) >= 1

        # The review contains the LLM output
        review_bodies = [
            kw["body"]
            for name, kw in api_tracker.calls
            if name == "post_issue_comment" and "Review" in kw.get("body", "")
        ]
        assert len(review_bodies) >= 1
        assert "input validation" in review_bodies[0].lower() or "JWT" in review_bodies[0]

    async def test_system_prompt_includes_pr_metadata(
        self,
        integration_settings: Settings,
        api_tracker: FakeApiTracker,
    ) -> None:
        """System prompt contains PR title, branches, body."""
        captured_prompt = None

        class CapturingAgent:
            def __init__(self, *args, **kwargs):
                pass

            async def run(self, system_prompt, user_message):
                nonlocal captured_prompt
                captured_prompt = system_prompt
                return "LGTM"

            async def collect_artifacts(self):
                return []

        container = make_mock_container({})

        handler, _ = _make_handler(
            api_tracker,
            integration_settings,
            llm_responses=[],
        )

        event = _make_pr_event()

        with (
            patch(
                "forge_bot.handlers.pull_request.ContainerManager",
                return_value=container,
            ),
            patch(
                "forge_bot.handlers.pull_request.AgentLoop",
                CapturingAgent,
            ),
            patch("forge_bot.handlers.pull_request.SmartRetriever"),
            patch("forge_bot.handlers.pull_request.RetrievalTool"),
        ):
            await handler.handle(event)

        assert captured_prompt is not None
        assert "Add login endpoint" in captured_prompt
        assert "feature/login" in captured_prompt
        assert "main" in captured_prompt
        assert "org/myapp" in captured_prompt
        assert "JWT" in captured_prompt or "authentication" in captured_prompt.lower()

    async def test_user_message_references_pr(
        self,
        integration_settings: Settings,
        api_tracker: FakeApiTracker,
    ) -> None:
        """User message sent to agent references the PR."""
        captured_msg = None

        class CapturingAgent:
            def __init__(self, *args, **kwargs):
                pass

            async def run(self, system_prompt, user_message):
                nonlocal captured_msg
                captured_msg = user_message
                return "Reviewed."

            async def collect_artifacts(self):
                return []

        container = make_mock_container({})

        handler, _ = _make_handler(
            api_tracker,
            integration_settings,
            llm_responses=[],
        )

        event = _make_pr_event()

        with (
            patch(
                "forge_bot.handlers.pull_request.ContainerManager",
                return_value=container,
            ),
            patch(
                "forge_bot.handlers.pull_request.AgentLoop",
                CapturingAgent,
            ),
            patch("forge_bot.handlers.pull_request.SmartRetriever"),
            patch("forge_bot.handlers.pull_request.RetrievalTool"),
        ):
            await handler.handle(event)

        assert captured_msg is not None
        assert "#42" in captured_msg
        assert "Add login endpoint" in captured_msg

    async def test_container_failure_posts_error(
        self,
        integration_settings: Settings,
        api_tracker: FakeApiTracker,
    ) -> None:
        """Container creation failure → error posted → container destroyed."""
        container = make_mock_container({})
        container.create = AsyncMock(side_effect=RuntimeError("No Docker"))

        handler, _ = _make_handler(
            api_tracker,
            integration_settings,
            llm_responses=[make_llm_text_response("Unreachable")],
        )

        event = _make_pr_event()

        with (
            patch(
                "forge_bot.handlers.pull_request.ContainerManager",
                return_value=container,
            ),
            patch("forge_bot.handlers.pull_request.SmartRetriever"),
            patch("forge_bot.handlers.pull_request.RetrievalTool"),
        ):
            await handler.handle(event)

        container.destroy.assert_awaited_once()

        # Error response posted
        posted = api_tracker.posted_comments
        assert any("error" in body.lower() for body in posted)

    async def test_status_finalized_done(
        self,
        integration_settings: Settings,
        api_tracker: FakeApiTracker,
    ) -> None:
        """Status comment finalized with 'Done' after successful review."""

        class SimpleAgent:
            def __init__(self, *args, **kwargs):
                pass

            async def run(self, system_prompt, user_message):
                return "LGTM."

            async def collect_artifacts(self):
                return []

        container = make_mock_container({})

        handler, _ = _make_handler(
            api_tracker,
            integration_settings,
            llm_responses=[],
        )

        event = _make_pr_event()

        with (
            patch(
                "forge_bot.handlers.pull_request.ContainerManager",
                return_value=container,
            ),
            patch(
                "forge_bot.handlers.pull_request.AgentLoop",
                SimpleAgent,
            ),
            patch("forge_bot.handlers.pull_request.SmartRetriever"),
            patch("forge_bot.handlers.pull_request.RetrievalTool"),
        ):
            await handler.handle(event)

        edits = api_tracker.edited_comments
        assert len(edits) >= 1
        assert "Done" in edits[-1]

    async def test_artifacts_attached_to_review(
        self,
        integration_settings: Settings,
        api_tracker: FakeApiTracker,
    ) -> None:
        """Artifacts from the agent are uploaded as attachments."""

        class ArtifactAgent:
            def __init__(self, *args, **kwargs):
                pass

            async def run(self, system_prompt, user_message):
                return "Review with attachment."

            async def collect_artifacts(self):
                return [("diff.patch", b"patch data")]

        container = make_mock_container({})

        handler, _ = _make_handler(
            api_tracker,
            integration_settings,
            llm_responses=[],
        )

        event = _make_pr_event()

        with (
            patch(
                "forge_bot.handlers.pull_request.ContainerManager",
                return_value=container,
            ),
            patch(
                "forge_bot.handlers.pull_request.AgentLoop",
                ArtifactAgent,
            ),
            patch("forge_bot.handlers.pull_request.SmartRetriever"),
            patch("forge_bot.handlers.pull_request.RetrievalTool"),
        ):
            await handler.handle(event)

        upload_calls = api_tracker.calls_for("upload_comment_attachment")
        assert len(upload_calls) == 1
        assert upload_calls[0]["attachment"] == ("diff.patch", b"patch data")
