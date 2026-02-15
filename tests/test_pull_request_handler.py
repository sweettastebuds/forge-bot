"""Tests for forge_bot.handlers.pull_request."""

from unittest.mock import AsyncMock

import pytest

from forge_bot.config import Settings
from forge_bot.handlers.pull_request import PullRequestHandler
from forge_bot.models import PullRequestEvent

SAMPLE_DIFF = """\
diff --git a/server.py b/server.py
--- a/server.py
+++ b/server.py
@@ -10,6 +10,7 @@
 import logging
+import os
"""


@pytest.fixture
def settings(monkeypatch: pytest.MonkeyPatch) -> Settings:
    monkeypatch.setenv("FORGE_INSTANCE_URL", "https://gitea.example.com")
    monkeypatch.setenv("FORGE_API_TOKEN", "test-token")
    monkeypatch.setenv("FORGE_WEBHOOK_SECRET", "test-secret")
    monkeypatch.setenv("LLM_API_KEY", "test-llm-key")
    return Settings()


@pytest.fixture
def pr_event(sample_pr_payload: dict) -> PullRequestEvent:
    return PullRequestEvent.model_validate(sample_pr_payload)


@pytest.fixture
def mock_forge():
    forge = AsyncMock()
    forge.get_pull_diff.return_value = SAMPLE_DIFF
    forge.get_pull_files.return_value = [
        {"filename": "server.py", "additions": 1, "deletions": 0},
    ]
    forge.post_comment.return_value = {"id": 99, "body": "review"}
    return forge


@pytest.fixture
def mock_llm():
    llm = AsyncMock()
    llm.chat.return_value = "Looks good — no major issues found."
    return llm


async def test_handle_fetches_diff_and_posts_review(
    pr_event: PullRequestEvent,
    mock_forge: AsyncMock,
    mock_llm: AsyncMock,
    settings: Settings,
):
    handler = PullRequestHandler(mock_forge, mock_llm, settings, "forge-bot")
    await handler.handle(pr_event)

    mock_forge.get_pull_diff.assert_awaited_once_with("owner", "repo", 1)
    mock_forge.get_pull_files.assert_awaited_once_with("owner", "repo", 1)

    # LLM called with system prompt from template + user message with diff.
    mock_llm.chat.assert_awaited_once()
    system_prompt = mock_llm.chat.call_args.args[0]
    user_message = mock_llm.chat.call_args.args[1]
    assert "owner/repo" in system_prompt
    assert "Test PR" in system_prompt
    assert "diff --git" in user_message
    assert "server.py" in user_message

    # Posts the review as a comment on the PR.
    mock_forge.post_comment.assert_awaited_once_with(
        "owner", "repo", 1, "Looks good — no major issues found."
    )


async def test_handle_includes_pr_description_in_message(
    pr_event: PullRequestEvent,
    mock_forge: AsyncMock,
    mock_llm: AsyncMock,
    settings: Settings,
):
    handler = PullRequestHandler(mock_forge, mock_llm, settings, "forge-bot")
    await handler.handle(pr_event)

    user_message = mock_llm.chat.call_args.args[1]
    assert "Test description" in user_message
    assert "feature" in user_message  # head branch
    assert "main" in user_message  # base branch


async def test_handle_skips_review_on_empty_diff(
    pr_event: PullRequestEvent,
    mock_llm: AsyncMock,
    settings: Settings,
):
    mock_forge = AsyncMock()
    mock_forge.get_pull_diff.return_value = ""
    mock_forge.get_pull_files.return_value = []

    handler = PullRequestHandler(mock_forge, mock_llm, settings, "forge-bot")
    await handler.handle(pr_event)

    mock_llm.chat.assert_not_awaited()
    mock_forge.post_comment.assert_not_awaited()


async def test_handle_truncates_large_diffs(
    pr_event: PullRequestEvent,
    mock_forge: AsyncMock,
    mock_llm: AsyncMock,
    settings: Settings,
):
    mock_forge.get_pull_diff.return_value = "x" * 50_000

    handler = PullRequestHandler(mock_forge, mock_llm, settings, "forge-bot")
    await handler.handle(pr_event)

    user_message = mock_llm.chat.call_args.args[1]
    assert "diff truncated" in user_message
    # The diff portion should be capped, not the full 50k.
    assert len(user_message) < 50_000


async def test_handle_posts_error_on_llm_failure(
    pr_event: PullRequestEvent,
    mock_forge: AsyncMock,
    settings: Settings,
):
    mock_llm = AsyncMock()
    mock_llm.chat.side_effect = RuntimeError("LLM down")

    handler = PullRequestHandler(mock_forge, mock_llm, settings, "forge-bot")
    await handler.handle(pr_event)

    mock_forge.post_comment.assert_awaited_once()
    posted_body = mock_forge.post_comment.call_args.args[3]
    assert "error" in posted_body.lower()


async def test_handle_survives_diff_fetch_failure(
    pr_event: PullRequestEvent,
    mock_llm: AsyncMock,
    settings: Settings,
):
    mock_forge = AsyncMock()
    mock_forge.get_pull_diff.side_effect = RuntimeError("API error")
    mock_forge.get_pull_files.return_value = []

    handler = PullRequestHandler(mock_forge, mock_llm, settings, "forge-bot")
    await handler.handle(pr_event)

    # Empty diff → skip review entirely.
    mock_llm.chat.assert_not_awaited()


async def test_handle_survives_post_comment_failure(
    pr_event: PullRequestEvent,
    mock_forge: AsyncMock,
    mock_llm: AsyncMock,
    settings: Settings,
):
    mock_forge.post_comment.side_effect = RuntimeError("API error")

    handler = PullRequestHandler(mock_forge, mock_llm, settings, "forge-bot")
    # Should not raise.
    await handler.handle(pr_event)
    mock_forge.post_comment.assert_awaited_once()


async def test_build_user_message_without_description(
    sample_pr_payload: dict,
    mock_forge: AsyncMock,
    mock_llm: AsyncMock,
    settings: Settings,
):
    sample_pr_payload["pull_request"]["body"] = ""
    event = PullRequestEvent.model_validate(sample_pr_payload)

    handler = PullRequestHandler(mock_forge, mock_llm, settings, "forge-bot")
    await handler.handle(event)

    user_message = mock_llm.chat.call_args.args[1]
    assert "Description" not in user_message
