"""Tests for forge_bot.handlers.issue_comment."""

from unittest.mock import AsyncMock

import pytest

from forge_bot.config import Settings
from forge_bot.handlers.issue_comment import IssueCommentHandler
from forge_bot.models import IssueCommentEvent


@pytest.fixture
def settings(monkeypatch: pytest.MonkeyPatch) -> Settings:
    monkeypatch.setenv("FORGE_INSTANCE_URL", "https://gitea.example.com")
    monkeypatch.setenv("FORGE_API_TOKEN", "test-token")
    monkeypatch.setenv("FORGE_WEBHOOK_SECRET", "test-secret")
    monkeypatch.setenv("LLM_API_KEY", "test-llm-key")
    return Settings()


@pytest.fixture
def comment_event(sample_comment_payload: dict) -> IssueCommentEvent:
    return IssueCommentEvent.model_validate(sample_comment_payload)


@pytest.fixture
def mock_forge():
    forge = AsyncMock()
    forge.get_issue_comments.return_value = [
        {
            "id": 1,
            "body": "I need help understanding the auth flow.",
            "user": {"login": "developer"},
            "created_at": "2026-01-01T00:00:00Z",
        },
        {
            "id": 2,
            "body": "@forge-bot what does this function do?",
            "user": {"login": "developer"},
            "created_at": "2026-01-01T00:01:00Z",
        },
    ]
    forge.post_comment.return_value = {"id": 3, "body": "LLM says hi"}
    return forge


@pytest.fixture
def mock_llm():
    llm = AsyncMock()
    llm.chat.return_value = "Here is my explanation of the auth flow."
    return llm


async def test_handle_fetches_comments_and_posts_reply(
    comment_event: IssueCommentEvent,
    mock_forge: AsyncMock,
    mock_llm: AsyncMock,
    settings: Settings,
):
    handler = IssueCommentHandler(mock_forge, mock_llm, settings, "forge-bot")
    await handler.handle(comment_event)

    # Should fetch conversation thread.
    mock_forge.get_issue_comments.assert_awaited_once_with("owner", "repo", 5)

    # Should call the LLM with system prompt and user message.
    mock_llm.chat.assert_awaited_once()
    call_args = mock_llm.chat.call_args
    system_prompt = call_args.args[0]
    user_message = call_args.args[1]
    assert "owner/repo" in system_prompt
    assert "#5" in system_prompt
    assert user_message == "@forge-bot what does this function do?"

    # Should post the LLM response as a comment.
    mock_forge.post_comment.assert_awaited_once_with(
        "owner", "repo", 5, "Here is my explanation of the auth flow."
    )


async def test_handle_posts_error_message_on_llm_failure(
    comment_event: IssueCommentEvent,
    mock_forge: AsyncMock,
    settings: Settings,
):
    mock_llm = AsyncMock()
    mock_llm.chat.side_effect = RuntimeError("LLM is down")

    handler = IssueCommentHandler(mock_forge, mock_llm, settings, "forge-bot")
    await handler.handle(comment_event)

    # Should still post a fallback error message.
    mock_forge.post_comment.assert_awaited_once()
    posted_body = mock_forge.post_comment.call_args.args[3]
    assert "error" in posted_body.lower()


async def test_handle_logs_on_post_comment_failure(
    comment_event: IssueCommentEvent,
    mock_llm: AsyncMock,
    settings: Settings,
):
    mock_forge = AsyncMock()
    mock_forge.get_issue_comments.return_value = []
    mock_forge.post_comment.side_effect = RuntimeError("API error")

    handler = IssueCommentHandler(mock_forge, mock_llm, settings, "forge-bot")

    # Should not raise — the handler catches and logs.
    await handler.handle(comment_event)
    mock_forge.post_comment.assert_awaited_once()
