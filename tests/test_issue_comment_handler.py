"""Tests for forge_bot.handlers.issue_comment."""

from unittest.mock import AsyncMock

import httpx
import pytest

from forge_bot.config import Settings
from forge_bot.handlers.issue_comment import (
    IssueCommentHandler,
    _extract_attachments,
    _extract_file_paths,
)
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
    forge.get_repo_tree.return_value = [
        {"path": "src/main.py", "type": "blob", "size": 100},
        {"path": "README.md", "type": "blob", "size": 50},
        {"path": "src", "type": "tree"},
    ]
    forge.get_file_content.side_effect = httpx.HTTPStatusError(
        "Not Found",
        request=httpx.Request("GET", "http://x"),
        response=httpx.Response(404),
    )
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


async def test_handle_includes_repo_tree_in_prompt(
    comment_event: IssueCommentEvent,
    mock_forge: AsyncMock,
    mock_llm: AsyncMock,
    settings: Settings,
):
    handler = IssueCommentHandler(mock_forge, mock_llm, settings, "forge-bot")
    await handler.handle(comment_event)

    mock_forge.get_repo_tree.assert_awaited_once()

    system_prompt = mock_llm.chat.call_args.args[0]
    # The tree should be in the system prompt (only blobs, not tree entries).
    assert "src/main.py" in system_prompt
    assert "README.md" in system_prompt


async def test_handle_fetches_referenced_files(
    mock_llm: AsyncMock,
    settings: Settings,
):
    """When a file path is mentioned in the comment, handler fetches its content."""
    payload = {
        "action": "created",
        "comment": {
            "id": 1,
            "body": "@forge-bot can you explain forge_bot/config.py?",
            "user": {"id": 1, "login": "developer"},
        },
        "issue": {
            "number": 5,
            "title": "Question",
            "body": "",
            "pull_request": None,
            "assignees": [],
        },
        "is_pull": False,
        "repository": {"full_name": "owner/repo"},
        "sender": {"id": 1, "login": "developer"},
    }
    event = IssueCommentEvent.model_validate(payload)

    mock_forge = AsyncMock()
    mock_forge.get_issue_comments.return_value = []
    mock_forge.get_repo_tree.return_value = []
    mock_forge.get_file_content.return_value = 'SECRET = "hello"\n'
    mock_forge.post_comment.return_value = {"id": 10}

    handler = IssueCommentHandler(mock_forge, mock_llm, settings, "forge-bot")
    await handler.handle(event)

    # Should have tried to fetch the referenced file.
    mock_forge.get_file_content.assert_awaited_once_with(
        "owner", "repo", "forge_bot/config.py"
    )

    # File content should appear in the system prompt.
    system_prompt = mock_llm.chat.call_args.args[0]
    assert "forge_bot/config.py" in system_prompt
    assert 'SECRET = "hello"' in system_prompt


async def test_handle_gracefully_handles_tree_failure(
    comment_event: IssueCommentEvent,
    mock_llm: AsyncMock,
    settings: Settings,
):
    """If the repo tree fetch fails, the handler still works."""
    mock_forge = AsyncMock()
    mock_forge.get_issue_comments.return_value = []
    mock_forge.get_repo_tree.side_effect = RuntimeError("API down")
    mock_forge.get_file_content.side_effect = RuntimeError("API down")
    mock_forge.post_comment.return_value = {"id": 10}

    handler = IssueCommentHandler(mock_forge, mock_llm, settings, "forge-bot")
    await handler.handle(comment_event)

    # Should still post a reply.
    mock_forge.post_comment.assert_awaited_once()
    mock_llm.chat.assert_awaited_once()


async def test_handle_includes_attachments_in_prompt(
    mock_llm: AsyncMock,
    settings: Settings,
):
    """Attachments in the issue body are surfaced in the system prompt."""
    payload = {
        "action": "created",
        "comment": {
            "id": 1,
            "body": "@forge-bot what does this screenshot show?",
            "user": {"id": 1, "login": "developer"},
        },
        "issue": {
            "number": 7,
            "title": "Bug with screenshot",
            "body": "See this:\n![error screenshot](/attachments/abc-123/screenshot.png)",
            "pull_request": None,
            "assignees": [],
        },
        "is_pull": False,
        "repository": {"full_name": "owner/repo"},
        "sender": {"id": 1, "login": "developer"},
    }
    event = IssueCommentEvent.model_validate(payload)

    mock_forge = AsyncMock()
    mock_forge.get_issue_comments.return_value = []
    mock_forge.get_repo_tree.return_value = []
    mock_forge.post_comment.return_value = {"id": 10}

    handler = IssueCommentHandler(mock_forge, mock_llm, settings, "forge-bot")
    await handler.handle(event)

    system_prompt = mock_llm.chat.call_args.args[0]
    assert "screenshot.png" in system_prompt
    assert "https://gitea.example.com/attachments/abc-123/screenshot.png" in system_prompt


# --- Unit tests for helper functions ---


def test_extract_file_paths_from_text():
    text = "Look at forge_bot/config.py and also src/main.py for details."
    paths = _extract_file_paths(text)
    assert "forge_bot/config.py" in paths
    assert "src/main.py" in paths


def test_extract_file_paths_in_backticks():
    text = "Check `forge_bot/server.py` for the webhook endpoint."
    paths = _extract_file_paths(text)
    assert "forge_bot/server.py" in paths


def test_extract_file_paths_deduplicates():
    text = "See config.py and also config.py again."
    paths = _extract_file_paths(text)
    assert paths.count("config.py") == 1


def test_extract_file_paths_no_matches():
    text = "This text has no file paths at all."
    assert _extract_file_paths(text) == []


def test_extract_attachments():
    text = "Here:\n![my image](/attachments/uuid-1/photo.png)\nand more"
    results = _extract_attachments(text, "https://gitea.example.com")
    assert len(results) == 1
    assert results[0]["alt"] == "my image"
    assert results[0]["url"] == "https://gitea.example.com/attachments/uuid-1/photo.png"


def test_extract_attachments_empty_alt():
    text = "![](/attachments/uuid-2/file.pdf)"
    results = _extract_attachments(text, "https://gitea.example.com")
    assert len(results) == 1
    assert results[0]["alt"] == "attachment"


def test_extract_attachments_none():
    text = "No attachments here."
    assert _extract_attachments(text, "https://gitea.example.com") == []
