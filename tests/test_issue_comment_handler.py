"""Tests for forge_bot.handlers.issue_comment."""

from unittest.mock import AsyncMock, MagicMock

import httpx
import pytest

from forge_bot.config import Settings
from forge_bot.handlers.issue_comment import (
    IssueCommentHandler,
    _context_limits,
    _extract_attachments,
    _extract_commit_shas,
    _extract_file_paths,
    _get_extension,
    _models_without_tool_support,
)
from forge_bot.models import IssueCommentEvent

# --- Helpers ---


def _mock_tool_response(content: str, tool_calls=None):
    """Build a mock ChatCompletion for chat_with_tools().

    When tool_calls is None/empty, the handler treats it as a final answer.
    """
    message = MagicMock()
    message.content = content
    message.tool_calls = tool_calls
    choice = MagicMock()
    choice.message = message
    resp = MagicMock()
    resp.choices = [choice]
    return resp


def _mock_tool_call(name: str, arguments: str, call_id: str = "call_1"):
    """Build a mock tool_call object (as returned by OpenAI)."""
    tc = MagicMock()
    tc.id = call_id
    tc.function.name = name
    tc.function.arguments = arguments
    return tc


# --- Fixtures ---


@pytest.fixture(autouse=True)
def _clear_tool_support_cache():
    """Ensure module-level tool-support cache is clean for each test."""
    _models_without_tool_support.clear()
    yield
    _models_without_tool_support.clear()


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
    # By default, file content fetch fails (no files referenced in default fixture).
    forge.get_file_content.return_value = "# Project\nSample content."
    forge.get_commit.side_effect = httpx.HTTPStatusError(
        "Not Found",
        request=httpx.Request("GET", "http://x"),
        response=httpx.Response(404),
    )
    forge.download_url.side_effect = httpx.HTTPStatusError(
        "Not Found",
        request=httpx.Request("GET", "http://x"),
        response=httpx.Response(404),
    )
    return forge


@pytest.fixture
def mock_llm():
    llm = AsyncMock()
    # chat() is still used for conversation summarization.
    llm.chat.return_value = "Summary of earlier discussion."
    # chat_with_tools() is used for the main response loop.
    llm.chat_with_tools.return_value = _mock_tool_response(
        "Here is my explanation of the auth flow."
    )
    return llm


def _make_event(
    comment_body: str = "@forge-bot hello",
    issue_body: str = "",
    default_branch: str = "main",
    is_pull: bool = False,
) -> IssueCommentEvent:
    """Helper to create a minimal IssueCommentEvent."""
    return IssueCommentEvent.model_validate({
        "action": "created",
        "comment": {
            "id": 1,
            "body": comment_body,
            "user": {"id": 1, "login": "developer"},
        },
        "issue": {
            "number": 5,
            "title": "Question",
            "body": issue_body,
            "pull_request": {"id": 1} if is_pull else None,
            "assignees": [],
        },
        "is_pull": is_pull,
        "repository": {
            "full_name": "owner/repo",
            "default_branch": default_branch,
        },
        "sender": {"id": 1, "login": "developer"},
    })


def _get_system_prompt(mock_llm: AsyncMock) -> str:
    """Extract the system prompt from the last chat_with_tools call."""
    messages = mock_llm.chat_with_tools.call_args.args[0]
    return messages[0]["content"]


def _get_user_message(mock_llm: AsyncMock) -> str:
    """Extract the user message from the last chat_with_tools call."""
    messages = mock_llm.chat_with_tools.call_args.args[0]
    return messages[1]["content"]


# --- Core handler tests ---


async def test_handle_fetches_comments_and_posts_reply(
    comment_event: IssueCommentEvent,
    mock_forge: AsyncMock,
    mock_llm: AsyncMock,
    settings: Settings,
):
    handler = IssueCommentHandler(mock_forge, mock_llm, settings, "forge-bot")
    await handler.handle(comment_event)

    mock_forge.get_issue_comments.assert_awaited_once_with("owner", "repo", 5)
    mock_llm.chat_with_tools.assert_awaited_once()

    system_prompt = _get_system_prompt(mock_llm)
    user_message = _get_user_message(mock_llm)
    assert "owner/repo" in system_prompt
    assert "#5" in system_prompt
    assert user_message == "@forge-bot what does this function do?"

    mock_forge.post_comment.assert_awaited_once_with(
        "owner", "repo", 5, "Here is my explanation of the auth flow."
    )


async def test_handle_posts_error_message_on_llm_failure(
    comment_event: IssueCommentEvent,
    mock_forge: AsyncMock,
    settings: Settings,
):
    mock_llm = AsyncMock()
    mock_llm.chat_with_tools.side_effect = RuntimeError("LLM is down")

    handler = IssueCommentHandler(mock_forge, mock_llm, settings, "forge-bot")
    await handler.handle(comment_event)

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
    mock_forge.get_repo_tree.return_value = []
    mock_forge.post_comment.side_effect = RuntimeError("API error")

    handler = IssueCommentHandler(mock_forge, mock_llm, settings, "forge-bot")
    await handler.handle(comment_event)
    mock_forge.post_comment.assert_awaited_once()


# --- Repo tree tests ---


async def test_handle_includes_repo_tree_in_prompt(
    comment_event: IssueCommentEvent,
    mock_forge: AsyncMock,
    mock_llm: AsyncMock,
    settings: Settings,
):
    handler = IssueCommentHandler(mock_forge, mock_llm, settings, "forge-bot")
    await handler.handle(comment_event)

    # default_branch in sample_comment_payload is "master"
    mock_forge.get_repo_tree.assert_awaited_once_with(
        "owner", "repo", ref="master"
    )

    system_prompt = _get_system_prompt(mock_llm)
    assert "src/main.py" in system_prompt
    assert "README.md" in system_prompt


async def test_fetch_repo_tree_uses_default_branch(
    mock_llm: AsyncMock,
    settings: Settings,
):
    """_fetch_repo_tree passes the repo's default_branch to the API."""
    event = _make_event(default_branch="develop")

    mock_forge = AsyncMock()
    mock_forge.get_issue_comments.return_value = []
    mock_forge.get_repo_tree.return_value = [
        {"path": "README.md", "type": "blob", "size": 50},
    ]
    mock_forge.get_file_content.return_value = "# Hello"
    mock_forge.post_comment.return_value = {"id": 10}

    handler = IssueCommentHandler(mock_forge, mock_llm, settings, "forge-bot")
    await handler.handle(event)

    mock_forge.get_repo_tree.assert_awaited_once_with(
        "owner", "repo", ref="develop"
    )


async def test_handle_gracefully_handles_tree_failure(
    comment_event: IssueCommentEvent,
    mock_llm: AsyncMock,
    settings: Settings,
):
    mock_forge = AsyncMock()
    mock_forge.get_issue_comments.return_value = []
    mock_forge.get_repo_tree.side_effect = RuntimeError("API down")
    mock_forge.get_file_content.side_effect = RuntimeError("API down")
    mock_forge.post_comment.return_value = {"id": 10}

    handler = IssueCommentHandler(mock_forge, mock_llm, settings, "forge-bot")
    await handler.handle(comment_event)

    mock_forge.post_comment.assert_awaited_once()
    mock_llm.chat_with_tools.assert_awaited_once()


# --- Grounding files tests ---


async def test_handle_fetches_grounding_files_when_no_paths_referenced(
    mock_llm: AsyncMock,
    settings: Settings,
):
    """When no file paths are mentioned, handler proactively fetches README.md etc."""
    event = _make_event(comment_body="@forge-bot what does this project do?")

    mock_forge = AsyncMock()
    mock_forge.get_issue_comments.return_value = []
    mock_forge.get_repo_tree.return_value = [
        {"path": "README.md", "type": "blob", "size": 500},
        {"path": "pyproject.toml", "type": "blob", "size": 200},
        {"path": "src/main.py", "type": "blob", "size": 100},
    ]
    mock_forge.get_file_content.return_value = "# My Project\nThis is a demo."
    mock_forge.post_comment.return_value = {"id": 10}

    handler = IssueCommentHandler(mock_forge, mock_llm, settings, "forge-bot")
    await handler.handle(event)

    file_content_calls = mock_forge.get_file_content.call_args_list
    fetched_paths = [call.args[2] for call in file_content_calls]
    assert "README.md" in fetched_paths

    system_prompt = _get_system_prompt(mock_llm)
    assert "My Project" in system_prompt


# --- File references tests ---


async def test_handle_fetches_referenced_files(
    mock_llm: AsyncMock,
    settings: Settings,
):
    event = _make_event(
        comment_body="@forge-bot can you explain forge_bot/config.py?"
    )

    mock_forge = AsyncMock()
    mock_forge.get_issue_comments.return_value = []
    mock_forge.get_repo_tree.return_value = []
    mock_forge.get_file_content.return_value = 'SECRET = "hello"\n'
    mock_forge.post_comment.return_value = {"id": 10}

    handler = IssueCommentHandler(mock_forge, mock_llm, settings, "forge-bot")
    await handler.handle(event)

    mock_forge.get_file_content.assert_any_await(
        "owner", "repo", "forge_bot/config.py", ref="main"
    )

    system_prompt = _get_system_prompt(mock_llm)
    assert "forge_bot/config.py" in system_prompt
    assert 'SECRET = "hello"' in system_prompt


# --- Commit tests ---


async def test_handle_fetches_referenced_commits(
    mock_llm: AsyncMock,
    settings: Settings,
):
    event = _make_event(
        comment_body="@forge-bot can you read commit 4a5bc21157?"
    )

    mock_forge = AsyncMock()
    mock_forge.get_issue_comments.return_value = []
    mock_forge.get_repo_tree.return_value = []
    mock_forge.get_commit.return_value = {
        "sha": "4a5bc21157abcdef1234567890abcdef12345678",
        "commit": {
            "message": "feat: add user authentication",
            "author": {
                "name": "Dev User",
                "date": "2026-01-15T10:00:00Z",
            },
        },
    }
    mock_forge.post_comment.return_value = {"id": 10}

    handler = IssueCommentHandler(mock_forge, mock_llm, settings, "forge-bot")
    await handler.handle(event)

    mock_forge.get_commit.assert_awaited_once_with(
        "owner", "repo", "4a5bc21157"
    )
    system_prompt = _get_system_prompt(mock_llm)
    assert "4a5bc21157" in system_prompt
    assert "add user authentication" in system_prompt


# --- Attachment tests ---


async def test_handle_includes_attachments_in_prompt(
    mock_llm: AsyncMock,
    settings: Settings,
):
    event = _make_event(
        comment_body="@forge-bot what does this screenshot show?",
        issue_body="See:\n![error](/attachments/abc-123/screenshot.png)",
    )

    mock_forge = AsyncMock()
    mock_forge.get_issue_comments.return_value = []
    mock_forge.get_repo_tree.return_value = []
    mock_forge.post_comment.return_value = {"id": 10}

    handler = IssueCommentHandler(mock_forge, mock_llm, settings, "forge-bot")
    await handler.handle(event)

    system_prompt = _get_system_prompt(mock_llm)
    assert "screenshot.png" in system_prompt


async def test_handle_downloads_text_attachments(
    mock_llm: AsyncMock,
    settings: Settings,
):
    event = _make_event(
        comment_body="@forge-bot review this doc",
        issue_body="Please review:\n![tdd](/attachments/uuid-1/tdd.md)",
    )

    mock_forge = AsyncMock()
    mock_forge.get_issue_comments.return_value = []
    mock_forge.get_repo_tree.return_value = []
    mock_forge.download_url.return_value = "# TDD Plan\nWrite tests first."
    mock_forge.post_comment.return_value = {"id": 10}

    handler = IssueCommentHandler(mock_forge, mock_llm, settings, "forge-bot")
    await handler.handle(event)

    mock_forge.download_url.assert_awaited_once()
    system_prompt = _get_system_prompt(mock_llm)
    assert "TDD Plan" in system_prompt


async def test_handle_survives_attachment_download_failure(
    mock_llm: AsyncMock,
    settings: Settings,
):
    event = _make_event(
        comment_body="@forge-bot check this",
        issue_body="![doc](/attachments/uuid/spec.md)",
    )

    mock_forge = AsyncMock()
    mock_forge.get_issue_comments.return_value = []
    mock_forge.get_repo_tree.return_value = []
    mock_forge.download_url.side_effect = RuntimeError("Network error")
    mock_forge.post_comment.return_value = {"id": 10}

    handler = IssueCommentHandler(mock_forge, mock_llm, settings, "forge-bot")
    await handler.handle(event)

    mock_llm.chat_with_tools.assert_awaited_once()
    mock_forge.post_comment.assert_awaited_once()


async def test_handle_truncates_long_bot_comments(
    mock_llm: AsyncMock,
    settings: Settings,
):
    """Long bot responses in history should be truncated to save context."""
    event = _make_event(comment_body="@forge-bot follow up question")
    long_bot_reply = "x" * 2000  # Way over _MAX_BOT_COMMENT_CHARS (500)

    mock_forge = AsyncMock()
    mock_forge.get_issue_comments.return_value = [
        {
            "id": 1,
            "body": "First question",
            "user": {"login": "developer"},
            "created_at": "2026-01-01T00:00:00Z",
        },
        {
            "id": 2,
            "body": long_bot_reply,
            "user": {"login": "forge-bot"},
            "created_at": "2026-01-01T00:01:00Z",
        },
    ]
    mock_forge.get_repo_tree.return_value = []
    mock_forge.post_comment.return_value = {"id": 10}

    handler = IssueCommentHandler(mock_forge, mock_llm, settings, "forge-bot")
    await handler.handle(event)

    system_prompt = _get_system_prompt(mock_llm)
    # The full 2000-char bot response should NOT appear.
    assert long_bot_reply not in system_prompt
    # The truncation marker should be present.
    assert "(response truncated)" in system_prompt


async def test_handle_preserves_short_bot_comments(
    mock_llm: AsyncMock,
    settings: Settings,
):
    """Short bot responses should NOT be truncated."""
    event = _make_event(comment_body="@forge-bot follow up")
    short_bot_reply = "Here is my short answer."

    mock_forge = AsyncMock()
    mock_forge.get_issue_comments.return_value = [
        {
            "id": 1,
            "body": short_bot_reply,
            "user": {"login": "forge-bot"},
            "created_at": "2026-01-01T00:00:00Z",
        },
    ]
    mock_forge.get_repo_tree.return_value = []
    mock_forge.post_comment.return_value = {"id": 10}

    handler = IssueCommentHandler(mock_forge, mock_llm, settings, "forge-bot")
    await handler.handle(event)

    system_prompt = _get_system_prompt(mock_llm)
    assert short_bot_reply in system_prompt
    assert "(response truncated)" not in system_prompt


# --- PR context tests ---


async def test_handle_fetches_pr_diff_when_is_pull(
    mock_llm: AsyncMock,
    settings: Settings,
):
    """When comment is on a PR, the handler fetches diff and files."""
    event = _make_event(
        comment_body="@forge-bot review this PR", is_pull=True,
    )

    mock_forge = AsyncMock()
    mock_forge.get_issue_comments.return_value = []
    mock_forge.get_repo_tree.return_value = [
        {"path": "src/app.py", "type": "blob", "size": 100},
    ]
    mock_forge.get_file_content.return_value = "# content"
    mock_forge.get_pull_diff.return_value = (
        "diff --git a/src/app.py b/src/app.py\n"
        "--- a/src/app.py\n"
        "+++ b/src/app.py\n"
        "@@ -1 +1,2 @@\n"
        " old line\n"
        "+new line\n"
    )
    mock_forge.get_pull_files.return_value = [
        {"filename": "src/app.py", "additions": 1, "deletions": 0},
    ]
    mock_forge.post_comment.return_value = {"id": 10}

    handler = IssueCommentHandler(mock_forge, mock_llm, settings, "forge-bot")
    await handler.handle(event)

    mock_forge.get_pull_diff.assert_awaited_once_with("owner", "repo", 5)
    mock_forge.get_pull_files.assert_awaited_once_with("owner", "repo", 5)

    system_prompt = _get_system_prompt(mock_llm)
    assert "PULL REQUEST DIFF" in system_prompt
    assert "+new line" in system_prompt
    assert "src/app.py" in system_prompt


async def test_handle_no_pr_diff_for_regular_issue(
    mock_forge: AsyncMock,
    mock_llm: AsyncMock,
    settings: Settings,
):
    """Regular issues should NOT fetch PR diff."""
    event = _make_event(comment_body="@forge-bot hello")

    handler = IssueCommentHandler(mock_forge, mock_llm, settings, "forge-bot")
    await handler.handle(event)

    mock_forge.get_pull_diff.assert_not_awaited()
    mock_forge.get_pull_files.assert_not_awaited()

    system_prompt = _get_system_prompt(mock_llm)
    assert "PULL REQUEST DIFF" not in system_prompt


async def test_handle_pr_diff_truncated(
    mock_llm: AsyncMock,
    settings: Settings,
):
    """Large PR diffs should be truncated."""
    event = _make_event(
        comment_body="@forge-bot check this", is_pull=True,
    )

    mock_forge = AsyncMock()
    mock_forge.get_issue_comments.return_value = []
    mock_forge.get_repo_tree.return_value = []
    mock_forge.get_pull_diff.return_value = "x" * 100_000
    mock_forge.get_pull_files.return_value = []
    mock_forge.post_comment.return_value = {"id": 10}

    handler = IssueCommentHandler(mock_forge, mock_llm, settings, "forge-bot")
    await handler.handle(event)

    system_prompt = _get_system_prompt(mock_llm)
    assert "(diff truncated)" in system_prompt


async def test_handle_pr_diff_fetch_failure_graceful(
    mock_llm: AsyncMock,
    settings: Settings,
):
    """Failed PR diff fetch should not crash the handler."""
    event = _make_event(
        comment_body="@forge-bot review", is_pull=True,
    )

    mock_forge = AsyncMock()
    mock_forge.get_issue_comments.return_value = []
    mock_forge.get_repo_tree.return_value = []
    mock_forge.get_pull_diff.side_effect = RuntimeError("404")
    mock_forge.get_pull_files.side_effect = RuntimeError("404")
    mock_forge.post_comment.return_value = {"id": 10}

    handler = IssueCommentHandler(mock_forge, mock_llm, settings, "forge-bot")
    await handler.handle(event)

    # Should still post a reply (just without PR diff).
    mock_forge.post_comment.assert_awaited_once()
    system_prompt = _get_system_prompt(mock_llm)
    assert "PULL REQUEST DIFF" not in system_prompt


# --- Tool loop tests ---


async def test_tool_loop_native_fetches_file(
    settings: Settings,
):
    """In native mode, when LLM returns a tool_call, handler executes it and re-prompts."""
    event = _make_event(comment_body="@forge-bot explain the server")

    mock_forge = AsyncMock()
    mock_forge.get_issue_comments.return_value = []
    mock_forge.get_repo_tree.return_value = [
        {"path": "forge_bot/server.py", "type": "blob", "size": 200},
        {"path": "README.md", "type": "blob", "size": 50},
    ]
    mock_forge.get_file_content.return_value = "# server code\napp = FastAPI()"
    mock_forge.post_comment.return_value = {"id": 10}

    # First call: LLM requests fetch_file tool.
    tc = _mock_tool_call(
        "fetch_file", '{"path": "forge_bot/server.py"}', "call_1",
    )
    resp1 = _mock_tool_response(None, tool_calls=[tc])
    resp1.choices[0].message.model_dump.return_value = {
        "role": "assistant",
        "content": None,
        "tool_calls": [
            {
                "id": "call_1",
                "function": {
                    "name": "fetch_file",
                    "arguments": '{"path": "forge_bot/server.py"}',
                },
            },
        ],
    }
    # Second call: LLM gives final answer.
    resp2 = _mock_tool_response(
        "The server uses FastAPI to handle webhooks."
    )

    mock_llm = AsyncMock()
    mock_llm.chat_with_tools.side_effect = [resp1, resp2]

    handler = IssueCommentHandler(mock_forge, mock_llm, settings, "forge-bot")
    await handler.handle(event)

    # LLM should have been called twice.
    assert mock_llm.chat_with_tools.await_count == 2

    # Posted reply should be the final answer.
    posted_body = mock_forge.post_comment.call_args.args[3]
    assert "FastAPI" in posted_body


async def test_tool_loop_prompt_mode(
    settings: Settings,
    monkeypatch: pytest.MonkeyPatch,
):
    """In prompt mode, handler parses ```tool blocks and executes tools."""
    monkeypatch.setenv("LLM_TOOL_MODE", "prompt")
    prompt_settings = Settings()
    event = _make_event(comment_body="@forge-bot explain the config")

    mock_forge = AsyncMock()
    mock_forge.get_issue_comments.return_value = []
    mock_forge.get_repo_tree.return_value = [
        {"path": "config.py", "type": "blob", "size": 100},
    ]
    mock_forge.get_file_content.return_value = "KEY = 'value'"
    mock_forge.post_comment.return_value = {"id": 10}

    # First call: LLM returns a tool block.
    resp1 = _mock_tool_response(
        'I need to read the config.\n```tool\n'
        '{"name": "fetch_file", "arguments": {"path": "config.py"}}\n```'
    )
    # Second call: LLM gives final answer.
    resp2 = _mock_tool_response("The config defines KEY = 'value'.")

    mock_llm = AsyncMock()
    mock_llm.chat_with_tools.side_effect = [resp1, resp2]

    handler = IssueCommentHandler(
        mock_forge, mock_llm, prompt_settings, "forge-bot",
    )
    await handler.handle(event)

    assert mock_llm.chat_with_tools.await_count == 2

    # First call should have tool_descriptions in the system prompt.
    first_messages = mock_llm.chat_with_tools.call_args_list[0].args[0]
    system_prompt = first_messages[0]["content"]
    assert "AVAILABLE TOOLS" in system_prompt
    assert "fetch_file" in system_prompt

    # First call should NOT have tools kwarg (prompt mode).
    first_kwargs = mock_llm.chat_with_tools.call_args_list[0].kwargs
    assert "tools" not in first_kwargs

    posted_body = mock_forge.post_comment.call_args.args[3]
    assert "KEY" in posted_body


async def test_tool_loop_auto_fallback(
    settings: Settings,
):
    """Auto mode: if native call fails, falls back to prompt mode."""
    event = _make_event(comment_body="@forge-bot hello")

    mock_forge = AsyncMock()
    mock_forge.get_issue_comments.return_value = []
    mock_forge.get_repo_tree.return_value = []
    mock_forge.post_comment.return_value = {"id": 10}

    mock_llm = AsyncMock()
    # First call (native): fails with API error.
    # Second call (prompt fallback): succeeds.
    mock_llm.chat_with_tools.side_effect = [
        RuntimeError("tools parameter not supported"),
        _mock_tool_response("Here's my answer via prompt mode."),
    ]

    handler = IssueCommentHandler(mock_forge, mock_llm, settings, "forge-bot")
    await handler.handle(event)

    # Should have been called twice (native fail + prompt success).
    assert mock_llm.chat_with_tools.await_count == 2

    # Second call should NOT have tools kwarg (prompt fallback).
    second_kwargs = mock_llm.chat_with_tools.call_args_list[1].kwargs
    assert "tools" not in second_kwargs

    posted_body = mock_forge.post_comment.call_args.args[3]
    assert "prompt mode" in posted_body


async def test_tool_loop_auto_caches_unsupported_model(
    settings: Settings,
):
    """After auto-fallback, the model is remembered so native isn't retried."""
    event = _make_event(comment_body="@forge-bot hello")

    mock_forge = AsyncMock()
    mock_forge.get_issue_comments.return_value = []
    mock_forge.get_repo_tree.return_value = []
    mock_forge.post_comment.return_value = {"id": 10}

    mock_llm = AsyncMock()
    # First request: native fails, triggers fallback + caching.
    mock_llm.chat_with_tools.side_effect = [
        RuntimeError("tools parameter not supported"),
        _mock_tool_response("First answer."),
    ]

    handler = IssueCommentHandler(mock_forge, mock_llm, settings, "forge-bot")
    await handler.handle(event)
    assert settings.llm_model in _models_without_tool_support

    # Second request: should go straight to prompt mode (1 call, not 2).
    mock_llm.reset_mock()
    mock_llm.chat_with_tools.return_value = _mock_tool_response("Second answer.")
    await handler.handle(event)
    assert mock_llm.chat_with_tools.await_count == 1


async def test_tool_loop_max_rounds(
    settings: Settings,
):
    """The tool loop stops after _MAX_TOOL_ROUNDS."""
    event = _make_event(comment_body="@forge-bot investigate")

    mock_forge = AsyncMock()
    mock_forge.get_issue_comments.return_value = []
    mock_forge.get_repo_tree.return_value = [
        {"path": f"{c}.py", "type": "blob", "size": 10}
        for c in "abcdef"
    ]
    mock_forge.get_file_content.return_value = "code"
    mock_forge.post_comment.return_value = {"id": 10}

    def _make_fetch_response(path: str, call_id: str):
        tc = _mock_tool_call("fetch_file", f'{{"path": "{path}"}}', call_id)
        resp = _mock_tool_response(None, tool_calls=[tc])
        resp.choices[0].message.model_dump.return_value = {
            "role": "assistant",
            "content": None,
            "tool_calls": [{
                "id": call_id,
                "function": {
                    "name": "fetch_file",
                    "arguments": f'{{"path": "{path}"}}',
                },
            }],
        }
        return resp

    mock_llm = AsyncMock()
    # Every call requests more files — should stop after MAX_TOOL_ROUNDS (5).
    mock_llm.chat_with_tools.side_effect = [
        _make_fetch_response("a.py", "c1"),
        _make_fetch_response("b.py", "c2"),
        _make_fetch_response("c.py", "c3"),
        _make_fetch_response("d.py", "c4"),
        _make_fetch_response("e.py", "c5"),
    ]

    handler = IssueCommentHandler(mock_forge, mock_llm, settings, "forge-bot")
    await handler.handle(event)

    # Max rounds is 5, so should have 5 chat_with_tools calls.
    assert mock_llm.chat_with_tools.await_count == 5

    # Final reply should be the fallback message.
    posted_body = mock_forge.post_comment.call_args.args[3]
    assert "wasn't able to form" in posted_body


async def test_tool_loop_cleans_markers(
    settings: Settings,
):
    """Stray [FETCH:] and ```tool markers are cleaned from final reply."""
    event = _make_event(comment_body="@forge-bot help")

    mock_forge = AsyncMock()
    mock_forge.get_issue_comments.return_value = []
    mock_forge.get_repo_tree.return_value = []
    mock_forge.post_comment.return_value = {"id": 10}

    mock_llm = AsyncMock()
    mock_llm.chat_with_tools.return_value = _mock_tool_response(
        "Here's the answer.\n[FETCH: nonexistent.py]\nMore text."
    )

    handler = IssueCommentHandler(mock_forge, mock_llm, settings, "forge-bot")
    await handler.handle(event)

    posted_body = mock_forge.post_comment.call_args.args[3]
    assert "[FETCH:" not in posted_body
    assert "Here's the answer." in posted_body


async def test_tool_loop_fetch_file_with_ref(
    settings: Settings,
):
    """FetchFileTool correctly passes ref parameter."""
    event = _make_event(comment_body="@forge-bot compare configs")

    mock_forge = AsyncMock()
    mock_forge.get_issue_comments.return_value = []
    mock_forge.get_repo_tree.return_value = [
        {"path": "config.yaml", "type": "blob", "size": 100},
    ]
    mock_forge.get_file_content.return_value = "key: value"
    mock_forge.post_comment.return_value = {"id": 10}

    tc = _mock_tool_call(
        "fetch_file",
        '{"path": "config.yaml", "ref": "develop"}',
        "call_1",
    )
    resp1 = _mock_tool_response(None, tool_calls=[tc])
    resp1.choices[0].message.model_dump.return_value = {
        "role": "assistant",
        "content": None,
        "tool_calls": [{
            "id": "call_1",
            "function": {
                "name": "fetch_file",
                "arguments": '{"path": "config.yaml", "ref": "develop"}',
            },
        }],
    }
    resp2 = _mock_tool_response("The config differs on develop.")

    mock_llm = AsyncMock()
    mock_llm.chat_with_tools.side_effect = [resp1, resp2]

    handler = IssueCommentHandler(mock_forge, mock_llm, settings, "forge-bot")
    await handler.handle(event)

    # Should have fetched with ref="develop"
    file_content_calls = mock_forge.get_file_content.call_args_list
    fetch_call = [
        c for c in file_content_calls
        if c.args[2] == "config.yaml" and c.kwargs.get("ref") == "develop"
    ]
    assert len(fetch_call) == 1


async def test_tool_loop_search_code(
    settings: Settings,
):
    """SearchCodeTool is invoked and results are used."""
    event = _make_event(comment_body="@forge-bot where is auth defined?")

    mock_forge = AsyncMock()
    mock_forge.get_issue_comments.return_value = []
    mock_forge.get_repo_tree.return_value = [
        {"path": "src/auth.py", "type": "blob", "size": 100},
        {"path": "src/main.py", "type": "blob", "size": 100},
    ]
    mock_forge.get_file_content.return_value = "def authenticate(): pass"
    mock_forge.post_comment.return_value = {"id": 10}

    tc = _mock_tool_call(
        "search_code",
        '{"query": "authenticate"}',
        "call_1",
    )
    resp1 = _mock_tool_response(None, tool_calls=[tc])
    resp1.choices[0].message.model_dump.return_value = {
        "role": "assistant",
        "content": None,
        "tool_calls": [{
            "id": "call_1",
            "function": {
                "name": "search_code",
                "arguments": '{"query": "authenticate"}',
            },
        }],
    }
    resp2 = _mock_tool_response(
        "The authenticate function is defined in src/auth.py."
    )

    mock_llm = AsyncMock()
    mock_llm.chat_with_tools.side_effect = [resp1, resp2]

    handler = IssueCommentHandler(mock_forge, mock_llm, settings, "forge-bot")
    await handler.handle(event)

    posted_body = mock_forge.post_comment.call_args.args[3]
    assert "authenticate" in posted_body


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
    assert (
        results[0]["url"]
        == "https://gitea.example.com/attachments/uuid-1/photo.png"
    )


def test_extract_attachments_empty_alt():
    text = "![](/attachments/uuid-2/file.pdf)"
    results = _extract_attachments(text, "https://gitea.example.com")
    assert len(results) == 1
    assert results[0]["alt"] == "attachment"


def test_extract_attachments_none():
    text = "No attachments here."
    assert _extract_attachments(text, "https://gitea.example.com") == []


def test_extract_commit_shas_basic():
    text = "Check commit 4a5bc21157 for the fix."
    shas = _extract_commit_shas(text)
    assert "4a5bc21157" in shas


def test_extract_commit_shas_full_sha():
    text = "See 4a5bc21157abcdef1234567890abcdef12345678"
    shas = _extract_commit_shas(text)
    assert "4a5bc21157abcdef1234567890abcdef12345678" in shas


def test_extract_commit_shas_no_match():
    text = "This has no commits."
    assert _extract_commit_shas(text) == []


def test_extract_commit_shas_too_short():
    text = "abc12 is too short to be a SHA."
    assert _extract_commit_shas(text) == []


def test_get_extension():
    assert _get_extension("https://gitea.com/attachments/uuid/file.md") == ".md"
    assert _get_extension("https://gitea.com/att/file.PNG") == ".png"
    assert _get_extension("https://gitea.com/att/noext") == ""
    assert _get_extension("file.py?v=2") == ".py"


# --- Context limits tests ---


def test_context_limits_small_window():
    limits = _context_limits(2048)
    assert limits["max_recent_comments"] == 2
    assert limits["summary_max_tokens"] == 128
    assert limits["max_tree_entries"] == 51
    assert limits["max_file_chars"] == 4096
    assert limits["max_grounding_file_chars"] == 2048
    assert limits["max_pr_diff_chars"] == 8192


def test_context_limits_large_window():
    limits = _context_limits(32768)
    assert limits["max_recent_comments"] == 10
    assert limits["summary_max_tokens"] == 512
    assert limits["max_tree_entries"] == 200
    assert limits["max_file_chars"] == 8000
    assert limits["max_grounding_file_chars"] == 4000
    assert limits["max_pr_diff_chars"] == 30_000


def test_context_limits_default_window():
    limits = _context_limits(8192)
    assert limits["max_recent_comments"] == 4
    assert limits["summary_max_tokens"] == 512
    assert limits["max_tree_entries"] == 200
    assert limits["max_pr_diff_chars"] == 30_000


# --- Conversation summarization tests ---


async def test_short_conversation_no_summarization(
    mock_llm: AsyncMock,
    settings: Settings,
):
    """When thread has <= max_recent_comments, no summarization occurs."""
    event = _make_event(comment_body="@forge-bot explain this")

    mock_forge = AsyncMock()
    mock_forge.get_issue_comments.return_value = [
        {
            "id": i,
            "body": f"Comment {i}",
            "user": {"login": "developer"},
            "created_at": f"2026-01-01T00:0{i}:00Z",
        }
        for i in range(3)
    ]
    mock_forge.get_repo_tree.return_value = []
    mock_forge.post_comment.return_value = {"id": 10}

    handler = IssueCommentHandler(mock_forge, mock_llm, settings, "forge-bot")
    await handler.handle(event)

    # chat() should NOT be called (no summarization needed).
    mock_llm.chat.assert_not_awaited()
    # chat_with_tools() called once (main response).
    assert mock_llm.chat_with_tools.await_count == 1
    system_prompt = _get_system_prompt(mock_llm)
    assert "EARLIER CONVERSATION" not in system_prompt


async def test_long_conversation_triggers_summarization(
    settings: Settings,
):
    """When thread exceeds max_recent_comments, old comments are summarized."""
    event = _make_event(comment_body="@forge-bot summarize progress")

    mock_forge = AsyncMock()
    mock_forge.get_issue_comments.return_value = [
        {
            "id": i,
            "body": f"Comment number {i} about the project",
            "user": {"login": "developer" if i % 2 == 0 else "forge-bot"},
            "created_at": f"2026-01-01T00:{i:02d}:00Z",
        }
        for i in range(10)
    ]
    mock_forge.get_repo_tree.return_value = []
    mock_forge.post_comment.return_value = {"id": 10}

    mock_llm = AsyncMock()
    # chat() for summarization:
    mock_llm.chat.return_value = (
        "The conversation covered project setup and configuration."
    )
    # chat_with_tools() for main response:
    mock_llm.chat_with_tools.return_value = _mock_tool_response(
        "Here is my response based on the context."
    )

    handler = IssueCommentHandler(mock_forge, mock_llm, settings, "forge-bot")
    await handler.handle(event)

    # chat() called once for summarization.
    mock_llm.chat.assert_awaited_once()
    # chat_with_tools() called once for main response.
    mock_llm.chat_with_tools.assert_awaited_once()

    # Summarization call.
    summary_system = mock_llm.chat.call_args.args[0]
    assert "CONVERSATION TO SUMMARIZE" in summary_system
    assert "forge-bot" in summary_system

    # Main response includes summary.
    main_system = _get_system_prompt(mock_llm)
    assert "EARLIER CONVERSATION" in main_system
    assert "project setup and configuration" in main_system


async def test_summarization_failure_falls_back(
    settings: Settings,
):
    """If summarization LLM call fails, handler falls back to truncation."""
    event = _make_event(comment_body="@forge-bot help")

    mock_forge = AsyncMock()
    mock_forge.get_issue_comments.return_value = [
        {
            "id": i,
            "body": f"Comment {i}",
            "user": {"login": "developer"},
            "created_at": f"2026-01-01T00:{i:02d}:00Z",
        }
        for i in range(10)
    ]
    mock_forge.get_repo_tree.return_value = []
    mock_forge.post_comment.return_value = {"id": 10}

    mock_llm = AsyncMock()
    # chat() for summarization fails:
    mock_llm.chat.side_effect = RuntimeError("LLM is down")
    # chat_with_tools() for main response succeeds:
    mock_llm.chat_with_tools.return_value = _mock_tool_response(
        "Here is my response."
    )

    handler = IssueCommentHandler(mock_forge, mock_llm, settings, "forge-bot")
    await handler.handle(event)

    # Should still post a reply (fell back on truncation summary).
    mock_forge.post_comment.assert_awaited_once()
    # The main call should have a fallback summary in it.
    main_system = _get_system_prompt(mock_llm)
    assert "EARLIER CONVERSATION" in main_system


# --- Prompt structure tests ---


async def test_prompt_structure_ground_truth_after_conversation(
    mock_llm: AsyncMock,
    settings: Settings,
):
    """Ground truth sections (file tree, file contents) appear after conversation."""
    event = _make_event(comment_body="@forge-bot what is this project?")

    mock_forge = AsyncMock()
    mock_forge.get_issue_comments.return_value = [
        {
            "id": 1,
            "body": "Some comment",
            "user": {"login": "developer"},
            "created_at": "2026-01-01T00:00:00Z",
        },
    ]
    mock_forge.get_repo_tree.return_value = [
        {"path": "README.md", "type": "blob", "size": 50},
        {"path": "src/main.py", "type": "blob", "size": 100},
    ]
    mock_forge.get_file_content.return_value = "# My Project"
    mock_forge.post_comment.return_value = {"id": 10}

    handler = IssueCommentHandler(mock_forge, mock_llm, settings, "forge-bot")
    await handler.handle(event)

    system_prompt = _get_system_prompt(mock_llm)

    # Verify ordering: conversation before ground truth sections.
    conv_pos = system_prompt.index("=== RECENT CONVERSATION ===")
    tree_pos = system_prompt.index("=== REPOSITORY FILE TREE")
    file_pos = system_prompt.index("=== FILE CONTENTS")

    assert conv_pos < tree_pos, "Conversation should appear before file tree"
    assert tree_pos < file_pos, "File tree should appear before file contents"


async def test_prompt_contains_anti_hallucination_rules(
    mock_llm: AsyncMock,
    settings: Settings,
):
    """The system prompt includes explicit anti-hallucination instructions."""
    event = _make_event(comment_body="@forge-bot hello")

    mock_forge = AsyncMock()
    mock_forge.get_issue_comments.return_value = []
    mock_forge.get_repo_tree.return_value = []
    mock_forge.post_comment.return_value = {"id": 10}

    handler = IssueCommentHandler(mock_forge, mock_llm, settings, "forge-bot")
    await handler.handle(event)

    system_prompt = _get_system_prompt(mock_llm)
    assert "MANDATORY RULES" in system_prompt
    assert "do NOT guess" in system_prompt
    assert "NEVER" in system_prompt
    assert "simulating" in system_prompt


async def test_summarization_prompt_filters_bot_claims(
    settings: Settings,
):
    """The conversation_summary.j2 template warns about bot hallucinations."""
    event = _make_event(comment_body="@forge-bot update")

    mock_forge = AsyncMock()
    mock_forge.get_issue_comments.return_value = [
        {
            "id": i,
            "body": f"Comment {i}",
            "user": {"login": "developer"},
            "created_at": f"2026-01-01T00:{i:02d}:00Z",
        }
        for i in range(10)
    ]
    mock_forge.get_repo_tree.return_value = []
    mock_forge.post_comment.return_value = {"id": 10}

    mock_llm = AsyncMock()
    mock_llm.chat.return_value = "Users discussed configuration changes."
    mock_llm.chat_with_tools.return_value = _mock_tool_response(
        "Here is my response."
    )

    handler = IssueCommentHandler(mock_forge, mock_llm, settings, "forge-bot")
    await handler.handle(event)

    # The summarization call's system prompt should mention hallucination risk.
    summary_system = mock_llm.chat.call_args.args[0]
    assert "hallucinated" in summary_system.lower()
    assert "forge-bot" in summary_system
