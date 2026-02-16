"""Tests for forge_bot.handlers.pull_request."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

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
def pr_event(sample_pr_payload: dict) -> PullRequestEvent:
    return PullRequestEvent.model_validate(sample_pr_payload)


def _make_api(
    *,
    diff: str = SAMPLE_DIFF,
    files: list | None = None,
    diff_error: Exception | None = None,
    post_error: Exception | None = None,
) -> MagicMock:
    """Mock GenericForgeClient for PR handler tests."""
    api = MagicMock()

    async def mock_call(endpoint_name, **kwargs):
        if endpoint_name == "get_pull_diff":
            if diff_error:
                raise diff_error
            return diff
        if endpoint_name == "get_pull_files":
            return files if files is not None else [
                {"filename": "server.py", "additions": 1, "deletions": 0},
            ]
        if endpoint_name == "post_issue_comment":
            if post_error:
                raise post_error
            return {"id": 99, "body": kwargs.get("body", "")}
        return {}

    api.call = AsyncMock(side_effect=mock_call)
    return api


def _make_llm(response: str = "Looks good — no major issues found.") -> AsyncMock:
    llm = AsyncMock()
    llm.chat.return_value = response
    return llm


def _make_settings() -> MagicMock:
    settings = MagicMock()
    settings.rag_enabled = False
    return settings


async def test_handle_fetches_diff_and_posts_review(
    pr_event: PullRequestEvent,
):
    api = _make_api()
    llm = _make_llm()
    handler = PullRequestHandler(api, llm, _make_settings(), "forge-bot")
    await handler.handle(pr_event)

    # Should have called get_pull_diff and get_pull_files
    call_names = [c.args[0] for c in api.call.call_args_list]
    assert "get_pull_diff" in call_names
    assert "get_pull_files" in call_names

    # LLM called with system prompt + user message
    llm.chat.assert_awaited_once()
    system_prompt = llm.chat.call_args.args[0]
    user_message = llm.chat.call_args.args[1]
    assert "owner/repo" in system_prompt
    assert "Test PR" in system_prompt
    assert "diff --git" in user_message
    assert "server.py" in user_message

    # Should post comment
    assert "post_issue_comment" in call_names


async def test_handle_includes_pr_description_in_message(
    pr_event: PullRequestEvent,
):
    api = _make_api()
    llm = _make_llm()
    handler = PullRequestHandler(api, llm, _make_settings(), "forge-bot")
    await handler.handle(pr_event)

    user_message = llm.chat.call_args.args[1]
    assert "Test description" in user_message
    assert "feature" in user_message  # head branch
    assert "main" in user_message  # base branch


async def test_handle_skips_review_on_empty_diff(
    pr_event: PullRequestEvent,
):
    api = _make_api(diff="")
    llm = _make_llm()
    handler = PullRequestHandler(api, llm, _make_settings(), "forge-bot")
    await handler.handle(pr_event)

    llm.chat.assert_not_awaited()


async def test_handle_truncates_large_diffs(
    pr_event: PullRequestEvent,
):
    api = _make_api(diff="x" * 50_000)
    llm = _make_llm()
    handler = PullRequestHandler(api, llm, _make_settings(), "forge-bot")
    await handler.handle(pr_event)

    user_message = llm.chat.call_args.args[1]
    assert "diff truncated" in user_message
    assert len(user_message) < 50_000


async def test_handle_posts_error_on_llm_failure(
    pr_event: PullRequestEvent,
):
    api = _make_api()
    llm = AsyncMock()
    llm.chat.side_effect = RuntimeError("LLM down")

    handler = PullRequestHandler(api, llm, _make_settings(), "forge-bot")
    await handler.handle(pr_event)

    # Should have posted an error comment
    post_calls = [
        c for c in api.call.call_args_list
        if c.args[0] == "post_issue_comment"
    ]
    assert len(post_calls) == 1
    assert "error" in post_calls[0].kwargs["body"].lower()


async def test_handle_survives_diff_fetch_failure(
    pr_event: PullRequestEvent,
):
    api = _make_api(diff_error=RuntimeError("API error"))
    llm = _make_llm()
    handler = PullRequestHandler(api, llm, _make_settings(), "forge-bot")
    await handler.handle(pr_event)

    # Empty diff → skip review entirely
    llm.chat.assert_not_awaited()


async def test_handle_survives_post_comment_failure(
    pr_event: PullRequestEvent,
):
    api = _make_api(post_error=RuntimeError("API error"))
    llm = _make_llm()
    handler = PullRequestHandler(api, llm, _make_settings(), "forge-bot")
    # Should not raise
    await handler.handle(pr_event)


async def test_build_user_message_without_description(
    sample_pr_payload: dict,
):
    sample_pr_payload["pull_request"]["body"] = ""
    event = PullRequestEvent.model_validate(sample_pr_payload)

    api = _make_api()
    llm = _make_llm()
    handler = PullRequestHandler(api, llm, _make_settings(), "forge-bot")
    await handler.handle(event)

    user_message = llm.chat.call_args.args[1]
    assert "Description" not in user_message
