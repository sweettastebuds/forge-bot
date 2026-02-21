"""Tests for forge_bot.handlers.pull_request (tool-loop based review)."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from forge_bot.container.manager import ExecResult
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


# -- Mock helpers --


def _make_api(
    *,
    diff: str = SAMPLE_DIFF,
    files: list | None = None,
    diff_error: Exception | None = None,
    post_error: Exception | None = None,
) -> MagicMock:
    """Mock GenericForgeClient for PR handler tests."""
    api = MagicMock()
    _comment_id_counter = {"value": 100}

    async def mock_call(endpoint_name, **kwargs):
        if endpoint_name == "get_pull_diff":
            if diff_error:
                raise diff_error
            return diff
        if endpoint_name == "get_pull_files":
            return (
                files
                if files is not None
                else [
                    {"filename": "server.py", "additions": 1, "deletions": 0},
                ]
            )
        if endpoint_name == "post_issue_comment":
            if post_error:
                raise post_error
            cid = _comment_id_counter["value"]
            _comment_id_counter["value"] += 1
            return {"id": cid, "body": kwargs.get("body", "")}
        if endpoint_name == "edit_issue_comment":
            return {}
        return {}

    api.call = AsyncMock(side_effect=mock_call)
    return api


def _make_llm_with_tools(response_text: str = "LGTM, no major issues.") -> AsyncMock:
    """Mock LLM that returns text on the first tool-loop round (no tool calls)."""
    llm = AsyncMock()

    # chat_with_tools returns an OpenAI-like response object.
    mock_message = MagicMock()
    mock_message.content = response_text
    mock_message.tool_calls = None

    mock_choice = MagicMock()
    mock_choice.message = mock_message

    mock_response = MagicMock()
    mock_response.choices = [mock_choice]

    llm.chat_with_tools = AsyncMock(return_value=mock_response)
    llm.chat = AsyncMock(return_value=response_text)

    return llm


def _make_settings() -> MagicMock:
    settings = MagicMock()
    settings.rag_enabled = False
    settings.smart_retrieval_max_parallel = 10
    settings.llm_context_window = 8192
    settings.sandbox_timeout = 60
    settings.sandbox_memory = "512m"
    settings.sandbox_cpus = 1.0
    settings.container_workspace_image = "forge-bot-workspace:latest"
    settings.container_network_enabled = True
    settings.forge_api_token = "test-token"
    return settings


def _make_container() -> MagicMock:
    """Mock ContainerManager with async lifecycle and exec."""
    cm = MagicMock()
    cm.create = AsyncMock()
    cm.destroy = AsyncMock()
    cm.clone_url = "https://test-token@gitea.example.com/owner/repo.git"
    cm.default_branch = "main"
    cm.exec = AsyncMock(
        return_value=ExecResult(
            exit_code=0,
            stdout="ok\n",
            stderr="",
            duration_seconds=0.1,
            command="test",
        )
    )
    return cm


@pytest.fixture
def mock_container():
    return _make_container()


@pytest.fixture
def patch_container(mock_container, monkeypatch):
    """Patch ContainerManager so it returns the mock without creating a real container."""
    mock_cls = MagicMock(return_value=mock_container)
    monkeypatch.setattr(
        "forge_bot.handlers.base.ContainerManager",
        mock_cls,
    )
    return mock_container


# -- Tests: tool-loop integration --


async def test_handle_uses_tool_loop_not_direct_chat(
    pr_event: PullRequestEvent,
    patch_container,
):
    """PR handler should use the tool-calling loop (chat_with_tools),
    not a direct llm.chat call."""
    api = _make_api()
    llm = _make_llm_with_tools("Good code, LGTM!")

    handler = PullRequestHandler(api, llm, _make_settings(), "forge-bot")
    await handler.handle(pr_event)

    # chat_with_tools should be the primary LLM interface.
    llm.chat_with_tools.assert_awaited()

    # A review comment should have been posted.
    post_calls = [
        c for c in api.call.call_args_list if c.args[0] == "post_issue_comment"
    ]
    assert len(post_calls) >= 1
    bodies = [c.kwargs.get("body", "") for c in post_calls]
    assert any("LGTM" in b for b in bodies)


async def test_handle_creates_and_destroys_container(
    pr_event: PullRequestEvent,
    patch_container,
):
    """Container lifecycle: created before tool loop, destroyed after."""
    api = _make_api()
    llm = _make_llm_with_tools()

    handler = PullRequestHandler(api, llm, _make_settings(), "forge-bot")
    await handler.handle(pr_event)

    patch_container.create.assert_awaited_once()
    patch_container.destroy.assert_awaited_once()


async def test_handle_posts_status_comment(
    pr_event: PullRequestEvent,
    patch_container,
):
    """Handler should post a status comment for observability."""
    api = _make_api()
    llm = _make_llm_with_tools()

    handler = PullRequestHandler(api, llm, _make_settings(), "forge-bot")
    await handler.handle(pr_event)

    # At least 2 post_issue_comment calls: status + response.
    post_calls = [
        c for c in api.call.call_args_list if c.args[0] == "post_issue_comment"
    ]
    assert len(post_calls) >= 2


async def test_handle_includes_diff_in_user_message(
    pr_event: PullRequestEvent,
    patch_container,
):
    """The diff and file summary should appear in the LLM prompt."""
    api = _make_api()
    llm = _make_llm_with_tools()

    handler = PullRequestHandler(api, llm, _make_settings(), "forge-bot")
    await handler.handle(pr_event)

    # Inspect the messages passed to chat_with_tools.
    call_kwargs = llm.chat_with_tools.call_args
    messages = call_kwargs.kwargs.get("messages") or call_kwargs.args[0]

    # Find the user message.
    user_msgs = [m for m in messages if m.get("role") == "user"]
    assert user_msgs
    user_content = user_msgs[0]["content"]

    assert "diff --git" in user_content
    assert "server.py" in user_content
    assert "Test PR" in user_content


async def test_handle_includes_pr_description_in_prompt(
    pr_event: PullRequestEvent,
    patch_container,
):
    api = _make_api()
    llm = _make_llm_with_tools()

    handler = PullRequestHandler(api, llm, _make_settings(), "forge-bot")
    await handler.handle(pr_event)

    call_kwargs = llm.chat_with_tools.call_args
    messages = call_kwargs.kwargs.get("messages") or call_kwargs.args[0]
    user_msgs = [m for m in messages if m.get("role") == "user"]
    user_content = user_msgs[0]["content"]

    assert "Test description" in user_content
    assert "feature" in user_content  # head branch
    assert "main" in user_content  # base branch


async def test_handle_registers_tools_including_smart_search(
    pr_event: PullRequestEvent,
    patch_container,
):
    """The tool registry should include smart_search alongside exec, api_call, etc."""
    api = _make_api()
    llm = _make_llm_with_tools()

    handler = PullRequestHandler(api, llm, _make_settings(), "forge-bot")
    await handler.handle(pr_event)

    # Inspect the tools schemas passed to chat_with_tools.
    call_kwargs = llm.chat_with_tools.call_args
    tools = call_kwargs.kwargs.get("tools", [])

    tool_names = {t["function"]["name"] for t in tools}
    assert "exec" in tool_names
    assert "smart_search" in tool_names
    assert "api_call" in tool_names
    assert "search_api" in tool_names


async def test_handle_skips_review_on_empty_diff(
    pr_event: PullRequestEvent,
    patch_container,
):
    """No review when the diff is empty — no container, no tool loop."""
    api = _make_api(diff="")
    llm = _make_llm_with_tools()

    handler = PullRequestHandler(api, llm, _make_settings(), "forge-bot")
    await handler.handle(pr_event)

    llm.chat_with_tools.assert_not_awaited()
    # Container should not be created for empty diffs.
    patch_container.create.assert_not_awaited()


async def test_handle_survives_diff_fetch_failure(
    pr_event: PullRequestEvent,
    patch_container,
):
    api = _make_api(diff_error=RuntimeError("API error"))
    llm = _make_llm_with_tools()

    handler = PullRequestHandler(api, llm, _make_settings(), "forge-bot")
    await handler.handle(pr_event)

    # Empty diff → skip review.
    llm.chat_with_tools.assert_not_awaited()


async def test_handle_container_creation_failure_posts_error(
    pr_event: PullRequestEvent,
    monkeypatch,
):
    """If container creation fails, post an error comment instead of crashing."""
    api = _make_api()
    llm = _make_llm_with_tools()
    settings = _make_settings()

    failing_container = _make_container()
    failing_container.create = AsyncMock(side_effect=RuntimeError("Docker down"))

    mock_cls = MagicMock(return_value=failing_container)
    monkeypatch.setattr(
        "forge_bot.handlers.base.ContainerManager",
        mock_cls,
    )

    handler = PullRequestHandler(api, llm, settings, "forge-bot")
    await handler.handle(pr_event)

    # Should post an error comment.
    post_calls = [
        c for c in api.call.call_args_list if c.args[0] == "post_issue_comment"
    ]
    assert len(post_calls) >= 1
    bodies = [c.kwargs.get("body", "") for c in post_calls]
    assert any("error" in b.lower() for b in bodies)

    # Container should be destroyed even on failure.
    failing_container.destroy.assert_awaited_once()


async def test_handle_survives_post_comment_failure(
    pr_event: PullRequestEvent,
    patch_container,
):
    api = _make_api(post_error=RuntimeError("API error"))
    llm = _make_llm_with_tools()

    handler = PullRequestHandler(api, llm, _make_settings(), "forge-bot")
    # Should not raise.
    await handler.handle(pr_event)


async def test_handle_truncates_huge_diffs_in_message(
    pr_event: PullRequestEvent,
    patch_container,
):
    """Very large diffs should be truncated in the user message to fit context,
    with a note that the LLM can fetch the full diff via tools."""
    huge_diff = "diff --git a/big.py b/big.py\n" + "+x\n" * 50_000
    api = _make_api(diff=huge_diff)
    llm = _make_llm_with_tools()

    handler = PullRequestHandler(api, llm, _make_settings(), "forge-bot")
    await handler.handle(pr_event)

    call_kwargs = llm.chat_with_tools.call_args
    messages = call_kwargs.kwargs.get("messages") or call_kwargs.args[0]
    user_msgs = [m for m in messages if m.get("role") == "user"]
    user_content = user_msgs[0]["content"]

    # Content should be less than the full diff.
    assert len(user_content) < len(huge_diff)
    # Should hint that the diff was truncated.
    assert "truncat" in user_content.lower()


async def test_build_user_message_format(sample_pr_payload: dict):
    """Static helper should format PR info + diff correctly."""
    event = PullRequestEvent.model_validate(sample_pr_payload)
    msg = PullRequestHandler._build_user_message(
        event,
        "- server.py (+1/-0)",
        "diff --git a/file.py b/file.py",
    )
    assert "PR #1" in msg
    assert "Test PR" in msg
    assert "Test description" in msg
    assert "feature" in msg
    assert "main" in msg
    assert "server.py" in msg
    assert "diff --git" in msg


async def test_build_user_message_without_description(sample_pr_payload: dict):
    sample_pr_payload["pull_request"]["body"] = ""
    event = PullRequestEvent.model_validate(sample_pr_payload)
    msg = PullRequestHandler._build_user_message(event, "", "some diff")
    assert "Description" not in msg
