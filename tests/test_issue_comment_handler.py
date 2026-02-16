"""Tests for the rewritten IssueCommentHandler."""

from __future__ import annotations

import json
from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from forge_bot.handlers.issue_comment import IssueCommentHandler
from forge_bot.tools.base import ToolResult


# -- Helpers --


def _make_event(
    *,
    body: str = "Hey @bot what does this repo do?",
    repo_full_name: str = "owner/repo",
    issue_number: int = 1,
    default_branch: str = "main",
    clone_url: str = "https://gitea.example.com/owner/repo.git",
    sender: str = "alice",
    issue_title: str = "Test issue",
) -> Any:
    """Build a minimal IssueCommentEvent-like object for testing."""
    return SimpleNamespace(
        action="created",
        comment=SimpleNamespace(body=body, id=100),
        issue=SimpleNamespace(number=issue_number, title=issue_title),
        repository=SimpleNamespace(
            full_name=repo_full_name,
            default_branch=default_branch,
            clone_url=clone_url,
        ),
        sender=SimpleNamespace(login=sender),
    )


def _make_llm_response(
    *,
    content: str = "Here is my answer.",
    tool_calls: list[dict] | None = None,
) -> Any:
    """Build an object mimicking an OpenAI ChatCompletion response."""
    message = SimpleNamespace(content=content, tool_calls=None)

    if tool_calls:
        tc_list = []
        for tc in tool_calls:
            args = tc.get("arguments", {})
            if isinstance(args, dict):
                args = json.dumps(args)
            tc_list.append(
                SimpleNamespace(
                    function=SimpleNamespace(
                        name=tc["name"],
                        arguments=args,
                    )
                )
            )
        message.tool_calls = tc_list
        message.content = None

    return SimpleNamespace(choices=[SimpleNamespace(message=message)])


def _make_handler() -> tuple[IssueCommentHandler, MagicMock, AsyncMock]:
    """Create handler with mocked dependencies."""
    api = MagicMock()
    api.search.return_value = []
    api.call = AsyncMock(return_value={})

    llm = AsyncMock()
    llm.chat = AsyncMock(return_value="Fallback response")
    llm.chat_with_tools = AsyncMock(
        return_value=_make_llm_response(content="Here is my answer.")
    )

    settings = MagicMock()
    settings.forge_api_token = "test-token"
    settings.container_network_enabled = True

    handler = IssueCommentHandler(
        api_client=api,
        llm_client=llm,
        settings=settings,
        bot_username="bot",
    )
    return handler, api, llm


# -- _extract_tool_calls --


class TestExtractToolCalls:
    def test_openai_native_tool_calls(self) -> None:
        response = _make_llm_response(
            tool_calls=[
                {"name": "exec", "arguments": {"command": "ls"}},
                {"name": "search_api", "arguments": {"keyword": "file"}},
            ]
        )
        calls = IssueCommentHandler._extract_tool_calls(response)
        assert len(calls) == 2
        assert calls[0]["name"] == "exec"
        assert calls[0]["arguments"]["command"] == "ls"
        assert calls[1]["name"] == "search_api"

    def test_openai_native_string_arguments(self) -> None:
        """Arguments can come as JSON strings from some providers."""
        response = _make_llm_response(
            tool_calls=[
                {"name": "exec", "arguments": '{"command": "pwd"}'},
            ]
        )
        calls = IssueCommentHandler._extract_tool_calls(response)
        assert len(calls) == 1
        assert calls[0]["arguments"]["command"] == "pwd"

    def test_openai_native_invalid_json_args(self) -> None:
        response = _make_llm_response(
            tool_calls=[
                {"name": "exec", "arguments": "not json"},
            ]
        )
        calls = IssueCommentHandler._extract_tool_calls(response)
        assert len(calls) == 1
        assert calls[0]["arguments"] == {"raw": "not json"}

    def test_prompt_mode_tool_blocks(self) -> None:
        text = (
            "Let me run a command.\n"
            "```tool\n"
            '{"name": "exec", "arguments": {"command": "ls"}}\n'
            "```\n"
            "And another:\n"
            "```tool\n"
            '{"name": "search_api", "arguments": {"keyword": "pull"}}\n'
            "```\n"
        )
        calls = IssueCommentHandler._extract_tool_calls(text)
        assert len(calls) == 2
        assert calls[0]["name"] == "exec"
        assert calls[1]["name"] == "search_api"

    def test_prompt_mode_invalid_json_skipped(self) -> None:
        text = "```tool\nnot json\n```\n"
        calls = IssueCommentHandler._extract_tool_calls(text)
        assert len(calls) == 0

    def test_prompt_mode_no_name_skipped(self) -> None:
        text = '```tool\n{"arguments": {"x": 1}}\n```\n'
        calls = IssueCommentHandler._extract_tool_calls(text)
        assert len(calls) == 0

    def test_no_tool_calls(self) -> None:
        response = _make_llm_response(content="Just text, no tools.")
        calls = IssueCommentHandler._extract_tool_calls(response)
        assert len(calls) == 0


# -- _extract_text --


class TestExtractText:
    def test_openai_response(self) -> None:
        response = _make_llm_response(content="Hello!")
        assert IssueCommentHandler._extract_text(response) == "Hello!"

    def test_string_response(self) -> None:
        assert IssueCommentHandler._extract_text("plain text") == "plain text"

    def test_empty_content(self) -> None:
        response = _make_llm_response(content="")
        assert IssueCommentHandler._extract_text(response) == ""


# -- _build_system_prompt --


class TestBuildSystemPrompt:
    def test_renders_template(self) -> None:
        handler, _, _ = _make_handler()
        event = _make_event()

        registry = MagicMock()
        registry.prompt_text.return_value = "## Tools\n- exec: run commands"

        prompt = handler._build_system_prompt(
            event, registry,
            clone_url="https://token@gitea.example.com/owner/repo.git",
            default_branch="main",
        )
        assert "owner/repo" in prompt
        assert "bot" in prompt
        assert "exec" in prompt
        assert "NEVER" in prompt
        assert "clone" in prompt.lower()
        assert "main" in prompt


# -- _tool_loop --


class TestToolLoop:
    @pytest.mark.asyncio
    async def test_simple_text_response(self) -> None:
        """LLM returns text without tool calls -> immediate return."""
        handler, api, llm = _make_handler()
        event = _make_event()

        llm.chat_with_tools.return_value = _make_llm_response(
            content="This repo is a webhook bot."
        )

        status = MagicMock()
        status.post_initial_status = AsyncMock()
        status.update_phase = AsyncMock()
        status.record_tool_call = AsyncMock()

        from forge_bot.tools.registry import ToolRegistry
        from forge_bot.tools.search_api import SearchApiTool

        registry = ToolRegistry()
        registry.register(SearchApiTool(api))

        reply, results = await handler._tool_loop(
            event=event, registry=registry, status=status,
            clone_url="https://token@gitea.example.com/owner/repo.git",
            default_branch="main",
        )
        assert reply == "This repo is a webhook bot."
        assert results == []

    @pytest.mark.asyncio
    async def test_tool_call_then_response(self) -> None:
        """LLM makes a tool call, gets result, then responds with text."""
        handler, api, llm = _make_handler()
        event = _make_event()

        llm.chat_with_tools.side_effect = [
            _make_llm_response(
                tool_calls=[
                    {"name": "search_api", "arguments": {"keyword": "file"}},
                ]
            ),
            _make_llm_response(content="I found some file endpoints."),
        ]

        status = MagicMock()
        status.update_phase = AsyncMock()
        status.record_tool_call = AsyncMock()

        from forge_bot.tools.registry import ToolRegistry
        from forge_bot.tools.search_api import SearchApiTool

        registry = ToolRegistry()
        registry.register(SearchApiTool(api))

        reply, results = await handler._tool_loop(
            event=event, registry=registry, status=status,
            clone_url="https://token@gitea.example.com/owner/repo.git",
            default_branch="main",
        )
        assert reply == "I found some file endpoints."
        assert len(results) == 1
        assert results[0][0] == "search_api"

    @pytest.mark.asyncio
    async def test_llm_error_fallback_to_chat(self) -> None:
        """If chat_with_tools fails, falls back to simple chat."""
        handler, api, llm = _make_handler()
        event = _make_event()

        llm.chat_with_tools.side_effect = Exception("API error")
        llm.chat.return_value = "Fallback answer"

        status = MagicMock()
        status.update_phase = AsyncMock()
        status.record_tool_call = AsyncMock()

        from forge_bot.tools.registry import ToolRegistry
        from forge_bot.tools.search_api import SearchApiTool

        registry = ToolRegistry()
        registry.register(SearchApiTool(api))

        reply, results = await handler._tool_loop(
            event=event, registry=registry, status=status,
            clone_url="https://token@gitea.example.com/owner/repo.git",
            default_branch="main",
        )
        assert reply == "Fallback answer"

    @pytest.mark.asyncio
    async def test_llm_total_failure(self) -> None:
        """If both chat_with_tools and chat fail, return error."""
        handler, api, llm = _make_handler()
        event = _make_event()

        llm.chat_with_tools.side_effect = Exception("API error")
        llm.chat.side_effect = Exception("Also failed")

        status = MagicMock()
        status.update_phase = AsyncMock()
        status.record_tool_call = AsyncMock()

        from forge_bot.tools.registry import ToolRegistry
        from forge_bot.tools.search_api import SearchApiTool

        registry = ToolRegistry()
        registry.register(SearchApiTool(api))

        reply, results = await handler._tool_loop(
            event=event, registry=registry, status=status,
            clone_url="https://token@gitea.example.com/owner/repo.git",
            default_branch="main",
        )
        assert "error" in reply.lower()

    @pytest.mark.asyncio
    async def test_duplicate_tool_call_skipped(self) -> None:
        """Duplicate tool calls should be skipped after 2 occurrences."""
        handler, api, llm = _make_handler()
        event = _make_event()

        same_call = {"name": "search_api", "arguments": {"keyword": "file"}}

        # LLM keeps requesting the same call, then eventually responds
        llm.chat_with_tools.side_effect = [
            _make_llm_response(tool_calls=[same_call]),
            _make_llm_response(tool_calls=[same_call]),
            _make_llm_response(tool_calls=[same_call]),
            _make_llm_response(content="Done."),
        ]

        status = MagicMock()
        status.update_phase = AsyncMock()
        status.record_tool_call = AsyncMock()

        from forge_bot.tools.registry import ToolRegistry
        from forge_bot.tools.search_api import SearchApiTool

        registry = ToolRegistry()
        registry.register(SearchApiTool(api))

        reply, results = await handler._tool_loop(
            event=event, registry=registry, status=status,
            clone_url="https://token@gitea.example.com/owner/repo.git",
            default_branch="main",
        )
        # Only the first 2 should execute (3rd is blocked as duplicate)
        assert len(results) == 2

    @pytest.mark.asyncio
    async def test_stuck_loop_breaks(self) -> None:
        """If LLM is stuck (no new calls), loop exits with fallback."""
        handler, api, llm = _make_handler()
        event = _make_event()

        same_call = {"name": "search_api", "arguments": {"keyword": "file"}}

        # All rounds produce the same call (which gets duplicated and skipped)
        responses = []
        for _ in range(12):
            responses.append(_make_llm_response(tool_calls=[same_call]))
        llm.chat_with_tools.side_effect = responses

        status = MagicMock()
        status.update_phase = AsyncMock()
        status.record_tool_call = AsyncMock()

        from forge_bot.tools.registry import ToolRegistry
        from forge_bot.tools.search_api import SearchApiTool

        registry = ToolRegistry()
        registry.register(SearchApiTool(api))

        reply, results = await handler._tool_loop(
            event=event, registry=registry, status=status,
            clone_url="https://token@gitea.example.com/owner/repo.git",
            default_branch="main",
        )
        # Should have broken out early due to stuck detection
        assert llm.chat_with_tools.call_count < 10


# -- _verify_and_maybe_retry --


class TestVerifyAndMaybeRetry:
    @pytest.mark.asyncio
    async def test_passes_clean_response(self) -> None:
        handler, _, llm = _make_handler()
        event = _make_event()
        status = MagicMock()
        status.update_phase = AsyncMock()

        result = await handler._verify_and_maybe_retry(
            "Everything looks good.",
            [],
            event,
            status,
        )
        assert result == "Everything looks good."
        llm.chat.assert_not_called()

    @pytest.mark.asyncio
    async def test_retries_on_hallucination(self) -> None:
        handler, _, llm = _make_handler()
        event = _make_event()
        status = MagicMock()
        status.update_phase = AsyncMock()

        # Retry produces a clean response
        llm.chat.return_value = "Tests actually failed with exit code 1."

        tool_results = [
            ("exec", ToolResult("exec", False, "Exit code: 1\nstdout:\nFAILED\n")),
        ]

        result = await handler._verify_and_maybe_retry(
            "All tests pass successfully.",
            tool_results,
            event,
            status,
        )
        llm.chat.assert_called_once()

    @pytest.mark.asyncio
    async def test_warning_banner_on_failed_retry(self) -> None:
        handler, _, llm = _make_handler()
        event = _make_event()
        status = MagicMock()
        status.update_phase = AsyncMock()

        # Retry also produces a bad response
        llm.chat.return_value = "Tests pass (still wrong)."

        tool_results = [
            ("exec", ToolResult("exec", False, "Exit code: 1\nstdout:\nFAILED\n")),
        ]

        result = await handler._verify_and_maybe_retry(
            "All tests pass.",
            tool_results,
            event,
            status,
        )
        assert ":warning:" in result
        assert "All tests pass." in result

    @pytest.mark.asyncio
    async def test_warning_banner_on_retry_error(self) -> None:
        handler, _, llm = _make_handler()
        event = _make_event()
        status = MagicMock()
        status.update_phase = AsyncMock()

        llm.chat.side_effect = Exception("LLM down")

        tool_results = [
            ("exec", ToolResult("exec", False, "Exit code: 1\nstdout:\nFAILED\n")),
        ]

        result = await handler._verify_and_maybe_retry(
            "All tests pass.",
            tool_results,
            event,
            status,
        )
        assert ":warning:" in result


# -- Full handle() flow --


class TestHandleFlow:
    @pytest.mark.asyncio
    async def test_happy_path(self) -> None:
        """Full handle() with mocked container and LLM."""
        handler, api, llm = _make_handler()
        event = _make_event()

        llm.chat_with_tools.return_value = _make_llm_response(
            content="This repo is a webhook bot."
        )

        mock_container = MagicMock()
        mock_container.create = AsyncMock()
        mock_container.destroy = AsyncMock()
        mock_container.exec = AsyncMock()
        mock_container.clone_url = "https://token@gitea.example.com/owner/repo.git"
        mock_container.default_branch = "main"

        mock_status = MagicMock()
        mock_status.post_initial_status = AsyncMock()
        mock_status.update_phase = AsyncMock()
        mock_status.record_tool_call = AsyncMock()
        mock_status.post_response = AsyncMock()
        mock_status.finalize_status = AsyncMock()

        with (
            patch(
                "forge_bot.handlers.issue_comment.ContainerManager",
                return_value=mock_container,
            ),
            patch(
                "forge_bot.handlers.issue_comment.StatusCommentManager",
                return_value=mock_status,
            ),
        ):
            await handler.handle(event)

        mock_status.post_initial_status.assert_called_once()
        mock_container.create.assert_called_once()
        mock_status.post_response.assert_called_once()
        mock_status.finalize_status.assert_called_once_with("Done")
        mock_container.destroy.assert_called_once()

    @pytest.mark.asyncio
    async def test_container_failure_posts_error(self) -> None:
        """If container creation fails, posts error and destroys."""
        handler, api, llm = _make_handler()
        event = _make_event()

        mock_container = MagicMock()
        mock_container.create = AsyncMock(side_effect=RuntimeError("Docker down"))
        mock_container.destroy = AsyncMock()
        mock_container.clone_url = "https://token@gitea.example.com/owner/repo.git"
        mock_container.default_branch = "main"

        mock_status = MagicMock()
        mock_status.post_initial_status = AsyncMock()
        mock_status.update_phase = AsyncMock()
        mock_status.post_response = AsyncMock()
        mock_status.finalize_status = AsyncMock()

        with (
            patch(
                "forge_bot.handlers.issue_comment.ContainerManager",
                return_value=mock_container,
            ),
            patch(
                "forge_bot.handlers.issue_comment.StatusCommentManager",
                return_value=mock_status,
            ),
        ):
            await handler.handle(event)

        mock_status.post_response.assert_called_once()
        response_text = mock_status.post_response.call_args[0][0]
        assert "couldn't" in response_text.lower() or "workspace" in response_text.lower()
        mock_container.destroy.assert_called_once()

    @pytest.mark.asyncio
    async def test_container_destroyed_on_exception(self) -> None:
        """Container is always destroyed even if processing fails."""
        handler, api, llm = _make_handler()
        event = _make_event()

        llm.chat_with_tools.side_effect = Exception("Unexpected error")
        llm.chat.side_effect = Exception("Also failed")

        mock_container = MagicMock()
        mock_container.create = AsyncMock()
        mock_container.destroy = AsyncMock()
        mock_container.clone_url = "https://token@gitea.example.com/owner/repo.git"
        mock_container.default_branch = "main"

        mock_status = MagicMock()
        mock_status.post_initial_status = AsyncMock()
        mock_status.update_phase = AsyncMock()
        mock_status.record_tool_call = AsyncMock()
        mock_status.post_response = AsyncMock()
        mock_status.finalize_status = AsyncMock()

        with (
            patch(
                "forge_bot.handlers.issue_comment.ContainerManager",
                return_value=mock_container,
            ),
            patch(
                "forge_bot.handlers.issue_comment.StatusCommentManager",
                return_value=mock_status,
            ),
        ):
            await handler.handle(event)

        mock_container.destroy.assert_called_once()

    @pytest.mark.asyncio
    async def test_status_comment_failure_non_fatal(self) -> None:
        """Failure to post initial status comment doesn't block processing."""
        handler, api, llm = _make_handler()
        event = _make_event()

        llm.chat_with_tools.return_value = _make_llm_response(
            content="My response."
        )

        mock_container = MagicMock()
        mock_container.create = AsyncMock()
        mock_container.destroy = AsyncMock()
        mock_container.clone_url = "https://token@gitea.example.com/owner/repo.git"
        mock_container.default_branch = "main"

        mock_status = MagicMock()
        mock_status.post_initial_status = AsyncMock(
            side_effect=RuntimeError("API down")
        )
        mock_status.update_phase = AsyncMock()
        mock_status.record_tool_call = AsyncMock()
        mock_status.post_response = AsyncMock()
        mock_status.finalize_status = AsyncMock()

        with (
            patch(
                "forge_bot.handlers.issue_comment.ContainerManager",
                return_value=mock_container,
            ),
            patch(
                "forge_bot.handlers.issue_comment.StatusCommentManager",
                return_value=mock_status,
            ),
        ):
            await handler.handle(event)

        # Should still post final response
        mock_status.post_response.assert_called_once()
