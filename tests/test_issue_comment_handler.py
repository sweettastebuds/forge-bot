"""Tests for the IssueCommentHandler lifecycle."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from forge_bot.handlers.issue_comment import IssueCommentHandler

# -- Helpers -----------------------------------------------------------------


def _make_event(
    *,
    body: str = "Hey @bot what does this repo do?",
    repo_full_name: str = "owner/repo",
    issue_number: int = 1,
    default_branch: str = "main",
    clone_url: str = "https://gitea.example.com/owner/repo.git",
    sender: str = "alice",
    issue_title: str = "Test issue",
    issue_body: str = "",
    comment_id: int = 100,
) -> Any:
    return SimpleNamespace(
        action="created",
        comment=SimpleNamespace(body=body, id=comment_id),
        issue=SimpleNamespace(
            number=issue_number,
            title=issue_title,
            body=issue_body,
        ),
        repository=SimpleNamespace(
            full_name=repo_full_name,
            default_branch=default_branch,
            clone_url=clone_url,
        ),
        sender=SimpleNamespace(login=sender),
    )


def _make_handler() -> tuple[IssueCommentHandler, MagicMock, AsyncMock]:
    api = MagicMock()
    api.call = AsyncMock(return_value={"id": 1})

    llm = AsyncMock()
    llm.chat_with_tools = AsyncMock()
    llm.chat = AsyncMock(return_value="Fallback")

    settings = MagicMock()
    settings.forge_api_token = "test-token"
    settings.forge_instance_url = "https://gitea.example.com"
    settings.container_network_enabled = True
    settings.llm_context_window = 8192

    handler = IssueCommentHandler(
        api_client=api,
        llm_client=llm,
        settings=settings,
        bot_username="bot",
    )
    return handler, api, llm


def _mock_status() -> MagicMock:
    status = MagicMock()
    status.post_initial_status = AsyncMock()
    status.update_phase = AsyncMock()
    status.record_tool_call = AsyncMock()
    status.post_response = AsyncMock()
    status.finalize_status = AsyncMock()
    status.update_todos = AsyncMock()
    status.attach_file = AsyncMock()
    status.todos = []
    return status


def _mock_container() -> MagicMock:
    container = MagicMock()
    container.create = AsyncMock()
    container.destroy = AsyncMock()
    container.exec = AsyncMock()
    container.clone_url = "https://token@gitea.example.com/owner/repo.git"
    container.default_branch = "main"
    return container


def _patches():
    """Return context managers that patch SmartRetriever and RetrievalTool."""
    return (
        patch("forge_bot.handlers.issue_comment.SmartRetriever"),
        patch("forge_bot.handlers.issue_comment.RetrievalTool"),
    )


# -- Tests -------------------------------------------------------------------


class TestHandleFlow:
    @pytest.mark.asyncio
    async def test_happy_path(self) -> None:
        """Full handle(): status -> container -> agent -> post -> finalize -> destroy."""
        handler, api, llm = _make_handler()
        event = _make_event()
        mock_status = _mock_status()
        mock_container = _mock_container()

        mock_agent = MagicMock()
        mock_agent.run = AsyncMock(return_value="Here is my answer.")
        mock_agent.collect_artifacts = AsyncMock(return_value=[])

        p_retriever, p_tool = _patches()
        with (
            patch(
                "forge_bot.handlers.issue_comment.ContainerManager",
                return_value=mock_container,
            ),
            patch(
                "forge_bot.handlers.issue_comment.StatusCommentManager",
                return_value=mock_status,
            ),
            patch(
                "forge_bot.handlers.issue_comment.AgentLoop",
                return_value=mock_agent,
            ),
            p_retriever,
            p_tool,
        ):
            await handler.handle(event)

        mock_status.post_initial_status.assert_awaited_once()
        mock_container.create.assert_awaited_once()
        mock_agent.run.assert_awaited_once()
        mock_status.post_response.assert_awaited_once_with("Here is my answer.")
        mock_status.finalize_status.assert_awaited_once_with("Done")
        mock_container.destroy.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_container_failure_posts_error(self) -> None:
        """Container creation failure -> error posted -> container destroyed."""
        handler, _, _ = _make_handler()
        event = _make_event()
        mock_status = _mock_status()
        mock_container = _mock_container()
        mock_container.create = AsyncMock(side_effect=RuntimeError("Docker down"))

        p_retriever, p_tool = _patches()
        with (
            patch(
                "forge_bot.handlers.issue_comment.ContainerManager",
                return_value=mock_container,
            ),
            patch(
                "forge_bot.handlers.issue_comment.StatusCommentManager",
                return_value=mock_status,
            ),
            p_retriever,
            p_tool,
        ):
            await handler.handle(event)

        mock_status.post_response.assert_awaited_once()
        body = mock_status.post_response.call_args[0][0]
        assert "error" in body.lower()
        mock_container.destroy.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_container_destroyed_on_exception(self) -> None:
        """Container is always destroyed even if agent raises."""
        handler, _, _ = _make_handler()
        event = _make_event()
        mock_status = _mock_status()
        mock_container = _mock_container()

        mock_agent = MagicMock()
        mock_agent.run = AsyncMock(side_effect=RuntimeError("Unexpected"))

        p_retriever, p_tool = _patches()
        with (
            patch(
                "forge_bot.handlers.issue_comment.ContainerManager",
                return_value=mock_container,
            ),
            patch(
                "forge_bot.handlers.issue_comment.StatusCommentManager",
                return_value=mock_status,
            ),
            patch(
                "forge_bot.handlers.issue_comment.AgentLoop",
                return_value=mock_agent,
            ),
            p_retriever,
            p_tool,
        ):
            await handler.handle(event)

        mock_container.destroy.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_status_failure_non_fatal(self) -> None:
        """Failure to post initial status doesn't block processing."""
        handler, _, _ = _make_handler()
        event = _make_event()
        mock_status = _mock_status()
        mock_status.post_initial_status = AsyncMock(
            side_effect=RuntimeError("API down"),
        )
        mock_container = _mock_container()

        mock_agent = MagicMock()
        mock_agent.run = AsyncMock(return_value="My response.")
        mock_agent.collect_artifacts = AsyncMock(return_value=[])

        p_retriever, p_tool = _patches()
        with (
            patch(
                "forge_bot.handlers.issue_comment.ContainerManager",
                return_value=mock_container,
            ),
            patch(
                "forge_bot.handlers.issue_comment.StatusCommentManager",
                return_value=mock_status,
            ),
            patch(
                "forge_bot.handlers.issue_comment.AgentLoop",
                return_value=mock_agent,
            ),
            p_retriever,
            p_tool,
        ):
            await handler.handle(event)

        mock_status.post_response.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_artifacts_attached(self) -> None:
        """Artifacts from agent.collect_artifacts() are uploaded."""
        handler, _, _ = _make_handler()
        event = _make_event()
        mock_status = _mock_status()
        mock_container = _mock_container()

        mock_agent = MagicMock()
        mock_agent.run = AsyncMock(return_value="Done.")
        mock_agent.collect_artifacts = AsyncMock(
            return_value=[("report.txt", b"hello"), ("log.csv", b"a,b")]
        )

        p_retriever, p_tool = _patches()
        with (
            patch(
                "forge_bot.handlers.issue_comment.ContainerManager",
                return_value=mock_container,
            ),
            patch(
                "forge_bot.handlers.issue_comment.StatusCommentManager",
                return_value=mock_status,
            ),
            patch(
                "forge_bot.handlers.issue_comment.AgentLoop",
                return_value=mock_agent,
            ),
            p_retriever,
            p_tool,
        ):
            await handler.handle(event)

        assert mock_status.attach_file.await_count == 2
        mock_status.attach_file.assert_any_await("report.txt", b"hello")
        mock_status.attach_file.assert_any_await("log.csv", b"a,b")


class TestConversationContext:
    @pytest.mark.asyncio
    async def test_conversation_included_in_user_message(self) -> None:
        """Issue comments are fetched and included in the user message."""
        handler, api, _ = _make_handler()
        event = _make_event(issue_body="This is the issue description.")

        # api.call returns issue comments for get_issue_comments
        api.call = AsyncMock(
            side_effect=lambda name, **kw: (
                [
                    {"id": 50, "user": {"login": "alice"}, "body": "First comment"},
                    {"id": 100, "user": {"login": "bob"}, "body": "Trigger comment"},
                ]
                if name == "get_issue_comments"
                else {"id": 1}
            )
        )

        result = await handler._build_user_message("owner", "repo", 1, event)

        assert "Conversation so far" in result
        assert "Issue description" in result
        assert "This is the issue description." in result
        assert "@alice: First comment" in result
        # The triggering comment (id=100) should be excluded
        assert "Trigger comment" not in result
        assert "Current request" in result
        assert event.comment.body in result

    @pytest.mark.asyncio
    async def test_api_failure_returns_comment_body(self) -> None:
        """If fetching comments fails, only the comment body is used."""
        handler, api, _ = _make_handler()
        event = _make_event()

        api.call = AsyncMock(side_effect=RuntimeError("API error"))

        result = await handler._build_user_message("owner", "repo", 1, event)

        # Should fall back to just the comment body
        assert result == event.comment.body

    @pytest.mark.asyncio
    async def test_no_issue_body_no_comments(self) -> None:
        """Empty issue body and no comments returns just the comment body."""
        handler, api, _ = _make_handler()
        event = _make_event(issue_body="")

        api.call = AsyncMock(return_value=[])

        result = await handler._build_user_message("owner", "repo", 1, event)
        assert result == event.comment.body

    @pytest.mark.asyncio
    async def test_budget_caps_long_threads(self) -> None:
        """Very long threads are truncated by the character budget."""
        handler, api, _ = _make_handler()
        # llm_context_window = 8192, budget = 8192 * 4 // 6 ≈ 5461 chars
        event = _make_event(issue_body="Short body.")

        # Generate many comments that exceed the budget
        many_comments = [{"id": i, "user": {"login": "user"}, "body": "x" * 300} for i in range(50)]
        api.call = AsyncMock(return_value=many_comments)

        result = await handler._fetch_conversation("owner", "repo", 1, event)

        # Should be under the budget
        budget = 8192 * 4 // 6
        assert len(result) <= budget + 500  # some tolerance for join separators

    @pytest.mark.asyncio
    async def test_non_list_response_handled(self) -> None:
        """If API returns non-list, conversation context is still valid."""
        handler, api, _ = _make_handler()
        event = _make_event(issue_body="My issue")

        api.call = AsyncMock(return_value={"error": "not found"})

        result = await handler._fetch_conversation("owner", "repo", 1, event)

        # Should include just the issue body
        assert "My issue" in result
