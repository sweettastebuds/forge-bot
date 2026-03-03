"""Tests for the PullRequestHandler lifecycle."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from forge_bot.handlers.pull_request import PullRequestHandler
from forge_bot.models import PullRequestEvent


@pytest.fixture
def pr_event(sample_pr_payload: dict) -> PullRequestEvent:
    return PullRequestEvent.model_validate(sample_pr_payload)


def _make_handler() -> tuple[PullRequestHandler, MagicMock, AsyncMock]:
    api = MagicMock()
    api.call = AsyncMock(return_value={"id": 1})

    llm = AsyncMock()

    settings = MagicMock()
    settings.forge_api_token = "test-token"
    settings.forge_instance_url = "https://gitea.example.com"
    settings.container_network_enabled = True
    settings.llm_context_window = 8192

    handler = PullRequestHandler(
        api_client=api,
        llm_client=llm,
        settings=settings,
        bot_username="forge-bot",
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
        patch("forge_bot.handlers.pull_request.SmartRetriever"),
        patch("forge_bot.handlers.pull_request.RetrievalTool"),
    )


class TestHandleFlow:
    @pytest.mark.asyncio
    async def test_happy_path(self, pr_event: PullRequestEvent) -> None:
        """Full handle(): status -> container -> agent -> post -> finalize -> destroy."""
        handler, _, _ = _make_handler()
        mock_status = _mock_status()
        mock_container = _mock_container()

        mock_agent = MagicMock()
        mock_agent.run = AsyncMock(return_value="Looks good, no issues found.")
        mock_agent.collect_artifacts = AsyncMock(return_value=[])

        p_retriever, p_tool = _patches()
        with (
            patch(
                "forge_bot.handlers.pull_request.ContainerManager",
                return_value=mock_container,
            ),
            patch(
                "forge_bot.handlers.pull_request.StatusCommentManager",
                return_value=mock_status,
            ),
            patch(
                "forge_bot.handlers.pull_request.AgentLoop",
                return_value=mock_agent,
            ),
            p_retriever,
            p_tool,
        ):
            await handler.handle(pr_event)

        mock_status.post_initial_status.assert_awaited_once()
        mock_container.create.assert_awaited_once()
        mock_agent.run.assert_awaited_once()
        mock_status.post_response.assert_awaited_once_with("Looks good, no issues found.")
        mock_status.finalize_status.assert_awaited_once_with("Done")
        mock_container.destroy.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_container_failure_posts_error(
        self,
        pr_event: PullRequestEvent,
    ) -> None:
        handler, _, _ = _make_handler()
        mock_status = _mock_status()
        mock_container = _mock_container()
        mock_container.create = AsyncMock(side_effect=RuntimeError("Docker down"))

        p_retriever, p_tool = _patches()
        with (
            patch(
                "forge_bot.handlers.pull_request.ContainerManager",
                return_value=mock_container,
            ),
            patch(
                "forge_bot.handlers.pull_request.StatusCommentManager",
                return_value=mock_status,
            ),
            p_retriever,
            p_tool,
        ):
            await handler.handle(pr_event)

        mock_status.post_response.assert_awaited_once()
        body = mock_status.post_response.call_args[0][0]
        assert "error" in body.lower()
        mock_container.destroy.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_container_destroyed_on_exception(
        self,
        pr_event: PullRequestEvent,
    ) -> None:
        handler, _, _ = _make_handler()
        mock_status = _mock_status()
        mock_container = _mock_container()

        mock_agent = MagicMock()
        mock_agent.run = AsyncMock(side_effect=RuntimeError("Unexpected"))

        p_retriever, p_tool = _patches()
        with (
            patch(
                "forge_bot.handlers.pull_request.ContainerManager",
                return_value=mock_container,
            ),
            patch(
                "forge_bot.handlers.pull_request.StatusCommentManager",
                return_value=mock_status,
            ),
            patch(
                "forge_bot.handlers.pull_request.AgentLoop",
                return_value=mock_agent,
            ),
            p_retriever,
            p_tool,
        ):
            await handler.handle(pr_event)

        mock_container.destroy.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_agent_receives_pr_metadata(
        self,
        pr_event: PullRequestEvent,
    ) -> None:
        """The system prompt contains PR metadata (title, branches)."""
        handler, _, _ = _make_handler()
        mock_status = _mock_status()
        mock_container = _mock_container()

        captured_agent = MagicMock()
        captured_agent.run = AsyncMock(return_value="Review complete.")
        captured_agent.collect_artifacts = AsyncMock(return_value=[])

        p_retriever, p_tool = _patches()
        with (
            patch(
                "forge_bot.handlers.pull_request.ContainerManager",
                return_value=mock_container,
            ),
            patch(
                "forge_bot.handlers.pull_request.StatusCommentManager",
                return_value=mock_status,
            ),
            patch(
                "forge_bot.handlers.pull_request.AgentLoop",
                return_value=captured_agent,
            ),
            p_retriever,
            p_tool,
        ):
            await handler.handle(pr_event)

        # agent.run() receives (system_prompt, user_message)
        captured_agent.run.assert_awaited_once()
        system_prompt = captured_agent.run.call_args[0][0]
        user_message = captured_agent.run.call_args[0][1]

        # System prompt should contain PR metadata from the template
        assert "Test PR" in system_prompt
        assert "feature" in system_prompt  # head branch
        assert "main" in system_prompt  # base branch
        assert "owner/repo" in system_prompt

        # User message should reference the PR
        assert "#1" in user_message

    @pytest.mark.asyncio
    async def test_artifacts_attached(self, pr_event: PullRequestEvent) -> None:
        """Artifacts from agent.collect_artifacts() are uploaded."""
        handler, _, _ = _make_handler()
        mock_status = _mock_status()
        mock_container = _mock_container()

        mock_agent = MagicMock()
        mock_agent.run = AsyncMock(return_value="Review done.")
        mock_agent.collect_artifacts = AsyncMock(return_value=[("diff.patch", b"patch content")])

        p_retriever, p_tool = _patches()
        with (
            patch(
                "forge_bot.handlers.pull_request.ContainerManager",
                return_value=mock_container,
            ),
            patch(
                "forge_bot.handlers.pull_request.StatusCommentManager",
                return_value=mock_status,
            ),
            patch(
                "forge_bot.handlers.pull_request.AgentLoop",
                return_value=mock_agent,
            ),
            p_retriever,
            p_tool,
        ):
            await handler.handle(pr_event)

        mock_status.attach_file.assert_awaited_once_with("diff.patch", b"patch content")
