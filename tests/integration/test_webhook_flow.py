"""Integration tests: webhook reception → routing → handler dispatch.

These tests send real HTTP requests through the FastAPI app and verify that
the correct handler is dispatched with properly parsed Pydantic models.
Only the external boundaries (Docker, LLM API, Gitea API) are mocked.
"""

from __future__ import annotations

import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from httpx import ASGITransport, AsyncClient

from forge_bot.server import app

from .conftest import (
    TEST_SECRET,
    FakeApiTracker,
    sign_payload,
    webhook_headers,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


@pytest.fixture
async def http_client(api_tracker: FakeApiTracker):
    """ASGI test client with lifespan — bot identity resolved via FakeApiTracker."""
    with (
        patch(
            "forge_bot.server.GenericForgeClient.call",
            side_effect=api_tracker.call,
        ),
        patch("forge_bot.server.GenericForgeClient.close", AsyncMock()),
        patch("forge_bot.server.LLMClient.close", AsyncMock()),
    ):
        async with app.router.lifespan_context(app):
            transport = ASGITransport(app=app)
            async with AsyncClient(transport=transport, base_url="http://test") as c:
                yield c


# ---------------------------------------------------------------------------
# Tests: webhook → handler dispatch
# ---------------------------------------------------------------------------


class TestWebhookToPRHandler:
    """Webhook → PullRequestHandler integration."""

    async def test_pr_opened_dispatches_handler(
        self,
        http_client: AsyncClient,
        pr_payload: dict,
    ) -> None:
        """A valid PR opened webhook reaches the PullRequestHandler."""
        body = json.dumps(pr_payload).encode()
        headers = webhook_headers(body, event="pull_request")

        handler_called = False
        original_handle = None

        async def capture_handle(self_handler, event):
            nonlocal handler_called
            handler_called = True
            assert event.pull_request.title == "Add login endpoint"
            assert event.repository.full_name == "org/myapp"
            assert event.number == 42

        with patch(
            "forge_bot.handlers.pull_request.PullRequestHandler.handle",
            capture_handle,
        ):
            resp = await http_client.post("/webhook", content=body, headers=headers)

        assert resp.status_code == 200
        # Background tasks run synchronously in the test client
        assert handler_called

    async def test_pr_synchronized_dispatches(
        self,
        http_client: AsyncClient,
        pr_payload: dict,
    ) -> None:
        """action=synchronized also triggers the PR handler."""
        pr_payload["action"] = "synchronized"
        body = json.dumps(pr_payload).encode()
        headers = webhook_headers(body, event="pull_request", delivery="sync-uuid")

        handler_called = False

        async def capture(self_handler, event):
            nonlocal handler_called
            handler_called = True

        with patch(
            "forge_bot.handlers.pull_request.PullRequestHandler.handle",
            capture,
        ):
            resp = await http_client.post("/webhook", content=body, headers=headers)

        assert resp.status_code == 200
        assert handler_called

    async def test_pr_closed_ignored(
        self,
        http_client: AsyncClient,
        pr_payload: dict,
    ) -> None:
        """action=closed should not dispatch to any handler."""
        pr_payload["action"] = "closed"
        body = json.dumps(pr_payload).encode()
        headers = webhook_headers(body, event="pull_request", delivery="closed-uuid")

        handler_called = False

        async def capture(self_handler, event):
            nonlocal handler_called
            handler_called = True

        with patch(
            "forge_bot.handlers.pull_request.PullRequestHandler.handle",
            capture,
        ):
            resp = await http_client.post("/webhook", content=body, headers=headers)

        assert resp.status_code == 200
        assert not handler_called


class TestWebhookToCommentHandler:
    """Webhook → IssueCommentHandler integration."""

    async def test_mention_dispatches_handler(
        self,
        http_client: AsyncClient,
        comment_payload: dict,
    ) -> None:
        """A comment mentioning @forge-bot reaches the IssueCommentHandler."""
        body = json.dumps(comment_payload).encode()
        headers = webhook_headers(body, event="issue_comment", delivery="comment-uuid")

        captured_event = None

        async def capture(self_handler, event):
            nonlocal captured_event
            captured_event = event

        with patch(
            "forge_bot.handlers.issue_comment.IssueCommentHandler.handle",
            capture,
        ):
            resp = await http_client.post("/webhook", content=body, headers=headers)

        assert resp.status_code == 200
        assert captured_event is not None
        assert captured_event.comment.body == "@forge-bot explain how the auth module works"
        assert captured_event.issue.number == 7

    async def test_no_mention_skipped(
        self,
        http_client: AsyncClient,
        comment_payload: dict,
    ) -> None:
        """A comment without @forge-bot is silently skipped."""
        comment_payload["comment"]["body"] = "Just a regular comment"
        body = json.dumps(comment_payload).encode()
        headers = webhook_headers(body, event="issue_comment", delivery="no-mention-uuid")

        handler_called = False

        async def capture(self_handler, event):
            nonlocal handler_called
            handler_called = True

        with patch(
            "forge_bot.handlers.issue_comment.IssueCommentHandler.handle",
            capture,
        ):
            resp = await http_client.post("/webhook", content=body, headers=headers)

        assert resp.status_code == 200
        assert not handler_called

    async def test_self_loop_guard(
        self,
        http_client: AsyncClient,
        self_sent_comment_payload: dict,
    ) -> None:
        """Comments from the bot itself are dropped by the self-loop guard."""
        body = json.dumps(self_sent_comment_payload).encode()
        headers = webhook_headers(body, event="issue_comment", delivery="self-uuid")

        handler_called = False

        async def capture(self_handler, event):
            nonlocal handler_called
            handler_called = True

        with patch(
            "forge_bot.handlers.issue_comment.IssueCommentHandler.handle",
            capture,
        ):
            resp = await http_client.post("/webhook", content=body, headers=headers)

        assert resp.status_code == 200
        assert not handler_called


class TestWebhookSecurity:
    """HMAC verification and deduplication in the integration path."""

    async def test_bad_signature_rejected(
        self,
        http_client: AsyncClient,
        pr_payload: dict,
    ) -> None:
        body = json.dumps(pr_payload).encode()
        headers = webhook_headers(body, event="pull_request")
        headers["x-gitea-signature"] = "tampered"

        resp = await http_client.post("/webhook", content=body, headers=headers)
        assert resp.status_code == 403

    async def test_duplicate_delivery_idempotent(
        self,
        http_client: AsyncClient,
        pr_payload: dict,
    ) -> None:
        """Same delivery ID sent twice — second is ignored."""
        body = json.dumps(pr_payload).encode()
        headers = webhook_headers(body, event="pull_request", delivery="dedup-test")

        handler_count = 0

        async def capture(self_handler, event):
            nonlocal handler_count
            handler_count += 1

        with patch(
            "forge_bot.handlers.pull_request.PullRequestHandler.handle",
            capture,
        ):
            resp1 = await http_client.post("/webhook", content=body, headers=headers)
            resp2 = await http_client.post("/webhook", content=body, headers=headers)

        assert resp1.status_code == 200
        assert resp2.status_code == 200
        assert "duplicate" in resp2.text.lower()
        assert handler_count == 1

    async def test_forgejo_headers_take_precedence(
        self,
        http_client: AsyncClient,
        pr_payload: dict,
    ) -> None:
        """When both Forgejo and Gitea headers present, Forgejo wins."""
        body = json.dumps(pr_payload).encode()
        headers = {
            "x-forgejo-event": "pull_request",
            "x-gitea-event": "issue_comment",
            "x-forgejo-signature": sign_payload(body),
            "x-gitea-signature": "wrong",
            "x-forgejo-delivery": "forgejo-uuid",
            "content-type": "application/json",
        }

        handler_called = False

        async def capture(self_handler, event):
            nonlocal handler_called
            handler_called = True

        with patch(
            "forge_bot.handlers.pull_request.PullRequestHandler.handle",
            capture,
        ):
            resp = await http_client.post("/webhook", content=body, headers=headers)

        assert resp.status_code == 200
        # The Forgejo event type "pull_request" should have been used
        assert handler_called
