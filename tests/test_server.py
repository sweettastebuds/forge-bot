"""Tests for forge_bot.server."""

import hashlib
import hmac
import json
from unittest.mock import AsyncMock, patch

import pytest
from httpx import ASGITransport, AsyncClient

from forge_bot.server import app

TEST_SECRET = "test-secret"
TEST_ENV = {
    "FORGE_INSTANCE_URL": "https://gitea.example.com",
    "FORGE_API_TOKEN": "test-token",
    "FORGE_WEBHOOK_SECRET": TEST_SECRET,
    "LLM_API_KEY": "test-llm-key",
}


def _sign(body: bytes, secret: str = TEST_SECRET) -> str:
    """Compute HMAC-SHA256 hex digest for a webhook payload."""
    return hmac.new(secret.encode(), body, hashlib.sha256).hexdigest()


def _webhook_headers(body: bytes, event: str = "pull_request", delivery: str = "uuid-1") -> dict:
    """Build standard Gitea webhook headers with valid signature."""
    return {
        "x-gitea-event": event,
        "x-gitea-signature": _sign(body),
        "x-gitea-delivery": delivery,
        "content-type": "application/json",
    }


@pytest.fixture(autouse=True)
def _set_env(monkeypatch: pytest.MonkeyPatch):
    for k, v in TEST_ENV.items():
        monkeypatch.setenv(k, v)


@pytest.fixture
async def client():
    mock_call = AsyncMock(return_value={"id": 1, "login": "forge-bot"})
    with (
        patch("forge_bot.server.GenericForgeClient.call", mock_call),
        patch("forge_bot.server.GenericForgeClient.close", AsyncMock()),
        patch("forge_bot.server.LLMClient.close", AsyncMock()),
    ):
        async with app.router.lifespan_context(app):
            transport = ASGITransport(app=app)
            async with AsyncClient(transport=transport, base_url="http://test") as c:
                yield c


async def test_health(client: AsyncClient):
    resp = await client.get("/health")
    assert resp.status_code == 200
    assert resp.json() == {"status": "ok"}


async def test_valid_webhook(client: AsyncClient, sample_pr_payload: dict):
    body = json.dumps(sample_pr_payload).encode()
    headers = _webhook_headers(body)

    resp = await client.post("/webhook", content=body, headers=headers)
    assert resp.status_code == 200
    assert resp.text == "OK"


async def test_invalid_signature_returns_403(client: AsyncClient, sample_pr_payload: dict):
    body = json.dumps(sample_pr_payload).encode()
    headers = _webhook_headers(body)
    headers["x-gitea-signature"] = "bad-signature"

    resp = await client.post("/webhook", content=body, headers=headers)
    assert resp.status_code == 403


async def test_missing_signature_returns_403(client: AsyncClient, sample_pr_payload: dict):
    body = json.dumps(sample_pr_payload).encode()
    headers = {
        "x-gitea-event": "pull_request",
        "x-gitea-delivery": "uuid-1",
        "content-type": "application/json",
    }

    resp = await client.post("/webhook", content=body, headers=headers)
    assert resp.status_code == 403


async def test_duplicate_delivery_returns_200(client: AsyncClient, sample_pr_payload: dict):
    body = json.dumps(sample_pr_payload).encode()
    headers = _webhook_headers(body, delivery="dup-uuid")

    # First request
    resp1 = await client.post("/webhook", content=body, headers=headers)
    assert resp1.status_code == 200
    assert resp1.text == "OK"

    # Second request with same delivery ID
    resp2 = await client.post("/webhook", content=body, headers=headers)
    assert resp2.status_code == 200
    assert "duplicate" in resp2.text.lower()


async def test_missing_event_type_returns_200(client: AsyncClient, sample_pr_payload: dict):
    body = json.dumps(sample_pr_payload).encode()
    headers = {
        "x-gitea-signature": _sign(body),
        "x-gitea-delivery": "uuid-no-event",
        "content-type": "application/json",
    }

    resp = await client.post("/webhook", content=body, headers=headers)
    assert resp.status_code == 200
    assert "no event type" in resp.text.lower()


async def test_invalid_json_returns_400(client: AsyncClient):
    body = b"not json at all"
    headers = {
        "x-gitea-event": "pull_request",
        "x-gitea-signature": _sign(body),
        "x-gitea-delivery": "uuid-bad-json",
        "content-type": "application/json",
    }

    resp = await client.post("/webhook", content=body, headers=headers)
    assert resp.status_code == 400


async def test_forgejo_headers_preferred(client: AsyncClient, sample_pr_payload: dict):
    """When both Forgejo and Gitea headers are present, Forgejo takes precedence."""
    body = json.dumps(sample_pr_payload).encode()
    headers = {
        "x-forgejo-event": "pull_request",
        "x-gitea-event": "issue_comment",  # should be ignored
        "x-forgejo-signature": _sign(body),
        "x-gitea-signature": "wrong-sig",  # should be ignored
        "x-forgejo-delivery": "forgejo-uuid",
        "content-type": "application/json",
    }

    resp = await client.post("/webhook", content=body, headers=headers)
    assert resp.status_code == 200
    assert resp.text == "OK"
