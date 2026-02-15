"""Tests for forge_bot.clients.forge."""

import json

import httpx
import pytest
from pytest_httpx import HTTPXMock

from forge_bot.clients.forge import ForgeClient
from forge_bot.config import Settings


@pytest.fixture
def settings(monkeypatch: pytest.MonkeyPatch) -> Settings:
    monkeypatch.setenv("FORGE_INSTANCE_URL", "https://gitea.example.com")
    monkeypatch.setenv("FORGE_API_TOKEN", "test-token")
    monkeypatch.setenv("FORGE_WEBHOOK_SECRET", "test-secret")
    monkeypatch.setenv("LLM_API_KEY", "test-llm-key")
    return Settings()


async def test_get_self_returns_bot_user(settings: Settings, httpx_mock: HTTPXMock):
    httpx_mock.add_response(
        url="https://gitea.example.com/api/v1/user",
        json={"id": 42, "login": "forge-bot"},
    )

    client = ForgeClient(settings)
    try:
        user = await client.get_self()
        assert user.id == 42
        assert user.login == "forge-bot"
    finally:
        await client.close()


async def test_get_self_sends_auth_header(settings: Settings, httpx_mock: HTTPXMock):
    httpx_mock.add_response(
        url="https://gitea.example.com/api/v1/user",
        json={"id": 1, "login": "bot"},
    )

    client = ForgeClient(settings)
    try:
        await client.get_self()
        request = httpx_mock.get_request()
        assert request is not None
        assert request.headers["authorization"] == "token test-token"
    finally:
        await client.close()


async def test_get_self_raises_on_401(settings: Settings, httpx_mock: HTTPXMock):
    httpx_mock.add_response(
        url="https://gitea.example.com/api/v1/user",
        status_code=401,
    )

    client = ForgeClient(settings)
    try:
        with pytest.raises(httpx.HTTPStatusError):
            await client.get_self()
    finally:
        await client.close()


async def test_get_issue_comments(settings: Settings, httpx_mock: HTTPXMock):
    httpx_mock.add_response(
        url="https://gitea.example.com/api/v1/repos/owner/repo/issues/5/comments",
        json=[
            {"id": 1, "body": "first comment", "user": {"login": "dev"}},
            {"id": 2, "body": "second comment", "user": {"login": "bot"}},
        ],
    )

    client = ForgeClient(settings)
    try:
        comments = await client.get_issue_comments("owner", "repo", 5)
        assert len(comments) == 2
        assert comments[0]["body"] == "first comment"
    finally:
        await client.close()


async def test_post_comment(settings: Settings, httpx_mock: HTTPXMock):
    httpx_mock.add_response(
        url="https://gitea.example.com/api/v1/repos/owner/repo/issues/5/comments",
        method="POST",
        json={"id": 10, "body": "bot reply"},
    )

    client = ForgeClient(settings)
    try:
        result = await client.post_comment("owner", "repo", 5, "bot reply")
        assert result["id"] == 10

        request = httpx_mock.get_request()
        assert request is not None
        assert json.loads(request.content) == {"body": "bot reply"}
    finally:
        await client.close()


async def test_post_comment_raises_on_error(settings: Settings, httpx_mock: HTTPXMock):
    httpx_mock.add_response(
        url="https://gitea.example.com/api/v1/repos/owner/repo/issues/5/comments",
        method="POST",
        status_code=403,
    )

    client = ForgeClient(settings)
    try:
        with pytest.raises(httpx.HTTPStatusError):
            await client.post_comment("owner", "repo", 5, "nope")
    finally:
        await client.close()
