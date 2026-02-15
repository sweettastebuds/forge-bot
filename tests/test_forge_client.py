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


async def test_get_pull_diff(settings: Settings, httpx_mock: HTTPXMock):
    diff_text = (
        "diff --git a/file.py b/file.py\n"
        "--- a/file.py\n+++ b/file.py\n"
        "@@ -1 +1 @@\n-old\n+new\n"
    )
    httpx_mock.add_response(
        url="https://gitea.example.com/api/v1/repos/owner/repo/pulls/3.diff",
        text=diff_text,
    )

    client = ForgeClient(settings)
    try:
        result = await client.get_pull_diff("owner", "repo", 3)
        assert result == diff_text
        assert "diff --git" in result
    finally:
        await client.close()


async def test_get_pull_files(settings: Settings, httpx_mock: HTTPXMock):
    httpx_mock.add_response(
        url="https://gitea.example.com/api/v1/repos/owner/repo/pulls/3/files",
        json=[
            {"filename": "file.py", "additions": 5, "deletions": 2},
            {"filename": "readme.md", "additions": 1, "deletions": 0},
        ],
    )

    client = ForgeClient(settings)
    try:
        files = await client.get_pull_files("owner", "repo", 3)
        assert len(files) == 2
        assert files[0]["filename"] == "file.py"
        assert files[1]["additions"] == 1
    finally:
        await client.close()


async def test_get_pull_diff_raises_on_404(settings: Settings, httpx_mock: HTTPXMock):
    httpx_mock.add_response(
        url="https://gitea.example.com/api/v1/repos/owner/repo/pulls/999.diff",
        status_code=404,
    )

    client = ForgeClient(settings)
    try:
        with pytest.raises(httpx.HTTPStatusError):
            await client.get_pull_diff("owner", "repo", 999)
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


async def test_get_file_content(settings: Settings, httpx_mock: HTTPXMock):
    httpx_mock.add_response(
        url="https://gitea.example.com/api/v1/repos/owner/repo/raw/src/main.py",
        text="print('hello')\n",
    )

    client = ForgeClient(settings)
    try:
        content = await client.get_file_content("owner", "repo", "src/main.py")
        assert content == "print('hello')\n"
    finally:
        await client.close()


async def test_get_file_content_with_ref(settings: Settings, httpx_mock: HTTPXMock):
    httpx_mock.add_response(
        url="https://gitea.example.com/api/v1/repos/owner/repo/raw/README.md?ref=develop",
        text="# Hello\n",
    )

    client = ForgeClient(settings)
    try:
        content = await client.get_file_content(
            "owner", "repo", "README.md", ref="develop"
        )
        assert content == "# Hello\n"
    finally:
        await client.close()


async def test_get_file_content_raises_on_404(settings: Settings, httpx_mock: HTTPXMock):
    httpx_mock.add_response(
        url="https://gitea.example.com/api/v1/repos/owner/repo/raw/missing.txt",
        status_code=404,
    )

    client = ForgeClient(settings)
    try:
        with pytest.raises(httpx.HTTPStatusError):
            await client.get_file_content("owner", "repo", "missing.txt")
    finally:
        await client.close()


async def test_get_repo_tree(settings: Settings, httpx_mock: HTTPXMock):
    httpx_mock.add_response(
        url="https://gitea.example.com/api/v1/repos/owner/repo/git/trees/main?recursive=true",
        json={
            "sha": "abc123",
            "tree": [
                {"path": "src/main.py", "type": "blob", "size": 100},
                {"path": "src/utils.py", "type": "blob", "size": 200},
                {"path": "README.md", "type": "blob", "size": 50},
            ],
        },
    )

    client = ForgeClient(settings)
    try:
        tree = await client.get_repo_tree("owner", "repo", "main")
        assert len(tree) == 3
        assert tree[0]["path"] == "src/main.py"
        assert tree[2]["type"] == "blob"
    finally:
        await client.close()


async def test_get_repo_tree_non_recursive(settings: Settings, httpx_mock: HTTPXMock):
    httpx_mock.add_response(
        url="https://gitea.example.com/api/v1/repos/owner/repo/git/trees/main",
        json={
            "sha": "abc123",
            "tree": [
                {"path": "src", "type": "tree"},
                {"path": "README.md", "type": "blob", "size": 50},
            ],
        },
    )

    client = ForgeClient(settings)
    try:
        tree = await client.get_repo_tree(
            "owner", "repo", "main", recursive=False
        )
        assert len(tree) == 2
        assert tree[0]["type"] == "tree"
    finally:
        await client.close()


async def test_get_commit(settings: Settings, httpx_mock: HTTPXMock):
    httpx_mock.add_response(
        url="https://gitea.example.com/api/v1/repos/owner/repo/git/commits/abc123def",
        json={
            "sha": "abc123def456789",
            "commit": {
                "message": "fix: resolve auth bug",
                "author": {
                    "name": "Dev",
                    "email": "dev@example.com",
                    "date": "2026-01-01T00:00:00Z",
                },
            },
        },
    )

    client = ForgeClient(settings)
    try:
        result = await client.get_commit("owner", "repo", "abc123def")
        assert result["sha"] == "abc123def456789"
        assert "resolve auth bug" in result["commit"]["message"]
    finally:
        await client.close()


async def test_get_commit_raises_on_404(settings: Settings, httpx_mock: HTTPXMock):
    httpx_mock.add_response(
        url="https://gitea.example.com/api/v1/repos/owner/repo/git/commits/nonexistent",
        status_code=404,
    )

    client = ForgeClient(settings)
    try:
        with pytest.raises(httpx.HTTPStatusError):
            await client.get_commit("owner", "repo", "nonexistent")
    finally:
        await client.close()


async def test_download_url(settings: Settings, httpx_mock: HTTPXMock):
    httpx_mock.add_response(
        url="https://gitea.example.com/attachments/uuid-123/notes.md",
        text="# Meeting Notes\nDiscussed auth flow.",
    )

    client = ForgeClient(settings)
    try:
        content = await client.download_url(
            "https://gitea.example.com/attachments/uuid-123/notes.md"
        )
        assert "Meeting Notes" in content
    finally:
        await client.close()
