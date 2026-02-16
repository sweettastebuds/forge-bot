"""Tests for forge_bot.tools.get_commit."""

from __future__ import annotations

from unittest.mock import AsyncMock

from forge_bot.tools.get_commit import GetCommitTool


def _make_tool(forge: AsyncMock | None = None) -> GetCommitTool:
    forge = forge or AsyncMock()
    return GetCommitTool(forge, "owner", "repo")


async def test_get_commit_success():
    forge = AsyncMock()
    forge.get_commit.return_value = {
        "sha": "abcdef123456",
        "commit": {
            "message": "feat: add authentication",
            "author": {
                "name": "Dev User",
                "date": "2026-01-15T10:00:00Z",
            },
        },
    }
    tool = _make_tool(forge)

    result = await tool.execute(sha="abcdef123456")

    assert result.success is True
    assert "abcdef123456" in result.content
    assert "Dev User" in result.content
    assert "add authentication" in result.content
    forge.get_commit.assert_awaited_once_with("owner", "repo", "abcdef123456")


async def test_get_commit_not_found():
    forge = AsyncMock()
    forge.get_commit.side_effect = RuntimeError("404 Not Found")
    tool = _make_tool(forge)

    result = await tool.execute(sha="deadbeef")

    assert result.success is False
    assert "Could not fetch commit" in result.content


async def test_get_commit_missing_sha():
    tool = _make_tool()
    result = await tool.execute()
    assert result.success is False
    assert "Missing required parameter" in result.content


async def test_get_commit_schema():
    tool = _make_tool()
    schema = tool.to_openai_schema()
    assert schema["function"]["name"] == "get_commit"
    assert "sha" in schema["function"]["parameters"]["properties"]
    assert schema["function"]["parameters"]["required"] == ["sha"]
