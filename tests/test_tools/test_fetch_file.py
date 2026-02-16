"""Tests for forge_bot.tools.fetch_file."""

from __future__ import annotations

from unittest.mock import AsyncMock

from forge_bot.tools.fetch_file import FetchFileTool


def _make_tool(
    forge: AsyncMock | None = None,
    owner: str = "owner",
    repo: str = "repo",
    default_branch: str = "main",
    max_chars: int = 8000,
) -> FetchFileTool:
    forge = forge or AsyncMock()
    return FetchFileTool(
        forge, owner, repo, default_branch, max_chars=max_chars,
    )


async def test_fetch_file_success():
    forge = AsyncMock()
    forge.get_file_content.return_value = "print('hello')"
    tool = _make_tool(forge)

    result = await tool.execute(path="src/main.py")

    assert result.success is True
    assert "print('hello')" in result.content
    forge.get_file_content.assert_awaited_once_with(
        "owner", "repo", "src/main.py", ref="main",
    )


async def test_fetch_file_with_ref():
    forge = AsyncMock()
    forge.get_file_content.return_value = "old code"
    tool = _make_tool(forge)

    result = await tool.execute(path="config.yaml", ref="develop")

    assert result.success is True
    forge.get_file_content.assert_awaited_once_with(
        "owner", "repo", "config.yaml", ref="develop",
    )


async def test_fetch_file_default_branch():
    """Empty ref should use the default branch."""
    forge = AsyncMock()
    forge.get_file_content.return_value = "content"
    tool = _make_tool(forge, default_branch="master")

    await tool.execute(path="README.md", ref="")

    forge.get_file_content.assert_awaited_once_with(
        "owner", "repo", "README.md", ref="master",
    )


async def test_fetch_file_truncates():
    forge = AsyncMock()
    forge.get_file_content.return_value = "x" * 10_000
    tool = _make_tool(forge, max_chars=100)

    result = await tool.execute(path="big.py")

    assert result.success is True
    assert len(result.content) < 200
    assert "(truncated)" in result.content


async def test_fetch_file_not_found():
    forge = AsyncMock()
    forge.get_file_content.side_effect = RuntimeError("404 Not Found")
    tool = _make_tool(forge)

    result = await tool.execute(path="nonexistent.py")

    assert result.success is False
    assert "Could not fetch" in result.content


async def test_fetch_file_missing_path():
    tool = _make_tool()
    result = await tool.execute()
    assert result.success is False
    assert "Missing required parameter" in result.content


async def test_fetch_file_schema():
    tool = _make_tool()
    schema = tool.to_openai_schema()
    assert schema["function"]["name"] == "fetch_file"
    assert "path" in schema["function"]["parameters"]["properties"]
    assert "path" in schema["function"]["parameters"]["required"]
    assert "ref" not in schema["function"]["parameters"]["required"]
