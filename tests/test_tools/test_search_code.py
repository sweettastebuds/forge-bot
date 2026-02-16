"""Tests for forge_bot.tools.search_code."""

from __future__ import annotations

from unittest.mock import AsyncMock

from forge_bot.tools.search_code import SearchCodeTool


def _make_tool(
    forge: AsyncMock | None = None,
    tree_paths: list[str] | None = None,
) -> SearchCodeTool:
    forge = forge or AsyncMock()
    paths = tree_paths or ["src/main.py", "src/utils.py", "README.md"]
    return SearchCodeTool(
        forge, "owner", "repo", "main", tree_paths=paths,
    )


async def test_search_finds_matches():
    forge = AsyncMock()
    forge.get_file_content.side_effect = [
        "def hello():\n    pass\n",
        "import hello\nfrom utils import something\n",
        "# README\n",
    ]
    tool = _make_tool(forge)

    result = await tool.execute(query="hello")

    assert result.success is True
    assert "hello" in result.content
    assert "src/main.py:1" in result.content


async def test_search_with_file_pattern():
    forge = AsyncMock()
    forge.get_file_content.return_value = "def hello(): pass"
    tool = _make_tool(forge)

    result = await tool.execute(query="hello", file_pattern="*.py")

    assert result.success is True
    # Only .py files should be searched (not README.md)
    calls = forge.get_file_content.call_args_list
    searched_paths = [c.args[2] for c in calls]
    assert "README.md" not in searched_paths


async def test_search_no_matches():
    forge = AsyncMock()
    forge.get_file_content.return_value = "nothing relevant here"
    tool = _make_tool(forge)

    result = await tool.execute(query="nonexistent_function")

    assert result.success is True
    assert "No matches found" in result.content


async def test_search_max_results():
    """Should cap at _MAX_RESULTS (10)."""
    forge = AsyncMock()
    # Each file has 5 matching lines.
    forge.get_file_content.return_value = "\n".join(
        f"line with match {i}" for i in range(5)
    )
    paths = [f"file{i}.py" for i in range(10)]
    tool = _make_tool(forge, tree_paths=paths)

    result = await tool.execute(query="match")

    assert result.success is True
    # Should have at most 10 results (lines like "file0.py:1: line with match 0").
    match_lines = [
        line for line in result.content.splitlines()
        if line and line[0] != "F" and ":" in line  # skip header
    ]
    assert len(match_lines) <= 10


async def test_search_max_files():
    """Should cap at _MAX_FILES (30)."""
    forge = AsyncMock()
    forge.get_file_content.return_value = "no match here"
    paths = [f"file{i}.py" for i in range(50)]
    tool = _make_tool(forge, tree_paths=paths)

    await tool.execute(query="anything")

    # Should have checked at most 30 files.
    assert forge.get_file_content.call_count <= 30


async def test_search_prioritizes_filename_matches():
    """Files with the query term in the name should be searched first."""
    forge = AsyncMock()
    forge.get_file_content.return_value = "some content"
    paths = ["src/other.py", "src/auth.py", "src/auth_utils.py", "README.md"]
    tool = _make_tool(forge, tree_paths=paths)

    await tool.execute(query="auth")

    # Files with "auth" in name should be searched first.
    calls = forge.get_file_content.call_args_list
    searched_paths = [c.args[2] for c in calls]
    # auth.py and auth_utils.py should come before other.py
    auth_idx = min(
        searched_paths.index(p)
        for p in searched_paths
        if "auth" in p
    )
    other_idx = searched_paths.index("src/other.py")
    assert auth_idx < other_idx


async def test_search_missing_query():
    tool = _make_tool()
    result = await tool.execute()
    assert result.success is False
    assert "Missing required parameter" in result.content


async def test_search_file_fetch_error_skipped():
    """Files that fail to fetch should be silently skipped."""
    forge = AsyncMock()
    forge.get_file_content.side_effect = [
        RuntimeError("404"),
        "def hello(): pass",
        "nothing here",
    ]
    tool = _make_tool(forge)

    result = await tool.execute(query="hello")

    assert result.success is True
    assert "hello" in result.content
