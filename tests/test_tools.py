"""Tests for the standard tool implementations."""

from __future__ import annotations

import sys

import pytest

from forge_bot.tools.standard import (
    AskUserTool,
    BashTool,
    EditFileTool,
    GlobTool,
    GrepTool,
    ReadFileTool,
    WriteFileTool,
)


@pytest.mark.asyncio
async def test_to_openai_schema_format(tmp_path) -> None:
    tool = ReadFileTool(tmp_path)
    schema = tool.to_openai_schema()

    assert schema["type"] == "function"
    function = schema["function"]
    assert function["name"] == "read_file"
    assert function["description"]
    params = function["parameters"]
    assert params["type"] == "object"
    assert params["required"] == ["path"]
    assert params["properties"]["path"]["type"] == "string"


@pytest.mark.asyncio
async def test_read_write_edit_cycle(tmp_path) -> None:
    writer = WriteFileTool(tmp_path)
    reader = ReadFileTool(tmp_path)
    editor = EditFileTool(tmp_path)

    write = await writer.execute(path="notes.txt", content="hello world")
    assert write.success

    read = await reader.execute(path="notes.txt")
    assert read.success
    assert "hello world" in read.content

    edit = await editor.execute(path="notes.txt", old="world", new="there")
    assert edit.success

    updated = await reader.execute(path="notes.txt")
    assert updated.success
    assert "hello there" in updated.content


@pytest.mark.asyncio
async def test_glob_lists_matches(tmp_path) -> None:
    folder = tmp_path / "src"
    folder.mkdir()
    (folder / "a.txt").write_text("hi")
    (folder / "b.py").write_text("print('hi')")

    tool = GlobTool(tmp_path)
    result = await tool.execute(pattern="**/*.txt")

    assert result.success
    assert str(folder / "a.txt") in result.content
    assert "b.py" not in result.content


@pytest.mark.asyncio
async def test_grep_finds_matches(tmp_path) -> None:
    target = tmp_path / "app.log"
    target.write_text("INFO start\nWARN issue\nINFO done\n")

    tool = GrepTool(tmp_path)
    result = await tool.execute(pattern="WARN", path="app.log")

    assert result.success
    assert "app.log:2:WARN issue" in result.content


@pytest.mark.asyncio
async def test_bash_runs_commands(tmp_path) -> None:
    tool = BashTool(tmp_path)
    result = await tool.execute(cmd="echo ready")

    assert result.success
    assert "ready" in result.content


@pytest.mark.asyncio
async def test_ask_user_noninteractive(monkeypatch) -> None:
    class _FakeStdin:
        def isatty(self) -> bool:
            return False

    monkeypatch.setattr(sys, "stdin", _FakeStdin())
    tool = AskUserTool()

    result = await tool.execute(question="ok?")

    assert not result.success
    assert "unavailable" in result.content.lower()
