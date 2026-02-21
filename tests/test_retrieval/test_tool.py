"""Tests for forge_bot.retrieval.tool."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

from forge_bot.container.manager import ExecResult
from forge_bot.retrieval.pipeline import SmartRetriever
from forge_bot.retrieval.tool import RetrievalTool


def _make_retriever() -> SmartRetriever:
    llm = AsyncMock()
    llm.chat.return_value = "keyword1\nkeyword2"
    return SmartRetriever(llm, context_window=8192)


def _make_container(
    *,
    cloned: bool = True,
    files: str = "/workspace/src/main.py\n/workspace/README.md",
    file_contents: str = (
        "=== /workspace/src/main.py ===\n"
        "def handle_error(e): log(e)\ndef other(): pass\n\n"
        "=== /workspace/README.md ===\n"
        "# My Project\n"
    ),
) -> MagicMock:
    """Mock ContainerManager that simulates a cloned workspace."""
    cm = MagicMock()

    async def mock_exec(command: str, timeout: int = 60, **_kw: object) -> ExecResult:
        if "test -d /workspace/.git" in command:
            return ExecResult(
                exit_code=0 if cloned else 1,
                stdout="ok\n" if cloned else "",
                stderr="",
                duration_seconds=0.1,
                command=command,
            )
        if command.startswith("find "):
            return ExecResult(
                exit_code=0,
                stdout=files,
                stderr="",
                duration_seconds=0.2,
                command=command,
            )
        if command.startswith("for f in "):
            return ExecResult(
                exit_code=0,
                stdout=file_contents,
                stderr="",
                duration_seconds=0.3,
                command=command,
            )
        return ExecResult(
            exit_code=0, stdout="", stderr="",
            duration_seconds=0.1, command=command,
        )

    cm.exec = AsyncMock(side_effect=mock_exec)
    return cm


async def test_tool_searches_workspace():
    """Tool discovers files, reads them, and applies retrieval."""
    retriever = _make_retriever()
    retriever.scan_and_answer = AsyncMock(return_value="Error handling uses log(e)")
    cm = _make_container()
    tool = RetrievalTool(retriever, cm)

    result = await tool.execute(query="how is error handling done")

    assert result.success
    assert result.tool_name == "smart_search"
    # Should have called scan_and_answer (Level 2) for small content.
    retriever.scan_and_answer.assert_awaited_once()


async def test_tool_missing_query():
    retriever = _make_retriever()
    cm = _make_container()
    tool = RetrievalTool(retriever, cm)

    result = await tool.execute()
    assert not result.success
    assert "query" in result.content.lower()


async def test_tool_repo_not_cloned():
    retriever = _make_retriever()
    cm = _make_container(cloned=False)
    tool = RetrievalTool(retriever, cm)

    result = await tool.execute(query="find something")
    assert not result.success
    assert "not cloned" in result.content.lower()


async def test_tool_no_files_found():
    retriever = _make_retriever()
    cm = _make_container(files="", file_contents="")
    tool = RetrievalTool(retriever, cm)

    result = await tool.execute(query="find something")
    assert result.success
    assert "no relevant files" in result.content.lower()


async def test_tool_with_path():
    """When path is provided, search is scoped to that directory."""
    retriever = _make_retriever()
    retriever.scan_and_answer = AsyncMock(return_value="Found it")
    cm = _make_container()
    tool = RetrievalTool(retriever, cm)

    result = await tool.execute(query="how does auth work", path="src/auth")

    assert result.success
    # The find command should include the path.
    find_call = [
        c for c in cm.exec.call_args_list
        if "find " in str(c)
    ]
    assert any("src/auth" in str(c) for c in find_call)


async def test_tool_openai_schema():
    retriever = _make_retriever()
    cm = _make_container()
    tool = RetrievalTool(retriever, cm)
    schema = tool.to_openai_schema()

    assert schema["type"] == "function"
    assert schema["function"]["name"] == "smart_search"
    params = schema["function"]["parameters"]
    assert "query" in params["properties"]
    assert "path" in params["properties"]
    assert "query" in params["required"]
    # path should be optional
    assert "path" not in params["required"]


async def test_extract_keywords():
    """Keyword extraction filters stop words and keeps meaningful terms."""
    keywords = RetrievalTool._extract_keywords(
        "how does the authentication middleware handle JWT tokens"
    )
    assert "authentication" in keywords
    assert "middleware" in keywords
    assert "JWT" in keywords
    assert "tokens" in keywords
    # Stop words filtered out
    assert "how" not in keywords
    assert "does" not in keywords
    assert "the" not in keywords
