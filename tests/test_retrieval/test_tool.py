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
            exit_code=0,
            stdout="",
            stderr="",
            duration_seconds=0.1,
            command=command,
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
    find_call = [c for c in cm.exec.call_args_list if "find " in str(c)]
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


async def test_large_repo_grep_narrowing():
    """When the file list exceeds _SMALL_REPO_THRESHOLD, grep narrows the set."""
    retriever = _make_retriever()
    retriever.scan_and_answer = AsyncMock(return_value="Auth uses JWT middleware")

    # Generate 60 files (exceeds _SMALL_REPO_THRESHOLD of 50).
    large_file_list = "\n".join(f"/workspace/src/file{i}.py" for i in range(60))

    # The grep-narrowed result: only two files match.
    grep_hits = "/workspace/src/file7.py\n/workspace/src/file42.py"

    # File contents returned when reading the narrowed set.
    narrowed_contents = (
        "=== /workspace/src/file7.py ===\n"
        "class AuthMiddleware: pass\n\n"
        "=== /workspace/src/file42.py ===\n"
        "def validate_jwt(token): return True\n"
    )

    cm = MagicMock()

    async def mock_exec(command: str, timeout: int = 60, **_kw: object) -> ExecResult:
        if "test -d /workspace/.git" in command:
            return ExecResult(
                exit_code=0,
                stdout="ok\n",
                stderr="",
                duration_seconds=0.1,
                command=command,
            )
        if command.startswith("find "):
            return ExecResult(
                exit_code=0,
                stdout=large_file_list,
                stderr="",
                duration_seconds=0.2,
                command=command,
            )
        if command.startswith("grep "):
            return ExecResult(
                exit_code=0,
                stdout=grep_hits,
                stderr="",
                duration_seconds=0.3,
                command=command,
            )
        if command.startswith("for f in "):
            return ExecResult(
                exit_code=0,
                stdout=narrowed_contents,
                stderr="",
                duration_seconds=0.3,
                command=command,
            )
        return ExecResult(
            exit_code=0,
            stdout="",
            stderr="",
            duration_seconds=0.1,
            command=command,
        )

    cm.exec = AsyncMock(side_effect=mock_exec)
    tool = RetrievalTool(retriever, cm)

    result = await tool.execute(query="how does authentication middleware handle JWT")

    assert result.success

    # Verify the grep command was issued (narrowing path taken).
    grep_calls = [c for c in cm.exec.call_args_list if "grep " in str(c)]
    assert grep_calls, "Expected grep narrowing call for large repo"

    # The read command ("for f in ...") should use only the narrowed files.
    read_calls = [c for c in cm.exec.call_args_list if "for f in " in str(c)]
    assert read_calls
    read_cmd_str = str(read_calls[0])
    assert "file7.py" in read_cmd_str
    assert "file42.py" in read_cmd_str
    # Should NOT include files that grep didn't match.
    assert "file0.py" not in read_cmd_str

    retriever.scan_and_answer.assert_awaited_once()


async def test_large_repo_grep_no_hits_falls_back():
    """When grep finds nothing in a large repo, fall back to first N files."""
    retriever = _make_retriever()
    retriever.scan_and_answer = AsyncMock(return_value="Fallback answer")

    # Generate 60 files (exceeds threshold).
    large_file_list = "\n".join(f"/workspace/src/mod{i}.py" for i in range(60))

    # Simulate the read output for the first _MAX_FILES (30) files.
    first_30_contents = "\n\n".join(
        f"=== /workspace/src/mod{i}.py ===\n# module {i}" for i in range(30)
    )

    cm = MagicMock()

    async def mock_exec(command: str, timeout: int = 60, **_kw: object) -> ExecResult:
        if "test -d /workspace/.git" in command:
            return ExecResult(
                exit_code=0,
                stdout="ok\n",
                stderr="",
                duration_seconds=0.1,
                command=command,
            )
        if command.startswith("find "):
            return ExecResult(
                exit_code=0,
                stdout=large_file_list,
                stderr="",
                duration_seconds=0.2,
                command=command,
            )
        if command.startswith("grep "):
            # Grep returns nothing -- no matches.
            return ExecResult(
                exit_code=1,
                stdout="",
                stderr="",
                duration_seconds=0.3,
                command=command,
            )
        if command.startswith("for f in "):
            return ExecResult(
                exit_code=0,
                stdout=first_30_contents,
                stderr="",
                duration_seconds=0.3,
                command=command,
            )
        return ExecResult(
            exit_code=0,
            stdout="",
            stderr="",
            duration_seconds=0.1,
            command=command,
        )

    cm.exec = AsyncMock(side_effect=mock_exec)
    tool = RetrievalTool(retriever, cm)

    result = await tool.execute(query="how does authentication work")

    assert result.success

    # Grep was still attempted.
    grep_calls = [c for c in cm.exec.call_args_list if "grep " in str(c)]
    assert grep_calls, "Grep should be attempted even if it finds nothing"

    # The read command should use the first 30 files (fallback).
    read_calls = [c for c in cm.exec.call_args_list if "for f in " in str(c)]
    assert read_calls
    read_cmd_str = str(read_calls[0])
    assert "mod0.py" in read_cmd_str
    assert "mod29.py" in read_cmd_str
    # File 30+ should NOT be included (capped at _MAX_FILES=30).
    assert "mod30.py" not in read_cmd_str


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
