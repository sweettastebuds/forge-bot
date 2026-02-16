"""Tests for the new tool implementations."""

from __future__ import annotations

import json
from unittest.mock import AsyncMock, MagicMock

import pytest

from forge_bot.api.schema import EndpointDef, EndpointParam
from forge_bot.container.manager import ExecResult
from forge_bot.status.formatter import TodoItem
from forge_bot.tools.api_call import ApiCallTool
from forge_bot.tools.base import ToolResult
from forge_bot.tools.exec_tool import ExecTool
from forge_bot.tools.registry import ToolRegistry
from forge_bot.tools.search_api import SearchApiTool
from forge_bot.tools.todo import TodoTool


# -- Helpers --

def _mock_api_client(
    *,
    search_results: list[EndpointDef] | None = None,
    call_result: dict | str = "",
    call_error: Exception | None = None,
) -> MagicMock:
    api = MagicMock()
    api.search.return_value = search_results or []
    if call_error:
        api.call = AsyncMock(side_effect=call_error)
    else:
        api.call = AsyncMock(return_value=call_result)
    return api


def _mock_container_manager(
    *,
    exec_result: ExecResult | None = None,
    exec_error: Exception | None = None,
) -> MagicMock:
    cm = MagicMock()
    if exec_error:
        cm.exec = AsyncMock(side_effect=exec_error)
    else:
        result = exec_result or ExecResult(
            exit_code=0,
            stdout="hello\n",
            stderr="",
            duration_seconds=0.5,
            command="echo hello",
        )
        cm.exec = AsyncMock(return_value=result)
    return cm


def _mock_status_manager() -> MagicMock:
    sm = MagicMock()
    sm.update_todos = AsyncMock()
    return sm


# -- SearchApiTool --

class TestSearchApiTool:
    @pytest.mark.asyncio
    async def test_search_found(self) -> None:
        ep = EndpointDef(
            name="get_file_content",
            method="GET",
            path="/repos/{owner}/{repo}/raw/{filepath}",
            description="Get raw file content",
            tags=["file"],
            params=[
                EndpointParam(name="filepath", type="string", location="path"),
            ],
        )
        api = _mock_api_client(search_results=[ep])
        tool = SearchApiTool(api)

        result = await tool.execute(keyword="file")
        assert result.success
        assert "get_file_content" in result.content
        assert "GET" in result.content
        assert "Found 1 endpoint" in result.content

    @pytest.mark.asyncio
    async def test_search_no_results(self) -> None:
        api = _mock_api_client(search_results=[])
        tool = SearchApiTool(api)

        result = await tool.execute(keyword="nonexistent")
        assert result.success
        assert "No API endpoints found" in result.content

    @pytest.mark.asyncio
    async def test_search_missing_keyword(self) -> None:
        api = _mock_api_client()
        tool = SearchApiTool(api)

        result = await tool.execute()
        assert not result.success
        assert "Missing keyword" in result.content


# -- ApiCallTool --

class TestApiCallTool:
    @pytest.mark.asyncio
    async def test_call_json_response(self) -> None:
        api = _mock_api_client(call_result={"login": "bot", "id": 1})
        tool = ApiCallTool(api, owner="owner", repo="repo")

        result = await tool.execute(endpoint="get_authenticated_user")
        assert result.success
        data = json.loads(result.content)
        assert data["login"] == "bot"

    @pytest.mark.asyncio
    async def test_call_text_response(self) -> None:
        api = _mock_api_client(call_result="diff --git a/file.py")
        tool = ApiCallTool(api, owner="owner", repo="repo")

        result = await tool.execute(
            endpoint="get_pull_diff",
            params='{"index": 1}',
        )
        assert result.success
        assert "diff --git" in result.content

    @pytest.mark.asyncio
    async def test_auto_inject_owner_repo(self) -> None:
        api = _mock_api_client(call_result={"tree": []})
        tool = ApiCallTool(api, owner="myowner", repo="myrepo")

        await tool.execute(
            endpoint="get_repo_tree",
            params='{"ref": "main"}',
        )

        call_kwargs = api.call.call_args
        assert call_kwargs[1]["owner"] == "myowner"
        assert call_kwargs[1]["repo"] == "myrepo"

    @pytest.mark.asyncio
    async def test_explicit_owner_not_overridden(self) -> None:
        api = _mock_api_client(call_result={})
        tool = ApiCallTool(api, owner="default", repo="default")

        await tool.execute(
            endpoint="get_repo",
            params='{"owner": "explicit", "repo": "explicit"}',
        )

        call_kwargs = api.call.call_args
        assert call_kwargs[1]["owner"] == "explicit"
        assert call_kwargs[1]["repo"] == "explicit"

    @pytest.mark.asyncio
    async def test_missing_endpoint(self) -> None:
        api = _mock_api_client()
        tool = ApiCallTool(api)

        result = await tool.execute()
        assert not result.success
        assert "Missing endpoint" in result.content

    @pytest.mark.asyncio
    async def test_invalid_params_json(self) -> None:
        api = _mock_api_client()
        tool = ApiCallTool(api)

        result = await tool.execute(endpoint="get_repo", params="not json")
        assert not result.success
        assert "Invalid params JSON" in result.content

    @pytest.mark.asyncio
    async def test_api_error(self) -> None:
        api = _mock_api_client(call_error=ValueError("Unknown endpoint"))
        tool = ApiCallTool(api)

        result = await tool.execute(endpoint="bad_endpoint")
        assert not result.success
        assert "API error" in result.content

    @pytest.mark.asyncio
    async def test_truncates_large_response(self) -> None:
        big_data = {"key": "x" * 10_000}
        api = _mock_api_client(call_result=big_data)
        tool = ApiCallTool(api)

        result = await tool.execute(endpoint="some_endpoint")
        assert result.success
        assert len(result.content) <= 8_000 + 50
        assert "truncated" in result.content


# -- ExecTool --

class TestExecTool:
    @pytest.mark.asyncio
    async def test_exec_success(self) -> None:
        cm = _mock_container_manager()
        tool = ExecTool(cm)

        result = await tool.execute(command="echo hello")
        assert result.success
        assert "hello" in result.content
        assert "[0.5s]" in result.content

    @pytest.mark.asyncio
    async def test_exec_failure(self) -> None:
        cm = _mock_container_manager(
            exec_result=ExecResult(
                exit_code=1,
                stdout="",
                stderr="error: not found",
                duration_seconds=0.3,
                command="false",
            )
        )
        tool = ExecTool(cm)

        result = await tool.execute(command="false")
        assert not result.success
        assert "Exit code: 1" in result.content
        assert "error: not found" in result.content

    @pytest.mark.asyncio
    async def test_missing_command(self) -> None:
        cm = _mock_container_manager()
        tool = ExecTool(cm)

        result = await tool.execute()
        assert not result.success
        assert "Missing command" in result.content

    @pytest.mark.asyncio
    async def test_blocked_command(self) -> None:
        cm = _mock_container_manager()
        tool = ExecTool(cm)

        result = await tool.execute(command="rm -rf /")
        assert not result.success
        assert "blocked" in result.content

    @pytest.mark.asyncio
    async def test_blocked_mkfs(self) -> None:
        cm = _mock_container_manager()
        tool = ExecTool(cm)

        result = await tool.execute(command="mkfs.ext4 /dev/sda")
        assert not result.success

    @pytest.mark.asyncio
    async def test_timeout_capped(self) -> None:
        cm = _mock_container_manager()
        tool = ExecTool(cm)

        await tool.execute(command="sleep 200", timeout=999)
        # Verify exec was called with timeout capped at 120
        cm.exec.assert_called_once()
        call_kwargs = cm.exec.call_args[1]
        assert call_kwargs["timeout"] == 120

    @pytest.mark.asyncio
    async def test_exec_error_handled(self) -> None:
        cm = _mock_container_manager(exec_error=RuntimeError("container gone"))
        tool = ExecTool(cm)

        result = await tool.execute(command="echo hello")
        assert not result.success
        assert "Exec error" in result.content

    @pytest.mark.asyncio
    async def test_no_output(self) -> None:
        cm = _mock_container_manager(
            exec_result=ExecResult(
                exit_code=0,
                stdout="",
                stderr="",
                duration_seconds=0.1,
                command="true",
            )
        )
        tool = ExecTool(cm)

        result = await tool.execute(command="true")
        assert result.success
        assert "(no output)" in result.content


# -- TodoTool --

class TestTodoTool:
    @pytest.mark.asyncio
    async def test_set(self) -> None:
        sm = _mock_status_manager()
        tool = TodoTool(sm)

        result = await tool.execute(
            action="set",
            items='["Fetch files", "Run tests", "Write response"]',
        )
        assert result.success
        assert "0/3 done" in result.content
        assert len(tool.items) == 3
        sm.update_todos.assert_called_once()

    @pytest.mark.asyncio
    async def test_check(self) -> None:
        sm = _mock_status_manager()
        tool = TodoTool(sm)

        await tool.execute(
            action="set", items='["Fetch files", "Run tests"]'
        )
        sm.update_todos.reset_mock()

        result = await tool.execute(action="check", items="Fetch files")
        assert result.success
        assert "1/2 done" in result.content
        assert tool.items[0].done is True
        assert tool.items[1].done is False

    @pytest.mark.asyncio
    async def test_check_not_found(self) -> None:
        sm = _mock_status_manager()
        tool = TodoTool(sm)

        await tool.execute(action="set", items='["Task A"]')
        result = await tool.execute(action="check", items="Nonexistent")
        assert not result.success
        assert "not found" in result.content

    @pytest.mark.asyncio
    async def test_add(self) -> None:
        sm = _mock_status_manager()
        tool = TodoTool(sm)

        await tool.execute(action="set", items='["Task A"]')
        result = await tool.execute(action="add", items="Task B")
        assert result.success
        assert len(tool.items) == 2
        assert tool.items[1].text == "Task B"

    @pytest.mark.asyncio
    async def test_add_empty(self) -> None:
        sm = _mock_status_manager()
        tool = TodoTool(sm)

        result = await tool.execute(action="add", items="")
        assert not result.success

    @pytest.mark.asyncio
    async def test_unknown_action(self) -> None:
        sm = _mock_status_manager()
        tool = TodoTool(sm)

        result = await tool.execute(action="delete", items="x")
        assert not result.success
        assert "Unknown action" in result.content

    @pytest.mark.asyncio
    async def test_set_invalid_json(self) -> None:
        sm = _mock_status_manager()
        tool = TodoTool(sm)

        result = await tool.execute(action="set", items="not json")
        assert not result.success
        assert "Invalid items JSON" in result.content

    @pytest.mark.asyncio
    async def test_set_non_array(self) -> None:
        sm = _mock_status_manager()
        tool = TodoTool(sm)

        result = await tool.execute(action="set", items='{"a": "b"}')
        assert not result.success
        assert "must be a JSON array" in result.content


# -- ToolRegistry --

class TestToolRegistry:
    @pytest.mark.asyncio
    async def test_register_and_execute(self) -> None:
        api = _mock_api_client(search_results=[])
        tool = SearchApiTool(api)

        registry = ToolRegistry()
        registry.register(tool)

        assert registry.get("search_api") is tool
        result = await registry.execute("search_api", {"keyword": "test"})
        assert result.success

    @pytest.mark.asyncio
    async def test_unknown_tool(self) -> None:
        registry = ToolRegistry()
        result = await registry.execute("nonexistent", {})
        assert not result.success
        assert "Unknown tool" in result.content

    def test_openai_schemas(self) -> None:
        api = _mock_api_client()
        tool = SearchApiTool(api)
        registry = ToolRegistry()
        registry.register(tool)

        schemas = registry.openai_schemas()
        assert len(schemas) == 1
        assert schemas[0]["type"] == "function"
        assert schemas[0]["function"]["name"] == "search_api"

    def test_prompt_text(self) -> None:
        api = _mock_api_client()
        tool = SearchApiTool(api)
        registry = ToolRegistry()
        registry.register(tool)

        text = registry.prompt_text()
        assert "search_api" in text
        assert "```tool" in text
