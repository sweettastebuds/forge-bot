"""Tests for forge_bot.tools.registry."""

from __future__ import annotations

from forge_bot.tools.base import BaseTool, ToolParameter, ToolResult
from forge_bot.tools.registry import ToolRegistry


class _EchoTool(BaseTool):
    """Simple tool that echoes its input."""

    name = "echo"
    description = "Echo the input back"
    parameters = [
        ToolParameter("text", "string", "Text to echo"),
    ]

    async def execute(self, **kwargs: object) -> ToolResult:
        return ToolResult(
            tool_name=self.name,
            success=True,
            content=str(kwargs.get("text", "")),
        )


class _FailTool(BaseTool):
    """Tool that always raises."""

    name = "fail"
    description = "Always fails"
    parameters = []

    async def execute(self, **kwargs: object) -> ToolResult:
        msg = "Intentional failure"
        raise RuntimeError(msg)


def test_register_and_get():
    registry = ToolRegistry()
    tool = _EchoTool()
    registry.register(tool)
    assert registry.get("echo") is tool


def test_get_unknown():
    registry = ToolRegistry()
    assert registry.get("nonexistent") is None


def test_openai_schemas():
    registry = ToolRegistry()
    registry.register(_EchoTool())
    schemas = registry.openai_schemas()
    assert len(schemas) == 1
    assert schemas[0]["type"] == "function"
    assert schemas[0]["function"]["name"] == "echo"


def test_openai_schemas_multiple():
    registry = ToolRegistry()
    registry.register(_EchoTool())
    registry.register(_FailTool())
    schemas = registry.openai_schemas()
    assert len(schemas) == 2


def test_prompt_text():
    registry = ToolRegistry()
    registry.register(_EchoTool())
    text = registry.prompt_text()
    assert "echo" in text
    assert "fetch_file" in text  # concrete example
    assert '```tool' in text
    assert "arguments" in text
    assert "RULES:" in text


async def test_execute_success():
    registry = ToolRegistry()
    registry.register(_EchoTool())
    result = await registry.execute("echo", {"text": "hello"})
    assert result.success is True
    assert result.content == "hello"


async def test_execute_unknown_tool():
    registry = ToolRegistry()
    result = await registry.execute("nonexistent", {})
    assert result.success is False
    assert "Unknown tool" in result.content


async def test_execute_catches_exceptions():
    registry = ToolRegistry()
    registry.register(_FailTool())
    result = await registry.execute("fail", {})
    assert result.success is False
    assert "Tool error" in result.content
