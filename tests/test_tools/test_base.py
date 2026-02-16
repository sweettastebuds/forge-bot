"""Tests for forge_bot.tools.base."""

from __future__ import annotations

from forge_bot.tools.base import BaseTool, ToolParameter, ToolResult


class _DummyTool(BaseTool):
    """Concrete tool for testing the base class."""

    name = "dummy"
    description = "A dummy tool for testing"
    parameters = [
        ToolParameter("required_arg", "string", "A required argument"),
        ToolParameter("optional_arg", "integer", "An optional argument", required=False),
    ]

    async def execute(self, **kwargs: object) -> ToolResult:
        return ToolResult(
            tool_name=self.name,
            success=True,
            content=f"got: {kwargs}",
        )


def test_tool_result_dataclass():
    result = ToolResult(tool_name="test", success=True, content="hello")
    assert result.tool_name == "test"
    assert result.success is True
    assert result.content == "hello"


def test_tool_result_failure():
    result = ToolResult(tool_name="test", success=False, content="error")
    assert result.success is False


def test_to_openai_schema():
    tool = _DummyTool()
    schema = tool.to_openai_schema()

    assert schema["type"] == "function"
    assert schema["function"]["name"] == "dummy"
    assert schema["function"]["description"] == "A dummy tool for testing"

    params = schema["function"]["parameters"]
    assert params["type"] == "object"
    assert "required_arg" in params["properties"]
    assert "optional_arg" in params["properties"]
    assert params["properties"]["required_arg"]["type"] == "string"
    assert params["properties"]["optional_arg"]["type"] == "integer"
    assert params["required"] == ["required_arg"]


def test_to_openai_schema_required_list():
    """Only required parameters should appear in the 'required' list."""
    tool = _DummyTool()
    schema = tool.to_openai_schema()
    assert "optional_arg" not in schema["function"]["parameters"]["required"]


def test_to_prompt_text():
    tool = _DummyTool()
    text = tool.to_prompt_text()

    assert "dummy" in text
    assert "required_arg: string" in text
    assert "optional_arg: integer (optional)" in text
    assert "A dummy tool for testing" in text


async def test_execute():
    tool = _DummyTool()
    result = await tool.execute(required_arg="hello", optional_arg=42)
    assert result.success is True
    assert "hello" in result.content
