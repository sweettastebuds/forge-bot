"""Tool registry for managing and dispatching tool calls."""

from __future__ import annotations

import logging

from forge_bot.tools.base import BaseTool, ToolResult

logger = logging.getLogger("forge_bot.tools.registry")


class ToolRegistry:
    """Per-request registry that holds tool instances with repo context.

    Created fresh for each webhook event so each tool has the correct
    owner/repo/branch context.
    """

    def __init__(self) -> None:
        self._tools: dict[str, BaseTool] = {}

    def register(self, tool: BaseTool) -> None:
        """Register a tool instance."""
        self._tools[tool.name] = tool

    def get(self, name: str) -> BaseTool | None:
        """Look up a tool by name."""
        return self._tools.get(name)

    def openai_schemas(self) -> list[dict]:
        """Return OpenAI-format tool schemas for native mode."""
        return [t.to_openai_schema() for t in self._tools.values()]

    def prompt_text(self) -> str:
        """Return a formatted block for prompt-based tool calling.

        Includes tool descriptions and a usage example showing the
        expected JSON format.
        """
        lines = ["You have access to the following tools:\n"]
        for tool in self._tools.values():
            lines.append(tool.to_prompt_text())
        lines.append(
            '\nTo use a tool, respond with a JSON block:\n'
            '```tool\n'
            '{"name": "tool_name", "arguments": {"param": "value"}}\n'
            '```\n'
            "You may use multiple tool blocks in a single response.\n"
            "After tools execute, you will receive results and can respond."
        )
        return "\n".join(lines)

    async def execute(self, name: str, arguments: dict) -> ToolResult:
        """Execute a registered tool by name.

        Returns a failure ToolResult if the tool is unknown or raises.
        """
        tool = self._tools.get(name)
        if not tool:
            logger.warning("Unknown tool requested: %s", name)
            return ToolResult(
                tool_name=name,
                success=False,
                content=f"Unknown tool: {name}",
            )
        try:
            return await tool.execute(**arguments)
        except Exception as exc:
            logger.warning("Tool %s raised: %s", name, exc)
            return ToolResult(
                tool_name=name,
                success=False,
                content=f"Tool error: {exc}",
            )
