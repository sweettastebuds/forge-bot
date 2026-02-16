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

        Includes tool descriptions and a concrete usage example showing the
        expected JSON format.  The instructions are explicit so that smaller
        models can reliably follow the format.
        """
        lines = ["You have access to the following tools:\n"]
        for tool in self._tools.values():
            lines.append(tool.to_prompt_text())
        lines.append(
            "\nTo use a tool, you MUST respond with a fenced code block "
            "whose language tag is exactly `tool` (not json, not python — "
            "just `tool`). Inside the block, write a single JSON object with "
            '"name" and "arguments" keys.\n'
            "\n"
            "EXAMPLE — reading a file:\n"
            "```tool\n"
            '{"name": "fetch_file", "arguments": {"path": "src/main.py"}}\n'
            "```\n"
            "\n"
            "EXAMPLE — searching code:\n"
            "```tool\n"
            '{"name": "search_code", "arguments": {"query": "def handle"}}\n'
            "```\n"
            "\n"
            "RULES:\n"
            "- You may include multiple ```tool blocks in one response.\n"
            "- After you use a tool, you will receive the results and can "
            "then answer the user's question.\n"
            "- Only use a tool when you need information you don't already "
            "have.\n"
            "- Do NOT wrap tool blocks inside other code blocks or markdown."
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
