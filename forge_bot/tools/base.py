"""Base classes for the tool-calling system."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import ClassVar


@dataclass
class ToolResult:
    """Result returned by a tool execution."""

    tool_name: str
    success: bool
    content: str


@dataclass
class ToolParameter:
    """Describes a single parameter for a tool."""

    name: str
    type: str  # "string", "integer", "boolean"
    description: str
    required: bool = True


class BaseTool(ABC):
    """Abstract base class for all tools.

    Subclasses set ``name``, ``description``, and ``parameters`` as class
    attributes and implement ``execute()``.
    """

    name: ClassVar[str] = ""
    description: ClassVar[str] = ""
    parameters: ClassVar[list[ToolParameter]] = []

    @abstractmethod
    async def execute(self, **kwargs: object) -> ToolResult:
        """Execute the tool with the given arguments."""

    def to_openai_schema(self) -> dict:
        """Convert to OpenAI function-calling tools format."""
        properties: dict[str, dict[str, str]] = {}
        required: list[str] = []
        for p in self.parameters:
            properties[p.name] = {
                "type": p.type,
                "description": p.description,
            }
            if p.required:
                required.append(p.name)

        schema: dict = {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": {
                    "type": "object",
                    "properties": properties,
                    "required": required,
                },
            },
        }
        return schema

    def to_prompt_text(self) -> str:
        """Generate plain-text description for prompt-based fallback."""
        parts = []
        for p in self.parameters:
            label = f"{p.name}: {p.type}"
            if not p.required:
                label += " (optional)"
            parts.append(label)
        params = ", ".join(parts)
        return f"- {self.name}({params}): {self.description}"
