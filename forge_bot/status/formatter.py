"""Markdown formatting helpers for status comment rendering."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class TodoItem:
    """A single todo item with completion status."""

    text: str
    done: bool = False


@dataclass
class ToolCallRecord:
    """Record of a single tool invocation for status display."""

    tool_name: str
    arguments_summary: str
    result_summary: str
    success: bool
    duration_seconds: float


def format_todo_list(todos: list[TodoItem]) -> str:
    """Render a todo list as GitHub-flavored markdown checkboxes."""
    if not todos:
        return ""
    lines = []
    for todo in todos:
        check = "x" if todo.done else " "
        lines.append(f"- [{check}] {todo.text}")
    return "\n".join(lines)


def abbreviate(text: str, max_chars: int = 100) -> str:
    """Shorten text for display, adding ellipsis if truncated."""
    if len(text) <= max_chars:
        return text
    return text[: max_chars - 3] + "..."


def format_tool_call(record: ToolCallRecord) -> str:
    """Format a single tool call record as a markdown list item."""
    icon = "+" if record.success else "x"
    args = abbreviate(record.arguments_summary, 60)
    line = (
        f"- [{icon}] `{record.tool_name}({args})` "
        f"({record.duration_seconds:.1f}s)"
    )
    if record.result_summary:
        result = abbreviate(record.result_summary, 100)
        # Escape any markdown in result summary
        result = result.replace("\n", " ")
        line += f"\n  > {result}"
    return line


def format_tool_calls_section(records: list[ToolCallRecord], max_shown: int = 10) -> str:
    """Render tool call history in a collapsible details block."""
    if not records:
        return ""
    shown = records[-max_shown:]
    lines = [
        f"<details><summary>Tool calls ({len(records)})</summary>\n",
    ]
    for record in shown:
        lines.append(format_tool_call(record))
    lines.append("\n</details>")
    return "\n".join(lines)
