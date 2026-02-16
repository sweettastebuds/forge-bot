"""TodoTool: manage a visible todo list in the status comment."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

from forge_bot.status.formatter import TodoItem
from forge_bot.tools.base import BaseTool, ToolParameter, ToolResult

if TYPE_CHECKING:
    from forge_bot.status.manager import StatusCommentManager


class TodoTool(BaseTool):
    """Create and update a todo list that is displayed in the status comment.

    Use this to show the user what steps you are taking and track
    progress.  The todo list is visible in real-time on the Gitea issue.
    """

    name = "todo"
    description = (
        "Create or update a todo list visible in the status comment. "
        "Use 'set' to replace the list, 'check' to mark an item done, "
        "'add' to append a new item."
    )
    parameters = [
        ToolParameter(
            "action",
            "string",
            "One of: 'set' (replace entire list), 'check' (mark item done), "
            "'add' (add a new item)",
        ),
        ToolParameter(
            "items",
            "string",
            "For 'set': JSON array of strings (the todo items). "
            "For 'check': the text of the item to mark done. "
            "For 'add': the text of the new item.",
        ),
    ]

    def __init__(self, status_manager: StatusCommentManager) -> None:
        self._status = status_manager
        self._todos: list[TodoItem] = []

    async def execute(self, **kwargs: object) -> ToolResult:
        action = str(kwargs.get("action", ""))
        items_raw = str(kwargs.get("items", ""))

        if action == "set":
            try:
                items_list = json.loads(items_raw)
                if not isinstance(items_list, list):
                    return ToolResult(
                        self.name, False, "Items must be a JSON array"
                    )
                self._todos = [TodoItem(text=str(t)) for t in items_list]
            except json.JSONDecodeError as exc:
                return ToolResult(
                    self.name, False, f"Invalid items JSON: {exc}"
                )
        elif action == "check":
            found = False
            for todo in self._todos:
                if todo.text == items_raw:
                    todo.done = True
                    found = True
                    break
            if not found:
                return ToolResult(
                    self.name,
                    False,
                    f"Todo item not found: '{items_raw}'",
                )
        elif action == "add":
            if not items_raw:
                return ToolResult(self.name, False, "Missing item text")
            self._todos.append(TodoItem(text=items_raw))
        else:
            return ToolResult(
                self.name,
                False,
                f"Unknown action '{action}'. Use 'set', 'check', or 'add'.",
            )

        await self._status.update_todos(self._todos)

        done_count = sum(1 for t in self._todos if t.done)
        return ToolResult(
            self.name,
            True,
            f"Todo list updated ({done_count}/{len(self._todos)} done)",
        )

    @property
    def items(self) -> list[TodoItem]:
        """Current todo items."""
        return list(self._todos)
