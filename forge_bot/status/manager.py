"""Two-comment status system for real-time observability.

Comment 1 (status): Posted immediately, edited as work progresses.
Comment 2 (response): Posted when the LLM produces a final answer.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from forge_bot.status.formatter import (
    TodoItem,
    ToolCallRecord,
    format_todo_list,
    format_tool_calls_section,
)

if TYPE_CHECKING:
    from forge_bot.api.client import GenericForgeClient

logger = logging.getLogger("forge_bot.status.manager")

_MAX_STATUS_CHARS = 4000


class StatusCommentManager:
    """Manages two Gitea comments per interaction for observability.

    The **status comment** is posted immediately and edited as work
    progresses, showing the current phase, todo list, and tool call
    history.  The **response comment** is posted separately when the
    LLM produces a final answer.
    """

    def __init__(
        self,
        api_client: GenericForgeClient,
        owner: str,
        repo: str,
        issue_index: int,
    ) -> None:
        self._api = api_client
        self._owner = owner
        self._repo = repo
        self._issue_index = issue_index
        self._status_comment_id: int | None = None
        self._todos: list[TodoItem] = []
        self._tool_calls: list[ToolCallRecord] = []
        self._current_phase: str = "Starting..."

    # -- public API --

    async def post_initial_status(self) -> None:
        """Post the initial status comment with a 'thinking...' state."""
        body = self._render_status()
        result = await self._api.call(
            "post_issue_comment",
            owner=self._owner,
            repo=self._repo,
            index=self._issue_index,
            body=body,
        )
        self._status_comment_id = result["id"]
        logger.info(
            "Status comment %d posted on %s/%s#%d",
            self._status_comment_id,
            self._owner,
            self._repo,
            self._issue_index,
        )

    async def update_phase(self, phase: str) -> None:
        """Update the current phase text and refresh the status comment."""
        self._current_phase = phase
        await self._update_status_comment()

    async def update_todos(self, todos: list[TodoItem]) -> None:
        """Replace the todo list and refresh the status comment."""
        self._todos = list(todos)
        await self._update_status_comment()

    async def record_tool_call(self, record: ToolCallRecord) -> None:
        """Append a tool call to the history and refresh the status comment."""
        self._tool_calls.append(record)
        await self._update_status_comment()

    async def post_response(self, body: str) -> dict:
        """Post the final LLM response as a separate comment."""
        return await self._api.call(
            "post_issue_comment",
            owner=self._owner,
            repo=self._repo,
            index=self._issue_index,
            body=body,
        )

    async def finalize_status(self, final_phase: str = "Done") -> None:
        """Mark the status comment as complete."""
        self._current_phase = final_phase
        await self._update_status_comment()

    # -- properties --

    @property
    def status_comment_id(self) -> int | None:
        return self._status_comment_id

    @property
    def todos(self) -> list[TodoItem]:
        return list(self._todos)

    @property
    def tool_calls(self) -> list[ToolCallRecord]:
        return list(self._tool_calls)

    # -- internals --

    async def _update_status_comment(self) -> None:
        """Edit the status comment with the current rendered state."""
        if not self._status_comment_id:
            return
        body = self._render_status()
        try:
            await self._api.call(
                "edit_issue_comment",
                owner=self._owner,
                repo=self._repo,
                id=self._status_comment_id,
                body=body,
            )
        except Exception:
            logger.warning(
                "Failed to update status comment %d",
                self._status_comment_id,
                exc_info=True,
            )

    def _render_status(self) -> str:
        """Render the status comment body as markdown."""
        lines: list[str] = []

        lines.append(f"**Status:** {self._current_phase}")
        lines.append("")

        if self._todos:
            lines.append("**Progress:**")
            lines.append(format_todo_list(self._todos))
            lines.append("")

        tool_section = format_tool_calls_section(self._tool_calls)
        if tool_section:
            lines.append(tool_section)

        body = "\n".join(lines)

        # Enforce character limit for small-model friendliness.
        if len(body) > _MAX_STATUS_CHARS:
            body = body[: _MAX_STATUS_CHARS - 20] + "\n\n... (trimmed)"

        return body
