"""GetCommitTool — retrieve information about a specific commit."""

from __future__ import annotations

from typing import TYPE_CHECKING

from forge_bot.tools.base import BaseTool, ToolParameter, ToolResult

if TYPE_CHECKING:
    from forge_bot.clients.forge import ForgeClient


class GetCommitTool(BaseTool):
    """Get information about a specific commit by SHA."""

    name = "get_commit"
    description = "Get information about a specific commit (message, author, date)"
    parameters = [
        ToolParameter("sha", "string", "The commit SHA (7 or more characters)"),
    ]

    def __init__(self, forge: ForgeClient, owner: str, repo: str) -> None:
        self._forge = forge
        self._owner = owner
        self._repo = repo

    async def execute(self, **kwargs: object) -> ToolResult:
        sha = str(kwargs.get("sha", ""))

        if not sha:
            return ToolResult(
                tool_name=self.name,
                success=False,
                content="Missing required parameter: sha",
            )

        try:
            data = await self._forge.get_commit(self._owner, self._repo, sha)
            commit = data.get("commit", {})
            message = commit.get("message", "")
            author = commit.get("author", {}).get("name", "unknown")
            date = commit.get("author", {}).get("date", "")
            full_sha = data.get("sha", sha)
            content = (
                f"Commit {full_sha[:12]} by {author} ({date}):\n{message}"
            )
            return ToolResult(
                tool_name=self.name,
                success=True,
                content=content,
            )
        except Exception as exc:
            return ToolResult(
                tool_name=self.name,
                success=False,
                content=f"Could not fetch commit {sha}: {exc}",
            )
