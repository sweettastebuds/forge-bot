"""FetchFileTool — retrieve file contents from the repository."""

from __future__ import annotations

from typing import TYPE_CHECKING

from forge_bot.tools.base import BaseTool, ToolParameter, ToolResult

if TYPE_CHECKING:
    from forge_bot.clients.forge import ForgeClient


class FetchFileTool(BaseTool):
    """Fetch the contents of a file from the repository.

    Replaces the V1 ``[FETCH: path]`` marker system.
    """

    name = "fetch_file"
    description = "Fetch the contents of a file from the repository"
    parameters = [
        ToolParameter("path", "string", "File path relative to the repository root"),
        ToolParameter(
            "ref", "string",
            "Branch name or commit SHA to read from (defaults to the repo's default branch)",
            required=False,
        ),
    ]

    def __init__(
        self,
        forge: ForgeClient,
        owner: str,
        repo: str,
        default_branch: str,
        *,
        max_chars: int = 8000,
    ) -> None:
        self._forge = forge
        self._owner = owner
        self._repo = repo
        self._default_branch = default_branch
        self._max_chars = max_chars

    async def execute(self, **kwargs: object) -> ToolResult:
        path = str(kwargs.get("path", ""))
        ref = str(kwargs.get("ref", "")) or self._default_branch

        if not path:
            return ToolResult(
                tool_name=self.name,
                success=False,
                content="Missing required parameter: path",
            )

        try:
            content = await self._forge.get_file_content(
                self._owner, self._repo, path, ref=ref,
            )
            if len(content) > self._max_chars:
                content = content[: self._max_chars] + "\n... (truncated)"
            return ToolResult(
                tool_name=self.name,
                success=True,
                content=content,
            )
        except Exception as exc:
            return ToolResult(
                tool_name=self.name,
                success=False,
                content=f"Could not fetch {path}: {exc}",
            )
