"""SearchCodeTool — search for text patterns across repository files."""

from __future__ import annotations

import fnmatch
from typing import TYPE_CHECKING

from forge_bot.tools.base import BaseTool, ToolParameter, ToolResult

if TYPE_CHECKING:
    from forge_bot.clients.forge import ForgeClient

_MAX_FILES = 30
_MAX_RESULTS = 10


class SearchCodeTool(BaseTool):
    """Search for a text pattern across repository files."""

    name = "search_code"
    description = (
        "Search for a text pattern across repository files. "
        "Returns matching file paths with line numbers and content."
    )
    parameters = [
        ToolParameter(
            "query", "string",
            "The text to search for (case-insensitive)",
        ),
        ToolParameter(
            "file_pattern", "string",
            "Glob pattern to filter files (e.g. '*.py', 'src/**/*.js')",
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
        tree_paths: list[str] | None = None,
    ) -> None:
        self._forge = forge
        self._owner = owner
        self._repo = repo
        self._default_branch = default_branch
        self._tree_paths = tree_paths or []

    async def execute(self, **kwargs: object) -> ToolResult:
        query = str(kwargs.get("query", ""))
        file_pattern = str(kwargs.get("file_pattern", ""))

        if not query:
            return ToolResult(
                tool_name=self.name,
                success=False,
                content="Missing required parameter: query",
            )

        # Filter paths by glob pattern.
        paths = list(self._tree_paths)
        if file_pattern:
            paths = [p for p in paths if fnmatch.fnmatch(p, file_pattern)]

        # Prioritize files whose name contains the query term.
        query_lower = query.lower()
        paths.sort(key=lambda p: query_lower not in p.lower())

        results: list[str] = []
        files_checked = 0
        for path in paths[:_MAX_FILES]:
            if len(results) >= _MAX_RESULTS:
                break
            try:
                content = await self._forge.get_file_content(
                    self._owner, self._repo, path, ref=self._default_branch,
                )
                files_checked += 1
                for i, line in enumerate(content.splitlines(), 1):
                    if query_lower in line.lower():
                        results.append(f"{path}:{i}: {line.strip()}")
                        if len(results) >= _MAX_RESULTS:
                            break
            except Exception:
                continue

        if not results:
            return ToolResult(
                tool_name=self.name,
                success=True,
                content=(
                    f"No matches found for '{query}' "
                    f"in {files_checked} file(s) searched."
                ),
            )

        header = f"Found {len(results)} match(es) in {files_checked} file(s):\n"
        return ToolResult(
            tool_name=self.name,
            success=True,
            content=header + "\n".join(results),
        )
