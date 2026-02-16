"""SearchApiTool: search available API endpoints by keyword."""

from __future__ import annotations

from typing import TYPE_CHECKING

from forge_bot.tools.base import BaseTool, ToolParameter, ToolResult

if TYPE_CHECKING:
    from forge_bot.api.client import GenericForgeClient


class SearchApiTool(BaseTool):
    """Search Gitea/Forgejo API endpoints by keyword.

    Returns matching endpoint names, methods, parameters, and descriptions
    so the LLM can discover which API calls are available.
    """

    name = "search_api"
    description = (
        "Search Gitea/Forgejo API endpoints by keyword. "
        "Returns matching endpoint names, methods, and descriptions. "
        "Use this to discover which API calls are available."
    )
    parameters = [
        ToolParameter(
            "keyword",
            "string",
            "Search term (e.g. 'pull request', 'label', 'branch')",
        ),
    ]

    def __init__(self, api_client: GenericForgeClient) -> None:
        self._api = api_client

    async def execute(self, **kwargs: object) -> ToolResult:
        keyword = str(kwargs.get("keyword", ""))
        if not keyword:
            return ToolResult(self.name, False, "Missing keyword")

        matches = self._api.search(keyword)
        if not matches:
            return ToolResult(
                self.name,
                True,
                f"No API endpoints found matching '{keyword}'",
            )

        lines = [f"Found {len(matches)} endpoint(s):\n"]
        for ep in matches[:10]:
            params_summary = ", ".join(
                f"{p.name}: {p.type}" + ("" if p.required else "?")
                for p in ep.params
            )
            lines.append(
                f"- {ep.method} {ep.name}({params_summary})\n"
                f"  {ep.description}"
            )
        return ToolResult(self.name, True, "\n".join(lines))
