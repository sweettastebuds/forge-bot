"""ApiCallTool: execute any YAML-defined Gitea/Forgejo API endpoint."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

from forge_bot.tools.base import BaseTool, ToolParameter, ToolResult

if TYPE_CHECKING:
    from forge_bot.api.client import GenericForgeClient


class ApiCallTool(BaseTool):
    """Execute any YAML-defined Gitea/Forgejo API endpoint.

    The LLM specifies the endpoint name and parameters.  ``owner`` and
    ``repo`` are auto-injected from the event context if not provided.
    """

    name = "api_call"
    description = (
        "Call a Gitea/Forgejo API endpoint by name. "
        "Use search_api first to find the right endpoint. "
        "Owner and repo are auto-filled from context if omitted."
    )
    parameters = [
        ToolParameter(
            "endpoint",
            "string",
            "API endpoint name (e.g. 'get_file_content', 'post_issue_comment')",
        ),
        ToolParameter(
            "params",
            "string",
            "JSON object of parameters for the endpoint (e.g. "
            '\'{"filepath": "src/main.py", "ref": "main"}\')',
            required=False,
        ),
    ]

    _MAX_RESULT_CHARS = 8_000

    def __init__(
        self,
        api_client: GenericForgeClient,
        *,
        owner: str = "",
        repo: str = "",
    ) -> None:
        self._api = api_client
        self._default_owner = owner
        self._default_repo = repo

    async def execute(self, **kwargs: object) -> ToolResult:
        endpoint = str(kwargs.get("endpoint", ""))
        params_raw = str(kwargs.get("params", "{}"))

        if not endpoint:
            return ToolResult(self.name, False, "Missing endpoint name")

        try:
            params = json.loads(params_raw)
        except json.JSONDecodeError as exc:
            return ToolResult(
                self.name, False, f"Invalid params JSON: {exc}"
            )

        # Auto-inject owner/repo from context if not provided.
        if "owner" not in params and self._default_owner:
            params["owner"] = self._default_owner
        if "repo" not in params and self._default_repo:
            params["repo"] = self._default_repo

        try:
            result = await self._api.call(endpoint, **params)
            if isinstance(result, (dict, list)):
                content = json.dumps(result, indent=2)
            else:
                content = str(result)
            if len(content) > self._MAX_RESULT_CHARS:
                content = (
                    content[: self._MAX_RESULT_CHARS] + "\n... (truncated)"
                )
            return ToolResult(self.name, True, content)
        except Exception as exc:
            return ToolResult(self.name, False, f"API error: {exc}")
