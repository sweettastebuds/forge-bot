"""HTTP client for the Gitea/Forgejo REST API."""

import logging
from typing import Any

import httpx

from forge_bot.config import Settings
from forge_bot.models import WebhookUser

logger = logging.getLogger("forge_bot.clients.forge")


class ForgeClient:
    """Async wrapper around the Gitea/Forgejo v1 API.

    Auth: ``Authorization: token {api_token}`` on every request.
    """

    def __init__(self, settings: Settings) -> None:
        base_url = settings.forge_instance_url.rstrip("/")
        self._api_base = f"{base_url}/api/v1"
        self._client = httpx.AsyncClient(
            base_url=self._api_base,
            headers={
                "Authorization": f"token {settings.forge_api_token}",
                "Accept": "application/json",
            },
            timeout=30.0,
        )

    async def get_self(self) -> WebhookUser:
        """GET /api/v1/user — resolve the authenticated bot's identity."""
        resp = await self._client.get("/user")
        resp.raise_for_status()
        data = resp.json()
        return WebhookUser(id=data["id"], login=data["login"])

    async def get_issue_comments(
        self, owner: str, repo: str, issue_index: int
    ) -> list[dict[str, Any]]:
        """GET /repos/{owner}/{repo}/issues/{index}/comments."""
        resp = await self._client.get(
            f"/repos/{owner}/{repo}/issues/{issue_index}/comments"
        )
        resp.raise_for_status()
        return resp.json()

    async def get_pull_diff(
        self, owner: str, repo: str, pull_index: int
    ) -> str:
        """GET /repos/{owner}/{repo}/pulls/{index}.diff — raw unified diff."""
        resp = await self._client.get(
            f"/repos/{owner}/{repo}/pulls/{pull_index}.diff",
            headers={"Accept": "text/plain"},
        )
        resp.raise_for_status()
        return resp.text

    async def get_pull_files(
        self, owner: str, repo: str, pull_index: int
    ) -> list[dict[str, Any]]:
        """GET /repos/{owner}/{repo}/pulls/{index}/files — changed file list."""
        resp = await self._client.get(
            f"/repos/{owner}/{repo}/pulls/{pull_index}/files"
        )
        resp.raise_for_status()
        return resp.json()

    async def post_comment(
        self, owner: str, repo: str, issue_index: int, body: str
    ) -> dict[str, Any]:
        """POST /repos/{owner}/{repo}/issues/{index}/comments."""
        resp = await self._client.post(
            f"/repos/{owner}/{repo}/issues/{issue_index}/comments",
            json={"body": body},
        )
        resp.raise_for_status()
        return resp.json()

    async def get_file_content(
        self, owner: str, repo: str, filepath: str, ref: str = ""
    ) -> str:
        """GET /repos/{owner}/{repo}/raw/{filepath}?ref={ref} — file content."""
        params = {}
        if ref:
            params["ref"] = ref
        resp = await self._client.get(
            f"/repos/{owner}/{repo}/raw/{filepath}",
            params=params,
            headers={"Accept": "text/plain"},
        )
        resp.raise_for_status()
        return resp.text

    async def get_repo_tree(
        self,
        owner: str,
        repo: str,
        ref: str = "main",
        *,
        recursive: bool = True,
    ) -> list[dict[str, Any]]:
        """GET /repos/{owner}/{repo}/git/trees/{ref} — repo file tree."""
        params = {"recursive": "true"} if recursive else {}
        resp = await self._client.get(
            f"/repos/{owner}/{repo}/git/trees/{ref}",
            params=params,
        )
        resp.raise_for_status()
        data = resp.json()
        return data.get("tree", [])

    async def get_commit(
        self, owner: str, repo: str, sha: str
    ) -> dict[str, Any]:
        """GET /repos/{owner}/{repo}/git/commits/{sha} — single commit info."""
        resp = await self._client.get(
            f"/repos/{owner}/{repo}/git/commits/{sha}"
        )
        resp.raise_for_status()
        return resp.json()

    async def download_url(self, url: str) -> str:
        """GET an arbitrary fully-qualified URL using the bot's auth token.

        Useful for downloading attachments from the Forge instance.
        """
        resp = await self._client.get(url)
        resp.raise_for_status()
        return resp.text

    async def close(self) -> None:
        """Close the underlying HTTP client."""
        await self._client.aclose()
