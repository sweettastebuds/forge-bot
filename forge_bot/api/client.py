"""YAML-driven generic async HTTP client for Gitea/Forgejo API."""

from __future__ import annotations

import logging
from typing import Any

import httpx

from forge_bot.api.loader import load_provider_definition
from forge_bot.api.schema import EndpointDef
from forge_bot.config import Settings

logger = logging.getLogger("forge_bot.api.client")


class GenericForgeClient:
    """Async HTTP client that loads endpoint definitions from YAML.

    Instead of one Python method per API endpoint, this client loads
    endpoint definitions from a YAML file and provides:

    - ``call(endpoint_name, **params)`` — execute any defined endpoint
    - ``search(keyword)`` — find endpoints by keyword
    - ``get_endpoint(name)`` — look up a single endpoint definition
    - ``list_endpoints()`` — list all registered endpoint names
    """

    def __init__(self, settings: Settings) -> None:
        base_url = settings.forge_instance_url.rstrip("/")
        self._base_url = base_url
        self._client = httpx.AsyncClient(
            headers={
                "Authorization": f"token {settings.forge_api_token}",
                "Accept": "application/json",
            },
            timeout=30.0,
        )
        self._endpoints: dict[str, EndpointDef] = {}
        self._base_path = "/api/v1"
        self._load_definitions(settings)

    def _load_definitions(self, settings: Settings) -> None:
        """Load endpoint definitions for the configured provider."""
        provider = settings.forge_provider
        api_def = load_provider_definition(provider)
        self._base_path = api_def.base_path
        for ep in api_def.endpoints:
            self._endpoints[ep.name] = ep
        logger.info(
            "Loaded %d API endpoints for provider '%s'",
            len(self._endpoints),
            provider,
        )

    async def call(
        self,
        endpoint_name: str,
        **params: Any,
    ) -> dict[str, Any] | list[Any] | str:
        """Execute a named API endpoint with the given parameters.

        Path params are interpolated into the URL template.
        Query params are appended as ``?key=value``.
        Body params are assembled into a JSON request body.

        Returns parsed JSON (dict/list) or raw text depending on
        the endpoint's ``response_type``.

        Raises:
            ValueError: Unknown endpoint or missing required parameter.
            httpx.HTTPStatusError: API returned a non-2xx status.
        """
        ep = self._endpoints.get(endpoint_name)
        if not ep:
            raise ValueError(f"Unknown API endpoint: {endpoint_name}")

        url_path = ep.path
        query_params: dict[str, str] = {}
        body_params: dict[str, Any] = {}

        form_params: dict[str, Any] = {}

        for param_def in ep.params:
            value = params.get(param_def.name, param_def.default)
            if value is None and param_def.required:
                raise ValueError(
                    f"Missing required parameter '{param_def.name}' for endpoint '{endpoint_name}'"
                )
            if value is None:
                continue

            if param_def.location == "path":
                url_path = url_path.replace(f"{{{param_def.name}}}", str(value))
            elif param_def.location == "query":
                query_params[param_def.name] = str(value)
            elif param_def.location == "body":
                body_params[param_def.name] = value
            elif param_def.location == "form":
                form_params[param_def.name] = value

        full_url = f"{self._base_url}{self._base_path}{url_path}"

        # Merge endpoint-specific headers with defaults.
        headers = dict(ep.headers) if ep.headers else {}

        # Build the request kwargs depending on content type.
        is_multipart = ep.content_type == "multipart/form-data" or form_params

        method = ep.method.upper()
        if method == "GET":
            resp = await self._client.get(full_url, params=query_params, headers=headers)
        elif method in ("POST", "PATCH", "PUT"):
            if is_multipart:
                files = self._build_files(form_params)
                request_method = getattr(self._client, method.lower())
                resp = await request_method(full_url, files=files, headers=headers)
            else:
                request_method = getattr(self._client, method.lower())
                resp = await request_method(full_url, json=body_params, headers=headers)
        elif method == "DELETE":
            resp = await self._client.delete(full_url, headers=headers)
        else:
            raise ValueError(f"Unsupported HTTP method: {method}")

        resp.raise_for_status()

        if ep.response_type == "text":
            return resp.text
        return resp.json()

    def search(self, keyword: str) -> list[EndpointDef]:
        """Search endpoint definitions by keyword.

        Matches against name, description, and tags (case-insensitive).
        Results are sorted by relevance score (highest first).
        """
        keyword_lower = keyword.lower()
        results: list[tuple[int, EndpointDef]] = []

        for ep in self._endpoints.values():
            score = 0
            if keyword_lower in ep.name.lower():
                score += 3
            if keyword_lower in ep.description.lower():
                score += 2
            if any(keyword_lower in tag.lower() for tag in ep.tags):
                score += 1
            if score > 0:
                results.append((score, ep))

        results.sort(key=lambda x: x[0], reverse=True)
        return [ep for _, ep in results]

    def get_endpoint(self, name: str) -> EndpointDef | None:
        """Get an endpoint definition by exact name."""
        return self._endpoints.get(name)

    def list_endpoints(self) -> list[str]:
        """List all registered endpoint names, sorted alphabetically."""
        return sorted(self._endpoints.keys())

    @staticmethod
    def _build_files(
        form_params: dict[str, Any],
    ) -> dict[str, tuple[str, Any]]:
        """Convert form params into httpx ``files`` dict for multipart upload.

        Values can be:
        - ``(filename, content_bytes)`` tuple — used as-is.
        - ``bytes`` — wrapped with a generic filename.
        - ``str`` — treated as a text field.
        """
        files: dict[str, tuple[str, Any]] = {}
        for name, value in form_params.items():
            if isinstance(value, tuple) and len(value) == 2:
                files[name] = value
            elif isinstance(value, bytes):
                files[name] = (name, value)
            else:
                files[name] = (name, str(value).encode())
        return files

    async def close(self) -> None:
        """Close the underlying HTTP client."""
        await self._client.aclose()
