"""Shared fixtures for integration tests.

These tests exercise real module interactions while mocking only the
external boundaries: Docker daemon, LLM HTTP API, and Gitea/Forgejo HTTP API.
"""

from __future__ import annotations

import hashlib
import hmac
import json
from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from httpx import ASGITransport, AsyncClient

from forge_bot.api.client import GenericForgeClient
from forge_bot.api.loader import clear_cache
from forge_bot.config import Settings
from forge_bot.container.manager import ExecResult
from forge_bot.server import app
from forge_bot.status.formatter import ToolCallRecord


# ---------------------------------------------------------------------------
# Settings
# ---------------------------------------------------------------------------

TEST_SECRET = "integration-test-secret"

TEST_ENV = {
    "FORGE_INSTANCE_URL": "https://gitea.example.com",
    "FORGE_API_TOKEN": "test-token",
    "FORGE_WEBHOOK_SECRET": TEST_SECRET,
    "LLM_API_KEY": "test-llm-key",
    "LLM_MODEL": "test-model",
    "LLM_BASE_URL": "https://llm.example.com/v1",
    "LLM_CONTEXT_WINDOW": "8192",
    "LLM_MAX_TOKENS": "1024",
    "CONTAINER_TIMEOUT": "60",
}


@pytest.fixture
def integration_settings(monkeypatch: pytest.MonkeyPatch) -> Settings:
    """Settings object pre-configured for integration tests."""
    for k, v in TEST_ENV.items():
        monkeypatch.setenv(k, v)
    return Settings()


@pytest.fixture(autouse=True)
def _set_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """Set required env vars for all integration tests."""
    for k, v in TEST_ENV.items():
        monkeypatch.setenv(k, v)


@pytest.fixture(autouse=True)
def _clear_api_cache() -> None:
    """Clear YAML loader cache between tests."""
    clear_cache()


# ---------------------------------------------------------------------------
# HMAC helpers
# ---------------------------------------------------------------------------


def sign_payload(body: bytes, secret: str = TEST_SECRET) -> str:
    """Compute HMAC-SHA256 hex digest."""
    return hmac.new(secret.encode(), body, hashlib.sha256).hexdigest()


def webhook_headers(
    body: bytes,
    event: str = "pull_request",
    delivery: str = "integ-uuid-1",
) -> dict[str, str]:
    """Build Gitea webhook headers with valid HMAC signature."""
    return {
        "x-gitea-event": event,
        "x-gitea-signature": sign_payload(body),
        "x-gitea-delivery": delivery,
        "content-type": "application/json",
    }


# ---------------------------------------------------------------------------
# Webhook payloads
# ---------------------------------------------------------------------------


@pytest.fixture
def pr_payload() -> dict[str, Any]:
    """Complete pull_request webhook payload."""
    return {
        "action": "opened",
        "number": 42,
        "pull_request": {
            "id": 42,
            "number": 42,
            "title": "Add login endpoint",
            "body": "Implements JWT-based authentication.",
            "state": "open",
            "user": {"id": 10, "login": "alice"},
            "head": {"ref": "feature/login", "sha": "aaa111"},
            "base": {"ref": "main", "sha": "bbb222"},
            "assignees": [],
            "requested_reviewers": [],
        },
        "repository": {
            "full_name": "org/myapp",
            "clone_url": "https://gitea.example.com/org/myapp.git",
            "default_branch": "main",
        },
        "sender": {"id": 10, "login": "alice"},
    }


@pytest.fixture
def comment_payload() -> dict[str, Any]:
    """Complete issue_comment webhook payload with @mention."""
    return {
        "action": "created",
        "comment": {
            "id": 200,
            "body": "@forge-bot explain how the auth module works",
            "user": {"id": 10, "login": "alice"},
        },
        "issue": {
            "number": 7,
            "title": "Auth module question",
            "body": "I need help understanding the auth flow.",
            "pull_request": None,
            "assignees": [],
        },
        "is_pull": False,
        "repository": {
            "full_name": "org/myapp",
            "clone_url": "https://gitea.example.com/org/myapp.git",
            "default_branch": "main",
        },
        "sender": {"id": 10, "login": "alice"},
    }


@pytest.fixture
def self_sent_comment_payload() -> dict[str, Any]:
    """Comment payload from the bot itself (should be dropped by self-loop guard)."""
    return {
        "action": "created",
        "comment": {
            "id": 300,
            "body": "@forge-bot follow-up",
            "user": {"id": 99, "login": "forge-bot"},
        },
        "issue": {
            "number": 7,
            "title": "Test issue",
            "body": "",
            "pull_request": None,
            "assignees": [],
        },
        "is_pull": False,
        "repository": {
            "full_name": "org/myapp",
            "clone_url": "https://gitea.example.com/org/myapp.git",
            "default_branch": "main",
        },
        "sender": {"id": 99, "login": "forge-bot"},
    }


# ---------------------------------------------------------------------------
# Mock LLM responses
# ---------------------------------------------------------------------------


def make_llm_text_response(content: str = "Here is my answer.") -> Any:
    """Create a mock LLM response with no tool calls (final text)."""
    msg = SimpleNamespace(content=content, tool_calls=None)
    msg.model_dump = lambda: {
        "role": "assistant",
        "content": content,
        "tool_calls": None,
    }
    return SimpleNamespace(choices=[SimpleNamespace(message=msg)])


def make_llm_tool_response(commands: list[str]) -> Any:
    """Create a mock LLM response requesting execute tool calls."""
    tcs = []
    for i, cmd in enumerate(commands):
        tcs.append(
            SimpleNamespace(
                id=f"call_{i}",
                function=SimpleNamespace(
                    name="execute",
                    arguments=json.dumps({"command": cmd}),
                ),
            )
        )
    msg = SimpleNamespace(content=None, tool_calls=tcs)
    msg.model_dump = lambda: {
        "role": "assistant",
        "content": None,
        "tool_calls": [
            {
                "id": tc.id,
                "type": "function",
                "function": {
                    "name": tc.function.name,
                    "arguments": tc.function.arguments,
                },
            }
            for tc in tcs
        ],
    }
    return SimpleNamespace(choices=[SimpleNamespace(message=msg)])


# ---------------------------------------------------------------------------
# Mock container
# ---------------------------------------------------------------------------


@dataclass
class FakeExecResult:
    """Lightweight replacement for container.manager.ExecResult."""

    exit_code: int = 0
    stdout: str = ""
    stderr: str = ""
    duration_seconds: float = 0.05
    command: str = ""


def make_mock_container(
    exec_results: dict[str, FakeExecResult] | None = None,
) -> MagicMock:
    """Build a ContainerManager mock with programmable exec responses.

    *exec_results* maps a substring to the result returned when the
    executed command contains that substring.  Lookups are first-match.
    """
    default = FakeExecResult(stdout="ok")
    mapping = exec_results or {}

    async def _exec(command: str, timeout: int | None = None, **kw: Any) -> FakeExecResult:
        for pattern, result in mapping.items():
            if pattern in command:
                return FakeExecResult(
                    exit_code=result.exit_code,
                    stdout=result.stdout,
                    stderr=result.stderr,
                    duration_seconds=result.duration_seconds,
                    command=command,
                )
        return FakeExecResult(stdout=default.stdout, command=command)

    container = MagicMock()
    container.create = AsyncMock()
    container.destroy = AsyncMock()
    container.exec = AsyncMock(side_effect=_exec)
    container.clone_url = "https://test-token@gitea.example.com/org/myapp.git"
    container.default_branch = "main"
    return container


# ---------------------------------------------------------------------------
# Mock Gitea API (tracks calls)
# ---------------------------------------------------------------------------


class FakeApiTracker:
    """Records all API calls for assertion and provides canned responses."""

    def __init__(self) -> None:
        self.calls: list[tuple[str, dict[str, Any]]] = []
        self._responses: dict[str, Any] = {
            "get_authenticated_user": {"id": 99, "login": "forge-bot"},
            "post_issue_comment": {"id": 1},
            "edit_issue_comment": {"id": 1},
            "create_issue_comment": {"id": 1},
            "get_issue_comments": [],
            "upload_comment_attachment": {"id": 1},
        }
        self._comment_counter = 0

    def set_response(self, endpoint: str, response: Any) -> None:
        self._responses[endpoint] = response

    async def call(self, endpoint: str, **kwargs: Any) -> Any:
        self.calls.append((endpoint, kwargs))
        if endpoint == "post_issue_comment":
            self._comment_counter += 1
            return {"id": self._comment_counter}
        return self._responses.get(endpoint, {})

    def calls_for(self, endpoint: str) -> list[dict[str, Any]]:
        return [kw for name, kw in self.calls if name == endpoint]

    @property
    def posted_comments(self) -> list[str]:
        return [kw["body"] for _, kw in self.calls if _ == "post_issue_comment"]

    @property
    def edited_comments(self) -> list[str]:
        return [kw.get("body", "") for _, kw in self.calls if _ == "edit_issue_comment"]


@pytest.fixture
def api_tracker() -> FakeApiTracker:
    return FakeApiTracker()
