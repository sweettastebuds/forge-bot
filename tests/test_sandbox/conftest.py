"""Shared fixtures for sandbox tests."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from forge_bot.config import Settings
from forge_bot.sandbox.images import ImageRegistry


@pytest.fixture()
def settings(monkeypatch: pytest.MonkeyPatch) -> Settings:
    """Minimal Settings for sandbox testing."""
    monkeypatch.setenv("FORGE_INSTANCE_URL", "https://gitea.example.com")
    monkeypatch.setenv("FORGE_API_TOKEN", "tok")
    monkeypatch.setenv("FORGE_WEBHOOK_SECRET", "sec")
    monkeypatch.setenv("LLM_API_KEY", "llm-key")
    monkeypatch.setenv("SANDBOX_ENABLED", "true")
    monkeypatch.setenv("SANDBOX_TIMEOUT", "30")
    monkeypatch.setenv("SANDBOX_MEMORY", "256m")
    monkeypatch.setenv("SANDBOX_CPUS", "0.5")
    monkeypatch.setenv("SANDBOX_IMAGES_FILE", "")
    # Clear any env-leaked defaults
    for var in ("LLM_BASE_URL", "LLM_MODEL", "LOG_LEVEL"):
        monkeypatch.delenv(var, raising=False)
    return Settings()


@pytest.fixture()
def registry() -> ImageRegistry:
    """Default image registry."""
    return ImageRegistry()


@pytest.fixture()
def mock_docker_client() -> MagicMock:
    """A mocked docker.DockerClient."""
    client = MagicMock()
    # Mock container object returned by containers.run()
    container = MagicMock()
    container.wait.return_value = {"StatusCode": 0}
    container.logs.side_effect = lambda stdout=True, stderr=True: (
        b"hello world\n" if stdout else b""
    )
    container.attrs = {"State": {"OOMKilled": False}}
    container.reload.return_value = None
    container.remove.return_value = None
    client.containers.run.return_value = container
    client.images.pull.return_value = MagicMock()
    client.close.return_value = None
    return client
