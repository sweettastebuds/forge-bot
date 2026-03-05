"""Tests for forge_bot.clients.llm."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from forge_bot.clients.llm import LLMClient
from forge_bot.config import Settings


@pytest.fixture
def settings(monkeypatch: pytest.MonkeyPatch) -> Settings:
    monkeypatch.setenv("FORGE_INSTANCE_URL", "https://gitea.example.com")
    monkeypatch.setenv("FORGE_API_TOKEN", "test-token")
    monkeypatch.setenv("FORGE_WEBHOOK_SECRET", "test-secret")
    monkeypatch.setenv("LLM_API_KEY", "test-llm-key")
    monkeypatch.setenv("LLM_BASE_URL", "https://api.example.com/v1")
    monkeypatch.setenv("LLM_MODEL", "test-model")
    monkeypatch.setenv("LLM_MAX_CONCURRENT", "2")
    return Settings()


def _mock_completion(content: str) -> MagicMock:
    """Build a mock ChatCompletion response."""
    choice = MagicMock()
    choice.message.content = content
    resp = MagicMock()
    resp.choices = [choice]
    return resp


async def test_chat_returns_response(settings: Settings):
    mock_create = AsyncMock(return_value=_mock_completion("Hello!"))
    with patch("forge_bot.clients.llm.AsyncOpenAI") as mock_cls:
        mock_cls.return_value.chat.completions.create = mock_create
        mock_cls.return_value.close = AsyncMock()

        client = LLMClient(settings)
        result = await client.chat("You are helpful.", "Hi there")
        await client.close()

    assert result == "Hello!"
    mock_create.assert_awaited_once()
    call_kwargs = mock_create.call_args.kwargs
    assert call_kwargs["model"] == "test-model"
    assert call_kwargs["messages"][0]["role"] == "system"
    assert call_kwargs["messages"][1]["role"] == "user"


async def test_chat_uses_custom_temperature(settings: Settings):
    mock_create = AsyncMock(return_value=_mock_completion("OK"))
    with patch("forge_bot.clients.llm.AsyncOpenAI") as mock_cls:
        mock_cls.return_value.chat.completions.create = mock_create
        mock_cls.return_value.close = AsyncMock()

        client = LLMClient(settings)
        await client.chat("sys", "usr", temperature=0.9, max_tokens=100)
        await client.close()

    call_kwargs = mock_create.call_args.kwargs
    assert call_kwargs["temperature"] == 0.9
    assert call_kwargs["max_tokens"] == 100


async def test_chat_handles_empty_content(settings: Settings):
    mock_create = AsyncMock(return_value=_mock_completion(None))
    with patch("forge_bot.clients.llm.AsyncOpenAI") as mock_cls:
        mock_cls.return_value.chat.completions.create = mock_create
        mock_cls.return_value.close = AsyncMock()

        client = LLMClient(settings)
        # None content should be coerced to empty string.
        choice = MagicMock()
        choice.message.content = None
        resp = MagicMock()
        resp.choices = [choice]
        mock_create.return_value = resp

        result = await client.chat("sys", "usr")
        await client.close()

    assert result == ""


async def test_chat_handles_empty_choices(settings: Settings):
    """LLM returning empty choices returns empty string."""
    resp = MagicMock()
    resp.choices = []
    mock_create = AsyncMock(return_value=resp)
    with patch("forge_bot.clients.llm.AsyncOpenAI") as mock_cls:
        mock_cls.return_value.chat.completions.create = mock_create
        mock_cls.return_value.close = AsyncMock()

        client = LLMClient(settings)
        result = await client.chat("sys", "usr")
        await client.close()

    assert result == ""
