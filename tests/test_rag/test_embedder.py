"""Tests for forge_bot.rag.embedder."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from forge_bot.config import Settings
from forge_bot.rag.embedder import Embedder


@pytest.fixture()
def api_settings(monkeypatch: pytest.MonkeyPatch) -> Settings:
    """Settings configured for API-based embeddings."""
    monkeypatch.setenv("FORGE_INSTANCE_URL", "https://gitea.example.com")
    monkeypatch.setenv("FORGE_API_TOKEN", "tok")
    monkeypatch.setenv("FORGE_WEBHOOK_SECRET", "sec")
    monkeypatch.setenv("LLM_API_KEY", "test-key")
    monkeypatch.setenv("RAG_EMBED_URL", "https://embed.example.com/v1")
    monkeypatch.setenv("RAG_EMBED_MODEL", "text-embedding-ada-002")
    for var in ("LLM_BASE_URL", "LLM_MODEL", "LOG_LEVEL"):
        monkeypatch.delenv(var, raising=False)
    return Settings()


@pytest.fixture()
def local_settings(monkeypatch: pytest.MonkeyPatch) -> Settings:
    """Settings configured for local sentence-transformers."""
    monkeypatch.setenv("FORGE_INSTANCE_URL", "https://gitea.example.com")
    monkeypatch.setenv("FORGE_API_TOKEN", "tok")
    monkeypatch.setenv("FORGE_WEBHOOK_SECRET", "sec")
    monkeypatch.setenv("LLM_API_KEY", "test-key")
    monkeypatch.setenv("RAG_EMBED_URL", "")
    monkeypatch.setenv("RAG_EMBED_MODEL", "all-MiniLM-L6-v2")
    for var in ("LLM_BASE_URL", "LLM_MODEL", "LOG_LEVEL"):
        monkeypatch.delenv(var, raising=False)
    return Settings()


@pytest.mark.asyncio
async def test_embed_empty_list(api_settings):
    embedder = Embedder(api_settings)
    result = await embedder.embed([])
    assert result == []


@pytest.mark.asyncio
async def test_embed_via_api(api_settings):
    """API-based embedding should call AsyncOpenAI embeddings.create."""
    embedder = Embedder(api_settings)

    mock_response = MagicMock()
    mock_response.data = [
        MagicMock(embedding=[0.1, 0.2, 0.3]),
        MagicMock(embedding=[0.4, 0.5, 0.6]),
    ]

    mock_client = AsyncMock()
    mock_client.embeddings.create.return_value = mock_response
    embedder._api_client = mock_client

    result = await embedder.embed(["hello", "world"])
    assert len(result) == 2
    assert result[0] == [0.1, 0.2, 0.3]
    mock_client.embeddings.create.assert_called_once_with(
        model="text-embedding-ada-002",
        input=["hello", "world"],
    )


@pytest.mark.asyncio
async def test_embed_query(api_settings):
    """embed_query should return a single vector."""
    embedder = Embedder(api_settings)

    mock_response = MagicMock()
    mock_response.data = [MagicMock(embedding=[0.1, 0.2])]
    mock_client = AsyncMock()
    mock_client.embeddings.create.return_value = mock_response
    embedder._api_client = mock_client

    result = await embedder.embed_query("test query")
    assert result == [0.1, 0.2]


class _FakeArray:
    """Minimal ndarray-like object with .tolist() for testing."""

    def __init__(self, data: list) -> None:
        self._data = data

    def tolist(self) -> list:
        return self._data

    def __iter__(self):
        return iter([_FakeArray(row) for row in self._data])


@pytest.mark.asyncio
async def test_embed_local_model(local_settings):
    """Local embedding should use sentence-transformers."""
    embedder = Embedder(local_settings)

    mock_model = MagicMock()
    mock_model.encode.return_value = _FakeArray([[0.1, 0.2], [0.3, 0.4]])
    embedder._local_model = mock_model

    result = await embedder.embed(["hello", "world"])
    assert len(result) == 2
    assert result[0] == pytest.approx([0.1, 0.2])
    mock_model.encode.assert_called_once_with(
        ["hello", "world"], convert_to_numpy=True,
    )
