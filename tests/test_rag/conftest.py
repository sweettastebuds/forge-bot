"""Shared fixtures for RAG tests."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from forge_bot.config import Settings


@pytest.fixture()
def settings(monkeypatch: pytest.MonkeyPatch) -> Settings:
    """Settings with RAG enabled."""
    monkeypatch.setenv("FORGE_INSTANCE_URL", "https://gitea.example.com")
    monkeypatch.setenv("FORGE_API_TOKEN", "tok")
    monkeypatch.setenv("FORGE_WEBHOOK_SECRET", "sec")
    monkeypatch.setenv("LLM_API_KEY", "llm-key")
    monkeypatch.setenv("RAG_ENABLED", "true")
    monkeypatch.setenv("RAG_CHUNK_SIZE", "500")
    monkeypatch.setenv("RAG_CHUNK_OVERLAP", "50")
    monkeypatch.setenv("RAG_TOP_K", "5")
    monkeypatch.setenv("RAG_MAX_CONTEXT_TOKENS", "2000")
    monkeypatch.setenv("RAG_STORE_PATH", "/tmp/test-chromadb")
    # Clear env leaks
    for var in ("LLM_BASE_URL", "LLM_MODEL", "LOG_LEVEL"):
        monkeypatch.delenv(var, raising=False)
    return Settings()


_DEFAULT_TREE = [
    {"path": "README.md", "type": "blob"},
    {"path": "src/main.py", "type": "blob"},
    {"path": "src/utils.py", "type": "blob"},
    {"path": "tests/test_main.py", "type": "blob"},
    {"path": "vendor/lib.js", "type": "blob"},
    {"path": "image.png", "type": "blob"},
]

_DEFAULT_FILES = {
    "README.md": "# My Project\nA sample project.",
    "src/main.py": (
        "def hello():\n    print('hello world')\n\ndef goodbye():\n    print('goodbye')\n"
    ),
    "src/utils.py": ("def add(a, b):\n    return a + b\n\ndef multiply(a, b):\n    return a * b\n"),
    "tests/test_main.py": ("from src.main import hello\n\ndef test_hello():\n    hello()\n"),
}


@pytest.fixture()
def mock_api_client() -> AsyncMock:
    """Mocked GenericForgeClient using call() dispatch."""
    client = AsyncMock()

    async def _call(endpoint_name, **params):
        if endpoint_name == "get_repo_tree":
            return {"tree": list(_DEFAULT_TREE)}
        elif endpoint_name == "get_file_content":
            filepath = params.get("filepath", "")
            if filepath in _DEFAULT_FILES:
                return _DEFAULT_FILES[filepath]
            raise Exception(f"File not found: {filepath}")
        raise ValueError(f"Unmocked endpoint: {endpoint_name}")

    client.call = AsyncMock(side_effect=_call)
    return client


@pytest.fixture()
def mock_embedder() -> MagicMock:
    """Mocked Embedder that returns fixed-length vectors."""
    embedder = AsyncMock()

    async def _embed(texts):
        # Return a simple deterministic embedding
        return [[float(i) / 10.0] * 8 for i, _ in enumerate(texts)]

    async def _embed_query(query):
        return [0.5] * 8

    embedder.embed.side_effect = _embed
    embedder.embed_query.side_effect = _embed_query
    return embedder
