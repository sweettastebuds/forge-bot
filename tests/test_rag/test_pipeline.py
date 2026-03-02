"""Tests for forge_bot.rag.pipeline."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from forge_bot.rag.pipeline import RAGPipeline


def _setup_pipeline(settings, mock_api_client, store=None, embedder=None, ingester=None):
    """Create a pipeline with pre-injected mocks (avoids lazy import issues)."""
    pipeline = RAGPipeline(settings, mock_api_client)
    pipeline._store = store or MagicMock()
    pipeline._embedder = embedder or AsyncMock()
    pipeline._chunker = MagicMock()
    pipeline._ingester = ingester or AsyncMock()
    return pipeline


@pytest.mark.asyncio
async def test_retrieve_returns_empty_when_not_indexed(settings, mock_api_client):
    """Retrieve on a non-indexed repo should return empty string."""
    mock_store = MagicMock()
    mock_store.collection_exists.return_value = False

    pipeline = _setup_pipeline(settings, mock_api_client, store=mock_store)
    result = await pipeline.retrieve("owner", "repo", "what does hello do?")

    assert result == ""


@pytest.mark.asyncio
async def test_ensure_indexed_skips_if_exists(settings, mock_api_client):
    """Should skip indexing if collection already has data."""
    mock_store = MagicMock()
    mock_store.collection_exists.return_value = True
    mock_ingester = AsyncMock()

    pipeline = _setup_pipeline(
        settings, mock_api_client, store=mock_store, ingester=mock_ingester,
    )
    count = await pipeline.ensure_indexed("owner", "repo", "main")

    assert count == 0
    mock_ingester.ingest_repo.assert_not_called()


@pytest.mark.asyncio
async def test_ensure_indexed_force(settings, mock_api_client):
    """Force indexing should re-index even if collection exists."""
    mock_store = MagicMock()
    mock_store.collection_exists.return_value = True
    mock_ingester = AsyncMock()
    mock_ingester.ingest_repo.return_value = 42

    pipeline = _setup_pipeline(
        settings, mock_api_client, store=mock_store, ingester=mock_ingester,
    )
    count = await pipeline.ensure_indexed("owner", "repo", "main", force=True)

    assert count == 42
    mock_ingester.ingest_repo.assert_called_once_with("owner", "repo", "main")


@pytest.mark.asyncio
async def test_retrieve_formats_context(settings, mock_api_client):
    """Retrieve should format hits as markdown."""
    mock_store = MagicMock()
    mock_store.collection_exists.return_value = True
    mock_collection = MagicMock()
    mock_store.get_or_create_collection.return_value = mock_collection

    # Configure store.query() to return hits (not collection.query)
    mock_store.query.return_value = [
        {
            "content": "def hello(): pass",
            "metadata": {
                "file_path": "src/main.py",
                "language": "python",
                "symbol_name": "hello",
                "start_line": 1,
                "end_line": 1,
                "token_count": 5,
            },
            "distance": 0.1,
        },
    ]

    mock_embedder = AsyncMock()
    mock_embedder.embed_query.return_value = [0.5] * 8

    pipeline = _setup_pipeline(
        settings, mock_api_client, store=mock_store, embedder=mock_embedder,
    )
    result = await pipeline.retrieve("owner", "repo", "what is hello?")

    assert "src/main.py" in result
    assert "hello" in result
    assert "```python" in result


@pytest.mark.asyncio
async def test_reindex_files(settings, mock_api_client):
    """reindex_files should delegate to ingester."""
    mock_ingester = AsyncMock()
    mock_ingester.ingest_files.return_value = 5

    pipeline = _setup_pipeline(settings, mock_api_client, ingester=mock_ingester)
    count = await pipeline.reindex_files("owner", "repo", "main", ["src/main.py"])

    assert count == 5
    mock_ingester.ingest_files.assert_called_once_with(
        "owner", "repo", "main", ["src/main.py"],
    )


def test_format_context_with_symbol():
    """_format_context should include symbol name when present."""
    hits = [
        {
            "content": "def greet(): pass",
            "metadata": {
                "file_path": "src/greet.py",
                "language": "python",
                "symbol_name": "greet",
                "start_line": 5,
                "end_line": 5,
                "token_count": 4,
            },
            "distance": 0.05,
        },
    ]
    result = RAGPipeline._format_context(hits)
    assert "`greet`" in result
    assert "src/greet.py" in result


def test_format_context_without_symbol():
    """_format_context should work without empty symbol name."""
    hits = [
        {
            "content": "some text",
            "metadata": {
                "file_path": "README.md",
                "language": "text",
                "symbol_name": "",
                "start_line": 1,
                "end_line": 3,
                "token_count": 2,
            },
            "distance": 0.2,
        },
    ]
    result = RAGPipeline._format_context(hits)
    assert "README.md" in result
