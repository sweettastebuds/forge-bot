"""Tests for forge_bot.rag.ingester."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from forge_bot.rag.chunker import Chunker
from forge_bot.rag.ingester import Ingester, _should_skip_path


def test_should_skip_vendor():
    assert _should_skip_path("vendor/lib/foo.go") is True


def test_should_skip_node_modules():
    assert _should_skip_path("node_modules/express/index.js") is True


def test_should_skip_binary_extension():
    assert _should_skip_path("images/logo.png") is True
    assert _should_skip_path("build/app.exe") is True
    assert _should_skip_path("lib/native.so") is True


def test_should_skip_min_js():
    assert _should_skip_path("dist/bundle.min.js") is True


def test_should_not_skip_normal_file():
    assert _should_skip_path("src/main.py") is False
    assert _should_skip_path("README.md") is False
    assert _should_skip_path("go.mod") is False


def test_should_skip_git():
    assert _should_skip_path(".git/config") is True


def test_should_skip_pycache():
    assert _should_skip_path("src/__pycache__/main.cpython-312.pyc") is True


@pytest.mark.asyncio
async def test_ingest_repo(mock_forge, mock_embedder):
    """Full ingestion should fetch files, chunk, embed, and store."""
    chunker = Chunker(chunk_size=500, overlap=50)
    store = MagicMock()
    collection = MagicMock()
    store.get_or_create_collection.return_value = collection

    ingester = Ingester(mock_forge, chunker, mock_embedder, store)
    count = await ingester.ingest_repo("owner", "repo", "main")

    # Should have fetched tree
    mock_forge.get_repo_tree.assert_called_once_with("owner", "repo", ref="main")

    # Should have fetched eligible files (skipping vendor/ and .png)
    assert mock_forge.get_file_content.call_count >= 3  # README, main.py, utils.py, test_main.py

    # Should have stored chunks
    assert count > 0
    store.upsert.assert_called_once()
    store.get_or_create_collection.assert_called_with("owner/repo")


@pytest.mark.asyncio
async def test_ingest_skips_binaries(mock_forge):
    """Binary files (.png) should be filtered out."""
    mock_forge.get_repo_tree.return_value = [
        {"path": "image.png", "type": "blob"},
        {"path": "video.mp4", "type": "blob"},
    ]

    chunker = Chunker()
    embedder = AsyncMock()
    store = MagicMock()

    ingester = Ingester(mock_forge, chunker, embedder, store)
    count = await ingester.ingest_repo("owner", "repo", "main")

    assert count == 0
    mock_forge.get_file_content.assert_not_called()


@pytest.mark.asyncio
async def test_ingest_skips_vendor(mock_forge):
    """Vendor directories should be filtered out."""
    mock_forge.get_repo_tree.return_value = [
        {"path": "vendor/lib.js", "type": "blob"},
        {"path": "node_modules/express/index.js", "type": "blob"},
    ]

    chunker = Chunker()
    embedder = AsyncMock()
    store = MagicMock()

    ingester = Ingester(mock_forge, chunker, embedder, store)
    count = await ingester.ingest_repo("owner", "repo", "main")

    assert count == 0
    mock_forge.get_file_content.assert_not_called()


@pytest.mark.asyncio
async def test_ingest_skips_large_files(mock_forge):
    """Files exceeding 100KB should be skipped."""
    mock_forge.get_repo_tree.return_value = [
        {"path": "big.py", "type": "blob"},
    ]
    # Return a file that exceeds _MAX_FILE_SIZE
    mock_forge.get_file_content.return_value = "x" * 200_000

    chunker = Chunker()
    embedder = AsyncMock()
    store = MagicMock()

    ingester = Ingester(mock_forge, chunker, embedder, store)
    count = await ingester.ingest_repo("owner", "repo", "main")

    assert count == 0


@pytest.mark.asyncio
async def test_ingest_files_selective(mock_forge, mock_embedder):
    """Selective file re-indexing should delete old chunks first."""
    chunker = Chunker(chunk_size=500, overlap=50)
    store = MagicMock()
    collection = MagicMock()
    store.get_or_create_collection.return_value = collection

    ingester = Ingester(mock_forge, chunker, mock_embedder, store)
    count = await ingester.ingest_files(
        "owner", "repo", "main", ["src/main.py"],
    )

    # Should have deleted old chunks for the file
    store.delete_by_file.assert_called_once_with(collection, "src/main.py")
    assert count > 0
