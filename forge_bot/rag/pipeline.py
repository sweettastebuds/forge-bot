"""RAG pipeline: orchestrates ingest, chunk, embed, store, and retrieve."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from forge_bot.api.client import GenericForgeClient
    from forge_bot.config import Settings

logger = logging.getLogger("forge_bot.rag.pipeline")


class RAGPipeline:
    """Orchestrate the full RAG workflow with lazy initialization."""

    def __init__(self, settings: Settings, api_client: GenericForgeClient) -> None:
        self._settings = settings
        self._api = api_client
        self._chunker = None
        self._embedder = None
        self._store = None
        self._ingester = None

    def _ensure_initialized(self) -> None:
        """Lazily initialize chunker, embedder, store, and ingester."""
        if self._store is not None:
            return

        from forge_bot.rag.chunker import Chunker
        from forge_bot.rag.embedder import Embedder
        from forge_bot.rag.ingester import Ingester
        from forge_bot.rag.store import VectorStore

        self._chunker = Chunker(
            chunk_size=self._settings.rag_chunk_size,
            overlap=self._settings.rag_chunk_overlap,
        )
        self._embedder = Embedder(self._settings)
        self._store = VectorStore(self._settings.rag_store_path)
        self._ingester = Ingester(
            self._api, self._chunker, self._embedder, self._store,
        )
        logger.info("RAG pipeline initialized")

    async def ensure_indexed(
        self,
        owner: str,
        repo: str,
        ref: str,
        *,
        force: bool = False,
    ) -> int:
        """Ensure the repository is indexed. Returns chunk count.

        If the collection already has data and *force* is False, skips
        re-indexing.
        """
        self._ensure_initialized()
        assert self._store is not None
        assert self._ingester is not None

        repo_full_name = f"{owner}/{repo}"

        if not force and self._store.collection_exists(repo_full_name):
            logger.info("Collection for %s already exists, skipping index", repo_full_name)
            return 0

        return await self._ingester.ingest_repo(owner, repo, ref)

    async def retrieve(
        self,
        owner: str,
        repo: str,
        query: str,
        top_k: int = 10,
    ) -> str:
        """Retrieve relevant code context for a query.

        Returns a formatted markdown string of code chunks.
        """
        self._ensure_initialized()
        assert self._embedder is not None
        assert self._store is not None

        repo_full_name = f"{owner}/{repo}"

        if not self._store.collection_exists(repo_full_name):
            return ""

        collection = self._store.get_or_create_collection(repo_full_name)
        query_embedding = await self._embedder.embed_query(query)
        hits = self._store.query(collection, query_embedding, top_k=top_k)

        if not hits:
            return ""

        return self._format_context(hits)

    async def reindex_files(
        self,
        owner: str,
        repo: str,
        ref: str,
        paths: list[str],
    ) -> int:
        """Re-index specific files (e.g., after a PR merge)."""
        self._ensure_initialized()
        assert self._ingester is not None

        return await self._ingester.ingest_files(owner, repo, ref, paths)

    @staticmethod
    def _format_context(hits: list[dict]) -> str:
        """Format retrieval hits as a markdown context string."""
        parts: list[str] = []
        total_tokens = 0

        for hit in hits:
            meta = hit["metadata"]
            content = hit["content"]
            token_count = meta.get("token_count", len(content) // 4)
            total_tokens += token_count

            header = f"### {meta['file_path']}"
            if meta.get("symbol_name"):
                header += f" — `{meta['symbol_name']}`"
            header += f" (lines {meta['start_line']}–{meta['end_line']})"

            parts.append(f"{header}\n```{meta['language']}\n{content}\n```")

        return "\n\n".join(parts)
