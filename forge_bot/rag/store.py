"""ChromaDB vector store wrapper."""

from __future__ import annotations

import logging
from typing import Any

from forge_bot.rag.chunker import CodeChunk

logger = logging.getLogger("forge_bot.rag.store")


def _collection_name(repo_full_name: str) -> str:
    """Convert 'owner/repo' to a valid ChromaDB collection name."""
    return repo_full_name.replace("/", "__")


class VectorStore:
    """Persistent ChromaDB wrapper for code chunk storage and retrieval."""

    def __init__(self, persist_path: str) -> None:
        import chromadb

        self._client = chromadb.PersistentClient(path=persist_path)
        logger.info("ChromaDB store initialized at %s", persist_path)

    def get_or_create_collection(self, repo_full_name: str) -> Any:
        """Get or create a collection for the given repository."""
        name = _collection_name(repo_full_name)
        return self._client.get_or_create_collection(
            name=name,
            metadata={"hnsw:space": "cosine"},
        )

    def collection_exists(self, repo_full_name: str) -> bool:
        """Check whether a collection for this repo already has data."""
        name = _collection_name(repo_full_name)
        try:
            coll = self._client.get_collection(name)
            return coll.count() > 0
        except Exception:
            return False

    def upsert(
        self,
        collection: Any,
        chunks: list[CodeChunk],
        embeddings: list[list[float]],
    ) -> None:
        """Upsert chunks with pre-computed embeddings into the collection."""
        if not chunks:
            return

        ids = [
            f"{c.file_path}:{c.start_line}-{c.end_line}" for c in chunks
        ]
        documents = [c.content for c in chunks]
        metadatas = [
            {
                "file_path": c.file_path,
                "language": c.language,
                "symbol_name": c.symbol_name or "",
                "start_line": c.start_line,
                "end_line": c.end_line,
                "token_count": c.token_count,
            }
            for c in chunks
        ]

        collection.upsert(
            ids=ids,
            documents=documents,
            embeddings=embeddings,
            metadatas=metadatas,
        )
        logger.info(
            "Upserted %d chunks into collection %s",
            len(chunks),
            collection.name,
        )

    def query(
        self,
        collection: Any,
        query_embedding: list[float],
        top_k: int = 10,
    ) -> list[dict[str, Any]]:
        """Query the collection and return top-K results with metadata."""
        results = collection.query(
            query_embeddings=[query_embedding],
            n_results=min(top_k, collection.count()) if collection.count() > 0 else top_k,
            include=["documents", "metadatas", "distances"],
        )

        hits: list[dict[str, Any]] = []
        if not results["documents"] or not results["documents"][0]:
            return hits

        for doc, meta, dist in zip(
            results["documents"][0],
            results["metadatas"][0],
            results["distances"][0],
            strict=True,
        ):
            hits.append({
                "content": doc,
                "metadata": meta,
                "distance": dist,
            })

        return hits

    def delete_by_file(
        self, collection: Any, file_path: str,
    ) -> None:
        """Delete all chunks for a specific file from the collection."""
        collection.delete(where={"file_path": file_path})
        logger.info(
            "Deleted chunks for %s from collection %s",
            file_path,
            collection.name,
        )
