"""Repository ingestion: fetch files via Forge API, chunk, embed, store."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from forge_bot.rag.chunker import Chunker, CodeChunk, detect_language

if TYPE_CHECKING:
    from forge_bot.clients.forge import ForgeClient
    from forge_bot.rag.embedder import Embedder
    from forge_bot.rag.store import VectorStore

logger = logging.getLogger("forge_bot.rag.ingester")

# File-level filters
_MAX_FILE_SIZE = 100_000  # 100 KB
_SKIP_DIRS = {"vendor", "node_modules", ".git", "__pycache__", ".venv", "venv"}
_SKIP_EXTENSIONS = {
    ".min.js", ".min.css", ".pb.go", ".png", ".jpg", ".jpeg", ".gif",
    ".ico", ".svg", ".woff", ".woff2", ".ttf", ".eot", ".mp3", ".mp4",
    ".zip", ".tar", ".gz", ".bz2", ".pdf", ".exe", ".dll", ".so",
    ".dylib", ".pyc", ".pyo", ".class", ".o", ".obj",
}
_EMBED_BATCH_SIZE = 32


def _should_skip_path(path: str) -> bool:
    """Return True if the file should be excluded from ingestion."""
    parts = path.split("/")
    for part in parts:
        if part in _SKIP_DIRS:
            return True

    # Check extension (including compound extensions like .min.js)
    filename = parts[-1].lower()
    for ext in _SKIP_EXTENSIONS:
        if filename.endswith(ext):
            return True

    return False


class Ingester:
    """Ingest a repository into the vector store via the Forge API."""

    def __init__(
        self,
        forge: ForgeClient,
        chunker: Chunker,
        embedder: Embedder,
        store: VectorStore,
    ) -> None:
        self._forge = forge
        self._chunker = chunker
        self._embedder = embedder
        self._store = store

    async def ingest_repo(
        self, owner: str, repo: str, ref: str,
    ) -> int:
        """Ingest all eligible files from a repository.

        Returns the number of chunks stored.
        """
        logger.info("Ingesting %s/%s (ref=%s)", owner, repo, ref)

        try:
            tree = await self._forge.get_repo_tree(owner, repo, ref=ref)
        except Exception:
            logger.exception("Failed to fetch repo tree for %s/%s", owner, repo)
            return 0

        file_paths = [
            e["path"]
            for e in tree
            if e.get("type") == "blob" and not _should_skip_path(e["path"])
        ]

        logger.info(
            "Ingesting %s/%s: %d eligible files from tree",
            owner, repo, len(file_paths),
        )

        return await self._ingest_file_list(
            owner, repo, ref, file_paths, f"{owner}/{repo}",
        )

    async def ingest_files(
        self, owner: str, repo: str, ref: str, file_paths: list[str],
    ) -> int:
        """Selectively re-index specific files.

        Returns the number of chunks stored.
        """
        repo_full_name = f"{owner}/{repo}"

        # Delete old chunks for these files first.
        collection = self._store.get_or_create_collection(repo_full_name)
        for path in file_paths:
            self._store.delete_by_file(collection, path)

        return await self._ingest_file_list(
            owner, repo, ref, file_paths, repo_full_name,
        )

    async def _ingest_file_list(
        self,
        owner: str,
        repo: str,
        ref: str,
        file_paths: list[str],
        repo_full_name: str,
    ) -> int:
        """Fetch, chunk, embed, and store a list of files."""
        all_chunks: list[CodeChunk] = []

        for path in file_paths:
            try:
                content = await self._forge.get_file_content(
                    owner, repo, path, ref=ref,
                )
            except Exception:
                logger.warning("Could not fetch %s, skipping", path)
                continue

            if len(content) > _MAX_FILE_SIZE:
                logger.debug("Skipping %s (too large: %d bytes)", path, len(content))
                continue

            language = detect_language(path)
            chunks = self._chunker.chunk_file(content, path, language)
            all_chunks.extend(chunks)

        if not all_chunks:
            logger.info("No chunks produced for %s", repo_full_name)
            return 0

        # Embed in batches.
        all_embeddings: list[list[float]] = []
        for i in range(0, len(all_chunks), _EMBED_BATCH_SIZE):
            batch = all_chunks[i : i + _EMBED_BATCH_SIZE]
            texts = [c.content for c in batch]
            embeddings = await self._embedder.embed(texts)
            all_embeddings.extend(embeddings)

        # Store.
        collection = self._store.get_or_create_collection(repo_full_name)
        self._store.upsert(collection, all_chunks, all_embeddings)

        logger.info(
            "Ingested %s: %d chunks stored",
            repo_full_name, len(all_chunks),
        )
        return len(all_chunks)
