"""Dual-backend embedding client (local sentence-transformers or API)."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from forge_bot.config import Settings

logger = logging.getLogger("forge_bot.rag.embedder")


class Embedder:
    """Compute embeddings using a local model or remote API.

    - If ``settings.rag_embed_url`` is set, uses the OpenAI-compatible
      ``/v1/embeddings`` endpoint.
    - Otherwise, loads a sentence-transformers model locally.
    """

    def __init__(self, settings: Settings) -> None:
        self._model_name = settings.rag_embed_model
        self._embed_url = settings.rag_embed_url
        self._api_key = settings.llm_api_key
        self._local_model: Any = None
        self._api_client: Any = None

    async def embed(self, texts: list[str]) -> list[list[float]]:
        """Embed a batch of texts and return vectors."""
        if not texts:
            return []
        if self._embed_url:
            return await self._embed_via_api(texts)
        return self._embed_local(texts)

    async def embed_query(self, query: str) -> list[float]:
        """Embed a single query string."""
        results = await self.embed([query])
        return results[0]

    async def _embed_via_api(self, texts: list[str]) -> list[list[float]]:
        """Call an OpenAI-compatible /v1/embeddings endpoint."""
        if self._api_client is None:
            from openai import AsyncOpenAI

            self._api_client = AsyncOpenAI(
                api_key=self._api_key,
                base_url=self._embed_url,
            )

        response = await self._api_client.embeddings.create(
            model=self._model_name,
            input=texts,
        )
        return [item.embedding for item in response.data]

    def _embed_local(self, texts: list[str]) -> list[list[float]]:
        """Compute embeddings using a local sentence-transformers model."""
        if self._local_model is None:
            from sentence_transformers import SentenceTransformer

            logger.info("Loading embedding model: %s", self._model_name)
            self._local_model = SentenceTransformer(self._model_name)
            logger.info("Embedding model loaded")

        embeddings = self._local_model.encode(texts, convert_to_numpy=True)
        return [e.tolist() for e in embeddings]
