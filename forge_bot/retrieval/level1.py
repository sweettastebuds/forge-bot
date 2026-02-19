"""Level 1 — BM25 keyword retrieval.

Documents are split into ~512-token chunks.  The LLM generates multilingual
keywords from the user's query, and BM25 ranks chunks by relevance.  Only
the top-ranked chunks fitting within the token budget are returned.

This is the fastest level: one LLM call for keyword expansion, then pure
Python scoring.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

from forge_bot.clients.llm import LLMClient
from forge_bot.retrieval.bm25 import BM25Index, _tokenize
from forge_bot.retrieval.chunking import Chunk
from forge_bot.retrieval.token_budget import TokenBudget

logger = logging.getLogger("forge_bot.retrieval.level1")

_KEYWORD_SYSTEM_PROMPT = (
    "You are a search keyword generator. Given a question or task, "
    "produce a flat list of search keywords (one per line) that would "
    "help find relevant code or documentation. Include synonyms, "
    "abbreviations, and related technical terms. Output ONLY the "
    "keywords, nothing else."
)


@dataclass
class RetrievalResult:
    """Result from a retrieval level."""

    chunks: list[Chunk]
    total_tokens: int
    query_keywords: list[str]


class Level1Retriever:
    """BM25-based keyword retrieval (fast, may miss semantic matches)."""

    def __init__(
        self,
        llm: LLMClient,
        budget: TokenBudget,
    ) -> None:
        self._llm = llm
        self._budget = budget

    async def retrieve(
        self,
        query: str,
        documents: list[Chunk],
        *,
        top_k: int = 20,
    ) -> RetrievalResult:
        """Rank *documents* by BM25 relevance to *query*.

        Steps:
        1. Ask LLM to expand *query* into search keywords.
        2. Build BM25 index over chunk contents.
        3. Return top-k chunks that fit within the token budget.
        """
        if not documents:
            return RetrievalResult(chunks=[], total_tokens=0, query_keywords=[])

        # Step 1: keyword expansion via LLM.
        keywords = await self._expand_keywords(query)
        logger.debug("Expanded keywords: %s", keywords)

        # Step 2: build index and score.
        texts = [c.content for c in documents]
        index = BM25Index.build(texts)
        ranked = index.query(keywords, top_k=top_k)

        # Step 3: collect top chunks within budget.
        result_chunks: list[Chunk] = []
        total_tokens = 0

        for doc_idx, _score in ranked:
            chunk = documents[doc_idx]
            if not self._budget.can_fit(chunk.token_count):
                break
            result_chunks.append(chunk)
            total_tokens += chunk.token_count
            self._budget.consume(chunk.token_count)

        logger.info(
            "Level-1 retrieved %d/%d chunks (%d tokens)",
            len(result_chunks),
            len(documents),
            total_tokens,
        )

        return RetrievalResult(
            chunks=result_chunks,
            total_tokens=total_tokens,
            query_keywords=keywords,
        )

    async def _expand_keywords(self, query: str) -> list[str]:
        """Ask the LLM to expand the query into search keywords."""
        try:
            response = await self._llm.chat(
                _KEYWORD_SYSTEM_PROMPT,
                query,
                max_tokens=256,
                temperature=0.1,
            )
            # Parse one keyword per line, lowercase, filter empties.
            raw = [line.strip().lower() for line in response.strip().split("\n") if line.strip()]
            # Also include the original query tokens.
            raw.extend(_tokenize(query))
            # Deduplicate while preserving order.
            seen: set[str] = set()
            keywords: list[str] = []
            for kw in raw:
                if kw not in seen:
                    seen.add(kw)
                    keywords.append(kw)
            return keywords
        except Exception:
            logger.warning(
                "Keyword expansion failed, falling back to raw tokenization",
                exc_info=True,
            )
            return _tokenize(query)
