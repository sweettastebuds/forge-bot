"""Level 2 — Parallel chunk relevance scanning.

Every ~1,000-token chunk is sent to the LLM *in parallel* with the
question: "Is this relevant?  If yes, extract the relevant sentences."
Chunks returning ``none`` are discarded.  The surviving sentences form a
refined search query for a second BM25 pass with ~300-token chunks, and
the final answer is generated from those results.

This brute-force approach guarantees nothing is missed, while parallel
execution keeps latency manageable.
"""

from __future__ import annotations

import asyncio
import logging

from forge_bot.clients.llm import LLMClient
from forge_bot.retrieval.bm25 import BM25Index, _tokenize
from forge_bot.retrieval.chunking import Chunk, chunk_text
from forge_bot.retrieval.token_budget import (
    TokenBudget,
    truncate_to_tokens,
)

logger = logging.getLogger("forge_bot.retrieval.level2")

_RELEVANCE_SYSTEM_PROMPT = (
    "You are a relevance filter.  Given a QUESTION and a CODE CHUNK, "
    "decide if the chunk is relevant to answering the question.\n\n"
    "If relevant: extract ONLY the relevant lines or sentences, verbatim.\n"
    "If not relevant: respond with exactly the word 'none'.\n\n"
    "Do NOT explain, summarize, or add commentary.  Output extracted "
    "lines or 'none'."
)

_ANSWER_SYSTEM_PROMPT = (
    "You are a technical assistant.  Using ONLY the provided context, "
    "answer the question.  If the context is insufficient, say so.  "
    "Do not fabricate information."
)


class Level2Scanner:
    """Parallel chunk scanner with refinement pass."""

    def __init__(
        self,
        llm: LLMClient,
        budget: TokenBudget,
        *,
        max_parallel: int = 10,
    ) -> None:
        self._llm = llm
        self._budget = budget
        self._max_parallel = max_parallel

    async def retrieve_and_answer(
        self,
        query: str,
        raw_text: str,
        *,
        source: str = "",
    ) -> str:
        """Scan *raw_text* for relevant content and answer *query*.

        Returns the LLM-generated answer string.
        """
        # Phase 1: chunk at ~1000 tokens for scanning.
        scan_chunks = chunk_text(
            raw_text,
            chunk_tokens=self._budget.level2_chunk_tokens,
            overlap_tokens=50,
            source=source,
        )

        if not scan_chunks:
            return "No content available to analyze."

        logger.info(
            "Level-2 scanning %d chunks (%s) for query",
            len(scan_chunks),
            source or "inline",
        )

        # Phase 2: parallel relevance scanning.
        relevant_sentences = await self._parallel_scan(query, scan_chunks)

        if not relevant_sentences:
            return "No relevant content found for your question."

        logger.info(
            "Level-2 found %d relevant extracts",
            len(relevant_sentences),
        )

        # Phase 3: refined BM25 pass with ~300-token chunks.
        refined_context = self._refine_with_bm25(
            query,
            relevant_sentences,
            raw_text,
        )

        # Phase 4: generate answer from refined context.
        answer = await self._generate_answer(query, refined_context)
        return answer

    async def scan_only(
        self,
        query: str,
        raw_text: str,
        *,
        source: str = "",
    ) -> list[str]:
        """Scan *raw_text* and return only the relevant extracted sentences.

        Useful when Level 3 needs raw extracts rather than a full answer.
        """
        scan_chunks = chunk_text(
            raw_text,
            chunk_tokens=self._budget.level2_chunk_tokens,
            overlap_tokens=50,
            source=source,
        )
        return await self._parallel_scan(query, scan_chunks)

    # -- Internal ----------------------------------------------------------

    async def _parallel_scan(
        self,
        query: str,
        chunks: list[Chunk],
    ) -> list[str]:
        """Send each chunk to the LLM in parallel for relevance filtering."""
        sem = asyncio.Semaphore(self._max_parallel)

        async def _check_one(chunk: Chunk) -> str | None:
            async with sem:
                user_msg = (
                    f"QUESTION:\n{query}\n\nCODE CHUNK (from {chunk.source}):\n{chunk.content}"
                )
                try:
                    result = await self._llm.chat(
                        _RELEVANCE_SYSTEM_PROMPT,
                        user_msg,
                        max_tokens=512,
                        temperature=0.0,
                    )
                    stripped = result.strip().lower()
                    if stripped == "none" or not stripped:
                        return None
                    return result.strip()
                except Exception:
                    logger.warning(
                        "Relevance check failed for chunk %d (%s)",
                        chunk.index,
                        chunk.source,
                        exc_info=True,
                    )
                    return None

        tasks = [_check_one(c) for c in chunks]
        results = await asyncio.gather(*tasks)

        return [r for r in results if r is not None]

    def _refine_with_bm25(
        self,
        query: str,
        relevant_sentences: list[str],
        full_text: str,
    ) -> str:
        """Use extracted sentences as a refined query for a second BM25 pass.

        Chunks the full text at ~300 tokens, builds a BM25 index, and
        queries using the relevant sentence tokens plus the original query.
        """
        # Build refined query from extracted sentences.
        combined_extracts = "\n".join(relevant_sentences)
        refined_keywords = _tokenize(combined_extracts + " " + query)

        # Chunk at finer granularity.
        fine_chunks = chunk_text(
            full_text,
            chunk_tokens=self._budget.level2_refine_chunk_tokens,
            overlap_tokens=30,
        )

        if not fine_chunks:
            return combined_extracts

        texts = [c.content for c in fine_chunks]
        index = BM25Index.build(texts)
        ranked = index.query(refined_keywords, top_k=15)

        # Collect results within budget.
        context_parts: list[str] = []
        total_tokens = 0
        max_tokens = self._budget.available

        for doc_idx, _score in ranked:
            chunk = fine_chunks[doc_idx]
            if total_tokens + chunk.token_count > max_tokens:
                break
            context_parts.append(chunk.content)
            total_tokens += chunk.token_count

        if not context_parts:
            # Fall back to the raw extracts if BM25 found nothing.
            return truncate_to_tokens(combined_extracts, max_tokens)

        return "\n---\n".join(context_parts)

    async def _generate_answer(self, query: str, context: str) -> str:
        """Generate a final answer from the refined context."""
        user_msg = f"CONTEXT:\n{context}\n\nQUESTION:\n{query}"
        budget_tokens = min(self._budget.available, 2048)
        try:
            return await self._llm.chat(
                _ANSWER_SYSTEM_PROMPT,
                user_msg,
                max_tokens=budget_tokens,
                temperature=0.2,
            )
        except Exception:
            logger.exception("Level-2 answer generation failed")
            return "Failed to generate an answer from the retrieved context."
