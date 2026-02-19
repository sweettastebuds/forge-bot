"""Orchestrator that chains Level 1 → Level 2 → Level 3 as needed.

Provides a single ``SmartRetriever`` facade that handlers call.  The
retriever decides which level to use based on the complexity of the
question and the size of the input.
"""

from __future__ import annotations

import logging

from forge_bot.clients.llm import LLMClient
from forge_bot.retrieval.chunking import chunk_diff_by_file, chunk_text
from forge_bot.retrieval.level1 import Level1Retriever, RetrievalResult
from forge_bot.retrieval.level2 import Level2Scanner
from forge_bot.retrieval.level3 import Level3Agent, SourceDocument
from forge_bot.retrieval.token_budget import TokenBudget, estimate_tokens

logger = logging.getLogger("forge_bot.retrieval.pipeline")


class SmartRetriever:
    """Facade over the three-level retrieval hierarchy.

    Instantiate once per request with the appropriate token budget,
    then call ``search_text``, ``review_diff``, or ``answer_complex``.
    """

    def __init__(
        self,
        llm: LLMClient,
        context_window: int = 8192,
    ) -> None:
        self._llm = llm
        self._context_window = context_window

    def _make_budget(self) -> TokenBudget:
        return TokenBudget(context_window=self._context_window)

    # -- Level 1: fast keyword search --------------------------------------

    async def search_text(
        self,
        query: str,
        text: str,
        *,
        source: str = "",
        chunk_tokens: int = 512,
        top_k: int = 20,
    ) -> RetrievalResult:
        """BM25 keyword search over *text* (Level 1).

        Returns the most relevant chunks within the token budget.
        Best for: quick lookups where keyword overlap is strong.
        """
        budget = self._make_budget()
        chunks = chunk_text(
            text,
            chunk_tokens=chunk_tokens,
            source=source,
        )
        retriever = Level1Retriever(self._llm, budget)
        return await retriever.retrieve(query, chunks, top_k=top_k)

    # -- Level 2: thorough parallel scan -----------------------------------

    async def scan_and_answer(
        self,
        query: str,
        text: str,
        *,
        source: str = "",
    ) -> str:
        """Parallel chunk scan + refined BM25 + answer (Level 2).

        Returns an LLM-generated answer grounded in the text.
        Best for: thorough analysis where you can't miss anything.
        """
        budget = self._make_budget()
        scanner = Level2Scanner(self._llm, budget)
        return await scanner.retrieve_and_answer(query, text, source=source)

    # -- Level 3: multi-hop reasoning --------------------------------------

    async def answer_complex(
        self,
        question: str,
        sources: dict[str, str],
    ) -> str:
        """Multi-hop decomposition across multiple sources (Level 3).

        *sources* is a ``{name: content}`` mapping.  The agent decides
        which sources to query and decomposes the question.

        Returns the synthesized answer.
        Best for: complex questions spanning multiple files/contexts.
        """
        budget = self._make_budget()
        scanner = Level2Scanner(self._llm, budget)
        agent = Level3Agent(self._llm, scanner, budget)
        docs = [SourceDocument(name=n, content=c) for n, c in sources.items()]
        return await agent.answer(question, docs)

    # -- Diff-specific helpers ---------------------------------------------

    async def review_diff(
        self,
        question: str,
        diff_text: str,
        *,
        file_contents: dict[str, str] | None = None,
    ) -> str:
        """Review a diff using the retrieval hierarchy.

        For small diffs (< budget), uses Level 2 for thorough scanning.
        For large diffs, uses Level 3 to decompose the review across
        per-file chunks and optional full file contents.

        Parameters
        ----------
        question:
            The review question (e.g. "Review this PR for bugs and
            inconsistencies").
        diff_text:
            The raw unified diff.
        file_contents:
            Optional ``{path: content}`` of full file contents for
            cross-referencing.  The agent retrieves from these as needed.
        """
        diff_tokens = estimate_tokens(diff_text)

        budget = self._make_budget()

        if diff_tokens <= budget.available:
            # Small diff — Level 2 is sufficient.
            logger.info(
                "Diff fits in budget (%d tokens), using Level 2",
                diff_tokens,
            )
            scanner = Level2Scanner(self._llm, budget)
            return await scanner.retrieve_and_answer(
                question,
                diff_text,
                source="diff",
            )

        # Large diff — use Level 3 with per-file sources.
        logger.info(
            "Large diff (%d tokens), using Level 3 multi-hop",
            diff_tokens,
        )
        sources: dict[str, str] = {}

        # Split diff by file.
        file_chunks = chunk_diff_by_file(diff_text)
        # Group chunks by source file.
        file_diffs: dict[str, list[str]] = {}
        for chunk in file_chunks:
            file_diffs.setdefault(chunk.source, []).append(chunk.content)

        for path, parts in file_diffs.items():
            sources[f"diff:{path}"] = "\n".join(parts)

        # Add full file contents if provided.
        if file_contents:
            for path, content in file_contents.items():
                sources[f"file:{path}"] = content

        return await self.answer_complex(question, sources)
