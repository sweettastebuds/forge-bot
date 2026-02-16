"""High-level retrieval interface for handlers."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from forge_bot.rag.pipeline import RAGPipeline

logger = logging.getLogger("forge_bot.rag.retriever")


class Retriever:
    """Query interface that handlers use to get RAG context."""

    def __init__(self, pipeline: RAGPipeline, max_context_tokens: int = 4000) -> None:
        self._pipeline = pipeline
        self._max_chars = max_context_tokens * 4  # ~4 chars per token

    async def get_context_for_issue(
        self,
        owner: str,
        repo: str,
        ref: str,
        question: str,
        top_k: int = 10,
    ) -> str:
        """Retrieve context relevant to an issue question.

        Ensures the repo is indexed, then queries with the question text.
        """
        await self._pipeline.ensure_indexed(owner, repo, ref)
        context = await self._pipeline.retrieve(owner, repo, question, top_k=top_k)
        return self._truncate(context)

    async def get_context_for_pr(
        self,
        owner: str,
        repo: str,
        ref: str,
        changed_files: list[str],
        pr_title: str = "",
        top_k: int = 10,
    ) -> str:
        """Retrieve context relevant to a PR.

        Queries with changed file paths and PR title to find related code.
        """
        await self._pipeline.ensure_indexed(owner, repo, ref)

        query = f"Files changed: {', '.join(changed_files)}"
        if pr_title:
            query = f"{pr_title}. {query}"

        context = await self._pipeline.retrieve(owner, repo, query, top_k=top_k)
        return self._truncate(context)

    def _truncate(self, context: str) -> str:
        """Truncate context to the configured max size."""
        if len(context) <= self._max_chars:
            return context
        truncated = context[: self._max_chars]
        # Try to cut at a code block boundary.
        last_fence = truncated.rfind("```\n")
        if last_fence > self._max_chars // 2:
            truncated = truncated[: last_fence + 4]
        return truncated + "\n\n*(context truncated)*"
