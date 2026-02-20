"""Tests for forge_bot.retrieval.level1."""

from __future__ import annotations

from unittest.mock import AsyncMock

from forge_bot.retrieval.chunking import Chunk
from forge_bot.retrieval.level1 import Level1Retriever
from forge_bot.retrieval.token_budget import TokenBudget


def _make_llm(keyword_response: str = "error\nhandling\nexception") -> AsyncMock:
    llm = AsyncMock()
    llm.chat.return_value = keyword_response
    return llm


def _make_chunks() -> list[Chunk]:
    return [
        Chunk(
            content="def handle_error(e): logging.error(str(e))",
            source="errors.py",
            index=0,
            token_count=12,
        ),
        Chunk(
            content="def connect_db(): return psycopg2.connect(dsn)",
            source="db.py",
            index=1,
            token_count=12,
        ),
        Chunk(
            content="class AuthError(Exception): pass",
            source="auth.py",
            index=2,
            token_count=8,
        ),
    ]


async def test_retrieve_returns_relevant_chunks():
    llm = _make_llm()
    budget = TokenBudget(context_window=8192)
    retriever = Level1Retriever(llm, budget)

    result = await retriever.retrieve("How are errors handled?", _make_chunks())

    assert len(result.chunks) >= 1
    # Error-related chunks should be ranked first
    sources = [c.source for c in result.chunks]
    assert "errors.py" in sources


async def test_retrieve_respects_budget():
    llm = _make_llm()
    budget = TokenBudget(context_window=2048, reserve_output=1024, reserve_system=512)
    # Available = 512 tokens
    retriever = Level1Retriever(llm, budget)

    # Create many large chunks
    chunks = [
        Chunk(content="x" * 2000, source=f"big_{i}.py", index=i, token_count=500) for i in range(10)
    ]
    result = await retriever.retrieve("find something", chunks)
    assert result.total_tokens <= budget.available + 500  # at most one over


async def test_retrieve_empty_docs():
    llm = _make_llm()
    budget = TokenBudget(context_window=8192)
    retriever = Level1Retriever(llm, budget)

    result = await retriever.retrieve("anything", [])
    assert result.chunks == []
    assert result.total_tokens == 0


async def test_retrieve_falls_back_on_llm_failure():
    llm = AsyncMock()
    llm.chat.side_effect = RuntimeError("LLM down")
    budget = TokenBudget(context_window=8192)
    retriever = Level1Retriever(llm, budget)

    # Should not raise — falls back to raw tokenization
    result = await retriever.retrieve("error handling", _make_chunks())
    assert len(result.query_keywords) > 0


async def test_retrieve_deduplicates_keywords():
    llm = _make_llm("error\nerror\nhandle")
    budget = TokenBudget(context_window=8192)
    retriever = Level1Retriever(llm, budget)

    result = await retriever.retrieve("error", _make_chunks())
    # Keywords should not have duplicates
    assert len(result.query_keywords) == len(set(result.query_keywords))
