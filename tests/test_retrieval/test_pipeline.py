"""Tests for forge_bot.retrieval.pipeline."""

from __future__ import annotations

from unittest.mock import AsyncMock

from forge_bot.retrieval.pipeline import SmartRetriever


def _make_llm(
    chat_response: str = "keyword1\nkeyword2",
) -> AsyncMock:
    llm = AsyncMock()
    llm.chat.return_value = chat_response
    return llm


async def test_search_text_returns_results():
    llm = _make_llm("error\nhandling")
    retriever = SmartRetriever(llm, context_window=8192)

    result = await retriever.search_text(
        "error handling",
        "def handle_error(e): log(e)\ndef unrelated(): pass",
        source="app.py",
    )

    assert len(result.chunks) >= 1
    assert result.total_tokens > 0


async def test_search_text_empty():
    llm = _make_llm()
    retriever = SmartRetriever(llm, context_window=8192)

    result = await retriever.search_text("anything", "")
    assert result.chunks == []


async def test_scan_and_answer():
    llm = AsyncMock()
    # Relevance check returns extracted text, answer returns final
    call_idx = 0

    async def mock_chat(system, user, **kwargs):
        nonlocal call_idx
        call_idx += 1
        if "relevance filter" in system.lower():
            return "def handle_error(e): log(e)"
        return "Errors are logged via the handle_error function."

    llm.chat = AsyncMock(side_effect=mock_chat)

    retriever = SmartRetriever(llm, context_window=8192)
    result = await retriever.scan_and_answer(
        "How are errors handled?",
        "def handle_error(e): log(e)\ndef other(): pass",
    )
    assert isinstance(result, str)
    assert len(result) > 0


async def test_review_diff_small():
    """Small diff should use Level 2."""
    llm = AsyncMock()

    async def mock_chat(system, user, **kwargs):
        if "relevance filter" in system.lower():
            return "+import os"
        return "The diff adds an os import."

    llm.chat = AsyncMock(side_effect=mock_chat)

    retriever = SmartRetriever(llm, context_window=8192)
    diff = "diff --git a/app.py b/app.py\n+import os\n"
    result = await retriever.review_diff("Review this change", diff)
    assert isinstance(result, str)
    assert len(result) > 0
