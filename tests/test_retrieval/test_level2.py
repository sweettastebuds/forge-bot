"""Tests for forge_bot.retrieval.level2."""

from __future__ import annotations

from unittest.mock import AsyncMock

from forge_bot.retrieval.level2 import Level2Scanner
from forge_bot.retrieval.token_budget import TokenBudget


def _make_llm(
    relevance_responses: list[str] | None = None,
    answer_response: str = "The function handles errors by logging them.",
) -> AsyncMock:
    """Mock LLM that returns relevance responses then an answer."""
    llm = AsyncMock()
    call_count = 0
    relevance = relevance_responses or []

    async def mock_chat(system, user, **kwargs):
        nonlocal call_count
        if "relevance filter" in system.lower():
            idx = min(call_count, len(relevance) - 1) if relevance else 0
            call_count += 1
            return relevance[idx] if relevance else "none"
        return answer_response

    llm.chat = AsyncMock(side_effect=mock_chat)
    return llm


async def test_scan_and_answer_with_relevant_content():
    llm = _make_llm(
        relevance_responses=[
            "def handle_error(e): logging.error(str(e))",
            "none",
        ],
        answer_response="Errors are handled by logging.",
    )
    budget = TokenBudget(context_window=8192)
    scanner = Level2Scanner(llm, budget)

    result = await scanner.retrieve_and_answer(
        "How are errors handled?",
        "def handle_error(e): logging.error(str(e))\n\ndef unrelated(): pass",
        source="app.py",
    )

    assert "Errors" in result or "error" in result.lower()


async def test_scan_and_answer_no_relevant_content():
    llm = _make_llm(relevance_responses=["none"])
    budget = TokenBudget(context_window=8192)
    scanner = Level2Scanner(llm, budget)

    result = await scanner.retrieve_and_answer(
        "How is quantum physics implemented?",
        "def add(a, b): return a + b",
    )

    assert "no relevant" in result.lower() or "not" in result.lower()


async def test_scan_only_returns_extracts():
    llm = _make_llm(
        relevance_responses=["logging.error(str(e))", "none"],
    )
    budget = TokenBudget(context_window=8192)
    scanner = Level2Scanner(llm, budget)

    extracts = await scanner.scan_only(
        "Where is error logging?",
        "def handle(e): logging.error(str(e))\n\ndef other(): pass",
    )

    assert len(extracts) >= 1


async def test_scan_handles_empty_text():
    llm = _make_llm()
    budget = TokenBudget(context_window=8192)
    scanner = Level2Scanner(llm, budget)

    result = await scanner.retrieve_and_answer("anything", "")
    assert "no content" in result.lower()
